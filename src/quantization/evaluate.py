"""Evaluate one PyTorch or ONNX token-classification artifact on CPU.

The module keeps the historical word-level BIO evaluation for comparability with
``src/training/evaluate_pii.py`` and adds a production-like character-span view.
The latter follows every subword and the grouping semantics of Transformers
``aggregation_strategy="simple"``. It also writes compact predictions (labels and
spans only, never the source text) so quantized variants can be compared with the
published FP32 checkpoint in a separate process.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import resource
import statistics
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Collection, Iterable, Sequence

from .metrics import (
    canonical_text_and_word_offsets,
    compute_span_metrics,
    evaluate_predictions,
    normalize_labels,
    simple_char_spans,
    spans,
    word_spans_to_char_spans,
)


@dataclass(frozen=True)
class ValidationRecord:
    tokens: list[str]
    labels: list[str]


class RssSampler:
    """Poll process RSS without keeping another model alive in the parent."""

    def __init__(self, interval_seconds: float = 0.02) -> None:
        self.interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.start_bytes = 0
        self.peak_bytes = 0

    @staticmethod
    def _rss() -> int:
        try:
            import psutil

            return int(psutil.Process(os.getpid()).memory_info().rss)
        except ImportError:
            return 0

    def __enter__(self) -> "RssSampler":
        self.start_bytes = self._rss()
        self.peak_bytes = self.start_bytes

        def sample() -> None:
            while not self._stop.wait(self.interval_seconds):
                self.peak_bytes = max(self.peak_bytes, self._rss())

        self._thread = threading.Thread(target=sample, name="rss-sampler", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.peak_bytes = max(self.peak_bytes, self._rss())
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1)


def _resource_max_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # Darwin reports bytes, Linux and the BSDs normally report KiB.
    return value if sys.platform == "darwin" else value * 1024


def _memory_snapshot() -> dict[str, int | None]:
    """Collect cheap process memory fields; Linux detail is supplied by smaps."""
    result: dict[str, int | None] = {
        "rss_bytes": RssSampler._rss(),
        "pss_bytes": None,
        "uss_bytes": None,
        "anonymous_bytes": None,
        "swap_bytes": None,
        "resource_max_rss_bytes": _resource_max_rss_bytes(),
    }
    try:
        from .profile import sample_process_memory

        result.update(sample_process_memory(os.getpid()))
    except (ImportError, OSError, PermissionError):
        pass
    result["resource_max_rss_bytes"] = _resource_max_rss_bytes()
    return result


def _artifact_stat_snapshot(path: Path) -> list[dict[str, int | str]]:
    """Snapshot artifact-directory metadata without reading model contents."""
    model_path = path.expanduser().resolve()
    directory = model_path.parent if model_path.is_file() or model_path.suffix == ".onnx" else model_path
    if not directory.is_dir():
        raise FileNotFoundError(f"Directory artefatto non trovata: {directory}")
    rows: list[dict[str, int | str]] = []
    for candidate in sorted(directory.iterdir(), key=lambda item: item.name):
        if not candidate.is_file():
            continue
        stat = candidate.stat()
        rows.append(
            {
                "path": str(candidate.resolve()),
                "inode": int(stat.st_ino),
                "bytes": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            }
        )
    return rows


def _file_stat_snapshot(path: Path) -> dict[str, int | str]:
    candidate = path.expanduser().resolve()
    if not candidate.is_file():
        raise FileNotFoundError(f"File non trovato: {candidate}")
    stat = candidate.stat()
    return {
        "path": str(candidate),
        "inode": int(stat.st_ino),
        "bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


_PREPROCESSOR_FILENAMES = {
    "added_tokens.json",
    "config.json",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "vocab.txt",
}


def _preprocessor_files(path: Path) -> list[Path]:
    """Return the local files that determine token IDs and output labels."""

    directory = path.expanduser().resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"Directory tokenizer non trovata: {directory}")
    files = sorted(
        (
            candidate
            for candidate in directory.iterdir()
            if candidate.is_file()
            and (
                candidate.name in _PREPROCESSOR_FILENAMES
                or candidate.name.startswith("tokenizer.")
                or candidate.suffix == ".model"
            )
        ),
        key=lambda candidate: candidate.name,
    )
    if not files:
        raise FileNotFoundError(
            f"Tokenizer/config locali non trovati in {directory}"
        )
    if not any(candidate.name == "config.json" for candidate in files):
        raise FileNotFoundError(f"config.json non trovato in {directory}")
    return files


def _preprocessor_stat_snapshot(path: Path) -> list[dict[str, int | str]]:
    rows: list[dict[str, int | str]] = []
    for candidate in _preprocessor_files(path):
        stat = candidate.stat()
        rows.append(
            {
                "path": str(candidate.resolve()),
                "inode": int(stat.st_ino),
                "bytes": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            }
        )
    return rows


def preprocessor_integrity(path: Path) -> dict[str, Any]:
    """Hash the tokenizer and label-map files used by an evaluation."""

    directory = path.expanduser().resolve()
    files = [
        {
            "path": str(candidate.resolve()),
            "bytes": candidate.stat().st_size,
            "sha256": _sha256(candidate),
        }
        for candidate in _preprocessor_files(directory)
    ]
    return {"path": str(directory), "files": files}


def load_validation(path: Path, limit: int | None = None) -> list[ValidationRecord]:
    records: list[ValidationRecord] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            tokens = row.get("tokens")
            labels = row.get("bio_labels")
            if not isinstance(tokens, list) or not isinstance(labels, list):
                raise ValueError(f"{path}:{line_number}: tokens/bio_labels mancanti")
            if not tokens or len(tokens) != len(labels):
                raise ValueError(f"{path}:{line_number}: array vuoti o disallineati")
            records.append(ValidationRecord(list(tokens), normalize_labels(list(labels))))
            if limit is not None and len(records) >= limit:
                break
    if not records:
        raise ValueError(f"Nessun record valido in {path}")
    return records


def _artifact_size(path: Path) -> int:
    if path.is_file():
        # External-data locations are arbitrary ONNX metadata, not necessarily
        # ``model.onnx.data``.  Reuse the graph-aware implementation.
        from .quantize import artifact_info

        return int(artifact_info(path)["bytes"])
    safetensors = list(path.glob("*.safetensors"))
    if safetensors:
        return sum(item.stat().st_size for item in safetensors)
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = index - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _word_predictions(
    token_predictions: Sequence[Sequence[int]],
    word_ids_by_row: Sequence[Sequence[int | None]],
    id2label: dict[int, str],
) -> list[list[str]]:
    documents: list[list[str]] = []
    for token_ids, word_ids in zip(token_predictions, word_ids_by_row):
        seen: set[int] = set()
        labels: list[str] = []
        for position, word_id in enumerate(word_ids):
            if word_id is None or word_id in seen:
                continue
            seen.add(word_id)
            labels.append(id2label[int(token_ids[position])])
        documents.append(labels)
    return documents


def _production_like_char_predictions(
    token_predictions: Sequence[Sequence[int]],
    offsets_by_row: Sequence[Sequence[Sequence[int]]],
    special_tokens_masks: Sequence[Sequence[int | bool]],
    word_ids_by_row: Sequence[Sequence[int | None]],
    records: Sequence[ValidationRecord],
    id2label: dict[int, str],
) -> list[set[tuple[str, int, int]]]:
    """Aggrega tutti i subword in span carattere come la pipeline ``simple``.

    Gli offset del tokenizer sono locali alle parole perche' il dataset e'
    token-list. Vengono proiettati sul testo canonico ``" ".join(tokens)``;
    questa conserva un sistema di coordinate esatto e riproducibile, pur non
    pretendendo di ricostruire whitespace/punteggiatura del testo originale.
    """

    row_counts = {
        len(token_predictions),
        len(offsets_by_row),
        len(special_tokens_masks),
        len(word_ids_by_row),
        len(records),
    }
    if len(row_counts) != 1:
        raise ValueError("Batch disallineato durante l'aggregazione production-like")

    documents: list[set[tuple[str, int, int]]] = []
    for token_ids, offsets, special_mask, word_ids, record in zip(
        token_predictions,
        offsets_by_row,
        special_tokens_masks,
        word_ids_by_row,
        records,
    ):
        _, word_offsets = canonical_text_and_word_offsets(record.tokens)
        token_labels = [id2label[int(token_id)] for token_id in token_ids]
        documents.append(
            simple_char_spans(
                token_labels,
                offsets,
                special_tokens_mask=special_mask,
                word_ids=word_ids,
                word_offsets=word_offsets,
            )
        )
    return documents


def _batched(items: Sequence[ValidationRecord], size: int) -> Iterable[Sequence[ValidationRecord]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _load_id2label(config_path: Path) -> dict[int, str]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    labels = config.get("id2label")
    if not isinstance(labels, dict):
        raise ValueError(f"id2label assente in {config_path}")
    return {int(key): str(value) for key, value in labels.items()}


def _find_onnx(path: Path) -> Path:
    if path.is_file() and path.suffix == ".onnx":
        return path
    candidates = sorted(path.glob("*.onnx"))
    if len(candidates) != 1:
        raise ValueError(f"Atteso un solo .onnx in {path}, trovati: {candidates}")
    return candidates[0]


def _run_torch(
    model_path: Path,
    tokenizer_path: Path,
    records: Sequence[ValidationRecord],
    batch_size: int,
    max_length: int,
    threads: int,
    warmup_batches: int = 0,
) -> tuple[list[list[str]], dict[str, Any]]:
    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    torch.set_grad_enabled(False)
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    load_started = time.perf_counter()
    # This is a ModernBERT/Gemma tokenizer (Metaspace), not a Mistral tokenizer.
    # Transformers 4.57.6 can emit a false-positive large-vocabulary warning;
    # explicitly keep the published regex/pre-tokenizer unchanged for parity.
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, local_files_only=True, fix_mistral_regex=False
    )
    model = AutoModelForTokenClassification.from_pretrained(
        model_path,
        local_files_only=True,
        attn_implementation="eager",
    ).cpu().eval()
    load_seconds = time.perf_counter() - load_started
    memory_after_load = _memory_snapshot()
    id2label = {int(key): str(value) for key, value in model.config.id2label.items()}

    def infer_batch(
        batch: Sequence[ValidationRecord],
    ) -> tuple[
        list[list[str]], list[set[tuple[str, int, int]]], float, float
    ]:
        request_started = time.perf_counter()
        encoded = tokenizer(
            [row.tokens for row in batch],
            is_split_into_words=True,
            truncation=True,
            max_length=max_length,
            padding=True,
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
            return_tensors="pt",
        )
        word_ids = [encoded.word_ids(batch_index=i) for i in range(len(batch))]
        offsets = encoded.pop("offset_mapping").cpu().tolist()
        special_tokens_masks = encoded.pop("special_tokens_mask").cpu().tolist()
        model_started = time.perf_counter()
        logits = model(**encoded).logits
        model_seconds = time.perf_counter() - model_started
        token_predictions = logits.argmax(-1).cpu().tolist()
        labels = _word_predictions(token_predictions, word_ids, id2label)
        char_spans = _production_like_char_predictions(
            token_predictions,
            offsets,
            special_tokens_masks,
            word_ids,
            batch,
            id2label,
        )
        return labels, char_spans, time.perf_counter() - request_started, model_seconds

    predictions: list[list[str]] = []
    model_latencies: list[float] = []
    request_latencies: list[float] = []
    warmup_model_latencies: list[float] = []
    warmup_request_latencies: list[float] = []
    production_like_char_spans: list[set[tuple[str, int, int]]] = []
    warmup_batch = records[: min(batch_size, len(records))]
    with torch.inference_mode():
        for _ in range(warmup_batches):
            _, _, request_seconds, model_seconds = infer_batch(warmup_batch)
            warmup_request_latencies.append(request_seconds)
            warmup_model_latencies.append(model_seconds)
        memory_after_warmup = _memory_snapshot()

        inference_started = time.perf_counter()
        for batch in _batched(records, batch_size):
            labels, char_spans, request_seconds, model_seconds = infer_batch(batch)
            predictions.extend(labels)
            production_like_char_spans.extend(char_spans)
            request_latencies.append(request_seconds)
            model_latencies.append(model_seconds)
    inference_seconds = time.perf_counter() - inference_started
    memory_after_inference = _memory_snapshot()
    return predictions, {
        "load_seconds": load_seconds,
        "inference_seconds": inference_seconds,
        "rss_after_load_bytes": memory_after_load["rss_bytes"],
        "batch_latencies_seconds": model_latencies,
        "request_latencies_seconds": request_latencies,
        "warmup_model_latencies_seconds": warmup_model_latencies,
        "warmup_request_latencies_seconds": warmup_request_latencies,
        "_production_like_char_spans": production_like_char_spans,
        "memory_snapshots": {
            "after_load": memory_after_load,
            "after_warmup": memory_after_warmup,
            "after_inference": memory_after_inference,
        },
    }


def _run_onnx(
    model_path: Path,
    tokenizer_path: Path,
    records: Sequence[ValidationRecord],
    batch_size: int,
    max_length: int,
    threads: int,
    warmup_batches: int = 0,
) -> tuple[list[list[str]], dict[str, Any]]:
    import numpy as np
    import onnxruntime as ort
    from transformers import AutoTokenizer

    onnx_path = _find_onnx(model_path)
    load_started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, local_files_only=True, fix_mistral_regex=False
    )
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session = ort.InferenceSession(
        str(onnx_path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    if session.get_providers() != ["CPUExecutionProvider"]:
        raise RuntimeError(f"Provider inatteso: {session.get_providers()}")
    load_seconds = time.perf_counter() - load_started
    memory_after_load = _memory_snapshot()
    # The canonical, source-verified checkpoint supplies both tokenizer and
    # id2label.  Variant directories contain only graph/runtime artifacts and
    # must not be able to silently remap logits after evaluation.
    config_path = tokenizer_path / "config.json"
    id2label = _load_id2label(config_path)
    input_names = {item.name for item in session.get_inputs()}

    def infer_batch(
        batch: Sequence[ValidationRecord],
    ) -> tuple[
        list[list[str]], list[set[tuple[str, int, int]]], float, float
    ]:
        request_started = time.perf_counter()
        encoded = tokenizer(
            [row.tokens for row in batch],
            is_split_into_words=True,
            truncation=True,
            max_length=max_length,
            padding=True,
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
            return_tensors="np",
        )
        word_ids = [encoded.word_ids(batch_index=i) for i in range(len(batch))]
        offsets = encoded.pop("offset_mapping").tolist()
        special_tokens_masks = encoded.pop("special_tokens_mask").tolist()
        inputs = {
            name: np.asarray(encoded[name], dtype=np.int64)
            for name in input_names
            if name in encoded
        }
        missing = input_names - inputs.keys()
        if missing:
            raise RuntimeError(f"Input ONNX non prodotti dal tokenizer: {sorted(missing)}")
        model_started = time.perf_counter()
        logits = session.run(None, inputs)[0]
        model_seconds = time.perf_counter() - model_started
        token_predictions = np.asarray(logits).argmax(-1).tolist()
        labels = _word_predictions(token_predictions, word_ids, id2label)
        char_spans = _production_like_char_predictions(
            token_predictions,
            offsets,
            special_tokens_masks,
            word_ids,
            batch,
            id2label,
        )
        return labels, char_spans, time.perf_counter() - request_started, model_seconds

    predictions: list[list[str]] = []
    model_latencies: list[float] = []
    request_latencies: list[float] = []
    warmup_model_latencies: list[float] = []
    warmup_request_latencies: list[float] = []
    production_like_char_spans: list[set[tuple[str, int, int]]] = []
    warmup_batch = records[: min(batch_size, len(records))]
    for _ in range(warmup_batches):
        _, _, request_seconds, model_seconds = infer_batch(warmup_batch)
        warmup_request_latencies.append(request_seconds)
        warmup_model_latencies.append(model_seconds)
    memory_after_warmup = _memory_snapshot()

    inference_started = time.perf_counter()
    for batch in _batched(records, batch_size):
        labels, char_spans, request_seconds, model_seconds = infer_batch(batch)
        predictions.extend(labels)
        production_like_char_spans.extend(char_spans)
        request_latencies.append(request_seconds)
        model_latencies.append(model_seconds)
    inference_seconds = time.perf_counter() - inference_started
    memory_after_inference = _memory_snapshot()
    return predictions, {
        "load_seconds": load_seconds,
        "inference_seconds": inference_seconds,
        "rss_after_load_bytes": memory_after_load["rss_bytes"],
        "batch_latencies_seconds": model_latencies,
        "request_latencies_seconds": request_latencies,
        "warmup_model_latencies_seconds": warmup_model_latencies,
        "warmup_request_latencies_seconds": warmup_request_latencies,
        "_production_like_char_spans": production_like_char_spans,
        "memory_snapshots": {
            "after_load": memory_after_load,
            "after_warmup": memory_after_warmup,
            "after_inference": memory_after_inference,
        },
        "onnxruntime_version": ort.__version__,
        "providers": session.get_providers(),
    }


def write_predictions(
    path: Path,
    predictions: Sequence[Sequence[str]],
    production_like_char_spans: Sequence[
        Collection[tuple[str, int, int]]
    ] | None = None,
) -> None:
    if (
        production_like_char_spans is not None
        and len(production_like_char_spans) != len(predictions)
    ):
        raise ValueError("Predizioni word e character-span disallineate")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for index, labels in enumerate(predictions):
            compact_spans = [list(span) for span in sorted(spans(list(labels)))]
            row: dict[str, Any] = {
                "id": index,
                "labels": list(labels),
                "spans": compact_spans,
            }
            if production_like_char_spans is not None:
                row["char_spans"] = [
                    list(span)
                    for span in sorted(production_like_char_spans[index])
                ]
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_predictions(path: Path) -> list[list[str]]:
    predictions: list[list[str]] = []
    with path.open(encoding="utf-8") as handle:
        for expected_id, line in enumerate(handle):
            row = json.loads(line)
            if row.get("id") != expected_id or not isinstance(row.get("labels"), list):
                raise ValueError(f"Predizioni non valide in {path}, riga {expected_id + 1}")
            predictions.append([str(label) for label in row["labels"]])
    return predictions


def read_production_char_predictions(
    path: Path,
) -> list[set[tuple[str, int, int]]] | None:
    """Read persisted model-only character spans, or ``None`` for legacy files."""

    documents: list[set[tuple[str, int, int]]] = []
    availability: bool | None = None
    with path.open(encoding="utf-8") as handle:
        for expected_id, line in enumerate(handle):
            row = json.loads(line)
            if row.get("id") != expected_id:
                raise ValueError(f"Predizioni non valide in {path}, riga {expected_id + 1}")
            raw_spans = row.get("char_spans")
            present = isinstance(raw_spans, list)
            if availability is None:
                availability = present
            elif availability != present:
                raise ValueError(f"char_spans presenti solo in parte del file {path}")
            if not present:
                continue
            parsed: set[tuple[str, int, int]] = set()
            for raw_span in raw_spans:
                if (
                    not isinstance(raw_span, list)
                    or len(raw_span) != 3
                    or not isinstance(raw_span[0], str)
                    or isinstance(raw_span[1], bool)
                    or isinstance(raw_span[2], bool)
                    or not isinstance(raw_span[1], int)
                    or not isinstance(raw_span[2], int)
                    or raw_span[1] < 0
                    or raw_span[2] <= raw_span[1]
                ):
                    raise ValueError(
                        f"char_span non valido in {path}, riga {expected_id + 1}: {raw_span!r}"
                    )
                parsed.add((raw_span[0], raw_span[1], raw_span[2]))
            documents.append(parsed)
    return documents if availability else None


def evaluate_variant(
    *,
    name: str,
    backend: str,
    model_path: Path,
    tokenizer_path: Path,
    validation_path: Path,
    predictions_path: Path,
    report_path: Path,
    batch_size: int = 16,
    max_length: int = 768,
    threads: int = 1,
    limit: int | None = None,
    warmup_batches: int = 0,
    defer_artifact_hash: bool = False,
) -> dict[str, Any]:
    if warmup_batches < 0:
        raise ValueError("warmup_batches deve essere >= 0")
    validation_stat_before = _file_stat_snapshot(validation_path)
    preprocessor_stat_before = _preprocessor_stat_snapshot(tokenizer_path)
    records = load_validation(validation_path, limit=limit)
    gold = [row.labels for row in records]
    artifact_stat_before = _artifact_stat_snapshot(model_path) if backend == "onnx" else None
    artifact_integrity: dict[str, Any] | None = None

    with RssSampler() as rss:
        memory_start = _memory_snapshot()
        if backend == "torch":
            predictions, performance = _run_torch(
                model_path,
                tokenizer_path,
                records,
                batch_size,
                max_length,
                threads,
                warmup_batches,
            )
        elif backend == "onnx":
            predictions, performance = _run_onnx(
                model_path,
                tokenizer_path,
                records,
                batch_size,
                max_length,
                threads,
                warmup_batches,
            )
        else:
            raise ValueError(f"Backend non supportato: {backend}")

    if len(predictions) != len(gold):
        raise RuntimeError(f"Predizioni {len(predictions)} != record {len(gold)}")
    for index, (expected, candidate) in enumerate(zip(gold, predictions)):
        if len(expected) != len(candidate):
            raise RuntimeError(
                f"Documento {index}: {len(candidate)} predizioni per {len(expected)} parole"
            )

    if _file_stat_snapshot(validation_path) != validation_stat_before:
        raise RuntimeError("Validation modificata durante l'inferenza")
    validation_sha256 = _sha256(validation_path)
    if _file_stat_snapshot(validation_path) != validation_stat_before:
        raise RuntimeError("Validation modificata durante il controllo hash")

    if _preprocessor_stat_snapshot(tokenizer_path) != preprocessor_stat_before:
        raise RuntimeError("Tokenizer/config modificati durante l'inferenza")
    preprocessor_manifest = preprocessor_integrity(tokenizer_path)
    if _preprocessor_stat_snapshot(tokenizer_path) != preprocessor_stat_before:
        raise RuntimeError("Tokenizer/config modificati durante il controllo hash")

    if artifact_stat_before is not None:
        artifact_stat_after_inference = _artifact_stat_snapshot(model_path)
        if artifact_stat_after_inference != artifact_stat_before:
            raise RuntimeError("Artefatto ONNX modificato durante l'inferenza")

    if artifact_stat_before is not None and not defer_artifact_hash:
        from .quantize import artifact_info

        # Hash only after the measured region: hashing before session creation
        # would intentionally warm the OS page cache and bias RSS/load results.
        artifact_integrity = artifact_info(model_path)
        if _artifact_stat_snapshot(model_path) != artifact_stat_before:
            raise RuntimeError("Artefatto ONNX modificato durante il controllo hash")

    model_latencies = performance.pop("batch_latencies_seconds")
    request_latencies = performance.pop("request_latencies_seconds", model_latencies)
    warmup_model_latencies = performance.pop("warmup_model_latencies_seconds", [])
    warmup_request_latencies = performance.pop("warmup_request_latencies_seconds", [])
    production_like_predictions = performance.pop(
        "_production_like_char_spans", None
    )
    memory_snapshots = performance.pop("memory_snapshots", {})
    semantic = evaluate_predictions(gold, predictions)
    if production_like_predictions is not None:
        if len(production_like_predictions) != len(records):
            raise RuntimeError(
                "Numero di predizioni production-like diverso dai record: "
                f"{len(production_like_predictions)} != {len(records)}"
            )
        production_like_gold = []
        for record in records:
            _, word_offsets = canonical_text_and_word_offsets(record.tokens)
            production_like_gold.append(
                word_spans_to_char_spans(record.labels, word_offsets)
            )
        semantic["production_like_char"] = compute_span_metrics(
            production_like_gold,
            production_like_predictions,
        )
    inference_seconds = float(performance["inference_seconds"])
    rss_after_load = int(
        performance.pop(
            "rss_after_load_bytes",
            memory_snapshots.get("after_load", {}).get("rss_bytes", 0),
        )
    )
    explicit_rss_values = [rss.peak_bytes, rss_after_load]
    explicit_rss_values.extend(
        int(snapshot.get("rss_bytes") or 0) for snapshot in memory_snapshots.values()
    )
    rss_peak_bytes = max(explicit_rss_values)
    report: dict[str, Any] = {
        "schema_version": 1,
        "variant": name,
        "status": "ok",
        "backend": backend,
        "model_path": str(model_path.resolve()),
        "tokenizer_path": str(tokenizer_path.resolve()),
        "preprocessor_integrity": preprocessor_manifest,
        "preprocessor_integrity_timing": "post-evaluation-with-stat-guard",
        "preprocessor_stat_guard": preprocessor_stat_before,
        "validation_path": str(validation_path.resolve()),
        "validation_integrity": {
            "path": str(validation_path.resolve()),
            "bytes": int(validation_stat_before["bytes"]),
            "sha256": validation_sha256,
        },
        "validation_integrity_timing": "post-evaluation-with-stat-guard",
        "validation_stat_guard": validation_stat_before,
        "documents": len(records),
        "batch_size": batch_size,
        "max_length": max_length,
        "threads": threads,
        "warmup_batches": warmup_batches,
        "artifact_size_bytes": (
            int(artifact_integrity["bytes"])
            if artifact_integrity is not None
            else (_artifact_size(model_path) if backend != "onnx" else None)
        ),
        "rss_start_bytes": rss.start_bytes,
        "rss_after_load_bytes": rss_after_load,
        "rss_peak_bytes": rss_peak_bytes,
        "load_seconds": performance.pop("load_seconds"),
        "inference_seconds": inference_seconds,
        "documents_per_second": len(records) / inference_seconds if inference_seconds else 0.0,
        "batch_latency_seconds": {
            "mean": statistics.fmean(model_latencies) if model_latencies else 0.0,
            "p50": _percentile(model_latencies, 0.50),
            "p95": _percentile(model_latencies, 0.95),
            "samples": len(model_latencies),
        },
        "model_latency_seconds": {
            "mean": statistics.fmean(model_latencies) if model_latencies else 0.0,
            "p50": _percentile(model_latencies, 0.50),
            "p95": _percentile(model_latencies, 0.95),
            "samples": len(model_latencies),
        },
        "request_latency_seconds": {
            "mean": statistics.fmean(request_latencies) if request_latencies else 0.0,
            "p50": _percentile(request_latencies, 0.50),
            "p95": _percentile(request_latencies, 0.95),
            "samples": len(request_latencies),
        },
        "warmup": {
            "batches": warmup_batches,
            "model_latency_seconds": warmup_model_latencies,
            "request_latency_seconds": warmup_request_latencies,
        },
        "memory": {
            "sampling_interval_seconds": rss.interval_seconds,
            "start": memory_start,
            **memory_snapshots,
            "sampled_peak_rss_bytes": rss.peak_bytes,
            "peak_rss_bytes": rss_peak_bytes,
        },
        "metrics": semantic,
        "production_like_evaluation": (
            {
                "status": "available",
                "view": "model-only",
                "aggregation_strategy": "simple-equivalent",
                "coordinates": "character offsets over canonical single-space token-list text",
                "uses_all_model_subwords": True,
                "includes_regex_checksum_detectors": False,
                "includes_production_merge": False,
                "additional_model_forward_pass": False,
                "limitations": [
                    "the validation rows contain only tokens and BIO labels, not original text or original character offsets",
                    "canonical text is exactly one-space joining; original whitespace and punctuation attachment cannot be recovered",
                    "many source tokens already contain legacy WordPiece ## markers, which are preserved literally to keep offsets auditable",
                    "this catches internal-subword fragmentation but is not a substitute for an independently annotated raw-text span corpus",
                ],
                "end_to_end_gate": {
                    "status": "not-computable-on-this-dataset",
                    "required_view": "raw text -> model spans + regex/checksum detectors -> production _merge -> exact gold character spans",
                    "reason": "the token-list dataset has neither original text nor human gold character offsets",
                },
            }
            if production_like_predictions is not None
            else {
                "status": "unavailable",
                "reason": "backend runner did not return subword offsets (legacy or mocked evaluation)",
            }
        ),
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            **performance,
        },
    }
    if artifact_integrity is not None:
        report["artifact_integrity"] = artifact_integrity
        report["artifact_integrity_timing"] = "post-evaluation-with-stat-guard"
    elif artifact_stat_before is not None:
        report["artifact_integrity_timing"] = "deferred-with-stat-guard"
    if artifact_stat_before is not None:
        report["artifact_stat_guard"] = artifact_stat_before
    report["rss_model_load_delta_bytes"] = max(
        0, report["rss_after_load_bytes"] - report["rss_start_bytes"]
    )
    report["rss_peak_delta_bytes"] = max(
        0, report["rss_peak_bytes"] - report["rss_start_bytes"]
    )
    report["rss_inference_delta_bytes"] = max(
        0, report["rss_peak_bytes"] - report["rss_after_load_bytes"]
    )
    write_predictions(
        predictions_path,
        predictions,
        production_like_char_spans=production_like_predictions,
    )
    report["predictions_path"] = str(predictions_path.resolve())
    report["prediction_digest"] = _sha256(predictions_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report
