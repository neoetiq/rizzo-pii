"""Build a reproducible, offline teacher-logit cache for student distillation.

The cache is deliberately tied to one completed student run.  Dataset,
tokenizer, label order, windowing parameters and every padded training window
are cryptographically identified.  A consumer must reject the cache if any of
those inputs changes.

Only local model directories are accepted.  The student checkpoint is used for
its tokenizer and provenance; it is never loaded as a model.  Teacher inference
runs in PyTorch FP32 (CPU by default) and the resulting fixed-shape logits are
stored as FP16 in one safetensors file.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import random
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from src.training.student_utils import (
    IGNORE_INDEX,
    StudentTrainingError,
    TOKENIZER_FIX_MISTRAL_REGEX,
    align_char_spans,
    load_label_contract,
    sha256_file,
    tokenizer_load_policy,
)


SCHEMA_VERSION = 1
CACHE_KIND = "pii-teacher-logits-cache"
TENSOR_FILENAME = "cache.safetensors"
MANIFEST_FILENAME = "manifest.json"
MANIFEST_HASH_FILENAME = "manifest.sha256"

_MODEL_WEIGHT_NAMES = ("model.safetensors", "pytorch_model.bin")
_TOKENIZER_FILENAMES = {
    "added_tokens.json",
    "merges.txt",
    "sentencepiece.bpe.model",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "vocab.txt",
}
_REQUIRED_TENSORS = {
    "teacher_logits": "F16",
    "input_ids": "I32",
    "attention_mask": "U8",
    "labels": "I16",
}
_EXPECTED_MANIFEST_PATHS = {
    "dataset_sha256": ("dataset", "sha256"),
    "student_model_sha256": ("student", "model", "sha256"),
    "student_tokenizer_sha256": ("student", "tokenizer", "sha256"),
    "teacher_model_sha256": ("teacher", "model", "sha256"),
    "teacher_tokenizer_sha256": ("teacher", "tokenizer", "sha256"),
    "max_length": ("preprocessing", "max_length"),
    "stride": ("preprocessing", "stride"),
    "seed": ("preprocessing", "seed"),
    "num_labels": ("label_contract", "num_labels"),
    "num_windows": ("counts", "windows"),
}

_SPECIAL_TOKEN_ROLES = (
    "bos",
    "eos",
    "unk",
    "sep",
    "pad",
    "cls",
    "mask",
)


class TeacherLogitsCacheError(StudentTrainingError):
    """The offline distillation cache violates its reproducibility contract."""


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _value_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _read_json_object(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise TeacherLogitsCacheError(f"{description} non trovato: {path}") from exc
    except json.JSONDecodeError as exc:
        raise TeacherLogitsCacheError(f"{description} non valido: {path}") from exc
    if not isinstance(value, dict):
        raise TeacherLogitsCacheError(f"{description} deve essere un oggetto JSON")
    return value


def _file_integrity(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    if not source.is_file() or source.is_symlink():
        raise TeacherLogitsCacheError(f"File locale regolare richiesto: {source}")
    return {
        "path": str(source),
        "bytes": int(source.stat().st_size),
        "sha256": sha256_file(source),
    }


def _read_stable_json_object(
    path: Path, description: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    before = _file_integrity(path)
    value = _read_json_object(path, description)
    after = _file_integrity(path)
    if before != after:
        raise TeacherLogitsCacheError(
            f"{description} mutato durante la lettura: {path}"
        )
    return value, before


def _software_versions() -> dict[str, Any]:
    result: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    for package in (
        "torch",
        "transformers",
        "tokenizers",
        "safetensors",
        "numpy",
    ):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    return result


def _model_file(name: str) -> bool:
    if name == "config.json" or name in _MODEL_WEIGHT_NAMES:
        return True
    return (
        (name.startswith("model-") and name.endswith(".safetensors"))
        or (name.startswith("pytorch_model-") and name.endswith(".bin"))
        or name in {"model.safetensors.index.json", "pytorch_model.bin.index.json"}
    )


def _model_weight_file(name: str) -> bool:
    return (
        name in _MODEL_WEIGHT_NAMES
        or (name.startswith("model-") and name.endswith(".safetensors"))
        or (name.startswith("pytorch_model-") and name.endswith(".bin"))
    )


def artifact_group_integrity(source: str | Path, kind: str) -> dict[str, Any]:
    """Hash the exact local files that define a model or tokenizer artifact.

    The group digest is the SHA256 of a canonical, path-relative list of each
    selected file's name, byte length and SHA256.  Consumers can independently
    reproduce the digest without trusting the cache manifest.
    """

    root = Path(source).expanduser().resolve()
    if not root.is_dir():
        raise TeacherLogitsCacheError(f"Artifact locale non trovato: {root}")
    if kind not in {"model", "tokenizer"}:
        raise TeacherLogitsCacheError(f"Gruppo artifact non supportato: {kind}")

    selected: list[Path] = []
    for item in sorted(root.iterdir(), key=lambda path: path.name):
        if item.is_symlink():
            raise TeacherLogitsCacheError(f"Symlink non ammesso nell'artifact: {item}")
        if not item.is_file():
            continue
        if (kind == "model" and _model_file(item.name)) or (
            kind == "tokenizer" and item.name in _TOKENIZER_FILENAMES
        ):
            selected.append(item)

    names = {path.name for path in selected}
    if kind == "model":
        if "config.json" not in names or not any(_model_weight_file(name) for name in names):
            raise TeacherLogitsCacheError(
                f"Artifact modello incompleto (config/pesi richiesti): {root}"
            )
    elif not ({"tokenizer.json", "vocab.json", "vocab.txt", "sentencepiece.bpe.model"} & names):
        raise TeacherLogitsCacheError(f"Artifact tokenizer incompleto: {root}")

    files = [
        {
            "path": path.name,
            "bytes": int(path.stat().st_size),
            "sha256": sha256_file(path),
        }
        for path in selected
    ]
    return {"path": str(root), "sha256": _value_sha256(files), "files": files}


def _added_token_record(token_id: int, value: Any) -> dict[str, Any]:
    content = getattr(value, "content", value)
    if not isinstance(content, str):
        content = str(content)
    return {
        "id": int(token_id),
        "content": content,
        "special": bool(getattr(value, "special", False)),
        "single_word": bool(getattr(value, "single_word", False)),
        "lstrip": bool(getattr(value, "lstrip", False)),
        "rstrip": bool(getattr(value, "rstrip", False)),
        "normalized": bool(getattr(value, "normalized", True)),
    }


def tokenizer_semantic_identity(tokenizer: Any) -> dict[str, Any]:
    """Describe the embedding-index contract of a loaded local tokenizer.

    Raw tokenizer files may be serialized differently while retaining the same
    vocabulary.  This identity therefore compares the canonical ID-to-token
    mapping, added-token metadata, special token IDs and declared sizes.
    """

    try:
        raw_vocab = tokenizer.get_vocab()
    except (AttributeError, TypeError, ValueError) as exc:
        raise TeacherLogitsCacheError("Tokenizer privo di get_vocab valido") from exc
    if not isinstance(raw_vocab, Mapping) or not raw_vocab:
        raise TeacherLogitsCacheError("Vocabolario tokenizer vuoto o non valido")

    id_to_token: dict[int, str] = {}
    for raw_token, raw_id in raw_vocab.items():
        if not isinstance(raw_token, str) or isinstance(raw_id, bool):
            raise TeacherLogitsCacheError("Mapping token->ID non canonico")
        try:
            token_id = int(raw_id)
        except (TypeError, ValueError) as exc:
            raise TeacherLogitsCacheError("ID tokenizer non intero") from exc
        if token_id < 0 or token_id in id_to_token:
            raise TeacherLogitsCacheError("ID tokenizer negativo o duplicato")
        id_to_token[token_id] = raw_token

    try:
        tokenizer_length = int(len(tokenizer))
        vocab_size = int(tokenizer.vocab_size)
    except (AttributeError, TypeError, ValueError) as exc:
        raise TeacherLogitsCacheError("Dimensioni tokenizer non disponibili") from exc
    expected_ids = set(range(tokenizer_length))
    if tokenizer_length < 1 or set(id_to_token) != expected_ids:
        raise TeacherLogitsCacheError(
            "Mapping tokenizer non contiguo o diverso da len(tokenizer)"
        )
    if vocab_size < 1 or vocab_size > tokenizer_length:
        raise TeacherLogitsCacheError("vocab_size tokenizer non valido")
    ordered_tokens = [id_to_token[index] for index in range(tokenizer_length)]

    try:
        raw_added_vocab = tokenizer.get_added_vocab()
    except (AttributeError, TypeError, ValueError) as exc:
        raise TeacherLogitsCacheError("Tokenizer privo di get_added_vocab valido") from exc
    if not isinstance(raw_added_vocab, Mapping):
        raise TeacherLogitsCacheError("Added vocabulary tokenizer non valido")
    added_vocab: list[dict[str, Any]] = []
    for token, raw_id in raw_added_vocab.items():
        if not isinstance(token, str) or isinstance(raw_id, bool):
            raise TeacherLogitsCacheError("Added token non valido")
        token_id = int(raw_id)
        if id_to_token.get(token_id) != token:
            raise TeacherLogitsCacheError("Added token divergente dal mapping principale")
        added_vocab.append({"id": token_id, "content": token})
    added_vocab.sort(key=lambda record: (record["id"], record["content"]))

    raw_decoder = getattr(tokenizer, "added_tokens_decoder", {})
    if raw_decoder is None:
        raw_decoder = {}
    if not isinstance(raw_decoder, Mapping):
        raise TeacherLogitsCacheError("added_tokens_decoder non valido")
    added_decoder = sorted(
        (_added_token_record(int(token_id), value) for token_id, value in raw_decoder.items()),
        key=lambda record: (record["id"], record["content"]),
    )
    for record in added_decoder:
        if id_to_token.get(int(record["id"])) != record["content"]:
            raise TeacherLogitsCacheError(
                "added_tokens_decoder divergente dal mapping principale"
            )

    special_tokens: dict[str, dict[str, Any] | None] = {}
    for role in _SPECIAL_TOKEN_ROLES:
        token = getattr(tokenizer, f"{role}_token", None)
        token_id = getattr(tokenizer, f"{role}_token_id", None)
        if token is None and token_id is None:
            special_tokens[role] = None
            continue
        if token is None or token_id is None or isinstance(token_id, bool):
            raise TeacherLogitsCacheError(f"Special token {role} incompleto")
        token_id = int(token_id)
        token_text = str(getattr(token, "content", token))
        if id_to_token.get(token_id) != token_text:
            raise TeacherLogitsCacheError(f"Special token {role} non mappa al suo ID")
        special_tokens[role] = {"token": token_text, "id": token_id}

    all_special_ids = [int(value) for value in getattr(tokenizer, "all_special_ids", [])]
    all_special_tokens = [
        str(getattr(value, "content", value))
        for value in getattr(tokenizer, "all_special_tokens", [])
    ]
    if len(all_special_ids) != len(all_special_tokens):
        raise TeacherLogitsCacheError("Liste all_special_tokens/ids disallineate")
    special_pairs = [
        {"id": token_id, "token": token}
        for token_id, token in zip(all_special_ids, all_special_tokens)
    ]
    for pair in special_pairs:
        if id_to_token.get(pair["id"]) != pair["token"]:
            raise TeacherLogitsCacheError("all_special_tokens non coerente col vocabolario")

    additional_ids = [
        int(value) for value in getattr(tokenizer, "additional_special_tokens_ids", [])
    ]
    additional_tokens = [
        str(getattr(value, "content", value))
        for value in getattr(tokenizer, "additional_special_tokens", [])
    ]
    if len(additional_ids) != len(additional_tokens):
        raise TeacherLogitsCacheError("Additional special token disallineati")
    additional_pairs = [
        {"id": token_id, "token": token}
        for token_id, token in zip(additional_ids, additional_tokens)
    ]
    for pair in additional_pairs:
        if id_to_token.get(pair["id"]) != pair["token"]:
            raise TeacherLogitsCacheError(
                "Additional special token non coerente col vocabolario"
            )

    payload = {
        "vocab_size": vocab_size,
        "tokenizer_length": tokenizer_length,
        "id_to_token": {
            "count": len(ordered_tokens),
            "sha256": _value_sha256(ordered_tokens),
        },
        "added_vocab": {
            "count": len(added_vocab),
            "sha256": _value_sha256(added_vocab),
        },
        "added_tokens_decoder": {
            "count": len(added_decoder),
            "sha256": _value_sha256(added_decoder),
        },
        "special_tokens": special_tokens,
        "all_special_tokens": special_pairs,
        "additional_special_tokens": additional_pairs,
        "model_input_names": [
            str(value) for value in getattr(tokenizer, "model_input_names", [])
        ],
        "padding_side": str(getattr(tokenizer, "padding_side", "right")),
    }
    return {**payload, "sha256": _value_sha256(payload)}


def assert_tokenizers_compatible(
    student_tokenizer: Any, teacher_tokenizer: Any
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fail closed unless teacher and student use an identical token-ID contract."""

    student_identity = tokenizer_semantic_identity(student_tokenizer)
    teacher_identity = tokenizer_semantic_identity(teacher_tokenizer)
    if student_identity != teacher_identity:
        differing = sorted(
            key
            for key in set(student_identity) | set(teacher_identity)
            if student_identity.get(key) != teacher_identity.get(key)
        )
        raise TeacherLogitsCacheError(
            "Tokenizer teacher/student incompatibili: " + ", ".join(differing)
        )
    return student_identity, teacher_identity


def _build_inputs_snapshot(
    *,
    run_manifest_path: Path,
    training_summary_path: Path,
    dataset_path: Path,
    student_path: Path,
    teacher_path: Path,
) -> dict[str, Any]:
    """Capture every mutable input used to derive the cache."""

    code_path = Path(__file__).resolve()
    student_utils_path = code_path.with_name("student_utils.py")
    return {
        "schema_version": 1,
        "run_manifest": _file_integrity(run_manifest_path),
        "training_summary": _file_integrity(training_summary_path),
        "dataset": _file_integrity(dataset_path),
        "student": {
            "model": artifact_group_integrity(student_path, "model"),
            "tokenizer": artifact_group_integrity(student_path, "tokenizer"),
        },
        "teacher": {
            "model": artifact_group_integrity(teacher_path, "model"),
            "tokenizer": artifact_group_integrity(teacher_path, "tokenizer"),
        },
        "code": {
            "generator": _file_integrity(code_path),
            "student_utils": _file_integrity(student_utils_path),
        },
        "software": _software_versions(),
    }


def _snapshot_changed_sections(
    expected: Mapping[str, Any], actual: Mapping[str, Any]
) -> list[str]:
    return sorted(
        key
        for key in set(expected) | set(actual)
        if expected.get(key) != actual.get(key)
    )


def _assert_inputs_unchanged(
    initial: Mapping[str, Any],
    *,
    run_manifest_path: Path,
    training_summary_path: Path,
    dataset_path: Path,
    student_path: Path,
    teacher_path: Path,
) -> None:
    current = _build_inputs_snapshot(
        run_manifest_path=run_manifest_path,
        training_summary_path=training_summary_path,
        dataset_path=dataset_path,
        student_path=student_path,
        teacher_path=teacher_path,
    )
    if current != initial:
        raise TeacherLogitsCacheError(
            "Input mutati durante la creazione della cache: "
            + ", ".join(_snapshot_changed_sections(initial, current))
        )


def window_alignment_sha256(
    *,
    input_ids: Sequence[int],
    attention_mask: Sequence[int],
    labels: Sequence[int],
    extra_inputs: Mapping[str, Sequence[int]] | None = None,
) -> str:
    """Return the canonical digest for one *padded* training window."""

    payload = {
        "input_ids": [int(value) for value in input_ids],
        "attention_mask": [int(value) for value in attention_mask],
        "labels": [int(value) for value in labels],
        "extra_inputs": {
            str(name): [int(value) for value in values]
            for name, values in sorted((extra_inputs or {}).items())
        },
    }
    lengths = {
        len(payload["input_ids"]),
        len(payload["attention_mask"]),
        len(payload["labels"]),
        *(len(values) for values in payload["extra_inputs"].values()),
    }
    if len(lengths) != 1:
        raise TeacherLogitsCacheError("Finestra disallineata durante il digest")
    return _value_sha256(payload)


def _iter_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    try:
        handle = path.open(encoding="utf-8")
    except FileNotFoundError as exc:
        raise TeacherLogitsCacheError(f"Dataset non trovato: {path}") from exc
    with handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise TeacherLogitsCacheError(f"{path}:{line_number}: riga vuota")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise TeacherLogitsCacheError(
                    f"{path}:{line_number}: JSON non valido"
                ) from exc
            if not isinstance(row, dict):
                raise TeacherLogitsCacheError(
                    f"{path}:{line_number}: il record deve essere un oggetto"
                )
            text = row.get("source_text")
            entities = row.get("entities", row.get("privacy_mask"))
            if not isinstance(text, str) or not isinstance(entities, list):
                raise TeacherLogitsCacheError(
                    f"{path}:{line_number}: richiesti source_text ed entities/privacy_mask"
                )
            yield line_number - 1, row


def _pad(values: Sequence[int], *, width: int, pad_value: int, side: str) -> list[int]:
    result = [int(value) for value in values]
    if len(result) > width:
        raise TeacherLogitsCacheError(
            f"Tokenizer ha prodotto {len(result)} token oltre max_length={width}"
        )
    padding = [int(pad_value)] * (width - len(result))
    if side == "right":
        return result + padding
    if side == "left":
        return padding + result
    raise TeacherLogitsCacheError(f"padding_side tokenizer non valido: {side!r}")


def tokenize_raw_span_windows(
    records: Sequence[tuple[int, Mapping[str, Any]]],
    *,
    tokenizer: Any,
    label2id: Mapping[str, int],
    max_length: int,
    stride: int,
    tokenize_batch_size: int = 128,
    dataset_sha256: str,
) -> list[dict[str, Any]]:
    """Tokenize raw spans with exactly the windowing call used by train_student."""

    if max_length < 16:
        raise TeacherLogitsCacheError("max_length deve essere almeno 16")
    if stride < 0 or stride >= max_length - 2:
        raise TeacherLogitsCacheError("stride non valido per max_length")
    if tokenize_batch_size < 1:
        raise TeacherLogitsCacheError("tokenize_batch_size deve essere positivo")
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if isinstance(pad_token_id, bool) or not isinstance(pad_token_id, int):
        raise TeacherLogitsCacheError("Il tokenizer deve definire pad_token_id intero")
    padding_side = str(getattr(tokenizer, "padding_side", "right"))

    windows: list[dict[str, Any]] = []
    per_record_counts: dict[int, int] = {}
    for start in range(0, len(records), tokenize_batch_size):
        batch = records[start : start + tokenize_batch_size]
        texts = [str(row["source_text"]) for _, row in batch]
        encoded = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            stride=stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
            padding=False,
        )
        if not isinstance(encoded, Mapping):
            raise TeacherLogitsCacheError("Output tokenizer non mappabile")
        try:
            sample_mapping = encoded["overflow_to_sample_mapping"]
            offsets = encoded["offset_mapping"]
            special_masks = encoded["special_tokens_mask"]
            input_ids_values = encoded["input_ids"]
            attention_values = encoded["attention_mask"]
        except KeyError as exc:
            raise TeacherLogitsCacheError(
                f"Output tokenizer privo di {exc.args[0]}"
            ) from exc
        count = len(sample_mapping)
        if not all(
            len(values) == count
            for values in (offsets, special_masks, input_ids_values, attention_values)
        ):
            raise TeacherLogitsCacheError("Output overflow tokenizer disallineato")
        extra_names = sorted(
            name
            for name in encoded
            if name
            not in {
                "overflow_to_sample_mapping",
                "offset_mapping",
                "special_tokens_mask",
                "input_ids",
                "attention_mask",
            }
        )
        unsupported = [name for name in extra_names if name != "token_type_ids"]
        if unsupported:
            raise TeacherLogitsCacheError(
                f"Input tokenizer inattesi, contratto non definito: {unsupported}"
            )

        for encoded_index, raw_sample_index in enumerate(sample_mapping):
            sample_index = int(raw_sample_index)
            if sample_index < 0 or sample_index >= len(batch):
                raise TeacherLogitsCacheError("overflow_to_sample_mapping fuori range")
            record_index, row = batch[sample_index]
            span_field = "entities" if "entities" in row else "privacy_mask"
            raw_input_ids = [int(value) for value in input_ids_values[encoded_index]]
            raw_attention = [int(value) for value in attention_values[encoded_index]]
            if len(raw_input_ids) != len(raw_attention):
                raise TeacherLogitsCacheError("input_ids e attention_mask disallineati")
            if any(value not in (0, 1) for value in raw_attention):
                raise TeacherLogitsCacheError("attention_mask deve contenere solo 0/1")
            raw_labels = align_char_spans(
                offsets[encoded_index],
                row[span_field],
                label2id,
                special_tokens_mask=special_masks[encoded_index],
            )
            padded_input_ids = _pad(
                raw_input_ids, width=max_length, pad_value=pad_token_id, side=padding_side
            )
            padded_attention = _pad(
                raw_attention, width=max_length, pad_value=0, side=padding_side
            )
            padded_labels = _pad(
                raw_labels, width=max_length, pad_value=IGNORE_INDEX, side=padding_side
            )
            extra_inputs = {
                name: _pad(
                    encoded[name][encoded_index],
                    width=max_length,
                    pad_value=int(getattr(tokenizer, "pad_token_type_id", 0) or 0),
                    side=padding_side,
                )
                for name in extra_names
            }
            record_window_index = per_record_counts.get(record_index, 0)
            per_record_counts[record_index] = record_window_index + 1
            record_sha256 = _value_sha256(row)
            window_id = _value_sha256(
                {
                    "dataset_sha256": dataset_sha256,
                    "record_index": record_index,
                    "record_sha256": record_sha256,
                    "record_window_index": record_window_index,
                    "max_length": max_length,
                    "stride": stride,
                }
            )
            windows.append(
                {
                    "record_index": int(record_index),
                    "record_window_index": int(record_window_index),
                    "record_sha256": record_sha256,
                    "window_id": window_id,
                    "length": int(sum(padded_attention)),
                    "input_ids": padded_input_ids,
                    "attention_mask": padded_attention,
                    "labels": padded_labels,
                    "extra_inputs": extra_inputs,
                    "alignment_sha256": window_alignment_sha256(
                        input_ids=padded_input_ids,
                        attention_mask=padded_attention,
                        labels=padded_labels,
                        extra_inputs=extra_inputs,
                    ),
                }
            )
    if not windows:
        raise TeacherLogitsCacheError("Il dataset non ha prodotto finestre")
    return windows


def _nested_value(value: Mapping[str, Any], path: Sequence[str]) -> Any:
    current: Any = value
    for component in path:
        if not isinstance(current, Mapping) or component not in current:
            raise TeacherLogitsCacheError(
                f"Manifest cache privo di {'.'.join(path)}"
            )
        current = current[component]
    return current


def _cache_identity_payload(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: manifest[key]
        for key in (
            "schema_version",
            "kind",
            "inputs_snapshot",
            "inputs_snapshot_sha256",
            "dataset",
            "student",
            "teacher",
            "label_contract",
            "preprocessing",
            "resolved_config_sha256",
            "counts",
            "window_order_sha256",
            "storage",
        )
    }


def _safe_cache_file(cache_dir: Path, relative_name: Any) -> Path:
    if not isinstance(relative_name, str):
        raise TeacherLogitsCacheError("Path storage non valido")
    candidate = Path(relative_name)
    if candidate.is_absolute() or len(candidate.parts) != 1 or candidate.name != relative_name:
        raise TeacherLogitsCacheError("Il path storage deve essere un singolo nome relativo")
    resolved = (cache_dir / candidate).resolve()
    if resolved.parent != cache_dir.resolve():
        raise TeacherLogitsCacheError("Path storage fuori dalla cache")
    return resolved


def load_and_validate_cache_manifest(
    cache_dir: str | Path,
    expected: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate bundle hashes and structural metadata without loading tensors."""

    root = Path(cache_dir).expanduser().resolve()
    if not root.is_dir():
        raise TeacherLogitsCacheError(f"Cache teacher non trovata: {root}")
    expected_files = {MANIFEST_FILENAME, MANIFEST_HASH_FILENAME, TENSOR_FILENAME}
    actual_files = {path.name for path in root.iterdir() if path.is_file()}
    if actual_files != expected_files or any(path.is_dir() for path in root.iterdir()):
        raise TeacherLogitsCacheError(
            f"Bundle cache inatteso: files={sorted(actual_files)}"
        )

    manifest_path = root / MANIFEST_FILENAME
    declared_manifest_hash = (root / MANIFEST_HASH_FILENAME).read_text(
        encoding="ascii"
    ).strip()
    if not _is_sha256(declared_manifest_hash) or sha256_file(manifest_path) != declared_manifest_hash:
        raise TeacherLogitsCacheError("Hash manifest cache non valido")
    manifest = _read_json_object(manifest_path, "Manifest cache teacher")
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("kind") != CACHE_KIND
        or manifest.get("status") != "complete"
    ):
        raise TeacherLogitsCacheError("Schema/status cache teacher non supportato")

    inputs_snapshot = manifest.get("inputs_snapshot")
    if (
        not isinstance(inputs_snapshot, Mapping)
        or manifest.get("inputs_snapshot_sha256") != _value_sha256(inputs_snapshot)
    ):
        raise TeacherLogitsCacheError("Snapshot input cache non valido")
    for owner in ("student", "teacher"):
        owner_manifest = manifest.get(owner)
        owner_snapshot = inputs_snapshot.get(owner)
        if not isinstance(owner_manifest, Mapping) or not isinstance(owner_snapshot, Mapping):
            raise TeacherLogitsCacheError(f"Provenance {owner} incompleta")
        for group in ("model", "tokenizer"):
            if owner_manifest.get(group) != owner_snapshot.get(group):
                raise TeacherLogitsCacheError(
                    f"Provenance {owner}.{group} diversa dallo snapshot"
                )
        semantic = owner_manifest.get("tokenizer_identity")
        if not isinstance(semantic, Mapping):
            raise TeacherLogitsCacheError(f"Identita' tokenizer {owner} mancante")
        semantic_payload = {key: value for key, value in semantic.items() if key != "sha256"}
        if semantic.get("sha256") != _value_sha256(semantic_payload):
            raise TeacherLogitsCacheError(f"Identita' tokenizer {owner} non valida")
    if manifest["student"]["tokenizer_identity"] != manifest["teacher"]["tokenizer_identity"]:
        raise TeacherLogitsCacheError("Tokenizer teacher/student incompatibili nel manifest")
    dataset_manifest = manifest.get("dataset")
    dataset_snapshot = inputs_snapshot.get("dataset")
    if not isinstance(dataset_manifest, Mapping) or not isinstance(dataset_snapshot, Mapping):
        raise TeacherLogitsCacheError("Provenance dataset incompleta")
    if any(
        dataset_manifest.get(key) != dataset_snapshot.get(key)
        for key in ("path", "bytes", "sha256")
    ):
        raise TeacherLogitsCacheError("Dataset diverso dallo snapshot input")

    storage = manifest.get("storage")
    if not isinstance(storage, Mapping) or not isinstance(storage.get("file"), Mapping):
        raise TeacherLogitsCacheError("Manifest privo del contratto storage")
    tensor_path = _safe_cache_file(root, storage["file"].get("path"))
    if tensor_path.name != TENSOR_FILENAME or not tensor_path.is_file():
        raise TeacherLogitsCacheError("File safetensors cache mancante")
    if (
        int(storage["file"].get("bytes", -1)) != tensor_path.stat().st_size
        or storage["file"].get("sha256") != sha256_file(tensor_path)
    ):
        raise TeacherLogitsCacheError("Integrita' safetensors cache fallita")

    counts = manifest.get("counts")
    preprocessing = manifest.get("preprocessing")
    label_contract = manifest.get("label_contract")
    tensor_specs = storage.get("tensors")
    if not all(isinstance(value, Mapping) for value in (counts, preprocessing, label_contract, tensor_specs)):
        raise TeacherLogitsCacheError("Manifest cache incompleto")
    num_windows = int(counts.get("windows", -1))
    max_length = int(preprocessing.get("max_length", -1))
    num_labels = int(label_contract.get("num_labels", -1))
    if num_windows < 1 or max_length < 1 or num_labels < 2:
        raise TeacherLogitsCacheError("Dimensioni cache non valide")
    ordered_labels = label_contract.get("labels")
    if (
        not isinstance(ordered_labels, list)
        or len(ordered_labels) != num_labels
        or not all(isinstance(label, str) and label for label in ordered_labels)
        or len(set(ordered_labels)) != num_labels
        or label_contract.get("sha256") != _value_sha256(ordered_labels)
    ):
        raise TeacherLogitsCacheError("Contratto label cache non valido")
    resolved_config = manifest.get("resolved_config")
    if (
        not isinstance(resolved_config, Mapping)
        or manifest.get("resolved_config_sha256") != _value_sha256(resolved_config)
    ):
        raise TeacherLogitsCacheError("Hash configurazione risolta non valido")
    required_shapes = {
        "teacher_logits": [num_windows, max_length, num_labels],
        "input_ids": [num_windows, max_length],
        "attention_mask": [num_windows, max_length],
        "labels": [num_windows, max_length],
    }
    for name, dtype in _REQUIRED_TENSORS.items():
        spec = tensor_specs.get(name)
        if not isinstance(spec, Mapping) or spec.get("dtype") != dtype:
            raise TeacherLogitsCacheError(f"Spec tensor {name} non valida")
        if spec.get("shape") != required_shapes[name]:
            raise TeacherLogitsCacheError(f"Shape tensor {name} non valida")
    for name, spec in tensor_specs.items():
        if name in _REQUIRED_TENSORS:
            continue
        if name != "token_type_ids" or not isinstance(spec, Mapping):
            raise TeacherLogitsCacheError(f"Tensor cache inatteso: {name}")
        if spec.get("dtype") != "I32" or spec.get("shape") != [num_windows, max_length]:
            raise TeacherLogitsCacheError("Spec token_type_ids non valida")

    windows = manifest.get("windows")
    if not isinstance(windows, list) or len(windows) != num_windows:
        raise TeacherLogitsCacheError("Indice finestre cache incompleto")
    order_payload: list[dict[str, str]] = []
    for index, window in enumerate(windows):
        if not isinstance(window, Mapping) or window.get("index") != index:
            raise TeacherLogitsCacheError("Ordine finestre cache non contiguo")
        if not _is_sha256(window.get("window_id")) or not _is_sha256(
            window.get("alignment_sha256")
        ):
            raise TeacherLogitsCacheError(f"Digest finestra {index} non valido")
        length = int(window.get("length", -1))
        if length < 1 or length > max_length:
            raise TeacherLogitsCacheError(f"Lunghezza finestra {index} non valida")
        order_payload.append(
            {
                "window_id": str(window["window_id"]),
                "alignment_sha256": str(window["alignment_sha256"]),
            }
        )
    if manifest.get("window_order_sha256") != _value_sha256(order_payload):
        raise TeacherLogitsCacheError("Hash ordine finestre non valido")
    identity = manifest.get("cache_identity_sha256")
    if not _is_sha256(identity) or identity != _value_sha256(
        _cache_identity_payload(manifest)
    ):
        raise TeacherLogitsCacheError("Identita' cache non valida")

    for key, expected_value in (expected or {}).items():
        path = _EXPECTED_MANIFEST_PATHS.get(key)
        if path is None:
            raise TeacherLogitsCacheError(f"Vincolo expected sconosciuto: {key}")
        actual_value = _nested_value(manifest, path)
        if actual_value != expected_value:
            raise TeacherLogitsCacheError(
                f"Cache incompatibile ({key}): actual={actual_value!r}, expected={expected_value!r}"
            )
    return manifest


def load_teacher_cache(
    cache_dir: str | Path,
    expected: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load and fully validate tensors, including every alignment digest."""

    manifest = load_and_validate_cache_manifest(cache_dir, expected)
    try:
        import torch
        from safetensors.torch import load_file
    except ImportError as exc:
        raise TeacherLogitsCacheError(
            "torch e safetensors sono richiesti per leggere la cache"
        ) from exc
    tensor_path = Path(cache_dir).expanduser().resolve() / TENSOR_FILENAME
    tensors = load_file(str(tensor_path), device="cpu")
    specs = manifest["storage"]["tensors"]
    if set(tensors) != set(specs):
        raise TeacherLogitsCacheError("Chiavi tensor diverse dal manifest")
    dtype_names = {
        torch.float16: "F16",
        torch.int32: "I32",
        torch.uint8: "U8",
        torch.int16: "I16",
    }
    for name, tensor in tensors.items():
        if list(tensor.shape) != specs[name]["shape"] or dtype_names.get(tensor.dtype) != specs[name]["dtype"]:
            raise TeacherLogitsCacheError(f"Tensor {name} diverso dal manifest")
    if not bool(torch.isfinite(tensors["teacher_logits"]).all().item()):
        raise TeacherLogitsCacheError("Logits teacher non finiti")
    if bool((tensors["input_ids"] < 0).any().item()):
        raise TeacherLogitsCacheError("input_ids negativi nella cache")
    attention = tensors["attention_mask"]
    if not bool(((attention == 0) | (attention == 1)).all().item()):
        raise TeacherLogitsCacheError("attention_mask non binaria nella cache")

    extra_names = sorted(set(tensors) - set(_REQUIRED_TENSORS))
    for index, window in enumerate(manifest["windows"]):
        extra = {
            name: tensors[name][index].tolist()
            for name in extra_names
        }
        digest = window_alignment_sha256(
            input_ids=tensors["input_ids"][index].tolist(),
            attention_mask=attention[index].tolist(),
            labels=tensors["labels"][index].tolist(),
            extra_inputs=extra,
        )
        if digest != window["alignment_sha256"]:
            raise TeacherLogitsCacheError(f"Allineamento finestra {index} non valido")
        if int(attention[index].sum().item()) != int(window["length"]):
            raise TeacherLogitsCacheError(f"Lunghezza finestra {index} non valida")
    return manifest, tensors


def _run_manifest_contract(
    run_manifest_path: Path,
    student_model: Path,
    dataset_override: Path | None,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    Path,
    Path,
    dict[str, Any],
    dict[str, dict[str, Any]],
]:
    run_root = run_manifest_path.parent.resolve()
    if run_manifest_path.resolve() != run_root / "run-manifest.json":
        raise TeacherLogitsCacheError(
            "--student-run-manifest deve essere esattamente <parent>/run-manifest.json"
        )
    if student_model.resolve() != run_root / "final":
        raise TeacherLogitsCacheError(
            "--student-model deve essere esattamente <parent>/final"
        )
    run_manifest, run_manifest_integrity = _read_stable_json_object(
        run_manifest_path, "Run manifest student"
    )
    if run_manifest.get("schema_version") != 1:
        raise TeacherLogitsCacheError("Schema run manifest student non supportato")
    training_summary_path = run_root / "training-summary.json"
    training_summary, training_summary_integrity = _read_stable_json_object(
        training_summary_path, "Training summary student"
    )
    if training_summary.get("status") != "complete":
        raise TeacherLogitsCacheError("Il parent student non e' completo")
    global_step = int(training_summary.get("global_step", -1))
    planned_steps = int(training_summary.get("planned_steps", -2))
    if global_step < 1 or global_step != planned_steps:
        raise TeacherLogitsCacheError(
            "Il parent student non ha completato tutti gli update"
        )
    if training_summary.get("input_format") != "raw-spans":
        raise TeacherLogitsCacheError("Il parent student non e' raw-spans")
    training = run_manifest.get("training")
    data = run_manifest.get("data")
    if not isinstance(training, Mapping) or not isinstance(data, Mapping):
        raise TeacherLogitsCacheError("Run manifest student incompleto")
    train_data = data.get("train")
    if not isinstance(train_data, Mapping):
        raise TeacherLogitsCacheError("Run manifest privo del dataset train")
    if training.get("input_format") not in {"raw-spans", "auto"}:
        raise TeacherLogitsCacheError("La distillation cache richiede raw-spans")
    if training.get("input_format") == "auto" and training.get("resolved_input_format") != "raw-spans":
        raise TeacherLogitsCacheError("Il formato risolto non e' raw-spans")

    dataset = (dataset_override or Path(str(train_data.get("path", "")))).expanduser().resolve()
    dataset_integrity = _file_integrity(dataset)
    if dataset_integrity["sha256"] != train_data.get("sha256"):
        raise TeacherLogitsCacheError("Hash dataset diverso dal run student")
    declared_path = Path(str(train_data.get("path", ""))).expanduser().resolve()
    if dataset_override is not None and dataset != declared_path:
        raise TeacherLogitsCacheError("--dataset deve coincidere col path del run student")

    required = ("max_length", "stride", "seed")
    if any(key not in training for key in required):
        raise TeacherLogitsCacheError("Run manifest privo del contratto windowing/seed")
    return (
        run_manifest,
        training_summary,
        training_summary_path,
        dataset,
        dict(training),
        {
            "run_manifest": run_manifest_integrity,
            "training_summary": training_summary_integrity,
            "dataset": dataset_integrity,
        },
    )


def create_teacher_logits_cache(
    *,
    student_model: str | Path,
    student_run_manifest: str | Path,
    teacher_model: str | Path,
    output_dir: str | Path,
    dataset: str | Path | None = None,
    device: str = "cpu",
    batch_size: int = 1,
    tokenize_batch_size: int = 128,
    tokenizer: Any | None = None,
    teacher_tokenizer: Any | None = None,
    model: Any | None = None,
    torch_module: Any | None = None,
    save_file_function: Any | None = None,
) -> dict[str, Any]:
    """Create a new cache directory; existing outputs are never overwritten."""

    student_path = Path(student_model).expanduser().resolve()
    run_manifest_path = Path(student_run_manifest).expanduser().resolve()
    teacher_path = Path(teacher_model).expanduser().resolve()
    output_path = Path(output_dir).expanduser().resolve()
    dataset_override = Path(dataset) if dataset is not None else None
    if output_path.exists():
        raise TeacherLogitsCacheError(f"Output gia' esistente, overwrite vietato: {output_path}")
    if batch_size < 1:
        raise TeacherLogitsCacheError("batch_size deve essere positivo")
    if device not in {"cpu", "mps"}:
        raise TeacherLogitsCacheError("device deve essere cpu o mps")

    (
        run_manifest,
        _training_summary,
        training_summary_path,
        dataset_path,
        training,
        contract_integrities,
    ) = _run_manifest_contract(run_manifest_path, student_path, dataset_override)
    initial_snapshot = _build_inputs_snapshot(
        run_manifest_path=run_manifest_path,
        training_summary_path=training_summary_path,
        dataset_path=dataset_path,
        student_path=student_path,
        teacher_path=teacher_path,
    )
    for key in ("run_manifest", "training_summary", "dataset"):
        if initial_snapshot[key] != contract_integrities[key]:
            raise TeacherLogitsCacheError(
                f"Input {key} mutato prima dello snapshot iniziale"
            )
    initial_snapshot_sha256 = _value_sha256(initial_snapshot)
    max_length = int(training["max_length"])
    stride = int(training["stride"])
    seed = int(training["seed"])
    if max_length < 16 or stride < 0 or stride >= max_length - 2:
        raise TeacherLogitsCacheError("Windowing student non valido")

    records = list(_iter_jsonl(dataset_path))
    declared_rows = int(run_manifest["data"]["train"].get("rows", -1))
    if declared_rows != len(records):
        raise TeacherLogitsCacheError(
            f"Numero righe dataset diverso dal run: {len(records)} != {declared_rows}"
        )
    train_limit = training.get("train_limit")
    if train_limit is not None and int(train_limit) < len(records):
        raise TeacherLogitsCacheError(
            "Run con train_limit inferiore alle righe: materializzare prima il subset selezionato"
        )

    student_config = student_path / "config.json"
    teacher_config = teacher_path / "config.json"
    student_label2id, student_id2label = load_label_contract(student_config)
    teacher_label2id, teacher_id2label = load_label_contract(teacher_config)
    if student_label2id != teacher_label2id or student_id2label != teacher_id2label:
        raise TeacherLogitsCacheError("Ordine label teacher/student diverso")
    ordered_labels = [student_id2label[index] for index in range(len(student_id2label))]

    student_model_integrity = initial_snapshot["student"]["model"]
    student_tokenizer_integrity = initial_snapshot["student"]["tokenizer"]
    teacher_model_integrity = initial_snapshot["teacher"]["model"]
    teacher_tokenizer_integrity = initial_snapshot["teacher"]["tokenizer"]
    dataset_integrity = {
        **initial_snapshot["dataset"],
        "rows": len(records),
    }

    try:
        if torch_module is None:
            import torch as torch_module
        if save_file_function is None:
            from safetensors.torch import save_file as save_file_function
        if tokenizer is None or teacher_tokenizer is None or model is None:
            from transformers import AutoModelForTokenClassification, AutoTokenizer
    except ImportError as exc:
        raise TeacherLogitsCacheError(
            "torch, transformers e safetensors sono richiesti per creare la cache"
        ) from exc

    torch = torch_module
    if device == "mps" and not bool(torch.backends.mps.is_available()):
        raise TeacherLogitsCacheError("MPS non disponibile")
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed % (2**32))
    except ImportError:
        pass
    torch.manual_seed(seed)
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(
            str(student_path),
            local_files_only=True,
            trust_remote_code=False,
            fix_mistral_regex=TOKENIZER_FIX_MISTRAL_REGEX,
        )
    if teacher_tokenizer is None:
        teacher_tokenizer = AutoTokenizer.from_pretrained(
            str(teacher_path),
            local_files_only=True,
            trust_remote_code=False,
            fix_mistral_regex=TOKENIZER_FIX_MISTRAL_REGEX,
        )
    student_tokenizer_identity, teacher_tokenizer_identity = (
        assert_tokenizers_compatible(tokenizer, teacher_tokenizer)
    )
    # The teacher tokenizer is needed only for the compatibility proof.  Keep
    # the student tokenizer, which defines the cached windows, and release the
    # duplicate before loading the teacher model.
    teacher_tokenizer = None
    windows = tokenize_raw_span_windows(
        records,
        tokenizer=tokenizer,
        label2id=student_label2id,
        max_length=max_length,
        stride=stride,
        tokenize_batch_size=tokenize_batch_size,
        dataset_sha256=dataset_integrity["sha256"],
    )

    if model is None:
        model = AutoModelForTokenClassification.from_pretrained(
            str(teacher_path),
            local_files_only=True,
            trust_remote_code=False,
            dtype=torch.float32,
        )
    model = model.to(device)
    model.float()
    model.eval()
    model_num_labels = int(getattr(getattr(model, "config", None), "num_labels", -1))
    if model_num_labels != len(ordered_labels):
        raise TeacherLogitsCacheError(
            f"Head teacher incompatibile: {model_num_labels} != {len(ordered_labels)}"
        )

    input_ids = torch.tensor(
        [window["input_ids"] for window in windows], dtype=torch.int32
    )
    attention_mask = torch.tensor(
        [window["attention_mask"] for window in windows], dtype=torch.uint8
    )
    labels = torch.tensor([window["labels"] for window in windows], dtype=torch.int16)
    extra_names = sorted(windows[0]["extra_inputs"])
    if any(sorted(window["extra_inputs"]) != extra_names for window in windows):
        raise TeacherLogitsCacheError("Input extra tokenizer non uniformi")
    extra_tensors = {
        name: torch.tensor(
            [window["extra_inputs"][name] for window in windows], dtype=torch.int32
        )
        for name in extra_names
    }

    logit_batches: list[Any] = []
    with torch.inference_mode():
        for start in range(0, len(windows), batch_size):
            end = min(start + batch_size, len(windows))
            model_inputs = {
                "input_ids": input_ids[start:end].to(device=device, dtype=torch.long),
                "attention_mask": attention_mask[start:end].to(
                    device=device, dtype=torch.long
                ),
                **{
                    name: tensor[start:end].to(device=device, dtype=torch.long)
                    for name, tensor in extra_tensors.items()
                },
            }
            output = model(**model_inputs)
            logits = output.logits if hasattr(output, "logits") else output[0]
            expected_shape = (end - start, max_length, len(ordered_labels))
            if tuple(logits.shape) != expected_shape:
                raise TeacherLogitsCacheError(
                    f"Shape logits teacher inattesa: {tuple(logits.shape)} != {expected_shape}"
                )
            logits_fp32 = logits.detach().to(device="cpu", dtype=torch.float32)
            if not bool(torch.isfinite(logits_fp32).all().item()):
                raise TeacherLogitsCacheError("Teacher ha prodotto logits non finiti")
            logits_fp16 = logits_fp32.to(dtype=torch.float16)
            if not bool(torch.isfinite(logits_fp16).all().item()):
                raise TeacherLogitsCacheError("Conversione FP16 dei logits non finita")
            logit_batches.append(logits_fp16)
    teacher_logits = torch.cat(logit_batches, dim=0).contiguous()
    _assert_inputs_unchanged(
        initial_snapshot,
        run_manifest_path=run_manifest_path,
        training_summary_path=training_summary_path,
        dataset_path=dataset_path,
        student_path=student_path,
        teacher_path=teacher_path,
    )

    tensors = {
        "teacher_logits": teacher_logits,
        "input_ids": input_ids.contiguous(),
        "attention_mask": attention_mask.contiguous(),
        "labels": labels.contiguous(),
        **{name: tensor.contiguous() for name, tensor in extra_tensors.items()},
    }
    tensor_specs = {
        name: {
            "dtype": {
                torch.float16: "F16",
                torch.int32: "I32",
                torch.uint8: "U8",
                torch.int16: "I16",
            }[tensor.dtype],
            "shape": list(tensor.shape),
        }
        for name, tensor in sorted(tensors.items())
    }
    public_windows = [
        {
            "index": index,
            "record_index": int(window["record_index"]),
            "record_window_index": int(window["record_window_index"]),
            "record_sha256": str(window["record_sha256"]),
            "window_id": str(window["window_id"]),
            "length": int(window["length"]),
            "alignment_sha256": str(window["alignment_sha256"]),
        }
        for index, window in enumerate(windows)
    ]
    window_order_sha256 = _value_sha256(
        [
            {
                "window_id": window["window_id"],
                "alignment_sha256": window["alignment_sha256"],
            }
            for window in public_windows
        ]
    )
    resolved_config = {
        "student_model_sha256": student_model_integrity["sha256"],
        "student_tokenizer_sha256": student_tokenizer_integrity["sha256"],
        "student_tokenizer_identity_sha256": student_tokenizer_identity["sha256"],
        "teacher_model_sha256": teacher_model_integrity["sha256"],
        "teacher_tokenizer_sha256": teacher_tokenizer_integrity["sha256"],
        "teacher_tokenizer_identity_sha256": teacher_tokenizer_identity["sha256"],
        "dataset_sha256": dataset_integrity["sha256"],
        "run_manifest_sha256": initial_snapshot["run_manifest"]["sha256"],
        "training_summary_sha256": initial_snapshot["training_summary"]["sha256"],
        "inputs_snapshot_sha256": initial_snapshot_sha256,
        "label_contract_sha256": _value_sha256(ordered_labels),
        "input_format": "raw-spans",
        "max_length": max_length,
        "stride": stride,
        "seed": seed,
        "batch_size": batch_size,
        "tokenize_batch_size": tokenize_batch_size,
        "device": device,
        "teacher_compute_dtype": "FP32",
        "storage_dtype": "FP16",
        "tokenizer_load_policy": tokenizer_load_policy(),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    staging_path = Path(
        tempfile.mkdtemp(
            prefix=f".{output_path.name}.tmp-", dir=str(output_path.parent)
        )
    )
    try:
        tensor_path = staging_path / TENSOR_FILENAME
        save_file_function(
            tensors,
            str(tensor_path),
            metadata={"kind": CACHE_KIND, "schema_version": str(SCHEMA_VERSION)},
        )
        tensor_file = {
            "path": TENSOR_FILENAME,
            "bytes": int(tensor_path.stat().st_size),
            "sha256": sha256_file(tensor_path),
        }
        manifest: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "kind": CACHE_KIND,
            "status": "complete",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "inputs_snapshot": initial_snapshot,
            "inputs_snapshot_sha256": initial_snapshot_sha256,
            "dataset": dataset_integrity,
            "student": {
                "run_manifest": initial_snapshot["run_manifest"],
                "training_summary": initial_snapshot["training_summary"],
                "model": student_model_integrity,
                "tokenizer": student_tokenizer_integrity,
                "tokenizer_identity": student_tokenizer_identity,
            },
            "teacher": {
                "model": teacher_model_integrity,
                "tokenizer": teacher_tokenizer_integrity,
                "tokenizer_identity": teacher_tokenizer_identity,
                "compute_dtype": "FP32",
                "device": device,
            },
            "label_contract": {
                "num_labels": len(ordered_labels),
                "labels": ordered_labels,
                "sha256": _value_sha256(ordered_labels),
            },
            "preprocessing": {
                "input_format": "raw-spans",
                "max_length": max_length,
                "stride": stride,
                "train_limit": train_limit,
                "seed": seed,
                "tokenize_batch_size": tokenize_batch_size,
                "padding": "fixed-max-length",
                "padding_side": str(getattr(tokenizer, "padding_side", "right")),
                "ignore_index": IGNORE_INDEX,
                "tokenizer_load_policy": tokenizer_load_policy(),
            },
            "resolved_config": resolved_config,
            "resolved_config_sha256": _value_sha256(resolved_config),
            "counts": {"records": len(records), "windows": len(windows)},
            "window_order_sha256": window_order_sha256,
            "windows": public_windows,
            "storage": {"file": tensor_file, "tensors": tensor_specs},
        }
        manifest["cache_identity_sha256"] = _value_sha256(
            _cache_identity_payload(manifest)
        )
        manifest_path = staging_path / MANIFEST_FILENAME
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        (staging_path / MANIFEST_HASH_FILENAME).write_text(
            sha256_file(manifest_path) + "\n", encoding="ascii"
        )
        load_and_validate_cache_manifest(staging_path)
        _assert_inputs_unchanged(
            initial_snapshot,
            run_manifest_path=run_manifest_path,
            training_summary_path=training_summary_path,
            dataset_path=dataset_path,
            student_path=student_path,
            teacher_path=teacher_path,
        )
        if output_path.exists():
            raise TeacherLogitsCacheError(
                f"Output comparso durante il run, overwrite vietato: {output_path}"
            )
        staging_path.rename(output_path)
    except BaseException:
        if staging_path.exists():
            shutil.rmtree(staging_path)
        raise
    return manifest


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student-model", required=True)
    parser.add_argument("--student-run-manifest", required=True)
    parser.add_argument("--teacher-model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--dataset",
        help="opzionale, deve coincidere esattamente col train dataset del run student",
    )
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--tokenize-batch-size", type=int, default=128)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    manifest = create_teacher_logits_cache(
        student_model=args.student_model,
        student_run_manifest=args.student_run_manifest,
        teacher_model=args.teacher_model,
        output_dir=args.output_dir,
        dataset=args.dataset,
        device=args.device,
        batch_size=args.batch_size,
        tokenize_batch_size=args.tokenize_batch_size,
    )
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "output_dir": str(Path(args.output_dir).expanduser().resolve()),
                "cache_identity_sha256": manifest["cache_identity_sha256"],
                "records": manifest["counts"]["records"],
                "windows": manifest["counts"]["windows"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except TeacherLogitsCacheError as exc:
        print(f"ERRORE: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
