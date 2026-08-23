"""Materializza un pilot raw-span deterministico da shard Parquet pinned.

La selezione prende i record con score SHA-256 piu' basso. Non dipende
dall'ordine dello shard o dal PRNG di una specifica versione di Datasets.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from src.training.student_utils import StudentTrainingError, sha256_file


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAIN = ROOT / "artifacts/training/sources/clean/data/train-00000-of-00016.parquet"
DEFAULT_VALIDATION = (
    ROOT / "artifacts/training/sources/clean/data/validation-00000-of-00001.parquet"
)
DEFAULT_OUTPUT = ROOT / "artifacts/training/data"


def _canonical_record(raw: dict[str, Any], source_row: int) -> tuple[str, dict[str, Any]]:
    text = raw.get("source_text")
    entities = raw.get("entities")
    if not isinstance(text, str) or not text:
        raise StudentTrainingError(f"Riga {source_row}: source_text mancante")
    if not isinstance(entities, list):
        raise StudentTrainingError(f"Riga {source_row}: entities mancanti")
    canonical_entities = []
    for index, entity in enumerate(entities):
        try:
            start = int(entity["start"])
            end = int(entity["end"])
            label = str(entity["label"])
        except (KeyError, TypeError, ValueError) as exc:
            raise StudentTrainingError(
                f"Riga {source_row}, entita' {index}: schema non valido"
            ) from exc
        if start < 0 or end <= start or end > len(text):
            raise StudentTrainingError(
                f"Riga {source_row}, entita' {index}: offset {start}:{end} fuori testo"
            )
        canonical_entities.append({"start": start, "end": end, "label": label})
    canonical_entities.sort(key=lambda item: (item["start"], item["end"], item["label"]))
    identity_payload = {
        "source_text": text,
        "entities": canonical_entities,
        "language": str(raw.get("language") or ""),
        "template_id": raw.get("template_id"),
    }
    canonical = json.dumps(
        identity_payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    content_hash = hashlib.sha256(canonical).hexdigest()
    return content_hash, {
        "record_id": content_hash,
        "source": "clean",
        "source_row": source_row,
        **identity_payload,
    }


def select_records(
    parquet_path: str | Path,
    *,
    count: int,
    salt: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if count < 1:
        raise StudentTrainingError("count deve essere positivo")
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise StudentTrainingError("pyarrow e' richiesto per costruire il pilot") from exc

    source = Path(parquet_path).resolve()
    parquet_file = parquet.ParquetFile(source)
    required = {"source_text", "entities", "language", "template_id"}
    missing = required - set(parquet_file.schema_arrow.names)
    if missing:
        raise StudentTrainingError(f"Colonne Parquet mancanti: {sorted(missing)}")

    heap: list[tuple[int, str, dict[str, Any]]] = []
    content_hashes: set[str] = set()
    scanned = 0
    duplicates = 0
    columns = sorted(required)
    for batch in parquet_file.iter_batches(batch_size=1024, columns=columns):
        for raw in batch.to_pylist():
            content_hash, record = _canonical_record(raw, scanned)
            scanned += 1
            if content_hash in content_hashes:
                duplicates += 1
                continue
            content_hashes.add(content_hash)
            score = int.from_bytes(
                hashlib.sha256(f"{salt}:{content_hash}".encode("ascii")).digest(), "big"
            )
            item = (-score, content_hash, record)
            if len(heap) < count:
                heapq.heappush(heap, item)
            elif score < -heap[0][0]:
                heapq.heapreplace(heap, item)

    if len(heap) != count:
        raise StudentTrainingError(
            f"Richiesti {count} record unici, disponibili soltanto {len(heap)}"
        )
    selected = [item[2] for item in heap]
    selected.sort(
        key=lambda record: hashlib.sha256(
            f"{salt}:{record['record_id']}".encode("ascii")
        ).digest()
    )
    return selected, {
        "path": str(source),
        "bytes": source.stat().st_size,
        "sha256": sha256_file(source),
        "rows_scanned": scanned,
        "duplicate_content_rows": duplicates,
        "selected_rows": len(selected),
        "salt": salt,
    }


def _jsonl_bytes(records: list[dict[str, Any]]) -> bytes:
    return b"".join(
        (
            json.dumps(record, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
            + "\n"
        ).encode("utf-8")
        for record in records
    )


def _write_exact(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise StudentTrainingError(f"Output esistente ma diverso: {path}")
        return
    descriptor, temporary_name = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _selection_stats(records: list[dict[str, Any]]) -> dict[str, Any]:
    labels = Counter(
        entity["label"] for record in records for entity in record.get("entities", [])
    )
    templates = {
        record.get("template_id")
        for record in records
        if record.get("template_id") is not None
    }
    return {
        "rows": len(records),
        "entities": sum(labels.values()),
        "labels": dict(sorted(labels.items())),
        "templates": len(templates),
        "mean_chars": sum(len(record["source_text"]) for record in records) / len(records),
    }


def _skeleton_hash(record: dict[str, Any]) -> str:
    text = record["source_text"]
    for entity in sorted(record["entities"], key=lambda item: item["start"], reverse=True):
        text = (
            text[: entity["start"]]
            + f"<{entity['label']}>"
            + text[entity["end"] :]
        )
    canonical = " ".join(text.split())
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-parquet", default=str(DEFAULT_TRAIN))
    parser.add_argument("--validation-parquet", default=str(DEFAULT_VALIDATION))
    parser.add_argument("--train-count", type=int, default=1024)
    parser.add_argument("--validation-count", type=int, default=512)
    parser.add_argument("--salt", default="rizzo-pii-mmbert-small-pilot-v1-20260822")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args(argv)

    output = Path(args.output_dir).resolve()
    train, train_source = select_records(
        args.train_parquet, count=args.train_count, salt=args.salt + ":train"
    )
    validation, validation_source = select_records(
        args.validation_parquet,
        count=args.validation_count,
        salt=args.salt + ":validation",
    )
    overlap = {record["record_id"] for record in train} & {
        record["record_id"] for record in validation
    }
    if overlap:
        raise StudentTrainingError(f"Leakage train/validation: {len(overlap)} record")
    skeleton_overlap = {_skeleton_hash(record) for record in train} & {
        _skeleton_hash(record) for record in validation
    }
    if skeleton_overlap:
        raise StudentTrainingError(
            f"Leakage skeleton train/validation: {len(skeleton_overlap)} template"
        )
    train_template_ids = {
        record.get("template_id")
        for record in train
        if record.get("template_id") is not None
    }
    validation_template_ids = {
        record.get("template_id")
        for record in validation
        if record.get("template_id") is not None
    }

    train_path = output / f"clean-train-{len(train)}.jsonl"
    validation_path = output / f"clean-validation-{len(validation)}.jsonl"
    _write_exact(train_path, _jsonl_bytes(train))
    _write_exact(validation_path, _jsonl_bytes(validation))
    manifest = {
        "schema_version": 1,
        "selection": "lowest salted SHA-256 of canonical content",
        "train_source": train_source,
        "validation_source": validation_source,
        "train": {
            "path": str(train_path),
            "bytes": train_path.stat().st_size,
            "sha256": sha256_file(train_path),
            **_selection_stats(train),
        },
        "validation": {
            "path": str(validation_path),
            "bytes": validation_path.stat().st_size,
            "sha256": sha256_file(validation_path),
            **_selection_stats(validation),
        },
        "content_overlap": 0,
        "skeleton_overlap": 0,
        "template_id_overlap": len(train_template_ids & validation_template_ids),
        "template_id_note": (
            "template_id is not globally unique across generator families; "
            "the canonical entity-replaced skeleton is the enforced leakage key"
        ),
    }
    manifest_path = output / "clean-pilot-manifest.json"
    _write_exact(
        manifest_path,
        (json.dumps(manifest, indent=2, ensure_ascii=False) + "\n").encode("utf-8"),
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except StudentTrainingError as exc:
        print(f"ERRORE: {exc}")
        raise SystemExit(2) from exc
