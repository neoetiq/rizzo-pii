"""Quality-first raw character-span evaluation for PII teacher/student models.

The evaluator intentionally keeps inference and comparison separate:

* ``evaluate`` runs one local PyTorch token-classification checkpoint and writes
  a compact predictions JSONL plus a provenance-rich report;
* ``compare`` reads two saved reports/prediction files and computes paired
  teacher/student disagreement without loading either model again.

Raw text is tokenized with offsets, overflow windows and stride.  Logits for an
identical subword offset are averaged across windows in a stable order before a
single production-like BIO aggregation is performed.  This avoids both the
"first subword only" shortcut and duplicate/partial entities at window seams.
No source text is written to the prediction artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import sys
import tempfile
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from src.quantization.metrics import (
    DROP_TYPES,
    TAG_MAP,
    compare_span_predictions,
    compute_span_metrics,
    simple_char_spans,
)
from src.training.student_utils import (
    TOKENIZER_FIX_MISTRAL_REGEX,
    sha256_file,
    tokenizer_load_policy,
)


SCHEMA_VERSION = 1
EVALUATION_SCHEMA = "rizzo-pii.student-evaluation"
COMPARISON_SCHEMA = "rizzo-pii.student-comparison"
PREDICTION_SCHEMA = "rizzo-pii.raw-char-spans"
PRIVACY_RESCORE_SCHEMA = "rizzo-pii.privacy-utility-rescore"
ID_DOC_PREFIX_TOLERANT_METRIC_NAME = (
    "id_doc_payload_exact_optional_number_prefix_v1"
)
ID_DOC_PREFIX_TOLERANT_POLICY: dict[str, Any] = {
    "scope": "ID_DOC_only",
    "base_match": "exact_entity_type_and_canonical_payload_character_offsets",
    "ignored_leading_prefixes_case_insensitive": ["n.", "n°", "nr.", "numero"],
    "ignored_whitespace": "outer_span_and_between_prefix_and_payload",
    "prefix_position": "start_of_ID_DOC_span_only",
    "empty_payload": "prefix_is_not_removed",
    "other_entity_types": "exact_character_span",
    "wrong_type": "always_error",
    "partial_payload_or_extra_non_whitespace_text": "always_error",
}
PRODUCTION_PRIVACY_UTILITY_METRIC_NAME = (
    "production_privacy_utility_unicode_alnum_coverage_v1"
)
PRODUCTION_PRIVACY_UTILITY_POLICY: dict[str, Any] = {
    "schema_version": 1,
    "gold_payload": (
        "teacher-taxonomy character spans; ID_DOC harmless leading number prefixes "
        "are removed with id_doc_payload_exact_optional_number_prefix_v1"
    ),
    "sensitive_character": "Unicode code point where Python str.isalnum() is true",
    "prediction_scope": "all normalized model PII spans; character positions are unioned",
    "type_agnostic_coverage": "gold payload characters covered by any predicted PII type",
    "type_correct_coverage": "gold payload characters covered by the same canonical PII type",
    "wrong_type": "privacy-covered but not type-correct",
    "partial_payload": "uncovered payload characters are leakage",
    "complete_entity": "all sensitive payload characters covered; zero-alnum entities excluded",
    "complete_document": "all sensitive payload characters covered; zero-PII documents excluded",
    "collateral_masking": (
        "predicted Unicode alphanumeric character positions outside every canonical "
        "gold PII payload"
    ),
    "mask_precision": "predicted positions on gold payload / all predicted alnum positions",
    "collateral_masking_ratio": (
        "collateral positions / all document alnum positions outside canonical gold payload"
    ),
    "per_tag_prediction_counts": (
        "independent per predicted type; global counts deduplicate overlapping predictions"
    ),
    "zero_denominator": None,
    "persistence": "aggregates_and_policy_only_no_text_or_record_ids",
}

_ID_DOC_OPTIONAL_NUMBER_PREFIX = re.compile(
    r"(?:n\s*(?:\.|°)|nr\s*\.|numero(?=\s))\s*",
    flags=re.IGNORECASE,
)

_MODEL_EXACT_FILENAMES = {
    "config.json",
    "model.safetensors",
    "model.safetensors.index.json",
    "pytorch_model.bin",
    "pytorch_model.bin.index.json",
}
_TOKENIZER_EXACT_FILENAMES = {
    "added_tokens.json",
    "merges.txt",
    "sentencepiece.bpe.model",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "vocab.txt",
}

Span = tuple[str, int, int]


class StudentEvaluationError(ValueError):
    """Raised when evaluation inputs or saved artifacts violate the contract."""


@dataclass(frozen=True)
class RawSpanRecord:
    record_id: str
    text: str
    gold_spans: frozenset[Span]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _value_digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _sorted_spans(values: Iterable[Span]) -> list[list[str | int]]:
    return [
        [entity_type, start, end]
        for entity_type, start, end in sorted(values, key=lambda span: (span[1], span[2], span[0]))
    ]


def _parse_spans(value: Any, *, context: str) -> frozenset[Span]:
    if not isinstance(value, list):
        raise StudentEvaluationError(f"{context}: attesa una lista di span")
    result: set[Span] = set()
    for index, raw_span in enumerate(value):
        if not isinstance(raw_span, (list, tuple)) or len(raw_span) != 3:
            raise StudentEvaluationError(f"{context}: span {index} non valido")
        entity_type, raw_start, raw_end = raw_span
        if not isinstance(entity_type, str) or not entity_type:
            raise StudentEvaluationError(f"{context}: tipo span {index} non valido")
        if (
            isinstance(raw_start, bool)
            or isinstance(raw_end, bool)
            or not isinstance(raw_start, int)
            or not isinstance(raw_end, int)
            or raw_start < 0
            or raw_end <= raw_start
        ):
            raise StudentEvaluationError(f"{context}: offset span {index} non validi")
        result.add((entity_type, raw_start, raw_end))
    return frozenset(result)


def _normalized_entity_type(raw_label: str) -> str | None:
    label = raw_label[2:] if raw_label.startswith(("B-", "I-")) else raw_label
    normalized = TAG_MAP.get(label, label)
    return None if normalized in DROP_TYPES else normalized


def _merge_normalized_gold_spans(text: str, spans: Iterable[Span]) -> frozenset[Span]:
    """Apply the teacher taxonomy's consecutive-entity fusion to raw spans.

    ``GIVENNAME`` followed by ``SURNAME`` becomes two ``FULLNAME`` annotations
    after the type map, while the model contract represents it as one BIO
    entity.  If the annotations are separated only by whitespace, fuse them and
    retain the exact outer character boundaries.  Other gaps remain explicit.
    """

    ordered = sorted(set(spans), key=lambda span: (span[1], span[2], span[0]))
    merged: list[Span] = []
    for entity_type, start, end in ordered:
        if merged and start < merged[-1][2]:
            raise StudentEvaluationError(
                "Entita' gold sovrapposte non supportate: "
                f"{merged[-1][1]}:{merged[-1][2]} e {start}:{end}"
            )
        if (
            merged
            and merged[-1][0] == entity_type
            and (start == merged[-1][2] or text[merged[-1][2] : start].isspace())
        ):
            previous_type, previous_start, _ = merged[-1]
            merged[-1] = (previous_type, previous_start, end)
        else:
            merged.append((entity_type, start, end))
    return frozenset(merged)


def _raw_record(
    row: Mapping[str, Any], *, row_index: int, source: Path
) -> RawSpanRecord:
    text = row.get("source_text")
    raw_entities = row.get("entities", row.get("privacy_mask"))
    if not isinstance(text, str) or not isinstance(raw_entities, list):
        raise StudentEvaluationError(
            f"{source}: record {row_index}: servono source_text ed entities/privacy_mask"
        )
    if not text:
        raise StudentEvaluationError(f"{source}: record {row_index}: source_text vuoto")

    raw_id = row.get("record_id", row.get("id", row_index))
    if isinstance(raw_id, (dict, list)) or raw_id is None:
        raise StudentEvaluationError(f"{source}: record {row_index}: id non valido")
    record_id = str(raw_id)

    spans: set[Span] = set()
    for entity_index, entity in enumerate(raw_entities):
        if not isinstance(entity, Mapping):
            raise StudentEvaluationError(
                f"{source}: record {row_index}: entita' {entity_index} non valida"
            )
        try:
            raw_start = entity["start"]
            raw_end = entity["end"]
            raw_label = entity["label"]
        except KeyError as exc:
            raise StudentEvaluationError(
                f"{source}: record {row_index}: entita' {entity_index} incompleta"
            ) from exc
        if (
            isinstance(raw_start, bool)
            or isinstance(raw_end, bool)
            or not isinstance(raw_start, int)
            or not isinstance(raw_end, int)
            or raw_start < 0
            or raw_end <= raw_start
            or raw_end > len(text)
            or not isinstance(raw_label, str)
        ):
            raise StudentEvaluationError(
                f"{source}: record {row_index}: entita' {entity_index} non valida"
            )
        entity_type = _normalized_entity_type(raw_label)
        if entity_type is not None:
            spans.add((entity_type, raw_start, raw_end))
    return RawSpanRecord(record_id, text, _merge_normalized_gold_spans(text, spans))


def _jsonl_rows(path: Path) -> Iterable[Mapping[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise StudentEvaluationError(
                    f"{path}:{line_number}: JSON non valido"
                ) from exc
            if not isinstance(row, Mapping):
                raise StudentEvaluationError(f"{path}:{line_number}: record non oggetto")
            yield row


def _parquet_rows(path: Path) -> Iterable[Mapping[str, Any]]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise StudentEvaluationError(
            "pyarrow e' richiesto per valutare un dataset Parquet"
        ) from exc
    parquet_file = parquet.ParquetFile(path)
    names = set(parquet_file.schema_arrow.names)
    span_field = "entities" if "entities" in names else "privacy_mask"
    required = {"source_text", span_field}
    if span_field not in names or not required <= names:
        raise StudentEvaluationError(
            f"{path}: Parquet raw-span privo di source_text/entities/privacy_mask"
        )
    optional = [name for name in ("record_id", "id") if name in names]
    columns = ["source_text", span_field, *optional]
    for batch in parquet_file.iter_batches(batch_size=512, columns=columns):
        yield from batch.to_pylist()


def load_raw_records(path: str | Path, limit: int | None = None) -> list[RawSpanRecord]:
    """Load raw text + character spans without importing ML dependencies."""

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise StudentEvaluationError(f"Dataset non trovato: {source}")
    if limit is not None and limit < 1:
        raise StudentEvaluationError("limit deve essere positivo")
    rows = (
        _parquet_rows(source)
        if source.suffix.lower() in {".parquet", ".pq"}
        else _jsonl_rows(source)
    )
    records: list[RawSpanRecord] = []
    seen_ids: set[str] = set()
    for row_index, row in enumerate(rows):
        record = _raw_record(row, row_index=row_index, source=source)
        if record.record_id in seen_ids:
            raise StudentEvaluationError(
                f"{source}: record_id duplicato: {record.record_id!r}"
            )
        seen_ids.add(record.record_id)
        records.append(record)
        if limit is not None and len(records) >= limit:
            break
    if not records:
        raise StudentEvaluationError(f"Dataset raw-span vuoto: {source}")
    return records


def _selection_identity(records: Sequence[RawSpanRecord]) -> str:
    rows = [
        {
            "id": record.record_id,
            "text_sha256": hashlib.sha256(record.text.encode("utf-8")).hexdigest(),
            "gold_spans": _sorted_spans(record.gold_spans),
        }
        for record in records
    ]
    return _value_digest(rows)


def dataset_identity(
    path: str | Path, records: Sequence[RawSpanRecord], limit: int | None
) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    return {
        "path": str(source),
        "bytes": source.stat().st_size,
        "sha256": sha256_file(source),
        "selected_records": len(records),
        "selection_sha256": _selection_identity(records),
        "limit": limit,
    }


def _is_model_file(path: Path) -> bool:
    name = path.name
    return (
        name in _MODEL_EXACT_FILENAMES
        or (name.startswith("model-") and name.endswith(".safetensors"))
        or (name.startswith("pytorch_model-") and name.endswith(".bin"))
    )


def _is_tokenizer_file(path: Path) -> bool:
    return (
        path.name in _TOKENIZER_EXACT_FILENAMES
        or path.name.startswith("tokenizer.")
        or path.suffix == ".model"
    )


def _artifact_identity(
    path: str | Path, *, kind: str, predicate: Any
) -> dict[str, Any]:
    root = Path(path).expanduser().resolve()
    if not root.is_dir():
        raise StudentEvaluationError(f"Directory {kind} non trovata: {root}")
    candidates = sorted(
        (item for item in root.iterdir() if item.is_file() and predicate(item)),
        key=lambda item: item.name,
    )
    if not candidates:
        raise StudentEvaluationError(f"File {kind} non trovati in {root}")
    files = [
        {
            "path": item.name,
            "bytes": item.stat().st_size,
            "sha256": sha256_file(item),
        }
        for item in candidates
    ]
    return {"path": str(root), "sha256": _value_digest(files), "files": files}


def model_identity(path: str | Path) -> dict[str, Any]:
    """Content hash of the local config and PyTorch weight files."""

    identity = _artifact_identity(path, kind="modello", predicate=_is_model_file)
    names = {item["path"] for item in identity["files"]}
    if "config.json" not in names:
        raise StudentEvaluationError(f"config.json del modello assente in {path}")
    if not any(
        name.endswith((".safetensors", ".bin")) for name in names
    ):
        raise StudentEvaluationError(f"Pesi PyTorch assenti in {path}")
    return identity


def tokenizer_identity(path: str | Path) -> dict[str, Any]:
    """Content hash of every local file that can alter token IDs/offsets."""

    return _artifact_identity(path, kind="tokenizer", predicate=_is_tokenizer_file)


def merge_overflow_windows(
    logits_by_window: Sequence[Sequence[Sequence[float]]],
    offsets_by_window: Sequence[Sequence[Sequence[int]]],
    special_tokens_masks: Sequence[Sequence[int | bool]],
    id2label: Mapping[int, str],
    *,
    text_length: int | None = None,
) -> set[Span]:
    """Merge overflow windows by exact token offset, then aggregate BIO once.

    Every non-special subword participates.  Repeated observations of the same
    absolute character interval are averaged at logit level.  Ties use the
    lower label ID.  Non-identical overlapping token intervals are rejected:
    silently choosing one would make offsets/tokenization drift invisible.
    """

    if not id2label:
        raise StudentEvaluationError("id2label vuoto")
    row_counts = {
        len(logits_by_window),
        len(offsets_by_window),
        len(special_tokens_masks),
    }
    if len(row_counts) != 1 or not logits_by_window:
        raise StudentEvaluationError("Finestre logits/offset/mask disallineate o vuote")

    label_ids = sorted(int(index) for index in id2label)
    if label_ids != list(range(len(label_ids))):
        raise StudentEvaluationError("Gli ID label devono essere contigui da zero")
    normalized_labels = {int(index): str(label) for index, label in id2label.items()}
    sums: dict[tuple[int, int], list[float]] = {}
    counts: Counter[tuple[int, int]] = Counter()

    for window_index, (window_logits, offsets, special_mask) in enumerate(
        zip(logits_by_window, offsets_by_window, special_tokens_masks)
    ):
        if len(window_logits) != len(offsets) or len(offsets) != len(special_mask):
            raise StudentEvaluationError(
                f"Finestra {window_index}: logits/offset/mask disallineati"
            )
        for token_index, (raw_logits, raw_offset, is_special) in enumerate(
            zip(window_logits, offsets, special_mask)
        ):
            if is_special:
                continue
            if not isinstance(raw_offset, (list, tuple)) or len(raw_offset) != 2:
                raise StudentEvaluationError(
                    f"Finestra {window_index}, token {token_index}: offset non valido"
                )
            start, end = int(raw_offset[0]), int(raw_offset[1])
            if end <= start:
                continue
            if start < 0 or (text_length is not None and end > text_length):
                raise StudentEvaluationError(
                    f"Finestra {window_index}, token {token_index}: offset fuori testo"
                )
            values = [float(value) for value in raw_logits]
            if len(values) != len(label_ids) or not all(math.isfinite(value) for value in values):
                raise StudentEvaluationError(
                    f"Finestra {window_index}, token {token_index}: logits non validi"
                )
            key = (start, end)
            if key not in sums:
                sums[key] = [0.0] * len(values)
            for label_index, value in enumerate(values):
                sums[key][label_index] += value
            counts[key] += 1

    ordered_offsets = sorted(sums)
    if not ordered_offsets:
        return set()
    previous = ordered_offsets[0]
    for current in ordered_offsets[1:]:
        if current[0] < previous[1]:
            raise StudentEvaluationError(
                "Tokenizzazione incoerente tra finestre: intervalli sovrapposti "
                f"{previous} e {current}"
            )
        previous = current

    labels: list[str] = []
    for offset in ordered_offsets:
        divisor = counts[offset]
        averages = [value / divisor for value in sums[offset]]
        prediction_id = max(range(len(averages)), key=lambda index: averages[index])
        labels.append(normalized_labels[prediction_id])
    return simple_char_spans(labels, ordered_offsets)


def normalize_model_spans_for_production(
    text: str, spans: Iterable[Span]
) -> set[Span]:
    """Apply the model-only boundary rules used by ``src.app.app._merge``.

    Gemma/ModernBERT's Metaspace tokens can include the whitespace immediately
    before a word in their offset.  The production service removes that outer
    whitespace and expands a span that cuts an alphanumeric word in half.  The
    raw evaluator must do the same or an otherwise correct entity is counted as
    both a false positive and a false negative merely because its first token
    starts one character before the human annotation.

    Regex/checksum precedence is deliberately *not* reproduced here: this is
    still the model-only view.  Punctuation is not stripped because production
    does not strip it either.
    """

    normalized: list[Span] = []
    for entity_type, raw_start, raw_end in spans:
        start, end = int(raw_start), int(raw_end)
        if start < 0 or end > len(text) or end <= start:
            raise StudentEvaluationError(
                f"Span modello fuori testo: {entity_type} {start}:{end}"
            )
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if end <= start:
            continue
        while (
            start > 0
            and (text[start - 1].isalnum() or text[start - 1] == "_")
            and (text[start].isalnum() or text[start] == "_")
        ):
            start -= 1
        while (
            end < len(text)
            and (text[end].isalnum() or text[end] == "_")
            and (text[end - 1].isalnum() or text[end - 1] == "_")
        ):
            end += 1
        normalized.append((entity_type, start, end))

    normalized.sort(key=lambda span: (span[1], -(span[2] - span[1]), span[0]))
    merged: list[Span] = []
    for entity_type, start, end in normalized:
        if merged and start < merged[-1][2]:
            previous_type, previous_start, previous_end = merged[-1]
            merged[-1] = (previous_type, previous_start, max(previous_end, end))
            continue
        if merged and start == merged[-1][2] and entity_type == merged[-1][0]:
            previous_type, previous_start, _ = merged[-1]
            merged[-1] = (previous_type, previous_start, end)
            continue
        merged.append((entity_type, start, end))
    return set(merged)


def canonicalize_id_doc_prefix_spans(
    text: str, spans: Iterable[Span]
) -> frozenset[Span]:
    """Canonicalize only harmless leading number markers on ``ID_DOC``.

    The returned spans still point into the original text.  For ``ID_DOC`` we
    trim outer whitespace and, at the beginning of the span only, one of the
    case-insensitive markers ``n.``, ``n°``, ``nr.`` or ``numero`` plus the
    whitespace following it.  Every other character is payload and therefore
    remains part of the exact comparison.  Other entity types are returned
    byte-for-byte unchanged, so a ``DOCID``/``IBAN`` prediction can never be
    forgiven by this policy.

    A marker is removed only when a non-whitespace payload remains.  This
    prevents malformed marker-only spans from collapsing to the same empty
    interval.
    """

    canonical: set[Span] = set()
    for entity_type, raw_start, raw_end in spans:
        start, end = int(raw_start), int(raw_end)
        if start < 0 or end > len(text) or end <= start:
            raise StudentEvaluationError(
                f"Span metrica fuori testo: {entity_type} {start}:{end}"
            )
        if entity_type != "ID_DOC":
            canonical.add((entity_type, start, end))
            continue

        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if end <= start:
            canonical.add((entity_type, int(raw_start), int(raw_end)))
            continue

        marker = _ID_DOC_OPTIONAL_NUMBER_PREFIX.match(text[start:end])
        if marker is not None:
            payload_start = start + marker.end()
            while payload_start < end and text[payload_start].isspace():
                payload_start += 1
            if payload_start < end:
                start = payload_start
        canonical.add((entity_type, start, end))
    return frozenset(canonical)


def compute_id_doc_prefix_tolerant_metric(
    records: Sequence[RawSpanRecord],
    predictions: Sequence[Iterable[Span]],
) -> dict[str, Any]:
    """Compute the optional-prefix view while keeping exact-span primary."""

    if len(records) != len(predictions):
        raise StudentEvaluationError("Record e predizioni disallineati per la metrica")

    gold_documents: list[frozenset[Span]] = []
    prediction_documents: list[frozenset[Span]] = []
    canonicalized_gold = 0
    canonicalized_predictions = 0
    for record, predicted in zip(records, predictions):
        predicted_spans = frozenset(predicted)
        canonical_gold = canonicalize_id_doc_prefix_spans(
            record.text, record.gold_spans
        )
        canonical_prediction = canonicalize_id_doc_prefix_spans(
            record.text, predicted_spans
        )
        canonicalized_gold += len(record.gold_spans - canonical_gold)
        canonicalized_predictions += len(predicted_spans - canonical_prediction)
        gold_documents.append(canonical_gold)
        prediction_documents.append(canonical_prediction)

    return {
        "name": ID_DOC_PREFIX_TOLERANT_METRIC_NAME,
        "policy": dict(ID_DOC_PREFIX_TOLERANT_POLICY),
        "canonicalized_spans": {
            "gold": canonicalized_gold,
            "predictions": canonicalized_predictions,
        },
        "metrics": compute_span_metrics(gold_documents, prediction_documents),
    }


def _optional_ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _alnum_positions(text: str, start: int, end: int) -> set[int]:
    return {index for index in range(start, end) if text[index].isalnum()}


def _validated_prediction_spans(
    text: str, spans: Iterable[Span]
) -> frozenset[Span]:
    validated: set[Span] = set()
    for index, raw_span in enumerate(spans):
        if not isinstance(raw_span, (tuple, list)) or len(raw_span) != 3:
            raise StudentEvaluationError(f"Prediction span {index} non valido")
        entity_type, raw_start, raw_end = raw_span
        if (
            not isinstance(entity_type, str)
            or not entity_type
            or isinstance(raw_start, bool)
            or isinstance(raw_end, bool)
            or not isinstance(raw_start, int)
            or not isinstance(raw_end, int)
            or raw_start < 0
            or raw_end <= raw_start
            or raw_end > len(text)
        ):
            raise StudentEvaluationError(f"Prediction span {index} fuori contratto")
        validated.add((entity_type, raw_start, raw_end))
    return frozenset(validated)


def _coverage_aggregate(
    counts: Mapping[str, int],
    *,
    prefix: str,
) -> dict[str, Any]:
    characters = int(counts.get("gold_sensitive_characters", 0))
    entities = int(counts.get("gold_entities_with_sensitive_payload", 0))
    documents = int(counts.get("documents_with_sensitive_gold", 0))
    covered_characters = int(counts.get(f"{prefix}_covered_characters", 0))
    covered_entities = int(counts.get(f"{prefix}_fully_covered_entities", 0))
    covered_documents = int(counts.get(f"{prefix}_fully_covered_documents", 0))
    return {
        "covered_characters": covered_characters,
        "uncovered_characters": characters - covered_characters,
        "character_coverage": _optional_ratio(covered_characters, characters),
        "fully_covered_entities": covered_entities,
        "entities_with_sensitive_payload": entities,
        "entity_coverage": _optional_ratio(covered_entities, entities),
        "fully_covered_documents": covered_documents,
        "documents_with_sensitive_gold": documents,
        "document_coverage": _optional_ratio(covered_documents, documents),
    }


def compute_production_privacy_utility_metric(
    records: Sequence[RawSpanRecord],
    predictions: Sequence[Iterable[Span]],
) -> dict[str, Any]:
    """Score production masking by Unicode payload characters, never raw text."""

    if len(records) != len(predictions):
        raise StudentEvaluationError(
            "Record e predizioni disallineati per privacy/utility"
        )

    totals: Counter[str] = Counter()
    per_tag: dict[str, Counter[str]] = {}
    for record, raw_predictions in zip(records, predictions):
        canonical_gold = canonicalize_id_doc_prefix_spans(
            record.text, record.gold_spans
        )
        predicted_spans = _validated_prediction_spans(
            record.text, raw_predictions
        )
        totals["documents"] += 1
        totals["gold_entities"] += len(canonical_gold)
        totals["gold_spans_canonicalized"] += len(
            record.gold_spans - canonical_gold
        )

        gold_entities: list[tuple[str, set[int]]] = []
        gold_by_tag: dict[str, set[int]] = {}
        gold_sensitive: set[int] = set()
        for entity_type, start, end in canonical_gold:
            positions = _alnum_positions(record.text, start, end)
            gold_entities.append((entity_type, positions))
            gold_sensitive.update(positions)
            gold_by_tag.setdefault(entity_type, set()).update(positions)
            tag_counts = per_tag.setdefault(entity_type, Counter())
            tag_counts["gold_entities"] += 1
            if positions:
                totals["gold_entities_with_sensitive_payload"] += 1
                tag_counts["gold_entities_with_sensitive_payload"] += 1
                tag_counts["gold_sensitive_characters"] += len(positions)
            else:
                totals["gold_entities_without_sensitive_payload_excluded"] += 1
                tag_counts[
                    "gold_entities_without_sensitive_payload_excluded"
                ] += 1

        predicted_by_tag: dict[str, set[int]] = {}
        predicted_alnum: set[int] = set()
        for entity_type, start, end in predicted_spans:
            positions = _alnum_positions(record.text, start, end)
            predicted_alnum.update(positions)
            predicted_by_tag.setdefault(entity_type, set()).update(positions)
            per_tag.setdefault(entity_type, Counter())

        privacy_covered = gold_sensitive & predicted_alnum
        type_correct_covered: set[int] = set()
        for entity_type, gold_positions in gold_by_tag.items():
            type_correct_covered.update(
                gold_positions & predicted_by_tag.get(entity_type, set())
            )
        collateral = predicted_alnum - gold_sensitive
        all_alnum = {
            index for index, character in enumerate(record.text) if character.isalnum()
        }
        non_pii_alnum = all_alnum - gold_sensitive

        totals["gold_sensitive_characters"] += len(gold_sensitive)
        totals["privacy_covered_characters"] += len(privacy_covered)
        totals["type_correct_covered_characters"] += len(type_correct_covered)
        totals["wrong_type_only_characters"] += len(
            privacy_covered - type_correct_covered
        )
        totals["predicted_alnum_characters"] += len(predicted_alnum)
        totals["predicted_on_gold_characters"] += len(
            predicted_alnum & gold_sensitive
        )
        totals["collateral_masked_characters"] += len(collateral)
        totals["non_pii_alnum_characters"] += len(non_pii_alnum)

        for entity_type, positions in gold_entities:
            if not positions:
                continue
            tag_counts = per_tag[entity_type]
            if positions <= predicted_alnum:
                totals["privacy_fully_covered_entities"] += 1
                tag_counts["privacy_fully_covered_entities"] += 1
            if positions <= predicted_by_tag.get(entity_type, set()):
                totals["type_correct_fully_covered_entities"] += 1
                tag_counts["type_correct_fully_covered_entities"] += 1

        if gold_sensitive:
            totals["documents_with_sensitive_gold"] += 1
            if gold_sensitive <= predicted_alnum:
                totals["privacy_fully_covered_documents"] += 1
            if gold_sensitive <= type_correct_covered:
                totals["type_correct_fully_covered_documents"] += 1

        for entity_type, gold_positions in gold_by_tag.items():
            if not gold_positions:
                continue
            tag_counts = per_tag[entity_type]
            tag_counts["documents_with_sensitive_gold"] += 1
            if gold_positions <= predicted_alnum:
                tag_counts["privacy_fully_covered_documents"] += 1
            if gold_positions <= predicted_by_tag.get(entity_type, set()):
                tag_counts["type_correct_fully_covered_documents"] += 1

        for entity_type in set(gold_by_tag) | set(predicted_by_tag):
            tag_counts = per_tag[entity_type]
            gold_positions = gold_by_tag.get(entity_type, set())
            predicted_positions = predicted_by_tag.get(entity_type, set())
            any_covered = gold_positions & predicted_alnum
            correct_covered = gold_positions & predicted_positions
            predicted_on_gold = predicted_positions & gold_sensitive
            predicted_collateral = predicted_positions - gold_sensitive
            tag_counts["privacy_covered_characters"] += len(any_covered)
            tag_counts["type_correct_covered_characters"] += len(
                correct_covered
            )
            tag_counts["wrong_type_only_characters"] += len(
                any_covered - correct_covered
            )
            tag_counts["predicted_alnum_characters"] += len(predicted_positions)
            tag_counts["predicted_on_gold_characters"] += len(predicted_on_gold)
            tag_counts["collateral_masked_characters"] += len(
                predicted_collateral
            )

    privacy = _coverage_aggregate(totals, prefix="privacy")
    privacy["leaked_characters"] = privacy["uncovered_characters"]
    type_correct = _coverage_aggregate(totals, prefix="type_correct")
    type_correct["wrong_type_but_privacy_covered_characters"] = int(
        totals["wrong_type_only_characters"]
    )
    utility = {
        "predicted_alnum_characters": int(totals["predicted_alnum_characters"]),
        "predicted_on_gold_characters": int(
            totals["predicted_on_gold_characters"]
        ),
        "collateral_masked_characters": int(
            totals["collateral_masked_characters"]
        ),
        "mask_precision": _optional_ratio(
            int(totals["predicted_on_gold_characters"]),
            int(totals["predicted_alnum_characters"]),
        ),
        "collateral_share_of_predicted": _optional_ratio(
            int(totals["collateral_masked_characters"]),
            int(totals["predicted_alnum_characters"]),
        ),
        "non_pii_alnum_characters": int(totals["non_pii_alnum_characters"]),
        "collateral_masking_ratio": _optional_ratio(
            int(totals["collateral_masked_characters"]),
            int(totals["non_pii_alnum_characters"]),
        ),
    }

    tag_breakdown: dict[str, Any] = {}
    for entity_type in sorted(per_tag):
        counts = per_tag[entity_type]
        tag_privacy = _coverage_aggregate(counts, prefix="privacy")
        tag_privacy["leaked_characters"] = tag_privacy["uncovered_characters"]
        tag_type_correct = _coverage_aggregate(counts, prefix="type_correct")
        tag_type_correct["wrong_type_but_privacy_covered_characters"] = int(
            counts["wrong_type_only_characters"]
        )
        predicted_count = int(counts["predicted_alnum_characters"])
        collateral_count = int(counts["collateral_masked_characters"])
        tag_breakdown[entity_type] = {
            "gold": {
                "entities": int(counts["gold_entities"]),
                "entities_with_sensitive_payload": int(
                    counts["gold_entities_with_sensitive_payload"]
                ),
                "entities_without_sensitive_payload_excluded": int(
                    counts["gold_entities_without_sensitive_payload_excluded"]
                ),
                "sensitive_characters": int(counts["gold_sensitive_characters"]),
            },
            "type_agnostic": tag_privacy,
            "type_correct": tag_type_correct,
            "prediction_utility": {
                "predicted_alnum_characters": predicted_count,
                "predicted_on_any_gold_characters": int(
                    counts["predicted_on_gold_characters"]
                ),
                "collateral_masked_characters": collateral_count,
                "mask_precision": _optional_ratio(
                    int(counts["predicted_on_gold_characters"]), predicted_count
                ),
                "collateral_share_of_predicted": _optional_ratio(
                    collateral_count, predicted_count
                ),
                "collateral_masking_ratio_global_non_pii_denominator": (
                    _optional_ratio(
                        collateral_count,
                        int(totals["non_pii_alnum_characters"]),
                    )
                ),
            },
        }

    return {
        "name": PRODUCTION_PRIVACY_UTILITY_METRIC_NAME,
        "policy": dict(PRODUCTION_PRIVACY_UTILITY_POLICY),
        "counts": {
            "documents": int(totals["documents"]),
            "documents_with_sensitive_gold": int(
                totals["documents_with_sensitive_gold"]
            ),
            "gold_entities": int(totals["gold_entities"]),
            "gold_entities_with_sensitive_payload": int(
                totals["gold_entities_with_sensitive_payload"]
            ),
            "gold_entities_without_sensitive_payload_excluded": int(
                totals["gold_entities_without_sensitive_payload_excluded"]
            ),
            "gold_spans_canonicalized": int(totals["gold_spans_canonicalized"]),
            "gold_sensitive_characters": int(totals["gold_sensitive_characters"]),
        },
        "type_agnostic": privacy,
        "type_correct": type_correct,
        "utility": utility,
        "per_tag": tag_breakdown,
    }


def _resolve_device(torch: Any, requested: str) -> str:
    if requested == "auto":
        return "mps" if torch.backends.mps.is_available() else "cpu"
    if requested == "mps" and not torch.backends.mps.is_available():
        raise StudentEvaluationError("MPS richiesto ma non disponibile")
    return requested


def _tokenize_windows(tokenizer: Any, text: str, max_length: int, stride: int) -> tuple[Any, Any, Any]:
    encoded = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        return_special_tokens_mask=True,
        padding=False,
    )
    try:
        offsets = encoded.pop("offset_mapping")
        special_masks = encoded.pop("special_tokens_mask")
    except KeyError as exc:
        raise StudentEvaluationError(
            "Il tokenizer non ha restituito offset_mapping/special_tokens_mask"
        ) from exc
    sample_mapping = encoded.pop("overflow_to_sample_mapping", None)
    if sample_mapping is not None and any(int(index) != 0 for index in sample_mapping):
        raise StudentEvaluationError("overflow_to_sample_mapping inatteso per un documento")
    return encoded, offsets, special_masks


def _infer_record(
    *,
    torch: Any,
    model: Any,
    tokenizer: Any,
    record: RawSpanRecord,
    id2label: Mapping[int, str],
    max_length: int,
    stride: int,
    window_batch_size: int,
    device: str,
) -> set[Span]:
    encoded, offsets, special_masks = _tokenize_windows(
        tokenizer, record.text, max_length, stride
    )
    input_ids = encoded.get("input_ids")
    if not isinstance(input_ids, list) or not input_ids or not isinstance(input_ids[0], list):
        raise StudentEvaluationError("Il tokenizer non ha prodotto finestre di input_ids")
    if len(input_ids) != len(offsets) or len(offsets) != len(special_masks):
        raise StudentEvaluationError("Finestre tokenizzate disallineate")

    allowed_keys = {"input_ids", "attention_mask", "token_type_ids"}
    features = [
        {
            key: values[window_index]
            for key, values in encoded.items()
            if key in allowed_keys
        }
        for window_index in range(len(input_ids))
    ]
    logits_by_window: list[list[list[float]]] = []
    for start in range(0, len(features), window_batch_size):
        feature_batch = features[start : start + window_batch_size]
        lengths = [len(feature["input_ids"]) for feature in feature_batch]
        padded = tokenizer.pad(feature_batch, padding=True, return_tensors="pt")
        padded = {key: value.to(device) for key, value in padded.items()}
        with torch.inference_mode():
            output = model(**padded).logits.detach().float().cpu()
        for row_index, length in enumerate(lengths):
            logits_by_window.append(output[row_index, :length].tolist())

    return normalize_model_spans_for_production(
        record.text,
        merge_overflow_windows(
            logits_by_window,
            offsets,
            special_masks,
            id2label,
            text_length=len(record.text),
        ),
    )


def _software_versions() -> dict[str, str | None]:
    result: dict[str, str | None] = {"python": sys.version.split()[0]}
    for package in ("torch", "transformers", "tokenizers", "safetensors"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    return result


def _atomic_write_text(path: Path, content: str) -> None:
    target = path.expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
        os.replace(temporary_name, target)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_write_text(
        path, json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    )


def write_predictions_artifact(
    path: str | Path,
    records: Sequence[RawSpanRecord],
    predictions: Sequence[Iterable[Span]],
) -> dict[str, Any]:
    if len(records) != len(predictions):
        raise StudentEvaluationError("Record e predizioni disallineati")
    lines: list[str] = []
    for record, predicted in zip(records, predictions):
        row = {
            "schema": PREDICTION_SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "id": record.record_id,
            "gold_spans": _sorted_spans(record.gold_spans),
            "prediction_spans": _sorted_spans(predicted),
        }
        lines.append(_canonical_json_bytes(row).decode("utf-8"))
    target = Path(path).expanduser().resolve()
    _atomic_write_text(target, "\n".join(lines) + "\n")
    return {
        "path": str(target),
        "format": "jsonl",
        "schema": PREDICTION_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "records": len(records),
        "bytes": target.stat().st_size,
        "sha256": sha256_file(target),
    }


def evaluate_model(
    *,
    model_path: str | Path,
    dataset_path: str | Path,
    output_report: str | Path,
    output_predictions: str | Path | None = None,
    tokenizer_path: str | Path | None = None,
    role: str = "model",
    max_length: int = 256,
    stride: int = 32,
    window_batch_size: int = 1,
    device: str = "cpu",
    threads: int | None = None,
    limit: int | None = None,
    attention_implementation: str = "eager",
) -> dict[str, Any]:
    """Run one local PyTorch model and persist metrics plus compact predictions."""

    if max_length < 8:
        raise StudentEvaluationError("max_length deve essere almeno 8")
    if stride < 0 or stride >= max_length - 2:
        raise StudentEvaluationError("stride non valido per max_length")
    if window_batch_size < 1:
        raise StudentEvaluationError("window_batch_size deve essere positivo")
    if threads is not None and threads < 1:
        raise StudentEvaluationError("threads deve essere positivo")

    model_root = Path(model_path).expanduser().resolve()
    tokenizer_root = Path(tokenizer_path or model_root).expanduser().resolve()
    report_target = Path(output_report).expanduser().resolve()
    predictions_target = (
        Path(output_predictions).expanduser().resolve()
        if output_predictions is not None
        else report_target.with_name(report_target.stem + ".predictions.jsonl")
    )
    if report_target == predictions_target:
        raise StudentEvaluationError("Report e predictions devono essere file distinti")

    records = load_raw_records(dataset_path, limit=limit)
    data_identity = dataset_identity(dataset_path, records, limit)
    checkpoint_identity = model_identity(model_root)
    preprocessing_identity = tokenizer_identity(tokenizer_root)

    try:
        import torch
        from transformers import AutoModelForTokenClassification, AutoTokenizer
    except ImportError as exc:
        raise StudentEvaluationError(
            "torch e transformers sono richiesti solo per il comando evaluate"
        ) from exc

    if threads is not None:
        torch.set_num_threads(threads)
    torch.set_grad_enabled(False)
    resolved_device = _resolve_device(torch, device)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_root,
        local_files_only=True,
        use_fast=True,
        fix_mistral_regex=TOKENIZER_FIX_MISTRAL_REGEX,
    )
    if not getattr(tokenizer, "is_fast", False):
        raise StudentEvaluationError("Serve un fast tokenizer con offset_mapping")
    model = AutoModelForTokenClassification.from_pretrained(
        model_root,
        local_files_only=True,
        attn_implementation=attention_implementation,
    ).to(resolved_device).eval()
    id2label = {
        int(index): str(label) for index, label in model.config.id2label.items()
    }
    if "O" not in id2label.values():
        raise StudentEvaluationError("La label map del modello non contiene O")

    started = time.perf_counter()
    predictions = [
        _infer_record(
            torch=torch,
            model=model,
            tokenizer=tokenizer,
            record=record,
            id2label=id2label,
            max_length=max_length,
            stride=stride,
            window_batch_size=window_batch_size,
            device=resolved_device,
        )
        for record in records
    ]
    inference_seconds = time.perf_counter() - started
    metrics = compute_span_metrics(
        [record.gold_spans for record in records], predictions
    )
    id_doc_prefix_tolerant = compute_id_doc_prefix_tolerant_metric(
        records, predictions
    )
    production_privacy_utility = compute_production_privacy_utility_metric(
        records, predictions
    )
    prediction_identity = write_predictions_artifact(
        predictions_target, records, predictions
    )
    report: dict[str, Any] = {
        "schema": EVALUATION_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": _utc_now(),
        "role": role,
        "model": checkpoint_identity,
        "tokenizer": preprocessing_identity,
        "dataset": data_identity,
        "evaluation": {
            "backend": "pytorch",
            "device": resolved_device,
            "max_length": max_length,
            "stride": stride,
            "window_batch_size": window_batch_size,
            "attention_implementation": attention_implementation,
            "tokenizer_load_policy": tokenizer_load_policy(),
            "merge_policy": "mean_logits_by_exact_offset_then_global_simple_bio",
            "all_subwords": True,
            "inference_seconds": inference_seconds,
        },
        "label_map": {str(index): label for index, label in sorted(id2label.items())},
        "metrics": metrics,
        "additional_metrics": {
            ID_DOC_PREFIX_TOLERANT_METRIC_NAME: id_doc_prefix_tolerant,
            PRODUCTION_PRIVACY_UTILITY_METRIC_NAME: production_privacy_utility,
        },
        "predictions": prediction_identity,
        "host": {"platform": platform.platform(), "machine": platform.machine()},
        "software": _software_versions(),
    }
    _write_json(report_target, report)
    return report


def _load_json_object(path: Path, *, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise StudentEvaluationError(f"{description} non trovato: {path}") from exc
    except json.JSONDecodeError as exc:
        raise StudentEvaluationError(f"{description} non e' JSON valido: {path}") from exc
    if not isinstance(value, dict):
        raise StudentEvaluationError(f"{description} deve essere un oggetto JSON")
    return value


def _prediction_path(
    report: Mapping[str, Any], report_path: Path, override: str | Path | None
) -> Path:
    if override is not None:
        return Path(override).expanduser().resolve()
    metadata = report.get("predictions")
    if not isinstance(metadata, Mapping) or not isinstance(metadata.get("path"), str):
        raise StudentEvaluationError(f"{report_path}: path predictions assente")
    candidate = Path(str(metadata["path"])).expanduser()
    if not candidate.is_absolute():
        candidate = report_path.parent / candidate
    return candidate.resolve()


def load_predictions_artifact(
    path: str | Path, *, expected_sha256: str | None = None
) -> list[dict[str, Any]]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise StudentEvaluationError(f"Predictions non trovate: {source}")
    actual_sha256 = sha256_file(source)
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise StudentEvaluationError(
            f"Digest predictions non corrispondente per {source}: "
            f"atteso {expected_sha256}, trovato {actual_sha256}"
        )
    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for line_number, row in enumerate(_jsonl_rows(source), start=1):
        if row.get("schema") != PREDICTION_SCHEMA or row.get("schema_version") != SCHEMA_VERSION:
            raise StudentEvaluationError(
                f"{source}:{line_number}: schema predictions non supportato"
            )
        record_id = str(row.get("id"))
        if record_id in seen_ids:
            raise StudentEvaluationError(
                f"{source}:{line_number}: id duplicato {record_id!r}"
            )
        seen_ids.add(record_id)
        rows.append(
            {
                "id": record_id,
                "gold_spans": _parse_spans(
                    row.get("gold_spans"), context=f"{source}:{line_number}:gold"
                ),
                "prediction_spans": _parse_spans(
                    row.get("prediction_spans"),
                    context=f"{source}:{line_number}:prediction",
                ),
            }
        )
    if not rows:
        raise StudentEvaluationError(f"Predictions vuote: {source}")
    return rows


def rescore_saved_predictions_privacy_utility(
    *,
    dataset_path: str | Path,
    predictions_path: str | Path,
    expected_predictions_sha256: str | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    """Offline rescore from saved spans; returns policy and aggregates only."""

    records = load_raw_records(dataset_path, limit=limit)
    rows = load_predictions_artifact(
        predictions_path,
        expected_sha256=expected_predictions_sha256,
    )
    records_by_id = {record.record_id: record for record in records}
    if len(records_by_id) != len(records):
        raise StudentEvaluationError("Dataset con record_id duplicati nel rescore")
    rows_by_id = {str(row["id"]): row for row in rows}
    if set(records_by_id) != set(rows_by_id):
        raise StudentEvaluationError(
            "Dataset e predictions hanno record_id differenti nel rescore"
        )
    ordered_predictions: list[frozenset[Span]] = []
    for record in records:
        row = rows_by_id[record.record_id]
        if row["gold_spans"] != record.gold_spans:
            raise StudentEvaluationError(
                "Gold dataset e predictions divergenti nel rescore"
            )
        ordered_predictions.append(row["prediction_spans"])
    return compute_production_privacy_utility_metric(records, ordered_predictions)


def write_privacy_rescore_report(
    *,
    dataset_path: str | Path,
    predictions_path: str | Path,
    output_report: str | Path,
    expected_predictions_sha256: str | None = None,
) -> dict[str, Any]:
    """Fail-closed offline rescore with atomic, aggregate-only persistence."""

    dataset_source = Path(dataset_path).expanduser().resolve()
    predictions_source = Path(predictions_path).expanduser().resolve()
    report_target = Path(output_report).expanduser().resolve()
    if report_target in {dataset_source, predictions_source}:
        raise StudentEvaluationError(
            "Il report rescore deve essere distinto da dataset e predictions"
        )
    if not dataset_source.is_file():
        raise StudentEvaluationError(f"Dataset rescore non trovato: {dataset_source}")
    if not predictions_source.is_file():
        raise StudentEvaluationError(
            f"Predictions rescore non trovate: {predictions_source}"
        )
    if expected_predictions_sha256 is not None and re.fullmatch(
        r"[0-9a-fA-F]{64}", expected_predictions_sha256
    ) is None:
        raise StudentEvaluationError("SHA256 atteso predictions non valido")

    expected_hash = (
        expected_predictions_sha256.lower()
        if expected_predictions_sha256 is not None
        else None
    )
    before = {
        "dataset": {
            "bytes": dataset_source.stat().st_size,
            "sha256": sha256_file(dataset_source),
        },
        "predictions": {
            "bytes": predictions_source.stat().st_size,
            "sha256": sha256_file(predictions_source),
        },
    }
    metric = rescore_saved_predictions_privacy_utility(
        dataset_path=dataset_source,
        predictions_path=predictions_source,
        expected_predictions_sha256=expected_hash,
    )
    records = load_raw_records(dataset_source)
    dataset_record = dataset_identity(dataset_source, records, None)
    after = {
        "dataset": {
            "bytes": dataset_source.stat().st_size,
            "sha256": sha256_file(dataset_source),
        },
        "predictions": {
            "bytes": predictions_source.stat().st_size,
            "sha256": sha256_file(predictions_source),
        },
    }
    if before != after or dataset_record["sha256"] != after["dataset"]["sha256"]:
        raise StudentEvaluationError("Input rescore modificati durante la lettura")

    report = {
        "schema": PRIVACY_RESCORE_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": _utc_now(),
        "dataset": dataset_record,
        "predictions": {
            "path": str(predictions_source),
            "format": "jsonl",
            "schema": PREDICTION_SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "records": len(records),
            "bytes": int(after["predictions"]["bytes"]),
            "sha256": str(after["predictions"]["sha256"]),
        },
        "verification": {
            "dataset_sha256_stable_during_rescore": True,
            "predictions_sha256_stable_during_rescore": True,
            "expected_predictions_sha256": expected_hash,
            "expected_predictions_sha256_verified": (
                expected_hash is not None
            ),
            "record_id_sets_exact_match": True,
            "gold_spans_exact_match": True,
        },
        "metric": metric,
    }
    _write_json(report_target, report)
    return report


def _metrics_delta(
    teacher: Mapping[str, Any], student: Mapping[str, Any]
) -> dict[str, Any]:
    tags = sorted(set(teacher.get("per_tag", {})) | set(student.get("per_tag", {})))
    return {
        "micro_f1": float(student["micro"]["f1"]) - float(teacher["micro"]["f1"]),
        "macro_f1": float(student["macro"]["f1"]) - float(teacher["macro"]["f1"]),
        "per_tag_f1": {
            tag: float(student.get("per_tag", {}).get(tag, {}).get("f1", 0.0))
            - float(teacher.get("per_tag", {}).get(tag, {}).get("f1", 0.0))
            for tag in tags
        },
        "per_tag_recall": {
            tag: float(student.get("per_tag", {}).get(tag, {}).get("recall", 0.0))
            - float(teacher.get("per_tag", {}).get(tag, {}).get("recall", 0.0))
            for tag in tags
        },
    }


def _paired_gold_disagreement(
    gold_docs: Sequence[frozenset[Span]],
    teacher_docs: Sequence[frozenset[Span]],
    student_docs: Sequence[frozenset[Span]],
) -> dict[str, Any]:
    outcomes: Counter[str] = Counter()
    span_counts: Counter[str] = Counter()
    for gold, teacher, student in zip(gold_docs, teacher_docs, student_docs):
        teacher_exact = teacher == gold
        student_exact = student == gold
        if teacher_exact and student_exact:
            outcomes["both_exact"] += 1
        elif teacher_exact:
            outcomes["teacher_only_exact"] += 1
        elif student_exact:
            outcomes["student_only_exact"] += 1
        else:
            outcomes["neither_exact"] += 1

        teacher_errors = len(teacher ^ gold)
        student_errors = len(student ^ gold)
        if teacher_errors < student_errors:
            outcomes["teacher_fewer_span_errors"] += 1
        elif student_errors < teacher_errors:
            outcomes["student_fewer_span_errors"] += 1
        else:
            outcomes["equal_span_errors"] += 1

        span_counts["shared_correct"] += len(gold & teacher & student)
        span_counts["teacher_only_correct"] += len((gold & teacher) - student)
        span_counts["student_only_correct"] += len((gold & student) - teacher)
        span_counts["shared_false_positive"] += len((teacher & student) - gold)
        span_counts["teacher_only_false_positive"] += len((teacher - gold) - student)
        span_counts["student_only_false_positive"] += len((student - gold) - teacher)
        span_counts["both_missed"] += len(gold - teacher - student)

    return {
        "documents": len(gold_docs),
        "document_outcomes": {
            key: outcomes[key]
            for key in (
                "both_exact",
                "teacher_only_exact",
                "student_only_exact",
                "neither_exact",
                "teacher_fewer_span_errors",
                "student_fewer_span_errors",
                "equal_span_errors",
            )
        },
        "span_outcomes": {
            key: span_counts[key]
            for key in (
                "shared_correct",
                "teacher_only_correct",
                "student_only_correct",
                "shared_false_positive",
                "teacher_only_false_positive",
                "student_only_false_positive",
                "both_missed",
            )
        },
    }


def compare_saved_evaluations(
    *,
    teacher_report: str | Path,
    student_report: str | Path,
    output_report: str | Path | None = None,
    teacher_predictions: str | Path | None = None,
    student_predictions: str | Path | None = None,
) -> dict[str, Any]:
    """Compare saved artifacts only; this function imports no ML framework."""

    teacher_report_path = Path(teacher_report).expanduser().resolve()
    student_report_path = Path(student_report).expanduser().resolve()
    teacher_meta = _load_json_object(teacher_report_path, description="Report teacher")
    student_meta = _load_json_object(student_report_path, description="Report student")
    for role, report, path in (
        ("teacher", teacher_meta, teacher_report_path),
        ("student", student_meta, student_report_path),
    ):
        if report.get("schema") != EVALUATION_SCHEMA or report.get("schema_version") != SCHEMA_VERSION:
            raise StudentEvaluationError(f"{path}: schema report {role} non supportato")
        if not isinstance(report.get("dataset"), Mapping):
            raise StudentEvaluationError(f"{path}: identita' dataset assente")

    teacher_dataset = teacher_meta["dataset"]
    student_dataset = student_meta["dataset"]
    for key in ("sha256", "selection_sha256", "selected_records"):
        if teacher_dataset.get(key) != student_dataset.get(key):
            raise StudentEvaluationError(
                f"Teacher e student non usano la stessa selezione dataset ({key})"
            )

    teacher_prediction_meta = teacher_meta.get("predictions")
    student_prediction_meta = student_meta.get("predictions")
    if not isinstance(teacher_prediction_meta, Mapping) or not isinstance(
        student_prediction_meta, Mapping
    ):
        raise StudentEvaluationError("Metadati predictions mancanti nei report")
    teacher_prediction_path = _prediction_path(
        teacher_meta, teacher_report_path, teacher_predictions
    )
    student_prediction_path = _prediction_path(
        student_meta, student_report_path, student_predictions
    )
    teacher_rows = load_predictions_artifact(
        teacher_prediction_path,
        expected_sha256=str(teacher_prediction_meta.get("sha256")),
    )
    student_rows = load_predictions_artifact(
        student_prediction_path,
        expected_sha256=str(student_prediction_meta.get("sha256")),
    )
    student_by_id = {row["id"]: row for row in student_rows}
    if {row["id"] for row in teacher_rows} != set(student_by_id):
        raise StudentEvaluationError("Teacher e student hanno record_id differenti")

    gold_docs: list[frozenset[Span]] = []
    teacher_docs: list[frozenset[Span]] = []
    student_docs: list[frozenset[Span]] = []
    for teacher_row in teacher_rows:
        student_row = student_by_id[teacher_row["id"]]
        if teacher_row["gold_spans"] != student_row["gold_spans"]:
            raise StudentEvaluationError(
                f"Gold diverso per il record {teacher_row['id']!r}"
            )
        gold_docs.append(teacher_row["gold_spans"])
        teacher_docs.append(teacher_row["prediction_spans"])
        student_docs.append(student_row["prediction_spans"])

    expected_records = int(teacher_dataset["selected_records"])
    if len(gold_docs) != expected_records:
        raise StudentEvaluationError(
            f"Predictions incomplete: attese {expected_records}, trovate {len(gold_docs)}"
        )
    teacher_metrics = compute_span_metrics(gold_docs, teacher_docs)
    student_metrics = compute_span_metrics(gold_docs, student_docs)
    model_disagreement = compare_span_predictions(teacher_docs, student_docs)
    model_disagreement["roles"] = {"reference": "teacher", "candidate": "student"}

    comparison: dict[str, Any] = {
        "schema": COMPARISON_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": _utc_now(),
        "dataset": dict(teacher_dataset),
        "teacher": {
            "report_path": str(teacher_report_path),
            "report_sha256": sha256_file(teacher_report_path),
            "model_sha256": teacher_meta.get("model", {}).get("sha256"),
            "tokenizer_sha256": teacher_meta.get("tokenizer", {}).get("sha256"),
            "predictions_path": str(teacher_prediction_path),
            "predictions_sha256": teacher_prediction_meta.get("sha256"),
            "metrics": teacher_metrics,
        },
        "student": {
            "report_path": str(student_report_path),
            "report_sha256": sha256_file(student_report_path),
            "model_sha256": student_meta.get("model", {}).get("sha256"),
            "tokenizer_sha256": student_meta.get("tokenizer", {}).get("sha256"),
            "predictions_path": str(student_prediction_path),
            "predictions_sha256": student_prediction_meta.get("sha256"),
            "metrics": student_metrics,
        },
        "quality_delta_student_minus_teacher": _metrics_delta(
            teacher_metrics, student_metrics
        ),
        "disagreement": {
            "teacher_vs_student": model_disagreement,
            "paired_against_gold": _paired_gold_disagreement(
                gold_docs, teacher_docs, student_docs
            ),
        },
    }
    if output_report is not None:
        _write_json(Path(output_report), comparison)
    return comparison


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    evaluate_parser = subparsers.add_parser(
        "evaluate", help="valuta un checkpoint PyTorch raw-span"
    )
    evaluate_parser.add_argument("--model-path", required=True)
    evaluate_parser.add_argument("--tokenizer-path")
    evaluate_parser.add_argument("--dataset-path", required=True)
    evaluate_parser.add_argument("--output-report", required=True)
    evaluate_parser.add_argument("--output-predictions")
    evaluate_parser.add_argument("--role", choices=("teacher", "student", "model"), default="model")
    evaluate_parser.add_argument("--max-length", type=int, default=256)
    evaluate_parser.add_argument("--stride", type=int, default=32)
    evaluate_parser.add_argument("--window-batch-size", type=int, default=1)
    evaluate_parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="cpu")
    evaluate_parser.add_argument("--threads", type=int)
    evaluate_parser.add_argument("--limit", type=int)
    evaluate_parser.add_argument(
        "--attention-implementation", choices=("eager", "sdpa"), default="eager"
    )

    compare_parser = subparsers.add_parser(
        "compare", help="confronta report salvati senza rifare inferenza"
    )
    compare_parser.add_argument("--teacher-report", required=True)
    compare_parser.add_argument("--student-report", required=True)
    compare_parser.add_argument("--teacher-predictions")
    compare_parser.add_argument("--student-predictions")
    compare_parser.add_argument("--output-report", required=True)

    rescore_parser = subparsers.add_parser(
        "rescore-privacy",
        help="ricalcola privacy/utility da dataset e predictions senza inferenza",
    )
    rescore_parser.add_argument("--dataset-path", required=True)
    rescore_parser.add_argument("--predictions-path", required=True)
    rescore_parser.add_argument("--output-report", required=True)
    rescore_parser.add_argument("--expected-predictions-sha256")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "evaluate":
        report = evaluate_model(
            model_path=args.model_path,
            tokenizer_path=args.tokenizer_path,
            dataset_path=args.dataset_path,
            output_report=args.output_report,
            output_predictions=args.output_predictions,
            role=args.role,
            max_length=args.max_length,
            stride=args.stride,
            window_batch_size=args.window_batch_size,
            device=args.device,
            threads=args.threads,
            limit=args.limit,
            attention_implementation=args.attention_implementation,
        )
        print(json.dumps({"report": str(Path(args.output_report).resolve()), "micro_f1": report["metrics"]["micro"]["f1"]}))
        return 0
    if args.command == "rescore-privacy":
        report = write_privacy_rescore_report(
            dataset_path=args.dataset_path,
            predictions_path=args.predictions_path,
            output_report=args.output_report,
            expected_predictions_sha256=args.expected_predictions_sha256,
        )
        print(
            json.dumps(
                {
                    "report": str(Path(args.output_report).resolve()),
                    "metric": report["metric"]["name"],
                    "type_agnostic_character_coverage": report["metric"][
                        "type_agnostic"
                    ]["character_coverage"],
                }
            )
        )
        return 0
    comparison = compare_saved_evaluations(
        teacher_report=args.teacher_report,
        student_report=args.student_report,
        teacher_predictions=args.teacher_predictions,
        student_predictions=args.student_predictions,
        output_report=args.output_report,
    )
    print(
        json.dumps(
            {
                "report": str(Path(args.output_report).resolve()),
                "micro_f1_delta": comparison["quality_delta_student_minus_teacher"]["micro_f1"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
