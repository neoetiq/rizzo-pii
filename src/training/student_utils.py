"""Utility pure per il fine-tuning riproducibile dello student mmBERT-small.

Il modulo non importa PyTorch, Transformers o Datasets: validazione dei dati,
contratto delle label e manifest possono quindi essere testati senza caricare il
modello da 140M parametri.
"""

from __future__ import annotations

import hashlib
import json
import platform
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from src.quantization.metrics import DROP_TYPES, TAG_MAP, normalize_labels


HASH_CHUNK_BYTES = 1024 * 1024
IGNORE_INDEX = -100
TOKENIZER_FIX_MISTRAL_REGEX = False
TOKENIZER_LOAD_POLICY_NAME = "preserve-modernbert-metaspace-v1"


def tokenizer_load_policy() -> dict[str, Any]:
    """Return the frozen tokenizer policy recorded by new training artifacts.

    Transformers 4.57.6 can misclassify this 256k-vocabulary ModernBERT/Gemma
    Metaspace tokenizer as Mistral.  The Mistral regex patch is incompatible
    with its standalone Metaspace pre-tokenizer, so every loader must preserve
    the published tokenizer explicitly.
    """

    return {
        "schema_version": 1,
        "name": TOKENIZER_LOAD_POLICY_NAME,
        "fix_mistral_regex": TOKENIZER_FIX_MISTRAL_REGEX,
    }


class StudentTrainingError(ValueError):
    """Il contratto di training dello student non e' valido."""


def sha256_file(path: str | Path) -> str:
    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_label_contract(config_path: str | Path) -> tuple[dict[str, int], dict[int, str]]:
    """Carica l'ordine esatto delle label dal teacher, senza ricostruirlo dai dati."""

    path = Path(config_path)
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise StudentTrainingError(f"Config teacher non trovato: {path}") from exc
    except json.JSONDecodeError as exc:
        raise StudentTrainingError(f"Config teacher non valido: {path}") from exc

    raw_label2id = config.get("label2id")
    raw_id2label = config.get("id2label")
    if not isinstance(raw_label2id, Mapping) or not isinstance(raw_id2label, Mapping):
        raise StudentTrainingError("Il config teacher non contiene label2id/id2label")

    try:
        label2id = {str(label): int(index) for label, index in raw_label2id.items()}
        id2label = {int(index): str(label) for index, label in raw_id2label.items()}
    except (TypeError, ValueError) as exc:
        raise StudentTrainingError("label2id/id2label del teacher non sono validi") from exc

    expected_ids = set(range(len(label2id)))
    if set(id2label) != expected_ids or set(label2id.values()) != expected_ids:
        raise StudentTrainingError("Gli ID delle label devono essere contigui da zero")
    if any(id2label[index] != label for label, index in label2id.items()):
        raise StudentTrainingError("label2id e id2label non sono inversi")
    if "O" not in label2id:
        raise StudentTrainingError("La label O e' obbligatoria")

    entity_types = {label[2:] for label in label2id if label.startswith("B-")}
    missing_inside = sorted(
        entity_type for entity_type in entity_types if f"I-{entity_type}" not in label2id
    )
    if missing_inside:
        raise StudentTrainingError(
            f"Label I- mancanti per le entita': {', '.join(missing_inside)}"
        )
    return label2id, id2label


def align_word_labels(
    word_ids: Sequence[int | None],
    labels: Sequence[str],
    label2id: Mapping[str, int],
) -> list[int]:
    """Propaga la supervisione a tutti i subword di una sequenza tokenizzata.

    Il primo subword conserva ``B-X``; quelli successivi della stessa parola
    ricevono ``I-X``. In questo modo CF, IBAN, email e codici lunghi non hanno
    porzioni escluse dalla loss.
    """

    normalized = normalize_labels(labels)
    aligned: list[int] = []
    previous_word_id: int | None = None
    for position, word_id in enumerate(word_ids):
        if word_id is None:
            aligned.append(IGNORE_INDEX)
            previous_word_id = None
            continue
        if word_id < 0 or word_id >= len(normalized):
            raise StudentTrainingError(
                f"word_id {word_id} fuori range alla posizione {position} "
                f"(parole={len(normalized)})"
            )
        label = normalized[word_id]
        if word_id == previous_word_id and label.startswith("B-"):
            label = "I-" + label[2:]
        try:
            aligned.append(int(label2id[label]))
        except KeyError as exc:
            raise StudentTrainingError(f"Label non prevista dal teacher: {label}") from exc
        previous_word_id = word_id
    return aligned


def _normalized_entity_type(raw_label: str) -> str | None:
    label = raw_label[2:] if raw_label.startswith(("B-", "I-")) else raw_label
    normalized = TAG_MAP.get(label, label)
    return None if normalized in DROP_TYPES else normalized


def align_char_spans(
    offsets: Sequence[Sequence[int]],
    entities: Sequence[Mapping[str, Any]],
    label2id: Mapping[str, int],
    *,
    special_tokens_mask: Sequence[int | bool] | None = None,
) -> list[int]:
    """Allinea entita' raw-text agli offset del tokenizer.

    Questo e' il percorso primario per clean e Ai4Privacy: evita di trattare i
    vecchi ``mbert_tokens`` con marker ``##`` come nuove parole. Gli span
    droppati dalla tassonomia diventano ``O``; padding e token speciali restano
    esclusi dalla loss.
    """

    if special_tokens_mask is not None and len(special_tokens_mask) != len(offsets):
        raise StudentTrainingError("offsets e special_tokens_mask sono disallineati")

    normalized_entities: list[tuple[int, int, str]] = []
    for entity_index, entity in enumerate(entities):
        if not isinstance(entity, Mapping):
            raise StudentTrainingError(f"Entita' {entity_index} non valida")
        try:
            start = int(entity["start"])
            end = int(entity["end"])
            raw_label = str(entity["label"])
        except (KeyError, TypeError, ValueError) as exc:
            raise StudentTrainingError(f"Entita' {entity_index} priva di start/end/label") from exc
        if start < 0 or end <= start:
            raise StudentTrainingError(
                f"Entita' {entity_index} con offset non valido: {start}:{end}"
            )
        entity_type = _normalized_entity_type(raw_label)
        if entity_type is None:
            continue
        if f"B-{entity_type}" not in label2id or f"I-{entity_type}" not in label2id:
            raise StudentTrainingError(f"Tipo entita' non previsto dal teacher: {entity_type}")
        normalized_entities.append((start, end, entity_type))
    normalized_entities.sort()

    for previous, current in zip(normalized_entities, normalized_entities[1:]):
        if current[0] < previous[1]:
            raise StudentTrainingError(
                "Entita' sovrapposte non supportate: "
                f"{previous[0]}:{previous[1]} e {current[0]}:{current[1]}"
            )

    token_labels: list[str] = []
    token_entity_indexes: list[int | None] = []
    for token_index, raw_offset in enumerate(offsets):
        if len(raw_offset) != 2:
            raise StudentTrainingError(f"Offset tokenizer non valido: {raw_offset!r}")
        start, end = int(raw_offset[0]), int(raw_offset[1])
        is_special = (
            bool(special_tokens_mask[token_index])
            if special_tokens_mask is not None
            else end <= start
        )
        if is_special or end <= start:
            token_labels.append("O")
            token_entity_indexes.append(None)
            continue

        matches = [
            (index, entity_type)
            for index, (entity_start, entity_end, entity_type) in enumerate(normalized_entities)
            if start < entity_end and end > entity_start
        ]
        if len(matches) > 1:
            raise StudentTrainingError(
                f"Token {token_index} ({start}:{end}) interseca piu' entita'"
            )
        if not matches:
            token_labels.append("O")
            token_entity_indexes.append(None)
            continue
        entity_index, entity_type = matches[0]
        prefix = "I-" if entity_index in token_entity_indexes else "B-"
        token_labels.append(prefix + entity_type)
        token_entity_indexes.append(entity_index)

    # Replica la policy storica di normalizzazione/fusione usata dal teacher e
    # dai gate correnti. Una futura correzione dei boundary sara' un esperimento
    # dati separato, non una modifica silenziosa del baseline.
    token_labels = normalize_labels(token_labels)
    aligned: list[int] = []
    for index, label in enumerate(token_labels):
        is_special = (
            bool(special_tokens_mask[index])
            if special_tokens_mask is not None
            else int(offsets[index][1]) <= int(offsets[index][0])
        )
        aligned.append(IGNORE_INDEX if is_special else int(label2id[label]))
    return aligned


def inspect_jsonl(
    path: str | Path,
    *,
    label2id: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Valida un JSONL token-classification e ne produce un'identita' stabile."""

    source = Path(path).resolve()
    rows = 0
    tokens = 0
    sources: Counter[str] = Counter()
    languages: Counter[str] = Counter()
    label_types: Counter[str] = Counter()

    try:
        handle = source.open("r", encoding="utf-8")
    except FileNotFoundError as exc:
        raise StudentTrainingError(f"Dataset non trovato: {source}") from exc

    with handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise StudentTrainingError(f"{source}:{line_number}: riga vuota")
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise StudentTrainingError(
                    f"{source}:{line_number}: JSON non valido"
                ) from exc
            if not isinstance(record, Mapping):
                raise StudentTrainingError(f"{source}:{line_number}: record non oggetto")
            row_tokens = record.get("tokens")
            row_labels = record.get("bio_labels")
            raw_entities = record.get("entities", record.get("privacy_mask"))
            if isinstance(row_tokens, list) and isinstance(row_labels, list):
                if not row_tokens or len(row_tokens) != len(row_labels):
                    raise StudentTrainingError(
                        f"{source}:{line_number}: tokens e label vuoti o disallineati"
                    )
                if not all(isinstance(value, str) for value in row_tokens + row_labels):
                    raise StudentTrainingError(
                        f"{source}:{line_number}: tokens e label devono essere stringhe"
                    )
                normalized = normalize_labels(row_labels)
                for label in normalized:
                    if label2id is not None and label not in label2id:
                        raise StudentTrainingError(
                            f"{source}:{line_number}: label non prevista dal teacher: {label}"
                        )
                    label_types[label[2:] if label != "O" else "O"] += 1
                tokens += len(row_tokens)
            elif isinstance(record.get("source_text"), str) and isinstance(raw_entities, list):
                for entity in raw_entities:
                    if not isinstance(entity, Mapping) or not isinstance(entity.get("label"), str):
                        raise StudentTrainingError(
                            f"{source}:{line_number}: entita' raw non valida"
                        )
                    entity_type = _normalized_entity_type(str(entity["label"]))
                    if entity_type is None:
                        continue
                    if label2id is not None and (
                        f"B-{entity_type}" not in label2id or f"I-{entity_type}" not in label2id
                    ):
                        raise StudentTrainingError(
                            f"{source}:{line_number}: tipo non previsto: {entity_type}"
                        )
                    label_types[entity_type] += 1
            else:
                raise StudentTrainingError(
                    f"{source}:{line_number}: schema token/BIO o raw-span assente"
                )
            if isinstance(record.get("source"), str):
                sources[str(record["source"])] += 1
            if isinstance(record.get("language"), str):
                languages[str(record["language"])] += 1
            rows += 1

    return {
        "path": str(source),
        "bytes": source.stat().st_size,
        "sha256": sha256_file(source),
        "rows": rows,
        "tokens": tokens,
        "sources": dict(sorted(sources.items())),
        "languages": dict(sorted(languages.items())),
        "label_tokens": dict(sorted(label_types.items())),
    }


def inspect_parquet(
    path: str | Path,
    *,
    label2id: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Scansiona Parquet a batch senza materializzare il corpus in RAM."""

    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise StudentTrainingError("pyarrow e' richiesto per ispezionare Parquet") from exc

    source = Path(path).resolve()
    if not source.is_file():
        raise StudentTrainingError(f"Dataset non trovato: {source}")
    parquet_file = parquet.ParquetFile(source)
    names = set(parquet_file.schema_arrow.names)
    is_legacy = {"tokens", "bio_labels"} <= names
    span_field = "entities" if "entities" in names else "privacy_mask" if "privacy_mask" in names else None
    is_raw = "source_text" in names and span_field is not None
    if not is_legacy and not is_raw:
        raise StudentTrainingError(f"Schema Parquet non supportato: {sorted(names)}")

    columns = []
    for candidate in ("tokens", "bio_labels", span_field, "source", "language"):
        if candidate and candidate in names and candidate not in columns:
            columns.append(candidate)
    rows = 0
    tokens = 0
    sources: Counter[str] = Counter()
    languages: Counter[str] = Counter()
    label_types: Counter[str] = Counter()
    for batch in parquet_file.iter_batches(batch_size=2048, columns=columns):
        for record in batch.to_pylist():
            if not is_raw:
                row_tokens = record["tokens"]
                row_labels = record["bio_labels"]
                if not row_tokens or len(row_tokens) != len(row_labels):
                    raise StudentTrainingError(
                        f"{source}: riga Parquet {rows} tokens/label disallineati"
                    )
                normalized = normalize_labels(row_labels)
                for label in normalized:
                    if label2id is not None and label not in label2id:
                        raise StudentTrainingError(
                            f"{source}: label Parquet non prevista: {label}"
                        )
                    label_types[label[2:] if label != "O" else "O"] += 1
                tokens += len(row_tokens)
            else:
                for entity in record.get(span_field) or []:
                    entity_type = _normalized_entity_type(str(entity["label"]))
                    if entity_type is None:
                        continue
                    if label2id is not None and (
                        f"B-{entity_type}" not in label2id or f"I-{entity_type}" not in label2id
                    ):
                        raise StudentTrainingError(
                            f"{source}: tipo Parquet non previsto: {entity_type}"
                        )
                    label_types[entity_type] += 1
            if isinstance(record.get("source"), str):
                sources[str(record["source"])] += 1
            if isinstance(record.get("language"), str):
                languages[str(record["language"])] += 1
            rows += 1

    schema_text = str(parquet_file.schema_arrow)
    return {
        "path": str(source),
        "format": "parquet",
        "bytes": source.stat().st_size,
        "sha256": sha256_file(source),
        "rows": rows,
        "tokens": tokens,
        "schema_sha256": hashlib.sha256(schema_text.encode("utf-8")).hexdigest(),
        "sources": dict(sorted(sources.items())),
        "languages": dict(sorted(languages.items())),
        "label_tokens": dict(sorted(label_types.items())),
    }


def inspect_training_source(
    path: str | Path,
    *,
    label2id: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    source = Path(path)
    if source.suffix.lower() in (".parquet", ".pq"):
        return inspect_parquet(source, label2id=label2id)
    return inspect_jsonl(source, label2id=label2id)


def file_manifest(directory: str | Path) -> list[dict[str, Any]]:
    """Hash deterministici dei file sorgente del backbone/tokenizer."""

    root = Path(directory).resolve()
    if not root.is_dir():
        raise StudentTrainingError(f"Directory modello non trovata: {root}")
    result: list[dict[str, Any]] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root)
        if relative.parts and relative.parts[0] == ".cache":
            continue
        result.append(
            {
                "path": relative.as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    if not result:
        raise StudentTrainingError(f"Directory modello vuota: {root}")
    return result


def build_run_manifest(
    *,
    model_source: str | Path,
    model_repo: str,
    model_revision: str,
    teacher_config: str | Path,
    train_path: str | Path,
    validation_path: str | Path,
    training_configuration: Mapping[str, Any],
    software: Mapping[str, Any],
) -> dict[str, Any]:
    """Costruisce il manifest che lega checkpoint, dati e recipe del run."""

    teacher = Path(teacher_config).resolve()
    label2id, _ = load_label_contract(teacher)
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "backbone": {
            "repo_id": model_repo,
            "revision": model_revision,
            "path": str(Path(model_source).resolve()),
            "files": file_manifest(model_source),
        },
        "teacher_label_contract": {
            "path": str(teacher),
            "sha256": sha256_file(teacher),
        },
        "data": {
            "train": inspect_training_source(train_path, label2id=label2id),
            "validation": inspect_training_source(validation_path, label2id=label2id),
        },
        "training": dict(training_configuration),
        "software": dict(software),
    }


def prediction_documents(
    prediction_ids: Iterable[Sequence[int]],
    label_ids: Iterable[Sequence[int]],
    id2label: Mapping[int, str],
) -> tuple[list[list[str]], list[list[str]]]:
    """Converte gli array del Trainer in documenti BIO, ignorando padding/speciali."""

    gold_documents: list[list[str]] = []
    predicted_documents: list[list[str]] = []
    for row_index, (predicted, golden) in enumerate(zip(prediction_ids, label_ids)):
        gold_row: list[str] = []
        prediction_row: list[str] = []
        if len(predicted) != len(golden):
            raise StudentTrainingError(
                f"Predizioni/label disallineate alla riga {row_index}"
            )
        for prediction_id, label_id in zip(predicted, golden):
            label_index = int(label_id)
            if label_index == IGNORE_INDEX:
                continue
            prediction_index = int(prediction_id)
            try:
                gold_row.append(id2label[label_index])
                prediction_row.append(id2label[prediction_index])
            except KeyError as exc:
                raise StudentTrainingError(
                    f"ID label fuori contratto alla riga {row_index}: {exc.args[0]}"
                ) from exc
        gold_documents.append(gold_row)
        predicted_documents.append(prediction_row)
    return gold_documents, predicted_documents
