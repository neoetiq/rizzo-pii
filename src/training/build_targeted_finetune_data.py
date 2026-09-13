"""Build deterministic targeted fine-tuning and sealed evaluation datasets.

The builder is deliberately model-free.  It mines only gold annotations from
immutable clean Parquet sources, excludes every previously materialized record
and skeleton, and publishes the four datasets plus one provenance manifest as
an atomic, immutable directory.

V1 outputs:

* ``d1-targeted-train-1024.jsonl``: all novel ``ID_DOC`` ``n. + 7 digits``
  records, matched contextual ``DOCID`` hard negatives, and protected replay;
* ``d0-control-train-1024.jsonl``: a disjoint control matched by character
  length bin and protected focus tag, without the target-shape oversampling;
* ``sealed-global-2048.jsonl``: representative validation holdout;
* ``sealed-id-doc-2048.jsonl``: disjoint ``ID_DOC`` challenge containing every
  remaining novel target-shape record before deterministic fill.

V2 is a separate, synthetic-context ablation.  Its training split contains
128 synthetic ``ID_DOC`` positives (32 each for n.+7 digits, CIE, passport and
driving-licence surfaces), 256 surface/context hard negatives, and 640 real
replay records.  Its sealed challenge uses template families that never occur
in training.  Synthetic spans always cover the identifier value only: harmless
numbering cues such as ``n.`` remain outside the gold span.

No model is loaded and no holdout is evaluated by this module.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from src.quantization.metrics import DROP_TYPES, TAG_MAP
from src.training.student_utils import StudentTrainingError, sha256_file


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAIN_PARQUET = (
    ROOT / "artifacts/training/sources/clean/data/train-00000-of-00016.parquet"
)
DEFAULT_VALIDATION_PARQUET = (
    ROOT / "artifacts/training/sources/clean/data/validation-00000-of-00001.parquet"
)
DEFAULT_OBSERVED_TRAIN = ROOT / "artifacts/training/data/clean-train-1024.jsonl"
DEFAULT_OBSERVED_VALIDATION = (
    ROOT / "artifacts/training/data/clean-validation-512.jsonl"
)
DEFAULT_OUTPUT_DIR = ROOT / "artifacts/training/data/id-doc-targeted-ft-v1"
DEFAULT_V2_OUTPUT_DIR = ROOT / "artifacts/training/data/id-doc-targeted-ft-v2"

CLEAN_REVISION = "50163bcda973efe818004d053cfa159f0927663f"
EXPECTED_TRAIN_SHA256 = "bc1467a66485b621d3a5077c7593a86fb92d05572f5da79c634eab2611e9b7e2"
EXPECTED_VALIDATION_SHA256 = (
    "30960e940acb44fd7217d2ba6e6c3df0999c04cdc07778b500ba36f40bb44041"
)
DEFAULT_SEED = 20260822
DEFAULT_SALT = "rizzo-pii-id-doc-targeted-ft-v1-20260822"
DEFAULT_V2_SALT = "rizzo-pii-id-doc-targeted-ft-v2-20260822"

TARGET_PATTERN_TEXT = r"(?i)^\s*n\.\s*\d{7}\s*$"
TARGET_PATTERN = re.compile(TARGET_PATTERN_TEXT)
HARD_NEGATIVE_VARIANTS = ("sentenza-n", "protocollo", "rg")
PROTECTED_TAGS = (
    "ID_DOC",
    "DOCID",
    "CF",
    "PIVA",
    "IBAN",
    "EMAIL",
    "TELEPHONENUM",
)
LENGTH_BINS = (
    (512, "0000-0511"),
    (1024, "0512-1023"),
    (1536, "1024-1535"),
    (2**63 - 1, "1536-plus"),
)

OUTPUT_FILENAMES = {
    "d1_targeted": "d1-targeted-train-1024.jsonl",
    "d0_control": "d0-control-train-1024.jsonl",
    "sealed_global": "sealed-global-2048.jsonl",
    "sealed_id_doc": "sealed-id-doc-2048.jsonl",
}

D2_OUTPUT_FILENAMES = {
    "d2_targeted": "d2-synthetic-targeted-train-1024.jsonl",
    "sealed_synthetic_challenge": "sealed-synthetic-id-doc-challenge-384.jsonl",
}

D2_FORMAT_FAMILIES = (
    "numeric-7",
    "cie-2l5d2l",
    "passport-2l7d",
    "driving-2l7d1l",
)
D2_NUMERIC_TRAP_LENGTHS = (5, 6, 7, 8, 12)
D2_CANONICAL_SPAN_POLICY = {
    "id": "identifier-value-only-v1",
    "gold_span": "identifier payload only",
    "outside_gold": [
        "document-type cue",
        "numbering cue (n., N., n°, Nº, nr., numero)",
        "separator punctuation and whitespace",
    ],
    "reason": (
        "redaction of the identifier payload is privacy-complete; harmless cue "
        "coverage is not required and must not drive type classification"
    ),
}
D2_EVALUATION_VIEWS = {
    "exact_span": "standard entity type plus exact character boundaries",
    "payload_sensitive_span": (
        "entity type plus identifier payload; differences limited to an adjacent "
        "harmless numbering cue are boundary-only, not a privacy miss"
    ),
}

D2_FORMAT_SOURCES = {
    "numeric-7": {
        "basis": "observed clean ID_DOC target family",
        "pattern": "7 digits",
    },
    "cie-2l5d2l": {
        "basis": "official CIE serial shape",
        "pattern": "2 letters, 5 digits, 2 letters",
        "url": (
            "https://www.cartaidentita.interno.gov.it/"
            "cose-la-carta/caratteristiche-del-documento/"
        ),
    },
    "passport-2l7d": {
        "basis": "repository/data surface family; syntactic detector example only",
        "pattern": "2 letters, 7 digits",
    },
    "driving-2l7d1l": {
        "basis": "existing repository generator surface family",
        "pattern": "2 letters, 7 digits, 1 letter",
    },
}


@dataclass(frozen=True)
class BuildConfig:
    train_parquet: Path = DEFAULT_TRAIN_PARQUET
    validation_parquet: Path = DEFAULT_VALIDATION_PARQUET
    observed_train: Path = DEFAULT_OBSERVED_TRAIN
    observed_validation: Path = DEFAULT_OBSERVED_VALIDATION
    output_dir: Path = DEFAULT_OUTPUT_DIR
    clean_revision: str = CLEAN_REVISION
    seed: int = DEFAULT_SEED
    salt: str = DEFAULT_SALT
    d1_rows: int = 1024
    d0_rows: int = 1024
    holdout_global_rows: int = 2048
    holdout_id_doc_rows: int = 2048
    replay_protected_floor: int = 32
    expected_target_train_docs: int | None = 226
    expected_target_train_entities: int | None = 237
    # Validation has 85/95 target docs/entities. Two target records are in
    # validation-512 and one additional record has a skeleton already exposed
    # there, leaving the genuinely novel 82/92 required by the sealed policy.
    expected_target_holdout_docs: int | None = 82
    expected_target_holdout_entities: int | None = 92
    expected_train_sha256: str | None = EXPECTED_TRAIN_SHA256
    expected_validation_sha256: str | None = EXPECTED_VALIDATION_SHA256


@dataclass(frozen=True)
class D2BuildConfig:
    prior_v1_dir: Path = DEFAULT_OUTPUT_DIR
    output_dir: Path = DEFAULT_V2_OUTPUT_DIR
    seed: int = DEFAULT_SEED
    salt: str = DEFAULT_V2_SALT
    train_positive_rows: int = 128
    train_hard_negative_rows: int = 256
    replay_rows: int = 640
    challenge_positive_rows: int = 128
    challenge_hard_negative_rows: int = 256
    replay_numeric_iban_floor: int = 64
    replay_alphanumeric_id_floor: int = 64


@dataclass(frozen=True)
class D2TemplateFamily:
    family_id: str
    document_family: str
    format_family: str
    positive_core: str
    exact_negative_family: str
    exact_negative_core: str
    boundary_negative_family: str
    boundary_negative_core: str


@dataclass(frozen=True)
class RecordMeta:
    record_id: str
    source_row: int
    split: str
    skeleton_sha256: str
    chars: int
    length_bin: str
    tags: tuple[str, ...]
    entity_counts: tuple[tuple[str, int], ...]
    target_entities: int
    id_doc_entities: int
    hard_negative_variant: str | None


@dataclass(frozen=True)
class SelectedRecord:
    meta: RecordMeta
    stratum: str
    focus: str
    match_level: str | None = None


@dataclass(frozen=True)
class ObservedSet:
    path: Path
    rows: int
    record_ids: frozenset[str]
    skeletons: frozenset[str]
    identity: Mapping[str, Any]


D2_TRAIN_SUBJECTS = (
    "il richiedente",
    "la parte presente",
    "il dichiarante",
    "la persona interessata",
)
D2_TRAIN_TAILS = (
    "acquisito agli atti",
    "verificato allo sportello",
    "riportato nel modulo",
    "allegato alla richiesta",
)
D2_CHALLENGE_SUBJECTS = (
    "il soggetto comparso",
    "la persona identificata",
    "il titolare indicato",
    "l'utente allo sportello",
)
D2_CHALLENGE_TAILS = (
    "registrato nel verbale",
    "trascritto nella scheda",
    "annotato nel fascicolo",
    "confermato durante il controllo",
)

D2_DOCUMENT_CUES = {
    "numeric-7": (
        "documento d'identità",
        "carta d'identita",
        "C.I.",
        "CI",
        "documento esibito",
        "estremi del documento",
        "DOCUMENTO DI IDENTITÀ",
        "doc. identita",
    ),
    "cie-2l5d2l": (
        "CIE",
        "carta d'identità elettronica",
        "C.I.E.",
        "cie",
        "carta identita elettronica",
        "C I E",
        "seriale CIE",
        "C.l.E.",
    ),
    "passport-2l7d": (
        "passaporto",
        "PASSAPORTO",
        "documento di viaggio",
        "passaporto italiano",
        "estremi passaporto",
        "passap0rto",
        "doc. viaggio",
        "titolo di viaggio",
    ),
    "driving-2l7d1l": (
        "patente",
        "patente di guida",
        "PATENTE",
        "licenza di guida",
        "estremi patente",
        "patentc",
        "doc. di guida",
        "titolo di guida",
    ),
}

# Every cue is outside the synthetic gold span.  Whitespace, punctuation,
# casing and common OCR artefacts are still represented in the input context.
D2_NUMBERING_CUES = (
    "n. ",
    "N. ",
    "n° ",
    "Nº ",
    "nr. ",
    "numero ",
    ": ",
    "\nn . ",
)
D2_SURFACE_VARIANTS = (
    "compact-upper",
    "compact-lower",
    "grouped-space",
    "grouped-hyphen",
)

D2_TRAIN_TEMPLATE_FAMILIES = (
    D2TemplateFamily(
        "train-numeric-document-presented",
        "document-presented",
        "numeric-7",
        "{subject}: {doc_cue} {cue}{identifier}; {tail}.",
        "train-negative-sentence",
        "{subject}: sentenza {cue}{identifier}; {tail}.",
        "train-boundary-protocol",
        "Protocollo {cue}{identifier}, riferimento {decoy}; {tail}.",
    ),
    D2TemplateFamily(
        "train-numeric-document-details",
        "document-details",
        "numeric-7",
        "Per {subject}, {doc_cue} {cue}{identifier}, {tail}.",
        "train-negative-rg",
        "Per {subject}, R.G. {cue}{identifier}, {tail}.",
        "train-boundary-repertory",
        "Repertorio {cue}{identifier}; frammento {decoy}; {tail}.",
    ),
    D2TemplateFamily(
        "train-cie-front-office",
        "cie",
        "cie-2l5d2l",
        "{doc_cue} di {subject} {cue}{identifier}; {tail}.",
        "train-negative-electronic-practice",
        "Pratica elettronica di {subject} {cue}{identifier}; {tail}.",
        "train-boundary-register",
        "Registro pratica {cue}{identifier}, codice {decoy}; {tail}.",
    ),
    D2TemplateFamily(
        "train-cie-identification",
        "cie",
        "cie-2l5d2l",
        "Identificazione di {subject} mediante {doc_cue} {cue}{identifier}; {tail}.",
        "train-negative-decree",
        "Decreto riferito a {subject} {cue}{identifier}; {tail}.",
        "train-boundary-order",
        "Ordine amministrativo {cue}{identifier}; nota {decoy}; {tail}.",
    ),
    D2TemplateFamily(
        "train-passport-check",
        "passport",
        "passport-2l7d",
        "{doc_cue} di {subject} {cue}{identifier}; {tail}.",
        "train-negative-case-file",
        "Fascicolo di {subject} {cue}{identifier}; {tail}.",
        "train-boundary-policy",
        "Polizza {cue}{identifier}, appendice {decoy}; {tail}.",
    ),
    D2TemplateFamily(
        "train-passport-details",
        "passport",
        "passport-2l7d",
        "Estremi {doc_cue} per {subject} {cue}{identifier}; {tail}.",
        "train-negative-application",
        "Istanza di {subject} {cue}{identifier}; {tail}.",
        "train-boundary-contract",
        "Contratto {cue}{identifier}; sezione {decoy}; {tail}.",
    ),
    D2TemplateFamily(
        "train-driving-control",
        "driving-licence",
        "driving-2l7d1l",
        "{subject}, {doc_cue} {cue}{identifier}; {tail}.",
        "train-negative-report",
        "{subject}, verbale {cue}{identifier}; {tail}.",
        "train-boundary-invoice",
        "Fattura {cue}{identifier}, voce {decoy}; {tail}.",
    ),
    D2TemplateFamily(
        "train-driving-details",
        "driving-licence",
        "driving-2l7d1l",
        "Controllati gli estremi della {doc_cue} di {subject} {cue}{identifier}; {tail}.",
        "train-negative-notice",
        "Controllati gli estremi dell'avviso di {subject} {cue}{identifier}; {tail}.",
        "train-boundary-proceeding",
        "Procedimento {cue}{identifier}; allegato {decoy}; {tail}.",
    ),
)

D2_CHALLENGE_TEMPLATE_FAMILIES = (
    D2TemplateFamily(
        "challenge-numeric-reception",
        "document-presented",
        "numeric-7",
        "All'accettazione {subject} mostra il {doc_cue} {cue}{identifier}; {tail}.",
        "challenge-negative-judgment",
        "All'accettazione per {subject} si registra il provvedimento {cue}{identifier}; {tail}.",
        "challenge-boundary-docket",
        "Numero di ruolo {cue}{identifier}; residuo OCR {decoy}; {tail}.",
    ),
    D2TemplateFamily(
        "challenge-numeric-form",
        "document-details",
        "numeric-7",
        "Nel campo identificazione di {subject}: {doc_cue} {cue}{identifier}; {tail}.",
        "challenge-negative-decision",
        "Nel campo decisione di {subject}: atto {cue}{identifier}; {tail}.",
        "challenge-boundary-filing",
        "Deposito {cue}{identifier}, traccia {decoy}; {tail}.",
    ),
    D2TemplateFamily(
        "challenge-cie-serial",
        "cie",
        "cie-2l5d2l",
        "Il seriale del {doc_cue} consegnato da {subject} è {cue}{identifier}; {tail}.",
        "challenge-negative-file-serial",
        "Il seriale del fascicolo di {subject} è {cue}{identifier}; {tail}.",
        "challenge-boundary-ledger",
        "Registro interno {cue}{identifier}; sigla {decoy}; {tail}.",
    ),
    D2TemplateFamily(
        "challenge-cie-ocr-line",
        "cie",
        "cie-2l5d2l",
        "Lettura OCR per {subject}\n{doc_cue} {cue}{identifier}\n{tail}.",
        "challenge-negative-ocr-line",
        "Lettura OCR atto per {subject}\npratica {cue}{identifier}\n{tail}.",
        "challenge-boundary-archive",
        "Archivio\n{cue}{identifier}\nframmento {decoy}; {tail}.",
    ),
    D2TemplateFamily(
        "challenge-passport-border",
        "passport",
        "passport-2l7d",
        "Per {subject} è stato letto il {doc_cue} {cue}{identifier}; {tail}.",
        "challenge-negative-cross-border-file",
        "Per {subject} è stato letto il fascicolo estero {cue}{identifier}; {tail}.",
        "challenge-boundary-claim",
        "Sinistro estero {cue}{identifier}, nota {decoy}; {tail}.",
    ),
    D2TemplateFamily(
        "challenge-passport-transcription",
        "passport",
        "passport-2l7d",
        "Trascrizione per {subject}: {doc_cue} {cue}{identifier}; {tail}.",
        "challenge-negative-transcription",
        "Trascrizione atto per {subject}: repertorio {cue}{identifier}; {tail}.",
        "challenge-boundary-booking",
        "Prenotazione {cue}{identifier}; token {decoy}; {tail}.",
    ),
    D2TemplateFamily(
        "challenge-driving-roadside",
        "driving-licence",
        "driving-2l7d1l",
        "Durante il controllo {subject} presenta la {doc_cue} {cue}{identifier}; {tail}.",
        "challenge-negative-road-report",
        "Durante il controllo di {subject} si compila il rapporto {cue}{identifier}; {tail}.",
        "challenge-boundary-ticket",
        "Sanzione {cue}{identifier}; tag {decoy}; {tail}.",
    ),
    D2TemplateFamily(
        "challenge-driving-registry",
        "driving-licence",
        "driving-2l7d1l",
        "Nel registro di {subject}, {doc_cue} {cue}{identifier}; {tail}.",
        "challenge-negative-administrative-register",
        "Nel registro amministrativo di {subject}, pratica {cue}{identifier}; {tail}.",
        "challenge-boundary-case",
        "Posizione amministrativa {cue}{identifier}; residuo {decoy}; {tail}.",
    ),
)


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")


def _normalized_type(raw_label: str) -> str | None:
    label = raw_label[2:] if raw_label.startswith(("B-", "I-")) else raw_label
    normalized = TAG_MAP.get(label, label)
    return None if normalized in DROP_TYPES else normalized


def _length_bin(chars: int) -> str:
    for upper, name in LENGTH_BINS:
        if chars < upper:
            return name
    raise AssertionError("length bin non raggiungibile")


def _canonical_entities(
    raw_entities: Any, *, text: str, context: str
) -> list[dict[str, Any]]:
    if not isinstance(raw_entities, list):
        raise StudentTrainingError(f"{context}: entities mancanti")
    entities: list[dict[str, Any]] = []
    for index, raw_entity in enumerate(raw_entities):
        if not isinstance(raw_entity, Mapping):
            raise StudentTrainingError(f"{context}: entita' {index} non valida")
        try:
            start = int(raw_entity["start"])
            end = int(raw_entity["end"])
            label = str(raw_entity["label"])
        except (KeyError, TypeError, ValueError) as exc:
            raise StudentTrainingError(
                f"{context}: entita' {index} priva di start/end/label"
            ) from exc
        if start < 0 or end <= start or end > len(text) or not label:
            raise StudentTrainingError(
                f"{context}: entita' {index} fuori testo ({start}:{end})"
            )
        entities.append({"start": start, "end": end, "label": label})
    entities.sort(key=lambda item: (item["start"], item["end"], item["label"]))
    for previous, current in zip(entities, entities[1:]):
        if current["start"] < previous["end"]:
            raise StudentTrainingError(
                f"{context}: entita' sovrapposte "
                f"{previous['start']}:{previous['end']} e "
                f"{current['start']}:{current['end']}"
            )
    return entities


def _identity_payload(raw: Mapping[str, Any], *, context: str) -> dict[str, Any]:
    text = raw.get("source_text")
    if not isinstance(text, str) or not text:
        raise StudentTrainingError(f"{context}: source_text mancante")
    return {
        "source_text": text,
        "entities": _canonical_entities(raw.get("entities"), text=text, context=context),
        "language": str(raw.get("language") or ""),
        "template_id": raw.get("template_id"),
    }


def _record_id(identity: Mapping[str, Any]) -> str:
    # This is intentionally byte-compatible with build_student_pilot.py.
    return hashlib.sha256(_canonical_json_bytes(identity)).hexdigest()


def _skeleton_sha256(identity: Mapping[str, Any]) -> str:
    text = str(identity["source_text"])
    entities = identity["entities"]
    for entity in sorted(entities, key=lambda item: int(item["start"]), reverse=True):
        normalized = _normalized_type(str(entity["label"]))
        replacement = "O" if normalized is None else normalized
        text = (
            text[: int(entity["start"])]
            + f"<{replacement}>"
            + text[int(entity["end"]) :]
        )
    canonical = " ".join(text.split())
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _target_entity_count(identity: Mapping[str, Any]) -> int:
    text = str(identity["source_text"])
    return sum(
        1
        for entity in identity["entities"]
        if _normalized_type(str(entity["label"])) == "ID_DOC"
        and TARGET_PATTERN.fullmatch(text[int(entity["start"]) : int(entity["end"])])
        is not None
    )


def _hard_negative_variant(identity: Mapping[str, Any]) -> str | None:
    text = str(identity["source_text"])
    found: set[str] = set()
    for entity in identity["entities"]:
        if _normalized_type(str(entity["label"])) != "DOCID":
            continue
        start, end = int(entity["start"]), int(entity["end"])
        value = text[start:end].casefold().strip()
        context = text[max(0, start - 40) : end].casefold()
        if re.search(r"\bsent(?:enza)?\.?\s*n\.", value) or re.search(
            r"\bsent(?:enza)?\.?\s*n\.\s*$", context
        ):
            found.add("sentenza-n")
        if re.search(r"\bprot(?:ocollo)?\.?", value) or re.search(
            r"\bprot(?:ocollo)?\.?\s*$", context
        ):
            found.add("protocollo")
        if re.search(r"(?:^|\s)r\.?\s*g\.?(?:\s|$)", value) or re.search(
            r"(?:^|\s)r\.?\s*g\.?\s*$", context
        ):
            found.add("rg")
    return next((variant for variant in HARD_NEGATIVE_VARIANTS if variant in found), None)


def _record_meta(
    raw: Mapping[str, Any], *, source_row: int, split: str
) -> tuple[RecordMeta, dict[str, Any]]:
    identity = _identity_payload(raw, context=f"{split} riga {source_row}")
    counts: Counter[str] = Counter()
    for entity in identity["entities"]:
        normalized = _normalized_type(str(entity["label"]))
        if normalized is not None:
            counts[normalized] += 1
    target_entities = _target_entity_count(identity)
    meta = RecordMeta(
        record_id=_record_id(identity),
        source_row=source_row,
        split=split,
        skeleton_sha256=_skeleton_sha256(identity),
        chars=len(str(identity["source_text"])),
        length_bin=_length_bin(len(str(identity["source_text"]))),
        tags=tuple(sorted(counts)),
        entity_counts=tuple(sorted(counts.items())),
        target_entities=target_entities,
        id_doc_entities=int(counts.get("ID_DOC", 0)),
        hard_negative_variant=_hard_negative_variant(identity),
    )
    return meta, identity


def _scan_parquet(path: Path, *, split: str) -> tuple[list[RecordMeta], dict[str, Any]]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise StudentTrainingError("pyarrow e' richiesto dal builder targeted") from exc

    source = path.expanduser().resolve()
    if not source.is_file():
        raise StudentTrainingError(f"Parquet {split} non trovato: {source}")
    parquet_file = parquet.ParquetFile(source)
    required = {"source_text", "entities", "language", "template_id"}
    missing = required - set(parquet_file.schema_arrow.names)
    if missing:
        raise StudentTrainingError(
            f"Parquet {split} privo delle colonne {sorted(missing)}"
        )

    metas: list[RecordMeta] = []
    seen_ids: set[str] = set()
    duplicate_ids = 0
    columns = sorted(required)
    source_row = 0
    for batch in parquet_file.iter_batches(batch_size=2048, columns=columns):
        for raw in batch.to_pylist():
            meta, _ = _record_meta(raw, source_row=source_row, split=split)
            source_row += 1
            if meta.record_id in seen_ids:
                duplicate_ids += 1
                continue
            seen_ids.add(meta.record_id)
            metas.append(meta)
    if duplicate_ids:
        raise StudentTrainingError(
            f"Parquet {split}: {duplicate_ids} record canonici duplicati"
        )
    return metas, {
        "path": str(source),
        "bytes": source.stat().st_size,
        "sha256": sha256_file(source),
        "rows": len(metas),
        "schema_sha256": hashlib.sha256(
            str(parquet_file.schema_arrow).encode("utf-8")
        ).hexdigest(),
    }


def _load_observed(path: Path, *, description: str) -> ObservedSet:
    source = path.expanduser().resolve()
    if not source.is_file():
        raise StudentTrainingError(f"{description} non trovato: {source}")
    ids: set[str] = set()
    skeletons: set[str] = set()
    rows = 0
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise StudentTrainingError(
                    f"{description} {source}:{line_number}: riga vuota"
                )
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise StudentTrainingError(
                    f"{description} {source}:{line_number}: JSON non valido"
                ) from exc
            if not isinstance(raw, Mapping):
                raise StudentTrainingError(
                    f"{description} {source}:{line_number}: record non oggetto"
                )
            identity = _identity_payload(
                raw, context=f"{description} riga {line_number}"
            )
            computed_id = _record_id(identity)
            declared_id = raw.get("record_id")
            if declared_id is not None and str(declared_id) != computed_id:
                raise StudentTrainingError(
                    f"{description} {source}:{line_number}: record_id non canonico"
                )
            if computed_id in ids:
                raise StudentTrainingError(
                    f"{description}: record duplicato {computed_id}"
                )
            ids.add(computed_id)
            skeletons.add(_skeleton_sha256(identity))
            rows += 1
    return ObservedSet(
        path=source,
        rows=rows,
        record_ids=frozenset(ids),
        skeletons=frozenset(skeletons),
        identity={
            "path": str(source),
            "bytes": source.stat().st_size,
            "sha256": sha256_file(source),
            "rows": rows,
            "record_ids_sha256": hashlib.sha256(
                _canonical_json_bytes(sorted(ids))
            ).hexdigest(),
            "skeletons_sha256": hashlib.sha256(
                _canonical_json_bytes(sorted(skeletons))
            ).hexdigest(),
        },
    )


def _score(salt: str, record_id: str) -> bytes:
    return hashlib.sha256(f"{salt}:{record_id}".encode("ascii")).digest()


def _sorted_candidates(
    records: Iterable[RecordMeta], *, salt: str
) -> list[RecordMeta]:
    return sorted(records, key=lambda record: (_score(salt, record.record_id), record.record_id))


def _take_unique_skeletons(
    records: Iterable[RecordMeta],
    *,
    count: int,
    salt: str,
    forbidden_ids: set[str] | frozenset[str] | None = None,
    forbidden_skeletons: set[str] | frozenset[str] | None = None,
) -> list[RecordMeta]:
    if count < 0:
        raise StudentTrainingError("count di selezione negativo")
    used_ids = set(forbidden_ids or ())
    used_skeletons = set(forbidden_skeletons or ())
    selected: list[RecordMeta] = []
    for record in _sorted_candidates(records, salt=salt):
        if record.record_id in used_ids or record.skeleton_sha256 in used_skeletons:
            continue
        selected.append(record)
        used_ids.add(record.record_id)
        used_skeletons.add(record.skeleton_sha256)
        if len(selected) == count:
            break
    if len(selected) != count:
        raise StudentTrainingError(
            f"Selezione {salt}: richiesti {count}, disponibili {len(selected)}"
        )
    return selected


def _balanced_hard_negatives(
    candidates: Sequence[RecordMeta],
    *,
    count: int,
    salt: str,
    forbidden_ids: set[str],
    forbidden_skeletons: set[str],
) -> list[SelectedRecord]:
    groups: dict[str, list[RecordMeta]] = {
        variant: [
            record
            for record in candidates
            if record.hard_negative_variant == variant
        ]
        for variant in HARD_NEGATIVE_VARIANTS
    }
    base, remainder = divmod(count, len(HARD_NEGATIVE_VARIANTS))
    selected: list[SelectedRecord] = []
    used_ids = set(forbidden_ids)
    used_skeletons = set(forbidden_skeletons)
    for index, variant in enumerate(HARD_NEGATIVE_VARIANTS):
        quota = base + (1 if index < remainder else 0)
        picked = _take_unique_skeletons(
            groups[variant],
            count=quota,
            salt=f"{salt}:{variant}",
            forbidden_ids=used_ids,
            forbidden_skeletons=used_skeletons,
        )
        for record in picked:
            selected.append(
                SelectedRecord(record, f"hard-negative:{variant}", "DOCID")
            )
            used_ids.add(record.record_id)
            used_skeletons.add(record.skeleton_sha256)
    return selected


def _select_replay(
    candidates: Sequence[RecordMeta],
    *,
    count: int,
    protected_floor: int,
    salt: str,
    forbidden_ids: set[str],
    forbidden_skeletons: set[str],
) -> list[SelectedRecord]:
    minimum = protected_floor * len(PROTECTED_TAGS)
    if minimum > count:
        raise StudentTrainingError(
            f"Replay {count} insufficiente per floor protetto {minimum}"
        )
    used_ids = set(forbidden_ids)
    used_skeletons = set(forbidden_skeletons)
    selected: list[SelectedRecord] = []
    for tag in PROTECTED_TAGS:
        picked = _take_unique_skeletons(
            (record for record in candidates if tag in record.tags),
            count=protected_floor,
            salt=f"{salt}:protected:{tag}",
            forbidden_ids=used_ids,
            forbidden_skeletons=used_skeletons,
        )
        for record in picked:
            selected.append(
                SelectedRecord(record, f"replay-protected:{tag}", tag)
            )
            used_ids.add(record.record_id)
            used_skeletons.add(record.skeleton_sha256)
    remaining = count - len(selected)
    picked = _take_unique_skeletons(
        candidates,
        count=remaining,
        salt=f"{salt}:general",
        forbidden_ids=used_ids,
        forbidden_skeletons=used_skeletons,
    )
    selected.extend(
        SelectedRecord(record, "replay-general", "GENERAL") for record in picked
    )
    return selected


def _select_control(
    candidates: Sequence[RecordMeta],
    *,
    d1: Sequence[SelectedRecord],
    count: int,
    salt: str,
) -> list[SelectedRecord]:
    if count != len(d1):
        raise StudentTrainingError(
            "D0 e D1 devono avere lo stesso numero di record per il controllo"
        )
    quotas = Counter((item.meta.length_bin, item.focus) for item in d1)
    available = list(candidates)
    used_ids: set[str] = set()
    used_skeletons: set[str] = set()
    selected: list[SelectedRecord] = []

    def eligible(record: RecordMeta, *, length_bin: str | None, focus: str | None) -> bool:
        if record.record_id in used_ids or record.skeleton_sha256 in used_skeletons:
            return False
        if length_bin is not None and record.length_bin != length_bin:
            return False
        return focus in (None, "GENERAL") or focus in record.tags

    for (length_bin, focus), quota in sorted(quotas.items()):
        needed = quota
        levels = (
            ("exact-length-and-focus", length_bin, focus),
            ("focus-only", None, focus),
            ("length-only", length_bin, None),
            ("uniform-fallback", None, None),
        )
        for level, requested_bin, requested_focus in levels:
            if needed == 0:
                break
            pool = [
                record
                for record in available
                if eligible(
                    record, length_bin=requested_bin, focus=requested_focus
                )
            ]
            take = min(needed, len(pool))
            if take == 0:
                continue
            picked = _take_unique_skeletons(
                pool,
                count=take,
                salt=f"{salt}:{length_bin}:{focus}:{level}",
                forbidden_ids=used_ids,
                forbidden_skeletons=used_skeletons,
            )
            for record in picked:
                selected.append(
                    SelectedRecord(
                        record,
                        f"control-match:{focus}:{length_bin}",
                        focus,
                        match_level=level,
                    )
                )
                used_ids.add(record.record_id)
                used_skeletons.add(record.skeleton_sha256)
            needed -= len(picked)
        if needed:
            raise StudentTrainingError(
                f"Controllo: impossibile coprire {length_bin}/{focus}, mancanti {needed}"
            )
    if len(selected) != count:
        raise StudentTrainingError(
            f"Controllo: attesi {count} record, ottenuti {len(selected)}"
        )
    return selected


def _assert_expected(name: str, actual: int, expected: int | None) -> None:
    if expected is not None and actual != expected:
        raise StudentTrainingError(f"{name}: atteso {expected}, trovato {actual}")


def _selection_stats(items: Sequence[SelectedRecord]) -> dict[str, Any]:
    entity_counts: Counter[str] = Counter()
    strata: Counter[str] = Counter()
    length_bins: Counter[str] = Counter()
    target_entities = 0
    id_doc_entities = 0
    for item in items:
        entity_counts.update(dict(item.meta.entity_counts))
        strata[item.stratum] += 1
        length_bins[item.meta.length_bin] += 1
        target_entities += item.meta.target_entities
        id_doc_entities += item.meta.id_doc_entities
    return {
        "rows": len(items),
        "mean_chars": (
            sum(item.meta.chars for item in items) / len(items) if items else 0.0
        ),
        "target_shape_docs": sum(item.meta.target_entities > 0 for item in items),
        "target_shape_entities": target_entities,
        "id_doc_entities": id_doc_entities,
        "entity_counts": dict(sorted(entity_counts.items())),
        "strata": dict(sorted(strata.items())),
        "length_bins": dict(sorted(length_bins.items())),
        "match_levels": dict(
            sorted(
                Counter(
                    item.match_level
                    for item in items
                    if item.match_level is not None
                ).items()
            )
        ),
    }


def _pairwise_overlaps(
    groups: Mapping[str, Sequence[RecordMeta] | ObservedSet]
) -> dict[str, Any]:
    normalized: dict[str, tuple[set[str], set[str]]] = {}
    for name, value in groups.items():
        if isinstance(value, ObservedSet):
            normalized[name] = (set(value.record_ids), set(value.skeletons))
        else:
            normalized[name] = (
                {record.record_id for record in value},
                {record.skeleton_sha256 for record in value},
            )
    result: dict[str, Any] = {}
    names = list(normalized)
    all_zero = True
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            record_overlap = len(normalized[left][0] & normalized[right][0])
            skeleton_overlap = len(normalized[left][1] & normalized[right][1])
            result[f"{left}__{right}"] = {
                "record_id_overlap": record_overlap,
                "normalized_skeleton_overlap": skeleton_overlap,
            }
            all_zero = all_zero and record_overlap == 0 and skeleton_overlap == 0
    result["all_pairwise_zero"] = all_zero
    return result


def _d2_digest_bytes(seed: int, key: str, count: int) -> bytes:
    output = bytearray()
    counter = 0
    while len(output) < count:
        output.extend(
            hashlib.sha256(f"{seed}:{key}:{counter}".encode("utf-8")).digest()
        )
        counter += 1
    return bytes(output[:count])


def _d2_symbols(seed: int, key: str, alphabet: str, count: int) -> str:
    return "".join(
        alphabet[value % len(alphabet)]
        for value in _d2_digest_bytes(seed, key, count)
    )


def _d2_compact_identifier(
    *,
    seed: int,
    key: str,
    format_family: str,
    numeric_digits: int | None = None,
) -> tuple[str, tuple[int, ...]]:
    letters = "ABCDEFGHJKLMNPRSTUVWXYZ"
    digits = "0123456789"
    if format_family == "numeric-7":
        length = 7 if numeric_digits is None else int(numeric_digits)
        if length < 1:
            raise StudentTrainingError("Lunghezza identificatore numerico D2 non valida")
        value = _d2_symbols(seed, f"{key}:digits", digits, length)
        split = max(1, length // 2)
        return value, (split, length - split)
    if numeric_digits is not None:
        raise StudentTrainingError("numeric_digits ammesso solo per numeric-7")
    if format_family == "cie-2l5d2l":
        return (
            _d2_symbols(seed, f"{key}:l1", letters, 2)
            + _d2_symbols(seed, f"{key}:d", digits, 5)
            + _d2_symbols(seed, f"{key}:l2", letters, 2),
            (2, 5, 2),
        )
    if format_family == "passport-2l7d":
        return (
            _d2_symbols(seed, f"{key}:l", letters, 2)
            + _d2_symbols(seed, f"{key}:d", digits, 7),
            (2, 7),
        )
    if format_family == "driving-2l7d1l":
        return (
            _d2_symbols(seed, f"{key}:l1", letters, 2)
            + _d2_symbols(seed, f"{key}:d", digits, 7)
            + _d2_symbols(seed, f"{key}:l2", letters, 1),
            (2, 7, 1),
        )
    raise StudentTrainingError(f"Famiglia formato D2 sconosciuta: {format_family}")


def _d2_grouped(value: str, groups: Sequence[int], separator: str) -> str:
    pieces: list[str] = []
    position = 0
    for length in groups:
        pieces.append(value[position : position + int(length)])
        position += int(length)
    if position != len(value) or any(not piece for piece in pieces):
        raise StudentTrainingError("Gruppi superficie D2 non coerenti")
    return separator.join(pieces)


def _d2_surface(
    compact: str, groups: Sequence[int], variant: str
) -> str:
    if variant == "compact-upper":
        return compact.upper()
    if variant == "compact-lower":
        return compact.lower()
    if variant == "grouped-space":
        return _d2_grouped(compact.upper(), groups, " ")
    if variant == "grouped-hyphen":
        return _d2_grouped(compact.upper(), groups, "-")
    raise StudentTrainingError(f"Variante superficie D2 sconosciuta: {variant}")


def _d2_decoy(seed: int, key: str) -> str:
    letters = _d2_symbols(seed, f"{key}:letters", "ABCDEFGHJKLMNPRSTUVWXYZ", 3)
    digits = _d2_symbols(seed, f"{key}:digits", "0123456789", 4)
    return f"{letters}-{digits}-X"


def _d2_record(
    *,
    text: str,
    identifier: str,
    label: str,
    template_id: str,
    dataset_role: str,
    sealed: bool,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    if label not in {"ID_DOC", "DOCID"}:
        raise StudentTrainingError(f"Label sintetica D2 inattesa: {label}")
    if not identifier or text.count(identifier) != 1:
        raise StudentTrainingError(
            f"Il payload sintetico deve comparire una volta: {template_id}"
        )
    start = text.index(identifier)
    raw = {
        "source_text": text,
        "entities": [{"start": start, "end": start + len(identifier), "label": label}],
        "language": "it",
        "template_id": template_id,
    }
    identity = _identity_payload(raw, context=f"sintetico D2 {template_id}")
    record_id = _record_id(identity)
    return {
        "record_id": record_id,
        "source": "synthetic-d2",
        **identity,
        "dataset_role": dataset_role,
        "selection_stratum": str(metadata["selection_stratum"]),
        "sealed": bool(sealed),
        "synthetic": True,
        "contains_real_pii": False,
        "gold_span_policy": D2_CANONICAL_SPAN_POLICY["id"],
        "identifier_payload": {
            "start": start,
            "end": start + len(identifier),
            "sha256": hashlib.sha256(identifier.encode("utf-8")).hexdigest(),
        },
        **dict(metadata),
    }


def _d2_template_sets(
    split: str,
) -> tuple[
    tuple[D2TemplateFamily, ...],
    tuple[str, ...],
    tuple[str, ...],
]:
    if split == "train":
        return D2_TRAIN_TEMPLATE_FAMILIES, D2_TRAIN_SUBJECTS, D2_TRAIN_TAILS
    if split == "challenge":
        return (
            D2_CHALLENGE_TEMPLATE_FAMILIES,
            D2_CHALLENGE_SUBJECTS,
            D2_CHALLENGE_TAILS,
        )
    raise StudentTrainingError(f"Split sintetico D2 sconosciuto: {split}")


def _d2_template_family_ids(
    specs: Sequence[D2TemplateFamily],
) -> set[str]:
    result: set[str] = set()
    for spec in specs:
        result.update(
            {
                spec.family_id,
                spec.exact_negative_family,
                spec.boundary_negative_family,
            }
        )
    return result


def _generate_d2_synthetic(
    *,
    config: D2BuildConfig,
    split: str,
    positive_rows: int,
    hard_negative_rows: int,
    dataset_role: str,
    sealed: bool,
) -> list[dict[str, Any]]:
    specs, subjects, tails = _d2_template_sets(split)
    if positive_rows < 1 or positive_rows % len(specs):
        raise StudentTrainingError(
            f"Positivi D2 {split} devono essere multipli di {len(specs)}"
        )
    if hard_negative_rows != positive_rows * 2:
        raise StudentTrainingError(
            f"D2 {split} richiede due hard-negative per positivo"
        )
    per_spec = positive_rows // len(specs)
    combinations = len(subjects) * len(tails)
    if per_spec > combinations:
        raise StudentTrainingError(
            f"Template D2 {split} insufficienti: {per_spec} > {combinations}"
        )

    generated: list[dict[str, Any]] = []
    family_counts: Counter[str] = Counter()
    for spec_index, spec in enumerate(specs):
        for local_index in range(per_spec):
            key = f"{config.salt}:seed={config.seed}:{split}:{spec.family_id}:{local_index}"
            compact, groups = _d2_compact_identifier(
                seed=config.seed,
                key=key,
                format_family=spec.format_family,
            )
            variant = D2_SURFACE_VARIANTS[(local_index + spec_index) % len(D2_SURFACE_VARIANTS)]
            identifier = _d2_surface(compact, groups, variant)
            cue_index = (local_index * 3 + spec_index) % len(D2_NUMBERING_CUES)
            cue = D2_NUMBERING_CUES[cue_index]
            doc_cues = D2_DOCUMENT_CUES[spec.format_family]
            doc_cue_index = (local_index + spec_index * 2) % len(doc_cues)
            doc_cue = doc_cues[doc_cue_index]
            subject = subjects[local_index // len(tails)]
            tail = tails[local_index % len(tails)]
            decoy = _d2_decoy(config.seed, key)
            pair_id = hashlib.sha256(f"{key}:pair".encode("utf-8")).hexdigest()
            base_format = {
                "subject": subject,
                "tail": tail,
                "doc_cue": doc_cue,
                "cue": cue,
                "identifier": identifier,
                "decoy": decoy,
            }
            common = {
                "pair_id": pair_id,
                "document_family": spec.document_family,
                "format_family": spec.format_family,
                "surface_variant": variant,
                "numbering_cue_variant": cue_index,
                "document_cue_variant": doc_cue_index,
            }

            positive_text = spec.positive_core.format(**base_format)
            generated.append(
                _d2_record(
                    text=positive_text,
                    identifier=identifier,
                    label="ID_DOC",
                    template_id=f"{spec.family_id}:{local_index}",
                    dataset_role=dataset_role,
                    sealed=sealed,
                    metadata={
                        **common,
                        "synthetic_polarity": "positive",
                        "template_family": spec.family_id,
                        "selection_stratum": f"synthetic-positive:{spec.format_family}",
                        "hard_negative_match": None,
                        "boundary_trap": None,
                    },
                )
            )
            family_counts[f"positive:{spec.format_family}"] += 1

            exact_text = spec.exact_negative_core.format(**base_format)
            generated.append(
                _d2_record(
                    text=exact_text,
                    identifier=identifier,
                    label="DOCID",
                    template_id=f"{spec.exact_negative_family}:{local_index}",
                    dataset_role=dataset_role,
                    sealed=sealed,
                    metadata={
                        **common,
                        "synthetic_polarity": "hard-negative-exact",
                        "template_family": spec.exact_negative_family,
                        "selection_stratum": (
                            f"synthetic-hard-negative-exact:{spec.format_family}"
                        ),
                        "hard_negative_match": (
                            "same identifier surface and value-only span length as positive"
                        ),
                        "boundary_trap": None,
                    },
                )
            )
            family_counts[f"hard-negative-exact:{spec.format_family}"] += 1

            trap_key = f"{key}:boundary"
            trap_digits: int | None = None
            if spec.format_family == "numeric-7":
                trap_digits = D2_NUMERIC_TRAP_LENGTHS[
                    (local_index + spec_index) % len(D2_NUMERIC_TRAP_LENGTHS)
                ]
            trap_compact, trap_groups = _d2_compact_identifier(
                seed=config.seed,
                key=trap_key,
                format_family=spec.format_family,
                numeric_digits=trap_digits,
            )
            trap_variant = D2_SURFACE_VARIANTS[
                (local_index + spec_index + 1) % len(D2_SURFACE_VARIANTS)
            ]
            trap_identifier = _d2_surface(trap_compact, trap_groups, trap_variant)
            trap_cue_index = (cue_index + 2) % len(D2_NUMBERING_CUES)
            trap_format = {
                **base_format,
                "cue": D2_NUMBERING_CUES[trap_cue_index],
                "identifier": trap_identifier,
            }
            trap_kind = (
                f"numeric-length-{trap_digits}-cue-and-punctuation-outside-span"
                if trap_digits is not None
                else "alphanumeric-fragment-adjacency-cue-outside-span"
            )
            trap_text = spec.boundary_negative_core.format(**trap_format)
            generated.append(
                _d2_record(
                    text=trap_text,
                    identifier=trap_identifier,
                    label="DOCID",
                    template_id=f"{spec.boundary_negative_family}:{local_index}",
                    dataset_role=dataset_role,
                    sealed=sealed,
                    metadata={
                        **common,
                        "surface_variant": trap_variant,
                        "numbering_cue_variant": trap_cue_index,
                        "synthetic_polarity": "hard-negative-boundary",
                        "template_family": spec.boundary_negative_family,
                        "selection_stratum": (
                            f"synthetic-hard-negative-boundary:{spec.format_family}"
                        ),
                        "hard_negative_match": "same format family as positive",
                        "boundary_trap": trap_kind,
                    },
                )
            )
            family_counts[f"hard-negative-boundary:{spec.format_family}"] += 1

    expected_per_format = positive_rows // len(D2_FORMAT_FAMILIES)
    for format_family in D2_FORMAT_FAMILIES:
        if family_counts[f"positive:{format_family}"] != expected_per_format:
            raise StudentTrainingError(
                f"Bilanciamento D2 {split} non valido per {format_family}"
            )
    if len(generated) != positive_rows + hard_negative_rows:
        raise StudentTrainingError(f"Conteggio sintetico D2 {split} non valido")
    return generated


def _read_jsonl_objects(path: Path, description: str) -> list[dict[str, Any]]:
    source = path.expanduser().resolve()
    if not source.is_file():
        raise StudentTrainingError(f"{description} non trovato: {source}")
    records: list[dict[str, Any]] = []
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise StudentTrainingError(
                    f"{description} {source}:{line_number}: riga vuota"
                )
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise StudentTrainingError(
                    f"{description} {source}:{line_number}: JSON non valido"
                ) from exc
            if not isinstance(value, dict):
                raise StudentTrainingError(
                    f"{description} {source}:{line_number}: record non oggetto"
                )
            records.append(value)
    if not records:
        raise StudentTrainingError(f"{description} vuoto: {source}")
    return records


def _d2_prior_output(
    root: Path,
    manifest: Mapping[str, Any],
    role: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    outputs = manifest.get("outputs")
    record = outputs.get(role) if isinstance(outputs, Mapping) else None
    if not isinstance(record, Mapping):
        raise StudentTrainingError(f"Manifest V1 privo di output {role}")
    filename = str(record.get("filename") or "")
    if not filename or Path(filename).name != filename:
        raise StudentTrainingError(f"Filename V1 non valido per {role}")
    source = (root / filename).resolve()
    if source.parent != root.resolve():
        raise StudentTrainingError(f"Path V1 fuori directory per {role}")
    integrity = {
        "path": str(source),
        "bytes": source.stat().st_size if source.is_file() else -1,
        "sha256": sha256_file(source) if source.is_file() else None,
    }
    if integrity["sha256"] != record.get("sha256"):
        raise StudentTrainingError(f"Hash output V1 divergente per {role}")
    records = _read_jsonl_objects(source, f"Output V1 {role}")
    stats = record.get("stats")
    if isinstance(stats, Mapping) and int(stats.get("rows", -1)) != len(records):
        raise StudentTrainingError(f"Conteggio output V1 divergente per {role}")
    return records, {**integrity, "rows": len(records), "role": role}


def _d2_replay_candidate(
    raw: Mapping[str, Any], *, origin_role: str, index: int
) -> dict[str, Any]:
    identity = _identity_payload(raw, context=f"V1 {origin_role} replay {index}")
    record_id = _record_id(identity)
    declared = raw.get("record_id")
    if declared is not None and str(declared) != record_id:
        raise StudentTrainingError(f"Record ID V1 non canonico: {origin_role}/{index}")
    text = str(identity["source_text"])
    numeric_iban = False
    alphanumeric_identifier = False
    for entity in identity["entities"]:
        normalized = _normalized_type(str(entity["label"]))
        value = text[int(entity["start"]) : int(entity["end"])]
        if normalized == "IBAN" and re.fullmatch(r"[\d\s./-]+", value):
            numeric_iban = len(re.sub(r"\D", "", value)) >= 6
        if normalized in {"ID_DOC", "DOCID"}:
            alphanumeric_identifier = alphanumeric_identifier or (
                re.search(r"[A-Za-z]", value) is not None
                and re.search(r"\d", value) is not None
            )
    return {
        "identity": identity,
        "record_id": record_id,
        "skeleton_sha256": _skeleton_sha256(identity),
        "origin_role": origin_role,
        "origin_stratum": str(raw.get("selection_stratum") or "unknown"),
        "origin_source_row": raw.get("source_row"),
        "numeric_iban": numeric_iban,
        "alphanumeric_identifier": alphanumeric_identifier,
    }


def _load_d2_replay_pool(
    config: D2BuildConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root = config.prior_v1_dir.expanduser().resolve()
    manifest_path = root / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise StudentTrainingError(f"Manifest V1 non trovato: {manifest_path}") from exc
    except json.JSONDecodeError as exc:
        raise StudentTrainingError(f"Manifest V1 non valido: {manifest_path}") from exc
    if not isinstance(manifest, Mapping):
        raise StudentTrainingError("Manifest V1 non oggetto")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("kind") != "pii-id-doc-targeted-finetune-data"
        or manifest.get("status") != "sealed"
    ):
        raise StudentTrainingError("Bundle V1 non compatibile con D2")
    d1_rows, d1_identity = _d2_prior_output(root, manifest, "d1_targeted")
    d0_rows, d0_identity = _d2_prior_output(root, manifest, "d0_control")

    candidates: list[dict[str, Any]] = []
    for index, raw in enumerate(d1_rows):
        if str(raw.get("selection_stratum") or "").startswith("replay-"):
            candidates.append(
                _d2_replay_candidate(raw, origin_role="d1_targeted_replay", index=index)
            )
    for index, raw in enumerate(d0_rows):
        candidates.append(
            _d2_replay_candidate(raw, origin_role="d0_control", index=index)
        )
    ids = [str(item["record_id"]) for item in candidates]
    skeletons = [str(item["skeleton_sha256"]) for item in candidates]
    if len(ids) != len(set(ids)) or len(skeletons) != len(set(skeletons)):
        raise StudentTrainingError("Pool replay V1 contiene duplicati o skeleton condivisi")
    return candidates, {
        "root": str(root),
        "manifest": {
            "path": str(manifest_path),
            "bytes": manifest_path.stat().st_size,
            "sha256": sha256_file(manifest_path),
        },
        "outputs": {
            "d1_targeted": d1_identity,
            "d0_control": d0_identity,
        },
        "candidate_rows": len(candidates),
    }


def _select_d2_replay(
    config: D2BuildConfig,
    candidates: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    if config.replay_rows > len(candidates):
        raise StudentTrainingError(
            f"Replay D2 insufficiente: {len(candidates)} < {config.replay_rows}"
        )
    effective_salt = f"{config.salt}:seed={config.seed}:replay"
    selected: list[dict[str, Any]] = []
    used_ids: set[str] = set()
    used_skeletons: set[str] = set()

    def take(feature: str | None, count: int, suffix: str) -> None:
        pool = [
            item
            for item in candidates
            if (feature is None or bool(item[feature]))
            and str(item["record_id"]) not in used_ids
            and str(item["skeleton_sha256"]) not in used_skeletons
        ]
        ordered = sorted(
            pool,
            key=lambda item: (
                _score(f"{effective_salt}:{suffix}", str(item["record_id"])),
                str(item["record_id"]),
            ),
        )
        if len(ordered) < count:
            raise StudentTrainingError(
                f"Replay D2 {suffix}: richiesti {count}, disponibili {len(ordered)}"
            )
        for item in ordered[:count]:
            selected.append(item)
            used_ids.add(str(item["record_id"]))
            used_skeletons.add(str(item["skeleton_sha256"]))

    take("numeric_iban", config.replay_numeric_iban_floor, "numeric-iban")
    take(
        "alphanumeric_identifier",
        config.replay_alphanumeric_id_floor,
        "alphanumeric-identifier",
    )
    take(None, config.replay_rows - len(selected), "general")
    if len(selected) != config.replay_rows:
        raise StudentTrainingError("Conteggio replay D2 non valido")
    return selected


def _d2_replay_payload(item: Mapping[str, Any]) -> dict[str, Any]:
    identity = dict(item["identity"])
    return {
        "record_id": str(item["record_id"]),
        "source": "clean-v1-replay",
        **identity,
        "dataset_role": "d2_targeted",
        "selection_stratum": f"real-replay:{item['origin_role']}",
        "sealed": False,
        "synthetic": False,
        "contains_real_pii": "source-dataset-contract",
        "replay_origin": {
            "role": str(item["origin_role"]),
            "stratum": str(item["origin_stratum"]),
            "source_row": item["origin_source_row"],
        },
        "replay_coverage": {
            "numeric_iban": bool(item["numeric_iban"]),
            "alphanumeric_identifier": bool(item["alphanumeric_identifier"]),
        },
    }


def _d2_entity_counts(records: Sequence[Mapping[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for index, raw in enumerate(records):
        identity = _identity_payload(raw, context=f"statistiche D2 {index}")
        for entity in identity["entities"]:
            normalized = _normalized_type(str(entity["label"]))
            if normalized is not None:
                counts[normalized] += 1
    return counts


def _d2_output_manifest(
    *,
    role: str,
    filename: str,
    records: Sequence[Mapping[str, Any]],
    payload: bytes,
    sealed: bool,
) -> dict[str, Any]:
    polarities = Counter(
        str(record.get("synthetic_polarity") or "real-replay")
        for record in records
    )
    sources = Counter(str(record.get("source")) for record in records)
    template_families = Counter(
        str(record["template_family"])
        for record in records
        if record.get("template_family") is not None
    )
    format_families = Counter(
        str(record["format_family"])
        for record in records
        if record.get("format_family") is not None
    )
    positive_format_families = Counter(
        str(record["format_family"])
        for record in records
        if record.get("synthetic_polarity") == "positive"
        and record.get("format_family") is not None
    )
    boundary_traps = Counter(
        str(record["boundary_trap"])
        for record in records
        if record.get("boundary_trap") is not None
    )
    ids: list[str] = []
    skeletons: list[str] = []
    public_records: list[dict[str, Any]] = []
    for index, raw in enumerate(records):
        identity = _identity_payload(raw, context=f"manifest D2 {role}/{index}")
        record_id = _record_id(identity)
        if str(raw.get("record_id")) != record_id:
            raise StudentTrainingError(f"Record ID D2 non canonico: {role}/{index}")
        skeleton = _skeleton_sha256(identity)
        ids.append(record_id)
        skeletons.append(skeleton)
        payload_meta = raw.get("identifier_payload")
        public_records.append(
            {
                "record_id": record_id,
                "normalized_skeleton_sha256": skeleton,
                "source": str(raw.get("source")),
                "stratum": str(raw.get("selection_stratum")),
                "synthetic_polarity": raw.get("synthetic_polarity"),
                "template_family": raw.get("template_family"),
                "document_family": raw.get("document_family"),
                "format_family": raw.get("format_family"),
                "pair_id": raw.get("pair_id"),
                "boundary_trap": raw.get("boundary_trap"),
                "identifier_payload_sha256": (
                    payload_meta.get("sha256")
                    if isinstance(payload_meta, Mapping)
                    else None
                ),
            }
        )
    if len(ids) != len(set(ids)) or len(skeletons) != len(set(skeletons)):
        raise StudentTrainingError(f"Output D2 {role} contiene duplicati")
    replay_records = [record for record in records if not bool(record.get("synthetic"))]
    replay_coverage = {
        "numeric_iban_rows": sum(
            bool(record.get("replay_coverage", {}).get("numeric_iban"))
            for record in replay_records
        ),
        "alphanumeric_identifier_rows": sum(
            bool(record.get("replay_coverage", {}).get("alphanumeric_identifier"))
            for record in replay_records
        ),
    }
    selection_payload = [
        {
            "record_id": item["record_id"],
            "normalized_skeleton_sha256": item["normalized_skeleton_sha256"],
            "stratum": item["stratum"],
            "template_family": item["template_family"],
        }
        for item in public_records
    ]
    return {
        "role": role,
        "filename": filename,
        "sealed": sealed,
        "consumed": False if sealed else None,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "selection_sha256": hashlib.sha256(
            _canonical_json_bytes(selection_payload)
        ).hexdigest(),
        "stats": {
            "rows": len(records),
            "synthetic_rows": sum(bool(record.get("synthetic")) for record in records),
            "real_replay_rows": len(replay_records),
            "polarities": dict(sorted(polarities.items())),
            "sources": dict(sorted(sources.items())),
            "entity_counts": dict(sorted(_d2_entity_counts(records).items())),
            "template_families": dict(sorted(template_families.items())),
            "format_families": dict(sorted(format_families.items())),
            "positive_format_families": dict(
                sorted(positive_format_families.items())
            ),
            "boundary_traps": dict(sorted(boundary_traps.items())),
            "replay_coverage": replay_coverage,
        },
        "records": public_records,
    }


def _validate_d2_config(config: D2BuildConfig) -> None:
    if config.seed < 0 or not config.salt:
        raise StudentTrainingError("Seed/salt D2 non validi")
    counts = (
        config.train_positive_rows,
        config.train_hard_negative_rows,
        config.replay_rows,
        config.challenge_positive_rows,
        config.challenge_hard_negative_rows,
    )
    if min(counts) < 1:
        raise StudentTrainingError("Dimensioni split D2 devono essere positive")
    if config.train_hard_negative_rows != config.train_positive_rows * 2:
        raise StudentTrainingError("Train D2 richiede due hard-negative per positivo")
    if config.challenge_hard_negative_rows != config.challenge_positive_rows * 2:
        raise StudentTrainingError("Challenge D2 richiede due hard-negative per positivo")
    if config.train_positive_rows % len(D2_FORMAT_FAMILIES):
        raise StudentTrainingError("Positivi train D2 non bilanciabili per formato")
    if config.challenge_positive_rows % len(D2_FORMAT_FAMILIES):
        raise StudentTrainingError("Positivi challenge D2 non bilanciabili per formato")
    if (
        config.replay_numeric_iban_floor < 0
        or config.replay_alphanumeric_id_floor < 0
        or config.replay_numeric_iban_floor + config.replay_alphanumeric_id_floor
        > config.replay_rows
    ):
        raise StudentTrainingError("Floor replay D2 non validi")


def build_v2_artifacts(
    config: D2BuildConfig,
) -> tuple[dict[str, Any], dict[str, bytes]]:
    _validate_d2_config(config)
    train_family_ids = _d2_template_family_ids(D2_TRAIN_TEMPLATE_FAMILIES)
    challenge_family_ids = _d2_template_family_ids(D2_CHALLENGE_TEMPLATE_FAMILIES)
    if train_family_ids & challenge_family_ids:
        raise StudentTrainingError("Famiglie template train/challenge D2 sovrapposte")

    synthetic_train = _generate_d2_synthetic(
        config=config,
        split="train",
        positive_rows=config.train_positive_rows,
        hard_negative_rows=config.train_hard_negative_rows,
        dataset_role="d2_targeted",
        sealed=False,
    )
    replay_pool, replay_source = _load_d2_replay_pool(config)
    replay_selection = _select_d2_replay(config, replay_pool)
    replay_records = [_d2_replay_payload(item) for item in replay_selection]
    train_records = synthetic_train + replay_records
    train_records.sort(
        key=lambda record: (
            _score(
                f"{config.salt}:seed={config.seed}:d2-train-order",
                str(record["record_id"]),
            ),
            str(record["record_id"]),
        )
    )
    expected_train_rows = (
        config.train_positive_rows
        + config.train_hard_negative_rows
        + config.replay_rows
    )
    if len(train_records) != expected_train_rows:
        raise StudentTrainingError("Conteggio train D2 non valido")

    challenge_records = _generate_d2_synthetic(
        config=config,
        split="challenge",
        positive_rows=config.challenge_positive_rows,
        hard_negative_rows=config.challenge_hard_negative_rows,
        dataset_role="sealed_synthetic_challenge",
        sealed=True,
    )
    challenge_records.sort(
        key=lambda record: (
            _score(
                f"{config.salt}:seed={config.seed}:d2-challenge-order",
                str(record["record_id"]),
            ),
            str(record["record_id"]),
        )
    )

    train_ids = {str(record["record_id"]) for record in train_records}
    challenge_ids = {str(record["record_id"]) for record in challenge_records}
    train_skeletons = {
        _skeleton_sha256(_identity_payload(record, context="train D2 overlap"))
        for record in train_records
    }
    challenge_skeletons = {
        _skeleton_sha256(_identity_payload(record, context="challenge D2 overlap"))
        for record in challenge_records
    }
    overlap_audit = {
        "train_challenge_record_id_overlap": len(train_ids & challenge_ids),
        "train_challenge_normalized_skeleton_overlap": len(
            train_skeletons & challenge_skeletons
        ),
        "template_family_overlap": len(train_family_ids & challenge_family_ids),
    }
    overlap_audit["all_required_zero"] = not any(overlap_audit.values())
    if not overlap_audit["all_required_zero"]:
        raise StudentTrainingError(f"Leakage D2 train/challenge: {overlap_audit}")

    output_records = {
        "d2_targeted": train_records,
        "sealed_synthetic_challenge": challenge_records,
    }
    payloads: dict[str, bytes] = {}
    outputs: dict[str, Any] = {}
    for role, records in output_records.items():
        sealed = role.startswith("sealed_")
        filename = D2_OUTPUT_FILENAMES[role]
        payload = _jsonl_bytes(records)
        payloads[filename] = payload
        outputs[role] = _d2_output_manifest(
            role=role,
            filename=filename,
            records=records,
            payload=payload,
            sealed=sealed,
        )

    train_stats = outputs["d2_targeted"]["stats"]
    expected_polarities = {
        "positive": config.train_positive_rows,
        "hard-negative-exact": config.train_positive_rows,
        "hard-negative-boundary": config.train_positive_rows,
        "real-replay": config.replay_rows,
    }
    if train_stats["polarities"] != expected_polarities:
        raise StudentTrainingError(
            f"Composizione train D2 divergente: {train_stats['polarities']}"
        )
    replay_coverage = train_stats["replay_coverage"]
    if replay_coverage["numeric_iban_rows"] < config.replay_numeric_iban_floor:
        raise StudentTrainingError("Floor IBAN numerici non rispettato nel replay D2")
    if (
        replay_coverage["alphanumeric_identifier_rows"]
        < config.replay_alphanumeric_id_floor
    ):
        raise StudentTrainingError(
            "Floor identificatori alfanumerici non rispettato nel replay D2"
        )

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "kind": "pii-id-doc-targeted-finetune-data-v2",
        "status": "sealed",
        "seed": config.seed,
        "recipe": {
            "train_rows": expected_train_rows,
            "synthetic_positive_rows": config.train_positive_rows,
            "synthetic_positive_per_format": (
                config.train_positive_rows // len(D2_FORMAT_FAMILIES)
            ),
            "synthetic_hard_negative_rows": config.train_hard_negative_rows,
            "hard_negative_policy": (
                "one exact surface/span matched DOCID plus one boundary trap per positive"
            ),
            "real_replay_rows": config.replay_rows,
            "challenge_rows": len(challenge_records),
            "challenge_positive_rows": config.challenge_positive_rows,
            "challenge_hard_negative_rows": config.challenge_hard_negative_rows,
        },
        "selection": {
            "algorithm": "lowest salted SHA-256 of canonical record_id",
            "salt": config.salt,
            "effective_salt": f"{config.salt}:seed={config.seed}",
            "replay_numeric_iban_floor": config.replay_numeric_iban_floor,
            "replay_alphanumeric_identifier_floor": (
                config.replay_alphanumeric_id_floor
            ),
        },
        "span_policy": D2_CANONICAL_SPAN_POLICY,
        "evaluation_views": D2_EVALUATION_VIEWS,
        "synthetic_generation": {
            "contains_real_pii": False,
            "values": "deterministic syntactic examples not linked to people",
            "format_families": D2_FORMAT_SOURCES,
            "numeric_boundary_trap_lengths": list(D2_NUMERIC_TRAP_LENGTHS),
            "surface_variants": list(D2_SURFACE_VARIANTS),
            "context_variation": [
                "case",
                "spacing",
                "punctuation",
                "line breaks",
                "accent loss",
                "limited OCR substitutions in harmless cues",
            ],
        },
        "template_family_split": {
            "train": sorted(train_family_ids),
            "challenge": sorted(challenge_family_ids),
            "disjoint": True,
        },
        "sources": {"v1_real_replay": replay_source},
        "overlap_audit": overlap_audit,
        "outputs": outputs,
        "immutability": {
            "publication": "atomic directory rename on same filesystem",
            "existing_output_policy": "accept byte-identical directory only",
            "sealed_challenge_must_not_be_evaluated_before_checkpoint_freeze": True,
        },
        "builder": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__)),
        },
    }
    manifest["recipe_sha256"] = hashlib.sha256(
        _canonical_json_bytes(
            {
                "seed": manifest["seed"],
                "recipe": manifest["recipe"],
                "selection": manifest["selection"],
                "span_policy": manifest["span_policy"],
                "template_family_split": manifest["template_family_split"],
                "source_manifest_sha256": replay_source["manifest"]["sha256"],
                "output_selection_sha256": {
                    role: value["selection_sha256"]
                    for role, value in outputs.items()
                },
            }
        )
    ).hexdigest()
    payloads["manifest.json"] = json.dumps(
        manifest, ensure_ascii=False, sort_keys=True, indent=2
    ).encode("utf-8") + b"\n"
    return manifest, payloads


def _validate_config(config: BuildConfig) -> None:
    if config.seed < 0 or not config.salt:
        raise StudentTrainingError("Seed/salt targeted non validi")
    if min(
        config.d1_rows,
        config.d0_rows,
        config.holdout_global_rows,
        config.holdout_id_doc_rows,
    ) < 1:
        raise StudentTrainingError("Le dimensioni dei quattro split devono essere positive")
    if config.d0_rows != config.d1_rows:
        raise StudentTrainingError("D0 e D1 devono avere la stessa dimensione")


def _select(config: BuildConfig) -> tuple[dict[str, list[SelectedRecord]], dict[str, Any]]:
    _validate_config(config)
    effective_salt = f"{config.salt}:seed={config.seed}"
    train_metas, train_identity = _scan_parquet(config.train_parquet, split="train")
    validation_metas, validation_identity = _scan_parquet(
        config.validation_parquet, split="validation"
    )
    if (
        config.expected_train_sha256 is not None
        and train_identity["sha256"] != config.expected_train_sha256
    ):
        raise StudentTrainingError("SHA-256 shard clean train diverso dal pinned")
    if (
        config.expected_validation_sha256 is not None
        and validation_identity["sha256"] != config.expected_validation_sha256
    ):
        raise StudentTrainingError("SHA-256 clean validation diverso dal pinned")

    observed_train = _load_observed(
        config.observed_train, description="Train osservato"
    )
    observed_validation = _load_observed(
        config.observed_validation, description="Validation osservata"
    )
    observed_overlap = _pairwise_overlaps(
        {"observed_train": observed_train, "observed_validation": observed_validation}
    )
    if not observed_overlap["all_pairwise_zero"]:
        raise StudentTrainingError("Train e validation osservati hanno leakage")

    novel_train = [
        record
        for record in train_metas
        if record.record_id not in observed_train.record_ids
        and record.skeleton_sha256 not in observed_train.skeletons
        and record.record_id not in observed_validation.record_ids
        and record.skeleton_sha256 not in observed_validation.skeletons
    ]
    target_train = [record for record in novel_train if record.target_entities > 0]
    _assert_expected(
        "Documenti target train novel",
        len(target_train),
        config.expected_target_train_docs,
    )
    _assert_expected(
        "Entita' target train novel",
        sum(record.target_entities for record in target_train),
        config.expected_target_train_entities,
    )
    if config.d1_rows < 2 * len(target_train):
        raise StudentTrainingError("D1 troppo piccolo per target + hard negative 1:1")

    target_train = _sorted_candidates(
        target_train, salt=f"{effective_salt}:d1:target"
    )
    d1: list[SelectedRecord] = [
        SelectedRecord(record, "target:id-doc-n-dot-7-digits", "ID_DOC")
        for record in target_train
    ]
    d1_ids = {record.record_id for record in target_train}
    d1_skeletons = {record.skeleton_sha256 for record in target_train}

    hard_candidates = [
        record
        for record in novel_train
        if record.record_id not in d1_ids
        and record.target_entities == 0
        and record.hard_negative_variant is not None
    ]
    hard = _balanced_hard_negatives(
        hard_candidates,
        count=len(target_train),
        salt=f"{effective_salt}:d1:hard-negative",
        forbidden_ids=d1_ids,
        forbidden_skeletons=d1_skeletons,
    )
    d1.extend(hard)
    d1_ids.update(item.meta.record_id for item in hard)
    d1_skeletons.update(item.meta.skeleton_sha256 for item in hard)

    replay_count = config.d1_rows - len(d1)
    replay_candidates = [
        record
        for record in novel_train
        if record.record_id not in d1_ids
        and record.target_entities == 0
        and record.hard_negative_variant is None
    ]
    replay = _select_replay(
        replay_candidates,
        count=replay_count,
        protected_floor=config.replay_protected_floor,
        salt=f"{effective_salt}:d1:replay",
        forbidden_ids=d1_ids,
        forbidden_skeletons=d1_skeletons,
    )
    d1.extend(replay)
    if len(d1) != config.d1_rows:
        raise StudentTrainingError("D1 non ha la dimensione richiesta")

    d1_ids = {item.meta.record_id for item in d1}
    d1_skeletons = {item.meta.skeleton_sha256 for item in d1}
    control_candidates = [
        record
        for record in novel_train
        if record.record_id not in d1_ids
        and record.skeleton_sha256 not in d1_skeletons
        and record.target_entities == 0
    ]
    d0 = _select_control(
        control_candidates,
        d1=d1,
        count=config.d0_rows,
        salt=f"{effective_salt}:d0:matched-control",
    )

    selected_train_ids = d1_ids | {item.meta.record_id for item in d0}
    selected_train_skeletons = d1_skeletons | {
        item.meta.skeleton_sha256 for item in d0
    }
    forbidden_validation_ids = (
        set(observed_train.record_ids)
        | set(observed_validation.record_ids)
        | selected_train_ids
    )
    forbidden_validation_skeletons = (
        set(observed_train.skeletons)
        | set(observed_validation.skeletons)
        | selected_train_skeletons
    )
    eligible_validation = [
        record
        for record in validation_metas
        if record.record_id not in forbidden_validation_ids
        and record.skeleton_sha256 not in forbidden_validation_skeletons
    ]
    target_holdout = [
        record for record in eligible_validation if record.target_entities > 0
    ]
    _assert_expected(
        "Documenti target holdout novel",
        len(target_holdout),
        config.expected_target_holdout_docs,
    )
    _assert_expected(
        "Entita' target holdout novel",
        sum(record.target_entities for record in target_holdout),
        config.expected_target_holdout_entities,
    )
    target_holdout = _sorted_candidates(
        target_holdout, salt=f"{effective_salt}:sealed:id-doc:target"
    )
    target_holdout_skeletons = [record.skeleton_sha256 for record in target_holdout]
    if len(set(target_holdout_skeletons)) != len(target_holdout_skeletons):
        raise StudentTrainingError("Target holdout con skeleton interni duplicati")
    if len(target_holdout) > config.holdout_id_doc_rows:
        raise StudentTrainingError("Challenge ID_DOC troppo piccolo per tutti i target")

    id_doc_fill = _take_unique_skeletons(
        (
            record
            for record in eligible_validation
            if record.id_doc_entities > 0 and record.target_entities == 0
        ),
        count=config.holdout_id_doc_rows - len(target_holdout),
        salt=f"{effective_salt}:sealed:id-doc:fill",
        forbidden_ids={record.record_id for record in target_holdout},
        forbidden_skeletons=set(target_holdout_skeletons),
    )
    sealed_id_doc = [
        SelectedRecord(record, "sealed-id-doc:target-n-dot-7-digits", "ID_DOC")
        for record in target_holdout
    ] + [
        SelectedRecord(record, "sealed-id-doc:other-id-doc", "ID_DOC")
        for record in id_doc_fill
    ]
    sealed_id_ids = {item.meta.record_id for item in sealed_id_doc}
    sealed_id_skeletons = {item.meta.skeleton_sha256 for item in sealed_id_doc}
    global_records = _take_unique_skeletons(
        eligible_validation,
        count=config.holdout_global_rows,
        salt=f"{effective_salt}:sealed:global",
        forbidden_ids=sealed_id_ids,
        forbidden_skeletons=sealed_id_skeletons,
    )
    sealed_global = [
        SelectedRecord(record, "sealed-global:uniform", "GENERAL")
        for record in global_records
    ]

    selections = {
        "d1_targeted": d1,
        "d0_control": d0,
        "sealed_global": sealed_global,
        "sealed_id_doc": sealed_id_doc,
    }
    overlaps = _pairwise_overlaps(
        {
            "observed_train": observed_train,
            "observed_validation": observed_validation,
            **{
                name: [item.meta for item in items]
                for name, items in selections.items()
            },
        }
    )
    if not overlaps["all_pairwise_zero"]:
        failures = {
            key: value
            for key, value in overlaps.items()
            if key != "all_pairwise_zero"
            and (value["record_id_overlap"] or value["normalized_skeleton_overlap"])
        }
        raise StudentTrainingError(f"Leakage fra split targeted: {failures}")

    context = {
        "sources": {
            "clean_revision": config.clean_revision,
            "train": train_identity,
            "validation": validation_identity,
            "observed_train": dict(observed_train.identity),
            "observed_validation": dict(observed_validation.identity),
        },
        "pool": {
            "novel_train_rows": len(novel_train),
            "eligible_validation_rows": len(eligible_validation),
            "target_train_docs": len(target_train),
            "target_train_entities": sum(
                record.target_entities for record in target_train
            ),
            "target_holdout_docs": len(target_holdout),
            "target_holdout_entities": sum(
                record.target_entities for record in target_holdout
            ),
        },
        "overlap_audit": overlaps,
    }
    return selections, context


def _load_payload_records(
    parquet_path: Path,
    *,
    split: str,
    selected: Sequence[SelectedRecord],
    dataset_role: str,
    sealed: bool,
) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise StudentTrainingError("pyarrow e' richiesto dal builder targeted") from exc

    by_id = {item.meta.record_id: item for item in selected}
    if len(by_id) != len(selected):
        raise StudentTrainingError(f"{dataset_role}: record selezionati duplicati")
    found: dict[str, dict[str, Any]] = {}
    source = parquet.ParquetFile(parquet_path.expanduser().resolve())
    required = sorted({"source_text", "entities", "language", "template_id"})
    row_index = 0
    for batch in source.iter_batches(batch_size=2048, columns=required):
        for raw in batch.to_pylist():
            meta, identity = _record_meta(raw, source_row=row_index, split=split)
            row_index += 1
            item = by_id.get(meta.record_id)
            if item is None:
                continue
            if meta != item.meta:
                raise StudentTrainingError(
                    f"{dataset_role}: sorgente cambiata per {meta.record_id}"
                )
            found[meta.record_id] = {
                "record_id": meta.record_id,
                "source": "clean",
                "source_split": split,
                "source_row": meta.source_row,
                **identity,
                "dataset_role": dataset_role,
                "selection_stratum": item.stratum,
                "sealed": sealed,
            }
    missing = set(by_id) - set(found)
    if missing:
        raise StudentTrainingError(
            f"{dataset_role}: {len(missing)} record non ritrovati nel Parquet"
        )
    return [found[item.meta.record_id] for item in selected]


def _jsonl_bytes(records: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(_canonical_json_bytes(record) + b"\n" for record in records)


def _selection_manifest(
    *,
    role: str,
    filename: str,
    items: Sequence[SelectedRecord],
    payload: bytes,
    sealed: bool,
) -> dict[str, Any]:
    return {
        "role": role,
        "filename": filename,
        "sealed": sealed,
        "consumed": False if sealed else None,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "selection_sha256": hashlib.sha256(
            _canonical_json_bytes(
                [
                    {
                        "record_id": item.meta.record_id,
                        "source_row": item.meta.source_row,
                        "normalized_skeleton_sha256": item.meta.skeleton_sha256,
                        "stratum": item.stratum,
                        "focus": item.focus,
                        "match_level": item.match_level,
                    }
                    for item in items
                ]
            )
        ).hexdigest(),
        "stats": _selection_stats(items),
        "records": [
            {
                "record_id": item.meta.record_id,
                "source_row": item.meta.source_row,
                "normalized_skeleton_sha256": item.meta.skeleton_sha256,
                "stratum": item.stratum,
                "focus": item.focus,
                "match_level": item.match_level,
                "chars": item.meta.chars,
                "length_bin": item.meta.length_bin,
                "tags": list(item.meta.tags),
                "target_shape_entities": item.meta.target_entities,
            }
            for item in items
        ],
    }


def build_artifacts(
    config: BuildConfig,
) -> tuple[dict[str, Any], dict[str, bytes]]:
    selections, context = _select(config)
    payloads: dict[str, bytes] = {}
    outputs: dict[str, Any] = {}
    for role, items in selections.items():
        sealed = role.startswith("sealed_")
        split = "validation" if sealed else "train"
        source_path = (
            config.validation_parquet if sealed else config.train_parquet
        )
        records = _load_payload_records(
            source_path,
            split=split,
            selected=items,
            dataset_role=role,
            sealed=sealed,
        )
        payload = _jsonl_bytes(records)
        filename = OUTPUT_FILENAMES[role]
        payloads[filename] = payload
        outputs[role] = _selection_manifest(
            role=role,
            filename=filename,
            items=items,
            payload=payload,
            sealed=sealed,
        )

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "kind": "pii-id-doc-targeted-finetune-data",
        "status": "sealed",
        "seed": config.seed,
        "selection": {
            "algorithm": "lowest salted SHA-256 of canonical record_id",
            "salt": config.salt,
            "effective_salt": f"{config.salt}:seed={config.seed}",
            "target_normalized_type": "ID_DOC",
            "target_entity_value_regex": TARGET_PATTERN_TEXT,
            "hard_negative_normalized_type": "DOCID",
            "hard_negative_variants": list(HARD_NEGATIVE_VARIANTS),
            "hard_negative_ratio_to_target_docs": "1:1",
            "protected_replay_tags": list(PROTECTED_TAGS),
            "replay_protected_floor_per_tag": config.replay_protected_floor,
            "control_matching": (
                "disjoint; target-shape excluded; quotas matched by char length bin "
                "and protected focus tag with documented deterministic fallback"
            ),
            "holdout_policy": (
                "exclude observed validation IDs and normalized skeletons; allocate all "
                "novel target-shape records to sealed ID_DOC first; unique skeletons; "
                "then fill ID_DOC and select disjoint global holdout"
            ),
        },
        **context,
        "outputs": outputs,
        "immutability": {
            "publication": "atomic directory rename on same filesystem",
            "existing_output_policy": "accept byte-identical directory only",
            "sealed_holdouts_must_not_be_evaluated_before_checkpoint_freeze": True,
        },
        "builder": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__)),
        },
    }
    payloads["manifest.json"] = json.dumps(
        manifest, ensure_ascii=False, sort_keys=True, indent=2
    ).encode("utf-8") + b"\n"
    return manifest, payloads


def _verify_existing(output_dir: Path, payloads: Mapping[str, bytes]) -> None:
    actual = {path.name for path in output_dir.iterdir() if path.is_file()}
    expected = set(payloads)
    if actual != expected or any(path.is_dir() for path in output_dir.iterdir()):
        raise StudentTrainingError(
            f"Output targeted esistente con contenuto inatteso: "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )
    for filename, payload in payloads.items():
        path = output_dir / filename
        if path.read_bytes() != payload:
            raise StudentTrainingError(
                f"Output targeted esistente ma diverso: {path}"
            )


def publish_atomic(output_dir: Path, payloads: Mapping[str, bytes]) -> None:
    target = output_dir.expanduser().resolve()
    if target.exists():
        if not target.is_dir():
            raise StudentTrainingError(f"Output targeted non e' directory: {target}")
        _verify_existing(target, payloads)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.", dir=str(target.parent))
    )
    try:
        for filename, payload in payloads.items():
            destination = temporary / filename
            with destination.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        descriptor = os.open(temporary, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        try:
            os.replace(temporary, target)
        except OSError:
            if target.is_dir():
                _verify_existing(target, payloads)
            else:
                raise
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def _summary(manifest: Mapping[str, Any], output_dir: Path | None) -> dict[str, Any]:
    if manifest.get("kind") == "pii-id-doc-targeted-finetune-data-v2":
        return {
            "status": manifest["status"],
            "output_dir": (
                str(output_dir.resolve()) if output_dir is not None else None
            ),
            "seed": manifest["seed"],
            "recipe": manifest["recipe"],
            "span_policy": manifest["span_policy"]["id"],
            "outputs": {
                role: {
                    "filename": value["filename"],
                    "sha256": value["sha256"],
                    "rows": value["stats"]["rows"],
                    "polarities": value["stats"]["polarities"],
                    "sealed": value["sealed"],
                }
                for role, value in manifest["outputs"].items()
            },
            "all_required_overlap_zero": manifest["overlap_audit"][
                "all_required_zero"
            ],
        }
    return {
        "status": manifest["status"],
        "output_dir": str(output_dir.resolve()) if output_dir is not None else None,
        "seed": manifest["seed"],
        "pool": manifest["pool"],
        "outputs": {
            role: {
                "filename": value["filename"],
                "sha256": value["sha256"],
                "rows": value["stats"]["rows"],
                "target_shape_docs": value["stats"]["target_shape_docs"],
                "target_shape_entities": value["stats"]["target_shape_entities"],
                "sealed": value["sealed"],
            }
            for role, value in manifest["outputs"].items()
        },
        "all_pairwise_overlap_zero": manifest["overlap_audit"][
            "all_pairwise_zero"
        ],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("inspect", "build"))
    parser.add_argument("--recipe", choices=("v1", "v2"), default="v1")
    parser.add_argument("--train-parquet", default=str(DEFAULT_TRAIN_PARQUET))
    parser.add_argument(
        "--validation-parquet", default=str(DEFAULT_VALIDATION_PARQUET)
    )
    parser.add_argument("--observed-train", default=str(DEFAULT_OBSERVED_TRAIN))
    parser.add_argument(
        "--observed-validation", default=str(DEFAULT_OBSERVED_VALIDATION)
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="default dipendente dalla recipe (V1 o V2)",
    )
    parser.add_argument("--clean-revision", default=CLEAN_REVISION)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--salt", default=None, help="default dipendente dalla recipe")
    parser.add_argument("--d1-rows", type=int, default=1024)
    parser.add_argument("--d0-rows", type=int, default=1024)
    parser.add_argument("--holdout-global-rows", type=int, default=2048)
    parser.add_argument("--holdout-id-doc-rows", type=int, default=2048)
    parser.add_argument("--replay-protected-floor", type=int, default=32)
    parser.add_argument("--expected-target-train-docs", type=int, default=226)
    parser.add_argument("--expected-target-train-entities", type=int, default=237)
    parser.add_argument("--expected-target-holdout-docs", type=int, default=82)
    parser.add_argument("--expected-target-holdout-entities", type=int, default=92)
    parser.add_argument("--expected-train-sha256", default=EXPECTED_TRAIN_SHA256)
    parser.add_argument(
        "--expected-validation-sha256", default=EXPECTED_VALIDATION_SHA256
    )
    parser.add_argument("--prior-v1-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--train-positive-rows", type=int, default=128)
    parser.add_argument("--train-hard-negative-rows", type=int, default=256)
    parser.add_argument("--replay-rows", type=int, default=640)
    parser.add_argument("--challenge-positive-rows", type=int, default=128)
    parser.add_argument("--challenge-hard-negative-rows", type=int, default=256)
    parser.add_argument("--replay-numeric-iban-floor", type=int, default=64)
    parser.add_argument("--replay-alphanumeric-id-floor", type=int, default=64)
    return parser


def _config_from_args(args: argparse.Namespace) -> BuildConfig:
    return BuildConfig(
        train_parquet=Path(args.train_parquet),
        validation_parquet=Path(args.validation_parquet),
        observed_train=Path(args.observed_train),
        observed_validation=Path(args.observed_validation),
        output_dir=Path(args.output_dir or DEFAULT_OUTPUT_DIR),
        clean_revision=str(args.clean_revision),
        seed=int(args.seed),
        salt=str(args.salt or DEFAULT_SALT),
        d1_rows=int(args.d1_rows),
        d0_rows=int(args.d0_rows),
        holdout_global_rows=int(args.holdout_global_rows),
        holdout_id_doc_rows=int(args.holdout_id_doc_rows),
        replay_protected_floor=int(args.replay_protected_floor),
        expected_target_train_docs=int(args.expected_target_train_docs),
        expected_target_train_entities=int(args.expected_target_train_entities),
        expected_target_holdout_docs=int(args.expected_target_holdout_docs),
        expected_target_holdout_entities=int(args.expected_target_holdout_entities),
        expected_train_sha256=(str(args.expected_train_sha256) or None),
        expected_validation_sha256=(str(args.expected_validation_sha256) or None),
    )


def _v2_config_from_args(args: argparse.Namespace) -> D2BuildConfig:
    return D2BuildConfig(
        prior_v1_dir=Path(args.prior_v1_dir),
        output_dir=Path(args.output_dir or DEFAULT_V2_OUTPUT_DIR),
        seed=int(args.seed),
        salt=str(args.salt or DEFAULT_V2_SALT),
        train_positive_rows=int(args.train_positive_rows),
        train_hard_negative_rows=int(args.train_hard_negative_rows),
        replay_rows=int(args.replay_rows),
        challenge_positive_rows=int(args.challenge_positive_rows),
        challenge_hard_negative_rows=int(args.challenge_hard_negative_rows),
        replay_numeric_iban_floor=int(args.replay_numeric_iban_floor),
        replay_alphanumeric_id_floor=int(args.replay_alphanumeric_id_floor),
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.recipe == "v2":
        config = _v2_config_from_args(args)
        manifest, payloads = build_v2_artifacts(config)
    else:
        config = _config_from_args(args)
        manifest, payloads = build_artifacts(config)
    output: Path | None = None
    if args.mode == "build":
        publish_atomic(config.output_dir, payloads)
        output = config.output_dir
    print(json.dumps(_summary(manifest, output), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except StudentTrainingError as exc:
        print(f"ERRORE: {exc}")
        raise SystemExit(2) from exc
