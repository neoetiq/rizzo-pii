"""Metriche entity-level senza dipendenze ML per regressioni di quantizzazione.

Le sequenze sono word-level e in formato BIO.  Gli span sono deliberatamente
considerati corretti solo se ``(tipo, start, end)`` coincide: e' la stessa
semantica usata da :mod:`src.training.evaluate_pii`, quindi una sovrapposizione
parziale non puo' mascherare una regressione.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Collection, Mapping, Sequence
from typing import Any


# Stessa tassonomia di ``src/training/evaluate_pii.py``.  Il dataset di
# validazione contiene ancora alcuni tag grezzi che il modello non emette.
TAG_MAP = {
    "GIVENNAME": "FULLNAME",
    "SURNAME": "FULLNAME",
    "GIUDICE": "FULLNAME",
    "AVVOCATO": "FULLNAME",
    "CONVENUTO": "FULLNAME",
    "ATTORE": "FULLNAME",
    "TESTIMONE": "FULLNAME",
    "SEX": "GENDER",
    "TAXNUM": "PIVA",
    "PEC": "EMAIL",
    "RG": "DOCID",
    "IDCARDNUM": "ID_DOC",
    "PASSPORTNUM": "ID_DOC",
    "DRIVERLICENSENUM": "ID_DOC",
    "SOCIALNUM": "ID_DOC",
    "CONTO": "IBAN",
    "CIG": "DOCID",
    "CUP": "DOCID",
    "POLIZZA": "DOCID",
    "MATRICOLA": "DOCID",
}
DROP_TYPES = {"TITLE", "TRIBUNAL"}

Span = tuple[str, int, int]


def canonical_text_and_word_offsets(
    tokens: Sequence[str],
) -> tuple[str, list[tuple[int, int]]]:
    """Costruisce il testo canonico misurabile di un record token-list.

    Il dataset pubblico non conserva il testo originale ne' la whitespace map.
    Usiamo quindi *esattamente* ``" ".join(tokens)`` e restituiamo gli offset
    assoluti di ciascun token in quel testo.  Non tentiamo di rimuovere marker
    ``##`` o di riattaccare la punteggiatura: sarebbe una ricostruzione
    indimostrabile e renderebbe gli offset locali del tokenizer incoerenti.
    """

    if isinstance(tokens, (str, bytes)):
        raise TypeError("tokens deve essere una sequenza di stringhe")
    if not all(isinstance(token, str) for token in tokens):
        raise TypeError("Ogni token deve essere una stringa")

    offsets: list[tuple[int, int]] = []
    cursor = 0
    for index, token in enumerate(tokens):
        if index:
            cursor += 1
        start = cursor
        cursor += len(token)
        offsets.append((start, cursor))
    return " ".join(tokens), offsets


def word_spans_to_char_spans(
    tags: Sequence[str], word_offsets: Sequence[tuple[int, int]]
) -> set[Span]:
    """Proietta gli span BIO word-level sugli offset del testo canonico."""

    if len(tags) != len(word_offsets):
        raise ValueError(
            f"Tag e offset parola disallineati: {len(tags)} != {len(word_offsets)}"
        )
    result: set[Span] = set()
    for entity_type, word_start, word_end in spans(tags):
        char_start = int(word_offsets[word_start][0])
        char_end = int(word_offsets[word_end - 1][1])
        if char_end <= char_start:
            raise ValueError(f"Span carattere vuoto per {entity_type}")
        result.add((entity_type, char_start, char_end))
    return result


def _bio_parts(label: str) -> tuple[str, str]:
    if label.startswith("B-"):
        return "B", label[2:]
    if label.startswith("I-"):
        return "I", label[2:]
    # E' la stessa convenzione di Transformers TokenClassificationPipeline:
    # una label senza prefisso e' trattata come continuazione.
    return "I", label


def simple_char_spans(
    labels: Sequence[str],
    offsets: Sequence[Sequence[int]],
    *,
    special_tokens_mask: Sequence[int | bool] | None = None,
    word_ids: Sequence[int | None] | None = None,
    word_offsets: Sequence[tuple[int, int]] | None = None,
    ignore_labels: Collection[str] = ("O",),
) -> set[Span]:
    """Replica gli span di ``aggregation_strategy="simple"``.

    A differenza della metrica storica, questa funzione legge **ogni subword**.
    Gli offset possono essere gia' assoluti (input testuale di produzione) oppure
    locali a parole pre-tokenizzate; nel secondo caso ``word_ids`` e
    ``word_offsets`` li riportano sul testo canonico del record.

    Score e testo decodificato non servono per la regressione esatta: la pipeline
    sceglie prima l'argmax per token, poi unisce ``B-/I-`` adiacenti dello stesso
    tipo e infine rimuove i gruppi ignorati.
    """

    lengths = {len(labels), len(offsets)}
    if special_tokens_mask is not None:
        lengths.add(len(special_tokens_mask))
    if word_ids is not None:
        lengths.add(len(word_ids))
    if len(lengths) != 1:
        raise ValueError("Label, offset, mask e word_ids devono avere la stessa lunghezza")
    if (word_ids is None) != (word_offsets is None):
        raise ValueError("word_ids e word_offsets devono essere forniti insieme")

    ignored = set(ignore_labels)
    result: set[Span] = set()
    current: tuple[str, int, int] | None = None

    def finish() -> None:
        nonlocal current
        if current is not None and current[0] not in ignored:
            result.add(current)
        current = None

    for index, (label, raw_offset) in enumerate(zip(labels, offsets)):
        if special_tokens_mask is not None and bool(special_tokens_mask[index]):
            continue
        if len(raw_offset) != 2:
            raise ValueError(f"Offset non valido in posizione {index}: {raw_offset!r}")
        start, end = int(raw_offset[0]), int(raw_offset[1])
        if word_ids is not None:
            word_id = word_ids[index]
            if word_id is None:
                continue
            if word_id < 0 or word_id >= len(word_offsets):
                raise ValueError(f"word_id fuori range in posizione {index}: {word_id}")
            base = int(word_offsets[word_id][0])
            start += base
            end += base
        if end <= start:
            # Un token normalizzato a lunghezza zero non deve saldare due
            # entita' separate. I token speciali sono gia' esclusi dalla mask.
            finish()
            continue

        bi, entity_type = _bio_parts(str(label))
        if current is not None and current[0] == entity_type and bi != "B":
            current = (current[0], current[1], end)
        else:
            finish()
            current = (entity_type, start, end)
    finish()
    return result


def normalize_labels(labels: Sequence[str]) -> list[str]:
    """Rimappa i tag grezzi alla tassonomia del modello e ricostruisce il BIO.

    Due tag consecutivi che diventano lo stesso tipo dopo la rimappatura sono
    una sola entita' (``B-FULLNAME, I-FULLNAME``), anche se in origine erano
    ad esempio ``GIVENNAME`` e ``SURNAME``.  E' identico alla normalizzazione
    effettuata dal valutatore FP32 esistente.
    """

    out: list[str] = []
    previous: str | None = None
    for label in labels:
        entity_type = TAG_MAP.get(label[2:], label[2:]) if label != "O" else None
        if entity_type is None or entity_type in DROP_TYPES:
            out.append("O")
            previous = None
            continue
        out.append(("I-" if entity_type == previous else "B-") + entity_type)
        previous = entity_type
    return out


def spans(tags: Sequence[str]) -> set[Span]:
    """Restituisce gli span BIO word-level come ``(tipo, start, end)``.

    Un ``I-TAG`` isolato apre un'entita', esattamente come nel valutatore di
    training; un cambio di tipo chiude quella corrente.  Questo rende la
    regressione robusta anche a sequenze BIO non perfettamente ben formate.
    """

    result: set[Span] = set()
    current: tuple[str, int] | None = None
    for index, tag in enumerate([*tags, "O"]):
        if tag == "O" or tag.startswith("B-") or (
            current is not None and tag[2:] != current[0]
        ):
            if current is not None:
                result.add((current[0], current[1], index))
                current = None
        if tag.startswith("B-") or (tag.startswith("I-") and current is None):
            current = (tag[2:], index)
    return result


def _validate_documents(
    gold_docs: Sequence[Sequence[str]], pred_docs: Sequence[Sequence[str]]
) -> None:
    """Controlla l'allineamento word-level prima di calcolare le metriche."""

    if isinstance(gold_docs, (str, bytes)) or isinstance(pred_docs, (str, bytes)):
        raise TypeError("gold_docs e pred_docs devono essere liste di sequenze BIO")
    if len(gold_docs) != len(pred_docs):
        raise ValueError(
            "Numero di documenti diverso: "
            f"gold={len(gold_docs)}, prediction={len(pred_docs)}"
        )
    for index, (gold, pred) in enumerate(zip(gold_docs, pred_docs)):
        if isinstance(gold, (str, bytes)) or isinstance(pred, (str, bytes)):
            raise TypeError(f"Documento {index}: attesa una sequenza di tag BIO")
        if len(gold) != len(pred):
            raise ValueError(
                f"Documento {index}: numero di token diverso "
                f"(gold={len(gold)}, prediction={len(pred)})"
            )


def _prf(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "support": tp + fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _validate_span_documents(
    gold_docs: Sequence[Collection[Span]], pred_docs: Sequence[Collection[Span]]
) -> None:
    if isinstance(gold_docs, (str, bytes)) or isinstance(pred_docs, (str, bytes)):
        raise TypeError("gold_docs e pred_docs devono essere liste di collezioni di span")
    if len(gold_docs) != len(pred_docs):
        raise ValueError(
            "Numero di documenti diverso: "
            f"gold={len(gold_docs)}, prediction={len(pred_docs)}"
        )
    for document_index, documents in enumerate(zip(gold_docs, pred_docs)):
        for side, document in zip(("gold", "prediction"), documents):
            if isinstance(document, (str, bytes)):
                raise TypeError(
                    f"Documento {document_index} {side}: attesa una collezione di span"
                )
            for span in document:
                if (
                    not isinstance(span, tuple)
                    or len(span) != 3
                    or not isinstance(span[0], str)
                    or isinstance(span[1], bool)
                    or isinstance(span[2], bool)
                    or not isinstance(span[1], int)
                    or not isinstance(span[2], int)
                    or span[1] < 0
                    or span[2] <= span[1]
                ):
                    raise ValueError(
                        f"Documento {document_index} {side}: span non valido {span!r}"
                    )


def compute_span_metrics(
    gold_docs: Sequence[Collection[Span]], pred_docs: Sequence[Collection[Span]]
) -> dict[str, Any]:
    """Calcola le stesse metriche entity-level su span gia' materializzati.

    Le coordinate possono essere word index oppure offset carattere: conta solo
    che baseline e candidato usino lo stesso sistema. Questo percorso permette
    al gate production-like di misurare gli offset esatti senza riconvertirli in
    una sequenza BIO artificiale.
    """

    _validate_span_documents(gold_docs, pred_docs)
    tp: Counter[str] = Counter()
    fp: Counter[str] = Counter()
    fn: Counter[str] = Counter()
    tag_names: set[str] = set()

    for gold, pred in zip(gold_docs, pred_docs):
        gold_spans = set(gold)
        pred_spans = set(pred)
        tag_names.update(span[0] for span in gold_spans | pred_spans)
        for entity_type, _, _ in gold_spans & pred_spans:
            tp[entity_type] += 1
        for entity_type, _, _ in pred_spans - gold_spans:
            fp[entity_type] += 1
        for entity_type, _, _ in gold_spans - pred_spans:
            fn[entity_type] += 1

    per_tag = {
        entity_type: _prf(tp[entity_type], fp[entity_type], fn[entity_type])
        for entity_type in sorted(tag_names)
    }
    total_tp, total_fp, total_fn = sum(tp.values()), sum(fp.values()), sum(fn.values())
    macro_values = list(per_tag.values())
    macro = {
        "tags": len(macro_values),
        "precision": (
            sum(float(item["precision"]) for item in macro_values) / len(macro_values)
            if macro_values
            else 0.0
        ),
        "recall": (
            sum(float(item["recall"]) for item in macro_values) / len(macro_values)
            if macro_values
            else 0.0
        ),
        "f1": (
            sum(float(item["f1"]) for item in macro_values) / len(macro_values)
            if macro_values
            else 0.0
        ),
    }
    return {
        "documents": len(gold_docs),
        "micro": _prf(total_tp, total_fp, total_fn),
        "macro": macro,
        "macro_f1": macro["f1"],
        "per_tag": per_tag,
    }


def compute_metrics(
    gold_docs: Sequence[Sequence[str]], pred_docs: Sequence[Sequence[str]]
) -> dict[str, Any]:
    """Calcola precision, recall e F1 entity-level con match esatto degli span.

    ``gold_docs`` e ``pred_docs`` devono avere stesso numero di documenti e
    token.  I tag presenti soltanto nelle predizioni compaiono comunque in
    ``per_tag`` (supporto zero), cosi' un nuovo falso positivo non resta
    invisibile nel report.
    """

    _validate_documents(gold_docs, pred_docs)
    return compute_span_metrics(
        [spans(gold) for gold in gold_docs],
        [spans(pred) for pred in pred_docs],
    )


def compare_predictions(
    golden_docs: Sequence[Sequence[str]], candidate_docs: Sequence[Sequence[str]]
) -> dict[str, Any]:
    """Confronta le predizioni candidate con il golden FP32, non con il gold.

    Gli span ``new`` sono rilevati soltanto dalla versione candidata, mentre
    ``removed`` erano presenti nel golden FP32 e non lo sono piu'.  La chiave
    ``agreement`` conta gli span identici e i documenti senza alcuna variazione.
    """

    _validate_documents(golden_docs, candidate_docs)
    return compare_span_predictions(
        [spans(golden) for golden in golden_docs],
        [spans(candidate) for candidate in candidate_docs],
    )


def compare_span_predictions(
    reference_docs: Sequence[Collection[Span]],
    candidate_docs: Sequence[Collection[Span]],
) -> dict[str, Any]:
    """Confronta due modelli su span esatti, senza chiamarne uno ground truth.

    E' il confronto adatto a teacher/student e student-FP32/student-quantizzato.
    La qualita' assoluta dei due lati deve essere misurata separatamente contro
    il gold umano; qui ``reference`` indica solo il lato di confronto.
    """

    _validate_span_documents(reference_docs, candidate_docs)
    by_tag: defaultdict[str, Counter[str]] = defaultdict(Counter)
    totals: Counter[str] = Counter()
    exact_documents = 0

    for reference, candidate in zip(reference_docs, candidate_docs):
        reference_spans = set(reference)
        candidate_spans = set(candidate)
        agreed = reference_spans & candidate_spans
        new = candidate_spans - reference_spans
        removed = reference_spans - candidate_spans
        exact_documents += int(reference_spans == candidate_spans)
        totals.update(
            golden=len(reference_spans),
            candidate=len(candidate_spans),
            agreed=len(agreed),
            new=len(new),
            removed=len(removed),
        )
        for key, values in (
            ("golden", reference_spans),
            ("candidate", candidate_spans),
            ("agreed", agreed),
            ("new", new),
            ("removed", removed),
        ):
            for entity_type, _, _ in values:
                by_tag[entity_type][key] += 1

    documents = len(reference_docs)
    return {
        "documents": documents,
        "agreement": {
            "exact_documents": exact_documents,
            "exact_document_rate": exact_documents / documents if documents else 0.0,
            "spans": dict(totals),
            "by_tag": {
                entity_type: {
                    key: counts[key]
                    for key in ("golden", "candidate", "agreed", "new", "removed")
                }
                for entity_type, counts in sorted(by_tag.items())
            },
        },
    }


def _metric(metrics: Mapping[str, Any], level: str, field: str) -> float:
    try:
        return float(metrics[level][field])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Metriche prive di {level}.{field}") from exc


def _append_failure(
    failures: list[dict[str, Any]], gate: str, actual: float | int, limit: float | int,
    *, tag: str | None = None,
) -> None:
    failure: dict[str, Any] = {"gate": gate, "actual": actual, "limit": limit}
    if tag is not None:
        failure["tag"] = tag
    failures.append(failure)


def apply_quality_gate(
    baseline_metrics: Mapping[str, Any],
    candidate_metrics: Mapping[str, Any],
    variant: str,
    config: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Applica soglie di regressione al candidato rispetto alla baseline FP32.

    ``baseline_metrics`` e ``candidate_metrics`` sono normalmente il risultato
    di :func:`compute_metrics` rispetto allo stesso gold.  ``config`` e'
    opzionale e puo' contenere:

    * ``max_micro_f1_drop`` e ``max_macro_f1_drop``;
    * ``min_micro_f1`` e ``min_macro_f1``;
    * ``max_per_tag_f1_drop``: numero valido per tutti i tag, oppure mappa
      ``{tag: limite}``;
    * ``min_per_tag_f1``: numero o mappa ``{tag: soglia}``.

    Le soglie sono inclusive: una perdita pari al limite passa il gate.

    Se entrambe le metriche contengono la vista ``production_like_char``, le
    stesse soglie vengono applicate anche agli span carattere ottenuti con la
    semantica ``aggregation_strategy="simple"``.  I nomi dei relativi failure
    hanno il prefisso ``production_like_char.``. La vista resta opzionale per
    poter rileggere i report storici, ma se e' presente solo da un lato il gate
    fallisce: non e' lecito confrontare due semantiche diverse.
    """

    config = config or {}
    if not isinstance(config, Mapping):
        raise TypeError("config deve essere una mapping o None")

    base_micro = _metric(baseline_metrics, "micro", "f1")
    candidate_micro = _metric(candidate_metrics, "micro", "f1")
    base_macro = _metric(baseline_metrics, "macro", "f1")
    candidate_macro = _metric(candidate_metrics, "macro", "f1")
    base_tags = baseline_metrics.get("per_tag", {})
    candidate_tags = candidate_metrics.get("per_tag", {})
    if not isinstance(base_tags, Mapping) or not isinstance(candidate_tags, Mapping):
        raise ValueError("Metriche prive di per_tag")

    all_tags = sorted(set(base_tags) | set(candidate_tags))
    deltas = {
        "micro_f1": candidate_micro - base_micro,
        "macro_f1": candidate_macro - base_macro,
        "per_tag_f1": {
            tag: float(candidate_tags.get(tag, {}).get("f1", 0.0))
            - float(base_tags.get(tag, {}).get("f1", 0.0))
            for tag in all_tags
        },
        "per_tag_recall": {
            tag: float(candidate_tags.get(tag, {}).get("recall", 0.0))
            - float(base_tags.get(tag, {}).get("recall", 0.0))
            for tag in all_tags
        },
    }
    failures: list[dict[str, Any]] = []

    for key, baseline, candidate in (
        ("micro", base_micro, candidate_micro),
        ("macro", base_macro, candidate_macro),
    ):
        drop_key = f"max_{key}_f1_drop"
        min_key = f"min_{key}_f1"
        if drop_key in config:
            limit = float(config[drop_key])
            actual = baseline - candidate
            if actual > limit:
                _append_failure(failures, drop_key, actual, limit)
        if min_key in config:
            limit = float(config[min_key])
            if candidate < limit:
                _append_failure(failures, min_key, candidate, limit)

    for key, is_drop in (("max_per_tag_f1_drop", True), ("min_per_tag_f1", False)):
        if key not in config:
            continue
        configured = config[key]
        limits = configured if isinstance(configured, Mapping) else {
            tag: configured for tag in sorted(set(base_tags) | set(candidate_tags))
        }
        for tag, raw_limit in limits.items():
            limit = float(raw_limit)
            baseline = float(base_tags.get(tag, {}).get("f1", 0.0))
            candidate = float(candidate_tags.get(tag, {}).get("f1", 0.0))
            actual = baseline - candidate if is_drop else candidate
            if (is_drop and actual > limit) or (not is_drop and actual < limit):
                _append_failure(failures, key, actual, limit, tag=str(tag))

    recall_limits = config.get("max_per_tag_recall_drop")
    if recall_limits is not None:
        limits = recall_limits if isinstance(recall_limits, Mapping) else {
            tag: recall_limits for tag in all_tags
        }
        for tag, raw_limit in limits.items():
            limit = float(raw_limit)
            baseline = float(base_tags.get(tag, {}).get("recall", 0.0))
            candidate = float(candidate_tags.get(tag, {}).get("recall", 0.0))
            actual = baseline - candidate
            if actual > limit:
                _append_failure(
                    failures, "max_per_tag_recall_drop", actual, limit, tag=str(tag)
                )

    production_baseline = baseline_metrics.get("production_like_char")
    production_candidate = candidate_metrics.get("production_like_char")
    production_gate: dict[str, Any] | None = None
    if (production_baseline is None) != (production_candidate is None):
        _append_failure(
            failures,
            "production_like_char.metrics_missing",
            int(
                production_baseline is not None
                and production_candidate is not None
            ),
            1,
        )
        production_gate = {
            "passed": False,
            "reason": "vista production_like_char presente solo in uno dei due report",
        }
    elif production_baseline is not None and production_candidate is not None:
        if not isinstance(production_baseline, Mapping) or not isinstance(
            production_candidate, Mapping
        ):
            raise ValueError("production_like_char deve contenere metriche valide")
        # La chiamata termina qui perche' le viste annidate prodotte dal runner
        # non contengono a loro volta ``production_like_char``.
        production_gate = apply_quality_gate(
            production_baseline,
            production_candidate,
            variant,
            config,
        )
        for failure in production_gate["failures"]:
            prefixed = dict(failure)
            prefixed["gate"] = f"production_like_char.{failure['gate']}"
            failures.append(prefixed)
        deltas["production_like_char"] = production_gate["deltas"]

    result = {
        "variant": variant,
        "passed": not failures,
        "failures": failures,
        "deltas": deltas,
        "baseline": {"micro_f1": base_micro, "macro_f1": base_macro},
        "candidate": {"micro_f1": candidate_micro, "macro_f1": candidate_macro},
    }
    if production_gate is not None:
        result["production_like_char"] = production_gate
    return result


# Nomi espliciti usati dal runner di benchmark.  Manteniamo le funzioni corte
# sopra per rendere il modulo comodo anche nei test e negli script interattivi.
evaluate_predictions = compute_metrics
compare_with_fp32 = compare_predictions
evaluate_quality_gates = apply_quality_gate
