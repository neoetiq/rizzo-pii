"""Run artifact evaluations in isolated subprocesses and build a regression report."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

from .evaluate import (
    preprocessor_integrity,
    read_predictions,
    read_production_char_predictions,
)
from .metrics import (
    compare_span_predictions,
    compare_with_fp32,
    evaluate_quality_gates,
)


_QUALITY_POLICY_KEYS = {
    "max_micro_f1_drop",
    "max_macro_f1_drop",
    "min_micro_f1",
    "min_macro_f1",
    "max_per_tag_f1_drop",
    "min_per_tag_f1",
    "max_per_tag_recall_drop",
    "min_entity_agreement",
    "max_new_spans",
    "max_removed_spans",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_verified_predictions(
    report: dict[str, Any],
) -> tuple[
    list[list[str]],
    list[set[tuple[str, int, int]]] | None,
    dict[str, Any],
]:
    """Read compact predictions and verify their evaluation-time digest.

    Legacy reports without a digest remain readable for diagnostics, but the
    caller must keep them non-promotable.  A present-but-different digest is
    corruption or stale state and therefore aborts report construction.
    """

    path = Path(report["predictions_path"])
    predictions = read_predictions(path)
    char_predictions = read_production_char_predictions(path)
    expected_documents = int(report.get("documents", 0))
    if len(predictions) != expected_documents:
        raise ValueError(
            f"Predizioni {report.get('variant')} disallineate dal report: "
            f"{len(predictions)} != {expected_documents}"
        )
    recorded = report.get("prediction_digest")
    current = _sha256(path)
    if recorded is None:
        return predictions, char_predictions, {
            "status": "legacy-missing-digest",
            "verified": False,
            "sha256_current": current,
        }
    if not isinstance(recorded, str) or not recorded:
        raise ValueError(f"Digest predizioni non valido per {report.get('variant')}")
    if recorded != current:
        raise ValueError(
            f"Digest predizioni diverso per {report.get('variant')}: "
            f"registrato {recorded}, corrente {current}"
        )
    return predictions, char_predictions, {
        "status": "verified",
        "verified": True,
        "sha256": current,
    }


def _verify_validation_integrity(report: dict[str, Any]) -> dict[str, Any]:
    """Verify that metrics still point to the exact evaluated validation bytes."""

    recorded = report.get("validation_integrity")
    raw_path = report.get("validation_path")
    if not raw_path:
        if recorded is not None:
            raise ValueError(
                f"validation_path assente per {report.get('variant')} con integrita registrata"
            )
        return {
            "status": "legacy-missing-path-and-integrity",
            "verified": False,
        }
    validation_path = Path(raw_path)
    current = {
        "path": str(validation_path.expanduser().resolve()),
        "bytes": validation_path.stat().st_size,
        "sha256": _sha256(validation_path),
    }
    if recorded is None:
        return {
            "status": "legacy-missing-integrity",
            "verified": False,
            "current": current,
        }
    if not isinstance(recorded, dict):
        raise ValueError(
            f"Integrita validation non valida per {report.get('variant')}"
        )
    normalized = {
        "path": str(Path(recorded.get("path", "")).expanduser().resolve()),
        "bytes": int(recorded.get("bytes", -1)),
        "sha256": str(recorded.get("sha256", "")),
    }
    if normalized != current:
        raise ValueError(
            f"Validation diversa da quella valutata per {report.get('variant')}"
        )
    return {
        "status": "verified",
        "verified": True,
        **current,
    }


def _verify_preprocessor_integrity(report: dict[str, Any]) -> dict[str, Any]:
    """Verify tokenizer bytes and the config/id2label used during evaluation."""

    recorded = report.get("preprocessor_integrity")
    raw_path = report.get("tokenizer_path")
    if not raw_path:
        if recorded is not None:
            raise ValueError(
                f"tokenizer_path assente per {report.get('variant')} con integrita registrata"
            )
        return {
            "status": "legacy-missing-path-and-integrity",
            "verified": False,
        }
    current = preprocessor_integrity(Path(raw_path))
    if recorded is None:
        return {
            "status": "legacy-missing-integrity",
            "verified": False,
            "current": current,
        }
    if not isinstance(recorded, dict):
        raise ValueError(
            f"Integrita tokenizer/config non valida per {report.get('variant')}"
        )
    if recorded != current:
        raise ValueError(
            f"Tokenizer/config diversi da quelli valutati per {report.get('variant')}"
        )
    return {
        "status": "verified",
        "verified": True,
        **current,
    }


def _run(command: Sequence[str], *, cwd: Path) -> None:
    completed = subprocess.run(list(command), cwd=cwd, check=False)
    if completed.returncode:
        raise RuntimeError(f"Comando fallito ({completed.returncode}): {' '.join(command)}")


def evaluate_isolated(
    *,
    root: Path,
    name: str,
    backend: str,
    model_path: Path,
    tokenizer_path: Path,
    validation_path: Path,
    output_dir: Path,
    batch_size: int,
    max_length: int,
    threads: int,
    limit: int | None = None,
) -> dict[str, Any]:
    metrics_path = output_dir / "metrics" / f"{name}.json"
    predictions_path = output_dir / "predictions" / f"{name}.jsonl"
    command = [
        sys.executable,
        "-m",
        "src.quantization.cli",
        "evaluate-one",
        "--name",
        name,
        "--backend",
        backend,
        "--model-path",
        str(model_path),
        "--tokenizer-path",
        str(tokenizer_path),
        "--validation-path",
        str(validation_path),
        "--predictions-out",
        str(predictions_path),
        "--report-out",
        str(metrics_path),
        "--batch-size",
        str(batch_size),
        "--max-length",
        str(max_length),
        "--threads",
        str(threads),
    ]
    if limit is not None:
        command.extend(["--limit", str(limit)])
    _run(command, cwd=root)
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def build_regression_report(
    *,
    variant_reports: Sequence[dict[str, Any]],
    output_dir: Path,
    gates_config: dict[str, Any],
    expected_documents: int | None = None,
    sources_verification: dict[str, Any] | None = None,
) -> dict[str, Any]:
    names = [str(report["variant"]) for report in variant_reports]
    if len(names) != len(set(names)):
        raise ValueError("Report duplicati per la stessa variante")
    by_name = {report["variant"]: report for report in variant_reports}
    if "torch-fp32" not in by_name:
        raise ValueError("Manca il golden torch-fp32")
    non_ok = [
        name for name, report in by_name.items() if report.get("status") != "ok"
    ]
    if non_ok:
        raise ValueError(
            "Varianti richieste senza valutazione completata: " + ", ".join(non_ok)
        )

    sources_integrity_complete = bool(
        isinstance(sources_verification, dict)
        and sources_verification.get("status") == "verified"
        and sources_verification.get("evaluation_guarded") is True
    )

    for item in by_name.values():
        rss_start = int(item.get("rss_start_bytes", 0))
        rss_after_load = int(item.get("rss_after_load_bytes", 0))
        rss_peak = int(item.get("rss_peak_bytes", 0))
        item.setdefault("rss_model_load_delta_bytes", max(0, rss_after_load - rss_start))
        item.setdefault("rss_peak_delta_bytes", max(0, rss_peak - rss_start))
        item.setdefault("rss_inference_delta_bytes", max(0, rss_peak - rss_after_load))

    document_counts = {
        name: int(item.get("documents", 0)) for name, item in by_name.items()
    }
    validation_integrity = {
        name: _verify_validation_integrity(item) for name, item in by_name.items()
    }
    preprocessor_integrity_by_variant = {
        name: _verify_preprocessor_integrity(item) for name, item in by_name.items()
    }
    verified_validation_hashes = {
        item["sha256"]
        for item in validation_integrity.values()
        if item.get("verified")
    }
    if len(verified_validation_hashes) > 1:
        raise ValueError("Le varianti sono state valutate su validation diverse")
    validation_integrity_complete = bool(
        validation_integrity
        and all(item.get("verified") for item in validation_integrity.values())
    )
    preprocessor_integrity_complete = bool(
        preprocessor_integrity_by_variant
        and all(
            item.get("verified")
            for item in preprocessor_integrity_by_variant.values()
        )
    )
    full_validation = bool(
        expected_documents is not None
        and expected_documents > 0
        and all(count == expected_documents for count in document_counts.values())
    )
    golden_report = by_name["torch-fp32"]
    (
        golden_predictions,
        golden_char_predictions,
        golden_prediction_integrity,
    ) = _read_verified_predictions(golden_report)
    prediction_integrity: dict[str, Any] = {
        "torch-fp32": golden_prediction_integrity
    }
    comparisons: dict[str, Any] = {}
    for name, report in by_name.items():
        if name == "torch-fp32":
            continue
        (
            candidate_predictions,
            candidate_char_predictions,
            candidate_prediction_integrity,
        ) = (
            _read_verified_predictions(report)
        )
        prediction_integrity[name] = candidate_prediction_integrity
        comparison = compare_with_fp32(golden_predictions, candidate_predictions)
        raw_variant_config = gates_config.get(name)
        variant_config = (
            raw_variant_config if isinstance(raw_variant_config, dict) else {}
        )
        quality_policy_complete = bool(
            variant_config
            and any(key in variant_config for key in _QUALITY_POLICY_KEYS)
        )
        gates = evaluate_quality_gates(
            golden_report["metrics"],
            report["metrics"],
            variant=name,
            config=variant_config,
        )
        if not quality_policy_complete:
            gates["passed"] = False
            gates["failures"].append(
                {
                    "gate": "quality_policy.required_for_release",
                    "actual": 0,
                    "limit": 1,
                }
            )
        production_like_metrics_complete = all(
            isinstance(item.get("metrics", {}).get("production_like_char"), dict)
            for item in (golden_report, report)
        )
        production_like_predictions_complete = bool(
            golden_char_predictions is not None
            and candidate_char_predictions is not None
        )
        production_like_complete = bool(
            production_like_metrics_complete
            and production_like_predictions_complete
        )
        production_like_comparison = (
            compare_span_predictions(
                golden_char_predictions,
                candidate_char_predictions,
            )
            if production_like_predictions_complete
            else None
        )
        if not production_like_complete:
            gates["passed"] = False
            gates["failures"].append(
                {
                    "gate": "production_like_char.metrics_required_for_release",
                    "actual": int(production_like_complete),
                    "limit": 1,
                }
            )
        prediction_digest_complete = bool(
            golden_prediction_integrity["verified"]
            and candidate_prediction_integrity["verified"]
        )
        if not prediction_digest_complete:
            gates["passed"] = False
            gates["failures"].append(
                {
                    "gate": "prediction_digest.required_for_release",
                    "actual": int(prediction_digest_complete),
                    "limit": 1,
                }
            )
        candidate_validation_complete = bool(
            validation_integrity["torch-fp32"]["verified"]
            and validation_integrity[name]["verified"]
        )
        if not candidate_validation_complete:
            gates["passed"] = False
            gates["failures"].append(
                {
                    "gate": "validation_integrity.required_for_release",
                    "actual": int(candidate_validation_complete),
                    "limit": 1,
                }
            )
        candidate_preprocessor_complete = bool(
            preprocessor_integrity_by_variant["torch-fp32"]["verified"]
            and preprocessor_integrity_by_variant[name]["verified"]
        )
        if not candidate_preprocessor_complete:
            gates["passed"] = False
            gates["failures"].append(
                {
                    "gate": "preprocessor_integrity.required_for_release",
                    "actual": int(candidate_preprocessor_complete),
                    "limit": 1,
                }
            )
        if not sources_integrity_complete:
            gates["passed"] = False
            gates["failures"].append(
                {
                    "gate": "sources_integrity.required_for_release",
                    "actual": int(sources_integrity_complete),
                    "limit": 1,
                }
            )
        release_enabled = bool(variant_config.get("release_enabled", True))
        if not release_enabled:
            gates["passed"] = False
            gates["failures"].append(
                {
                    "gate": "variant.release_enabled",
                    "actual": 0,
                    "limit": 1,
                }
            )
        artifact_evaluation_complete = bool(
            report.get("backend") != "onnx"
            or (
                isinstance(report.get("artifact_integrity"), dict)
                and report.get("artifact_integrity_timing")
                == "post-evaluation-with-stat-guard"
                and report.get("artifact_integrity_current_verified") is True
            )
        )
        if not artifact_evaluation_complete:
            gates["passed"] = False
            gates["failures"].append(
                {
                    "gate": "artifact_integrity.evaluation_time_required_for_release",
                    "actual": int(artifact_evaluation_complete),
                    "limit": 1,
                }
            )
        span_counts = comparison["agreement"]["spans"]
        char_span_counts = (
            production_like_comparison["agreement"]["spans"]
            if production_like_comparison is not None
            else None
        )
        min_agreement = variant_config.get("min_entity_agreement")
        if min_agreement is not None:
            # Dice/F1 agreement is symmetric: added spans count just as much as
            # removed spans.  agreed/golden alone would ignore false additions.
            denominator = span_counts.get("golden", 0) + span_counts.get("candidate", 0)
            agreement = (
                2 * span_counts.get("agreed", 0) / denominator
                if denominator
                else 1.0
            )
            gates["entity_agreement_f1"] = agreement
            if agreement < float(min_agreement):
                gates["passed"] = False
                gates["failures"].append(
                    {
                        "gate": "min_entity_agreement",
                        "actual": agreement,
                        "limit": float(min_agreement),
                    }
                )
            if char_span_counts is not None:
                char_denominator = char_span_counts.get(
                    "golden", 0
                ) + char_span_counts.get("candidate", 0)
                char_agreement = (
                    2 * char_span_counts.get("agreed", 0) / char_denominator
                    if char_denominator
                    else 1.0
                )
                gates["production_like_char_entity_agreement_f1"] = (
                    char_agreement
                )
                if char_agreement < float(min_agreement):
                    gates["passed"] = False
                    gates["failures"].append(
                        {
                            "gate": "production_like_char.min_entity_agreement",
                            "actual": char_agreement,
                            "limit": float(min_agreement),
                        }
                    )
        for gate_name, span_name in (
            ("max_new_spans", "new"),
            ("max_removed_spans", "removed"),
        ):
            if gate_name not in variant_config:
                continue
            actual = int(span_counts.get(span_name, 0))
            limit = int(variant_config[gate_name])
            if actual > limit:
                gates["passed"] = False
                gates["failures"].append(
                    {"gate": gate_name, "actual": actual, "limit": limit}
                )
            if char_span_counts is not None:
                char_actual = int(char_span_counts.get(span_name, 0))
                if char_actual > limit:
                    gates["passed"] = False
                    gates["failures"].append(
                        {
                            "gate": f"production_like_char.{gate_name}",
                            "actual": char_actual,
                            "limit": limit,
                        }
                    )
        comparisons[name] = {
            "prediction_regression": comparison,
            "production_like_prediction_regression": production_like_comparison,
            "quality_gate": gates,
            "quality_evaluation_complete": production_like_complete,
            "quality_policy_complete": quality_policy_complete,
            "prediction_integrity_complete": prediction_digest_complete,
            "validation_integrity_complete": candidate_validation_complete,
            "preprocessor_integrity_complete": candidate_preprocessor_complete,
            "sources_integrity_complete": sources_integrity_complete,
            "artifact_evaluation_integrity_complete": artifact_evaluation_complete,
            "release_enabled": release_enabled,
            "release_eligible": bool(
                full_validation
                and quality_policy_complete
                and production_like_complete
                and prediction_digest_complete
                and candidate_validation_complete
                and candidate_preprocessor_complete
                and sources_integrity_complete
                and artifact_evaluation_complete
                and release_enabled
                and gates["passed"]
            ),
        }

    report = {
        "schema_version": 1,
        "golden": "torch-fp32",
        "validation_scope": {
            "kind": "full-validation" if full_validation else "smoke",
            "expected_documents": expected_documents,
            "documents_by_variant": document_counts,
            "release_eligible": full_validation,
        },
        "validation_integrity": {
            "complete": validation_integrity_complete,
            "variants": validation_integrity,
        },
        "preprocessor_integrity": {
            "complete": preprocessor_integrity_complete,
            "variants": preprocessor_integrity_by_variant,
        },
        "sources_verification": sources_verification,
        "sources_integrity_complete": sources_integrity_complete,
        "prediction_integrity": prediction_integrity,
        "variants": by_name,
        "comparisons": comparisons,
        "fp8": {
            "variant": "fp8",
            "status": "skipped",
            "reason": (
                "ONNX Runtime CPUExecutionProvider non offre un percorso FP8 nativo "
                "general-purpose; un file compresso con dequantizzazione non misurerebbe "
                "un reale deployment CPU FP8."
            ),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "report.md").write_text(_render_markdown(report), encoding="utf-8")
    return report


def _format_mib(value: int | float | None) -> str:
    return "n/d" if value is None else f"{float(value) / 1024 / 1024:.1f}"


def _format_f1(metrics: Any, level: str = "micro") -> str:
    if not isinstance(metrics, dict):
        return "n/d"
    value = metrics.get(level, {}).get("f1")
    return "n/d" if value is None else f"{float(value):.6f}"


def _render_markdown(report: dict[str, Any]) -> str:
    scope = report.get("validation_scope", {})
    expected = scope.get("expected_documents")
    scope_label = "FULL VALIDATION" if scope.get("release_eligible") else "SMOKE - NOT FOR RELEASE"
    integrity_timings = sorted(
        {
            str(item["artifact_integrity_timing"])
            for item in report.get("variants", {}).values()
            if item.get("backend") == "onnx" and item.get("artifact_integrity_timing")
        }
    )
    integrity_label = ", ".join(integrity_timings) if integrity_timings else "n/d"
    rows = [
        "# Quantization regression report",
        "",
        "Golden semantico: `torch-fp32`. Tutte le inferenze sono CPU-only e ogni variante gira in un processo isolato.",
        "",
        f"Scope: **{scope_label}**" + (f" (attesi {expected} documenti)." if expected else "."),
        "",
        f"Integrità artefatti ONNX: **{integrity_label}**. Hash e dimensioni sono verificati rispetto ai file correnti.",
        "",
        "Integrità semantica: validation, tokenizer, configurazione/id2label e file predizioni devono coincidere byte per byte con quelli usati durante l'inferenza.",
        "",
        "| Variante | Documenti | Stato | word micro-F1 | word macro-F1 | char micro-F1 | char macro-F1 | Dimensione MiB | RSS load delta MiB | RSS peak delta MiB | docs/s | Gate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    comparisons = report.get("comparisons", {})
    for name, item in report["variants"].items():
        metrics = item.get("metrics", {})
        char_metrics = metrics.get("production_like_char")
        gate = comparisons.get(name, {}).get("quality_gate", {})
        rows.append(
            "| {name} | {documents} | {status} | {word_micro} | {word_macro} | {char_micro} | {char_macro} | {size} | {rss_load} | {rss_peak} | {dps:.2f} | {gate} |".format(
                name=name,
                documents=item.get("documents", 0),
                status=item.get("status", "unknown"),
                word_micro=_format_f1(metrics, "micro"),
                word_macro=_format_f1(metrics, "macro"),
                char_micro=_format_f1(char_metrics, "micro"),
                char_macro=_format_f1(char_metrics, "macro"),
                size=_format_mib(item.get("artifact_size_bytes")),
                rss_load=_format_mib(item.get("rss_model_load_delta_bytes")),
                rss_peak=_format_mib(item.get("rss_peak_delta_bytes")),
                dps=float(item.get("documents_per_second", 0.0)),
                gate=(
                    "SMOKE"
                    if not scope.get("release_eligible")
                    else (("PASS" if gate.get("passed") else "FAIL") if gate else "golden")
                ),
            )
        )
    rows.extend(
        [
            "",
            "> Le metriche `char` usano tutti i subword e span esatti sul testo canonico token-list. Sono model-only: il gate end-to-end dopo regex/checksum e merge richiede raw text con offset umani.",
            "",
            "## FP8",
            "",
            f"**Skipped:** {report['fp8']['reason']}",
            "",
            "> Latenza e RSS vanno confermati sulla VPS x86 target: una misura su macOS/ARM non rappresenta i kernel AVX2/VNNI.",
            "",
        ]
    )
    return "\n".join(rows)
