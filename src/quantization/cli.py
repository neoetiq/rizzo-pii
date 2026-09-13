"""Single entry point for the reproducible CPU quantization experiment."""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
from pathlib import Path
from typing import Any, Sequence

from .sources import ROOT, load_config


VARIANT_ALIASES = {
    "fp32": "onnx-fp32",
    "int8": "onnx-int8",
    "int8-linear": "onnx-int8",
    "int8-weight": "onnx-int8-weight",
    "int8-weight-only": "onnx-int8-weight",
    "int8-full": "onnx-int8-full",
    "int8-embedding": "onnx-int8-full",
    "int4": "onnx-int4",
    "int4-a8": "onnx-int4",
    "int4-matmul-gather": "onnx-int4",
    "int4-a32": "onnx-int4-a32",
    "int2": "onnx-int2",
    "int2-a8": "onnx-int2",
}


def _output_dir(config: dict[str, Any], override: Path | None) -> Path:
    path = override or Path(config["output_dir"])
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _layout(output_dir: Path) -> dict[str, Path]:
    return {
        "root": output_dir,
        "checkpoint": output_dir / "model",
        "validation": output_dir / "data" / "validation" / "validation_real.jsonl",
        "calibration": output_dir / "data" / "subsets" / "train_subset_10k.jsonl",
        "onnx-fp32": output_dir / "onnx-fp32",
        "onnx-int8": output_dir / "onnx-int8",
        "onnx-int8-weight": output_dir / "onnx-int8-weight",
        "onnx-int8-full": output_dir / "onnx-int8-full",
        "onnx-int4": output_dir / "onnx-int4",
        "onnx-int4-a32": output_dir / "onnx-int4-a32",
        "onnx-int2": output_dir / "onnx-int2",
        "regression": output_dir / "regression",
    }


def _variant_definitions(paths: dict[str, Path]) -> dict[str, tuple[str, Path, Path]]:
    # Every backend uses the same pinned tokenizer and label map.  Copies next
    # to ONNX graphs are packaging conveniences, not semantic authorities.
    canonical_preprocessor = paths["checkpoint"]
    return {
        "torch-fp32": ("torch", paths["checkpoint"], canonical_preprocessor),
        "onnx-fp32": ("onnx", paths["onnx-fp32"] / "model.onnx", canonical_preprocessor),
        "onnx-int8": ("onnx", paths["onnx-int8"] / "model.onnx", canonical_preprocessor),
        "onnx-int8-weight": (
            "onnx",
            paths["onnx-int8-weight"] / "model.onnx",
            canonical_preprocessor,
        ),
        "onnx-int8-full": (
            "onnx",
            paths["onnx-int8-full"] / "model.onnx",
            canonical_preprocessor,
        ),
        "onnx-int4": ("onnx", paths["onnx-int4"] / "model.onnx", canonical_preprocessor),
        "onnx-int4-a32": (
            "onnx",
            paths["onnx-int4-a32"] / "model.onnx",
            canonical_preprocessor,
        ),
        "onnx-int2": ("onnx", paths["onnx-int2"] / "model.onnx", canonical_preprocessor),
    }


def _torch_artifact_integrity(paths: dict[str, Path]) -> dict[str, Any]:
    """Describe the pinned PyTorch checkpoint without hashing it a second time."""
    lock_path = paths["root"] / "sources.lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    files = []
    for item in lock.get("model", {}).get("files", []):
        relative = Path(str(item["path"]))
        files.append(
            {
                "path": str((paths["checkpoint"] / relative).resolve()),
                "bytes": int(item["bytes"]),
                "sha256": str(item["sha256"]),
            }
        )
    if not files:
        raise RuntimeError(f"Manifest checkpoint PyTorch vuoto: {lock_path}")
    return {
        "model": str(paths["checkpoint"].resolve()),
        "bytes": sum(int(item["bytes"]) for item in files),
        "files": files,
        "revision": lock.get("model", {}).get("revision_resolved"),
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _require(path: Path, description: str) -> Path:
    if not path.exists():
        raise SystemExit(f"{description} non trovato: {path}")
    return path


def _copy_preprocessor(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    names = {
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "merges.txt",
    }
    for item in source.iterdir():
        if item.is_file() and (item.name in names or item.suffix == ".model"):
            shutil.copy2(item, destination / item.name)


def _attach_artifact_integrity(
    reports: Sequence[dict[str, Any]], *, allow_backfill: bool = False
) -> list[str]:
    """Verify evaluation-time hashes, or explicitly backfill a legacy run once."""
    from .quantize import artifact_info

    backfilled: list[str] = []
    for report in reports:
        if report.get("backend") != "onnx":
            continue
        current = artifact_info(Path(report["model_path"]))
        recorded = report.get("artifact_integrity")
        if recorded is None:
            if not allow_backfill:
                raise SystemExit(
                    f"Metriche {report.get('variant')} prive dell'hash evaluation-time; "
                    "rieseguire l'inferenza o usare una sola volta --allow-integrity-backfill"
                )
            report["artifact_integrity"] = current
            report["artifact_integrity_timing"] = "backfilled-after-evaluation"
            backfilled.append(str(report.get("variant")))
        elif recorded != current:
            raise SystemExit(
                f"Artefatto {report.get('variant')} diverso da quello legato alle metriche"
            )
        report["artifact_integrity_current_verified"] = True
    return backfilled


def _int8_weight_base_is_current(paths: dict[str, Path]) -> bool:
    """Return whether the reusable INT8 base and its FP32 source match its report."""
    base_model = paths["onnx-int8-weight"] / "model.onnx"
    report_path = paths["onnx-int8-weight"] / "quantization-report.json"
    fp32_model = paths["onnx-fp32"] / "model.onnx"
    if not base_model.is_file() or not report_path.is_file() or not fp32_model.is_file():
        return False
    try:
        from .quantize import artifact_info

        report = json.loads(report_path.read_text(encoding="utf-8"))
        return bool(
            report.get("artifact") == artifact_info(base_model)
            and report.get("source") == artifact_info(fp32_model)
        )
    except Exception:
        # An unreadable graph/report is not a reusable cache entry.  Rebuilding
        # it yields the normal, more specific quantization error if dependencies
        # or the FP32 source are actually unavailable.
        return False


def command_sources(args: argparse.Namespace, config: dict[str, Any], paths: dict[str, Path]) -> None:
    from .sources import prepare_sources

    result = prepare_sources(args.config, paths["root"])
    print(json.dumps({"status": "ok", "lock_path": result["lock_path"]}, indent=2))


def command_export(args: argparse.Namespace, config: dict[str, Any], paths: dict[str, Path]) -> None:
    from .export import export_model
    from .sources import verify_sources

    sources_verification = verify_sources(args.config, paths["root"])
    checkpoint = _require(paths["checkpoint"], "Checkpoint FP32")
    runtime = config["runtime"]
    report = export_model(
        checkpoint,
        paths["onnx-fp32"],
        opset=args.opset,
        validation_shapes=runtime["validation_shapes"],
        prefer_optimum=not args.torch_only,
    )
    report["sources_verification"] = sources_verification
    _write_json(paths["onnx-fp32"] / "export-report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def _quantize_one(name: str, config: dict[str, Any], paths: dict[str, Path]) -> dict[str, Any]:
    from .quantize import (
        quantize_int2,
        quantize_int4,
        quantize_int8,
        quantize_int8_weight_only,
    )

    fp32_model = _require(paths["onnx-fp32"] / "model.onnx", "ONNX FP32")
    destination = paths[name]
    destination.mkdir(parents=True, exist_ok=True)
    runtime = config["runtime"]
    if name == "onnx-int8":
        report = quantize_int8(
            fp32_model,
            destination / "model.onnx",
            validation_shapes=runtime["validation_shapes"],
        )
        report["embedding_quantized"] = False
        report["label"] = "INT8 dynamic linear-only"
    elif name == "onnx-int8-weight":
        report = quantize_int8_weight_only(
            fp32_model,
            destination / "model.onnx",
            block_size=int(runtime["int4_block_size"]),
            validation_shapes=runtime["validation_shapes"],
        )
        report["embedding_quantized"] = False
        report["label"] = "INT8 MatMulNBits W8A8 dinamico (embedding FP32)"
    elif name == "onnx-int8-full":
        from .embedding import quantize_embedding_int8

        base_model = _require(
            paths["onnx-int8-weight"] / "model.onnx",
            "ONNX INT8 weight-only usato come base per l'embedding INT8",
        )
        base_report_path = _require(
            paths["onnx-int8-weight"] / "quantization-report.json",
            "Report INT8 weight-only",
        )
        report = quantize_embedding_int8(
            base_model,
            destination / "model.onnx",
            validation_shapes=runtime["validation_shapes"],
        )
        base_report = json.loads(base_report_path.read_text(encoding="utf-8"))
        report["matmul_quantization"] = base_report.get("coverage")
        report["base_variant"] = "onnx-int8-weight"
        report["embedding_quantized"] = True
        report["accuracy_level"] = 4
        report["activation_quantization"] = "dynamic INT8 inside MatMulNBits"
        report["compute_path"] = "W8A8 MatMul + embedding W8 per-row"
        report["label"] = "INT8 completo: MatMul W8A8 dinamico + embedding W8 per-row"
    elif name in {"onnx-int4", "onnx-int4-a32"}:
        accuracy_level = 1 if name == "onnx-int4-a32" else 4
        report = quantize_int4(
            fp32_model,
            destination / "model.onnx",
            block_size=int(runtime["int4_block_size"]),
            accuracy_level=accuracy_level,
            validation_shapes=runtime["validation_shapes"],
        )
        gather_count = report["coverage"]["quantized_result"].get("GatherBlockQuantized", 0)
        if gather_count < 1:
            raise RuntimeError(
                "INT4 incompleto: nessun GatherBlockQuantized; l'embedding dominante e rimasto FP32"
            )
        report["embedding_quantized"] = True
        report["label"] = (
            "INT4 DEFAULT symmetric affine: W4A32 MatMul + W4 Gather"
            if accuracy_level == 1
            else "INT4 DEFAULT symmetric affine: W4A8 MatMul + W4 Gather"
        )
    elif name == "onnx-int2":
        report = quantize_int2(
            fp32_model,
            destination / "model.onnx",
            block_size=int(runtime["int4_block_size"]),
            accuracy_level=4,
            validation_shapes=runtime["validation_shapes"],
        )
        gather_count = report["coverage"]["quantized_result"].get(
            "GatherBlockQuantized", 0
        )
        if gather_count < 1:
            raise RuntimeError(
                "INT2 incompleto: nessun GatherBlockQuantized; embedding rimasto FP32"
            )
        report["embedding_quantized"] = True
        report["label"] = (
            "Mixed-bit DEFAULT: W2A8 MatMul + W4 embedding Gather"
        )
    else:
        raise ValueError(name)
    _copy_preprocessor(paths["onnx-fp32"], destination)
    _write_json(destination / "quantization-report.json", report)
    return report


def command_quantize(args: argparse.Namespace, config: dict[str, Any], paths: dict[str, Path]) -> None:
    from .sources import verify_sources

    sources_verification = verify_sources(args.config, paths["root"])
    requested = (
        [
            "onnx-int8",
            "onnx-int8-weight",
            "onnx-int8-full",
            "onnx-int4-a32",
            "onnx-int4",
        ]
        if args.variant == "all"
        else [VARIANT_ALIASES[args.variant]]
    )
    reports: dict[str, Any] = {}
    for name in (
        "onnx-int8",
        "onnx-int8-weight",
        "onnx-int8-full",
        "onnx-int2",
        "onnx-int4-a32",
        "onnx-int4",
    ):
        existing = paths[name] / "quantization-report.json"
        if existing.is_file() and name not in requested:
            reports[name] = json.loads(existing.read_text(encoding="utf-8"))
    if "onnx-int8-full" in requested and "onnx-int8-weight" not in requested:
        if not _int8_weight_base_is_current(paths):
            reports["onnx-int8-weight"] = _quantize_one(
                "onnx-int8-weight", config, paths
            )
    reports.update({name: _quantize_one(name, config, paths) for name in requested})
    for name, report in reports.items():
        # Migrate reports produced by the first spike, whose label called the
        # ORT DEFAULT affine config "RTN" even though RTN is a distinct API.
        if name in {"onnx-int8-weight", "onnx-int2", "onnx-int4-a32", "onnx-int4"} and report.get("algorithm") == "RTN symmetric weight-only":
            report["algorithm"] = "ONNX Runtime DEFAULT symmetric affine weight-only"
            report["algorithm_id"] = "DEFAULT"
        if name in {"onnx-int4-a32", "onnx-int4"}:
            accuracy_level = 1 if name == "onnx-int4-a32" else 4
            report["accuracy_level"] = accuracy_level
            report["activation_quantization"] = (
                "none (FP32 MatMul input)"
                if accuracy_level == 1
                else "dynamic INT8 inside MatMulNBits"
            )
            report["compute_path"] = (
                "W4A32 MatMul; Gather dequantizes embeddings to FP32"
                if accuracy_level == 1
                else "W4A8 MatMul; Gather dequantizes embeddings to FP32"
            )
            report["label"] = (
                "INT4 DEFAULT symmetric affine: W4A32 MatMul + W4 Gather"
                if accuracy_level == 1
                else "INT4 DEFAULT symmetric affine: W4A8 MatMul + W4 Gather"
            )
        if name == "onnx-int2":
            report["experimental"] = True
            report["default_pipeline"] = False
            report["accuracy_level"] = 4
            report["activation_quantization"] = "dynamic INT8 inside MatMulNBits"
            report["compute_path"] = (
                "W2A8 MatMul; W4 Gather dequantizes embeddings to FP32"
            )
            report["label"] = (
                "Mixed-bit DEFAULT: W2A8 MatMul + W4 embedding Gather"
            )
        if name in {"onnx-int8-weight", "onnx-int8-full"}:
            report["accuracy_level"] = 4
            report["activation_quantization"] = "dynamic INT8 inside MatMulNBits"
            report["compute_path"] = "W8A8 MatMul; other activations remain FP32"
            report["label"] = (
                "INT8 completo: MatMul W8A8 dinamico + embedding W8 per-row"
                if name == "onnx-int8-full"
                else "INT8 MatMulNBits W8A8 dinamico (embedding FP32)"
            )
        report["sources_verification"] = sources_verification
        _write_json(paths[name] / "quantization-report.json", report)
    fp8 = {
        "variant": "fp8",
        "status": "skipped",
        "reason": "nessun percorso FP8 nativo nel provider CPU general-purpose di ONNX Runtime",
    }
    _write_json(paths["root"] / "quantization-report.json", {"sources_verification": sources_verification, "variants": reports, "fp8": fp8})
    print(json.dumps({"variants": reports, "fp8": fp8}, ensure_ascii=False, indent=2))


def command_evaluate_one(args: argparse.Namespace) -> None:
    from .evaluate import evaluate_variant

    report = evaluate_variant(
        name=args.name,
        backend=args.backend,
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        validation_path=args.validation_path,
        predictions_path=args.predictions_out,
        report_path=args.report_out,
        batch_size=args.batch_size,
        max_length=args.max_length,
        threads=args.threads,
        limit=args.limit,
        warmup_batches=args.warmup_batches,
        defer_artifact_hash=args.defer_artifact_hash,
    )
    print(
        json.dumps(
            {
                "variant": report["variant"],
                "status": report["status"],
                "micro_f1": report["metrics"]["micro"]["f1"],
                "report": str(args.report_out),
            },
            ensure_ascii=False,
        )
    )


def _parse_variants(values: Sequence[str] | None) -> list[str]:
    if not values:
        return [
            "torch-fp32",
            "onnx-fp32",
            "onnx-int8",
            "onnx-int8-weight",
            "onnx-int8-full",
            "onnx-int4-a32",
            "onnx-int4",
        ]
    result: list[str] = []
    for raw in values:
        for value in raw.split(","):
            value = value.strip()
            if value == "torch-fp32":
                canonical = value
            else:
                canonical = VARIANT_ALIASES.get(value, value)
            if canonical not in {
                "torch-fp32",
                "onnx-fp32",
                "onnx-int8",
                "onnx-int8-weight",
                "onnx-int8-full",
                "onnx-int2",
                "onnx-int4-a32",
                "onnx-int4",
            }:
                raise SystemExit(f"Variante sconosciuta: {value}")
            if canonical not in result:
                result.append(canonical)
    if "torch-fp32" not in result:
        result.insert(0, "torch-fp32")
    return result


def command_regress(args: argparse.Namespace, config: dict[str, Any], paths: dict[str, Path]) -> None:
    from .benchmark import build_regression_report, evaluate_isolated
    from .sources import verify_sources

    sources_preflight = verify_sources(args.config, paths["root"])
    variants = _parse_variants(args.variants)
    validation = _require(paths["validation"], "Validation 7k")
    runtime = config["runtime"]
    batch_size = args.batch_size or int(runtime["batch_size"])
    max_length = int(runtime["max_length"])
    threads = args.threads or int(runtime["threads"])
    definitions = _variant_definitions(paths)
    reports = []
    for name in variants:
        backend, model_path, tokenizer_path = definitions[name]
        _require(model_path, f"Artefatto {name}")
        _require(tokenizer_path, f"Tokenizer {name}")
        print(f"Valuto {name} in un processo CPU isolato...", flush=True)
        reports.append(
            evaluate_isolated(
                root=ROOT,
                name=name,
                backend=backend,
                model_path=model_path,
                tokenizer_path=tokenizer_path,
                validation_path=validation,
                output_dir=paths["regression"],
                batch_size=batch_size,
                max_length=max_length,
                threads=threads,
                limit=args.limit,
            )
        )
    _attach_artifact_integrity(reports)
    sources_postflight = verify_sources(args.config, paths["root"])
    if sources_postflight != sources_preflight:
        raise SystemExit("Sorgenti modificate durante la regressione")
    sources_verification = {
        **sources_preflight,
        "evaluation_guarded": True,
        "verification_timing": "pre-and-post-regression",
        "postflight_lock_sha256": sources_postflight["lock_sha256"],
    }
    for item in reports:
        item["sources_verification"] = sources_verification
        _write_json(
            paths["regression"] / "metrics" / f"{item['variant']}.json",
            item,
        )
    report = build_regression_report(
        variant_reports=reports,
        output_dir=paths["regression"],
        gates_config=config.get("gates", {}),
        expected_documents=int(config["dataset"]["validation"]["expected_rows"]),
        sources_verification=sources_verification,
    )
    failed = [
        name
        for name, comparison in report["comparisons"].items()
        if not comparison.get("release_eligible", False)
    ]
    print(json.dumps({"status": "ok", "report": str(paths["regression"] / "report.md"), "failed_or_ineligible": failed, "comparisons": report["comparisons"]}, ensure_ascii=False, indent=2))
    if args.enforce_gates and failed:
        raise SystemExit("Gate non superati o run non promuovibile: " + ", ".join(failed))


def command_report(args: argparse.Namespace, config: dict[str, Any], paths: dict[str, Path]) -> None:
    """Rebuild comparisons/gates from existing isolated inference outputs."""
    if args.profile:
        _rebuild_profile_report(args, paths)
        return

    from .benchmark import build_regression_report
    from .sources import verify_sources

    output_dir = paths["regression"]
    requested = args.variants
    variants = _parse_variants(requested)
    reports = []
    for name in variants:
        metrics_path = _require(
            output_dir / "metrics" / f"{name}.json",
            f"Metriche {name}",
        )
        reports.append(json.loads(metrics_path.read_text(encoding="utf-8")))
    backfilled = _attach_artifact_integrity(
        reports, allow_backfill=args.allow_integrity_backfill
    )
    for item in reports:
        _write_json(output_dir / "metrics" / f"{item['variant']}.json", item)
    current_sources = verify_sources(args.config, paths["root"])
    recorded_sources = [
        item.get("sources_verification")
        for item in reports
        if isinstance(item.get("sources_verification"), dict)
    ]
    evaluation_guarded = bool(
        len(recorded_sources) == len(reports)
        and recorded_sources
        and all(item == recorded_sources[0] for item in recorded_sources)
        and recorded_sources[0].get("evaluation_guarded") is True
        and recorded_sources[0].get("lock_sha256")
        == current_sources.get("lock_sha256")
    )
    sources_verification = {
        **current_sources,
        "evaluation_guarded": evaluation_guarded,
        "verification_timing": "report-rebuild-current-plus-recorded-evaluation",
        "evaluation_lock_sha256": (
            recorded_sources[0].get("lock_sha256") if recorded_sources else None
        ),
    }
    report = build_regression_report(
        variant_reports=reports,
        output_dir=output_dir,
        gates_config=config.get("gates", {}),
        expected_documents=int(config["dataset"]["validation"]["expected_rows"]),
        sources_verification=sources_verification,
    )
    failed = [
        name
        for name, comparison in report["comparisons"].items()
        if not comparison.get("release_eligible", False)
    ]
    print(json.dumps({"status": "ok", "report": str(output_dir / "report.md"), "failed_or_ineligible": failed, "integrity_backfilled": backfilled}, ensure_ascii=False, indent=2))
    if args.enforce_gates and failed:
        raise SystemExit("Gate non superati o run non promuovibile: " + ", ".join(failed))


def command_profile(args: argparse.Namespace, config: dict[str, Any], paths: dict[str, Path]) -> None:
    """Measure a repeated, process-isolated REST-like batch-1 profile."""
    from datetime import datetime, timezone
    from uuid import uuid4

    from .profile import (
        aggregate_profile_runs,
        build_profile_schedule,
        run_profile_subprocess,
        write_profile_report,
    )
    from .quantize import artifact_info
    from .sources import verify_sources

    requested = args.variants or [
        "torch-fp32",
        "int8-weight",
        "int8-full",
        "int4-a32",
        "int4-a8",
    ]
    variants = _parse_variants(requested)
    if args.documents < 1:
        raise SystemExit("--documents deve essere positivo")
    repeats = 2 * len(variants) if args.repeats is None else args.repeats
    if repeats < 1:
        raise SystemExit("--repeats deve essere positivo")
    expected_validation_rows = int(config["dataset"]["validation"]["expected_rows"])
    if args.documents > expected_validation_rows:
        raise SystemExit(
            f"--documents supera la validation disponibile ({expected_validation_rows})"
        )
    if args.warmup_batches < 0:
        raise SystemExit("--warmup-batches deve essere >= 0")

    sources_preflight = verify_sources(args.config, paths["root"])
    runtime = config["runtime"]
    definitions = _variant_definitions(paths)
    max_length = int(runtime["max_length"])
    threads = args.threads or int(runtime["threads"])
    output_dir = paths["root"] / "service-profile"
    suite_id = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        + "-"
        + uuid4().hex[:12]
    )
    suite_dir = (output_dir / "suites" / suite_id).resolve()
    suite_dir.mkdir(parents=True, exist_ok=False)
    validation = _require(paths["validation"], "Validation")

    # Hash every requested artifact before scheduling any run.  This proves
    # identity and gives all formats the same documented warm-cache policy.
    integrity: dict[str, dict[str, Any]] = {}
    for name in variants:
        backend, model_path, tokenizer_path = definitions[name]
        _require(model_path, f"Artefatto {name}")
        _require(tokenizer_path, f"Tokenizer {name}")
        integrity[name] = (
            _torch_artifact_integrity(paths)
            if backend == "torch"
            else artifact_info(model_path)
        )

    schedule = build_profile_schedule(
        variants, repeats=repeats, seed=args.seed, order=args.order
    )
    configuration = {
        "documents": args.documents,
        "batch_size": 1,
        "max_length": max_length,
        "threads": threads,
        "warmup_batches": args.warmup_batches,
        "memory_sample_ms": args.memory_sample_ms,
    }
    reports: list[dict[str, Any]] = []
    report_files: list[Path] = []
    for slot in schedule:
        name = str(slot["variant"])
        round_index = int(slot["round"])
        position = int(slot["position"])
        backend, model_path, tokenizer_path = definitions[name]
        report_path = (
            suite_dir
            / "runs"
            / f"round-{round_index + 1:03d}"
            / f"{position + 1:02d}-{name}.json"
        )
        predictions_path = (
            suite_dir
            / "predictions"
            / f"round-{round_index + 1:03d}"
            / f"{position + 1:02d}-{name}.jsonl"
        )
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
            str(validation),
            "--predictions-out",
            str(predictions_path),
            "--report-out",
            str(report_path),
            "--batch-size",
            "1",
            "--max-length",
            str(max_length),
            "--threads",
            str(threads),
            "--limit",
            str(args.documents),
            "--warmup-batches",
            str(args.warmup_batches),
            "--defer-artifact-hash",
        ]
        print(
            f"Profilo {round_index + 1}/{repeats}, posizione {position + 1}: {name}...",
            flush=True,
        )
        process_result = run_profile_subprocess(
            command, ROOT, sample_interval_ms=args.memory_sample_ms
        )
        if process_result["returncode"]:
            tail = str(process_result["stderr"])[-4000:]
            raise RuntimeError(f"Profilo {name} fallito:\n{tail}")
        child = json.loads(report_path.read_text(encoding="utf-8"))
        if Path(str(child.get("predictions_path", ""))).resolve() != predictions_path.resolve():
            raise RuntimeError(
                f"Il run {name} ha referenziato un file di predizione inatteso"
            )
        child["internal_memory"] = child.pop("memory", {})
        child.update(
            {
                "suite_id": suite_id,
                "round": round_index,
                "position": position,
                "configuration": configuration,
                "artifact_integrity": integrity[name],
                "artifact_integrity_timing": "preflight-verified; post-suite-pending",
                "artifact_size_bytes": int(integrity[name]["bytes"]),
                "returncode": int(process_result["returncode"]),
                "process": {
                    "pid": process_result["pid"],
                    "elapsed_seconds": process_result["elapsed_seconds"],
                    "stdout": process_result["stdout"],
                    "stderr": process_result["stderr"],
                },
                "memory": process_result["memory"],
            }
        )
        _write_json(report_path, child)
        reports.append(child)
        report_files.append(report_path)

    # Recheck only after every measured process has terminated.  A mismatch
    # invalidates the suite instead of associating stale metrics with new bytes.
    sources_postflight = verify_sources(args.config, paths["root"])
    for name in variants:
        backend, model_path, _ = definitions[name]
        postflight = (
            _torch_artifact_integrity(paths)
            if backend == "torch"
            else artifact_info(model_path)
        )
        if postflight != integrity[name]:
            raise RuntimeError(f"Artefatto {name} modificato durante il profilo")
    if sources_postflight["lock_sha256"] != sources_preflight["lock_sha256"]:
        raise RuntimeError("Lock delle sorgenti modificato durante il profilo")

    for child, report_path in zip(reports, report_files):
        child["artifact_integrity_timing"] = "preflight-and-post-suite-verified"
        _write_json(report_path, child)

    methodology = {
        "suite_id": suite_id,
        "suite_root": str(suite_dir),
        "order": args.order,
        "seed": args.seed,
        "process_isolation": "fresh child process for every variant and repeat",
        "warmup": "excluded from latency and throughput; included in process peak memory",
        "cache_policy": "preflight-verified-warm-os-page-cache",
        "integrity": "artifacts hashed before scheduling and after the complete suite; per-run ONNX stat guard",
        "memory": "external parent polling; Linux smaps_rollup PSS/USS/anonymous when available",
        "host": platform.platform(),
        "python": platform.python_version(),
    }
    report = aggregate_profile_runs(reports, variants, repeats, methodology)
    report["sources_preflight"] = sources_preflight
    report["sources_postflight"] = sources_postflight
    write_profile_report(report, suite_dir)
    write_profile_report(report, output_dir)
    summary = {
        name: {
            "docs_per_second_median": item["statistics"]["documents_per_second"]["median"],
            "peak_rss_mib_median": (
                float(item["statistics"]["peak_rss_bytes"]["median"]) / 1024 / 1024
            ),
        }
        for name, item in report["variants"].items()
    }
    print(
        json.dumps(
            {
                "status": "ok",
                "suite_id": suite_id,
                "profile": str(output_dir / "report.md"),
                "suite_profile": str(suite_dir / "report.md"),
                "documents": args.documents,
                "repeats": repeats,
                "summary": summary,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _rebuild_profile_report(args: argparse.Namespace, paths: dict[str, Path]) -> None:
    """Revalidate and rewrite an existing repeated service profile."""
    from .profile import aggregate_profile_runs, write_profile_report
    from .quantize import artifact_info
    from .sources import verify_sources

    output_dir = paths["root"] / "service-profile"
    report_path = _require(output_dir / "report.json", "Report profilo ripetuto")
    existing = json.loads(report_path.read_text(encoding="utf-8"))
    if existing.get("kind") != "cpu-service-profile":
        raise SystemExit("Il report esistente usa il vecchio schema: rieseguire profile")
    if int(existing.get("schema_version", 0)) < 2:
        raise SystemExit(
            "Il report non lega i file di predizione a una suite: rieseguire profile"
        )
    suite_id = existing.get("suite_id")
    raw_suite_root = existing.get("suite_root")
    if not isinstance(suite_id, str) or not suite_id or not isinstance(raw_suite_root, str):
        raise SystemExit("Identita della suite mancante: rieseguire profile")
    try:
        suite_root = Path(raw_suite_root).resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise SystemExit(f"Directory della suite non disponibile: {raw_suite_root}") from exc
    suites_root = (output_dir / "suites").resolve()
    if suite_root.parent != suites_root or suite_root.name != suite_id:
        raise SystemExit("Il report punta a una namespace di suite non valida")
    methodology = existing.get("methodology")
    if not isinstance(methodology, dict):
        raise SystemExit("Metodologia della suite mancante: rieseguire profile")
    try:
        methodology_root = Path(str(methodology.get("suite_root", ""))).resolve(
            strict=True
        )
    except (FileNotFoundError, OSError) as exc:
        raise SystemExit("Namespace della metodologia non disponibile") from exc
    if methodology.get("suite_id") != suite_id or methodology_root != suite_root:
        raise SystemExit("Identita della suite incoerente tra report e metodologia")
    variants = list(existing.get("variants_order", []))
    if args.variants and _parse_variants(args.variants) != variants:
        raise SystemExit("Per cambiare le varianti occorre rieseguire profile")
    definitions = _variant_definitions(paths)
    current_sources = verify_sources(args.config, paths["root"])
    for timing in ("sources_preflight", "sources_postflight"):
        recorded_sources = existing.get(timing)
        if not isinstance(recorded_sources, dict) or recorded_sources != current_sources:
            raise SystemExit(
                f"Sorgenti {timing} assenti o diverse da quelle del profilo"
            )
    current_integrity: dict[str, dict[str, Any]] = {}
    for name in variants:
        backend, model_path, _ = definitions[name]
        current = (
            _torch_artifact_integrity(paths)
            if backend == "torch"
            else artifact_info(model_path)
        )
        current_integrity[name] = current
        if current != existing["variants"][name]["artifact_integrity"]:
            raise SystemExit(f"Artefatto {name} diverso da quello profilato")
    try:
        rebuilt = aggregate_profile_runs(
            existing["runs"],
            variants,
            int(existing["repeats"]),
            existing["methodology"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise SystemExit(f"Suite del profilo non verificabile: {exc}") from exc
    for name in variants:
        if rebuilt["variants"][name]["artifact_integrity"] != current_integrity[name]:
            raise SystemExit(
                f"Run della suite {name} legati a un artefatto diverso da quello corrente"
            )
    rebuilt["sources_preflight"] = existing.get("sources_preflight")
    rebuilt["sources_postflight"] = existing.get("sources_postflight")
    write_profile_report(rebuilt, suite_root)
    write_profile_report(rebuilt, output_dir)
    print(json.dumps({"status": "ok", "profile": str(output_dir / "report.md")}, indent=2))


def command_all(args: argparse.Namespace, config: dict[str, Any], paths: dict[str, Path]) -> None:
    command_sources(args, config, paths)
    export_args = argparse.Namespace(
        opset=args.opset, torch_only=args.torch_only, config=args.config
    )
    command_export(export_args, config, paths)
    quant_args = argparse.Namespace(variant="all", config=args.config)
    command_quantize(quant_args, config, paths)
    regress_args = argparse.Namespace(
        variants=None,
        batch_size=args.batch_size,
        threads=args.threads,
        limit=args.limit,
        enforce_gates=args.enforce_gates,
        config=args.config,
    )
    command_regress(regress_args, config, paths)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("sources", help="scarica e verifica modello/dataset bloccati")

    export_parser = subparsers.add_parser("export", help="esporta e valida ONNX FP32")
    export_parser.add_argument("--opset", type=int, default=18)
    export_parser.add_argument("--torch-only", action="store_true")

    quantize_parser = subparsers.add_parser("quantize", help="crea candidati INT8/INT4")
    quantize_parser.add_argument(
        "--variant",
        choices=[
            "all",
            "int8",
            "int8-linear",
            "int8-weight",
            "int8-weight-only",
            "int8-full",
            "int8-embedding",
            "int2",
            "int2-a8",
            "int4",
            "int4-a8",
            "int4-a32",
            "int4-matmul-gather",
        ],
        default="all",
    )

    evaluate_parser = subparsers.add_parser("evaluate-one", help=argparse.SUPPRESS)
    evaluate_parser.add_argument("--name", required=True)
    evaluate_parser.add_argument("--backend", choices=["torch", "onnx"], required=True)
    evaluate_parser.add_argument("--model-path", type=Path, required=True)
    evaluate_parser.add_argument("--tokenizer-path", type=Path, required=True)
    evaluate_parser.add_argument("--validation-path", type=Path, required=True)
    evaluate_parser.add_argument("--predictions-out", type=Path, required=True)
    evaluate_parser.add_argument("--report-out", type=Path, required=True)
    evaluate_parser.add_argument("--batch-size", type=int, default=16)
    evaluate_parser.add_argument("--max-length", type=int, default=768)
    evaluate_parser.add_argument("--threads", type=int, default=1)
    evaluate_parser.add_argument("--limit", type=int, default=None)
    evaluate_parser.add_argument("--warmup-batches", type=int, default=0)
    evaluate_parser.add_argument("--defer-artifact-hash", action="store_true")

    regress_parser = subparsers.add_parser("regress", help="regressione qualità/RAM/CPU isolata")
    regress_parser.add_argument("--variants", nargs="*", default=None)
    regress_parser.add_argument("--batch-size", type=int, default=None)
    regress_parser.add_argument("--threads", type=int, default=None)
    regress_parser.add_argument("--limit", type=int, default=None)
    regress_parser.add_argument("--enforce-gates", action="store_true")

    benchmark_parser = subparsers.add_parser("benchmark", help="alias di regress")
    benchmark_parser.add_argument("--variants", nargs="*", default=None)
    benchmark_parser.add_argument("--batch-size", type=int, default=None)
    benchmark_parser.add_argument("--threads", type=int, default=None)
    benchmark_parser.add_argument("--limit", type=int, default=None)
    benchmark_parser.add_argument("--enforce-gates", action="store_true")

    report_parser = subparsers.add_parser("report", help="rigenera gate/report senza inferenza")
    report_parser.add_argument("--variants", nargs="*", default=None)
    report_parser.add_argument("--enforce-gates", action="store_true")
    report_parser.add_argument("--profile", action="store_true", help="usa gli output service-profile")
    report_parser.add_argument(
        "--allow-integrity-backfill",
        action="store_true",
        help="lega esplicitamente un vecchio run agli artefatti correnti",
    )

    profile_parser = subparsers.add_parser("profile", help="profilo REST batch-1 su un subset")
    profile_parser.add_argument("--variants", nargs="*", default=None)
    profile_parser.add_argument("--documents", type=int, default=256)
    profile_parser.add_argument("--threads", type=int, default=None)
    profile_parser.add_argument(
        "--repeats",
        type=int,
        default=None,
        help="repeat per variante; default: 2 x numero varianti",
    )
    profile_parser.add_argument("--warmup-batches", type=int, default=8)
    profile_parser.add_argument("--seed", type=int, default=20260822)
    profile_parser.add_argument(
        "--order", choices=["balanced", "random", "fixed"], default="balanced"
    )
    profile_parser.add_argument("--memory-sample-ms", type=int, default=25)

    all_parser = subparsers.add_parser("all", help="sources + export + quantize + regress")
    all_parser.add_argument("--opset", type=int, default=18)
    all_parser.add_argument("--torch-only", action="store_true")
    all_parser.add_argument("--batch-size", type=int, default=None)
    all_parser.add_argument("--threads", type=int, default=None)
    all_parser.add_argument("--limit", type=int, default=None)
    all_parser.add_argument("--enforce-gates", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_config(args.config)
    paths = _layout(_output_dir(config, args.output_dir))
    if args.command == "evaluate-one":
        command_evaluate_one(args)
        return 0
    commands = {
        "sources": command_sources,
        "export": command_export,
        "quantize": command_quantize,
        "regress": command_regress,
        "benchmark": command_regress,
        "report": command_report,
        "profile": command_profile,
        "all": command_all,
    }
    commands[args.command](args, config, paths)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
