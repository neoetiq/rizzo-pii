"""Fine-tuning riproducibile di mmBERT-small per PII token classification.

Il comando e' separato dal trainer CUDA legacy. Supporta MPS, usa un contratto
di 45 label fissato dal teacher e non materializza il corpus in liste Python.
Il dataset locale 10k e' esplicitamente un proxy meccanico; i run di qualita'
devono usare raw text + span carattere da clean/Ai4Privacy.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import re
import shutil
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.quantization.metrics import compute_metrics
from src.training.student_utils import (
    StudentTrainingError,
    TOKENIZER_FIX_MISTRAL_REGEX,
    align_char_spans,
    align_word_labels,
    build_run_manifest,
    load_label_contract,
    prediction_documents,
    sha256_file,
    tokenizer_load_policy,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path(__file__).with_name("student-small-mps.json")
LR_SCHEDULER_TYPE = "linear"
CHECKPOINT_REQUIRED_FILES = (
    "config.json",
    "optimizer.pt",
    "rng_state.pth",
    "scheduler.pt",
    "trainer_state.json",
    "training_args.bin",
)
CHECKPOINT_WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin")


def _root_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _load_config(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        config = json.loads(source.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise StudentTrainingError(f"Config student non trovato: {source}") from exc
    except json.JSONDecodeError as exc:
        raise StudentTrainingError(f"Config student non valido: {source}") from exc
    if config.get("schema_version") != 1:
        raise StudentTrainingError("schema_version student non supportata")
    for key in ("backbone", "teacher", "data", "training"):
        if not isinstance(config.get(key), Mapping):
            raise StudentTrainingError(f"Sezione config mancante: {key}")
    return config


def _software_versions() -> dict[str, str | None]:
    result: dict[str, str | None] = {"python": sys.version.split()[0]}
    for package in (
        "torch",
        "transformers",
        "accelerate",
        "datasets",
        "pyarrow",
        "safetensors",
        "numpy",
    ):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    return result


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("inspect", "probe", "train"))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--model-source")
    parser.add_argument(
        "--parent-run",
        help=(
            "run completo da cui caricare soltanto i pesi di <run>/final; "
            "optimizer e scheduler ripartono da zero"
        ),
    )
    parser.add_argument("--train-file")
    parser.add_argument("--validation-file")
    parser.add_argument(
        "--input-format",
        choices=("auto", "raw-spans", "legacy-token-bio"),
        default="auto",
    )
    parser.add_argument("--output-dir")
    parser.add_argument("--cache-dir", default="artifacts/training/cache")
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    parser.add_argument("--precision", choices=("fp32", "bf16"))
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--stride", type=int)
    parser.add_argument("--epochs", type=float)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--validation-limit", type=int)
    parser.add_argument("--gradient-accumulation-steps", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument(
        "--optimizer", choices=("adamw_torch_fused", "adamw_torch", "adafactor")
    )
    parser.add_argument("--freeze-embeddings", action="store_true")
    parser.add_argument("--empty-cache-steps", type=int)
    parser.add_argument(
        "--resume-exact",
        metavar="CHECKPOINT",
        help="continua esattamente il checkpoint nella sua run directory originale",
    )
    parser.add_argument("--resume-from-checkpoint", help=argparse.SUPPRESS)
    parser.add_argument("--allow-legacy-proxy", action="store_true")
    args = parser.parse_args(argv)
    if args.resume_exact and args.resume_from_checkpoint:
        parser.error("usare un solo flag fra --resume-exact e --resume-from-checkpoint")
    if args.resume_from_checkpoint:
        warnings.warn(
            "--resume-from-checkpoint e' deprecato e ora applica gli stessi controlli "
            "fail-closed di --resume-exact; aggiornare il comando",
            FutureWarning,
            stacklevel=2,
        )
        args.resume_exact = args.resume_from_checkpoint
    if args.resume_exact and args.parent_run:
        parser.error("--parent-run (weights-only) e --resume-exact sono incompatibili")
    return args


def _resolved_configuration(args: argparse.Namespace, config: Mapping[str, Any]) -> dict[str, Any]:
    training = dict(config["training"])
    backbone = config["backbone"]
    teacher = config["teacher"]
    data = config["data"]
    parent_run = _root_path(args.parent_run) if args.parent_run else None
    default_model_source = (
        parent_run / "final" if parent_run else _root_path(backbone["local_path"])
    )
    model_source = _root_path(args.model_source) if args.model_source else default_model_source
    train_file = _root_path(args.train_file or data["pilot_train"])
    validation_file = _root_path(args.validation_file or data["pilot_validation"])
    return {
        "model_source": str(model_source),
        "parent_run": str(parent_run) if parent_run else None,
        "model_repo": str(backbone["repo_id"]),
        "model_revision": str(backbone["revision"]),
        "teacher_config": str(_root_path(teacher["config_path"])),
        "train_file": str(train_file),
        "validation_file": str(validation_file),
        "input_format": args.input_format,
        "cache_dir": str(_root_path(args.cache_dir)),
        "device": args.device,
        "precision": args.precision or training["precision"],
        "max_length": args.max_length or int(training["max_length"]),
        "stride": args.stride if args.stride is not None else int(training["stride"]),
        "epochs": args.epochs if args.epochs is not None else float(training["epochs"]),
        "max_steps": int(args.max_steps),
        "train_limit": args.train_limit,
        "validation_limit": args.validation_limit,
        "gradient_accumulation_steps": (
            args.gradient_accumulation_steps
            or int(training["gradient_accumulation_steps"])
        ),
        "train_batch_size": int(training["train_batch_size"]),
        "eval_batch_size": int(training["eval_batch_size"]),
        "learning_rate": (
            args.learning_rate
            if args.learning_rate is not None
            else float(training["learning_rate"])
        ),
        "weight_decay": float(training["weight_decay"]),
        "warmup_ratio": float(training["warmup_ratio"]),
        "optimizer": str(args.optimizer or training["optimizer"]),
        "attention_implementation": str(training["attention_implementation"]),
        "gradient_checkpointing": bool(training["gradient_checkpointing"]),
        "mps_memory_fraction": float(training["mps_memory_fraction"]),
        "torch_empty_cache_steps": int(
            args.empty_cache_steps
            if args.empty_cache_steps is not None
            else training["torch_empty_cache_steps"]
        ),
        "max_swap_growth_bytes": int(training["max_swap_growth_bytes"]),
        "min_available_memory_bytes": int(training["min_available_memory_bytes"]),
        "seed": int(training["seed"]),
        "freeze_embeddings": bool(args.freeze_embeddings),
        "legacy_proxy_authorized": bool(args.allow_legacy_proxy),
        "legacy_proxy_declared_by_config": bool(data.get("pilot_is_legacy_proxy")),
        "tokenizer_load_policy": tokenizer_load_policy(),
    }


def _validate_resolved(values: Mapping[str, Any], mode: str) -> None:
    for key in ("model_source", "teacher_config", "train_file", "validation_file"):
        if not Path(str(values[key])).exists():
            raise StudentTrainingError(f"Path richiesto non trovato ({key}): {values[key]}")
    if int(values["max_length"]) < 16:
        raise StudentTrainingError("max_length deve essere almeno 16")
    if int(values["stride"]) < 0 or int(values["stride"]) >= int(values["max_length"]) - 2:
        raise StudentTrainingError("stride non valido per max_length")
    if int(values["gradient_accumulation_steps"]) < 1:
        raise StudentTrainingError("gradient_accumulation_steps deve essere positivo")
    if int(values["torch_empty_cache_steps"]) < 1:
        raise StudentTrainingError("torch_empty_cache_steps deve essere positivo")
    if values["precision"] == "bf16" and values["device"] == "cpu":
        raise StudentTrainingError("Il profilo CPU deve partire in fp32")
    if mode in ("probe", "train"):
        source = Path(str(values["model_source"]))
        if not any((source / name).is_file() for name in ("model.safetensors", "pytorch_model.bin")):
            raise StudentTrainingError(
                f"Pesi mmBERT-small mancanti in {source}; eseguire il download pinned"
            )


def _detect_input_format(columns: set[str], requested: str) -> str:
    if requested != "auto":
        selected = requested
    elif "source_text" in columns and ("entities" in columns or "privacy_mask" in columns):
        selected = "raw-spans"
    elif {"tokens", "bio_labels"} <= columns:
        selected = "legacy-token-bio"
    else:
        raise StudentTrainingError(f"Schema dataset non riconosciuto: {sorted(columns)}")
    if selected == "raw-spans" and not (
        "source_text" in columns and ("entities" in columns or "privacy_mask" in columns)
    ):
        raise StudentTrainingError("raw-spans richiede source_text ed entities/privacy_mask")
    if selected == "legacy-token-bio" and not {"tokens", "bio_labels"} <= columns:
        raise StudentTrainingError("legacy-token-bio richiede tokens e bio_labels")
    return selected


def _select_limit(dataset: Any, limit: int | None, seed: int) -> Any:
    if limit is None or limit >= len(dataset):
        return dataset
    if limit < 1:
        raise StudentTrainingError("I limiti dataset devono essere positivi")
    return dataset.shuffle(seed=seed).select(range(limit))


def _dataset_loader_name(path: str | Path) -> str:
    return "parquet" if Path(path).suffix.lower() in (".parquet", ".pq") else "json"


def _load_training_splits(
    load_dataset: Any,
    *,
    train_file: str | Path,
    validation_file: str | Path,
    cache_dir: str | Path,
) -> dict[str, Any]:
    """Carica ogni split separatamente, senza imporre uno schema Arrow comune."""

    result: dict[str, Any] = {}
    for split, source in (
        ("train", train_file),
        ("validation", validation_file),
    ):
        loaded = load_dataset(
            _dataset_loader_name(source),
            data_files={split: str(source)},
            cache_dir=str(cache_dir),
        )
        try:
            result[split] = loaded[split]
        except (KeyError, TypeError) as exc:
            raise StudentTrainingError(
                f"Loader dataset privo dello split richiesto: {split}"
            ) from exc
    return result


def _select_training_input_columns(dataset: Any, input_format: str) -> Any:
    """Rimuove metadata estranei conservando l'input completo consumato dal trainer."""

    columns = set(dataset.column_names)
    if input_format == "raw-spans":
        span_field = "entities" if "entities" in columns else "privacy_mask"
        selected = ["source_text", span_field]
    elif input_format == "legacy-token-bio":
        selected = ["tokens", "bio_labels"]
    else:
        raise StudentTrainingError(f"Formato input non supportato: {input_format}")
    missing = [name for name in selected if name not in columns]
    if missing:
        raise StudentTrainingError(
            f"Campi input mancanti per {input_format}: {missing}"
        )
    return dataset.select_columns(selected)


def _tokenize_dataset(
    dataset: Any,
    *,
    tokenizer: Any,
    label2id: Mapping[str, int],
    input_format: str,
    max_length: int,
    stride: int,
    description: str,
) -> Any:
    columns = list(dataset.column_names)

    def tokenize_batch(batch: Mapping[str, list[Any]]) -> dict[str, Any]:
        if input_format == "raw-spans":
            encoded = tokenizer(
                batch["source_text"],
                truncation=True,
                max_length=max_length,
                stride=stride,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                return_special_tokens_mask=True,
                padding=False,
            )
            sample_mapping = encoded.pop("overflow_to_sample_mapping")
            offsets = encoded.pop("offset_mapping")
            special_masks = encoded.pop("special_tokens_mask")
            span_field = "entities" if "entities" in batch else "privacy_mask"
            encoded["labels"] = [
                align_char_spans(
                    offsets[index],
                    batch[span_field][sample_index],
                    label2id,
                    special_tokens_mask=special_masks[index],
                )
                for index, sample_index in enumerate(sample_mapping)
            ]
            return encoded

        encoded = tokenizer(
            batch["tokens"],
            is_split_into_words=True,
            truncation=True,
            max_length=max_length,
            stride=stride,
            return_overflowing_tokens=True,
            padding=False,
        )
        sample_mapping = encoded.pop("overflow_to_sample_mapping")
        aligned = []
        for encoded_index, sample_index in enumerate(sample_mapping):
            aligned.append(
                align_word_labels(
                    encoded.word_ids(encoded_index),
                    batch["bio_labels"][sample_index],
                    label2id,
                )
            )
        encoded["labels"] = aligned
        return encoded

    return dataset.map(
        tokenize_batch,
        batched=True,
        batch_size=128,
        remove_columns=columns,
        desc=description,
    )


def _memory_snapshot(
    torch: Any, psutil: Any, device: str
) -> dict[str, int | float | None]:
    process = psutil.Process()
    result: dict[str, int | float | None] = {
        "rss_bytes": int(process.memory_info().rss),
        "system_available_bytes": int(psutil.virtual_memory().available),
        "system_memory_percent": float(psutil.virtual_memory().percent),
        "swap_used_bytes": int(psutil.swap_memory().used),
        "mps_current_allocated_bytes": None,
        "mps_driver_allocated_bytes": None,
        "mps_recommended_max_bytes": None,
    }
    if device == "mps":
        result.update(
            mps_current_allocated_bytes=int(torch.mps.current_allocated_memory()),
            mps_driver_allocated_bytes=int(torch.mps.driver_allocated_memory()),
            mps_recommended_max_bytes=int(torch.mps.recommended_max_memory()),
        )
    return result


def _memory_guard_policy(
    *,
    max_swap_growth_bytes: int,
    min_available_memory_bytes: int,
) -> dict[str, Any]:
    """Descrive la policy stateful registrata insieme ai campioni del run."""

    return {
        "schema_version": 1,
        "swap_growth": {
            "max_bytes": int(max_swap_growth_bytes),
            "comparison": "growth_bytes > max_bytes",
            "trigger": "immediate",
        },
        "available_memory": {
            "min_bytes": int(min_available_memory_bytes),
            "comparison": "system_available_bytes < min_bytes",
            "consecutive_observations_required": 2,
            "reset_counter_on_recovery": True,
            "trigger": "consecutive_observations",
        },
    }


def _new_memory_guard_state(
    policy: Mapping[str, Any], *, swap_at_start: int
) -> dict[str, Any]:
    return {
        "triggered": False,
        "reason": None,
        "phase": None,
        "step": None,
        "policy": dict(policy),
        "swap_at_start_bytes": int(swap_at_start),
        "observation_count": 0,
        "available_memory_consecutive_below": 0,
        "available_memory_max_consecutive_below": 0,
        "samples": [],
    }


def _observe_memory_guard(
    snapshot: Mapping[str, Any],
    *,
    phase: str,
    memory_guard: dict[str, Any],
    step: int | None = None,
) -> str | None:
    """Registra un campione e applica swap immediato / RAM bassa persistente."""

    policy = memory_guard.get("policy")
    if not isinstance(policy, Mapping):
        raise StudentTrainingError("Memory guard privo della policy")
    swap_policy = policy.get("swap_growth")
    available_policy = policy.get("available_memory")
    if not isinstance(swap_policy, Mapping) or not isinstance(
        available_policy, Mapping
    ):
        raise StudentTrainingError("Memory guard policy incompleta")

    swap_used = int(snapshot.get("swap_used_bytes") or 0)
    available = int(snapshot.get("system_available_bytes") or 0)
    swap_growth = swap_used - int(memory_guard["swap_at_start_bytes"])
    below = available < int(available_policy["min_bytes"])
    consecutive = int(memory_guard["available_memory_consecutive_below"])
    consecutive = consecutive + 1 if below else 0
    memory_guard["available_memory_consecutive_below"] = consecutive
    memory_guard["available_memory_max_consecutive_below"] = max(
        int(memory_guard["available_memory_max_consecutive_below"]), consecutive
    )
    observation = int(memory_guard["observation_count"]) + 1
    memory_guard["observation_count"] = observation
    sample = {
        "observation": observation,
        "phase": str(phase),
        "step": None if step is None else int(step),
        "system_available_bytes": available,
        "available_below_threshold": below,
        "available_consecutive_below": consecutive,
        "swap_used_bytes": swap_used,
        "swap_growth_bytes": swap_growth,
    }
    samples = memory_guard.get("samples")
    if not isinstance(samples, list):
        raise StudentTrainingError("Memory guard samples non validi")
    samples.append(sample)

    reason: str | None = None
    if swap_growth > int(swap_policy["max_bytes"]):
        reason = f"swap growth {swap_growth} bytes"
    elif consecutive >= int(
        available_policy["consecutive_observations_required"]
    ):
        reason = (
            f"available memory {available} bytes below threshold for "
            f"{consecutive} consecutive observations"
        )
    if reason is not None and not bool(memory_guard["triggered"]):
        memory_guard.update(
            triggered=True,
            reason=reason,
            phase=str(phase),
            step=None if step is None else int(step),
            trigger_observation=observation,
        )
    if bool(memory_guard["triggered"]):
        return str(memory_guard["reason"])
    return None


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _read_json_object(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise StudentTrainingError(f"{description} non trovato: {path}") from exc
    except json.JSONDecodeError as exc:
        raise StudentTrainingError(f"{description} non valido: {path}") from exc
    if not isinstance(value, dict):
        raise StudentTrainingError(f"{description} deve essere un oggetto JSON: {path}")
    return value


def _file_integrity(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise StudentTrainingError(f"File richiesto non trovato: {path}")
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _require_file_identity(
    recorded: Any,
    current: Mapping[str, Any],
    *,
    description: str,
) -> None:
    if not isinstance(recorded, Mapping):
        raise StudentTrainingError(f"{description} non registrato dal parent")
    try:
        same_path = Path(str(recorded["path"])).expanduser().resolve() == Path(
            str(current["path"])
        ).resolve()
        same_bytes = int(recorded["bytes"]) == int(current["bytes"])
        same_hash = str(recorded["sha256"]) == str(current["sha256"])
    except (KeyError, TypeError, ValueError) as exc:
        raise StudentTrainingError(
            f"{description} incompleto nel parent"
        ) from exc
    if not (same_path and same_bytes and same_hash):
        raise StudentTrainingError(f"{description} modificato dopo il run parent")


def _require_artifact_identity(
    recorded: Any,
    current: Mapping[str, Any],
    *,
    description: str,
) -> None:
    if not isinstance(recorded, Mapping):
        raise StudentTrainingError(f"{description} non registrato dal parent")
    try:
        same_path = Path(str(recorded["path"])).expanduser().resolve() == Path(
            str(current["path"])
        ).resolve()
        same_hash = str(recorded["sha256"]) == str(current["sha256"])
        same_files = list(recorded["files"]) == list(current["files"])
    except (KeyError, TypeError, ValueError) as exc:
        raise StudentTrainingError(
            f"{description} incompleto nel parent"
        ) from exc
    if not (same_path and same_hash and same_files):
        raise StudentTrainingError(f"{description} modificato dopo il run parent")


def _weights_only_parent_context(
    parent_run: str | Path,
    *,
    model_source: str | Path,
) -> dict[str, Any]:
    """Valida un run completo e lega esattamente ``model_source`` a ``run/final``."""

    from src.training.cache_teacher_logits import artifact_group_integrity

    run_dir = _root_path(parent_run).expanduser().resolve()
    if not run_dir.is_dir():
        raise StudentTrainingError(f"Run parent non trovato: {run_dir}")
    final_dir = (run_dir / "final").resolve()
    if Path(model_source).expanduser().resolve() != final_dir:
        raise StudentTrainingError(
            "--model-source deve coincidere esattamente con <parent-run>/final"
        )

    manifest_path = run_dir / "run-manifest.json"
    summary_path = run_dir / "training-summary.json"
    manifest = _read_json_object(manifest_path, "Manifest parent weights-only")
    summary = _read_json_object(summary_path, "Training summary parent weights-only")
    if manifest.get("schema_version") != 1:
        raise StudentTrainingError("schema_version del parent non supportata")
    manifest_status = manifest.get("status")
    if manifest_status != "complete" or manifest.get("result_status") != "complete":
        raise StudentTrainingError("Il manifest parent non e' completo")
    if summary.get("status") != "complete":
        raise StudentTrainingError("Il training parent non e' completo")
    try:
        global_step = int(summary["global_step"])
        planned_steps = int(summary["planned_steps"])
        total_parameters = int(summary["total_parameters"])
        trainable_parameters = int(summary["trainable_parameters"])
    except (KeyError, TypeError, ValueError) as exc:
        raise StudentTrainingError("Training summary parent incompleto") from exc
    if (
        global_step < 1
        or global_step != planned_steps
        or total_parameters < 1
        or trainable_parameters < 1
    ):
        raise StudentTrainingError("Training summary parent non completato")

    manifest_integrity = _file_integrity(manifest_path)
    summary_integrity = _file_integrity(summary_path)
    recorded_summary = manifest.get("training_summary")
    _require_file_identity(
        recorded_summary,
        summary_integrity,
        description="Training summary finale",
    )

    final_model = summary.get("final_model")
    if not isinstance(final_model, Mapping):
        raise StudentTrainingError("Training summary parent privo di final_model")
    model_integrity = artifact_group_integrity(final_dir, "model")
    tokenizer_integrity = artifact_group_integrity(final_dir, "tokenizer")
    _require_artifact_identity(
        final_model.get("model"),
        model_integrity,
        description="Artifact modello finale",
    )
    _require_artifact_identity(
        final_model.get("tokenizer"),
        tokenizer_integrity,
        description="Artifact tokenizer finale",
    )

    return {
        "schema_version": 1,
        "kind": "weights-only-fine-tune-parent",
        "run_dir": str(run_dir),
        "run_manifest": manifest_integrity,
        "training_summary": summary_integrity,
        "model": model_integrity,
        "tokenizer": tokenizer_integrity,
        "initialization": "weights-only",
        "optimizer_state_loaded": False,
        "scheduler_state_loaded": False,
    }


def _checkpoint_integrity(
    checkpoint: str | Path, *, require_incomplete: bool = True
) -> dict[str, Any]:
    path = Path(checkpoint).expanduser().resolve()
    if not path.is_dir() or path.parent.name != "checkpoints":
        raise StudentTrainingError(
            "--resume-exact deve puntare a <run>/checkpoints/checkpoint-N"
        )
    match = re.fullmatch(r"checkpoint-(\d+)", path.name)
    if match is None:
        raise StudentTrainingError(f"Nome checkpoint non valido: {path.name}")
    missing = [name for name in CHECKPOINT_REQUIRED_FILES if not (path / name).is_file()]
    weights = [name for name in CHECKPOINT_WEIGHT_FILES if (path / name).is_file()]
    if missing or len(weights) != 1:
        raise StudentTrainingError(
            f"Checkpoint incompleto: missing={missing}, weight_files={weights}"
        )
    files = [_file_integrity(item) for item in sorted(path.iterdir()) if item.is_file()]
    canonical = json.dumps(
        [{"name": Path(item["path"]).name, "bytes": item["bytes"], "sha256": item["sha256"]} for item in files],
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    state = _read_json_object(path / "trainer_state.json", "trainer_state")
    step = int(match.group(1))
    global_step = int(state.get("global_step", -1))
    planned_steps = int(state.get("max_steps", -1))
    invalid_progress = planned_steps < global_step or (
        require_incomplete and planned_steps == global_step
    )
    if global_step != step or global_step < 1 or invalid_progress:
        raise StudentTrainingError(
            f"Stato checkpoint non continuabile: dir={step}, global={global_step}, planned={planned_steps}"
        )
    return {
        "path": str(path),
        "run_dir": str(path.parent.parent.resolve()),
        "step": global_step,
        "planned_steps": planned_steps,
        "completion_ratio": global_step / planned_steps,
        "sha256": hashlib.sha256(canonical).hexdigest(),
        "files": files,
        "trainer_state": state,
    }


def _authoritative_model_record(
    source: str | Path,
    *,
    kind: str,
    step: int,
) -> dict[str, Any]:
    """Identifica i pesi/tokenizer autorevoli senza duplicare il checkpoint."""

    from src.training.cache_teacher_logits import artifact_group_integrity

    if kind not in {"final-directory", "trainer-checkpoint"}:
        raise StudentTrainingError(f"Tipo artifact autorevole non valido: {kind}")
    root = Path(source).expanduser().resolve()
    if step < 1:
        raise StudentTrainingError("Step artifact autorevole non valido")
    return {
        "schema_version": 1,
        "kind": kind,
        "path": str(root),
        "step": int(step),
        "model": artifact_group_integrity(root, "model"),
        "tokenizer": artifact_group_integrity(root, "tokenizer"),
    }


def _validate_authoritative_checkpoint(
    authoritative: Any,
    checkpoint: Mapping[str, Any],
) -> None:
    """Lega il record compatto ai file gia' verificati da checkpoint_integrity."""

    if not isinstance(authoritative, Mapping):
        raise StudentTrainingError("Summary resume privo dell'artifact autorevole")
    try:
        same_kind = authoritative["kind"] == "trainer-checkpoint"
        same_path = Path(str(authoritative["path"])).resolve() == Path(
            str(checkpoint["path"])
        ).resolve()
        same_step = int(authoritative["step"]) == int(checkpoint["step"])
        checkpoint_files = {
            Path(str(item["path"])).name: (int(item["bytes"]), str(item["sha256"]))
            for item in checkpoint["files"]
        }
        groups = (authoritative["model"], authoritative["tokenizer"])
    except (KeyError, TypeError, ValueError) as exc:
        raise StudentTrainingError("Artifact autorevole resume incompleto") from exc
    if not (same_kind and same_path and same_step):
        raise StudentTrainingError("Artifact autorevole divergente dal checkpoint finale")
    for group in groups:
        if not isinstance(group, Mapping) or not isinstance(group.get("files"), list):
            raise StudentTrainingError("Gruppo artifact autorevole incompleto")
        if Path(str(group.get("path", ""))).resolve() != Path(
            str(checkpoint["path"])
        ).resolve():
            raise StudentTrainingError("Path gruppo artifact divergente dal checkpoint")
        for item in group["files"]:
            if not isinstance(item, Mapping):
                raise StudentTrainingError("File artifact autorevole non valido")
            name = Path(str(item.get("path", ""))).name
            current = checkpoint_files.get(name)
            expected = (int(item.get("bytes", -1)), str(item.get("sha256", "")))
            if current != expected:
                raise StudentTrainingError(
                    f"File artifact autorevole divergente dal checkpoint: {name}"
                )


def _explicit_options(argv: Sequence[str]) -> set[str]:
    return {
        value.split("=", 1)[0]
        for value in argv
        if isinstance(value, str) and value.startswith("--")
    }


def _critical_contract(
    training: Mapping[str, Any],
    data: Mapping[str, Any],
    *,
    trainable_parameters: int,
) -> dict[str, Any]:
    train = data.get("train")
    validation = data.get("validation")
    if not isinstance(train, Mapping) or not isinstance(validation, Mapping):
        raise StudentTrainingError("Manifest parent privo del contratto dati")
    return {
        "optimizer": str(training["optimizer"]),
        "freeze_embeddings": bool(training.get("freeze_embeddings", False)),
        "trainable_parameters": int(trainable_parameters),
        "train_path": str(Path(str(train["path"])).resolve()),
        "train_sha256": str(train["sha256"]),
        "validation_path": str(Path(str(validation["path"])).resolve()),
        "validation_sha256": str(validation["sha256"]),
        "max_length": int(training["max_length"]),
        "stride": int(training["stride"]),
        "train_batch_size": int(training["train_batch_size"]),
        "eval_batch_size": int(training["eval_batch_size"]),
        "gradient_accumulation_steps": int(training["gradient_accumulation_steps"]),
        "learning_rate": float(training["learning_rate"]),
        "weight_decay": float(training["weight_decay"]),
        "warmup_ratio": float(training["warmup_ratio"]),
        "lr_scheduler_type": str(training.get("lr_scheduler_type", LR_SCHEDULER_TYPE)),
        "epochs": float(training["epochs"]),
        "max_steps": int(training["max_steps"]),
        "seed": int(training["seed"]),
        "precision": str(training["precision"]),
    }


def _resume_context(
    args: argparse.Namespace,
    values: dict[str, Any],
    argv: Sequence[str],
) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    checkpoint = _checkpoint_integrity(str(args.resume_exact))
    run_dir = Path(checkpoint["run_dir"])
    if args.mode != "train":
        raise StudentTrainingError("--resume-exact e' ammesso soltanto in modalita train")
    if args.output_dir and _root_path(args.output_dir).resolve() != run_dir:
        raise StudentTrainingError("Il resume exact deve riusare la run directory parent")

    manifest_path = run_dir / "run-manifest.json"
    summary_path = run_dir / "training-summary.json"
    manifest = _read_json_object(manifest_path, "Manifest parent")
    summary = _read_json_object(summary_path, "Training summary parent")
    parent_training = manifest.get("training")
    if manifest.get("schema_version") != 1 or not isinstance(parent_training, Mapping):
        raise StudentTrainingError("Manifest parent non supportato")

    options = _explicit_options(argv)
    option_keys = {
        "--model-source": "model_source",
        "--train-file": "train_file",
        "--validation-file": "validation_file",
        "--input-format": "input_format",
        "--precision": "precision",
        "--max-length": "max_length",
        "--stride": "stride",
        "--epochs": "epochs",
        "--max-steps": "max_steps",
        "--train-limit": "train_limit",
        "--validation-limit": "validation_limit",
        "--gradient-accumulation-steps": "gradient_accumulation_steps",
        "--learning-rate": "learning_rate",
        "--optimizer": "optimizer",
        "--freeze-embeddings": "freeze_embeddings",
    }
    mismatches: list[str] = []
    for option, key in option_keys.items():
        if option not in options or key not in parent_training:
            continue
        current = values[key]
        expected = parent_training[key]
        if key.endswith("_file") or key == "model_source":
            equal = Path(str(current)).resolve() == Path(str(expected)).resolve()
        else:
            equal = current == expected
        if not equal:
            mismatches.append(f"{key}: current={current!r}, parent={expected!r}")
    if mismatches:
        raise StudentTrainingError("Override incompatibili col resume exact: " + "; ".join(mismatches))

    inherited = dict(values)
    for key, value in parent_training.items():
        if key in inherited and key not in {"config_sha256", "resolved_input_format"}:
            inherited[key] = value

    expected = _critical_contract(
        parent_training,
        manifest.get("data", {}),
        trainable_parameters=int(summary["trainable_parameters"]),
    )
    actual_data = {
        "train": {
            "path": inherited["train_file"],
            "sha256": sha256_file(inherited["train_file"]),
        },
        "validation": {
            "path": inherited["validation_file"],
            "sha256": sha256_file(inherited["validation_file"]),
        },
    }
    actual = _critical_contract(
        inherited,
        actual_data,
        trainable_parameters=int(summary["trainable_parameters"]),
    )
    if actual != expected:
        differences = [key for key in expected if actual.get(key) != expected[key]]
        raise StudentTrainingError(f"Contratto resume exact divergente: {differences}")
    teacher_record = manifest.get("teacher_label_contract", {})
    if sha256_file(inherited["teacher_config"]) != teacher_record.get("sha256"):
        raise StudentTrainingError("Contratto label teacher modificato dal run parent")
    fine_tune_parent = manifest.get("fine_tune_parent")
    if fine_tune_parent is not None and not isinstance(fine_tune_parent, Mapping):
        raise StudentTrainingError("Provenienza fine-tune parent non valida")

    context = {
        "checkpoint": checkpoint,
        "parent_manifest": manifest,
        "parent_manifest_integrity": _file_integrity(manifest_path),
        "parent_summary_integrity": _file_integrity(summary_path),
        "contract": expected,
        "expected_total_parameters": int(summary["total_parameters"]),
        "expected_trainable_parameters": int(summary["trainable_parameters"]),
        "fine_tune_parent": (
            dict(fine_tune_parent) if isinstance(fine_tune_parent, Mapping) else None
        ),
        "argv": list(argv),
        "code": _file_integrity(Path(__file__)),
    }
    return inherited, run_dir, context


def _progress_status(global_step: int, planned_steps: int, guard_triggered: bool) -> str:
    if planned_steps < 1 or global_step < 0 or global_step > planned_steps:
        raise StudentTrainingError("Progresso Trainer non valido")
    if global_step == planned_steps:
        return "complete"
    if guard_triggered:
        return "partial_guard_stop"
    raise StudentTrainingError(
        f"Training incompleto senza memory guard: {global_step}/{planned_steps}"
    )


def _begin_continuation(run_dir: Path, context: dict[str, Any]) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    path = run_dir / "continuations" / f"{stamp}-from-step-{context['checkpoint']['step']}.json"
    record = {
        "schema_version": 1,
        "kind": "exact-resume-continuation",
        "status": "running",
        "release_eligible": False,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "argv": context["argv"],
        "code": context["code"],
        "parent_manifest": context["parent_manifest_integrity"],
        "parent_summary": context["parent_summary_integrity"],
        "checkpoint_before": context["checkpoint"],
        "critical_contract": context["contract"],
        "pre": {
            "global_step": context["checkpoint"]["step"],
            "planned_steps": context["checkpoint"]["planned_steps"],
            "completion_ratio": context["checkpoint"]["completion_ratio"],
        },
    }
    _write_json(path, record)
    context["continuation_path"] = str(path)
    context["continuation_record"] = record
    return path


def _remove_redundant_final(run_dir: Path, context: dict[str, Any]) -> None:
    """Rimuove solo il duplicato byte-identico gia' ricostruibile dal checkpoint."""

    final_dir = run_dir / "final"
    if not final_dir.exists():
        return
    checkpoint_files = {
        Path(item["path"]).name: item for item in context["checkpoint"]["files"]
    }
    final_entries = sorted(final_dir.iterdir())
    if not final_entries or any(not item.is_file() for item in final_entries):
        raise StudentTrainingError(f"Directory final inattesa o vuota: {final_dir}")
    final_files = final_entries
    checked: list[dict[str, Any]] = []
    for path in final_files:
        current = _file_integrity(path)
        parent = checkpoint_files.get(path.name)
        if parent is None or (current["bytes"], current["sha256"]) != (
            parent["bytes"],
            parent["sha256"],
        ):
            raise StudentTrainingError(
                f"Non rimuovo final: {path.name} non e' identico al checkpoint"
            )
        checked.append(current)
    shutil.rmtree(final_dir)
    record = context["continuation_record"]
    record["disk_safety"] = {
        "same_run_directory": True,
        "redundant_final_removed": str(final_dir),
        "verified_files": checked,
        "bytes_reclaimed": sum(int(item["bytes"]) for item in checked),
    }
    _write_json(Path(context["continuation_path"]), record)


def _finish_continuation(
    run_dir: Path,
    context: dict[str, Any],
    summary: Mapping[str, Any],
) -> None:
    record = dict(context["continuation_record"])
    global_step = int(summary["global_step"])
    final_checkpoint = _checkpoint_integrity(
        run_dir / "checkpoints" / f"checkpoint-{global_step}",
        require_incomplete=False,
    )
    authoritative = summary.get("authoritative_model")
    _validate_authoritative_checkpoint(authoritative, final_checkpoint)
    if not isinstance(authoritative, Mapping) or summary.get("final_model") != {
        "model": authoritative.get("model"),
        "tokenizer": authoritative.get("tokenizer"),
    }:
        raise StudentTrainingError("final_model non coincide con l'artifact autorevole")
    current_manifest = _file_integrity(run_dir / "run-manifest.json")
    if current_manifest["sha256"] != context["parent_manifest_integrity"]["sha256"]:
        raise StudentTrainingError("Manifest parent modificato durante la continuation")
    current_code = _file_integrity(Path(__file__))
    if current_code["sha256"] != context["code"]["sha256"]:
        raise StudentTrainingError("Codice trainer modificato durante la continuation")
    record.update(
        status=str(summary["status"]),
        completed_at_utc=datetime.now(timezone.utc).isoformat(),
        post={
            "global_step": global_step,
            "planned_steps": int(summary["planned_steps"]),
            "completion_ratio": float(summary["completion_ratio"]),
            "memory_guard": summary["memory_guard"],
            "checkpoint": final_checkpoint,
            "training_summary": _file_integrity(run_dir / "training-summary.json"),
        },
    )
    _write_json(Path(context["continuation_path"]), record)


def _finalize_resumed_manifest(
    run_dir: Path,
    context: Mapping[str, Any],
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    """Aggiorna il puntatore di testa mantenendo la continuation come catena audit."""

    manifest_path = run_dir / "run-manifest.json"
    manifest_integrity = _file_integrity(manifest_path)
    if manifest_integrity["sha256"] != context["parent_manifest_integrity"]["sha256"]:
        raise StudentTrainingError("Manifest parent modificato prima del commit resume")
    manifest = _read_json_object(manifest_path, "Manifest da finalizzare dopo resume")
    continuation_path = Path(str(context.get("continuation_path", ""))).resolve()
    continuation = _read_json_object(continuation_path, "Continuation completata")
    summary_integrity = _file_integrity(run_dir / "training-summary.json")
    post = continuation.get("post")
    if (
        continuation.get("status") != summary.get("status")
        or not isinstance(post, Mapping)
        or post.get("training_summary") != summary_integrity
    ):
        raise StudentTrainingError("Continuation e summary finale non coincidono")
    authoritative = summary.get("authoritative_model")
    if not isinstance(authoritative, Mapping):
        raise StudentTrainingError("Summary finale privo dell'artifact autorevole")

    history_value = manifest.get("continuation_history", [])
    if not isinstance(history_value, list):
        raise StudentTrainingError("Cronologia continuation non valida nel manifest")
    continuation_integrity = _file_integrity(continuation_path)
    history = list(history_value)
    history.append(
        {
            "record": continuation_integrity,
            "from_step": int(context["checkpoint"]["step"]),
            "to_step": int(summary["global_step"]),
            "result_status": str(summary["status"]),
        }
    )
    finalized = dict(manifest)
    finalized.update(
        status="complete",
        result_status=str(summary["status"]),
        release_eligible=False,
        completed_at_utc=str(continuation["completed_at_utc"]),
        training_summary=summary_integrity,
        authoritative_model=dict(authoritative),
        latest_continuation=continuation_integrity,
        continuation_history=history,
    )
    _write_json(manifest_path, finalized)
    return finalized


def _fail_continuation(context: Mapping[str, Any], exc: BaseException) -> None:
    path_value = context.get("continuation_path")
    record_value = context.get("continuation_record")
    if not path_value or not isinstance(record_value, Mapping):
        return
    record = dict(record_value)
    record.update(
        status="failed",
        release_eligible=False,
        completed_at_utc=datetime.now(timezone.utc).isoformat(),
        post={"error_type": type(exc).__name__, "error": str(exc)},
    )
    _write_json(Path(str(path_value)), record)


def _run_training(
    args: argparse.Namespace,
    values: dict[str, Any],
    run_dir: Path,
    resume_context: Mapping[str, Any] | None = None,
    fine_tune_parent: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    # Queste variabili devono essere impostate prima di importare torch.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "0")

    import numpy as np
    import psutil
    import torch
    from datasets import load_dataset
    from transformers import (
        AutoConfig,
        AutoModelForTokenClassification,
        AutoTokenizer,
        DataCollatorForTokenClassification,
        Trainer,
        TrainerCallback,
        TrainingArguments,
        set_seed,
    )

    requested_device = str(values["device"])
    if requested_device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        device = requested_device
    if device == "mps" and not torch.backends.mps.is_available():
        raise StudentTrainingError(
            "MPS non disponibile in questo processo. Nel runner sandboxato e' atteso; "
            "eseguire il comando sul processo host."
        )
    if device == "cpu" and values["precision"] != "fp32":
        raise StudentTrainingError("CPU richiede precision=fp32 nel primo baseline")
    if device == "mps":
        torch.mps.set_per_process_memory_fraction(float(values["mps_memory_fraction"]))

    set_seed(int(values["seed"]), deterministic=True)
    label2id, id2label = load_label_contract(values["teacher_config"])
    tokenizer = AutoTokenizer.from_pretrained(
        values["model_source"],
        trust_remote_code=False,
        fix_mistral_regex=TOKENIZER_FIX_MISTRAL_REGEX,
    )

    raw = _load_training_splits(
        load_dataset,
        train_file=values["train_file"],
        validation_file=values["validation_file"],
        cache_dir=values["cache_dir"],
    )
    raw["train"] = _select_limit(raw["train"], values["train_limit"], int(values["seed"]))
    raw["validation"] = _select_limit(
        raw["validation"], values["validation_limit"], int(values["seed"]) + 1
    )
    input_format = _detect_input_format(
        set(raw["train"].column_names), values["input_format"]
    )
    validation_input_format = _detect_input_format(
        set(raw["validation"].column_names), values["input_format"]
    )
    if validation_input_format != input_format:
        raise StudentTrainingError(
            "Train e validation usano formati input differenti: "
            f"{input_format} != {validation_input_format}"
        )
    raw["train"] = _select_training_input_columns(raw["train"], input_format)
    raw["validation"] = _select_training_input_columns(
        raw["validation"], input_format
    )
    values["resolved_input_format"] = input_format
    if input_format == "legacy-token-bio" and not values["legacy_proxy_authorized"]:
        raise StudentTrainingError(
            "Il dataset e' un proxy legacy tokens/BIO. Ripetere con --allow-legacy-proxy "
            "solo per probe/pilot; non usarlo come evidenza finale di qualita'."
        )

    tokenized_train = _tokenize_dataset(
        raw["train"],
        tokenizer=tokenizer,
        label2id=label2id,
        input_format=input_format,
        max_length=int(values["max_length"]),
        stride=int(values["stride"]),
        description="Tokenizzazione train student",
    )
    tokenized_validation = _tokenize_dataset(
        raw["validation"],
        tokenizer=tokenizer,
        label2id=label2id,
        input_format=input_format,
        max_length=int(values["max_length"]),
        stride=int(values["stride"]),
        description="Tokenizzazione validation student",
    )

    model_config = AutoConfig.from_pretrained(values["model_source"], trust_remote_code=False)
    model_config.num_labels = len(label2id)
    model_config.label2id = dict(label2id)
    model_config.id2label = dict(id2label)
    model, loading_info = AutoModelForTokenClassification.from_pretrained(
        values["model_source"],
        config=model_config,
        trust_remote_code=False,
        attn_implementation=values["attention_implementation"],
        output_loading_info=True,
    )
    missing = list(loading_info.get("missing_keys", []))
    unexpected = list(loading_info.get("unexpected_keys", []))
    mismatched = list(loading_info.get("mismatched_keys", []))
    errors = list(loading_info.get("error_msgs", []))
    if errors or mismatched:
        raise StudentTrainingError(
            f"Caricamento backbone non pulito: errors={errors}, mismatched={mismatched}"
        )
    if any("classifier" not in key for key in missing):
        raise StudentTrainingError(f"Pesi backbone mancanti inattesi: {missing}")
    if any("decoder" not in key for key in unexpected):
        raise StudentTrainingError(f"Pesi backbone inutilizzati inattesi: {unexpected}")

    if values["freeze_embeddings"]:
        for parameter in model.get_input_embeddings().parameters():
            parameter.requires_grad = False
    if values["gradient_checkpointing"]:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    if resume_context is not None:
        if total_parameters != int(resume_context["expected_total_parameters"]):
            raise StudentTrainingError("Numero totale di parametri diverso dal run parent")
        if trainable_parameters != int(resume_context["expected_trainable_parameters"]):
            raise StudentTrainingError("Trainable set diverso dal run parent")

        batches_per_epoch = math.ceil(
            len(tokenized_train) / int(values["train_batch_size"])
        )
        updates_per_epoch = math.ceil(
            batches_per_epoch / int(values["gradient_accumulation_steps"])
        )
        planned_steps = (
            int(values["max_steps"])
            if int(values["max_steps"]) > 0
            else math.ceil(float(values["epochs"]) * updates_per_epoch)
        )
        if planned_steps != int(resume_context["checkpoint"]["planned_steps"]):
            raise StudentTrainingError(
                "Numero di update pianificati diverso dal checkpoint: "
                f"{planned_steps} != {resume_context['checkpoint']['planned_steps']}"
            )
    after_model_load = _memory_snapshot(torch, psutil, device)
    memory_timeline: list[dict[str, Any]] = [
        {"event": "after_model_load", **after_model_load}
    ]
    swap_at_start = int(after_model_load["swap_used_bytes"] or 0)
    memory_guard_policy = _memory_guard_policy(
        max_swap_growth_bytes=int(values["max_swap_growth_bytes"]),
        min_available_memory_bytes=int(values["min_available_memory_bytes"]),
    )
    memory_guard = _new_memory_guard_state(
        memory_guard_policy, swap_at_start=swap_at_start
    )

    def preprocess_logits(logits: Any, _: Any) -> Any:
        if isinstance(logits, tuple):
            logits = logits[0]
        return logits.argmax(dim=-1)

    def trainer_metrics(evaluation: Any) -> dict[str, float]:
        predictions = evaluation.predictions
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        gold_docs, prediction_docs = prediction_documents(
            predictions, evaluation.label_ids, id2label
        )
        metrics = compute_metrics(gold_docs, prediction_docs)
        critical = ("CF", "PIVA", "IBAN", "DOCID", "ID_DOC", "EMAIL", "TELEPHONENUM")
        recalls = [
            float(metrics["per_tag"][tag]["recall"])
            for tag in critical
            if tag in metrics["per_tag"] and int(metrics["per_tag"][tag]["support"]) > 0
        ]
        return {
            "f1_micro": float(metrics["micro"]["f1"]),
            "f1_macro": float(metrics["macro"]["f1"]),
            "recall_micro": float(metrics["micro"]["recall"]),
            "critical_recall_min": min(recalls) if recalls else 0.0,
        }

    class MemoryCallback(TrainerCallback):
        def on_log(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            snapshot = _memory_snapshot(torch, psutil, device)
            memory_timeline.append(
                {"event": "log", "step": int(state.global_step), **snapshot}
            )
            reason = _observe_memory_guard(
                snapshot,
                phase="trainer_log",
                step=int(state.global_step),
                memory_guard=memory_guard,
            )
            if reason is not None:
                control.should_training_stop = True
            return control

    class ExactResumeCallback(TrainerCallback):
        def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            # Il vecchio best farebbe conservare checkpoint-20 e cancellare quello
            # appena scritto con save_total_limit=1. Non influenza optimizer,
            # scheduler, RNG o data-skip e viene ricalcolato alla prossima eval.
            state.best_metric = None
            state.best_model_checkpoint = None
            return control

    is_probe = args.mode == "probe"
    if is_probe:
        eval_strategy = "no"
        save_strategy = "no"
        max_steps = values["max_steps"] if int(values["max_steps"]) > 0 else 1
    else:
        eval_strategy = "epoch"
        save_strategy = "epoch"
        max_steps = int(values["max_steps"])

    training_args = TrainingArguments(
        output_dir=str(run_dir / "checkpoints"),
        overwrite_output_dir=False,
        do_train=True,
        do_eval=not is_probe,
        eval_strategy=eval_strategy,
        save_strategy=save_strategy,
        save_total_limit=1,
        load_best_model_at_end=not is_probe and resume_context is None,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        per_device_train_batch_size=int(values["train_batch_size"]),
        per_device_eval_batch_size=int(values["eval_batch_size"]),
        gradient_accumulation_steps=int(values["gradient_accumulation_steps"]),
        num_train_epochs=float(values["epochs"]),
        max_steps=max_steps,
        learning_rate=float(values["learning_rate"]),
        weight_decay=float(values["weight_decay"]),
        warmup_ratio=float(values["warmup_ratio"]),
        max_grad_norm=1.0,
        lr_scheduler_type=LR_SCHEDULER_TYPE,
        optim=str(values["optimizer"]),
        bf16=(values["precision"] == "bf16"),
        fp16=False,
        use_cpu=(device == "cpu"),
        gradient_checkpointing=bool(values["gradient_checkpointing"]),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        eval_accumulation_steps=16,
        logging_strategy="steps",
        logging_steps=1 if is_probe else 10,
        report_to=[],
        seed=int(values["seed"]),
        data_seed=int(values["seed"]),
        save_safetensors=True,
        torch_empty_cache_steps=int(values["torch_empty_cache_steps"]),
        torch_compile=False,
        run_name=run_dir.name,
    )
    collator = DataCollatorForTokenClassification(tokenizer=tokenizer, padding=True)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=None if is_probe else tokenized_validation,
        data_collator=collator,
        processing_class=tokenizer,
        compute_metrics=None if is_probe else trainer_metrics,
        preprocess_logits_for_metrics=None if is_probe else preprocess_logits,
        callbacks=[MemoryCallback(), *([ExactResumeCallback()] if resume_context else [])],
    )

    started = time.monotonic()
    train_result = trainer.train(resume_from_checkpoint=args.resume_exact)
    if device == "mps":
        torch.mps.synchronize()
    elapsed = time.monotonic() - started
    memory_timeline.append(
        {"event": "after_train", **_memory_snapshot(torch, psutil, device)}
    )

    global_step = int(trainer.state.global_step)
    planned_steps = int(trainer.state.max_steps)
    final_model: dict[str, Any] | None = None
    authoritative_model: dict[str, Any] | None = None
    if not is_probe and resume_context is None:
        final_dir = run_dir / "final"
        trainer.save_model(str(final_dir))
        tokenizer.save_pretrained(str(final_dir))
        authoritative_model = _authoritative_model_record(
            final_dir,
            kind="final-directory",
            step=global_step,
        )
    elif resume_context is not None:
        checkpoint_dir = run_dir / "checkpoints" / f"checkpoint-{global_step}"
        authoritative_model = _authoritative_model_record(
            checkpoint_dir,
            kind="trainer-checkpoint",
            step=global_step,
        )
    if authoritative_model is not None:
        final_model = {
            "model": authoritative_model["model"],
            "tokenizer": authoritative_model["tokenizer"],
        }

    status = _progress_status(global_step, planned_steps, bool(memory_guard["triggered"]))

    summary = {
        "status": status,
        "release_eligible": False,
        "mode": args.mode,
        "device": device,
        "precision": values["precision"],
        "input_format": input_format,
        "legacy_proxy": input_format == "legacy-token-bio",
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "freeze_embeddings": bool(values["freeze_embeddings"]),
        "tokenizer_load_policy": values["tokenizer_load_policy"],
        "train_rows": len(raw["train"]),
        "validation_rows": len(raw["validation"]),
        "train_windows": len(tokenized_train),
        "validation_windows": len(tokenized_validation),
        "elapsed_seconds": elapsed,
        "global_step": global_step,
        "planned_steps": planned_steps,
        "completion_ratio": global_step / planned_steps,
        "resume_exact": bool(resume_context),
        "initialization": (
            "exact-checkpoint"
            if resume_context
            else "weights-only" if fine_tune_parent else "backbone"
        ),
        "resume_checkpoint": (
            {
                "path": resume_context["checkpoint"]["path"],
                "step": resume_context["checkpoint"]["step"],
                "sha256": resume_context["checkpoint"]["sha256"],
            }
            if resume_context
            else None
        ),
        "fine_tune_parent": dict(fine_tune_parent) if fine_tune_parent else None,
        "parent_initialization": "weights-only" if fine_tune_parent else None,
        "optimizer_state_loaded": bool(resume_context),
        "scheduler_state_loaded": bool(resume_context),
        "trainer_metrics": dict(train_result.metrics),
        "loading_info": {
            "missing_keys": missing,
            "unexpected_keys": unexpected,
            "mismatched_keys": mismatched,
        },
        "memory_timeline": memory_timeline,
        "memory_guard": memory_guard,
        "final_model": final_model,
        "authoritative_model": authoritative_model,
    }
    _write_json(run_dir / "training-summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = _parse_args(raw_argv)
    config = _load_config(args.config)
    values = _resolved_configuration(args, config)

    if args.mode == "probe":
        values["train_limit"] = values["train_limit"] or 4
        values["validation_limit"] = values["validation_limit"] or 4
        values["max_steps"] = values["max_steps"] if int(values["max_steps"]) > 0 else 1
    fine_tune_parent: dict[str, Any] | None = None
    if args.parent_run:
        fine_tune_parent = _weights_only_parent_context(
            args.parent_run,
            model_source=values["model_source"],
        )
    resume_context: dict[str, Any] | None = None
    if args.resume_exact:
        values, run_dir, resume_context = _resume_context(args, values, raw_argv)
        inherited_parent = resume_context.get("fine_tune_parent")
        fine_tune_parent = (
            dict(inherited_parent)
            if isinstance(inherited_parent, Mapping)
            else None
        )
        args.resume_exact = str(Path(args.resume_exact).expanduser().resolve())
    else:
        config_bytes = json.dumps(values, sort_keys=True, separators=(",", ":")).encode("utf-8")
        config_digest = hashlib.sha256(config_bytes).hexdigest()
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        default_output = ROOT / "artifacts" / "training" / "runs" / f"{stamp}-{config_digest[:10]}"
        run_dir = _root_path(args.output_dir) if args.output_dir else default_output
    _validate_resolved(values, args.mode)

    free_bytes = shutil.disk_usage(run_dir.parent if run_dir.parent.exists() else ROOT).free
    if args.mode == "train" and free_bytes < 5 * 1024**3:
        raise StudentTrainingError(
            f"Spazio libero insufficiente per checkpoint e cache: {free_bytes / 1024**3:.1f} GiB"
        )

    if resume_context is None:
        manifest = build_run_manifest(
            model_source=values["model_source"],
            model_repo=values["model_repo"],
            model_revision=values["model_revision"],
            teacher_config=values["teacher_config"],
            train_path=values["train_file"],
            validation_path=values["validation_file"],
            training_configuration={**values, "config_sha256": config_digest},
            software=_software_versions(),
        )
        manifest.update(
            kind="pii-student-supervised-finetune",
            status="prepared",
            release_eligible=False,
        )
        manifest["preflight"] = {
            "free_disk_bytes": free_bytes,
            "model_source_sha256": {
                item["path"]: item["sha256"] for item in manifest["backbone"]["files"]
            },
        }
        manifest["fine_tune_parent"] = fine_tune_parent
        run_dir.mkdir(parents=True, exist_ok=False)
        _write_json(run_dir / "run-manifest.json", manifest)
    else:
        manifest = resume_context["parent_manifest"]
        _begin_continuation(run_dir, resume_context)
        _remove_redundant_final(run_dir, resume_context)

    if args.mode == "inspect":
        print(json.dumps({"status": "ok", "run_dir": str(run_dir), "manifest": manifest}, indent=2))
        return 0

    try:
        summary = _run_training(
            args,
            values,
            run_dir,
            resume_context,
            fine_tune_parent,
        )
        if resume_context is not None:
            _finish_continuation(run_dir, resume_context, summary)
    except BaseException as exc:
        if resume_context is not None:
            _fail_continuation(resume_context, exc)
        raise
    manifest_after = json.loads((run_dir / "run-manifest.json").read_text(encoding="utf-8"))
    for section, path_key in (("train", "train_file"), ("validation", "validation_file")):
        if sha256_file(values[path_key]) != manifest_after["data"][section]["sha256"]:
            raise StudentTrainingError(f"Dataset {section} modificato durante il run")
    if fine_tune_parent is not None:
        parent_model = fine_tune_parent.get("model")
        if not isinstance(parent_model, Mapping):
            raise StudentTrainingError("Provenienza parent priva dell'artifact modello")
        parent_after = _weights_only_parent_context(
            str(fine_tune_parent["run_dir"]),
            model_source=str(parent_model["path"]),
        )
        if parent_after != fine_tune_parent:
            raise StudentTrainingError("Artifact parent modificati durante il run")
    if resume_context is None:
        manifest_after.update(
            status="complete",
            result_status=str(summary["status"]),
            completed_at_utc=datetime.now(timezone.utc).isoformat(),
            training_summary=_file_integrity(run_dir / "training-summary.json"),
        )
        _write_json(run_dir / "run-manifest.json", manifest_after)
    else:
        _finalize_resumed_manifest(run_dir, resume_context, summary)
    print(json.dumps({"run_dir": str(run_dir), **summary}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except StudentTrainingError as exc:
        print(f"ERRORE: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
