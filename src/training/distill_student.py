"""Fine-tuning distillato, riproducibile e fail-closed, dello student PII.

Questo stadio parte esclusivamente dai pesi finali di un run student completo:
optimizer e scheduler vengono sempre ricreati.  Per ``alpha > 0`` usa una cache
offline di logits del teacher, gia' proiettati sui token dello student; il
teacher non viene mai caricato in questo processo.  ``alpha = 0`` e' il
controllo gold-only e non richiede ne' apre la cache teacher.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.quantization.metrics import compute_metrics
from src.training.student_utils import (
    IGNORE_INDEX,
    StudentTrainingError,
    TOKENIZER_FIX_MISTRAL_REGEX,
    load_label_contract,
    prediction_documents,
    sha256_file,
    tokenizer_load_policy,
)
from src.training.train_student import (
    LR_SCHEDULER_TYPE,
    _detect_input_format,
    _memory_snapshot,
    _progress_status,
    _select_limit,
    _tokenize_dataset,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SEED = 20260822
DEFAULT_ALPHA = 0.25
DEFAULT_TEMPERATURE = 2.0
DEFAULT_LEARNING_RATE = 1e-5
DEFAULT_EPOCHS = 0.25
DEFAULT_KD_MODE = "raw"
DEFAULT_KD_MIN_COVERAGE = 0.96
DEFAULT_KD_MAX_COVERAGE = 0.98
KD_MODES = ("raw", "type-boundary-normalized")
DISTILL_INDEX = "__distill_index"
RUN_KIND = "pii-student-distillation-finetune"


def _root_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


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


def _file_integrity(path: str | Path) -> dict[str, Any]:
    source = Path(path).resolve()
    if not source.is_file():
        raise StudentTrainingError(f"File richiesto non trovato: {source}")
    return {
        "path": str(source),
        "bytes": int(source.stat().st_size),
        "sha256": sha256_file(source),
    }


def _memory_guard_policy(
    *,
    max_swap_growth_bytes: int,
    min_available_memory_bytes: int,
) -> dict[str, Any]:
    """Contratto immutabile del guard, registrato nel run manifest."""

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
    """Aggiorna counter/campioni e applica la policy stateful del guard."""

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


def _enforce_pretraining_memory_guard(
    snapshot: Mapping[str, Any],
    *,
    phase: str,
    memory_guard: dict[str, Any],
) -> None:
    """Blocca prima di ``Trainer.train`` e conserva causa/fase nel record run."""

    reason = _observe_memory_guard(
        snapshot,
        phase=phase,
        memory_guard=memory_guard,
    )
    if reason is None:
        return
    error = StudentTrainingError(f"Memory guard pre-training ({phase}): {reason}")
    error.memory_guard = memory_guard  # type: ignore[attr-defined]
    raise error


def _finalize_run_lifecycle(
    run_dir: str | Path,
    *,
    lifecycle_status: str,
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    """Transizione atomica prepared -> complete/failed legata all'hash summary."""

    if lifecycle_status not in {"complete", "failed"}:
        raise StudentTrainingError(f"Lifecycle status non valido: {lifecycle_status}")
    result_status = str(summary.get("status", ""))
    if lifecycle_status == "failed":
        if result_status != "failed":
            raise StudentTrainingError("Lifecycle failed richiede summary failed")
    elif result_status not in {"complete", "partial_guard_stop"}:
        raise StudentTrainingError(
            "Lifecycle complete richiede un risultato complete/partial_guard_stop"
        )

    root = Path(run_dir).resolve()
    manifest_path = root / "run-manifest.json"
    summary_path = root / "training-summary.json"
    manifest = _read_json_object(manifest_path, "Run manifest distillation")
    if manifest.get("kind") != RUN_KIND or manifest.get("status") != "prepared":
        raise StudentTrainingError("Transizione lifecycle ammessa solo da prepared")
    on_disk_summary = _read_json_object(summary_path, "Summary distillation")
    if dict(on_disk_summary) != dict(summary):
        raise StudentTrainingError("Summary in memoria e su disco divergenti")

    finalized = dict(manifest)
    finalized.update(
        status=lifecycle_status,
        result_status=result_status,
        completed_at_utc=datetime.now(timezone.utc).isoformat(),
        training_summary=_file_integrity(summary_path),
    )
    _write_json(manifest_path, finalized)
    return finalized


def _software_versions() -> dict[str, str | None]:
    result: dict[str, str | None] = {"python": sys.version.split()[0]}
    for package in (
        "torch",
        "transformers",
        "accelerate",
        "datasets",
        "safetensors",
        "numpy",
        "psutil",
    ):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    return result


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("inspect", "train"))
    parser.add_argument(
        "--parent-run",
        required=True,
        help="run student completo; i soli pesi in <run>/final inizializzano lo stadio",
    )
    parser.add_argument(
        "--teacher-cache",
        help="cache offline; obbligatoria se alpha > 0, non letta se alpha = 0",
    )
    parser.add_argument("--output-dir")
    parser.add_argument("--dataset-cache-dir", default="artifacts/training/cache")
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--kd-mode", choices=KD_MODES, default=DEFAULT_KD_MODE)
    parser.add_argument(
        "--kd-min-coverage", type=float, default=DEFAULT_KD_MIN_COVERAGE
    )
    parser.add_argument(
        "--kd-max-coverage", type=float, default=DEFAULT_KD_MAX_COVERAGE
    )
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--epochs", type=float, default=DEFAULT_EPOCHS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-steps", type=int, default=-1)
    return parser.parse_args(list(argv) if argv is not None else None)


def _validate_recipe(args: argparse.Namespace) -> None:
    alpha = float(args.alpha)
    if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise StudentTrainingError("alpha deve essere finito e compreso fra 0 e 1")
    temperature = float(args.temperature)
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise StudentTrainingError("temperature deve essere finita e positiva")
    learning_rate = float(args.learning_rate)
    if not math.isfinite(learning_rate) or learning_rate <= 0.0:
        raise StudentTrainingError("learning_rate deve essere finito e positivo")
    epochs = float(args.epochs)
    if not math.isfinite(epochs) or epochs <= 0.0:
        raise StudentTrainingError("epochs deve essere finito e positivo")
    if int(args.seed) < 0:
        raise StudentTrainingError("seed deve essere non negativo")
    if int(args.max_steps) == 0 or int(args.max_steps) < -1:
        raise StudentTrainingError("max_steps deve essere -1 oppure positivo")
    min_coverage = float(args.kd_min_coverage)
    max_coverage = float(args.kd_max_coverage)
    if (
        not math.isfinite(min_coverage)
        or not math.isfinite(max_coverage)
        or not 0.0 <= min_coverage <= max_coverage <= 1.0
    ):
        raise StudentTrainingError(
            "La coverage KD deve rispettare 0 <= min <= max <= 1"
        )
    if alpha > 0.0 and not args.teacher_cache:
        raise StudentTrainingError("--teacher-cache e' obbligatoria quando alpha > 0")


def _labels_in_id_order(label2id: Mapping[str, int]) -> list[str]:
    labels = [""] * len(label2id)
    for label, index in label2id.items():
        labels[int(index)] = str(label)
    if any(not label for label in labels):
        raise StudentTrainingError("Contratto label non contiguo")
    return labels


def _load_parent_context(parent_run: str | Path) -> dict[str, Any]:
    """Valida il parent completo e descrive l'inizializzazione weights-only."""

    from src.training.cache_teacher_logits import artifact_group_integrity

    run_dir = _root_path(parent_run)
    if not run_dir.is_dir():
        raise StudentTrainingError(f"Run parent non trovato: {run_dir}")
    manifest_path = run_dir / "run-manifest.json"
    summary_path = run_dir / "training-summary.json"
    manifest = _read_json_object(manifest_path, "Manifest parent")
    summary = _read_json_object(summary_path, "Summary parent")
    if manifest.get("schema_version") != 1:
        raise StudentTrainingError("schema_version del parent non supportata")
    if summary.get("status") != "complete":
        raise StudentTrainingError("Il parent deve essere un run completo")
    if int(summary.get("global_step", -1)) != int(summary.get("planned_steps", -2)):
        raise StudentTrainingError("Il parent non ha completato tutti gli update pianificati")

    training = manifest.get("training")
    data = manifest.get("data")
    teacher_contract = manifest.get("teacher_label_contract")
    if not isinstance(training, Mapping) or not isinstance(data, Mapping):
        raise StudentTrainingError("Manifest parent privo di training/data")
    if not isinstance(teacher_contract, Mapping):
        raise StudentTrainingError("Manifest parent privo del contratto label")
    if summary.get("input_format") != "raw-spans":
        raise StudentTrainingError("La distillation richiede un parent raw-spans")

    required_training = (
        "train_file",
        "validation_file",
        "max_length",
        "stride",
        "train_batch_size",
        "eval_batch_size",
        "gradient_accumulation_steps",
        "weight_decay",
        "warmup_ratio",
        "attention_implementation",
        "gradient_checkpointing",
        "mps_memory_fraction",
        "torch_empty_cache_steps",
        "max_swap_growth_bytes",
        "min_available_memory_bytes",
        "seed",
    )
    missing = [key for key in required_training if key not in training]
    if missing:
        raise StudentTrainingError(f"Manifest parent incompleto: {missing}")

    data_records: dict[str, dict[str, Any]] = {}
    for split, training_key in (("train", "train_file"), ("validation", "validation_file")):
        record = data.get(split)
        if not isinstance(record, Mapping):
            raise StudentTrainingError(f"Manifest parent privo del dataset {split}")
        path = Path(str(training[training_key])).expanduser().resolve()
        if Path(str(record.get("path", ""))).expanduser().resolve() != path:
            raise StudentTrainingError(f"Path {split} divergente nel manifest parent")
        current = _file_integrity(path)
        if current["sha256"] != record.get("sha256"):
            raise StudentTrainingError(f"Dataset {split} modificato dopo il run parent")
        data_records[split] = {**dict(record), **current}

    contract_path = Path(str(teacher_contract.get("path", ""))).expanduser().resolve()
    contract_integrity = _file_integrity(contract_path)
    if contract_integrity["sha256"] != teacher_contract.get("sha256"):
        raise StudentTrainingError("Contratto label modificato dopo il run parent")
    label2id, id2label = load_label_contract(contract_path)

    final_dir = run_dir / "final"
    model_integrity = artifact_group_integrity(final_dir, "model")
    tokenizer_integrity = artifact_group_integrity(final_dir, "tokenizer")
    final_label2id, final_id2label = load_label_contract(final_dir / "config.json")
    if final_label2id != label2id or final_id2label != id2label:
        raise StudentTrainingError("Le label del modello final divergono dal teacher contract")

    return {
        "run_dir": str(run_dir),
        "manifest": manifest,
        "summary": summary,
        "manifest_integrity": _file_integrity(manifest_path),
        "summary_integrity": _file_integrity(summary_path),
        "final_dir": str(final_dir.resolve()),
        "model_integrity": model_integrity,
        "tokenizer_integrity": tokenizer_integrity,
        "teacher_label_contract": contract_integrity,
        "label2id": label2id,
        "id2label": id2label,
        "labels": _labels_in_id_order(label2id),
        "data": data_records,
        "training": dict(training),
    }


def _resolved_configuration(
    args: argparse.Namespace, parent: Mapping[str, Any]
) -> dict[str, Any]:
    inherited = parent["training"]
    parent_seed = int(inherited["seed"])
    if int(args.seed) != parent_seed:
        raise StudentTrainingError(
            f"Il seed deve restare quello del parent: {args.seed} != {parent_seed}"
        )
    precision = str(args.precision)
    if args.device == "cpu" and precision != "fp32":
        raise StudentTrainingError("CPU richiede --precision fp32")
    if int(inherited["max_length"]) < 16:
        raise StudentTrainingError("max_length parent non valido")
    if not 0 <= int(inherited["stride"]) < int(inherited["max_length"]) - 2:
        raise StudentTrainingError("stride parent non valido")

    return {
        "parent_model": str(parent["final_dir"]),
        "train_file": str(Path(str(inherited["train_file"])).resolve()),
        "validation_file": str(Path(str(inherited["validation_file"])).resolve()),
        "input_format": "raw-spans",
        "dataset_cache_dir": str(_root_path(args.dataset_cache_dir)),
        "teacher_cache": (
            str(_root_path(args.teacher_cache)) if float(args.alpha) > 0.0 else None
        ),
        "cache_loaded": False,
        "device": str(args.device),
        "precision": precision,
        "max_length": int(inherited["max_length"]),
        "stride": int(inherited["stride"]),
        "train_limit": inherited.get("train_limit"),
        "validation_limit": inherited.get("validation_limit"),
        "train_batch_size": int(inherited["train_batch_size"]),
        "eval_batch_size": int(inherited["eval_batch_size"]),
        "gradient_accumulation_steps": int(inherited["gradient_accumulation_steps"]),
        "gradient_checkpointing": bool(inherited["gradient_checkpointing"]),
        "attention_implementation": str(inherited["attention_implementation"]),
        "weight_decay": float(inherited["weight_decay"]),
        "warmup_ratio": float(inherited["warmup_ratio"]),
        "mps_memory_fraction": float(inherited["mps_memory_fraction"]),
        "torch_empty_cache_steps": int(inherited["torch_empty_cache_steps"]),
        "max_swap_growth_bytes": int(inherited["max_swap_growth_bytes"]),
        "min_available_memory_bytes": int(inherited["min_available_memory_bytes"]),
        "optimizer": "adafactor",
        "lr_scheduler_type": LR_SCHEDULER_TYPE,
        "learning_rate": float(args.learning_rate),
        "epochs": float(args.epochs),
        "max_steps": int(args.max_steps),
        "alpha": float(args.alpha),
        "temperature": float(args.temperature),
        "kd_mode": str(args.kd_mode),
        "kd_min_coverage": float(args.kd_min_coverage),
        "kd_max_coverage": float(args.kd_max_coverage),
        "tokenizer_load_policy": tokenizer_load_policy(),
        "seed": int(args.seed),
    }


def _bio_label_contract(label_names: Sequence[str]) -> dict[str, Any]:
    """Valida BIO e costruisce lookup stabili per tipo e prefisso."""

    names = [str(label) for label in label_names]
    if not names or len(set(names)) != len(names):
        raise StudentTrainingError("Contratto label KD vuoto o con duplicati")

    o_ids = [index for index, label in enumerate(names) if label == "O"]
    if len(o_ids) != 1:
        raise StudentTrainingError("Contratto label KD deve contenere un solo O")

    pairs: dict[str, dict[str, int]] = {}
    ordered_types: list[str] = []
    for index, label in enumerate(names):
        if label == "O":
            continue
        prefix, separator, entity_type = label.partition("-")
        if separator != "-" or prefix not in {"B", "I"} or not entity_type:
            raise StudentTrainingError(f"Label BIO non valida: {label!r}")
        if entity_type not in pairs:
            pairs[entity_type] = {}
            ordered_types.append(entity_type)
        if prefix in pairs[entity_type]:
            raise StudentTrainingError(
                f"Contratto label KD duplica {prefix}-{entity_type}"
            )
        pairs[entity_type][prefix] = index

    incomplete = [
        entity_type
        for entity_type in ordered_types
        if set(pairs[entity_type]) != {"B", "I"}
    ]
    if incomplete:
        raise StudentTrainingError(
            "Contratto label KD privo di coppie B/I complete: "
            + ", ".join(incomplete)
        )

    o_type_id = len(ordered_types)
    type_ids = [o_type_id] * len(names)
    prefix_ids = [0] * len(names)  # O=0, B=1, I=2
    pair_ids: list[tuple[int, int]] = []
    opposite_ids = list(range(len(names)))
    for type_id, entity_type in enumerate(ordered_types):
        b_id = pairs[entity_type]["B"]
        i_id = pairs[entity_type]["I"]
        pair_ids.append((b_id, i_id))
        opposite_ids[b_id] = i_id
        opposite_ids[i_id] = b_id
        type_ids[b_id] = type_id
        type_ids[i_id] = type_id
        prefix_ids[b_id] = 1
        prefix_ids[i_id] = 2
    return {
        "labels": names,
        "o_id": o_ids[0],
        "o_type_id": o_type_id,
        "entity_types": ordered_types,
        "pairs": pair_ids,
        "opposite_ids": opposite_ids,
        "type_ids": type_ids,
        "prefix_ids": prefix_ids,
    }


def _type_boundary_normalized_targets(
    teacher_logits: Any,
    labels: Any,
    *,
    label_names: Sequence[str],
    temperature: float,
) -> tuple[Any, Any, Any]:
    """Restituisce target normalizzati, mask KD attiva e hard label teacher.

    La proiezione e' intenzionalmente minima: avviene solo se hard teacher e
    gold differiscono esclusivamente nel prefisso BIO dello stesso tipo X. In
    quel caso somma ``p(B-X)+p(I-X)`` sulla classe gold e azzera l'altro
    prefisso; tutte le altre classi restano identiche. Exact agreement e O/O
    mantengono esattamente le probabilita' teacher originali.
    """

    import torch
    import torch.nn.functional as functional

    if teacher_logits.ndim != 3 or labels.ndim != 2:
        raise StudentTrainingError("Shape teacher logits/labels KD non valida")
    if tuple(teacher_logits.shape[:2]) != tuple(labels.shape):
        raise StudentTrainingError("Teacher logits e labels KD disallineati")
    if not math.isfinite(float(temperature)) or float(temperature) <= 0.0:
        raise StudentTrainingError("temperature non valida")

    contract = _bio_label_contract(label_names)
    num_labels = int(teacher_logits.shape[-1])
    if num_labels != len(contract["labels"]):
        raise StudentTrainingError(
            "Dimensione logits divergente dal contratto label KD"
        )

    supervised = labels.ne(IGNORE_INDEX)
    safe_labels = torch.where(
        supervised, labels, torch.full_like(labels, int(contract["o_id"]))
    ).to(dtype=torch.long)
    invalid_labels = supervised & ((safe_labels < 0) | (safe_labels >= num_labels))
    if bool(invalid_labels.any().item()):
        raise StudentTrainingError("ID gold fuori dal contratto label")

    teacher_float = teacher_logits.float()
    invalid_teacher = (~torch.isfinite(teacher_float).all(dim=-1)) & supervised
    if bool(invalid_teacher.any().item()):
        raise StudentTrainingError("Teacher logits non finiti sui token supervisionati")
    sanitized_teacher = torch.where(
        supervised.unsqueeze(-1), teacher_float, torch.zeros_like(teacher_float)
    )
    teacher_probs = functional.softmax(
        sanitized_teacher / float(temperature), dim=-1
    )
    hard_labels = sanitized_teacher.argmax(dim=-1)

    type_lookup = torch.tensor(
        contract["type_ids"], dtype=torch.long, device=teacher_logits.device
    )
    gold_types = type_lookup[safe_labels]
    teacher_types = type_lookup[hard_labels]
    active = supervised & gold_types.eq(teacher_types)
    bio_only = active & hard_labels.ne(safe_labels) & gold_types.ne(
        int(contract["o_type_id"])
    )
    opposite_lookup = torch.tensor(
        contract["opposite_ids"], dtype=torch.long, device=teacher_logits.device
    )
    opposite_labels = opposite_lookup[safe_labels]
    opposite_probability = torch.gather(
        teacher_probs, dim=-1, index=opposite_labels.unsqueeze(-1)
    ).squeeze(-1)
    gold_one_hot = functional.one_hot(safe_labels, num_classes=num_labels).to(
        dtype=teacher_probs.dtype
    )
    opposite_one_hot = functional.one_hot(
        opposite_labels, num_classes=num_labels
    ).to(dtype=teacher_probs.dtype)
    projected = teacher_probs + opposite_probability.unsqueeze(-1) * (
        gold_one_hot - opposite_one_hot
    )
    normalized = torch.where(bio_only.unsqueeze(-1), projected, teacher_probs)
    return normalized, active, hard_labels


def _audit_kd_cache_targets(
    *,
    teacher_logits: Any,
    labels: Any,
    input_ids: Any,
    tokenizer: Any,
    label_names: Sequence[str],
    temperature: float,
    kd_mode: str,
    min_coverage: float,
    max_coverage: float,
) -> dict[str, Any]:
    """Audit completo e fail-closed dei target KD prima del caricamento modello."""

    import torch
    import torch.nn.functional as functional

    if kd_mode not in KD_MODES:
        raise StudentTrainingError(f"Modalita' KD non valida: {kd_mode}")
    if tuple(teacher_logits.shape[:2]) != tuple(labels.shape) or tuple(
        input_ids.shape
    ) != tuple(labels.shape):
        raise StudentTrainingError("Tensor cache disallineati nell'audit KD")
    contract = _bio_label_contract(label_names)
    if int(teacher_logits.shape[-1]) != len(contract["labels"]):
        raise StudentTrainingError(
            "Dimensione logits divergente dal contratto label KD"
        )

    supervised = labels.ne(IGNORE_INDEX)
    supervised_count = int(supervised.sum().item())
    if supervised_count == 0:
        raise StudentTrainingError("Cache KD priva di token supervisionati")
    safe_labels = torch.where(
        supervised,
        labels.to(dtype=torch.long),
        torch.full_like(labels, int(contract["o_id"]), dtype=torch.long),
    )
    invalid_labels = supervised & (
        (safe_labels < 0) | (safe_labels >= len(contract["labels"]))
    )
    if bool(invalid_labels.any().item()):
        raise StudentTrainingError("ID gold fuori dal contratto label")

    type_lookup = torch.tensor(contract["type_ids"], dtype=torch.long)
    hard_labels = teacher_logits.float().argmax(dim=-1).to(dtype=torch.long)
    gold_types = type_lookup[safe_labels]
    teacher_types = type_lookup[hard_labels]
    same_type = supervised & gold_types.eq(teacher_types)
    type_mismatch = supervised & ~same_type
    exact_agreement = supervised & hard_labels.eq(safe_labels)
    bio_only_same_type = same_type & ~exact_agreement
    active = supervised if kd_mode == "raw" else same_type
    excluded_type_mismatch = (
        torch.zeros_like(type_mismatch) if kd_mode == "raw" else type_mismatch
    )
    active_per_window = active.sum(dim=1)
    zero_active_windows = int(active_per_window.eq(0).sum().item())
    min_active_tokens_per_window = int(active_per_window.min().item())
    max_active_tokens_per_window = int(active_per_window.max().item())

    # Il controllo usa gli ID realmente presenti, senza assumere che il marker
    # SentencePiece abbia un ID fisso nel vocabolario.
    marker_ids: list[int] = []
    for token_id in torch.unique(input_ids[supervised].to(dtype=torch.long)).tolist():
        try:
            token = tokenizer.convert_ids_to_tokens(int(token_id))
        except Exception as exc:
            raise StudentTrainingError(
                f"Tokenizer non converte l'ID {token_id} durante l'audit KD"
            ) from exc
        if token == "▁":
            marker_ids.append(int(token_id))
    boundary_marker = torch.zeros_like(supervised)
    for marker_id in marker_ids:
        boundary_marker = boundary_marker | input_ids.eq(marker_id)
    gold_o = safe_labels.eq(int(contract["o_id"]))
    teacher_entity = teacher_types.ne(int(contract["o_type_id"]))
    gold_o_boundary_teacher_entity = (
        supervised & boundary_marker & gold_o & teacher_entity
    )
    active_type_mismatch = active & type_mismatch
    active_boundary_conflict = active & gold_o_boundary_teacher_entity

    tolerance = 1e-5
    max_probability_sum_error = 0.0
    max_target_sum_error = 0.0
    chunk_size = 64
    for start in range(0, int(labels.shape[0]), chunk_size):
        stop = min(start + chunk_size, int(labels.shape[0]))
        chunk_mask = supervised[start:stop]
        chunk_logits = teacher_logits[start:stop].float()
        invalid_logits = (~torch.isfinite(chunk_logits).all(dim=-1)) & chunk_mask
        if bool(invalid_logits.any().item()):
            raise StudentTrainingError(
                "Teacher logits non finiti sui token supervisionati nell'audit KD"
            )
        sanitized = torch.where(
            chunk_mask.unsqueeze(-1), chunk_logits, torch.zeros_like(chunk_logits)
        )
        probabilities = functional.softmax(
            sanitized / float(temperature), dim=-1
        )
        if kd_mode == "type-boundary-normalized":
            targets, chunk_active, _ = _type_boundary_normalized_targets(
                chunk_logits,
                labels[start:stop],
                label_names=label_names,
                temperature=temperature,
            )
            if not torch.equal(chunk_active.cpu(), active[start:stop].cpu()):
                raise StudentTrainingError("Mask KD divergente durante l'audit")
            chunk_exact_active = exact_agreement[start:stop] & chunk_active
            if bool(chunk_exact_active.any().item()) and not torch.equal(
                targets[chunk_exact_active], probabilities[chunk_exact_active]
            ):
                raise StudentTrainingError(
                    "La normalizzazione ha modificato target KD exact-agreement"
                )
            chunk_projected = bio_only_same_type[start:stop]
            if bool(chunk_projected.any().item()):
                opposite_lookup = torch.tensor(
                    contract["opposite_ids"], dtype=torch.long
                )
                chunk_gold = safe_labels[start:stop]
                chunk_opposite = opposite_lookup[chunk_gold]
                class_ids = torch.arange(len(contract["labels"])).view(1, 1, -1)
                allowed_change = class_ids.eq(chunk_gold.unsqueeze(-1)) | class_ids.eq(
                    chunk_opposite.unsqueeze(-1)
                )
                forbidden_change = (
                    chunk_projected.unsqueeze(-1)
                    & ~allowed_change
                    & targets.ne(probabilities)
                )
                if bool(forbidden_change.any().item()):
                    raise StudentTrainingError(
                        "La normalizzazione ha modificato classi estranee alla coppia BIO"
                    )
        else:
            targets = probabilities

        selected_probabilities = torch.where(
            chunk_mask.unsqueeze(-1),
            probabilities,
            torch.zeros_like(probabilities),
        )
        selected_targets = torch.where(
            chunk_mask.unsqueeze(-1), targets, torch.zeros_like(targets)
        )
        if not bool(torch.isfinite(selected_probabilities).all().item()) or not bool(
            torch.isfinite(selected_targets).all().item()
        ):
            raise StudentTrainingError("Probabilita' KD non finite nel preflight")
        probability_errors = torch.where(
            chunk_mask,
            (probabilities.sum(dim=-1) - 1.0).abs(),
            torch.zeros_like(chunk_mask, dtype=probabilities.dtype),
        )
        target_errors = torch.where(
            chunk_mask,
            (targets.sum(dim=-1) - 1.0).abs(),
            torch.zeros_like(chunk_mask, dtype=targets.dtype),
        )
        max_probability_sum_error = max(
            max_probability_sum_error, float(probability_errors.max().item())
        )
        max_target_sum_error = max(
            max_target_sum_error, float(target_errors.max().item())
        )

    if max_probability_sum_error > tolerance or max_target_sum_error > tolerance:
        raise StudentTrainingError(
            "Le probabilita' KD non conservano massa unitaria: "
            f"teacher={max_probability_sum_error}, target={max_target_sum_error}"
        )
    if kd_mode == "type-boundary-normalized":
        if bool(active_type_mismatch.any().item()):
            raise StudentTrainingError("Il gate KD ha attivato un mismatch di tipo")
        if bool(active_boundary_conflict.any().item()):
            raise StudentTrainingError(
                "Il gate KD ha attivato un boundary ▁ gold-O/teacher-entita'"
            )
        if zero_active_windows:
            raise StudentTrainingError(
                "Cache KD con finestre prive di token attivi: "
                f"{zero_active_windows}"
            )

    active_count = int(active.sum().item())
    coverage = active_count / supervised_count
    if kd_mode == "type-boundary-normalized" and not (
        float(min_coverage) <= coverage <= float(max_coverage)
    ):
        raise StudentTrainingError(
            "Coverage KD normalizzata fuori intervallo: "
            f"{coverage:.8f} non in [{min_coverage:.8f}, {max_coverage:.8f}]"
        )
    return {
        "schema_version": 1,
        "mode": kd_mode,
        "temperature": float(temperature),
        "denominator": "cache_window_token_observations_with_label_ne_ignore_index",
        "supervised_tokens": supervised_count,
        "active_kd_tokens": active_count,
        "coverage": coverage,
        "zero_active_windows": zero_active_windows,
        "min_active_tokens_per_window": min_active_tokens_per_window,
        "max_active_tokens_per_window": max_active_tokens_per_window,
        "coverage_bounds_enforced": kd_mode == "type-boundary-normalized",
        "min_coverage": float(min_coverage),
        "max_coverage": float(max_coverage),
        "exact_agreement_tokens": int(exact_agreement.sum().item()),
        "exact_active_unchanged_tokens": int((exact_agreement & active).sum().item()),
        "bio_only_same_type_tokens": int(bio_only_same_type.sum().item()),
        "projected_bio_only_tokens": (
            int(bio_only_same_type.sum().item())
            if kd_mode == "type-boundary-normalized"
            else 0
        ),
        "type_mismatch_tokens": int(type_mismatch.sum().item()),
        "type_mismatch_excluded_tokens": int(excluded_type_mismatch.sum().item()),
        "active_type_mismatch_tokens": int(active_type_mismatch.sum().item()),
        "gold_o_boundary_teacher_entity_tokens": int(
            gold_o_boundary_teacher_entity.sum().item()
        ),
        "active_gold_o_boundary_teacher_entity_tokens": int(
            active_boundary_conflict.sum().item()
        ),
        "boundary_marker": "▁",
        "boundary_marker_token_ids": marker_ids,
        "probabilities_finite": True,
        "probability_sum_tolerance": tolerance,
        "max_teacher_probability_sum_error": max_probability_sum_error,
        "max_target_probability_sum_error": max_target_sum_error,
    }


def compute_distillation_loss(
    student_logits: Any,
    labels: Any,
    *,
    teacher_logits: Any | None,
    alpha: float,
    temperature: float,
    kd_mode: str = DEFAULT_KD_MODE,
    label_names: Sequence[str] | None = None,
) -> Any:
    """Calcola CE gold-dominante + KL, esclusivamente sui token supervisionati."""

    import torch
    import torch.nn.functional as functional

    if not 0.0 <= float(alpha) <= 1.0:
        raise StudentTrainingError("alpha fuori intervallo")
    if not math.isfinite(float(temperature)) or float(temperature) <= 0.0:
        raise StudentTrainingError("temperature non valida")
    if kd_mode not in KD_MODES:
        raise StudentTrainingError(f"Modalita' KD non valida: {kd_mode}")
    if student_logits.ndim != 3 or labels.ndim != 2:
        raise StudentTrainingError("Shape student logits/labels non valida")
    if tuple(student_logits.shape[:2]) != tuple(labels.shape):
        raise StudentTrainingError("Student logits e labels disallineati")

    mask = labels.ne(IGNORE_INDEX)
    if not bool(mask.any().item()):
        raise StudentTrainingError("Batch privo di token supervisionati")
    num_labels = int(student_logits.shape[-1])
    safe_labels = torch.where(mask, labels, torch.zeros_like(labels)).to(dtype=torch.long)
    invalid_labels = mask & ((safe_labels < 0) | (safe_labels >= num_labels))
    if bool(invalid_labels.any().item()):
        raise StudentTrainingError("ID gold fuori dal contratto label")

    # Niente boolean/advanced indexing: il backward MPS di index_put non offre
    # un kernel deterministic. CE ignora direttamente -100 e resta equivalente
    # alla media sui soli token supervisionati.
    student_float = student_logits.float()
    gold_loss = functional.cross_entropy(
        student_float.reshape(-1, num_labels),
        labels.to(dtype=torch.long).reshape(-1),
        ignore_index=IGNORE_INDEX,
    )
    if float(alpha) == 0.0:
        return gold_loss
    if teacher_logits is None:
        raise StudentTrainingError("Teacher logits mancanti con alpha > 0")
    if tuple(teacher_logits.shape) != tuple(student_logits.shape):
        raise StudentTrainingError("Teacher e student logits hanno shape diversa")
    teacher_float = teacher_logits.float()
    invalid_teacher = (~torch.isfinite(teacher_float).all(dim=-1)) & mask
    if bool(invalid_teacher.any().item()):
        raise StudentTrainingError("Teacher logits non finiti sui token supervisionati")

    if kd_mode == "type-boundary-normalized":
        if label_names is None:
            raise StudentTrainingError(
                "Contratto label richiesto per KD type-boundary-normalized"
            )
        teacher_targets, active_mask, _ = _type_boundary_normalized_targets(
            teacher_float,
            labels,
            label_names=label_names,
            temperature=temperature,
        )
        # Il preflight verifica la stessa condizione per ogni finestra della
        # cache. Manteniamo anche la loss fail-closed per evitare che un uso
        # futuro cambi silenziosamente la formula o il peso della CE.
        if not bool(active_mask.any().item()):
            raise StudentTrainingError("Batch KD privo di token attivi")
        student_for_kl = torch.where(
            active_mask.unsqueeze(-1), student_float, torch.zeros_like(student_float)
        )
        targets_for_kl = torch.where(
            active_mask.unsqueeze(-1),
            teacher_targets,
            torch.zeros_like(teacher_targets),
        )
        scale = float(temperature)
        per_class_kl = functional.kl_div(
            functional.log_softmax(student_for_kl / scale, dim=-1),
            targets_for_kl,
            reduction="none",
        )
        per_token_kl = per_class_kl.sum(dim=-1)
        float_mask = active_mask.to(dtype=per_token_kl.dtype)
        kl_loss = (per_token_kl * float_mask).sum() / float_mask.sum()
        return (
            (1.0 - float(alpha)) * gold_loss
            + float(alpha) * scale**2 * kl_loss
        )

    # Compatibilita' metodologica: il ramo raw conserva esattamente la formula
    # e l'ordine delle operazioni del primo esperimento KD.
    teacher_for_kl = torch.where(
        mask.unsqueeze(-1), teacher_float, torch.zeros_like(teacher_float)
    )
    student_for_kl = torch.where(
        mask.unsqueeze(-1), student_float, torch.zeros_like(student_float)
    )

    scale = float(temperature)
    per_class_kl = functional.kl_div(
        functional.log_softmax(student_for_kl / scale, dim=-1),
        functional.softmax(teacher_for_kl / scale, dim=-1),
        reduction="none",
    )
    per_token_kl = per_class_kl.sum(dim=-1)
    float_mask = mask.to(dtype=per_token_kl.dtype)
    supervised_count = float_mask.sum()
    kl_loss = (per_token_kl * float_mask).sum() / supervised_count
    return (1.0 - float(alpha)) * gold_loss + float(alpha) * scale**2 * kl_loss


def _pad_window(
    feature: Mapping[str, Sequence[int]],
    *,
    max_length: int,
    pad_token_id: int,
    padding_side: str,
    pad_token_type_id: int = 0,
) -> dict[str, list[int]]:
    length = len(feature["input_ids"])
    if length < 1 or length > max_length:
        raise StudentTrainingError(f"Lunghezza finestra non valida: {length}")
    if len(feature.get("attention_mask", [])) != length or len(feature["labels"]) != length:
        raise StudentTrainingError("Finestra tokenizzata internamente disallineata")
    padding = max_length - length
    left = padding_side == "left"
    if padding_side not in ("left", "right"):
        raise StudentTrainingError(f"padding_side non supportato: {padding_side}")

    def padded(values: Sequence[int], fill: int) -> list[int]:
        pad = [fill] * padding
        materialized = [int(value) for value in values]
        return pad + materialized if left else materialized + pad

    result = {
        "input_ids": padded(feature["input_ids"], pad_token_id),
        "attention_mask": padded(feature["attention_mask"], 0),
        "labels": padded(feature["labels"], IGNORE_INDEX),
    }
    if "token_type_ids" in feature:
        result["token_type_ids"] = padded(feature["token_type_ids"], pad_token_type_id)
    return result


def _validate_teacher_cache(
    cache_dir: str | Path,
    *,
    tokenized_train: Any,
    tokenizer: Any,
    parent: Mapping[str, Any],
    values: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Valida identita' e allineamento completo prima di restituire i logits."""

    import torch

    from src.training.cache_teacher_logits import (
        load_teacher_cache,
        window_alignment_sha256,
    )

    expected = {
        "dataset_sha256": str(parent["data"]["train"]["sha256"]),
        "student_model_sha256": str(parent["model_integrity"]["sha256"]),
        "student_tokenizer_sha256": str(parent["tokenizer_integrity"]["sha256"]),
        "max_length": int(values["max_length"]),
        "stride": int(values["stride"]),
        "seed": int(values["seed"]),
        "num_labels": len(parent["labels"]),
        "num_windows": len(tokenized_train),
    }
    manifest, tensors = load_teacher_cache(cache_dir, expected=expected)
    windows = manifest.get("windows")
    if not isinstance(windows, list) or len(windows) != len(tokenized_train):
        raise StudentTrainingError("Indice finestre della cache incompleto")
    label_contract = manifest.get("label_contract")
    if not isinstance(label_contract, Mapping) or label_contract.get("labels") != list(
        parent["labels"]
    ):
        raise StudentTrainingError("Ordine label della cache divergente dal parent")
    student_record = manifest.get("student")
    cache_run_manifest = (
        student_record.get("run_manifest")
        if isinstance(student_record, Mapping)
        else None
    )
    if not isinstance(cache_run_manifest, Mapping) or cache_run_manifest.get(
        "sha256"
    ) != parent["manifest_integrity"]["sha256"]:
        raise StudentTrainingError("La cache appartiene a un run student diverso")
    preprocessing = manifest.get("preprocessing")
    if not isinstance(preprocessing, Mapping):
        raise StudentTrainingError("Cache priva del contratto preprocessing")
    expected_preprocessing = {
        "input_format": "raw-spans",
        "max_length": int(values["max_length"]),
        "stride": int(values["stride"]),
        "train_limit": values["train_limit"],
        "seed": int(values["seed"]),
        "padding_side": str(getattr(tokenizer, "padding_side", "right")),
        "ignore_index": IGNORE_INDEX,
    }
    mismatched_preprocessing = [
        key
        for key, expected_value in expected_preprocessing.items()
        if preprocessing.get(key) != expected_value
    ]
    if mismatched_preprocessing:
        raise StudentTrainingError(
            "Preprocessing cache divergente: " + ", ".join(mismatched_preprocessing)
        )

    required_tensors = {"teacher_logits", "input_ids", "attention_mask", "labels"}
    if not required_tensors <= set(tensors):
        raise StudentTrainingError(
            f"Tensor cache mancanti: {sorted(required_tensors - set(tensors))}"
        )
    teacher_logits = tensors["teacher_logits"]
    expected_shape = (
        len(tokenized_train),
        int(values["max_length"]),
        len(parent["labels"]),
    )
    if tuple(teacher_logits.shape) != expected_shape:
        raise StudentTrainingError(
            f"Shape teacher_logits inattesa: {tuple(teacher_logits.shape)} != {expected_shape}"
        )

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        raise StudentTrainingError("Tokenizer parent privo di pad_token_id")
    padding_side = str(getattr(tokenizer, "padding_side", "right"))
    pad_token_type_id = int(getattr(tokenizer, "pad_token_type_id", 0) or 0)
    cache_has_types = "token_type_ids" in tensors

    for index in range(len(tokenized_train)):
        feature = tokenized_train[index]
        padded = _pad_window(
            feature,
            max_length=int(values["max_length"]),
            pad_token_id=int(pad_token_id),
            padding_side=padding_side,
            pad_token_type_id=pad_token_type_id,
        )
        has_types = "token_type_ids" in padded
        if has_types != cache_has_types:
            raise StudentTrainingError(
                f"token_type_ids divergente nella cache alla finestra {index}"
            )
        extras = {"token_type_ids": padded["token_type_ids"]} if has_types else None
        alignment = window_alignment_sha256(
            input_ids=padded["input_ids"],
            attention_mask=padded["attention_mask"],
            labels=padded["labels"],
            extra_inputs=extras,
        )
        record = windows[index]
        if not isinstance(record, Mapping):
            raise StudentTrainingError(f"Record finestra cache non valido: {index}")
        if int(record.get("index", -1)) != index:
            raise StudentTrainingError(f"Ordine finestre cache divergente: {index}")
        if int(record.get("length", -1)) != len(feature["input_ids"]):
            raise StudentTrainingError(f"Lunghezza cache divergente: {index}")
        if record.get("alignment_sha256") != alignment:
            raise StudentTrainingError(f"Fingerprint cache divergente: finestra {index}")

        for name in ("input_ids", "attention_mask", "labels"):
            expected_tensor = torch.tensor(padded[name], dtype=torch.int64)
            if not torch.equal(tensors[name][index].to(torch.int64), expected_tensor):
                raise StudentTrainingError(
                    f"Tensor {name} cache divergente: finestra {index}"
                )
        if has_types:
            expected_types = torch.tensor(padded["token_type_ids"], dtype=torch.int64)
            if not torch.equal(tensors["token_type_ids"][index].to(torch.int64), expected_types):
                raise StudentTrainingError(
                    f"Tensor token_type_ids cache divergente: finestra {index}"
                )
    return manifest, tensors


class _IndexedDataset:
    def __init__(self, dataset: Any) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        feature = dict(self.dataset[index])
        feature[DISTILL_INDEX] = int(index)
        return feature


class DistillationCollator:
    """Collator dinamico che aggancia i logits cached mediante indice stabile."""

    def __init__(
        self,
        base_collator: Any,
        *,
        teacher_logits: Any | None,
        padding_side: str,
    ) -> None:
        self.base_collator = base_collator
        self.teacher_logits = teacher_logits
        self.padding_side = padding_side

    def __call__(self, features: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        copied = [dict(feature) for feature in features]
        indexes = [feature.pop(DISTILL_INDEX, None) for feature in copied]
        has_index = [index is not None for index in indexes]
        if any(has_index) and not all(has_index):
            raise StudentTrainingError("Batch misto con indici cache incompleti")
        batch = self.base_collator(copied)
        if not any(has_index):
            return batch
        if self.teacher_logits is None:
            raise StudentTrainingError("Dataset indicizzato senza teacher logits")

        import torch

        selected = self.teacher_logits[
            torch.tensor([int(index) for index in indexes], dtype=torch.long)
        ]
        batch_length = int(batch["input_ids"].shape[1])
        if batch_length > int(selected.shape[1]):
            raise StudentTrainingError("Batch piu' lungo della cache teacher")
        if self.padding_side == "right":
            selected = selected[:, :batch_length, :]
        elif self.padding_side == "left":
            selected = selected[:, -batch_length:, :]
        else:
            raise StudentTrainingError(f"padding_side non supportato: {self.padding_side}")
        batch["teacher_logits"] = selected
        return batch


def _build_trainer_class(
    base_trainer: type,
    *,
    alpha: float,
    temperature: float,
    kd_mode: str,
    label_names: Sequence[str],
) -> type:
    class DistillationTrainer(base_trainer):
        def compute_loss(
            self,
            model: Any,
            inputs: Mapping[str, Any],
            return_outputs: bool = False,
            num_items_in_batch: Any | None = None,
        ) -> Any:
            del num_items_in_batch
            model_inputs = dict(inputs)
            teacher_logits = model_inputs.pop("teacher_logits", None)
            labels = model_inputs.get("labels")
            if labels is None:
                raise StudentTrainingError("Batch privo delle label gold")
            outputs = model(**model_inputs)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs["logits"]
            effective_teacher = teacher_logits if model.training and alpha > 0.0 else None
            effective_alpha = alpha if model.training else 0.0
            loss = compute_distillation_loss(
                logits,
                labels,
                teacher_logits=effective_teacher,
                alpha=effective_alpha,
                temperature=temperature,
                kd_mode=kd_mode,
                label_names=label_names,
            )
            return (loss, outputs) if return_outputs else loss

    return DistillationTrainer


def _code_integrity() -> list[dict[str, Any]]:
    from src.training import cache_teacher_logits, student_utils, train_student

    return [
        _file_integrity(Path(__file__)),
        _file_integrity(Path(cache_teacher_logits.__file__)),
        _file_integrity(Path(train_student.__file__)),
        _file_integrity(Path(student_utils.__file__)),
    ]


def _run_training(
    args: argparse.Namespace,
    parent: Mapping[str, Any],
    values: dict[str, Any],
    run_dir: Path,
) -> dict[str, Any]:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "0")

    import psutil
    import torch
    from datasets import load_dataset
    from transformers import (
        AutoModelForTokenClassification,
        AutoTokenizer,
        DataCollatorForTokenClassification,
        Trainer,
        TrainerCallback,
        TrainingArguments,
        set_seed,
    )

    requested_device = str(values["device"])
    device = (
        "mps"
        if requested_device == "auto" and torch.backends.mps.is_available()
        else "cpu" if requested_device == "auto" else requested_device
    )
    if device == "mps" and not torch.backends.mps.is_available():
        raise StudentTrainingError("MPS non disponibile in questo processo")
    if device == "cpu" and values["precision"] != "fp32":
        raise StudentTrainingError("CPU richiede precision fp32")
    if device == "mps":
        torch.mps.set_per_process_memory_fraction(float(values["mps_memory_fraction"]))
    set_seed(int(values["seed"]), deterministic=True)
    baseline_snapshot = _memory_snapshot(torch, psutil, device)
    memory_timeline: list[dict[str, Any]] = [
        {"event": "preflight_start", **baseline_snapshot}
    ]
    swap_at_start = int(baseline_snapshot["swap_used_bytes"] or 0)
    memory_guard_policy = _memory_guard_policy(
        max_swap_growth_bytes=values["max_swap_growth_bytes"],
        min_available_memory_bytes=values["min_available_memory_bytes"],
    )
    memory_guard = _new_memory_guard_state(
        memory_guard_policy, swap_at_start=swap_at_start
    )

    tokenizer = AutoTokenizer.from_pretrained(
        values["parent_model"],
        trust_remote_code=False,
        fix_mistral_regex=TOKENIZER_FIX_MISTRAL_REGEX,
    )
    suffix = Path(values["train_file"]).suffix.lower()
    loader_name = "parquet" if suffix in (".parquet", ".pq") else "json"
    raw = load_dataset(
        loader_name,
        data_files={
            "train": values["train_file"],
            "validation": values["validation_file"],
        },
        cache_dir=values["dataset_cache_dir"],
    )
    raw["train"] = _select_limit(raw["train"], values["train_limit"], values["seed"])
    raw["validation"] = _select_limit(
        raw["validation"], values["validation_limit"], values["seed"] + 1
    )
    input_format = _detect_input_format(set(raw["train"].column_names), "raw-spans")
    tokenized_train = _tokenize_dataset(
        raw["train"],
        tokenizer=tokenizer,
        label2id=parent["label2id"],
        input_format=input_format,
        max_length=values["max_length"],
        stride=values["stride"],
        description="Tokenizzazione train distillation",
    )
    tokenized_validation = _tokenize_dataset(
        raw["validation"],
        tokenizer=tokenizer,
        label2id=parent["label2id"],
        input_format=input_format,
        max_length=values["max_length"],
        stride=values["stride"],
        description="Tokenizzazione validation distillation",
    )

    cache_manifest: dict[str, Any] | None = None
    cache_tensors: dict[str, Any] | None = None
    kd_preflight: dict[str, Any] | None = None
    if float(values["alpha"]) > 0.0:
        cache_manifest, cache_tensors = _validate_teacher_cache(
            values["teacher_cache"],
            tokenized_train=tokenized_train,
            tokenizer=tokenizer,
            parent=parent,
            values=values,
        )
        values["cache_loaded"] = True
        kd_preflight = _audit_kd_cache_targets(
            teacher_logits=cache_tensors["teacher_logits"],
            labels=cache_tensors["labels"],
            input_ids=cache_tensors["input_ids"],
            tokenizer=tokenizer,
            label_names=parent["labels"],
            temperature=values["temperature"],
            kd_mode=values["kd_mode"],
            min_coverage=values["kd_min_coverage"],
            max_coverage=values["kd_max_coverage"],
        )

    after_data_snapshot = _memory_snapshot(torch, psutil, device)
    memory_timeline.append({"event": "after_data_and_cache", **after_data_snapshot})

    code_before = _code_integrity()
    recipe = dict(values)
    recipe["parent_initialization"] = "weights-only"
    recipe["optimizer_state_loaded"] = False
    recipe["scheduler_state_loaded"] = False
    recipe["teacher_loaded"] = False
    manifest = {
        "schema_version": 1,
        "kind": RUN_KIND,
        "status": "prepared",
        "release_eligible": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "host": {"platform": platform.platform(), "machine": platform.machine()},
        "parent": {
            "run_dir": parent["run_dir"],
            "run_manifest": parent["manifest_integrity"],
            "training_summary": parent["summary_integrity"],
            "model": parent["model_integrity"],
            "tokenizer": parent["tokenizer_integrity"],
            "initialization": "weights-only",
            "optimizer_state_loaded": False,
            "scheduler_state_loaded": False,
        },
        "teacher_cache": (
            None
            if cache_manifest is None
            else {
                "path": str(Path(values["teacher_cache"]).resolve()),
                "manifest": cache_manifest,
                "loaded": True,
            }
        ),
        "teacher_loaded": False,
        "kd_preflight": kd_preflight,
        "memory_guard_policy": memory_guard_policy,
        "label_contract": {
            **parent["teacher_label_contract"],
            "labels": list(parent["labels"]),
        },
        "data": parent["data"],
        "preprocessing": {
            "input_format": input_format,
            "max_length": values["max_length"],
            "stride": values["stride"],
            "train_rows": len(raw["train"]),
            "validation_rows": len(raw["validation"]),
            "train_windows": len(tokenized_train),
            "validation_windows": len(tokenized_validation),
        },
        "training": recipe,
        "software": _software_versions(),
        "code": code_before,
    }
    manifest["recipe_sha256"] = _canonical_sha256(
        {
            key: manifest[key]
            for key in (
                "kind",
                "parent",
                "teacher_cache",
                "teacher_loaded",
                "kd_preflight",
                "memory_guard_policy",
                "label_contract",
                "data",
                "preprocessing",
                "training",
                "software",
                "code",
            )
        }
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    _write_json(run_dir / "run-manifest.json", manifest)
    _enforce_pretraining_memory_guard(
        after_data_snapshot,
        phase="after_data_and_cache",
        memory_guard=memory_guard,
    )

    model, loading_info = AutoModelForTokenClassification.from_pretrained(
        values["parent_model"],
        trust_remote_code=False,
        attn_implementation=values["attention_implementation"],
        output_loading_info=True,
    )
    missing = list(loading_info.get("missing_keys", []))
    unexpected = list(loading_info.get("unexpected_keys", []))
    mismatched = list(loading_info.get("mismatched_keys", []))
    errors = list(loading_info.get("error_msgs", []))
    if missing or unexpected or mismatched or errors:
        raise StudentTrainingError(
            "Caricamento weights-only non pulito: "
            f"missing={missing}, unexpected={unexpected}, mismatched={mismatched}, errors={errors}"
        )
    if values["gradient_checkpointing"]:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    if total_parameters != int(parent["summary"]["total_parameters"]):
        raise StudentTrainingError("Numero parametri diverso dal parent")
    if trainable_parameters != int(parent["summary"]["trainable_parameters"]):
        raise StudentTrainingError("Trainable set diverso dal parent")

    after_model_snapshot = _memory_snapshot(torch, psutil, device)
    memory_timeline.append({"event": "after_model_load", **after_model_snapshot})
    _enforce_pretraining_memory_guard(
        after_model_snapshot,
        phase="after_model_load",
        memory_guard=memory_guard,
    )

    def preprocess_logits(logits: Any, _: Any) -> Any:
        return (logits[0] if isinstance(logits, tuple) else logits).argmax(dim=-1)

    def trainer_metrics(evaluation: Any) -> dict[str, float]:
        predictions = evaluation.predictions
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        gold_docs, prediction_docs = prediction_documents(
            predictions, evaluation.label_ids, parent["id2label"]
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
            memory_timeline.append({"event": "log", "step": int(state.global_step), **snapshot})
            reason = _observe_memory_guard(
                snapshot,
                phase="trainer_log",
                step=int(state.global_step),
                memory_guard=memory_guard,
            )
            if reason is not None:
                control.should_training_stop = True
            return control

    teacher_logits = cache_tensors["teacher_logits"] if cache_tensors is not None else None
    train_dataset = _IndexedDataset(tokenized_train) if teacher_logits is not None else tokenized_train
    base_collator = DataCollatorForTokenClassification(tokenizer=tokenizer, padding=True)
    collator = DistillationCollator(
        base_collator,
        teacher_logits=teacher_logits,
        padding_side=str(tokenizer.padding_side),
    )
    DistillationTrainer = _build_trainer_class(
        Trainer,
        alpha=values["alpha"],
        temperature=values["temperature"],
        kd_mode=values["kd_mode"],
        label_names=parent["labels"],
    )
    training_args = TrainingArguments(
        output_dir=str(run_dir / "checkpoints"),
        overwrite_output_dir=False,
        do_train=True,
        do_eval=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        per_device_train_batch_size=values["train_batch_size"],
        per_device_eval_batch_size=values["eval_batch_size"],
        gradient_accumulation_steps=values["gradient_accumulation_steps"],
        num_train_epochs=values["epochs"],
        max_steps=values["max_steps"],
        learning_rate=values["learning_rate"],
        weight_decay=values["weight_decay"],
        warmup_ratio=values["warmup_ratio"],
        max_grad_norm=1.0,
        lr_scheduler_type=values["lr_scheduler_type"],
        optim=values["optimizer"],
        bf16=values["precision"] == "bf16",
        fp16=False,
        use_cpu=device == "cpu",
        gradient_checkpointing=values["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        eval_accumulation_steps=16,
        logging_strategy="steps",
        logging_steps=10,
        report_to=[],
        seed=values["seed"],
        data_seed=values["seed"],
        save_safetensors=True,
        torch_empty_cache_steps=values["torch_empty_cache_steps"],
        torch_compile=False,
        remove_unused_columns=False,
        run_name=run_dir.name,
    )
    trainer = DistillationTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=tokenized_validation,
        data_collator=collator,
        processing_class=tokenizer,
        compute_metrics=trainer_metrics,
        preprocess_logits_for_metrics=preprocess_logits,
        callbacks=[MemoryCallback()],
    )

    started = time.monotonic()
    train_result = trainer.train()  # intenzionalmente: optimizer/scheduler nuovi
    if device == "mps":
        torch.mps.synchronize()
    elapsed = time.monotonic() - started
    memory_timeline.append({"event": "after_train", **_memory_snapshot(torch, psutil, device)})
    final_dir = run_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    global_step = int(trainer.state.global_step)
    planned_steps = int(trainer.state.max_steps)
    status = _progress_status(global_step, planned_steps, bool(memory_guard["triggered"]))
    summary = {
        "status": status,
        "release_eligible": False,
        "kind": RUN_KIND,
        "device": device,
        "precision": values["precision"],
        "alpha": values["alpha"],
        "temperature": values["temperature"],
        "kd_mode": values["kd_mode"],
        "kd_preflight": kd_preflight,
        "tokenizer_load_policy": values["tokenizer_load_policy"],
        "gold_weight": 1.0 - values["alpha"],
        "teacher_cache_required": values["alpha"] > 0.0,
        "teacher_cache_loaded": bool(values["cache_loaded"]),
        "teacher_loaded": False,
        "parent_initialization": "weights-only",
        "optimizer_state_loaded": False,
        "scheduler_state_loaded": False,
        "optimizer": values["optimizer"],
        "learning_rate": values["learning_rate"],
        "epochs": values["epochs"],
        "seed": values["seed"],
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "train_rows": len(raw["train"]),
        "validation_rows": len(raw["validation"]),
        "train_windows": len(tokenized_train),
        "validation_windows": len(tokenized_validation),
        "elapsed_seconds": elapsed,
        "global_step": global_step,
        "planned_steps": planned_steps,
        "completion_ratio": global_step / planned_steps,
        "trainer_metrics": dict(train_result.metrics),
        "loading_info": loading_info,
        "memory_timeline": memory_timeline,
        "memory_guard": memory_guard,
        "run_manifest_recipe_sha256": manifest["recipe_sha256"],
        "final_model": {
            "model": __import__(
                "src.training.cache_teacher_logits", fromlist=["artifact_group_integrity"]
            ).artifact_group_integrity(final_dir, "model"),
            "tokenizer": __import__(
                "src.training.cache_teacher_logits", fromlist=["artifact_group_integrity"]
            ).artifact_group_integrity(final_dir, "tokenizer"),
        },
    }

    # Verifiche finali: nessun input, parent, cache o codice puo' cambiare in corsa.
    if _file_integrity(parent["manifest_integrity"]["path"])["sha256"] != parent["manifest_integrity"]["sha256"]:
        raise StudentTrainingError("Manifest parent modificato durante il run")
    if _file_integrity(parent["summary_integrity"]["path"]) != parent["summary_integrity"]:
        raise StudentTrainingError("Training summary parent modificato durante il run")
    if _file_integrity(parent["teacher_label_contract"]["path"]) != parent["teacher_label_contract"]:
        raise StudentTrainingError("Contratto label parent modificato durante il run")
    from src.training.cache_teacher_logits import artifact_group_integrity

    if artifact_group_integrity(parent["final_dir"], "model") != parent["model_integrity"]:
        raise StudentTrainingError("Pesi parent modificati durante il run")
    if artifact_group_integrity(parent["final_dir"], "tokenizer") != parent["tokenizer_integrity"]:
        raise StudentTrainingError("Tokenizer parent modificato durante il run")
    for split in ("train", "validation"):
        if sha256_file(parent["data"][split]["path"]) != parent["data"][split]["sha256"]:
            raise StudentTrainingError(f"Dataset {split} modificato durante il run")
    if _code_integrity() != code_before:
        raise StudentTrainingError("Codice distillation modificato durante il run")
    if values["alpha"] > 0.0:
        from src.training.cache_teacher_logits import load_and_validate_cache_manifest

        refreshed = load_and_validate_cache_manifest(values["teacher_cache"])
        if refreshed.get("cache_identity_sha256") != cache_manifest.get("cache_identity_sha256"):
            raise StudentTrainingError("Cache teacher modificata durante il run")
    _write_json(run_dir / "training-summary.json", summary)
    _finalize_run_lifecycle(
        run_dir,
        lifecycle_status="complete",
        summary=summary,
    )
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    _validate_recipe(args)
    parent = _load_parent_context(args.parent_run)
    values = _resolved_configuration(args, parent)

    if args.mode == "inspect":
        # Inspect non apre intenzionalmente cache/model: dichiara solo il contratto.
        print(
            json.dumps(
                {
                    "status": "ok",
                    "parent": parent["run_dir"],
                    "parent_model_sha256": parent["model_integrity"]["sha256"],
                    "teacher_cache_required": values["alpha"] > 0.0,
                    "teacher_cache_loaded": False,
                    "recipe": values,
                },
                indent=2,
            )
        )
        return 0

    free_bytes = shutil.disk_usage(ROOT).free
    if free_bytes < 3 * 1024**3:
        raise StudentTrainingError(
            f"Spazio libero insufficiente: {free_bytes / 1024**3:.1f} GiB"
        )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    identity = _canonical_sha256(
        {
            "parent": parent["model_integrity"]["sha256"],
            "alpha": values["alpha"],
            "temperature": values["temperature"],
            "kd_mode": values["kd_mode"],
            "kd_min_coverage": values["kd_min_coverage"],
            "kd_max_coverage": values["kd_max_coverage"],
            "learning_rate": values["learning_rate"],
            "epochs": values["epochs"],
            "seed": values["seed"],
        }
    )
    run_dir = (
        _root_path(args.output_dir)
        if args.output_dir
        else ROOT / "artifacts" / "training" / "runs" / f"{stamp}-distill-{identity[:10]}"
    )
    if run_dir.exists():
        raise StudentTrainingError(f"Output gia' esistente: {run_dir}")
    try:
        summary = _run_training(args, parent, values, run_dir)
    except BaseException as exc:
        if run_dir.is_dir():
            manifest_path = run_dir / "run-manifest.json"
            if manifest_path.is_file():
                current_manifest = _read_json_object(
                    manifest_path, "Run manifest distillation"
                )
                if current_manifest.get("status") == "prepared":
                    failure_summary = {
                        "status": "failed",
                        "release_eligible": False,
                        "kind": RUN_KIND,
                        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "run_manifest_recipe_sha256": current_manifest.get(
                            "recipe_sha256"
                        ),
                    }
                    failed_guard = getattr(exc, "memory_guard", None)
                    if isinstance(failed_guard, Mapping):
                        failure_summary["memory_guard"] = dict(failed_guard)
                    _write_json(
                        run_dir / "training-summary.json", failure_summary
                    )
                    _finalize_run_lifecycle(
                        run_dir,
                        lifecycle_status="failed",
                        summary=failure_summary,
                    )
        raise
    print(json.dumps({"run_dir": str(run_dir), **summary}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except StudentTrainingError as exc:
        print(f"ERRORE: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
