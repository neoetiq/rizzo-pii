"""Pinned source acquisition and integrity checks for quantization experiments.

The module intentionally has no eager ML imports.  prepare_sources imports
huggingface_hub only when it is actually asked to download.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path(__file__).with_name("config.json")
LOCK_NAME = "sources.lock.json"
HASH_CHUNK_BYTES = 1024 * 1024


class SourceValidationError(ValueError):
    """A pinned source does not match the contract required by the benchmark."""


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load the committed, human-readable quantization source contract."""
    path = Path(config_path) if config_path else DEFAULT_CONFIG
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SourceValidationError(f"Configurazione sorgenti non trovata: {path}") from exc
    except json.JSONDecodeError as exc:
        raise SourceValidationError(f"JSON non valido in {path}: {exc}") from exc

    if config.get("schema_version") != 1:
        raise SourceValidationError("schema_version non supportata")
    for section in ("model", "dataset", "validation"):
        if not isinstance(config.get(section), dict):
            raise SourceValidationError(f"Sezione obbligatoria assente: {section}")
    return config


def _resolve_output_dir(config: dict[str, Any], output_dir: str | Path | None) -> Path:
    requested = Path(output_dir) if output_dir else Path(config["output_dir"])
    return requested if requested.is_absolute() else ROOT / requested


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_manifest(directory: Path) -> list[dict[str, Any]]:
    """Return deterministic hashes for every downloaded model file."""
    manifest = []
    for path in sorted(item for item in directory.rglob("*") if item.is_file()):
        relative = path.relative_to(directory)
        # snapshot_download(local_dir=...) stores transport metadata in .cache/.
        # It is not part of the source artifact and can vary between hub versions.
        if relative.parts[0] == ".cache":
            continue
        manifest.append(
            {
                "path": relative.as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    if not manifest:
        raise SourceValidationError(f"Nessun file modello trovato in {directory}")
    return manifest


def verify_sources(
    config_path: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Recheck pinned revisions, hashes and sizes before using local sources."""
    config = load_config(config_path)
    destination = _resolve_output_dir(config, output_dir).resolve()
    lock_path = destination / LOCK_NAME
    try:
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SourceValidationError(
            f"Lock sorgenti non trovato: {lock_path}; eseguire prima il comando sources"
        ) from exc
    except json.JSONDecodeError as exc:
        raise SourceValidationError(f"Lock sorgenti non valido: {lock_path}") from exc

    model_lock = lock.get("model", {})
    dataset_lock = lock.get("dataset", {})
    if model_lock.get("revision_resolved") != config["model"]["revision"]:
        raise SourceValidationError("Revisione modello nel lock diversa dalla configurazione")
    if dataset_lock.get("revision_resolved") != config["dataset"]["revision"]:
        raise SourceValidationError("Revisione dataset nel lock diversa dalla configurazione")

    expected_model_files = model_lock.get("files")
    if not isinstance(expected_model_files, list):
        raise SourceValidationError("Manifest modello assente nel lock")
    actual_model_files = _file_manifest(destination / "model")
    if actual_model_files != expected_model_files:
        raise SourceValidationError(
            "Checkpoint locale diverso dal manifest bloccato; rieseguire sources in una directory pulita"
        )

    checked_dataset_files: dict[str, dict[str, Any]] = {}
    locked_dataset_files = dataset_lock.get("files", {})
    for purpose in ("validation", "calibration"):
        expected = locked_dataset_files.get(purpose)
        if not isinstance(expected, dict):
            raise SourceValidationError(f"File dataset {purpose} assente nel lock")
        configured = config["dataset"][purpose]
        if int(expected.get("rows", -1)) != int(configured["expected_rows"]):
            raise SourceValidationError(f"Conteggio {purpose} nel lock non coerente")
        relative = Path(str(expected.get("path", "")))
        candidate = (destination / relative).resolve()
        try:
            candidate.relative_to(destination)
        except ValueError as exc:
            raise SourceValidationError(f"Path {purpose} fuori dalla directory artefatti") from exc
        if not candidate.is_file():
            raise SourceValidationError(f"File dataset {purpose} mancante: {candidate}")
        actual = {"bytes": candidate.stat().st_size, "sha256": _sha256(candidate)}
        if actual["bytes"] != expected.get("bytes") or actual["sha256"] != expected.get("sha256"):
            raise SourceValidationError(f"Hash o dimensione non validi per {purpose}: {candidate}")
        checked_dataset_files[purpose] = {"path": relative.as_posix(), **actual}

    return {
        "status": "verified",
        "lock_path": str(lock_path),
        "lock_sha256": _sha256(lock_path),
        "model_files": len(actual_model_files),
        "dataset_files": checked_dataset_files,
        "model_revision": config["model"]["revision"],
        "dataset_revision": config["dataset"]["revision"],
    }


def validate_jsonl(
    path: str | Path, *, expected_rows: int, required_fields: list[str]
) -> dict[str, Any]:
    """Validate row count and the word-level token-classification shape."""
    source = Path(path)
    rows = 0
    tokens_total = 0
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise SourceValidationError(f"{source}:{line_number}: riga vuota")
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SourceValidationError(f"{source}:{line_number}: JSON non valido") from exc
            if not isinstance(record, dict):
                raise SourceValidationError(f"{source}:{line_number}: record non oggetto")
            missing = [field for field in required_fields if field not in record]
            if missing:
                raise SourceValidationError(f"{source}:{line_number}: campi mancanti {missing}")
            tokens, labels = record["tokens"], record["bio_labels"]
            if not isinstance(tokens, list) or not isinstance(labels, list):
                raise SourceValidationError(
                    f"{source}:{line_number}: tokens/bio_labels devono essere liste"
                )
            if not tokens or len(tokens) != len(labels):
                raise SourceValidationError(
                    f"{source}:{line_number}: {len(tokens)} token e {len(labels)} label"
                )
            if not all(isinstance(token, str) for token in tokens):
                raise SourceValidationError(f"{source}:{line_number}: token non testuale")
            if not all(isinstance(label, str) for label in labels):
                raise SourceValidationError(f"{source}:{line_number}: label non testuale")
            rows += 1
            tokens_total += len(tokens)

    if rows != expected_rows:
        raise SourceValidationError(f"{source}: attese {expected_rows} righe, trovate {rows}")
    return {
        "rows": rows,
        "tokens": tokens_total,
        "bytes": source.stat().st_size,
        "sha256": _sha256(source),
    }


def _software_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    for distribution in (
        "huggingface_hub",
        "transformers",
        "torch",
        "accelerate",
        "onnx",
        "onnx-ir",
        "onnxruntime",
        "optimum",
        "optimum-onnx",
        "numpy",
        "datasets",
    ):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None
    return versions


def prepare_sources(
    config_path: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Download the pinned sources, validate them, and write sources.lock.json.

    Config pins immutable Git commits.  The generated lock adds SHA-256/byte length
    for every fetched file and the software environment that created the experiment.
    """
    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "Dipendenza mancante: installare requirements-quantization.txt"
        ) from exc

    config = load_config(config_path)
    destination = _resolve_output_dir(config, output_dir)
    model_destination = destination / "model"
    data_destination = destination / "data"
    model_destination.mkdir(parents=True, exist_ok=True)
    data_destination.mkdir(parents=True, exist_ok=True)

    model = config["model"]
    snapshot_download(
        repo_id=model["repo_id"],
        repo_type="model",
        revision=model["revision"],
        allow_patterns=model["allow_patterns"],
        local_dir=model_destination,
    )
    model_files = _file_manifest(model_destination)
    if not any(item["path"].endswith(".safetensors") for item in model_files):
        raise SourceValidationError("Snapshot modello privo di pesi .safetensors")
    if not any(item["path"] == "config.json" for item in model_files):
        raise SourceValidationError("Snapshot modello privo di config.json")

    dataset = config["dataset"]
    required_fields = config["validation"]["required_fields"]
    data_lock: dict[str, Any] = {}
    for purpose in ("validation", "calibration"):
        spec = dataset[purpose]
        local_path = Path(
            hf_hub_download(
                repo_id=dataset["repo_id"],
                repo_type="dataset",
                filename=spec["path"],
                revision=dataset["revision"],
                local_dir=data_destination,
            )
        )
        result = validate_jsonl(
            local_path,
            expected_rows=spec["expected_rows"],
            required_fields=required_fields,
        )
        result["path"] = local_path.relative_to(destination).as_posix()
        data_lock[purpose] = result

    lock = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": "python -m src.quantization.cli sources",
        "python_executable": sys.executable,
        "software": _software_versions(),
        "model": {
            "repo_id": model["repo_id"],
            "revision_name": model["revision_name"],
            "revision_resolved": model["revision"],
            "files": model_files,
        },
        "dataset": {
            "repo_id": dataset["repo_id"],
            "revision_name": dataset["revision_name"],
            "revision_resolved": dataset["revision"],
            "files": data_lock,
        },
    }
    lock_path = destination / LOCK_NAME
    lock_path.write_text(json.dumps(lock, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"output_dir": str(destination), "lock_path": str(lock_path), "lock": lock}
