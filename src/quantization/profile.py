"""Statistically robust, process-isolated CPU service profiling helpers.

This module deliberately stays separate from the semantic regression runner.  A
profile run is expected to describe one fresh child process with this shape::

    {
        "variant": "onnx-int8-full",
        "round": 0,
        "position": 1,
        "configuration": {...},
        "prediction_digest": "sha256:...",
        "artifact_integrity": {"files": [{"path": ..., "sha256": ...}]},
        "load_seconds": 1.2,
        "documents_per_second": 35.0,
        "request_latency_seconds": {"p95": 0.05},
        "model_latency_seconds": {"p95": 0.04},
        "memory": {"peak": {"rss_bytes": ..., "pss_bytes": ...}},
    }

The orchestrator can attach the memory result returned by
``run_profile_subprocess`` to the child report.  Full artifact hashes are
verified before scheduling and after the complete suite.  Consequently every
variant follows the same documented warm-cache policy and no artifact is hashed
immediately before only its own run.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import random
import statistics
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


MEMORY_KEYS = (
    "rss_bytes",
    "pss_bytes",
    "uss_bytes",
    "private_clean_bytes",
    "private_dirty_bytes",
    "private_hugetlb_bytes",
    "anonymous_bytes",
    "swap_bytes",
)

PROFILE_METRICS = {
    "load_seconds": ("load_seconds",),
    "documents_per_second": ("documents_per_second",),
    "request_latency_p95_seconds": ("request_latency_seconds", "p95"),
    "model_latency_p95_seconds": ("model_latency_seconds", "p95"),
    "peak_rss_bytes": ("memory", "peak", "rss_bytes"),
    "peak_pss_bytes": ("memory", "peak", "pss_bytes"),
    "peak_uss_bytes": ("memory", "peak", "uss_bytes"),
    "peak_anonymous_bytes": ("memory", "peak", "anonymous_bytes"),
}


def build_profile_schedule(
    variants: Sequence[str],
    repeats: int,
    seed: int,
    order: str = "balanced",
) -> list[dict[str, int | str]]:
    """Return a deterministic schedule containing every variant in every round.

    ``balanced`` uses randomized cyclic Latin-square blocks.  In a complete
    block of ``len(variants)`` rounds every variant occupies every position once;
    the following block reverses the base order to counter persistent neighbour
    effects.  ``random`` performs a seeded shuffle per round, while ``fixed`` is
    mainly useful for diagnostics.
    """

    names = list(variants)
    if not names:
        raise ValueError("Serve almeno una variante")
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError("I nomi delle varianti devono essere stringhe non vuote")
    if len(set(names)) != len(names):
        raise ValueError("Le varianti devono essere univoche")
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ValueError("repeats deve essere un intero positivo")
    if order not in {"balanced", "random", "fixed"}:
        raise ValueError(f"Ordine del profilo non supportato: {order}")

    rng = random.Random(seed)
    schedule: list[dict[str, int | str]] = []
    width = len(names)
    pair_bases: dict[int, list[str]] = {}

    for round_index in range(repeats):
        if order == "fixed":
            round_variants = list(names)
        elif order == "random":
            round_variants = list(names)
            rng.shuffle(round_variants)
        else:
            block = round_index // width
            pair = block // 2
            if pair not in pair_bases:
                base = list(names)
                rng.shuffle(base)
                pair_bases[pair] = base
            base = list(pair_bases[pair])
            if block % 2:
                base.reverse()
            shift = round_index % width
            round_variants = base[shift:] + base[:shift]

        schedule.extend(
            {
                "round": round_index,
                "position": position,
                "variant": variant,
            }
            for position, variant in enumerate(round_variants)
        )
    return schedule


def _memory_template() -> dict[str, int | None]:
    return {key: None for key in MEMORY_KEYS}


def _quantity_to_bytes(value: str, unit: str | None) -> int:
    number = int(value)
    normalized = (unit or "B").lower()
    factors = {"b": 1, "kb": 1024, "mb": 1024**2, "gb": 1024**3}
    if normalized not in factors:
        raise ValueError(f"Unita di memoria non supportata: {unit}")
    return number * factors[normalized]


def parse_smaps_rollup(text: str) -> dict[str, int | None]:
    """Parse Linux ``smaps_rollup`` values and normalize them to bytes."""

    wanted = {
        "Rss": "rss_bytes",
        "Pss": "pss_bytes",
        "Private_Clean": "private_clean_bytes",
        "Private_Dirty": "private_dirty_bytes",
        "Private_Hugetlb": "private_hugetlb_bytes",
        "Anonymous": "anonymous_bytes",
        "Swap": "swap_bytes",
    }
    result = _memory_template()
    for raw_line in text.splitlines():
        if ":" not in raw_line:
            continue
        raw_key, raw_value = raw_line.split(":", 1)
        output_key = wanted.get(raw_key.strip())
        if output_key is None:
            continue
        fields = raw_value.split()
        if not fields:
            continue
        try:
            result[output_key] = _quantity_to_bytes(
                fields[0], fields[1] if len(fields) > 1 else None
            )
        except (ValueError, TypeError):
            # A partially written or future kernel field must not make the
            # profiler fail; the missing metric remains explicitly unavailable.
            result[output_key] = None

    private_clean = result["private_clean_bytes"]
    private_dirty = result["private_dirty_bytes"]
    private_hugetlb = result["private_hugetlb_bytes"]
    if (
        private_clean is not None
        and private_dirty is not None
        and private_hugetlb is not None
    ):
        result["uss_bytes"] = private_clean + private_dirty + private_hugetlb
    return result


def sample_process_memory(pid: int) -> dict[str, int | None]:
    """Return one cross-platform process-memory snapshot.

    Linux uses ``smaps_rollup`` first because it exposes the distinction needed
    for VPS sizing.  ``psutil`` supplies RSS/PSS/USS where the platform supports
    them and acts as a fallback when ``/proc`` disappears as the child exits.
    An unavailable metric is represented by ``None``, never by a misleading
    zero.
    """

    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        raise ValueError("pid deve essere un intero positivo")

    result = _memory_template()
    if sys.platform.startswith("linux"):
        try:
            result.update(
                parse_smaps_rollup(
                    Path(f"/proc/{pid}/smaps_rollup").read_text(encoding="utf-8")
                )
            )
        except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
            pass

    try:
        import psutil

        process = psutil.Process(pid)
        try:
            info = process.memory_full_info()
        except (psutil.AccessDenied, AttributeError):
            info = process.memory_info()
        for attribute, key in (
            ("rss", "rss_bytes"),
            ("pss", "pss_bytes"),
            ("uss", "uss_bytes"),
            ("swap", "swap_bytes"),
        ):
            value = getattr(info, attribute, None)
            if result[key] is None and value is not None:
                result[key] = int(value)
    except (ImportError, ProcessLookupError, PermissionError, OSError):
        pass
    except Exception as exc:
        # psutil uses platform-specific exception subclasses.  Do not import it
        # only to enumerate them when the process commonly vanishes between
        # ``poll`` and sampling, but do not hide unrelated programming errors.
        if exc.__class__.__module__.startswith("psutil"):
            pass
        else:
            raise
    return result


def run_profile_subprocess(
    command: Sequence[str],
    cwd: Path,
    sample_interval_ms: int,
) -> dict[str, Any]:
    """Run one fresh child and sample independent peak memory externally."""

    argv = [str(item) for item in command]
    if not argv:
        raise ValueError("Il comando del profilo non puo essere vuoto")
    if (
        isinstance(sample_interval_ms, bool)
        or not isinstance(sample_interval_ms, int)
        or sample_interval_ms < 1
    ):
        raise ValueError("sample_interval_ms deve essere un intero positivo")

    peaks = _memory_template()
    samples = 0
    started = time.perf_counter()
    with tempfile.TemporaryFile(mode="w+b") as stdout_file, tempfile.TemporaryFile(
        mode="w+b"
    ) as stderr_file:
        process = subprocess.Popen(
            argv,
            cwd=str(cwd),
            stdout=stdout_file,
            stderr=stderr_file,
            shell=False,
        )
        while True:
            snapshot = sample_process_memory(process.pid)
            samples += 1
            if any(value is not None for value in snapshot.values()):
                for key in MEMORY_KEYS:
                    value = snapshot[key]
                    if value is not None and (peaks[key] is None or value > peaks[key]):
                        peaks[key] = value
            returncode = process.poll()
            if returncode is not None:
                break
            time.sleep(sample_interval_ms / 1000)

        # ``poll`` reaps the process on supported Python platforms, but wait is
        # explicit here so the returned code is never ambiguous.
        returncode = process.wait()
        stdout_file.seek(0)
        stderr_file.seek(0)
        stdout = stdout_file.read().decode("utf-8", errors="replace")
        stderr = stderr_file.read().decode("utf-8", errors="replace")

    return {
        "command": argv,
        "cwd": str(Path(cwd).resolve()),
        "pid": process.pid,
        "returncode": returncode,
        "status": "ok" if returncode == 0 else "failed",
        "elapsed_seconds": time.perf_counter() - started,
        "stdout": stdout,
        "stderr": stderr,
        "memory": {
            "sample_interval_ms": sample_interval_ms,
            "samples": samples,
            "peak": peaks,
        },
    }


def _percentile(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = index - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def summarize(values: Sequence[int | float | None]) -> dict[str, int | float | None]:
    """Summarize finite values, ignoring only explicitly unavailable entries."""

    usable: list[float] = []
    for value in values:
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"Valore statistico non numerico: {value!r}")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"Valore statistico non finito: {value!r}")
        usable.append(number)

    empty: dict[str, int | float | None] = {
        "count": 0,
        "min": None,
        "p05": None,
        "median": None,
        "mean": None,
        "p95": None,
        "max": None,
        "stdev": None,
        "cv": None,
    }
    if not usable:
        return empty

    mean = statistics.fmean(usable)
    stdev = statistics.pstdev(usable)
    return {
        "count": len(usable),
        "min": min(usable),
        "p05": _percentile(usable, 0.05),
        "median": statistics.median(usable),
        "mean": mean,
        "p95": _percentile(usable, 0.95),
        "max": max(usable),
        "stdev": stdev,
        "cv": stdev / abs(mean) if mean else None,
    }


def _canonical(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise ValueError("Il profilo contiene dati non serializzabili in JSON") from exc


def _artifact_hash_identity(value: Any) -> tuple[tuple[str, str], ...]:
    if isinstance(value, str) and value:
        return (("artifact", value),)
    if not isinstance(value, Mapping):
        raise ValueError("artifact_integrity deve contenere hash SHA-256")
    direct = value.get("sha256")
    if isinstance(direct, str) and direct:
        return ((str(value.get("path", "artifact")), direct),)
    files = value.get("files")
    if not isinstance(files, Sequence) or isinstance(files, (str, bytes)):
        raise ValueError("artifact_integrity.files mancante")
    hashes: list[tuple[str, str]] = []
    for index, item in enumerate(files):
        if not isinstance(item, Mapping) or not isinstance(item.get("sha256"), str):
            raise ValueError("Ogni file dell'artefatto deve avere sha256")
        digest = str(item["sha256"])
        if not digest:
            raise ValueError("Hash SHA-256 vuoto")
        hashes.append((str(item.get("path", index)), digest))
    if not hashes:
        raise ValueError("artifact_integrity.files non puo essere vuoto")
    return tuple(sorted(hashes))


def _normalize_sha256(value: Any, description: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{description} deve essere uno SHA-256")
    digest = value.removeprefix("sha256:").lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{description} non e uno SHA-256 valido")
    return digest


def _prediction_file_integrity(path: Path) -> dict[str, Any]:
    """Hash one prediction file while guarding against replacement during I/O."""

    try:
        before = path.stat()
    except (FileNotFoundError, OSError) as exc:
        raise ValueError(f"File di predizione non leggibile: {path}") from exc
    if not path.is_file():
        raise ValueError(f"Il riferimento alle predizioni non e un file: {path}")

    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        after = path.stat()
    except (FileNotFoundError, OSError) as exc:
        raise ValueError(f"File di predizione non leggibile: {path}") from exc

    stat_identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    stat_identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if stat_identity_after != stat_identity_before:
        raise ValueError(f"File di predizione modificato durante il controllo: {path}")
    return {
        "path": str(path),
        "bytes": int(after.st_size),
        "sha256": digest.hexdigest(),
    }


def _verify_prediction_reference(
    run: Mapping[str, Any], *, suite_id: str, suite_root: Path
) -> dict[str, Any]:
    """Bind a run to one immutable prediction file inside its profile suite."""

    if run.get("suite_id") != suite_id:
        raise ValueError(
            f"Run appartenente a una suite diversa: {run.get('suite_id')!r} != {suite_id!r}"
        )
    raw_path = run.get("predictions_path")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError("Ogni run deve referenziare predictions_path")
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        raise ValueError("predictions_path deve essere assoluto")
    try:
        resolved = candidate.resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise ValueError(f"File di predizione non leggibile: {candidate}") from exc
    if not resolved.is_relative_to(suite_root):
        raise ValueError(
            f"File di predizione esterno alla suite {suite_id}: {resolved}"
        )

    integrity = _prediction_file_integrity(resolved)
    expected = _normalize_sha256(
        run.get("prediction_digest"), "prediction_digest"
    )
    if not hmac.compare_digest(expected, str(integrity["sha256"])):
        raise ValueError(
            f"SHA-256 del file di predizione non corrispondente: {resolved}"
        )
    return integrity


def _nested(run: Mapping[str, Any], path: Sequence[str]) -> Any:
    value: Any = run
    for key in path:
        if not isinstance(value, Mapping) or key not in value:
            return None
        value = value[key]
    return value


def _metric_value(
    run: Mapping[str, Any], metric: str, *, optional: bool = False
) -> int | float | None:
    value = _nested(run, PROFILE_METRICS[metric])
    if value is None:
        if optional:
            return None
        raise ValueError(f"Metrica obbligatoria mancante: {metric}")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Metrica {metric} non numerica")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"Metrica {metric} non valida: {value!r}")
    return value


def aggregate_profile_runs(
    runs: Sequence[Mapping[str, Any]],
    variants: Sequence[str],
    repeats: int,
    methodology: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and aggregate one fresh-process run per variant and round."""

    names = list(variants)
    # Reuse schedule validation for names/repeats without imposing its order on
    # already executed runs.
    build_profile_schedule(names, repeats, seed=0)
    if not isinstance(methodology, Mapping):
        raise ValueError("methodology deve essere una mappa")
    suite_id = methodology.get("suite_id")
    raw_suite_root = methodology.get("suite_root")
    if not isinstance(suite_id, str) or not suite_id:
        raise ValueError("methodology.suite_id deve identificare la suite corrente")
    if not isinstance(raw_suite_root, str) or not raw_suite_root:
        raise ValueError("methodology.suite_root deve identificare la suite corrente")
    suite_root_candidate = Path(raw_suite_root)
    if not suite_root_candidate.is_absolute():
        raise ValueError("methodology.suite_root deve essere assoluto")
    try:
        suite_root = suite_root_candidate.resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise ValueError(f"Directory della suite non leggibile: {suite_root_candidate}") from exc
    if not suite_root.is_dir():
        raise ValueError(f"La root della suite non e una directory: {suite_root}")
    if len(runs) != len(names) * repeats:
        raise ValueError(
            f"Attesi {len(names) * repeats} run, ricevuti {len(runs)}"
        )

    expected_positions = set(range(len(names)))
    expected_variants = set(names)
    by_round: dict[int, list[Mapping[str, Any]]] = {
        round_index: [] for round_index in range(repeats)
    }
    seen_slots: set[tuple[int, int]] = set()
    configuration_identity: str | None = None
    configuration: Mapping[str, Any] | None = None
    digests: dict[str, str] = {}
    prediction_integrity: dict[str, list[dict[str, Any]]] = {
        name: [] for name in names
    }
    prediction_paths: set[str] = set()
    hash_identities: dict[str, tuple[tuple[str, str], ...]] = {}
    artifact_integrity: dict[str, Any] = {}

    for run in runs:
        variant = run.get("variant")
        if variant not in expected_variants:
            raise ValueError(f"Variante inattesa nel profilo: {variant!r}")
        round_index = run.get("round")
        position = run.get("position")
        if (
            isinstance(round_index, bool)
            or not isinstance(round_index, int)
            or round_index not in by_round
        ):
            raise ValueError(f"Round non valido: {round_index!r}")
        if (
            isinstance(position, bool)
            or not isinstance(position, int)
            or position not in expected_positions
        ):
            raise ValueError(f"Posizione non valida: {position!r}")
        slot = (round_index, position)
        if slot in seen_slots:
            raise ValueError(f"Slot del profilo duplicato: {slot}")
        seen_slots.add(slot)
        by_round[round_index].append(run)

        if run.get("returncode", 0) != 0 or run.get("status", "ok") != "ok":
            raise ValueError(f"Run fallito per {variant}, round {round_index}")
        run_configuration = run.get("configuration")
        if not isinstance(run_configuration, Mapping):
            raise ValueError("Ogni run deve includere configuration")
        current_configuration_identity = _canonical(run_configuration)
        if configuration_identity is None:
            configuration_identity = current_configuration_identity
            configuration = run_configuration
        elif current_configuration_identity != configuration_identity:
            raise ValueError("Configurazione incoerente tra i run")

        verified_prediction = _verify_prediction_reference(
            run, suite_id=suite_id, suite_root=suite_root
        )
        verified_path = str(verified_prediction["path"])
        if verified_path in prediction_paths:
            raise ValueError(
                f"File di predizione riutilizzato da piu run: {verified_path}"
            )
        prediction_paths.add(verified_path)
        prediction_integrity[str(variant)].append(verified_prediction)

        digest = _normalize_sha256(run.get("prediction_digest"), "prediction_digest")
        previous_digest = digests.setdefault(str(variant), digest)
        if previous_digest != digest:
            raise ValueError(f"Prediction digest incoerente per {variant}")

        integrity = run.get("artifact_integrity")
        identity = _artifact_hash_identity(integrity)
        previous_identity = hash_identities.setdefault(str(variant), identity)
        if previous_identity != identity:
            raise ValueError(f"Hash artefatto incoerente per {variant}")
        artifact_integrity.setdefault(str(variant), integrity)

    for round_index, round_runs in by_round.items():
        round_variants = {str(item["variant"]) for item in round_runs}
        round_positions = {int(item["position"]) for item in round_runs}
        if round_variants != expected_variants or round_positions != expected_positions:
            raise ValueError(f"Round {round_index} incompleto o duplicato")

    sorted_runs = sorted(runs, key=lambda item: (int(item["round"]), int(item["position"])))
    optional_metrics = {
        "peak_pss_bytes",
        "peak_uss_bytes",
        "peak_anonymous_bytes",
    }
    variant_reports: dict[str, Any] = {}
    for variant in names:
        variant_runs = [item for item in sorted_runs if item["variant"] == variant]
        metric_summaries = {
            metric: summarize(
                [
                    _metric_value(
                        item,
                        metric,
                        optional=metric in optional_metrics,
                    )
                    for item in variant_runs
                ]
            )
            for metric in PROFILE_METRICS
        }
        variant_reports[variant] = {
            "runs": len(variant_runs),
            "prediction_digest": digests[variant],
            "prediction_files": prediction_integrity[variant],
            "artifact_integrity": artifact_integrity[variant],
            "statistics": metric_summaries,
        }

    return {
        "schema_version": 2,
        "kind": "cpu-service-profile",
        "suite_id": suite_id,
        "suite_root": str(suite_root),
        "configuration": dict(configuration or {}),
        "methodology": dict(methodology),
        "variants_order": names,
        "repeats": repeats,
        "validation": {
            "expected_runs": len(names) * repeats,
            "actual_runs": len(runs),
            "configuration_consistent": True,
            "prediction_digests_consistent": True,
            "prediction_files_verified": True,
            "prediction_files_suite_bound": True,
            "artifact_hashes_consistent": True,
        },
        "variants": variant_reports,
        "runs": [dict(item) for item in sorted_runs],
    }


def _format_number(value: int | float | None, digits: int = 2) -> str:
    return "n/d" if value is None else f"{float(value):.{digits}f}"


def _format_mib(value: int | float | None) -> str:
    return "n/d" if value is None else f"{float(value) / 1024 / 1024:.1f}"


def _memory_triplet(summary: Mapping[str, Any]) -> str:
    return "/".join(
        _format_mib(summary.get(key)) for key in ("median", "p95", "max")
    )


def _latency_pair(summary: Mapping[str, Any]) -> str:
    return "/".join(
        "n/d" if summary.get(key) is None else f"{float(summary[key]) * 1000:.1f}"
        for key in ("median", "max")
    )


def _render_profile_markdown(report: Mapping[str, Any]) -> str:
    methodology = report.get("methodology", {})
    rows = [
        "# CPU service profile",
        "",
        (
            f"Processi freschi: **{report.get('repeats', 0)} repeat per variante**. "
            f"Ordine: `{methodology.get('order', 'n/d')}`; "
            f"seed: `{methodology.get('seed', 'n/d')}`; "
            f"page cache: `{methodology.get('cache_policy', 'uncontrolled-os-page-cache')}`."
        ),
        "",
        "Le terne di memoria sono mediana/p95/massimo. Le coppie di latenza sono mediana/massimo del p95 di ciascun processo.",
        "",
        "| Variante | Run | load s med/p95/max | docs/s p05/med | request p95 ms med/max | model p95 ms med/max | RSS MiB med/p95/max | PSS MiB med/p95/max | USS MiB med/p95/max | anon MiB med/p95/max |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in report.get("variants_order", []):
        item = report["variants"][variant]
        stats = item["statistics"]
        load = stats["load_seconds"]
        throughput = stats["documents_per_second"]
        rows.append(
            "| {variant} | {runs} | {load} | {dps} | {request} | {model} | {rss} | {pss} | {uss} | {anon} |".format(
                variant=variant,
                runs=item["runs"],
                load="/".join(
                    _format_number(load.get(key)) for key in ("median", "p95", "max")
                ),
                dps="/".join(
                    _format_number(throughput.get(key)) for key in ("p05", "median")
                ),
                request=_latency_pair(stats["request_latency_p95_seconds"]),
                model=_latency_pair(stats["model_latency_p95_seconds"]),
                rss=_memory_triplet(stats["peak_rss_bytes"]),
                pss=_memory_triplet(stats["peak_pss_bytes"]),
                uss=_memory_triplet(stats["peak_uss_bytes"]),
                anon=_memory_triplet(stats["peak_anonymous_bytes"]),
            )
        )
    cache_policy = str(
        methodology.get("cache_policy", "uncontrolled-os-page-cache")
    )
    cache_description = (
        "profilo warm-cache controllato"
        if cache_policy == "preflight-verified-warm-os-page-cache"
        else "politica di page cache dichiarata dal report"
    )
    rows.extend(
        [
            "",
            f"> `{cache_policy}` descrive un {cache_description}, non un cold-start riproducibile.",
            "",
        ]
    )
    return "\n".join(rows)


def write_profile_report(
    report: Mapping[str, Any], output_dir: Path
) -> dict[str, Path]:
    """Write machine-readable and Markdown aggregate reports."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    json_path = destination / "report.json"
    markdown_path = destination / "report.md"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    markdown_path.write_text(_render_profile_markdown(report), encoding="utf-8")
    return {"json": json_path, "markdown": markdown_path}
