# -*- coding: utf-8 -*-
"""Create and validate CPU-oriented INT8 and INT4 ONNX artifacts.

INT8 uses ONNX Runtime dynamic, signed, per-channel weight quantization for
MatMul/Gemm.  Low-bit weight-only QOperators are reported together with their
``accuracy_level`` because ONNX Runtime can dynamically quantize MatMul inputs
to INT8 even when the stored weights alone are 4 or 8 bit.

All ONNX/ORT imports are lazy so importing this module remains inexpensive.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import logging
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence


DEFAULT_VALIDATION_SHAPES: tuple[tuple[int, int], ...] = ((1, 8), (2, 32))
HASH_CHUNK_BYTES = 1024 * 1024
FP8_SKIPPED: dict[str, str] = {
    "status": "skipped",
    "reason": (
        "ONNX Runtime CPUExecutionProvider non offre un percorso FP8 "
        "general-purpose per questo modello; produrre un artefatto FP8 non "
        "eseguibile nativamente su CPU falserebbe il confronto."
    ),
}


class QuantizationError(RuntimeError):
    """Raised when an ONNX artifact cannot be quantized or validated."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_model(path: str | Path) -> Path:
    model_path = Path(path).expanduser().resolve()
    if not model_path.is_file():
        raise QuantizationError(f"Modello ONNX FP32 non trovato: {model_path}")
    if model_path.suffix.lower() != ".onnx":
        raise QuantizationError(f"Il modello sorgente deve essere .onnx: {model_path}")
    return model_path


def _clear_generated_model(path: Path) -> None:
    """Remove only files belonging to a previous generated ONNX artifact.

    ONNX Runtime opens external-data files in append mode in some releases.  A
    rerun without this cleanup would therefore report an artificially doubled
    artifact even though the graph references only the newest byte range.
    """
    for candidate in path.parent.glob(path.name + "*"):
        if candidate.is_file():
            candidate.unlink()


def _ort_version() -> str:
    try:
        return importlib.metadata.version("onnxruntime")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _graph_details(path: Path) -> tuple[Any, Counter[str], dict[str, int]]:
    try:
        import onnx
    except ImportError as exc:
        raise QuantizationError("Installare onnx per ispezionare gli artefatti") from exc
    try:
        model = onnx.load_model(str(path), load_external_data=False)
    except Exception as exc:
        raise QuantizationError(f"Impossibile leggere il grafo ONNX {path}: {exc}") from exc
    op_counts = Counter(node.op_type for node in model.graph.node)
    initializers = {tensor.name: int(tensor.data_type) for tensor in model.graph.initializer}
    return model, op_counts, initializers


def _eligible_counts(path: Path) -> dict[str, int]:
    model, _, initializers = _graph_details(path)
    eligible = Counter[str]()
    # TensorProto.FLOAT=1 and TensorProto.FLOAT16=10.  ORT's weight-only
    # quantizers intentionally skip integer shape/index constants.
    floating_weight_types = {1, 10}
    for node in model.graph.node:
        if node.op_type in {"MatMul", "Gemm"} and len(node.input) > 1:
            if initializers.get(node.input[1]) in floating_weight_types:
                eligible[node.op_type] += 1
        elif node.op_type == "Gather" and node.input:
            if initializers.get(node.input[0]) in floating_weight_types:
                eligible[node.op_type] += 1
    return dict(sorted(eligible.items()))


def _external_locations(model: Any) -> set[str]:
    locations: set[str] = set()
    for tensor in model.graph.initializer:
        for item in tensor.external_data:
            if item.key == "location" and item.value:
                locations.add(item.value)
    return locations


def _nbits_operator_configs(path: Path) -> dict[str, Any]:
    """Summarize stored bits/accuracy attributes for auditable reports."""
    model, _, _ = _graph_details(path)
    matmul: Counter[tuple[int, int]] = Counter()
    gathers = 0
    for node in model.graph.node:
        attributes = {attribute.name: int(attribute.i) for attribute in node.attribute}
        if node.op_type == "MatMulNBits":
            matmul[(attributes.get("bits", 4), attributes.get("accuracy_level", 0))] += 1
        elif node.op_type == "GatherBlockQuantized":
            # The ORT contrib operator is a fixed 4-bit Gather in ORT 1.29 and
            # consequently has no configurable ``bits`` attribute.
            gathers += 1
    return {
        "MatMulNBits": [
            {"bits": bits, "accuracy_level": level, "nodes": count}
            for (bits, level), count in sorted(matmul.items())
        ],
        "GatherBlockQuantized": {"bits": 4, "nodes": gathers},
    }


def artifact_info(path: str | Path) -> dict[str, Any]:
    """Return graph coverage inputs and the real ONNX+external-data size."""
    model_path = Path(path).expanduser().resolve()
    model, op_counts, _ = _graph_details(model_path)
    files = {model_path}
    for location in _external_locations(model):
        candidate = (model_path.parent / location).resolve()
        try:
            candidate.relative_to(model_path.parent.resolve())
        except ValueError as exc:
            raise QuantizationError(
                f"External data ONNX fuori dalla directory dell'artefatto: {location}"
            ) from exc
        if not candidate.is_file():
            raise QuantizationError(f"External data ONNX mancante: {candidate}")
        files.add(candidate)
    file_rows = [
        {"path": str(file), "bytes": file.stat().st_size, "sha256": _sha256(file)}
        for file in sorted(files, key=lambda item: str(item))
    ]
    return {
        "model": str(model_path),
        "bytes": sum(row["bytes"] for row in file_rows),
        "files": file_rows,
        "op_counts": dict(sorted(op_counts.items())),
    }


def _validate_cpu(
    path: Path, validation_shapes: Iterable[Sequence[int]]
) -> list[dict[str, Any]]:
    try:
        from .export import ExportError, validate_onnx_cpu
    except ImportError:  # Direct execution: python src/quantization/quantize.py
        from export import ExportError, validate_onnx_cpu  # type: ignore[no-redef]

    try:
        return validate_onnx_cpu(path, validation_shapes)
    except ExportError as exc:
        raise QuantizationError(str(exc)) from exc


def _coverage(
    *, source: Path, result: Path, precision: str
) -> dict[str, Any]:
    eligible = _eligible_counts(source)
    _, result_ops, _ = _graph_details(result)
    if precision == "int8":
        quantized_by_type = {
            "MatMulInteger": result_ops.get("MatMulInteger", 0),
            "QLinearMatMul": result_ops.get("QLinearMatMul", 0),
            "QGemm": result_ops.get("QGemm", 0),
        }
        quantized_total = sum(quantized_by_type.values())
        eligible_total = eligible.get("MatMul", 0) + eligible.get("Gemm", 0)
        targets = ["MatMul", "Gemm"]
    elif precision in {"int2", "int4", "int8-weight-only"}:
        quantized_by_type = {
            "MatMulNBits": result_ops.get("MatMulNBits", 0),
            "GatherBlockQuantized": result_ops.get("GatherBlockQuantized", 0),
        }
        quantized_total = sum(quantized_by_type.values())
        targets = (
            ["MatMul", "Gather"]
            if precision in {"int2", "int4"}
            else ["MatMul"]
        )
        eligible_total = sum(eligible.get(target, 0) for target in targets)
    else:
        raise ValueError(f"Precisione non gestita: {precision}")

    # A valid graph with no converted operators is almost always a configuration
    # error.  Fail early instead of publishing an FP32 file labelled as quantized.
    if eligible_total and quantized_total == 0:
        raise QuantizationError(
            f"Il grafo {precision.upper()} non contiene operatori quantizzati; "
            f"operatori sorgente eleggibili: {eligible}"
        )
    return {
        "targets": targets,
        "eligible_source": eligible,
        "eligible_target_total": eligible_total,
        "quantized_result": quantized_by_type,
        "quantized_total": quantized_total,
        "coverage_ratio": (
            min(1.0, quantized_total / eligible_total) if eligible_total else None
        ),
        "remaining_fp_operators": {
            target: result_ops.get(target, 0) for target in targets
        },
    }


def quantize_int8(
    fp32_model: str | Path,
    output_model: str | Path,
    *,
    validation_shapes: Iterable[Sequence[int]] = DEFAULT_VALIDATION_SHAPES,
) -> dict[str, Any]:
    """Create dynamic per-channel signed INT8 weights for MatMul/Gemm."""
    source = _source_model(fp32_model)
    destination = Path(output_model).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    _clear_generated_model(destination)
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic
    except ImportError as exc:
        raise QuantizationError(
            "Quantizzazione INT8 non disponibile: installare onnxruntime"
        ) from exc

    try:
        quantize_dynamic(
            model_input=str(source),
            model_output=str(destination),
            per_channel=True,
            weight_type=QuantType.QInt8,
            op_types_to_quantize=["MatMul", "Gemm"],
            use_external_data_format=True,
            extra_options={"MatMulConstBOnly": True},
        )
    except Exception as exc:
        raise QuantizationError(f"Quantizzazione dinamica INT8 fallita: {exc}") from exc
    if not destination.is_file():
        raise QuantizationError(f"La quantizzazione INT8 non ha creato {destination}")

    coverage = _coverage(source=source, result=destination, precision="int8")
    validation = _validate_cpu(destination, validation_shapes)
    return {
        "status": "ok",
        "precision": "int8",
        "algorithm": "ONNX Runtime dynamic per-channel QInt8",
        "weight_type": "QInt8",
        "activation_quantization": "dynamic",
        "onnxruntime_version": _ort_version(),
        "source": artifact_info(source),
        "artifact": artifact_info(destination),
        "coverage": coverage,
        "cpu_validation": validation,
    }


def _nbits_quantizer(
    source: Path,
    destination: Path,
    block_size: int,
    accuracy_level: int,
    bits: int,
    op_types: tuple[str, ...] = ("MatMul", "Gather"),
) -> str:
    """Run the current ORT n-bit API, falling back to its pre-rename module."""
    try:
        from onnxruntime.quantization import matmul_nbits_quantizer, quant_utils

        config = matmul_nbits_quantizer.DefaultWeightOnlyQuantConfig(
            block_size=block_size,
            is_symmetric=True,
            accuracy_level=accuracy_level,
            quant_format=quant_utils.QuantFormat.QOperator,
            op_types_to_quantize=op_types,
            quant_axes=tuple(
                (op_type, 0 if op_type == "MatMul" else 1)
                for op_type in op_types
            ),
            bits=bits,
        )
        quantizer = matmul_nbits_quantizer.MatMulNBitsQuantizer(
            model=str(source), algo_config=config
        )
        logging.getLogger("onnxruntime.quantization.matmul_nbits_quantizer").setLevel(
            logging.WARNING
        )
        api_name = "matmul_nbits_quantizer.MatMulNBitsQuantizer"
    except (ImportError, AttributeError):
        if bits != 4:
            raise QuantizationError(
                f"La release ONNX Runtime installata non espone MatMulNBits per {bits} bit"
            )
        try:
            from onnxruntime.quantization import (
                matmul_4bits_quantizer,
                quant_utils,
            )

            config = matmul_4bits_quantizer.DefaultWeightOnlyQuantConfig(
                block_size=block_size,
                is_symmetric=True,
                accuracy_level=accuracy_level,
                quant_format=quant_utils.QuantFormat.QOperator,
                op_types_to_quantize=op_types,
                quant_axes=tuple(
                    (op_type, 0 if op_type == "MatMul" else 1)
                    for op_type in op_types
                ),
            )
            model = quant_utils.load_model_with_shape_infer(source)
            quantizer = matmul_4bits_quantizer.MatMul4BitsQuantizer(
                model=model, nodes_to_exclude=None, nodes_to_include=None, algo_config=config
            )
            api_name = "matmul_4bits_quantizer.MatMul4BitsQuantizer"
        except (ImportError, AttributeError) as exc:
            raise QuantizationError(
                "API INT4 di ONNX Runtime non disponibile. Serve una release ORT "
                "con MatMulNBits e GatherBlockQuantized (ORT >= 1.20 per Gather)."
            ) from exc

    try:
        quantizer.process()
        quantizer.model.save_model_to_file(str(destination), True)
    except Exception as exc:
        raise QuantizationError(
            f"Quantizzazione DEFAULT INT{bits} fallita: {exc}"
        ) from exc
    return api_name


def _int8_weight_only_quantizer(source: Path, destination: Path, block_size: int) -> str:
    """Use MatMulNBits with 8-bit weights and dynamic INT8 MatMul inputs."""
    try:
        from onnxruntime.quantization import matmul_nbits_quantizer, quant_utils
    except (ImportError, AttributeError) as exc:
        raise QuantizationError(
            "API weight-only INT8 non disponibile: serve MatMulNBitsQuantizer"
        ) from exc

    try:
        config = matmul_nbits_quantizer.DefaultWeightOnlyQuantConfig(
            block_size=block_size,
            is_symmetric=True,
            accuracy_level=4,
            quant_format=quant_utils.QuantFormat.QOperator,
            op_types_to_quantize=("MatMul",),
            quant_axes=(("MatMul", 0),),
            bits=8,
        )
        quantizer = matmul_nbits_quantizer.MatMulNBitsQuantizer(
            model=str(source), algo_config=config
        )
        logging.getLogger("onnxruntime.quantization.matmul_nbits_quantizer").setLevel(
            logging.WARNING
        )
        quantizer.process()
        quantizer.model.save_model_to_file(str(destination), True)
    except Exception as exc:
        raise QuantizationError(f"Quantizzazione weight-only INT8 fallita: {exc}") from exc
    return "matmul_nbits_quantizer.MatMulNBitsQuantizer"


def quantize_int8_weight_only(
    fp32_model: str | Path,
    output_model: str | Path,
    *,
    block_size: int = 128,
    validation_shapes: Iterable[Sequence[int]] = DEFAULT_VALIDATION_SHAPES,
) -> dict[str, Any]:
    """Create symmetric W8A8 MatMulNBits operators (dynamic input quantization)."""
    if block_size < 16 or block_size & (block_size - 1):
        raise ValueError("block_size INT8 deve essere una potenza di due >= 16")
    source = _source_model(fp32_model)
    destination = Path(output_model).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    _clear_generated_model(destination)
    api_name = _int8_weight_only_quantizer(source, destination, block_size)
    if not destination.is_file():
        raise QuantizationError(f"La quantizzazione weight-only INT8 non ha creato {destination}")
    coverage = _coverage(source=source, result=destination, precision="int8-weight-only")
    validation = _validate_cpu(destination, validation_shapes)
    return {
        "status": "ok",
        "precision": "int8",
        "algorithm": "ONNX Runtime DEFAULT symmetric affine weight-only",
        "algorithm_id": "DEFAULT",
        "quant_format": "QOperator",
        "block_size": block_size,
        "op_types": ["MatMul"],
        "accuracy_level": 4,
        "activation_quantization": "dynamic INT8 inside MatMulNBits",
        "compute_path": "W8A8 MatMul; non-MatMul activations remain FP32",
        "api": api_name,
        "onnxruntime_version": _ort_version(),
        "source": artifact_info(source),
        "artifact": artifact_info(destination),
        "coverage": coverage,
        "cpu_validation": validation,
    }


def quantize_int4(
    fp32_model: str | Path,
    output_model: str | Path,
    *,
    block_size: int = 128,
    accuracy_level: int = 4,
    validation_shapes: Iterable[Sequence[int]] = DEFAULT_VALIDATION_SHAPES,
) -> dict[str, Any]:
    """Create symmetric DEFAULT affine INT4 QOperators for MatMul/Gather weights.

    ``accuracy_level=1`` keeps MatMul inputs FP32 (W4A32).  Level 4 asks the
    CPU kernel to dynamically quantize them to INT8 (W4A8).  Gather always
    returns floating-point embeddings to the rest of the graph.
    """
    if block_size < 16 or block_size & (block_size - 1):
        raise ValueError("block_size INT4 deve essere una potenza di due >= 16")
    if accuracy_level not in {1, 4}:
        raise ValueError("accuracy_level INT4 supportato: 1 (W4A32) o 4 (W4A8)")
    source = _source_model(fp32_model)
    destination = Path(output_model).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    _clear_generated_model(destination)
    api_name = _nbits_quantizer(
        source, destination, block_size, accuracy_level, bits=4
    )
    if not destination.is_file():
        raise QuantizationError(f"La quantizzazione INT4 non ha creato {destination}")

    coverage = _coverage(source=source, result=destination, precision="int4")
    validation = _validate_cpu(destination, validation_shapes)
    return {
        "status": "ok",
        "precision": "int4",
        "algorithm": "ONNX Runtime DEFAULT symmetric affine weight-only",
        "algorithm_id": "DEFAULT",
        "quant_format": "QOperator",
        "block_size": block_size,
        "accuracy_level": accuracy_level,
        "op_types": ["MatMul", "Gather"],
        "activation_quantization": (
            "none (FP32 MatMul input)"
            if accuracy_level == 1
            else "dynamic INT8 inside MatMulNBits"
        ),
        "compute_path": (
            "W4A32 MatMul; Gather dequantizes embeddings to FP32"
            if accuracy_level == 1
            else "W4A8 MatMul; Gather dequantizes embeddings to FP32"
        ),
        "api": api_name,
        "onnxruntime_version": _ort_version(),
        "source": artifact_info(source),
        "artifact": artifact_info(destination),
        "coverage": coverage,
        "operator_configs": _nbits_operator_configs(destination),
        "cpu_validation": validation,
    }


def quantize_int2(
    fp32_model: str | Path,
    output_model: str | Path,
    *,
    block_size: int = 128,
    accuracy_level: int = 4,
    validation_shapes: Iterable[Sequence[int]] = DEFAULT_VALIDATION_SHAPES,
) -> dict[str, Any]:
    """Create stock-runtime W2 MatMul plus W4 Gather operators.

    ORT 1.29 supports 2-bit ``MatMulNBits`` but only 4-bit
    ``GatherBlockQuantized``.  The mixed artifact is explicit about this rather
    than leaving the dominant embedding FP32 or pretending it is all INT2.
    """
    if block_size < 16 or block_size & (block_size - 1):
        raise ValueError("block_size INT2 deve essere una potenza di due >= 16")
    if accuracy_level not in {1, 4}:
        raise ValueError("accuracy_level INT2 supportato: 1 (W2A32) o 4 (W2A8)")
    source = _source_model(fp32_model)
    destination = Path(output_model).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    _clear_generated_model(destination)
    with tempfile.TemporaryDirectory(
        prefix="rizzo-int2-matmul-", dir=str(destination.parent)
    ) as temporary:
        intermediate = Path(temporary) / "model.onnx"
        matmul_api = _nbits_quantizer(
            source,
            intermediate,
            block_size,
            accuracy_level,
            bits=2,
            op_types=("MatMul",),
        )
        gather_api = _nbits_quantizer(
            intermediate,
            destination,
            block_size,
            accuracy_level,
            bits=4,
            op_types=("Gather",),
        )
    if not destination.is_file():
        raise QuantizationError(f"La quantizzazione INT2 non ha creato {destination}")

    coverage = _coverage(source=source, result=destination, precision="int2")
    operator_configs = _nbits_operator_configs(destination)
    matmul_configs = operator_configs["MatMulNBits"]
    if not matmul_configs or any(item["bits"] != 2 for item in matmul_configs):
        raise QuantizationError("Artefatto mixed-bit privo di MatMulNBits W2 verificabili")
    if int(operator_configs["GatherBlockQuantized"]["nodes"]) < 1:
        raise QuantizationError("Artefatto mixed-bit privo di embedding Gather W4")
    validation = _validate_cpu(destination, validation_shapes)
    return {
        "status": "ok",
        "experimental": True,
        "default_pipeline": False,
        "precision": "mixed-w2-w4",
        "algorithm": "ONNX Runtime DEFAULT symmetric affine mixed-bit",
        "algorithm_id": "DEFAULT",
        "quant_format": "QOperator",
        "block_size": block_size,
        "accuracy_level": accuracy_level,
        "stored_bits": {"MatMul": 2, "Gather_embedding": 4},
        "op_types": ["MatMul", "Gather"],
        "activation_quantization": (
            "none (FP32 MatMul input)"
            if accuracy_level == 1
            else "dynamic INT8 inside MatMulNBits"
        ),
        "compute_path": (
            "W2A32 MatMul; W4 Gather dequantizes embeddings to FP32"
            if accuracy_level == 1
            else "W2A8 MatMul; W4 Gather dequantizes embeddings to FP32"
        ),
        "api": {"MatMul_W2": matmul_api, "Gather_W4": gather_api},
        "onnxruntime_version": _ort_version(),
        "source": artifact_info(source),
        "artifact": artifact_info(destination),
        "coverage": coverage,
        "operator_configs": operator_configs,
        "cpu_validation": validation,
    }


def quantize_models(
    fp32_model: str | Path,
    output_dir: str | Path,
    *,
    block_size: int = 128,
    validation_shapes: Iterable[Sequence[int]] = DEFAULT_VALIDATION_SHAPES,
) -> dict[str, Any]:
    """Build both CPU artifacts and explicitly record FP8 as skipped."""
    source = _source_model(fp32_model)
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    # Materialise once because callers may pass a generator.
    shapes = tuple(tuple(int(dim) for dim in shape) for shape in validation_shapes)
    int8 = quantize_int8(
        source, destination / "model.int8.onnx", validation_shapes=shapes
    )
    int4 = quantize_int4(
        source,
        destination / "model.int4.onnx",
        block_size=block_size,
        validation_shapes=shapes,
    )
    report = {
        "status": "ok",
        "source_model": str(source),
        "variants": {"int8": int8, "int4": int4},
        "fp8": dict(FP8_SKIPPED),
    }
    report_path = destination / "quantization-report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report["report"] = str(report_path)
    return report


def _parse_shape(value: str) -> tuple[int, int]:
    try:
        batch, sequence = value.lower().replace("x", ",").split(",", 1)
        return int(batch), int(sequence)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("Usare BATCHxSEQUENCE, ad esempio 2x128") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Crea ONNX INT8/INT4 CPU e verifica caricamento, shape e copertura."
    )
    parser.add_argument("fp32_model", type=Path, help="modello ONNX FP32")
    parser.add_argument("output_dir", type=Path, help="directory degli artefatti")
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument(
        "--validation-shape",
        action="append",
        type=_parse_shape,
        dest="validation_shapes",
        help="shape CPU da verificare, ripetibile (default: 1x8 e 2x32)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = quantize_models(
            args.fp32_model,
            args.output_dir,
            block_size=args.block_size,
            validation_shapes=args.validation_shapes or DEFAULT_VALIDATION_SHAPES,
        )
    except (QuantizationError, ValueError) as exc:
        raise SystemExit(f"Errore quantizzazione: {exc}") from exc
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
