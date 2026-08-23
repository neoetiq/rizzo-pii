# -*- coding: utf-8 -*-
"""Quantize FP32 ONNX embedding tables to symmetric per-row INT8.

The transformation deliberately uses only standard ONNX operators so the
result remains portable to ONNX Runtime's CPUExecutionProvider::

    Gather(INT8 table, ids) -> Cast(FP32) --+
                                                Mul -> original FP32 output
    Gather(FP32 scales, ids) -------------------+

Each vocabulary row has one FP32 scale and an implicit zero-point of zero.
MatMulNBits nodes already present in a weight-only model are left untouched.
"""

from __future__ import annotations

import copy
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

from .quantize import (
    DEFAULT_VALIDATION_SHAPES,
    QuantizationError,
    _source_model,
    _validate_cpu,
    artifact_info,
)


EMBEDDING_ROW_CHUNK_SIZE = 4096


def _safe_output_paths(destination: Path) -> tuple[Path, Path]:
    """Return the exact two paths produced by this module.

    Avoiding a prefix glob is intentional: a file such as
    ``model.onnx.notes`` must never be removed as a side effect of a rerun.
    """
    return destination, destination.with_name(destination.name + ".data")


def _clear_output(destination: Path) -> None:
    for candidate in _safe_output_paths(destination):
        if candidate.is_file():
            candidate.unlink()


def _gather_axis(node: Any) -> int:
    for attribute in node.attribute:
        if attribute.name == "axis":
            return int(attribute.i)
    return 0


def _all_value_names(graph: Any) -> set[str]:
    names = {
        value.name
        for values in (graph.input, graph.output, graph.value_info, graph.initializer)
        for value in values
        if value.name
    }
    for node in graph.node:
        names.update(name for name in node.input if name)
        names.update(name for name in node.output if name)
    return names


def _unique(base: str, used: set[str]) -> str:
    candidate = base
    suffix = 2
    while candidate in used:
        candidate = f"{base}_{suffix}"
        suffix += 1
    used.add(candidate)
    return candidate


def _candidate_gathers(model: Any) -> dict[str, list[int]]:
    """Find FP32 rank-2 initializer tables used only by axis-0 Gather nodes."""
    try:
        import onnx
    except ImportError as exc:  # pragma: no cover - handled by public entrypoint
        raise QuantizationError("Installare onnx per quantizzare gli embedding") from exc

    graph = model.graph
    initializers = {tensor.name: tensor for tensor in graph.initializer}
    graph_inputs = {value.name for value in graph.input}
    uses: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for node_index, node in enumerate(graph.node):
        for input_index, name in enumerate(node.input):
            if name in initializers:
                uses[name].append((node_index, input_index))

    candidates: dict[str, list[int]] = {}
    for name, tensor in initializers.items():
        if (
            tensor.data_type != onnx.TensorProto.FLOAT
            or len(tensor.dims) != 2
            or name in graph_inputs
        ):
            continue
        consumers = uses.get(name, [])
        if not consumers:
            continue
        gather_indices: list[int] = []
        safe = True
        for node_index, input_index in consumers:
            node = graph.node[node_index]
            if (
                input_index != 0
                or node.op_type != "Gather"
                or node.domain not in {"", "ai.onnx"}
                or _gather_axis(node) != 0
                or len(node.input) < 2
                or len(node.output) != 1
                or not node.output[0]
            ):
                safe = False
                break
            gather_indices.append(node_index)
        if safe:
            candidates[name] = gather_indices
    return candidates


def _quantize_rows(weight: Any) -> tuple[Any, Any, dict[str, float | int]]:
    import numpy as np

    values = np.asarray(weight, dtype=np.float32)
    if values.ndim != 2:
        raise QuantizationError(
            f"Embedding con rank {values.ndim}: attesa matrice [vocab, hidden]"
        )
    if values.shape[0] < 1 or values.shape[1] < 1:
        raise QuantizationError("L'embedding FP32 deve avere vocab e hidden size positivi")

    # The real embedding is hundreds of MiB.  Keep temporary arrays bounded to
    # one row chunk instead of materialising full-size abs/division/restored/error
    # tensors alongside the FP32 input and INT8 output.
    quantized = np.empty(values.shape, dtype=np.int8)
    scales = np.empty((values.shape[0], 1), dtype=np.float32)
    max_abs_error = 0.0
    absolute_error_sum = 0.0
    zero_rows = 0
    chunk_size = max(1, int(EMBEDDING_ROW_CHUNK_SIZE))
    for start in range(0, values.shape[0], chunk_size):
        stop = min(start + chunk_size, values.shape[0])
        source_chunk = values[start:stop]

        # One reusable FP32 buffer is sufficient for row maxima, rounding and
        # reconstruction error.  The row-sized maxima/masks are negligible.
        work = np.abs(source_chunk)
        row_max = np.max(work, axis=1, keepdims=True)
        if not np.isfinite(row_max).all():
            raise QuantizationError("L'embedding FP32 contiene valori NaN o infiniti")

        chunk_scales = scales[start:stop]
        np.divide(row_max, 127.0, out=chunk_scales)
        zero_mask = row_max == 0.0
        zero_rows += int(np.count_nonzero(zero_mask))
        chunk_scales[zero_mask] = 1.0

        np.divide(source_chunk, chunk_scales, out=work)
        np.rint(work, out=work)
        np.clip(work, -127.0, 127.0, out=work)
        quantized[start:stop] = work

        np.multiply(quantized[start:stop], chunk_scales, out=work)
        np.subtract(source_chunk, work, out=work)
        np.abs(work, out=work)
        max_abs_error = max(max_abs_error, float(np.max(work)))
        absolute_error_sum += float(np.sum(work, dtype=np.float64))

    original_bytes = int(values.nbytes)
    quantized_bytes = int(quantized.nbytes + scales.nbytes)
    statistics: dict[str, float | int] = {
        "rows": int(values.shape[0]),
        "hidden_size": int(values.shape[1]),
        "original_weight_bytes": original_bytes,
        "quantized_weight_bytes": quantized_bytes,
        "weight_size_ratio": quantized_bytes / original_bytes if original_bytes else 0.0,
        "max_abs_error": max_abs_error,
        "mean_abs_error": absolute_error_sum / int(values.size),
        "zero_rows": zero_rows,
    }
    return quantized, scales, statistics


def _clone_gather(node: Any, *, table: str, output: str, name: str) -> Any:
    clone = copy.deepcopy(node)
    clone.input[0] = table
    clone.output[0] = output
    clone.name = name
    return clone


def _transform(
    model: Any,
    *,
    initializer_name: str | None,
    source_directory: Path,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    try:
        import numpy as np
        import onnx
        from onnx import helper, numpy_helper
    except ImportError as exc:
        raise QuantizationError(
            "Quantizzazione embedding INT8 non disponibile: installare onnx e numpy"
        ) from exc

    graph = model.graph
    candidates = _candidate_gathers(model)
    eligible_gathers = sum(len(indices) for indices in candidates.values())
    if initializer_name is not None:
        if initializer_name not in candidates:
            available = ", ".join(sorted(candidates)) or "nessuno"
            raise QuantizationError(
                f"Initializer embedding non eleggibile: {initializer_name!r}; "
                f"candidati: {available}"
            )
        selected = {initializer_name: candidates[initializer_name]}
    else:
        selected = candidates
    if not selected:
        raise QuantizationError(
            "Nessun Gather axis=0 su initializer FP32 [vocab, hidden] eleggibile"
        )

    used_values = _all_value_names(graph)
    used_node_names = {node.name for node in graph.node if node.name}
    initializers = {tensor.name: tensor for tensor in graph.initializer}
    replacements: dict[int, list[Any]] = {}
    new_initializers: dict[str, tuple[Any, Any]] = {}
    tensor_reports: list[dict[str, Any]] = []

    for weight_name, gather_indices in selected.items():
        tensor = initializers[weight_name]
        try:
            values = numpy_helper.to_array(tensor, base_dir=str(source_directory))
        except Exception as exc:
            raise QuantizationError(
                f"Impossibile leggere l'external data di {weight_name!r}: {exc}"
            ) from exc
        quantized, scales, statistics = _quantize_rows(values)
        quantized_name = _unique(weight_name + "__rowwise_int8", used_values)
        scales_name = _unique(weight_name + "__rowwise_scale", used_values)
        quantized_tensor = numpy_helper.from_array(
            np.ascontiguousarray(quantized), name=quantized_name
        )
        scales_tensor = numpy_helper.from_array(
            np.ascontiguousarray(scales), name=scales_name
        )
        new_initializers[weight_name] = (quantized_tensor, scales_tensor)

        gather_reports: list[dict[str, str]] = []
        for node_index in gather_indices:
            original = graph.node[node_index]
            output_name = original.output[0]
            quantized_output = _unique(output_name + "__int8_values", used_values)
            scale_output = _unique(output_name + "__row_scales", used_values)
            cast_output = _unique(output_name + "__dequantized", used_values)
            base_node_name = original.name or f"Gather_{node_index}"
            values_node_name = _unique(base_node_name + "__int8_values", used_node_names)
            scales_node_name = _unique(base_node_name + "__row_scales", used_node_names)
            cast_node_name = _unique(base_node_name + "__cast_fp32", used_node_names)
            mul_node_name = _unique(base_node_name + "__mul_scale", used_node_names)

            values_gather = _clone_gather(
                original,
                table=quantized_name,
                output=quantized_output,
                name=values_node_name,
            )
            scales_gather = _clone_gather(
                original,
                table=scales_name,
                output=scale_output,
                name=scales_node_name,
            )
            cast = helper.make_node(
                "Cast",
                [quantized_output],
                [cast_output],
                name=cast_node_name,
                to=onnx.TensorProto.FLOAT,
            )
            multiply = helper.make_node(
                "Mul",
                [cast_output, scale_output],
                [output_name],
                name=mul_node_name,
            )
            replacements[node_index] = [values_gather, scales_gather, cast, multiply]
            gather_reports.append(
                {
                    "original_node": original.name,
                    "output": output_name,
                    "quantized_gather": values_node_name,
                    "scale_gather": scales_node_name,
                    "cast": cast_node_name,
                    "multiply": mul_node_name,
                    "quantized_output": quantized_output,
                    "scale_output": scale_output,
                    "cast_output": cast_output,
                }
            )

        tensor_reports.append(
            {
                "original_initializer": weight_name,
                "quantized_initializer": quantized_name,
                "scale_initializer": scales_name,
                **statistics,
                "gathers": gather_reports,
            }
        )

    rewritten_nodes: list[Any] = []
    for node_index, node in enumerate(graph.node):
        rewritten_nodes.extend(replacements.get(node_index, [node]))
    del graph.node[:]
    graph.node.extend(rewritten_nodes)

    rewritten_initializers: list[Any] = []
    for tensor in graph.initializer:
        replacement = new_initializers.get(tensor.name)
        if replacement is None:
            rewritten_initializers.append(tensor)
        else:
            rewritten_initializers.extend(replacement)
    del graph.initializer[:]
    graph.initializer.extend(rewritten_initializers)

    coverage = {
        "eligible_embedding_initializers": len(candidates),
        "selected_embedding_initializers": len(selected),
        "eligible_gathers": eligible_gathers,
        "quantized_gathers": sum(len(indices) for indices in selected.values()),
    }
    return tensor_reports, coverage


def _validate_structure(
    path: Path, tensor_reports: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    try:
        import onnx
    except ImportError as exc:  # pragma: no cover - public entrypoint imports it
        raise QuantizationError("Installare onnx per validare l'artefatto") from exc

    try:
        onnx.checker.check_model(str(path))
        model = onnx.load_model(str(path), load_external_data=False)
    except Exception as exc:
        raise QuantizationError(f"Validazione strutturale ONNX fallita: {exc}") from exc

    initializers = {tensor.name: tensor for tensor in model.graph.initializer}
    nodes_by_name = {node.name: node for node in model.graph.node if node.name}
    for row in tensor_reports:
        original_name = str(row["original_initializer"])
        quantized_name = str(row["quantized_initializer"])
        scale_name = str(row["scale_initializer"])
        if original_name in initializers:
            raise QuantizationError(
                f"Initializer FP32 originale ancora presente: {original_name}"
            )
        quantized = initializers.get(quantized_name)
        scales = initializers.get(scale_name)
        if quantized is None or quantized.data_type != onnx.TensorProto.INT8:
            raise QuantizationError(f"Initializer INT8 non valido: {quantized_name}")
        if scales is None or scales.data_type != onnx.TensorProto.FLOAT:
            raise QuantizationError(f"Initializer scale FP32 non valido: {scale_name}")
        if list(quantized.dims) != [row["rows"], row["hidden_size"]]:
            raise QuantizationError(f"Shape INT8 non valida per {quantized_name}")
        if list(scales.dims) != [row["rows"], 1]:
            raise QuantizationError(f"Shape scale non valida per {scale_name}")

        for gather in row["gathers"]:
            values_gather = nodes_by_name.get(gather["quantized_gather"])
            scales_gather = nodes_by_name.get(gather["scale_gather"])
            cast = nodes_by_name.get(gather["cast"])
            multiply = nodes_by_name.get(gather["multiply"])
            expected_nodes = (
                (values_gather, gather["quantized_gather"], "Gather"),
                (scales_gather, gather["scale_gather"], "Gather"),
                (cast, gather["cast"], "Cast"),
                (multiply, gather["multiply"], "Mul"),
            )
            for node, name, op_type in expected_nodes:
                if node is None or node.op_type != op_type:
                    raise QuantizationError(
                        f"Nodo di dequantizzazione mancante/non valido: {name} ({op_type})"
                    )
            assert values_gather is not None
            assert scales_gather is not None
            assert cast is not None
            assert multiply is not None
            wiring_is_valid = (
                list(values_gather.input[:1]) == [quantized_name]
                and list(values_gather.output) == [gather["quantized_output"]]
                and list(scales_gather.input[:1]) == [scale_name]
                and list(scales_gather.output) == [gather["scale_output"]]
                and list(cast.input) == [gather["quantized_output"]]
                and list(cast.output) == [gather["cast_output"]]
                and list(multiply.input)
                == [gather["cast_output"], gather["scale_output"]]
                and list(multiply.output) == [gather["output"]]
            )
            cast_to_float = any(
                attribute.name == "to"
                and int(attribute.i) == onnx.TensorProto.FLOAT
                for attribute in cast.attribute
            )
            if not wiring_is_valid or not cast_to_float:
                raise QuantizationError(
                    f"Cablaggio dequantizzazione non valido per {gather['output']}"
                )

    op_counts = Counter(node.op_type for node in model.graph.node)
    return {
        "onnx_checker": "ok",
        "op_counts": dict(sorted(op_counts.items())),
        "original_fp32_initializers_removed": True,
    }


def quantize_embedding_int8(
    input_model: str | Path,
    output_model: str | Path,
    *,
    initializer_name: str | None = None,
    validation_shapes: Iterable[Sequence[int]] = DEFAULT_VALIDATION_SHAPES,
) -> dict[str, Any]:
    """Compress FP32 embedding Gather weights in an ONNX CPU model.

    ``input_model`` may already contain external data and custom MatMulNBits
    nodes.  The entire model is loaded and re-saved to one destination external
    data file, which also prevents stale offsets or references to the source
    artifact.  In-place conversion is rejected before any cleanup.
    """
    source = _source_model(input_model)
    destination = Path(output_model).expanduser().resolve()
    if destination == source:
        raise QuantizationError("Input e output devono essere file distinti")
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_report = artifact_info(source)
    source_files = {Path(row["path"]).resolve() for row in source_report["files"]}
    collisions = source_files.intersection(_safe_output_paths(destination))
    if collisions:
        paths = ", ".join(str(path) for path in sorted(collisions))
        raise QuantizationError(f"L'output sovrascriverebbe external data sorgente: {paths}")

    try:
        import onnx
    except ImportError as exc:
        raise QuantizationError(
            "Quantizzazione embedding INT8 non disponibile: installare onnx"
        ) from exc

    try:
        model = onnx.load_model(str(source), load_external_data=True)
    except Exception as exc:
        raise QuantizationError(f"Impossibile caricare il modello ONNX {source}: {exc}") from exc

    tensor_reports, coverage = _transform(
        model,
        initializer_name=initializer_name,
        source_directory=source.parent,
    )
    quantized_gathers = coverage["quantized_gathers"]
    coverage["coverage_ratio"] = (
        quantized_gathers / coverage["eligible_gathers"]
        if coverage["eligible_gathers"]
        else None
    )

    external_path = destination.with_name(destination.name + ".data")
    _clear_output(destination)
    try:
        onnx.save_model(
            model,
            str(destination),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=external_path.name,
            size_threshold=0,
            convert_attribute=False,
        )
        if not destination.is_file() or not external_path.is_file():
            raise QuantizationError(
                f"Salvataggio incompleto: attesi {destination} e {external_path}"
            )
        structure = _validate_structure(destination, tensor_reports)
        cpu_validation = _validate_cpu(destination, validation_shapes)
        output_report = artifact_info(destination)
    except Exception as exc:
        _clear_output(destination)
        if isinstance(exc, QuantizationError):
            raise
        raise QuantizationError(f"Creazione embedding INT8 fallita: {exc}") from exc

    original_weight_bytes = sum(
        int(row["original_weight_bytes"]) for row in tensor_reports
    )
    quantized_weight_bytes = sum(
        int(row["quantized_weight_bytes"]) for row in tensor_reports
    )
    return {
        "status": "ok",
        "precision": "int8",
        "algorithm": "symmetric per-row embedding weight quantization",
        "zero_point": 0,
        "activation_quantization": "none (dequantized to FP32 after Gather)",
        "operators": ["Gather", "Cast", "Mul"],
        "source": source_report,
        "artifact": output_report,
        "coverage": coverage,
        "embedding_tensors": tensor_reports,
        "embedding_storage": {
            "original_weight_bytes": original_weight_bytes,
            "quantized_weight_bytes": quantized_weight_bytes,
            "bytes_saved": original_weight_bytes - quantized_weight_bytes,
            "size_ratio": (
                quantized_weight_bytes / original_weight_bytes
                if original_weight_bytes
                else None
            ),
        },
        "structural_validation": structure,
        "cpu_validation": cpu_validation,
    }


__all__ = ["quantize_embedding_int8"]
