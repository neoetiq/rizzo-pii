# -*- coding: utf-8 -*-
"""Honest CPU probes for PyTorch/ATen and ONNX Runtime low-bit paths.

This module deliberately separates three different claims which are often
collapsed into the word "INT4":

* packed weight *storage*;
* the dtype used by activations and accumulators;
* the kernel which actually executes the matrix multiplication.

The PyTorch linear path below uses the stock private ATen CPU W4A32 operator
(``aten::_weight_int4pack_mm_for_cpu``).  It is a useful capability and kernel
probe, not a stable public deployment API.  Embeddings are genuinely packed
two values per byte, but selected rows are unpacked and dequantized with normal
PyTorch tensor operations: there is no claim of a fused INT4 embedding kernel.

The ONNX matrix changes only ``MatMulNBits.accuracy_level`` on a single packed
artifact.  Level 1 is W4A32, while level 4 dynamically quantizes activations and
uses the W4A8 path.  Hashing the initializers verifies that the two sessions use
identical packed weights, which isolates the engine/compute-path difference.

INT6 is estimate/probe-only.  A ``torch.int6`` shell dtype or six-bit values in
a uint8 tensor do not constitute packed storage or a usable CPU kernel.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import statistics
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


SUPPORTED_ATEN_GROUP_SIZES = (32, 64, 128, 256)
DEFAULT_MODERNBERT_SHAPES: tuple[tuple[int, int, int], ...] = (
    # (rows/out_features, columns/in_features, repetitions)
    (2304, 768, 44),
    (768, 768, 23),
    (768, 1152, 22),
    (45, 768, 1),
    (256_000, 768, 1),  # token embedding
)


class LowBitProbeError(RuntimeError):
    """Raised when a requested low-bit path is unavailable or misleading."""


def _validate_positive(name: str, value: int) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} deve essere > 0")
    return value


def _tensor_storage_bytes(tensor: Tensor) -> int:
    """Return allocated storage, not merely ``numel * element_size``."""
    try:
        return int(tensor.untyped_storage().nbytes())
    except (AttributeError, RuntimeError):
        return int(tensor.numel() * tensor.element_size())


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("percentile richiede almeno un valore")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _timing_summary_ms(samples_ns: Sequence[int]) -> dict[str, float | int]:
    samples = [float(value) / 1_000_000.0 for value in samples_ns]
    if not samples:
        raise ValueError("servono campioni temporali")
    return {
        "count": len(samples),
        "min_ms": min(samples),
        "median_ms": statistics.median(samples),
        "mean_ms": statistics.fmean(samples),
        "p95_ms": _percentile(samples, 0.95),
        "max_ms": max(samples),
    }


def _benchmark_callable(
    function: Callable[[], Any], *, warmup: int, repeats: int
) -> dict[str, float | int]:
    warmup = _validate_positive("warmup", warmup)
    repeats = _validate_positive("repeats", repeats)
    with torch.inference_mode():
        for _ in range(warmup):
            function()
        samples: list[int] = []
        for _ in range(repeats):
            started = time.perf_counter_ns()
            function()
            samples.append(time.perf_counter_ns() - started)
    return _timing_summary_ms(samples)


def _benchmark_callables_interleaved(
    functions: dict[str, Callable[[], Any]], *, warmup: int, repeats: int
) -> tuple[dict[str, Any], dict[str, dict[str, float | int]], dict[str, Any]]:
    """Time variants in a cyclic order so no path is always first or last."""
    warmup = _validate_positive("warmup", warmup)
    repeats = _validate_positive("repeats", repeats)
    names = list(functions)
    if not names:
        raise ValueError("serve almeno una variante")
    samples: dict[str, list[int]] = {name: [] for name in names}
    outputs: dict[str, Any] = {}
    with torch.inference_mode():
        for round_index in range(warmup):
            offset = round_index % len(names)
            for name in names[offset:] + names[:offset]:
                functions[name]()
        for round_index in range(repeats):
            offset = round_index % len(names)
            for name in names[offset:] + names[:offset]:
                started = time.perf_counter_ns()
                outputs[name] = functions[name]()
                samples[name].append(time.perf_counter_ns() - started)
    return (
        outputs,
        {name: _timing_summary_ms(values) for name, values in samples.items()},
        {
            "order": "cyclic interleaving",
            "variants": names,
            "position_balance_complete": repeats % len(names) == 0,
        },
    )


def aten_int4_capability(*, execute_smoke: bool = True) -> dict[str, Any]:
    """Report whether the current PyTorch build exposes a working CPU W4A32 op."""
    names = (
        "_convert_weight_to_int4pack_for_cpu",
        "_weight_int4pack_mm_for_cpu",
    )
    schemas: dict[str, str | None] = {}
    for name in names:
        try:
            operation = getattr(torch.ops.aten, name)
            schemas[name] = str(operation.default._schema)
        except (AttributeError, RuntimeError):
            schemas[name] = None
    exposed = all(schemas.values())
    smoke_ok = False
    smoke_error: str | None = None
    if exposed and execute_smoke:
        try:
            weight = torch.linspace(-1.0, 1.0, 16 * 128).reshape(16, 128)
            packed, scale_zero, _, _ = _pack_aten_int4_weight(
                weight, 128, audit_dequantized=False
            )
            output = torch.ops.aten._weight_int4pack_mm_for_cpu(
                torch.ones(1, 128), packed, 128, scale_zero
            )
            smoke_ok = output.shape == (1, 16) and bool(torch.isfinite(output).all())
        except Exception as exc:  # A probe must return evidence, not mask backend errors.
            smoke_error = f"{type(exc).__name__}: {exc}"
    elif exposed:
        smoke_ok = True
    return {
        "available": bool(exposed and smoke_ok),
        "ops_exposed": exposed,
        "smoke_executed": bool(execute_smoke and exposed),
        "smoke_ok": smoke_ok,
        "smoke_error": smoke_error,
        "schemas": schemas,
        "torch_version": torch.__version__,
        "compute_path": "W4A32 with FP32 activations/output",
        "stability": "private ATen API; pin and re-probe the PyTorch version",
    }


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _registered_cpu_op(qualified_name: str) -> bool:
    """Conservatively inspect the dispatcher without creating fake op handles."""
    try:
        return bool(
            torch._C._dispatch_has_kernel_for_dispatch_key(qualified_name, "CPU")
        )
    except (AttributeError, RuntimeError):
        return False


def probe_int6_capability(*, numel: int = 128) -> dict[str, Any]:
    """Probe INT6 storage/API signals without claiming an unverified packed path.

    ``packed_kernel_verified`` intentionally remains false: this module has no
    stock, cross-platform W6 CPU operator that can be executed and audited.
    """
    numel = _validate_positive("numel", numel)
    theoretical_bytes = math.ceil(numel * 6 / 8)
    dtype = getattr(torch, "int6", None)
    native: dict[str, Any] = {
        "dtype_exposed": dtype is not None,
        "dtype": str(dtype) if dtype is not None else None,
        "numel": numel,
        "theoretical_packed_bytes": theoretical_bytes,
        "allocated_storage_bytes": None,
        "element_size_bytes": None,
        "exact_packed_storage_observed": False,
        "allocation_error": None,
    }
    if dtype is not None:
        try:
            tensor = torch.empty(numel, dtype=dtype, device="cpu")
            allocated = _tensor_storage_bytes(tensor)
            native.update(
                {
                    "allocated_storage_bytes": allocated,
                    "element_size_bytes": tensor.element_size(),
                    # Conservative by design: padding is not accepted as proof.
                    "exact_packed_storage_observed": allocated == theoretical_bytes,
                }
            )
        except Exception as exc:
            native["allocation_error"] = f"{type(exc).__name__}: {exc}"

    torchao_version = _package_version("torchao")
    candidate_ops = (
        "torchao::_pack_8bit_act_6bit_weight",
        "torchao::_linear_8bit_act_6bit_weight",
        "torchao::_linear_fp_act_6bit_weight",
        "fbgemm::_pack_8bit_act_6bit_weight",
        "fbgemm::_linear_8bit_act_6bit_weight",
    )
    registered_ops = [name for name in candidate_ops if _registered_cpu_op(name)]

    onnx_int6 = False
    onnx_version = _package_version("onnx")
    if onnx_version is not None:
        try:
            from onnx import TensorProto

            onnx_int6 = hasattr(TensorProto, "INT6")
        except ImportError:
            pass

    ort_version = _package_version("onnxruntime")
    ort_symbols: list[str] = []
    if ort_version is not None:
        try:
            from onnxruntime.capi import _pybind_state as ort_pybind

            ort_symbols = [
                symbol
                for symbol in (
                    "quantize_matmul_2bits",
                    "quantize_matmul_4bits",
                    "quantize_matmul_6bits",
                    "quantize_matmul_8bits",
                )
                if hasattr(ort_pybind, symbol)
            ]
        except ImportError:
            pass

    packed_kernel_verified = False
    return {
        "status": "estimate-only",
        "native_torch": native,
        "torchao": {
            "installed_version": torchao_version,
            "candidate_cpu_ops_registered": registered_ops,
        },
        "onnx": {
            "version": onnx_version,
            "TensorProto_INT6": onnx_int6,
        },
        "onnxruntime": {
            "version": ort_version,
            "lowbit_pybind_symbols": ort_symbols,
            "quantize_matmul_6bits": "quantize_matmul_6bits" in ort_symbols,
        },
        "packed_kernel_verified": packed_kernel_verified,
        "deployment_ready": False,
        "reason": (
            "Nessun kernel CPU W6 packed e relativo formato di serializzazione "
            "sono stati eseguiti e verificati. uint8 con valori 0..63 resta INT8."
        ),
    }


def modernbert_weight_shapes() -> list[tuple[int, int]]:
    """Expand the audited ModernBERT linear + embedding weight shapes."""
    return [
        (rows, columns)
        for rows, columns, repetitions in DEFAULT_MODERNBERT_SHAPES
        for _ in range(repetitions)
    ]


def estimate_groupwise_storage(
    weight_shapes: Iterable[Sequence[int]],
    *,
    bits: int,
    group_size: int = 128,
    scale_bytes: int = 4,
    zero_point_bytes: int = 0,
) -> dict[str, Any]:
    """Estimate row-wise packed weight and group metadata bytes.

    This is a format estimate, not evidence that a matching kernel exists.
    Each row is independently byte-aligned; scale/zero-point metadata is one
    value per ``group_size`` columns.
    """
    bits = _validate_positive("bits", bits)
    if bits > 8:
        raise ValueError("bits deve essere compreso tra 1 e 8")
    group_size = _validate_positive("group_size", group_size)
    if scale_bytes < 0 or zero_point_bytes < 0:
        raise ValueError("i byte dei metadati non possono essere negativi")

    rows_report: list[dict[str, int]] = []
    element_count = 0
    packed_bytes = 0
    group_count = 0
    for index, shape in enumerate(weight_shapes):
        if len(shape) != 2:
            raise ValueError(f"shape {index} deve avere due dimensioni")
        rows = _validate_positive(f"shape[{index}].rows", int(shape[0]))
        columns = _validate_positive(f"shape[{index}].columns", int(shape[1]))
        row_packed_bytes = math.ceil(columns * bits / 8)
        groups_per_row = math.ceil(columns / group_size)
        shape_packed = rows * row_packed_bytes
        shape_groups = rows * groups_per_row
        element_count += rows * columns
        packed_bytes += shape_packed
        group_count += shape_groups
        rows_report.append(
            {
                "rows": rows,
                "columns": columns,
                "elements": rows * columns,
                "packed_weight_bytes": shape_packed,
                "groups": shape_groups,
            }
        )

    ideal_global_bytes = math.ceil(element_count * bits / 8)
    metadata_bytes = group_count * (scale_bytes + zero_point_bytes)
    total_bytes = packed_bytes + metadata_bytes
    return {
        "estimate_only": True,
        "bits": bits,
        "group_size": group_size,
        "tensor_count": len(rows_report),
        "elements": element_count,
        "ideal_global_weight_bytes": ideal_global_bytes,
        "row_aligned_packed_weight_bytes": packed_bytes,
        "row_alignment_overhead_bytes": packed_bytes - ideal_global_bytes,
        "groups": group_count,
        "scale_bytes_per_group": scale_bytes,
        "zero_point_bytes_per_group": zero_point_bytes,
        "metadata_bytes": metadata_bytes,
        "total_bytes": total_bytes,
        "total_mib": total_bytes / 1024**2,
        "fp32_weight_bytes": element_count * 4,
        "ratio_to_fp32": total_bytes / (element_count * 4) if element_count else None,
        "shapes": rows_report,
        "warning": "stima di formato; non prova storage packed o kernel CPU disponibili",
    }


def modernbert_lowbit_storage_matrix(*, group_size: int = 128) -> dict[str, Any]:
    """Return directly comparable symmetric W4/W6/W8 storage estimates."""
    shapes = modernbert_weight_shapes()
    return {
        f"w{bits}_symmetric": estimate_groupwise_storage(
            shapes,
            bits=bits,
            group_size=group_size,
            scale_bytes=4,
            zero_point_bytes=0,
        )
        for bits in (4, 6, 8)
    }


def _pack_aten_int4_weight(
    weight: Tensor, group_size: int, *, audit_dequantized: bool = True
) -> tuple[Tensor, Tensor, tuple[int, int], Tensor | None]:
    """Pack a 2-D FP32 CPU weight for ATen's asymmetric W4A32 operator.

    Returns packed data, ATen scale/zero tensor, original shape and a detached
    dequantized reference.  The latter is only for numerical auditing and is
    not retained by :class:`AtenInt4Linear`.
    """
    if group_size not in SUPPORTED_ATEN_GROUP_SIZES:
        raise ValueError(
            f"group_size ATen supportato: {SUPPORTED_ATEN_GROUP_SIZES}"
        )
    if weight.ndim != 2:
        raise ValueError("il peso Linear deve essere bidimensionale")
    if weight.device.type != "cpu" or weight.dtype != torch.float32:
        raise ValueError("il probe ATen richiede pesi CPU FP32")
    if not bool(torch.isfinite(weight).all()):
        raise ValueError("il peso contiene valori non finiti")
    original_rows, original_columns = (int(dim) for dim in weight.shape)
    padded_rows = math.ceil(original_rows / 16) * 16
    padded_columns = math.ceil(original_columns / group_size) * group_size
    padded = F.pad(
        weight.detach().contiguous(),
        (0, padded_columns - original_columns, 0, padded_rows - original_rows),
    )
    groups = padded.reshape(padded_rows, -1, group_size)
    minima = groups.amin(dim=-1, keepdim=True)
    maxima = groups.amax(dim=-1, keepdim=True)
    ranges = maxima - minima
    # A scale of one represents a constant group exactly with q=0 and avoids
    # division by zero.  It affects neither output because all q values are 0.
    scales = torch.where(ranges > 0, ranges / 15.0, torch.ones_like(ranges))
    quantized = torch.round((groups - minima) / scales).clamp_(0, 15).to(torch.int32)
    quantized_matrix = quantized.reshape(padded_rows, padded_columns).contiguous()
    try:
        packed = torch.ops.aten._convert_weight_to_int4pack_for_cpu(
            quantized_matrix, 1
        )
    except (AttributeError, RuntimeError) as exc:
        raise LowBitProbeError(
            "Questo build PyTorch non espone un packer ATen INT4 CPU funzionante"
        ) from exc
    zeros = minima + scales * 8.0
    scale_zero = torch.cat((scales, zeros), dim=-1).transpose(0, 1).contiguous()
    dequantized = None
    if audit_dequantized:
        dequantized = (quantized.to(torch.float32) * scales + minima).reshape(
            padded_rows, padded_columns
        )
    return packed, scale_zero, (original_rows, original_columns), dequantized


class AtenInt4Linear(nn.Module):
    """CPU-only FP32-input Linear backed by private ATen packed W4A32."""

    def __init__(
        self,
        *,
        packed_weight: Tensor,
        scales_and_zeros: Tensor,
        in_features: int,
        out_features: int,
        padded_in_features: int,
        padded_out_features: int,
        group_size: int,
        bias: Tensor | None,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.padded_in_features = int(padded_in_features)
        self.padded_out_features = int(padded_out_features)
        self.group_size = int(group_size)
        self.register_buffer("packed_weight", packed_weight)
        self.register_buffer("scales_and_zeros", scales_and_zeros)
        self.register_buffer("bias", bias)

    @classmethod
    def from_float(cls, module: nn.Linear, *, group_size: int = 128) -> "AtenInt4Linear":
        if module.weight.device.type != "cpu" or module.weight.dtype != torch.float32:
            raise ValueError("AtenInt4Linear richiede un nn.Linear CPU FP32")
        packed, scale_zero, original, _ = _pack_aten_int4_weight(
            module.weight.detach(), group_size, audit_dequantized=False
        )
        padded_out = int(scale_zero.shape[1])
        padded_in = int(scale_zero.shape[0]) * group_size
        bias = module.bias.detach().clone() if module.bias is not None else None
        return cls(
            packed_weight=packed,
            scales_and_zeros=scale_zero,
            in_features=original[1],
            out_features=original[0],
            padded_in_features=padded_in,
            padded_out_features=padded_out,
            group_size=group_size,
            bias=bias,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.device.type != "cpu" or inputs.dtype != torch.float32:
            raise LowBitProbeError("AtenInt4Linear accetta solo attivazioni CPU FP32")
        if inputs.shape[-1] != self.in_features:
            raise ValueError(
                f"dimensione input {inputs.shape[-1]}, attesa {self.in_features}"
            )
        leading_shape = inputs.shape[:-1]
        matrix = inputs.reshape(-1, self.in_features)
        if self.padded_in_features != self.in_features:
            matrix = F.pad(matrix, (0, self.padded_in_features - self.in_features))
        try:
            result = torch.ops.aten._weight_int4pack_mm_for_cpu(
                matrix,
                self.packed_weight,
                self.group_size,
                self.scales_and_zeros,
            )
        except (AttributeError, RuntimeError) as exc:
            raise LowBitProbeError("Kernel ATen W4A32 CPU non eseguibile") from exc
        result = result[:, : self.out_features]
        if self.bias is not None:
            result = result + self.bias
        return result.reshape(*leading_shape, self.out_features)

    def storage_report(self) -> dict[str, Any]:
        packed = _tensor_storage_bytes(self.packed_weight)
        metadata = _tensor_storage_bytes(self.scales_and_zeros)
        bias = _tensor_storage_bytes(self.bias) if self.bias is not None else 0
        return {
            "packed_weight_bytes": packed,
            "metadata_bytes": metadata,
            "bias_bytes": bias,
            "total_bytes": packed + metadata + bias,
            "original_fp32_weight_bytes": self.in_features * self.out_features * 4,
            "kernel": "aten::_weight_int4pack_mm_for_cpu",
            "activation_dtype": "float32",
        }


class PackedInt4Embedding(nn.Module):
    """Packed W4 embedding with selected-row FP32 dequantization.

    This reduces persistent weight storage.  It is explicitly *not* a fused
    low-bit embedding CPU kernel.
    """

    def __init__(
        self,
        *,
        packed_weight: Tensor,
        scales: Tensor,
        minima: Tensor,
        embedding_dim: int,
        padded_embedding_dim: int,
        group_size: int,
        padding_idx: int | None,
    ) -> None:
        super().__init__()
        self.num_embeddings = int(packed_weight.shape[0])
        self.embedding_dim = int(embedding_dim)
        self.padded_embedding_dim = int(padded_embedding_dim)
        self.group_size = int(group_size)
        self.padding_idx = padding_idx
        self.register_buffer("packed_weight", packed_weight)
        self.register_buffer("scales", scales)
        self.register_buffer("minima", minima)

    @classmethod
    def from_float(
        cls,
        module: nn.Embedding,
        *,
        group_size: int = 128,
        chunk_rows: int = 4096,
    ) -> "PackedInt4Embedding":
        group_size = _validate_positive("group_size", group_size)
        chunk_rows = _validate_positive("chunk_rows", chunk_rows)
        if group_size % 2:
            raise ValueError("group_size embedding deve essere pari")
        if module.max_norm is not None:
            raise ValueError("nn.Embedding con max_norm non è coperto dal probe INT4")
        weight = module.weight.detach()
        if weight.device.type != "cpu" or weight.dtype != torch.float32:
            raise ValueError("PackedInt4Embedding richiede pesi CPU FP32")
        if not bool(torch.isfinite(weight).all()):
            raise ValueError("l'embedding contiene valori non finiti")
        rows, columns = (int(dim) for dim in weight.shape)
        padded_columns = math.ceil(columns / group_size) * group_size
        group_count = padded_columns // group_size
        packed_weight = torch.empty((rows, padded_columns // 2), dtype=torch.uint8)
        scales_out = torch.empty((rows, group_count), dtype=torch.float32)
        minima_out = torch.empty((rows, group_count), dtype=torch.float32)

        for start in range(0, rows, chunk_rows):
            stop = min(rows, start + chunk_rows)
            chunk = F.pad(weight[start:stop].contiguous(), (0, padded_columns - columns))
            groups = chunk.reshape(stop - start, group_count, group_size)
            minima = groups.amin(dim=-1, keepdim=True)
            maxima = groups.amax(dim=-1, keepdim=True)
            ranges = maxima - minima
            scales = torch.where(ranges > 0, ranges / 15.0, torch.ones_like(ranges))
            quantized = torch.round((groups - minima) / scales).clamp_(0, 15)
            quantized = quantized.to(torch.uint8).reshape(stop - start, padded_columns)
            low = quantized[:, 0::2]
            high = quantized[:, 1::2]
            packed_weight[start:stop].copy_(low | (high << 4))
            scales_out[start:stop].copy_(scales.squeeze(-1))
            minima_out[start:stop].copy_(minima.squeeze(-1))

        return cls(
            packed_weight=packed_weight,
            scales=scales_out,
            minima=minima_out,
            embedding_dim=columns,
            padded_embedding_dim=padded_columns,
            group_size=group_size,
            padding_idx=module.padding_idx,
        )

    def forward(self, indices: Tensor) -> Tensor:
        if indices.device.type != "cpu" or indices.dtype not in (torch.int32, torch.int64):
            raise LowBitProbeError("PackedInt4Embedding richiede indici CPU int32/int64")
        selected = self.packed_weight[indices]
        quantized = torch.empty(
            (*selected.shape[:-1], self.padded_embedding_dim), dtype=torch.uint8
        )
        quantized[..., 0::2] = selected & 0x0F
        quantized[..., 1::2] = selected >> 4
        group_count = self.padded_embedding_dim // self.group_size
        grouped = quantized.to(torch.float32).reshape(
            *indices.shape, group_count, self.group_size
        )
        scales = self.scales[indices].unsqueeze(-1)
        minima = self.minima[indices].unsqueeze(-1)
        output = (grouped * scales + minima).reshape(
            *indices.shape, self.padded_embedding_dim
        )
        return output[..., : self.embedding_dim]

    def storage_report(self) -> dict[str, Any]:
        packed = _tensor_storage_bytes(self.packed_weight)
        scales = _tensor_storage_bytes(self.scales)
        minima = _tensor_storage_bytes(self.minima)
        original = self.num_embeddings * self.embedding_dim * 4
        return {
            "packed_weight_bytes": packed,
            "scale_bytes": scales,
            "minimum_bytes": minima,
            "total_bytes": packed + scales + minima,
            "original_fp32_weight_bytes": original,
            "ratio_to_original_weight": (packed + scales + minima) / original,
            "compute": "selected-row unpack/dequant to FP32; no fused INT4 kernel",
        }


def _count_target_modules(model: nn.Module) -> tuple[int, int]:
    return (
        sum(isinstance(module, nn.Linear) for module in model.modules()),
        sum(isinstance(module, nn.Embedding) for module in model.modules()),
    )


def _preflight_int4_model(model: nn.Module) -> None:
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Embedding)):
            if module.weight.device.type != "cpu" or module.weight.dtype != torch.float32:
                raise ValueError(f"{name or '<root>'}: il peso deve essere CPU FP32")
        if isinstance(module, nn.Embedding) and module.max_norm is not None:
            raise ValueError(f"{name or '<root>'}: max_norm non supportato")


def quantize_model_int4_aten(
    model: nn.Module,
    *,
    group_size: int = 128,
    embedding_chunk_rows: int = 4096,
    require_weight_storage_coverage: bool = True,
) -> dict[str, Any]:
    """Replace all Linear/Embedding weights with runnable packed INT4 modules.

    The operation mutates ``model``.  Linear compute uses the ATen W4A32
    kernel.  Embedding storage is packed W4, but lookup dequantization is not a
    fused low-bit kernel; the report and assertion helper preserve that fact.
    """
    if group_size not in SUPPORTED_ATEN_GROUP_SIZES:
        raise ValueError(
            f"group_size ATen supportato: {SUPPORTED_ATEN_GROUP_SIZES}"
        )
    _validate_positive("embedding_chunk_rows", embedding_chunk_rows)
    capability = aten_int4_capability()
    if not capability["available"]:
        raise LowBitProbeError(
            f"Kernel ATen INT4 CPU indisponibile: {capability['smoke_error']}"
        )
    _preflight_int4_model(model)
    linear_before, embedding_before = _count_target_modules(model)
    linear_reports: list[dict[str, Any]] = []
    embedding_reports: list[dict[str, Any]] = []

    def replace(parent: nn.Module) -> None:
        for name, child in list(parent.named_children()):
            if isinstance(child, nn.Linear):
                converted = AtenInt4Linear.from_float(child, group_size=group_size)
                linear_reports.append(converted.storage_report())
                setattr(parent, name, converted)
            elif isinstance(child, nn.Embedding):
                converted_embedding = PackedInt4Embedding.from_float(
                    child,
                    group_size=group_size,
                    chunk_rows=embedding_chunk_rows,
                )
                embedding_reports.append(converted_embedding.storage_report())
                setattr(parent, name, converted_embedding)
            else:
                replace(child)

    replace(model)
    linear_after, embedding_after = _count_target_modules(model)
    linear_converted = len(linear_reports)
    embedding_converted = len(embedding_reports)
    storage_coverage_complete = (
        linear_after == 0
        and embedding_after == 0
        and linear_converted == linear_before
        and embedding_converted == embedding_before
    )
    if require_weight_storage_coverage and not storage_coverage_complete:
        raise AssertionError(
            "Copertura storage INT4 incompleta: "
            f"Linear {linear_converted}/{linear_before}, "
            f"Embedding {embedding_converted}/{embedding_before}, "
            f"residui {linear_after}/{embedding_after}"
        )

    persistent_bytes = sum(row["total_bytes"] for row in linear_reports) + sum(
        row["total_bytes"] for row in embedding_reports
    )
    original_bytes = sum(
        row["original_fp32_weight_bytes"] for row in linear_reports
    ) + sum(row["original_fp32_weight_bytes"] for row in embedding_reports)
    fused_kernel_coverage_complete = embedding_before == 0
    return {
        "status": "ok",
        "mutates_model": True,
        "group_size": group_size,
        "linear": {
            "before": linear_before,
            "converted": linear_converted,
            "remaining_fp32_modules": linear_after,
            "kernel": "ATen W4A32",
        },
        "embedding": {
            "before": embedding_before,
            "converted": embedding_converted,
            "remaining_fp32_modules": embedding_after,
            "kernel": None,
            "compute": "packed W4 lookup followed by FP32 dequantization",
        },
        "weight_storage_coverage_complete": storage_coverage_complete,
        "fused_lowbit_kernel_coverage_complete": fused_kernel_coverage_complete,
        "fused_kernel_coverage_reason": (
            None
            if fused_kernel_coverage_complete
            else "Embedding packed ma senza kernel fused low-bit PyTorch CPU"
        ),
        "target_weight_bytes_before": original_bytes,
        "target_persistent_bytes_after": persistent_bytes,
        "persistent_ratio": persistent_bytes / original_bytes if original_bytes else None,
        "serialization": (
            "buffer packed e metadati presenti nello state_dict; ricostruire prima "
            "la stessa architettura convertita"
        ),
    }


def assert_fused_int4_kernel_coverage(report: dict[str, Any]) -> None:
    """Fail if a report could be misread as fully fused low-bit execution."""
    if not report.get("fused_lowbit_kernel_coverage_complete", False):
        raise AssertionError(
            str(report.get("fused_kernel_coverage_reason") or "copertura kernel incompleta")
        )


def benchmark_aten_int4_shape(
    *,
    m: int,
    k: int,
    n: int,
    group_size: int = 128,
    threads: int = 1,
    warmup: int = 8,
    repeats: int = 30,
    seed: int = 20260822,
) -> dict[str, Any]:
    """Microbenchmark the same FP32 inputs/weights with FP32 and ATen W4A32."""
    m = _validate_positive("m", m)
    k = _validate_positive("k", k)
    n = _validate_positive("n", n)
    threads = _validate_positive("threads", threads)
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    inputs = torch.randn((m, k), generator=generator, dtype=torch.float32)
    linear = nn.Linear(k, n, bias=False, dtype=torch.float32)
    with torch.no_grad():
        linear.weight.copy_(torch.randn((n, k), generator=generator))
    pack_started = time.perf_counter_ns()
    packed, scale_zero, original, dequantized_weight = _pack_aten_int4_weight(
        linear.weight.detach(), group_size, audit_dequantized=True
    )
    pack_ms = (time.perf_counter_ns() - pack_started) / 1_000_000.0
    if dequantized_weight is None:  # Defensive: requested explicitly above.
        raise AssertionError("riferimento dequantizzato mancante")
    quantized = AtenInt4Linear(
        packed_weight=packed,
        scales_and_zeros=scale_zero,
        in_features=original[1],
        out_features=original[0],
        padded_in_features=int(scale_zero.shape[0]) * group_size,
        padded_out_features=int(scale_zero.shape[1]),
        group_size=group_size,
        bias=None,
    )

    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(threads)
        with torch.inference_mode():
            expected = linear(inputs)
            actual = quantized(inputs)
            dequantized_reference = inputs @ dequantized_weight[:n, :k].T
        _, timings, schedule = _benchmark_callables_interleaved(
            {"fp32": lambda: linear(inputs), "aten_w4a32": lambda: quantized(inputs)},
            warmup=warmup,
            repeats=repeats,
        )
        fp32_timing = timings["fp32"]
        int4_timing = timings["aten_w4a32"]
    finally:
        torch.set_num_threads(previous_threads)

    total_errors = (actual - expected).abs()
    quantization_errors = (dequantized_reference - expected).abs()
    kernel_errors = (actual - dequantized_reference).abs()
    fp32_median = float(fp32_timing["median_ms"])
    int4_median = float(int4_timing["median_ms"])
    return {
        "shape": {"m": m, "k": k, "n": n},
        "threads": threads,
        "group_size": group_size,
        "same_inputs_and_source_weights": True,
        "one_time_pack_ms_excluded_from_inference": pack_ms,
        "fp32": {
            "kernel": "torch.nn.functional.linear / ATen FP32",
            "timing": fp32_timing,
        },
        "aten_w4a32": {
            "kernel": "aten::_weight_int4pack_mm_for_cpu",
            "activation_dtype": "float32",
            "output_dtype": str(actual.dtype),
            "quantization_scheme": "groupwise asymmetric affine uint4",
            "timing": int4_timing,
            "storage": quantized.storage_report(),
        },
        "speedup_fp32_over_w4a32": fp32_median / int4_median,
        "timing_schedule": schedule,
        "cross_engine_limit": (
            "ATen uses groupwise asymmetric affine packing here; compare engines "
            "only after matching quantized tensors, not from this speedup alone"
        ),
        "numerical_isolation": {
            "quantization_only_dequantized_reference_vs_fp32": {
                "max_abs": float(quantization_errors.max()),
                "mean_abs": float(quantization_errors.mean()),
            },
            "aten_kernel_vs_dequantized_reference": {
                "max_abs": float(kernel_errors.max()),
                "mean_abs": float(kernel_errors.mean()),
            },
            "end_to_end_w4a32_vs_fp32": {
                "max_abs": float(total_errors.max()),
                "mean_abs": float(total_errors.mean()),
            },
        },
        "scope": "kernel microbenchmark; excludes tokenizer, model load and service",
    }


def benchmark_aten_modernbert_shapes(
    *,
    token_rows: Sequence[int] = (1, 64, 128),
    group_size: int = 128,
    threads: int = 1,
    warmup: int = 8,
    repeats: int = 30,
) -> list[dict[str, Any]]:
    """Probe representative ModernBERT Linear shapes, including its 45-way head."""
    shapes = ((768, 2304), (768, 768), (1152, 768), (768, 45))
    return [
        benchmark_aten_int4_shape(
            m=int(m),
            k=k,
            n=n,
            group_size=group_size,
            threads=threads,
            warmup=warmup,
            repeats=repeats,
        )
        for m in token_rows
        for k, n in shapes
    ]


def _onnx_initializer_digest(model_path: Path) -> str:
    import onnx
    from onnx import numpy_helper

    model = onnx.load_model(str(model_path), load_external_data=True)
    digest = hashlib.sha256()
    for initializer in sorted(model.graph.initializer, key=lambda item: item.name):
        array = numpy_helper.to_array(initializer)
        digest.update(initializer.name.encode("utf-8"))
        digest.update(str(initializer.data_type).encode("ascii"))
        digest.update(json.dumps(list(initializer.dims)).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _set_matmul_nbits_accuracy_level(model: Any, accuracy_level: int) -> int:
    from onnx import helper

    if accuracy_level not in (1, 4):
        raise ValueError("accuracy_level deve essere 1 (W4A32) o 4 (W4A8)")
    changed = 0
    for node in model.graph.node:
        if node.op_type != "MatMulNBits":
            continue
        retained = [attribute for attribute in node.attribute if attribute.name != "accuracy_level"]
        del node.attribute[:]
        node.attribute.extend(retained)
        node.attribute.append(helper.make_attribute("accuracy_level", accuracy_level))
        changed += 1
    if not changed:
        raise LowBitProbeError("Il grafo non contiene MatMulNBits")
    return changed


def clone_onnx_int4_accuracy_level(
    source_model: str | Path,
    destination_model: str | Path,
    *,
    accuracy_level: int,
) -> dict[str, Any]:
    """Clone packed INT4 initializers and change only the compute-path attribute."""
    import onnx

    source = Path(source_model).expanduser().resolve()
    destination = Path(destination_model).expanduser().resolve()
    if source == destination:
        raise ValueError("sorgente e destinazione devono essere file distinti")
    if not source.is_file():
        raise FileNotFoundError(source)
    if destination.exists() or destination.with_name(destination.name + ".data").exists():
        raise FileExistsError(f"destinazione già esistente: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_digest = _onnx_initializer_digest(source)
    model = onnx.load_model(str(source), load_external_data=True)
    changed = _set_matmul_nbits_accuracy_level(model, accuracy_level)
    onnx.save_model(
        model,
        str(destination),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=destination.name + ".data",
        size_threshold=0,
    )
    destination_digest = _onnx_initializer_digest(destination)
    if source_digest != destination_digest:
        raise AssertionError("Gli initializer sono cambiati clonando accuracy_level")
    return {
        "source": str(source),
        "destination": str(destination),
        "accuracy_level": accuracy_level,
        "MatMulNBits_nodes_changed": changed,
        "initializer_sha256": destination_digest,
        "initializers_identical": True,
    }


def _write_synthetic_matmul(path: Path, weight: Any, m: int, k: int, n: int) -> None:
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    inputs = helper.make_tensor_value_info("X", TensorProto.FLOAT, [m, k])
    outputs = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [m, n])
    node = helper.make_node("MatMul", ["X", "W"], ["Y"], name="MatMulProbe")
    graph = helper.make_graph(
        [node],
        "lowbit_probe",
        [inputs],
        [outputs],
        [numpy_helper.from_array(weight, name="W")],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 21)],
        producer_name="rizzo-pii-lowbit-probe",
    )
    model.ir_version = min(model.ir_version, 10)
    onnx.save_model(model, str(path))


def _quantize_synthetic_int4(
    source: Path, destination: Path, *, block_size: int, accuracy_level: int
) -> None:
    from onnxruntime.quantization import matmul_nbits_quantizer, quant_utils

    config = matmul_nbits_quantizer.DefaultWeightOnlyQuantConfig(
        block_size=block_size,
        is_symmetric=True,
        accuracy_level=accuracy_level,
        quant_format=quant_utils.QuantFormat.QOperator,
        op_types_to_quantize=("MatMul",),
        quant_axes=(("MatMul", 0),),
        bits=4,
    )
    quantizer = matmul_nbits_quantizer.MatMulNBitsQuantizer(
        model=str(source), algo_config=config
    )
    quantizer.process()
    quantizer.model.save_model_to_file(str(destination), True)


def _ort_session(model: Path, *, threads: int) -> Any:
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return ort.InferenceSession(
        str(model), sess_options=options, providers=["CPUExecutionProvider"]
    )


def benchmark_onnx_accuracy_matrix(
    *,
    m: int,
    k: int,
    n: int,
    block_size: int = 128,
    threads: int = 1,
    warmup: int = 8,
    repeats: int = 30,
    seed: int = 20260822,
) -> dict[str, Any]:
    """Compare FP32, W4A32(level 1), and W4A8(level 4) on identical weights."""
    try:
        import numpy as np
        import onnx  # noqa: F401 - validates the optional stack before work.
        import onnxruntime as ort
    except ImportError as exc:
        raise LowBitProbeError("Servono numpy, onnx e onnxruntime") from exc
    for name, value in (("m", m), ("k", k), ("n", n), ("threads", threads)):
        _validate_positive(name, value)
    if block_size < 16 or block_size & (block_size - 1):
        raise ValueError("block_size deve essere una potenza di due >= 16")
    if k % block_size:
        raise ValueError("k deve essere divisibile per block_size per questo probe")

    generator = np.random.default_rng(int(seed))
    inputs = generator.standard_normal((m, k), dtype=np.float32)
    weight = generator.standard_normal((k, n), dtype=np.float32)
    with tempfile.TemporaryDirectory(prefix="rizzo-lowbit-") as temporary:
        root = Path(temporary)
        fp32_path = root / "matmul.fp32.onnx"
        level1_path = root / "matmul.int4.acc1.onnx"
        level4_path = root / "matmul.int4.acc4.onnx"
        _write_synthetic_matmul(fp32_path, weight, m, k, n)
        quantize_started = time.perf_counter_ns()
        _quantize_synthetic_int4(
            fp32_path,
            level1_path,
            block_size=block_size,
            accuracy_level=1,
        )
        quantize_ms = (time.perf_counter_ns() - quantize_started) / 1_000_000.0
        clone_started = time.perf_counter_ns()
        clone = clone_onnx_int4_accuracy_level(
            level1_path, level4_path, accuracy_level=4
        )
        clone_ms = (time.perf_counter_ns() - clone_started) / 1_000_000.0

        rows: dict[str, Any] = {}
        sessions: dict[str, Any] = {}
        load_order: list[str] = []
        for variant, path, accuracy_level, compute in (
            ("fp32", fp32_path, None, "FP32"),
            ("int4_acc1", level1_path, 1, "W4A32"),
            ("int4_acc4", level4_path, 4, "W4A8 dynamic activation quantization"),
        ):
            load_started = time.perf_counter_ns()
            session = _ort_session(path, threads=threads)
            load_ms = (time.perf_counter_ns() - load_started) / 1_000_000.0
            sessions[variant] = session
            load_order.append(variant)
            rows[variant] = {
                "accuracy_level": accuracy_level,
                "compute_path": compute,
                "load_ms": load_ms,
            }

        functions = {
            variant: (
                lambda session=session: session.run(
                    None, {session.get_inputs()[0].name: inputs}
                )[0]
            )
            for variant, session in sessions.items()
        }
        outputs, timings, schedule = _benchmark_callables_interleaved(
            functions, warmup=warmup, repeats=repeats
        )
        for variant, timing in timings.items():
            rows[variant]["timing"] = timing

        baseline = outputs["fp32"]
        for variant in ("int4_acc1", "int4_acc4"):
            error = np.abs(outputs[variant] - baseline)
            rows[variant]["numerical_error_vs_fp32"] = {
                "max_abs": float(error.max()),
                "mean_abs": float(error.mean()),
            }
        acc4_vs_acc1 = np.abs(outputs["int4_acc4"] - outputs["int4_acc1"])
        rows["int4_acc4"]["numerical_error_vs_identical_weights_acc1"] = {
            "max_abs": float(acc4_vs_acc1.max()),
            "mean_abs": float(acc4_vs_acc1.mean()),
        }
        digest1 = _onnx_initializer_digest(level1_path)
        digest4 = _onnx_initializer_digest(level4_path)
        if digest1 != digest4:
            raise AssertionError("acc1 e acc4 non usano gli stessi initializer")
        del sessions

    fp32_median = float(rows["fp32"]["timing"]["median_ms"])
    for variant in ("int4_acc1", "int4_acc4"):
        median = float(rows[variant]["timing"]["median_ms"])
        rows[variant]["speedup_vs_fp32"] = fp32_median / median
    rows["int4_acc4"]["speedup_vs_acc1"] = (
        float(rows["int4_acc1"]["timing"]["median_ms"])
        / float(rows["int4_acc4"]["timing"]["median_ms"])
    )
    return {
        "shape": {"m": m, "k": k, "n": n},
        "block_size": block_size,
        "threads": threads,
        "onnxruntime_version": ort.__version__,
        "same_fp32_inputs": True,
        "same_packed_int4_initializers": digest1 == digest4,
        "packed_initializer_sha256": clone["initializer_sha256"],
        "quantization_scheme": "ORT DEFAULT groupwise symmetric affine W4",
        "one_time_build_ms_excluded_from_inference": {
            "quantize_acc1": quantize_ms,
            "clone_and_change_attribute_only_acc4": clone_ms,
        },
        "engine_isolation": (
            "acc1 and acc4 differ only by MatMulNBits accuracy_level; packed "
            "initializer digest is identical"
        ),
        "timing_schedule": schedule,
        "load_timing_warning": (
            f"load order fisso {load_order}; load_ms diagnostico, non usare per speedup"
        ),
        "variants": rows,
        "scope": "synthetic MatMul kernel/engine probe; not end-to-end ModernBERT",
    }


def probe_modernbert_int4_e2e(
    model_path: str | Path,
    *,
    batch_size: int = 1,
    sequence_length: int = 64,
    threads: int = 1,
    warmup: int = 2,
    repeats: int = 5,
    group_size: int = 128,
    seed: int = 20260822,
) -> dict[str, Any]:
    """Run a synthetic-input smoke/regression of a local HF token classifier.

    This is not a dataset regression.  Its purpose is to prove runnable
    end-to-end coverage and expose prediction/logit drift before the slower
    validation corpus is run by the existing evaluation workflow.
    """
    try:
        from transformers import AutoModelForTokenClassification
    except ImportError as exc:
        raise LowBitProbeError("Installare transformers per il probe end-to-end") from exc
    model_path = Path(model_path).expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    batch_size = _validate_positive("batch_size", batch_size)
    sequence_length = _validate_positive("sequence_length", sequence_length)
    threads = _validate_positive("threads", threads)
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    model = AutoModelForTokenClassification.from_pretrained(
        str(model_path), local_files_only=True
    ).eval().to(device="cpu", dtype=torch.float32)
    vocab_size = int(model.config.vocab_size)
    input_ids = torch.randint(
        0,
        vocab_size,
        (batch_size, sequence_length),
        generator=generator,
        dtype=torch.int64,
    )
    attention_mask = torch.ones_like(input_ids)

    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(threads)

        def run() -> Tensor:
            return model(input_ids=input_ids, attention_mask=attention_mask).logits

        with torch.inference_mode():
            fp32_logits = run().detach().clone()
        fp32_timing = _benchmark_callable(run, warmup=warmup, repeats=repeats)
        coverage = quantize_model_int4_aten(model, group_size=group_size)
        gc.collect()
        with torch.inference_mode():
            int4_logits = run().detach().clone()
        int4_timing = _benchmark_callable(run, warmup=warmup, repeats=repeats)
    finally:
        torch.set_num_threads(previous_threads)

    errors = (int4_logits - fp32_logits).abs()
    fp32_labels = fp32_logits.argmax(dim=-1)
    int4_labels = int4_logits.argmax(dim=-1)
    return {
        "status": "ok",
        "model_path": str(model_path),
        "input_shape": [batch_size, sequence_length],
        "threads": threads,
        "coverage": coverage,
        "fp32_timing": fp32_timing,
        "int4_timing": int4_timing,
        "speedup_vs_fp32": (
            float(fp32_timing["median_ms"]) / float(int4_timing["median_ms"])
        ),
        "logit_error": {
            "max_abs": float(errors.max()),
            "mean_abs": float(errors.mean()),
        },
        "label_agreement": float((fp32_labels == int4_labels).float().mean()),
        "regression_scope": (
            "synthetic-input smoke only; use the validation dataset for quality gates"
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Probe CPU low-bit: ATen W4A32, ORT acc1/acc4 e capability INT6."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    micro = subparsers.add_parser("microbench", help="FP32 vs ATen W4A32")
    micro.add_argument("--m", type=int, default=64)
    micro.add_argument("--k", type=int, default=768)
    micro.add_argument("--n", type=int, default=2304)
    micro.add_argument("--group-size", type=int, default=128)
    micro.add_argument("--threads", type=int, default=1)
    micro.add_argument("--warmup", type=int, default=8)
    micro.add_argument("--repeats", type=int, default=30)

    onnx_matrix = subparsers.add_parser(
        "onnx-matrix", help="FP32 vs INT4 accuracy_level 1 e 4"
    )
    onnx_matrix.add_argument("--m", type=int, default=64)
    onnx_matrix.add_argument("--k", type=int, default=768)
    onnx_matrix.add_argument("--n", type=int, default=2304)
    onnx_matrix.add_argument("--block-size", type=int, default=128)
    onnx_matrix.add_argument("--threads", type=int, default=1)
    onnx_matrix.add_argument("--warmup", type=int, default=8)
    onnx_matrix.add_argument("--repeats", type=int, default=30)

    int6 = subparsers.add_parser("int6", help="capability probe e stima storage")
    int6.add_argument("--group-size", type=int, default=128)

    e2e = subparsers.add_parser("e2e", help="smoke PyTorch ModernBERT INT4")
    e2e.add_argument("model_path", type=Path)
    e2e.add_argument("--batch-size", type=int, default=1)
    e2e.add_argument("--sequence-length", type=int, default=64)
    e2e.add_argument("--group-size", type=int, default=128)
    e2e.add_argument("--threads", type=int, default=1)
    e2e.add_argument("--warmup", type=int, default=2)
    e2e.add_argument("--repeats", type=int, default=5)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "microbench":
        report = benchmark_aten_int4_shape(
            m=args.m,
            k=args.k,
            n=args.n,
            group_size=args.group_size,
            threads=args.threads,
            warmup=args.warmup,
            repeats=args.repeats,
        )
    elif args.command == "onnx-matrix":
        report = benchmark_onnx_accuracy_matrix(
            m=args.m,
            k=args.k,
            n=args.n,
            block_size=args.block_size,
            threads=args.threads,
            warmup=args.warmup,
            repeats=args.repeats,
        )
    elif args.command == "int6":
        report = {
            "capability": probe_int6_capability(),
            "storage": modernbert_lowbit_storage_matrix(group_size=args.group_size),
        }
    elif args.command == "e2e":
        report = probe_modernbert_int4_e2e(
            args.model_path,
            batch_size=args.batch_size,
            sequence_length=args.sequence_length,
            group_size=args.group_size,
            threads=args.threads,
            warmup=args.warmup,
            repeats=args.repeats,
        )
    else:  # pragma: no cover - argparse enforces the choices.
        raise AssertionError(args.command)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
