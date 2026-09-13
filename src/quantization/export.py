# -*- coding: utf-8 -*-
"""Export a local Rizzo PII checkpoint to an ONNX FP32 CPU model.

The module intentionally has no ML imports at import time.  This keeps commands
such as ``--help`` usable in small deployment environments and makes missing
optional dependencies fail only when their functionality is requested.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Iterable, Sequence


DEFAULT_VALIDATION_SHAPES: tuple[tuple[int, int], ...] = ((1, 8), (2, 32))


class ExportError(RuntimeError):
    """Raised when a checkpoint cannot be exported or validated."""


def _local_checkpoint(path: str | Path) -> Path:
    checkpoint = Path(path).expanduser().resolve()
    if not checkpoint.is_dir():
        raise ExportError(
            f"Checkpoint locale non trovato: {checkpoint}. "
            "Questo comando non accetta direttamente un model ID Hugging Face."
        )
    config = checkpoint / "config.json"
    if not config.is_file():
        raise ExportError(f"config.json non trovato nel checkpoint: {checkpoint}")
    return checkpoint


def _normalise_shapes(shapes: Iterable[Sequence[int]]) -> tuple[tuple[int, int], ...]:
    normalised: list[tuple[int, int]] = []
    for shape in shapes:
        if len(shape) != 2:
            raise ValueError(f"Una forma di test deve essere BATCH,SEQUENCE; ricevuta {shape!r}")
        batch, sequence = int(shape[0]), int(shape[1])
        if batch < 1 or sequence < 1:
            raise ValueError(f"Batch e sequence devono essere positivi; ricevuti {(batch, sequence)}")
        normalised.append((batch, sequence))
    if not normalised:
        raise ValueError("Specificare almeno una forma di validazione")
    return tuple(normalised)


def _find_exported_model(output_dir: Path) -> Path:
    preferred = output_dir / "model.onnx"
    if preferred.is_file():
        return preferred
    candidates = sorted(output_dir.glob("*.onnx"))
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise ExportError(f"L'esportatore non ha prodotto alcun file ONNX in {output_dir}")
    names = ", ".join(path.name for path in candidates)
    raise ExportError(f"Esportazione ambigua: trovati piu file ONNX in {output_dir}: {names}")


def _copy_preprocessor_files(checkpoint: Path, output_dir: Path) -> None:
    """Save tokenizer/config through Transformers, with a safe file-copy fallback."""
    try:
        from transformers import AutoConfig, AutoTokenizer

        config = AutoConfig.from_pretrained(
            str(checkpoint), local_files_only=True, trust_remote_code=False
        )
        tokenizer = AutoTokenizer.from_pretrained(
            str(checkpoint), local_files_only=True, trust_remote_code=False,
            fix_mistral_regex=False,
        )
        config.save_pretrained(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))
        return
    except Exception as exc:
        # Export may have succeeded even if a non-standard tokenizer cannot be
        # instantiated.  Preserve all standard tokenizer/config assets verbatim.
        copied = False
        patterns = (
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "added_tokens.json",
            "vocab.*",
            "merges.txt",
            "sentencepiece.*",
            "*.model",
        )
        for pattern in patterns:
            for source in checkpoint.glob(pattern):
                if source.is_file():
                    shutil.copy2(source, output_dir / source.name)
                    copied = True
        if not copied or not (output_dir / "config.json").is_file():
            raise ExportError(
                "Impossibile copiare tokenizer e config dal checkpoint locale"
            ) from exc


def _export_with_optimum(checkpoint: Path, output_dir: Path, opset: int) -> Path:
    try:
        from optimum.exporters.onnx import main_export
    except ImportError as exc:
        raise ExportError(
            "Optimum non installato (serve `optimum[onnxruntime]`); "
            "verra tentato il fallback torch.onnx."
        ) from exc

    try:
        main_export(
            model_name_or_path=str(checkpoint),
            output=output_dir,
            task="token-classification",
            opset=opset,
            device="cpu",
            dtype="fp32",
            framework="pt",
            do_validation=False,
            local_files_only=True,
            trust_remote_code=False,
        )
    except Exception as exc:
        raise ExportError(f"Export Optimum non riuscito: {exc}") from exc
    return _find_exported_model(output_dir)


def _export_with_torch(checkpoint: Path, output_dir: Path, opset: int) -> Path:
    try:
        import torch
        from transformers import AutoModelForTokenClassification
    except ImportError as exc:
        raise ExportError(
            "Fallback ONNX non disponibile: installare torch e transformers"
        ) from exc

    model_path = output_dir / "model.onnx"
    try:
        model = AutoModelForTokenClassification.from_pretrained(
            str(checkpoint),
            local_files_only=True,
            trust_remote_code=False,
            torch_dtype=torch.float32,
            attn_implementation="eager",
        ).cpu().eval()

        # Some Transformers versions expose the implementation only through the
        # config/private field.  Setting it is harmless when the constructor has
        # already honoured ``attn_implementation``.
        if hasattr(model.config, "_attn_implementation"):
            model.config._attn_implementation = "eager"

        class _LogitsOnly(torch.nn.Module):
            def __init__(self, inner: Any) -> None:
                super().__init__()
                self.inner = inner

            def forward(self, input_ids: Any, attention_mask: Any) -> Any:
                return self.inner(
                    input_ids=input_ids, attention_mask=attention_mask, return_dict=True
                ).logits

        wrapper = _LogitsOnly(model).eval()
        dummy_ids = torch.zeros((1, 16), dtype=torch.long)
        dummy_mask = torch.ones((1, 16), dtype=torch.long)
        with torch.inference_mode():
            torch.onnx.export(
                wrapper,
                (dummy_ids, dummy_mask),
                str(model_path),
                input_names=["input_ids", "attention_mask"],
                output_names=["logits"],
                dynamic_axes={
                    "input_ids": {0: "batch", 1: "sequence"},
                    "attention_mask": {0: "batch", 1: "sequence"},
                    "logits": {0: "batch", 1: "sequence"},
                },
                export_params=True,
                do_constant_folding=True,
                opset_version=opset,
                dynamo=False,
            )
    except Exception as exc:
        raise ExportError(f"Fallback torch.onnx non riuscito: {exc}") from exc
    if not model_path.is_file():
        raise ExportError(f"torch.onnx non ha creato {model_path}")
    return model_path


def _numpy_dtype(ort_type: str) -> Any:
    import numpy as np

    mapping = {
        "tensor(int64)": np.int64,
        "tensor(int32)": np.int32,
        "tensor(float)": np.float32,
        "tensor(float16)": np.float16,
        "tensor(bool)": np.bool_,
    }
    try:
        return mapping[ort_type]
    except KeyError as exc:
        raise ExportError(f"Tipo input ONNX non gestito nella validazione: {ort_type}") from exc


def validate_onnx_cpu(
    model_path: str | Path,
    shapes: Iterable[Sequence[int]] = DEFAULT_VALIDATION_SHAPES,
) -> list[dict[str, Any]]:
    """Load an ONNX model on CPU and run deterministic shape smoke tests."""
    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError as exc:
        raise ExportError(
            "Validazione CPU impossibile: installare numpy e onnxruntime"
        ) from exc

    path = Path(model_path).expanduser().resolve()
    if not path.is_file():
        raise ExportError(f"Modello ONNX non trovato: {path}")
    validation_shapes = _normalise_shapes(shapes)

    try:
        session = ort.InferenceSession(
            str(path), providers=["CPUExecutionProvider"]
        )
    except Exception as exc:
        raise ExportError(f"ONNX Runtime CPU non riesce a caricare {path}: {exc}") from exc

    available = session.get_providers()
    if "CPUExecutionProvider" not in available:
        raise ExportError(f"CPUExecutionProvider non disponibile; provider attivi: {available}")

    results: list[dict[str, Any]] = []
    for batch, sequence in validation_shapes:
        feeds: dict[str, Any] = {}
        for input_meta in session.get_inputs():
            rank = len(input_meta.shape)
            if rank != 2:
                raise ExportError(
                    f"Input {input_meta.name!r} con rank {rank}: atteso rank 2 per token classification"
                )
            actual_shape = (
                input_meta.shape[0] if isinstance(input_meta.shape[0], int) else batch,
                input_meta.shape[1] if isinstance(input_meta.shape[1], int) else sequence,
            )
            dtype = _numpy_dtype(input_meta.type)
            name = input_meta.name.lower()
            if "attention_mask" in name:
                value = np.ones(actual_shape, dtype=dtype)
            elif "position_ids" in name:
                value = np.broadcast_to(
                    np.arange(actual_shape[1], dtype=dtype), actual_shape
                ).copy()
            else:
                value = np.zeros(actual_shape, dtype=dtype)
            feeds[input_meta.name] = value

        try:
            outputs = session.run(None, feeds)
        except Exception as exc:
            raise ExportError(
                f"Inferenza CPU fallita per shape batch={batch}, sequence={sequence}: {exc}"
            ) from exc
        if not outputs:
            raise ExportError("Il modello ONNX non ha restituito output")
        logits_shape = tuple(int(dim) for dim in outputs[0].shape)
        if len(logits_shape) != 3:
            raise ExportError(f"Output logits inatteso: shape {logits_shape}, atteso rank 3")
        expected_batch = next(iter(feeds.values())).shape[0]
        expected_sequence = next(iter(feeds.values())).shape[1]
        if logits_shape[:2] != (expected_batch, expected_sequence):
            raise ExportError(
                f"Output logits {logits_shape} non coerente con input "
                f"{(expected_batch, expected_sequence)}"
            )
        results.append(
            {
                "requested_shape": [batch, sequence],
                "input_shape": [expected_batch, expected_sequence],
                "logits_shape": list(logits_shape),
            }
        )
    return results


def export_model(
    checkpoint: str | Path,
    output_dir: str | Path,
    *,
    opset: int = 18,
    validation_shapes: Iterable[Sequence[int]] = DEFAULT_VALIDATION_SHAPES,
    prefer_optimum: bool = True,
) -> dict[str, Any]:
    """Export ``checkpoint`` and return a machine-readable export report."""
    if opset < 18:
        raise ValueError("ModernBERT richiede un opset moderno; usare opset >= 18")
    source = _local_checkpoint(checkpoint)
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)

    warnings: list[str] = []
    model_path: Path | None = None
    method = "torch.onnx"
    if prefer_optimum:
        try:
            model_path = _export_with_optimum(source, destination, opset)
            method = "optimum"
        except ExportError as exc:
            warnings.append(str(exc))

    if model_path is None:
        try:
            model_path = _export_with_torch(source, destination, opset)
        except ExportError as exc:
            detail = f"; errore Optimum: {warnings[-1]}" if warnings else ""
            raise ExportError(f"Nessun percorso di export riuscito: {exc}{detail}") from exc

    _copy_preprocessor_files(source, destination)
    validation = validate_onnx_cpu(model_path, validation_shapes)
    return {
        "status": "ok",
        "format": "onnx",
        "precision": "fp32",
        "method": method,
        "checkpoint": str(source),
        "model": str(model_path),
        "model_bytes": model_path.stat().st_size,
        "opset": opset,
        "cpu_validation": validation,
        "warnings": warnings,
    }


def _parse_shape(value: str) -> tuple[int, int]:
    try:
        batch, sequence = value.lower().replace("x", ",").split(",", 1)
        return int(batch), int(sequence)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("Usare BATCHxSEQUENCE, ad esempio 2x128") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Esporta un checkpoint ModernBERT token-classification locale in ONNX FP32."
    )
    parser.add_argument("checkpoint", type=Path, help="directory locale del checkpoint")
    parser.add_argument("output_dir", type=Path, help="directory degli artefatti ONNX")
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument(
        "--validation-shape",
        action="append",
        type=_parse_shape,
        dest="validation_shapes",
        help="shape CPU da verificare, ripetibile (default: 1x8 e 2x32)",
    )
    parser.add_argument(
        "--torch-only", action="store_true", help="salta Optimum e usa direttamente torch.onnx"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    shapes = args.validation_shapes or DEFAULT_VALIDATION_SHAPES
    try:
        report = export_model(
            args.checkpoint,
            args.output_dir,
            opset=args.opset,
            validation_shapes=shapes,
            prefer_optimum=not args.torch_only,
        )
    except (ExportError, ValueError) as exc:
        raise SystemExit(f"Errore export: {exc}") from exc
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
