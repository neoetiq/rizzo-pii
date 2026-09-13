"""Lightweight integrity tests for isolated model evaluation."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.quantization import evaluate


class QuantizationEvaluationTests(unittest.TestCase):
    def test_onnx_artifact_replacement_during_inference_is_rejected(self):
        before = [{"path": "/tmp/model.onnx", "bytes": 10, "inode": 1, "mtime_ns": 1}]
        after = [{"path": "/tmp/model.onnx", "bytes": 10, "inode": 1, "mtime_ns": 2}]
        performance = {
            "batch_latencies_seconds": [0.01],
            "inference_seconds": 0.01,
            "load_seconds": 0.01,
            "rss_after_load_bytes": 0,
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report_path = root / "metrics.json"
            (root / "validation.jsonl").write_text("{}\n", encoding="utf-8")
            (root / "config.json").write_text("{}\n", encoding="utf-8")
            with (
                patch.object(
                    evaluate,
                    "load_validation",
                    return_value=[evaluate.ValidationRecord(["token"], ["O"])],
                ),
                patch.object(evaluate, "_run_onnx", return_value=([["O"]], performance)),
                patch.object(evaluate, "_artifact_stat_snapshot", side_effect=[before, after]),
            ):
                with self.assertRaisesRegex(RuntimeError, "modificato durante"):
                    evaluate.evaluate_variant(
                        name="onnx-test",
                        backend="onnx",
                        model_path=root / "model.onnx",
                        tokenizer_path=root,
                        validation_path=root / "validation.jsonl",
                        predictions_path=root / "predictions.jsonl",
                        report_path=report_path,
                    )

            self.assertFalse(report_path.exists())

    def test_onnx_metrics_record_evaluation_time_integrity(self):
        artifact = {
            "model": "/tmp/model.onnx",
            "bytes": 10,
            "files": [{"path": "/tmp/model.onnx", "bytes": 10, "sha256": "same"}],
            "op_counts": {"MatMul": 1},
        }
        performance = {
            "batch_latencies_seconds": [0.01],
            "inference_seconds": 0.01,
            "load_seconds": 0.01,
            "rss_after_load_bytes": 0,
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "validation.jsonl").write_text("{}\n", encoding="utf-8")
            (root / "config.json").write_text("{}\n", encoding="utf-8")
            with (
                patch.object(
                    evaluate,
                    "load_validation",
                    return_value=[evaluate.ValidationRecord(["token"], ["O"])],
                ),
                patch.object(evaluate, "_run_onnx", return_value=([["O"]], performance)),
                patch.object(
                    evaluate,
                    "_artifact_stat_snapshot",
                    return_value=[{"path": "/tmp/model.onnx", "bytes": 10, "inode": 1, "mtime_ns": 1}],
                ),
                patch("src.quantization.quantize.artifact_info", return_value=artifact),
            ):
                report = evaluate.evaluate_variant(
                    name="onnx-test",
                    backend="onnx",
                    model_path=root / "model.onnx",
                    tokenizer_path=root,
                    validation_path=root / "validation.jsonl",
                    predictions_path=root / "predictions.jsonl",
                    report_path=root / "metrics.json",
                )

        self.assertEqual(artifact, report["artifact_integrity"])
        self.assertEqual("post-evaluation-with-stat-guard", report["artifact_integrity_timing"])

    def test_deferred_hash_does_not_read_artifact_and_peak_includes_explicit_load(self):
        performance = {
            "batch_latencies_seconds": [0.01],
            "request_latencies_seconds": [0.02],
            "inference_seconds": 0.02,
            "load_seconds": 0.01,
            "rss_after_load_bytes": 200,
            "memory_snapshots": {
                "after_load": {"rss_bytes": 200},
                "after_warmup": {"rss_bytes": 180},
                "after_inference": {"rss_bytes": 190},
            },
        }

        class FakeSampler:
            interval_seconds = 0.02
            start_bytes = 100
            peak_bytes = 120

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return None

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "validation.jsonl").write_text("{}\n", encoding="utf-8")
            (root / "config.json").write_text("{}\n", encoding="utf-8")
            stat = [{"path": "/tmp/model.onnx", "bytes": 10, "inode": 1, "mtime_ns": 1}]
            with (
                patch.object(
                    evaluate,
                    "load_validation",
                    return_value=[evaluate.ValidationRecord(["token"], ["O"])],
                ),
                patch.object(evaluate, "_run_onnx", return_value=([["O"]], performance)) as runner,
                patch.object(evaluate, "_artifact_stat_snapshot", return_value=stat),
                patch.object(evaluate, "_memory_snapshot", return_value={"rss_bytes": 100}),
                patch.object(evaluate, "RssSampler", return_value=FakeSampler()),
                patch("src.quantization.quantize.artifact_info") as artifact_info,
            ):
                report = evaluate.evaluate_variant(
                    name="onnx-test",
                    backend="onnx",
                    model_path=root / "model.onnx",
                    tokenizer_path=root,
                    validation_path=root / "validation.jsonl",
                    predictions_path=root / "predictions.jsonl",
                    report_path=root / "metrics.json",
                    warmup_batches=3,
                    defer_artifact_hash=True,
                )

        artifact_info.assert_not_called()
        self.assertEqual(3, runner.call_args.args[-1])
        self.assertEqual(200, report["rss_peak_bytes"])
        self.assertEqual(100, report["rss_peak_delta_bytes"])
        self.assertEqual("deferred-with-stat-guard", report["artifact_integrity_timing"])

    def test_validation_replacement_during_inference_is_rejected(self):
        performance = {
            "batch_latencies_seconds": [0.01],
            "inference_seconds": 0.01,
            "load_seconds": 0.01,
            "rss_after_load_bytes": 0,
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            validation = root / "validation.jsonl"
            validation.write_text("before\n", encoding="utf-8")
            (root / "config.json").write_text("{}\n", encoding="utf-8")

            def mutate_validation(*_args, **_kwargs):
                validation.write_text("after-with-different-bytes\n", encoding="utf-8")
                return [["O"]], performance

            with (
                patch.object(
                    evaluate,
                    "load_validation",
                    return_value=[evaluate.ValidationRecord(["token"], ["O"])],
                ),
                patch.object(evaluate, "_run_onnx", side_effect=mutate_validation),
                patch.object(
                    evaluate,
                    "_artifact_stat_snapshot",
                    return_value=[
                        {
                            "path": "/tmp/model.onnx",
                            "bytes": 10,
                            "inode": 1,
                            "mtime_ns": 1,
                        }
                    ],
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "Validation modificata"):
                    evaluate.evaluate_variant(
                        name="onnx-test",
                        backend="onnx",
                        model_path=root / "model.onnx",
                        tokenizer_path=root,
                        validation_path=validation,
                        predictions_path=root / "predictions.jsonl",
                        report_path=root / "metrics.json",
                        defer_artifact_hash=True,
                    )


if __name__ == "__main__":
    unittest.main()
