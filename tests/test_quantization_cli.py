"""Lightweight CLI orchestration tests without loading ML dependencies."""

from __future__ import annotations

import argparse
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import call, patch

from src.quantization import cli


class QuantizationCliTests(unittest.TestCase):
    def test_all_backends_use_the_canonical_checkpoint_preprocessor(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = cli._layout(Path(temporary))
            definitions = cli._variant_definitions(paths)

        self.assertTrue(definitions)
        self.assertTrue(
            all(
                tokenizer_path == paths["checkpoint"]
                for _, _, tokenizer_path in definitions.values()
            )
        )

    def test_profile_repeats_zero_fails_instead_of_using_the_default(self):
        args = cli.build_parser().parse_args(
            ["profile", "--variants", "onnx-fp32", "--repeats", "0"]
        )
        with self.assertRaisesRegex(SystemExit, "--repeats deve essere positivo"):
            cli.command_profile(args, {}, {"root": Path("unused")})

    def test_profile_rejects_more_documents_than_the_validation(self):
        args = cli.build_parser().parse_args(
            ["profile", "--variants", "onnx-fp32", "--documents", "7001"]
        )
        config = {"dataset": {"validation": {"expected_rows": 7000}}}
        with self.assertRaisesRegex(SystemExit, "supera la validation"):
            cli.command_profile(args, config, {"root": Path("unused")})

    def test_all_propagates_custom_config_to_every_phase(self):
        config_path = Path("custom-quantization.json")
        args = argparse.Namespace(
            config=config_path,
            opset=18,
            torch_only=False,
            batch_size=16,
            threads=4,
            limit=None,
            enforce_gates=False,
        )
        with (
            patch.object(cli, "command_sources") as sources,
            patch.object(cli, "command_export") as export,
            patch.object(cli, "command_quantize") as quantize,
            patch.object(cli, "command_regress") as regress,
        ):
            cli.command_all(args, {"schema_version": 1}, {"root": Path("artifacts")})

        self.assertEqual(config_path, sources.call_args.args[0].config)
        self.assertEqual(config_path, export.call_args.args[0].config)
        self.assertEqual(config_path, quantize.call_args.args[0].config)
        self.assertEqual(config_path, regress.call_args.args[0].config)

    def test_artifact_integrity_requires_evaluation_time_hash_or_explicit_backfill(self):
        current = {
            "model": "/tmp/model.onnx",
            "bytes": 123,
            "files": [{"path": "/tmp/model.onnx", "bytes": 123, "sha256": "abc"}],
            "op_counts": {"MatMulNBits": 90},
        }
        report = {
            "variant": "onnx-int8-full",
            "backend": "onnx",
            "model_path": "/tmp/model.onnx",
        }
        with patch("src.quantization.quantize.artifact_info", return_value=current):
            with self.assertRaises(SystemExit):
                cli._attach_artifact_integrity([report])

            self.assertEqual(
                ["onnx-int8-full"],
                cli._attach_artifact_integrity([report], allow_backfill=True),
            )
            self.assertEqual(current, report["artifact_integrity"])
            self.assertEqual("backfilled-after-evaluation", report["artifact_integrity_timing"])
            self.assertTrue(report["artifact_integrity_current_verified"])
            self.assertEqual([], cli._attach_artifact_integrity([report]))

            report["artifact_integrity"] = {**current, "bytes": 124}
            with self.assertRaises(SystemExit):
                cli._attach_artifact_integrity([report])

    def test_single_int8_full_builds_missing_weight_base_first(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = cli._layout(Path(temporary))
            args = argparse.Namespace(config=None, variant="int8-full")

            def fake_quantize(name, _config, _paths):
                return {"variant": name, "status": "ok"}

            with (
                patch("src.quantization.sources.verify_sources", return_value={"status": "ok"}),
                patch.object(cli, "_quantize_one", side_effect=fake_quantize) as quantize,
            ):
                cli.command_quantize(args, {"runtime": {}}, paths)

        self.assertEqual(
            [call("onnx-int8-weight", {"runtime": {}}, paths), call("onnx-int8-full", {"runtime": {}}, paths)],
            quantize.call_args_list,
        )

    def test_single_int8_full_rebuilds_a_stale_weight_base(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = cli._layout(Path(temporary))
            base_dir = paths["onnx-int8-weight"]
            fp32_dir = paths["onnx-fp32"]
            base_dir.mkdir(parents=True)
            fp32_dir.mkdir(parents=True)
            (base_dir / "model.onnx").touch()
            (fp32_dir / "model.onnx").touch()
            cli._write_json(
                base_dir / "quantization-report.json",
                {"source": {"bytes": 1}, "artifact": {"bytes": 2}},
            )
            args = argparse.Namespace(config=None, variant="int8-full")

            with (
                patch("src.quantization.sources.verify_sources", return_value={"status": "ok"}),
                patch("src.quantization.quantize.artifact_info", return_value={"bytes": 3}),
                patch.object(
                    cli,
                    "_quantize_one",
                    side_effect=lambda name, _config, _paths: {"variant": name, "status": "ok"},
                ) as quantize,
            ):
                cli.command_quantize(args, {"runtime": {}}, paths)

        self.assertEqual(
            ["onnx-int8-weight", "onnx-int8-full"],
            [item.args[0] for item in quantize.call_args_list],
        )

    def test_single_int8_full_reuses_a_verified_weight_base(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = cli._layout(Path(temporary))
            base_dir = paths["onnx-int8-weight"]
            fp32_dir = paths["onnx-fp32"]
            base_dir.mkdir(parents=True)
            fp32_dir.mkdir(parents=True)
            base_model = base_dir / "model.onnx"
            fp32_model = fp32_dir / "model.onnx"
            base_model.touch()
            fp32_model.touch()
            base_info = {"model": str(base_model.resolve()), "bytes": 2}
            source_info = {"model": str(fp32_model.resolve()), "bytes": 1}
            cli._write_json(
                base_dir / "quantization-report.json",
                {"source": source_info, "artifact": base_info},
            )
            args = argparse.Namespace(config=None, variant="int8-full")

            def fake_info(path):
                return base_info if Path(path) == base_model else source_info

            with (
                patch("src.quantization.sources.verify_sources", return_value={"status": "ok"}),
                patch("src.quantization.quantize.artifact_info", side_effect=fake_info),
                patch.object(
                    cli,
                    "_quantize_one",
                    side_effect=lambda name, _config, _paths: {"variant": name, "status": "ok"},
                ) as quantize,
            ):
                cli.command_quantize(args, {"runtime": {}}, paths)

        self.assertEqual(["onnx-int8-full"], [item.args[0] for item in quantize.call_args_list])

    def test_all_quantizes_each_variant_once(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = cli._layout(Path(temporary))
            args = argparse.Namespace(config=None, variant="all")

            with (
                patch("src.quantization.sources.verify_sources", return_value={"status": "ok"}),
                patch.object(
                    cli,
                    "_quantize_one",
                    side_effect=lambda name, _config, _paths: {"variant": name, "status": "ok"},
                ) as quantize,
            ):
                cli.command_quantize(args, {"runtime": {}}, paths)

        self.assertEqual(
            [
                "onnx-int8",
                "onnx-int8-weight",
                "onnx-int8-full",
                "onnx-int4-a32",
                "onnx-int4",
            ],
            [item.args[0] for item in quantize.call_args_list],
        )

    def test_profile_rebuild_rehashes_the_referenced_prediction_file(self):
        from src.quantization.profile import (
            aggregate_profile_runs,
            write_profile_report,
        )

        with tempfile.TemporaryDirectory() as temporary:
            paths = cli._layout(Path(temporary))
            output_dir = paths["root"] / "service-profile"
            suite_id = "suite-rebuild"
            suite_root = output_dir / "suites" / suite_id
            prediction_path = suite_root / "predictions" / "round-001" / "01-onnx-fp32.jsonl"
            prediction_path.parent.mkdir(parents=True)
            prediction_path.write_text('{"id":0,"labels":["O"]}\n', encoding="utf-8")
            digest = hashlib.sha256(prediction_path.read_bytes()).hexdigest()

            model_path = paths["onnx-fp32"] / "model.onnx"
            model_path.parent.mkdir(parents=True)
            model_path.touch()
            artifact = {
                "model": str(model_path.resolve()),
                "bytes": 0,
                "files": [
                    {
                        "path": str(model_path.resolve()),
                        "bytes": 0,
                        "sha256": "model-hash",
                    }
                ],
            }
            run = {
                "variant": "onnx-fp32",
                "suite_id": suite_id,
                "round": 0,
                "position": 0,
                "status": "ok",
                "returncode": 0,
                "configuration": {
                    "documents": 1,
                    "batch_size": 1,
                    "threads": 1,
                    "warmup_batches": 0,
                },
                "predictions_path": str(prediction_path.resolve()),
                "prediction_digest": digest,
                "artifact_integrity": artifact,
                "load_seconds": 0.1,
                "documents_per_second": 10.0,
                "request_latency_seconds": {"p95": 0.01},
                "model_latency_seconds": {"p95": 0.009},
                "memory": {
                    "peak": {
                        "rss_bytes": 1024,
                        "pss_bytes": 900,
                        "uss_bytes": 800,
                        "anonymous_bytes": 700,
                    }
                },
            }
            methodology = {
                "suite_id": suite_id,
                "suite_root": str(suite_root.resolve()),
                "order": "fixed",
            }
            report = aggregate_profile_runs(
                [run], ["onnx-fp32"], 1, methodology
            )
            source_state = {"status": "verified", "lock_sha256": "source-hash"}
            report["sources_preflight"] = source_state
            report["sources_postflight"] = source_state
            write_profile_report(report, suite_root)
            write_profile_report(report, output_dir)

            prediction_path.write_text(
                prediction_path.read_text(encoding="utf-8") + "stale\n",
                encoding="utf-8",
            )
            args = argparse.Namespace(config=None, variants=None)
            with (
                patch(
                    "src.quantization.sources.verify_sources",
                    return_value=source_state,
                ),
                patch(
                    "src.quantization.quantize.artifact_info",
                    return_value=artifact,
                ),
            ):
                with self.assertRaisesRegex(SystemExit, "SHA-256"):
                    cli._rebuild_profile_report(args, paths)

    def test_regression_verifies_sources_before_and_after_the_suite(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = cli._layout(Path(temporary))
            args = argparse.Namespace(
                config=None,
                variants=["torch-fp32", "onnx-fp32"],
                batch_size=1,
                threads=1,
                limit=1,
                enforce_gates=False,
            )
            config = {
                "runtime": {"batch_size": 1, "max_length": 8, "threads": 1},
                "dataset": {"validation": {"expected_rows": 7000}},
                "gates": {},
            }
            source_state = {"status": "verified", "lock_sha256": "same"}

            def fake_evaluate(**kwargs):
                return {"variant": kwargs["name"], "status": "ok"}

            with (
                patch.object(cli, "_require", side_effect=lambda path, _label: path),
                patch.object(cli, "_attach_artifact_integrity"),
                patch(
                    "src.quantization.sources.verify_sources",
                    side_effect=[source_state, source_state],
                ) as verify,
                patch(
                    "src.quantization.benchmark.evaluate_isolated",
                    side_effect=fake_evaluate,
                ),
                patch(
                    "src.quantization.benchmark.build_regression_report",
                    return_value={"comparisons": {}},
                ) as build,
            ):
                cli.command_regress(args, config, paths)

        self.assertEqual(2, verify.call_count)
        guarded = build.call_args.kwargs["sources_verification"]
        self.assertTrue(guarded["evaluation_guarded"])
        self.assertEqual("pre-and-post-regression", guarded["verification_timing"])

    def test_regression_rejects_source_drift_during_the_suite(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = cli._layout(Path(temporary))
            args = argparse.Namespace(
                config=None,
                variants=["torch-fp32"],
                batch_size=1,
                threads=1,
                limit=1,
                enforce_gates=False,
            )
            config = {
                "runtime": {"batch_size": 1, "max_length": 8, "threads": 1},
                "dataset": {"validation": {"expected_rows": 7000}},
                "gates": {},
            }

            with (
                patch.object(cli, "_require", side_effect=lambda path, _label: path),
                patch.object(cli, "_attach_artifact_integrity"),
                patch(
                    "src.quantization.sources.verify_sources",
                    side_effect=[
                        {"status": "verified", "lock_sha256": "before"},
                        {"status": "verified", "lock_sha256": "after"},
                    ],
                ),
                patch(
                    "src.quantization.benchmark.evaluate_isolated",
                    return_value={"variant": "torch-fp32", "status": "ok"},
                ),
                patch(
                    "src.quantization.benchmark.build_regression_report"
                ) as build,
            ):
                with self.assertRaisesRegex(SystemExit, "Sorgenti modificate"):
                    cli.command_regress(args, config, paths)

        build.assert_not_called()


if __name__ == "__main__":
    unittest.main()
