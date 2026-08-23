"""Synthetic tests for repeatable, process-isolated service profiling."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from src.quantization.profile import (
    aggregate_profile_runs,
    build_profile_schedule,
    parse_smaps_rollup,
    run_profile_subprocess,
    sample_process_memory,
    summarize,
    write_profile_report,
)


class ProfileScheduleTests(unittest.TestCase):
    def test_balanced_schedule_is_deterministic_and_position_balanced(self):
        variants = ["torch-fp32", "onnx-int8-full", "onnx-int4"]
        first = build_profile_schedule(variants, repeats=6, seed=1234)
        second = build_profile_schedule(variants, repeats=6, seed=1234)

        self.assertEqual(first, second)
        self.assertEqual(18, len(first))
        for round_index in range(6):
            round_rows = [row for row in first if row["round"] == round_index]
            self.assertEqual(set(variants), {row["variant"] for row in round_rows})
            self.assertEqual({0, 1, 2}, {row["position"] for row in round_rows})

        positions = Counter((row["variant"], row["position"]) for row in first)
        self.assertTrue(all(count == 2 for count in positions.values()))

    def test_random_schedule_uses_seed_and_fixed_preserves_input_order(self):
        variants = ["a", "b", "c", "d"]
        random_first = build_profile_schedule(variants, 5, seed=11, order="random")
        random_second = build_profile_schedule(variants, 5, seed=11, order="random")
        random_other = build_profile_schedule(variants, 5, seed=12, order="random")
        fixed = build_profile_schedule(variants, 2, seed=999, order="fixed")

        self.assertEqual(random_first, random_second)
        self.assertNotEqual(random_first, random_other)
        self.assertEqual(
            variants,
            [row["variant"] for row in fixed if row["round"] == 0],
        )
        self.assertEqual(
            variants,
            [row["variant"] for row in fixed if row["round"] == 1],
        )

    def test_schedule_rejects_invalid_arguments(self):
        for variants, repeats, order in (
            ([], 1, "balanced"),
            (["a", "a"], 1, "balanced"),
            (["a"], 0, "balanced"),
            (["a"], 1, "unknown"),
        ):
            with self.subTest(variants=variants, repeats=repeats, order=order):
                with self.assertRaises(ValueError):
                    build_profile_schedule(variants, repeats, seed=1, order=order)


class MemorySamplingTests(unittest.TestCase):
    def test_parse_smaps_rollup_returns_bytes_and_uss(self):
        parsed = parse_smaps_rollup(
            """
00400000-00401000 r--p 00000000 00:00 0 [rollup]
Rss:                100 kB
Pss:                 80 kB
Private_Clean:       11 kB
Private_Dirty:       29 kB
Private_Hugetlb:      5 kB
Anonymous:           31 kB
Swap:                 3 kB
Shared_Clean:        60 kB
"""
        )

        self.assertEqual(100 * 1024, parsed["rss_bytes"])
        self.assertEqual(80 * 1024, parsed["pss_bytes"])
        self.assertEqual(45 * 1024, parsed["uss_bytes"])
        self.assertEqual(5 * 1024, parsed["private_hugetlb_bytes"])
        self.assertEqual(31 * 1024, parsed["anonymous_bytes"])
        self.assertEqual(3 * 1024, parsed["swap_bytes"])

    def test_parse_smaps_rollup_preserves_unavailable_metrics_as_none(self):
        parsed = parse_smaps_rollup("Rss: 7 kB\nPrivate_Dirty: malformed kB\n")

        self.assertEqual(7 * 1024, parsed["rss_bytes"])
        self.assertIsNone(parsed["pss_bytes"])
        self.assertIsNone(parsed["uss_bytes"])
        self.assertIsNone(parsed["anonymous_bytes"])

    def test_sample_current_process_returns_rss_when_supported(self):
        sampled = sample_process_memory(os.getpid())
        if sampled["rss_bytes"] is None:
            self.skipTest("RSS del processo non disponibile su questa piattaforma")
        self.assertGreater(sampled["rss_bytes"], 0)

    def test_profile_subprocess_captures_output_and_external_peak(self):
        command = [
            sys.executable,
            "-c",
            (
                "import time; payload=bytearray(8*1024*1024); "
                "print('child-ready', flush=True); time.sleep(0.10)"
            ),
        ]
        result = run_profile_subprocess(command, Path.cwd(), sample_interval_ms=5)

        self.assertEqual(0, result["returncode"])
        self.assertEqual("ok", result["status"])
        self.assertIn("child-ready", result["stdout"])
        self.assertGreater(result["memory"]["samples"], 0)
        if result["memory"]["peak"]["rss_bytes"] is None:
            self.skipTest("RSS del child non disponibile su questa piattaforma")
        self.assertGreater(result["memory"]["peak"]["rss_bytes"], 0)


class StatisticsTests(unittest.TestCase):
    def test_summarize_ignores_none_and_computes_interpolated_percentiles(self):
        summary = summarize([1, None, 2, 3, 4])

        self.assertEqual(4, summary["count"])
        self.assertEqual(1.0, summary["min"])
        self.assertAlmostEqual(1.15, summary["p05"])
        self.assertEqual(2.5, summary["median"])
        self.assertEqual(2.5, summary["mean"])
        self.assertAlmostEqual(3.85, summary["p95"])
        self.assertEqual(4.0, summary["max"])
        self.assertAlmostEqual(1.118033988749895, summary["stdev"])
        self.assertAlmostEqual(summary["stdev"] / 2.5, summary["cv"])

    def test_summarize_empty_and_zero_mean(self):
        empty = summarize([None, None])
        zero_mean = summarize([-1, 1])

        self.assertEqual(0, empty["count"])
        self.assertTrue(all(value is None for key, value in empty.items() if key != "count"))
        self.assertIsNone(zero_mean["cv"])

    def test_summarize_rejects_non_finite_and_non_numeric_values(self):
        with self.assertRaises(ValueError):
            summarize([float("nan")])
        with self.assertRaises(TypeError):
            summarize(["1"])  # type: ignore[list-item]


class ProfileAggregationTests(unittest.TestCase):
    variants = ["torch-fp32", "onnx-int8-full"]
    repeats = 2

    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.suite_id = "suite-current"
        cls.suite_root = (
            Path(cls.temporary.name) / "service-profile" / "suites" / cls.suite_id
        ).resolve()
        cls.suite_root.mkdir(parents=True)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    @classmethod
    def methodology(cls, **overrides):
        return {
            "suite_id": cls.suite_id,
            "suite_root": str(cls.suite_root),
            "order": "fixed",
            **overrides,
        }

    @classmethod
    def make_runs(cls):
        schedule = build_profile_schedule(
            cls.variants, cls.repeats, seed=7, order="fixed"
        )
        runs = []
        for row in schedule:
            variant = str(row["variant"])
            round_index = int(row["round"])
            base = 1 if variant == "torch-fp32" else 2
            prediction_path = (
                cls.suite_root
                / "predictions"
                / f"round-{round_index + 1:03d}"
                / f"{int(row['position']) + 1:02d}-{variant}.jsonl"
            )
            prediction_path.parent.mkdir(parents=True, exist_ok=True)
            prediction_path.write_text(
                json.dumps({"variant": variant}, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            runs.append(
                {
                    **row,
                    "suite_id": cls.suite_id,
                    "status": "ok",
                    "returncode": 0,
                    "configuration": {
                        "documents": 16,
                        "batch_size": 1,
                        "threads": 2,
                        "warmup_batches": 3,
                    },
                    "predictions_path": str(prediction_path.resolve()),
                    "prediction_digest": hashlib.sha256(
                        prediction_path.read_bytes()
                    ).hexdigest(),
                    "artifact_integrity": {
                        "files": [
                            {
                                "path": f"/{variant}/model",
                                "sha256": f"hash-{variant}",
                            }
                        ]
                    },
                    "load_seconds": base + round_index,
                    "documents_per_second": 10 * base + round_index,
                    "request_latency_seconds": {
                        "p95": 0.010 * base + round_index / 1000
                    },
                    "model_latency_seconds": {
                        "p95": 0.008 * base + round_index / 1000
                    },
                    "memory": {
                        "peak": {
                            "rss_bytes": (100 * base + round_index) * 1024**2,
                            "pss_bytes": (80 * base + round_index) * 1024**2,
                            "uss_bytes": (60 * base + round_index) * 1024**2,
                            "anonymous_bytes": (
                                None
                                if variant == "torch-fp32"
                                else (50 * base + round_index) * 1024**2
                            ),
                        }
                    },
                }
            )
        return runs

    def test_aggregate_validates_and_summarizes_each_variant(self):
        report = aggregate_profile_runs(
            self.make_runs(),
            self.variants,
            self.repeats,
            self.methodology(
                seed=7,
                cache_policy="uncontrolled-os-page-cache",
            ),
        )

        self.assertEqual("cpu-service-profile", report["kind"])
        self.assertEqual(2, report["schema_version"])
        self.assertEqual(self.suite_id, report["suite_id"])
        self.assertEqual(4, report["validation"]["actual_runs"])
        self.assertTrue(report["validation"]["prediction_files_verified"])
        torch = report["variants"]["torch-fp32"]
        self.assertEqual(2, torch["runs"])
        self.assertEqual(1.5, torch["statistics"]["load_seconds"]["median"])
        self.assertEqual(
            100.5 * 1024**2,
            torch["statistics"]["peak_rss_bytes"]["median"],
        )
        self.assertEqual(
            0,
            torch["statistics"]["peak_anonymous_bytes"]["count"],
        )
        self.assertEqual(
            2,
            report["variants"]["onnx-int8-full"]["statistics"][
                "peak_anonymous_bytes"
            ]["count"],
        )
        self.assertEqual(
            [(0, 0), (0, 1), (1, 0), (1, 1)],
            [(row["round"], row["position"]) for row in report["runs"]],
        )

    def test_aggregate_rejects_configuration_digest_and_hash_drift(self):
        mutations = []

        config_drift = self.make_runs()
        config_drift[-1]["configuration"]["threads"] = 3
        mutations.append((config_drift, "Configurazione"))

        digest_drift = self.make_runs()
        digest_drift[-1]["prediction_digest"] = "different"
        mutations.append((digest_drift, "digest"))

        hash_drift = self.make_runs()
        hash_drift[-1]["artifact_integrity"]["files"][0]["sha256"] = "different"
        mutations.append((hash_drift, "Hash"))

        for runs, message in mutations:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    aggregate_profile_runs(
                        runs,
                        self.variants,
                        self.repeats,
                        self.methodology(),
                    )

    def test_aggregate_rejects_missing_or_duplicate_schedule_slots(self):
        missing = self.make_runs()[:-1]
        with self.assertRaisesRegex(ValueError, "Attesi"):
            aggregate_profile_runs(
                missing, self.variants, self.repeats, self.methodology()
            )

        duplicate = self.make_runs()
        duplicate[-1]["round"] = duplicate[0]["round"]
        duplicate[-1]["position"] = duplicate[0]["position"]
        with self.assertRaisesRegex(ValueError, "duplicato"):
            aggregate_profile_runs(
                duplicate, self.variants, self.repeats, self.methodology()
            )

    def test_aggregate_hashes_prediction_files_and_rejects_stale_content(self):
        runs = self.make_runs()
        prediction_path = Path(runs[-1]["predictions_path"])
        original = prediction_path.read_bytes()
        prediction_path.write_bytes(original + b"stale\n")
        try:
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                aggregate_profile_runs(
                    runs,
                    self.variants,
                    self.repeats,
                    self.methodology(),
                )
        finally:
            prediction_path.write_bytes(original)

    def test_aggregate_rejects_cross_suite_or_reused_prediction_file(self):
        cross_suite = self.make_runs()
        cross_suite[-1]["suite_id"] = "suite-old"
        with self.assertRaisesRegex(ValueError, "suite diversa"):
            aggregate_profile_runs(
                cross_suite,
                self.variants,
                self.repeats,
                self.methodology(),
            )

        reused = self.make_runs()
        reused[-1]["predictions_path"] = reused[1]["predictions_path"]
        reused[-1]["prediction_digest"] = reused[1]["prediction_digest"]
        with self.assertRaisesRegex(ValueError, "riutilizzato"):
            aggregate_profile_runs(
                reused,
                self.variants,
                self.repeats,
                self.methodology(),
            )

    def test_write_profile_report_creates_json_and_markdown(self):
        report = aggregate_profile_runs(
            self.make_runs(),
            self.variants,
            self.repeats,
            self.methodology(
                seed=7,
                cache_policy="uncontrolled-os-page-cache",
            ),
        )
        with tempfile.TemporaryDirectory() as temporary:
            written = write_profile_report(report, Path(temporary))
            loaded = json.loads(written["json"].read_text(encoding="utf-8"))
            markdown = written["markdown"].read_text(encoding="utf-8")

        self.assertEqual("cpu-service-profile", loaded["kind"])
        self.assertIn("onnx-int8-full", markdown)
        self.assertIn("uncontrolled-os-page-cache", markdown)

    def test_aggregate_does_not_mutate_input_runs(self):
        runs = self.make_runs()
        before = copy.deepcopy(runs)
        aggregate_profile_runs(
            runs, self.variants, self.repeats, self.methodology()
        )
        self.assertEqual(before, runs)


if __name__ == "__main__":
    unittest.main()
