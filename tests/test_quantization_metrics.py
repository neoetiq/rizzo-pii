"""Test unitari del nucleo di regressione, senza torch/transformers."""

from __future__ import annotations

import hashlib
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.quantization.benchmark import build_regression_report
from src.quantization.evaluate import preprocessor_integrity, write_predictions
from src.quantization.metrics import (
    apply_quality_gate,
    compare_predictions,
    compute_metrics,
    compute_span_metrics,
    normalize_labels,
    spans,
)


class QuantizationMetricsTests(unittest.TestCase):
    @staticmethod
    def _sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    @classmethod
    def _integrity(cls, path: Path) -> dict[str, object]:
        return {
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": cls._sha256(path),
        }

    def test_normalize_labels_matches_training_taxonomy(self):
        self.assertEqual(
            [
                "B-FULLNAME", "I-FULLNAME", "O", "B-FULLNAME", "O",
                "B-PIVA", "I-PIVA",
            ],
            normalize_labels([
                "B-GIVENNAME", "I-SURNAME", "B-TITLE", "I-FULLNAME", "O",
                "I-TAXNUM", "I-TAXNUM",
            ]),
        )

    def test_spans_recovers_invalid_i_tag_and_splits_on_type_change(self):
        self.assertEqual(
            {("NAME", 0, 2), ("EMAIL", 2, 3), ("EMAIL", 3, 5)},
            spans(["I-NAME", "I-NAME", "I-EMAIL", "B-EMAIL", "I-EMAIL"]),
        )

    def test_compute_metrics_is_exact_entity_level_with_micro_macro_and_tags(self):
        gold = [
            ["B-NAME", "I-NAME", "O", "B-EMAIL"],
            ["O", "B-CF", "I-CF"],
        ]
        prediction = [
            ["B-NAME", "I-NAME", "O", "B-PHONE"],
            ["O", "B-CF", "I-CF"],
        ]
        metrics = compute_metrics(gold, prediction)

        self.assertEqual({"tp": 2, "fp": 1, "fn": 1, "support": 3},
                         {key: metrics["micro"][key] for key in ("tp", "fp", "fn", "support")})
        self.assertAlmostEqual(2 / 3, metrics["micro"]["precision"])
        self.assertAlmostEqual(2 / 3, metrics["micro"]["recall"])
        self.assertAlmostEqual(2 / 3, metrics["micro"]["f1"])
        self.assertAlmostEqual(0.5, metrics["macro"]["f1"])
        self.assertAlmostEqual(0.5, metrics["macro_f1"])
        self.assertEqual(0, metrics["per_tag"]["PHONE"]["support"])
        self.assertEqual(1, metrics["per_tag"]["PHONE"]["fp"])

    def test_compare_predictions_reports_agreement_new_and_removed_spans(self):
        fp32 = [
            ["B-NAME", "I-NAME", "O", "B-EMAIL"],
            ["B-CF"],
        ]
        int8 = [
            ["B-NAME", "I-NAME", "O", "B-PHONE"],
            ["B-CF"],
        ]
        comparison = compare_predictions(fp32, int8)["agreement"]

        self.assertEqual(1, comparison["exact_documents"])
        self.assertEqual(0.5, comparison["exact_document_rate"])
        self.assertEqual(
            {"golden": 3, "candidate": 3, "agreed": 2, "new": 1, "removed": 1},
            comparison["spans"],
        )
        self.assertEqual(1, comparison["by_tag"]["PHONE"]["new"])
        self.assertEqual(1, comparison["by_tag"]["EMAIL"]["removed"])

    def test_quality_gate_detects_global_and_per_tag_regression(self):
        gold = [["B-NAME", "O", "B-EMAIL"], ["B-CF"]]
        fp32 = [["B-NAME", "O", "B-EMAIL"], ["B-CF"]]
        int4 = [["B-NAME", "O", "B-PHONE"], ["B-CF"]]
        result = apply_quality_gate(
            compute_metrics(gold, fp32),
            compute_metrics(gold, int4),
            "int4",
            {"max_micro_f1_drop": 0.10, "min_per_tag_f1": {"EMAIL": 0.90}},
        )

        self.assertFalse(result["passed"])
        self.assertEqual({"max_micro_f1_drop", "min_per_tag_f1"},
                         {failure["gate"] for failure in result["failures"]})
        self.assertEqual("EMAIL", next(
            failure["tag"] for failure in result["failures"]
            if failure["gate"] == "min_per_tag_f1"
        ))

    def test_quality_gate_can_protect_critical_tag_recall(self):
        gold = [["B-CF"], ["B-PIVA"], ["B-IBAN"]]
        fp32 = [["B-CF"], ["B-PIVA"], ["B-IBAN"]]
        candidate = [["O"], ["B-PIVA"], ["B-IBAN"]]
        result = apply_quality_gate(
            compute_metrics(gold, fp32),
            compute_metrics(gold, candidate),
            "int8",
            {"max_per_tag_recall_drop": {"CF": 0.005}},
        )

        self.assertFalse(result["passed"])
        self.assertEqual("CF", result["failures"][0]["tag"])
        self.assertEqual("max_per_tag_recall_drop", result["failures"][0]["gate"])

    def test_misaligned_documents_fail_fast(self):
        with self.assertRaisesRegex(ValueError, "Numero di documenti diverso"):
            compute_metrics([["O"]], [])
        with self.assertRaisesRegex(ValueError, "Documento 0"):
            compare_predictions([["O"]], [["O", "O"]])

    def test_report_marks_limited_run_as_smoke_and_uses_symmetric_agreement(self):
        golden = [["B-CF", "O"]]
        candidate = [["B-CF", "B-EMAIL"]]
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            golden_path = root / "golden.jsonl"
            candidate_path = root / "candidate.jsonl"
            validation_path = root / "validation.jsonl"
            validation_path.write_text("{}\n", encoding="utf-8")
            write_predictions(golden_path, golden)
            write_predictions(candidate_path, candidate)
            reports = [
                {
                    "variant": "torch-fp32",
                    "status": "ok",
                    "documents": 1,
                    "predictions_path": str(golden_path),
                    "validation_path": str(validation_path),
                    "metrics": compute_metrics(golden, golden),
                },
                {
                    "variant": "onnx-fp32",
                    "status": "ok",
                    "documents": 1,
                    "predictions_path": str(candidate_path),
                    "validation_path": str(validation_path),
                    "metrics": compute_metrics(golden, candidate),
                },
            ]
            report = build_regression_report(
                variant_reports=reports,
                output_dir=root / "report",
                gates_config={"onnx-fp32": {"min_entity_agreement": 0.9}},
                expected_documents=7000,
            )
            markdown = (root / "report" / "report.md").read_text(encoding="utf-8")

        comparison = report["comparisons"]["onnx-fp32"]
        self.assertEqual("smoke", report["validation_scope"]["kind"])
        self.assertFalse(comparison["release_eligible"])
        self.assertAlmostEqual(2 / 3, comparison["quality_gate"]["entity_agreement_f1"])
        self.assertFalse(comparison["quality_gate"]["passed"])
        self.assertIn("char micro-F1", markdown)
        self.assertIn("Sono model-only", markdown)

    def test_report_rejects_a_modified_prediction_file(self):
        labels = [["B-CF"]]
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            validation = root / "validation.jsonl"
            validation.write_text("{}\n", encoding="utf-8")
            predictions = root / "predictions.jsonl"
            write_predictions(predictions, labels, [{("CF", 0, 16)}])
            report = {
                "variant": "torch-fp32",
                "status": "ok",
                "documents": 1,
                "predictions_path": str(predictions),
                "prediction_digest": self._sha256(predictions),
                "validation_path": str(validation),
                "validation_integrity": self._integrity(validation),
                "metrics": {
                    **compute_metrics(labels, labels),
                    "production_like_char": compute_span_metrics(
                        [{("CF", 0, 16)}], [{("CF", 0, 16)}]
                    ),
                },
            }
            predictions.write_text('{"id":0,"labels":["O"],"spans":[],"char_spans":[]}\n', encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "Digest predizioni diverso"):
                build_regression_report(
                    variant_reports=[report],
                    output_dir=root / "report",
                    gates_config={},
                    expected_documents=1,
                )

    def test_report_rejects_validation_bytes_different_from_evaluation(self):
        labels = [["O"]]
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            validation = root / "validation.jsonl"
            validation.write_text("before\n", encoding="utf-8")
            recorded_validation = self._integrity(validation)
            predictions = root / "predictions.jsonl"
            write_predictions(predictions, labels, [set()])
            validation.write_text("after\n", encoding="utf-8")
            report = {
                "variant": "torch-fp32",
                "status": "ok",
                "documents": 1,
                "predictions_path": str(predictions),
                "prediction_digest": self._sha256(predictions),
                "validation_path": str(validation),
                "validation_integrity": recorded_validation,
                "metrics": {
                    **compute_metrics(labels, labels),
                    "production_like_char": compute_span_metrics([set()], [set()]),
                },
            }

            with self.assertRaisesRegex(ValueError, "Validation diversa"):
                build_regression_report(
                    variant_reports=[report],
                    output_dir=root / "report",
                    gates_config={},
                    expected_documents=1,
                )

    def test_report_gates_persisted_char_span_fragmentation(self):
        word_labels = [["B-CF"]]
        gold_char = [{("CF", 0, 16)}]
        candidate_char = [{("CF", 0, 3), ("DOCID", 3, 16)}]
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            validation = root / "validation.jsonl"
            validation.write_text("{}\n", encoding="utf-8")
            validation_integrity = self._integrity(validation)
            preprocessor = root / "preprocessor"
            preprocessor.mkdir()
            (preprocessor / "config.json").write_text("{}\n", encoding="utf-8")
            (preprocessor / "tokenizer.json").write_text("{}\n", encoding="utf-8")
            preprocessor_manifest = preprocessor_integrity(preprocessor)
            golden_path = root / "golden.jsonl"
            candidate_path = root / "candidate.jsonl"
            write_predictions(golden_path, word_labels, gold_char)
            write_predictions(candidate_path, word_labels, candidate_char)

            def make_report(name: str, path: Path, char_spans):
                return {
                    "variant": name,
                    "status": "ok",
                    "documents": 1,
                    "predictions_path": str(path),
                    "prediction_digest": self._sha256(path),
                    "validation_path": str(validation),
                    "validation_integrity": validation_integrity,
                    "tokenizer_path": str(preprocessor),
                    "preprocessor_integrity": preprocessor_manifest,
                    "metrics": {
                        **compute_metrics(word_labels, word_labels),
                        "production_like_char": compute_span_metrics(
                            gold_char, char_spans
                        ),
                    },
                }

            report = build_regression_report(
                variant_reports=[
                    make_report("torch-fp32", golden_path, gold_char),
                    make_report("onnx-fp32", candidate_path, candidate_char),
                ],
                output_dir=root / "report",
                gates_config={
                    "onnx-fp32": {
                        "max_micro_f1_drop": 0.0,
                        "max_new_spans": 0,
                        "max_removed_spans": 0,
                    }
                },
                expected_documents=1,
                sources_verification={
                    "status": "verified",
                    "evaluation_guarded": True,
                },
            )

        comparison = report["comparisons"]["onnx-fp32"]
        self.assertEqual(
            1,
            comparison["prediction_regression"]["agreement"]["exact_documents"],
        )
        char_counts = comparison["production_like_prediction_regression"][
            "agreement"
        ]["spans"]
        self.assertEqual(2, char_counts["new"])
        self.assertEqual(1, char_counts["removed"])
        self.assertFalse(comparison["release_eligible"])
        self.assertIn(
            "production_like_char.max_new_spans",
            {item["gate"] for item in comparison["quality_gate"]["failures"]},
        )

    def test_report_rejects_tokenizer_or_label_map_changed_after_evaluation(self):
        labels = [["O"]]
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            validation = root / "validation.jsonl"
            validation.write_text("{}\n", encoding="utf-8")
            preprocessor = root / "preprocessor"
            preprocessor.mkdir()
            config_path = preprocessor / "config.json"
            config_path.write_text('{"id2label":{"0":"O"}}\n', encoding="utf-8")
            (preprocessor / "tokenizer.json").write_text("{}\n", encoding="utf-8")
            recorded_preprocessor = preprocessor_integrity(preprocessor)
            predictions = root / "predictions.jsonl"
            write_predictions(predictions, labels, [set()])
            config_path.write_text(
                '{"id2label":{"0":"B-CF"}}\n', encoding="utf-8"
            )
            report = {
                "variant": "torch-fp32",
                "status": "ok",
                "documents": 1,
                "predictions_path": str(predictions),
                "prediction_digest": self._sha256(predictions),
                "validation_path": str(validation),
                "validation_integrity": self._integrity(validation),
                "tokenizer_path": str(preprocessor),
                "preprocessor_integrity": recorded_preprocessor,
                "metrics": {
                    **compute_metrics(labels, labels),
                    "production_like_char": compute_span_metrics([set()], [set()]),
                },
            }

            with self.assertRaisesRegex(ValueError, "Tokenizer/config diversi"):
                build_regression_report(
                    variant_reports=[report],
                    output_dir=root / "report",
                    gates_config={},
                    expected_documents=1,
                )

    def test_release_is_possible_only_with_complete_quality_and_integrity(self):
        labels = [["B-CF"]]
        char_spans = [{("CF", 0, 16)}]
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            validation = root / "validation.jsonl"
            validation.write_text("{}\n", encoding="utf-8")
            validation_manifest = self._integrity(validation)
            preprocessor = root / "preprocessor"
            preprocessor.mkdir()
            (preprocessor / "config.json").write_text("{}\n", encoding="utf-8")
            (preprocessor / "tokenizer.json").write_text("{}\n", encoding="utf-8")
            preprocessor_manifest = preprocessor_integrity(preprocessor)

            reports = []
            for name, backend in (("torch-fp32", "torch"), ("onnx-fp32", "onnx")):
                predictions = root / f"{name}.jsonl"
                write_predictions(predictions, labels, char_spans)
                item = {
                    "variant": name,
                    "backend": backend,
                    "status": "ok",
                    "documents": 1,
                    "predictions_path": str(predictions),
                    "prediction_digest": self._sha256(predictions),
                    "validation_path": str(validation),
                    "validation_integrity": validation_manifest,
                    "tokenizer_path": str(preprocessor),
                    "preprocessor_integrity": preprocessor_manifest,
                    "metrics": {
                        **compute_metrics(labels, labels),
                        "production_like_char": compute_span_metrics(
                            char_spans, char_spans
                        ),
                    },
                }
                if backend == "onnx":
                    item["artifact_integrity"] = {"files": [{"sha256": "abc"}]}
                    item["artifact_integrity_timing"] = (
                        "post-evaluation-with-stat-guard"
                    )
                    item["artifact_integrity_current_verified"] = True
                reports.append(item)

            report = build_regression_report(
                variant_reports=reports,
                output_dir=root / "report",
                gates_config={
                    "onnx-fp32": {
                        "max_micro_f1_drop": 0.0,
                        "max_macro_f1_drop": 0.0,
                        "max_new_spans": 0,
                        "max_removed_spans": 0,
                    }
                },
                expected_documents=1,
                sources_verification={
                    "status": "verified",
                    "evaluation_guarded": True,
                },
            )

        comparison = report["comparisons"]["onnx-fp32"]
        self.assertTrue(comparison["quality_gate"]["passed"])
        self.assertTrue(comparison["release_eligible"])

    def test_missing_variant_quality_policy_is_fail_closed(self):
        labels = [["B-CF"]]
        char_spans = [{("CF", 0, 16)}]
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            validation = root / "validation.jsonl"
            validation.write_text("{}\n", encoding="utf-8")
            preprocessor = root / "preprocessor"
            preprocessor.mkdir()
            (preprocessor / "config.json").write_text("{}\n", encoding="utf-8")
            (preprocessor / "tokenizer.json").write_text("{}\n", encoding="utf-8")

            reports = []
            for name, backend in (("torch-fp32", "torch"), ("custom", "onnx")):
                predictions = root / f"{name}.jsonl"
                write_predictions(predictions, labels, char_spans)
                item = {
                    "variant": name,
                    "backend": backend,
                    "status": "ok",
                    "documents": 1,
                    "predictions_path": str(predictions),
                    "prediction_digest": self._sha256(predictions),
                    "validation_path": str(validation),
                    "validation_integrity": self._integrity(validation),
                    "tokenizer_path": str(preprocessor),
                    "preprocessor_integrity": preprocessor_integrity(preprocessor),
                    "metrics": {
                        **compute_metrics(labels, labels),
                        "production_like_char": compute_span_metrics(
                            char_spans, char_spans
                        ),
                    },
                }
                if backend == "onnx":
                    item.update(
                        {
                            "artifact_integrity": {"files": [{"sha256": "abc"}]},
                            "artifact_integrity_timing": "post-evaluation-with-stat-guard",
                            "artifact_integrity_current_verified": True,
                        }
                    )
                reports.append(item)

            report = build_regression_report(
                variant_reports=reports,
                output_dir=root / "report",
                gates_config={},
                expected_documents=1,
                sources_verification={
                    "status": "verified",
                    "evaluation_guarded": True,
                },
            )

        comparison = report["comparisons"]["custom"]
        self.assertFalse(comparison["quality_policy_complete"])
        self.assertFalse(comparison["release_eligible"])
        self.assertIn(
            "quality_policy.required_for_release",
            {item["gate"] for item in comparison["quality_gate"]["failures"]},
        )

    def test_non_ok_requested_variant_fails_instead_of_being_skipped(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            reports = [
                {"variant": "torch-fp32", "status": "ok"},
                {"variant": "onnx-int4", "status": "failed"},
            ]
            with self.assertRaisesRegex(ValueError, "onnx-int4"):
                build_regression_report(
                    variant_reports=reports,
                    output_dir=root,
                    gates_config={"onnx-int4": {"max_micro_f1_drop": 0.002}},
                    expected_documents=7000,
                )


if __name__ == "__main__":
    unittest.main()
