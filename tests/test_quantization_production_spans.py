"""Regression tests for the production-like, all-subword quality view."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from src.quantization import evaluate
from src.quantization.evaluate import (
    ValidationRecord,
    _production_like_char_predictions,
)
from src.quantization.metrics import (
    apply_quality_gate,
    canonical_text_and_word_offsets,
    compare_span_predictions,
    compute_metrics,
    compute_span_metrics,
    normalize_labels,
    simple_char_spans,
    word_spans_to_char_spans,
)


class ProductionLikeSpanTests(unittest.TestCase):
    def test_quantization_taxonomy_matches_v15_docid_mappings(self):
        self.assertEqual(
            ["B-DOCID", "I-DOCID", "I-DOCID", "I-DOCID"],
            normalize_labels(
                ["B-CIG", "B-CUP", "B-POLIZZA", "B-MATRICOLA"]
            ),
        )

    def test_simple_aggregation_keeps_all_subwords_for_critical_entities(self):
        text = (
            "Asia Argento RSSMRA85H12F205Z "
            "IT60X0542811101000000123456 mario.rossi@example.it"
        )
        full_start = text.index("Asia")
        full_end = text.index("Argento") + len("Argento")
        cf_start = text.index("RSSMRA85H12F205Z")
        cf_end = cf_start + len("RSSMRA85H12F205Z")
        iban_start = text.index("IT60X0542811101000000123456")
        iban_end = iban_start + len("IT60X0542811101000000123456")
        email_start = text.index("mario.rossi@example.it")
        email_end = email_start + len("mario.rossi@example.it")

        labels = [
            "O",
            "B-FULLNAME", "I-FULLNAME", "I-FULLNAME", "I-FULLNAME",
            "B-CF", "I-CF", "I-CF",
            "B-IBAN", "I-IBAN", "I-IBAN",
            "B-EMAIL", "I-EMAIL", "I-EMAIL", "I-EMAIL",
            "O",
        ]
        offsets = [
            (0, 0),
            (full_start, full_start + 2),
            (full_start + 2, full_start + 4),
            (text.index("Argento"), text.index("Argento") + 3),
            (text.index("Argento") + 3, full_end),
            (cf_start, cf_start + 3),
            (cf_start + 3, cf_start + 10),
            (cf_start + 10, cf_end),
            (iban_start, iban_start + 4),
            (iban_start + 4, iban_start + 15),
            (iban_start + 15, iban_end),
            (email_start, email_start + 5),
            (email_start + 5, email_start + 6),
            (email_start + 6, email_start + 19),
            (email_start + 19, email_end),
            (0, 0),
        ]
        spans = simple_char_spans(
            labels,
            offsets,
            special_tokens_mask=[1, *([0] * 14), 1],
        )

        self.assertEqual(
            {
                ("FULLNAME", full_start, full_end),
                ("CF", cf_start, cf_end),
                ("IBAN", iban_start, iban_end),
                ("EMAIL", email_start, email_end),
            },
            spans,
        )

    def test_pretokenized_local_offsets_are_projected_to_canonical_text(self):
        record = ValidationRecord(
            ["Asia", "Argento", "RSSMRA85H12F205Z"],
            ["B-FULLNAME", "I-FULLNAME", "B-CF"],
        )
        text, word_offsets = canonical_text_and_word_offsets(record.tokens)
        self.assertEqual("Asia Argento RSSMRA85H12F205Z", text)
        self.assertEqual([(0, 4), (5, 12), (13, 29)], word_offsets)

        predicted = _production_like_char_predictions(
            token_predictions=[[0, 1, 2, 2, 3, 4, 0]],
            offsets_by_row=[[
                (0, 0), (0, 4), (0, 3), (3, 7), (0, 3), (3, 16), (0, 0)
            ]],
            special_tokens_masks=[[1, 0, 0, 0, 0, 0, 1]],
            word_ids_by_row=[[None, 0, 1, 1, 2, 2, None]],
            records=[record],
            id2label={
                0: "O",
                1: "B-FULLNAME",
                2: "I-FULLNAME",
                3: "B-CF",
                4: "I-CF",
            },
        )
        self.assertEqual(
            [{("FULLNAME", 0, 12), ("CF", 13, 29)}],
            predicted,
        )
        self.assertEqual(
            predicted[0],
            word_spans_to_char_spans(record.labels, word_offsets),
        )

    def test_quality_gate_catches_fragmentation_hidden_by_first_subword_metric(self):
        # La metrica storica vede una sola parola e quindi solo B-CF: passa.
        word_gold = [["B-CF"]]
        baseline = compute_metrics(word_gold, [["B-CF"]])
        candidate = compute_metrics(word_gold, [["B-CF"]])

        gold_char = [{("CF", 0, 16)}]
        baseline["production_like_char"] = compute_span_metrics(
            gold_char, [{("CF", 0, 16)}]
        )
        # Il primo subword e' CF, ma un subword interno cambia tipo: la risposta
        # reale contiene due frammenti e nessuno coincide col CF completo.
        candidate["production_like_char"] = compute_span_metrics(
            gold_char,
            [simple_char_spans(
                ["B-CF", "I-ID_DOC"],
                [(0, 3), (3, 16)],
            )],
        )

        gate = apply_quality_gate(
            baseline,
            candidate,
            "student-fp32",
            {"max_micro_f1_drop": 0.0},
        )
        self.assertFalse(gate["passed"])
        self.assertEqual(1.0, candidate["micro"]["f1"])
        self.assertEqual(0.0, candidate["production_like_char"]["micro"]["f1"])
        self.assertIn(
            "production_like_char.max_micro_f1_drop",
            {failure["gate"] for failure in gate["failures"]},
        )

    def test_teacher_student_disagreement_is_separate_from_human_gold_quality(self):
        teacher = [{("FULLNAME", 0, 12), ("CF", 13, 29)}]
        student = [{("FULLNAME", 0, 12), ("CF", 13, 20)}]
        disagreement = compare_span_predictions(teacher, student)["agreement"]

        self.assertEqual(1, disagreement["spans"]["agreed"])
        self.assertEqual(1, disagreement["spans"]["new"])
        self.assertEqual(1, disagreement["spans"]["removed"])
        self.assertEqual(0, disagreement["exact_documents"])

    def test_evaluation_report_embeds_model_only_char_metrics_and_e2e_limit(self):
        performance = {
            "batch_latencies_seconds": [0.01],
            "request_latencies_seconds": [0.02],
            "inference_seconds": 0.02,
            "load_seconds": 0.01,
            "rss_after_load_bytes": 0,
            "_production_like_char_spans": [{("CF", 0, 16)}],
        }
        stat = [{"path": "/tmp/model.onnx", "bytes": 10, "inode": 1, "mtime_ns": 1}]
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "validation.jsonl").write_text("{}\n", encoding="utf-8")
            (root / "config.json").write_text("{}\n", encoding="utf-8")
            with (
                patch.object(
                    evaluate,
                    "load_validation",
                    return_value=[
                        ValidationRecord(["RSSMRA85H12F205Z"], ["B-CF"])
                    ],
                ),
                patch.object(
                    evaluate,
                    "_run_onnx",
                    return_value=([["B-CF"]], performance),
                ),
                patch.object(evaluate, "_artifact_stat_snapshot", return_value=stat),
            ):
                report = evaluate.evaluate_variant(
                    name="student-fp32",
                    backend="onnx",
                    model_path=root / "model.onnx",
                    tokenizer_path=root,
                    validation_path=root / "validation.jsonl",
                    predictions_path=root / "predictions.jsonl",
                    report_path=root / "metrics.json",
                    defer_artifact_hash=True,
                )

        self.assertEqual(1.0, report["metrics"]["micro"]["f1"])
        self.assertEqual(
            1.0,
            report["metrics"]["production_like_char"]["micro"]["f1"],
        )
        view = report["production_like_evaluation"]
        self.assertEqual("model-only", view["view"])
        self.assertFalse(view["includes_production_merge"])
        self.assertEqual(
            "not-computable-on-this-dataset",
            view["end_to_end_gate"]["status"],
        )


if __name__ == "__main__":
    unittest.main()
