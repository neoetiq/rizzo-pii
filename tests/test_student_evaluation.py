import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from src.training.student_utils import sha256_file

from src.training.evaluate_student import (
    EVALUATION_SCHEMA,
    ID_DOC_PREFIX_TOLERANT_METRIC_NAME,
    PRIVACY_RESCORE_SCHEMA,
    PRODUCTION_PRIVACY_UTILITY_METRIC_NAME,
    SCHEMA_VERSION,
    RawSpanRecord,
    StudentEvaluationError,
    canonicalize_id_doc_prefix_spans,
    compare_saved_evaluations,
    compute_id_doc_prefix_tolerant_metric,
    compute_production_privacy_utility_metric,
    dataset_identity,
    load_raw_records,
    main as evaluation_main,
    merge_overflow_windows,
    normalize_model_spans_for_production,
    rescore_saved_predictions_privacy_utility,
    model_identity,
    tokenizer_identity,
    write_predictions_artifact,
)


ID2LABEL = {
    0: "O",
    1: "B-FULLNAME",
    2: "I-FULLNAME",
    3: "B-CF",
    4: "I-CF",
}


def _logits(label_id: int) -> list[float]:
    values = [-4.0] * len(ID2LABEL)
    values[label_id] = 4.0
    return values


class StudentEvaluationTests(unittest.TestCase):
    def test_raw_loader_keeps_exact_offsets_and_teacher_taxonomy(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "validation.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "record_id": "doc-1",
                        "source_text": "Mario Rossi, RG 123",
                        "entities": [
                            {"start": 0, "end": 11, "label": "ATTORE"},
                            {"start": 13, "end": 15, "label": "TITLE"},
                            {"start": 16, "end": 19, "label": "RG"},
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            records = load_raw_records(path)

        self.assertEqual(records[0].record_id, "doc-1")
        self.assertEqual(
            records[0].gold_spans,
            frozenset({("FULLNAME", 0, 11), ("DOCID", 16, 19)}),
        )

    def test_raw_loader_fuses_adjacent_givenname_and_surname(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "validation.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "source_text": "Mario Rossi e Anna",
                        "entities": [
                            {"start": 0, "end": 5, "label": "GIVENNAME"},
                            {"start": 6, "end": 11, "label": "SURNAME"},
                            {"start": 14, "end": 18, "label": "GIVENNAME"},
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            records = load_raw_records(path)

        self.assertEqual(
            records[0].gold_spans,
            frozenset({("FULLNAME", 0, 11), ("FULLNAME", 14, 18)}),
        )

    def test_overflow_merge_uses_all_subwords_and_removes_duplicates(self):
        logits = [
            [
                _logits(0),
                _logits(1),
                _logits(2),
                _logits(0),
                _logits(0),
            ],
            [
                _logits(0),
                _logits(2),
                _logits(0),
                _logits(3),
                _logits(0),
            ],
        ]
        offsets = [
            [(0, 0), (0, 3), (3, 5), (6, 10), (0, 0)],
            [(0, 0), (3, 5), (6, 10), (11, 14), (0, 0)],
        ]
        masks = [[1, 0, 0, 0, 1], [1, 0, 0, 0, 1]]

        spans = merge_overflow_windows(
            logits, offsets, masks, ID2LABEL, text_length=14
        )

        self.assertEqual(spans, {("FULLNAME", 0, 5), ("CF", 11, 14)})

    def test_overlap_conflicts_are_averaged_with_stable_label_tie(self):
        # At the repeated offset, the mean is tied between O (id 0) and B-CF
        # (id 3); the documented lower-ID tie break deterministically selects O.
        first = [0.0] * len(ID2LABEL)
        second = [0.0] * len(ID2LABEL)
        first[0] = 2.0
        second[3] = 2.0
        spans = merge_overflow_windows(
            [[first], [second]],
            [[(0, 4)], [(0, 4)]],
            [[0], [0]],
            ID2LABEL,
            text_length=4,
        )
        self.assertEqual(spans, set())

    def test_nonidentical_overlapping_token_offsets_are_rejected(self):
        with self.assertRaises(StudentEvaluationError):
            merge_overflow_windows(
                [[_logits(0), _logits(0)]],
                [[(0, 4), (3, 6)]],
                [[0, 0]],
                ID2LABEL,
                text_length=6,
            )

    def test_production_boundaries_trim_metaspace_and_expand_partial_word(self):
        text = " Mario Rossi e Novara,"
        normalized = normalize_model_spans_for_production(
            text,
            {
                ("FULLNAME", 0, 12),  # leading space from Metaspace
                ("CITY", 15, 18),  # only "Nov" inside "Novara"
            },
        )
        self.assertEqual(
            normalized,
            {("FULLNAME", 1, 12), ("CITY", 15, 21)},
        )

    def test_production_boundaries_do_not_hide_punctuation_errors(self):
        text = "Societa S.n.c., continua"
        normalized = normalize_model_spans_for_production(
            text, {("ORG", 7, 15)}
        )
        self.assertEqual(normalized, {("ORG", 8, 15)})

    def test_id_doc_metric_accepts_only_supported_optional_number_markers(self):
        chunks = (
            ("n. 1234567", "1234567"),
            ("N° AB123", "AB123"),
            ("nr. ZX9", "ZX9"),
            ("NUMERO Q7", "Q7"),
        )
        text = " \t" + " | ".join(chunk for chunk, _ in chunks) + "  "
        gold = set()
        predictions = set()
        cursor = 0
        for chunk, payload in chunks:
            start = text.index(chunk, cursor)
            end = start + len(chunk)
            payload_start = text.index(payload, start, end)
            gold.add(("ID_DOC", start, end))
            predictions.add(("ID_DOC", payload_start, end))
            cursor = end

        record = RawSpanRecord("doc", text, frozenset(gold))
        result = compute_id_doc_prefix_tolerant_metric(
            [record], [predictions]
        )

        self.assertEqual(result["name"], ID_DOC_PREFIX_TOLERANT_METRIC_NAME)
        self.assertEqual(result["metrics"]["micro"]["f1"], 1.0)
        self.assertEqual(result["metrics"]["micro"]["tp"], 4)
        self.assertEqual(result["canonicalized_spans"]["gold"], 4)
        self.assertEqual(result["canonicalized_spans"]["predictions"], 0)

    def test_id_doc_metric_keeps_partial_wrong_type_and_invading_text_as_errors(self):
        text = "Documento n. 1234567"
        gold_start = text.index("n.")
        payload_start = text.index("1234567")
        end = len(text)
        record = RawSpanRecord(
            "doc", text, frozenset({("ID_DOC", gold_start, end)})
        )
        cases = {
            "partial_payload": {("ID_DOC", payload_start + 1, end)},
            "wrong_type": {("DOCID", payload_start, end)},
            "invading_text": {("ID_DOC", 0, end)},
        }

        for name, prediction in cases.items():
            with self.subTest(name=name):
                metrics = compute_id_doc_prefix_tolerant_metric(
                    [record], [prediction]
                )["metrics"]["micro"]
                self.assertEqual(metrics["tp"], 0)
                self.assertEqual(metrics["fp"], 1)
                self.assertEqual(metrics["fn"], 1)

    def test_id_doc_metric_does_not_relax_docid_or_unlisted_prefixes(self):
        text = "n. 1234567; num. 7654321; numero: 2468135"
        first_start = 0
        first_payload = text.index("1234567")
        first_end = first_payload + len("1234567")
        num_start = text.index("num.")
        num_payload = text.index("7654321")
        num_end = num_payload + len("7654321")
        numero_start = text.index("numero:")
        numero_payload = text.index("2468135")
        numero_end = numero_payload + len("2468135")

        canonical = canonicalize_id_doc_prefix_spans(
            text,
            {
                ("DOCID", first_start, first_end),
                ("ID_DOC", num_start, num_end),
                ("ID_DOC", numero_start, numero_end),
            },
        )

        self.assertIn(("DOCID", first_start, first_end), canonical)
        self.assertIn(("ID_DOC", num_start, num_end), canonical)
        self.assertIn(("ID_DOC", numero_start, numero_end), canonical)
        self.assertNotIn(("DOCID", first_payload, first_end), canonical)
        self.assertNotIn(("ID_DOC", num_payload, num_end), canonical)
        self.assertNotIn(("ID_DOC", numero_payload, numero_end), canonical)

    def test_production_privacy_utility_handles_prefix_wrong_type_partial_fp_and_docs(self):
        first_text = "n. AB12 extra"
        first_payload_start = first_text.index("AB12")
        first_payload_end = first_payload_start + len("AB12")
        first_extra_start = first_text.index("extra")
        first_record = RawSpanRecord(
            "first",
            first_text,
            frozenset({("ID_DOC", 0, first_payload_end)}),
        )
        first_predictions = {
            ("DOCID", first_payload_start, first_payload_end),
            ("FULLNAME", first_extra_start, len(first_text)),
        }

        second_text = "ID ZX9é"
        second_payload_start = second_text.index("ZX9é")
        second_payload_end = len(second_text)
        second_record = RawSpanRecord(
            "second",
            second_text,
            frozenset(
                {("ID_DOC", second_payload_start, second_payload_end)}
            ),
        )
        second_predictions = {
            ("ID_DOC", second_payload_start, second_payload_end - 1)
        }

        result = compute_production_privacy_utility_metric(
            [first_record, second_record],
            [first_predictions, second_predictions],
        )

        self.assertEqual(result["name"], PRODUCTION_PRIVACY_UTILITY_METRIC_NAME)
        self.assertEqual(result["counts"]["documents"], 2)
        self.assertEqual(result["counts"]["gold_spans_canonicalized"], 1)
        self.assertEqual(result["counts"]["gold_sensitive_characters"], 8)
        self.assertEqual(result["type_agnostic"]["covered_characters"], 7)
        self.assertEqual(result["type_agnostic"]["leaked_characters"], 1)
        self.assertEqual(result["type_agnostic"]["character_coverage"], 7 / 8)
        self.assertEqual(result["type_correct"]["covered_characters"], 3)
        self.assertEqual(
            result["type_correct"][
                "wrong_type_but_privacy_covered_characters"
            ],
            4,
        )
        self.assertEqual(result["type_agnostic"]["fully_covered_entities"], 1)
        self.assertEqual(result["type_correct"]["fully_covered_entities"], 0)
        self.assertEqual(result["type_agnostic"]["fully_covered_documents"], 1)
        self.assertEqual(result["type_correct"]["fully_covered_documents"], 0)

        utility = result["utility"]
        self.assertEqual(utility["predicted_alnum_characters"], 12)
        self.assertEqual(utility["predicted_on_gold_characters"], 7)
        self.assertEqual(utility["collateral_masked_characters"], 5)
        self.assertEqual(utility["mask_precision"], 7 / 12)
        self.assertEqual(utility["non_pii_alnum_characters"], 8)
        self.assertEqual(utility["collateral_masking_ratio"], 5 / 8)

        id_doc = result["per_tag"]["ID_DOC"]
        self.assertEqual(id_doc["gold"]["sensitive_characters"], 8)
        self.assertEqual(id_doc["type_agnostic"]["covered_characters"], 7)
        self.assertEqual(id_doc["type_correct"]["covered_characters"], 3)
        self.assertEqual(
            result["per_tag"]["DOCID"]["prediction_utility"][
                "predicted_on_any_gold_characters"
            ],
            4,
        )
        self.assertEqual(
            result["per_tag"]["FULLNAME"]["prediction_utility"][
                "collateral_masked_characters"
            ],
            5,
        )

    def test_production_privacy_utility_can_rescore_saved_predictions_offline(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "data.jsonl"
            secret_text = "numero Q7 RISERVATO"
            payload_start = secret_text.index("Q7")
            payload_end = payload_start + 2
            dataset.write_text(
                json.dumps(
                    {
                        "record_id": "offline-doc",
                        "source_text": secret_text,
                        "entities": [
                            {
                                "start": 0,
                                "end": payload_end,
                                "label": "ID_DOC",
                            }
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            records = load_raw_records(dataset)
            predictions_path = root / "predictions.jsonl"
            prediction_meta = write_predictions_artifact(
                predictions_path,
                records,
                [{("ID_DOC", payload_start, payload_end)}],
            )

            rescored = rescore_saved_predictions_privacy_utility(
                dataset_path=dataset,
                predictions_path=predictions_path,
                expected_predictions_sha256=prediction_meta["sha256"],
            )

            self.assertEqual(rescored["type_agnostic"]["character_coverage"], 1.0)
            self.assertEqual(rescored["type_correct"]["character_coverage"], 1.0)
            self.assertNotIn("offline-doc", json.dumps(rescored))
            self.assertNotIn("RISERVATO", json.dumps(rescored))

    def test_rescore_privacy_cli_writes_atomic_aggregate_only_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "data.jsonl"
            source_text = "n. A1 TESTO_NON_PERSISTIBILE"
            payload_start = source_text.index("A1")
            payload_end = payload_start + 2
            dataset.write_text(
                json.dumps(
                    {
                        "record_id": "private-record-id",
                        "source_text": source_text,
                        "entities": [
                            {"start": 0, "end": payload_end, "label": "ID_DOC"}
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            records = load_raw_records(dataset)
            predictions = root / "predictions.jsonl"
            prediction_meta = write_predictions_artifact(
                predictions,
                records,
                [{("ID_DOC", payload_start, payload_end)}],
            )
            output = root / "privacy-report.json"

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = evaluation_main(
                    [
                        "rescore-privacy",
                        "--dataset-path",
                        str(dataset),
                        "--predictions-path",
                        str(predictions),
                        "--expected-predictions-sha256",
                        prediction_meta["sha256"].upper(),
                        "--output-report",
                        str(output),
                    ]
                )

            self.assertEqual(exit_code, 0)
            self.assertTrue(output.is_file())
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["schema"], PRIVACY_RESCORE_SCHEMA)
            self.assertEqual(report["schema_version"], SCHEMA_VERSION)
            self.assertEqual(report["dataset"]["sha256"], sha256_file(dataset))
            self.assertEqual(
                report["predictions"]["sha256"], prediction_meta["sha256"]
            )
            self.assertTrue(
                report["verification"]["record_id_sets_exact_match"]
            )
            self.assertTrue(report["verification"]["gold_spans_exact_match"])
            self.assertEqual(
                report["metric"]["type_agnostic"]["character_coverage"], 1.0
            )
            serialized = json.dumps(report, ensure_ascii=False)
            self.assertNotIn("private-record-id", serialized)
            self.assertNotIn("TESTO_NON_PERSISTIBILE", serialized)
            self.assertIn(str(output.resolve()), stdout.getvalue())

    def test_rescore_privacy_cli_is_fail_closed_for_hash_ids_and_gold(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "data.jsonl"
            text = "Documento B7"
            payload_start = text.index("B7")
            payload_end = len(text)
            dataset.write_text(
                json.dumps(
                    {
                        "record_id": "expected-id",
                        "source_text": text,
                        "entities": [
                            {
                                "start": payload_start,
                                "end": payload_end,
                                "label": "ID_DOC",
                            }
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            records = load_raw_records(dataset)
            correct_predictions = root / "correct.jsonl"
            write_predictions_artifact(
                correct_predictions,
                records,
                [{("ID_DOC", payload_start, payload_end)}],
            )

            wrong_id_predictions = root / "wrong-id.jsonl"
            wrong_id_record = RawSpanRecord(
                "different-id", text, records[0].gold_spans
            )
            write_predictions_artifact(
                wrong_id_predictions,
                [wrong_id_record],
                [{("ID_DOC", payload_start, payload_end)}],
            )

            wrong_gold_predictions = root / "wrong-gold.jsonl"
            wrong_gold_record = RawSpanRecord(
                "expected-id",
                text,
                frozenset({("ID_DOC", payload_start, payload_end - 1)}),
            )
            write_predictions_artifact(
                wrong_gold_predictions,
                [wrong_gold_record],
                [{("ID_DOC", payload_start, payload_end - 1)}],
            )

            cases = (
                ("hash", correct_predictions, ["--expected-predictions-sha256", "0" * 64]),
                ("record_ids", wrong_id_predictions, []),
                ("gold", wrong_gold_predictions, []),
            )
            for name, predictions, extra_args in cases:
                with self.subTest(name=name):
                    output = root / f"failed-{name}.json"
                    with self.assertRaises(StudentEvaluationError):
                        evaluation_main(
                            [
                                "rescore-privacy",
                                "--dataset-path",
                                str(dataset),
                                "--predictions-path",
                                str(predictions),
                                "--output-report",
                                str(output),
                                *extra_args,
                            ]
                        )
                    self.assertFalse(output.exists())

    def test_model_tokenizer_and_dataset_content_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model"
            model.mkdir()
            (model / "config.json").write_text("{}", encoding="utf-8")
            (model / "model.safetensors").write_bytes(b"weights")
            (model / "tokenizer.json").write_text("{}", encoding="utf-8")
            (model / "tokenizer_config.json").write_text("{}", encoding="utf-8")
            dataset = root / "data.jsonl"
            dataset.write_text(
                json.dumps(
                    {
                        "record_id": "x",
                        "source_text": "Mario",
                        "entities": [{"start": 0, "end": 5, "label": "FULLNAME"}],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            records = load_raw_records(dataset)
            model_hash = model_identity(model)
            tokenizer_hash = tokenizer_identity(model)
            data_hash = dataset_identity(dataset, records, None)

        self.assertEqual(len(model_hash["sha256"]), 64)
        self.assertEqual(len(tokenizer_hash["sha256"]), 64)
        self.assertEqual(len(data_hash["sha256"]), 64)
        self.assertEqual(len(data_hash["selection_sha256"]), 64)
        self.assertNotEqual(model_hash["sha256"], tokenizer_hash["sha256"])

    def test_saved_reports_compare_without_model_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "data.jsonl"
            dataset.write_text(
                json.dumps(
                    {
                        "record_id": "doc",
                        "source_text": "Mario CF123",
                        "entities": [
                            {"start": 0, "end": 5, "label": "FULLNAME"},
                            {"start": 6, "end": 11, "label": "CF"},
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            records = load_raw_records(dataset)
            identity = dataset_identity(dataset, records, None)
            teacher_predictions = root / "teacher.jsonl"
            student_predictions = root / "student.jsonl"
            teacher_pred_meta = write_predictions_artifact(
                teacher_predictions,
                records,
                [{("FULLNAME", 0, 5), ("CF", 6, 11)}],
            )
            student_pred_meta = write_predictions_artifact(
                student_predictions,
                records,
                [{("FULLNAME", 0, 5)}],
            )
            teacher_report = root / "teacher-report.json"
            student_report = root / "student-report.json"
            common = {
                "schema": EVALUATION_SCHEMA,
                "schema_version": SCHEMA_VERSION,
                "dataset": identity,
                "tokenizer": {"sha256": "tokenizer"},
            }
            teacher_report.write_text(
                json.dumps(
                    {
                        **common,
                        "model": {"sha256": "teacher"},
                        "predictions": teacher_pred_meta,
                    }
                ),
                encoding="utf-8",
            )
            student_report.write_text(
                json.dumps(
                    {
                        **common,
                        "model": {"sha256": "student"},
                        "predictions": student_pred_meta,
                    }
                ),
                encoding="utf-8",
            )
            output = root / "comparison.json"

            comparison = compare_saved_evaluations(
                teacher_report=teacher_report,
                student_report=student_report,
                output_report=output,
            )

            self.assertTrue(output.is_file())
            self.assertEqual(comparison["teacher"]["metrics"]["micro"]["f1"], 1.0)
            self.assertLess(comparison["student"]["metrics"]["micro"]["f1"], 1.0)
            paired = comparison["disagreement"]["paired_against_gold"]
            self.assertEqual(paired["document_outcomes"]["teacher_only_exact"], 1)
            agreement = comparison["disagreement"]["teacher_vs_student"]["agreement"]
            self.assertEqual(agreement["spans"]["removed"], 1)

    def test_compare_rejects_tampered_predictions_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prediction = root / "prediction.jsonl"
            record = RawSpanRecord("x", "Mario", frozenset({("FULLNAME", 0, 5)}))
            metadata = write_predictions_artifact(
                prediction, [record], [{("FULLNAME", 0, 5)}]
            )
            dataset = {
                "sha256": "dataset",
                "selection_sha256": "selection",
                "selected_records": 1,
            }
            report = {
                "schema": EVALUATION_SCHEMA,
                "schema_version": SCHEMA_VERSION,
                "dataset": dataset,
                "model": {"sha256": "model"},
                "tokenizer": {"sha256": "tokenizer"},
                "predictions": metadata,
            }
            teacher = root / "teacher.json"
            student = root / "student.json"
            teacher.write_text(json.dumps(report), encoding="utf-8")
            student.write_text(json.dumps(report), encoding="utf-8")
            prediction.write_text(prediction.read_text() + " ", encoding="utf-8")

            with self.assertRaises(StudentEvaluationError):
                compare_saved_evaluations(
                    teacher_report=teacher,
                    student_report=student,
                )


if __name__ == "__main__":
    unittest.main()
