import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from src.training import build_targeted_finetune_data as targeted
from src.training.student_utils import StudentTrainingError


def _record(uid: str, label: str, value: str) -> dict:
    prefix = f"Documento {uid}: "
    suffix = f"; contesto verificabile {uid}."
    text = prefix + value + suffix
    start = len(prefix)
    return {
        "source_text": text,
        "entities": [
            {
                "start": start,
                "end": start + len(value),
                "label": label,
                "value": value,
            }
        ],
        "language": "it",
        "template_id": f"template-{uid}",
    }


def _write_parquet(path: Path, rows: list[dict]) -> None:
    pq.write_table(pa.Table.from_pylist(rows), path)


def _write_observed(path: Path, rows: list[dict]) -> None:
    payloads = []
    for index, row in enumerate(rows):
        identity = targeted._identity_payload(row, context=f"observed {index}")
        payloads.append(
            {
                "record_id": targeted._record_id(identity),
                **identity,
            }
        )
    path.write_text(
        "".join(
            json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n"
            for payload in payloads
        ),
        encoding="utf-8",
    )


def _v1_replay_payload(
    row: dict, *, source_row: int, stratum: str
) -> dict:
    identity = targeted._identity_payload(row, context=f"V1 replay {source_row}")
    return {
        "record_id": targeted._record_id(identity),
        "source": "clean",
        "source_split": "train",
        "source_row": source_row,
        **identity,
        "dataset_role": "d1_targeted",
        "selection_stratum": stratum,
        "sealed": False,
    }


def _write_jsonl(path: Path, records: list[dict]) -> bytes:
    payload = b"".join(
        targeted._canonical_json_bytes(record) + b"\n" for record in records
    )
    path.write_bytes(payload)
    return payload


def _write_fake_v1_bundle(root: Path) -> Path:
    root.mkdir()
    records = []
    source_row = 0
    for index in range(12):
        records.append(
            _v1_replay_payload(
                _record(f"v1-iban-{index}", "IBAN", f"0011223344{index:02d}"),
                source_row=source_row,
                stratum="replay-protected:IBAN",
            )
        )
        source_row += 1
    for index in range(12):
        records.append(
            _v1_replay_payload(
                _record(f"v1-alphanumeric-{index}", "ID_DOC", f"ZX{index:07d}"),
                source_row=source_row,
                stratum="replay-protected:ID_DOC",
            )
        )
        source_row += 1
    for index in range(24):
        records.append(
            _v1_replay_payload(
                _record(f"v1-general-{index}", "FULLNAME", f"Persona Test {index}"),
                source_row=source_row,
                stratum="replay-general",
            )
        )
        source_row += 1

    d1_records = records[:24]
    d0_records = records[24:]
    payloads = {
        "d1_targeted": (
            targeted.OUTPUT_FILENAMES["d1_targeted"],
            _write_jsonl(
                root / targeted.OUTPUT_FILENAMES["d1_targeted"], d1_records
            ),
            len(d1_records),
        ),
        "d0_control": (
            targeted.OUTPUT_FILENAMES["d0_control"],
            _write_jsonl(
                root / targeted.OUTPUT_FILENAMES["d0_control"], d0_records
            ),
            len(d0_records),
        ),
    }
    manifest = {
        "schema_version": 1,
        "kind": "pii-id-doc-targeted-finetune-data",
        "status": "sealed",
        "outputs": {
            role: {
                "filename": filename,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "stats": {"rows": rows},
            }
            for role, (filename, payload, rows) in payloads.items()
        },
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return root


class TargetedFinetuneDataTests(unittest.TestCase):
    def _fixture(self, root: Path) -> targeted.BuildConfig:
        observed_train_row = _record("observed-train", "FULLNAME", "Mario Rossi")
        train_rows = [observed_train_row]

        # Every target must be selected in D1; aliases exercise normalized ID_DOC.
        train_rows.extend(
            [
                _record("target-shared", "IDCARDNUM", "n. 1234567"),
                _record("target-1", "ID_DOC", "n. 2345678"),
                _record("target-2", "PASSPORTNUM", " n. 3456789 "),
            ]
        )

        hard_values = {
            "sentenza": "Sent. n. 1200/2024",
            "protocollo": "Prot. 2024-1200",
            "rg": "RG 1200/2024",
        }
        for variant, value in hard_values.items():
            for index in range(6):
                train_rows.append(
                    _record(f"hard-{variant}-{index}", "DOCID", value + f"-{index}")
                )

        protected_values = {
            "ID_DOC": "CIE AB1234567",
            "DOCID": "DOC-12345",
            "CF": "RSSMRA80A01H501U",
            "PIVA": "12345678901",
            "IBAN": "IT60X0542811101000000123456",
            "EMAIL": "mario@example.test",
            "TELEPHONENUM": "+39 0200000000",
        }
        for label, value in protected_values.items():
            for index in range(10):
                train_rows.append(
                    _record(f"protected-{label}-{index}", label, value + str(index))
                )
        for index in range(50):
            train_rows.append(
                _record(f"general-{index}", "FULLNAME", f"Persona Generale {index}")
            )

        observed_validation_row = _record(
            "observed-validation-shape", "ID_DOC", "n. 4567890"
        )
        validation_rows = [
            observed_validation_row,
            # Different content, same normalized skeleton as validation-512.
            _record("observed-validation-shape", "ID_DOC", "n. 4567891"),
            # Same skeleton as a D1 target; it must not enter either holdout.
            _record("target-shared", "IDCARDNUM", "n. 4567892"),
            _record("holdout-target-1", "ID_DOC", "n. 5678901"),
            _record("holdout-target-2", "DRIVERLICENSENUM", "n. 6789012"),
        ]
        for index in range(12):
            validation_rows.append(
                _record(f"holdout-id-{index}", "ID_DOC", f"CIE ZZ{index:07d}")
            )
        for index in range(30):
            validation_rows.append(
                _record(f"holdout-global-{index}", "FULLNAME", f"Nome Globale {index}")
            )

        train_path = root / "train.parquet"
        validation_path = root / "validation.parquet"
        observed_train_path = root / "train-observed.jsonl"
        observed_validation_path = root / "validation-observed.jsonl"
        _write_parquet(train_path, train_rows)
        _write_parquet(validation_path, validation_rows)
        _write_observed(observed_train_path, [observed_train_row])
        _write_observed(observed_validation_path, [observed_validation_row])
        return targeted.BuildConfig(
            train_parquet=train_path,
            validation_parquet=validation_path,
            observed_train=observed_train_path,
            observed_validation=observed_validation_path,
            output_dir=root / "targeted-output",
            clean_revision="synthetic-test-revision",
            seed=7,
            salt="targeted-unit-test",
            d1_rows=20,
            d0_rows=20,
            holdout_global_rows=4,
            holdout_id_doc_rows=4,
            replay_protected_floor=1,
            expected_target_train_docs=3,
            expected_target_train_entities=3,
            expected_target_holdout_docs=2,
            expected_target_holdout_entities=2,
            expected_train_sha256=None,
            expected_validation_sha256=None,
        )

    def test_build_is_deterministic_disjoint_and_stratified(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self._fixture(Path(directory))
            manifest, payloads = targeted.build_artifacts(config)
            repeated_manifest, repeated_payloads = targeted.build_artifacts(config)

            self.assertEqual(payloads, repeated_payloads)
            self.assertEqual(manifest, repeated_manifest)
            self.assertTrue(manifest["overlap_audit"]["all_pairwise_zero"])
            self.assertEqual(manifest["pool"]["target_train_docs"], 3)
            self.assertEqual(manifest["pool"]["target_holdout_docs"], 2)
            self.assertEqual(
                manifest["selection"]["effective_salt"],
                "targeted-unit-test:seed=7",
            )

            outputs = manifest["outputs"]
            self.assertEqual(outputs["d1_targeted"]["stats"]["rows"], 20)
            self.assertEqual(
                outputs["d1_targeted"]["stats"]["target_shape_docs"], 3
            )
            self.assertEqual(
                sum(
                    count
                    for name, count in outputs["d1_targeted"]["stats"][
                        "strata"
                    ].items()
                    if name.startswith("hard-negative:")
                ),
                3,
            )
            self.assertEqual(
                sum(
                    count
                    for name, count in outputs["d1_targeted"]["stats"][
                        "strata"
                    ].items()
                    if name.startswith("replay-")
                ),
                14,
            )
            self.assertEqual(outputs["d0_control"]["stats"]["rows"], 20)
            self.assertEqual(
                outputs["d0_control"]["stats"]["target_shape_docs"], 0
            )
            self.assertEqual(outputs["sealed_global"]["stats"]["rows"], 4)
            self.assertEqual(outputs["sealed_id_doc"]["stats"]["rows"], 4)
            self.assertEqual(
                outputs["sealed_id_doc"]["stats"]["target_shape_docs"], 2
            )

            all_ids = []
            all_skeletons = []
            for output in outputs.values():
                self.assertEqual(
                    hashlib.sha256(payloads[output["filename"]]).hexdigest(),
                    output["sha256"],
                )
                all_ids.extend(item["record_id"] for item in output["records"])
                all_skeletons.extend(
                    item["normalized_skeleton_sha256"]
                    for item in output["records"]
                )
            self.assertEqual(len(all_ids), len(set(all_ids)))
            self.assertEqual(len(all_skeletons), len(set(all_skeletons)))

    def test_seed_changes_salted_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self._fixture(Path(directory))
            first_manifest, first_payloads = targeted.build_artifacts(config)
            second_manifest, second_payloads = targeted.build_artifacts(
                replace(config, seed=8)
            )

            self.assertNotEqual(
                first_manifest["selection"]["effective_salt"],
                second_manifest["selection"]["effective_salt"],
            )
            self.assertNotEqual(
                first_payloads[targeted.OUTPUT_FILENAMES["d0_control"]],
                second_payloads[targeted.OUTPUT_FILENAMES["d0_control"]],
            )

    def test_atomic_publish_is_idempotent_and_rejects_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self._fixture(Path(directory))
            _, payloads = targeted.build_artifacts(config)

            targeted.publish_atomic(config.output_dir, payloads)
            targeted.publish_atomic(config.output_dir, payloads)
            self.assertEqual(
                {path.name for path in config.output_dir.iterdir()}, set(payloads)
            )

            changed = config.output_dir / targeted.OUTPUT_FILENAMES["d1_targeted"]
            changed.write_bytes(changed.read_bytes() + b"{}\n")
            with self.assertRaisesRegex(StudentTrainingError, "esistente ma diverso"):
                targeted.publish_atomic(config.output_dir, payloads)

    def test_expected_target_count_is_a_hard_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self._fixture(Path(directory))
            with self.assertRaisesRegex(
                StudentTrainingError, "Documenti target train novel"
            ):
                targeted.build_artifacts(
                    replace(config, expected_target_train_docs=4)
                )

    def test_target_and_hard_negative_normalization(self):
        target = _record("normalization-target", "IDCARDNUM", " n. 1234567 ")
        target_identity = targeted._identity_payload(target, context="target")
        self.assertEqual(targeted._target_entity_count(target_identity), 1)

        variants = {
            "sentenza-n": "Sentenza n. 44/2024",
            "protocollo": "Protocollo 123/2024",
            "rg": "R.G. 123/2024",
        }
        for expected, value in variants.items():
            with self.subTest(variant=expected):
                raw = _record(f"hard-{expected}", "DOCID", value)
                identity = targeted._identity_payload(raw, context=expected)
                self.assertEqual(targeted._hard_negative_variant(identity), expected)


class TargetedFinetuneDataV2Tests(unittest.TestCase):
    def _fixture(self, root: Path) -> targeted.D2BuildConfig:
        prior_v1_dir = _write_fake_v1_bundle(root / "v1")
        return targeted.D2BuildConfig(
            prior_v1_dir=prior_v1_dir,
            output_dir=root / "v2",
            seed=20260822,
            salt="d2-unit-test",
            train_positive_rows=8,
            train_hard_negative_rows=16,
            replay_rows=16,
            challenge_positive_rows=8,
            challenge_hard_negative_rows=16,
            replay_numeric_iban_floor=2,
            replay_alphanumeric_id_floor=2,
        )

    @staticmethod
    def _jsonl_records(payload: bytes) -> list[dict]:
        return [json.loads(line) for line in payload.decode("utf-8").splitlines()]

    def test_default_recipe_contract_is_1024_and_balanced(self):
        config = targeted.D2BuildConfig()
        self.assertEqual(
            config.train_positive_rows
            + config.train_hard_negative_rows
            + config.replay_rows,
            1024,
        )
        self.assertEqual(config.train_positive_rows, 128)
        self.assertEqual(config.train_positive_rows // 4, 32)
        self.assertEqual(config.train_hard_negative_rows, 256)
        self.assertEqual(config.replay_rows, 640)

    def test_v2_is_deterministic_balanced_and_hash_pinned(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self._fixture(Path(directory))
            manifest, payloads = targeted.build_v2_artifacts(config)
            repeated_manifest, repeated_payloads = targeted.build_v2_artifacts(config)

            self.assertEqual(manifest, repeated_manifest)
            self.assertEqual(payloads, repeated_payloads)
            self.assertEqual(manifest["seed"], 20260822)
            self.assertEqual(manifest["span_policy"]["id"], "identifier-value-only-v1")
            self.assertEqual(
                set(manifest["evaluation_views"]),
                {"exact_span", "payload_sensitive_span"},
            )
            train = manifest["outputs"]["d2_targeted"]
            challenge = manifest["outputs"]["sealed_synthetic_challenge"]
            self.assertEqual(train["stats"]["rows"], 40)
            self.assertEqual(
                train["stats"]["polarities"],
                {
                    "hard-negative-boundary": 8,
                    "hard-negative-exact": 8,
                    "positive": 8,
                    "real-replay": 16,
                },
            )
            self.assertEqual(
                train["stats"]["positive_format_families"],
                {family: 2 for family in targeted.D2_FORMAT_FAMILIES},
            )
            self.assertEqual(challenge["stats"]["rows"], 24)
            for output in manifest["outputs"].values():
                self.assertEqual(
                    hashlib.sha256(payloads[output["filename"]]).hexdigest(),
                    output["sha256"],
                )

    def test_value_only_spans_and_matched_docid_pairs(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self._fixture(Path(directory))
            _, payloads = targeted.build_v2_artifacts(config)
            records = self._jsonl_records(
                payloads[targeted.D2_OUTPUT_FILENAMES["d2_targeted"]]
            )
            synthetic = [record for record in records if record["synthetic"]]
            pairs: dict[str, list[dict]] = {}
            for record in synthetic:
                entity = record["entities"][0]
                payload_meta = record["identifier_payload"]
                self.assertEqual(entity["start"], payload_meta["start"])
                self.assertEqual(entity["end"], payload_meta["end"])
                self.assertEqual(record["gold_span_policy"], "identifier-value-only-v1")
                self.assertEqual(
                    hashlib.sha256(
                        record["source_text"][entity["start"] : entity["end"]].encode(
                            "utf-8"
                        )
                    ).hexdigest(),
                    payload_meta["sha256"],
                )
                expected_label = (
                    "ID_DOC"
                    if record["synthetic_polarity"] == "positive"
                    else "DOCID"
                )
                self.assertEqual(entity["label"], expected_label)
                pairs.setdefault(record["pair_id"], []).append(record)

            self.assertEqual(len(pairs), 8)
            for pair in pairs.values():
                by_polarity = {
                    record["synthetic_polarity"]: record for record in pair
                }
                self.assertEqual(
                    set(by_polarity),
                    {"positive", "hard-negative-exact", "hard-negative-boundary"},
                )
                self.assertEqual(
                    by_polarity["positive"]["identifier_payload"]["sha256"],
                    by_polarity["hard-negative-exact"]["identifier_payload"]["sha256"],
                )

    def test_challenge_families_are_disjoint_and_replay_floors_hold(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self._fixture(Path(directory))
            manifest, _ = targeted.build_v2_artifacts(config)
            family_split = manifest["template_family_split"]
            self.assertTrue(family_split["disjoint"])
            self.assertFalse(set(family_split["train"]) & set(family_split["challenge"]))
            self.assertTrue(manifest["overlap_audit"]["all_required_zero"])
            self.assertTrue(
                manifest["outputs"]["sealed_synthetic_challenge"]["sealed"]
            )
            coverage = manifest["outputs"]["d2_targeted"]["stats"][
                "replay_coverage"
            ]
            self.assertGreaterEqual(coverage["numeric_iban_rows"], 2)
            self.assertGreaterEqual(coverage["alphanumeric_identifier_rows"], 2)

    def test_numeric_boundary_traps_cover_all_required_lengths(self):
        config = targeted.D2BuildConfig()
        records = targeted._generate_d2_synthetic(
            config=config,
            split="train",
            positive_rows=config.train_positive_rows,
            hard_negative_rows=config.train_hard_negative_rows,
            dataset_role="d2_targeted",
            sealed=False,
        )
        lengths = {
            int(record["boundary_trap"].split("-")[2])
            for record in records
            if str(record.get("boundary_trap") or "").startswith("numeric-length-")
        }
        self.assertEqual(lengths, set(targeted.D2_NUMERIC_TRAP_LENGTHS))

    def test_seed_changes_v2_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self._fixture(Path(directory))
            _, first = targeted.build_v2_artifacts(config)
            _, second = targeted.build_v2_artifacts(replace(config, seed=20260823))
            self.assertNotEqual(
                first[targeted.D2_OUTPUT_FILENAMES["d2_targeted"]],
                second[targeted.D2_OUTPUT_FILENAMES["d2_targeted"]],
            )


if __name__ == "__main__":
    unittest.main()
