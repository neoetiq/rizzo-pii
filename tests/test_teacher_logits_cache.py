import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from src.training.cache_teacher_logits import (
    CACHE_KIND,
    TeacherLogitsCacheError,
    artifact_group_integrity,
    assert_tokenizers_compatible,
    create_teacher_logits_cache,
    load_and_validate_cache_manifest,
    load_teacher_cache,
    window_alignment_sha256,
)
from src.training.student_utils import sha256_file, tokenizer_load_policy


LABEL2ID = {"B-FULLNAME": 0, "I-FULLNAME": 1, "O": 2}
ID2LABEL = {str(index): label for label, index in LABEL2ID.items()}


class FakeTokenizer:
    pad_token_type_id = 0
    padding_side = "right"

    def __init__(self, *, replacement_at_10=None):
        self._vocab = {f"tok-{index}": index for index in range(64)}
        if replacement_at_10 is not None:
            self._vocab.pop("tok-10")
            self._vocab[str(replacement_at_10)] = 10
        self.vocab_size = 64
        self.model_input_names = ["input_ids", "attention_mask", "token_type_ids"]
        self.pad_token, self.pad_token_id = "tok-0", 0
        self.bos_token, self.bos_token_id = "tok-1", 1
        self.eos_token, self.eos_token_id = "tok-2", 2
        self.sep_token, self.sep_token_id = "tok-2", 2
        self.cls_token, self.cls_token_id = "tok-1", 1
        self.unk_token, self.unk_token_id = "tok-3", 3
        self.mask_token, self.mask_token_id = "tok-4", 4
        self.all_special_ids = [0, 2, 1, 3, 4]
        self.all_special_tokens = ["tok-0", "tok-2", "tok-1", "tok-3", "tok-4"]
        self.additional_special_tokens_ids = []
        self.additional_special_tokens = []
        self.added_tokens_decoder = {
            index: SimpleNamespace(
                content=f"tok-{index}",
                special=True,
                single_word=False,
                lstrip=False,
                rstrip=False,
                normalized=False,
            )
            for index in range(5)
        }

    def __len__(self):
        return len(self._vocab)

    def get_vocab(self):
        return dict(self._vocab)

    def get_added_vocab(self):
        return {f"tok-{index}": index for index in range(5)}

    def __call__(self, texts, **kwargs):
        mapping = []
        input_ids = []
        attention_mask = []
        offsets = []
        special_tokens_mask = []
        token_type_ids = []
        for sample_index, text in enumerate(texts):
            # Tiny deterministic stand-in; documents in these tests fit in one
            # window, while offsets exercise the real raw-span aligner.
            ids = [1, *[10 + (ord(char) % 50) for char in text], 2]
            mapping.append(sample_index)
            input_ids.append(ids)
            attention_mask.append([1] * len(ids))
            offsets.append([(0, 0), *[(i, i + 1) for i in range(len(text))], (0, 0)])
            special_tokens_mask.append([1, *([0] * len(text)), 1])
            token_type_ids.append([0] * len(ids))
        return {
            "overflow_to_sample_mapping": mapping,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "offset_mapping": offsets,
            "special_tokens_mask": special_tokens_mask,
            "token_type_ids": token_type_ids,
        }


class TinyTeacher:
    def __init__(self, torch, *, on_call=None):
        self.torch = torch
        self.config = SimpleNamespace(num_labels=len(LABEL2ID))
        self.on_call = on_call
        self.calls = 0

    def to(self, device):
        self.device = device
        return self

    def float(self):
        return self

    def eval(self):
        return self

    def __call__(self, *, input_ids, attention_mask, token_type_ids):
        self.calls += 1
        if self.on_call is not None:
            self.on_call()
        values = input_ids.to(dtype=self.torch.float32)
        logits = self.torch.stack((values / 10.0, -values / 10.0, values * 0.0), dim=-1)
        return SimpleNamespace(logits=logits)


class TeacherLogitsCacheTests(unittest.TestCase):
    def _fixture(self, root: Path):
        import torch

        run = root / "student-run"
        student = run / "final"
        teacher = root / "teacher"
        student.mkdir(parents=True)
        teacher.mkdir()
        config = {"label2id": LABEL2ID, "id2label": ID2LABEL, "num_labels": 3}
        for model_dir in (student, teacher):
            (model_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
            (model_dir / "model.safetensors").write_bytes(b"synthetic-model")
            (model_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
            (model_dir / "tokenizer_config.json").write_text("{}", encoding="utf-8")

        dataset = root / "train.jsonl"
        rows = [
            {
                "source_text": "Mario",
                "entities": [{"start": 0, "end": 5, "label": "FULLNAME"}],
            },
            {"source_text": "nessuno", "entities": []},
        ]
        dataset.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        run_manifest = run / "run-manifest.json"
        run_manifest.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "data": {
                        "train": {
                            "path": str(dataset),
                            "sha256": sha256_file(dataset),
                            "rows": len(rows),
                        }
                    },
                    "training": {
                        "input_format": "raw-spans",
                        "max_length": 16,
                        "stride": 2,
                        "seed": 20260822,
                        "train_limit": len(rows),
                    },
                }
            ),
            encoding="utf-8",
        )
        training_summary = run / "training-summary.json"
        training_summary.write_text(
            json.dumps(
                {
                    "status": "complete",
                    "global_step": 2,
                    "planned_steps": 2,
                    "input_format": "raw-spans",
                }
            ),
            encoding="utf-8",
        )
        return {
            "torch": torch,
            "run": run,
            "student": student,
            "teacher": teacher,
            "dataset": dataset,
            "run_manifest": run_manifest,
            "training_summary": training_summary,
            "output": root / "cache",
        }

    def _create(
        self,
        fixture,
        *,
        teacher_tokenizer=None,
        model=None,
        student_model=None,
        save_file_function=None,
    ):
        return create_teacher_logits_cache(
            student_model=student_model or fixture["student"],
            student_run_manifest=fixture["run_manifest"],
            teacher_model=fixture["teacher"],
            output_dir=fixture["output"],
            tokenizer=FakeTokenizer(),
            teacher_tokenizer=teacher_tokenizer or FakeTokenizer(),
            model=model or TinyTeacher(fixture["torch"]),
            torch_module=fixture["torch"],
            save_file_function=save_file_function,
            batch_size=2,
            tokenize_batch_size=1,
        )

    def test_builds_fixed_shape_fp16_cache_and_validates_alignment(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self._fixture(Path(directory))
            created = self._create(fixture)

            self.assertEqual(created["kind"], CACHE_KIND)
            self.assertEqual(created["preprocessing"]["seed"], 20260822)
            self.assertEqual(
                created["preprocessing"]["tokenizer_load_policy"],
                tokenizer_load_policy(),
            )
            self.assertEqual(
                created["resolved_config"]["tokenizer_load_policy"],
                tokenizer_load_policy(),
            )
            self.assertEqual(created["counts"], {"records": 2, "windows": 2})
            manifest, tensors = load_teacher_cache(
                fixture["output"],
                expected={
                    "dataset_sha256": sha256_file(fixture["dataset"]),
                    "student_model_sha256": artifact_group_integrity(
                        fixture["student"], "model"
                    )["sha256"],
                    "student_tokenizer_sha256": artifact_group_integrity(
                        fixture["student"], "tokenizer"
                    )["sha256"],
                    "teacher_model_sha256": artifact_group_integrity(
                        fixture["teacher"], "model"
                    )["sha256"],
                    "teacher_tokenizer_sha256": artifact_group_integrity(
                        fixture["teacher"], "tokenizer"
                    )["sha256"],
                    "max_length": 16,
                    "stride": 2,
                    "seed": 20260822,
                    "num_labels": 3,
                    "num_windows": 2,
                },
            )
            self.assertEqual(tensors["teacher_logits"].dtype, fixture["torch"].float16)
            self.assertEqual(tuple(tensors["teacher_logits"].shape), (2, 16, 3))
            self.assertEqual(tuple(tensors["labels"].shape), (2, 16))
            self.assertIn("token_type_ids", tensors)
            self.assertEqual(
                manifest["student"]["tokenizer_identity"],
                manifest["teacher"]["tokenizer_identity"],
            )
            self.assertEqual(
                manifest["inputs_snapshot_sha256"],
                manifest["resolved_config"]["inputs_snapshot_sha256"],
            )
            self.assertEqual(
                manifest["windows"][0]["alignment_sha256"],
                window_alignment_sha256(
                    input_ids=tensors["input_ids"][0].tolist(),
                    attention_mask=tensors["attention_mask"][0].tolist(),
                    labels=tensors["labels"][0].tolist(),
                    extra_inputs={
                        "token_type_ids": tensors["token_type_ids"][0].tolist()
                    },
                ),
            )

    def test_output_is_new_only_and_dataset_contract_is_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self._fixture(Path(directory))
            self._create(fixture)
            with self.assertRaisesRegex(TeacherLogitsCacheError, "overwrite"):
                self._create(fixture)

        with tempfile.TemporaryDirectory() as directory:
            fixture = self._fixture(Path(directory))
            fixture["dataset"].write_text(
                fixture["dataset"].read_text(encoding="utf-8")
                + json.dumps({"source_text": "extra", "entities": []})
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(TeacherLogitsCacheError, "Hash dataset"):
                self._create(fixture)
            self.assertFalse(fixture["output"].exists())

    def test_tampering_and_expected_mismatch_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self._fixture(Path(directory))
            self._create(fixture)
            with self.assertRaisesRegex(TeacherLogitsCacheError, "incompatibile"):
                load_and_validate_cache_manifest(
                    fixture["output"], {"seed": 7}
                )

            cache_file = fixture["output"] / "cache.safetensors"
            payload = bytearray(cache_file.read_bytes())
            payload[-1] ^= 1
            cache_file.write_bytes(payload)
            with self.assertRaisesRegex(TeacherLogitsCacheError, "Integrita"):
                load_and_validate_cache_manifest(fixture["output"])

    def test_label_order_mismatch_is_rejected_before_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self._fixture(Path(directory))
            mismatched = {
                "label2id": {"B-FULLNAME": 1, "I-FULLNAME": 0, "O": 2},
                "id2label": {"0": "I-FULLNAME", "1": "B-FULLNAME", "2": "O"},
                "num_labels": 3,
            }
            (fixture["teacher"] / "config.json").write_text(
                json.dumps(mismatched), encoding="utf-8"
            )
            with self.assertRaisesRegex(TeacherLogitsCacheError, "Ordine label"):
                self._create(fixture)
            self.assertFalse(fixture["output"].exists())

    def test_teacher_tokenizer_mapping_mismatch_is_rejected_before_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self._fixture(Path(directory))
            model = TinyTeacher(fixture["torch"])
            with self.assertRaisesRegex(
                TeacherLogitsCacheError, "Tokenizer teacher/student incompatibili"
            ):
                self._create(
                    fixture,
                    teacher_tokenizer=FakeTokenizer(replacement_at_10="changed-token"),
                    model=model,
                )
            self.assertEqual(model.calls, 0)
            self.assertFalse(fixture["output"].exists())

    def test_added_tokens_special_ids_and_vocab_size_are_part_of_identity(self):
        mutations = []
        changed_added = FakeTokenizer()
        changed_added.added_tokens_decoder[4].lstrip = True
        mutations.append(changed_added)
        changed_special = FakeTokenizer()
        changed_special.pad_token_id = 5
        mutations.append(changed_special)
        changed_size = FakeTokenizer()
        changed_size.vocab_size = 63
        mutations.append(changed_size)

        for teacher_tokenizer in mutations:
            with self.subTest(tokenizer=teacher_tokenizer):
                with self.assertRaises(TeacherLogitsCacheError):
                    assert_tokenizers_compatible(FakeTokenizer(), teacher_tokenizer)

    def test_input_mutation_during_inference_aborts_without_publishing(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self._fixture(Path(directory))
            weight_path = fixture["teacher"] / "model.safetensors"

            def mutate_teacher():
                weight_path.write_bytes(b"mutated-during-inference")

            model = TinyTeacher(fixture["torch"], on_call=mutate_teacher)
            with self.assertRaisesRegex(TeacherLogitsCacheError, "Input mutati"):
                self._create(fixture, model=model)
            self.assertGreater(model.calls, 0)
            self.assertFalse(fixture["output"].exists())
            self.assertFalse(any(fixture["output"].parent.glob(".cache.tmp-*")))

    def test_final_rehash_catches_mutation_while_writing_staging_cache(self):
        from safetensors.torch import save_file

        with tempfile.TemporaryDirectory() as directory:
            fixture = self._fixture(Path(directory))

            def save_then_mutate(tensors, path, metadata):
                save_file(tensors, path, metadata=metadata)
                fixture["dataset"].write_text(
                    fixture["dataset"].read_text(encoding="utf-8") + " ",
                    encoding="utf-8",
                )

            with self.assertRaisesRegex(TeacherLogitsCacheError, "Input mutati"):
                self._create(fixture, save_file_function=save_then_mutate)
            self.assertFalse(fixture["output"].exists())
            self.assertFalse(any(fixture["output"].parent.glob(".cache.tmp-*")))

    def test_parent_must_be_complete_and_model_must_be_exact_final(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self._fixture(Path(directory))
            fixture["training_summary"].write_text(
                json.dumps(
                    {
                        "status": "partial_guard_stop",
                        "global_step": 1,
                        "planned_steps": 2,
                        "input_format": "raw-spans",
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(TeacherLogitsCacheError, "non e' completo"):
                self._create(fixture)

        with tempfile.TemporaryDirectory() as directory:
            fixture = self._fixture(Path(directory))
            with self.assertRaisesRegex(TeacherLogitsCacheError, "esattamente <parent>/final"):
                self._create(fixture, student_model=fixture["run"])


if __name__ == "__main__":
    unittest.main()
