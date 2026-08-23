import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.training.distill_student import (
    DEFAULT_ALPHA,
    DEFAULT_EPOCHS,
    DEFAULT_KD_MAX_COVERAGE,
    DEFAULT_KD_MIN_COVERAGE,
    DEFAULT_KD_MODE,
    DEFAULT_LEARNING_RATE,
    DEFAULT_SEED,
    DEFAULT_TEMPERATURE,
    DISTILL_INDEX,
    DistillationCollator,
    _IndexedDataset,
    _audit_kd_cache_targets,
    _bio_label_contract,
    _enforce_pretraining_memory_guard,
    _finalize_run_lifecycle,
    _load_parent_context,
    _memory_guard_policy,
    _new_memory_guard_state,
    _observe_memory_guard,
    _pad_window,
    _parse_args,
    _resolved_configuration,
    _type_boundary_normalized_targets,
    _validate_teacher_cache,
    _validate_recipe,
    compute_distillation_loss,
)
from src.training.student_utils import (
    IGNORE_INDEX,
    StudentTrainingError,
    sha256_file,
    tokenizer_load_policy,
)


LABELS = ["B-FULLNAME", "I-FULLNAME", "O"]
LABEL2ID = {label: index for index, label in enumerate(LABELS)}
ID2LABEL = {str(index): label for label, index in LABEL2ID.items()}


class StudentDistillationTests(unittest.TestCase):
    def _args(self, *extra: str):
        return _parse_args(["train", "--parent-run", "/tmp/parent", *extra])

    def _memory_guard(self):
        policy = _memory_guard_policy(
            max_swap_growth_bytes=20,
            min_available_memory_bytes=500,
        )
        return policy, _new_memory_guard_state(policy, swap_at_start=100)

    def _parent_fixture(self, root: Path, *, status: str = "complete") -> Path:
        run = root / "parent"
        final = run / "final"
        final.mkdir(parents=True)
        config = {"label2id": LABEL2ID, "id2label": ID2LABEL, "model_type": "test"}
        (final / "config.json").write_text(json.dumps(config), encoding="utf-8")
        (final / "model.safetensors").write_bytes(b"synthetic-weights")
        (final / "tokenizer.json").write_text("{}", encoding="utf-8")
        (final / "tokenizer_config.json").write_text("{}", encoding="utf-8")

        contract = root / "teacher-config.json"
        contract.write_text(json.dumps(config), encoding="utf-8")
        train = root / "train.jsonl"
        validation = root / "validation.jsonl"
        train.write_text('{"source_text":"Mario","entities":[]}\n', encoding="utf-8")
        validation.write_text(
            '{"source_text":"Luigi","entities":[]}\n', encoding="utf-8"
        )
        training = {
            "train_file": str(train),
            "validation_file": str(validation),
            "max_length": 256,
            "stride": 32,
            "train_limit": 1,
            "validation_limit": 1,
            "train_batch_size": 1,
            "eval_batch_size": 1,
            "gradient_accumulation_steps": 4,
            "weight_decay": 0.01,
            "warmup_ratio": 0.05,
            "attention_implementation": "sdpa",
            "gradient_checkpointing": True,
            "mps_memory_fraction": 0.75,
            "torch_empty_cache_steps": 1,
            "max_swap_growth_bytes": 2 * 1024**3,
            "min_available_memory_bytes": 512 * 1024**2,
            "seed": DEFAULT_SEED,
        }
        manifest = {
            "schema_version": 1,
            "teacher_label_contract": {
                "path": str(contract),
                "sha256": sha256_file(contract),
            },
            "data": {
                "train": {"path": str(train), "sha256": sha256_file(train), "rows": 1},
                "validation": {
                    "path": str(validation),
                    "sha256": sha256_file(validation),
                    "rows": 1,
                },
            },
            "training": training,
        }
        (run / "run-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        (run / "training-summary.json").write_text(
            json.dumps(
                {
                    "status": status,
                    "global_step": 10,
                    "planned_steps": 10,
                    "input_format": "raw-spans",
                    "total_parameters": 123,
                }
            ),
            encoding="utf-8",
        )
        return run

    def test_cli_defaults_are_the_frozen_first_recipe(self):
        args = self._args("--teacher-cache", "/tmp/cache")
        self.assertEqual(args.alpha, DEFAULT_ALPHA)
        self.assertEqual(args.temperature, DEFAULT_TEMPERATURE)
        self.assertEqual(args.learning_rate, DEFAULT_LEARNING_RATE)
        self.assertEqual(args.epochs, DEFAULT_EPOCHS)
        self.assertEqual(args.seed, DEFAULT_SEED)
        self.assertEqual(args.kd_mode, DEFAULT_KD_MODE)
        self.assertEqual(args.kd_min_coverage, DEFAULT_KD_MIN_COVERAGE)
        self.assertEqual(args.kd_max_coverage, DEFAULT_KD_MAX_COVERAGE)
        _validate_recipe(args)

    def test_positive_alpha_requires_cache_but_gold_only_does_not(self):
        with self.assertRaisesRegex(StudentTrainingError, "teacher-cache"):
            _validate_recipe(self._args())
        gold_only = self._args("--alpha", "0")
        _validate_recipe(gold_only)
        self.assertIsNone(gold_only.teacher_cache)

    def test_gold_only_does_not_resolve_or_load_supplied_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = _load_parent_context(self._parent_fixture(Path(directory)))
            args = self._args(
                "--alpha",
                "0",
                "--teacher-cache",
                "/definitely/not/a/cache",
                "--precision",
                "fp32",
                "--device",
                "cpu",
            )
            _validate_recipe(args)
            values = _resolved_configuration(args, parent)
        self.assertEqual(values["alpha"], 0.0)
        self.assertIsNone(values["teacher_cache"])
        self.assertFalse(values["cache_loaded"])
        self.assertEqual(values["tokenizer_load_policy"], tokenizer_load_policy())

    def test_recipe_rejects_invalid_hyperparameters(self):
        for option, value in (
            ("--alpha", "-0.1"),
            ("--alpha", "1.1"),
            ("--temperature", "0"),
            ("--learning-rate", "0"),
            ("--epochs", "0"),
            ("--max-steps", "0"),
            ("--kd-min-coverage", "-0.1"),
            ("--kd-max-coverage", "1.1"),
        ):
            with self.subTest(option=option, value=value):
                with self.assertRaises(StudentTrainingError):
                    _validate_recipe(
                        self._args(
                            "--teacher-cache", "/tmp/cache", option, value
                        )
                    )

        with self.assertRaisesRegex(StudentTrainingError, "min <= max"):
            _validate_recipe(
                self._args(
                    "--teacher-cache",
                    "/tmp/cache",
                    "--kd-min-coverage",
                    "0.99",
                    "--kd-max-coverage",
                    "0.98",
                )
            )

    def test_available_memory_single_low_is_recorded_without_stop(self):
        policy, guard = self._memory_guard()
        reason = _observe_memory_guard(
            {"swap_used_bytes": 110, "system_available_bytes": 499},
            phase="trainer_log",
            step=10,
            memory_guard=guard,
        )
        self.assertIsNone(reason)
        self.assertFalse(guard["triggered"])
        self.assertEqual(guard["available_memory_consecutive_below"], 1)
        self.assertEqual(guard["observation_count"], 1)
        self.assertEqual(guard["samples"][0]["step"], 10)
        self.assertTrue(guard["samples"][0]["available_below_threshold"])
        self.assertEqual(
            policy["available_memory"]["consecutive_observations_required"], 2
        )
        self.assertTrue(policy["available_memory"]["reset_counter_on_recovery"])

    def test_available_memory_recovery_resets_counter(self):
        _, guard = self._memory_guard()
        _observe_memory_guard(
            {"swap_used_bytes": 110, "system_available_bytes": 499},
            phase="trainer_log",
            memory_guard=guard,
        )
        reason = _observe_memory_guard(
            {"swap_used_bytes": 110, "system_available_bytes": 900},
            phase="trainer_log",
            memory_guard=guard,
        )
        self.assertIsNone(reason)
        self.assertFalse(guard["triggered"])
        self.assertEqual(guard["available_memory_consecutive_below"], 0)
        self.assertEqual(guard["available_memory_max_consecutive_below"], 1)
        self.assertEqual(
            [sample["available_consecutive_below"] for sample in guard["samples"]],
            [1, 0],
        )

    def test_available_memory_two_consecutive_lows_trigger(self):
        _, guard = self._memory_guard()
        _enforce_pretraining_memory_guard(
            {"swap_used_bytes": 110, "system_available_bytes": 499},
            phase="after_data_and_cache",
            memory_guard=guard,
        )
        with self.assertRaisesRegex(StudentTrainingError, "2 consecutive") as caught:
            _enforce_pretraining_memory_guard(
                {"swap_used_bytes": 111, "system_available_bytes": 450},
                phase="after_model_load",
                memory_guard=guard,
            )
        self.assertTrue(guard["triggered"])
        self.assertEqual(guard["phase"], "after_model_load")
        self.assertEqual(guard["trigger_observation"], 2)
        self.assertIs(getattr(caught.exception, "memory_guard"), guard)

    def test_swap_growth_triggers_on_first_observation(self):
        policy, guard = self._memory_guard()
        reason = _observe_memory_guard(
            {"swap_used_bytes": 121, "system_available_bytes": 1_000},
            phase="after_data_and_cache",
            memory_guard=guard,
        )
        self.assertIn("swap growth", reason)
        self.assertTrue(guard["triggered"])
        self.assertEqual(guard["trigger_observation"], 1)
        self.assertEqual(policy["swap_growth"]["trigger"], "immediate")

    def test_lifecycle_atomically_links_complete_manifest_and_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            manifest = {
                "schema_version": 1,
                "kind": "pii-student-distillation-finetune",
                "status": "prepared",
                "recipe_sha256": "a" * 64,
            }
            summary = {
                "status": "complete",
                "release_eligible": False,
                "kind": "pii-student-distillation-finetune",
            }
            (run / "run-manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            (run / "training-summary.json").write_text(
                json.dumps(summary), encoding="utf-8"
            )
            finalized = _finalize_run_lifecycle(
                run, lifecycle_status="complete", summary=summary
            )
            on_disk = json.loads(
                (run / "run-manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(finalized, on_disk)
            self.assertEqual(on_disk["status"], "complete")
            self.assertEqual(on_disk["result_status"], "complete")
            self.assertEqual(
                on_disk["training_summary"]["sha256"],
                sha256_file(run / "training-summary.json"),
            )
            self.assertEqual(on_disk["recipe_sha256"], "a" * 64)
            with self.assertRaisesRegex(StudentTrainingError, "prepared"):
                _finalize_run_lifecycle(
                    run, lifecycle_status="complete", summary=summary
                )

    def test_lifecycle_failed_requires_and_hashes_failed_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / "run-manifest.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "kind": "pii-student-distillation-finetune",
                        "status": "prepared",
                    }
                ),
                encoding="utf-8",
            )
            complete = {"status": "complete"}
            (run / "training-summary.json").write_text(
                json.dumps(complete), encoding="utf-8"
            )
            with self.assertRaisesRegex(StudentTrainingError, "failed"):
                _finalize_run_lifecycle(
                    run, lifecycle_status="failed", summary=complete
                )

            failed = {"status": "failed", "error": "synthetic"}
            (run / "training-summary.json").write_text(
                json.dumps(failed), encoding="utf-8"
            )
            finalized = _finalize_run_lifecycle(
                run, lifecycle_status="failed", summary=failed
            )
            self.assertEqual(finalized["status"], "failed")
            self.assertEqual(finalized["result_status"], "failed")

    def test_parent_must_be_complete_and_untampered(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            partial = self._parent_fixture(root, status="partial_guard_stop")
            with self.assertRaisesRegex(StudentTrainingError, "completo"):
                _load_parent_context(partial)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = self._parent_fixture(root)
            (root / "train.jsonl").write_text("changed\n", encoding="utf-8")
            with self.assertRaisesRegex(StudentTrainingError, "modificato"):
                _load_parent_context(run)

    def test_parent_context_is_explicitly_weights_only_and_seed_is_fixed(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = _load_parent_context(self._parent_fixture(Path(directory)))
            args = self._args(
                "--alpha",
                "0",
                "--precision",
                "fp32",
                "--device",
                "cpu",
            )
            values = _resolved_configuration(args, parent)
            self.assertEqual(values["optimizer"], "adafactor")
            self.assertEqual(values["seed"], DEFAULT_SEED)
            self.assertEqual(parent["summary"]["status"], "complete")

            normalized_args = self._args(
                "--teacher-cache",
                "/tmp/cache",
                "--kd-mode",
                "type-boundary-normalized",
                "--kd-min-coverage",
                "0.95",
                "--kd-max-coverage",
                "0.99",
                "--precision",
                "fp32",
                "--device",
                "cpu",
            )
            normalized = _resolved_configuration(normalized_args, parent)
            self.assertEqual(normalized["kd_mode"], "type-boundary-normalized")
            self.assertEqual(normalized["kd_min_coverage"], 0.95)
            self.assertEqual(normalized["kd_max_coverage"], 0.99)

            args.seed += 1
            with self.assertRaisesRegex(StudentTrainingError, "seed"):
                _resolved_configuration(args, parent)

    def test_gold_only_loss_matches_masked_cross_entropy_without_teacher(self):
        import torch
        import torch.nn.functional as functional

        logits = torch.tensor(
            [[[3.0, 0.0, -1.0], [100.0, -100.0, 0.0], [0.0, 2.0, -1.0]]]
        )
        labels = torch.tensor([[0, IGNORE_INDEX, 1]])
        actual = compute_distillation_loss(
            logits,
            labels,
            teacher_logits=None,
            alpha=0.0,
            temperature=2.0,
        )
        expected = functional.cross_entropy(logits[0, [0, 2]], torch.tensor([0, 1]))
        torch.testing.assert_close(actual, expected)

    def test_distillation_formula_and_ignore_mask_are_exact(self):
        import torch
        import torch.nn.functional as functional

        student = torch.tensor(
            [[[2.0, 0.0, -1.0], [0.0, 0.0, 0.0], [-1.0, 2.0, 0.0]]]
        )
        labels = torch.tensor([[0, IGNORE_INDEX, 1]])
        teacher = torch.tensor(
            [[[3.0, -1.0, 0.0], [9999.0, -9999.0, 1.0], [0.0, 4.0, -1.0]]]
        )
        alpha = 0.25
        temperature = 2.0
        actual = compute_distillation_loss(
            student,
            labels,
            teacher_logits=teacher,
            alpha=alpha,
            temperature=temperature,
        )
        selected_student = student[0, [0, 2]].float()
        selected_teacher = teacher[0, [0, 2]].float()
        ce = functional.cross_entropy(selected_student, torch.tensor([0, 1]))
        kl = functional.kl_div(
            functional.log_softmax(selected_student / temperature, dim=-1),
            functional.softmax(selected_teacher / temperature, dim=-1),
            reduction="batchmean",
        )
        expected = (1 - alpha) * ce + alpha * temperature**2 * kl
        torch.testing.assert_close(actual, expected)

        changed_only_on_ignored = teacher.clone()
        changed_only_on_ignored[0, 1] = torch.tensor([-1e6, 1e6, 0.0])
        unchanged = compute_distillation_loss(
            student,
            labels,
            teacher_logits=changed_only_on_ignored,
            alpha=alpha,
            temperature=temperature,
        )
        torch.testing.assert_close(actual, unchanged)

        nan_only_on_ignored = teacher.clone()
        nan_only_on_ignored[0, 1, 0] = float("nan")
        still_unchanged = compute_distillation_loss(
            student,
            labels,
            teacher_logits=nan_only_on_ignored,
            alpha=alpha,
            temperature=temperature,
        )
        torch.testing.assert_close(actual, still_unchanged)

    def test_index_free_loss_matches_masked_reference_loss_and_gradient(self):
        import torch
        import torch.nn.functional as functional

        alpha = 0.25
        temperature = 2.0
        labels = torch.tensor(
            [[0, IGNORE_INDEX, 2], [1, 0, IGNORE_INDEX]], dtype=torch.long
        )
        initial = torch.tensor(
            [
                [[2.0, -1.0, 0.5], [5.0, -4.0, 1.0], [0.0, -2.0, 3.0]],
                [[-1.0, 2.5, 0.0], [1.5, 0.0, -0.5], [8.0, -3.0, 2.0]],
            ],
            dtype=torch.float64,
        )
        teacher = torch.tensor(
            [
                [[3.0, -2.0, 0.0], [100.0, -100.0, 0.0], [0.0, -1.0, 4.0]],
                [[-2.0, 4.0, 0.0], [2.0, 0.0, -1.0], [-50.0, 50.0, 0.0]],
            ],
            dtype=torch.float64,
        )

        index_free_logits = initial.clone().requires_grad_(True)
        index_free = compute_distillation_loss(
            index_free_logits,
            labels,
            teacher_logits=teacher,
            alpha=alpha,
            temperature=temperature,
        )
        index_free.backward()

        reference_logits = initial.clone().requires_grad_(True)
        mask = labels.ne(IGNORE_INDEX)
        selected_student = reference_logits[mask]
        selected_teacher = teacher[mask]
        selected_labels = labels[mask]
        ce = functional.cross_entropy(selected_student.float(), selected_labels)
        kl = functional.kl_div(
            functional.log_softmax(selected_student.float() / temperature, dim=-1),
            functional.softmax(selected_teacher.float() / temperature, dim=-1),
            reduction="batchmean",
        )
        reference = (1.0 - alpha) * ce + alpha * temperature**2 * kl
        reference.backward()

        torch.testing.assert_close(index_free, reference)
        torch.testing.assert_close(index_free_logits.grad, reference_logits.grad)
        self.assertTrue(
            torch.equal(
                index_free_logits.grad[~mask],
                torch.zeros_like(index_free_logits.grad[~mask]),
            )
        )

    def test_boundary_normalization_is_minimal_and_conserves_mass(self):
        import torch
        import torch.nn.functional as functional

        labels = ["B-X", "I-X", "B-Y", "I-Y", "O"]
        teacher = torch.tensor(
            [
                [
                    [0.2, 3.0, 0.7, -0.4, -1.0],
                    [-0.2, -1.0, 3.2, 0.5, -2.0],
                    [-1.0, -2.0, -3.0, -4.0, 4.0],
                    [4.0, 0.0, -1.0, -2.0, -3.0],
                ]
            ],
            dtype=torch.float64,
        )
        gold = torch.tensor([[0, 3, 4, 0]])
        targets, active, hard = _type_boundary_normalized_targets(
            teacher,
            gold,
            label_names=labels,
            temperature=2.0,
        )
        raw = functional.softmax(teacher.float() / 2.0, dim=-1)
        expected_b = torch.stack(
            (raw[0, 0, 0] + raw[0, 0, 1], torch.tensor(0.0),
             raw[0, 0, 2], raw[0, 0, 3], raw[0, 0, 4])
        )
        expected_i = torch.stack(
            (raw[0, 1, 0], raw[0, 1, 1],
             torch.tensor(0.0), raw[0, 1, 2] + raw[0, 1, 3], raw[0, 1, 4])
        )
        torch.testing.assert_close(targets[0, 0], expected_b)
        torch.testing.assert_close(targets[0, 1], expected_i)
        torch.testing.assert_close(targets[0, 2], raw[0, 2])
        # Un exact-active entity target must remain exactly unchanged too.
        self.assertTrue(torch.equal(targets[0, 3], raw[0, 3]))
        # Other entity types are untouched on projected tokens.
        self.assertTrue(torch.equal(targets[0, 0, 2:], raw[0, 0, 2:]))
        self.assertTrue(torch.equal(targets[0, 1, :2], raw[0, 1, :2]))
        torch.testing.assert_close(
            targets.sum(dim=-1), torch.ones_like(targets[..., 0])
        )
        self.assertTrue(
            torch.equal(active, torch.tensor([[True, True, True, True]]))
        )
        self.assertTrue(torch.equal(hard, torch.tensor([[1, 2, 4, 0]])))

    def test_normalized_loss_value_gradient_and_type_mask_match_reference(self):
        import torch
        import torch.nn.functional as functional

        names = ["B-X", "I-X", "B-Y", "I-Y", "O"]
        gold = torch.tensor([[0, 2, 4, IGNORE_INDEX]])
        teacher = torch.tensor(
            [[
                [0.0, 4.0, -1.0, -2.0, -3.0],  # same X, BIO-only
                [4.0, -1.0, 1.0, 0.0, -2.0],   # X vs gold Y: excluded
                [-2.0, -1.0, -3.0, -4.0, 3.0], # O exact
                [100.0, -100.0, 0.0, 0.0, 0.0],
            ]],
            dtype=torch.float64,
        )
        initial = torch.tensor(
            [[
                [1.0, 0.0, -0.5, -1.0, 0.2],
                [-0.4, 0.5, 1.2, -0.3, 0.0],
                [-1.0, -0.5, 0.0, 0.5, 1.0],
                [9.0, -9.0, 1.0, 0.0, 0.0],
            ]],
            dtype=torch.float64,
        )
        alpha = 0.1
        temperature = 2.0

        actual_logits = initial.clone().requires_grad_(True)
        actual = compute_distillation_loss(
            actual_logits,
            gold,
            teacher_logits=teacher,
            alpha=alpha,
            temperature=temperature,
            kd_mode="type-boundary-normalized",
            label_names=names,
        )
        actual.backward()

        reference_logits = initial.clone().requires_grad_(True)
        targets, active, _ = _type_boundary_normalized_targets(
            teacher,
            gold,
            label_names=names,
            temperature=temperature,
        )
        ce = functional.cross_entropy(
            reference_logits.float().reshape(-1, len(names)),
            gold.reshape(-1),
            ignore_index=IGNORE_INDEX,
        )
        kl = functional.kl_div(
            functional.log_softmax(
                reference_logits.float()[active] / temperature, dim=-1
            ),
            targets[active],
            reduction="batchmean",
        )
        expected = (1.0 - alpha) * ce + alpha * temperature**2 * kl
        expected.backward()
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(actual_logits.grad, reference_logits.grad)
        self.assertTrue(torch.equal(active, torch.tensor([[True, False, True, False]])))

    def test_normalized_loss_with_no_active_teacher_token_fails_closed(self):
        import torch

        names = ["B-X", "I-X", "B-Y", "I-Y", "O"]
        logits = torch.tensor(
            [[[1.0, 0.0, -1.0, -2.0, 0.5], [0.0, 1.0, 0.5, -1.0, 2.0]]]
        )
        gold = torch.tensor([[0, 4]])
        teacher = torch.tensor(
            [[[-2.0, -3.0, 4.0, 0.0, -1.0], [4.0, 0.0, -2.0, -3.0, -1.0]]]
        )
        with self.assertRaisesRegex(StudentTrainingError, "privo di token attivi"):
            compute_distillation_loss(
                logits,
                gold,
                teacher_logits=teacher,
                alpha=0.25,
                temperature=2.0,
                kd_mode="type-boundary-normalized",
                label_names=names,
            )

    def test_normalized_mode_rejects_malformed_bio_contract(self):
        import torch

        with self.assertRaisesRegex(StudentTrainingError, "coppie B/I"):
            _bio_label_contract(["B-X", "O"])
        with self.assertRaisesRegex(StudentTrainingError, "BIO"):
            compute_distillation_loss(
                torch.zeros((1, 1, 2)),
                torch.tensor([[0]]),
                teacher_logits=torch.zeros((1, 1, 2)),
                alpha=0.25,
                temperature=2.0,
                kd_mode="type-boundary-normalized",
                label_names=["PERSON", "O"],
            )

    def test_normalized_preflight_reports_gate_stats_and_enforces_coverage(self):
        import torch

        names = ["B-X", "I-X", "B-Y", "I-Y", "O"]
        gold = torch.tensor([[0, 0, 1, 4, 4, IGNORE_INDEX]])
        hard = [0, 1, 2, 4, 0, 0]
        teacher = torch.full((1, 6, len(names)), -5.0)
        for position, label_id in enumerate(hard):
            teacher[0, position, label_id] = 5.0
        input_ids = torch.tensor([[10, 11, 12, 13, 99, 0]])

        class Tokenizer:
            def convert_ids_to_tokens(self, token_id):
                return "▁" if token_id == 99 else f"t{token_id}"

        stats = _audit_kd_cache_targets(
            teacher_logits=teacher,
            labels=gold,
            input_ids=input_ids,
            tokenizer=Tokenizer(),
            label_names=names,
            temperature=2.0,
            kd_mode="type-boundary-normalized",
            min_coverage=0.59,
            max_coverage=0.61,
        )
        self.assertEqual(stats["supervised_tokens"], 5)
        self.assertEqual(stats["active_kd_tokens"], 3)
        self.assertEqual(stats["coverage"], 0.6)
        self.assertEqual(stats["zero_active_windows"], 0)
        self.assertEqual(stats["min_active_tokens_per_window"], 3)
        self.assertEqual(stats["max_active_tokens_per_window"], 3)
        self.assertEqual(stats["exact_agreement_tokens"], 2)
        self.assertEqual(stats["exact_active_unchanged_tokens"], 2)
        self.assertEqual(stats["bio_only_same_type_tokens"], 1)
        self.assertEqual(stats["projected_bio_only_tokens"], 1)
        self.assertEqual(stats["type_mismatch_excluded_tokens"], 2)
        self.assertEqual(stats["active_type_mismatch_tokens"], 0)
        self.assertEqual(stats["gold_o_boundary_teacher_entity_tokens"], 1)
        self.assertEqual(
            stats["active_gold_o_boundary_teacher_entity_tokens"], 0
        )
        self.assertEqual(stats["boundary_marker_token_ids"], [99])
        self.assertTrue(stats["probabilities_finite"])

        with self.assertRaisesRegex(StudentTrainingError, "Coverage"):
            _audit_kd_cache_targets(
                teacher_logits=teacher,
                labels=gold,
                input_ids=input_ids,
                tokenizer=Tokenizer(),
                label_names=names,
                temperature=2.0,
                kd_mode="type-boundary-normalized",
                min_coverage=0.96,
                max_coverage=0.98,
            )

        no_active_teacher = torch.full_like(teacher, -5.0)
        # Tutti i token supervisionati hanno un hard type diverso dal gold.
        for position, label_id in enumerate([2, 2, 2, 0, 0, 0]):
            no_active_teacher[0, position, label_id] = 5.0
        with self.assertRaisesRegex(StudentTrainingError, "finestre prive"):
            _audit_kd_cache_targets(
                teacher_logits=no_active_teacher,
                labels=gold,
                input_ids=input_ids,
                tokenizer=Tokenizer(),
                label_names=names,
                temperature=2.0,
                kd_mode="type-boundary-normalized",
                min_coverage=0.0,
                max_coverage=1.0,
            )

    def test_distillation_loss_fails_closed(self):
        import torch

        student = torch.zeros((1, 2, 3))
        labels = torch.tensor([[0, 1]])
        with self.assertRaisesRegex(StudentTrainingError, "mancanti"):
            compute_distillation_loss(
                student, labels, teacher_logits=None, alpha=0.25, temperature=2.0
            )
        with self.assertRaisesRegex(StudentTrainingError, "shape"):
            compute_distillation_loss(
                student,
                labels,
                teacher_logits=torch.zeros((1, 3, 3)),
                alpha=0.25,
                temperature=2.0,
            )
        with self.assertRaisesRegex(StudentTrainingError, "non finiti"):
            teacher = torch.zeros_like(student)
            teacher[0, 0, 0] = float("nan")
            compute_distillation_loss(
                student, labels, teacher_logits=teacher, alpha=0.25, temperature=2.0
            )
        with self.assertRaisesRegex(StudentTrainingError, "privo"):
            compute_distillation_loss(
                student,
                torch.full_like(labels, IGNORE_INDEX),
                teacher_logits=torch.zeros_like(student),
                alpha=0.25,
                temperature=2.0,
            )

    def test_padding_contract_handles_both_sides(self):
        feature = {
            "input_ids": [10, 11],
            "attention_mask": [1, 1],
            "labels": [0, 2],
            "token_type_ids": [0, 0],
        }
        right = _pad_window(
            feature,
            max_length=4,
            pad_token_id=7,
            padding_side="right",
        )
        self.assertEqual(right["input_ids"], [10, 11, 7, 7])
        self.assertEqual(right["labels"], [0, 2, IGNORE_INDEX, IGNORE_INDEX])
        left = _pad_window(
            feature,
            max_length=4,
            pad_token_id=7,
            padding_side="left",
        )
        self.assertEqual(left["input_ids"], [7, 7, 10, 11])
        self.assertEqual(left["attention_mask"], [0, 0, 1, 1])

    def test_indexed_dataset_is_stable_and_does_not_mutate_source(self):
        source = [{"input_ids": [1, 2], "labels": [0, 2]}]
        indexed = _IndexedDataset(source)
        row = indexed[0]
        self.assertEqual(row[DISTILL_INDEX], 0)
        self.assertNotIn(DISTILL_INDEX, source[0])

    def test_collator_adds_only_the_selected_teacher_windows(self):
        import torch

        class BaseCollator:
            def __call__(self, features):
                return {
                    "input_ids": torch.tensor([feature["input_ids"] for feature in features]),
                    "labels": torch.tensor([feature["labels"] for feature in features]),
                }

        logits = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)
        collator = DistillationCollator(
            BaseCollator(), teacher_logits=logits, padding_side="right"
        )
        batch = collator(
            [
                {"input_ids": [1, 2], "labels": [0, 2], DISTILL_INDEX: 1},
            ]
        )
        torch.testing.assert_close(batch["teacher_logits"], logits[[1], :2])
        self.assertNotIn(DISTILL_INDEX, batch)

        gold_batch = collator([{"input_ids": [1, 2], "labels": [0, 2]}])
        self.assertNotIn("teacher_logits", gold_batch)

    def test_trainer_revalidates_every_cached_window_against_tokenized_data(self):
        import torch

        from src.training.cache_teacher_logits import window_alignment_sha256

        tokenized = [
            {"input_ids": [10, 11], "attention_mask": [1, 1], "labels": [0, 2]}
        ]
        padded = {
            "input_ids": [10, 11, 7, 7],
            "attention_mask": [1, 1, 0, 0],
            "labels": [0, 2, IGNORE_INDEX, IGNORE_INDEX],
        }
        alignment = window_alignment_sha256(**padded)
        manifest = {
            "student": {"run_manifest": {"sha256": "a" * 64}},
            "label_contract": {"labels": LABELS},
            "preprocessing": {
                "input_format": "raw-spans",
                "max_length": 4,
                "stride": 1,
                "train_limit": 1,
                "seed": DEFAULT_SEED,
                "padding_side": "right",
                "ignore_index": IGNORE_INDEX,
            },
            "windows": [
                {"index": 0, "length": 2, "alignment_sha256": alignment}
            ],
        }
        tensors = {
            "teacher_logits": torch.zeros((1, 4, len(LABELS)), dtype=torch.float16),
            "input_ids": torch.tensor([padded["input_ids"]], dtype=torch.int32),
            "attention_mask": torch.tensor(
                [padded["attention_mask"]], dtype=torch.uint8
            ),
            "labels": torch.tensor([padded["labels"]], dtype=torch.int16),
        }
        parent = {
            "data": {"train": {"sha256": "b" * 64}},
            "model_integrity": {"sha256": "c" * 64},
            "tokenizer_integrity": {"sha256": "d" * 64},
            "manifest_integrity": {"sha256": "a" * 64},
            "labels": LABELS,
        }
        values = {
            "max_length": 4,
            "stride": 1,
            "seed": DEFAULT_SEED,
            "train_limit": 1,
        }
        tokenizer = SimpleNamespace(
            pad_token_id=7, padding_side="right", pad_token_type_id=0
        )
        with patch(
            "src.training.cache_teacher_logits.load_teacher_cache",
            return_value=(manifest, tensors),
        ):
            loaded_manifest, loaded = _validate_teacher_cache(
                "/synthetic/cache",
                tokenized_train=tokenized,
                tokenizer=tokenizer,
                parent=parent,
                values=values,
            )
        self.assertIs(loaded_manifest, manifest)
        self.assertIs(loaded, tensors)

        corrupted = dict(tensors)
        corrupted["input_ids"] = tensors["input_ids"].clone()
        corrupted["input_ids"][0, 0] = 99
        with patch(
            "src.training.cache_teacher_logits.load_teacher_cache",
            return_value=(manifest, corrupted),
        ):
            with self.assertRaisesRegex(StudentTrainingError, "input_ids"):
                _validate_teacher_cache(
                    "/synthetic/cache",
                    tokenized_train=tokenized,
                    tokenizer=tokenizer,
                    parent=parent,
                    values=values,
                )


if __name__ == "__main__":
    unittest.main()
