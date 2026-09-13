"""Regression tests for the honest PyTorch/ONNX low-bit capability probe."""

from __future__ import annotations

import importlib.util
import unittest


HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_ONNX_STACK = HAS_TORCH and all(
    importlib.util.find_spec(package) is not None
    for package in ("numpy", "onnx", "onnxruntime")
)

if HAS_TORCH:
    import torch

    from src.quantization.lowbit_probe import (
        AtenInt4Linear,
        PackedInt4Embedding,
        _pack_aten_int4_weight,
        assert_fused_int4_kernel_coverage,
        aten_int4_capability,
        benchmark_aten_int4_shape,
        benchmark_onnx_accuracy_matrix,
        estimate_groupwise_storage,
        modernbert_lowbit_storage_matrix,
        probe_int6_capability,
        quantize_model_int4_aten,
    )


@unittest.skipUnless(HAS_TORCH, "PyTorch non installato")
class StorageAndCapabilityTests(unittest.TestCase):
    def test_w6_estimate_counts_packed_bits_and_metadata_separately(self):
        report = estimate_groupwise_storage(
            [(2, 128)], bits=6, group_size=128, scale_bytes=4
        )

        self.assertTrue(report["estimate_only"])
        self.assertEqual(192, report["row_aligned_packed_weight_bytes"])
        self.assertEqual(2, report["groups"])
        self.assertEqual(8, report["metadata_bytes"])
        self.assertEqual(200, report["total_bytes"])
        self.assertAlmostEqual(200 / 1024, report["ratio_to_fp32"])

    def test_modernbert_storage_matrix_matches_audited_parameter_count(self):
        matrix = modernbert_lowbit_storage_matrix(group_size=128)

        self.assertEqual(307_529_472, matrix["w6_symmetric"]["elements"])
        self.assertEqual(163_375_032, matrix["w4_symmetric"]["total_bytes"])
        self.assertEqual(240_257_400, matrix["w6_symmetric"]["total_bytes"])
        self.assertEqual(317_139_768, matrix["w8_symmetric"]["total_bytes"])
        self.assertLess(
            matrix["w4_symmetric"]["total_bytes"],
            matrix["w6_symmetric"]["total_bytes"],
        )
        self.assertLess(
            matrix["w6_symmetric"]["total_bytes"],
            matrix["w8_symmetric"]["total_bytes"],
        )

    def test_int6_probe_never_promotes_shell_dtype_or_uint8_to_deployment(self):
        report = probe_int6_capability(numel=128)
        native = report["native_torch"]

        self.assertEqual("estimate-only", report["status"])
        self.assertFalse(report["packed_kernel_verified"])
        self.assertFalse(report["deployment_ready"])
        self.assertIn("uint8", report["reason"])
        if native["allocated_storage_bytes"] != native["theoretical_packed_bytes"]:
            self.assertFalse(native["exact_packed_storage_observed"])


@unittest.skipUnless(HAS_TORCH, "PyTorch non installato")
class AtenInt4Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        capability = aten_int4_capability()
        if not capability["available"]:
            raise unittest.SkipTest(f"ATen W4A32 non disponibile: {capability}")

    def setUp(self):
        torch.manual_seed(7)

    def test_private_kernel_matches_explicit_dequant_and_pads_45_way_head(self):
        weight = torch.randn(45, 768, dtype=torch.float32)
        inputs = torch.randn(3, 768, dtype=torch.float32)
        packed, scales_zeros, original, dequantized = _pack_aten_int4_weight(
            weight, 128
        )

        actual = torch.ops.aten._weight_int4pack_mm_for_cpu(
            inputs, packed, 128, scales_zeros
        )[:, :45]
        expected = inputs @ dequantized[:45, :768].T

        self.assertEqual((45, 768), original)
        self.assertEqual(48, scales_zeros.shape[1])
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2e-5)

    def test_linear_wrapper_preserves_shape_and_serializes_only_packed_weight(self):
        source = torch.nn.Linear(128, 7, bias=True, dtype=torch.float32)
        converted = AtenInt4Linear.from_float(source, group_size=128)
        inputs = torch.randn(2, 3, 128)
        output = converted(inputs)

        self.assertEqual((2, 3, 7), tuple(output.shape))
        self.assertTrue(bool(torch.isfinite(output).all()))
        self.assertNotIn("weight", converted.state_dict())
        self.assertIn("packed_weight", converted.state_dict())
        self.assertLess(
            converted.storage_report()["packed_weight_bytes"],
            source.weight.numel() * source.weight.element_size(),
        )

    def test_embedding_is_packed_but_fused_kernel_coverage_is_explicitly_false(self):
        class ToyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(32, 128)
                self.classifier = torch.nn.Linear(128, 7)

            def forward(self, indices):
                return self.classifier(self.embedding(indices))

        model = ToyModel().eval()
        indices = torch.tensor([[0, 1, 31]], dtype=torch.int64)
        with torch.inference_mode():
            baseline = model(indices)
        report = quantize_model_int4_aten(
            model, group_size=128, embedding_chunk_rows=5
        )
        with torch.inference_mode():
            quantized = model(indices)

        self.assertIsInstance(model.embedding, PackedInt4Embedding)
        self.assertIsInstance(model.classifier, AtenInt4Linear)
        self.assertEqual((1, 3, 7), tuple(quantized.shape))
        self.assertTrue(report["weight_storage_coverage_complete"])
        self.assertFalse(report["fused_lowbit_kernel_coverage_complete"])
        self.assertEqual(0, report["linear"]["remaining_fp32_modules"])
        self.assertEqual(0, report["embedding"]["remaining_fp32_modules"])
        self.assertLess(float((quantized - baseline).abs().mean()), 0.25)
        with self.assertRaisesRegex(AssertionError, "Embedding packed"):
            assert_fused_int4_kernel_coverage(report)

    def test_microbenchmark_reports_real_kernel_and_does_not_assume_speedup(self):
        report = benchmark_aten_int4_shape(
            m=2, k=128, n=7, threads=1, warmup=1, repeats=2
        )

        self.assertEqual(
            "aten::_weight_int4pack_mm_for_cpu", report["aten_w4a32"]["kernel"]
        )
        self.assertEqual(2, report["fp32"]["timing"]["count"])
        self.assertEqual(2, report["aten_w4a32"]["timing"]["count"])
        self.assertGreater(report["fp32"]["timing"]["median_ms"], 0)
        self.assertGreater(report["aten_w4a32"]["timing"]["median_ms"], 0)


@unittest.skipUnless(HAS_ONNX_STACK, "onnx/numpy/onnxruntime non installati")
class OnnxAccuracyMatrixTests(unittest.TestCase):
    def test_acc1_and_acc4_share_exact_packed_initializers(self):
        report = benchmark_onnx_accuracy_matrix(
            m=2,
            k=128,
            n=16,
            block_size=128,
            threads=1,
            warmup=1,
            repeats=2,
        )

        self.assertTrue(report["same_fp32_inputs"])
        self.assertTrue(report["same_packed_int4_initializers"])
        self.assertEqual(64, len(report["packed_initializer_sha256"]))
        self.assertIsNone(report["variants"]["fp32"]["accuracy_level"])
        self.assertEqual(1, report["variants"]["int4_acc1"]["accuracy_level"])
        self.assertEqual(4, report["variants"]["int4_acc4"]["accuracy_level"])
        self.assertEqual("W4A32", report["variants"]["int4_acc1"]["compute_path"])
        self.assertIn("W4A8", report["variants"]["int4_acc4"]["compute_path"])
        for variant in report["variants"].values():
            self.assertEqual(2, variant["timing"]["count"])
            self.assertGreater(variant["timing"]["median_ms"], 0)


if __name__ == "__main__":
    unittest.main()
