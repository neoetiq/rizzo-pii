"""Synthetic regression tests for standard-ONNX INT8 embedding conversion."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import src.quantization.embedding as embedding_quantization
from src.quantization.embedding import quantize_embedding_int8
from src.quantization.quantize import QuantizationError


HAS_ONNX_STACK = all(
    importlib.util.find_spec(package) is not None
    for package in ("numpy", "onnx", "onnxruntime")
)


@unittest.skipUnless(HAS_ONNX_STACK, "onnx/numpy/onnxruntime non installati")
class EmbeddingQuantizationTests(unittest.TestCase):
    def _write_model(self, path: Path):
        import numpy as np
        import onnx
        from onnx import TensorProto, helper, numpy_helper

        # Row zero also exercises the zero-row scale path.
        embedding = np.asarray(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.11, -0.73, 1.20, 0.41],
                [-1.87, 0.32, 0.71, -0.08],
                [2.13, -1.14, 0.02, 0.93],
                [-0.39, 0.61, -0.27, 0.08],
            ],
            dtype=np.float32,
        )
        ids = helper.make_tensor_value_info(
            "input_ids", TensorProto.INT64, ["batch", "sequence"]
        )
        output = helper.make_tensor_value_info(
            "embeddings", TensorProto.FLOAT, ["batch", "sequence", 4]
        )
        gather = helper.make_node(
            "Gather",
            ["word_embeddings", "input_ids"],
            ["embeddings"],
            name="WordEmbeddingGather",
            axis=0,
        )
        graph = helper.make_graph(
            [gather],
            "synthetic_embedding",
            [ids],
            [output],
            [numpy_helper.from_array(embedding, name="word_embeddings")],
        )
        model = helper.make_model(
            graph,
            opset_imports=[helper.make_opsetid("", 18)],
            producer_name="test",
        )
        # Keep the fixture compatible with a wider range of ORT releases.
        model.ir_version = min(model.ir_version, 10)
        onnx.save_model(
            model,
            str(path),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=path.name + ".data",
            size_threshold=0,
        )
        return embedding

    def test_row_quantization_is_identical_across_multiple_chunks(self):
        import numpy as np

        values = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [0.1, -0.2, 0.7],
                [-1.4, 0.3, 0.2],
                [2.1, -0.4, 0.9],
                [-0.3, 0.8, -0.6],
            ],
            dtype=np.float32,
        )
        with patch.object(embedding_quantization, "EMBEDDING_ROW_CHUNK_SIZE", 2):
            chunked_q, chunked_scales, chunked_stats = (
                embedding_quantization._quantize_rows(values)
            )
        with patch.object(embedding_quantization, "EMBEDDING_ROW_CHUNK_SIZE", 100):
            single_q, single_scales, single_stats = (
                embedding_quantization._quantize_rows(values)
            )

        np.testing.assert_array_equal(chunked_q, single_q)
        np.testing.assert_array_equal(chunked_scales, single_scales)
        self.assertEqual(chunked_stats, single_stats)

    def test_quantizes_external_embedding_and_preserves_numerical_inference(self):
        import numpy as np
        import onnx
        import onnxruntime as ort
        from onnx import numpy_helper

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.onnx"
            destination = root / "embedding-int8.onnx"
            unrelated = root / "embedding-int8.onnx.keep"
            embedding = self._write_model(source)
            unrelated.write_text("preserve", encoding="utf-8")

            report = quantize_embedding_int8(
                source,
                destination,
                validation_shapes=((1, 3), (2, 2)),
            )

            ids = np.asarray([[0, 1, 2], [3, 4, 1]], dtype=np.int64)
            original_session = ort.InferenceSession(
                str(source), providers=["CPUExecutionProvider"]
            )
            quantized_session = ort.InferenceSession(
                str(destination), providers=["CPUExecutionProvider"]
            )
            expected = original_session.run(None, {"input_ids": ids})[0]
            actual = quantized_session.run(None, {"input_ids": ids})[0]

            max_scale = float(np.max(np.max(np.abs(embedding), axis=1) / 127.0))
            np.testing.assert_allclose(actual, expected, rtol=0.0, atol=max_scale / 2 + 1e-7)
            np.testing.assert_array_equal(actual[0, 0], np.zeros(4, dtype=np.float32))

            converted = onnx.load_model(str(destination), load_external_data=True)
            tensors = {tensor.name: tensor for tensor in converted.graph.initializer}
            self.assertNotIn("word_embeddings", tensors)
            self.assertEqual(2, len(tensors))
            int8_tensor = next(
                tensor for tensor in tensors.values()
                if tensor.data_type == onnx.TensorProto.INT8
            )
            scale_tensor = next(
                tensor for tensor in tensors.values()
                if tensor.data_type == onnx.TensorProto.FLOAT
            )
            self.assertEqual([5, 4], list(int8_tensor.dims))
            self.assertEqual([5, 1], list(scale_tensor.dims))
            self.assertEqual(np.int8, numpy_helper.to_array(int8_tensor).dtype)
            self.assertEqual(1.0, float(numpy_helper.to_array(scale_tensor)[0, 0]))

            op_counts = report["structural_validation"]["op_counts"]
            self.assertEqual(2, op_counts["Gather"])
            self.assertEqual(1, op_counts["Cast"])
            self.assertEqual(1, op_counts["Mul"])
            self.assertEqual(1.0, report["coverage"]["coverage_ratio"])
            self.assertEqual(80, report["embedding_storage"]["original_weight_bytes"])
            self.assertEqual(40, report["embedding_storage"]["quantized_weight_bytes"])
            self.assertEqual(0.5, report["embedding_storage"]["size_ratio"])
            self.assertGreater(report["artifact"]["bytes"], 0)
            self.assertTrue(destination.with_name(destination.name + ".data").is_file())
            self.assertEqual("preserve", unrelated.read_text(encoding="utf-8"))

    def test_in_place_conversion_is_rejected_before_cleanup(self):
        with TemporaryDirectory() as temporary:
            source = Path(temporary) / "source.onnx"
            self._write_model(source)
            before = source.read_bytes()
            external = source.with_name(source.name + ".data")
            external_before = external.read_bytes()

            with self.assertRaisesRegex(QuantizationError, "file distinti"):
                quantize_embedding_int8(source, source)

            self.assertEqual(before, source.read_bytes())
            self.assertEqual(external_before, external.read_bytes())


if __name__ == "__main__":
    unittest.main()
