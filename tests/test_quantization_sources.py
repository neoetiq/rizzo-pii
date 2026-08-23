"""Integrity checks for the pinned local model and validation sources."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from src.quantization.sources import SourceValidationError, verify_sources


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class QuantizationSourceTests(unittest.TestCase):
    def test_verify_sources_detects_checkpoint_tampering(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifacts = root / "artifacts"
            model_dir = artifacts / "model"
            data_dir = artifacts / "data"
            model_dir.mkdir(parents=True)
            data_dir.mkdir(parents=True)
            config_file = model_dir / "config.json"
            weights_file = model_dir / "model.safetensors"
            config_file.write_text("{}\n", encoding="utf-8")
            weights_file.write_bytes(b"weights")

            dataset_rows = {}
            for purpose, filename in (("validation", "validation.jsonl"), ("calibration", "calibration.jsonl")):
                path = data_dir / filename
                path.write_text('{"tokens":["x"],"bio_labels":["O"]}\n', encoding="utf-8")
                dataset_rows[purpose] = {
                    "path": path.relative_to(artifacts).as_posix(),
                    "rows": 1,
                    "tokens": 1,
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }

            model_manifest = [
                {
                    "path": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
                for path in sorted(model_dir.iterdir())
            ]
            revision_model = "a" * 40
            revision_dataset = "b" * 40
            lock = {
                "schema_version": 1,
                "model": {
                    "revision_resolved": revision_model,
                    "files": model_manifest,
                },
                "dataset": {
                    "revision_resolved": revision_dataset,
                    "files": dataset_rows,
                },
            }
            (artifacts / "sources.lock.json").write_text(
                json.dumps(lock), encoding="utf-8"
            )
            config = {
                "schema_version": 1,
                "output_dir": str(artifacts),
                "model": {"revision": revision_model},
                "dataset": {
                    "revision": revision_dataset,
                    "validation": {"expected_rows": 1},
                    "calibration": {"expected_rows": 1},
                },
                "validation": {},
            }
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")

            result = verify_sources(config_path, artifacts)
            self.assertEqual("verified", result["status"])

            weights_file.write_bytes(b"tampered")
            with self.assertRaisesRegex(SourceValidationError, "manifest bloccato"):
                verify_sources(config_path, artifacts)


if __name__ == "__main__":
    unittest.main()
