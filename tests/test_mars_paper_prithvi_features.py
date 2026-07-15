from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import merge_mars_paper_prithvi_cls_features as merge  # noqa: E402
from extract_mars_paper_prithvi_cls_features import cls_features  # noqa: E402


class PaperPrithviFeatureTests(unittest.TestCase):
    def test_cls_features_follow_four_block_schema(self) -> None:
        outputs = [torch.zeros((2, 5, 192), dtype=torch.float32) for _ in range(12)]
        for block in (3, 6, 9, 12):
            outputs[block - 1][:, 0] = float(block)
        values = cls_features(outputs)
        self.assertEqual(values.shape, (2, 768))
        self.assertTrue(torch.all(values[:, :192] == 3.0))
        self.assertTrue(torch.all(values[:, -192:] == 12.0))

    def test_paper_merge_is_contiguous_and_label_independent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = []
            common = {
                "feature_names": np.asarray(["a", "b"]),
                "shard_count": np.asarray(5),
                "total_available_rows": np.asarray(5),
                "foundation_revision": np.asarray("revision"),
                "checkpoint_sha256": np.asarray("checkpoint"),
                "foundation_receipt_sha256": np.asarray("foundation-receipt"),
                "sealed_manifest_sha256": np.asarray("manifest"),
                "test_acquisition_receipt_sha256": np.asarray("test-receipt"),
                "input_contract": np.asarray("input"),
                "nir_transfer_contract": np.asarray("nir"),
                "missing_reference_datetime_policy": np.asarray("neutral"),
            }
            for index in range(5):
                path = root / f"shard{index}.npz"
                np.savez_compressed(
                    path,
                    features=np.asarray([[index, index + 1]], dtype=np.float16),
                    sample_ids=np.asarray([f"sample-{index}"]),
                    groups=np.asarray([f"site-{index}"]),
                    sensors=np.asarray([index % 2], dtype=np.uint8),
                    shard_index=np.asarray(index),
                    shard_start=np.asarray(index),
                    shard_end=np.asarray(index + 1),
                    missing_reference_datetime_rows=np.asarray(index == 3),
                    **common,
                )
                inputs.append(path.name)
            output = root / "merged.npz"
            receipt = root / "receipt.json"
            argv = [
                "merge_mars_paper_prithvi_cls_features.py",
                *inputs,
                "--output",
                output.name,
                "--receipt",
                receipt.name,
            ]
            with mock.patch.object(sys, "argv", argv), mock.patch.object(
                merge, "repo_root", return_value=root
            ), mock.patch.object(merge, "EXPECTED_ROWS", 5), mock.patch.object(
                merge, "EXPECTED_FEATURES", 2
            ):
                self.assertEqual(merge.main(), 0)
            with np.load(output, allow_pickle=False) as result:
                self.assertNotIn("labels", result.files)
                self.assertEqual(result["features"].shape, (5, 2))
                self.assertEqual(
                    result["sample_ids"].astype(str).tolist(),
                    [f"sample-{i}" for i in range(5)],
                )
                self.assertEqual(int(result["missing_reference_datetime_rows"].item()), 1)
            self.assertTrue(receipt.is_file())


if __name__ == "__main__":
    unittest.main()
