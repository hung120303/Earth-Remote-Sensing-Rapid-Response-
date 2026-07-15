from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import merge_mars_prithvi_scene_features as merge  # noqa: E402


class PrithviShardMergeTests(unittest.TestCase):
    def test_five_verified_fold_shards_merge_once_each(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs: list[str] = []
            common = {
                "feature_names": np.asarray(["a", "b"]),
                "foundation_revision": np.asarray("revision"),
                "checkpoint_sha256": np.asarray("checkpoint"),
                "foundation_receipt_sha256": np.asarray("receipt"),
                "manifest_sha256": np.asarray("manifest"),
                "protocol_sha256": np.asarray("protocol"),
                "input_contract": np.asarray("input"),
                "nir_transfer_contract": np.asarray("nir-transfer"),
                "missing_reference_datetime_policy": np.asarray("neutral"),
            }
            for fold in range(5):
                path = root / f"fold{fold}.npz"
                np.savez_compressed(
                    path,
                    features=np.asarray([[fold, fold + 1]], dtype=np.float16),
                    labels=np.asarray([fold % 2], dtype=np.uint8),
                    sensors=np.asarray([fold % 2], dtype=np.uint8),
                    sample_ids=np.asarray([f"sample-{fold}"]),
                    groups=np.asarray([f"site-{fold}"]),
                    folds=np.asarray([fold], dtype=np.uint8),
                    missing_reference_datetime_rows=np.asarray(fold == 2),
                    **common,
                )
                inputs.append(path.name)
            output = root / "merged.npz"
            receipt = root / "receipt.json"
            argv = [
                "merge_mars_prithvi_scene_features.py",
                *inputs,
                "--expected-rows",
                "5",
                "--output",
                output.name,
                "--receipt",
                receipt.name,
            ]
            with mock.patch.object(sys, "argv", argv), mock.patch.object(
                merge, "repo_root", return_value=root
            ):
                self.assertEqual(merge.main(), 0)
            with np.load(output, allow_pickle=False) as result:
                self.assertEqual(result["features"].shape, (5, 2))
                self.assertEqual(
                    result["sample_ids"].astype(str).tolist(),
                    [f"sample-{i}" for i in range(5)],
                )
                self.assertEqual(result["folds"].tolist(), list(range(5)))
                self.assertEqual(int(result["missing_reference_datetime_rows"].item()), 1)
                self.assertEqual(result["source_shard_sha256"].shape, (5,))
            self.assertTrue(receipt.is_file())


if __name__ == "__main__":
    unittest.main()
