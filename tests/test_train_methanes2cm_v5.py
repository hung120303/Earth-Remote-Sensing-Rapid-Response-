from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "tools", ROOT / "EarthRemoteSensingRapidResponse"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from methanes2cm_v5_model import MethaneS2CMV5Model  # noqa: E402
from analyze_methanes2cm_v5_signal import robust_scene_score  # noqa: E402
from train_methanes2cm_v5 import (  # noqa: E402
    PackedMethaneS2CMDataset,
    choose_threshold_at_fpr,
    read_manifest,
    segmentation_first_loss,
)


class MethaneS2CMV5TrainingTests(unittest.TestCase):
    def test_packed_dataset_and_loss_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            packed_path = Path(directory) / "train.h5"
            with h5py.File(packed_path, "w") as packed:
                packed.attrs["source_revision"] = (
                    "ee9a96d4994ca6bc45725c1e92d7a06258131eaf"
                )
                packed.create_dataset("sample_id", data=np.asarray([11], dtype=np.int64))
                packed.create_dataset("label", data=np.asarray([1], dtype=np.uint8))
                for name in ("target", "reference90", "reference365"):
                    values = np.full((1, 12, 32, 32), 3000, dtype=np.uint16)
                    values[:, 11, 10:14, 12:17] = 4500
                    packed.create_dataset(name, data=values)
                mask = np.zeros((1, 32, 32), dtype=np.uint8)
                mask[0, 10:14, 12:17] = 1
                packed.create_dataset("mask", data=mask)
            records = [
                {
                    "id": "11",
                    "label": "1",
                    "group_id": "geo25_test",
                    "exact_location_id": "latlon:1,2",
                }
            ]
            dataset = PackedMethaneS2CMDataset(
                packed_path, records, augment=False, seed=1
            )
            sample = dataset[0]
            self.assertEqual(sample["inputs"].shape, (20, 32, 32))
            self.assertEqual(float(sample["presence"]), 1.0)
            model = MethaneS2CMV5Model()
            batch = {
                key: value.unsqueeze(0)
                for key, value in sample.items()
                if isinstance(value, torch.Tensor)
            }
            output = model(batch["inputs"], batch["observable"])
            loss, parts = segmentation_first_loss(output, batch)
            self.assertTrue(torch.isfinite(loss))
            self.assertIn("scene_bce", parts)
            full_mask = dict(batch)
            full_mask["mask"] = torch.ones_like(batch["mask"])
            full_mask["presence"] = torch.ones_like(batch["presence"])
            full_loss, full_parts = segmentation_first_loss(output, full_mask)
            self.assertTrue(torch.isfinite(full_loss))
            self.assertGreaterEqual(full_parts["hard_negative_bce"], 0.0)

    def test_manifest_and_fpr_threshold_contracts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.jsonl"
            rows = [
                {
                    "id": "2",
                    "label": "0",
                    "research_role": "internal_development",
                    "source_revision": "ee9a96d4994ca6bc45725c1e92d7a06258131eaf",
                },
                {
                    "id": "1",
                    "label": "1",
                    "research_role": "internal_fitting",
                    "source_revision": "ee9a96d4994ca6bc45725c1e92d7a06258131eaf",
                },
            ]
            manifest.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            self.assertEqual([row["id"] for row in read_manifest(manifest)], ["1", "2"])
        selected = choose_threshold_at_fpr(
            np.asarray([1, 1, 0, 0], dtype=np.uint8),
            np.asarray([0.9, 0.7, 0.8, 0.1]),
            0.0,
        )
        self.assertEqual(selected["recall"], 0.5)
        self.assertEqual(selected["false_positive_rate"], 0.0)

    def test_robust_scene_score_uses_only_observable_top_pixels(self) -> None:
        evidence = np.zeros((1, 4, 4), dtype=np.float32)
        observable = np.ones_like(evidence, dtype=bool)
        evidence[0, 0, 0] = 3.0
        evidence[0, 0, 1] = 2.0
        observable[0, 0, 0] = False
        score = robust_scene_score(
            evidence, observable, topk_fraction=0.125, max_weight=0.0
        )
        self.assertEqual(float(score[0]), 1.0)


if __name__ == "__main__":
    unittest.main()
