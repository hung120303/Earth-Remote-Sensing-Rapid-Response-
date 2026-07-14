from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from train_mars_paper_residual import (  # noqa: E402
    sampling_weights,
    smoke_subset,
    successor_loss,
    verify_acquisition_receipt,
)


class MarsPaperResidualTrainingTests(unittest.TestCase):
    def test_sampling_equalizes_label_sensor_strata(self) -> None:
        records = [
            {"label_state": "PLUME", "sensor_family": "Sentinel-2"},
            {"label_state": "PLUME", "sensor_family": "Sentinel-2"},
            {"label_state": "NO_PLUME", "sensor_family": "Landsat"},
        ]
        weights = sampling_weights(records)
        self.assertEqual(weights.tolist(), [0.5, 0.5, 1.0])

    def test_smoke_subset_keeps_both_labels(self) -> None:
        records = [
            {"label_state": "PLUME", "sensor_family": "Sentinel-2"},
            {"label_state": "NO_PLUME", "sensor_family": "Sentinel-2"},
        ]
        self.assertEqual(smoke_subset(records, 1), records)

    def test_negative_upward_change_is_penalized(self) -> None:
        logits = torch.ones(1, 1, 4, 4, requires_grad=True)
        output = {
            "segmentation_logits": logits,
            "baseline_logits": torch.zeros_like(logits),
            "correction_logits": logits,
            "scene_logit": torch.zeros(1, requires_grad=True),
        }
        batch = {
            "mask": torch.zeros_like(logits),
            "observable": torch.ones_like(logits),
            "presence": torch.zeros(1),
        }
        loss, parts = successor_loss(output, batch)
        self.assertGreater(parts["negative_upward_penalty"], 0.0)
        loss.backward()
        self.assertIsNotNone(logits.grad)

    def test_receipt_must_match_development_manifest(self) -> None:
        import json
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "receipt.json"
            path.write_text(
                json.dumps(
                    {
                        "result": {
                            "ok": True,
                            "manifest_filter": {"sha256": "expected"},
                        }
                    }
                ),
                encoding="utf-8",
            )
            verify_acquisition_receipt(path, "expected")
            with self.assertRaises(ValueError):
                verify_acquisition_receipt(path, "different")


if __name__ == "__main__":
    unittest.main()
