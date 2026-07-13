from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "tools", ROOT / "EarthRemoteSensingRapidResponse"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from train_mars_v4 import (  # noqa: E402
    choose_threshold_at_fpr,
    development_decision,
    hard_negative_segmentation_loss,
    masked_weighted_segmentation_loss,
)


class TrainMarsV4Tests(unittest.TestCase):
    def test_hard_negative_loss_focuses_false_alarm_pixels(self) -> None:
        target = torch.zeros(1, 1, 10, 10)
        target[:, :, 4:6, 4:6] = 1
        observable = torch.ones_like(target)
        quiet = torch.full_like(target, -4.0, requires_grad=True)
        noisy = quiet.detach().clone()
        noisy[:, :, 0, 0] = 8.0
        noisy.requires_grad_(True)
        quiet_loss, _ = hard_negative_segmentation_loss(quiet, target, observable)
        noisy_loss, _ = hard_negative_segmentation_loss(noisy, target, observable)
        self.assertGreater(float(noisy_loss.detach()), float(quiet_loss.detach()))
        noisy_loss.backward()
        self.assertTrue(torch.isfinite(noisy.grad).all())

    def test_threshold_selection_respects_no_plume_budget(self) -> None:
        labels = np.asarray([1, 1, 0, 0, 0, 0], dtype=np.uint8)
        scores = np.asarray([0.9, 0.7, 0.8, 0.6, 0.2, 0.1])
        result = choose_threshold_at_fpr(labels, scores, 0.25)
        self.assertEqual(result["threshold"], 0.7)
        self.assertEqual(result["recall"], 1.0)
        self.assertLessEqual(result["false_positive_rate"], 0.25)

    def test_weighted_segmentation_loss_rewards_detected_plume(self) -> None:
        target = torch.zeros(1, 1, 8, 8)
        target[:, :, 3:5, 3:5] = 1
        observable = torch.ones_like(target)
        missed = torch.full_like(target, -5.0)
        detected = missed.clone()
        detected[target > 0.5] = 5.0
        missed_loss, _ = masked_weighted_segmentation_loss(missed, target, observable)
        detected_loss, _ = masked_weighted_segmentation_loss(detected, target, observable)
        self.assertLess(float(detected_loss.detach()), float(missed_loss.detach()))

    def test_development_decision_rejects_internal_regression(self) -> None:
        validation = {
            "average_precision": 0.2,
            "auroc": 0.6,
            "positive_pixel_dice": 0.1,
            "operating_points": {"0.05": {"recall": 0.1}},
        }
        reference = {
            "mean": {
                "average_precision": 0.8,
                "auroc": 0.9,
                "positive_pixel_dice": 0.5,
                "recall_at_fpr5": 0.8,
            }
        }
        checks, decision = development_decision(validation, reference)
        self.assertFalse(any(checks.values()))
        self.assertTrue(decision.startswith("Reject"))


if __name__ == "__main__":
    unittest.main()
