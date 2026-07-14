from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from train_mars_hard_scene_ranker import (  # noqa: E402
    hard_example_masks,
    robust_checks,
    select_robust_candidate,
)


class HardSceneRankerTests(unittest.TestCase):
    def test_hard_masks_respect_labels(self) -> None:
        labels = np.asarray([1, 1, 0, 0, 0, 0], dtype=np.uint8)
        scores = np.asarray([0.1, 0.9, 0.8, 0.7, 0.2, 0.1])
        positive, negative, _ = hard_example_masks(labels, scores)
        self.assertFalse(np.any(positive & (labels == 0)))
        self.assertFalse(np.any(negative & (labels == 1)))

    def test_robust_gate_requires_three_tp_margin(self) -> None:
        candidate = {"checks": {"base": True}, "delta": {"recall_at_fpr_0_0713": 2 / 100}}
        self.assertFalse(robust_checks(candidate, 100)["minimum_three_tp_recall_margin"])
        candidate["delta"]["recall_at_fpr_0_0713"] = 3 / 100
        self.assertTrue(robust_checks(candidate, 100)["minimum_three_tp_recall_margin"])

    def test_selection_prefers_robust_pass(self) -> None:
        failing = {"robust_checks": {"a": False}, "rank": [1, 1, 1]}
        passing = {"robust_checks": {"a": True}, "rank": [0, 0, 0]}
        self.assertIs(select_robust_candidate([failing, passing]), passing)


if __name__ == "__main__":
    unittest.main()
