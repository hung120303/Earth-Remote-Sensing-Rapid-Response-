from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from analyze_mars_mask_thresholds import choose_threshold, component_mask_at


class MarsMaskThresholdTests(unittest.TestCase):
    def test_component_rule_removes_small_regions(self) -> None:
        score = np.zeros((8, 8), dtype=np.float32)
        score[1:3, 1:3] = 0.9
        score[4:7, 4:7] = 0.8
        result = component_mask_at(score, 0.5, 5)
        self.assertEqual(int(result.sum()), 9)
        self.assertFalse(bool(result[1, 1]))
        self.assertTrue(bool(result[5, 5]))

    def test_selection_prioritizes_worst_fold_then_sensor(self) -> None:
        summaries = {
            "0.5": {"delta": 0.0, "folds": {"2": {"delta": 0.0}}, "sensors": {"S": {"delta": 0.0}}},
            "0.6": {"delta": 0.1, "folds": {"2": {"delta": 0.01}}, "sensors": {"S": {"delta": 0.02}}},
            "0.7": {"delta": 0.2, "folds": {"2": {"delta": -0.01}}, "sensors": {"S": {"delta": 0.03}}},
        }
        selected, _ = choose_threshold(summaries, "0.5")
        self.assertEqual(selected, "0.6")


if __name__ == "__main__":
    unittest.main()
