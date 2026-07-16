from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from evaluate_unep_mars_positive_baseline import aggregate, retained_mask


class UnepPositiveBaselineTests(unittest.TestCase):
    def test_retained_mask_removes_small_components(self) -> None:
        probability = np.zeros((20, 20), dtype=np.float32)
        probability[0:2, 0:2] = 0.9
        probability[5:15, 5:15] = 0.9
        result = retained_mask(probability, 0.8, 100)
        self.assertEqual(int(np.count_nonzero(result)), 100)
        self.assertFalse(bool(result[0, 0]))

    def test_aggregate_uses_global_pixel_counts(self) -> None:
        rows = [
            {"endpoint": {"detected": True, "intersection": 2, "predicted": 3, "truth": 4}},
            {"endpoint": {"detected": False, "intersection": 0, "predicted": 0, "truth": 2}},
        ]
        result = aggregate(rows, "endpoint")
        self.assertEqual(result["positive_recall"], 0.5)
        self.assertEqual(result["pixel_iou"], 2 / 7)


if __name__ == "__main__":
    unittest.main()
