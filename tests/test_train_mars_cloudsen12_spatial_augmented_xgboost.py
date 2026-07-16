from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from train_mars_cloudsen12_spatial_augmented_xgboost import negative_confirmation


class CloudsenSpatialAugmentedTests(unittest.TestCase):
    def test_negative_confirmation_requires_count_and_margin_noninferiority(self) -> None:
        current = np.asarray([0.1, 0.2, 0.7])
        better = np.asarray([0.05, 0.1, 0.6])
        result = negative_confirmation(current, better, 0.65, 0.65)
        self.assertTrue(result["passed"])
        worse = negative_confirmation(current, np.asarray([0.7, 0.8, 0.9]), 0.65, 0.65)
        self.assertFalse(worse["passed"])


if __name__ == "__main__":
    unittest.main()
