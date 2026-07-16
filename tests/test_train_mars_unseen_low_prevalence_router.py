from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from train_mars_unseen_low_prevalence_router import low_prevalence_mask


class UnseenLowPrevalenceRouterTests(unittest.TestCase):
    def test_mask_selects_whole_sites_by_label_rate(self) -> None:
        labels = np.asarray([0, 0, 1, 0, 0, 0], dtype=np.uint8)
        groups = np.asarray(["low", "low", "high", "high", "zero", "zero"])
        result = low_prevalence_mask(labels, groups, 0.1)
        self.assertEqual(result.tolist(), [True, True, False, False, True, True])


if __name__ == "__main__":
    unittest.main()
