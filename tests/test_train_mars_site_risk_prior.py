from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from train_mars_site_risk_prior import build_site_table, site_prior_scores


class SiteRiskPriorTests(unittest.TestCase):
    def test_site_table_has_one_row_per_group_and_any_positive_label(self) -> None:
        table = build_site_table(
            np.asarray([0.1, 0.8, 0.2]),
            np.asarray([0.2, 0.7, 0.1]),
            np.asarray([0, 1, 0]),
            np.asarray(["a", "a", "b"]),
            labels=np.asarray([0, 1, 0]),
            folds=np.asarray([2, 2, 3]),
        )
        self.assertEqual(table["features"].shape, (2, 32))
        self.assertEqual(table["labels"].tolist(), [1, 0])
        self.assertEqual(table["folds"].tolist(), [2, 3])

    def test_higher_site_risk_increases_same_scene_score(self) -> None:
        current = np.asarray([0.1, 0.1])
        risk = np.asarray([0.2, 0.8])
        result = site_prior_scores(current, risk, np.asarray([0, 1]), 0.5)
        self.assertLess(result[0], result[1])


if __name__ == "__main__":
    unittest.main()
