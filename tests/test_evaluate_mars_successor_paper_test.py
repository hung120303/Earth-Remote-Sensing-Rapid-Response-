from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from evaluate_mars_successor_paper_test import (
    average_precision_from_cumulative,
    plan_cumulative,
    score_plan,
)


class MarsPaperSuccessorTests(unittest.TestCase):
    def test_weighted_ap_matches_sklearn_with_ties(self) -> None:
        labels = np.asarray([1, 0, 1, 0, 1], dtype=np.uint8)
        scores = np.asarray([0.9, 0.8, 0.8, 0.2, 0.1])
        sites = np.asarray([0, 0, 1, 2, 2], dtype=np.int64)
        draws = np.asarray([[1, 2, 0], [0, 1, 2]], dtype=np.int32)
        plan = score_plan(labels, scores, sites)
        tp, fp, total = plan_cumulative(draws, plan)
        observed = average_precision_from_cumulative(tp, fp, total)
        expected = [
            average_precision_score(labels, scores, sample_weight=draw[sites])
            for draw in draws
        ]
        np.testing.assert_allclose(observed, expected, rtol=0, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
