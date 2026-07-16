from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from evaluate_cloudsen12_fresh_test_scene_heads import compare_stratum, score_summary


class FreshSceneHeadEvaluationTests(unittest.TestCase):
    def test_suppressed_candidate_passes_calibration_aware_stratum(self) -> None:
        current = np.asarray([0.01, 0.02, 0.30])
        candidate = np.asarray([0.004, 0.009, 0.24])
        result = compare_stratum(
            current,
            candidate,
            np.ones(3, dtype=bool),
            current_threshold=0.28,
            candidate_threshold=0.23,
        )
        self.assertTrue(all(result["checks"].values()))
        self.assertEqual(result["current"]["false_positives"], 1)
        self.assertEqual(result["candidate"]["false_positives"], 1)

    def test_score_summary_uses_frozen_threshold(self) -> None:
        result = score_summary(np.asarray([0.1, 0.2, 0.3]), 0.2)
        self.assertEqual(result["false_positives"], 2)
        self.assertAlmostEqual(result["false_positive_rate"], 2 / 3)


if __name__ == "__main__":
    unittest.main()
