from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from finalize_mars_joint_calibration_aware import calibration_aware_negative_safety


class CalibrationAwareSafetyTests(unittest.TestCase):
    def test_suppressed_scores_pass_across_different_probability_scales(self) -> None:
        current = np.asarray([0.005, 0.01, 0.02, 0.30])
        candidate = np.asarray([0.002, 0.004, 0.009, 0.25])
        result = calibration_aware_negative_safety(
            current,
            candidate,
            current_threshold=0.28,
            candidate_threshold=0.23,
        )
        self.assertTrue(result["passed"])
        self.assertEqual(result["current_false_positives"], 1)
        self.assertEqual(result["candidate_false_positives"], 1)
        self.assertLess(result["candidate_raw_score_p95"], result["current_raw_score_p95"])
        self.assertLess(
            result["candidate_logit_threshold_margin_p95"],
            result["current_logit_threshold_margin_p95"],
        )


if __name__ == "__main__":
    unittest.main()
