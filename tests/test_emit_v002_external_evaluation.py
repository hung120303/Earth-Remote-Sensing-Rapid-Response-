from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from evaluate_emit_v002_external import (
    FIXED_SEEDS,
    SEED_REPORT_PATHS,
    connected_mask,
    paired_bootstrap,
    positive_metrics,
    tracked_dirty,
)


class ExternalEvaluationContractTests(unittest.TestCase):
    @patch("evaluate_emit_v002_external.subprocess.check_output", return_value="")
    def test_dirty_check_uses_repository_line_endings(self, check_output) -> None:
        self.assertFalse(tracked_dirty(ROOT))
        command = check_output.call_args.args[0]
        self.assertIn("core.autocrlf=true", command)

    def test_every_fixed_seed_has_one_explicit_report_pair(self) -> None:
        self.assertEqual(set(SEED_REPORT_PATHS), set(FIXED_SEEDS))
        self.assertEqual(
            SEED_REPORT_PATHS[303],
            (
                "reports/experiments/mars_v3_validation.json",
                "reports/experiments/mars_v3_proposal_validation.json",
            ),
        )

    def test_connected_mask_applies_observability_and_minimum_area(self) -> None:
        probability = np.zeros((8, 8), dtype=np.float32)
        probability[1:4, 1:4] = 0.9
        probability[6, 6] = 0.95
        observable = np.ones((8, 8), dtype=bool)
        observable[1, 1] = False
        result = connected_mask(probability, observable, 0.5, 4)
        self.assertEqual(int(np.count_nonzero(result)), 8)
        self.assertFalse(result[6, 6])

    def test_positive_metrics_reports_recall_without_fake_negative_metrics(self) -> None:
        result = positive_metrics(np.asarray([True, False, True]), 3)
        self.assertEqual(result["true_positive"], 2)
        self.assertEqual(result["false_negative"], 1)
        self.assertAlmostEqual(result["recall"], 2 / 3)

    def test_paired_bootstrap_is_deterministic_and_paired(self) -> None:
        candidate = [np.asarray([1, 1, 0, 1], dtype=bool) for _ in range(5)]
        baseline = np.asarray([1, 0, 0, 1], dtype=bool)
        first = paired_bootstrap(candidate, baseline, 100)
        second = paired_bootstrap(candidate, baseline, 100)
        self.assertEqual(first, second)
        self.assertGreater(first["recall_delta"]["mean"], 0.0)


if __name__ == "__main__":
    unittest.main()
