from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "tools", ROOT / "EarthRemoteSensingRapidResponse"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from evaluate_mars_v4_3_ensemble import (  # noqa: E402
    calibrated_ensemble,
    group_held_audit,
    select_pixel_rule,
)


class MarsV43EnsembleTests(unittest.TestCase):
    def test_calibrated_ensemble_is_invariant_to_monotone_seed_scaling(self) -> None:
        base = np.asarray(
            [[0.1, 10.0, -4.0], [0.2, 20.0, -3.0], [0.3, 30.0, -2.0], [0.4, 40.0, -1.0]]
        )
        transformed = base.copy()
        transformed[:, 1] = np.exp(transformed[:, 1] / 20.0)
        transformed[:, 2] = transformed[:, 2] * 100.0 + 7.0
        np.testing.assert_allclose(
            calibrated_ensemble(base, base),
            calibrated_ensemble(transformed, transformed),
        )

    def test_group_held_audit_scores_every_scene_without_group_overlap(self) -> None:
        groups = np.repeat(np.asarray([f"g{index:02d}" for index in range(20)]), 4)
        labels = np.tile(np.asarray([1, 0, 0, 0], dtype=np.uint8), 20)
        signal = labels.astype(np.float64) + np.linspace(0.0, 0.01, labels.size)
        scores = np.column_stack([signal, signal * 3 + 2, np.square(signal + 1)])
        result = group_held_audit(labels, groups, scores)
        self.assertGreater(result["ranking"]["average_precision"], 0.95)
        self.assertEqual(len(result["folds"]), 5)
        self.assertEqual(
            sum(fold["held_out_scenes"] for fold in result["folds"]), labels.size
        )

    def test_pixel_rule_uses_lower_threshold_to_break_dice_tie(self) -> None:
        intersections = np.asarray([5, 5, 4, 4, 4, 4, 4, 4, 4], dtype=np.float64)
        predicted = np.asarray([10, 10, 9, 9, 9, 9, 9, 9, 9], dtype=np.float64)
        result = select_pixel_rule(intersections, predicted, truth_pixels=10)
        self.assertEqual(result["selected"]["threshold"], 0.1)


if __name__ == "__main__":
    unittest.main()
