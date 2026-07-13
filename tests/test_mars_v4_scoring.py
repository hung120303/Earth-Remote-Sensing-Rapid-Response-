from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "tools", ROOT / "EarthRemoteSensingRapidResponse"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from analyze_mars_v4_scoring import empirical_percentile, nested_score_selection  # noqa: E402
from mars_v4_scoring import CANDIDATE_FORMULAS, scene_score_candidates  # noqa: E402


class MarsV4ScoringTests(unittest.TestCase):
    def test_candidate_schema_is_finite_and_ignores_unobservable_hot_pixel(self) -> None:
        logits = np.full((40, 40), -4.0, dtype=np.float32)
        observable = np.ones_like(logits, dtype=bool)
        baseline = scene_score_candidates(logits, observable)
        logits[0, 0] = 30.0
        observable[0, 0] = False
        result = scene_score_candidates(logits, observable)
        self.assertEqual(tuple(result), tuple(CANDIDATE_FORMULAS))
        self.assertTrue(np.all(np.isfinite(list(result.values()))))
        self.assertAlmostEqual(
            result["current_top_0_5pct_blend"],
            baseline["current_top_0_5pct_blend"],
            places=6,
        )

    def test_component_excess_rewards_connected_support(self) -> None:
        isolated = np.full((20, 20), -8.0, dtype=np.float32)
        connected = isolated.copy()
        locations = (
            (1, 1),
            (1, 5),
            (1, 9),
            (5, 1),
            (5, 5),
            (5, 9),
            (9, 1),
            (9, 5),
            (9, 9),
        )
        for row, col in locations:
            isolated[row, col] = 3.0
        connected[5:8, 5:8] = 3.0
        valid = np.ones_like(isolated, dtype=bool)
        isolated_score = scene_score_candidates(isolated, valid)["component_excess_p80"]
        connected_score = scene_score_candidates(connected, valid)["component_excess_p80"]
        self.assertGreater(connected_score, isolated_score * 8)

    def test_empirical_percentile_is_monotone(self) -> None:
        result = empirical_percentile(
            np.asarray([1.0, 2.0, 3.0]), np.asarray([0.0, 2.0, 4.0])
        )
        np.testing.assert_allclose(result, np.asarray([0.0, 2 / 3, 1.0]))

    def test_nested_selection_keeps_groups_held_out(self) -> None:
        groups = np.repeat(np.asarray([f"g{index:02d}" for index in range(20)]), 4)
        labels = np.tile(np.asarray([1, 0, 0, 0], dtype=np.uint8), 20)
        useful = labels.astype(np.float64) + np.linspace(0, 0.01, labels.size)
        noise = np.tile(np.asarray([0.1, 0.9, 0.2, 0.3]), 20)
        result = nested_score_selection(
            labels,
            groups,
            ["useful", "noise"],
            np.column_stack([useful, noise]),
        )
        self.assertGreater(result["ranking"]["average_precision"], 0.95)
        self.assertEqual(len(result["folds"]), 5)
        self.assertTrue(all(fold["held_out_groups"] == 4 for fold in result["folds"]))


if __name__ == "__main__":
    unittest.main()
