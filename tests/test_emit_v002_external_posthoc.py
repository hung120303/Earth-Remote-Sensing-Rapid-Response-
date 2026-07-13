from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from analyze_emit_v002_external_posthoc import (  # noqa: E402
    FROZEN_CONFIRMATION_SCOPE,
    binary_auc,
    distribution_comparison,
    mask_signal_features,
    rank_strata_indices,
    time_offset_strata,
)


class EmitV002ExternalPosthocTests(unittest.TestCase):
    def test_frozen_confirmation_scope_matches_committed_receipt(self) -> None:
        payload = (ROOT / "reports/experiments/emit_v002_external_confirmation.json").read_text(
            encoding="utf-8"
        )
        self.assertIn(f'"scope": "{FROZEN_CONFIRMATION_SCOPE}"', payload)

    def test_binary_auc_is_tie_aware(self) -> None:
        labels = np.asarray([1, 1, 0, 0])
        self.assertAlmostEqual(binary_auc(labels, np.asarray([2.0, 1.0, 1.0, 0.0])), 0.875)

    def test_mask_signal_features_are_direction_free(self) -> None:
        plume = np.asarray([[1, 1], [0, 0]], dtype=bool)
        observable = np.ones((2, 2), dtype=bool)
        high = mask_signal_features(
            np.asarray([[3.0, 2.0], [1.0, 0.0]]), observable, plume
        )
        low = mask_signal_features(
            np.asarray([[0.0, 1.0], [2.0, 3.0]]), observable, plume
        )
        self.assertEqual(high["mbmp_mask_direction_free_auc"], 1.0)
        self.assertEqual(low["mbmp_mask_direction_free_auc"], 1.0)
        self.assertGreater(high["mbmp_mask_signed_median_contrast"], 0.0)
        self.assertLess(low["mbmp_mask_signed_median_contrast"], 0.0)

    def test_rank_strata_are_complete_and_deterministic(self) -> None:
        first = rank_strata_indices([4.0, 1.0, 3.0, 2.0, 2.0, 9.0], ("a", "b", "c"))
        second = rank_strata_indices([4.0, 1.0, 3.0, 2.0, 2.0, 9.0], ("a", "b", "c"))
        np.testing.assert_array_equal(np.concatenate(first), np.concatenate(second))
        self.assertEqual(sorted(np.concatenate(first).tolist()), list(range(6)))

    def test_time_bins_keep_boundaries_predeclared(self) -> None:
        rows = []
        for value in (0.5, 1.0, 1.5, 2.0, 3.0):
            rows.append(
                {
                    "absolute_time_offset_hours": value,
                    "released_prediction": 0,
                    "seed_predictions": [0, 0, 0, 0, 0],
                }
            )
        result = time_offset_strata(rows)
        self.assertEqual([item["samples"] for item in result], [2, 2, 1])

    def test_distribution_comparison_reports_effect_direction(self) -> None:
        external = [{"x": 3.0}, {"x": 4.0}]
        strict = [{"x": 1.0}, {"x": 2.0}]
        result = distribution_comparison(external, strict, "x")
        self.assertEqual(result["cliffs_delta_external_vs_mars"], 1.0)
        self.assertEqual(result["external_minus_mars_median"], 2.0)


if __name__ == "__main__":
    unittest.main()
