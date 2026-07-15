from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from evaluate_mars_successor_paper_test import (  # noqa: E402
    average_precision_from_cumulative,
    candidate_pixel_counts,
    diagnostic_pixel_counts,
    mask_threshold_for_sensor,
    plan_cumulative,
    predict_scene_head,
    score_plan,
)


class MarsPaperSuccessorTests(unittest.TestCase):
    def test_mask_threshold_supports_scalar_and_sensor_specific_configs(self) -> None:
        self.assertEqual(mask_threshold_for_sensor({"mask_probability_threshold": 0.7}, 0), 0.7)
        architecture = {
            "mask_probability_threshold_by_sensor": {"Sentinel-2": 0.8, "Landsat": 0.7}
        }
        self.assertEqual(mask_threshold_for_sensor(architecture, 0), 0.8)
        self.assertEqual(mask_threshold_for_sensor(architecture, 1), 0.7)

    def test_scene_head_supports_direct_estimators(self) -> None:
        class DirectEstimator:
            def predict_proba(self, features: np.ndarray) -> np.ndarray:
                return np.column_stack((1.0 - features[:, 0], features[:, 0]))

        features = np.asarray([[0.2], [0.8]])
        np.testing.assert_allclose(predict_scene_head(DirectEstimator(), features), [0.2, 0.8])


    def test_candidate_pixel_counts_are_mutually_exclusive(self) -> None:
        observable = np.asarray([[True, True], [True, False]])
        truth = np.asarray([[True, False], [False, False]])
        prediction = np.asarray([[True, True], [False, True]])
        result = candidate_pixel_counts(
            prediction, truth, observable, truth_available=True
        )
        self.assertEqual(result, {"truth_available": True, "tp": 1, "fp": 1, "fn": 0})

    def test_missing_truth_keeps_all_observable_predictions_adversarial(self) -> None:
        observable = np.asarray([[True, True], [True, False]])
        prediction = np.asarray([[True, True], [False, True]])
        result = candidate_pixel_counts(
            prediction,
            np.zeros_like(prediction),
            observable,
            truth_available=False,
        )
        self.assertEqual(result, {"truth_available": False, "tp": 0, "fp": 2, "fn": 0})

    def test_diagnostic_counts_apply_missing_truth_policy(self) -> None:
        raw = np.asarray([[[3, 2, 1], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
        result = diagnostic_pixel_counts(
            raw,
            np.asarray([True, False]),
            np.asarray([4, 20]),
        )
        np.testing.assert_array_equal(result[:, 0], raw[:, 0])
        np.testing.assert_array_equal(result[:, 1], [[0, 5, 20], [0, 11, 20]])

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
