from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from train_mars_v4_cascade import (  # noqa: E402
    choose_threshold_at_fpr,
    largest_component_pixels,
    metrics,
    physics_features,
)


class MarsV4CascadeTests(unittest.TestCase):
    def test_largest_component_uses_eight_connectivity(self) -> None:
        mask = np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=bool)
        self.assertEqual(largest_component_pixels(mask), 3)

    def test_physics_features_are_label_blind_and_finite(self) -> None:
        mbmp = np.asarray([[0.94, 0.97], [1.0, 1.02]], dtype=np.float32)
        target = np.ones((6, 2, 2), dtype=np.float32)
        reference = np.full((6, 2, 2), 0.9, dtype=np.float32)
        names, values = physics_features(
            mbmp,
            target,
            reference,
            np.ones((2, 2), dtype=bool),
            cloud_fraction=0.0,
            wind_speed_m_s=3.0,
            reference_interval_days_value=5.0,
        )
        self.assertEqual(len(names), len(values))
        self.assertTrue(np.all(np.isfinite(values)))
        self.assertFalse(any("label" in name or "plume_mask" in name for name in names))
        self.assertEqual(values[names.index("mbmp_largest_component_le_95")], 1.0)

    def test_threshold_selection_respects_fpr_and_maximizes_recall(self) -> None:
        labels = np.asarray([1, 1, 0, 0, 0, 0], dtype=np.uint8)
        scores = np.asarray([0.9, 0.7, 0.8, 0.6, 0.2, 0.1])
        result = choose_threshold_at_fpr(labels, scores, 0.25)
        self.assertLessEqual(result["training_fpr"], 0.25)
        self.assertEqual(result["training_recall"], 1.0)
        self.assertEqual(result["threshold"], 0.7)

    def test_metrics_count_no_plume_errors_explicitly(self) -> None:
        labels = np.asarray([1, 1, 0, 0], dtype=np.uint8)
        scores = np.asarray([0.9, 0.1, 0.8, 0.2])
        result = metrics(labels, scores, scores >= 0.5)
        self.assertEqual(result["tp"], 1)
        self.assertEqual(result["fp"], 1)
        self.assertEqual(result["recall"], 0.5)
        self.assertEqual(result["false_positive_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
