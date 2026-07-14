from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from train_mars_context_scene_ranker import augment_site_context, context_feature_names, leave_one_out_max  # noqa: E402


class ContextSceneRankerTests(unittest.TestCase):
    def test_leave_one_out_max_does_not_leak_unique_self_maximum(self) -> None:
        values = np.asarray([[1.0, 3.0], [2.0, 2.0], [3.0, 1.0]])
        result = leave_one_out_max(values)
        np.testing.assert_array_equal(result[2], np.asarray([2.0, 3.0]))
        np.testing.assert_array_equal(result[0], np.asarray([3.0, 2.0]))

    def test_context_is_computed_within_group(self) -> None:
        names = np.asarray([
            "primary_connected_score", "released_connected_score", "primary_top_100_mean",
            "primary_top_500_mean", "primary_area_above_0.3", "primary_area_above_0.5",
            "input_0_mean", "input_0_top_100_mean", "logit_delta_valid_mean", "clear_fraction",
        ])
        values = np.vstack((np.zeros(10), np.ones(10), np.full(10, 100.0)))
        augmented, output_names = augment_site_context(values, names, np.asarray(["a", "a", "b"]))
        self.assertEqual(augmented.shape[1], 10 + len(context_feature_names()))
        self.assertEqual(len(output_names), augmented.shape[1])
        self.assertLess(float(augmented[0, 10]), 2.0)
        self.assertEqual(float(augmented[2, 10]), 100.0)


if __name__ == "__main__":
    unittest.main()
