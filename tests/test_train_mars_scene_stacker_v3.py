from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from train_mars_scene_stacker_v3 import stack_features  # noqa: E402


class MarsSceneStackerV3Tests(unittest.TestCase):
    def test_sensor_domain_features_are_aligned_and_finite(self) -> None:
        primary = np.asarray([0.2, 0.3])
        legacy = np.asarray([0.4, 0.5])
        new = np.asarray([0.6, 0.7])
        sensors = np.asarray([0, 1])
        offshore = np.asarray([False, True])
        features, names = stack_features(
            primary, legacy, new, sensors, offshore, "sensor_domain"
        )
        self.assertEqual(features.shape, (2, 12))
        self.assertEqual(len(names), 12)
        self.assertTrue(np.isfinite(features).all())
        self.assertEqual(features[0, names.index("is_landsat")], 0.0)
        self.assertEqual(features[1, names.index("is_landsat")], 1.0)
        self.assertEqual(features[0, names.index("is_offshore")], 0.0)
        self.assertEqual(features[1, names.index("is_offshore")], 1.0)

    def test_unknown_feature_set_is_rejected(self) -> None:
        values = np.asarray([0.5])
        with self.assertRaises(ValueError):
            stack_features(values, values, values, np.asarray([0]), np.asarray([False]), "bad")


if __name__ == "__main__":
    unittest.main()
