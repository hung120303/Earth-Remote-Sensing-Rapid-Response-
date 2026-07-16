from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from train_mars_joint_external_augmented_xgboost import joint_training_arrays


class JointExternalAugmentationTests(unittest.TestCase):
    def test_joint_arrays_preserve_labels_and_declared_weights(self) -> None:
        original = {
            "features": np.arange(12, dtype=np.float64).reshape(4, 3),
            "labels": np.asarray([0, 1, 0, 1], dtype=np.uint8),
        }
        positive = {
            "features": np.ones((2, 3), dtype=np.float64),
            "labels": np.ones(2, dtype=np.uint8),
        }
        negative = {
            "features": np.zeros((3, 3), dtype=np.float64),
            "labels": np.zeros(3, dtype=np.uint8),
        }
        features, labels, weights = joint_training_arrays(
            original,
            positive,
            negative,
            np.asarray([True, False, True, False]),
            positive_multiplier=2.5,
            negative_multiplier=0.75,
        )
        self.assertEqual(features.shape, (7, 3))
        np.testing.assert_array_equal(labels, [0, 0, 1, 1, 0, 0, 0])
        np.testing.assert_allclose(weights, [1, 1, 2.5, 2.5, 0.75, 0.75, 0.75])


if __name__ == "__main__":
    unittest.main()
