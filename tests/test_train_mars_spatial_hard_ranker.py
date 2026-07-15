from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from train_mars_spatial_hard_ranker import hard_negative_probabilities  # noqa: E402


class MarsSpatialHardRankerTests(unittest.TestCase):
    def test_higher_scores_receive_more_hard_negative_mass(self) -> None:
        probabilities = hard_negative_probabilities(np.asarray([0.1, 0.5, 0.9]), 1.0)
        self.assertGreater(probabilities[2], probabilities[1])
        self.assertGreater(probabilities[1], probabilities[0])
        self.assertAlmostEqual(float(probabilities.sum()), 1.0)

    def test_zero_hard_fraction_is_uniform(self) -> None:
        probabilities = hard_negative_probabilities(np.asarray([0.1, 0.9]), 0.0)
        np.testing.assert_allclose(probabilities, [0.5, 0.5])


if __name__ == "__main__":
    unittest.main()
