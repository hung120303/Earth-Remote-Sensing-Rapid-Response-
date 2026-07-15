from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from train_mars_crossfold_hard_positive_bagging import (  # noqa: E402
    hard_positive_weights,
)


class CrossfoldHardPositiveBaggingTests(unittest.TestCase):
    def test_only_missed_training_positives_are_upweighted(self) -> None:
        labels = np.asarray([0] * 20 + [1, 1], dtype=np.uint8)
        scores = np.concatenate((np.linspace(1.0, 0.0, 20), [0.99, 0.05]))
        weights, threshold, count = hard_positive_weights(labels, scores, 4.0)
        self.assertGreater(threshold, 0.05)
        self.assertEqual(count, 1)
        np.testing.assert_array_equal(weights, [1] * 21 + [4])


if __name__ == "__main__":
    unittest.main()
