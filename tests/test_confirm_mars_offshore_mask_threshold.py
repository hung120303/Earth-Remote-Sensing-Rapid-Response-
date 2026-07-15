from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from confirm_mars_offshore_mask_threshold import route_offshore_counts  # noqa: E402


class OffshoreMaskThresholdTests(unittest.TestCase):
    def test_only_offshore_rows_change(self) -> None:
        baseline = np.asarray([[1, 2, 3], [4, 5, 6]])
        offshore_counts = np.asarray([[7, 8, 9], [10, 11, 12]])
        offshore = np.asarray([False, True])
        actual = route_offshore_counts(baseline, offshore_counts, offshore)
        np.testing.assert_array_equal(actual, [[1, 2, 3], [10, 11, 12]])


if __name__ == "__main__":
    unittest.main()
