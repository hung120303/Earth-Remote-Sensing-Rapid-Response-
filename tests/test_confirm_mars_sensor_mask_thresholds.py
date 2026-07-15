from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from confirm_mars_sensor_mask_thresholds import route_sensor_counts  # noqa: E402


class SensorMaskThresholdTests(unittest.TestCase):
    def test_only_sentinel_rows_change(self) -> None:
        baseline = np.asarray([[1, 2, 3], [4, 5, 6]])
        sentinel = np.asarray([[7, 8, 9], [10, 11, 12]])
        sensors = np.asarray([0, 1])
        actual = route_sensor_counts(baseline, sentinel, sensors)
        np.testing.assert_array_equal(actual, [[7, 8, 9], [4, 5, 6]])


if __name__ == "__main__":
    unittest.main()
