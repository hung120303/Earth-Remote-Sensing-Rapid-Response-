from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "tools", ROOT / "EarthRemoteSensingRapidResponse"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from aggregate_methanes2cm_v5_1 import (  # noqa: E402
    binary_metrics,
    empirical_percentile,
    stable_group_folds,
)


class MethaneS2CMV51AggregationTests(unittest.TestCase):
    def test_empirical_percentile_is_right_continuous_and_monotonic(self) -> None:
        calibrated = empirical_percentile(
            np.asarray([0.1, 0.2, 0.2, 0.8]),
            np.asarray([0.0, 0.2, 0.5, 1.0]),
        )
        np.testing.assert_allclose(calibrated, [0.0, 0.75, 0.75, 1.0])

    def test_group_folds_never_split_a_group(self) -> None:
        groups = np.asarray(["a", "a", "b", "c", "b"])
        folds = stable_group_folds(groups)
        self.assertEqual(int(folds[0]), int(folds[1]))
        self.assertEqual(int(folds[2]), int(folds[4]))

    def test_binary_metrics_include_no_plume_error_rate(self) -> None:
        result = binary_metrics(
            np.asarray([1, 1, 0, 0]), np.asarray([1, 0, 1, 0])
        )
        self.assertEqual(result["recall"], 0.5)
        self.assertEqual(result["false_positive_rate"], 0.5)
        self.assertEqual(result["precision"], 0.5)


if __name__ == "__main__":
    unittest.main()
