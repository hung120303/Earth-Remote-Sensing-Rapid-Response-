from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from analyze_mars_v3_strict_posthoc import (  # noqa: E402
    rank_strata_indices,
    rate_summary,
    zero_nonzero_strata,
)


class MarsV3StrictPosthocTests(unittest.TestCase):
    def test_rank_strata_are_complete_disjoint_and_deterministic(self) -> None:
        values = [4.0, 1.0, 3.0, 2.0, 2.0, 9.0]
        first = rank_strata_indices(values, ("low", "middle", "high"))
        second = rank_strata_indices(values, ("low", "middle", "high"))
        np.testing.assert_array_equal(np.concatenate(first), np.concatenate(second))
        self.assertEqual(sorted(np.concatenate(first).tolist()), list(range(6)))
        self.assertEqual([len(item) for item in first], [2, 2, 2])

    def test_rank_strata_reject_nonfinite_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite"):
            rank_strata_indices([1.0, np.nan, 2.0], ("a", "b", "c"))

    def test_rate_summary_averages_fixed_seed_decisions(self) -> None:
        baseline = np.asarray([1, 0, 1], dtype=np.uint8)
        candidates = np.asarray([[1, 0, 0], [0, 0, 1]], dtype=np.uint8)
        result = rate_summary(np.asarray([0, 2]), baseline, candidates)
        self.assertEqual(result["samples"], 2)
        self.assertAlmostEqual(result["released_mars_s2l_rate"], 1.0)
        self.assertAlmostEqual(result["ersrr_seed_mean_rate"], 0.5)

    def test_zero_nonzero_strata_do_not_split_tied_zeros(self) -> None:
        rows = [{"cloud": 0.0}, {"cloud": 0.0}, {"cloud": 0.2}]
        baseline = np.asarray([0, 1, 1], dtype=np.uint8)
        candidates = np.asarray([[0, 0, 1], [0, 1, 1]], dtype=np.uint8)
        result = zero_nonzero_strata(
            rows,
            np.arange(3),
            "cloud",
            baseline,
            candidates,
        )
        self.assertEqual([item["name"] for item in result], ["zero", "nonzero"])
        self.assertEqual([item["samples"] for item in result], [2, 1])


if __name__ == "__main__":
    unittest.main()
