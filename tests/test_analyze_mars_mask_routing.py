from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from analyze_mars_mask_routing import (  # noqa: E402
    paired_group_bootstrap,
    pixel_counts,
    select_candidate,
)


class MarsMaskRoutingTests(unittest.TestCase):
    def test_pixel_counts_are_mutually_exclusive(self) -> None:
        prediction = np.asarray([[True, True], [False, False]])
        truth = np.asarray([[True, False], [True, False]])
        observable = np.ones((2, 2), dtype=bool)
        np.testing.assert_array_equal(pixel_counts(prediction, truth, observable), [1, 1, 1])

    def test_group_bootstrap_is_paired_and_deterministic(self) -> None:
        baseline = np.asarray([[5, 5, 0], [4, 6, 0], [6, 4, 0], [5, 5, 0]])
        candidate = np.asarray([[7, 3, 0], [6, 4, 0], [8, 2, 0], [7, 3, 0]])
        groups = np.asarray(["a", "a", "b", "c"])
        first = paired_group_bootstrap(
            baseline, candidate, groups, replicates=200, seed=17, confidence=0.95
        )
        second = paired_group_bootstrap(
            baseline, candidate, groups, replicates=200, seed=17, confidence=0.95
        )
        self.assertEqual(first, second)
        self.assertGreater(first["lower"], 0.0)

    def test_selection_prefers_passing_confidence_and_domains(self) -> None:
        def candidate(lower: float, delta: float) -> dict:
            return {
                "delta": delta,
                "folds": {"2": {"delta": delta}, "3": {"delta": delta}},
                "sensors": {"Sentinel-2": {"delta": delta}, "Landsat": {"delta": delta}},
                "paired_group_bootstrap_delta": {"lower": lower},
            }

        key, _ = select_candidate({"0.6": candidate(-0.01, 0.2), "0.7": candidate(0.01, 0.1)})
        self.assertEqual(key, "0.7")


if __name__ == "__main__":
    unittest.main()
