from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from train_mars_scene_ranker import (  # noqa: E402
    blend_scores,
    select_candidate,
    site_cell_weights,
)


class SceneRankerTests(unittest.TestCase):
    def test_site_cell_weights_equalize_cells(self) -> None:
        weights = site_cell_weights(
            np.asarray(["a", "a", "a", "b"]),
            np.asarray([0, 0, 1, 0]),
            np.asarray([0, 0, 0, 1]),
        )
        self.assertAlmostEqual(float(weights[:2].sum()), float(weights[2]))
        self.assertAlmostEqual(float(weights[2]), float(weights[3]))

    def test_blend_has_exact_endpoints(self) -> None:
        primary = np.asarray([0.2, 0.8])
        head = np.asarray([0.7, 0.3])
        np.testing.assert_array_equal(blend_scores(primary, head, 0.0), primary)
        np.testing.assert_array_equal(blend_scores(primary, head, 1.0), head)

    def test_selection_prefers_a_passing_candidate(self) -> None:
        failing = {"checks": {"a": False}, "rank": [1.0, 1.0, 1.0]}
        passing = {"checks": {"a": True}, "rank": [0.1, 0.1, 0.1]}
        self.assertIs(select_candidate([failing, passing]), passing)


if __name__ == "__main__":
    unittest.main()
