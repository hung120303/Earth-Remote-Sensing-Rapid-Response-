from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from train_mars_oof_scene_ensemble_v2 import (  # noqa: E402
    ap_group_bootstrap,
    sample_weights,
)


class OofSceneEnsembleV2Tests(unittest.TestCase):
    def test_group_weights_equalize_group_mass(self) -> None:
        groups = np.asarray(["a", "a", "b"])
        labels = np.asarray([0, 1, 0])
        sensors = np.asarray([0, 0, 1])
        weights = sample_weights("group", groups, labels, sensors)
        self.assertAlmostEqual(float(weights[:2].sum()), float(weights[2]))

    def test_ap_group_bootstrap_detects_strictly_better_ranking(self) -> None:
        labels = np.asarray([1, 0, 1, 0, 1, 0], dtype=np.uint8)
        baseline = np.asarray([0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
        candidate = np.asarray([0.9, 0.2, 0.8, 0.1, 0.7, 0.0])
        groups = np.asarray(["a", "a", "b", "b", "c", "c"])
        result = ap_group_bootstrap(
            labels, baseline, candidate, groups, replicates=200, seed=7
        )
        self.assertGreater(result["mean"], 0.0)
        self.assertGreaterEqual(result["lower"], 0.0)


if __name__ == "__main__":
    unittest.main()
