from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from train_mars_target_weighted_scene_head import density_ratio_weights  # noqa: E402


class TargetWeightedSceneHeadTests(unittest.TestCase):
    def test_density_weights_are_positive_normalized_and_target_directed(self) -> None:
        rng = np.random.default_rng(7)
        source = np.concatenate((rng.normal(-2, 0.2, (40, 2)), rng.normal(2, 0.2, (40, 2))))
        target = rng.normal(2, 0.2, (40, 2))
        weights, audit = density_ratio_weights(
            source, target, gamma=1.0, clip_lower=0.1, clip_upper=10.0
        )
        self.assertTrue(np.isfinite(weights).all())
        self.assertTrue(np.all(weights > 0.0))
        self.assertAlmostEqual(float(weights.mean()), 1.0)
        self.assertGreater(float(weights[40:].mean()), float(weights[:40].mean()))
        self.assertGreater(audit["domain_auc"], 0.7)


if __name__ == "__main__":
    unittest.main()
