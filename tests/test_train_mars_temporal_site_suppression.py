from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from train_mars_temporal_site_suppression import temporal_site_suppression


class TemporalSiteSuppressionTests(unittest.TestCase):
    def test_rule_never_raises_a_scene_score(self) -> None:
        scores = np.asarray([0.01, 0.1, 0.8, 0.9])
        groups = np.asarray(["low", "low", "high", "high"])
        result = temporal_site_suppression(scores, groups, top_k=1, cutoff=0.5, weight=1.0)
        self.assertTrue(np.all(result <= scores + 1e-15))
        self.assertLess(result[0], scores[0])
        np.testing.assert_allclose(result[2:], scores[2:], rtol=0.0, atol=1e-15)


if __name__ == "__main__":
    unittest.main()
