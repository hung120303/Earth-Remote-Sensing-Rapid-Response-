from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from train_mars_temporal_site_prior import temporal_site_prior


class TemporalSitePriorTests(unittest.TestCase):
    def test_top_k_prior_is_group_local_and_label_free(self) -> None:
        scores = np.asarray([0.1, 0.8, 0.2, 0.3])
        groups = np.asarray(["a", "a", "b", "b"])
        result = temporal_site_prior(scores, groups, top_k=1, weight=0.5)
        self.assertGreater(result[0], 0.1)
        self.assertGreater(result[1], 0.8)
        self.assertLess(result[2], 0.2)
        self.assertLess(result[3], 0.3)

    def test_singleton_groups_preserve_score_order(self) -> None:
        scores = np.asarray([0.01, 0.2, 0.8])
        groups = np.asarray(["a", "b", "c"])
        result = temporal_site_prior(scores, groups, top_k=3, weight=0.5)
        self.assertEqual(np.argsort(result).tolist(), np.argsort(scores).tolist())

    def test_sites_below_minimum_history_are_unchanged(self) -> None:
        scores = np.asarray([0.1, 0.8, 0.2])
        groups = np.asarray(["a", "a", "b"])
        result = temporal_site_prior(scores, groups, top_k=1, weight=0.5, min_site_size=3)
        np.testing.assert_allclose(result, scores, rtol=0.0, atol=1e-15)


if __name__ == "__main__":
    unittest.main()
