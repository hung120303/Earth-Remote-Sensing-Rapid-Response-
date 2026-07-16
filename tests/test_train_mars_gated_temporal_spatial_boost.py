from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from train_mars_gated_temporal_spatial_boost import one_sided_site_boost


class GatedTemporalSpatialBoostTests(unittest.TestCase):
    def test_boost_never_lowers_and_sparse_site_is_exact(self) -> None:
        spatial = np.asarray([0.8, 0.2, 0.1])
        groups = np.asarray(["large", "large", "small"])
        result = one_sided_site_boost(
            spatial, groups, min_site_size=2, top_k=1,
            confidence_cutoff=0.5, weight=0.5,
        )
        self.assertTrue(np.all(result >= spatial - 1e-14))
        self.assertEqual(result[2], spatial[2])
        self.assertGreater(result[0], spatial[0])


if __name__ == "__main__":
    unittest.main()
