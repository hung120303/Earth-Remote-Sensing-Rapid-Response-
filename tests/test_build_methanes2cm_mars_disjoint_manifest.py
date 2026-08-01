from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from build_methanes2cm_mars_disjoint_manifest import (  # noqa: E402
    nearest_site_distances_km,
)


class MethaneS2CMDisjointManifestTests(unittest.TestCase):
    def test_nearest_haversine_distance_and_index(self) -> None:
        source = np.asarray([[0.0, 0.0], [0.0, 2.0]])
        target = np.asarray([[0.0, 0.1], [0.0, 3.0]])
        distance, index = nearest_site_distances_km(source, target)
        np.testing.assert_array_equal(index, [0, 1])
        self.assertAlmostEqual(float(distance[0]), 11.1195, places=3)
        self.assertAlmostEqual(float(distance[1]), 111.1951, places=3)

    def test_rejects_nonfinite_coordinates(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-empty and finite"):
            nearest_site_distances_km(
                np.asarray([[np.nan, 0.0]]), np.asarray([[0.0, 0.0]])
            )


if __name__ == "__main__":
    unittest.main()
