from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from confirm_mars_scene_domain_routing import route_scores  # noqa: E402


class SceneDomainRoutingTests(unittest.TestCase):
    def test_offshore_shift_reduces_scores_and_sensor_weights_route(self) -> None:
        legacy = np.asarray([0.2, 0.2, 0.2])
        new = np.asarray([0.8, 0.8, 0.8])
        sensors = np.asarray([0, 1, 0])
        offshore = np.asarray([False, False, True])
        routed = route_scores(
            legacy,
            new,
            sensors,
            offshore,
            sentinel_new_weight=0.0,
            landsat_new_weight=1.0,
            offshore_logit_shift=-4.0,
        )
        self.assertAlmostEqual(routed[0], 0.2)
        self.assertAlmostEqual(routed[1], 0.8)
        self.assertLess(routed[2], 0.2)


if __name__ == "__main__":
    unittest.main()
