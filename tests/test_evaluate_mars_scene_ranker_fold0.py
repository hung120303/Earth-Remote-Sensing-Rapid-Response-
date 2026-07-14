from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from evaluate_mars_scene_ranker_fold0 import (  # noqa: E402
    assert_primary_identity,
    primary_identity_values,
)


class SceneRankerFold0Tests(unittest.TestCase):
    def test_primary_identity_is_exact(self) -> None:
        metrics = {
            "average_precision": 0.8,
            "operating_point": {"recall": 0.7, "false_positive_rate": 0.06},
            "sensor_average_precision": {"Sentinel-2": 0.75, "Landsat": 0.9},
        }
        summary = {
            "candidate": {
                "average_precision": 0.8,
                "operating_points": {"0.0713": {"recall": 0.7, "false_positive_rate": 0.06}},
            },
            "sensor_strata": {
                "Sentinel-2": {"candidate": {"average_precision": 0.75}},
                "Landsat": {"candidate": {"average_precision": 0.9}},
            },
        }
        assert_primary_identity(metrics, summary)
        metrics["average_precision"] += 1e-12
        with self.assertRaises(RuntimeError):
            assert_primary_identity(metrics, summary)


if __name__ == "__main__":
    unittest.main()
