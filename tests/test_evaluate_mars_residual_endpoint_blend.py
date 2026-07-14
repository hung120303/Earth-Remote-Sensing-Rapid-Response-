from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from evaluate_mars_residual_endpoint_blend import (  # noqa: E402
    assert_endpoint_identity,
    select_beta,
)


def summary(rank: tuple[float, float, float], passes: bool) -> dict:
    return {
        "delta": {
            "average_precision": rank[1] - rank[0],
            "pixel_iou": rank[0],
            "recall_at_fpr_0_0713": rank[2],
        },
        "promotion_checks": {"gate": passes},
    }


class EndpointBlendTests(unittest.TestCase):
    def test_selection_prefers_any_fully_passing_interior_candidate(self) -> None:
        values = {
            "0": summary((0.0, 0.0, 0.0), False),
            "0.25": summary((0.1, 0.2, 0.0), False),
            "0.5": summary((0.01, 0.02, 0.01), True),
            "1": summary((-0.1, -0.2, -0.1), False),
        }
        key, _ = select_beta(values)
        self.assertEqual(key, "0.5")

    def test_selection_rejects_endpoint_only_grid(self) -> None:
        with self.assertRaises(ValueError):
            select_beta({"0": summary((0, 0, 0), False), "1": summary((0, 0, 0), False)})

    def test_endpoint_identity_is_exact(self) -> None:
        endpoint = {
            "candidate": {
                "average_precision": 0.4,
                "operating_points": {"0.0713": {"recall": 0.7}},
                "pixel_fixed_0_5": {"intersection_over_union": 0.3},
            },
            "sensor_strata": {
                name: {
                    "candidate": {
                        "average_precision": 0.5,
                        "pixel_fixed_0_5": {"intersection_over_union": 0.2},
                    }
                }
                for name in ("Landsat", "Sentinel-2")
            },
        }
        assert_endpoint_identity(endpoint, endpoint, "same")
        changed = {**endpoint, "candidate": {**endpoint["candidate"], "average_precision": 0.40001}}
        with self.assertRaises(RuntimeError):
            assert_endpoint_identity(changed, endpoint, "changed")


if __name__ == "__main__":
    unittest.main()
