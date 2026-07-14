from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from evaluate_mars_residual_trust_region import (  # noqa: E402
    assert_matches_artifact_baseline,
    balanced_rank,
    promotion_checks,
    select_alpha,
)


def summary(ap: float, iou: float, recall: float) -> dict[str, object]:
    return {
        "delta": {
            "average_precision": ap,
            "pixel_iou": iou,
            "recall_at_fpr_0_0713": recall,
        },
        "sensor_strata": {
            name: {
                "eligible_for_promotion": True,
                "delta": {"average_precision": ap, "pixel_iou": iou},
            }
            for name in ("Sentinel-2", "Landsat")
        },
    }


class MarsResidualTrustRegionTests(unittest.TestCase):
    def test_rank_balances_ap_and_iou_before_recall(self) -> None:
        self.assertGreater(
            balanced_rank(summary(0.02, 0.01, 0.0)),
            balanced_rank(summary(0.20, -0.01, 0.5)),
        )

    def test_zero_alpha_is_never_selected(self) -> None:
        key, _ = select_alpha(
            {"0": summary(0.0, 0.0, 0.0), "0.25": summary(0.01, 0.01, 0.01)}
        )
        self.assertEqual(key, "0.25")

    def test_promotion_requires_all_point_and_sensor_gates(self) -> None:
        self.assertTrue(all(promotion_checks(summary(0.01, 0.01, 0.01)).values()))
        failed = summary(0.01, -0.001, 0.01)
        self.assertFalse(promotion_checks(failed)["pixel_iou_higher"])

    def test_artifact_baseline_assertion_detects_metric_drift(self) -> None:
        sensor_metric = {
            "average_precision": 0.8,
            "pixel_fixed_0_5": {"intersection_over_union": 0.4},
        }
        actual = {
            "candidate": {
                **sensor_metric,
                "operating_points": {"0.0713": {"recall": 0.9}},
            },
            "sensor_strata": {
                name: {"candidate": sensor_metric.copy()}
                for name in ("Sentinel-2", "Landsat")
            },
        }
        expected = {
            "released_baseline": actual["candidate"],
            "sensor_strata": {
                name: {"released_baseline": sensor_metric.copy()}
                for name in ("Sentinel-2", "Landsat")
            },
        }
        assert_matches_artifact_baseline(actual, expected)
        expected["released_baseline"] = {
            **actual["candidate"],
            "average_precision": 0.79,
        }
        with self.assertRaisesRegex(RuntimeError, "does not match"):
            assert_matches_artifact_baseline(actual, expected)


if __name__ == "__main__":
    unittest.main()
