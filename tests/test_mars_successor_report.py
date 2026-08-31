from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from build_mars_successor_report import build_data, render  # noqa: E402


def descriptive_fixture() -> dict:
    ap = {
        "candidate": 0.91,
        "comparator": 0.90,
        "delta": 0.01,
    }
    return {
        "decision": "PASS_DEVELOPMENT",
        "claim_language": "Development passed; official superiority remains unevaluated.",
        "forbidden_claim": "ERSRR outperforms MARS-S2L on the authors' official benchmark.",
        "scene": {
            "rows": 100,
            "positive": 20,
            "negative": 80,
            "pooled_average_precision": ap,
            "fold_average_precision": {"3": ap, "4": ap},
            "sensor_average_precision": {"Sentinel-2": ap, "Landsat": ap},
            "matched_fpr_recall_curve": [
                {
                    "fpr": value,
                    "candidate_recall": 0.8,
                    "comparator_recall": 0.75,
                    "delta": 0.05,
                }
                for value in (0.005, 0.01, 0.02, 0.05, 0.10)
            ],
            "matched_fpr_mean_delta": 0.05,
        },
        "dense": {
            "candidate": {
                "true_positive": 100,
                "false_positive": 20,
                "false_negative": 30,
                "intersection_over_union": 2 / 3,
            },
            "comparator": {
                "true_positive": 90,
                "false_positive": 25,
                "false_negative": 35,
                "intersection_over_union": 0.60,
            },
            "iou_delta": 1 / 15,
        },
        "frozen_metrics": {
            "pooled_ap_delta": 0.01,
            "matched_fpr_recall_delta": 0.05,
            "dense_iou_delta": 1 / 15,
            "checks": {
                "pooled_ap_delta_gte_0_003": True,
                "each_fold_ap_positive": True,
                "each_sensor_ap_positive": True,
                "matched_fpr_recall_nonnegative": True,
                "ap_bootstrap_lower_positive": True,
                "dense_iou_delta_positive": True,
                "dense_bootstrap_lower_positive": True,
            },
            "passed": True,
        },
        "provenance": {
            "reporting_protocol_sha256": "a" * 64,
            "compact_result_sha256": "b" * 64,
            "candidate_predictions": {
                "path": ".research/candidate.npz",
                "sha256": "c" * 64,
            },
            "endpoint_states": {
                "path": ".research/endpoints.pt",
                "sha256": "d" * 64,
            },
        },
    }


class MarsSuccessorReportTests(unittest.TestCase):
    def test_build_data_never_promotes_development_to_official_superiority(self) -> None:
        data = build_data(ROOT, descriptive=descriptive_fixture())
        self.assertTrue(data["status"]["development_passed"])
        self.assertFalse(data["status"]["official_superiority_established"])
        self.assertEqual(data["paper"]["revision"], "v3")
        self.assertAlmostEqual(
            data["paper"]["reconstructed"]["full"]["average_precision"],
            0.6410196024104218,
        )
        self.assertFalse(data["prior_official"]["test_only_sites"]["passed"])

    def test_template_is_self_contained_and_preserves_claim_boundary(self) -> None:
        template = (ROOT / "tools/templates/ersrr_mars_successor_report.html").read_text(
            encoding="utf-8"
        )
        output = render(template, build_data(ROOT, descriptive=descriptive_fixture()))
        self.assertNotIn("__ERSRR_SUCCESSOR_DATA__", output)
        self.assertNotIn('src="http', output)
        self.assertNotIn("@import", output)
        self.assertEqual(output.count("<section "), 8)
        self.assertIn("Official MARS-S2L superiority remains unestablished.", output)
        self.assertIn('"official_superiority_established":false', output)
        self.assertIn('"decision":"PASS_DEVELOPMENT"', output)
        self.assertIn("architecture schematic, not a model prediction", output.lower())

    def test_render_requires_exactly_one_placeholder(self) -> None:
        with self.assertRaises(ValueError):
            render("<html></html>", {"ok": True})


if __name__ == "__main__":
    unittest.main()
