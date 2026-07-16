from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from train_mars_unep_positive_augmented_xgboost import original_checks


class UnepAugmentedXgboostTests(unittest.TestCase):
    def result(self, ap: float, recall: float, sensor: float) -> dict:
        return {
            "versus_current": {
                "delta": {
                    "average_precision": ap,
                    "recall_at_fpr_0_0713": recall,
                    "sensor_average_precision": {
                        "Sentinel-2": sensor,
                        "Landsat": sensor,
                    },
                }
            },
            "paired_group_bootstrap_ap_delta_vs_current": {"lower": 0.001},
        }

    def test_selection_allows_zero_recall_delta(self) -> None:
        checks = original_checks(self.result(0.002, 0.0, -0.002), confirmation=False)
        self.assertTrue(all(checks.values()))

    def test_confirmation_requires_strict_recall_and_positive_interval(self) -> None:
        result = self.result(0.003, 0.0, 0.0)
        checks = original_checks(result, confirmation=True)
        self.assertFalse(checks["recall_gate"])
        result["versus_current"]["delta"]["recall_at_fpr_0_0713"] = 0.001
        result["paired_group_bootstrap_ap_delta_vs_current"]["lower"] = 0.0
        checks = original_checks(result, confirmation=True)
        self.assertFalse(checks["paired_ap_lower_positive"])


if __name__ == "__main__":
    unittest.main()
