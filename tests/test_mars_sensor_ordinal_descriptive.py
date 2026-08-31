from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from audit_mars_sensor_ordinal_descriptive import (  # noqa: E402
    absolute_dense_metrics,
    absolute_scene_metrics,
    assert_nested_close,
    validate_access_ledger,
)


class MarsSensorOrdinalDescriptiveTests(unittest.TestCase):
    def test_absolute_scene_metrics_preserve_strata_and_curve(self) -> None:
        labels = np.asarray([1, 0, 1, 0, 1, 0, 1, 0], dtype=np.uint8)
        candidate = np.asarray([0.95, 0.05, 0.85, 0.15, 0.90, 0.10, 0.80, 0.20])
        comparator = np.asarray([0.80, 0.20, 0.70, 0.30, 0.75, 0.25, 0.65, 0.35])
        folds = np.asarray([3, 3, 4, 4, 3, 3, 4, 4], dtype=np.uint8)
        sensors = np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.uint8)
        result = absolute_scene_metrics(
            labels, candidate, comparator, folds, sensors, [0.25, 0.50]
        )
        self.assertEqual(result["rows"], 8)
        self.assertEqual(set(result["fold_average_precision"]), {"3", "4"})
        self.assertEqual(
            set(result["sensor_average_precision"]), {"Sentinel-2", "Landsat"}
        )
        self.assertEqual(len(result["matched_fpr_recall_curve"]), 2)
        self.assertGreaterEqual(result["pooled_average_precision"]["delta"], 0.0)

    def test_absolute_dense_metrics_reaggregate_counts(self) -> None:
        candidate = np.asarray([[5, 1, 2], [3, 2, 1]], dtype=np.int64)
        comparator = np.asarray([[4, 2, 3], [2, 3, 2]], dtype=np.int64)
        result = absolute_dense_metrics(candidate, comparator)
        self.assertEqual(result["candidate"]["true_positive"], 8)
        self.assertAlmostEqual(result["candidate"]["intersection_over_union"], 8 / 14)
        self.assertGreater(result["iou_delta"], 0.0)

    def test_nested_comparison_rejects_a_numeric_change(self) -> None:
        assert_nested_close(
            {"metric": 0.5, "checks": [True, False]},
            {"metric": 0.5 + 1e-13, "checks": [True, False]},
            tolerance=1e-12,
        )
        with self.assertRaises(AssertionError):
            assert_nested_close(
                {"metric": 0.5}, {"metric": 0.5001}, tolerance=1e-12
            )

    def test_access_ledger_requires_exact_safe_boundary(self) -> None:
        ledger = {
            "comparator_integrity_bytes_hashed": True,
            "comparator_values_decoded": True,
            "held_folds_opened": [3, 4],
            "folds_0_1_2_opened": False,
            "external_or_official_evidence_opened": False,
        }
        validate_access_ledger(ledger)
        ledger["external_or_official_evidence_opened"] = True
        with self.assertRaises(RuntimeError):
            validate_access_ledger(ledger)


if __name__ == "__main__":
    unittest.main()
