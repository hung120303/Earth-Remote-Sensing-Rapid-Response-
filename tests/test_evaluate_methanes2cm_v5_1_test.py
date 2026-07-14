from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "tools", ROOT / "EarthRemoteSensingRapidResponse"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from evaluate_methanes2cm_v5_1_test import (  # noqa: E402
    build_v4_batch,
    model_metrics,
)
from acquire_methanes2cm_v5_test import verified_freeze  # noqa: E402
from analyze_methanes2cm_v5_1_test_posthoc import frozen_operating_points  # noqa: E402


class MethaneS2CMV51LocationTestEvaluatorTests(unittest.TestCase):
    def test_pretest_report_identities_match_frozen_files(self) -> None:
        ensemble = verified_freeze(ROOT)
        self.assertTrue(ensemble["freeze"]["location_test_still_sealed"])

    def test_v4_compatibility_batch_uses_frozen_channel_contract(self) -> None:
        values = torch.arange(2 * 20 * 4 * 4, dtype=torch.float32).reshape(2, 20, 4, 4)
        converted = build_v4_batch(values)
        self.assertEqual(tuple(converted.shape), (2, 16, 4, 4))
        torch.testing.assert_close(converted[:, 0:1], values[:, 0:1])
        torch.testing.assert_close(converted[:, 1:13], values[:, 2:14])
        torch.testing.assert_close(converted[:, 13:15], torch.full((2, 2, 4, 4), 0.5))
        torch.testing.assert_close(converted[:, 15:16], torch.zeros((2, 1, 4, 4)))

    def test_metrics_count_no_plume_errors_and_dense_overlap(self) -> None:
        labels = np.asarray([1, 0], dtype=np.uint8)
        scores = np.asarray([0.9, 0.1], dtype=np.float32)
        decisions = np.asarray([True, False])
        truth = np.zeros((2, 2, 2), dtype=bool)
        truth[0, 0, 0] = True
        observable = np.ones_like(truth)
        probability = np.full((2, 2, 2), 0.1, dtype=np.float32)
        probability[0, 0, 0] = 0.9
        pixel_decision = probability >= 0.5
        metrics, per_scene = model_metrics(
            labels,
            scores,
            decisions,
            probability,
            pixel_decision,
            truth,
            observable,
        )
        self.assertEqual(metrics["scene"]["recall"], 1.0)
        self.assertEqual(metrics["scene"]["false_positive_rate"], 0.0)
        self.assertEqual(metrics["pixel"]["dice"], 1.0)
        self.assertEqual(metrics["pixel"]["intersection_over_union"], 1.0)
        np.testing.assert_array_equal(per_scene["pixel_truth"], [1, 0])

    def test_posthoc_uses_only_already_frozen_thresholds(self) -> None:
        result = frozen_operating_points(
            np.asarray([1, 1, 0, 0], dtype=np.uint8),
            np.asarray([0.9, 0.6, 0.4, 0.1]),
            np.asarray([0.05]),
            np.asarray([0.5]),
            {
                "0.05": {
                    "recall": 0.75,
                    "false_positive_rate": 0.05,
                    "precision": 0.8,
                }
            },
        )
        self.assertEqual(result["0.05"]["threshold_frozen_on_development"], 0.5)
        self.assertEqual(result["0.05"]["location_test"]["recall"], 1.0)
        self.assertEqual(result["0.05"]["location_test"]["false_positive_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
