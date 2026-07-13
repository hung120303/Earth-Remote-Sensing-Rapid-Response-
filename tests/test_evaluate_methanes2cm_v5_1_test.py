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


class MethaneS2CMV51LocationTestEvaluatorTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
