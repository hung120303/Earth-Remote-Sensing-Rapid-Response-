from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "tools", ROOT / "EarthRemoteSensingRapidResponse"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from evaluate_mars_v4_3_strict import (  # noqa: E402
    align_released_cache,
    paired_group_bootstrap,
)


class MarsV43StrictTests(unittest.TestCase):
    def test_released_cache_alignment_follows_candidate_order(self) -> None:
        cache = {
            "sample_ids": np.asarray(["b", "a"]),
            "groups": np.asarray(["g2", "g1"]),
            "labels": np.asarray([0, 1], dtype=np.uint8),
            "scores": np.asarray([0.2, 0.9]),
            "predictions": np.asarray([0, 1], dtype=np.uint8),
        }
        aligned = align_released_cache(cache, ["a", "b"])
        self.assertEqual(list(aligned["sample_ids"]), ["a", "b"])
        self.assertEqual(list(aligned["labels"]), [1, 0])

    def test_paired_bootstrap_reports_candidate_advantage(self) -> None:
        groups = np.repeat(np.asarray([f"g{index:02d}" for index in range(20)]), 4)
        labels = np.tile(np.asarray([1, 0, 0, 0], dtype=np.uint8), 20)
        candidate_scores = labels.astype(float) * 0.8 + 0.1
        baseline_scores = np.tile(np.asarray([0.4, 0.5, 0.3, 0.2]), 20)
        result = paired_group_bootstrap(
            labels,
            groups,
            candidate_scores,
            candidate_scores >= 0.5,
            baseline_scores,
            baseline_scores >= 0.45,
        )
        self.assertGreater(result["recall_delta"]["mean"], 0.0)
        self.assertLess(result["false_positive_rate_delta"]["mean"], 0.0)
        self.assertGreater(result["average_precision_delta"]["mean"], 0.0)


if __name__ == "__main__":
    unittest.main()
