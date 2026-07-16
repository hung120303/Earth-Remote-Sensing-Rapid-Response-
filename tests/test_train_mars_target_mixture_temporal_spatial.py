from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from train_mars_target_mixture_temporal_spatial import build_cohort_plans


class TargetMixtureTests(unittest.TestCase):
    def test_plan_has_frozen_site_mixture_and_scene_cap(self) -> None:
        groups = np.repeat(np.asarray([f"s{i}" for i in range(8)]), 3)
        folds = np.zeros(groups.size, dtype=np.uint8)
        labels = np.zeros(groups.size, dtype=np.uint8)
        labels[0] = 1
        labels[3] = 1
        plans = build_cohort_plans(
            labels, groups, folds, [0], replicates=3, sites_per_fold=4,
            positive_sites_per_fold=1, maximum_scenes_per_site=2, seed=7,
        )
        for plan in plans:
            rows = plan[0]
            self.assertEqual(rows.size, 8)
            selected_groups = np.unique(groups[rows])
            self.assertEqual(selected_groups.size, 4)
            positive_groups = sum(bool(np.any(labels[rows][groups[rows] == group])) for group in selected_groups)
            self.assertEqual(positive_groups, 1)


if __name__ == "__main__":
    unittest.main()
