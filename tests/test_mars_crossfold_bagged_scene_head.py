from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from train_mars_crossfold_bagged_scene_head import (  # noqa: E402
    aggregate_predictions,
    nested_training_folds,
)


class CrossfoldBaggedSceneHeadTests(unittest.TestCase):
    def test_nested_plans_never_include_holdout(self) -> None:
        plans = nested_training_folds(2)
        self.assertEqual(len(plans), 4)
        self.assertTrue(all(2 not in plan for plan in plans))
        self.assertTrue(all(len(plan) == 3 for plan in plans))
        self.assertEqual(set().union(*(set(plan) for plan in plans)), {0, 1, 3, 4})

    def test_aggregations_are_finite_and_bounded(self) -> None:
        predictions = np.asarray([[0.1, 0.8], [0.2, 0.7], [0.3, 0.9]])
        for mode in ("mean_probability", "mean_logit", "median_probability"):
            values = aggregate_predictions(predictions, mode)
            self.assertEqual(values.shape, (2,))
            self.assertTrue(np.all((0.0 <= values) & (values <= 1.0)))
            self.assertTrue(np.isfinite(values).all())


if __name__ == "__main__":
    unittest.main()
