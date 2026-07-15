from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from train_mars_xgboost_scene_head import MODEL_SPECS, build_model  # noqa: E402


class XGBoostSceneHeadTests(unittest.TestCase):
    def test_specs_are_unique_and_conservatively_regularized(self) -> None:
        names = [spec["name"] for spec in MODEL_SPECS]
        self.assertEqual(len(names), len(set(names)))
        self.assertTrue(all(spec["min_child_weight"] >= 10.0 for spec in MODEL_SPECS))
        self.assertTrue(all(spec["learning_rate"] <= 0.04 for spec in MODEL_SPECS))

    def test_model_has_frozen_leakage_safe_training_contract(self) -> None:
        model = build_model(MODEL_SPECS[0], seed=17)
        params = model.get_params()
        self.assertEqual(params["objective"], "binary:logistic")
        self.assertEqual(params["tree_method"], "hist")
        self.assertEqual(params["device"], "cpu")
        self.assertEqual(params["random_state"], 17)
        self.assertEqual(params["reg_lambda"], 10.0)
        self.assertEqual(params["subsample"], 0.8)


if __name__ == "__main__":
    unittest.main()
