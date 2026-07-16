from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools/train_mars_cloudsen12_negative_augmented_xgboost.py"
SPEC = importlib.util.spec_from_file_location("train_cloudsen", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_augmented_fit_arrays_weights_only_cloud_rows() -> None:
    mars = np.asarray([[1.0], [2.0]])
    labels = np.asarray([0, 1], dtype=np.uint8)
    cloud = np.asarray([[3.0], [4.0], [5.0]])

    features, output_labels, weights = MODULE.augmented_fit_arrays(
        mars, labels, cloud, 4.0
    )

    np.testing.assert_allclose(features[:, 0], [1, 2, 3, 4, 5])
    np.testing.assert_array_equal(output_labels, [0, 1, 0, 0, 0])
    np.testing.assert_allclose(weights, [1, 1, 4, 4, 4])


def test_augmented_fit_arrays_rejects_nonpositive_multiplier() -> None:
    with pytest.raises(ValueError, match="positive"):
        MODULE.augmented_fit_arrays(
            np.ones((1, 2)), np.zeros(1), np.ones((1, 2)), 0.0
        )


def test_frozen_model_builder_is_low_capacity() -> None:
    spec = {
        "n_estimators": 10,
        "max_depth": 2,
        "learning_rate": 0.03,
        "min_child_weight": 20.0,
    }
    fixed = {
        "tree_method": "hist",
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 20.0,
        "n_jobs": 1,
    }
    model = MODULE.build_model(spec, fixed, seed=7)
    assert model.max_depth == 2
    assert model.reg_lambda == 20.0
    assert model.random_state == 7
