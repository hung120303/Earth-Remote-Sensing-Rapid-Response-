from __future__ import annotations

import numpy as np

from tools.analyze_mars_prior_reference_complementarity import (
    fixed_aggregations,
    reference_set_features,
    rescue_cell,
)


def test_fixed_aggregations_use_only_valid_prior_views_and_fallback() -> None:
    scores = np.asarray(
        [
            [0.1, 0.2, 0.8, 0.4, 0.0, 0.0],
            [0.3, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    mask = np.asarray(
        [
            [1, 1, 1, 1, 0, 0],
            [1, 0, 0, 0, 0, 0],
        ],
        dtype=bool,
    )
    distances = np.asarray([[0.01, 0.02, 0.03, np.nan, np.nan], [np.nan] * 5])
    result = fixed_aggregations(scores, mask, distances, softmax_temperature=0.02)
    np.testing.assert_allclose(result["nearest"], [0.2, 0.3])
    np.testing.assert_allclose(result["median"], [0.4, 0.3])
    np.testing.assert_allclose(result["top2_mean"], [0.6, 0.3])
    np.testing.assert_allclose(result["maximum"], [0.8, 0.3])
    assert 0.2 < result["similarity_weighted"][0] < 0.8


def test_reference_set_features_are_finite_with_fallback() -> None:
    values = np.asarray(
        [
            [[0.1, 1.0], [0.2, 2.0], [0.4, 4.0]],
            [[0.3, 3.0], [0.0, 0.0], [0.0, 0.0]],
        ]
    )
    mask = np.asarray([[1, 1, 1], [1, 0, 0]], dtype=bool)
    distances = np.asarray([[0.1, 0.2], [np.nan, np.nan]])
    features, names = reference_set_features(
        values, mask, distances, np.asarray(["score", "area"])
    )
    assert features.shape == (2, 12)
    assert names.shape == (12,)
    assert np.isfinite(features).all()
    assert features[1, -4] == 0.0


def test_rescue_cell_counts_groups_and_folds() -> None:
    selection = np.asarray([1, 1, 1, 0], dtype=bool)
    labels = np.asarray([1, 0, 1, 1], dtype=np.uint8)
    groups = np.asarray(["a", "b", "a", "c"])
    folds = np.asarray([3, 3, 4, 4], dtype=np.uint8)
    result = rescue_cell(selection, labels, groups, folds)
    assert result["rows"] == 3
    assert result["positives"] == 2
    assert result["negatives"] == 1
    assert result["positive_groups"] == 1
    assert result["positive_fold_counts"] == {"3": 1, "4": 1}
