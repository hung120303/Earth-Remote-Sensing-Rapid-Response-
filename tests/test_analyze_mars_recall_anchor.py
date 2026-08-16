from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from analyze_mars_recall_anchor import (  # noqa: E402
    align_feature_rows,
    decision_table,
    standardized_feature_contrasts,
)


def test_align_feature_rows_preserves_champion_order() -> None:
    indices = align_feature_rows(
        np.asarray(["c", "a"]),
        np.asarray(["a", "b", "c"]),
        np.asarray([3, 2, 4], dtype=np.uint8),
    )
    np.testing.assert_array_equal(indices, np.asarray([2, 0]))


def test_decision_table_partitions_rows_and_labels() -> None:
    result = decision_table(
        np.asarray([True, True, False, False]),
        np.asarray([True, False, True, False]),
        np.asarray([1, 0, 1, 0], dtype=np.uint8),
    )
    assert sum(value["rows"] for value in result.values()) == 4
    assert result["both"]["positives"] == 1
    assert result["first_only"]["negatives"] == 1
    assert result["second_only"]["positives"] == 1


def test_feature_contrast_orders_absolute_effect() -> None:
    features = np.asarray(
        [[4.0, 1.0], [5.0, 3.0], [0.0, 0.0], [1.0, 2.0]], dtype=np.float64
    )
    contrasts = standardized_feature_contrasts(
        features,
        np.asarray(["strong", "weak"]),
        np.asarray([True, True, False, False]),
        np.asarray([False, False, True, True]),
        2,
    )
    assert contrasts[0]["feature"] == "strong"
    assert contrasts[0]["standardized_mean_difference"] > 0
