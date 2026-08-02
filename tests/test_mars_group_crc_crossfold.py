from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from evaluate_mars_group_crc_crossfold import (  # noqa: E402
    crc_threshold_fast,
    crossfold_curve,
    threshold_metrics,
)
from calibrate_mars_v6_group_risk import crc_threshold  # noqa: E402


def test_threshold_metrics_weights_physical_groups_equally() -> None:
    scores = np.asarray([0.9, 0.1, 0.1, 0.1, 0.8, 0.2])
    labels = np.asarray([0, 0, 0, 0, 1, 1])
    groups = np.asarray(["large", "small", "small", "small", "p1", "p2"])
    result = threshold_metrics(scores, labels, groups, threshold=0.5)
    assert result["crop_false_positive_rate"] == pytest.approx(0.25)
    assert result["group_balanced_false_positive_rate"] == pytest.approx(0.5)
    assert result["crop_recall"] == pytest.approx(0.5)


def test_crossfold_thresholds_never_use_confirmation_labels() -> None:
    scores = np.asarray(
        [0.9, 0.8, 0.7, 0.1, 0.9, 0.8, 0.2, 0.1] * 2,
        dtype=float,
    )
    labels = np.asarray([1, 1, 0, 0, 1, 1, 0, 0] * 2, dtype=int)
    groups = np.asarray(
        ["3p1", "3p2", "3n1", "3n2", "3p3", "3p4", "3n3", "3n4"]
        + ["4p1", "4p2", "4n1", "4n2", "4p3", "4p4", "4n3", "4n4"]
    )
    folds = np.asarray([3] * 8 + [4] * 8)
    sensors = np.asarray([0, 1, 0, 1, 0, 1, 0, 1] * 2)
    first = crossfold_curve(scores, labels, groups, folds, sensors, [0.4])
    altered = labels.copy()
    altered[folds == 4] = 1 - altered[folds == 4]
    second = crossfold_curve(scores, altered, groups, folds, sensors, [0.4])
    first_threshold = first["directions"]["fold3_to_fold4"]["curve"][0][
        "calibration"
    ]["threshold"]
    second_threshold = second["directions"]["fold3_to_fold4"]["curve"][0][
        "calibration"
    ]["threshold"]
    assert first_threshold == second_threshold


def test_crossfold_requires_exact_folds_three_and_four() -> None:
    with pytest.raises(ValueError, match="Expected folds"):
        crossfold_curve(
            np.asarray([0.9, 0.1]),
            np.asarray([1, 0]),
            np.asarray(["a", "b"]),
            np.asarray([2, 3]),
            np.asarray([0, 0]),
            [0.5],
        )


@pytest.mark.parametrize("alpha", [0.2, 0.4, 0.6, 0.9])
def test_fast_crc_matches_reference_scan(alpha: float) -> None:
    scores = np.asarray([0.99, 0.81, 0.7, 0.7, 0.4, 0.3, 0.2, 0.1])
    labels = np.asarray([1, 1, 0, 0, 0, 0, 0, 0])
    groups = np.asarray(["p1", "p2", "n1", "n1", "n2", "n2", "n3", "n3"])
    expected = crc_threshold(scores, labels, groups, alpha)
    actual = crc_threshold_fast(scores, labels, groups, alpha)
    assert actual["feasible"] is expected["feasible"]
    assert actual.get("threshold") == expected.get("threshold")
    if actual["feasible"]:
        assert actual["crc_expected_risk_bound"] == pytest.approx(
            expected["crc_expected_risk_bound"]
        )
