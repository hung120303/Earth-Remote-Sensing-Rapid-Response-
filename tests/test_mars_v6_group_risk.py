from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from calibrate_mars_v6_group_risk import (  # noqa: E402
    calibration_report,
    crc_threshold,
    group_losses,
)


def fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scores = np.asarray([0.9, 0.8, 0.7, 0.6, 0.4, 0.3, 0.2, 0.1])
    labels = np.asarray([1, 1, 0, 0, 0, 0, 0, 0])
    groups = np.asarray(["p1", "p2", "n1", "n1", "n2", "n2", "n3", "n3"])
    return scores, labels, groups


def test_group_loss_weights_groups_not_crops() -> None:
    scores = np.asarray([0.9, 0.8, 0.1, 0.1, 0.1])
    labels = np.zeros(5, dtype=int)
    groups = np.asarray(["large", "small", "small", "small", "small"])
    losses = group_losses(scores, labels, groups, 0.5)
    assert sorted(losses.tolist()) == [0.25, 1.0]
    assert losses.mean() == pytest.approx(0.625)


def test_crc_selects_most_permissive_feasible_threshold() -> None:
    scores, labels, groups = fixture()
    result = crc_threshold(scores, labels, groups, alpha=0.40)
    assert result["feasible"] is True
    assert result["crc_expected_risk_bound"] <= 0.40
    lower = np.nextafter(result["threshold"], -np.inf)
    lower_losses = group_losses(scores, labels, groups, lower)
    lower_bound = (len(lower_losses) * lower_losses.mean() + 1.0) / (len(lower_losses) + 1.0)
    assert lower_bound > 0.40


def test_crc_reports_infeasible_below_finite_sample_floor() -> None:
    scores, labels, groups = fixture()
    result = crc_threshold(scores, labels, groups, alpha=0.20)
    assert result["feasible"] is False
    assert result["minimum_achievable_crc_bound"] == pytest.approx(0.25)


def test_mondrian_diagnostics_require_enough_negative_groups() -> None:
    scores, labels, groups = fixture()
    products = np.asarray(["a", "b", "a", "a", "a", "a", "b", "b"])
    report = calibration_report(
        scores,
        labels,
        groups,
        [0.4],
        strata={"products": products},
        minimum_stratum_negative_groups=2,
    )
    assert report["strata"]["products"]["a"]["eligible"] is True
    assert report["strata"]["products"]["b"]["eligible"] is False
    assert "exchangeable" in report["guarantee_scope"]

