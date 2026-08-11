from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
SPEC = importlib.util.spec_from_file_location(
    "group_crc_transport",
    TOOLS / "evaluate_methanes2cm_v5_1_group_crc_transport.py",
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_pooled_threshold_is_most_permissive_feasible() -> None:
    scores = np.asarray([0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
    labels = np.asarray([1, 1, 0, 0, 0, 0])
    threshold = MODULE.pooled_fpr_threshold(scores, labels, 0.25)
    negative = scores[labels == 0]
    assert np.mean(negative >= threshold) <= 0.25
    assert np.mean(negative >= np.nextafter(threshold, -np.inf)) > 0.25


def test_operating_metrics_balance_groups_not_crops() -> None:
    scores = np.asarray([0.9, 0.9, 0.1, 0.1, 0.1, 0.1, 0.8, 0.2])
    labels = np.asarray([0, 0, 0, 0, 0, 0, 1, 1])
    groups = np.asarray(["small", "large", "large", "large", "large", "large", "p", "p"])
    result = MODULE.operating_metrics(scores, labels, groups, 0.5)
    assert result["false_positive_rate"] == pytest.approx(2 / 6)
    assert result["group_balanced_false_positive_rate"] == pytest.approx(0.6)
    assert result["recall"] == pytest.approx(0.5)


def test_paired_group_bootstrap_is_deterministic_and_paired() -> None:
    scores = np.asarray([0.9, 0.7, 0.6, 0.4, 0.8, 0.5, 0.3, 0.2])
    labels = np.asarray([1, 0, 1, 0, 1, 0, 1, 0])
    groups = np.asarray(["a", "a", "b", "b", "c", "c", "d", "d"])
    first = MODULE.paired_group_bootstrap(scores, labels, groups, 0.75, 0.45, 500, 17, 0.95)
    second = MODULE.paired_group_bootstrap(scores, labels, groups, 0.75, 0.45, 500, 17, 0.95)
    assert first == second
    assert first["group_balanced_fpr_delta"]["point"] <= 0.0
    assert first["group_balanced_recall_delta"]["point"] <= 0.0


def test_partition_requires_exact_disjoint_24_by_24() -> None:
    calibration = [f"c{i}" for i in range(24)]
    confirmation = [f"h{i}" for i in range(24)]
    rows = [{"group_id": group} for group in calibration + confirmation]
    receipt = {
        "held_confirmation_rows": 48,
        "methanes2cm_confirmation_partition": {
            "risk_calibration_groups": calibration,
            "confirmation_groups": confirmation,
        },
    }
    observed = MODULE.validate_partition(rows, receipt)
    assert observed == (set(calibration), set(confirmation))
    receipt["methanes2cm_confirmation_partition"]["confirmation_groups"][0] = "c0"
    with pytest.raises(ValueError, match="overlap"):
        MODULE.validate_partition(rows, receipt)


def test_fit_and_evaluate_require_distinct_authorization_states(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="does not authorize calibration"):
        MODULE.fit(tmp_path, {"status": "draft"}, {})
    with pytest.raises(ValueError, match="does not authorize confirmation"):
        MODULE.evaluate(tmp_path, {"status": "fit_authorized_before_confirmation_score_label_join"}, {})


def test_implementation_does_not_reference_opened_location_test_artifacts() -> None:
    source = (TOOLS / "evaluate_methanes2cm_v5_1_group_crc_transport.py").read_text(
        encoding="utf-8"
    )
    assert "v5_1_location_test_predictions" not in source
    assert "methanes2cm_v5_1_location_test.json" not in source
