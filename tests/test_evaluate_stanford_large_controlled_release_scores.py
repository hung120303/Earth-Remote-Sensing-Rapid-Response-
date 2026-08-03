from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from evaluate_stanford_large_controlled_release_scores import (
    binary_metrics,
    paired_date_bootstrap,
    score_model,
    superiority_gate,
    validate_score_bundle,
    wilson_interval,
)


def test_wilson_interval_is_bounded_and_contains_observed_fraction() -> None:
    lower, upper = wilson_interval(7, 10, confidence=0.95)
    assert 0.0 <= lower < 0.7 < upper <= 1.0
    assert wilson_interval(0, 0, confidence=0.95) == (None, None)


def test_binary_metrics_honors_strict_and_inclusive_thresholds() -> None:
    labels = np.asarray([0, 0, 1, 1], dtype=np.int8)
    scores = np.asarray([0.5, 0.6, 0.5, 0.9], dtype=np.float64)
    strict = binary_metrics(labels, scores, threshold=0.5, comparator=">")
    inclusive = binary_metrics(labels, scores, threshold=0.5, comparator=">=")
    assert strict["confusion"] == {"tp": 1, "tn": 1, "fp": 1, "fn": 1}
    assert inclusive["confusion"] == {"tp": 2, "tn": 0, "fp": 2, "fn": 0}


def test_score_model_reports_wilson_intervals() -> None:
    labels = np.asarray([0, 0, 1, 1], dtype=np.int8)
    scores = np.asarray([0.1, 0.8, 0.7, 0.9], dtype=np.float64)
    result = score_model(labels, scores, threshold=0.5, comparator=">=")
    assert result["average_precision"] == pytest.approx(5.0 / 6.0)
    assert result["fixed_threshold"]["recall"] == 1.0
    assert result["fixed_threshold"]["false_positive_rate"] == 0.5
    assert result["fixed_threshold"]["recall_interval_95"][0] < 1.0


def test_paired_date_bootstrap_is_deterministic_and_paired() -> None:
    labels = np.asarray([0, 1, 0, 1, 0, 1], dtype=np.int8)
    baseline = np.asarray([0.1, 0.6, 0.2, 0.7, 0.3, 0.8], dtype=np.float64)
    candidate = np.asarray([0.05, 0.8, 0.1, 0.85, 0.2, 0.9], dtype=np.float64)
    dates = np.asarray(["2025-01-01", "2025-01-01", "2025-01-02", "2025-01-02", "2025-01-03", "2025-01-03"])
    first = paired_date_bootstrap(
        labels,
        baseline,
        candidate,
        dates,
        baseline_threshold=0.5,
        baseline_comparator=">",
        candidate_threshold=0.5,
        candidate_comparator=">=",
        replicates=500,
        seed=123,
    )
    second = paired_date_bootstrap(
        labels,
        baseline,
        candidate,
        dates,
        baseline_threshold=0.5,
        baseline_comparator=">",
        candidate_threshold=0.5,
        candidate_comparator=">=",
        replicates=500,
        seed=123,
    )
    assert first == second
    assert first["average_precision_delta"]["valid_replicates"] > 0


def test_superiority_requires_all_three_interval_gates() -> None:
    passed = {
        "average_precision_delta": {"lower": 0.01, "upper": 0.1},
        "recall_delta": {"lower": 0.01, "upper": 0.2},
        "false_positive_rate_delta": {"lower": -0.1, "upper": 0.0},
    }
    failed_fpr = {
        **passed,
        "false_positive_rate_delta": {"lower": -0.1, "upper": 0.01},
    }
    assert superiority_gate(passed)["passed"] is True
    assert superiority_gate(failed_fpr)["passed"] is False


def test_validate_score_bundle_rejects_missing_or_duplicate_events(tmp_path: Path) -> None:
    crop = {
        "summary": {"complete_pairs": 2, "errors": 0},
        "samples": [{"event_id": "a"}, {"event_id": "b"}],
    }
    crop_path = tmp_path / "crop.json"
    crop_path.write_text(json.dumps(crop), encoding="utf-8")
    arrays = {
        "event_ids": np.asarray(["a", "b"]),
        "released_mars_v3_scores": np.asarray([0.1, 0.2]),
        "gaussian_dofa_scores": np.asarray([0.2, 0.3]),
        "calibrated_spatial_prithvi_scores": np.asarray([0.3, 0.4]),
    }
    validate_score_bundle(arrays, crop_path, expected_rows=2)
    with pytest.raises(ValueError, match="duplicate"):
        validate_score_bundle({**arrays, "event_ids": np.asarray(["a", "a"])}, crop_path, expected_rows=2)
    with pytest.raises(ValueError, match="event IDs"):
        validate_score_bundle({**arrays, "event_ids": np.asarray(["a", "c"])}, crop_path, expected_rows=2)
