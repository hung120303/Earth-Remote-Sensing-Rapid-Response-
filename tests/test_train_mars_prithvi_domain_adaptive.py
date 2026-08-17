from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from train_mars_prithvi_domain_adaptive import (
    corrected_scores,
    load_champion,
    record_weights,
    standalone_replication_checks,
)


def test_zero_correction_exactly_preserves_champion() -> None:
    baseline = np.asarray([0.01, 0.2, 0.7, 0.99], dtype=np.float64)
    result = corrected_scores(baseline, np.zeros_like(baseline), 0.5)
    assert np.allclose(result, baseline, rtol=0.0, atol=1e-15)


def test_correction_strength_and_shapes_are_validated() -> None:
    baseline = np.asarray([0.2, 0.8])
    correction = np.asarray([-0.4, 0.4])
    weak = corrected_scores(baseline, correction, 0.25)
    strong = corrected_scores(baseline, correction, 1.0)
    assert strong[0] < weak[0] < baseline[0]
    assert strong[1] > weak[1] > baseline[1]
    with pytest.raises(ValueError, match="strength"):
        corrected_scores(baseline, correction, 0.0)
    with pytest.raises(ValueError, match="shapes"):
        corrected_scores(baseline, correction[:1], 0.5)


def test_error_aware_weights_are_finite_and_normalized() -> None:
    records = [
        {
            "sample_id": "a",
            "group_id": "g1",
            "label_state": "NO_PLUME",
            "sensor_family": "Sentinel-2",
        },
        {
            "sample_id": "b",
            "group_id": "g2",
            "label_state": "PLUME",
            "sensor_family": "Landsat",
        },
        {
            "sample_id": "c",
            "group_id": "g2",
            "label_state": "PLUME",
            "sensor_family": "Landsat",
        },
    ]
    values = record_weights(records, {"a": 0.9, "b": 0.1, "c": 0.9})
    assert np.isfinite(values.numpy()).all()
    assert values.mean().item() == pytest.approx(1.0)
    assert values[1] > values[2]


def test_champion_loader_rejects_non_selection_folds(tmp_path: Path) -> None:
    path = tmp_path / "scores.npz"
    np.savez_compressed(
        path,
        sample_ids=np.asarray(["x", "y"]),
        labels=np.asarray([0, 1], dtype=np.uint8),
        sensors=np.asarray([0, 1], dtype=np.uint8),
        groups=np.asarray(["g1", "g2"]),
        folds=np.asarray([2, 3], dtype=np.uint8),
        champion_scores=np.asarray([0.1, 0.9]),
    )
    with pytest.raises(ValueError, match="folds 3/4"):
        load_champion(path)


def test_replication_requires_every_frozen_pilot_check() -> None:
    checks = {
        "pooled_ap": True,
        "every_fold_ap": True,
        "every_sensor_ap": False,
        "pooled_recall": True,
        "every_fold_recall": True,
        "paired_site_ap": False,
    }
    result = standalone_replication_checks({"checks": checks})
    assert result == {f"seed_two_{name}": value for name, value in checks.items()}
    assert not all(result.values())
