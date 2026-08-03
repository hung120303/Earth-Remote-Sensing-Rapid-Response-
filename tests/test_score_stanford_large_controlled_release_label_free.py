from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from score_stanford_large_controlled_release_label_free import (  # noqa: E402
    FROZEN_BANDS,
    assert_no_outcome_data,
    audit_deployability,
    build_mars_input,
    calibrated_spatial_prithvi_score,
    compose_gaussian_dofa_score,
    connected_scene_score,
    oof_extratrees_current_score,
    parse_args,
    released_scene_decision,
    validate_frozen_bindings,
    write_label_free_outputs,
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ConstantHead:
    def __init__(self, probabilities: list[float]) -> None:
        self.probabilities = np.asarray(probabilities, dtype=np.float64)

    def predict_proba(self, values: np.ndarray) -> np.ndarray:
        assert values.shape[0] == self.probabilities.size
        return np.column_stack((1.0 - self.probabilities, self.probabilities))


def test_build_mars_input_preserves_frozen_dn_wind_cloud_and_observability_contract() -> None:
    target = np.full((6, 4, 4), 1000, dtype=np.uint16)
    reference = np.full((6, 4, 4), 500, dtype=np.uint16)
    target[:, 0, 0] = 0

    model_input, observable, physical_pair = build_mars_input(target, reference)

    assert FROZEN_BANDS == ("B02", "B03", "B04", "B08", "B11", "B12")
    assert model_input.shape == (16, 4, 4)
    # Physical Sentinel-2 L1C reflectance is DN / 10,000; released MARS input is
    # physical reflectance / 0.5, hence DN / 5,000.
    assert np.allclose(physical_pair[:6, 1:, 1:], 0.1)
    assert np.allclose(physical_pair[6:, 1:, 1:], 0.05)
    assert np.allclose(model_input[1:7, 1:, 1:], 0.2)
    assert np.allclose(model_input[7:13, 1:, 1:], 0.1)
    assert np.all(model_input[13] == 0.5)
    assert np.all(model_input[14] == 0.5)
    assert np.all(model_input[15] == 0.0)
    assert not bool(observable[0, 0])
    assert bool(observable[1, 1])
    assert np.isfinite(model_input).all()


def test_released_connected_score_and_decision_use_strict_frozen_rule() -> None:
    score = np.full((16, 16), 0.1, dtype=np.float32)
    score[2:12, 3:13] = 0.75  # exactly 100 8-connected pixels

    observed = connected_scene_score(score)

    assert abs(observed - 0.75) <= 1e-3
    assert released_scene_decision(observed)
    assert not released_scene_decision(0.5)


def test_oof_extratrees_current_score_uses_frozen_logit_blend() -> None:
    artifact: dict[str, Any] = {
        "schema_version": 1,
        "architecture": "mars_oof_scene_ensemble_v2",
        "blend_lambda": 0.625,
        "augmented_feature_names": ["a", "b"],
        "primary_feature": "primary_connected_score",
        "fitted": ConstantHead([0.8, 0.2]),
    }
    features = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    primary = np.asarray([0.6, 0.4], dtype=np.float64)

    observed = oof_extratrees_current_score(
        artifact,
        augmented_features=features,
        augmented_feature_names=["a", "b"],
        primary_scores=primary,
    )

    expected_logit = (
        0.375 * np.log(primary / (1.0 - primary))
        + 0.625 * np.log(np.asarray([0.8, 0.2]) / np.asarray([0.2, 0.8]))
    )
    expected = 1.0 / (1.0 + np.exp(-expected_logit))
    assert np.allclose(observed, expected)
    with pytest.raises(ValueError, match="feature schema"):
        oof_extratrees_current_score(
            artifact,
            augmented_features=features,
            augmented_feature_names=["b", "a"],
            primary_scores=primary,
        )


def test_gaussian_dofa_composition_preserves_below_gate_scores() -> None:
    current = np.asarray([0.10, 0.30, 0.60, 0.90], dtype=np.float64)
    gaussian_logits = np.asarray([100.0, -2.0, 0.5, -0.5], dtype=np.float64)
    dofa_raw = np.asarray([0.99, 0.20, 0.80, 0.40], dtype=np.float64)

    observed = compose_gaussian_dofa_score(
        current,
        gaussian_logits,
        dofa_raw,
        gaussian_strength=0.1,
        final_gate=0.25,
        dofa_gate=0.5,
        dofa_weight=0.05,
    )

    assert observed.shape == current.shape
    assert observed[0] == current[0]
    assert np.all(observed[1:] >= 0.25)
    assert np.isfinite(observed).all()
    assert np.all((observed >= 0.0) & (observed <= 1.0))


def test_spatial_prithvi_composition_is_frozen_logit_blend_then_offset() -> None:
    spatial = np.asarray([0.2, 0.8], dtype=np.float64)
    prithvi = np.asarray([0.7, 0.3], dtype=np.float64)

    raw, calibrated = calibrated_spatial_prithvi_score(
        spatial,
        prithvi,
        prithvi_weight=0.75,
        logit_offset=-0.5044762783678565,
    )

    assert np.all(calibrated < raw)
    assert np.isfinite(raw).all() and np.isfinite(calibrated).all()


def test_outcome_guard_rejects_values_but_allows_negative_access_attestations() -> None:
    assert_no_outcome_data(
        {
            "event_id": ["event-a"],
            "labels_accessed": False,
            "outcome_blindness": {"detector_outcomes_accessed": False},
        }
    )
    for forbidden in (
        {"label": [0]},
        {"nested": {"metered_ch4_kgh": 1000.0}},
        {"truth_stratum": "primary_positive"},
        {"official_test_outcome": "positive"},
    ):
        with pytest.raises(ValueError, match="forbidden outcome"):
            assert_no_outcome_data(forbidden)


def test_frozen_binding_validation_hashes_every_contract_and_rejects_escape(tmp_path: Path) -> None:
    first = tmp_path / "a.bin"
    second = tmp_path / "b.bin"
    first.write_bytes(b"a")
    second.write_bytes(b"b")
    protocol = {
        "one": {"path": "a.bin", "sha256": digest(first)},
        "nested": {"two": {"path": "b.bin", "sha256": digest(second)}},
    }

    records = validate_frozen_bindings(tmp_path, protocol)

    assert {record["binding"] for record in records} == {"one", "nested.two"}
    assert all(record["verified"] for record in records)
    second.write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_frozen_bindings(tmp_path, protocol)
    protocol["nested"]["two"]["path"] = "../escape.bin"
    with pytest.raises(ValueError, match="beneath repository root"):
        validate_frozen_bindings(tmp_path, protocol)


def test_current_frozen_protocol_is_deployable_without_shortcuts() -> None:
    protocol = json.loads(
        (ROOT / "configs/stanford_large_controlled_release_scoring_protocol.json").read_text(
            encoding="utf-8"
        )
    )

    audit = audit_deployability(protocol)

    assert audit["deployable"] is True
    assert audit["blockers"] == []
    assert audit["no_shortcut_used"] is True


def test_writer_emits_only_label_free_arrays_manifest_and_compact_receipt(
    tmp_path: Path,
) -> None:
    arrays = {
        "event_ids": np.asarray(["event-a", "event-b"]),
        "released_mars_v3_scores": np.asarray([0.1, 0.8]),
        "released_mars_v3_decisions": np.asarray([0, 1], dtype=np.uint8),
        "current_oof_extratrees_scores": np.asarray([0.2, 0.7]),
        "gaussian_dofa_scores": np.asarray([0.2, 0.75]),
        "gaussian_dofa_decisions": np.asarray([1, 1], dtype=np.uint8),
        "calibrated_spatial_prithvi_scores": np.asarray([0.05, 0.6]),
        "calibrated_spatial_prithvi_decisions": np.asarray([0, 1], dtype=np.uint8),
    }
    score_path = tmp_path / ".research/scores/scores.npz"
    manifest_path = tmp_path / ".research/scores/manifest.json"
    receipt_path = tmp_path / "reports/receipt.json"

    receipt = write_label_free_outputs(
        root=tmp_path,
        arrays=arrays,
        sample_manifest=[
            {
                "event_id": "event-a",
                "target_sha256": "a" * 64,
                "reference_sha256": "b" * 64,
            },
            {
                "event_id": "event-b",
                "target_sha256": "c" * 64,
                "reference_sha256": "d" * 64,
            },
        ],
        score_path=score_path,
        manifest_path=manifest_path,
        receipt_path=receipt_path,
        protocol_sha256="e" * 64,
        script_sha256="f" * 64,
    )

    with np.load(score_path, allow_pickle=False) as cache:
        assert set(cache.files) == set(arrays) | {"schema_version"}
        assert all(
            token not in " ".join(cache.files).lower()
            for token in ("label", "truth", "metered", "outcome", "release_rate")
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    written_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert manifest["rows"] == 2
    assert written_receipt == receipt
    assert written_receipt["labels_or_outcomes_accessed"] is False
    assert written_receipt["scores"]["sha256"] == digest(score_path)
    assert written_receipt["manifest"]["sha256"] == digest(manifest_path)
    assert_no_outcome_data(manifest)
    assert_no_outcome_data(written_receipt)


def test_cli_exposes_dry_run_and_positive_limit() -> None:
    args = parse_args(["--dry-run", "--limit", "3"])
    assert args.dry_run is True
    assert args.limit == 3
    with pytest.raises(SystemExit):
        parse_args(["--limit", "0"])
