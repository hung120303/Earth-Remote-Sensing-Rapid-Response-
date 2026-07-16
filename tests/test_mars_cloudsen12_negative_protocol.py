from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/mars_cloudsen12_negative_augmented_xgboost_protocol.json"


def test_cloudsen12_protocol_preserves_sealed_and_feature_boundaries() -> None:
    payload = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    partitions = payload["partitions"]
    features = payload["feature_contract"]
    invariants = " ".join(payload["invariants"]).lower()

    assert payload["status"] == "frozen_before_source_extraction_or_model_fitting"
    assert "test only" in partitions["cloudsen12_sealed_external"]
    assert "inaccessible" in partitions["cloudsen12_sealed_external"]
    assert "paper-test" in invariants
    assert "offshore" not in features["allowed"]
    assert "country" not in features["allowed"]
    assert "offshore" in features["prohibited"]
    assert payload["base_architecture"]["dense_mask_change"] is False


def test_cloudsen12_protocol_uses_only_operational_statistics() -> None:
    payload = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    allowed = payload["feature_contract"]["allowed"]

    assert len(allowed) == 32
    assert len(set(allowed)) == len(allowed)
    assert allowed[:2] == ["wind_u", "wind_v"]
    assert set(allowed[-2:]) == {"cloudmask_0.0", "cloudmask_1.0"}
    assert not any("plume" in name.lower() or "ch4" in name.lower() for name in allowed)
