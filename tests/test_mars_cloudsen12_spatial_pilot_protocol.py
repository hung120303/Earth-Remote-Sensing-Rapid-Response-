from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/mars_cloudsen12_spatial_pilot_protocol.json"


def test_spatial_pilot_is_bounded_and_keeps_test_sealed() -> None:
    payload = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    assert payload["status"] == "frozen_before_pilot_selection_or_pixel_acquisition"
    assert payload["sample_sizes"] == {"train": 384, "validation": 128, "sealed_test": 0}
    assert payload["eligibility"]["published_test_partition_accessed"] is False
    assert payload["eligibility"]["minimum_distance_from_any_mars_emitter_site_km"] == 25
    assert payload["storage"]["bulk_data_ignored"] is True
    assert "No paper cache" in payload["go_no_go"]["paper_access"]


def test_spatial_pilot_preserves_released_band_contract() -> None:
    payload = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    bands = payload["pixel_contract"]["band_order"]
    assert len(bands) == 12
    assert bands[:6] == ["B02", "B03", "B04", "B08", "B11", "B12"]
    assert bands[6:] == [f"{band}_bg" for band in bands[:6]]
    assert payload["pixel_contract"]["label_state"] == "NO_PLUME"
