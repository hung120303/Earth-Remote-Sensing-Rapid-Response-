from __future__ import annotations

import numpy as np

from tools.extract_mars_prior_reference_scores import (
    SAFE_ASSET_ROLES,
    released_input,
    safe_manifest_record,
)


def test_safe_manifest_record_drops_outcomes_and_non_input_assets() -> None:
    raw = {
        "sample_id": "sample",
        "group_id": "group",
        "sensor_family": "Sentinel-2",
        "target_datetime": "2020-01-01T00:00:00+00:00",
        "target_scene_id": "scene",
        "wind_u": 1.0,
        "wind_v": 2.0,
        "band_order": ["band"] * 12,
        "label_state": "PLUME",
        "assets": [
            {"role": "image", "path": "image.tif"},
            {"role": "cloud_mask", "path": "cloud.tif"},
            {"role": "plume_mask", "path": "forbidden-mask.tif"},
            {"role": "methane_enhancement", "path": "forbidden-enhancement.tif"},
        ],
    }
    safe = safe_manifest_record(raw)
    assert "label_state" not in safe
    assert set(safe["assets"]) == SAFE_ASSET_ROLES
    assert "forbidden" not in repr(safe)


def test_released_input_preserves_exact_channel_contract() -> None:
    target = np.full((6, 4, 5), 5000, dtype=np.uint16)
    reference = np.full((6, 4, 5), 2500, dtype=np.uint16)
    cloud = np.zeros((4, 5), dtype=np.uint8)
    cloud[0, 0] = 2
    values = released_input(target, reference, (8.0, -4.0), cloud)
    assert values.shape == (16, 4, 5)
    np.testing.assert_allclose(values[0], 1.0)
    np.testing.assert_allclose(values[1:7], 1.0)
    np.testing.assert_allclose(values[7:13], 0.5)
    np.testing.assert_allclose(values[13], 1.0)
    np.testing.assert_allclose(values[14], -0.5)
    assert values[15, 0, 0] == 1.0
    assert np.sum(values[15]) == 1.0
