from __future__ import annotations

import numpy as np
import pytest

from tools.build_mars_prior_reference_bank import (
    descriptor,
    reference_distance,
    scene_grid_key,
    select_prior_references,
)


def metadata_row(
    sample_id: str,
    timestamp: float,
    *,
    sensor: str = "Sentinel-2",
    clear: float = 100.0,
    transform: list[float] | None = None,
) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "fold": 3,
        "physical_location_id": "site",
        "sensor_family": sensor,
        "grid_key": "S2:32SKA" if sensor == "Sentinel-2" else "Landsat:190030",
        "target_scene_id": sample_id,
        "target_datetime": f"2020-01-{int(timestamp):02d}T00:00:00+00:00",
        "timestamp": timestamp,
        "percentage_clear": clear,
        "crs": "EPSG:32632",
        "transform": transform or [10.0, 0.0, 1.0, 0.0, -10.0, 2.0],
    }


def test_scene_grid_key_parses_both_sensor_contracts() -> None:
    assert (
        scene_grid_key("S2A_MSIL1C_20200101T000000_N0000_R000_T32SKA_20200101T000000")
        == "S2:32SKA"
    )
    assert (
        scene_grid_key("LC09_L1TP_190030_20200101_20200101_02_T1") == "Landsat:190030"
    )
    with pytest.raises(ValueError, match="Cannot infer"):
        scene_grid_key("unknown")


def test_descriptor_is_invariant_to_common_positive_scale() -> None:
    values = np.arange(1, 65, dtype=np.float32).reshape(4, 4, 4)
    first, first_medians = descriptor(values)
    second, second_medians = descriptor(values * 3.0)
    np.testing.assert_allclose(first, second)
    np.testing.assert_allclose(second_medians, first_medians * 3.0)


def test_reference_distance_penalizes_shape_and_radiometry() -> None:
    target = np.zeros(8, dtype=np.float32)
    medians = np.ones(4, dtype=np.float32)
    candidates = np.stack([target, np.ones(8, dtype=np.float32)])
    scales = np.stack([medians * 2.0, medians])
    distances = reference_distance(
        target,
        medians,
        candidates,
        scales,
        radiometric_weight=0.25,
    )
    assert distances[0] == pytest.approx(0.25 * np.log(2.0))
    assert distances[1] == pytest.approx(1.0)


def test_selection_is_strictly_prior_clear_exact_grid_and_sentinel_only() -> None:
    rows = [
        metadata_row("prior_good", 1.0),
        metadata_row("prior_cloudy", 2.0, clear=90.0),
        metadata_row(
            "prior_shifted",
            3.0,
            transform=[10.0, 0.0, 11.0, 0.0, -10.0, 2.0],
        ),
        metadata_row("target", 4.0),
        metadata_row("future", 5.0),
        metadata_row("landsat_prior", 1.0, sensor="Landsat"),
        metadata_row("landsat_target", 2.0, sensor="Landsat"),
    ]
    descriptors = np.stack(
        [
            np.full(8, 0.1),
            np.full(8, 0.2),
            np.full(8, 0.3),
            np.zeros(8),
            np.zeros(8),
            np.zeros(8),
            np.zeros(8),
        ]
    ).astype(np.float32)
    medians = np.ones((len(rows), 4), dtype=np.float32)
    selections, summary = select_prior_references(
        rows,
        descriptors,
        medians,
        descriptors,
        medians,
        minimum_percentage_clear=95.0,
        recent_pool_size=10,
        selected_references=5,
        radiometric_weight=0.25,
    )
    target = next(row for row in selections if row["sample_id"] == "target")
    assert target["selected_sample_ids"] == ["prior_good"]
    assert target["strictly_prior_clear_candidates"] == 2
    assert target["exact_grid_candidates"] == 1
    landsat = next(row for row in selections if row["sample_id"] == "landsat_target")
    assert landsat["selected_sample_ids"] == []
    assert landsat["fallback_to_original_only"] is True
    assert summary["sentinel_rows_with_selected_reference"] == 3
