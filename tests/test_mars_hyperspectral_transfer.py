from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from tools.acquire_mars_hyperspectral_metadata import safe_output_path
from tools.acquire_mars_hyperspectral_train_labels import (
    allowed_patterns,
    validate_download,
)
from tools.audit_mars_hyperspectral_transfer import (
    haversine_km,
    parse_datetime,
    read_mars_observations,
)
from tools.audit_mars_hyperspectral_train_masks import (
    geographic_group_ids,
    read_mask_fact,
)
from tools.query_mars_hyperspectral_cdse import summarize
from tools.query_mars_hyperspectral_stage_b_cdse import (
    _deduplicated_pairs,
    build_query_groups,
    point_in_bbox,
)


def test_safe_output_path_rejects_parent_escape(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unsafe repository path"):
        safe_output_path(tmp_path, "../secret")


def test_haversine_distance_is_reasonable() -> None:
    assert haversine_km(0.0, 0.0, 0.0, 1.0) == pytest.approx(111.195, rel=1e-4)


def test_parse_datetime_normalizes_zulu() -> None:
    assert parse_datetime("2024-01-02T03:04:05Z").isoformat() == (
        "2024-01-02T03:04:05+00:00"
    )


def test_mars_reader_uses_declared_safe_schema(tmp_path: Path) -> None:
    path = tmp_path / "mars.csv"
    fields = [
        "id_location",
        "location_name",
        "country",
        "lon",
        "lat",
        "satellite",
        "tile",
        "tile_date",
        "split_name",
        "isplume",
        "plume",
        "ch4_fluxrate",
        "ch4_fluxrate_std",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "id_location": "loc",
                "location_name": "site",
                "country": "Country",
                "lon": "2",
                "lat": "1",
                "satellite": "S2A",
                "tile": "scene",
                "tile_date": "2024-01-02T03:04:05+00:00",
                "split_name": "train_2023",
                "isplume": "True",
                "plume": "secret",
                "ch4_fluxrate": "999",
                "ch4_fluxrate_std": "1",
            }
        )
    rows = read_mars_observations(path)
    assert len(rows) == 1
    assert rows[0].location_name == "site"
    assert not hasattr(rows[0], "isplume")


def test_catalog_summary_counts_temporal_tiers() -> None:
    summary = summarize(
        [
            {
                "sample_ids": ["a", "b"],
                "products": [{"id": "p1", "offset_hours": 0.2}],
            },
            {
                "sample_ids": ["c"],
                "products": [{"id": "p2", "offset_hours": 0.8}],
            },
            {"sample_ids": ["d"], "products": []},
        ],
        3,
    )
    assert summary["hsi_samples_with_candidate"] == 3
    assert summary["within_15_minutes"] == 2
    assert summary["within_1_hour"] == 3
    assert summary["unique_sentinel_products"] == 2


def test_train_label_patterns_are_exact_and_minimal() -> None:
    patterns = allowed_patterns({"sample": "EMIT/folder"})
    assert patterns == [
        "EMIT/folder/info.json",
        "EMIT/folder/plumemask.tif",
    ]


def test_train_label_validation_reports_missing_authoritative_mask(
    tmp_path: Path,
) -> None:
    folder = tmp_path / "EMIT" / "folder"
    folder.mkdir(parents=True)
    (folder / "info.json").write_text("{}", encoding="utf-8")
    validation = validate_download(
        output_dir=tmp_path,
        expected_patterns=[
            "EMIT/folder/info.json",
            "EMIT/folder/plumemask.tif",
        ],
    )
    assert validation["missing_mask_files"] == [
        "EMIT/folder/plumemask.tif"
    ]


def test_mask_truth_and_georeferencing_come_from_authoritative_raster(
    tmp_path: Path,
) -> None:
    path = tmp_path / "sample" / "plumemask.tif"
    path.parent.mkdir()
    values = np.zeros((4, 4), dtype=np.uint8)
    values[1, 2] = 1
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=4,
        height=4,
        count=1,
        dtype="uint8",
        crs="EPSG:4326",
        transform=from_origin(10.0, 20.0, 0.1, 0.1),
    ) as target:
        target.write(values, 1)

    fact = read_mask_fact(sample_id="sample", mask_path=path, label_root=tmp_path)
    assert fact.label_state == "PLUME"
    assert fact.positive_pixels == 1
    assert fact.longitude == pytest.approx(10.2)
    assert fact.latitude == pytest.approx(19.8)


def test_geographic_grouping_uses_connected_25km_components() -> None:
    groups = geographic_group_ids(
        {
            "a": (0.0, 0.0),
            "b": (0.0, 0.20),
            "c": (0.0, 0.40),
            "far": (5.0, 5.0),
        },
        25.0,
    )
    assert groups["a"] == groups["b"] == groups["c"]
    assert groups["far"] != groups["a"]


def test_stage_b_groups_crops_from_the_same_hsi_granule() -> None:
    records = [
        {
            "sample_id": sample_id,
            "eligible_for_target_catalog": True,
            "sensor": "EMIT",
            "tile": "granule",
            "timestamp": "2024-01-01T00:00:00+00:00",
            "longitude": longitude,
            "latitude": 1.0,
        }
        for sample_id, longitude in (("a", 2.0), ("b", 2.1))
    ]
    groups = build_query_groups(records)
    assert len(groups) == 1
    assert [point.sample_id for point in groups[0].points] == ["a", "b"]

    many_records = [
        {
            **records[0],
            "sample_id": f"sample_{index:02d}",
            "longitude": 2.0 + index / 100,
        }
        for index in range(21)
    ]
    chunked = build_query_groups(many_records)
    assert [len(group.points) for group in chunked] == [20, 1]


def test_stage_b_bbox_handles_antimeridian() -> None:
    assert point_in_bbox(179.5, 0.0, [179.0, -1.0, -179.0, 1.0])
    assert point_in_bbox(-179.5, 0.0, [179.0, -1.0, -179.0, 1.0])
    assert not point_in_bbox(0.0, 0.0, [179.0, -1.0, -179.0, 1.0])


def test_stage_b_deduplicates_tiles_from_one_acquisition_and_limits_negatives() -> None:
    mask_records = [
        {
            "sample_id": "negative",
            "label_state": "NO_PLUME",
            "sensor": "EMIT",
            "tile": "hsi",
            "timestamp": "2024-01-01T00:00:00+00:00",
            "country": "X",
            "group_id": "g",
            "novel_beyond_all_mars_25km": True,
        }
    ]
    query_records = [
        {
            "products": [
                {
                    "id": "cloudy-tile",
                    "datetime": "2024-01-01T00:30:00Z",
                    "offset_hours": 0.5,
                    "cloud_cover": 80.0,
                    "covered_sample_ids": ["negative"],
                },
                {
                    "id": "clearer-tile",
                    "datetime": "2024-01-01T00:30:00Z",
                    "offset_hours": 0.5,
                    "cloud_cover": 10.0,
                    "covered_sample_ids": ["negative"],
                },
                {
                    "id": "too-late",
                    "datetime": "2024-01-01T02:00:00Z",
                    "offset_hours": 2.0,
                    "cloud_cover": 0.0,
                    "covered_sample_ids": ["negative"],
                },
            ]
        }
    ]
    pairs = _deduplicated_pairs(mask_records, query_records)
    assert len(pairs) == 1
    assert pairs[0]["target_product_id"] == "clearer-tile"
    assert pairs[0]["scene_supervision"] == "absence_high_confidence"


def test_stage_b_keeps_independent_target_sensors_at_the_same_time() -> None:
    mask_records = [
        {
            "sample_id": "positive",
            "label_state": "PLUME",
            "sensor": "EMIT",
            "tile": "hsi",
            "timestamp": "2024-01-01T00:00:00+00:00",
            "country": "X",
            "group_id": "g",
            "novel_beyond_all_mars_25km": True,
        }
    ]
    query_records = [
        {
            "target_sensor": target_sensor,
            "products": [
                {
                    "id": f"{target_sensor}-product",
                    "datetime": "2024-01-01T00:10:00Z",
                    "offset_hours": 1 / 6,
                    "cloud_cover": 0.0,
                    "covered_sample_ids": ["positive"],
                }
            ],
        }
        for target_sensor in ("sentinel2", "landsat")
    ]
    pairs = _deduplicated_pairs(mask_records, query_records)
    assert len(pairs) == 2
    assert {pair["target_sensor"] for pair in pairs} == {"sentinel2", "landsat"}
