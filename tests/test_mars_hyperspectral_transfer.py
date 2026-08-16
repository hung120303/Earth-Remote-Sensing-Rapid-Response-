from __future__ import annotations

import csv
from pathlib import Path

import pytest

from tools.acquire_mars_hyperspectral_metadata import safe_output_path
from tools.acquire_mars_hyperspectral_train_labels import allowed_patterns
from tools.audit_mars_hyperspectral_transfer import (
    haversine_km,
    parse_datetime,
    read_mars_observations,
)
from tools.query_mars_hyperspectral_cdse import summarize


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
