from __future__ import annotations

import csv
import struct
import zipfile
from pathlib import Path

import pytest

from tools.audit_zhao_dsan_2025_dataset import (
    audit_archive,
    audit_mars_overlap,
    haversine_km,
    load_mars_sites,
    safe_zip_member,
)


def minimal_png_header(width: int = 430, height: int = 430) -> bytes:
    return (
        b"\x89PNG\r\n\x1a\n"
        + struct.pack(">I", 13)
        + b"IHDR"
        + struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
        + b"\x00\x00\x00\x00"
    )


def test_haversine_identity_and_one_degree() -> None:
    assert haversine_km(31.0, 5.0, 31.0, 5.0) == pytest.approx(0.0)
    assert haversine_km(0.0, 0.0, 1.0, 0.0) == pytest.approx(111.195, abs=0.001)


def test_safe_zip_member_rejects_traversal() -> None:
    with pytest.raises(ValueError, match="Unsafe ZIP member"):
        safe_zip_member("../escape.png")


def test_archive_counts_and_png_contract(tmp_path: Path) -> None:
    archive_path = tmp_path / "dataset.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("Dataset#1/plume-containing/D1_20200101.png", minimal_png_header())
        archive.writestr("Dataset#1/plume-free/D1_20200102.png", minimal_png_header())
    table = {
        "expected_total_rows": 2,
        "sites": [
            {
                "dataset_index": 1,
                "plume_containing": 1,
                "plume_free": 1,
            }
        ],
    }
    result = audit_archive(archive_path, table)
    assert result["image_members"] == 2
    assert result["dimensions"] == {"430x430": 2}
    assert result["same_site_same_date_label_conflict_count"] == 0
    assert result["dense_masks_present"] is False


def test_overlap_excludes_development_and_official_test() -> None:
    zhao_sites = [
        {
            "dataset_index": 1,
            "field": "A",
            "country": "X",
            "latitude": 10.0,
            "longitude": 10.0,
            "plume_containing": 1,
            "plume_free": 2,
        },
        {
            "dataset_index": 2,
            "field": "B",
            "country": "Y",
            "latitude": 40.0,
            "longitude": 40.0,
            "plume_containing": 2,
            "plume_free": 3,
        },
    ]
    mars_sites = [
        {
            "id_location": "development",
            "split_name": "train_2023",
            "latitude": 10.01,
            "longitude": 10.01,
        },
        {
            "id_location": "test",
            "split_name": "test_2023",
            "latitude": 10.02,
            "longitude": 10.02,
        },
    ]
    result = audit_mars_overlap(zhao_sites, mars_sites, 25.0)
    assert result["eligible_dataset_indices"] == [2]
    assert result["eligible_rows"] == 5
    assert result["sites"][0]["overlaps_development_within_25_km"] is True
    assert result["sites"][0]["overlaps_official_test_within_25_km"] is True


def test_mars_loader_reads_only_frozen_identity_geography_columns(tmp_path: Path) -> None:
    path = tmp_path / "mars.csv"
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(
            target,
            fieldnames=["id_location", "split_name", "lat", "lon", "isplume"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "id_location": "one",
                "split_name": "test_2023",
                "lat": 1.0,
                "lon": 2.0,
                "isplume": "secret",
            }
        )
    sites, counts = load_mars_sites(
        path, ["id_location", "split_name", "lat", "lon"]
    )
    assert sites == [
        {
            "id_location": "one",
            "split_name": "test_2023",
            "latitude": 1.0,
            "longitude": 2.0,
        }
    ]
    assert counts == {"test_2023": 1}
    assert "isplume" not in sites[0]
