#!/usr/bin/env python3
"""Partition GeoTIFFs by whether they contain a known CAFO point.

CAFO CSV coordinates are interpreted as WGS84 longitude/latitude and projected
into each raster's CRS before testing its bounds.  Outputs are derived copies;
the input tree and CSV files are never modified.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import rasterio
from rasterio.warp import transform

LATITUDE_NAMES = {"lat", "latitude", "latdec", "lat_facili", "y"}
LONGITUDE_NAMES = {"lon", "long", "longitude", "londec", "lon_facili", "x"}


def _coordinate_column(fieldnames: list[str], candidates: set[str]) -> str | None:
    normalized = {column.lower().strip(): column for column in fieldnames}
    return next((normalized[name] for name in sorted(candidates) if name in normalized), None)


def load_cafo_points(csv_path: Path) -> list[tuple[float, float]]:
    """Read valid WGS84 ``(longitude, latitude)`` points from one CSV."""
    points: list[tuple[float, float]] = []
    with csv_path.open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        fieldnames = list(reader.fieldnames or [])
        latitude_column = _coordinate_column(fieldnames, LATITUDE_NAMES)
        longitude_column = _coordinate_column(fieldnames, LONGITUDE_NAMES)
        if latitude_column is None or longitude_column is None:
            raise ValueError(f"Could not detect latitude/longitude columns in {csv_path}")
        for row in reader:
            try:
                latitude = float(row[latitude_column])
                longitude = float(row[longitude_column])
            except (KeyError, TypeError, ValueError):
                continue
            if -90.0 <= latitude <= 90.0 and -180.0 <= longitude <= 180.0:
                points.append((longitude, latitude))
    return points


def contains_cafo(raster_path: Path, points: list[tuple[float, float]]) -> bool:
    """Return whether any WGS84 point falls inside the raster bounds."""
    if not points:
        return False
    with rasterio.open(raster_path) as source:
        if source.crs is None:
            raise ValueError(f"Raster has no CRS: {raster_path}")
        longitudes, latitudes = zip(*points)
        xs, ys = transform("EPSG:4326", source.crs, longitudes, latitudes)
        bounds = source.bounds
    return any(bounds.left <= x <= bounds.right and bounds.bottom <= y <= bounds.top for x, y in zip(xs, ys))


def partition(
    input_dir: Path,
    cafo_csv_dir: Path,
    positive_dir: Path,
    negative_dir: Path,
) -> dict[str, int]:
    csv_files = sorted(cafo_csv_dir.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CAFO CSV files found under {cafo_csv_dir}")
    points = [point for csv_path in csv_files for point in load_cafo_points(csv_path)]
    if not points:
        raise ValueError("CAFO CSV files contain no valid coordinates")

    rasters = sorted(input_dir.rglob("*.tif")) + sorted(input_dir.rglob("*.tiff"))
    if not rasters:
        raise FileNotFoundError(f"No GeoTIFFs found under {input_dir}")

    counts = {"rasters": len(rasters), "cafo": 0, "non_cafo": 0, "points": len(points)}
    for raster_path in rasters:
        is_positive = contains_cafo(raster_path, points)
        key = "cafo" if is_positive else "non_cafo"
        destination_root = positive_dir if is_positive else negative_dir
        relative = raster_path.relative_to(input_dir)
        destination = destination_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(raster_path, destination)
        counts[key] += 1
    return counts


def build_parser() -> argparse.ArgumentParser:
    data_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=data_root / "train_test_s2_0")
    parser.add_argument("--cafo-csv-dir", type=Path, default=data_root / "cafo_csv")
    parser.add_argument("--positive-dir", type=Path, default=data_root / "images_with_cafo")
    parser.add_argument("--negative-dir", type=Path, default=data_root / "images_non_cafo")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = partition(
        args.input_dir.resolve(),
        args.cafo_csv_dir.resolve(),
        args.positive_dir.resolve(),
        args.negative_dir.resolve(),
    )
    print(json.dumps({"ok": True, **result}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
