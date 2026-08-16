"""Audit train-only MARS-Hyperspectral masks and recover crop georeferencing.

The authoritative scene label is derived only from ``plumemask.tif`` pixels.
Source CSV plume flags and flux fields are never read by this module.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable

import numpy as np
import rasterio
from rasterio.warp import transform as transform_points

from tools.acquire_mars_hyperspectral_train_labels import read_train_folders
from tools.audit_mars_hyperspectral_transfer import (
    HsiSample,
    MarsObservation,
    haversine_km,
    nearest_distance_km,
    read_hsi_samples,
    read_mars_observations,
    representative_locations,
)


class UnionFind:
    def __init__(self, values: Iterable[str]) -> None:
        self.parent = {value: value for value in values}
        self.rank = {value: 0 for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


@dataclass(frozen=True)
class MaskFact:
    sample_id: str
    relative_path: str
    crs: str
    width: int
    height: int
    longitude: float
    latitude: float
    positive_pixels: int
    total_pixels: int
    minimum_value: float
    maximum_value: float
    distinct_values: tuple[float, ...]

    @property
    def label_state(self) -> str:
        return "PLUME" if self.positive_pixels else "NO_PLUME"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mask_center_wgs84(source: rasterio.io.DatasetReader) -> tuple[float, float]:
    if source.crs is None:
        raise ValueError(f"Mask has no CRS: {source.name}")
    center_x, center_y = source.xy((source.height - 1) / 2, (source.width - 1) / 2)
    longitudes, latitudes = transform_points(
        source.crs, "EPSG:4326", [center_x], [center_y]
    )
    longitude, latitude = float(longitudes[0]), float(latitudes[0])
    if not (-180.0 <= longitude <= 180.0 and -90.0 <= latitude <= 90.0):
        raise ValueError(f"Invalid WGS84 mask center: {(longitude, latitude)}")
    return longitude, latitude


def read_mask_fact(
    *, sample_id: str, mask_path: Path, label_root: Path
) -> MaskFact:
    with rasterio.open(mask_path) as source:
        if source.count != 1:
            raise ValueError(f"Expected one mask band for {sample_id}, got {source.count}")
        values = source.read(1)
        if values.size == 0 or not np.isfinite(values).all():
            raise ValueError(f"Mask is empty or non-finite: {sample_id}")
        minimum = float(values.min())
        maximum = float(values.max())
        if minimum < 0:
            raise ValueError(f"Mask contains negative values: {sample_id}")
        unique = np.unique(values)
        if len(unique) > 16:
            raise ValueError(
                f"Mask has {len(unique)} values and is not label-like: {sample_id}"
            )
        longitude, latitude = mask_center_wgs84(source)
        return MaskFact(
            sample_id=sample_id,
            relative_path=mask_path.relative_to(label_root).as_posix(),
            crs=str(source.crs),
            width=int(source.width),
            height=int(source.height),
            longitude=longitude,
            latitude=latitude,
            positive_pixels=int(np.count_nonzero(values > 0)),
            total_pixels=int(values.size),
            minimum_value=minimum,
            maximum_value=maximum,
            distinct_values=tuple(float(value) for value in unique),
        )


def geographic_group_ids(
    coordinates: dict[str, tuple[float, float]], radius_km: float
) -> dict[str, str]:
    """Assign exact connected components with a latitude/longitude spatial index."""

    identifiers = sorted(coordinates)
    groups = UnionFind(identifiers)
    cell_degrees = radius_km / 111.195
    longitude_cell_count = math.ceil(360.0 / cell_degrees)
    cells: dict[tuple[int, int], list[str]] = defaultdict(list)
    for identifier in identifiers:
        latitude, longitude = coordinates[identifier]
        lat_cell = math.floor((latitude + 90.0) / cell_degrees)
        lon_cell = math.floor((longitude + 180.0) / cell_degrees) % longitude_cell_count
        latitude_span = 1
        longitude_span = max(
            1,
            math.ceil(1.0 / max(abs(math.cos(math.radians(latitude))), 0.05)),
        )
        for delta_lat in range(-latitude_span, latitude_span + 1):
            for delta_lon in range(-longitude_span, longitude_span + 1):
                neighbor_lon = (lon_cell + delta_lon) % longitude_cell_count
                for other in cells.get((lat_cell + delta_lat, neighbor_lon), []):
                    if haversine_km(
                        latitude,
                        longitude,
                        coordinates[other][0],
                        coordinates[other][1],
                    ) <= radius_km:
                        groups.union(identifier, other)
        cells[(lat_cell, lon_cell)].append(identifier)

    members: dict[str, list[str]] = defaultdict(list)
    for identifier in identifiers:
        members[groups.find(identifier)].append(identifier)
    result: dict[str, str] = {}
    for component in members.values():
        ordered = sorted(component)
        identity = hashlib.sha256("\0".join(ordered).encode("utf-8")).hexdigest()[:16]
        for identifier in ordered:
            result[identifier] = f"geo25_{identity}"
    return result


def _protected_coordinates(mars: Iterable[MarsObservation]) -> tuple[set[str], list[tuple[float, float]]]:
    observations = list(mars)
    locations = representative_locations(observations)
    names = {
        row.location_name
        for row in observations
        if row.split_name.lower().startswith("test")
    }
    return names, [locations[name] for name in sorted(names)]


def audit_train_masks(
    *,
    metadata_root: Path,
    label_root: Path,
    mars_manifest: Path,
    protocol_path: Path,
    stage_a_report_path: Path,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    stage_a = json.loads(stage_a_report_path.read_text(encoding="utf-8"))
    if not stage_a.get("pass"):
        raise ValueError("Stage A did not pass; train-mask audit is not authorized")
    filters = protocol["stage_b_label_and_catalog_audit"]["filters"]
    minimum_clear = float(filters["minimum_hyperspectral_percentage_clear"])
    exclusion_km = float(filters["mars_protected_exclusion_radius_km"])
    group_radius_km = float(filters["group_radius_km"])

    samples = {
        sample.sample_id: sample
        for sample in read_hsi_samples(metadata_root)
        if sample.published_split == "train"
    }
    train_folders = read_train_folders(metadata_root)
    if set(samples) != set(train_folders):
        raise ValueError("Train CSV identity mismatch between metadata and folder manifests")

    facts: dict[str, MaskFact] = {}
    missing_masks: list[str] = []
    for sample_id, relative_folder in sorted(train_folders.items()):
        mask_path = label_root.joinpath(
            *PurePosixPath(relative_folder).parts, "plumemask.tif"
        )
        if not mask_path.exists():
            missing_masks.append(sample_id)
            continue
        facts[sample_id] = read_mask_fact(
            sample_id=sample_id, mask_path=mask_path, label_root=label_root
        )
    if missing_masks:
        raise ValueError(
            f"Authoritative train masks are incomplete: {len(missing_masks)} missing"
        )

    mars = read_mars_observations(mars_manifest)
    mars_locations = representative_locations(mars)
    all_mars_coords = list(mars_locations.values())
    protected_names, protected_coords = _protected_coordinates(mars)

    eligible_coordinates: dict[str, tuple[float, float]] = {}
    records: list[dict[str, object]] = []
    csv_coordinate_offsets_km: list[float] = []
    for sample_id in sorted(samples):
        sample: HsiSample = samples[sample_id]
        fact = facts[sample_id]
        nearest_mars = nearest_distance_km(
            fact.latitude, fact.longitude, all_mars_coords
        )
        protected_distance = nearest_distance_km(
            fact.latitude, fact.longitude, protected_coords
        )
        protected = (
            sample.location_name in protected_names
            or protected_distance <= exclusion_km
        )
        eligible = sample.percentage_clear >= minimum_clear and not protected
        if eligible:
            eligible_coordinates[sample_id] = (fact.latitude, fact.longitude)
        if sample.latitude is not None and sample.longitude is not None:
            csv_coordinate_offsets_km.append(
                haversine_km(
                    fact.latitude,
                    fact.longitude,
                    sample.latitude,
                    sample.longitude,
                )
            )
        record = {
            **asdict(fact),
            "distinct_values": list(fact.distinct_values),
            "label_state": fact.label_state,
            "sensor": sample.sensor,
            "tile": sample.tile,
            "published_split": sample.published_split,
            "timestamp": sample.timestamp.isoformat(),
            "location_name": sample.location_name,
            "country": sample.country,
            "sector": sample.sector,
            "percentage_clear": sample.percentage_clear,
            "coordinate_source": "authoritative_mask_georeferencing",
            "mars_test_protected": protected,
            "nearest_mars_test_km": protected_distance,
            "nearest_any_mars_km": nearest_mars,
            "novel_beyond_all_mars_25km": nearest_mars > group_radius_km,
            "eligible_for_target_catalog": eligible,
        }
        records.append(record)

    group_by_sample = geographic_group_ids(eligible_coordinates, group_radius_km)
    for record in records:
        record["group_id"] = group_by_sample.get(str(record["sample_id"]))

    eligible_records = [
        record for record in records if record["eligible_for_target_catalog"]
    ]
    novel_groups = {
        record["group_id"]
        for record in eligible_records
        if record["novel_beyond_all_mars_25km"]
    }
    label_counts = Counter(str(record["label_state"]) for record in records)
    eligible_label_counts = Counter(
        str(record["label_state"]) for record in eligible_records
    )
    csv_coordinate_offsets_km.sort()
    summary: dict[str, object] = {
        "schema_version": 1,
        "scope": "published_train_masks_only_no_retrievals_no_validation_or_test_labels",
        "label_contract": "plumemask_positive_pixel_presence_only",
        "source_revision": protocol["source"]["revision"],
        "source_license": protocol["source"]["license"],
        "train_samples": len(records),
        "mask_files": len(facts),
        "label_counts": dict(sorted(label_counts.items())),
        "eligible_target_catalog_samples": len(eligible_records),
        "eligible_label_counts": dict(sorted(eligible_label_counts.items())),
        "protected_samples": sum(bool(record["mars_test_protected"]) for record in records),
        "eligible_geographic_groups_25km": len(set(group_by_sample.values())),
        "eligible_novel_groups_beyond_all_mars_25km": len(novel_groups),
        "eligible_countries": len(
            {str(record["country"]) for record in eligible_records if record["country"]}
        ),
        "csv_vs_mask_center_offset_km": {
            "count": len(csv_coordinate_offsets_km),
            "median": (
                csv_coordinate_offsets_km[len(csv_coordinate_offsets_km) // 2]
                if csv_coordinate_offsets_km
                else None
            ),
            "maximum": max(csv_coordinate_offsets_km, default=None),
        },
        "inputs": {
            "protocol": {
                "path": protocol_path.as_posix(),
                "sha256": sha256_file(protocol_path),
            },
            "stage_a_report": {
                "path": stage_a_report_path.as_posix(),
                "sha256": sha256_file(stage_a_report_path),
            },
            "train_label_manifest": {
                "path": (label_root / "train_label_manifest.json").as_posix(),
                "sha256": (
                    sha256_file(label_root / "train_label_manifest.json")
                    if (label_root / "train_label_manifest.json").exists()
                    else None
                ),
            },
        },
        "claim_boundary": (
            "This audit establishes train-mask truth and georeferencing only. "
            "It does not establish a valid target-sensor pair or authorize modeling."
        ),
    }
    value_patterns = Counter(
        tuple(float(value) for value in record["distinct_values"])
        for record in records
    )
    summary["mask_value_patterns"] = {
        str(pattern): count for pattern, count in sorted(value_patterns.items())
    }
    return records, summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metadata-root",
        type=Path,
        default=Path(".research/mars_hyperspectral_transfer"),
    )
    parser.add_argument(
        "--label-root",
        type=Path,
        default=Path(".research/mars_hyperspectral_transfer/train_labels"),
    )
    parser.add_argument(
        "--mars-manifest",
        type=Path,
        default=Path(
            ".research/source_audit_20260715/mars_s2l_current_validated_images_all.csv"
        ),
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("configs/mars_hyperspectral_transfer_acquisition_protocol.json"),
    )
    parser.add_argument(
        "--stage-a-report",
        type=Path,
        default=Path("reports/acquisition/mars_hyperspectral_transfer_stage_a.json"),
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=Path(
            ".research/mars_hyperspectral_transfer/train_mask_catalog.jsonl"
        ),
    )
    parser.add_argument(
        "--output-summary",
        type=Path,
        default=Path(
            ".research/mars_hyperspectral_transfer/train_mask_summary.json"
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    records, summary = audit_train_masks(
        metadata_root=args.metadata_root,
        label_root=args.label_root,
        mars_manifest=args.mars_manifest,
        protocol_path=args.protocol,
        stage_a_report_path=args.stage_a_report,
    )
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.output_jsonl.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    args.output_summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
