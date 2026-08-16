"""Run the preregistered metadata-only MARS-Hyperspectral transfer audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


REVISION = "74b3d3132d135fee1761df1dadb7d662a4b5245b"
SAFE_MARS_COLUMNS = {
    "id_location",
    "location_name",
    "country",
    "lon",
    "lat",
    "satellite",
    "tile",
    "tile_date",
    "split_name",
}
FORBIDDEN_MARS_COLUMNS = {"isplume", "plume", "ch4_fluxrate", "ch4_fluxrate_std"}


@dataclass(frozen=True)
class HsiSample:
    sample_id: str
    sensor: str
    published_split: str
    location_name: str
    country: str
    sector: str
    timestamp: datetime
    percentage_clear: float
    latitude: float | None
    longitude: float | None


@dataclass(frozen=True)
class MarsObservation:
    location_id: str
    location_name: str
    country: str
    latitude: float
    longitude: float
    sensor: str
    scene_id: str
    timestamp: datetime
    split_name: str


def parse_datetime(value: str) -> datetime:
    text = value.strip().replace("Z", "+00:00")
    return datetime.fromisoformat(text)


def optional_float(value: str | None) -> float | None:
    if value is None or not value.strip():
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_id_set(path: Path) -> set[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "id" not in reader.fieldnames:
            raise ValueError(f"Missing id column in {path}")
        return {row["id"].strip() for row in reader if row.get("id", "").strip()}


def read_hsi_samples(metadata_root: Path) -> list[HsiSample]:
    samples: list[HsiSample] = []
    emit_files = {
        "train": metadata_root / "EMIT/train_t_v4a.csv",
        "validation": metadata_root / "EMIT/val_t_v4a.csv",
        "test": metadata_root / "EMIT/test_t_v4a.csv",
    }
    for split, path in emit_files.items():
        samples.extend(read_hsi_csv(path, sensor="EMIT", split_by_id={split: None}))

    for sensor in ("EnMAP", "PRISMA"):
        train_ids = read_id_set(metadata_root / sensor / "train_s_v4a.csv")
        testval_ids = read_id_set(metadata_root / sensor / "testval_s_v4a.csv")
        overlap = train_ids & testval_ids
        if overlap:
            raise ValueError(f"{sensor} published train/testval ID overlap: {len(overlap)}")
        samples.extend(
            read_hsi_csv(
                metadata_root / sensor / "all_events.csv",
                sensor=sensor,
                split_by_id={"train": train_ids, "testval": testval_ids},
            )
        )
    sample_ids = [sample.sample_id for sample in samples]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Duplicate HSI sample IDs across source sensors")
    return samples


def read_hsi_csv(
    path: Path,
    *,
    sensor: str,
    split_by_id: dict[str, set[str] | None],
) -> list[HsiSample]:
    result: list[HsiSample] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "id",
            "location_name",
            "country",
            "sector",
            "tile_date",
            "percentage_clear",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing HSI columns in {path}: {sorted(missing)}")
        for row in reader:
            sample_id = row["id"].strip()
            matching = [
                split
                for split, ids in split_by_id.items()
                if ids is None or sample_id in ids
            ]
            if len(matching) > 1:
                raise ValueError(
                    f"Expected at most one published split for {sample_id}, got {matching}"
                )
            published_split = matching[0] if matching else "unassigned"
            result.append(
                HsiSample(
                    sample_id=sample_id,
                    sensor=sensor,
                    published_split=published_split,
                    location_name=row["location_name"].strip(),
                    country=row["country"].strip(),
                    sector=row["sector"].strip(),
                    timestamp=parse_datetime(row["tile_date"]),
                    percentage_clear=float(row["percentage_clear"]),
                    latitude=optional_float(row.get("location_lat")),
                    longitude=optional_float(row.get("location_lon")),
                )
            )
    return result


def read_mars_observations(path: Path) -> list[MarsObservation]:
    observations: list[MarsObservation] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        if not SAFE_MARS_COLUMNS <= fields:
            raise ValueError(f"Missing safe MARS columns: {sorted(SAFE_MARS_COLUMNS-fields)}")
        if not FORBIDDEN_MARS_COLUMNS <= fields:
            raise ValueError("Expected source schema no longer contains declared forbidden columns")
        for source_row in reader:
            row = {key: source_row[key] for key in SAFE_MARS_COLUMNS}
            latitude = optional_float(row["lat"])
            longitude = optional_float(row["lon"])
            if latitude is None or longitude is None:
                continue
            observations.append(
                MarsObservation(
                    location_id=row["id_location"].strip(),
                    location_name=row["location_name"].strip(),
                    country=row["country"].strip(),
                    latitude=latitude,
                    longitude=longitude,
                    sensor=row["satellite"].strip(),
                    scene_id=row["tile"].strip(),
                    timestamp=parse_datetime(row["tile_date"]),
                    split_name=row["split_name"].strip(),
                )
            )
    return observations


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_km = 6371.0088
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = phi2 - phi1
    dlambda = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    )
    return 2.0 * radius_km * math.asin(math.sqrt(a))


def representative_locations(
    observations: Iterable[MarsObservation],
) -> dict[str, tuple[float, float]]:
    buckets: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in observations:
        buckets[row.location_name].append((row.latitude, row.longitude))
    result: dict[str, tuple[float, float]] = {}
    for name, coords in buckets.items():
        result[name] = (
            sum(value[0] for value in coords) / len(coords),
            sum(value[1] for value in coords) / len(coords),
        )
    return result


def nearest_distance_km(
    latitude: float,
    longitude: float,
    candidates: Iterable[tuple[float, float]],
) -> float:
    return min(
        (haversine_km(latitude, longitude, lat, lon) for lat, lon in candidates),
        default=math.inf,
    )


def audit(
    *,
    metadata_root: Path,
    mars_manifest: Path,
    protocol_path: Path,
    catalog_summary_path: Path | None = None,
) -> dict[str, object]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol["source"]["revision"] != REVISION:
        raise ValueError("Protocol revision differs from audit implementation")
    acquisition = json.loads(
        (metadata_root / "metadata_manifest.json").read_text(encoding="utf-8")
    )
    if acquisition["revision"] != REVISION:
        raise ValueError("Acquired metadata revision differs from frozen revision")

    samples = read_hsi_samples(metadata_root)
    mars = read_mars_observations(mars_manifest)
    mars_by_name: dict[str, list[MarsObservation]] = defaultdict(list)
    for observation in mars:
        mars_by_name[observation.location_name].append(observation)
    mars_locations = representative_locations(mars)
    protected_names = {
        row.location_name for row in mars if row.split_name.lower().startswith("test")
    }
    protected_coords = [mars_locations[name] for name in protected_names]

    resolved: list[tuple[HsiSample, float, float, str]] = []
    resolved_by_sample: dict[str, tuple[float, float]] = {}
    unresolved = 0
    for sample in samples:
        if sample.latitude is not None and sample.longitude is not None:
            resolved.append((sample, sample.latitude, sample.longitude, "source_csv"))
            resolved_by_sample[sample.sample_id] = (sample.latitude, sample.longitude)
        elif sample.location_name in mars_locations:
            lat, lon = mars_locations[sample.location_name]
            resolved.append((sample, lat, lon, "exact_mars_location_identity"))
            resolved_by_sample[sample.sample_id] = (lat, lon)
        else:
            unresolved += 1

    overlap_by_identity = sum(sample.location_name in protected_names for sample in samples)
    protected_resolved = 0
    protected_sample_ids: set[str] = set()
    nonprotected_location_keys: set[tuple[str, str]] = set()
    for sample, lat, lon, _method in resolved:
        protected = sample.location_name in protected_names or nearest_distance_km(
            lat, lon, protected_coords
        ) <= float(protocol["stage_b_label_and_catalog_audit"]["filters"]["mars_protected_exclusion_radius_km"])
        if protected:
            protected_resolved += 1
            protected_sample_ids.add(sample.sample_id)
        else:
            nonprotected_location_keys.add((sample.sensor, sample.location_name))

    offsets = (15.0 / 60.0, 1.0, 6.0)
    matched_sample_ids: dict[float, set[str]] = {threshold: set() for threshold in offsets}
    matched_pairs: dict[float, int] = {threshold: 0 for threshold in offsets}
    minimum_clear = float(
        protocol["stage_b_label_and_catalog_audit"]["filters"]
        ["minimum_hyperspectral_percentage_clear"]
    )
    eligible_samples = [
        sample
        for sample in samples
        if sample.published_split == "train"
        and sample.percentage_clear >= minimum_clear
        and sample.sample_id in resolved_by_sample
        and sample.sample_id not in protected_sample_ids
    ]
    for sample in eligible_samples:
        for observation in mars_by_name.get(sample.location_name, []):
            delta_hours = abs((sample.timestamp - observation.timestamp).total_seconds()) / 3600.0
            for threshold in offsets:
                if delta_hours <= threshold:
                    matched_sample_ids[threshold].add(sample.sample_id)
                    matched_pairs[threshold] += 1

    stage = protocol["stage_a_metadata_audit"]
    catalog_matches: dict[str, object] | None = None
    if catalog_summary_path is not None and catalog_summary_path.exists():
        catalog_matches = json.loads(catalog_summary_path.read_text(encoding="utf-8"))
        if catalog_matches.get("scope") != "metadata_only_no_target_assets":
            raise ValueError("Unexpected target catalog summary scope")
        if int(catalog_matches.get("eligible_hsi_samples", -1)) != len(eligible_samples):
            raise ValueError("Target catalog eligible-sample denominator mismatch")
    metrics = {
        "samples": len(samples),
        "unique_location_names": len({sample.location_name for sample in samples}),
        "countries": len({sample.country for sample in samples if sample.country}),
        "sectors": dict(sorted(Counter(sample.sector for sample in samples).items())),
        "by_sensor": dict(sorted(Counter(sample.sensor for sample in samples).items())),
        "by_published_split": dict(
            sorted(Counter(sample.published_split for sample in samples).items())
        ),
        "by_sensor_and_split": {
            f"{sensor}:{split}": count
            for (sensor, split), count in sorted(
                Counter((sample.sensor, sample.published_split) for sample in samples).items()
            )
        },
        "coordinate_resolved_samples": len(resolved),
        "coordinate_unresolved_samples": unresolved,
        "coordinate_resolved_non_mars_test_locations": len(nonprotected_location_keys),
        "eligible_train_samples_after_clear_and_mars_test_exclusion": len(
            eligible_samples
        ),
        "mars_test_overlap_samples_by_exact_identity": overlap_by_identity,
        "mars_test_overlap_samples_after_25km_check": protected_resolved,
        "existing_target_matches": {
            "within_15_minutes": {
                "hsi_samples": len(matched_sample_ids[offsets[0]]),
                "pairs": matched_pairs[offsets[0]],
            },
            "within_1_hour": {
                "hsi_samples": len(matched_sample_ids[offsets[1]]),
                "pairs": matched_pairs[offsets[1]],
            },
            "within_6_hours": {
                "hsi_samples": len(matched_sample_ids[offsets[2]]),
                "pairs": matched_pairs[offsets[2]],
            },
        },
        "catalog_target_matches": catalog_matches,
    }
    eligible_match_count = max(
        int(metrics["existing_target_matches"]["within_6_hours"]["hsi_samples"]),
        int(catalog_matches["within_6_hours"]) if catalog_matches else 0,
    )
    gates = {
        "minimum_total_samples": metrics["samples"]
        >= int(stage["gates"]["minimum_total_samples"]),
        "minimum_countries": metrics["countries"]
        >= int(stage["gates"]["minimum_countries"]),
        "minimum_coordinate_resolved_non_mars_test_locations": metrics[
            "coordinate_resolved_non_mars_test_locations"
        ]
        >= int(stage["gates"]["minimum_coordinate_resolved_non_mars_test_locations"]),
        "minimum_existing_or_catalog_query_candidates_within_6_hours": eligible_match_count
        >= int(
            stage["gates"][
                "minimum_existing_or_catalog_query_candidates_within_6_hours"
            ]
        ),
    }
    return {
        "schema_version": 1,
        "protocol": str(protocol_path).replace("\\", "/"),
        "protocol_sha256": sha256_file(protocol_path),
        "metadata_manifest": {
            "path": str(metadata_root / "metadata_manifest.json").replace("\\", "/"),
            "sha256": sha256_file(metadata_root / "metadata_manifest.json"),
            "files": len(acquisition["files"]),
            "bytes": int(acquisition["total_bytes"]),
        },
        "catalog_summary": (
            {
                "path": str(catalog_summary_path).replace("\\", "/"),
                "sha256": sha256_file(catalog_summary_path),
            }
            if catalog_matches is not None and catalog_summary_path is not None
            else None
        ),
        "source_revision": REVISION,
        "scope": "metadata_only_no_rasters_no_hsi_mask_truth_no_mars_outcomes",
        "metrics": metrics,
        "gates": gates,
        "pass": all(gates.values()),
        "interpretation": (
            "PASS authorizes only Stage B label/georeference and target-catalog audit. "
            "It does not authorize model training or establish transferability."
        ),
    }


def write_markdown(report: dict[str, object], path: Path) -> None:
    metrics = report["metrics"]
    matches = metrics["existing_target_matches"]
    catalog = metrics.get("catalog_target_matches")
    lines = [
        "# MARS-Hyperspectral transfer Stage A",
        "",
        f"**Decision:** {'PASS' if report['pass'] else 'FAIL'}",
        "",
        "This was a metadata-only audit. No hyperspectral raster, mask truth, MARS model score, or protected MARS outcome was read.",
        "",
        "## Coverage",
        "",
        f"- Samples: {metrics['samples']:,}",
        f"- Unique location names: {metrics['unique_location_names']:,}",
        f"- Countries: {metrics['countries']:,}",
        f"- Coordinate-resolved samples: {metrics['coordinate_resolved_samples']:,}",
        f"- Coordinate-unresolved samples: {metrics['coordinate_unresolved_samples']:,}",
        f"- Resolved non-protected locations: {metrics['coordinate_resolved_non_mars_test_locations']:,}",
        f"- Eligible train samples after clear/protected-site filters: {metrics['eligible_train_samples_after_clear_and_mars_test_exclusion']:,}",
        f"- Exact MARS official-test identity overlaps: {metrics['mars_test_overlap_samples_by_exact_identity']:,}",
        f"- Resolved overlaps after 25 km exclusion: {metrics['mars_test_overlap_samples_after_25km_check']:,}",
        "",
        "## Existing local target acquisitions",
        "",
        f"- Within 15 minutes: {matches['within_15_minutes']['hsi_samples']:,} HSI samples / {matches['within_15_minutes']['pairs']:,} pairs",
        f"- Within 1 hour: {matches['within_1_hour']['hsi_samples']:,} HSI samples / {matches['within_1_hour']['pairs']:,} pairs",
        f"- Within 6 hours: {matches['within_6_hours']['hsi_samples']:,} HSI samples / {matches['within_6_hours']['pairs']:,} pairs",
    ]
    if catalog:
        lines.extend(
            [
                "",
                "## Public Copernicus catalog",
                "",
                f"- Eligible HSI training observations queried: {catalog['eligible_hsi_samples']:,}",
                f"- Sentinel-2 L1C candidates within 15 minutes: {catalog['within_15_minutes']:,}",
                f"- Sentinel-2 L1C candidates within 1 hour: {catalog['within_1_hour']:,}",
                f"- Sentinel-2 L1C candidates within 6 hours: {catalog['within_6_hours']:,}",
                f"- Unique Sentinel-2 L1C products: {catalog['unique_sentinel_products']:,}",
            ]
        )
    lines.extend(
        [
            "",
            "## Frozen gates",
            "",
        ]
    )
    lines.extend(
        f"- {'PASS' if passed else 'FAIL'} `{name}`"
        for name, passed in report["gates"].items()
    )
    lines.extend(["", "## Claim boundary", "", report["interpretation"], ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metadata-root",
        type=Path,
        default=Path(".research/mars_hyperspectral_transfer"),
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
        "--catalog-summary",
        type=Path,
        default=Path(
            ".research/mars_hyperspectral_transfer/cdse_s2_l1c_summary.json"
        ),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("reports/acquisition/mars_hyperspectral_transfer_stage_a.json"),
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=Path("reports/acquisition/MARS_HYPERSPECTRAL_TRANSFER_STAGE_A.md"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = audit(
        metadata_root=args.metadata_root,
        mars_manifest=args.mars_manifest,
        protocol_path=args.protocol,
        catalog_summary_path=args.catalog_summary,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_markdown(report, args.output_markdown)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
