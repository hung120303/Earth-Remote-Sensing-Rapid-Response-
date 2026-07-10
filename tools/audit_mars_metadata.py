#!/usr/bin/env python3
"""Audit pinned MARS-S2L metadata without downloading the image corpus."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from acquire_mars_metadata import (
    DEFAULT_OUTPUT,
    MANIFEST_NAME,
    REPO_ID,
    REVISION,
    checked_output_dir,
    repo_root,
    sha256,
    verify_files,
)

DEFAULT_JSON = Path("reports/acquisition/mars_s2l_metadata_audit.json")
DEFAULT_MARKDOWN = Path("reports/acquisition/MARS_S2L_METADATA_AUDIT.md")
OFFICIAL_SPLITS = ("train", "val", "test")
CSV_FIELD_LIMIT = sys.maxsize
GEOGRAPHIC_GROUP_RADIUS_KM = 25.0
RECOMMENDED_MIN_CLEAR_PCT = 80.0
REQUIRED_IMAGE_COLUMNS = {
    "s2path",
    "plumepath",
    "cloudmaskpath",
    "ch4path",
    "percentage_clear",
    "tile",
    "isplume",
    "satellite",
    "tile_date",
    "id_location",
    "location_name",
    "country",
    "lon",
    "lat",
    "observability",
    "background_image_tile",
    "crs",
    "width",
    "height",
    "plume",
    "split_name",
    "case_study",
    "id_loc_image",
}
REQUIRED_PLUME_COLUMNS = {
    "id_plume",
    "id_loc_image",
    "id_source",
    "tile",
    "tile_date",
    "satellite",
    "validated",
    "geometry",
    "detection_institution",
}


def bool_value(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"Expected boolean text, got {value!r}")


def numeric(value: str, *, default: float | None = None) -> float | None:
    value = value.strip()
    if not value:
        return default
    try:
        result = float(value)
    except ValueError as exc:
        raise ValueError(f"Expected numeric text, got {value!r}") from exc
    return result if math.isfinite(result) else default


def product_domain(row: dict[str, str]) -> str:
    tile = row["tile"]
    satellite = row["satellite"]
    if satellite.startswith("S2") and "MSIL1C" in tile:
        return "Sentinel-2_MSI_L1C"
    if satellite.startswith("LC") and "_L1TP_" in tile:
        return "Landsat_Collection2_L1TP"
    if satellite.startswith("LC") and "_L1GT_" in tile:
        return "Landsat_Collection2_L1GT"
    return f"other:{satellite}"


def quantiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "p10": None, "median": None, "p90": None, "max": None, "mean": None}
    ordered = sorted(values)

    def at(fraction: float) -> float:
        index = fraction * (len(ordered) - 1)
        lower = int(math.floor(index))
        upper = int(math.ceil(index))
        if lower == upper:
            return ordered[lower]
        weight = index - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    return {
        "min": round(ordered[0], 6),
        "p10": round(at(0.10), 6),
        "median": round(at(0.50), 6),
        "p90": round(at(0.90), 6),
        "max": round(ordered[-1], 6),
        "mean": round(statistics.fmean(ordered), 6),
    }


def counter_dict(counter: Counter[str]) -> dict[str, int]:
    return {key: counter[key] for key in sorted(counter)}


def open_csv(path: Path) -> tuple[csv.DictReader, Any]:
    source = path.open("r", encoding="utf-8", newline="")
    reader = csv.DictReader(source)
    return reader, source


def require_columns(path: Path, observed: Iterable[str] | None, required: set[str]) -> None:
    fields = set(observed or [])
    missing = sorted(required - fields)
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {missing}")


def audit_image_csv(path: Path, expected_split: str | None) -> tuple[dict[str, Any], dict[str, Any]]:
    csv.field_size_limit(CSV_FIELD_LIMIT)
    reader, source = open_csv(path)
    require_columns(path, reader.fieldnames, REQUIRED_IMAGE_COLUMNS)
    counts: Counter[str] = Counter()
    satellites: Counter[str] = Counter()
    products: Counter[str] = Counter()
    observability: Counter[str] = Counter()
    countries: Counter[str] = Counter()
    cases: Counter[str] = Counter()
    dimensions: Counter[str] = Counter()
    split_names: Counter[str] = Counter()
    clear_values: list[float] = []
    ids: set[str] = set()
    locations: set[str] = set()
    scenes: set[str] = set()
    positive_ids: set[str] = set()
    location_rows: Counter[str] = Counter()
    location_positive: Counter[str] = Counter()
    location_coordinates: dict[str, list[tuple[float, float]]] = defaultdict(list)
    rows_by_location: dict[str, Counter[str]] = defaultdict(Counter)

    try:
        for row_number, row in enumerate(reader, start=2):
            counts["rows"] += 1
            item_id = row["id_loc_image"].strip()
            if not item_id:
                counts["missing_item_id"] += 1
            elif item_id in ids:
                counts["duplicate_item_id"] += 1
            ids.add(item_id)

            is_plume = bool_value(row["isplume"])
            label = "positive" if is_plume else "negative"
            counts[label] += 1
            if is_plume:
                positive_ids.add(item_id)

            plume_path_present = bool(row["plumepath"].strip())
            ch4_path_present = bool(row["ch4path"].strip())
            plume_geometry_present = row["plume"].strip() not in {"", "MULTIPOLYGON EMPTY"}
            if is_plume and not plume_path_present:
                counts["positive_missing_plume_path"] += 1
            if is_plume and not ch4_path_present:
                counts["positive_missing_ch4_path"] += 1
            if is_plume and not plume_geometry_present:
                counts["positive_missing_plume_geometry"] += 1
            if not is_plume and (plume_path_present or ch4_path_present or plume_geometry_present):
                counts["negative_with_positive_asset"] += 1

            satellite = row["satellite"].strip() or "<missing>"
            domain = product_domain(row)
            observed = row["observability"].strip() or "<missing>"
            satellites[satellite] += 1
            products[domain] += 1
            observability[observed] += 1
            countries[row["country"].strip() or "<missing>"] += 1
            cases[row["case_study"].strip() or "<missing>"] += 1
            split_value = row["split_name"].strip() or "<missing>"
            split_names[split_value] += 1
            if expected_split is not None and split_value != f"{expected_split}_2023":
                counts["split_name_mismatch"] += 1

            width = row["width"].strip() or "?"
            height = row["height"].strip() or "?"
            dimensions[f"{width}x{height}"] += 1
            scene = row["tile"].strip()
            scenes.add(scene)
            location = row["id_location"].strip()
            locations.add(location)
            location_rows[location] += 1
            location_positive[location] += int(is_plume)
            rows_by_location[location][label] += 1
            latitude = numeric(row["lat"])
            longitude = numeric(row["lon"])
            if latitude is not None and longitude is not None:
                location_coordinates[location].append((latitude, longitude))
            else:
                counts["missing_coordinates"] += 1

            clear_pct = numeric(row["percentage_clear"])
            if clear_pct is not None:
                clear_values.append(clear_pct)
            else:
                counts["missing_clear_percentage"] += 1
            background_present = bool(row["background_image_tile"].strip())
            counts["background_present" if background_present else "background_missing"] += 1
            s2 = satellite.startswith("S2") and domain == "Sentinel-2_MSI_L1C"
            if s2:
                counts["sentinel2_rows"] += 1
                counts[f"sentinel2_{label}"] += 1
            recommended = (
                s2
                and observed == "clear"
                and clear_pct is not None
                and clear_pct >= RECOMMENDED_MIN_CLEAR_PCT
                and background_present
            )
            if recommended:
                counts["recommended_s2_rows"] += 1
                counts[f"recommended_s2_{label}"] += 1
                rows_by_location[location][f"recommended_{label}"] += 1
    except Exception as exc:
        raise ValueError(f"Failed to audit {path.name} at row {row_number}: {exc}") from exc
    finally:
        source.close()

    coordinates = {
        location: (
            statistics.fmean(point[0] for point in points),
            statistics.fmean(point[1] for point in points),
        )
        for location, points in location_coordinates.items()
        if points
    }
    summary = {
        "file": path.name,
        "counts": counter_dict(counts),
        "positive_fraction": round(counts["positive"] / counts["rows"], 8) if counts["rows"] else None,
        "unique_item_ids": len(ids),
        "unique_locations": len(locations),
        "unique_scenes": len(scenes),
        "satellites": counter_dict(satellites),
        "product_domains": counter_dict(products),
        "observability": counter_dict(observability),
        "dimensions": counter_dict(dimensions),
        "split_names": counter_dict(split_names),
        "clear_percentage": quantiles(clear_values),
        "countries": counter_dict(countries),
        "case_studies": counter_dict(cases),
    }
    internal = {
        "ids": ids,
        "positive_ids": positive_ids,
        "locations": locations,
        "scenes": scenes,
        "coordinates": coordinates,
        "rows_by_location": rows_by_location,
        "location_rows": location_rows,
        "location_positive": location_positive,
    }
    return summary, internal


def audit_plume_csv(path: Path) -> tuple[dict[str, Any], set[str]]:
    csv.field_size_limit(CSV_FIELD_LIMIT)
    reader, source = open_csv(path)
    require_columns(path, reader.fieldnames, REQUIRED_PLUME_COLUMNS)
    counts: Counter[str] = Counter()
    satellites: Counter[str] = Counter()
    institutions: Counter[str] = Counter()
    plume_ids: set[str] = set()
    image_ids: set[str] = set()
    source_ids: set[str] = set()
    try:
        for row_number, row in enumerate(reader, start=2):
            counts["rows"] += 1
            plume_id = row["id_plume"].strip()
            if plume_id in plume_ids:
                counts["duplicate_plume_id"] += 1
            plume_ids.add(plume_id)
            image_ids.add(row["id_loc_image"].strip())
            source_ids.add(row["id_source"].strip())
            satellites[row["satellite"].strip() or "<missing>"] += 1
            institutions[row["detection_institution"].strip() or "<missing>"] += 1
            if not bool_value(row["validated"]):
                counts["not_validated"] += 1
            if row["geometry"].strip() in {"", "MULTIPOLYGON EMPTY"}:
                counts["missing_geometry"] += 1
    except Exception as exc:
        raise ValueError(f"Failed to audit {path.name} at row {row_number}: {exc}") from exc
    finally:
        source.close()
    return (
        {
            "file": path.name,
            "counts": counter_dict(counts),
            "unique_plume_ids": len(plume_ids),
            "unique_image_ids": len(image_ids),
            "unique_source_ids": len(source_ids),
            "satellites": counter_dict(satellites),
            "detection_institutions": counter_dict(institutions),
        },
        image_ids,
    )


def haversine_km(first: tuple[float, float], second: tuple[float, float]) -> float:
    lat1, lon1 = map(math.radians, first)
    lat2, lon2 = map(math.radians, second)
    delta_lat = lat2 - lat1
    delta_lon = lon2 - lon1
    value = (
        math.sin(delta_lat / 2.0) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(delta_lon / 2.0) ** 2
    )
    return 2.0 * 6371.0088 * math.asin(math.sqrt(value))


def geographic_components(
    split_internal: dict[str, dict[str, Any]], radius_km: float
) -> dict[str, Any]:
    coordinates: dict[str, tuple[float, float]] = {}
    location_splits: dict[str, set[str]] = defaultdict(set)
    for split, internal in split_internal.items():
        coordinates.update(internal["coordinates"])
        for location in internal["locations"]:
            location_splits[location].add(split)
    locations = sorted(coordinates)
    parent = list(range(len(locations)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for left in range(len(locations)):
        for right in range(left + 1, len(locations)):
            if haversine_km(coordinates[locations[left]], coordinates[locations[right]]) <= radius_km:
                union(left, right)

    components: dict[int, list[str]] = defaultdict(list)
    for index, location in enumerate(locations):
        components[find(index)].append(location)
    split_patterns: Counter[str] = Counter()
    cross_split = 0
    largest = 0
    for members in components.values():
        splits = sorted({split for location in members for split in location_splits[location]})
        split_patterns["+".join(splits)] += 1
        cross_split += int(len(splits) > 1)
        largest = max(largest, len(members))
    return {
        "radius_km": radius_km,
        "locations_with_coordinates": len(locations),
        "component_count": len(components),
        "cross_split_components": cross_split,
        "largest_component_locations": largest,
        "component_split_patterns": counter_dict(split_patterns),
    }


def git_commit(root: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


def tracked_dirty(root: Path) -> bool:
    output = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=root, text=True
    )
    return bool(output.strip())


def safe_output(root: Path, value: str) -> Path:
    path = (root / value).resolve()
    if root not in path.parents:
        raise ValueError("Report output must resolve beneath the repository root")
    return path


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_markdown(path: Path, audit: dict[str, Any]) -> None:
    splits = audit["official_splits"]
    overlap = audit["split_isolation"]
    global_info = audit["global_integrity"]
    lines = [
        "# MARS-S2L pinned metadata audit",
        "",
        f"- Source: `{REPO_ID}`",
        f"- Revision: `{REVISION}`",
        f"- Generated: `{audit['generated_at_utc']}`",
        f"- Metadata files: {audit['source_integrity']['file_count']} / {audit['source_integrity']['total_bytes']:,} bytes verified",
        "",
        "## Official split summary",
        "",
        "| Split | Rows | Plume | No plume | Plume % | Locations | S2 L1C | Recommended S2 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split in OFFICIAL_SPLITS:
        item = splits[split]
        counts = item["counts"]
        lines.append(
            f"| {split} | {counts['rows']:,} | {counts['positive']:,} | {counts['negative']:,} | "
            f"{100.0 * item['positive_fraction']:.2f}% | {item['unique_locations']:,} | "
            f"{counts.get('sentinel2_rows', 0):,} | {counts.get('recommended_s2_rows', 0):,} |"
        )
    lines.extend(
        [
            "",
            "The official split union contains "
            f"{global_info['official_split_rows']:,} unique image-location items. The full metadata "
            f"table contains {global_info['all_rows']:,}; {global_info['unassigned_rows']:,} rows are "
            "outside the released train/validation/test CSVs and must not be silently added.",
            "",
            "## What the metadata proves",
            "",
            f"- Explicit real negatives: {global_info['official_negative_rows']:,} official-split rows.",
            f"- Positive images: {global_info['official_positive_rows']:,}; validated plume table: "
            f"{global_info['plume_rows']:,} plume records on {global_info['plume_image_ids']:,} images.",
            f"- Plume-table linkage is incomplete for {global_info['official_positive_images_missing_from_plume_table']:,} "
            "official positive images; keep those out of object/flux analyses until reconciled.",
            "- Sentinel-2 inputs are MSI L1C; Landsat Collection-2 L1 products are a separate domain.",
            "- Exact acquisition-scene overlap across official splits is zero.",
            f"- Physical location overlap is not zero: train/validation {overlap['location_overlap_counts']['train_val']}, "
            f"train/test {overlap['location_overlap_counts']['train_test']}, validation/test "
            f"{overlap['location_overlap_counts']['val_test']}.",
            f"- Test-only physical locations: {overlap['test_unseen_locations']:,}; these contain "
            f"{overlap['test_unseen_rows']:,} rows ({overlap['test_unseen_positive_rows']:,} plume, "
            f"{overlap['test_unseen_negative_rows']:,} no plume).",
            f"- A {GEOGRAPHIC_GROUP_RADIUS_KM:g} km location graph yields "
            f"{overlap['geographic_components']['component_count']:,} components; "
            f"{overlap['geographic_components']['cross_split_components']:,} cross official splits.",
            "",
            "## Architecture and evaluation consequences",
            "",
            "1. Start with the Sentinel-2 L1C target/reference cohort only; do not mix Landsat or the existing L2A pilot.",
            "2. Preserve the official test set for published comparability, but report the test-only-location subset as the primary geographic-transfer result.",
            "3. Rebuild group IDs as connected components of physical location and 25 km proximity before any ERSRR cross-validation.",
            "4. Train on reviewed negatives and calibrate no-plume thresholds on validation only.",
            "5. Exclude non-clear, low-clear, or missing-reference rows from the first model; evaluate them later as observability/abstention cases.",
            "6. Download only assets referenced by the selected Sentinel-2 cohort rather than mirroring the full mixed-sensor repository.",
            "",
            "## Integrity findings",
            "",
            f"- Duplicate item IDs in official splits: {global_info['official_duplicate_item_ids']}",
            f"- Exact scenes crossing official splits: {sum(overlap['scene_overlap_counts'].values())}",
            f"- Official items missing from the full table: {global_info['official_items_missing_from_all']}",
            f"- Positive asset-contract violations: {global_info['positive_asset_contract_violations']}",
            f"- Negative rows carrying positive assets: {global_info['negative_asset_contract_violations']}",
            f"- Full-table positive images missing plume-table records: "
            f"{global_info['full_positive_images_missing_from_plume_table']}",
            "",
            "The metadata gate passes for a selective Sentinel-2 asset import. It does not authorize a model claim; image-band and raster-grid contracts must be verified on downloaded assets next.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_audit(root: Path, input_dir: Path) -> dict[str, Any]:
    verified = verify_files(input_dir)
    local_manifest = input_dir / MANIFEST_NAME
    manifest = json.loads(local_manifest.read_text(encoding="utf-8"))
    if manifest.get("source", {}).get("revision") != REVISION:
        raise ValueError("Local metadata manifest revision does not match the pinned audit revision")

    split_summaries: dict[str, dict[str, Any]] = {}
    split_internal: dict[str, dict[str, Any]] = {}
    for split in OFFICIAL_SPLITS:
        summary, internal = audit_image_csv(input_dir / f"{split}.csv", split)
        split_summaries[split] = summary
        split_internal[split] = internal

    all_summary, all_internal = audit_image_csv(input_dir / "validated_images_all.csv", None)
    plume_summary, plume_image_ids = audit_plume_csv(input_dir / "validated_images_plumes.csv")
    official_ids = set().union(*(split_internal[name]["ids"] for name in OFFICIAL_SPLITS))
    official_positive_ids = set().union(
        *(split_internal[name]["positive_ids"] for name in OFFICIAL_SPLITS)
    )
    official_locations = {
        name: split_internal[name]["locations"] for name in OFFICIAL_SPLITS
    }
    official_scenes = {name: split_internal[name]["scenes"] for name in OFFICIAL_SPLITS}
    train_validation_locations = official_locations["train"] | official_locations["val"]
    test_unseen_locations = official_locations["test"] - train_validation_locations
    test_rows_by_location = split_internal["test"]["rows_by_location"]
    test_unseen_rows = sum(
        test_rows_by_location[location]["positive"] + test_rows_by_location[location]["negative"]
        for location in test_unseen_locations
    )
    test_unseen_positive = sum(
        test_rows_by_location[location]["positive"] for location in test_unseen_locations
    )
    test_unseen_negative = sum(
        test_rows_by_location[location]["negative"] for location in test_unseen_locations
    )
    all_counts = all_summary["counts"]
    official_counts = Counter()
    for split in OFFICIAL_SPLITS:
        official_counts.update(split_summaries[split]["counts"])

    audit = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "repository": REPO_ID,
            "repository_url": f"https://huggingface.co/datasets/{REPO_ID}",
            "revision": REVISION,
            "license": "CC-BY-NC-SA-4.0",
            "local_manifest_sha256": sha256(local_manifest),
        },
        "source_integrity": {
            "file_count": len(verified),
            "total_bytes": sum(int(item["size"]) for item in verified),
            "all_expected_sizes_match": True,
            "all_declared_lfs_hashes_match": True,
            "files": verified,
        },
        "official_splits": split_summaries,
        "full_metadata_table": all_summary,
        "plume_table": plume_summary,
        "global_integrity": {
            "official_split_rows": sum(split_summaries[name]["counts"]["rows"] for name in OFFICIAL_SPLITS),
            "official_unique_item_ids": len(official_ids),
            "official_duplicate_item_ids": sum(
                split_summaries[name]["counts"].get("duplicate_item_id", 0) for name in OFFICIAL_SPLITS
            ),
            "official_positive_rows": official_counts["positive"],
            "official_negative_rows": official_counts["negative"],
            "all_rows": all_counts["rows"],
            "all_unique_item_ids": all_summary["unique_item_ids"],
            "unassigned_rows": len(all_internal["ids"] - official_ids),
            "official_items_missing_from_all": len(official_ids - all_internal["ids"]),
            "official_positive_images_missing_from_plume_table": len(official_positive_ids - plume_image_ids),
            "plume_images_outside_official_positive_ids": len(plume_image_ids - official_positive_ids),
            "full_positive_images_missing_from_plume_table": len(
                all_internal["positive_ids"] - plume_image_ids
            ),
            "plume_images_not_marked_positive_in_full_table": len(
                plume_image_ids - all_internal["positive_ids"]
            ),
            "positive_images_missing_plume_table_by_split": {
                name: len(split_internal[name]["positive_ids"] - plume_image_ids)
                for name in OFFICIAL_SPLITS
            },
            "plume_rows": plume_summary["counts"]["rows"],
            "plume_image_ids": plume_summary["unique_image_ids"],
            "positive_asset_contract_violations": sum(
                split_summaries[name]["counts"].get("positive_missing_plume_path", 0)
                + split_summaries[name]["counts"].get("positive_missing_ch4_path", 0)
                + split_summaries[name]["counts"].get("positive_missing_plume_geometry", 0)
                for name in OFFICIAL_SPLITS
            ),
            "negative_asset_contract_violations": sum(
                split_summaries[name]["counts"].get("negative_with_positive_asset", 0)
                for name in OFFICIAL_SPLITS
            ),
        },
        "split_isolation": {
            "location_overlap_counts": {
                "train_val": len(official_locations["train"] & official_locations["val"]),
                "train_test": len(official_locations["train"] & official_locations["test"]),
                "val_test": len(official_locations["val"] & official_locations["test"]),
            },
            "scene_overlap_counts": {
                "train_val": len(official_scenes["train"] & official_scenes["val"]),
                "train_test": len(official_scenes["train"] & official_scenes["test"]),
                "val_test": len(official_scenes["val"] & official_scenes["test"]),
            },
            "test_unseen_locations": len(test_unseen_locations),
            "test_unseen_rows": test_unseen_rows,
            "test_unseen_positive_rows": test_unseen_positive,
            "test_unseen_negative_rows": test_unseen_negative,
            "geographic_components": geographic_components(
                split_internal, GEOGRAPHIC_GROUP_RADIUS_KM
            ),
        },
        "recommended_initial_cohort": {
            "sensor": "Sentinel-2 MSI",
            "product_level": "L1C",
            "observability": "clear",
            "minimum_clear_percentage": RECOMMENDED_MIN_CLEAR_PCT,
            "requires_background_reference": True,
            "counts_by_split": {
                name: {
                    "rows": split_summaries[name]["counts"].get("recommended_s2_rows", 0),
                    "positive": split_summaries[name]["counts"].get("recommended_s2_positive", 0),
                    "negative": split_summaries[name]["counts"].get("recommended_s2_negative", 0),
                }
                for name in OFFICIAL_SPLITS
            },
        },
        "decisions": [
            "Use MARS-S2L as the primary real positive/negative corpus.",
            "Start with Sentinel-2 MSI L1C only; keep Landsat and ERSRR L2A separate.",
            "Use test-only physical locations as the primary geographic-transfer subset.",
            "Recompute 25 km connected location groups for ERSRR cross-validation.",
            "Treat non-clear and missing-reference rows as observability/abstention cases.",
            "Download only assets referenced by the selected Sentinel-2 cohort.",
        ],
        "provenance": {
            "git_commit": git_commit(root),
            "git_tracked_worktree_dirty_at_start": tracked_dirty(root),
            "script": "tools/audit_mars_metadata.py",
            "script_sha256": sha256(Path(__file__)),
            "python": sys.version.split()[0],
        },
    }
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default=DEFAULT_OUTPUT.as_posix())
    parser.add_argument("--output-json", default=DEFAULT_JSON.as_posix())
    parser.add_argument("--output-markdown", default=DEFAULT_MARKDOWN.as_posix())
    parser.add_argument("--dry-run", action="store_true", help="Audit without writing reports")
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args()
    root = repo_root()
    try:
        input_dir = checked_output_dir(root, args.input_dir)
        output_json = safe_output(root, args.output_json)
        output_markdown = safe_output(root, args.output_markdown)
        audit = build_audit(root, input_dir)
        if not args.dry_run:
            write_json(output_json, audit)
            write_markdown(output_markdown, audit)
    except (csv.Error, FileNotFoundError, OSError, ValueError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, indent=None if args.compact else 2))
        return 2
    payload = {
        "ok": True,
        "dry_run": bool(args.dry_run),
        "revision": REVISION,
        "official_rows": audit["global_integrity"]["official_split_rows"],
        "official_positive_rows": audit["global_integrity"]["official_positive_rows"],
        "official_negative_rows": audit["global_integrity"]["official_negative_rows"],
        "test_unseen_locations": audit["split_isolation"]["test_unseen_locations"],
        "recommended_initial_cohort": audit["recommended_initial_cohort"]["counts_by_split"],
        "output_json": None if args.dry_run else output_json.relative_to(root).as_posix(),
        "output_markdown": None if args.dry_run else output_markdown.relative_to(root).as_posix(),
    }
    print(json.dumps(payload, indent=None if args.compact else 2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
