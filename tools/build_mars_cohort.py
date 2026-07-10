#!/usr/bin/env python3
"""Freeze the publication MARS-S2L Sentinel-2 cohort and remote asset inventory.

The detailed sample manifest and remote catalog are intentionally written under
the ignored acquisition root. Compact reports containing counts, byte totals,
identities, and split-isolation evidence are suitable for version control. This
tool inventories remote files but does not download the raster corpus.
"""

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
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from acquire_mars_metadata import (
    DEFAULT_OUTPUT,
    REPO_ID,
    REVISION,
    USER_AGENT,
    checked_output_dir,
    repo_root,
    sha256,
    source_url,
    verify_files,
)
from acquire_mars_pilot import MIN_CLEAR_PCT, parse_bool, recommended_s2

OFFICIAL_SPLITS = ("train", "val", "test")
GROUP_RADIUS_KM = 25.0
CATALOG_BATCH_SIZE = 500
CATALOG_WORKERS = 4
COHORT_MANIFEST = "publication_s2_cohort.jsonl"
REMOTE_CATALOG = "publication_s2_remote_catalog.jsonl"
DEFAULT_JSON = Path("reports/acquisition/mars_s2l_publication_cohort.json")
DEFAULT_MARKDOWN = Path("reports/acquisition/MARS_S2L_PUBLICATION_COHORT.md")
PATHS_INFO_URL = (
    f"https://huggingface.co/api/datasets/{REPO_ID}/paths-info/{REVISION}"
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


def git_commit(root: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


def tracked_dirty(root: Path) -> bool:
    output = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=root, text=True
    )
    return bool(output.strip())


def safe_report(root: Path, value: str) -> Path:
    result = (root / value).resolve()
    if root not in result.parents:
        raise ValueError("Report output must resolve beneath the repository root")
    return result


def haversine_km(left: tuple[float, float], right: tuple[float, float]) -> float:
    lat1, lon1 = map(math.radians, left)
    lat2, lon2 = map(math.radians, right)
    delta_lat = lat2 - lat1
    delta_lon = lon2 - lon1
    value = (
        math.sin(delta_lat / 2.0) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(delta_lon / 2.0) ** 2
    )
    return 6371.0088 * 2.0 * math.asin(min(1.0, math.sqrt(value)))


def read_cohort(metadata_dir: Path) -> tuple[list[dict[str, Any]], dict[str, tuple[float, float]]]:
    csv.field_size_limit(sys.maxsize)
    rows: list[dict[str, Any]] = []
    locations: dict[str, tuple[float, float]] = {}
    train_validation_locations: set[str] = set()
    raw_by_split: dict[str, list[dict[str, str]]] = {}
    for split in OFFICIAL_SPLITS:
        with (metadata_dir / f"{split}.csv").open("r", encoding="utf-8", newline="") as source:
            raw_by_split[split] = list(csv.DictReader(source))
        if split in ("train", "val"):
            train_validation_locations.update(row["id_location"] for row in raw_by_split[split])

    for split in OFFICIAL_SPLITS:
        for row in raw_by_split[split]:
            if not recommended_s2(row):
                continue
            positive = parse_bool(row["isplume"])
            location = row["id_location"]
            coordinate = (float(row["lat"]), float(row["lon"]))
            previous = locations.setdefault(location, coordinate)
            if haversine_km(previous, coordinate) > 0.1:
                raise ValueError(f"Location {location} has inconsistent coordinates")
            assets = [
                {"role": "image", "path": row["s2path"]},
                {"role": "cloud_mask", "path": row["cloudmaskpath"]},
            ]
            if positive:
                assets.extend(
                    [
                        {"role": "plume_mask", "path": row["plumepath"]},
                        {"role": "methane_enhancement", "path": row["ch4path"]},
                    ]
                )
            if any(not item["path"].strip() for item in assets):
                raise ValueError(f"Selected sample has a missing asset: {row['id_loc_image']}")
            rows.append(
                {
                    "sample_id": row["id_loc_image"],
                    "source_dataset": REPO_ID,
                    "source_revision": REVISION,
                    "license": "CC-BY-NC-SA-4.0",
                    "sensor": "Sentinel-2 MSI",
                    "product_level": "L1C",
                    "target_scene_id": row["tile"],
                    "reference_scene_id": row["background_image_tile"],
                    "target_datetime": row["tile_date"],
                    "band_order": [
                        "B02",
                        "B03",
                        "B04",
                        "B08",
                        "B11",
                        "B12",
                        "B02_bg",
                        "B03_bg",
                        "B04_bg",
                        "B08_bg",
                        "B11_bg",
                        "B12_bg",
                    ],
                    "crs": row["crs"],
                    "width": int(row["width"]),
                    "height": int(row["height"]),
                    "observability": row["observability"],
                    "cloud_fraction": round(1.0 - float(row["percentage_clear"]) / 100.0, 8),
                    "label_state": "PLUME" if positive else "NO_PLUME",
                    "label_source": "MARS-S2L reviewed release",
                    "physical_location_id": location,
                    "country": row["country"],
                    "longitude": coordinate[1],
                    "latitude": coordinate[0],
                    "split": split,
                    "test_only_location": split == "test" and location not in train_validation_locations,
                    "assets": assets,
                }
            )
    rows.sort(key=lambda item: (OFFICIAL_SPLITS.index(item["split"]), item["sample_id"]))
    if len({item["sample_id"] for item in rows}) != len(rows):
        raise ValueError("Cohort contains duplicate sample IDs")
    return rows, locations


def assign_geographic_groups(
    rows: list[dict[str, Any]], locations: dict[str, tuple[float, float]]
) -> dict[str, Any]:
    identifiers = sorted(locations)
    groups = UnionFind(identifiers)
    for index, left in enumerate(identifiers):
        for right in identifiers[index + 1 :]:
            if haversine_km(locations[left], locations[right]) <= GROUP_RADIUS_KM:
                groups.union(left, right)
    components: dict[str, list[str]] = defaultdict(list)
    for location in identifiers:
        components[groups.find(location)].append(location)
    group_by_location: dict[str, str] = {}
    for members in components.values():
        ordered = sorted(members)
        identity = hashlib.sha256("\0".join(ordered).encode("utf-8")).hexdigest()[:16]
        group_id = f"geo25_{identity}"
        for location in ordered:
            group_by_location[location] = group_id
    split_by_group: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        row["group_id"] = group_by_location[row["physical_location_id"]]
        split_by_group[row["group_id"]].add(row["split"])
    patterns = Counter("+".join(sorted(value)) for value in split_by_group.values())
    sizes = Counter(row["group_id"] for row in rows)
    return {
        "radius_km": GROUP_RADIUS_KM,
        "physical_locations": len(locations),
        "group_count": len(components),
        "cross_split_groups": sum(len(value) > 1 for value in split_by_group.values()),
        "largest_group_samples": max(sizes.values()),
        "split_patterns": dict(sorted(patterns.items())),
    }


def asset_owners(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    owners: dict[str, dict[str, Any]] = {}
    for row in rows:
        for asset in row["assets"]:
            path = asset["path"]
            pure = PurePosixPath(path)
            if pure.is_absolute() or ".." in pure.parts:
                raise ValueError(f"Unsafe asset path in cohort: {path}")
            observed = owners.setdefault(
                path,
                {"roles": set(), "splits": set(), "labels": set(), "sample_count": 0},
            )
            observed["roles"].add(asset["role"])
            observed["splits"].add(row["split"])
            observed["labels"].add(row["label_state"])
            observed["sample_count"] += 1
    return owners


def batches(values: list[str], size: int) -> Iterable[list[str]]:
    for offset in range(0, len(values), size):
        yield values[offset : offset + size]


def simplify_remote(item: dict[str, Any]) -> dict[str, Any]:
    lfs = item.get("lfs")
    return {
        "path": item["path"],
        "size": int(item["size"]),
        "remote_oid": lfs["oid"] if lfs else item["oid"],
        "remote_oid_type": "sha256_lfs" if lfs else "git_blob_sha1",
        "xet_hash": item.get("xetHash"),
        "source_url": source_url(item["path"]),
    }


def request_batch(paths: list[str]) -> list[dict[str, Any]]:
    body = json.dumps({"paths": paths, "expand": True}, separators=(",", ":")).encode("utf-8")
    request = urllib.request.Request(
        PATHS_INFO_URL,
        data=body,
        headers={"Content-Type": "application/json", "User-Agent": USER_AGENT},
        method="POST",
    )
    for attempt in range(5):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                payload = json.load(response)
            return [simplify_remote(item) for item in payload if item.get("type") == "file"]
        except (TimeoutError, urllib.error.HTTPError, urllib.error.URLError) as exc:
            if isinstance(exc, urllib.error.HTTPError) and exc.code not in (429, 500, 502, 503, 504):
                raise
            if attempt == 4:
                raise
            time.sleep(2**attempt)
    raise RuntimeError("Unreachable catalog retry state")


def load_catalog(path: Path) -> dict[str, dict[str, Any]]:
    catalog: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return catalog
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid catalog JSONL at line {line_number}") from exc
            catalog[item["path"]] = item
    return catalog


def resolve_catalog(
    paths: list[str], catalog_path: Path, *, offline: bool, refresh: bool
) -> dict[str, dict[str, Any]]:
    catalog = {} if refresh else load_catalog(catalog_path)
    required = set(paths)
    catalog = {path: item for path, item in catalog.items() if path in required}
    missing = sorted(required - set(catalog))
    if missing and offline:
        raise ValueError(f"Remote catalog is missing {len(missing):,} required assets in offline mode")
    jobs = list(batches(missing, CATALOG_BATCH_SIZE))
    if jobs:
        print(
            f"Resolving {len(missing):,} remote assets in {len(jobs):,} API batches...",
            file=sys.stderr,
            flush=True,
        )
        with ThreadPoolExecutor(max_workers=CATALOG_WORKERS) as executor:
            futures = {executor.submit(request_batch, job): job for job in jobs}
            completed = 0
            for future in as_completed(futures):
                requested = futures[future]
                result = future.result()
                returned = {item["path"] for item in result}
                absent = set(requested) - returned
                if absent:
                    raise ValueError(
                        f"Pinned repository did not return {len(absent)} requested assets; "
                        f"first missing: {sorted(absent)[0]}"
                    )
                catalog.update({item["path"]: item for item in result})
                completed += 1
                if completed % 20 == 0 or completed == len(jobs):
                    print(
                        f"Resolved {completed:,}/{len(jobs):,} catalog batches",
                        file=sys.stderr,
                        flush=True,
                    )
    if set(catalog) != required:
        raise ValueError("Resolved catalog does not exactly match the cohort asset set")
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = catalog_path.with_suffix(catalog_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as target:
        for path in sorted(catalog):
            target.write(json.dumps(catalog[path], sort_keys=True, separators=(",", ":")) + "\n")
    os.replace(temporary, catalog_path)
    return catalog


def write_cohort_manifest(
    path: Path, rows: list[dict[str, Any]], catalog: dict[str, dict[str, Any]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as target:
        for row in rows:
            enriched = dict(row)
            enriched["assets"] = [
                {**asset, **catalog[asset["path"]]} for asset in row["assets"]
            ]
            target.write(json.dumps(enriched, sort_keys=True, separators=(",", ":")) + "\n")
    os.replace(temporary, path)


def number_summary(values: list[int]) -> dict[str, int | float]:
    ordered = sorted(values)
    return {
        "min": ordered[0],
        "median": int(statistics.median(ordered)),
        "mean": round(statistics.fmean(ordered), 3),
        "max": ordered[-1],
    }


def build_report(
    root: Path,
    rows: list[dict[str, Any]],
    owners: dict[str, dict[str, Any]],
    catalog: dict[str, dict[str, Any]],
    group_summary: dict[str, Any],
    manifest_path: Path,
    catalog_path: Path,
) -> dict[str, Any]:
    split_labels = Counter(f"{row['split']}:{row['label_state']}" for row in rows)
    bytes_by_role: Counter[str] = Counter()
    assets_by_role: Counter[str] = Counter()
    bytes_by_split: Counter[str] = Counter()
    assets_by_split: Counter[str] = Counter()
    lfs_count = 0
    test_only_rows = [row for row in rows if row["test_only_location"]]
    for path, owner in owners.items():
        remote = catalog[path]
        size = int(remote["size"])
        lfs_count += int(remote["remote_oid_type"] == "sha256_lfs")
        for role in owner["roles"]:
            bytes_by_role[role] += size
            assets_by_role[role] += 1
        for split in owner["splits"]:
            bytes_by_split[split] += size
            assets_by_split[split] += 1
    total_bytes = sum(int(item["size"]) for item in catalog.values())
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "repository": REPO_ID,
            "repository_url": f"https://huggingface.co/datasets/{REPO_ID}",
            "revision": REVISION,
            "license": "CC-BY-NC-SA-4.0",
        },
        "selection": {
            "official_splits_only": True,
            "sensor": "Sentinel-2 MSI",
            "product_level": "L1C",
            "observability": "clear",
            "minimum_clear_percentage": MIN_CLEAR_PCT,
            "requires_background_reference": True,
        },
        "cohort": {
            "sample_count": len(rows),
            "positive_samples": sum(row["label_state"] == "PLUME" for row in rows),
            "negative_samples": sum(row["label_state"] == "NO_PLUME" for row in rows),
            "test_only_location_samples": len(test_only_rows),
            "test_only_locations": len(
                {row["physical_location_id"] for row in test_only_rows}
            ),
            "test_only_location_positive_samples": sum(
                row["label_state"] == "PLUME" for row in test_only_rows
            ),
            "test_only_location_negative_samples": sum(
                row["label_state"] == "NO_PLUME" for row in test_only_rows
            ),
            "split_label_counts": dict(sorted(split_labels.items())),
            "grouping": group_summary,
        },
        "remote_assets": {
            "unique_asset_count": len(catalog),
            "total_bytes": total_bytes,
            "decimal_gb": round(total_bytes / 1_000_000_000, 3),
            "binary_gib": round(total_bytes / (1024**3), 3),
            "lfs_sha256_asset_count": lfs_count,
            "git_blob_sha1_asset_count": len(catalog) - lfs_count,
            "asset_size_bytes": number_summary([int(item["size"]) for item in catalog.values()]),
            "asset_counts_by_role": dict(sorted(assets_by_role.items())),
            "bytes_by_role": dict(sorted(bytes_by_role.items())),
            "asset_counts_by_split": dict(sorted(assets_by_split.items())),
            "bytes_by_split": dict(sorted(bytes_by_split.items())),
        },
        "local_ignored_artifacts": {
            "cohort_manifest": manifest_path.relative_to(root).as_posix(),
            "cohort_manifest_sha256": sha256(manifest_path),
            "cohort_manifest_bytes": manifest_path.stat().st_size,
            "remote_catalog": catalog_path.relative_to(root).as_posix(),
            "remote_catalog_sha256": sha256(catalog_path),
            "remote_catalog_bytes": catalog_path.stat().st_size,
        },
        "decisions": [
            "Authorize only this frozen Sentinel-2 cohort, not the full mixed-sensor repository.",
            "Use the test-only-location subset for the primary geographic-transfer result.",
            "Create an internal group-disjoint train/validation split; the official validation sites overlap training.",
            "Use the 25 km connected-component group IDs for ERSRR resampling and uncertainty intervals.",
            "Verify downloaded LFS files against their declared SHA-256 and all files against size.",
        ],
        "provenance": {
            "git_commit": git_commit(root),
            "git_tracked_worktree_dirty_at_start": tracked_dirty(root),
            "script": "tools/build_mars_cohort.py",
            "script_sha256": sha256(Path(__file__)),
            "python": sys.version.split()[0],
        },
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    cohort = report["cohort"]
    remote = report["remote_assets"]
    groups = cohort["grouping"]
    counts = cohort["split_label_counts"]
    lines = [
        "# MARS-S2L frozen publication cohort",
        "",
        f"- Source revision: `{REVISION}`",
        f"- Samples: {cohort['sample_count']:,} ({cohort['positive_samples']:,} plume / {cohort['negative_samples']:,} no plume)",
        f"- Unique remote assets: {remote['unique_asset_count']:,}",
        f"- Exact transfer size: {remote['total_bytes']:,} bytes ({remote['decimal_gb']:.3f} GB / {remote['binary_gib']:.3f} GiB)",
        f"- Detailed manifest SHA-256: `{report['local_ignored_artifacts']['cohort_manifest_sha256']}`",
        f"- Remote catalog SHA-256: `{report['local_ignored_artifacts']['remote_catalog_sha256']}`",
        "",
        "## Selection",
        "",
        "Official split rows only; Sentinel-2 MSI L1C; `observability=clear`; at least 80% clear; background reference required.",
        "",
        "| Split | Plume | No plume |",
        "|---|---:|---:|",
    ]
    for split in OFFICIAL_SPLITS:
        lines.append(
            f"| {split} | {counts.get(split + ':PLUME', 0):,} | "
            f"{counts.get(split + ':NO_PLUME', 0):,} |"
        )
    lines.extend(
        [
            "",
            "## Split and grouping warning",
            "",
            f"The cohort contains {groups['physical_locations']:,} physical locations and "
            f"{groups['group_count']:,} connected 25 km groups. {groups['cross_split_groups']:,} "
            "groups cross official split boundaries. The official validation locations are not "
            "isolated from training, so create a group-disjoint internal validation split for model "
            "selection. Preserve the official validation/test results for comparison, use the "
            f"{cohort['test_only_locations']:,} test-only locations ({cohort['test_only_location_samples']:,} "
            f"samples: {cohort['test_only_location_positive_samples']:,} plume / "
            f"{cohort['test_only_location_negative_samples']:,} no plume) for the primary "
            "geographic-transfer claim, and never tune thresholds on either test view.",
            "",
            "## Transfer decision",
            "",
            "This manifest is the proposed maximum first S2 scope. It inventories files only and does not download the raster corpus. A large transfer should proceed only after the exact byte total above is accepted. Raw files and both detailed JSONL manifests remain ignored by Git.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-dir", default=DEFAULT_OUTPUT.as_posix())
    parser.add_argument("--output-json", default=DEFAULT_JSON.as_posix())
    parser.add_argument("--output-markdown", default=DEFAULT_MARKDOWN.as_posix())
    parser.add_argument("--offline", action="store_true", help="Require a complete local remote catalog")
    parser.add_argument("--refresh-catalog", action="store_true", help="Ignore and rebuild the remote catalog")
    parser.add_argument("--dry-run", action="store_true", help="Select and group rows without network or writes")
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args()
    if args.offline and args.refresh_catalog:
        parser.error("--offline and --refresh-catalog are mutually exclusive")
    root = repo_root()
    try:
        metadata_dir = checked_output_dir(root, args.metadata_dir)
        verify_files(metadata_dir)
        output_json = safe_report(root, args.output_json)
        output_markdown = safe_report(root, args.output_markdown)
        rows, locations = read_cohort(metadata_dir)
        group_summary = assign_geographic_groups(rows, locations)
        owners = asset_owners(rows)
        if args.dry_run:
            payload = {
                "ok": True,
                "dry_run": True,
                "sample_count": len(rows),
                "positive_samples": sum(row["label_state"] == "PLUME" for row in rows),
                "negative_samples": sum(row["label_state"] == "NO_PLUME" for row in rows),
                "unique_asset_count": len(owners),
                "grouping": group_summary,
            }
            print(json.dumps(payload, indent=None if args.compact else 2, sort_keys=True))
            return 0
        catalog_path = metadata_dir / REMOTE_CATALOG
        manifest_path = metadata_dir / COHORT_MANIFEST
        catalog = resolve_catalog(
            sorted(owners), catalog_path, offline=args.offline, refresh=args.refresh_catalog
        )
        write_cohort_manifest(manifest_path, rows, catalog)
        report = build_report(
            root, rows, owners, catalog, group_summary, manifest_path, catalog_path
        )
        write_json(output_json, report)
        write_markdown(output_markdown, report)
    except (
        csv.Error,
        FileNotFoundError,
        OSError,
        RuntimeError,
        ValueError,
        urllib.error.URLError,
    ) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, indent=None if args.compact else 2))
        return 2
    payload = {
        "ok": True,
        "dry_run": False,
        "sample_count": report["cohort"]["sample_count"],
        "unique_asset_count": report["remote_assets"]["unique_asset_count"],
        "total_bytes": report["remote_assets"]["total_bytes"],
        "binary_gib": report["remote_assets"]["binary_gib"],
        "output_json": output_json.relative_to(root).as_posix(),
        "output_markdown": output_markdown.relative_to(root).as_posix(),
    }
    print(json.dumps(payload, indent=None if args.compact else 2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
