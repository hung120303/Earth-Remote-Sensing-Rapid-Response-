#!/usr/bin/env python3
"""Audit and freeze the post-2024 UNEP MARS exact-product plume cohort."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


DEFAULT_PROTOCOL = Path("configs/unep_mars_post2024_protocol.json")
DEFAULT_CATALOG_DIR = Path(".research/unep_mars_post2024/extracted")
DEFAULT_RAW_DIR = Path(".research/unep_mars_post2024/raw")
DEFAULT_MANIFEST = Path(".research/unep_mars_post2024/eligible_manifest.jsonl")
DEFAULT_JSON = Path("reports/acquisition/unep_mars_post2024_catalog_audit.json")
DEFAULT_MARKDOWN = Path("reports/acquisition/UNEP_MARS_POST2024_CATALOG_AUDIT.md")

S2_RE = re.compile(r"^S2[ABC]_MSIL1C_\d{8}T\d{6}_")
LANDSAT_RE = re.compile(r"^LC0[89]_L1(?:TP|GT|GS)_")
ALLOWED_GEOMETRIES = {"Polygon", "MultiPolygon"}


def repo_root() -> Path:
    value = subprocess.check_output(
        ["git", "rev-parse", "--show-toplevel"], text=True
    ).strip()
    if not value:
        raise RuntimeError("Could not resolve repository root")
    return Path(value).resolve()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_path(root: Path, value: str) -> Path:
    path = (root / value).resolve()
    if path != root and root not in path.parents:
        raise ValueError(f"Path must remain beneath repository root: {value}")
    return path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def parse_time(value: str) -> datetime:
    normalized = value.strip().replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def haversine_km(left: tuple[float, float], right: tuple[float, float]) -> float:
    lon1, lat1 = map(math.radians, left)
    lon2, lat2 = map(math.radians, right)
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    value = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(
        dlon / 2
    ) ** 2
    return 6371.0088 * 2 * math.asin(min(1.0, math.sqrt(value)))


def minimum_distance_km(
    point: tuple[float, float], references: Iterable[tuple[float, float]]
) -> float:
    return min((haversine_km(point, other) for other in references), default=math.inf)


def valid_product_pair(satellite: str, target: str, background: str) -> bool:
    if not target or not background or target == background:
        return False
    if satellite == "Sentinel-2 - ESA":
        return bool(S2_RE.match(target) and S2_RE.match(background))
    if satellite == "Landsat - NASA/USGS":
        return bool(LANDSAT_RE.match(target) and LANDSAT_RE.match(background))
    return False


def split_bucket(group_id: str) -> int:
    identity = hashlib.sha256(
        f"ERSRR-UNEP-MARS-v1|{group_id}".encode("utf-8")
    ).digest()
    return int.from_bytes(identity[:8], "big") % 10


def role_for_bucket(bucket: int) -> str:
    if bucket <= 7:
        return "auxiliary_training"
    if bucket == 8:
        return "development"
    return "sealed_external"


def connected_source_groups(
    source_points: dict[str, tuple[float, float]], radius_km: float
) -> dict[str, str]:
    names = sorted(source_points)
    parent = list(range(len(names)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    for left in range(len(names)):
        for right in range(left + 1, len(names)):
            if (
                haversine_km(source_points[names[left]], source_points[names[right]])
                <= radius_km
            ):
                union(left, right)

    members: dict[int, list[str]] = defaultdict(list)
    for index, name in enumerate(names):
        members[find(index)].append(name)
    result: dict[str, str] = {}
    for component in members.values():
        canonical = "|".join(sorted(component))
        identity = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
        group_id = f"unep25_{identity}"
        for name in component:
            result[name] = group_id
    return result


def paper_exclusions(path: Path) -> tuple[set[str], list[tuple[float, float]]]:
    rows = read_csv(path)
    targets = {row["tile"].strip() for row in rows if row.get("tile", "").strip()}
    locations = {
        (round(float(row["lon"]), 7), round(float(row["lat"]), 7))
        for row in rows
        if row.get("lon", "").strip() and row.get("lat", "").strip()
    }
    return targets, sorted(locations)


def geometry_index(path: Path) -> tuple[dict[str, dict[str, Any] | None], Counter[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("type") != "FeatureCollection":
        raise ValueError("Expected a GeoJSON FeatureCollection")
    result: dict[str, dict[str, Any] | None] = {}
    types: Counter[str] = Counter()
    for feature in payload.get("features", []):
        plume_id = str(feature.get("properties", {}).get("id_plume", "")).strip()
        if not plume_id or plume_id in result:
            raise ValueError("GeoJSON plume IDs must be nonempty and unique")
        geometry = feature.get("geometry")
        geometry_type = str((geometry or {}).get("type") or "None")
        types[geometry_type] += 1
        result[plume_id] = geometry
    return result, types


def select_samples(
    rows: list[dict[str, str]],
    geometries: dict[str, dict[str, Any] | None],
    *,
    cutoff: datetime,
    allowed_satellites: set[str],
    required_fields: list[str],
    paper_targets: set[str],
    paper_locations: list[tuple[float, float]],
    exclusion_km: float,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    rejected: Counter[str] = Counter()
    accepted: list[dict[str, Any]] = []
    for row in rows:
        try:
            observed = parse_time(row.get("tile_date", ""))
        except (TypeError, ValueError):
            rejected["invalid_tile_date"] += 1
            continue
        if observed < cutoff:
            rejected["before_cutoff"] += 1
            continue
        satellite = row.get("satellite", "").strip()
        if satellite not in allowed_satellites:
            rejected["disallowed_satellite"] += 1
            continue
        if row.get("actionable", "").strip().casefold() != "yes":
            rejected["not_expert_actionable"] += 1
            continue
        if any(not row.get(field, "").strip() for field in required_fields):
            rejected["missing_required_field"] += 1
            continue
        target = row["tile"].strip()
        background = row["tile_background"].strip()
        if not valid_product_pair(satellite, target, background):
            rejected["invalid_product_pair"] += 1
            continue
        if target in paper_targets:
            rejected["exact_paper_test_target"] += 1
            continue
        try:
            point = (float(row["lon"]), float(row["lat"]))
        except ValueError:
            rejected["invalid_coordinate"] += 1
            continue
        distance = minimum_distance_km(point, paper_locations)
        if distance < exclusion_km:
            rejected["within_paper_test_exclusion"] += 1
            continue
        plume_id = row["id_plume"].strip()
        if plume_id not in geometries:
            raise ValueError(f"CSV plume absent from GeoJSON: {plume_id}")
        geometry = geometries[plume_id]
        accepted.append(
            {
                "plume_id": plume_id,
                "source_name": row["source_name"].strip(),
                "satellite": satellite,
                "sensor_family": "Sentinel-2" if satellite.startswith("Sentinel-2") else "Landsat",
                "tile_date": observed.isoformat(),
                "target_product": target,
                "background_product": background,
                "longitude": point[0],
                "latitude": point[1],
                "minimum_paper_test_distance_km": distance,
                "geometry": geometry,
                "pixel_truth_eligible": (geometry or {}).get("type") in ALLOWED_GEOMETRIES,
            }
        )

    by_sample: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in accepted:
        by_sample[(record["target_product"], record["source_name"])].append(record)
    source_coordinates: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for records in by_sample.values():
        for record in records:
            source_coordinates[record["source_name"]].append(
                (record["longitude"], record["latitude"])
            )
    source_points = {
        source: (
            sum(point[0] for point in points) / len(points),
            sum(point[1] for point in points) / len(points),
        )
        for source, points in source_coordinates.items()
    }
    groups = connected_source_groups(source_points, exclusion_km)

    samples: list[dict[str, Any]] = []
    for (target, source), records in sorted(by_sample.items()):
        first = records[0]
        if any(record["background_product"] != first["background_product"] for record in records):
            raise ValueError(f"One target/source sample has multiple backgrounds: {target} {source}")
        group_id = groups[source]
        bucket = split_bucket(group_id)
        samples.append(
            {
                "sample_id": hashlib.sha256(f"{target}|{source}".encode("utf-8")).hexdigest()[:24],
                "target_product": target,
                "background_product": first["background_product"],
                "tile_date": first["tile_date"],
                "satellite": first["satellite"],
                "sensor_family": first["sensor_family"],
                "source_name": source,
                "source_center": [
                    round(source_points[source][0], 7),
                    round(source_points[source][1], 7),
                ],
                "minimum_paper_test_distance_km": round(
                    min(record["minimum_paper_test_distance_km"] for record in records), 6
                ),
                "group_id": group_id,
                "split_bucket": bucket,
                "research_role": role_for_bucket(bucket),
                "plume_ids": sorted(record["plume_id"] for record in records),
                "plume_geometries": [
                    record["geometry"] for record in records if record["pixel_truth_eligible"]
                ],
                "pixel_truth_eligible": any(
                    record["pixel_truth_eligible"] for record in records
                ),
            }
        )
    rejected["accepted_plume_records"] = len(accepted)
    rejected["merged_duplicate_plume_records"] = len(accepted) - len(samples)
    return samples, rejected


def counts_by(samples: list[dict[str, Any]], field: str) -> dict[str, int]:
    return dict(sorted(Counter(str(sample[field]) for sample in samples).items()))


def write_markdown(report: dict[str, Any], path: Path) -> None:
    counts = report["cohort"]
    rejection = report["selection_audit"]
    lines = [
        "# UNEP MARS post-2024 catalog audit",
        "",
        f"Generated: {report['generated_at_utc']}.",
        "",
        "## Result",
        "",
        f"- Catalog plume rows: **{report['catalog']['plume_rows']:,}**.",
        f"- Eligible exact-product plume records: **{rejection['accepted_plume_records']:,}**.",
        f"- Source-crop samples after merging: **{counts['samples']:,}**.",
        f"- Independent 25 km groups: **{counts['groups']:,}**.",
        f"- Pixel-truth samples with polygon geometry: **{counts['pixel_truth_samples']:,}**.",
        f"- Sentinel-2 / Landsat samples: **{counts['by_sensor'].get('Sentinel-2', 0):,} / {counts['by_sensor'].get('Landsat', 0):,}**.",
        "",
        "## Fixed roles",
        "",
    ]
    for role, count in counts["by_role"].items():
        lines.append(f"- {role}: **{count:,}** samples.")
    lines.extend(["", "## Sequential exclusions", ""])
    for reason, count in sorted(rejection.items()):
        if reason not in {"accepted_plume_records", "merged_duplicate_plume_records"}:
            lines.append(f"- {reason}: {count:,}")
    lines.extend(
        [
            "",
            "## Integrity",
            "",
            f"- Pinned paper test SHA-256: `{report['paper_exclusion']['paper_test_sha256']}`.",
            f"- Minimum accepted distance from a paper-test location: **{counts['minimum_paper_test_distance_km']:.3f} km**.",
            f"- Eligible manifest SHA-256: `{report['manifest']['sha256']}`.",
            "- No catalog absence was interpreted as a no-plume label.",
            "- The sealed-external role remains positive-only and must stay unread during model selection.",
            "- Bulk catalogs and the row-level manifest remain under ignored `.research/` storage.",
            "",
            "Source: UNEP IMEO Eye on Methane MARS sources and plumes, CC BY-NC-SA 4.0 (non-commercial).",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL.as_posix())
    parser.add_argument("--catalog-dir", default=DEFAULT_CATALOG_DIR.as_posix())
    parser.add_argument("--raw-dir", default=DEFAULT_RAW_DIR.as_posix())
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST.as_posix())
    parser.add_argument("--output-json", default=DEFAULT_JSON.as_posix())
    parser.add_argument("--output-markdown", default=DEFAULT_MARKDOWN.as_posix())
    args = parser.parse_args()

    root = repo_root()
    protocol_path = safe_path(root, args.protocol)
    catalog_dir = safe_path(root, args.catalog_dir)
    raw_dir = safe_path(root, args.raw_dir)
    manifest_path = safe_path(root, args.manifest)
    output_json = safe_path(root, args.output_json)
    output_markdown = safe_path(root, args.output_markdown)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    paper_path = safe_path(root, protocol["paper_comparator_exclusion"]["paper_test_csv"])
    expected_paper_sha = protocol["paper_comparator_exclusion"]["paper_test_csv_sha256"]
    if sha256(paper_path) != expected_paper_sha:
        raise ValueError("Pinned paper-test CSV checksum mismatch")

    csv_path = catalog_dir / "unep_methanedata_detected_plumes.csv"
    geojson_path = catalog_dir / "unep_methanedata_detected_plumes.geojson"
    rows = read_csv(csv_path)
    geometries, geometry_types = geometry_index(geojson_path)
    if len(rows) != len(geometries):
        raise ValueError("CSV and GeoJSON plume counts differ")
    paper_targets, paper_locations = paper_exclusions(paper_path)
    selection = protocol["catalog_selection"]
    cutoff = parse_time(selection["minimum_tile_date_utc"])
    samples, rejection = select_samples(
        rows,
        geometries,
        cutoff=cutoff,
        allowed_satellites=set(selection["allowed_satellites"]),
        required_fields=list(selection["required_nonempty_fields"]),
        paper_targets=paper_targets,
        paper_locations=paper_locations,
        exclusion_km=float(
            protocol["paper_comparator_exclusion"][
                "minimum_distance_from_every_paper_test_location_km"
            ]
        ),
    )
    if not samples:
        raise ValueError("Frozen selection produced no samples")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="\n") as target:
        for sample in samples:
            target.write(json.dumps(sample, sort_keys=True, separators=(",", ":")) + "\n")

    archive_hashes = {
        path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in sorted(raw_dir.glob("*.zip"))
    }
    report = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "catalog_audited; imagery_not_yet_acquired",
        "protocol": {"path": args.protocol, "sha256": sha256(protocol_path)},
        "catalog": {
            "plume_rows": len(rows),
            "csv": {"bytes": csv_path.stat().st_size, "sha256": sha256(csv_path)},
            "geojson": {
                "bytes": geojson_path.stat().st_size,
                "sha256": sha256(geojson_path),
                "geometry_types": dict(sorted(geometry_types.items())),
            },
            "archives": archive_hashes,
        },
        "paper_exclusion": {
            "paper_test_sha256": expected_paper_sha,
            "paper_test_targets": len(paper_targets),
            "paper_test_locations": len(paper_locations),
        },
        "selection_audit": dict(sorted(rejection.items())),
        "cohort": {
            "samples": len(samples),
            "groups": len({sample["group_id"] for sample in samples}),
            "sources": len({sample["source_name"] for sample in samples}),
            "pixel_truth_samples": sum(sample["pixel_truth_eligible"] for sample in samples),
            "by_sensor": counts_by(samples, "sensor_family"),
            "by_role": counts_by(samples, "research_role"),
            "minimum_paper_test_distance_km": min(
                sample["minimum_paper_test_distance_km"] for sample in samples
            ),
        },
        "manifest": {
            "path": args.manifest,
            "bytes": manifest_path.stat().st_size,
            "sha256": sha256(manifest_path),
        },
        "label_semantics": {
            "all_samples": "expert-actionable positive plume",
            "no_plume_from_catalog_absence": False,
            "external_metrics": "positive recall and mask IoU only",
        },
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(report, output_markdown)
    print(json.dumps(report["cohort"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
