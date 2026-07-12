#!/usr/bin/env python3
"""Build a prediction-blind EMIT V002/Sentinel-2 external-candidate manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import requests

from acquire_v002_pilot import (
    CMR_COLLECTION_ID,
    CMR_GRANULES_URL,
    EARTH_SEARCH_URL,
    S2_BANDS,
    S2_COLLECTION,
    bbox_center,
    parse_datetime,
    plume_feature,
    repo_root,
)

DEFAULT_JSON = Path("reports/acquisition/emit_v002_time_aligned_candidates.json")
DEFAULT_MARKDOWN = Path("reports/acquisition/EMIT_V002_TIME_ALIGNED_CANDIDATES.md")
PILOT_REPORT = Path("reports/acquisition/emit_v002_2026_07_batch.json")
REQUIRED_ASSETS = {asset for _, asset in S2_BANDS} | {"scl"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_output(root: Path, value: str) -> Path:
    path = (root / value).resolve()
    if root not in path.parents:
        raise ValueError("Output must resolve beneath the repository root")
    return path


def request_json(
    method: str,
    url: str,
    *,
    params: dict[str, Any] | None = None,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            response = requests.request(
                method,
                url,
                params=params,
                json=payload,
                headers={"Accept": "application/json", "User-Agent": "ERSRR-research/1.0"},
                timeout=60,
            )
            response.raise_for_status()
            result = response.json()
            if not isinstance(result, dict):
                raise ValueError(f"Expected object response from {url}")
            return result
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(1.0 * (2**attempt))
    assert last_error is not None
    raise last_error


def cmr_records() -> list[dict[str, Any]]:
    payload = request_json(
        "GET",
        CMR_GRANULES_URL,
        params={
            "collection_concept_id": CMR_COLLECTION_ID,
            "page_size": 2000,
            "sort_key[]": "start_date",
        },
    )
    records = [item.get("umm", {}) for item in payload.get("items", [])]
    records = [
        item
        for item in records
        if item.get("GranuleUR", "").startswith("EMIT_L2B_CH4PLM_002_")
        and item.get("CollectionReference", {}).get("Version") == "002"
    ]
    if not records:
        raise ValueError("CMR returned no EMIT CH4PLM V002 records")
    return records


def pilot_ids(root: Path) -> set[str]:
    path = root / PILOT_REPORT
    if not path.is_file():
        return set()
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(item["granule_id"]) for item in payload.get("granules", [])}


def circular_hours(left: float, right: float) -> float:
    difference = abs(left - right) % 24.0
    return min(difference, 24.0 - difference)


def query_order(records: list[dict[str, Any]], excluded: set[str]) -> list[dict[str, Any]]:
    buckets: dict[tuple[int, int], list[tuple[float, str, dict[str, Any]]]] = defaultdict(list)
    for record in records:
        if record.get("GranuleUR") in excluded:
            continue
        try:
            feature = plume_feature(record)
            observed = parse_datetime(feature["properties"]["datetime"])
        except (KeyError, TypeError, ValueError):
            continue
        lon, lat = bbox_center(feature["bbox"])
        local_solar_hour = (observed.hour + observed.minute / 60.0 + lon / 15.0) % 24.0
        solar_distance = circular_hours(local_solar_hour, 10.5)
        cell = (math.floor((lat + 90.0) / 10.0), math.floor((lon + 180.0) / 10.0))
        buckets[cell].append((solar_distance, str(record["GranuleUR"]), record))
    queues = {
        cell: deque(item[2] for item in sorted(values, key=lambda value: (value[0], value[1])))
        for cell, values in buckets.items()
    }
    ordered: list[dict[str, Any]] = []
    cells = sorted(queues)
    while cells:
        next_cells: list[tuple[int, int]] = []
        for cell in cells:
            queue = queues[cell]
            if queue:
                ordered.append(queue.popleft())
            if queue:
                next_cells.append(cell)
        cells = next_cells
    return ordered


def source_scenes(record: dict[str, Any]) -> list[str]:
    additional = {
        item.get("Name"): item.get("Values", [])
        for item in record.get("AdditionalAttributes", [])
        if item.get("Name")
    }
    return sorted(str(value) for value in additional.get("SOURCE_SCENES", []) if value)


def query_one(
    record: dict[str, Any], *, max_offset_hours: float, max_cloud: float
) -> dict[str, Any]:
    granule_id = str(record.get("GranuleUR"))
    try:
        feature = plume_feature(record)
        observed = parse_datetime(feature["properties"]["datetime"])
        start = observed - timedelta(hours=max_offset_hours)
        end = observed + timedelta(hours=max_offset_hours)
        response = request_json(
            "POST",
            EARTH_SEARCH_URL,
            payload={
                "collections": [S2_COLLECTION],
                "bbox": feature["bbox"],
                "datetime": (
                    f"{start.isoformat().replace('+00:00', 'Z')}/"
                    f"{end.isoformat().replace('+00:00', 'Z')}"
                ),
                "limit": 100,
                "query": {"eo:cloud_cover": {"lte": max_cloud}},
            },
        )
        center_lon, center_lat = bbox_center(feature["bbox"])
        candidates = []
        for item in response.get("features", []):
            bbox = item.get("bbox")
            assets = item.get("assets", {})
            if not bbox or len(bbox) < 4 or not REQUIRED_ASSETS.issubset(assets):
                continue
            if not (bbox[0] <= center_lon <= bbox[2] and bbox[1] <= center_lat <= bbox[3]):
                continue
            scene_time = parse_datetime(item["properties"]["datetime"])
            offset = (scene_time - observed).total_seconds() / 3600.0
            if abs(offset) > max_offset_hours + 1e-9:
                continue
            candidates.append(
                (
                    abs(offset),
                    float(item.get("properties", {}).get("eo:cloud_cover", 100.0)),
                    str(item.get("id", "")),
                    offset,
                    scene_time,
                    item,
                )
            )
        if not candidates:
            return {"granule_id": granule_id, "status": "no_time_aligned_s2"}
        _, cloud, scene_id, offset, scene_time, item = min(candidates)
        self_url = next(
            (link.get("href") for link in item.get("links", []) if link.get("rel") == "self"),
            None,
        )
        return {
            "status": "candidate",
            "granule_id": granule_id,
            "plume_id": feature["properties"].get("plume_id"),
            "source_scenes": source_scenes(record),
            "emit_datetime": observed.isoformat(),
            "bbox": [round(float(value), 7) for value in feature["bbox"]],
            "center": [round(center_lon, 7), round(center_lat, 7)],
            "emit_cloud_cover_pct": record.get("CloudCover"),
            "s2_scene_id": scene_id,
            "s2_datetime": scene_time.isoformat(),
            "offset_hours": round(float(offset), 6),
            "scene_cloud_cover_pct": round(cloud, 6),
            "s2_stac_item": self_url,
            "required_asset_roles": sorted(REQUIRED_ASSETS),
        }
    except Exception as exc:  # keep discovery resilient and report every failure
        return {
            "granule_id": granule_id,
            "status": "query_error",
            "error_type": type(exc).__name__,
            "error": str(exc)[:500],
        }


def haversine_km(left: list[float], right: list[float]) -> float:
    lon1, lat1 = map(math.radians, left)
    lon2, lat2 = map(math.radians, right)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    value = math.sin(dlat / 2.0) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(
        dlon / 2.0
    ) ** 2
    return 6371.0088 * 2.0 * math.asin(min(1.0, math.sqrt(value)))


def connected_groups(records: list[dict[str, Any]], radius_km: float) -> list[list[int]]:
    parent = list(range(len(records)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    for left in range(len(records)):
        for right in range(left + 1, len(records)):
            if haversine_km(records[left]["center"], records[right]["center"]) <= radius_km:
                union(left, right)
    groups: dict[int, list[int]] = defaultdict(list)
    for index in range(len(records)):
        groups[find(index)].append(index)
    return sorted(groups.values(), key=lambda indices: min(records[index]["granule_id"] for index in indices))


def independent_best(records: list[dict[str, Any]], radius_km: float) -> list[dict[str, Any]]:
    source_deduplicated: dict[str, dict[str, Any]] = {}
    for record in records:
        source_key = "|".join(record.get("source_scenes") or [record["granule_id"]])
        current = source_deduplicated.get(source_key)
        score = (abs(record["offset_hours"]), record["scene_cloud_cover_pct"], record["granule_id"])
        if current is None or score < (
            abs(current["offset_hours"]),
            current["scene_cloud_cover_pct"],
            current["granule_id"],
        ):
            source_deduplicated[source_key] = record
    scene_deduplicated: dict[str, dict[str, Any]] = {}
    for record in source_deduplicated.values():
        scene_key = record["s2_scene_id"]
        current = scene_deduplicated.get(scene_key)
        score = (abs(record["offset_hours"]), record["scene_cloud_cover_pct"], record["granule_id"])
        if current is None or score < (
            abs(current["offset_hours"]),
            current["scene_cloud_cover_pct"],
            current["granule_id"],
        ):
            scene_deduplicated[scene_key] = record
    values = sorted(scene_deduplicated.values(), key=lambda item: item["granule_id"])
    selected = []
    for group_number, indices in enumerate(connected_groups(values, radius_km), start=1):
        best = min(
            (values[index] for index in indices),
            key=lambda item: (
                abs(item["offset_hours"]),
                item["scene_cloud_cover_pct"],
                item["granule_id"],
            ),
        )
        selected.append({**best, "group_id": f"emit25km-{group_number:04d}"})
    return sorted(
        selected,
        key=lambda item: (
            abs(item["offset_hours"]),
            item["scene_cloud_cover_pct"],
            item["granule_id"],
        ),
    )


def write_json(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    summary = report["summary"]
    lines = [
        "# EMIT V002 time-aligned external candidates",
        "",
        f"- Public EMIT V002 catalog records: {summary['catalog_records']:,}",
        f"- CMR records queried against Sentinel-2: {summary['queries_completed']:,}",
        f"- Raw time-aligned matches: {summary['raw_candidates']:,}",
        f"- Independent source/25 km candidates: {summary['independent_candidates']:,}",
        f"- Query errors: {summary['query_errors']:,}",
        f"- Time gate: +/-{report['contract']['max_offset_hours']:.1f} h",
        f"- Scene-cloud prefilter: <={report['contract']['max_scene_cloud_pct']:.1f}%",
        "",
        "Candidate selection uses only public CMR geometry/time metadata and public Sentinel-2 catalog metadata. No detector checkpoint or prediction participates in selection.",
        "",
        "| Group | EMIT plume | Sentinel-2 scene | Offset h | Cloud % | Center lon/lat |",
        "|---|---|---|---:|---:|---|",
    ]
    for item in report["candidates"]:
        lines.append(
            f"| `{item['group_id']}` | `{item['granule_id']}` | `{item['s2_scene_id']}` | "
            f"{item['offset_hours']:.3f} | {item['scene_cloud_cover_pct']:.2f} | "
            f"{item['center'][0]:.3f}, {item['center'][1]:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Use boundary",
            "",
            "These are acquisition candidates, not accepted labels. Each scene must still pass local ROI-clear, common EMIT enhancement/uncertainty/sensitivity support, exact product-grid, plume-containment, deduplication, and two-annotator review gates. An absent catalog plume cannot create a `NO_PLUME` label.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build(args: argparse.Namespace) -> dict[str, Any]:
    root = repo_root()
    records = cmr_records()
    excluded = pilot_ids(root)
    ordered = query_order(records, excluded)
    outcomes: list[dict[str, Any]] = []
    queries_completed = 0
    for start in range(0, min(len(ordered), args.max_queries), args.workers):
        batch = ordered[start : start + args.workers]
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            results = list(
                executor.map(
                    lambda record: query_one(
                        record,
                        max_offset_hours=args.max_offset_hours,
                        max_cloud=args.max_cloud,
                    ),
                    batch,
                )
            )
        outcomes.extend(results)
        queries_completed += len(results)
        candidates = [item for item in outcomes if item["status"] == "candidate"]
        independent = independent_best(candidates, args.group_radius_km) if candidates else []
        if len(independent) >= args.target_groups:
            break
    candidates = [item for item in outcomes if item["status"] == "candidate"]
    independent = independent_best(candidates, args.group_radius_km)
    selected = independent[: args.target_groups]
    status_counts = Counter(item["status"] for item in outcomes)
    return {
        "schema_version": 1,
        "scope": "prediction_blind_emit_v002_sentinel2_external_candidate_discovery",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract": {
            "emit_collection": "EMITL2BCH4PLM.002",
            "cmr_collection_concept_id": CMR_COLLECTION_ID,
            "sentinel2_collection": S2_COLLECTION,
            "max_offset_hours": args.max_offset_hours,
            "max_scene_cloud_pct": args.max_cloud,
            "group_radius_km": args.group_radius_km,
            "one_candidate_per_source_scene": True,
            "one_candidate_per_sentinel2_scene": True,
            "pilot_granules_excluded": sorted(excluded),
            "selection_inputs": "public CMR and STAC metadata only; no model predictions",
        },
        "summary": {
            "catalog_records": len(records),
            "eligible_query_records": len(ordered),
            "queries_completed": queries_completed,
            "raw_candidates": len(candidates),
            "independent_candidates_available": len(independent),
            "independent_candidates": len(selected),
            "target_groups": args.target_groups,
            "query_errors": status_counts.get("query_error", 0),
            "status_counts": dict(sorted(status_counts.items())),
            "target_met": len(selected) >= args.target_groups,
        },
        "candidates": selected,
        "query_errors": [item for item in outcomes if item["status"] == "query_error"],
        "provenance": {
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True
            ).strip(),
            "script": "tools/build_emit_v002_time_aligned_candidates.py",
            "script_sha256": sha256(Path(__file__)),
            "python": subprocess.check_output(
                [sys.executable, "-c", "import sys; print(sys.version.split()[0])"], text=True
            ).strip(),
            "requests": requests.__version__,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-groups", type=int, default=80)
    parser.add_argument("--max-queries", type=int, default=800)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-offset-hours", type=float, default=6.0)
    parser.add_argument("--max-cloud", type=float, default=20.0)
    parser.add_argument("--group-radius-km", type=float, default=25.0)
    parser.add_argument("--output-json", default=DEFAULT_JSON.as_posix())
    parser.add_argument("--output-markdown", default=DEFAULT_MARKDOWN.as_posix())
    args = parser.parse_args()
    if min(args.target_groups, args.max_queries, args.workers) <= 0:
        parser.error("target-groups, max-queries, and workers must be positive")
    if not 0 < args.max_offset_hours <= 24:
        parser.error("max-offset-hours must be in (0, 24]")
    if not 0 <= args.max_cloud <= 100:
        parser.error("max-cloud must be in [0, 100]")
    if not 0 < args.group_radius_km <= 100:
        parser.error("group-radius-km must be in (0, 100]")
    root = repo_root()
    try:
        report = build(args)
        output_json = safe_output(root, args.output_json)
        output_markdown = safe_output(root, args.output_markdown)
        write_json(output_json, report)
        write_markdown(output_markdown, report)
    except (OSError, ValueError, requests.RequestException, json.JSONDecodeError) as exc:
        print(json.dumps({"ok": False, "error": str(exc), "error_type": type(exc).__name__}))
        return 2
    print(
        json.dumps(
            {
                "ok": report["summary"]["target_met"],
                "catalog_records": report["summary"]["catalog_records"],
                "queries_completed": report["summary"]["queries_completed"],
                "raw_candidates": report["summary"]["raw_candidates"],
                "independent_candidates": report["summary"]["independent_candidates"],
                "query_errors": report["summary"]["query_errors"],
                "output_json": output_json.relative_to(root).as_posix(),
                "output_markdown": output_markdown.relative_to(root).as_posix(),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["summary"]["target_met"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
