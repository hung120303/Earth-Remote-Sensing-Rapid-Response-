#!/usr/bin/env python3
"""Audit frozen Evanston Sentinel-2 L1C targets without opening outcomes or imagery."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, time, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from build_stanford_large_controlled_release_l1c_pairs import (
    EARTH_SEARCH_URL,
    L1C_COLLECTION,
    REQUIRED_ASSETS,
    assert_no_outcome_fields,
    compact_l1c_item,
    contains_center,
    mgrs_tile,
    parse_datetime,
    request_json,
    sha256,
    validate_catalog_item,
)

DEFAULT_PROTOCOL = Path(
    "configs/stanford_evanston_sentinel2_l1c_target_audit_protocol.json"
)
DEFAULT_MANIFEST = Path(
    ".research/stanford_controlled_release_2024_2025/evanston_l1c_audit/target_manifest.json"
)
DEFAULT_REPORT = Path(
    "reports/acquisition/stanford_evanston_sentinel2_l1c_target_audit.json"
)
EXPECTED_PLATFORMS = {"S2A", "S2B"}


def repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def load_protocol(path: Path) -> dict[str, Any]:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("status") != (
        "frozen_before_evanston_flow_summary_access_and_before_detector_inference"
    ):
        raise ValueError("Evanston target-audit protocol is not frozen")
    site = protocol.get("site")
    events = protocol.get("events")
    contract = protocol.get("catalog_contract")
    if not isinstance(site, dict) or not isinstance(events, list) or not events:
        raise ValueError("Protocol lacks frozen site/events")
    if not isinstance(contract, dict):
        raise ValueError("Protocol lacks catalog contract")
    if contract.get("endpoint") != EARTH_SEARCH_URL:
        raise ValueError("Catalog endpoint differs from frozen Earth Search endpoint")
    if contract.get("collection") != L1C_COLLECTION:
        raise ValueError("Catalog collection differs from frozen Sentinel-2 L1C collection")
    if set(contract.get("required_assets", [])) != REQUIRED_ASSETS:
        raise ValueError("Required L1C assets differ from the frozen contract")
    try:
        longitude = float(site["longitude"])
        latitude = float(site["latitude"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Protocol site coordinate is invalid") from exc
    if not (-180.0 <= longitude <= 180.0 and -90.0 <= latitude <= 90.0):
        raise ValueError("Protocol site coordinate is out of range")

    observed_ids: set[str] = set()
    for event in events:
        if not isinstance(event, dict):
            raise ValueError("Each frozen event must be an object")
        event_id = str(event.get("event_id", ""))
        platform = str(event.get("platform", ""))
        try:
            event_date = date.fromisoformat(str(event["utc_date"]))
        except (KeyError, ValueError) as exc:
            raise ValueError(f"Invalid frozen UTC date for {event_id!r}") from exc
        expected_id = f"{event_date:%m%d%Y}_{platform}"
        if platform not in EXPECTED_PLATFORMS or event_id != expected_id:
            raise ValueError(f"Frozen event identity is inconsistent: {event_id!r}")
        if event_id in observed_ids:
            raise ValueError(f"Duplicate frozen event: {event_id}")
        observed_ids.add(event_id)
    assert_no_outcome_fields(events)
    return protocol


def search_target_items(event: dict[str, Any], center: list[float]) -> list[dict[str, Any]]:
    event_date = date.fromisoformat(str(event["utc_date"]))
    start = datetime.combine(event_date, time.min, tzinfo=timezone.utc)
    end = datetime.combine(event_date, time.max, tzinfo=timezone.utc)
    response = request_json(
        "POST",
        EARTH_SEARCH_URL,
        payload={
            "collections": [L1C_COLLECTION],
            "intersects": {"type": "Point", "coordinates": center},
            "datetime": (
                f"{start.isoformat().replace('+00:00', 'Z')}/"
                f"{end.isoformat().replace('+00:00', 'Z')}"
            ),
            "limit": 100,
        },
    )
    features = response.get("features")
    if not isinstance(features, list):
        raise ValueError("Earth Search response has a non-list features member")
    return features


def select_target_item(
    features: list[dict[str, Any]],
    event: dict[str, Any],
    center: list[float],
) -> tuple[dict[str, Any], int]:
    platform = str(event["platform"])
    event_date = date.fromisoformat(str(event["utc_date"]))
    expected_compact_date = event_date.strftime("%Y%m%d")
    eligible: list[tuple[float, str, dict[str, Any]]] = []

    for item in features:
        if not isinstance(item, dict):
            continue
        scene_id = str(item.get("id", ""))
        if not scene_id.startswith(f"{platform}_"):
            continue
        try:
            if scene_id.split("_")[2] != expected_compact_date:
                continue
            mgrs_tile(scene_id)
            acquired = parse_datetime(str(item.get("properties", {}).get("datetime", "")))
            cloud = float(item.get("properties", {})["eo:cloud_cover"])
        except (KeyError, TypeError, ValueError):
            continue
        if acquired.date() != event_date or not contains_center(item, center):
            continue
        if item.get("collection") not in {None, L1C_COLLECTION}:
            continue
        if not REQUIRED_ASSETS.issubset(item.get("assets", {})):
            continue
        eligible.append((cloud, scene_id, item))

    if not eligible:
        raise ValueError(
            f"No exact-date {platform} Sentinel-2 L1C product covers {event['event_id']}"
        )
    _, selected_id, selected = min(eligible, key=lambda row: (row[0], row[1]))
    validate_catalog_item(selected, expected_id=selected_id, center=center)
    return selected, len(eligible)


def audit_event(event: dict[str, Any], center: list[float]) -> dict[str, Any]:
    features = search_target_items(event, center)
    selected, eligible_count = select_target_item(features, event, center)
    scene_id = str(selected["id"])
    compact = compact_l1c_item(selected, expected_id=scene_id, center=center)
    result = {
        "event_id": event["event_id"],
        "utc_date": event["utc_date"],
        "platform": event["platform"],
        "catalog_features_returned": len(features),
        "eligible_exact_products": eligible_count,
        "target": compact,
    }
    assert_no_outcome_fields(result)
    return result


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_audit(
    protocol_path: Path,
    manifest_path: Path,
    report_path: Path,
    *,
    limit: int | None = None,
) -> dict[str, Any]:
    protocol = load_protocol(protocol_path)
    center = [float(protocol["site"]["longitude"]), float(protocol["site"]["latitude"])]
    events = list(protocol["events"])
    if limit is not None:
        events = events[:limit]

    rows = [audit_event(event, center) for event in events]
    selected_ids = [row["target"]["scene_id"] for row in rows]
    if len(selected_ids) != len(set(selected_ids)):
        raise ValueError("Target audit selected a duplicate Sentinel-2 scene")
    tiles = sorted({row["target"]["mgrs_tile"] for row in rows})
    manifest = {
        "schema_version": 1,
        "status": "complete_outcome_blind_target_audit",
        "center": center,
        "rows": rows,
    }
    assert_no_outcome_fields(manifest["rows"])
    write_json(manifest_path, manifest)

    report = {
        "schema_version": 1,
        "status": "complete_outcome_blind_target_audit",
        "scope": protocol["scope"],
        "claim_boundary": protocol["claim_boundary"],
        "site": {
            "name": protocol["site"]["name"],
            "address": protocol["site"]["address"],
            "center": center,
            "coordinate_match_score": protocol["site"]["geocoder_match_score"],
            "coordinate_match_type": protocol["site"]["geocoder_match_type"],
        },
        "summary": {
            "frozen_events": len(protocol["events"]),
            "audited_events": len(rows),
            "validated_targets": len(rows),
            "unique_targets": len(set(selected_ids)),
            "mgrs_tiles": tiles,
            "imagery_downloaded": False,
            "release_or_summary_files_opened": False,
        },
        "bindings": {
            "protocol": {
                "path": protocol_path.relative_to(ROOT).as_posix(),
                "sha256": sha256(protocol_path),
            },
            "manifest": {
                "path": manifest_path.relative_to(ROOT).as_posix(),
                "sha256": sha256(manifest_path),
            },
            "script": {
                "path": Path(__file__).resolve().relative_to(ROOT).as_posix(),
                "sha256": sha256(Path(__file__).resolve()),
            },
        },
    }
    write_json(report_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL.as_posix())
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST.as_posix())
    parser.add_argument("--report", default=DEFAULT_REPORT.as_posix())
    parser.add_argument("--limit", type=positive_int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    protocol_path = repo_path(args.protocol)
    protocol = load_protocol(protocol_path)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "dry_run",
                    "frozen_events": len(protocol["events"]),
                    "protocol_sha256": sha256(protocol_path),
                },
                sort_keys=True,
            )
        )
        return

    report = run_audit(
        protocol_path,
        repo_path(args.manifest),
        repo_path(args.report),
        limit=args.limit,
    )
    print(json.dumps(report["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
