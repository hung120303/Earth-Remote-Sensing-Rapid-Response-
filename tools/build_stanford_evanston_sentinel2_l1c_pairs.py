#!/usr/bin/env python3
"""Freeze prior-only Evanston Sentinel-2 L1C target/reference pairs."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from build_stanford_large_controlled_release_l1c_pairs import (
    L1C_BANDS,
    assert_no_outcome_fields,
    pair_one,
    sha256,
    source_targets,
    validate_protocol,
)

DEFAULT_PROTOCOL = Path("configs/stanford_evanston_sentinel2_l1c_pair_protocol.json")
DEFAULT_MANIFEST = Path(
    ".research/stanford_controlled_release_2024_2025/evanston_l1c_stress/pair_manifest.json"
)
DEFAULT_REPORT = Path("reports/acquisition/stanford_evanston_sentinel2_l1c_pairs.json")


def repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def load_bound_protocol(path: Path) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("status") != (
        "frozen_before_evanston_reference_selection_imagery_access_and_detector_inference"
    ):
        raise ValueError("Evanston pair protocol is not frozen")
    frozen = validate_protocol(protocol)
    binding = protocol.get("source", {}).get("target_manifest", {})
    manifest_path = repo_path(str(binding.get("path", "")))
    if not manifest_path.is_file():
        raise ValueError("Bound Evanston target manifest does not exist")
    observed_hash = sha256(manifest_path)
    if observed_hash != binding.get("sha256"):
        raise ValueError("Bound Evanston target manifest hash mismatch")
    target_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if target_manifest.get("status") != "complete_outcome_blind_target_audit":
        raise ValueError("Bound Evanston target audit is incomplete")
    rows = target_manifest.get("rows")
    if not isinstance(rows, list) or len(rows) != int(binding.get("rows", -1)):
        raise ValueError("Bound Evanston target row count mismatch")
    return protocol, manifest_path, target_manifest


def source_records(target_manifest: dict[str, Any]) -> list[dict[str, Any]]:
    center = target_manifest.get("center")
    rows = target_manifest.get("rows")
    if not isinstance(center, list) or len(center) != 2 or not isinstance(rows, list):
        raise ValueError("Target manifest lacks center/rows")
    longitude, latitude = map(float, center)
    records: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("target"), dict):
            raise ValueError("Invalid target-audit row")
        target = row["target"]
        if str(target.get("datetime", ""))[:10] != str(row.get("utc_date", "")):
            raise ValueError(f"Target date mismatch for {row.get('event_id')}")
        records.append(
            {
                "release_id": row["event_id"],
                "sensor": "Sentinel-2",
                "observed_at_utc": target["datetime"],
                "latitude": latitude,
                "longitude": longitude,
                "target": {
                    "status": "resolved",
                    "id": target["scene_id"],
                    "datetime": target["datetime"],
                },
            }
        )
    assert_no_outcome_fields(records)
    return records


def write_json(path: Path, payload: dict[str, Any]) -> None:
    assert_no_outcome_fields(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_pairs(protocol_path: Path, manifest_path: Path, *, workers: int) -> dict[str, Any]:
    protocol, target_manifest_path, target_manifest = load_bound_protocol(protocol_path)
    frozen = validate_protocol(protocol)
    targets, excluded_target_ids = source_targets(source_records(target_manifest))
    reference = frozen["reference"]
    seasonal = reference["seasonal_fallback"]
    excluded_utc_dates = {
        str(value) for value in reference["additional_campaign_exclusions"]["utc_dates"]
    }
    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(
            executor.map(
                lambda target: pair_one(
                    target,
                    excluded_target_ids=excluded_target_ids,
                    min_gap_hours=float(reference["minimum_gap_hours"]),
                    max_lookback_days=int(reference["maximum_lookback_days"]),
                    seasonal_min_lookback_days=int(seasonal["minimum_lookback_days"]),
                    seasonal_max_lookback_days=int(seasonal["maximum_lookback_days"]),
                    seasonal_target_gap_days=int(seasonal["target_gap_days"]),
                    excluded_utc_dates=excluded_utc_dates,
                    max_cloud=float(reference["maximum_catalog_eo_cloud_cover_pct"]),
                ),
                targets,
            )
        )
    pairs = sorted(
        (row for row in results if row["status"] == "paired"),
        key=lambda row: (row["target"]["datetime"], row["event_id"]),
    )
    errors = sorted(
        (row for row in results if row["status"] != "paired"),
        key=lambda row: str(row.get("event_id")),
    )
    target_ids = {row["target"]["scene_id"] for row in pairs}
    reference_ids = {row["reference"]["scene_id"] for row in pairs}
    forbidden_overlap = reference_ids & excluded_target_ids
    date_overlap = sorted(
        {
            row["reference"]["datetime"][:10]
            for row in pairs
            if row["reference"]["datetime"][:10] in excluded_utc_dates
        }
    )
    if forbidden_overlap or date_overlap:
        raise ValueError("Reference selection violated frozen Evanston exclusions")
    tier_counts: dict[str, int] = {}
    tile_counts: dict[str, int] = {}
    for row in pairs:
        tier = str(row["reference_tier"])
        tile = str(row["target"]["mgrs_tile"])
        tier_counts[tier] = tier_counts.get(tier, 0) + 1
        tile_counts[tile] = tile_counts.get(tile, 0) + 1

    manifest = {
        "schema_version": 1,
        "scope": "outcome_blind_stanford_evanston_sentinel2_l1c_target_reference_pairs",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "bindings": {
            "target_manifest": {
                "path": target_manifest_path.relative_to(ROOT).as_posix(),
                "sha256": sha256(target_manifest_path),
            },
            "protocol": {
                "path": protocol_path.relative_to(ROOT).as_posix(),
                "sha256": sha256(protocol_path),
            },
            "script": {
                "path": Path(__file__).resolve().relative_to(ROOT).as_posix(),
                "sha256": sha256(Path(__file__).resolve()),
            },
        },
        "contract": {
            "reference_policy": "prior-only same-MGRS L1C; 1 hour through 31 days; catalog cloud <=20%; deterministic gap/cloud/ID order; 334-410 day seasonal fallback only if primary window empty",
            "excluded_reference_scene_ids": sorted(excluded_target_ids),
            "excluded_reference_utc_dates": sorted(excluded_utc_dates),
            "spectral_product": "Sentinel-2 Level-1C top-of-atmosphere raw DN",
            "band_order": list(L1C_BANDS),
        },
        "outcome_blindness": {
            "release_or_summary_files_opened": False,
            "detector_outputs_accessed": False,
            "event_selection_changed": False,
        },
        "summary": {
            "eligible_targets": len(targets),
            "complete_pairs": len(pairs),
            "pair_errors": len(errors),
            "unique_target_scene_ids": len(target_ids),
            "unique_reference_scene_ids": len(reference_ids),
            "references_matching_target_scene_ids": len(forbidden_overlap),
            "references_matching_campaign_dates": len(date_overlap),
            "reference_tier_counts": dict(sorted(tier_counts.items())),
            "target_mgrs_tile_counts": dict(sorted(tile_counts.items())),
            "all_pairs_complete": len(pairs) == len(targets) and not errors,
        },
        "pairs": pairs,
        "errors": errors,
        "runtime": {
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "python": sys.version.split()[0],
        },
    }
    assert_no_outcome_fields(manifest)
    write_json(manifest_path, manifest)
    return manifest


def build_report(protocol_path: Path, manifest_path: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    report = {
        "schema_version": 1,
        "status": "frozen_complete" if manifest["summary"]["all_pairs_complete"] else "frozen_with_pair_errors",
        "scope": manifest["scope"],
        "claim_boundary": "Outcome-blind pairing only; no imagery, release summaries, rates, scores, or performance claims.",
        "bindings": {
            "protocol": {
                "path": protocol_path.relative_to(ROOT).as_posix(),
                "sha256": sha256(protocol_path),
            },
            "pair_manifest": {
                "path": manifest_path.relative_to(ROOT).as_posix(),
                "sha256": sha256(manifest_path),
                "tracked": False,
            },
            "script": manifest["bindings"]["script"],
            "target_manifest": manifest["bindings"]["target_manifest"],
        },
        "contract": manifest["contract"],
        "outcome_blindness": manifest["outcome_blindness"],
        "summary": manifest["summary"],
    }
    assert_no_outcome_fields(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL.as_posix())
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST.as_posix())
    parser.add_argument("--report", default=DEFAULT_REPORT.as_posix())
    parser.add_argument("--workers", type=positive_int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    protocol_path = repo_path(args.protocol)
    _, _, target_manifest = load_bound_protocol(protocol_path)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "dry_run",
                    "protocol_sha256": sha256(protocol_path),
                    "frozen_targets": len(target_manifest["rows"]),
                },
                sort_keys=True,
            )
        )
        return

    manifest_path = repo_path(args.manifest)
    manifest = build_pairs(protocol_path, manifest_path, workers=args.workers)
    report = build_report(protocol_path, manifest_path, manifest)
    write_json(repo_path(args.report), report)
    print(json.dumps(report["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
