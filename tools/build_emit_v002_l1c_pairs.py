#!/usr/bin/env python3
"""Freeze L1C target/reference pairs for prediction-blind EMIT candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path
from typing import Any

import requests

from acquire_v002_pilot import EARTH_SEARCH_URL, parse_datetime, repo_root

DEFAULT_INPUT = Path("reports/acquisition/emit_v002_time_aligned_candidates.json")
DEFAULT_JSON = Path("reports/acquisition/emit_v002_l1c_pairs.json")
DEFAULT_MARKDOWN = Path("reports/acquisition/EMIT_V002_L1C_PAIRS.md")
L1C_COLLECTION = "sentinel-2-l1c"
L2A_COLLECTION = "sentinel-2-l2a"
L1C_BANDS = {
    "B02": "blue",
    "B03": "green",
    "B04": "red",
    "B08": "nir",
    "B11": "swir16",
    "B12": "swir22",
}
REQUIRED_L1C_ASSETS = set(L1C_BANDS.values())
REQUIRED_L2A_ASSETS = {"scl"}
SCENE_ID = re.compile(r"^(S2[ABC])_(\d{2}[A-Z]{3})_(\d{8})_(\d+)_(L1C|L2A)$")
L1C_HTTPS_PREFIX = "https://sentinel-s2-l1c.s3.amazonaws.com/"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
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
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    error: Exception | None = None
    for attempt in range(3):
        try:
            response = requests.request(
                method,
                url,
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
            error = exc
            if attempt < 2:
                time.sleep(float(2**attempt))
    assert error is not None
    raise error


def item_url(collection: str, scene_id: str) -> str:
    return (
        "https://earth-search.aws.element84.com/v1/collections/"
        f"{collection}/items/{scene_id}"
    )


def mgrs_tile(scene_id: str) -> str:
    match = SCENE_ID.fullmatch(scene_id)
    if match is None:
        raise ValueError(f"Unexpected Sentinel-2 item ID: {scene_id}")
    return match.group(2)


def corresponding_l1c_id(l2a_id: str) -> str:
    match = SCENE_ID.fullmatch(l2a_id)
    if match is None or match.group(5) != "L2A":
        raise ValueError(f"Expected an L2A Sentinel-2 item ID, got {l2a_id}")
    return l2a_id[:-3] + "L1C"


def contains_center(item: dict[str, Any], center: list[float]) -> bool:
    bbox = item.get("bbox")
    return bool(
        bbox
        and len(bbox) >= 4
        and bbox[0] <= center[0] <= bbox[2]
        and bbox[1] <= center[1] <= bbox[3]
    )


def validate_item(
    item: dict[str, Any],
    *,
    expected_id: str,
    center: list[float],
    required_assets: set[str],
) -> None:
    if item.get("id") != expected_id:
        raise ValueError(f"STAC returned {item.get('id')!r}, expected {expected_id!r}")
    if not contains_center(item, center):
        raise ValueError(f"Scene {expected_id} does not cover plume center")
    missing = sorted(required_assets - set(item.get("assets", {})))
    if missing:
        raise ValueError(f"Scene {expected_id} lacks required assets: {missing}")


def public_item(collection: str, scene_id: str) -> dict[str, Any]:
    return request_json("GET", item_url(collection, scene_id))


def official_l1c_url(catalog_href: str) -> tuple[str, bool]:
    prefixes = (
        "s3://sentinel-s2-l1c/",
        "s3://sentinel-s2-l2a/",
        "https://sentinel-s2-l1c.s3.amazonaws.com/",
        "https://sentinel-s2-l2a.s3.amazonaws.com/",
        "https://roda.sentinel-hub.com/sentinel-s2-l1c/",
    )
    for prefix in prefixes:
        if catalog_href.startswith(prefix):
            path = catalog_href[len(prefix) :]
            corrected = "l2a" in prefix
            return L1C_HTTPS_PREFIX + path, corrected
    raise ValueError(f"Unsupported Sentinel-2 L1C catalog href: {catalog_href}")


def validated_l1c_assets(item: dict[str, Any]) -> dict[str, Any]:
    assets = item.get("assets", {})
    spectral: dict[str, str] = {}
    corrections = 0
    for band, role in L1C_BANDS.items():
        url, corrected = official_l1c_url(str(assets[role]["href"]))
        spectral[band] = url
        corrections += int(corrected)
    tileinfo_href = str(assets.get("tileinfo_metadata", {}).get("href", ""))
    tileinfo_url, tileinfo_corrected = official_l1c_url(tileinfo_href)
    tileinfo = request_json("GET", tileinfo_url)
    expected_product = str(item.get("properties", {}).get("s2:product_uri", "")).removesuffix(
        ".SAFE"
    )
    observed_product = str(tileinfo.get("productName", ""))
    if not expected_product or observed_product != expected_product:
        raise ValueError(
            f"Official L1C tile metadata mismatch: {observed_product!r} vs {expected_product!r}"
        )
    return {
        "spectral_assets": spectral,
        "tileinfo_metadata": tileinfo_url,
        "product_name": observed_product,
        "catalog_asset_hrefs_corrected_from_l2a_bucket": corrections
        + int(tileinfo_corrected),
    }


def reference_l2a(
    candidate: dict[str, Any], *, max_cloud: float, max_reference_days: int
) -> dict[str, Any]:
    target_time = parse_datetime(candidate["s2_datetime"])
    target_tile = mgrs_tile(candidate["s2_scene_id"])
    start = target_time - timedelta(days=max_reference_days)
    end = target_time - timedelta(hours=1)
    response = request_json(
        "POST",
        EARTH_SEARCH_URL,
        payload={
            "collections": [L2A_COLLECTION],
            "bbox": candidate["bbox"],
            "datetime": (
                f"{start.isoformat().replace('+00:00', 'Z')}/"
                f"{end.isoformat().replace('+00:00', 'Z')}"
            ),
            "limit": 100,
            "query": {"eo:cloud_cover": {"lte": max_cloud}},
        },
    )
    eligible = []
    for item in response.get("features", []):
        scene_id = str(item.get("id", ""))
        if scene_id == candidate["s2_scene_id"]:
            continue
        try:
            tile = mgrs_tile(scene_id)
        except ValueError:
            continue
        if tile != target_tile or not contains_center(item, candidate["center"]):
            continue
        if not REQUIRED_L2A_ASSETS.issubset(item.get("assets", {})):
            continue
        scene_time = parse_datetime(item["properties"]["datetime"])
        gap_hours = (target_time - scene_time).total_seconds() / 3600.0
        if not 1.0 <= gap_hours <= max_reference_days * 24.0 + 1e-9:
            continue
        eligible.append(
            (
                gap_hours,
                float(item.get("properties", {}).get("eo:cloud_cover", 100.0)),
                scene_id,
                scene_time,
                item,
            )
        )
    if not eligible:
        raise ValueError(
            f"No prior same-tile L2A reference within {max_reference_days} days for "
            f"{candidate['s2_scene_id']}"
        )
    return min(eligible, key=lambda value: (value[0], value[1], value[2]))[-1]


def compact_item(
    l2a: dict[str, Any], l1c: dict[str, Any], *, center: list[float]
) -> dict[str, Any]:
    l2a_id = str(l2a["id"])
    l1c_id = str(l1c["id"])
    validate_item(
        l2a,
        expected_id=l2a_id,
        center=center,
        required_assets=REQUIRED_L2A_ASSETS,
    )
    validate_item(
        l1c,
        expected_id=l1c_id,
        center=center,
        required_assets=REQUIRED_L1C_ASSETS,
    )
    l2a_time = parse_datetime(l2a["properties"]["datetime"])
    l1c_time = parse_datetime(l1c["properties"]["datetime"])
    acquisition_difference = abs((l1c_time - l2a_time).total_seconds())
    l1c_datatake = l1c.get("properties", {}).get("s2:datatake_id")
    l2a_datatake = l2a.get("properties", {}).get("s2:datatake_id")
    if mgrs_tile(l1c_id) != mgrs_tile(l2a_id):
        raise ValueError(f"L1C/L2A MGRS mismatch: {l1c_id} vs {l2a_id}")
    if not l1c_datatake or l1c_datatake != l2a_datatake:
        raise ValueError(f"L1C/L2A datatake mismatch: {l1c_id} vs {l2a_id}")
    if acquisition_difference > 30.0:
        raise ValueError(f"L1C/L2A time mismatch: {l1c_id} vs {l2a_id}")
    l1c_assets = validated_l1c_assets(l1c)
    return {
        "l1c_scene_id": l1c_id,
        "l1c_stac_item": item_url(L1C_COLLECTION, l1c_id),
        "l2a_scene_id": l2a_id,
        "l2a_stac_item": item_url(L2A_COLLECTION, l2a_id),
        "datetime": l1c_time.isoformat(),
        "mgrs_tile": mgrs_tile(l1c_id),
        "sentinel2_datatake_id": l1c_datatake,
        "scene_cloud_cover_pct": round(
            float(l2a.get("properties", {}).get("eo:cloud_cover", 100.0)), 6
        ),
        "l1c_l2a_acquisition_difference_seconds": round(acquisition_difference, 6),
        "l1c_spectral_asset_roles": dict(L1C_BANDS),
        "l1c_spectral_assets": l1c_assets["spectral_assets"],
        "l1c_tileinfo_metadata": l1c_assets["tileinfo_metadata"],
        "l1c_product_name": l1c_assets["product_name"],
        "l1c_catalog_asset_hrefs_corrected_from_l2a_bucket": l1c_assets[
            "catalog_asset_hrefs_corrected_from_l2a_bucket"
        ],
        "l2a_observability_asset_role": "scl",
        "l2a_scl_asset": str(l2a["assets"]["scl"]["href"]),
    }


def pair_one(
    candidate: dict[str, Any], *, max_cloud: float, max_reference_days: int
) -> dict[str, Any]:
    try:
        target_l2a = public_item(L2A_COLLECTION, candidate["s2_scene_id"])
        target_l1c_id = corresponding_l1c_id(candidate["s2_scene_id"])
        target_l1c = public_item(L1C_COLLECTION, target_l1c_id)
        reference_l2a_item = reference_l2a(
            candidate,
            max_cloud=max_cloud,
            max_reference_days=max_reference_days,
        )
        reference_l1c_id = corresponding_l1c_id(str(reference_l2a_item["id"]))
        reference_l1c = public_item(L1C_COLLECTION, reference_l1c_id)
        target = compact_item(target_l2a, target_l1c, center=candidate["center"])
        reference = compact_item(
            reference_l2a_item, reference_l1c, center=candidate["center"]
        )
        target_time = parse_datetime(target["datetime"])
        reference_time = parse_datetime(reference["datetime"])
        return {
            "status": "paired",
            "group_id": candidate["group_id"],
            "granule_id": candidate["granule_id"],
            "plume_id": candidate.get("plume_id"),
            "source_scenes": candidate.get("source_scenes", []),
            "center": candidate["center"],
            "bbox": candidate["bbox"],
            "emit_datetime": candidate["emit_datetime"],
            "emit_to_target_offset_hours": candidate["offset_hours"],
            "target": target,
            "reference": reference,
            "reference_to_target_gap_hours": round(
                (target_time - reference_time).total_seconds() / 3600.0, 6
            ),
        }
    except Exception as exc:
        return {
            "status": "pair_error",
            "group_id": candidate.get("group_id"),
            "granule_id": candidate.get("granule_id"),
            "error_type": type(exc).__name__,
            "error": str(exc)[:500],
        }


def write_json(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    summary = report["summary"]
    lines = [
        "# EMIT external Sentinel-2 L1C pair manifest",
        "",
        f"- Prediction-blind EMIT candidates: {summary['input_candidates']}",
        f"- Complete target/reference pairs: {summary['complete_pairs']}",
        f"- Pairing errors: {summary['pair_errors']}",
        f"- Unique L1C targets: {summary['unique_l1c_targets']}",
        f"- Unique L1C references: {summary['unique_l1c_references']}",
        f"- Catalog L1C hrefs corrected from the L2A bucket: {summary['catalog_l1c_hrefs_corrected_from_l2a_bucket']}",
        f"- Maximum reference lookback: {report['contract']['max_reference_days']} days",
        "",
        "## Architecture contract",
        "",
        "The detector inputs use six Sentinel-2 L1C/TOA bands (`B02,B03,B04,B08,B11,B12`) to match the MARS-S2L training product. Each L1C target/reference is paired with the co-temporal same-tile L2A item only to obtain the SCL observability mask. References are the nearest prior same-tile scenes passing the catalog cloud prefilter; no future reference and no model prediction is used.",
        "",
        "The current Earth Search L1C item identities are valid, but some of their asset hrefs resolve to the L2A bucket. The manifest therefore reconstructs every tile path against the official public `sentinel-s2-l1c` bucket and verifies each official `tileInfo.json` product name against the STAC L1C product URI before accepting the pair.",
        "",
        "| Group | EMIT plume | L1C target | L1C reference | EMIT offset h | Reference gap h |",
        "|---|---|---|---|---:|---:|",
    ]
    for item in report["pairs"]:
        lines.append(
            f"| `{item['group_id']}` | `{item['granule_id']}` | "
            f"`{item['target']['l1c_scene_id']}` | `{item['reference']['l1c_scene_id']}` | "
            f"{item['emit_to_target_offset_hours']:.3f} | "
            f"{item['reference_to_target_gap_hours']:.2f} |"
        )
    lines.extend(
        [
            "",
            "This is a catalog contract, not a passed raster-quality gate. Public L1C bands and co-temporal L2A SCL crops must still be downloaded, regridded to the native 200x200 10 m model grid, and checked for local clear support before protected EMIT expansion.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build(args: argparse.Namespace) -> dict[str, Any]:
    root = repo_root()
    input_path = safe_output(root, args.input)
    source = json.loads(input_path.read_text(encoding="utf-8"))
    candidates = source.get("candidates", [])
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        outcomes = list(
            executor.map(
                lambda candidate: pair_one(
                    candidate,
                    max_cloud=args.max_cloud,
                    max_reference_days=args.max_reference_days,
                ),
                candidates,
            )
        )
    pairs = sorted(
        (item for item in outcomes if item["status"] == "paired"),
        key=lambda item: item["group_id"],
    )
    errors = [item for item in outcomes if item["status"] != "paired"]
    target_ids = {item["target"]["l1c_scene_id"] for item in pairs}
    reference_ids = {item["reference"]["l1c_scene_id"] for item in pairs}
    corrected_hrefs = sum(
        int(item[role]["l1c_catalog_asset_hrefs_corrected_from_l2a_bucket"])
        for item in pairs
        for role in ("target", "reference")
    )
    return {
        "schema_version": 1,
        "scope": "prediction_blind_emit_external_l1c_target_reference_pairs",
        "contract": {
            "input_manifest": input_path.relative_to(root).as_posix(),
            "input_manifest_sha256": sha256(input_path),
            "spectral_product": "Sentinel-2 L1C top-of-atmosphere",
            "spectral_band_order": list(L1C_BANDS),
            "observability_product": "co-temporal same-tile Sentinel-2 L2A SCL",
            "l1c_asset_authority": "official public sentinel-s2-l1c AWS bucket with tileInfo product-name verification",
            "reference_policy": "nearest prior same-tile scene; no future reference",
            "max_reference_days": args.max_reference_days,
            "max_catalog_scene_cloud_pct": args.max_cloud,
            "selection_inputs": "public CMR/STAC metadata only; no detector predictions",
        },
        "summary": {
            "input_candidates": len(candidates),
            "complete_pairs": len(pairs),
            "pair_errors": len(errors),
            "unique_l1c_targets": len(target_ids),
            "unique_l1c_references": len(reference_ids),
            "catalog_l1c_hrefs_corrected_from_l2a_bucket": corrected_hrefs,
            "all_pairs_complete": len(pairs) == len(candidates) and not errors,
        },
        "pairs": pairs,
        "errors": errors,
        "provenance": {
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True
            ).strip(),
            "script": "tools/build_emit_v002_l1c_pairs.py",
            "script_sha256": sha256(Path(__file__)),
            "python": sys.version.split()[0],
            "requests": requests.__version__,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_INPUT.as_posix())
    parser.add_argument("--output-json", default=DEFAULT_JSON.as_posix())
    parser.add_argument("--output-markdown", default=DEFAULT_MARKDOWN.as_posix())
    parser.add_argument("--max-reference-days", type=int, default=31)
    parser.add_argument("--max-cloud", type=float, default=20.0)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.max_reference_days <= 90:
        parser.error("max-reference-days must be 1..90")
    if not 0 <= args.max_cloud <= 100:
        parser.error("max-cloud must be in [0,100]")
    if args.workers <= 0:
        parser.error("workers must be positive")
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
                "ok": report["summary"]["all_pairs_complete"],
                "input_candidates": report["summary"]["input_candidates"],
                "complete_pairs": report["summary"]["complete_pairs"],
                "pair_errors": report["summary"]["pair_errors"],
                "output_json": output_json.relative_to(root).as_posix(),
                "output_markdown": output_markdown.relative_to(root).as_posix(),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["summary"]["all_pairs_complete"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
