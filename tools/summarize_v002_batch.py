#!/usr/bin/env python3
"""Create a small, token-free integrity manifest for an ignored V002 batch."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import rasterio
from rasterio.warp import transform_geom

from acquire_v002_pilot import BAND_ORDER, geometry_bounds, repo_root, sha256, write_json


def safe_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if root != path and root not in path.parents:
        raise ValueError(f"Path escapes repository root: {relative}")
    return path


def tracked_worktree_dirty(root: Path) -> bool:
    result = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return bool(result.stdout.strip())


def git_commit(root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def scene_identity(root: Path, scene: dict[str, Any]) -> dict[str, Any]:
    stack_path = safe_path(root, scene["stack"])
    mask_path = safe_path(root, scene["mask"])
    observed_stack_hash = sha256(stack_path)
    observed_mask_hash = sha256(mask_path)
    if observed_stack_hash != scene["stack_sha256"] or observed_mask_hash != scene["mask_sha256"]:
        raise ValueError(f"Hash mismatch for {scene['scene_id']}")
    with rasterio.open(mask_path) as source:
        mask_positive_pixels = int((source.read(1) > 0).sum())
        pixels = source.width * source.height
    return {
        "scene_id": scene["scene_id"],
        "datetime": scene["datetime"],
        "offset_hours": float(scene["offset_hours"]),
        "roi_clear_pct": float(scene["roi_clear_pct"]),
        "scene_cloud_cover_pct": float(scene["scene_cloud_cover_pct"]),
        "product_level": scene["product_level"],
        "band_order": scene["band_order"],
        "resolution_m": float(scene["resolution_m"]),
        "shape": scene["shape"],
        "crs": scene["crs"],
        "transform": scene["transform"],
        "stack_sha256": observed_stack_hash,
        "mask_sha256": observed_mask_hash,
        "mask_positive_pixels": mask_positive_pixels,
        "mask_positive_pct": 100.0 * mask_positive_pixels / pixels,
    }


def summarize_granule(root: Path, manifest_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    plume_path = safe_path(root, manifest["plume_geojson"])
    plume_hash = sha256(plume_path)
    if plume_hash != manifest["plume_geojson_sha256"]:
        raise ValueError(f"Plume GeoJSON hash mismatch: {manifest['granule_id']}")
    feature = json.loads(plume_path.read_text(encoding="utf-8"))
    scenes = [scene_identity(root, scene) for scene in manifest["scenes"]]
    if len(scenes) != 2:
        raise ValueError(f"Expected two bracketing scenes for {manifest['granule_id']}; got {len(scenes)}")

    first_mask_path = safe_path(root, manifest["scenes"][0]["mask"])
    with rasterio.open(first_mask_path) as source:
        projected = transform_geom("EPSG:4326", source.crs, feature["geometry"], precision=3)
        plume_bounds = geometry_bounds(projected)
        grid_bounds = (source.bounds.left, source.bounds.bottom, source.bounds.right, source.bounds.top)
        resolution = abs(float(source.transform.a))
    margins = (
        plume_bounds[0] - grid_bounds[0],
        plume_bounds[1] - grid_bounds[1],
        grid_bounds[2] - plume_bounds[2],
        grid_bounds[3] - plume_bounds[3],
    )
    fully_contained = min(margins) >= 0.0
    min_clear = min(scene["roi_clear_pct"] for scene in scenes)
    mask_pct = scenes[0]["mask_positive_pct"]
    flags = {
        "bracketed": min(scene["offset_hours"] for scene in scenes) <= 0 <= max(
            scene["offset_hours"] for scene in scenes
        ),
        "minimum_roi_clear_pct_at_least_70": min_clear >= 70.0,
        "mask_coverage_between_1_and_50_pct": 1.0 <= mask_pct <= 50.0,
        "plume_fully_contained": fully_contained,
    }
    return {
        "granule_id": manifest["granule_id"],
        "emit_datetime": manifest["emit_datetime"],
        "emit_collection": manifest["emit_collection"],
        "label_type": manifest["label_type"],
        "source_manifest_sha256": sha256(manifest_path),
        "plume_geojson_sha256": plume_hash,
        "plume_bounds_projected": [round(float(value), 3) for value in plume_bounds],
        "grid_bounds_projected": [round(float(value), 3) for value in grid_bounds],
        "plume_min_margin_pixels": round(min(margins) / resolution, 3),
        "quality_flags": flags,
        "benchmark_eligible": all(flags.values()),
        "scenes": scenes,
    }


def build_summary(batch_dir: Path) -> dict[str, Any]:
    root = repo_root()
    manifests = sorted(batch_dir.glob("*/manifest.json"))
    if not manifests:
        raise FileNotFoundError(f"No manifests found under {batch_dir}")
    granules = [summarize_granule(root, path) for path in manifests]
    exclusions: Counter[str] = Counter()
    for granule in granules:
        for flag, passed in granule["quality_flags"].items():
            if not passed:
                exclusions[flag] += 1

    identity = hashlib.sha256()
    for granule in granules:
        identity.update(granule["granule_id"].encode("utf-8") + b"\0")
        identity.update(granule["source_manifest_sha256"].encode("ascii") + b"\0")
        identity.update(granule["plume_geojson_sha256"].encode("ascii") + b"\0")
        for scene in granule["scenes"]:
            identity.update(scene["stack_sha256"].encode("ascii") + b"\0")
            identity.update(scene["mask_sha256"].encode("ascii") + b"\0")

    first_manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
    selection = first_manifest["sentinel2_selection"]
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "batch_name": batch_dir.name,
        "emit_collection": "EMITL2BCH4PLM.002",
        "cmr_collection_concept_id": "C3242707413-LPCLOUD",
        "sentinel2_collection": "sentinel-2-l2a",
        "band_order": list(BAND_ORDER),
        "acquisition_contract": {
            "temporal_mode": selection["temporal_mode"],
            "window_days": selection["window_days"],
            "max_scene_cloud_pct": selection["max_scene_cloud_pct"],
            "tile_size": granules[0]["scenes"][0]["shape"],
            "resolution_m": granules[0]["scenes"][0]["resolution_m"],
            "protected_emit_cogs_included": False,
        },
        "integrity": {
            "granules": len(granules),
            "sentinel2_stacks": sum(len(item["scenes"]) for item in granules),
            "hash_mismatches": 0,
            "benchmark_eligible_groups": sum(item["benchmark_eligible"] for item in granules),
            "quality_failures": dict(exclusions),
            "batch_identity_sha256": identity.hexdigest(),
        },
        "provenance": {
            "git_commit": git_commit(root),
            "git_tracked_worktree_dirty": tracked_worktree_dirty(root),
            "script": "tools/summarize_v002_batch.py",
            "script_sha256": sha256(Path(__file__)),
            "python": sys.version.split()[0],
            "rasterio": rasterio.__version__,
        },
        "granules": granules,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--batch-dir",
        default="EarthRemoteSensingRapidResponse/Data Collection/s2_emit_pairs/emit-v002-2026-07",
    )
    parser.add_argument(
        "--output",
        default="reports/acquisition/emit_v002_2026_07_batch.json",
    )
    args = parser.parse_args()
    root = repo_root()
    batch_base = (
        root / "EarthRemoteSensingRapidResponse" / "Data Collection" / "s2_emit_pairs"
    ).resolve()
    batch_dir = (root / args.batch_dir).resolve()
    if batch_base not in batch_dir.parents:
        parser.error("batch-dir must resolve beneath the ignored s2_emit_pairs directory")
    summary = build_summary(batch_dir)
    output = (root / args.output).resolve()
    if root not in output.parents:
        parser.error("output must resolve beneath the repository root")
    write_json(output, summary)
    print(
        json.dumps(
            {
                "ok": True,
                "output": output.relative_to(root).as_posix(),
                "granules": summary["integrity"]["granules"],
                "scenes": summary["integrity"]["sentinel2_stacks"],
                "benchmark_eligible_groups": summary["integrity"]["benchmark_eligible_groups"],
                "batch_identity_sha256": summary["integrity"]["batch_identity_sha256"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
