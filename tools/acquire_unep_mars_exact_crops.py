#!/usr/bin/env python3
"""Acquire resumable 200x200 exact-product crops for nonsealed scene cohorts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.warp import transform_geom

from acquire_emit_v002_l1c_crops import (
    aligned_grid,
    read_to_grid,
    safe_repo_path,
    sha256,
    write_raster,
)


DEFAULT_COHORT = Path(".research/unep_mars_post2024/eligible_manifest.jsonl")
DEFAULT_ASSETS = Path(".research/unep_mars_post2024/nonsealed_exact_assets.jsonl")
DEFAULT_OUTPUT = Path(".research/unep_mars_post2024/crops")
DEFAULT_JSON = Path("reports/acquisition/unep_mars_post2024_crop_acquisition.json")
DEFAULT_MARKDOWN = Path("reports/acquisition/UNEP_MARS_POST2024_CROP_ACQUISITION.md")

SIZE = 200
RESOLUTION = 10.0
S2_SOURCE_BANDS = ("B02", "B03", "B04", "B08", "B11", "B12")
S2_DESCRIPTIONS = S2_SOURCE_BANDS
LANDSAT_SOURCE_BANDS = ("B2", "B3", "B4", "B5", "B6", "B7")
LANDSAT_DESCRIPTIONS = ("B02", "B03", "B04", "B05", "B06", "B07")


def repo_root() -> Path:
    value = subprocess.check_output(
        ["git", "rev-parse", "--show-toplevel"], text=True
    ).strip()
    if not value:
        raise RuntimeError("Could not resolve repository root")
    return Path(value).resolve()


def file_record(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def stable_identity(cohort: dict[str, Any], assets: dict[str, Any]) -> str:
    payload = {"cohort": cohort, "assets": assets}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def source_contract(sensor: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if sensor == "Sentinel-2":
        return S2_SOURCE_BANDS, S2_DESCRIPTIONS
    if sensor == "Landsat":
        return LANDSAT_SOURCE_BANDS, LANDSAT_DESCRIPTIONS
    raise ValueError(f"Unsupported sensor: {sensor}")


def landsat_cloud_classes(qa: np.ndarray) -> np.ndarray:
    """Map C2 QA_PIXEL to MARS-compatible 0=clear, 1=cloud/invalid."""
    if not np.issubdtype(qa.dtype, np.integer):
        raise ValueError("QA_PIXEL must be an integer array")
    nonclear_bits = sum(1 << bit for bit in (0, 1, 2, 3, 4, 5))
    return ((qa.astype(np.uint16) & nonclear_bits) != 0).astype(np.uint8)


def read_stack(
    assets: dict[str, str],
    bands: tuple[str, ...],
    *,
    crs: Any,
    transform: Any,
) -> np.ndarray:
    return np.stack(
        [
            read_to_grid(
                assets[band],
                crs=crs,
                transform=transform,
                size=SIZE,
                resampling=Resampling.bilinear,
                dtype="uint16",
            )
            for band in bands
        ]
    )


def rasterized_plumes(
    geometries: list[dict[str, Any]], *, crs: Any, transform: Any
) -> np.ndarray:
    projected = [
        transform_geom("EPSG:4326", crs, geometry, precision=3)
        for geometry in geometries
        if geometry and geometry.get("type") in {"Polygon", "MultiPolygon"}
    ]
    if not projected:
        return np.zeros((SIZE, SIZE), dtype=np.uint8)
    return rasterize(
        [(geometry, 1) for geometry in projected],
        out_shape=(SIZE, SIZE),
        transform=transform,
        fill=0,
        dtype="uint8",
        all_touched=True,
    )


def geometry_gate(label_state: str, plume_pixels: int) -> bool:
    """Require label-consistent geometry without weakening positive-cohort checks."""
    if label_state == "NO_PLUME":
        return plume_pixels == 0
    if label_state == "PLUME":
        return plume_pixels > 0
    raise ValueError(f"Unsupported label_state: {label_state}")


def select_shard(
    records: list[dict[str, Any]], shard_count: int, shard_index: int
) -> list[dict[str, Any]]:
    """Select a deterministic disjoint shard from sample-id-sorted records."""
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("Shard count must be positive and index must be in range")
    return [
        record
        for position, record in enumerate(records)
        if position % shard_count == shard_index
    ]


def verify_cached(path: Path, expected_identity: str, root: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest["input_identity_sha256"] != expected_identity:
        raise ValueError(f"Cached crop identity mismatch: {path}")
    for record in manifest["assets"].values():
        asset = safe_repo_path(root, record["path"])
        if asset.stat().st_size != record["bytes"] or sha256(asset) != record["sha256"]:
            raise ValueError(f"Cached crop asset mismatch: {asset}")
    return manifest


def acquire_one(
    root: Path,
    output_root: Path,
    cohort: dict[str, Any],
    assets: dict[str, Any],
    *,
    min_valid_fraction: float,
    overwrite: bool,
) -> dict[str, Any]:
    sample_id = cohort["sample_id"]
    identity = stable_identity(cohort, assets)
    scene_dir = output_root / sample_id
    manifest_path = scene_dir / "manifest.json"
    if manifest_path.is_file() and not overwrite:
        return verify_cached(manifest_path, identity, root)

    if cohort["research_role"] == "sealed_external" or assets["research_role"] == "sealed_external":
        raise ValueError("Sealed-external sample reached crop acquisition")
    if assets["status"] != "resolved":
        raise ValueError(f"Unresolved sample reached crop acquisition: {sample_id}")
    sensor = cohort["sensor_family"]
    bands, descriptions = source_contract(sensor)
    target_assets = assets["target"]["assets"]
    reference_assets = assets["reference"]["assets"]
    crs, transform = aligned_grid(
        target_assets[bands[0]], cohort["source_center"], size=SIZE, resolution=RESOLUTION
    )
    target = read_stack(target_assets, bands, crs=crs, transform=transform)
    reference = read_stack(reference_assets, bands, crs=crs, transform=transform)
    image = np.concatenate([target, reference], axis=0)
    valid = np.all(image > 0, axis=0)
    label_state = cohort.get("label_state", "PLUME")
    mask = rasterized_plumes(cohort["plume_geometries"], crs=crs, transform=transform)
    plume_pixels = int(np.count_nonzero(mask))
    valid_fraction = float(np.mean(valid))
    plume_valid_fraction = float(np.mean(valid[mask > 0])) if plume_pixels else None

    scene_dir.mkdir(parents=True, exist_ok=True)
    image_path = scene_dir / "image.tif"
    mask_path = scene_dir / "plume_mask.tif"
    write_raster(
        image_path,
        image,
        crs=crs,
        transform=transform,
        descriptions=list(descriptions) + [f"{band}_bg" for band in descriptions],
        nodata=0,
    )
    write_raster(
        mask_path,
        mask,
        crs=crs,
        transform=transform,
        descriptions=[
            "NO_PLUME_ZERO_MASK" if label_state == "NO_PLUME" else "UNEP_IMEO_MARS_PLUME"
        ],
        nodata=0,
    )
    output_assets = {
        "image": file_record(image_path, root),
        "plume_mask": file_record(mask_path, root),
    }
    cloud_status = "cloudsen12_pending"
    clear_fraction: float | None = None
    plume_clear_fraction: float | None = None
    if sensor == "Landsat":
        target_qa = read_to_grid(
            target_assets["QA_PIXEL"],
            crs=crs,
            transform=transform,
            size=SIZE,
            resampling=Resampling.nearest,
            dtype="uint16",
        )
        reference_qa = read_to_grid(
            reference_assets["QA_PIXEL"],
            crs=crs,
            transform=transform,
            size=SIZE,
            resampling=Resampling.nearest,
            dtype="uint16",
        )
        cloud = np.maximum(
            landsat_cloud_classes(target_qa), landsat_cloud_classes(reference_qa)
        )
        cloud[~valid] = 1
        cloud_path = scene_dir / "cloud_mask.tif"
        write_raster(
            cloud_path,
            cloud,
            crs=crs,
            transform=transform,
            descriptions=["USGS_C2_QA_PIXEL_TARGET_REFERENCE_UNION"],
            nodata=1,
        )
        output_assets["cloud_mask"] = file_record(cloud_path, root)
        clear = cloud == 0
        clear_fraction = float(np.mean(clear))
        plume_clear_fraction = float(np.mean(clear[mask > 0])) if plume_pixels else None
        cloud_status = "landsat_qa_complete"

    geometry_pass = geometry_gate(label_state, plume_pixels)
    radiometry_gate = valid_fraction >= min_valid_fraction and (
        label_state == "NO_PLUME"
        or (
            plume_valid_fraction is not None
            and plume_valid_fraction >= min_valid_fraction
        )
    )
    gate_pass = geometry_pass and radiometry_gate
    manifest = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "sample_id": sample_id,
        "group_id": cohort["group_id"],
        "research_role": cohort["research_role"],
        "sensor_family": sensor,
        "source_name": cohort["source_name"],
        "target_product": cohort["target_product"],
        "background_product": cohort["background_product"],
        "label_state": label_state,
        "plume_ids": cohort.get("plume_ids", []),
        "source_center": cohort["source_center"],
        "input_identity_sha256": identity,
        "product_contract": {
            "shape": [SIZE, SIZE],
            "resolution_m": RESOLUTION,
            "crs": crs.to_string(),
            "transform": list(transform)[:6],
            "band_order": list(descriptions) + [f"{band}_bg" for band in descriptions],
            "dtype": "uint16",
            "resampling": "bilinear spectral; nearest QA; all_touched polygon rasterization",
        },
        "quality": {
            "gate_pass_before_cloud": gate_pass,
            "radiometric_valid_fraction": round(valid_fraction, 8),
            "plume_radiometric_valid_fraction": (
                None
                if plume_valid_fraction is None
                else round(plume_valid_fraction, 8)
            ),
            "plume_pixels": plume_pixels,
            "cloud_status": cloud_status,
            "clear_fraction": None if clear_fraction is None else round(clear_fraction, 8),
            "plume_clear_fraction": (
                None if plume_clear_fraction is None else round(plume_clear_fraction, 8)
            ),
        },
        "assets": output_assets,
    }
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, manifest_path)
    return manifest


def write_markdown(report: dict[str, Any], path: Path) -> None:
    summary = report["summary"]
    negative_only = set(summary["by_label_state"]) == {"NO_PLUME"}
    lines = [
        (
            "# CloudSEN12+ clear-scene spatial-pilot crop acquisition"
            if negative_only
            else "# UNEP MARS post-2024 exact crop acquisition"
        ),
        "",
        f"Generated: {report['generated_at_utc']}.",
        "",
        "## Result",
        "",
        f"- Fully resolved nonsealed samples attempted: **{summary['attempted']:,}**.",
        f"- Crops acquired and hash-verified: **{summary['acquired']:,}**.",
        f"- Pre-cloud radiometry/geometry gate pass: **{summary['gate_pass_before_cloud']:,}**.",
        f"- Acquisition errors: **{summary['errors']:,}**.",
        f"- Ignored raster bytes: **{summary['raster_bytes']:,}**.",
        "",
        "## Contract",
        "",
        "- Exact target and background products; no product substitution.",
        "- 200x200 pixels at 10 m in the target product CRS (2x2 km).",
        "- Twelve uint16 bands: six target then six reference bands.",
        (
            "- Explicit clear-scene negatives require an identically gridded zero plume mask."
            if negative_only
            else "- UNEP MultiPolygon plume truth is rasterized on the identical grid."
        ),
        "- Landsat cloud support is the target/reference union of QA_PIXEL fill, dilated-cloud, cirrus, cloud, shadow, and snow bits.",
        (
            "- Published CloudSEN12+ all-clear labels remain the negative cloud contract."
            if negative_only
            else "- Sentinel-2 CloudSEN12+ masks remain a separate required acquisition gate."
        ),
        "- Sealed-external samples were excluded.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", default=DEFAULT_COHORT.as_posix())
    parser.add_argument("--assets", default=DEFAULT_ASSETS.as_posix())
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT.as_posix())
    parser.add_argument("--output-json", default=DEFAULT_JSON.as_posix())
    parser.add_argument("--output-markdown", default=DEFAULT_MARKDOWN.as_posix())
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--min-valid-fraction", type=float, default=0.8)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.workers <= 12:
        parser.error("--workers must be between 1 and 12")
    if not 0.0 <= args.min_valid_fraction <= 1.0:
        parser.error("--min-valid-fraction must be in [0,1]")
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        parser.error("--shard-count must be positive and --shard-index must be in range")

    root = repo_root()
    cohort_path = safe_repo_path(root, args.cohort)
    assets_path = safe_repo_path(root, args.assets)
    output_root = safe_repo_path(root, args.output_dir)
    output_json = safe_repo_path(root, args.output_json)
    output_markdown = safe_repo_path(root, args.output_markdown)
    cohort = {
        item["sample_id"]: item
        for item in (
            json.loads(line) for line in cohort_path.read_text(encoding="utf-8").splitlines()
        )
        if item["research_role"] != "sealed_external"
    }
    resolved = [
        item
        for item in (
            json.loads(line) for line in assets_path.read_text(encoding="utf-8").splitlines()
        )
        if item["status"] == "resolved"
    ]
    resolved.sort(key=lambda item: item["sample_id"])
    resolved = select_shard(resolved, args.shard_count, args.shard_index)
    if args.limit is not None:
        resolved = resolved[: args.limit]
    if any(item["sample_id"] not in cohort for item in resolved):
        raise ValueError("Resolved asset manifest is not a subset of the nonsealed cohort")

    manifests: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    output_root.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                acquire_one,
                root,
                output_root,
                cohort[item["sample_id"]],
                item,
                min_valid_fraction=args.min_valid_fraction,
                overwrite=args.overwrite,
            ): item["sample_id"]
            for item in resolved
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            sample_id = futures[future]
            try:
                manifests.append(future.result())
            except Exception as exc:
                errors.append(
                    {
                        "sample_id": sample_id,
                        "error_type": type(exc).__name__,
                        "error": str(exc)[:500],
                    }
                )
            if completed % 10 == 0 or completed == len(futures):
                print(f"cropped {completed}/{len(futures)} errors={len(errors)}", flush=True)
    manifests.sort(key=lambda item: item["sample_id"])
    raster_bytes = sum(
        asset["bytes"] for manifest in manifests for asset in manifest["assets"].values()
    )
    report = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "nonsealed crops acquired; Sentinel-2 cloud gate pending",
        "inputs": {
            "cohort": {"path": args.cohort, "sha256": sha256(cohort_path)},
            "assets": {"path": args.assets, "sha256": sha256(assets_path)},
        },
        "contract": {
            "shape": [SIZE, SIZE],
            "resolution_m": RESOLUTION,
            "min_valid_fraction": args.min_valid_fraction,
            "exact_product_substitution": False,
            "sealed_external_accessed": False,
            "shard_count": args.shard_count,
            "shard_index": args.shard_index,
        },
        "summary": {
            "attempted": len(resolved),
            "acquired": len(manifests),
            "gate_pass_before_cloud": sum(
                item["quality"]["gate_pass_before_cloud"] for item in manifests
            ),
            "by_sensor": dict(sorted(Counter(item["sensor_family"] for item in manifests).items())),
            "by_role": dict(sorted(Counter(item["research_role"] for item in manifests).items())),
            "by_label_state": dict(
                sorted(Counter(item.get("label_state", "PLUME") for item in manifests).items())
            ),
            "errors": len(errors),
            "raster_bytes": raster_bytes,
        },
        "errors": errors,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(report, output_markdown)
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    if errors:
        raise RuntimeError(f"{len(errors)} crop acquisitions failed; see {output_json}")


if __name__ == "__main__":
    main()
