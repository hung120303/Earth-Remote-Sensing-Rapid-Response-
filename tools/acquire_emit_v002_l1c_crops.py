#!/usr/bin/env python3
"""Acquire and audit public L1C/SCL crops for frozen EMIT external pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from affine import Affine
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform as transform_points
from rasterio.warp import transform_geom

from acquire_v002_pilot import cmr_granule, geometry_bounds, plume_feature, repo_root

DEFAULT_PAIRS = Path("reports/acquisition/emit_v002_l1c_pairs.json")
DEFAULT_OUTPUT = Path(
    "EarthRemoteSensingRapidResponse/Data Collection/s2_emit_pairs/"
    "emit-v002-external-l1c-2026-07"
)
DEFAULT_JSON = Path("reports/acquisition/emit_v002_l1c_raster_gate.json")
DEFAULT_MARKDOWN = Path("reports/acquisition/EMIT_V002_L1C_RASTER_GATE.md")
BAND_ORDER = ("B02", "B03", "B04", "B08", "B11", "B12")
NON_CLEAR_SCL = {0, 1, 3, 8, 9, 10, 11}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_repo_path(root: Path, value: str) -> Path:
    path = (root / value).resolve()
    if path != root and root not in path.parents:
        raise ValueError(f"Path must resolve beneath repository root: {value}")
    return path


def gdal_env() -> dict[str, str]:
    return {
        "AWS_NO_SIGN_REQUEST": "YES",
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
        "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.jp2",
        "GDAL_HTTP_MAX_RETRY": "5",
        "GDAL_HTTP_RETRY_DELAY": "1",
        "VSI_CACHE": "TRUE",
        "VSI_CACHE_SIZE": str(64 * 1024 * 1024),
    }


def aligned_grid(
    reference_url: str,
    center: list[float],
    *,
    size: int,
    resolution: float,
) -> tuple[Any, Affine]:
    with rasterio.Env(**gdal_env()), rasterio.open(reference_url) as source:
        if source.crs is None:
            raise ValueError(f"Reference raster has no CRS: {reference_url}")
        xs, ys = transform_points("EPSG:4326", source.crs, [center[0]], [center[1]])
        half = size * resolution / 2.0
        left = math.floor((xs[0] - half) / resolution) * resolution
        top = math.ceil((ys[0] + half) / resolution) * resolution
        return source.crs, Affine(resolution, 0.0, left, 0.0, -resolution, top)


def read_to_grid(
    url: str,
    *,
    crs: Any,
    transform: Affine,
    size: int,
    resampling: Resampling,
    dtype: str,
) -> np.ndarray:
    with rasterio.Env(**gdal_env()), rasterio.open(url) as source:
        source_nodata = source.nodata if source.nodata is not None else 0
        with WarpedVRT(
            source,
            crs=crs,
            transform=transform,
            width=size,
            height=size,
            src_nodata=source_nodata,
            nodata=0,
            resampling=resampling,
        ) as vrt:
            return vrt.read(1, out_dtype=dtype)


def write_raster(
    path: Path,
    data: np.ndarray,
    *,
    crs: Any,
    transform: Affine,
    descriptions: list[str],
    nodata: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if data.ndim == 2:
        data = data[None, ...]
    profile = {
        "driver": "GTiff",
        "width": data.shape[2],
        "height": data.shape[1],
        "count": data.shape[0],
        "dtype": str(data.dtype),
        "crs": crs,
        "transform": transform,
        "nodata": nodata,
        "compress": "deflate",
        "predictor": 2 if np.issubdtype(data.dtype, np.integer) and data.dtype.itemsize > 1 else 1,
        "tiled": True,
        "blockxsize": 128,
        "blockysize": 128,
    }
    with rasterio.open(temporary, "w", **profile) as target:
        target.write(data)
        for index, description in enumerate(descriptions, start=1):
            target.set_band_description(index, description)
    os.replace(temporary, path)


def asset_record(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def verify_asset(root: Path, record: dict[str, Any]) -> Path:
    path = (root / record["path"]).resolve()
    if root not in path.parents:
        raise ValueError(f"Manifest asset escapes repository root: {record['path']}")
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size != int(record["bytes"]):
        raise ValueError(f"Size mismatch: {path}")
    if sha256(path) != record["sha256"]:
        raise ValueError(f"SHA-256 mismatch: {path}")
    return path


def verify_local_manifest(root: Path, path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    for record in manifest["assets"].values():
        verify_asset(root, record)
    return manifest


def crop_stack(
    assets: dict[str, str], *, crs: Any, transform: Affine, size: int
) -> np.ndarray:
    return np.stack(
        [
            read_to_grid(
                assets[band],
                crs=crs,
                transform=transform,
                size=size,
                resampling=Resampling.bilinear,
                dtype="uint16",
            )
            for band in BAND_ORDER
        ]
    )


def clear_support(stack: np.ndarray, scl: np.ndarray) -> dict[str, Any]:
    radiometry_valid = np.all(stack > 0, axis=0)
    scl_valid = scl != 0
    clear = radiometry_valid & scl_valid & ~np.isin(scl, list(NON_CLEAR_SCL))
    return {
        "radiometry_valid": radiometry_valid,
        "scl_valid": scl_valid,
        "clear": clear,
        "radiometry_valid_fraction": round(float(np.mean(radiometry_valid)), 8),
        "scl_valid_fraction": round(float(np.mean(scl_valid)), 8),
        "clear_fraction": round(float(np.mean(clear)), 8),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def acquire_one(
    root: Path,
    output_root: Path,
    pair: dict[str, Any],
    *,
    size: int,
    resolution: float,
    min_clear_fraction: float,
    overwrite: bool,
    verify_only: bool,
) -> dict[str, Any]:
    scene_dir = output_root / pair["granule_id"]
    manifest_path = scene_dir / "manifest.json"
    if manifest_path.is_file() and not overwrite:
        manifest = verify_local_manifest(root, manifest_path)
        return {"status": "verified_cached", "manifest": manifest}
    if verify_only:
        raise FileNotFoundError(f"Missing local manifest: {manifest_path}")

    record = cmr_granule(pair["granule_id"])
    feature = plume_feature(record)
    target_assets = pair["target"]["l1c_spectral_assets"]
    reference_assets = pair["reference"]["l1c_spectral_assets"]
    crs, transform = aligned_grid(
        target_assets["B02"], pair["center"], size=size, resolution=resolution
    )
    target_stack = crop_stack(target_assets, crs=crs, transform=transform, size=size)
    reference_stack = crop_stack(reference_assets, crs=crs, transform=transform, size=size)
    target_scl = read_to_grid(
        pair["target"]["l2a_scl_asset"],
        crs=crs,
        transform=transform,
        size=size,
        resampling=Resampling.nearest,
        dtype="uint8",
    )
    reference_scl = read_to_grid(
        pair["reference"]["l2a_scl_asset"],
        crs=crs,
        transform=transform,
        size=size,
        resampling=Resampling.nearest,
        dtype="uint8",
    )
    target_support = clear_support(target_stack, target_scl)
    reference_support = clear_support(reference_stack, reference_scl)

    projected = transform_geom("EPSG:4326", crs, feature["geometry"], precision=3)
    plume_bounds = geometry_bounds(projected)
    grid_bounds = rasterio.transform.array_bounds(size, size, transform)
    margins = (
        plume_bounds[0] - grid_bounds[0],
        plume_bounds[1] - grid_bounds[1],
        grid_bounds[2] - plume_bounds[2],
        grid_bounds[3] - plume_bounds[3],
    )
    plume_fully_contained = min(margins) >= 0.0
    plume_mask = rasterize(
        [(projected, 1)],
        out_shape=(size, size),
        transform=transform,
        fill=0,
        dtype="uint8",
        all_touched=True,
    )
    plume_pixels = int(np.count_nonzero(plume_mask))
    target_clear_on_plume = float(
        np.count_nonzero(target_support["clear"] & (plume_mask > 0)) / max(plume_pixels, 1)
    )
    gate_checks = {
        "target_local_clear": target_support["clear_fraction"] >= min_clear_fraction,
        "reference_local_clear": reference_support["clear_fraction"] >= min_clear_fraction,
        "plume_fully_contained": plume_fully_contained,
        "plume_mask_nonempty": plume_pixels > 0,
        "target_plume_support_clear": target_clear_on_plume >= min_clear_fraction,
    }

    target_path = scene_dir / "target.l1c.tif"
    reference_path = scene_dir / "reference.l1c.tif"
    target_scl_path = scene_dir / "target.l2a_scl.tif"
    reference_scl_path = scene_dir / "reference.l2a_scl.tif"
    plume_path = scene_dir / "plume_mask.tif"
    write_raster(
        target_path,
        target_stack,
        crs=crs,
        transform=transform,
        descriptions=list(BAND_ORDER),
        nodata=0,
    )
    write_raster(
        reference_path,
        reference_stack,
        crs=crs,
        transform=transform,
        descriptions=[f"{band}_reference" for band in BAND_ORDER],
        nodata=0,
    )
    write_raster(
        target_scl_path,
        target_scl,
        crs=crs,
        transform=transform,
        descriptions=["SCL_target"],
        nodata=0,
    )
    write_raster(
        reference_scl_path,
        reference_scl,
        crs=crs,
        transform=transform,
        descriptions=["SCL_reference"],
        nodata=0,
    )
    write_raster(
        plume_path,
        plume_mask,
        crs=crs,
        transform=transform,
        descriptions=["EMIT_V002_CMR_PLUME_MASK"],
        nodata=0,
    )
    assets = {
        "target_l1c": asset_record(target_path, root),
        "reference_l1c": asset_record(reference_path, root),
        "target_l2a_scl": asset_record(target_scl_path, root),
        "reference_l2a_scl": asset_record(reference_scl_path, root),
        "plume_mask": asset_record(plume_path, root),
    }
    manifest = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "group_id": pair["group_id"],
        "granule_id": pair["granule_id"],
        "plume_id": pair.get("plume_id"),
        "source_scenes": pair.get("source_scenes", []),
        "emit_datetime": pair["emit_datetime"],
        "emit_to_target_offset_hours": pair["emit_to_target_offset_hours"],
        "reference_to_target_gap_hours": pair["reference_to_target_gap_hours"],
        "target_scene_id": pair["target"]["l1c_scene_id"],
        "reference_scene_id": pair["reference"]["l1c_scene_id"],
        "product_contract": {
            "spectral_product": "Sentinel-2 L1C TOA",
            "band_order": list(BAND_ORDER),
            "observability_product": "co-temporal Sentinel-2 L2A SCL",
            "shape": [size, size],
            "resolution_m": resolution,
            "crs": crs.to_string(),
            "transform": [float(value) for value in transform[:6]],
        },
        "quality": {
            "target": {
                key: value for key, value in target_support.items() if not isinstance(value, np.ndarray)
            },
            "reference": {
                key: value
                for key, value in reference_support.items()
                if not isinstance(value, np.ndarray)
            },
            "target_clear_fraction_on_plume": round(target_clear_on_plume, 8),
            "plume_pixels": plume_pixels,
            "plume_positive_fraction": round(plume_pixels / float(size * size), 8),
            "plume_fully_contained": plume_fully_contained,
            "plume_min_margin_pixels": round(min(margins) / resolution, 6),
            "gate_checks": gate_checks,
            "gate_pass": all(gate_checks.values()),
        },
        "assets": assets,
    }
    write_json(manifest_path, manifest)
    return {"status": "acquired", "manifest": manifest}


def summarize(
    root: Path,
    pairs_path: Path,
    output_root: Path,
    results: list[dict[str, Any]],
    *,
    size: int,
    resolution: float,
    min_clear_fraction: float,
) -> dict[str, Any]:
    manifests = [item["manifest"] for item in results if "manifest" in item]
    errors = [item for item in results if "error" in item]
    gate_failures: Counter[str] = Counter()
    for manifest in manifests:
        for check, passed in manifest["quality"]["gate_checks"].items():
            if not passed:
                gate_failures[check] += 1
    return {
        "schema_version": 1,
        "scope": "public_emit_external_l1c_l2a_scl_raster_gate",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract": {
            "pairs_manifest": pairs_path.relative_to(root).as_posix(),
            "pairs_manifest_sha256": sha256(pairs_path),
            "ignored_output_root": output_root.relative_to(root).as_posix(),
            "spectral_product": "Sentinel-2 L1C TOA",
            "band_order": list(BAND_ORDER),
            "observability_product": "co-temporal Sentinel-2 L2A SCL",
            "shape": [size, size],
            "resolution_m": resolution,
            "minimum_local_clear_fraction": min_clear_fraction,
            "selection_inputs": "public metadata and rasters only; no model predictions",
        },
        "summary": {
            "requested_pairs": len(results),
            "verified_pairs": len(manifests),
            "gate_pass_pairs": sum(item["quality"]["gate_pass"] for item in manifests),
            "gate_fail_pairs": sum(not item["quality"]["gate_pass"] for item in manifests),
            "acquisition_errors": len(errors),
            "gate_failure_counts": dict(sorted(gate_failures.items())),
            "raw_cropped_assets": sum(len(item["assets"]) for item in manifests),
            "raw_cropped_bytes": sum(
                int(asset["bytes"])
                for item in manifests
                for asset in item["assets"].values()
            ),
            "minimum_50_group_goal_met": sum(item["quality"]["gate_pass"] for item in manifests)
            >= 50,
        },
        "samples": [
            {
                "group_id": item["group_id"],
                "granule_id": item["granule_id"],
                "target_scene_id": item["target_scene_id"],
                "reference_scene_id": item["reference_scene_id"],
                "target_clear_fraction": item["quality"]["target"]["clear_fraction"],
                "reference_clear_fraction": item["quality"]["reference"]["clear_fraction"],
                "target_clear_fraction_on_plume": item["quality"][
                    "target_clear_fraction_on_plume"
                ],
                "plume_pixels": item["quality"]["plume_pixels"],
                "plume_min_margin_pixels": item["quality"]["plume_min_margin_pixels"],
                "gate_checks": item["quality"]["gate_checks"],
                "gate_pass": item["quality"]["gate_pass"],
                "manifest_sha256": sha256(
                    output_root / item["granule_id"] / "manifest.json"
                ),
            }
            for item in manifests
        ],
        "errors": errors,
        "provenance": {
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True
            ).strip(),
            "script": "tools/acquire_emit_v002_l1c_crops.py",
            "script_sha256": sha256(Path(__file__)),
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "rasterio": rasterio.__version__,
        },
    }


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    summary = report["summary"]
    lines = [
        "# EMIT external L1C raster-quality gate",
        "",
        f"- Requested pairs: {summary['requested_pairs']}",
        f"- Verified local pairs: {summary['verified_pairs']}",
        f"- Gate-pass pairs: {summary['gate_pass_pairs']}",
        f"- Gate-fail pairs: {summary['gate_fail_pairs']}",
        f"- Acquisition errors: {summary['acquisition_errors']}",
        f"- Ignored cropped assets: {summary['raw_cropped_assets']} / {summary['raw_cropped_bytes']:,} bytes",
        f"- Minimum 50-group acquisition goal: `{'pass' if summary['minimum_50_group_goal_met'] else 'not_yet'}`",
        "",
        "The gate uses public six-band L1C target/reference radiometry on the native 200x200, 10 m detector grid and co-temporal L2A SCL only for observability. Selection and quality filtering remain prediction-blind.",
        "",
        "| Group | Target clear | Reference clear | Plume clear | Margin px | Gate |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for item in report["samples"]:
        lines.append(
            f"| `{item['group_id']}` | {100 * item['target_clear_fraction']:.1f}% | "
            f"{100 * item['reference_clear_fraction']:.1f}% | "
            f"{100 * item['target_clear_fraction_on_plume']:.1f}% | "
            f"{item['plume_min_margin_pixels']:.1f} | "
            f"{'pass' if item['gate_pass'] else 'fail'} |"
        )
    if summary["gate_failure_counts"]:
        lines.extend(["", "## Exclusion counts", ""])
        lines.extend(
            f"- `{name}`: {count}" for name, count in summary["gate_failure_counts"].items()
        )
    lines.extend(
        [
            "",
            "Passing this raster gate does not create a locked paper label. Protected EMIT enhancement/uncertainty/sensitivity acquisition, wind reanalysis, source deduplication, and two-annotator review remain required before the one-time external evaluation.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", default=DEFAULT_PAIRS.as_posix())
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT.as_posix())
    parser.add_argument("--output-json", default=DEFAULT_JSON.as_posix())
    parser.add_argument("--output-markdown", default=DEFAULT_MARKDOWN.as_posix())
    parser.add_argument("--size", type=int, default=200)
    parser.add_argument("--resolution", type=float, default=10.0)
    parser.add_argument("--min-clear-fraction", type=float, default=0.70)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.size != 200 or args.resolution != 10.0:
        parser.error("External primary contract is fixed at 200 pixels and 10 m")
    if not 0 <= args.min_clear_fraction <= 1:
        parser.error("min-clear-fraction must be in [0,1]")
    if args.workers <= 0 or (args.limit is not None and args.limit <= 0):
        parser.error("workers and limit must be positive")
    root = repo_root()
    pairs_path = safe_repo_path(root, args.pairs)
    output_root = safe_repo_path(root, args.output_dir)
    output_json = safe_repo_path(root, args.output_json)
    output_markdown = safe_repo_path(root, args.output_markdown)
    pair_report = json.loads(pairs_path.read_text(encoding="utf-8"))
    pairs = list(pair_report.get("pairs", []))
    if args.limit is not None:
        pairs = pairs[: args.limit]

    def checked(pair: dict[str, Any]) -> dict[str, Any]:
        try:
            return acquire_one(
                root,
                output_root,
                pair,
                size=args.size,
                resolution=args.resolution,
                min_clear_fraction=args.min_clear_fraction,
                overwrite=args.overwrite,
                verify_only=args.verify_only,
            )
        except Exception as exc:
            return {
                "status": "error",
                "group_id": pair.get("group_id"),
                "granule_id": pair.get("granule_id"),
                "error_type": type(exc).__name__,
                "error": str(exc)[:1000],
            }

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        results = list(executor.map(checked, pairs))
    report = summarize(
        root,
        pairs_path,
        output_root,
        results,
        size=args.size,
        resolution=args.resolution,
        min_clear_fraction=args.min_clear_fraction,
    )
    write_json(output_json, report)
    write_markdown(output_markdown, report)
    result = {
        "ok": report["summary"]["acquisition_errors"] == 0,
        "requested_pairs": report["summary"]["requested_pairs"],
        "verified_pairs": report["summary"]["verified_pairs"],
        "gate_pass_pairs": report["summary"]["gate_pass_pairs"],
        "gate_fail_pairs": report["summary"]["gate_fail_pairs"],
        "acquisition_errors": report["summary"]["acquisition_errors"],
        "output_json": output_json.relative_to(root).as_posix(),
        "output_markdown": output_markdown.relative_to(root).as_posix(),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
