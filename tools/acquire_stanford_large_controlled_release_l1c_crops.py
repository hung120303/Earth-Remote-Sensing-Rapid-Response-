#!/usr/bin/env python3
"""Acquire frozen Stanford Sentinel-2 L1C crops with window-only reads.

This stage is deliberately outcome-blind. It accepts the label-free frozen pair
manifest, verifies its tracked receipt, reads only the required six-band
256x256 target/reference windows, and writes all raster data below .research/.
No Stanford release labels or detector outputs are read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from affine import Affine
from rasterio.enums import Resampling
from rasterio.warp import reproject, transform as transform_points, transform_bounds
from rasterio.windows import Window, from_bounds

BAND_ORDER = ("B02", "B03", "B04", "B08", "B11", "B12")
TEN_METER_BANDS = frozenset(("B02", "B03", "B04", "B08"))
TWENTY_METER_BANDS = frozenset(("B11", "B12"))
BANNED_OUTCOME_KEYS = frozenset(
    (
        "emission_rate_kg_h",
        "primary_label",
        "truth_class",
        "release_label",
        "release_rate",
        "model_score",
        "prediction",
        "detector_output",
    )
)
DEFAULT_PAIRS = Path(
    ".research/stanford_controlled_release_2024_2025/l1c_stress/pair_manifest.json"
)
DEFAULT_PAIR_RECEIPT = Path(
    "reports/acquisition/stanford_large_controlled_release_l1c_pairs.json"
)
DEFAULT_PROTOCOL = Path("configs/stanford_large_controlled_release_protocol.json")
DEFAULT_OUTPUT_ROOT = Path(
    ".research/stanford_controlled_release_2024_2025/l1c_stress/crops"
)
DEFAULT_MANIFEST = Path(
    ".research/stanford_controlled_release_2024_2025/l1c_stress/crop_manifest.json"
)
DEFAULT_JSON = Path(
    "reports/acquisition/stanford_large_controlled_release_l1c_crops.json"
)
DEFAULT_MARKDOWN = Path(
    "reports/acquisition/STANFORD_LARGE_CONTROLLED_RELEASE_L1C_CROPS.md"
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


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


def assert_ignored_storage(root: Path, path: Path) -> None:
    relative = path.relative_to(root).as_posix()
    completed = subprocess.run(
        ["git", "check-ignore", "--quiet", "--", relative],
        cwd=root,
        check=False,
    )
    if completed.returncode != 0:
        raise ValueError(f"Bulk raster path is not git-ignored: {relative}")


def assert_no_outcome_fields(value: Any, *, trail: str = "root") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = str(key).strip().lower()
            if normalized in BANNED_OUTCOME_KEYS:
                raise ValueError(f"Outcome field is forbidden at {trail}.{key}")
            assert_no_outcome_fields(item, trail=f"{trail}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            assert_no_outcome_fields(item, trail=f"{trail}[{index}]")


def gdal_env() -> dict[str, str]:
    return {
        "AWS_NO_SIGN_REQUEST": "YES",
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
        "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".jp2,.tif,.tiff",
        "GDAL_HTTP_MAX_RETRY": "5",
        "GDAL_HTTP_RETRY_DELAY": "1",
        "VSI_CACHE": "TRUE",
        "VSI_CACHE_SIZE": str(64 * 1024 * 1024),
    }


def band_resampling(band: str) -> Resampling:
    if band in TEN_METER_BANDS:
        return Resampling.nearest
    if band in TWENTY_METER_BANDS:
        return Resampling.bilinear
    raise ValueError(f"Unsupported Sentinel-2 band: {band}")


def centered_native_window(
    transform: Affine,
    *,
    width: int,
    height: int,
    projected_x: float,
    projected_y: float,
    size: int,
) -> Window:
    if size <= 0 or size % 2:
        raise ValueError("Crop size must be a positive even integer")
    col, row = (~transform) * (projected_x, projected_y)
    col_off = int(math.floor(col)) - size // 2
    row_off = int(math.floor(row)) - size // 2
    if col_off < 0 or row_off < 0 or col_off + size > width or row_off + size > height:
        raise ValueError("Centered crop would leave the native raster extent")
    return Window(col_off, row_off, size, size)


def read_native_window(source: Any, *, window: Window, size: int) -> np.ndarray:
    if int(window.width) != size or int(window.height) != size:
        raise ValueError(f"Native window must be exactly {size}x{size}")
    data = source.read(
        1,
        window=window,
        boundless=False,
        out_dtype="uint16",
    )
    if data.shape != (size, size):
        raise ValueError(f"Native read returned {data.shape}, expected {(size, size)}")
    if data.dtype != np.uint16:
        raise ValueError("Native L1C read did not preserve raw uint16")
    return data


def stack_bands(bands: dict[str, np.ndarray], *, size: int) -> np.ndarray:
    missing = set(BAND_ORDER) - set(bands)
    if missing:
        raise ValueError(f"Missing L1C bands: {sorted(missing)}")
    ordered: list[np.ndarray] = []
    for band in BAND_ORDER:
        data = bands[band]
        if data.dtype != np.uint16:
            raise ValueError(f"{band} must remain raw uint16")
        if data.shape != (size, size):
            raise ValueError(f"{band} must be {size}x{size}, got {data.shape}")
        ordered.append(data)
    result = np.stack(ordered, axis=0)
    if result.dtype != np.uint16:
        raise ValueError("L1C stack must remain raw uint16")
    return result


def _clipped_source_window(
    source: Any,
    *,
    dst_crs: Any,
    dst_transform: Affine,
    size: int,
    padding_pixels: int,
) -> Window:
    left, bottom, right, top = rasterio.transform.array_bounds(size, size, dst_transform)
    if source.crs is None:
        raise ValueError("Remote L1C asset has no CRS")
    if source.crs != dst_crs:
        left, bottom, right, top = transform_bounds(
            dst_crs,
            source.crs,
            left,
            bottom,
            right,
            top,
            densify_pts=21,
        )
    fractional = from_bounds(left, bottom, right, top, transform=source.transform)
    col0 = max(0, int(math.floor(fractional.col_off)) - padding_pixels)
    row0 = max(0, int(math.floor(fractional.row_off)) - padding_pixels)
    col1 = min(
        source.width,
        int(math.ceil(fractional.col_off + fractional.width)) + padding_pixels,
    )
    row1 = min(
        source.height,
        int(math.ceil(fractional.row_off + fractional.height)) + padding_pixels,
    )
    if col1 <= col0 or row1 <= row0:
        raise ValueError("Target grid does not overlap remote L1C asset")
    return Window(col0, row0, col1 - col0, row1 - row0)


def read_to_target_grid(
    url: str,
    *,
    dst_crs: Any,
    dst_transform: Affine,
    size: int,
    resampling: Resampling,
) -> np.ndarray:
    padding = 1 if resampling is Resampling.bilinear else 0
    with rasterio.Env(**gdal_env()), rasterio.open(url) as source:
        window = _clipped_source_window(
            source,
            dst_crs=dst_crs,
            dst_transform=dst_transform,
            size=size,
            padding_pixels=padding,
        )
        source_data = source.read(
            1,
            window=window,
            boundless=False,
            out_dtype="uint16",
        )
        if source_data.dtype != np.uint16:
            raise ValueError(f"Remote L1C asset is not raw uint16: {url}")
        destination = np.zeros((size, size), dtype=np.uint16)
        reproject(
            source=source_data,
            destination=destination,
            src_transform=source.window_transform(window),
            src_crs=source.crs,
            src_nodata=source.nodata if source.nodata is not None else 0,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            dst_nodata=0,
            resampling=resampling,
        )
    return destination


def target_grid(
    b02_url: str,
    center: list[float],
    *,
    size: int,
) -> tuple[Any, Affine, np.ndarray]:
    with rasterio.Env(**gdal_env()), rasterio.open(b02_url) as source:
        if source.crs is None:
            raise ValueError("Target B02 has no CRS")
        x, y = transform_points("EPSG:4326", source.crs, [center[0]], [center[1]])
        window = centered_native_window(
            source.transform,
            width=source.width,
            height=source.height,
            projected_x=float(x[0]),
            projected_y=float(y[0]),
            size=size,
        )
        b02 = read_native_window(source, window=window, size=size)
        transform = source.window_transform(window)
        resolution_x = abs(float(transform.a))
        resolution_y = abs(float(transform.e))
        if abs(resolution_x - 10.0) > 1e-6 or abs(resolution_y - 10.0) > 1e-6:
            raise ValueError(
                f"Target B02 grid is not native 10 m: {resolution_x}, {resolution_y}"
            )
        return source.crs, transform, b02


def crop_pair(pair: dict[str, Any], *, size: int) -> tuple[np.ndarray, np.ndarray, Any, Affine]:
    center = pair["center"]
    target_assets = pair["target"]["spectral_assets"]
    reference_assets = pair["reference"]["spectral_assets"]
    if set(target_assets) != set(BAND_ORDER) or set(reference_assets) != set(BAND_ORDER):
        raise ValueError("Pair spectral assets do not match the frozen six-band order")
    crs, transform, target_b02 = target_grid(target_assets["B02"], center, size=size)
    target_bands = {"B02": target_b02}
    for band in BAND_ORDER[1:]:
        target_bands[band] = read_to_target_grid(
            target_assets[band],
            dst_crs=crs,
            dst_transform=transform,
            size=size,
            resampling=band_resampling(band),
        )
    reference_bands = {
        band: read_to_target_grid(
            reference_assets[band],
            dst_crs=crs,
            dst_transform=transform,
            size=size,
            resampling=band_resampling(band),
        )
        for band in BAND_ORDER
    }
    return (
        stack_bands(target_bands, size=size),
        stack_bands(reference_bands, size=size),
        crs,
        transform,
    )


def write_raster(
    path: Path,
    data: np.ndarray,
    *,
    crs: Any,
    transform: Affine,
    descriptions: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    profile = {
        "driver": "GTiff",
        "width": data.shape[2],
        "height": data.shape[1],
        "count": data.shape[0],
        "dtype": "uint16",
        "crs": crs,
        "transform": transform,
        "nodata": 0,
        "compress": "deflate",
        "predictor": 2,
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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    assert_no_outcome_fields(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def safe_event_dir_name(event_id: str) -> str:
    result = re.sub(r"[^A-Za-z0-9_.-]+", "_", event_id).strip("._")
    if not result:
        raise ValueError(f"Unsafe empty event directory name from {event_id!r}")
    return result


def verify_local_pair(root: Path, manifest_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert_no_outcome_fields(manifest)
    for record in manifest["assets"].values():
        verify_asset(root, record)
    return manifest


def acquire_one(
    root: Path,
    output_root: Path,
    pair: dict[str, Any],
    *,
    size: int,
    overwrite: bool,
    verify_only: bool,
) -> dict[str, Any]:
    event_id = str(pair["event_id"])
    event_dir = output_root / safe_event_dir_name(event_id)
    manifest_path = event_dir / "manifest.json"
    if manifest_path.is_file() and not overwrite:
        manifest = verify_local_pair(root, manifest_path)
        return {"status": "verified_cached", "manifest": manifest}
    if verify_only:
        raise FileNotFoundError(f"Missing local crop manifest: {manifest_path}")
    target, reference, crs, transform = crop_pair(pair, size=size)
    target_path = event_dir / "target.l1c.tif"
    reference_path = event_dir / "reference.l1c.tif"
    write_raster(
        target_path,
        target,
        crs=crs,
        transform=transform,
        descriptions=list(BAND_ORDER),
    )
    write_raster(
        reference_path,
        reference,
        crs=crs,
        transform=transform,
        descriptions=[f"{band}_reference" for band in BAND_ORDER],
    )
    target_valid = float(np.mean(np.all(target > 0, axis=0)))
    reference_valid = float(np.mean(np.all(reference > 0, axis=0)))
    manifest = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "event_id": event_id,
        "center": pair["center"],
        "target_scene_id": pair["target"]["scene_id"],
        "reference_scene_id": pair["reference"]["scene_id"],
        "reference_to_target_gap_hours": pair["reference_to_target_gap_hours"],
        "reference_tier": pair["reference_tier"],
        "product_contract": {
            "spectral_product": "Sentinel-2 Level-1C top-of-atmosphere raw DN",
            "band_order": list(BAND_ORDER),
            "shape": [size, size],
            "grid": "target B02 native 10 m grid",
            "crs": crs.to_string(),
            "transform": [float(value) for value in transform[:6]],
            "ten_meter_resampling": "nearest; target B02 native window",
            "twenty_meter_resampling": "bilinear",
            "radiometry": "raw uint16; no scale or processing-offset correction",
            "cloud_product": None,
        },
        "radiometric_observability": {
            "target_all_band_nonzero_fraction": round(target_valid, 8),
            "reference_all_band_nonzero_fraction": round(reference_valid, 8),
        },
        "assets": {
            "target_l1c": asset_record(target_path, root),
            "reference_l1c": asset_record(reference_path, root),
        },
    }
    write_json(manifest_path, manifest)
    return {"status": "acquired", "manifest": manifest}


def expected_pair_count(protocol: dict[str, Any]) -> int:
    """Return a cohort-bound count while preserving the original 169-row default."""
    binding = protocol.get("source", {}).get("target_manifest", {})
    if isinstance(binding, dict) and "rows" in binding:
        count = int(binding["rows"])
        if count <= 0:
            raise ValueError("Protocol target-manifest row count must be positive")
        return count
    return 169


def pair_manifest_binding(pair_receipt: dict[str, Any]) -> dict[str, Any]:
    """Accept both the original receipt and newer namespaced binding shape."""
    direct = pair_receipt.get("pair_manifest")
    if isinstance(direct, dict):
        return direct
    nested = pair_receipt.get("bindings", {}).get("pair_manifest")
    if isinstance(nested, dict):
        return nested
    raise ValueError("Pair receipt lacks a pair-manifest binding")


def load_and_validate_inputs(
    root: Path,
    pairs_path: Path,
    pair_receipt_path: Path,
    protocol_path: Path,
) -> tuple[list[dict[str, Any]], int]:
    pair_manifest = json.loads(pairs_path.read_text(encoding="utf-8"))
    pair_receipt = json.loads(pair_receipt_path.read_text(encoding="utf-8"))
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    assert_no_outcome_fields(pair_manifest)
    if pair_manifest_binding(pair_receipt)["sha256"] != sha256(pairs_path):
        raise ValueError("Pair manifest SHA-256 does not match tracked freeze receipt")
    if not pair_manifest["summary"]["all_pairs_complete"]:
        raise ValueError("Pair freeze is incomplete")
    pairs = pair_manifest["pairs"]
    expected_pairs = expected_pair_count(protocol)
    if len(pairs) != expected_pairs:
        raise ValueError(f"Frozen pair count changed: {len(pairs)}")
    crop = protocol["sentinel_2_l1c_stress_acquisition_contract"]["crop"]
    if crop["shape_pixels"] != [256, 256]:
        raise ValueError("Protocol crop shape changed")
    if crop["band_order"] != list(BAND_ORDER):
        raise ValueError("Protocol band order changed")
    if crop["radiometry"] != (
        "raw uint16 Level-1C DNs; no Level-2, SCL, processing-offset, or scale correction"
    ):
        raise ValueError("Protocol radiometry contract changed")
    return pairs, int(crop["shape_pixels"][0])


def build_crop_manifest(
    root: Path,
    pairs_path: Path,
    pair_receipt_path: Path,
    protocol_path: Path,
    output_root: Path,
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    pair_scope = str(json.loads(pairs_path.read_text(encoding="utf-8")).get("scope", ""))
    scope = (
        pair_scope.removesuffix("_target_reference_pairs") + "_windowed_crops"
        if pair_scope.endswith("_target_reference_pairs")
        else "outcome_blind_stanford_2025_sentinel2_l1c_windowed_crops"
    )
    manifests = [item["manifest"] for item in results if "manifest" in item]
    errors = [item for item in results if "error" in item]
    statuses: dict[str, int] = {}
    for item in results:
        status = str(item["status"])
        statuses[status] = statuses.get(status, 0) + 1
    target_valid = [
        float(item["radiometric_observability"]["target_all_band_nonzero_fraction"])
        for item in manifests
    ]
    reference_valid = [
        float(item["radiometric_observability"]["reference_all_band_nonzero_fraction"])
        for item in manifests
    ]
    asset_bytes = sum(
        int(asset["bytes"])
        for item in manifests
        for asset in item["assets"].values()
    )
    payload = {
        "schema_version": 1,
        "scope": scope,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "bindings": {
            "pair_manifest": {
                "path": pairs_path.relative_to(root).as_posix(),
                "sha256": sha256(pairs_path),
            },
            "pair_receipt": {
                "path": pair_receipt_path.relative_to(root).as_posix(),
                "sha256": sha256(pair_receipt_path),
            },
            "protocol": {
                "path": protocol_path.relative_to(root).as_posix(),
                "sha256": sha256(protocol_path),
            },
            "script": {
                "path": Path(__file__).resolve().relative_to(root).as_posix(),
                "sha256": sha256(Path(__file__).resolve()),
            },
        },
        "contract": {
            "ignored_output_root": output_root.relative_to(root).as_posix(),
            "window_only": True,
            "full_products_downloaded": False,
            "spectral_product": "Sentinel-2 Level-1C top-of-atmosphere raw DN",
            "band_order": list(BAND_ORDER),
            "shape": [256, 256],
            "target_grid": "target B02 native 10 m grid",
            "ten_meter_resampling": "nearest; target B02 native window",
            "twenty_meter_resampling": "bilinear",
            "cloud_or_l2_assets_acquired": False,
            "landsat_status": (
                "pending exact USGS EROS Collection 2 Level-1 authentication; "
                "no Landsat L2 substitution"
            ),
        },
        "outcome_blindness": {
            "release_labels_or_rates_accessed": False,
            "detector_outcomes_accessed": False,
            "statement": "No release outcomes or detector outcomes were accessed.",
        },
        "summary": {
            "requested_pairs": len(results),
            "verified_pairs": len(manifests),
            "acquisition_errors": len(errors),
            "all_pairs_complete": len(manifests) == len(results) and not errors,
            "status_counts": dict(sorted(statuses.items())),
            "ignored_crop_assets": 2 * len(manifests),
            "ignored_crop_bytes": asset_bytes,
            "target_all_band_nonzero_fraction_min": (
                None if not target_valid else round(min(target_valid), 8)
            ),
            "target_all_band_nonzero_fraction_median": (
                None if not target_valid else round(float(np.median(target_valid)), 8)
            ),
            "reference_all_band_nonzero_fraction_min": (
                None if not reference_valid else round(min(reference_valid), 8)
            ),
            "reference_all_band_nonzero_fraction_median": (
                None if not reference_valid else round(float(np.median(reference_valid)), 8)
            ),
        },
        "samples": [
            {
                "event_id": item["event_id"],
                "target_scene_id": item["target_scene_id"],
                "reference_scene_id": item["reference_scene_id"],
                "reference_tier": item["reference_tier"],
                "target_all_band_nonzero_fraction": item["radiometric_observability"][
                    "target_all_band_nonzero_fraction"
                ],
                "reference_all_band_nonzero_fraction": item[
                    "radiometric_observability"
                ]["reference_all_band_nonzero_fraction"],
                "assets": item["assets"],
            }
            for item in manifests
        ],
        "errors": errors,
        "runtime": {
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True
            ).strip(),
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "rasterio": rasterio.__version__,
        },
    }
    assert_no_outcome_fields(payload)
    return payload


def compact_receipt(root: Path, manifest_path: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "schema_version": 1,
        "scope": manifest["scope"],
        "generated_at_utc": manifest["generated_at_utc"],
        "status": (
            "acquisition_complete"
            if manifest["summary"]["all_pairs_complete"]
            else "acquisition_incomplete"
        ),
        "bindings": manifest["bindings"],
        "crop_manifest": {
            "path": manifest_path.relative_to(root).as_posix(),
            "sha256": sha256(manifest_path),
            "tracked": False,
        },
        "contract": manifest["contract"],
        "outcome_blindness": manifest["outcome_blindness"],
        "summary": manifest["summary"],
    }
    assert_no_outcome_fields(payload)
    return payload


def write_markdown(path: Path, receipt: dict[str, Any]) -> None:
    summary = receipt["summary"]
    lines = [
        "# Stanford controlled-release Sentinel-2 L1C windowed crops",
        "",
        f"- Status: `{receipt['status']}`",
        f"- Requested/verified pairs: {summary['requested_pairs']} / {summary['verified_pairs']}",
        f"- Acquisition errors: {summary['acquisition_errors']}",
        f"- Ignored crop assets: {summary['ignored_crop_assets']}",
        f"- Ignored crop bytes: {summary['ignored_crop_bytes']:,}",
        f"- Target all-band nonzero fraction, min/median: {summary['target_all_band_nonzero_fraction_min']} / {summary['target_all_band_nonzero_fraction_median']}",
        f"- Reference all-band nonzero fraction, min/median: {summary['reference_all_band_nonzero_fraction_min']} / {summary['reference_all_band_nonzero_fraction_median']}",
        f"- Ignored crop manifest: `{receipt['crop_manifest']['path']}`",
        f"- Crop manifest SHA-256: `{receipt['crop_manifest']['sha256']}`",
        "",
        "Only six-band 256x256 Sentinel-2 Level-1C windows were read and retained. No full product, Level-2A, SCL, cloud product, release label, release rate, or detector output was accessed.",
        "",
        "Ten-meter bands use nearest/native target-grid sampling; B11/B12 use bilinear resampling. Raw uint16 L1C DNs are preserved without scale or processing-offset correction.",
        "",
        "Landsat remains pending exact USGS EROS Collection 2 Level-1 authentication. No Landsat Level-2 substitute is permitted.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", default=DEFAULT_PAIRS.as_posix())
    parser.add_argument("--pair-receipt", default=DEFAULT_PAIR_RECEIPT.as_posix())
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL.as_posix())
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT.as_posix())
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST.as_posix())
    parser.add_argument("--output-json", default=DEFAULT_JSON.as_posix())
    parser.add_argument("--output-markdown", default=DEFAULT_MARKDOWN.as_posix())
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Development smoke only: process the first N frozen pairs",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.workers <= 0:
        parser.error("workers must be positive")
    if args.limit is not None and args.limit <= 0:
        parser.error("limit must be positive")
    root = repo_root()
    try:
        pairs_path = safe_repo_path(root, args.pairs)
        pair_receipt_path = safe_repo_path(root, args.pair_receipt)
        protocol_path = safe_repo_path(root, args.protocol)
        output_root = safe_repo_path(root, args.output_root)
        manifest_path = safe_repo_path(root, args.manifest)
        output_json = safe_repo_path(root, args.output_json)
        output_markdown = safe_repo_path(root, args.output_markdown)
        assert_ignored_storage(root, output_root)
        assert_ignored_storage(root, manifest_path)
        pairs, size = load_and_validate_inputs(
            root,
            pairs_path,
            pair_receipt_path,
            protocol_path,
        )
        if args.limit is not None:
            if args.limit > len(pairs):
                raise ValueError(
                    f"limit exceeds the frozen pair count: {args.limit} > {len(pairs)}"
                )
            pairs = pairs[: args.limit]

        def run(pair: dict[str, Any]) -> dict[str, Any]:
            try:
                return acquire_one(
                    root,
                    output_root,
                    pair,
                    size=size,
                    overwrite=args.overwrite,
                    verify_only=args.verify_only,
                )
            except Exception as exc:
                return {
                    "status": "acquisition_error",
                    "event_id": pair.get("event_id"),
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:1000],
                }

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            results = list(executor.map(run, pairs))
        manifest = build_crop_manifest(
            root,
            pairs_path,
            pair_receipt_path,
            protocol_path,
            output_root,
            results,
        )
        write_json(manifest_path, manifest)
        receipt = compact_receipt(root, manifest_path, manifest)
        write_json(output_json, receipt)
        write_markdown(output_markdown, receipt)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(
            json.dumps(
                {"ok": False, "error_type": type(exc).__name__, "error": str(exc)},
                sort_keys=True,
            )
        )
        return 2
    result = {
        "ok": receipt["summary"]["all_pairs_complete"],
        "requested_pairs": receipt["summary"]["requested_pairs"],
        "verified_pairs": receipt["summary"]["verified_pairs"],
        "acquisition_errors": receipt["summary"]["acquisition_errors"],
        "manifest": manifest_path.relative_to(root).as_posix(),
        "output_json": output_json.relative_to(root).as_posix(),
        "output_markdown": output_markdown.relative_to(root).as_posix(),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["ok"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
