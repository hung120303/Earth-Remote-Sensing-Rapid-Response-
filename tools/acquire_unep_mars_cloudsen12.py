#!/usr/bin/env python3
"""Generate frozen CloudSEN12+ observability masks for UNEP MARS S2 crops."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import torch
from rasterio.enums import Resampling

from acquire_emit_v002_cloudsen12 import (
    BAND_ORDER,
    EXTRA_BANDS,
    LOCAL_BANDS,
    MODEL_FILE,
    MODEL_NAME,
    MODEL_REPOSITORY,
    MODEL_SHA256,
    ordered_stack,
)
from acquire_emit_v002_l1c_crops import read_to_grid, safe_repo_path, sha256, write_raster


DEFAULT_ASSETS = Path(".research/unep_mars_post2024/nonsealed_exact_assets.jsonl")
DEFAULT_CROP_ROOT = Path(".research/unep_mars_post2024/crops")
DEFAULT_WEIGHTS = Path(
    "EarthRemoteSensingRapidResponse/Data Collection/s2_emit_pairs/"
    "emit-v002-external-l1c-2026-07/_cloudsen12_weights"
)
DEFAULT_JSON = Path("reports/acquisition/unep_mars_post2024_cloudsen12.json")
DEFAULT_MARKDOWN = Path("reports/acquisition/UNEP_MARS_POST2024_CLOUDSEN12.md")
MIN_SCENE_CLEAR = 0.8
MIN_PLUME_CLEAR = 0.8


def repo_root() -> Path:
    value = subprocess.check_output(
        ["git", "rev-parse", "--show-toplevel"], text=True
    ).strip()
    if not value:
        raise RuntimeError("Could not resolve repository root")
    return Path(value).resolve()


def pair_identity(asset_record: dict[str, Any], crop_manifest: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            {"asset_record": asset_record, "crop_manifest": crop_manifest},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def extra_band_url(b02_url: str, band: str) -> str:
    if band not in EXTRA_BANDS:
        raise ValueError(f"Not a CloudSEN12 extra band: {band}")
    suffix = "/B02.jp2"
    if not b02_url.startswith("https://sentinel-s2-l1c.s3.amazonaws.com/"):
        raise ValueError("CloudSEN12 inputs must use the official public L1C bucket")
    if not b02_url.endswith(suffix):
        raise ValueError(f"Unexpected Sentinel-2 B02 URL: {b02_url}")
    return b02_url[: -len(suffix)] + f"/{band}.jp2"


def local_target(path: Path) -> dict[str, np.ndarray]:
    with rasterio.open(path) as source:
        descriptions = tuple(source.descriptions)
        if descriptions[:6] != LOCAL_BANDS or source.count != 12:
            raise ValueError(f"Unexpected exact crop band order: {descriptions}")
        values = source.read(indexes=range(1, 7), out_dtype="uint16")
    return {band: values[index] for index, band in enumerate(LOCAL_BANDS)}


def verify_sidecar(
    root: Path, path: Path, *, expected_identity: str
) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest["input_identity_sha256"] != expected_identity:
        raise ValueError(f"CloudSEN12 input identity changed: {path}")
    record = manifest["asset"]
    asset = safe_repo_path(root, record["path"])
    if asset.stat().st_size != record["bytes"] or sha256(asset) != record["sha256"]:
        raise ValueError(f"CloudSEN12 asset verification failed: {asset}")
    return manifest


def acquire_one(
    root: Path,
    crop_root: Path,
    asset_record: dict[str, Any],
    model: Any,
    model_lock: threading.Lock,
    *,
    band_workers: int,
    overwrite: bool,
) -> dict[str, Any]:
    sample_id = asset_record["sample_id"]
    crop_dir = crop_root / sample_id
    crop_manifest_path = crop_dir / "manifest.json"
    crop_manifest = json.loads(crop_manifest_path.read_text(encoding="utf-8"))
    if crop_manifest["research_role"] == "sealed_external":
        raise ValueError("Sealed-external crop reached CloudSEN12 acquisition")
    if crop_manifest["sensor_family"] != "Sentinel-2":
        raise ValueError("CloudSEN12 received a non-Sentinel sample")
    identity = pair_identity(asset_record, crop_manifest)
    sidecar = crop_dir / "cloudsen12.manifest.json"
    if sidecar.is_file() and not overwrite:
        return verify_sidecar(root, sidecar, expected_identity=identity)

    contract = crop_manifest["product_contract"]
    crs = contract["crs"]
    transform = rasterio.Affine(*contract["transform"])
    size = int(contract["shape"][0])
    image_path = safe_repo_path(root, crop_manifest["assets"]["image"]["path"])
    mask_path = safe_repo_path(root, crop_manifest["assets"]["plume_mask"]["path"])
    local = local_target(image_path)
    b02 = asset_record["target"]["assets"]["B02"]

    def read_extra(band: str) -> tuple[str, np.ndarray]:
        return (
            band,
            read_to_grid(
                extra_band_url(b02, band),
                crs=crs,
                transform=transform,
                size=size,
                resampling=Resampling.bilinear,
                dtype="uint16",
            ),
        )

    with ThreadPoolExecutor(max_workers=band_workers) as executor:
        extras = dict(executor.map(read_extra, EXTRA_BANDS))
    stack = ordered_stack(local, extras)
    invalid = np.any(stack == 0, axis=0)
    with model_lock:
        cloud = np.asarray(
            model.predict(stack.astype(np.float32) / 10_000.0), dtype=np.uint8
        )
    if cloud.shape != (size, size) or not set(np.unique(cloud)).issubset({0, 1, 2, 3}):
        raise ValueError(f"Invalid CloudSEN12 prediction for {sample_id}")
    cloud[invalid] = 4
    with rasterio.open(mask_path) as source:
        plume = source.read(1) > 0
    clear = cloud == 0
    scene_clear = float(np.mean(clear))
    plume_pixels = int(np.count_nonzero(plume))
    plume_clear = float(np.mean(clear[plume])) if plume_pixels else 0.0
    gate_pass = (
        crop_manifest["quality"]["gate_pass_before_cloud"]
        and scene_clear >= MIN_SCENE_CLEAR
        and plume_clear >= MIN_PLUME_CLEAR
    )
    output_path = crop_dir / "cloud_mask.tif"
    write_raster(
        output_path,
        cloud,
        crs=crs,
        transform=transform,
        descriptions=["CloudSEN12_UNetMobV2_V2"],
        nodata=4,
    )
    output_record = {
        "path": output_path.relative_to(root).as_posix(),
        "bytes": output_path.stat().st_size,
        "sha256": sha256(output_path),
    }
    counts = Counter(int(value) for value in cloud.ravel())
    manifest = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "sample_id": sample_id,
        "group_id": crop_manifest["group_id"],
        "research_role": crop_manifest["research_role"],
        "input_identity_sha256": identity,
        "model": {
            "name": MODEL_NAME,
            "repository": MODEL_REPOSITORY,
            "file": MODEL_FILE,
            "sha256": MODEL_SHA256,
            "band_order": list(BAND_ORDER),
            "input_scaling": "L1C digital number / 10000",
        },
        "quality": {
            "gate_pass": gate_pass,
            "minimum_scene_clear_fraction": MIN_SCENE_CLEAR,
            "minimum_plume_clear_fraction": MIN_PLUME_CLEAR,
            "scene_clear_fraction": round(scene_clear, 8),
            "plume_clear_fraction": round(plume_clear, 8),
            "plume_pixels": plume_pixels,
            "class_counts": {str(key): counts[key] for key in sorted(counts)},
        },
        "asset": output_record,
    }
    temporary = sidecar.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, sidecar)
    return manifest


def write_markdown(report: dict[str, Any], path: Path) -> None:
    summary = report["summary"]
    lines = [
        "# UNEP MARS post-2024 CloudSEN12+ acquisition",
        "",
        f"Generated: {report['generated_at_utc']}.",
        "",
        "## Result",
        "",
        f"- Exact Sentinel-2 crops processed: **{summary['processed']:,}**.",
        f"- Cloud/radiometry/geometry gate pass: **{summary['gate_pass']:,}**.",
        f"- Gate failures retained in the audit: **{summary['gate_fail']:,}**.",
        f"- Acquisition errors: **{summary['errors']:,}**.",
        f"- Median scene/plume clear fraction: **{summary['scene_clear_median']:.3f} / {summary['plume_clear_median']:.3f}**.",
        f"- Auxiliary-training gate pass: **{summary['gate_pass_by_role'].get('auxiliary_training', 0):,}**.",
        f"- Development gate pass: **{summary['gate_pass_by_role'].get('development', 0):,}**.",
        "",
        "## Frozen contract",
        "",
        f"- Model: `{MODEL_REPOSITORY}/{MODEL_FILE}` (`{MODEL_SHA256}`).",
        "- Thirteen Sentinel-2 L1C bands in the published CloudSEN12 order, scaled by 1/10,000.",
        f"- Scene and plume clear-fraction gates: **{MIN_SCENE_CLEAR:.2f} / {MIN_PLUME_CLEAR:.2f}**.",
        "- Cloud classes are 0 clear, 1 thick cloud, 2 thin cloud, 3 shadow, and 4 invalid.",
        "- Cloud predictions affect observability only and are not methane labels or model inputs beyond the released cloud channel.",
        "- Sealed-external samples were excluded.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", default=DEFAULT_ASSETS.as_posix())
    parser.add_argument("--crop-root", default=DEFAULT_CROP_ROOT.as_posix())
    parser.add_argument("--weights-dir", default=DEFAULT_WEIGHTS.as_posix())
    parser.add_argument("--output-json", default=DEFAULT_JSON.as_posix())
    parser.add_argument("--output-markdown", default=DEFAULT_MARKDOWN.as_posix())
    parser.add_argument("--candidate-workers", type=int, default=4)
    parser.add_argument("--band-workers", type=int, default=2)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    root = repo_root()
    assets_path = safe_repo_path(root, args.assets)
    crop_root = safe_repo_path(root, args.crop_root)
    weights_dir = safe_repo_path(root, args.weights_dir)
    output_json = safe_repo_path(root, args.output_json)
    output_markdown = safe_repo_path(root, args.output_markdown)
    records = [
        item
        for item in (
            json.loads(line) for line in assets_path.read_text(encoding="utf-8").splitlines()
        )
        if item["status"] == "resolved"
        and item["sensor_family"] == "Sentinel-2"
        and item["research_role"] != "sealed_external"
    ]
    records.sort(key=lambda item: item["sample_id"])
    if args.limit is not None:
        records = records[: args.limit]
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    try:
        from cloudsen12_models import cloudsen12
    except ImportError as exc:
        raise RuntimeError("Install cloudsen12-models==1.0.2 from requirements.txt") from exc
    model = cloudsen12.load_model_by_name(
        MODEL_NAME, weights_folder=str(weights_dir), device=torch.device(device)
    )
    if tuple(model.bands) != BAND_ORDER:
        raise ValueError(f"CloudSEN12 band contract changed: {model.bands}")
    if sha256(weights_dir / MODEL_FILE) != MODEL_SHA256:
        raise ValueError("CloudSEN12 weight SHA-256 mismatch")
    model_lock = threading.Lock()

    completed: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=args.candidate_workers) as executor:
        futures = {
            executor.submit(
                acquire_one,
                root,
                crop_root,
                record,
                model,
                model_lock,
                band_workers=args.band_workers,
                overwrite=args.overwrite,
            ): record["sample_id"]
            for record in records
        }
        for count, future in enumerate(as_completed(futures), start=1):
            sample_id = futures[future]
            try:
                completed.append(future.result())
            except Exception as exc:
                errors.append(
                    {
                        "sample_id": sample_id,
                        "error_type": type(exc).__name__,
                        "error": str(exc)[:500],
                    }
                )
            if count % 10 == 0 or count == len(futures):
                print(f"clouded {count}/{len(futures)} errors={len(errors)}", flush=True)
    completed.sort(key=lambda item: item["sample_id"])
    gate_pass_records = [item for item in completed if item["quality"]["gate_pass"]]
    gate_fail_records = [item for item in completed if not item["quality"]["gate_pass"]]
    report = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "nonsealed Sentinel-2 CloudSEN12 acquisition complete",
        "inputs": {
            "assets": {"path": args.assets, "sha256": sha256(assets_path)},
            "crop_root": args.crop_root,
        },
        "contract": {
            "model_name": MODEL_NAME,
            "model_repository": MODEL_REPOSITORY,
            "model_file": MODEL_FILE,
            "model_sha256": MODEL_SHA256,
            "band_order": list(BAND_ORDER),
            "minimum_scene_clear_fraction": MIN_SCENE_CLEAR,
            "minimum_plume_clear_fraction": MIN_PLUME_CLEAR,
            "sealed_external_accessed": False,
        },
        "summary": {
            "processed": len(completed),
            "gate_pass": len(gate_pass_records),
            "gate_fail": len(gate_fail_records),
            "gate_pass_by_role": dict(
                sorted(Counter(item["research_role"] for item in gate_pass_records).items())
            ),
            "gate_fail_by_role": dict(
                sorted(Counter(item["research_role"] for item in gate_fail_records).items())
            ),
            "scene_clear_median": round(
                float(np.median([item["quality"]["scene_clear_fraction"] for item in completed])),
                8,
            ),
            "plume_clear_median": round(
                float(np.median([item["quality"]["plume_clear_fraction"] for item in completed])),
                8,
            ),
            "by_role": dict(sorted(Counter(item["research_role"] for item in completed).items())),
            "errors": len(errors),
            "mask_bytes": sum(item["asset"]["bytes"] for item in completed),
        },
        "errors": errors,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(report, output_markdown)
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    if errors:
        raise RuntimeError(f"{len(errors)} CloudSEN12 acquisitions failed")


if __name__ == "__main__":
    main()
