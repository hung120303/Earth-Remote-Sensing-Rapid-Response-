#!/usr/bin/env python3
"""Generate the exact MARS Sentinel-2 CloudSEN12+ input for external crops."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import torch
from rasterio.enums import Resampling

from acquire_emit_v002_l1c_crops import (
    read_to_grid,
    safe_repo_path,
    sha256,
    verify_local_manifest,
    write_raster,
)

DEFAULT_PAIRS = Path("reports/acquisition/emit_v002_l1c_pairs.json")
DEFAULT_CROP_ROOT = Path(
    "EarthRemoteSensingRapidResponse/Data Collection/s2_emit_pairs/"
    "emit-v002-external-l1c-2026-07"
)
DEFAULT_JSON = Path("reports/acquisition/emit_v002_cloudsen12_acquisition.json")
DEFAULT_MARKDOWN = Path("reports/acquisition/EMIT_V002_CLOUDSEN12_ACQUISITION.md")
MODEL_NAME = "UNetMobV2_V2"
MODEL_REPOSITORY = "isp-uv-es/cloudsen12_models"
MODEL_FILE = "UNetMobV2_V2.pt"
MODEL_SHA256 = "218fa69aa3c7212d4e690b48af88ac6f3c976fc50d07f275b8fd623909183d7a"
BAND_ORDER = (
    "B01",
    "B02",
    "B03",
    "B04",
    "B05",
    "B06",
    "B07",
    "B08",
    "B8A",
    "B09",
    "B10",
    "B11",
    "B12",
)
LOCAL_BANDS = ("B02", "B03", "B04", "B08", "B11", "B12")
EXTRA_BANDS = tuple(item for item in BAND_ORDER if item not in LOCAL_BANDS)


def repo_root() -> Path:
    value = subprocess.check_output(
        ["git", "rev-parse", "--show-toplevel"], text=True
    ).strip()
    return Path(value).resolve()


def official_band_url(tileinfo_url: str, band: str) -> str:
    if band not in BAND_ORDER:
        raise ValueError(f"Unsupported Sentinel-2 L1C band: {band}")
    suffix = "/tileInfo.json"
    if not tileinfo_url.startswith("https://sentinel-s2-l1c.s3.amazonaws.com/"):
        raise ValueError("CloudSEN12 inputs must use the official public L1C bucket")
    if not tileinfo_url.endswith(suffix):
        raise ValueError(f"Unexpected L1C tileInfo URL: {tileinfo_url}")
    return tileinfo_url[: -len(suffix)] + f"/{band}.jp2"


def local_target_bands(path: Path) -> dict[str, np.ndarray]:
    with rasterio.open(path) as source:
        descriptions = list(source.descriptions)
        if descriptions != list(LOCAL_BANDS):
            raise ValueError(f"Unexpected local target band order: {descriptions}")
        values = source.read(out_dtype="uint16")
    return {band: values[index] for index, band in enumerate(LOCAL_BANDS)}


def ordered_stack(
    local: dict[str, np.ndarray], extras: dict[str, np.ndarray]
) -> np.ndarray:
    values = {**local, **extras}
    missing = sorted(set(BAND_ORDER) - set(values))
    if missing:
        raise ValueError(f"CloudSEN12 stack lacks bands: {missing}")
    shapes = {values[band].shape for band in BAND_ORDER}
    if len(shapes) != 1:
        raise ValueError(f"CloudSEN12 bands have inconsistent shapes: {shapes}")
    return np.stack([values[band] for band in BAND_ORDER])


def asset_record(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def verify_sidecar(root: Path, path: Path, expected_request: str) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest["pair_sha256"] != expected_request:
        raise ValueError(f"CloudSEN12 pair identity changed: {path}")
    asset = safe_repo_path(root, manifest["asset"]["path"])
    if asset.stat().st_size != manifest["asset"]["bytes"] or sha256(asset) != manifest["asset"]["sha256"]:
        raise ValueError(f"CloudSEN12 asset verification failed: {asset}")
    return manifest


def pair_sha256(pair: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(pair, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def acquire_one(
    root: Path,
    crop_root: Path,
    pair: dict[str, Any],
    model: Any,
    model_lock: threading.Lock,
    *,
    band_workers: int,
    overwrite: bool,
    verify_only: bool,
) -> dict[str, Any] | None:
    scene_dir = crop_root / pair["granule_id"]
    crop_manifest_path = scene_dir / "manifest.json"
    crop_manifest = verify_local_manifest(root, crop_manifest_path)
    if not crop_manifest["quality"]["gate_pass"]:
        return None
    identity = pair_sha256(pair)
    sidecar_path = scene_dir / "cloudsen12.manifest.json"
    if sidecar_path.is_file() and not overwrite:
        return verify_sidecar(root, sidecar_path, identity)
    if verify_only:
        raise FileNotFoundError(sidecar_path)
    contract = crop_manifest["product_contract"]
    transform = rasterio.Affine(*contract["transform"])
    crs = contract["crs"]
    size = int(contract["shape"][0])
    target_path = safe_repo_path(root, crop_manifest["assets"]["target_l1c"]["path"])
    local = local_target_bands(target_path)
    tileinfo = pair["target"]["l1c_tileinfo_metadata"]

    def read_band(band: str) -> tuple[str, np.ndarray]:
        return (
            band,
            read_to_grid(
                official_band_url(tileinfo, band),
                crs=crs,
                transform=transform,
                size=size,
                resampling=Resampling.bilinear,
                dtype="uint16",
            ),
        )

    with ThreadPoolExecutor(max_workers=band_workers) as executor:
        extras = dict(executor.map(read_band, EXTRA_BANDS))
    stack = ordered_stack(local, extras)
    invalid = np.any(stack == 0, axis=0)
    with model_lock:
        prediction = np.asarray(
            model.predict(stack.astype(np.float32) / 10_000.0), dtype=np.uint8
        )
    if prediction.shape != (size, size) or not set(np.unique(prediction)).issubset({0, 1, 2, 3}):
        raise ValueError(f"Invalid CloudSEN12 prediction for {pair['group_id']}")
    prediction[invalid] = 4
    output_path = scene_dir / "target.cloudsen12.tif"
    write_raster(
        output_path,
        prediction,
        crs=crs,
        transform=transform,
        descriptions=["CloudSEN12_UNetMobV2_V2"],
        nodata=4,
    )
    counts = Counter(int(value) for value in prediction.ravel())
    fractions = {
        name: round(counts[value] / float(prediction.size), 8)
        for value, name in enumerate(("clear", "thick_cloud", "thin_cloud", "cloud_shadow", "invalid"))
    }
    manifest = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "group_id": pair["group_id"],
        "granule_id": pair["granule_id"],
        "target_scene_id": pair["target"]["l1c_scene_id"],
        "pair_sha256": identity,
        "model": {
            "name": MODEL_NAME,
            "repository": MODEL_REPOSITORY,
            "file": MODEL_FILE,
            "sha256": MODEL_SHA256,
            "band_order": list(BAND_ORDER),
            "input_scaling": "L1C digital number / 10000",
        },
        "class_fractions": fractions,
        "asset": asset_record(output_path, root),
    }
    temporary = sidecar_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, sidecar_path)
    return manifest


def compact_report(source: dict[str, Any], records: list[dict[str, Any]]) -> dict[str, Any]:
    output = []
    for item in sorted(records, key=lambda value: value["group_id"]):
        output.append(
            {
                "group_id": item["group_id"],
                "granule_id": item["granule_id"],
                "target_scene_id": item["target_scene_id"],
                "pair_sha256": item["pair_sha256"],
                "class_fractions": item["class_fractions"],
                "asset": item["asset"],
            }
        )
    return {
        "contract": {
            "input_pairs_sha256": hashlib.sha256(
                json.dumps(source, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
            "eligibility": "predeclared L2A-SCL/radiometry/containment gate pass only",
            "eligibility_uses_model_predictions": False,
            "model_name": MODEL_NAME,
            "model_repository": MODEL_REPOSITORY,
            "model_file": MODEL_FILE,
            "model_sha256": MODEL_SHA256,
            "band_order": list(BAND_ORDER),
        },
        "summary": {
            "generated": len(output),
            "hash_verified": len(output),
            "raw_mask_bytes": sum(item["asset"]["bytes"] for item in output),
        },
        "records": output,
    }


def markdown(result: dict[str, Any]) -> str:
    summary = result["summary"]
    contract = result["contract"]
    return f"""# EMIT V002 external CloudSEN12 acquisition

## Result

- Exact MARS Sentinel-2 cloud model: **{contract['model_name']}**.
- Prediction-blind gate-pass scenes processed: **{summary['generated']}**.
- Hash-verified masks: **{summary['hash_verified']}**.
- Ignored raw mask bytes: **{summary['raw_mask_bytes']:,}**.

Eligibility was fixed from independent L2A SCL, radiometric validity, and plume containment. CloudSEN12 output was generated only afterward as a model input, so it cannot alter cohort membership. The 13-band order, input scaling, model repository, and exact weight hash are frozen in the JSON report.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", default=DEFAULT_PAIRS.as_posix())
    parser.add_argument("--crop-root", default=DEFAULT_CROP_ROOT.as_posix())
    parser.add_argument("--output-json", default=DEFAULT_JSON.as_posix())
    parser.add_argument("--output-markdown", default=DEFAULT_MARKDOWN.as_posix())
    parser.add_argument("--weights-dir")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--candidate-workers", type=int, default=4)
    parser.add_argument("--band-workers", type=int, default=2)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    root = repo_root()
    pairs_path = safe_repo_path(root, args.pairs)
    source = json.loads(pairs_path.read_text(encoding="utf-8"))
    pairs = [item for item in source["pairs"] if item["status"] == "paired"]
    if args.limit is not None:
        pairs = pairs[: args.limit]
    crop_root = safe_repo_path(root, args.crop_root)
    weights_dir = (
        safe_repo_path(root, args.weights_dir)
        if args.weights_dir
        else crop_root / "_cloudsen12_weights"
    )
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
    weights_path = weights_dir / MODEL_FILE
    if sha256(weights_path) != MODEL_SHA256:
        raise ValueError("CloudSEN12 weight SHA-256 does not match the frozen model")
    model_lock = threading.Lock()

    def run(pair: dict[str, Any]) -> dict[str, Any] | None:
        return acquire_one(
            root,
            crop_root,
            pair,
            model,
            model_lock,
            band_workers=args.band_workers,
            overwrite=args.overwrite,
            verify_only=args.verify_only,
        )

    with ThreadPoolExecutor(max_workers=args.candidate_workers) as executor:
        completed = list(executor.map(run, pairs))
    records = [item for item in completed if item is not None]
    print(json.dumps({"candidates": len(pairs), "eligible_complete": len(records)}), flush=True)
    result = compact_report(source, records)
    json_path = safe_repo_path(root, args.output_json)
    markdown_path = safe_repo_path(root, args.output_markdown)
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(markdown(result), encoding="utf-8")
    print(json.dumps(result["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
