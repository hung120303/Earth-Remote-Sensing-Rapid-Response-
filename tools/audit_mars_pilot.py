#!/usr/bin/env python3
"""Verify and audit the deterministic MARS-S2L raster-contract pilot.

This script is intentionally separate from acquisition. It proves the local
files still match the pinned manifest, then inspects the raster grids, band
order, dtypes, nodata values, mask domains, and positive-label support. The
raw rasters remain under the ignored acquisition tree; only compact evidence
reports are written to Git-visible paths.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import rasterio

from acquire_mars_metadata import DEFAULT_OUTPUT, REPO_ID, REVISION, checked_output_dir, repo_root, sha256
from acquire_mars_pilot import PILOT_MANIFEST, safe_asset_path, verify_manifest

DEFAULT_JSON = Path("reports/acquisition/mars_s2l_contract_pilot.json")
DEFAULT_MARKDOWN = Path("reports/acquisition/MARS_S2L_CONTRACT_PILOT.md")
EXPECTED_IMAGE_DESCRIPTIONS = (
    "B02",
    "B03",
    "B04",
    "B08",
    "B11",
    "B12",
    "B02_bg",
    "B03_bg",
    "B04_bg",
    "B08_bg",
    "B11_bg",
    "B12_bg",
)
EXPECTED_POSITIVE_ROLES = {"image", "cloud_mask", "plume_mask", "methane_enhancement"}
EXPECTED_NEGATIVE_ROLES = {"image", "cloud_mask"}


def git_commit(root: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


def tracked_dirty(root: Path) -> bool:
    output = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=root, text=True
    )
    return bool(output.strip())


def safe_output(root: Path, value: str) -> Path:
    path = (root / value).resolve()
    if root not in path.parents:
        raise ValueError("Report output must resolve beneath the repository root")
    return path


def transform_list(transform: Any) -> list[float]:
    return [round(float(value), 12) for value in transform[:6]]


def finite_summary(values: np.ndarray) -> dict[str, float | int | None]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"count": 0, "min": None, "mean": None, "max": None}
    return {
        "count": int(finite.size),
        "min": round(float(np.min(finite)), 6),
        "mean": round(float(np.mean(finite)), 6),
        "max": round(float(np.max(finite)), 6),
    }


def raster_header(dataset: rasterio.io.DatasetReader) -> dict[str, Any]:
    return {
        "width": dataset.width,
        "height": dataset.height,
        "count": dataset.count,
        "crs": None if dataset.crs is None else dataset.crs.to_string(),
        "transform": transform_list(dataset.transform),
        "resolution": [abs(float(dataset.transform.a)), abs(float(dataset.transform.e))],
        "dtypes": list(dataset.dtypes),
        "nodata": dataset.nodata,
        "descriptions": list(dataset.descriptions),
    }


def same_grid(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return all(
        left[key] == right[key]
        for key in ("width", "height", "crs", "transform", "resolution")
    )


def inspect_sample(
    metadata_dir: Path, sample: dict[str, Any]
) -> tuple[dict[str, Any], list[str], list[str]]:
    sample_id = sample["id_loc_image"]
    label = sample["label"]
    expected_roles = EXPECTED_POSITIVE_ROLES if label == "positive" else EXPECTED_NEGATIVE_ROLES
    role_paths = {item["role"]: item["path"] for item in sample["assets"]}
    violations: list[str] = []
    warnings: list[str] = []
    if set(role_paths) != expected_roles:
        violations.append(
            f"{sample_id}: expected roles {sorted(expected_roles)}, got {sorted(role_paths)}"
        )

    details: dict[str, Any] = {
        "id_loc_image": sample_id,
        "id_location": sample["id_location"],
        "split": sample["split"],
        "label": label,
        "unseen_test_location": bool(sample["unseen_test_location"]),
        "metadata_percentage_clear": round(float(sample["percentage_clear"]), 6),
        "assets": role_paths,
    }

    with rasterio.open(safe_asset_path(metadata_dir, role_paths["image"])) as source:
        image_header = raster_header(source)
        image = source.read()
    details["image"] = image_header
    if image_header["count"] != 12:
        violations.append(f"{sample_id}: image has {image_header['count']} bands, expected 12")
    if tuple(image_header["descriptions"]) != EXPECTED_IMAGE_DESCRIPTIONS:
        violations.append(f"{sample_id}: unexpected image band descriptions")
    if set(image_header["dtypes"]) != {"uint16"}:
        violations.append(f"{sample_id}: image dtype is {image_header['dtypes']}, expected uint16")
    if image_header["nodata"] != 0.0:
        violations.append(f"{sample_id}: image nodata is {image_header['nodata']}, expected 0")
    if image_header["width"] != int(sample["width"]) or image_header["height"] != int(sample["height"]):
        violations.append(f"{sample_id}: image dimensions disagree with metadata")
    if image_header["crs"] != sample["crs"]:
        violations.append(
            f"{sample_id}: image CRS {image_header['crs']} disagrees with metadata {sample['crs']}"
        )

    target = image[:6]
    reference = image[6:12]
    target_valid = np.all(target != 0, axis=0)
    reference_valid = np.all(reference != 0, axis=0)
    paired_valid = target_valid & reference_valid
    paired_count = int(np.count_nonzero(paired_valid))
    target_reference_abs_difference = (
        np.abs(target.astype(np.float64) - reference.astype(np.float64))[:, paired_valid]
        if paired_count
        else np.array([], dtype=np.float64)
    )
    details["image_values"] = {
        "target_all_bands_valid_fraction": round(float(np.mean(target_valid)), 8),
        "reference_all_bands_valid_fraction": round(float(np.mean(reference_valid)), 8),
        "paired_all_bands_valid_fraction": round(float(np.mean(paired_valid)), 8),
        "target_nonzero": finite_summary(target[target != 0]),
        "reference_nonzero": finite_summary(reference[reference != 0]),
        "target_reference_absolute_difference": finite_summary(target_reference_abs_difference),
    }

    with rasterio.open(safe_asset_path(metadata_dir, role_paths["cloud_mask"])) as source:
        cloud_header = raster_header(source)
        cloud = source.read(1)
    cloud_values, cloud_counts = np.unique(cloud, return_counts=True)
    cloud_histogram = {
        str(int(value)): int(count) for value, count in zip(cloud_values, cloud_counts)
    }
    observed_clear = 100.0 * float(np.count_nonzero(cloud == 0)) / float(cloud.size)
    details["cloud_mask"] = {
        **cloud_header,
        "value_histogram": cloud_histogram,
        "observed_zero_value_percentage": round(observed_clear, 6),
        "metadata_clear_difference_percentage_points": round(
            observed_clear - float(sample["percentage_clear"]), 6
        ),
    }
    if cloud_header["count"] != 1 or set(cloud_header["dtypes"]) != {"uint8"}:
        violations.append(f"{sample_id}: cloud mask is not one-band uint8")
    if not set(int(value) for value in cloud_values).issubset({0, 1}):
        violations.append(f"{sample_id}: cloud mask contains values outside 0/1")
    if cloud_header["descriptions"] == [None]:
        warnings.append(f"{sample_id}: cloud mask has no descriptive band tag")
    elif cloud_header["descriptions"] != ["Cloudmask"]:
        violations.append(f"{sample_id}: unexpected cloud-mask description")
    if cloud_header["nodata"] in (0.0, 1.0):
        warnings.append(
            f"{sample_id}: cloud-mask nodata metadata overlaps an encoded 0/1 class; "
            "do not derive validity from GDAL nodata"
        )
    if not same_grid(image_header, cloud_header):
        violations.append(f"{sample_id}: image/cloud-mask grids differ")

    if label == "positive" and {"plume_mask", "methane_enhancement"}.issubset(role_paths):
        with rasterio.open(safe_asset_path(metadata_dir, role_paths["plume_mask"])) as source:
            plume_header = raster_header(source)
            plume = source.read(1)
        plume_values, plume_counts = np.unique(plume, return_counts=True)
        plume_pixels = int(np.count_nonzero(plume == 1))
        details["plume_mask"] = {
            **plume_header,
            "value_histogram": {
                str(int(value)): int(count)
                for value, count in zip(plume_values, plume_counts)
            },
            "positive_pixels": plume_pixels,
            "positive_area_fraction": round(plume_pixels / float(plume.size), 8),
        }
        if plume_header["count"] != 1 or set(plume_header["dtypes"]) != {"uint8"}:
            violations.append(f"{sample_id}: plume mask is not one-band uint8")
        if not set(int(value) for value in plume_values).issubset({0, 1}):
            violations.append(f"{sample_id}: plume mask contains values outside 0/1")
        if plume_pixels == 0:
            violations.append(f"{sample_id}: positive sample has an empty plume mask")
        if plume_header["descriptions"] == [None]:
            warnings.append(f"{sample_id}: plume mask has no descriptive band tag")
        elif plume_header["descriptions"] != ["Plumemask"]:
            violations.append(f"{sample_id}: unexpected plume-mask description")
        if not same_grid(image_header, plume_header):
            violations.append(f"{sample_id}: image/plume-mask grids differ")

        with rasterio.open(
            safe_asset_path(metadata_dir, role_paths["methane_enhancement"])
        ) as source:
            methane_header = raster_header(source)
            methane = source.read(1)
        methane_valid = np.isfinite(methane) & (methane != methane_header["nodata"])
        plume_methane = methane[(plume == 1) & np.isfinite(methane)]
        details["methane_enhancement"] = {
            **methane_header,
            "valid_values": finite_summary(methane[methane_valid]),
            "values_on_plume_mask": finite_summary(plume_methane),
            "nonzero_pixels": int(np.count_nonzero(methane_valid)),
        }
        if methane_header["count"] != 1 or set(methane_header["dtypes"]) != {"float64"}:
            violations.append(f"{sample_id}: methane enhancement is not one-band float64")
        if methane_header["descriptions"] == [None]:
            warnings.append(
                f"{sample_id}: methane enhancement has no units/description band tag"
            )
        elif methane_header["descriptions"] != ["DeltaCH4(ppm)"]:
            violations.append(f"{sample_id}: unexpected methane-enhancement units/description")
        if not same_grid(image_header, methane_header):
            violations.append(f"{sample_id}: image/methane-enhancement grids differ")
        if not np.all(np.isfinite(methane)):
            violations.append(f"{sample_id}: methane enhancement contains non-finite values")

    details["contract_ok"] = not violations
    details["warnings"] = warnings
    return details, violations, warnings


def aggregate(
    samples: list[dict[str, Any]], violations: list[str], warnings: list[str]
) -> dict[str, Any]:
    split_labels = Counter(f"{item['split']}:{item['label']}" for item in samples)
    crs_counts = Counter(item["image"]["crs"] for item in samples)
    dimensions = Counter(f"{item['image']['width']}x{item['image']['height']}" for item in samples)
    cloud_values: Counter[str] = Counter()
    cloud_nodata: Counter[str] = Counter()
    role_descriptions: dict[str, Counter[str]] = {
        "image": Counter(),
        "cloud_mask": Counter(),
        "plume_mask": Counter(),
        "methane_enhancement": Counter(),
    }
    clear_differences: list[float] = []
    paired_valid: list[float] = []
    plume_fractions: list[float] = []
    plume_methane_maxima: list[float] = []
    for item in samples:
        cloud_values.update(item["cloud_mask"]["value_histogram"])
        cloud_nodata[str(item["cloud_mask"]["nodata"])] += 1
        for role in role_descriptions:
            if role in item:
                role_descriptions[role][json.dumps(item[role]["descriptions"])] += 1
        clear_differences.append(item["cloud_mask"]["metadata_clear_difference_percentage_points"])
        paired_valid.append(item["image_values"]["paired_all_bands_valid_fraction"])
        if item["label"] == "positive":
            plume_fractions.append(item["plume_mask"]["positive_area_fraction"])
            maximum = item["methane_enhancement"]["values_on_plume_mask"]["max"]
            if maximum is not None:
                plume_methane_maxima.append(float(maximum))
    return {
        "sample_count": len(samples),
        "positive_samples": sum(item["label"] == "positive" for item in samples),
        "negative_samples": sum(item["label"] == "negative" for item in samples),
        "unseen_test_samples": sum(bool(item["unseen_test_location"]) for item in samples),
        "split_label_counts": dict(sorted(split_labels.items())),
        "crs_counts": dict(sorted(crs_counts.items())),
        "dimension_counts": dict(sorted(dimensions.items())),
        "cloud_mask_value_histogram": dict(sorted(cloud_values.items())),
        "cloud_mask_nodata_counts": dict(sorted(cloud_nodata.items())),
        "band_description_variants": {
            role: dict(sorted(counts.items())) for role, counts in role_descriptions.items()
        },
        "paired_valid_fraction": finite_summary(np.asarray(paired_valid)),
        "metadata_vs_raster_clear_difference_percentage_points": finite_summary(
            np.asarray(clear_differences)
        ),
        "positive_plume_area_fraction": finite_summary(np.asarray(plume_fractions)),
        "positive_plume_methane_max_ppm": finite_summary(np.asarray(plume_methane_maxima)),
        "contract_violation_count": len(violations),
        "contract_violations": violations,
        "warning_count": len(warnings),
        "warnings": warnings,
        "all_samples_pass_contract": not violations,
    }


def build_audit(root: Path, metadata_dir: Path) -> dict[str, Any]:
    manifest = verify_manifest(metadata_dir)
    manifest_path = metadata_dir / PILOT_MANIFEST
    samples: list[dict[str, Any]] = []
    violations: list[str] = []
    warnings: list[str] = []
    for sample in manifest["samples"]:
        inspected, sample_violations, sample_warnings = inspect_sample(metadata_dir, sample)
        samples.append(inspected)
        violations.extend(sample_violations)
        warnings.extend(sample_warnings)
    summary = aggregate(samples, violations, warnings)
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "repository": REPO_ID,
            "repository_url": f"https://huggingface.co/datasets/{REPO_ID}",
            "revision": REVISION,
            "license": manifest["source"]["license"],
            "pilot_manifest_sha256": sha256(manifest_path),
            "pilot_identity_sha256": manifest["integrity"]["pilot_identity_sha256"],
        },
        "source_integrity": {
            "asset_count": manifest["integrity"]["asset_count"],
            "total_bytes": manifest["integrity"]["total_bytes"],
            "all_assets_match_manifest_sha256": True,
        },
        "selection": manifest["selection"],
        "contract": {
            "sensor": "Sentinel-2 MSI",
            "product_level": "L1C",
            "grid": "200 x 200 pixels at 10 m in sample-local UTM",
            "image_layout": "12-band uint16 target/reference pair",
            "target_bands": list(EXPECTED_IMAGE_DESCRIPTIONS[:6]),
            "reference_bands": list(EXPECTED_IMAGE_DESCRIPTIONS[6:]),
            "cloud_mask": "one-band uint8; observed values 0/1; explicit classes override ambiguous nodata metadata",
            "positive_assets": "binary plume mask plus DeltaCH4(ppm) float64 raster",
            "negative_assets": "image and cloud mask only; the adapter must create the zero target in memory",
        },
        "summary": summary,
        "samples": samples,
        "decisions": [
            "Build a new MARS-S2L adapter around the native 12-band target/reference contract.",
            "Include B08; do not truncate the paper model to the legacy ERSRR five-band contract.",
            "Keep negative targets implicit and create zero masks only inside the loader.",
            "Use cloud masks as observability inputs and exclude invalid pixels from losses and metrics.",
            "Do not use rasterio/GDAL validity masks for cloud semantics because nodata can overlap the clear class.",
            "Keep the existing five-band L2A compact model as a separate baseline artifact.",
        ],
        "provenance": {
            "git_commit": git_commit(root),
            "git_tracked_worktree_dirty_at_start": tracked_dirty(root),
            "script": "tools/audit_mars_pilot.py",
            "script_sha256": sha256(Path(__file__)),
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "rasterio": rasterio.__version__,
        },
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_markdown(path: Path, audit: dict[str, Any]) -> None:
    summary = audit["summary"]
    contract = audit["contract"]
    lines = [
        "# MARS-S2L raster-contract pilot audit",
        "",
        f"- Source: `{REPO_ID}`",
        f"- Revision: `{REVISION}`",
        f"- Pilot identity: `{audit['source']['pilot_identity_sha256']}`",
        f"- Assets: {audit['source_integrity']['asset_count']} / {audit['source_integrity']['total_bytes']:,} bytes, all SHA-256 verified",
        f"- Samples: {summary['sample_count']} ({summary['positive_samples']} plume / {summary['negative_samples']} no plume)",
        "",
        "## Verified native product contract",
        "",
        f"- Image: {contract['grid']}; {contract['image_layout']}.",
        f"- Target bands: `{', '.join(contract['target_bands'])}`.",
        f"- Reference bands: `{', '.join(contract['reference_bands'])}`.",
        f"- Cloud mask: {contract['cloud_mask']}.",
        f"- Positive label assets: {contract['positive_assets']}.",
        f"- Negative label assets: {contract['negative_assets']}.",
        "",
        "| Split | Plume | No plume |",
        "|---|---:|---:|",
    ]
    for split in ("train", "val", "test"):
        lines.append(
            f"| {split} | {summary['split_label_counts'].get(split + ':positive', 0)} | "
            f"{summary['split_label_counts'].get(split + ':negative', 0)} |"
        )
    lines.extend(
        [
            "",
            "## Gate result",
            "",
            f"- Contract violations: {summary['contract_violation_count']}.",
            f"- Non-fatal metadata warnings: {summary['warning_count']}.",
            f"- All samples pass: `{str(summary['all_samples_pass_contract']).lower()}`.",
            f"- Paired all-band valid fraction: mean {summary['paired_valid_fraction']['mean']:.6f}, "
            f"range {summary['paired_valid_fraction']['min']:.6f}-{summary['paired_valid_fraction']['max']:.6f}.",
            f"- Positive plume area fraction: mean {summary['positive_plume_area_fraction']['mean']:.6f}, "
            f"range {summary['positive_plume_area_fraction']['min']:.6f}-{summary['positive_plume_area_fraction']['max']:.6f}.",
            "",
            "The raster gate passes if the violation count remains zero. This validates the adapter contract, not model accuracy.",
            "",
            "## Architecture consequences",
            "",
            "1. The paper model needs a native MARS adapter: six target bands plus the corresponding six background bands.",
            "2. B08 is part of the released data and should be retained; the legacy five-band ERSRR model remains a separate baseline.",
            "3. Negative samples intentionally omit plume and enhancement files; the loader must synthesize a zero mask in memory without inventing a raw label asset.",
            "4. Cloud and nodata support must gate both loss and evaluation. Product level remains explicitly Sentinel-2 MSI L1C.",
            "5. Some rasters omit descriptive band tags, and cloud nodata can overlap the clear class; resolve roles from the pinned manifest and interpret mask classes explicitly.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default=DEFAULT_OUTPUT.as_posix())
    parser.add_argument("--output-json", default=DEFAULT_JSON.as_posix())
    parser.add_argument("--output-markdown", default=DEFAULT_MARKDOWN.as_posix())
    parser.add_argument("--dry-run", action="store_true", help="Audit without writing reports")
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args()
    root = repo_root()
    try:
        input_dir = checked_output_dir(root, args.input_dir)
        output_json = safe_output(root, args.output_json)
        output_markdown = safe_output(root, args.output_markdown)
        audit = build_audit(root, input_dir)
        if not args.dry_run:
            write_json(output_json, audit)
            write_markdown(output_markdown, audit)
    except (FileNotFoundError, OSError, RuntimeError, ValueError, rasterio.errors.RasterioError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, indent=None if args.compact else 2))
        return 2
    payload = {
        "ok": audit["summary"]["all_samples_pass_contract"],
        "dry_run": bool(args.dry_run),
        "sample_count": audit["summary"]["sample_count"],
        "asset_count": audit["source_integrity"]["asset_count"],
        "contract_violation_count": audit["summary"]["contract_violation_count"],
        "output_json": None if args.dry_run else output_json.relative_to(root).as_posix(),
        "output_markdown": None if args.dry_run else output_markdown.relative_to(root).as_posix(),
    }
    print(json.dumps(payload, indent=None if args.compact else 2, sort_keys=True))
    return 0 if payload["ok"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
