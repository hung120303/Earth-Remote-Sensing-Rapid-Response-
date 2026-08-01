#!/usr/bin/env python3
"""Audit fit-fold MARS masks before freezing Gaussian-plume parameter ranges."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "EarthRemoteSensingRapidResponse"
if str(MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_ROOT))

from mars_s2l_adapter import (  # noqa: E402
    iter_development_manifest,
    role_paths,
    safe_asset_path,
    validate_image_band_order,
    validate_positive_mask,
)
from acquire_mars_metadata import sha256  # noqa: E402


DEFAULT_PROTOCOL = Path("configs/mars_gaussian_plume_morphology_audit.json")
QUANTILES = (0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99)


def summary(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"count": 0}
    result: dict[str, float | int] = {
        "count": int(array.size),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
        "mean": float(np.mean(array)),
    }
    for quantile, value in zip(QUANTILES, np.quantile(array, QUANTILES)):
        result[f"q{int(round(quantile * 100)):02d}"] = float(value)
    return result


def mask_geometry(
    mask: np.ndarray,
    resolution_m: float,
    wind_u: float,
    wind_v: float,
) -> dict[str, float | int]:
    rows, cols = np.nonzero(mask)
    if rows.size == 0:
        raise ValueError("Cannot audit an empty positive mask")
    height = int(rows.max() - rows.min() + 1)
    width = int(cols.max() - cols.min() + 1)
    components, component_count = ndimage.label(mask)
    component_sizes = np.bincount(components.ravel())[1:]
    xy = np.column_stack(
        [cols.astype(np.float64) * resolution_m, rows.astype(np.float64) * resolution_m]
    )
    if xy.shape[0] < 2:
        eigenvalues = np.zeros(2, dtype=np.float64)
        major_vector = np.asarray([1.0, 0.0])
    else:
        covariance = np.cov(xy, rowvar=False, bias=True)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = np.maximum(eigenvalues[order], 0.0)
        major_vector = eigenvectors[:, order[0]]
    major_sigma = math.sqrt(float(eigenvalues[0]))
    minor_sigma = math.sqrt(float(eigenvalues[1]))
    image_wind = np.asarray([wind_u, -wind_v], dtype=np.float64)
    wind_speed = float(np.linalg.norm(image_wind))
    alignment = (
        float(abs(np.dot(major_vector, image_wind / wind_speed)))
        if wind_speed >= 0.5
        else float("nan")
    )
    return {
        "area_pixels": int(rows.size),
        "area_m2": float(rows.size * resolution_m * resolution_m),
        "bbox_height_m": float(height * resolution_m),
        "bbox_width_m": float(width * resolution_m),
        "bbox_occupancy": float(rows.size / (height * width)),
        "component_count": int(component_count),
        "largest_component_fraction": float(component_sizes.max() / rows.size),
        "major_4sigma_m": float(4.0 * major_sigma),
        "minor_4sigma_m": float(4.0 * minor_sigma),
        "moment_aspect_ratio": float((major_sigma + resolution_m) / (minor_sigma + resolution_m)),
        "major_axis_wind_alignment": alignment,
    }


def mbmp_spectral_diagnostics(
    swir_pair: np.ndarray, mask: np.ndarray
) -> dict[str, float]:
    """Return unitless plume contrast against the same scene's background."""
    values = np.asarray(swir_pair, dtype=np.float32)
    if values.shape[0] != 4 or values.shape[1:] != mask.shape:
        raise ValueError("Expected target/reference B11/B12 on the mask grid")
    target_b11, target_b12, reference_b11, reference_b12 = values
    valid = np.all(values > 0, axis=0)
    inside = valid & mask
    outside = valid & ~mask
    if np.count_nonzero(inside) < 16 or np.count_nonzero(outside) < 128:
        return {}
    target_ratio = np.ones(mask.shape, dtype=np.float32)
    reference_ratio = np.ones(mask.shape, dtype=np.float32)
    target_ratio[valid] = target_b12[valid] / target_b11[valid]
    reference_ratio[valid] = reference_b12[valid] / reference_b11[valid]
    target_ratio[valid] /= max(float(np.median(target_ratio[valid])), 1e-8)
    reference_ratio[valid] /= max(float(np.median(reference_ratio[valid])), 1e-8)
    mbmp = np.ones(mask.shape, dtype=np.float32)
    mbmp[valid] = target_ratio[valid] / np.maximum(reference_ratio[valid], 1e-8)
    outside_values = mbmp[outside]
    outside_median = float(np.median(outside_values))
    inside_median = float(np.median(mbmp[inside]))
    noise = 1.4826 * float(np.median(np.abs(outside_values - outside_median)))
    contrast = outside_median - inside_median
    return {
        "mbmp_absorption_contrast": contrast,
        "mbmp_absorption_robust_snr": contrast / max(noise, 1e-6),
        "mbmp_background_robust_noise": noise,
    }


def _markdown(report: dict[str, Any]) -> str:
    geometry = report["geometry"]
    lines = [
        "# MARS Gaussian-plume morphology audit",
        "",
        "This is a fit-fold-only parameterization audit. It does not score a candidate model and did not load folds 0, 1, or 2 or the official paper test.",
        "",
        f"- Positive scenes: **{report['cohort']['positive_scenes']:,}**",
        f"- Non-empty masks audited: **{report['cohort']['nonempty_masks']:,}**",
        f"- Empty positive masks excluded from geometry: **{report['cohort']['empty_positive_masks']:,}**",
        f"- Physical sites: **{report['cohort']['sites']:,}**",
        f"- Folds: **{', '.join(map(str, report['cohort']['folds']))}**",
        f"- Sentinel-2 / Landsat: **{report['cohort']['sensor_counts'].get('Sentinel-2', 0):,} / {report['cohort']['sensor_counts'].get('Landsat', 0):,}**",
        "",
        "## Geometry anchors",
        "",
        "| Quantity | q05 | q25 | q50 | q75 | q95 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key in (
        "area_pixels",
        "major_4sigma_m",
        "minor_4sigma_m",
        "moment_aspect_ratio",
        "major_axis_wind_alignment",
    ):
        item = geometry[key]
        lines.append(
            f"| {key} | {item['q05']:.4g} | {item['q25']:.4g} | {item['q50']:.4g} | {item['q75']:.4g} | {item['q95']:.4g} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The audited quantiles may bound a synthetic morphology bank, but they may not be optimized against held-fold outcomes. Enhancement values are reported without physical units because the released MARS assets contain a documented ppm/ppb metadata conflict.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL.as_posix())
    args = parser.parse_args()
    protocol_path = (ROOT / args.protocol).resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    inputs = {
        name: (ROOT / contract["path"]).resolve()
        for name, contract in protocol["inputs"].items()
    }
    for name, path in inputs.items():
        expected = protocol["inputs"][name]["sha256"]
        if expected != "directory_verified_by_acquisition_receipt" and sha256(path) != expected:
            raise ValueError(f"Input hash mismatch: {name}")
    fold_payload = json.loads(inputs["fold_protocol"].read_text(encoding="utf-8"))
    group_to_fold = {
        str(item["group_id"]): int(item["fold"])
        for item in fold_payload["assignments"]
    }
    allowed_folds = set(map(int, protocol["allowed_folds"]))
    records = [
        record
        for record in iter_development_manifest(inputs["manifest"])
        if group_to_fold[str(record["group_id"])] in allowed_folds
        and record["label_state"] == "PLUME"
        and bool(record.get("pixel_truth_available", True))
    ]
    metrics: dict[str, list[float]] = {}
    sensor_counts: Counter[str] = Counter()
    fold_counts: Counter[int] = Counter()
    sites: set[str] = set()
    resolution_counts: Counter[str] = Counter()
    empty_positive_masks = 0
    for index, record in enumerate(records, start=1):
        fold = group_to_fold[str(record["group_id"])]
        paths = role_paths(record)
        image_path = safe_asset_path(inputs["metadata_root"], paths["image"])
        mask_path = safe_asset_path(inputs["metadata_root"], paths["plume_mask"])
        with rasterio.open(image_path) as image_source:
            if image_source.count != 12:
                raise ValueError(f"Expected 12 image bands: {record['sample_id']}")
            validate_image_band_order(record, tuple(image_source.descriptions))
            image_grid = (
                image_source.width,
                image_source.height,
                image_source.crs,
                tuple(image_source.transform)[:6],
            )
            resolution_m = float(abs(image_source.transform.a))
            swir_pair = image_source.read((5, 6, 11, 12))
        with rasterio.open(mask_path) as mask_source:
            mask_grid = (
                mask_source.width,
                mask_source.height,
                mask_source.crs,
                tuple(mask_source.transform)[:6],
            )
            if mask_source.count != 1 or mask_grid != image_grid:
                raise ValueError(f"Mask grid mismatch: {record['sample_id']}")
            plume_mask = validate_positive_mask(
                mask_source.read(1), str(record["sample_id"]), allow_empty=True
            )
        if not np.any(plume_mask):
            empty_positive_masks += 1
            continue
        wind_u = float(record.get("wind_u") or 0.0)
        wind_v = float(record.get("wind_v") or 0.0)
        values = mask_geometry(plume_mask, resolution_m, wind_u, wind_v)
        values.update(mbmp_spectral_diagnostics(swir_pair, plume_mask))
        for name, value in values.items():
            if isinstance(value, (int, float)) and np.isfinite(value):
                metrics.setdefault(name, []).append(float(value))
        sensor_counts[str(record["sensor_family"])] += 1
        fold_counts[fold] += 1
        sites.add(str(record["group_id"]))
        resolution_counts[f"{resolution_m:g}"] += 1
        if index % 256 == 0:
            print(json.dumps({"loaded": index, "total": len(records)}), flush=True)

    report = {
        "schema_version": 1,
        "scope": "fit-fold-only MARS positive-mask morphology audit for Gaussian-plume parameterization",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "cohort": {
            "folds": sorted(allowed_folds),
            "positive_scenes": len(records),
            "nonempty_masks": len(records) - empty_positive_masks,
            "empty_positive_masks": empty_positive_masks,
            "sites": len(sites),
            "fold_counts": {str(key): value for key, value in sorted(fold_counts.items())},
            "sensor_counts": dict(sorted(sensor_counts.items())),
            "resolution_m_counts": dict(sorted(resolution_counts.items())),
        },
        "geometry": {name: summary(values) for name, values in sorted(metrics.items())},
        "provenance": {
            "protocol_sha256": sha256(protocol_path),
            "script_sha256": sha256(Path(__file__).resolve()),
            "manifest_sha256": sha256(inputs["manifest"]),
            "fold_protocol_sha256": sha256(inputs["fold_protocol"]),
        },
        "invariants": [
            "Only positive pixel-truth rows assigned to folds 3 and 4 were loaded.",
            "Scene-positive rows with empty pixel masks were counted and excluded from geometry.",
            "No candidate predictions, scores, losses, or benchmark outcomes were computed.",
            "Folds 0, 1, and 2 and every official-test asset were excluded.",
            "The audit fixes parameter ranges only; it cannot select a model from held outcomes.",
            "Enhancement values have no claimed physical unit because the public metadata conflict is unresolved.",
        ],
    }
    outputs = {
        name: (ROOT / value).resolve() for name, value in protocol["outputs"].items()
    }
    for path in outputs.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    outputs["json"].write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    outputs["markdown"].write_text(_markdown(report), encoding="utf-8")
    print(json.dumps({"ok": True, **report["cohort"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
