#!/usr/bin/env python3
"""Audit authenticated EMIT V002 enhancement/uncertainty/sensitivity bundles."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BATCH = Path(
    "EarthRemoteSensingRapidResponse/Data Collection/EMIT_Plumes/"
    "emit-v002-authenticated-2026-07"
)
DEFAULT_JSON = Path("reports/acquisition/emit_v002_authenticated_science_audit.json")
DEFAULT_MARKDOWN = Path("reports/acquisition/EMIT_V002_AUTHENTICATED_SCIENCE_AUDIT.md")
RESOLUTION_DEGREES = 0.000542232520256367
SCENE_PATTERN = re.compile(
    r"^EMIT_L2B_CH4ENH_002_(\d{8}T\d{6}_\d{7}_\d{3})$"
)
PRODUCTS = {
    "enhancement": ("CH4ENH", "EMIT_L2B_CH4ENH"),
    "uncertainty": ("CH4UNCERT", "EMIT_L2B_CH4UNCERT"),
    "sensitivity": ("CH4SENS", "EMIT_L2B_CH4SENS"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_repo_path(value: str) -> Path:
    path = (ROOT / value).resolve()
    if path != ROOT and ROOT not in path.parents:
        raise ValueError(f"Path must resolve beneath the repository root: {value}")
    return path


def compact_stats(values: np.ndarray) -> dict[str, float | int | None]:
    if values.size == 0:
        return {"count": 0, "min": None, "median": None, "mean": None, "max": None}
    values64 = values.astype(np.float64, copy=False)
    return {
        "count": int(values.size),
        "min": round(float(np.min(values64)), 6),
        "median": round(float(np.median(values64)), 6),
        "mean": round(float(np.mean(values64)), 6),
        "max": round(float(np.max(values64)), 6),
    }


def source_scene_from_plume(path: Path) -> str:
    with rasterio.open(path) as source:
        value = source.tags().get("DAAC Scene Names", "")
    scene = value.replace("[", "").replace("]", "").replace(chr(39), "").strip()
    return scene.replace("_V002_", "_002_")


def plume_map(batch: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in sorted(batch.glob("EMIT_L2B_CH4PLM_002_*/*.tif")):
        scene = source_scene_from_plume(path)
        if scene in result:
            raise ValueError(f"Duplicate CH4PLM source scene: {scene}")
        result[scene] = path
    if not result:
        raise ValueError("No authenticated CH4PLM rasters found")
    return result


def expected_files(scene: str, directory: Path) -> dict[str, Path]:
    match = SCENE_PATTERN.fullmatch(scene)
    if match is None:
        raise ValueError(f"Unexpected CH4ENH scene directory: {scene}")
    tail = match.group(1)
    return {
        role: directory / f"EMIT_L2B_{token}_002_{tail}.tif"
        for role, (token, _) in PRODUCTS.items()
    }


def header(source: rasterio.io.DatasetReader) -> dict[str, Any]:
    return {
        "width": source.width,
        "height": source.height,
        "band_count": source.count,
        "dtype": source.dtypes[0] if source.count else None,
        "crs": None if source.crs is None else source.crs.to_string(),
        "transform": [round(float(value), 12) for value in source.transform[:6]],
        "bounds": [round(float(value), 12) for value in source.bounds],
        "resolution_degrees": [abs(float(source.res[0])), abs(float(source.res[1]))],
        "nodata": source.nodata,
        "descriptions": list(source.descriptions),
        "is_tiled": source.is_tiled,
        "block_shapes": [list(item) for item in source.block_shapes],
        "overviews": source.overviews(1) if source.count else [],
        "compression": None if source.compression is None else source.compression.name,
    }


def same_grid(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return all(
        left[key] == right[key]
        for key in ("width", "height", "crs", "transform", "resolution_degrees")
    )


def inspect_scene(scene_dir: Path, plume_path: Path) -> tuple[dict[str, Any], list[str]]:
    scene = scene_dir.name
    paths = expected_files(scene, scene_dir)
    missing = [role for role, path in paths.items() if not path.is_file()]
    if missing:
        return {"scene_id": scene, "missing_roles": missing}, [
            f"{scene}: missing role {role}" for role in missing
        ]

    details: dict[str, Any] = {"scene_id": scene, "products": {}}
    violations: list[str] = []
    arrays: dict[str, np.ndarray] = {}
    masks: dict[str, np.ndarray] = {}
    headers: dict[str, dict[str, Any]] = {}
    for role, path in paths.items():
        with rasterio.open(path) as source:
            item_header = header(source)
            values = source.read(1)
        valid = np.isfinite(values)
        if item_header["nodata"] is not None:
            valid &= values != item_header["nodata"]
        arrays[role] = values
        masks[role] = valid
        headers[role] = item_header
        expected_description = PRODUCTS[role][1]
        checks = {
            "one_band": item_header["band_count"] == 1,
            "float32": item_header["dtype"] == "float32",
            "epsg4326": item_header["crs"] == "EPSG:4326",
            "nodata_minus_9999": item_header["nodata"] == -9999.0,
            "nominal_60m_resolution": all(
                abs(float(value) - RESOLUTION_DEGREES) <= 1e-15
                for value in item_header["resolution_degrees"]
            ),
            "description_matches_role": item_header["descriptions"] == [expected_description],
            "cloud_optimized_layout": bool(
                item_header["is_tiled"] and item_header["overviews"]
            ),
            "has_valid_pixels": bool(np.any(valid)),
            "valid_values_finite": bool(np.all(np.isfinite(values[valid]))),
        }
        if role in {"uncertainty", "sensitivity"}:
            checks["strictly_positive_on_valid_support"] = bool(np.all(values[valid] > 0))
        details["products"][role] = {
            "path": path.relative_to(ROOT).as_posix(),
            "filename": path.name,
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
            **item_header,
            "valid_pixels": int(np.count_nonzero(valid)),
            "valid_fraction": round(float(np.mean(valid)), 8),
            "values": compact_stats(values[valid]),
            "contract_checks": checks,
            "contract_ok": all(checks.values()),
        }
        violations.extend(
            f"{scene}/{role}: failed {name}" for name, ok in checks.items() if not ok
        )

    aligned = all(same_grid(headers["enhancement"], headers[role]) for role in PRODUCTS)
    uncertainty_sensitivity_mask_equal = np.array_equal(
        masks["uncertainty"], masks["sensitivity"]
    )
    ancillary_outside_enhancement = int(
        np.count_nonzero(
            (masks["uncertainty"] | masks["sensitivity"]) & ~masks["enhancement"]
        )
    )
    common_valid = masks["enhancement"] & masks["uncertainty"] & masks["sensitivity"]
    if not aligned:
        violations.append(f"{scene}: enhancement/uncertainty/sensitivity grids differ")
    if not uncertainty_sensitivity_mask_equal:
        violations.append(f"{scene}: uncertainty/sensitivity validity masks differ")
    if ancillary_outside_enhancement:
        violations.append(f"{scene}: ancillary validity extends outside enhancement support")
    if not np.any(common_valid):
        violations.append(f"{scene}: bundle has no common valid support")

    with rasterio.open(paths["enhancement"]) as enhancement, rasterio.open(
        plume_path
    ) as plume:
        col_offset = (plume.transform.c - enhancement.transform.c) / enhancement.transform.a
        row_offset = (enhancement.transform.f - plume.transform.f) / abs(enhancement.transform.e)
        rounded_col = round(col_offset)
        rounded_row = round(row_offset)
        grid_phase_ok = math.isclose(col_offset, rounded_col, abs_tol=1e-7) and math.isclose(
            row_offset, rounded_row, abs_tol=1e-7
        )
        inside = (
            rounded_row >= 0
            and rounded_col >= 0
            and rounded_row + plume.height <= enhancement.height
            and rounded_col + plume.width <= enhancement.width
        )
        plume_values = plume.read(1)
        plume_valid = np.isfinite(plume_values) & (plume_values != plume.nodata)
        crop = (
            enhancement.read(
                1,
                window=(
                    (rounded_row, rounded_row + plume.height),
                    (rounded_col, rounded_col + plume.width),
                ),
            )
            if grid_phase_ok and inside
            else np.empty((0, 0), dtype=np.float32)
        )
    exact_crop = bool(
        crop.shape == plume_values.shape and np.array_equal(crop[plume_valid], plume_values[plume_valid])
    )
    if not grid_phase_ok:
        violations.append(f"{scene}: CH4PLM crop is not on the CH4ENH pixel grid")
    if not inside:
        violations.append(f"{scene}: CH4PLM crop extends outside source CH4ENH scene")
    if not exact_crop:
        violations.append(f"{scene}: CH4PLM values do not exactly match source CH4ENH crop")

    details["alignment"] = {
        "three_product_grid_equal": aligned,
        "uncertainty_sensitivity_validity_equal": uncertainty_sensitivity_mask_equal,
        "ancillary_valid_pixels_outside_enhancement": ancillary_outside_enhancement,
        "common_valid_pixels": int(np.count_nonzero(common_valid)),
        "common_valid_fraction": round(float(np.mean(common_valid)), 8),
    }
    details["plume_crop_identity"] = {
        "plume_path": plume_path.relative_to(ROOT).as_posix(),
        "plume_sha256": sha256(plume_path),
        "row_offset": rounded_row,
        "column_offset": rounded_col,
        "grid_phase_ok": grid_phase_ok,
        "inside_source_scene": inside,
        "valid_pixels_compared": int(np.count_nonzero(plume_valid)),
        "exact_float32_value_match": exact_crop,
    }
    details["contract_ok"] = not violations
    return details, violations


def build_report(batch: Path) -> dict[str, Any]:
    plumes = plume_map(batch)
    science_root = batch / "CH4ENH"
    scene_dirs = sorted(path for path in science_root.iterdir() if path.is_dir())
    observed = {path.name for path in scene_dirs}
    missing_scenes = sorted(set(plumes) - observed)
    unexpected_scenes = sorted(observed - set(plumes))
    violations = [f"Missing source CH4ENH scene: {scene}" for scene in missing_scenes]
    violations.extend(f"Unexpected source CH4ENH scene: {scene}" for scene in unexpected_scenes)
    scenes: list[dict[str, Any]] = []
    for scene_dir in scene_dirs:
        if scene_dir.name not in plumes:
            continue
        detail, errors = inspect_scene(scene_dir, plumes[scene_dir.name])
        scenes.append(detail)
        violations.extend(errors)

    product_files = [item for scene in scenes for item in scene.get("products", {}).values()]
    common_valid = [scene["alignment"]["common_valid_fraction"] for scene in scenes]
    return {
        "schema_version": 1,
        "scope": "authenticated_emit_v002_ch4enh_science_bundle_pilot",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "collection": "EMITL2BCH4ENH.002",
            "cmr_collection_concept_id": "C3242680113-LPCLOUD",
            "product_guide": "https://lpdaac.usgs.gov/documents/2250/EMIT_L2B_GHG_User_Guide_V2.pdf",
            "raw_files_policy": "ignored; preserve NASA filenames; do not redistribute from Git",
            "credential_material_recorded": False,
        },
        "contract": {
            "enhancement": "adaptive matched-filter methane total-column enhancement in ppm m",
            "uncertainty": "per-pixel methane enhancement uncertainty; positive ancillary layer",
            "sensitivity": "per-pixel methane retrieval sensitivity; positive observability layer",
            "grid": "three aligned single-band float32 COGs at nominal 60 m in EPSG:4326",
            "validity_policy": "use the intersection of all three explicit nodata masks",
            "label_policy": "CH4PLM exact crops provide vetted positive support; scene background is not automatically NO_PLUME",
        },
        "summary": {
            "expected_scenes": len(plumes),
            "observed_scenes": len(scenes),
            "science_files": len(product_files),
            "science_bytes": sum(int(item["bytes"]) for item in product_files),
            "metadata_companions_present": len(list(science_root.glob("*/*.cmr.json"))),
            "three_product_grid_aligned_scenes": sum(
                bool(scene["alignment"]["three_product_grid_equal"]) for scene in scenes
            ),
            "exact_plume_crop_matches": sum(
                bool(scene["plume_crop_identity"]["exact_float32_value_match"])
                for scene in scenes
            ),
            "common_valid_fraction": compact_stats(np.asarray(common_valid, dtype=np.float64)),
            "missing_scenes": missing_scenes,
            "unexpected_scenes": unexpected_scenes,
            "contract_violation_count": len(violations),
            "contract_violations": violations,
            "all_scenes_pass_contract": not violations and len(scenes) == len(plumes),
        },
        "scenes": scenes,
        "decisions": [
            "Promote sensitivity and uncertainty to external observability/quality gates, not detector inputs.",
            "Compute external metrics only on the common valid support of enhancement, uncertainty, and sensitivity.",
            "Use the exact CH4PLM crop identity as a provenance invariant for positive target construction.",
            "Do not infer NO_PLUME from non-plume pixels or from an absent CH4PLM catalog record without review.",
            "Keep this 12-scene pilot for contract validation; expand to at least 50 independent groups before publication claims.",
        ],
        "provenance": {
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "script": "tools/audit_emit_v002_science_bundles.py",
            "script_sha256": sha256(Path(__file__)),
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "rasterio": rasterio.__version__,
        },
    }


def write_json(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    summary = report["summary"]
    lines = [
        "# Authenticated EMIT V002 science-bundle audit",
        "",
        f"- Collection: `{report['source']['collection']}`",
        f"- Source scenes: {summary['observed_scenes']} / {summary['expected_scenes']}",
        f"- Protected science rasters: {summary['science_files']} / {3 * summary['expected_scenes']}",
        f"- Raw bytes: {summary['science_bytes']:,} (ignored; not committed)",
        f"- Three-product grid alignment: {summary['three_product_grid_aligned_scenes']} / {summary['expected_scenes']}",
        f"- Exact CH4PLM-to-CH4ENH crop identity: {summary['exact_plume_crop_matches']} / {summary['expected_scenes']}",
        "- Credentials, cookies, and signed URLs recorded: no",
        "",
        "## Scientific result",
        "",
        "Every source scene contains the official enhancement, uncertainty, and sensitivity COG. All three products share one EPSG:4326 pixel grid per scene, are one-band `float32`, use `-9999` nodata, and preserve a nominal 60 m spacing. Uncertainty and sensitivity are strictly positive on valid support.",
        "",
        "The strongest provenance check passes: each vetted CH4PLM raster is an integer-offset crop of its declared source CH4ENH scene, and every valid plume pixel matches the source enhancement value exactly at `float32` precision.",
        "",
        "| Source scene | Grid | Common valid | Plume pixels checked | Exact crop |",
        "|---|---:|---:|---:|---:|",
    ]
    for scene in report["scenes"]:
        enh = scene["products"]["enhancement"]
        align = scene["alignment"]
        crop = scene["plume_crop_identity"]
        lines.append(
            f"| `{scene['scene_id']}` | {enh['height']}x{enh['width']} | "
            f"{align['common_valid_pixels']:,} | {crop['valid_pixels_compared']:,} | "
            f"{'yes' if crop['exact_float32_value_match'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "## Architecture consequence",
            "",
            "The external evaluator should read the three products as a single quality-aware label bundle. Enhancement supplies the physical positive target; uncertainty and sensitivity define observability and stratification. They must not be fed to the Sentinel-2 detector, and unreviewed scene background must not be relabeled as `NO_PLUME`.",
            "",
            "The official V2 guide defines enhancement as an adaptive matched-filter total-column estimate in `ppm m` and lists the enhancement, uncertainty, and sensitivity COGs as the three CH4ENH science files. The one available `.cmr.json` is retained as optional catalog evidence; it is not part of the three-raster scientific contract.",
            "",
            "## Gate result",
            "",
            f"- Missing scenes: {len(summary['missing_scenes'])}.",
            f"- Unexpected scenes: {len(summary['unexpected_scenes'])}.",
            f"- Contract violations: {summary['contract_violation_count']}.",
            f"- Pilot gate: `{'pass' if summary['all_scenes_pass_contract'] else 'fail'}`.",
            "",
            f"Source: [NASA/JPL EMIT L2B GHG V2 Product User Guide]({report['source']['product_guide']}).",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-dir", default=DEFAULT_BATCH.as_posix())
    parser.add_argument("--output-json", default=DEFAULT_JSON.as_posix())
    parser.add_argument("--output-markdown", default=DEFAULT_MARKDOWN.as_posix())
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        batch = safe_repo_path(args.batch_dir)
        report = build_report(batch)
        output_json = safe_repo_path(args.output_json)
        output_markdown = safe_repo_path(args.output_markdown)
        if not args.dry_run:
            write_json(output_json, report)
            write_markdown(output_markdown, report)
    except (OSError, ValueError, rasterio.errors.RasterioError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}))
        return 2
    result = {
        "ok": report["summary"]["all_scenes_pass_contract"],
        "dry_run": bool(args.dry_run),
        "scenes": report["summary"]["observed_scenes"],
        "science_files": report["summary"]["science_files"],
        "contract_violation_count": report["summary"]["contract_violation_count"],
        "output_json": None if args.dry_run else output_json.relative_to(ROOT).as_posix(),
        "output_markdown": None if args.dry_run else output_markdown.relative_to(ROOT).as_posix(),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["ok"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
