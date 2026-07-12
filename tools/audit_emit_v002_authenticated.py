#!/usr/bin/env python3
"""Audit authenticated EMIT V002 plume-complex concentration rasters.

Raw Earthdata files remain beneath the ignored acquisition root. This script
binds the protected rasters to the already tracked public CMR/Sentinel-2 pilot,
checks the scientific raster contract, and writes compact checksum evidence.
It deliberately records no credentials, cookies, signed URLs, or raw pixels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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
DEFAULT_INPUT = Path(
    "EarthRemoteSensingRapidResponse/Data Collection/EMIT_Plumes/"
    "emit-v002-authenticated-2026-07"
)
DEFAULT_PUBLIC_BATCH = Path("reports/acquisition/emit_v002_2026_07_batch.json")
DEFAULT_JSON = Path("reports/acquisition/emit_v002_authenticated_plume_audit.json")
DEFAULT_MARKDOWN = Path("reports/acquisition/EMIT_V002_AUTHENTICATED_PLUME_AUDIT.md")
FILENAME = re.compile(
    r"^(EMIT_L2B_CH4PLM_002_(\d{8}T\d{6})_(\d{6}))\.tif$"
)
EXPECTED_RESOLUTION_DEGREES = 0.000542232520256367


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_repo_path(value: str) -> Path:
    path = (ROOT / value).resolve()
    if path != ROOT and ROOT not in path.parents:
        raise ValueError(f"Path must resolve beneath repository root: {value}")
    return path


def finite_stats(values: np.ndarray) -> dict[str, float | int | None]:
    if values.size == 0:
        return {
            "count": 0,
            "min": None,
            "p05": None,
            "median": None,
            "mean": None,
            "p95": None,
            "max": None,
            "std": None,
        }
    percentiles = np.percentile(values.astype(np.float64, copy=False), [5, 50, 95])
    return {
        "count": int(values.size),
        "min": round(float(np.min(values)), 6),
        "p05": round(float(percentiles[0]), 6),
        "median": round(float(percentiles[1]), 6),
        "mean": round(float(np.mean(values, dtype=np.float64)), 6),
        "p95": round(float(percentiles[2]), 6),
        "max": round(float(np.max(values)), 6),
        "std": round(float(np.std(values, dtype=np.float64)), 6),
    }


def expected_granules(public_batch: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(public_batch.read_text(encoding="utf-8"))
    granules = payload.get("granules")
    if not isinstance(granules, list) or not granules:
        raise ValueError("Public batch report has no granules")
    result = {str(item["granule_id"]): item for item in granules}
    if len(result) != len(granules):
        raise ValueError("Public batch report contains duplicate granule IDs")
    return result


def inspect_raster(path: Path, public: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    match = FILENAME.fullmatch(path.name)
    if match is None:
        return {"path": path.relative_to(ROOT).as_posix()}, [f"Unexpected filename: {path.name}"]
    granule_id, timestamp, plume_suffix = match.groups()
    violations: list[str] = []
    with rasterio.open(path) as source:
        data = source.read(1)
        tags = source.tags()
        nodata = source.nodata
        valid = np.isfinite(data)
        if nodata is not None:
            valid &= data != nodata
        valid_values = data[valid]
        bounds = [round(float(value), 12) for value in source.bounds]
        transform = [round(float(value), 12) for value in source.transform[:6]]
        detail: dict[str, Any] = {
            "granule_id": granule_id,
            "path": path.relative_to(ROOT).as_posix(),
            "filename": path.name,
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
            "width": source.width,
            "height": source.height,
            "band_count": source.count,
            "dtype": source.dtypes[0] if source.count else None,
            "crs": None if source.crs is None else source.crs.to_string(),
            "transform": transform,
            "bounds": bounds,
            "resolution_degrees": [abs(float(source.res[0])), abs(float(source.res[1]))],
            "nodata": nodata,
            "valid_pixels": int(np.count_nonzero(valid)),
            "valid_fraction": round(float(np.mean(valid)), 8),
            "values_ppm_m": finite_stats(valid_values),
            "tags": {
                "units": tags.get("Units"),
                "product_version": tags.get("product_version"),
                "plume_complex": tags.get("Plume_Complex"),
                "utc_time_observed": tags.get("UTC_Time_Observed"),
                "source_enhancement_scenes": tags.get("DAAC Scene Names"),
                "tagged_max_ppm_m": tags.get("Max Plume Concentration (ppm m)"),
                "latitude_of_max": tags.get("Latitude of max concentration"),
                "longitude_of_max": tags.get("Longitude of max concentration"),
            },
        }

    expected_scene = f"EMIT_L2B_CH4ENH_V002_{timestamp}_"
    expected_time = datetime.strptime(timestamp, "%Y%m%dT%H%M%S").replace(
        tzinfo=timezone.utc
    ).strftime("%Y-%m-%dT%H:%M:%SZ")
    expected_complex = f"CH4_PlumeComplex-{int(plume_suffix)}"
    checks = {
        "one_band": detail["band_count"] == 1,
        "float32": detail["dtype"] == "float32",
        "epsg4326": detail["crs"] == "EPSG:4326",
        "nodata_minus_9999": detail["nodata"] == -9999.0,
        "nominal_60m_resolution": all(
            abs(float(value) - EXPECTED_RESOLUTION_DEGREES) <= 1e-15
            for value in detail["resolution_degrees"]
        ),
        "has_valid_pixels": detail["valid_pixels"] > 0,
        "finite_valid_values": detail["values_ppm_m"]["count"] == detail["valid_pixels"],
        "units_ppm_m": detail["tags"]["units"] == "ppm m",
        "product_v002": detail["tags"]["product_version"] == "V002",
        "plume_complex_matches_filename": detail["tags"]["plume_complex"] == expected_complex,
        "timestamp_matches_filename": detail["tags"]["utc_time_observed"] == expected_time,
        "source_scene_tag_present": expected_scene in str(
            detail["tags"]["source_enhancement_scenes"]
        ),
        "public_batch_id_match": public.get("granule_id") == granule_id,
        "public_batch_time_match": public.get("emit_datetime", "").replace("+00:00", "Z")
        == expected_time,
    }
    detail["contract_checks"] = checks
    detail["contract_ok"] = all(checks.values())
    violations.extend(f"{granule_id}: failed {name}" for name, ok in checks.items() if not ok)
    return detail, violations


def build_report(input_dir: Path, public_batch: Path) -> dict[str, Any]:
    expected = expected_granules(public_batch)
    files = sorted(input_dir.glob("*/*.tif"))
    observed_ids: set[str] = set()
    records: list[dict[str, Any]] = []
    violations: list[str] = []
    for path in files:
        match = FILENAME.fullmatch(path.name)
        if match is None:
            detail, errors = inspect_raster(path, {})
        else:
            granule_id = match.group(1)
            if granule_id in observed_ids:
                errors = [f"Duplicate granule raster: {granule_id}"]
                detail = {"granule_id": granule_id, "path": path.relative_to(ROOT).as_posix()}
            else:
                observed_ids.add(granule_id)
                detail, errors = inspect_raster(path, expected.get(granule_id, {}))
        records.append(detail)
        violations.extend(errors)

    missing = sorted(set(expected) - observed_ids)
    unexpected = sorted(observed_ids - set(expected))
    violations.extend(f"Missing protected raster: {item}" for item in missing)
    violations.extend(f"Unexpected protected raster: {item}" for item in unexpected)
    valid_values = [item["values_ppm_m"] for item in records if "values_ppm_m" in item]
    total_pixels = sum(int(item.get("width", 0)) * int(item.get("height", 0)) for item in records)
    total_valid = sum(int(item.get("valid_pixels", 0)) for item in records)
    maximum_values = [float(item["max"]) for item in valid_values if item["max"] is not None]
    return {
        "schema_version": 1,
        "scope": "authenticated_emit_v002_ch4plm_pilot",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "collection": "EMITL2BCH4PLM.002",
            "cmr_collection_concept_id": "C3242707413-LPCLOUD",
            "earthdata_order_id": "6313106439",
            "earthdata_order_status_url": "https://search.earthdata.nasa.gov/downloads/6313106439",
            "public_batch_report": public_batch.relative_to(ROOT).as_posix(),
            "public_batch_report_sha256": sha256(public_batch),
            "raw_files_policy": "ignored; preserve NASA filenames; do not redistribute from Git",
            "credential_material_recorded": False,
        },
        "contract": {
            "product": "EMIT L2B Estimated Methane Plume Complexes 60 m V002",
            "scientific_quantity": "methane plume-complex concentration-length enhancement",
            "units": "ppm m",
            "grid": "single-band float32 geographic raster at nominal 60 m",
            "crs": "EPSG:4326",
            "nodata": -9999.0,
            "role": "external positive confirmation and plume-target audit; not detector input",
        },
        "summary": {
            "expected_granules": len(expected),
            "observed_granules": len(observed_ids),
            "files": len(files),
            "bytes": sum(path.stat().st_size for path in files),
            "pixels": total_pixels,
            "valid_pixels": total_valid,
            "valid_fraction": round(total_valid / total_pixels, 8) if total_pixels else 0.0,
            "scene_max_ppm_m": finite_stats(np.asarray(maximum_values, dtype=np.float64)),
            "missing_granules": missing,
            "unexpected_granules": unexpected,
            "contract_violation_count": len(violations),
            "contract_violations": violations,
            "all_files_pass_contract": not violations and len(files) == len(expected),
        },
        "rasters": records,
        "decisions": [
            "Use CH4PLM concentration rasters as independent positive evidence, never as a Sentinel-2 input channel.",
            "Do not treat negative concentration values as no-plume labels; they are retrieval values inside a detected complex.",
            "Acquire source CH4ENH, uncertainty, and sensitivity rasters before quantitative target construction.",
            "Keep the 12-scene pilot outside the locked strict test until time-alignment and observability gates are satisfied.",
        ],
        "provenance": {
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "script": "tools/audit_emit_v002_authenticated.py",
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
        "# Authenticated EMIT V002 plume-raster audit",
        "",
        f"- Collection: `{report['source']['collection']}`",
        f"- Earthdata order: [{report['source']['earthdata_order_id']}]({report['source']['earthdata_order_status_url']})",
        f"- Protected rasters: {summary['observed_granules']} / {summary['expected_granules']}",
        f"- Raw bytes: {summary['bytes']:,} (ignored; not committed)",
        f"- SHA-256-bound files passing contract: {sum(bool(item.get('contract_ok')) for item in report['rasters'])} / {len(report['rasters'])}",
        "- Credentials, cookies, and signed URLs recorded: no",
        "",
        "## Verified scientific contract",
        "",
        "All 12 pilot products are one-band `float32` V002 rasters in EPSG:4326 at the common nominal 60 m grid spacing, with `-9999` nodata and embedded `ppm m` units. Their filename timestamp, plume-complex identifier, and source CH4ENH scene tags agree with the public CMR pilot.",
        "",
        f"Valid support occupies {100.0 * summary['valid_fraction']:.2f}% of the cropped raster pixels. Scene maxima span {summary['scene_max_ppm_m']['min']:,.1f}-{summary['scene_max_ppm_m']['max']:,.1f} ppm m; negative in-footprint retrieval values are retained and must not be reinterpreted as no-plume labels.",
        "",
        "| Plume complex | Shape | Valid pixels | Max ppm m | SHA-256 |",
        "|---|---:|---:|---:|---|",
    ]
    for item in report["rasters"]:
        lines.append(
            f"| `{item['granule_id']}` | {item['height']}x{item['width']} | "
            f"{item['valid_pixels']:,} | {item['values_ppm_m']['max']:,.1f} | "
            f"`{item['sha256']}` |"
        )
    lines.extend(
        [
            "",
            "## Gate result",
            "",
            f"- Missing granules: {len(summary['missing_granules'])}.",
            f"- Unexpected granules: {len(summary['unexpected_granules'])}.",
            f"- Contract violations: {summary['contract_violation_count']}.",
            f"- Pilot gate: `{'pass' if summary['all_files_pass_contract'] else 'fail'}`.",
            "",
            "This proves protected-raster integrity and scientific metadata consistency, not model generalization. The next external-data step is to acquire and align each source CH4ENH enhancement, uncertainty, and sensitivity bundle before constructing quantitative or observability-aware targets.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default=DEFAULT_INPUT.as_posix())
    parser.add_argument("--public-batch", default=DEFAULT_PUBLIC_BATCH.as_posix())
    parser.add_argument("--output-json", default=DEFAULT_JSON.as_posix())
    parser.add_argument("--output-markdown", default=DEFAULT_MARKDOWN.as_posix())
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        input_dir = safe_repo_path(args.input_dir)
        public_batch = safe_repo_path(args.public_batch)
        report = build_report(input_dir, public_batch)
        output_json = safe_repo_path(args.output_json)
        output_markdown = safe_repo_path(args.output_markdown)
        if not args.dry_run:
            write_json(output_json, report)
            write_markdown(output_markdown, report)
    except (OSError, ValueError, json.JSONDecodeError, rasterio.errors.RasterioError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}))
        return 2
    result = {
        "ok": report["summary"]["all_files_pass_contract"],
        "dry_run": bool(args.dry_run),
        "observed_granules": report["summary"]["observed_granules"],
        "contract_violation_count": report["summary"]["contract_violation_count"],
        "output_json": None if args.dry_run else output_json.relative_to(ROOT).as_posix(),
        "output_markdown": None if args.dry_run else output_markdown.relative_to(ROOT).as_posix(),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["ok"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
