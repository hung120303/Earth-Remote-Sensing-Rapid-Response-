#!/usr/bin/env python3
"""Read-only audit for ERSRR paired Sentinel-2/EMIT GeoTIFF datasets.

The script scans split directories containing paired 6-band GeoTIFFs, writes a
per-file manifest CSV, and writes machine/human-readable summaries. It never
modifies source imagery.

Default assumptions match the current ERSRR dataset:
  - split folders live under EarthRemoteSensingRapidResponse/Dataset
  - paired tiles have 5 Sentinel-2 bands followed by 1 EMIT CH4 target band
  - EMIT nodata pixels are usually encoded as -9999, even when metadata disagrees
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import rasterio

DEFAULT_DATASET_DIR = Path("EarthRemoteSensingRapidResponse/Dataset")
DEFAULT_OUTPUT_DIR = Path("reports/dataset_audit")
DEFAULT_SPLITS = ("train_test", "validation")
DEFAULT_BAND_NAMES = ("B2", "B3", "B4", "B11", "B12", "EMIT_CH4")
S2_PREFIX_RE = re.compile(r"^(?P<s2_start>\d{8}T\d{6})_(?P<s2_end>\d{8}T\d{6})_(?P<s2_tile>T\d{2}[A-Z]{3})")
EMIT_RE = re.compile(r"(?P<emit_id>EMIT_L2B_CH4PLM_\d+_(?P<emit_time>\d{8}T\d{6})_(?P<emit_orbit>\d+))")
GRID_RE = re.compile(r"grid_(?P<grid>\d+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help="Dataset root containing split subdirectories. Default: %(default)s",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for manifest and summary outputs. Default: %(default)s",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(DEFAULT_SPLITS),
        help="Split directories to scan under --dataset-dir. Default: %(default)s",
    )
    parser.add_argument(
        "--band-names",
        nargs="+",
        default=list(DEFAULT_BAND_NAMES),
        help="Names for bands in order. Last band is treated as target by default.",
    )
    parser.add_argument(
        "--target-band",
        type=int,
        default=-1,
        help="1-based target band index, or -1 for last band. Default: %(default)s",
    )
    parser.add_argument(
        "--target-nodata",
        type=float,
        default=-9999.0,
        help="Pixel value to treat as target nodata, in addition to metadata nodata. Default: %(default)s",
    )
    parser.add_argument(
        "--positive-threshold",
        type=float,
        default=0.0,
        help="Target value above which pixels count as positive plume pixels. Default: %(default)s",
    )
    return parser.parse_args()


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def as_jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def percentile(values: np.ndarray, q: float) -> float | None:
    if values.size == 0:
        return None
    return safe_float(np.percentile(values, q))


def finite_values(arr: np.ndarray, nodata_values: Iterable[float | None] = ()) -> np.ndarray:
    mask = np.isfinite(arr)
    for nodata in nodata_values:
        if nodata is None:
            continue
        mask &= arr != nodata
    return arr[mask]


def parse_filename(path: Path) -> dict[str, Any]:
    stem = path.stem
    info: dict[str, Any] = {
        "s2_start": None,
        "s2_end": None,
        "s2_tile": None,
        "emit_id": None,
        "emit_time": None,
        "emit_orbit": None,
        "grid": None,
        "days_between_s2_emit": None,
    }

    s2_match = S2_PREFIX_RE.search(stem)
    if s2_match:
        info.update(s2_match.groupdict())

    emit_match = EMIT_RE.search(stem)
    if emit_match:
        info.update(emit_match.groupdict())

    grid_match = GRID_RE.search(stem)
    if grid_match:
        info["grid"] = int(grid_match.group("grid"))

    if info["s2_start"] and info["emit_time"]:
        try:
            s2_dt = datetime.strptime(info["s2_start"], "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
            emit_dt = datetime.strptime(info["emit_time"], "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
            info["days_between_s2_emit"] = safe_float((emit_dt - s2_dt).total_seconds() / 86400.0)
        except ValueError:
            pass

    return info


def audit_file(path: Path, repo_root: Path, split: str, args: argparse.Namespace) -> dict[str, Any]:
    with rasterio.open(path) as src:
        data = src.read()
        profile_nodata = safe_float(src.nodata)
        target_band_index = src.count if args.target_band == -1 else args.target_band
        if target_band_index < 1 or target_band_index > src.count:
            raise ValueError(f"target band {target_band_index} is outside 1..{src.count}: {path}")

        target = data[target_band_index - 1].astype("float64", copy=False)
        target_valid_mask = np.isfinite(target) & (target != args.target_nodata)
        if profile_nodata is not None:
            target_valid_mask &= target != profile_nodata
        target_valid = target[target_valid_mask]
        target_nodata_pixels = int(target.size - target_valid.size)
        target_positive_pixels = int(np.count_nonzero(target_valid > args.positive_threshold))

        row: dict[str, Any] = {
            "split": split,
            "path": str(path.relative_to(repo_root)),
            "filename": path.name,
            "width": src.width,
            "height": src.height,
            "band_count": src.count,
            "dtype": ",".join(src.dtypes),
            "crs": str(src.crs) if src.crs else None,
            "profile_nodata": profile_nodata,
            "bounds_left": safe_float(src.bounds.left),
            "bounds_bottom": safe_float(src.bounds.bottom),
            "bounds_right": safe_float(src.bounds.right),
            "bounds_top": safe_float(src.bounds.top),
            "target_band": target_band_index,
            "target_valid_pixels": int(target_valid.size),
            "target_total_pixels": int(target.size),
            "target_valid_pct": safe_float(100.0 * target_valid.size / target.size),
            "target_nodata_pixels": target_nodata_pixels,
            "target_nodata_pct": safe_float(100.0 * target_nodata_pixels / target.size),
            "target_positive_pixels": target_positive_pixels,
            "target_positive_pct_of_valid": safe_float(100.0 * target_positive_pixels / target_valid.size) if target_valid.size else None,
            "target_min": safe_float(np.min(target_valid)) if target_valid.size else None,
            "target_mean": safe_float(np.mean(target_valid)) if target_valid.size else None,
            "target_p50": percentile(target_valid, 50),
            "target_p95": percentile(target_valid, 95),
            "target_p99": percentile(target_valid, 99),
            "target_max": safe_float(np.max(target_valid)) if target_valid.size else None,
            "target_has_metadata_mismatch": bool(
                profile_nodata is None or (profile_nodata != args.target_nodata and np.any(target == args.target_nodata))
            ),
        }
        row.update(parse_filename(path))

        # Per-band source summaries. These stay compact enough for CSV while making
        # outlier/nodata issues obvious without reading every raster manually.
        for idx in range(src.count):
            band_name = args.band_names[idx] if idx < len(args.band_names) else f"band_{idx + 1}"
            band = data[idx].astype("float64", copy=False)
            nodata_candidates = [profile_nodata]
            if idx == target_band_index - 1:
                nodata_candidates.append(args.target_nodata)
            values = finite_values(band, nodata_candidates)
            prefix = f"{band_name.lower()}_"
            row[prefix + "min"] = safe_float(np.min(values)) if values.size else None
            row[prefix + "mean"] = safe_float(np.mean(values)) if values.size else None
            row[prefix + "p95"] = percentile(values, 95)
            row[prefix + "max"] = safe_float(np.max(values)) if values.size else None
            row[prefix + "zero_pct"] = safe_float(100.0 * np.count_nonzero(band == 0) / band.size)

        return row


def discover_files(dataset_dir: Path, splits: list[str]) -> list[tuple[str, Path]]:
    files: list[tuple[str, Path]] = []
    for split in splits:
        split_dir = dataset_dir / split
        if not split_dir.exists():
            continue
        for path in sorted(split_dir.glob("*.tif*")):
            files.append((split, path))
    return files


def numeric_summary(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    vals = np.array([row[field] for row in rows if row.get(field) is not None], dtype="float64")
    if vals.size == 0:
        return {"count": 0, "min": None, "mean": None, "p50": None, "p95": None, "max": None}
    return {
        "count": int(vals.size),
        "min": safe_float(np.min(vals)),
        "mean": safe_float(np.mean(vals)),
        "p50": percentile(vals, 50),
        "p95": percentile(vals, 95),
        "max": safe_float(np.max(vals)),
    }


def build_summary(rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_split[row["split"]].append(row)

    summary: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_dir": str(args.dataset_dir),
        "file_count": len(rows),
        "splits": {},
        "overall": {},
        "warnings": [],
    }

    for split, split_rows in sorted(by_split.items()):
        tiles = Counter(row.get("s2_tile") for row in split_rows if row.get("s2_tile"))
        emits = Counter(row.get("emit_id") for row in split_rows if row.get("emit_id"))
        summary["splits"][split] = {
            "file_count": len(split_rows),
            "target_valid_pct": numeric_summary(split_rows, "target_valid_pct"),
            "target_nodata_pct": numeric_summary(split_rows, "target_nodata_pct"),
            "target_positive_pct_of_valid": numeric_summary(split_rows, "target_positive_pct_of_valid"),
            "target_max": numeric_summary(split_rows, "target_max"),
            "days_between_s2_emit": numeric_summary(split_rows, "days_between_s2_emit"),
            "unique_s2_tiles": len(tiles),
            "top_s2_tiles": tiles.most_common(10),
            "unique_emit_ids": len(emits),
            "duplicate_emit_ids": [[key, count] for key, count in emits.items() if count > 1],
            "metadata_mismatch_files": sum(1 for row in split_rows if row.get("target_has_metadata_mismatch")),
        }

    summary["overall"] = {
        "target_valid_pct": numeric_summary(rows, "target_valid_pct"),
        "target_nodata_pct": numeric_summary(rows, "target_nodata_pct"),
        "target_positive_pct_of_valid": numeric_summary(rows, "target_positive_pct_of_valid"),
        "target_max": numeric_summary(rows, "target_max"),
        "days_between_s2_emit": numeric_summary(rows, "days_between_s2_emit"),
    }

    if len(rows) < 500:
        summary["warnings"].append("Dataset is small for supervised remote-sensing CV; prioritize more pairs, negatives, and/or synthetic plume generation before architecture-heavy work.")
    for split, split_info in summary["splits"].items():
        if split_info["file_count"] < 10:
            summary["warnings"].append(f"Split '{split}' has only {split_info['file_count']} files; metrics will be unstable.")
        if split_info["metadata_mismatch_files"]:
            summary["warnings"].append(f"Split '{split}' has {split_info['metadata_mismatch_files']} files where EMIT -9999 nodata appears inconsistent with GeoTIFF nodata metadata.")

    return summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: as_jsonable(row.get(key)) for key in fieldnames})


def write_markdown(path: Path, summary: dict[str, Any], manifest_name: str) -> None:
    lines = [
        "# ERSRR Dataset Audit",
        "",
        f"Generated: `{summary['generated_at_utc']}`",
        f"Dataset: `{summary['dataset_dir']}`",
        f"Manifest: `{manifest_name}`",
        "",
        "## Overall",
        "",
        f"- Files: {summary['file_count']}",
        f"- Target valid % mean: {summary['overall']['target_valid_pct']['mean']:.2f}" if summary['overall']['target_valid_pct']['mean'] is not None else "- Target valid % mean: n/a",
        f"- Target nodata % mean: {summary['overall']['target_nodata_pct']['mean']:.2f}" if summary['overall']['target_nodata_pct']['mean'] is not None else "- Target nodata % mean: n/a",
        f"- Target positive % of valid mean: {summary['overall']['target_positive_pct_of_valid']['mean']:.2f}" if summary['overall']['target_positive_pct_of_valid']['mean'] is not None else "- Target positive % of valid mean: n/a",
        "",
        "## Splits",
        "",
    ]

    for split, info in summary["splits"].items():
        valid = info["target_valid_pct"]
        nodata = info["target_nodata_pct"]
        delta = info["days_between_s2_emit"]
        lines.extend([
            f"### {split}",
            "",
            f"- Files: {info['file_count']}",
            f"- Unique S2 tiles: {info['unique_s2_tiles']}",
            f"- Unique EMIT ids: {info['unique_emit_ids']}",
            f"- Files with target metadata/nodata mismatch: {info['metadata_mismatch_files']}",
            f"- Target valid % min/mean/max: {valid['min']:.2f} / {valid['mean']:.2f} / {valid['max']:.2f}" if valid['mean'] is not None else "- Target valid % min/mean/max: n/a",
            f"- Target nodata % min/mean/max: {nodata['min']:.2f} / {nodata['mean']:.2f} / {nodata['max']:.2f}" if nodata['mean'] is not None else "- Target nodata % min/mean/max: n/a",
            f"- S2→EMIT day delta min/mean/max: {delta['min']:.2f} / {delta['mean']:.2f} / {delta['max']:.2f}" if delta['mean'] is not None else "- S2→EMIT day delta min/mean/max: n/a",
            "",
        ])

    if summary["warnings"]:
        lines.extend(["## Warnings", ""])
        lines.extend(f"- {warning}" for warning in summary["warnings"])
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    repo_root = Path.cwd().resolve()
    dataset_dir = args.dataset_dir if args.dataset_dir.is_absolute() else repo_root / args.dataset_dir
    output_dir = args.output_dir if args.output_dir.is_absolute() else repo_root / args.output_dir
    args.dataset_dir = dataset_dir.relative_to(repo_root) if dataset_dir.is_relative_to(repo_root) else dataset_dir

    files = discover_files(dataset_dir, args.splits)
    if not files:
        raise SystemExit(f"No GeoTIFFs found under {dataset_dir} for splits {args.splits}")

    rows = [audit_file(path, repo_root, split, args) for split, path in files]
    summary = build_summary(rows, args)

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "pairings_manifest.csv"
    summary_json_path = output_dir / "summary.json"
    summary_md_path = output_dir / "SUMMARY.md"

    write_csv(manifest_path, rows)
    summary_json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown(summary_md_path, summary, manifest_path.name)

    print(f"Audited {len(rows)} files")
    print(f"Wrote {manifest_path}")
    print(f"Wrote {summary_json_path}")
    print(f"Wrote {summary_md_path}")


if __name__ == "__main__":
    main()
