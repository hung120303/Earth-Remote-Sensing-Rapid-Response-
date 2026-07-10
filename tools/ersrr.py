#!/usr/bin/env python3
"""Agent-friendly CLI for ERSRR data hygiene and dataset workflows.

This CLI is deliberately small and local-first. It provides reliable commands
that future agents or humans can run without editing hardcoded paths in the
legacy scripts.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_DATASET_DIR = Path("EarthRemoteSensingRapidResponse/Dataset")
DEFAULT_AUDIT_DIR = Path("reports/dataset_audit")
DEFAULT_SPLITS = ["train_test", "validation"]
DEFAULT_BANDS = ["B2", "B3", "B4", "B11", "B12", "EMIT_CH4"]
DATA_COLLECTION_DIR = Path("EarthRemoteSensingRapidResponse/Data Collection")


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    return value


def print_json(payload: dict[str, Any], *, compact: bool = False) -> None:
    kwargs = {"default": _json_default, "sort_keys": True}
    if not compact:
        kwargs["indent"] = 2
    print(json.dumps(payload, **kwargs))


def repo_root(start: Path | None = None) -> Path:
    start = (start or Path.cwd()).resolve()
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=start,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        if out:
            return Path(out).resolve()
    except Exception:
        pass
    return start


def rel_or_abs(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def git_lines(root: Path, args: list[str]) -> list[str]:
    try:
        out = subprocess.check_output(["git", *args], cwd=root, stderr=subprocess.DEVNULL, text=True)
    except Exception:
        return []
    return [line for line in out.splitlines() if line]


def load_audit_module(root: Path):
    path = root / "tools" / "audit_dataset.py"
    if not path.exists():
        raise RuntimeError(f"Missing audit module: {path}")
    spec = importlib.util.spec_from_file_location("ersrr_audit_dataset", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import audit module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def dataset_counts(root: Path, dataset_dir: Path, splits: list[str]) -> dict[str, int]:
    base = dataset_dir if dataset_dir.is_absolute() else root / dataset_dir
    counts: dict[str, int] = {}
    for split in splits:
        split_dir = base / split
        counts[split] = len(list(split_dir.glob("*.tif*"))) if split_dir.exists() else 0
    return counts


def command_status(args: argparse.Namespace) -> int:
    root = repo_root(Path(args.repo_root) if args.repo_root else None)
    dataset_dir = Path(args.dataset_dir)
    counts = dataset_counts(root, dataset_dir, args.splits)
    tracked_node_modules = git_lines(root, ["ls-files", "*node_modules*"])
    status_lines = git_lines(root, ["status", "--short"])
    ee_credentials = Path.home() / ".config" / "earthengine" / "credentials"
    data_collection = root / DATA_COLLECTION_DIR

    payload = {
        "ok": True,
        "repo_root": str(root),
        "branch": (git_lines(root, ["branch", "--show-current"]) or [None])[0],
        "dirty_paths": len(status_lines),
        "dataset_dir": rel_or_abs((root / dataset_dir) if not dataset_dir.is_absolute() else dataset_dir, root),
        "dataset_counts": counts,
        "total_dataset_tifs": sum(counts.values()),
        "data_collection_dir_exists": data_collection.exists(),
        "earthengine_credentials_present": ee_credentials.exists(),
        "tracked_node_modules_files": len(tracked_node_modules),
        "tracked_node_modules_note": "Run git rm -r --cached ERSRR_Website/node_modules after explicit approval if you want node_modules removed from the repository index." if tracked_node_modules else None,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    print_json(payload, compact=args.compact)
    return 0


def command_audit(args: argparse.Namespace) -> int:
    root = repo_root(Path(args.repo_root) if args.repo_root else None)
    module = load_audit_module(root)
    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    dataset_abs = dataset_dir if dataset_dir.is_absolute() else root / dataset_dir
    output_abs = output_dir if output_dir.is_absolute() else root / output_dir

    audit_args = argparse.Namespace(
        dataset_dir=dataset_abs.relative_to(root) if dataset_abs.resolve().is_relative_to(root) else dataset_abs,
        output_dir=output_abs.relative_to(root) if output_abs.resolve().is_relative_to(root) else output_abs,
        splits=args.splits,
        band_names=args.band_names,
        target_band=args.target_band,
        target_nodata=args.target_nodata,
        positive_threshold=args.positive_threshold,
    )

    files = module.discover_files(dataset_abs, args.splits)
    if not files:
        payload = {"ok": False, "error": f"No GeoTIFFs found under {dataset_abs} for splits {args.splits}"}
        print_json(payload, compact=args.compact)
        return 2

    rows = [module.audit_file(path, root, split, audit_args) for split, path in files]
    summary = module.build_summary(rows, audit_args)

    if not args.dry_run:
        output_abs.mkdir(parents=True, exist_ok=True)
        manifest_path = output_abs / "pairings_manifest.csv"
        summary_json_path = output_abs / "summary.json"
        summary_md_path = output_abs / "SUMMARY.md"
        module.write_csv(manifest_path, rows)
        summary_json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        module.write_markdown(summary_md_path, summary, manifest_path.name)
    else:
        manifest_path = summary_json_path = summary_md_path = None

    payload = {
        "ok": True,
        "files_audited": len(rows),
        "dataset_dir": str(audit_args.dataset_dir),
        "output_dir": str(audit_args.output_dir),
        "dry_run": args.dry_run,
        "manifest_path": manifest_path,
        "summary_json_path": summary_json_path,
        "summary_md_path": summary_md_path,
        "warnings": summary.get("warnings", []),
        "splits": {name: info["file_count"] for name, info in summary.get("splits", {}).items()},
    }
    print_json(payload, compact=args.compact)
    return 0


def acquisition_guide() -> str:
    return """# ERSRR Data Acquisition Guide

This project learns a mapping from Sentinel-2 multispectral imagery to EMIT methane plume targets. More model work will not help much until the dataset grows beyond the current ~100 paired tiles.

## Primary data sources

1. EMIT methane plume complexes in Google Earth Engine
   - Dataset ID: `NASA/EMIT/L2B/CH4PLM`
   - Catalog: https://developers.google.com/earth-engine/datasets/catalog/NASA_EMIT_L2B_CH4PLM
   - Use this when you want the fastest path from plume geometries to paired Sentinel-2 exports.

2. EMIT L2B CH4PLM V002 from NASA Earthdata / LP DAAC
   - Catalog: https://www.earthdata.nasa.gov/data/catalog/lpcloud-emitl2bch4plm-002
   - DOI: https://doi.org/10.5067/EMIT/EMITL2BCH4PLM.002
   - Use this when you want original COG + GeoJSON granules locally.
   - Earthdata account required for downloads: https://urs.earthdata.nasa.gov/
   - Earthdata Search: https://search.earthdata.nasa.gov/

3. Sentinel-2 L2A / harmonized imagery
   - Google Earth Engine collection used by the repo: `COPERNICUS/S2_HARMONIZED`
   - Copernicus Browser for manual downloads/checks: https://browser.dataspace.copernicus.eu/
   - Copernicus Sentinel-2 docs: https://dataspace.copernicus.eu/data-collections/copernicus-sentinel-missions/sentinel-2

## What to collect

Prioritize EMIT plume scenes that have:
- non-trivial valid CH4 plume coverage;
- visible plume concentration above background;
- usable Sentinel-2 imagery within a small time window;
- low cloud cover;
- diverse regions/surfaces, not just repeated neighboring tiles.

Also collect negatives:
- Sentinel-2 tiles from similar geographies and seasons with no known methane plume;
- target CH4 band should be zero with an explicit valid mask or reliable nodata metadata.

## Manual download workflow if you need to grab data yourself

1. Create/sign into NASA Earthdata:
   https://urs.earthdata.nasa.gov/

2. Open the EMIT CH4PLM V002 catalog:
   https://www.earthdata.nasa.gov/data/catalog/lpcloud-emitl2bch4plm-002

3. Click the Earthdata Search / data access option for the product.

4. In Earthdata Search:
   - search for `EMITL2BCH4PLM` or use the product page's direct search link;
   - filter dates from 2022-08 onward;
   - select plume granules in regions with likely Sentinel-2 coverage;
   - download both the `.tif` COG and the companion `.json` / GeoJSON metadata when available.

5. Put each batch under:
   `EarthRemoteSensingRapidResponse/Data Collection/EMIT_Plumes/<batch-name>/`

6. Authenticate Google Earth Engine locally:
   `earthengine authenticate`

7. Use the repo's pairing/processing workflow or future CLI collection command to export matching Sentinel-2 tiles, then run:
   `python tools/ersrr.py audit`

## Recommended next targets

- Grow validation from 3 files to at least 50 held-out tiles.
- Keep held-out validation geographically/source separated from training.
- Add 200+ negative/background Sentinel-2 samples before investing in heavier architectures.
- Track every pair in the audit manifest; do not train on unlabeled mystery TIFFs.
"""


def command_guide(args: argparse.Namespace) -> int:
    root = repo_root(Path(args.repo_root) if args.repo_root else None)
    canonical_guide = root / "docs" / "DATA_ACQUISITION.md"
    text = canonical_guide.read_text(encoding="utf-8") if canonical_guide.is_file() else acquisition_guide()
    if args.output:
        out = Path(args.output)
        if not out.is_absolute():
            out = root / out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        payload = {"ok": True, "output": out, "bytes_written": out.stat().st_size}
        print_json(payload, compact=args.compact)
    else:
        print(text)
    return 0


def command_init_batch(args: argparse.Namespace) -> int:
    root = repo_root(Path(args.repo_root) if args.repo_root else None)
    batch = args.name.strip().replace(" ", "_")
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", batch) is None or ".." in batch:
        print_json({"ok": False, "error": "Batch name must be a safe 1-64 character slug"}, compact=args.compact)
        return 2
    base = (root / DATA_COLLECTION_DIR).resolve()
    paths = [
        (base / "EMIT_Plumes" / batch).resolve(),
        (base / "s2_emit_pairs" / batch).resolve(),
        (base / "unpaired_EMIT" / batch).resolve(),
    ]
    if any(base not in path.parents for path in paths):
        print_json({"ok": False, "error": "Batch path escaped the data directory"}, compact=args.compact)
        return 2
    if not args.dry_run:
        for path in paths:
            path.mkdir(parents=True, exist_ok=True)
    payload = {
        "ok": True,
        "dry_run": args.dry_run,
        "batch": batch,
        "paths": [rel_or_abs(path, root) for path in paths],
        "next_step": f"Place downloaded EMIT CH4PLM .tif/.json files in {rel_or_abs(paths[0], root)}",
    }
    print_json(payload, compact=args.compact)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ERSRR local data workflow CLI")
    parser.add_argument("--repo-root", help="Repository root. Defaults to git rev-parse from cwd.")
    parser.add_argument("--compact", action="store_true", help="Emit compact JSON for JSON-producing commands.")
    sub = parser.add_subparsers(dest="command", required=True)

    status = sub.add_parser("status", help="Inspect repo/data prerequisites without modifying files")
    status.add_argument("--compact", action="store_true", help="Emit compact JSON")
    status.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR))
    status.add_argument("--splits", nargs="+", default=DEFAULT_SPLITS)
    status.set_defaults(func=command_status)

    audit = sub.add_parser("audit", help="Audit paired GeoTIFF dataset and write manifest/summary")
    audit.add_argument("--compact", action="store_true", help="Emit compact JSON")
    audit.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR))
    audit.add_argument("--output-dir", default=str(DEFAULT_AUDIT_DIR))
    audit.add_argument("--splits", nargs="+", default=DEFAULT_SPLITS)
    audit.add_argument("--band-names", nargs="+", default=DEFAULT_BANDS)
    audit.add_argument("--target-band", type=int, default=-1)
    audit.add_argument("--target-nodata", type=float, default=-9999.0)
    audit.add_argument("--positive-threshold", type=float, default=0.0)
    audit.add_argument("--dry-run", action="store_true", help="Audit in memory without writing reports")
    audit.set_defaults(func=command_audit)

    guide = sub.add_parser("guide", help="Print or write exact data acquisition instructions")
    guide.add_argument("--compact", action="store_true", help="Emit compact JSON when --output is used")
    guide.add_argument("--output", help="Optional markdown output path, e.g. docs/DATA_ACQUISITION.md")
    guide.set_defaults(func=command_guide)

    init_batch = sub.add_parser("init-batch", help="Create safe local folders for a new EMIT/S2 collection batch")
    init_batch.add_argument("--compact", action="store_true", help="Emit compact JSON")
    init_batch.add_argument("name", help="Batch name, e.g. emit-v002-2026-06")
    init_batch.add_argument("--dry-run", action="store_true")
    init_batch.set_defaults(func=command_init_batch)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except Exception as exc:
        print_json({"ok": False, "error": str(exc)}, compact=getattr(args, "compact", False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
