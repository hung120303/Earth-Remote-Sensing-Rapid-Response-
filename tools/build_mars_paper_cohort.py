#!/usr/bin/env python3
"""Build the mixed-sensor MARS-S2L paper development and test cohort.

Bulk manifests and the remote asset catalog are written beneath the ignored
MARS directory.  The tracked report contains only counts, byte totals,
cryptographic identities, and the rules needed to reproduce acquisition.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from acquire_mars_metadata import (
    DEFAULT_OUTPUT,
    REPO_ID,
    REVISION as ASSET_REVISION,
    checked_output_dir,
    repo_root,
    sha256,
    verify_files,
)
from audit_mars_paper_benchmark import (
    ARCHIVE_REVISION,
    DEFAULT_CONFIG,
    DEFAULT_METADATA,
    DEFAULT_OFFSHORE,
    DEFAULT_ONSHORE,
    PUBLISHED,
    PUBLIC_METADATA_REVISION,
    index_unique,
    parse_bool,
    read_csv,
    reconstruct,
)
from build_mars_cohort import asset_owners, resolve_catalog, write_cohort_manifest

MANIFEST_NAME = "paper_v3_mixed_samples.jsonl"
DEVELOPMENT_MANIFEST_NAME = "paper_v3_development_samples.jsonl"
SEALED_TEST_MANIFEST_NAME = "paper_v3_sealed_test_samples.jsonl"
CATALOG_NAME = "paper_v3_mixed_remote_catalog.jsonl"
PAPER_TABLE_S3_COUNTS = {
    "training_rows": 38_366,
    "training_positive": 3_433,
    "validation_rows": 6_034,
    "validation_positive": 288,
}
DEFAULT_JSON = Path("reports/acquisition/mars_s2l_paper_v3_mixed_cohort.json")
DEFAULT_MARKDOWN = Path("reports/acquisition/MARS_S2L_PAPER_V3_MIXED_COHORT.md")
DEFAULT_ASSET_METADATA = DEFAULT_OUTPUT / "validated_images_all.csv"
DEFAULT_TRAIN_SPLIT = DEFAULT_OUTPUT / "train.csv"
DEFAULT_VALIDATION_SPLIT = DEFAULT_OUTPUT / "val.csv"
PATHS_INFO_URL = (
    f"https://huggingface.co/api/datasets/{REPO_ID}/paths-info/{ASSET_REVISION}"
)
def role_counts(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for role in sorted({row["research_role"] for row in rows}):
        selected = [row for row in rows if row["research_role"] == role]
        result[role] = {
            "rows": len(selected),
            "positive": sum(row["label_state"] == "PLUME" for row in selected),
            "negative": sum(row["label_state"] == "NO_PLUME" for row in selected),
            "sites": len({row["physical_location_id"] for row in selected}),
            "sentinel2": sum(row["sensor_family"] == "Sentinel-2" for row in selected),
            "landsat": sum(row["sensor_family"] == "Landsat" for row in selected),
        }
    return result


def site_group(location: str) -> str:
    identity = hashlib.sha256(location.encode("utf-8")).hexdigest()[:16]
    return f"site_{identity}"


def assets_for(
    meta: dict[str, str], positive: bool, *, include_pixel_truth: bool
) -> list[dict[str, str]]:
    assets = [
        {"role": "image", "path": meta["s2path"]},
        {"role": "cloud_mask", "path": meta["cloudmaskpath"]},
    ]
    if positive and include_pixel_truth:
        assets.extend(
            [
                {"role": "plume_mask", "path": meta["plumepath"]},
                {"role": "methane_enhancement", "path": meta["ch4path"]},
            ]
        )
    for asset in assets:
        value = str(asset["path"]).strip()
        pure = PurePosixPath(value)
        if not value or pure.is_absolute() or ".." in pure.parts:
            raise ValueError(f"Missing or unsafe {asset['role']} path for {meta['id_loc_image']}")
    return assets


def common_row(
    meta: dict[str, str],
    *,
    role: str,
    positive: bool,
    test_only: bool,
    label_source: str,
    pixel_truth_available: bool,
) -> dict[str, Any]:
    satellite = str(meta["satellite"])
    sentinel = satellite.startswith("S2")
    location = str(meta["location_name"]).strip()
    native_bands = (
        ["B02", "B03", "B04", "B08", "B11", "B12"]
        if sentinel
        else ["B02", "B03", "B04", "B05", "B06", "B07"]
    )
    return {
        "sample_id": meta["id_loc_image"],
        "source_dataset": REPO_ID,
        "source_revision": PUBLIC_METADATA_REVISION,
        "license": "CC-BY-NC-SA-4.0",
        "research_role": role,
        "split": role,
        "sensor_family": "Sentinel-2" if sentinel else "Landsat",
        "satellite": satellite,
        "product_level": "L1C" if sentinel else "Collection 2 TOA",
        "target_scene_id": meta["tile"],
        "reference_scene_id": meta["background_image_tile"],
        "target_datetime": meta["tile_date"],
        "raw_band_order": native_bands,
        "band_order": native_bands + [f"{band}_bg" for band in native_bands],
        "model_band_order": ["blue", "green", "red", "nir", "swir1", "swir2"],
        "input_contract": "MBMP + current/reference six-band TOA + wind_u/v + cloud mask = 16 channels",
        "crs": meta["crs"],
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "observability": meta["observability"],
        "percentage_clear": float(meta["percentage_clear"]),
        "label_state": "PLUME" if positive else "NO_PLUME",
        "label_source": label_source,
        "pixel_truth_available": pixel_truth_available,
        "physical_location_id": location,
        "group_id": site_group(location),
        "country": meta["country"],
        "longitude": float(meta["lon"]),
        "latitude": float(meta["lat"]),
        "wind_u": float(meta["wind_u"]),
        "wind_v": float(meta["wind_v"]),
        "solar_zenith_angle": float(meta["sza"]),
        "view_zenith_angle": float(meta["vza"]),
        "test_only_site": test_only,
        "assets": assets_for(
            meta, positive, include_pixel_truth=pixel_truth_available
        ),
    }


def build_rows(
    paper_metadata_path: Path,
    asset_metadata_path: Path,
    train_split_path: Path,
    validation_split_path: Path,
    onshore_path: Path,
    offshore_path: Path,
    config_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    paper_rows, paper_audit = reconstruct(
        paper_metadata_path, onshore_path, offshore_path, config_path
    )
    asset_metadata = index_unique(read_csv(asset_metadata_path), "id_loc_image")
    public_splits = {
        "development_training": index_unique(
            read_csv(train_split_path), "id_loc_image"
        ),
        "development_validation": index_unique(
            read_csv(validation_split_path), "id_loc_image"
        ),
    }
    config = json.loads(config_path.read_text(encoding="utf-8"))
    paper_by_id = {str(row["id_loc_image"]): row for row in paper_rows}

    rows: list[dict[str, Any]] = []
    excluded = Counter()
    unavailable_paper_asset_ids = set(paper_audit["missing_public_metadata_ids"])
    current_split_label_disagreements = 0
    public_split_ids: set[str] = set()
    for role, split_rows in public_splits.items():
        overlap = public_split_ids & set(split_rows)
        if overlap:
            raise ValueError(f"Public train/validation splits overlap on {len(overlap)} IDs")
        public_split_ids.update(split_rows)
        for sample_id, split_meta in split_rows.items():
            if sample_id in paper_by_id:
                raise ValueError("Public development split overlaps the sealed paper test")
            meta = asset_metadata.get(sample_id)
            if meta is None:
                raise ValueError(f"Public {role} row lacks current asset metadata: {sample_id}")
            split_positive = parse_bool(split_meta["isplume"])
            asset_positive = parse_bool(meta["isplume"])
            current_split_label_disagreements += split_positive != asset_positive
            rows.append(
                common_row(
                    meta,
                    role=role,
                    positive=split_positive,
                    test_only=False,
                    label_source=f"pinned public {role} split CSV",
                    pixel_truth_available=split_positive,
                )
            )

    for sample_id, paper in paper_by_id.items():
        meta = asset_metadata.get(sample_id)
        if meta is None:
            excluded["paper_test_no_current_asset_metadata"] += 1
            unavailable_paper_asset_ids.add(sample_id)
            continue
        positive = bool(int(paper["target"]))
        pixel_truth_available = bool(
            positive
            and parse_bool(meta["isplume"])
            and str(meta["plumepath"]).strip()
            and str(meta["ch4path"]).strip()
        )
        row = common_row(
            meta,
            role="sealed_paper_test",
            positive=positive,
            test_only=bool(paper["test_only_site"]),
            label_source="paper-era archived evaluation target",
            pixel_truth_available=pixel_truth_available,
        )
        row["paper_baseline"] = {
            "source": paper["baseline_source"],
            "scene_score": float(paper["scene_pred"]),
            "pixel_tp": int(float(paper["TP"])),
            "pixel_fp": int(float(paper["FP"])),
            "pixel_tn": int(float(paper["TN"])),
            "pixel_fn": int(float(paper["FN"])),
        }
        rows.append(row)

    rows.sort(key=lambda row: (row["research_role"], row["sample_id"]))
    if len({row["sample_id"] for row in rows}) != len(rows):
        raise ValueError("Mixed paper cohort contains duplicate sample IDs")
    test_rows = [row for row in rows if row["research_role"] == "sealed_paper_test"]
    expected_public_test_rows = PUBLISHED["full"]["rows"] - len(
        unavailable_paper_asset_ids
    )
    if len(test_rows) != expected_public_test_rows:
        raise ValueError("Public paper-test coverage differs from the benchmark lock")
    if sum(row["test_only_site"] for row in test_rows) != PUBLISHED["test_only_sites"]["rows"]:
        raise ValueError("Public test-only view no longer matches the paper")
    train_sites = {
        row["physical_location_id"]
        for row in rows
        if row["research_role"] == "development_training"
    }
    val_sites = {
        row["physical_location_id"]
        for row in rows
        if row["research_role"] == "development_validation"
    }
    return rows, {
        "excluded": dict(sorted(excluded.items())),
        "role_counts": role_counts(rows),
        "development_train_validation_site_overlap": len(train_sites & val_sites),
        "public_development_split_label_disagreements_vs_asset_metadata": current_split_label_disagreements,
        "public_development_split_sha256": {
            "train.csv": sha256(train_split_path),
            "val.csv": sha256(validation_split_path),
        },
        "released_checkpoint_internal_counts": {
            "training_rows": int(config["n_samples_train"]),
            "training_positive": int(config["n_pos_train"]),
            "validation_rows": int(config["n_samples_val"]),
            "validation_positive": int(config["n_pos_val"]),
            "source": str(config["csv_path"]),
        },
        "paper_table_s3_counts": PAPER_TABLE_S3_COUNTS,
        "paper_test_public_rows": len(test_rows),
        "paper_test_unavailable_rows": len(unavailable_paper_asset_ids),
        "paper_test_unavailable_ids": sorted(unavailable_paper_asset_ids),
        "paper_test_label_disagreements_vs_public_metadata": paper_audit[
            "paper_target_vs_public_metadata_label_disagreements"
        ],
        "paper_test_positive_rows_without_pixel_truth": sum(
            row["research_role"] == "sealed_paper_test"
            and row["label_state"] == "PLUME"
            and not row["pixel_truth_available"]
            for row in rows
        ),
        "current_asset_metadata_rows_used": len(rows),
        "historical_asset_metadata_fallback_rows": 0,
        "historical_asset_metadata_fallback_ids": [],
    }


def write_manifest(path: Path, rows: list[dict[str, Any]], catalog: dict[str, dict[str, Any]]) -> None:
    write_cohort_manifest(path, rows, catalog)


def build_report(
    root: Path,
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    catalog: dict[str, dict[str, Any]],
    manifest_path: Path,
    development_manifest_path: Path,
    sealed_test_manifest_path: Path,
    catalog_path: Path,
) -> dict[str, Any]:
    owners = asset_owners(rows)
    bytes_by_role: Counter[str] = Counter()
    assets_by_role: Counter[str] = Counter()
    bytes_by_research_role: Counter[str] = Counter()
    for path, owner in owners.items():
        size = int(catalog[path]["size"])
        for role in owner["roles"]:
            bytes_by_role[role] += size
            assets_by_role[role] += 1
        for role in owner["splits"]:
            bytes_by_research_role[role] += size
    return {
        "schema_version": 1,
        "status": "frozen_before_mixed_sensor_training_and_test_access",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "repository": REPO_ID,
            "paper_identity_metadata_revision": PUBLIC_METADATA_REVISION,
            "asset_metadata_revision": ASSET_REVISION,
            "prediction_archive_revision": ARCHIVE_REVISION,
            "license": "CC-BY-NC-SA-4.0",
        },
        "selection": {
            "development_training": "complete pinned public train.csv; candidate fitting only",
            "development_validation": "complete pinned public val.csv; candidate development only",
            "sealed_paper_test": "paper archive IDs and targets; never used for selection",
            "sensors": ["Sentinel-2 MSI L1C", "Landsat Collection 2 TOA"],
            "input_channels": 16,
        },
        "cohort": summary,
        "remote_assets": {
            "unique_asset_count": len(catalog),
            "total_bytes": sum(int(item["size"]) for item in catalog.values()),
            "binary_gib": round(sum(int(item["size"]) for item in catalog.values()) / 1024**3, 3),
            "asset_counts_by_role": dict(sorted(assets_by_role.items())),
            "bytes_by_asset_role": dict(sorted(bytes_by_role.items())),
            "bytes_by_research_role_nonexclusive": dict(sorted(bytes_by_research_role.items())),
        },
        "local_ignored_artifacts": {
            "acquisition_manifest": manifest_path.relative_to(root).as_posix(),
            "acquisition_manifest_sha256": sha256(manifest_path),
            "acquisition_manifest_bytes": manifest_path.stat().st_size,
            "development_manifest": development_manifest_path.relative_to(root).as_posix(),
            "development_manifest_sha256": sha256(development_manifest_path),
            "development_manifest_bytes": development_manifest_path.stat().st_size,
            "sealed_test_manifest": sealed_test_manifest_path.relative_to(root).as_posix(),
            "sealed_test_manifest_sha256": sha256(sealed_test_manifest_path),
            "sealed_test_manifest_bytes": sealed_test_manifest_path.stat().st_size,
            "remote_catalog": catalog_path.relative_to(root).as_posix(),
            "remote_catalog_sha256": sha256(catalog_path),
            "remote_catalog_bytes": catalog_path.stat().st_size,
        },
        "firewall": {
            "development_may_read": ["development_training", "development_validation"],
            "sealed_until_freeze": ["sealed_paper_test"],
            "missing_test_policy": "five unavailable historical scene rasters and all missing pixel truth receive adversarial candidate outcomes",
            "test_tuning": "prohibited",
        },
        "paper_training_snapshot_limit": {
            "released_checkpoint_config": summary["released_checkpoint_internal_counts"],
            "paper_table_s3": summary["paper_table_s3_counts"],
            "public_release_counts": summary["role_counts"],
            "interpretation": "the checkpoint names an unavailable internal Azure March-2025 CSV, its validation counts differ from paper Table S3, and the public train/val CSVs differ from both; public splits are used reproducibly for successor development, while the paper test assignment is reconstructed exactly",
        },
        "provenance": {
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
            "script": "tools/build_mars_paper_cohort.py",
            "script_sha256": sha256(Path(__file__)),
            "python": sys.version.split()[0],
        },
    }


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# MARS-S2L paper-v3 mixed-sensor cohort",
        "",
        f"- Remote assets: {report['remote_assets']['unique_asset_count']:,} / {report['remote_assets']['binary_gib']:.3f} GiB",
        f"- Acquisition manifest SHA-256: `{report['local_ignored_artifacts']['acquisition_manifest_sha256']}`",
        f"- Development-only manifest SHA-256: `{report['local_ignored_artifacts']['development_manifest_sha256']}`",
        f"- Sealed-test manifest SHA-256: `{report['local_ignored_artifacts']['sealed_test_manifest_sha256']}`",
        f"- Catalog SHA-256: `{report['local_ignored_artifacts']['remote_catalog_sha256']}`",
        "",
        "| Research role | Rows | Plume | No plume | Sites | Sentinel-2 | Landsat |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for role, item in report["cohort"]["role_counts"].items():
        lines.append(
            f"| {role} | {item['rows']:,} | {item['positive']:,} | {item['negative']:,} | {item['sites']:,} | {item['sentinel2']:,} | {item['landsat']:,} |"
        )
    lines.extend(
        [
            "",
            f"The exact paper archive covers 43,529 test scenes. Current released raster paths remain available for {report['cohort']['paper_test_public_rows']:,}; the {report['cohort']['paper_test_unavailable_rows']} unavailable historical scenes are preserved in the benchmark lock and scored adversarially for the candidate. Paper-era targets override later public labels on sealed test scenes.",
            "",
            "Development code reads only the physically separate development manifest. The sealed paper-test manifest is downloaded for eventual one-shot inference but cannot be opened for architecture, checkpoint, calibration, threshold, or postprocessing selection.",
            "",
            f"The released checkpoint config names an internal `{report['cohort']['released_checkpoint_internal_counts']['source']}` snapshot ({report['cohort']['released_checkpoint_internal_counts']['training_rows']:,} training / {report['cohort']['released_checkpoint_internal_counts']['validation_rows']:,} validation rows), while paper Table S3 reports {report['cohort']['paper_table_s3_counts']['training_rows']:,} / {report['cohort']['paper_table_s3_counts']['validation_rows']:,}. The private snapshot is not in the public release, whose pinned files contain {report['cohort']['role_counts']['development_training']['rows']:,} / {report['cohort']['role_counts']['development_validation']['rows']:,}. Successor development uses those complete public splits, and the exact archived paper test remains the comparison target.",
            "",
            "Acquisition: `python tools/acquire_mars_cohort.py --catalog-file paper_v3_mixed_remote_catalog.jsonl --workers 8 --receipt reports/acquisition/mars_s2l_paper_v3_mixed_download.json`.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-dir", default=DEFAULT_OUTPUT.as_posix())
    parser.add_argument("--source-metadata", default=DEFAULT_METADATA.as_posix())
    parser.add_argument("--asset-metadata", default=DEFAULT_ASSET_METADATA.as_posix())
    parser.add_argument("--train-split", default=DEFAULT_TRAIN_SPLIT.as_posix())
    parser.add_argument("--validation-split", default=DEFAULT_VALIDATION_SPLIT.as_posix())
    parser.add_argument("--onshore-predictions", default=DEFAULT_ONSHORE.as_posix())
    parser.add_argument("--offshore-predictions", default=DEFAULT_OFFSHORE.as_posix())
    parser.add_argument("--config", default=DEFAULT_CONFIG.as_posix())
    parser.add_argument("--output-json", default=DEFAULT_JSON.as_posix())
    parser.add_argument("--output-markdown", default=DEFAULT_MARKDOWN.as_posix())
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--refresh-catalog", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    root = repo_root()
    metadata_dir = checked_output_dir(root, args.metadata_dir)
    verify_files(metadata_dir)
    source_paths = [
        (root / value).resolve()
        for value in (
            args.source_metadata,
            args.asset_metadata,
            args.train_split,
            args.validation_split,
            args.onshore_predictions,
            args.offshore_predictions,
            args.config,
        )
    ]
    rows, summary = build_rows(*source_paths)
    manifest_path = metadata_dir / MANIFEST_NAME
    development_manifest_path = metadata_dir / DEVELOPMENT_MANIFEST_NAME
    sealed_test_manifest_path = metadata_dir / SEALED_TEST_MANIFEST_NAME
    catalog_path = metadata_dir / CATALOG_NAME
    owners = asset_owners(rows)
    catalog = resolve_catalog(
        sorted(owners),
        catalog_path,
        offline=args.offline or args.verify_only,
        refresh=args.refresh_catalog,
        paths_info_url=PATHS_INFO_URL,
        revision=ASSET_REVISION,
    )
    if args.verify_only:
        partitions = {
            manifest_path: rows,
            development_manifest_path: [
                row for row in rows if row["research_role"] != "sealed_paper_test"
            ],
            sealed_test_manifest_path: [
                row for row in rows if row["research_role"] == "sealed_paper_test"
            ],
        }
        for frozen, partition_rows in partitions.items():
            if not frozen.is_file():
                raise FileNotFoundError(f"Missing frozen manifest: {frozen}")
            expected = frozen.with_name(frozen.name + ".verify.tmp")
            try:
                write_manifest(expected, partition_rows, catalog)
                if sha256(expected) != sha256(frozen):
                    raise ValueError(f"Frozen manifest identity changed: {frozen.name}")
            finally:
                expected.unlink(missing_ok=True)
        print(json.dumps({"ok": True, "rows": len(rows), "acquisition_manifest_sha256": sha256(manifest_path), "development_manifest_sha256": sha256(development_manifest_path), "sealed_test_manifest_sha256": sha256(sealed_test_manifest_path), "catalog_sha256": sha256(catalog_path)}, sort_keys=True))
        return 0

    write_manifest(manifest_path, rows, catalog)
    write_manifest(
        development_manifest_path,
        [row for row in rows if row["research_role"] != "sealed_paper_test"],
        catalog,
    )
    write_manifest(
        sealed_test_manifest_path,
        [row for row in rows if row["research_role"] == "sealed_paper_test"],
        catalog,
    )
    report = build_report(
        root,
        rows,
        summary,
        catalog,
        manifest_path,
        development_manifest_path,
        sealed_test_manifest_path,
        catalog_path,
    )
    output_json = (root / args.output_json).resolve()
    output_markdown = (root / args.output_markdown).resolve()
    write_json(output_json, report)
    write_markdown(output_markdown, report)
    print(json.dumps({"ok": True, "rows": len(rows), "role_counts": summary["role_counts"], "remote_assets": report["remote_assets"], "acquisition_manifest_sha256": report["local_ignored_artifacts"]["acquisition_manifest_sha256"], "development_manifest_sha256": report["local_ignored_artifacts"]["development_manifest_sha256"], "sealed_test_manifest_sha256": report["local_ignored_artifacts"]["sealed_test_manifest_sha256"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
