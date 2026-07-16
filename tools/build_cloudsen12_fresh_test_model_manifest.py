#!/usr/bin/env python3
"""Build the frozen loader manifest for the fresh CloudSEN12+ no-plume test."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio

from build_cloudsen12_spatial_model_manifest import (
    MARS_BAND_ORDER,
    asset_record,
    iter_jsonl,
    model_record,
    repo_root,
    safe_repo_path,
    sha256,
    verify_asset,
    write_jsonl,
)


DEFAULT_PROTOCOL = Path("configs/cloudsen12_fresh_test_model_manifest_protocol.json")
FRESH_ROLE = "fresh_external_test"


def build_zero_cloud_proxy(
    root: Path, crop: dict[str, Any], *, overwrite: bool
) -> dict[str, Any]:
    """Create an explicit zero proxy; never describe it as published cloud truth."""
    image_path = verify_asset(root, crop["assets"]["image"])
    plume_path = verify_asset(root, crop["assets"]["plume_mask"])
    with rasterio.open(plume_path) as source:
        if source.count != 1 or np.any(source.read(1) != 0):
            raise ValueError(f"Fresh no-plume crop has nonzero plume truth: {crop['sample_id']}")
    with rasterio.open(image_path) as source:
        if source.count != 12 or tuple(source.descriptions) != MARS_BAND_ORDER:
            raise ValueError(f"Image contract changed for {crop['sample_id']}")
        profile = source.profile.copy()
        grid = (source.width, source.height, source.crs, tuple(source.transform)[:6])
    output = image_path.parent / "cloud_mask_proxy_zero.tif"
    description = "ZERO_PROXY_PUBLISHED_SPATIAL_CLOUD_MASK_UNAVAILABLE"
    if output.is_file() and not overwrite:
        with rasterio.open(output) as source:
            cached_grid = (
                source.width,
                source.height,
                source.crs,
                tuple(source.transform)[:6],
            )
            valid = (
                source.count == 1
                and cached_grid == grid
                and source.descriptions == (description,)
                and not np.any(source.read(1))
            )
        if not valid:
            raise ValueError(f"Cached zero cloud proxy changed: {output}")
        return asset_record(output, root)
    profile.update(count=1, dtype="uint8", nodata=4, compress="deflate", predictor=2)
    temporary = output.with_suffix(".tif.tmp")
    with rasterio.open(temporary, "w", **profile) as target:
        target.write(np.zeros((profile["height"], profile["width"]), dtype=np.uint8), 1)
        target.set_band_description(1, description)
    os.replace(temporary, output)
    return asset_record(output, root)


def published_cloud_lookup(metadata_path: Path, stats_path: Path) -> dict[str, dict[str, int]]:
    metadata = pd.read_csv(
        metadata_path,
        usecols=["id_loc_image", "location_name"],
        low_memory=False,
    )
    stats = pd.read_csv(
        stats_path,
        usecols=["id_loc_image", "cloudmask_0.0", "cloudmask_1.0"],
        low_memory=False,
    ).rename(columns={"id_loc_image": "location_name"})
    stats[["cloudmask_0.0", "cloudmask_1.0"]] = stats[
        ["cloudmask_0.0", "cloudmask_1.0"]
    ].fillna(0.0)
    joined = metadata.merge(stats, on="location_name", how="inner", validate="one_to_one")
    result: dict[str, dict[str, int]] = {}
    for row in joined.to_dict("records"):
        clear = int(row["cloudmask_0.0"])
        nonclear = int(row["cloudmask_1.0"])
        if clear + nonclear != 40000:
            raise ValueError(f"Published cloud pixel count changed: {row['id_loc_image']}")
        result[str(row["id_loc_image"])] = {"clear": clear, "nonclear": nonclear}
    return result


def fresh_model_record(
    root: Path,
    cohort: dict[str, Any],
    crop: dict[str, Any],
    cloud_asset: dict[str, Any],
    cloud_counts: dict[str, int],
) -> dict[str, Any]:
    record = model_record(
        root,
        cohort,
        crop,
        cloud_asset,
        allowed_roles=frozenset({FRESH_ROLE}),
    )
    record.update(
        {
            "cloud_mask_source": (
                "predeclared exact-grid zero proxy; producer spatial CloudSEN12+ TIFF "
                "is not publicly downloadable"
            ),
            "label_source": "CloudSEN12+ published scene-level NO_PLUME truth",
            "observability": (
                "published no-plume scene; published aggregate cloud composition retained; "
                "radiometric-valid fraction at least 0.8"
            ),
            "published_all_clear": cloud_counts["nonclear"] == 0,
            "published_clear_pixels": cloud_counts["clear"],
            "published_nonclear_pixels": cloud_counts["nonclear"],
            "source_dataset": "CloudSEN12+ fresh published no-plume test cohort",
            "split": "cloudsen12_fresh_external_test",
        }
    )
    return record


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    summary = report["summary"]
    lines = [
        "# CloudSEN12+ fresh-test model manifest",
        "",
        f"- Loader-compatible fresh no-plume rows: **{summary['rows']:,}**.",
        f"- Unavailable exact-product rows retained for bounds: **{summary['unavailable']:,}**.",
        f"- Available rows with published non-clear pixels: **{summary['nonclear_rows']:,}**.",
        f"- Published available-pixel composition: **{summary['clear_pixels']:,}** clear / **{summary['nonclear_pixels']:,}** non-clear.",
        f"- Missing-wind rows explicitly zero-filled: **{summary['wind_zero_imputed_rows']:,}**.",
        "- Spatial cloud input: predeclared all-zero proxy on the exact crop grid; this is not presented as the unavailable producer CloudSEN12+ TIFF.",
        "- Fixed current and candidate heads will receive identical proxy inputs; results are stratified by the published aggregate cloud composition.",
        "- Exact MARS paper cache accessed: **no**.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL.as_posix())
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    root = repo_root()
    protocol_path = safe_repo_path(root, args.protocol)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if sha256(Path(__file__).resolve()) != protocol["builder"]["sha256"]:
        raise ValueError("Fresh-test model-manifest builder hash mismatch")
    paths: dict[str, Path] = {}
    for name, source in protocol["inputs"].items():
        path = safe_repo_path(root, source["path"])
        if sha256(path) != source["sha256"]:
            raise ValueError(f"Frozen input hash mismatch: {name}")
        paths[name] = path
    crop_report = json.loads(paths["crop_report"].read_text(encoding="utf-8"))
    if (
        crop_report["summary"]["acquired"] != protocol["expected"]["available_rows"]
        or crop_report["summary"]["errors"] != 0
        or crop_report["summary"]["gate_pass_before_cloud"]
        != protocol["expected"]["available_rows"]
    ):
        raise ValueError("Frozen crop-acquisition receipt failed")
    cohort = {record["sample_id"]: record for record in iter_jsonl(paths["cohort"])}
    if len(cohort) != protocol["expected"]["full_rows"]:
        raise ValueError("Fresh-test cohort size changed")
    exact_assets = list(iter_jsonl(paths["exact_assets"]))
    resolved_ids = {
        record["sample_id"] for record in exact_assets if record["status"] == "resolved"
    }
    if len(exact_assets) != len(cohort) or len(resolved_ids) != protocol["expected"]["available_rows"]:
        raise ValueError("Frozen exact-product partition changed")
    clouds = published_cloud_lookup(paths["metadata"], paths["stats"])
    crop_root = safe_repo_path(root, protocol["outputs"]["crop_root"])
    records: list[dict[str, Any]] = []
    unavailable: list[str] = []
    for sample_id in sorted(cohort):
        source = cohort[sample_id]
        if source["research_role"] != FRESH_ROLE or source["label_state"] != "NO_PLUME":
            raise ValueError(f"Unexpected fresh-test role/label: {sample_id}")
        crop_path = crop_root / sample_id / "manifest.json"
        if sample_id not in resolved_ids:
            if crop_path.is_file():
                raise ValueError(f"Unresolved identity has a crop: {sample_id}")
            unavailable.append(sample_id)
            continue
        if not crop_path.is_file():
            raise ValueError(f"Resolved identity is missing its acquired crop: {sample_id}")
        crop = json.loads(crop_path.read_text(encoding="utf-8"))
        cloud_asset = build_zero_cloud_proxy(root, crop, overwrite=args.overwrite)
        records.append(fresh_model_record(root, source, crop, cloud_asset, clouds[sample_id]))
    expected = protocol["expected"]
    if len(records) != expected["available_rows"] or len(unavailable) != expected["unavailable_rows"]:
        raise ValueError("Available/unavailable fresh-test partition changed")
    records.sort(key=lambda record: record["sample_id"])
    output = safe_repo_path(root, protocol["outputs"]["manifest"])
    write_jsonl(output, records)
    summary = {
        "rows": len(records),
        "groups": len({record["group_id"] for record in records}),
        "unavailable": len(unavailable),
        "all_clear_rows": sum(record["published_all_clear"] for record in records),
        "nonclear_rows": sum(not record["published_all_clear"] for record in records),
        "clear_pixels": sum(record["published_clear_pixels"] for record in records),
        "nonclear_pixels": sum(record["published_nonclear_pixels"] for record in records),
        "wind_zero_imputed_rows": sum(record["wind_imputed"] for record in records),
        "wind_zero_imputed_components": sum(len(record["wind_imputed_components"]) for record in records),
    }
    for key in ("all_clear_rows", "nonclear_rows", "clear_pixels", "nonclear_pixels", "wind_zero_imputed_rows", "wind_zero_imputed_components"):
        if summary[key] != expected[key]:
            raise ValueError(f"Fresh-test summary contract changed: {key}")
    report = {
        "schema_version": 1,
        "status": "fresh no-plume loader manifest complete",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {"path": args.protocol, "sha256": sha256(protocol_path)},
        "artifact": {"path": protocol["outputs"]["manifest"], "bytes": output.stat().st_size, "sha256": sha256(output)},
        "summary": summary,
        "unavailable_sample_ids": unavailable,
        "cloud_input_contract": protocol["cloud_input_contract"],
        "paper_test_accessed": False,
    }
    output_json = safe_repo_path(root, protocol["outputs"]["report_json"])
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(safe_repo_path(root, protocol["outputs"]["report_markdown"]), report)
    print(json.dumps({"ok": True, **summary}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
