#!/usr/bin/env python3
"""Build loader-compatible manifests for the CloudSEN12+ clear spatial pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

import numpy as np
import rasterio


DEFAULT_COHORT = Path(".research/cloudsen12_spatial_pilot/selected_manifest.jsonl")
DEFAULT_CROP_ROOT = Path(".research/cloudsen12_spatial_pilot/crops")
DEFAULT_AUXILIARY = Path(
    ".research/cloudsen12_spatial_pilot/model_auxiliary_training.jsonl"
)
DEFAULT_DEVELOPMENT = Path(
    ".research/cloudsen12_spatial_pilot/model_development.jsonl"
)
DEFAULT_JSON = Path("reports/acquisition/cloudsen12_spatial_pilot_model_manifest.json")
DEFAULT_MARKDOWN = Path(
    "reports/acquisition/CLOUDSEN12_SPATIAL_PILOT_MODEL_MANIFEST.md"
)
MARS_BAND_ORDER = (
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
ALLOWED_ROLES = frozenset({"auxiliary_training", "development"})


def repo_root() -> Path:
    value = subprocess.check_output(
        ["git", "rev-parse", "--show-toplevel"], text=True
    ).strip()
    if not value:
        raise RuntimeError("Could not resolve repository root")
    return Path(value).resolve()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc


def safe_repo_path(root: Path, relative_path: str) -> Path:
    relative = PurePosixPath(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe repo-relative path: {relative_path}")
    result = (root / Path(*relative.parts)).resolve()
    if os.path.commonpath([str(root), str(result)]) != str(root):
        raise ValueError(f"Path escapes repository root: {relative_path}")
    return result


def asset_record(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def verify_asset(root: Path, record: dict[str, Any]) -> Path:
    path = safe_repo_path(root, str(record["path"]))
    if path.stat().st_size != int(record["bytes"]):
        raise ValueError(f"Asset byte count changed: {path}")
    if sha256(path) != str(record["sha256"]):
        raise ValueError(f"Asset SHA-256 changed: {path}")
    return path


def build_clear_mask(root: Path, crop: dict[str, Any], *, overwrite: bool) -> dict[str, Any]:
    """Materialize the published all-clear label on the exact crop grid."""
    image_path = verify_asset(root, crop["assets"]["image"])
    plume_path = verify_asset(root, crop["assets"]["plume_mask"])
    with rasterio.open(plume_path) as source:
        if source.count != 1 or np.any(source.read(1) != 0):
            raise ValueError(f"Clear-scene crop has nonzero plume truth: {crop['sample_id']}")
    output = image_path.parent / "cloud_mask.tif"
    with rasterio.open(image_path) as source:
        if source.count != 12 or tuple(source.descriptions) != MARS_BAND_ORDER:
            raise ValueError(f"Image contract changed for {crop['sample_id']}")
        profile = source.profile.copy()
        image_grid = (source.width, source.height, source.crs, tuple(source.transform)[:6])
    if output.is_file() and not overwrite:
        with rasterio.open(output) as source:
            output_grid = (
                source.width,
                source.height,
                source.crs,
                tuple(source.transform)[:6],
            )
            valid = source.count == 1 and output_grid == image_grid and not np.any(source.read(1))
        if not valid:
            raise ValueError(f"Cached clear mask contract changed: {output}")
        return asset_record(output, root)
    profile.update(count=1, dtype="uint8", nodata=4, compress="deflate", predictor=2)
    temporary = output.with_suffix(".tif.tmp")
    with rasterio.open(temporary, "w", **profile) as target:
        target.write(np.zeros((profile["height"], profile["width"]), dtype=np.uint8), 1)
        target.set_band_description(1, "CLOUDSEN12_PUBLISHED_ALL_CLEAR")
    os.replace(temporary, output)
    return asset_record(output, root)


def model_record(
    root: Path,
    cohort: dict[str, Any],
    crop: dict[str, Any],
    cloud_asset: dict[str, Any],
) -> dict[str, Any]:
    if cohort["research_role"] not in ALLOWED_ROLES:
        raise ValueError(f"Unsupported role: {cohort['research_role']}")
    identity = (cohort["sample_id"], cohort["group_id"], cohort["research_role"])
    if identity != (crop["sample_id"], crop["group_id"], crop["research_role"]):
        raise ValueError(f"Crop identity mismatch for {cohort['sample_id']}")
    if cohort.get("label_state") != "NO_PLUME" or crop.get("label_state") != "NO_PLUME":
        raise ValueError(f"Non-negative row reached builder: {cohort['sample_id']}")
    if not crop["quality"]["gate_pass_before_cloud"]:
        raise ValueError(f"Failed crop reached builder: {cohort['sample_id']}")
    if crop["quality"]["plume_pixels"] != 0:
        raise ValueError(f"Clear-scene crop has plume pixels: {cohort['sample_id']}")
    if tuple(crop["product_contract"]["band_order"]) != MARS_BAND_ORDER:
        raise ValueError(f"Band order changed for {cohort['sample_id']}")
    source_grid = cohort["source_grid"]
    if (
        crop["product_contract"]["crs"] != source_grid["crs"]
        or crop["product_contract"]["transform"] != source_grid["transform"]
        or crop["product_contract"]["shape"]
        != [source_grid["height"], source_grid["width"]]
    ):
        raise ValueError(f"Published grid changed for {cohort['sample_id']}")
    if crop["target_product"] != cohort["target_product"]:
        raise ValueError(f"Target product changed for {cohort['sample_id']}")
    if crop["background_product"] != cohort["background_product"]:
        raise ValueError(f"Background product changed for {cohort['sample_id']}")
    image = crop["assets"]["image"]
    verify_asset(root, image)
    verify_asset(root, cloud_asset)
    longitude, latitude = cohort["source_center"]
    target = cohort["target_product"]
    role = cohort["research_role"]
    return {
        "assets": [
            {"path": image["path"], "role": "image", "size": image["bytes"]},
            {
                "path": cloud_asset["path"],
                "role": "cloud_mask",
                "size": cloud_asset["bytes"],
            },
        ],
        "band_order": list(MARS_BAND_ORDER),
        "group_id": cohort["group_id"],
        "input_contract": "released MARS-S2L 16-channel Sentinel-2 contract",
        "label_source": "CloudSEN12+ published 40,000-pixel all-clear label",
        "label_state": "NO_PLUME",
        "latitude": latitude,
        "longitude": longitude,
        "observability": "published all-clear; radiometric-valid fraction at least 0.8",
        "physical_location_id": cohort["group_id"],
        "pixel_truth_available": True,
        "reference_scene_id": cohort["background_product"],
        "research_role": role,
        "sample_id": cohort["sample_id"],
        "satellite": target[:3],
        "sensor_family": "Sentinel-2",
        "source_dataset": "CloudSEN12+ clear-scene false-positive cohort",
        "source_name": cohort["source_name"],
        "split": f"cloudsen12_spatial_{role}",
        "target_datetime": cohort["tile_date"],
        "target_scene_id": target,
        "wind_source": "published MARS-S2L CloudSEN12+ statistics",
        "wind_u": float(cohort["wind_u"]),
        "wind_v": float(cohort["wind_v"]),
    }


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(
            json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    summary = report["summary"]
    lines = [
        "# CloudSEN12+ spatial-pilot model manifest",
        "",
        f"Generated: {report['generated_at_utc']}.",
        "",
        f"- Loader-compatible clear-scene negatives: **{summary['rows']:,}**.",
        f"- Auxiliary training: **{summary['by_role']['auxiliary_training']:,}**.",
        f"- Development confirmation: **{summary['by_role']['development']:,}**.",
        f"- Radiometry or availability exclusions: **{summary['excluded']:,}**.",
        "- Published CloudSEN12+ test rows accessed: **0**.",
        "",
        "Each output row binds the exact 12-band crop and an identically gridded zero cloud mask by hash. The zero mask is valid only because every frozen source row has exactly 40,000 published clear pixels and zero non-clear pixels.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", default=DEFAULT_COHORT.as_posix())
    parser.add_argument("--crop-root", default=DEFAULT_CROP_ROOT.as_posix())
    parser.add_argument("--auxiliary", default=DEFAULT_AUXILIARY.as_posix())
    parser.add_argument("--development", default=DEFAULT_DEVELOPMENT.as_posix())
    parser.add_argument("--output-json", default=DEFAULT_JSON.as_posix())
    parser.add_argument("--output-markdown", default=DEFAULT_MARKDOWN.as_posix())
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    root = repo_root()
    cohort_path = safe_repo_path(root, args.cohort)
    crop_root = safe_repo_path(root, args.crop_root)
    records: dict[str, list[dict[str, Any]]] = {role: [] for role in ALLOWED_ROLES}
    cohort_counts: Counter[str] = Counter()
    for cohort in iter_jsonl(cohort_path):
        role = str(cohort["research_role"])
        if role not in ALLOWED_ROLES:
            raise ValueError(f"Sealed or unsupported role reached builder: {role}")
        cohort_counts[role] += 1
        crop_path = crop_root / str(cohort["sample_id"]) / "manifest.json"
        if not crop_path.is_file():
            continue
        crop = json.loads(crop_path.read_text(encoding="utf-8"))
        if not crop["quality"]["gate_pass_before_cloud"]:
            continue
        cloud_asset = build_clear_mask(root, crop, overwrite=args.overwrite)
        records[role].append(model_record(root, cohort, crop, cloud_asset))
    for values in records.values():
        values.sort(key=lambda value: value["sample_id"])

    auxiliary = safe_repo_path(root, args.auxiliary)
    development = safe_repo_path(root, args.development)
    write_jsonl(auxiliary, records["auxiliary_training"])
    write_jsonl(development, records["development"])
    all_records = records["auxiliary_training"] + records["development"]
    report = {
        "schema_version": 1,
        "status": "nonsealed clear-scene loader manifests complete",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "cohort_path": args.cohort,
            "cohort_sha256": sha256(cohort_path),
            "crop_root": args.crop_root,
        },
        "contract": {
            "band_order": list(MARS_BAND_ORDER),
            "label_scope": "negative-only",
            "cloud_label": "published all-clear represented as an exact-grid zero mask",
            "sealed_external_assets_accessed": False,
        },
        "artifacts": {
            "auxiliary": {
                "path": args.auxiliary,
                "sha256": sha256(auxiliary),
                "bytes": auxiliary.stat().st_size,
            },
            "development": {
                "path": args.development,
                "sha256": sha256(development),
                "bytes": development.stat().st_size,
            },
        },
        "summary": {
            "rows": len(all_records),
            "by_role": {role: len(records[role]) for role in sorted(records)},
            "groups_by_role": {
                role: len({record["group_id"] for record in records[role]})
                for role in sorted(records)
            },
            "cohort_by_role": dict(sorted(cohort_counts.items())),
            "excluded": sum(cohort_counts.values()) - len(all_records),
        },
    }
    output_json = safe_repo_path(root, args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(safe_repo_path(root, args.output_markdown), report)
    print(json.dumps({"ok": True, **report["summary"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
