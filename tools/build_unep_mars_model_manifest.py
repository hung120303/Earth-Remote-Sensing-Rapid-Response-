#!/usr/bin/env python3
"""Build loader-compatible manifests for nonsealed UNEP MARS exact crops."""

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


DEFAULT_COHORT = Path(".research/unep_mars_post2024/eligible_manifest.jsonl")
DEFAULT_CROP_ROOT = Path(".research/unep_mars_post2024/crops")
DEFAULT_AUXILIARY = Path(
    ".research/unep_mars_post2024/model_auxiliary_training.jsonl"
)
DEFAULT_DEVELOPMENT = Path(
    ".research/unep_mars_post2024/model_development.jsonl"
)
DEFAULT_JSON = Path("reports/acquisition/unep_mars_post2024_model_manifest.json")
DEFAULT_MARKDOWN = Path("reports/acquisition/UNEP_MARS_POST2024_MODEL_MANIFEST.md")
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


def verify_asset(root: Path, record: dict[str, Any]) -> Path:
    path = safe_repo_path(root, str(record["path"]))
    if path.stat().st_size != int(record["bytes"]):
        raise ValueError(f"Asset byte count changed: {path}")
    if sha256(path) != str(record["sha256"]):
        raise ValueError(f"Asset SHA-256 changed: {path}")
    return path


def model_record(
    root: Path,
    cohort: dict[str, Any],
    crop: dict[str, Any],
    cloud: dict[str, Any],
) -> dict[str, Any]:
    identity = (cohort["sample_id"], cohort["group_id"], cohort["research_role"])
    if identity != (crop["sample_id"], crop["group_id"], crop["research_role"]):
        raise ValueError(f"Crop identity mismatch for {cohort['sample_id']}")
    if identity != (cloud["sample_id"], cloud["group_id"], cloud["research_role"]):
        raise ValueError(f"Cloud identity mismatch for {cohort['sample_id']}")
    if cohort["research_role"] not in ALLOWED_ROLES:
        raise ValueError(f"Non-development role reached builder: {cohort['research_role']}")
    if crop["sensor_family"] != "Sentinel-2" or not cloud["quality"]["gate_pass"]:
        raise ValueError(f"Ineligible crop reached builder: {cohort['sample_id']}")
    if tuple(crop["product_contract"]["band_order"]) != MARS_BAND_ORDER:
        raise ValueError(f"Band order changed for {cohort['sample_id']}")
    if crop["target_product"] != cohort["target_product"]:
        raise ValueError(f"Target product changed for {cohort['sample_id']}")
    if crop["background_product"] != cohort["background_product"]:
        raise ValueError(f"Background product changed for {cohort['sample_id']}")

    image = crop["assets"]["image"]
    plume = crop["assets"]["plume_mask"]
    cloud_asset = cloud["asset"]
    for asset in (image, plume, cloud_asset):
        verify_asset(root, asset)
    longitude, latitude = cohort["source_center"]
    role = cohort["research_role"]
    target = cohort["target_product"]
    return {
        "assets": [
            {"path": image["path"], "role": "image", "size": image["bytes"]},
            {
                "path": cloud_asset["path"],
                "role": "cloud_mask",
                "size": cloud_asset["bytes"],
            },
            {
                "path": plume["path"],
                "role": "plume_mask",
                "size": plume["bytes"],
            },
        ],
        "band_order": list(MARS_BAND_ORDER),
        "group_id": cohort["group_id"],
        "input_contract": (
            "released MARS-S2L 16 channels; catalog wind prohibited and neutralized"
        ),
        "label_source": "UNEP Eye on Methane MARS polygon",
        "label_state": "PLUME",
        "latitude": latitude,
        "longitude": longitude,
        "observability": "CloudSEN12 gate pass",
        "physical_location_id": cohort["group_id"],
        "pixel_truth_available": True,
        "reference_scene_id": cohort["background_product"],
        "research_role": role,
        "sample_id": cohort["sample_id"],
        "satellite": target[:3],
        "sensor_family": "Sentinel-2",
        "source_dataset": "UNEP Eye on Methane MARS plumes",
        "source_name": cohort["source_name"],
        "split": f"unep_mars_post2024_{role}",
        "target_datetime": cohort["tile_date"],
        "target_scene_id": target,
        "wind_source": "unavailable_zero_for_frozen_protocol",
        "wind_u": 0.0,
        "wind_v": 0.0,
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
        "# UNEP MARS post-2024 model manifest",
        "",
        f"Generated: {report['generated_at_utc']}.",
        "",
        "## Result",
        "",
        f"- Loader-compatible exact Sentinel-2 positives: **{summary['rows']:,}**.",
        f"- Auxiliary training: **{summary['by_role']['auxiliary_training']:,}** rows across **{summary['groups_by_role']['auxiliary_training']:,}** groups.",
        f"- Development confirmation: **{summary['by_role']['development']:,}** rows across **{summary['groups_by_role']['development']:,}** groups.",
        "- Sealed-external crop directories and assets accessed: **0**.",
        "",
        "## Compatibility contract",
        "",
        "- Image, polygon mask, and CloudSEN12 sidecars are byte- and SHA-256-verified before inclusion.",
        "- The released twelve-band ordering and reflectance scaling are unchanged.",
        "- Catalog flux, wind, and polygon geometry are not model features. Wind channels are explicitly zero-filled only to preserve the released 16-channel shape.",
        "- The output manifests remain ignored bulk metadata; this report records their identities.",
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
    args = parser.parse_args()

    root = repo_root()
    cohort_path = (root / args.cohort).resolve()
    crop_root = (root / args.crop_root).resolve()
    records: dict[str, list[dict[str, Any]]] = {role: [] for role in ALLOWED_ROLES}
    eligible_seen = Counter()
    for cohort in iter_jsonl(cohort_path):
        role = str(cohort["research_role"])
        eligible_seen[role] += 1
        if role not in ALLOWED_ROLES:
            continue
        crop_dir = crop_root / str(cohort["sample_id"])
        crop_path = crop_dir / "manifest.json"
        cloud_path = crop_dir / "cloudsen12.manifest.json"
        if not crop_path.is_file() or not cloud_path.is_file():
            continue
        crop = json.loads(crop_path.read_text(encoding="utf-8"))
        cloud = json.loads(cloud_path.read_text(encoding="utf-8"))
        if not cloud["quality"]["gate_pass"]:
            continue
        records[role].append(model_record(root, cohort, crop, cloud))
    for values in records.values():
        values.sort(key=lambda value: value["sample_id"])

    auxiliary = (root / args.auxiliary).resolve()
    development = (root / args.development).resolve()
    write_jsonl(auxiliary, records["auxiliary_training"])
    write_jsonl(development, records["development"])
    all_records = records["auxiliary_training"] + records["development"]
    report = {
        "schema_version": 1,
        "status": "nonsealed loader manifests complete",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "cohort_path": cohort_path.relative_to(root).as_posix(),
            "cohort_sha256": sha256(cohort_path),
            "crop_root": crop_root.relative_to(root).as_posix(),
        },
        "contract": {
            "band_order": list(MARS_BAND_ORDER),
            "label_scope": "positive-only",
            "catalog_flux_wind_geometry_as_features": False,
            "wind_channels": "zero-filled for released tensor compatibility",
            "sealed_external_assets_accessed": False,
        },
        "artifacts": {
            "auxiliary": {
                "path": auxiliary.relative_to(root).as_posix(),
                "sha256": sha256(auxiliary),
                "bytes": auxiliary.stat().st_size,
            },
            "development": {
                "path": development.relative_to(root).as_posix(),
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
            "eligible_catalog_by_role": dict(sorted(eligible_seen.items())),
        },
    }
    output_json = (root / args.output_json).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown((root / args.output_markdown).resolve(), report)
    print(json.dumps({"ok": True, **report["summary"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
