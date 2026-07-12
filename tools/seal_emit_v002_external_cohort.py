#!/usr/bin/env python3
"""Seal the prediction-blind external cohort after exact model-input observability."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import rasterio

DEFAULT_RASTER_GATE = Path("reports/acquisition/emit_v002_l1c_raster_gate.json")
DEFAULT_CLOUD = Path("reports/acquisition/emit_v002_cloudsen12_acquisition.json")
DEFAULT_RAW_ROOT = Path(
    "EarthRemoteSensingRapidResponse/Data Collection/s2_emit_pairs/"
    "emit-v002-external-l1c-2026-07"
)
DEFAULT_JSON = Path("reports/acquisition/emit_v002_external_cohort_seal.json")
DEFAULT_MARKDOWN = Path("reports/acquisition/EMIT_V002_EXTERNAL_COHORT_SEAL.md")


def repo_root() -> Path:
    value = subprocess.check_output(
        ["git", "rev-parse", "--show-toplevel"], text=True
    ).strip()
    return Path(value).resolve()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_path(root: Path, value: str) -> Path:
    result = (root / value).resolve()
    if result != root and root not in result.parents:
        raise ValueError("Path must resolve beneath the repository root")
    return result


def verified_asset(root: Path, record: dict[str, Any]) -> Path:
    path = safe_path(root, record["path"])
    if not path.is_file() or path.stat().st_size != int(record["bytes"]):
        raise ValueError(f"Missing or size-mismatched external asset: {path}")
    if sha256(path) != record["sha256"]:
        raise ValueError(f"External asset hash mismatch: {path}")
    return path


def model_observability(
    target: np.ndarray,
    reference: np.ndarray,
    cloud: np.ndarray,
    plume: np.ndarray,
) -> dict[str, float]:
    if target.shape != reference.shape or target.ndim != 3:
        raise ValueError("Target/reference stacks must be matching band-first arrays")
    shape = target.shape[1:]
    if cloud.shape != shape or plume.shape != shape:
        raise ValueError("Cloud/plume masks must match the spectral grid")
    plume_mask = plume > 0
    plume_pixels = int(np.count_nonzero(plume_mask))
    if plume_pixels == 0:
        raise ValueError("External plume mask is empty")
    radiometric = np.all(target != 0, axis=0) & np.all(reference != 0, axis=0)
    observable = radiometric & (cloud == 0)
    return {
        "model_observable_fraction": round(float(np.mean(observable)), 8),
        "model_observable_fraction_on_plume": round(
            float(np.count_nonzero(observable & plume_mask) / plume_pixels), 8
        ),
    }


def read(path: Path, count: int) -> tuple[np.ndarray, tuple[Any, ...]]:
    with rasterio.open(path) as source:
        if source.count != count:
            raise ValueError(f"Expected {count} bands in {path}")
        return source.read(), (
            source.width,
            source.height,
            source.crs,
            tuple(source.transform)[:6],
        )


def build(
    root: Path,
    raster_gate_path: Path,
    cloud_path: Path,
    raw_root: Path,
    min_observable: float,
) -> dict[str, Any]:
    raster_gate = json.loads(raster_gate_path.read_text(encoding="utf-8"))
    cloud_report = json.loads(cloud_path.read_text(encoding="utf-8"))
    cloud_by_group = {item["group_id"]: item for item in cloud_report["records"]}
    records = []
    exclusions: Counter[str] = Counter()
    initial_pass = [item for item in raster_gate["samples"] if item["gate_pass"]]
    for sample in initial_pass:
        group_id = sample["group_id"]
        cloud_record = cloud_by_group.get(group_id)
        if cloud_record is None:
            raise ValueError(f"Missing CloudSEN12 record for {group_id}")
        scene_dir = raw_root / sample["granule_id"]
        manifest_path = scene_dir / "manifest.json"
        crop = json.loads(manifest_path.read_text(encoding="utf-8"))
        if sha256(manifest_path) != sample["manifest_sha256"]:
            raise ValueError(f"Crop manifest identity changed for {group_id}")
        target_path = verified_asset(root, crop["assets"]["target_l1c"])
        reference_path = verified_asset(root, crop["assets"]["reference_l1c"])
        plume_path = verified_asset(root, crop["assets"]["plume_mask"])
        model_cloud_path = verified_asset(root, cloud_record["asset"])
        target, target_grid = read(target_path, 6)
        reference, reference_grid = read(reference_path, 6)
        plume, plume_grid = read(plume_path, 1)
        model_cloud, cloud_grid = read(model_cloud_path, 1)
        if len({target_grid, reference_grid, plume_grid, cloud_grid}) != 1:
            raise ValueError(f"Final external grids disagree for {group_id}")
        quality = model_observability(target, reference, model_cloud[0], plume[0])
        checks = {
            "model_local_observable": quality["model_observable_fraction"] >= min_observable,
            "model_plume_support_observable": (
                quality["model_observable_fraction_on_plume"] >= min_observable
            ),
        }
        for name, passed in checks.items():
            if not passed:
                exclusions[name] += 1
        records.append(
            {
                "group_id": group_id,
                "granule_id": sample["granule_id"],
                "target_scene_id": sample["target_scene_id"],
                "reference_scene_id": sample["reference_scene_id"],
                "preliminary_scl_gate_pass": True,
                "model_input_gate_checks": checks,
                "final_gate_pass": all(checks.values()),
                **quality,
                "crop_manifest_sha256": sample["manifest_sha256"],
                "cloud_mask_sha256": cloud_record["asset"]["sha256"],
                "plume_pixels": sample["plume_pixels"],
            }
        )
    records.sort(key=lambda item: item["group_id"])
    retained = [item for item in records if item["final_gate_pass"]]
    if len(retained) < 50:
        raise ValueError(f"Final external cohort has only {len(retained)} groups")
    return {
        "contract": {
            "raster_gate_manifest": raster_gate_path.relative_to(root).as_posix(),
            "raster_gate_manifest_sha256": sha256(raster_gate_path),
            "cloud_manifest": cloud_path.relative_to(root).as_posix(),
            "cloud_manifest_sha256": sha256(cloud_path),
            "minimum_model_observable_fraction": min_observable,
            "requirements": [
                "preliminary L2A-SCL/radiometry/containment gate pass",
                "target/reference radiometry valid and CloudSEN12 clear over at least 70% of scene",
                "target/reference radiometry valid and CloudSEN12 clear over at least 70% of EMIT plume",
            ],
            "selection_uses_methane_detector_predictions": False,
            "purpose": "independent positive-confirmation cohort; not a no-plume benchmark",
        },
        "summary": {
            "preliminary_gate_pass": len(initial_pass),
            "final_gate_pass": len(retained),
            "final_gate_fail": len(records) - len(retained),
            "minimum_50_group_goal": "pass",
            "exclusion_counts": dict(sorted(exclusions.items())),
        },
        "records": records,
    }


def markdown(result: dict[str, Any]) -> str:
    summary = result["summary"]
    rows = []
    for item in result["records"]:
        rows.append(
            "| `{group_id}` | {scene:.1%} | {plume:.1%} | {status} |".format(
                group_id=item["group_id"],
                scene=item["model_observable_fraction"],
                plume=item["model_observable_fraction_on_plume"],
                status="pass" if item["final_gate_pass"] else "fail",
            )
        )
    exclusions = "\n".join(
        f"- `{name}`: {count}" for name, count in summary["exclusion_counts"].items()
    )
    return f"""# EMIT V002 external cohort seal

## Result

- Preliminary SCL/radiometry/containment gate: **{summary['preliminary_gate_pass']}**.
- Final exact-model-input gate: **{summary['final_gate_pass']} pass / {summary['final_gate_fail']} fail**.
- Minimum 50-independent-group goal: **{summary['minimum_50_group_goal']}**.
- No methane-detector prediction was computed or consulted.

The final seal requires at least 70% valid target/reference radiometry and CloudSEN12-clear support across both the full detector window and the EMIT plume mask. This resolves a prediction-blind observability disagreement between Sentinel-2 L2A SCL and the exact cloud model used by MARS.

## Exclusions

{exclusions}

## Frozen records

| Group | Model observable | Plume observable | Gate |
|---|---:|---:|---:|
{chr(10).join(rows)}

This is an independent positive-confirmation cohort. It does not estimate no-plume false-positive rate; the sealed MARS-S2L strict cohort provides that benchmark.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raster-gate", default=DEFAULT_RASTER_GATE.as_posix())
    parser.add_argument("--cloud", default=DEFAULT_CLOUD.as_posix())
    parser.add_argument("--raw-root", default=DEFAULT_RAW_ROOT.as_posix())
    parser.add_argument("--minimum-observable", type=float, default=0.70)
    parser.add_argument("--output-json", default=DEFAULT_JSON.as_posix())
    parser.add_argument("--output-markdown", default=DEFAULT_MARKDOWN.as_posix())
    args = parser.parse_args()
    root = repo_root()
    result = build(
        root,
        safe_path(root, args.raster_gate),
        safe_path(root, args.cloud),
        safe_path(root, args.raw_root),
        args.minimum_observable,
    )
    json_path = safe_path(root, args.output_json)
    markdown_path = safe_path(root, args.output_markdown)
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(markdown(result), encoding="utf-8")
    print(json.dumps(result["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
