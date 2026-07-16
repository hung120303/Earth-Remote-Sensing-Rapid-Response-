#!/usr/bin/env python3
"""Extract label-free common scene statistics for frozen CloudSEN12 research."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "EarthRemoteSensingRapidResponse"
for path in (MODEL_ROOT, ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from acquire_mars_metadata import repo_root, sha256  # noqa: E402


DEFAULT_PROTOCOL = Path("configs/mars_cloudsen12_negative_augmented_xgboost_protocol.json")
DEFAULT_MARS_STATS = Path(".research/source_audit_20260715/mars_stats_dataset.csv")
DEFAULT_CLOUD_METADATA = Path(".research/source_audit_20260715/cloudsen12_clear_images.csv")
DEFAULT_CLOUD_STATS = Path(".research/source_audit_20260715/cloudsen12_stats_dataset.csv")
DEFAULT_MARS_OUTPUT = Path("outputs/mars_cloudsen12_common_stats_development.npz")
DEFAULT_CLOUD_OUTPUT = Path("outputs/cloudsen12_common_stats_nonsealed.npz")
DEFAULT_JSON = Path("reports/acquisition/mars_cloudsen12_common_stats.json")
DEFAULT_MARKDOWN = Path("reports/acquisition/MARS_CLOUDSEN12_COMMON_STATS.md")

DEVELOPMENT_CACHES = (
    Path("outputs/mars_scene_features_folds234.npz"),
    Path("outputs/mars_scene_features_fold0.npz"),
    Path("outputs/mars_scene_features_fold1_crossfit.npz"),
)
ALLOWED_CLOUD_SPLITS = frozenset({"train", "validation"})
SEALED_CLOUD_SPLIT = "test"


def atomic_savez(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def feature_frame(path: Path, feature_names: list[str]) -> pd.DataFrame:
    """Load only the identifier and preregistered, outcome-free columns."""
    frame = pd.read_csv(
        path,
        usecols=["id_loc_image", *feature_names],
        low_memory=False,
    )
    if frame["id_loc_image"].duplicated().any():
        raise ValueError(f"Duplicate scene identifiers in {path}")
    # A missing cloud-class count means that class was absent from the raster.
    for name in ("cloudmask_0.0", "cloudmask_1.0"):
        frame[name] = frame[name].fillna(0.0)
    values = frame[feature_names].to_numpy(dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError(f"Non-finite preregistered features in {path}")
    frame.loc[:, feature_names] = values
    return frame.set_index("id_loc_image")


def ordered_features(
    frame: pd.DataFrame, ids: np.ndarray, feature_names: list[str]
) -> np.ndarray:
    identifiers = np.asarray(ids).astype(str)
    missing = [identifier for identifier in identifiers if identifier not in frame.index]
    if missing:
        raise ValueError(f"Missing {len(missing)} required scene statistics; first={missing[0]}")
    values = frame.loc[identifiers, feature_names].to_numpy(dtype=np.float32)
    if values.shape != (identifiers.size, len(feature_names)) or not np.isfinite(values).all():
        raise ValueError("Aligned scene-statistic matrix is invalid")
    return values


def development_ids(root: Path) -> np.ndarray:
    parts = []
    for relative in DEVELOPMENT_CACHES:
        path = root / relative
        with np.load(path, allow_pickle=False) as cache:
            parts.append(cache["sample_ids"].astype(str))
    values = np.concatenate(parts)
    if np.unique(values).size != values.size:
        raise ValueError("Development feature caches contain duplicate scene IDs")
    return values


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    summary = report["summary"]
    lines = [
        "# MARS / CloudSEN12+ common-statistic extraction",
        "",
        "The extracted caches contain only preregistered summaries of operational input channels. No paper-test label or CloudSEN12 test feature is present.",
        "",
        f"- MARS development rows: **{summary['mars_development_rows']:,}**.",
        f"- CloudSEN12 training rows: **{summary['cloudsen12_training_rows']:,}**.",
        f"- CloudSEN12 validation rows: **{summary['cloudsen12_validation_rows']:,}**.",
        f"- CloudSEN12 sealed test rows retained only in source metadata: **{summary['cloudsen12_sealed_rows']:,}**.",
        f"- Published clear metadata rows without a statistics row: **{summary['cloudsen12_missing_stats_rows']:,}**.",
        f"- Allowed feature width: **{summary['feature_width']}**.",
        "",
        "CloudSEN12 ROI identities are disjoint across train, validation, and sealed test partitions. Every metadata and statistics row is explicitly no-plume.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL.as_posix())
    parser.add_argument("--mars-stats", default=DEFAULT_MARS_STATS.as_posix())
    parser.add_argument("--cloud-metadata", default=DEFAULT_CLOUD_METADATA.as_posix())
    parser.add_argument("--cloud-stats", default=DEFAULT_CLOUD_STATS.as_posix())
    parser.add_argument("--mars-output", default=DEFAULT_MARS_OUTPUT.as_posix())
    parser.add_argument("--cloud-output", default=DEFAULT_CLOUD_OUTPUT.as_posix())
    parser.add_argument("--output-json", default=DEFAULT_JSON.as_posix())
    parser.add_argument("--output-markdown", default=DEFAULT_MARKDOWN.as_posix())
    args = parser.parse_args()

    root = repo_root()
    paths = {
        "protocol": (root / args.protocol).resolve(),
        "mars_stats": (root / args.mars_stats).resolve(),
        "cloud_metadata": (root / args.cloud_metadata).resolve(),
        "cloud_stats": (root / args.cloud_stats).resolve(),
    }
    protocol = json.loads(paths["protocol"].read_text(encoding="utf-8"))
    expected = {
        "mars_stats": protocol["source_revisions"]["mars_s2l"]["stats_sha256"],
        "cloud_metadata": protocol["source_revisions"]["cloudsen12_clear"]["metadata_sha256"],
        "cloud_stats": protocol["source_revisions"]["cloudsen12_clear"]["stats_sha256"],
    }
    for name, digest in expected.items():
        if sha256(paths[name]) != digest:
            raise ValueError(f"Frozen {name} source hash mismatch")

    feature_names = list(protocol["feature_contract"]["allowed"])
    mars_frame = feature_frame(paths["mars_stats"], feature_names)
    mars_ids = development_ids(root)
    mars_features = ordered_features(mars_frame, mars_ids, feature_names)

    metadata = pd.read_csv(
        paths["cloud_metadata"],
        usecols=["id_loc_image", "location_name", "roi_id", "split_name", "isplume"],
        low_memory=False,
    )
    if metadata["id_loc_image"].duplicated().any():
        raise ValueError("CloudSEN12 metadata contains duplicate scene IDs")
    if metadata["location_name"].duplicated().any():
        raise ValueError("CloudSEN12 metadata contains duplicate statistics keys")
    if metadata["isplume"].astype(bool).any():
        raise ValueError("CloudSEN12 clear metadata contains a positive label")
    observed_splits = set(metadata["split_name"].astype(str))
    if observed_splits != set(ALLOWED_CLOUD_SPLITS) | {SEALED_CLOUD_SPLIT}:
        raise ValueError(f"Unexpected CloudSEN12 partitions: {sorted(observed_splits)}")
    roi_sets = {
        split: set(part["roi_id"].astype(str))
        for split, part in metadata.groupby("split_name")
    }
    if any(
        roi_sets[left] & roi_sets[right]
        for left in roi_sets
        for right in roi_sets
        if left < right
    ):
        raise ValueError("CloudSEN12 ROI identity crosses published partitions")

    cloud_frame = feature_frame(paths["cloud_stats"], feature_names)
    stats_ids = set(cloud_frame.index.astype(str))
    metadata_ids = set(metadata["location_name"].astype(str))
    if not stats_ids.issubset(metadata_ids):
        raise ValueError("CloudSEN12 statistics contain an unknown scene ID")
    selected = metadata[metadata["split_name"].isin(ALLOWED_CLOUD_SPLITS)].copy()
    selected = selected[selected["location_name"].isin(stats_ids)].sort_values("id_loc_image")
    if (selected["split_name"] == SEALED_CLOUD_SPLIT).any():
        raise ValueError("Sealed CloudSEN12 test row reached feature extraction")
    cloud_ids = selected["id_loc_image"].astype(str).to_numpy()
    cloud_stats_ids = selected["location_name"].astype(str).to_numpy()
    cloud_features = ordered_features(cloud_frame, cloud_stats_ids, feature_names)

    mars_output = (root / args.mars_output).resolve()
    cloud_output = (root / args.cloud_output).resolve()
    atomic_savez(
        mars_output,
        sample_ids=mars_ids,
        features=mars_features,
        feature_names=np.asarray(feature_names),
        protocol_sha256=np.asarray(sha256(paths["protocol"])),
    )
    atomic_savez(
        cloud_output,
        sample_ids=cloud_ids,
        source_stats_ids=cloud_stats_ids,
        group_ids=selected["roi_id"].astype(str).to_numpy(),
        splits=selected["split_name"].astype(str).to_numpy(),
        features=cloud_features,
        feature_names=np.asarray(feature_names),
        labels=np.zeros(cloud_ids.size, dtype=np.uint8),
        protocol_sha256=np.asarray(sha256(paths["protocol"])),
    )

    split_counts = metadata["split_name"].value_counts().to_dict()
    selected_counts = selected["split_name"].value_counts().to_dict()
    report = {
        "schema_version": 1,
        "status": "nonsealed common-statistic caches extracted",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "mars_development_rows": int(mars_ids.size),
            "cloudsen12_metadata_rows": int(len(metadata)),
            "cloudsen12_stats_rows": int(len(cloud_frame)),
            "cloudsen12_training_rows": int(selected_counts.get("train", 0)),
            "cloudsen12_validation_rows": int(selected_counts.get("validation", 0)),
            "cloudsen12_sealed_rows": int(split_counts.get("test", 0)),
            "cloudsen12_missing_stats_rows": int(len(metadata_ids - stats_ids)),
            "cloudsen12_all_negative": True,
            "cloudsen12_roi_partitions_disjoint": True,
            "feature_width": len(feature_names),
        },
        "outputs": {
            "mars_development": {
                "path": args.mars_output,
                "bytes": mars_output.stat().st_size,
                "sha256": sha256(mars_output),
            },
            "cloudsen12_nonsealed": {
                "path": args.cloud_output,
                "bytes": cloud_output.stat().st_size,
                "sha256": sha256(cloud_output),
            },
        },
        "source_hashes": expected,
        "protocol_sha256": sha256(paths["protocol"]),
        "script_sha256": sha256(Path(__file__).resolve()),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "sealed_cloudsen12_test_features_extracted": False,
        "paper_test_labels_loaded": False,
    }
    write_json((root / args.output_json).resolve(), report)
    write_markdown((root / args.output_markdown).resolve(), report)
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
