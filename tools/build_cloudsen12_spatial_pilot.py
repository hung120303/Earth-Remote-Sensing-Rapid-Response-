#!/usr/bin/env python3
"""Select the frozen spatially disjoint CloudSEN12+ crop pilot."""

from __future__ import annotations

import argparse
import hashlib
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
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from acquire_mars_metadata import repo_root, sha256


DEFAULT_PROTOCOL = Path("configs/mars_cloudsen12_spatial_pilot_protocol.json")
DEFAULT_CLOUD_METADATA = Path(".research/source_audit_20260715/cloudsen12_clear_images.csv")
DEFAULT_CLOUD_STATS = Path(".research/source_audit_20260715/cloudsen12_stats_dataset.csv")
DEFAULT_MARS_METADATA = Path(
    "EarthRemoteSensingRapidResponse/Data Collection/s2_emit_pairs/publication-v1/external/MARS-S2L/validated_images_all.csv"
)
DEFAULT_OUTPUT = Path(".research/cloudsen12_spatial_pilot/selected_manifest.jsonl")
DEFAULT_JSON = Path("reports/acquisition/cloudsen12_spatial_pilot_selection.json")
DEFAULT_MARKDOWN = Path("reports/acquisition/CLOUDSEN12_SPATIAL_PILOT_SELECTION.md")


def stable_hash(seed: str, identity: str) -> str:
    return hashlib.sha256(f"{seed}|{identity}".encode("utf-8")).hexdigest()


def haversine_min_km(
    lon: np.ndarray, lat: np.ndarray, reference_lon: np.ndarray, reference_lat: np.ndarray
) -> np.ndarray:
    """Return each query point's minimum great-circle distance to references."""
    output = np.full(lon.shape, np.inf, dtype=np.float64)
    ref_lon = np.radians(reference_lon.astype(np.float64))
    ref_lat = np.radians(reference_lat.astype(np.float64))
    for start in range(0, lon.size, 512):
        stop = min(start + 512, lon.size)
        qlon = np.radians(lon[start:stop].astype(np.float64))[:, None]
        qlat = np.radians(lat[start:stop].astype(np.float64))[:, None]
        dlon = ref_lon[None, :] - qlon
        dlat = ref_lat[None, :] - qlat
        value = np.sin(dlat / 2.0) ** 2 + np.cos(qlat) * np.cos(ref_lat)[None, :] * np.sin(dlon / 2.0) ** 2
        distance = 2.0 * 6371.0088 * np.arcsin(np.sqrt(np.clip(value, 0.0, 1.0)))
        output[start:stop] = distance.min(axis=1)
    return output


def select_partition(
    frame: pd.DataFrame, size: int, hard_fraction: float, seed: str
) -> pd.DataFrame:
    if size <= 0 or len(frame) < size or not 0.0 <= hard_fraction <= 1.0:
        raise ValueError("Invalid pilot selection request")
    values = frame.copy()
    values["stable_hash"] = [
        stable_hash(seed, identity) for identity in values["id_loc_image"].astype(str)
    ]
    hard_count = int(round(size * hard_fraction))
    hard = values.sort_values(
        ["MBMP_std", "MBMP_max", "stable_hash"],
        ascending=[False, False, True],
    ).head(hard_count).copy()
    hard["selection_stratum"] = "hard_mbmp"
    remaining = values[~values["id_loc_image"].isin(hard["id_loc_image"])].copy()
    country_queues = {
        country: part.sort_values("stable_hash").to_dict("records")
        for country, part in remaining.groupby("country", dropna=False)
    }
    diverse_records: list[dict[str, Any]] = []
    target = size - hard_count
    while len(diverse_records) < target and any(country_queues.values()):
        for country in sorted(country_queues, key=str):
            queue = country_queues[country]
            if queue and len(diverse_records) < target:
                diverse_records.append(queue.pop(0))
    diverse = pd.DataFrame(diverse_records)
    if len(diverse) != target:
        raise RuntimeError("Country-round-robin selection did not reach target size")
    diverse["selection_stratum"] = "country_diverse"
    selected = pd.concat([hard, diverse], ignore_index=True)
    if len(selected) != size or selected["id_loc_image"].duplicated().any():
        raise RuntimeError("Pilot selection has an invalid size or duplicate")
    return selected.sort_values("id_loc_image").reset_index(drop=True)


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(item, sort_keys=True, separators=(",", ":")) + "\n" for item in records),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    summary = report["summary"]
    lines = [
        "# CloudSEN12+ spatial pilot selection",
        "",
        "The bounded pilot was selected before pixel acquisition. CloudSEN12 test rows were excluded, and every selected crop is at least 25 km from every MARS emitter site.",
        "",
        f"- Eligible all-clear, spatially disjoint rows: **{summary['eligible_rows']:,}**.",
        f"- Auxiliary training: **{summary['by_role']['auxiliary_training']:,}** rows.",
        f"- Development: **{summary['by_role']['development']:,}** rows.",
        f"- Countries represented: **{summary['countries']:,}**.",
        f"- Minimum observed distance to a MARS emitter: **{summary['minimum_distance_to_mars_km']:.2f} km**.",
        f"- Sealed CloudSEN12 test rows selected/accessed: **0**.",
        "",
        "Half of each partition targets high-MBMP-variance clear backgrounds; half is selected by deterministic country round-robin. Country and location metadata are sampling-only and prohibited model inputs.",
        "",
        "Each selected row preserves the producer-published CRS and six-value affine transform for the exact 200x200 grid.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL.as_posix())
    parser.add_argument("--cloud-metadata", default=DEFAULT_CLOUD_METADATA.as_posix())
    parser.add_argument("--cloud-stats", default=DEFAULT_CLOUD_STATS.as_posix())
    parser.add_argument("--mars-metadata", default=DEFAULT_MARS_METADATA.as_posix())
    parser.add_argument("--output", default=DEFAULT_OUTPUT.as_posix())
    parser.add_argument("--output-json", default=DEFAULT_JSON.as_posix())
    parser.add_argument("--output-markdown", default=DEFAULT_MARKDOWN.as_posix())
    args = parser.parse_args()
    root = repo_root()
    paths = {
        "protocol": (root / args.protocol).resolve(),
        "cloud_metadata": (root / args.cloud_metadata).resolve(),
        "cloud_stats": (root / args.cloud_stats).resolve(),
        "mars_metadata": (root / args.mars_metadata).resolve(),
    }
    protocol = json.loads(paths["protocol"].read_text(encoding="utf-8"))
    expected = {
        "cloud_metadata": protocol["sources"]["metadata_sha256"],
        "cloud_stats": protocol["sources"]["stats_sha256"],
        "mars_metadata": protocol["sources"]["mars_metadata_sha256"],
    }
    for name, digest in expected.items():
        if sha256(paths[name]) != digest:
            raise ValueError(f"Frozen {name} hash mismatch")

    metadata_columns = [
        "id_loc_image", "location_name", "roi_id", "split_name", "isplume",
        "satellite", "tile", "background_image_tile", "tile_date", "country",
        "lon", "lat", "wind_u", "wind_v", "crs", "transform_a", "transform_b",
        "transform_c", "transform_d", "transform_e", "transform_f", "width", "height",
    ]
    metadata = pd.read_csv(paths["cloud_metadata"], usecols=metadata_columns, low_memory=False)
    stats = pd.read_csv(
        paths["cloud_stats"],
        usecols=["id_loc_image", "cloudmask_0.0", "cloudmask_1.0", "MBMP_std", "MBMP_max"],
        low_memory=False,
    ).rename(columns={"id_loc_image": "location_name"})
    stats[["cloudmask_0.0", "cloudmask_1.0"]] = stats[["cloudmask_0.0", "cloudmask_1.0"]].fillna(0.0)
    joined = metadata.merge(stats, on="location_name", how="inner", validate="one_to_one")
    if joined["isplume"].astype(bool).any():
        raise ValueError("Positive row reached CloudSEN12 negative pilot")
    allowed_splits = set(protocol["eligibility"]["published_partitions_allowed"])
    eligible = joined[
        joined["split_name"].isin(allowed_splits)
        & (joined["cloudmask_0.0"] == protocol["eligibility"]["cloudmask_clear_pixels"])
        & (joined["cloudmask_1.0"] == protocol["eligibility"]["cloudmask_nonclear_pixels"])
        & joined["satellite"].astype(str).str.startswith("S2")
    ].copy()

    mars = pd.read_csv(paths["mars_metadata"], usecols=["id_location", "lon", "lat"], low_memory=False)
    mars = mars.dropna(subset=["lon", "lat"]).drop_duplicates("id_location")
    eligible["distance_to_mars_km"] = haversine_min_km(
        eligible["lon"].to_numpy(), eligible["lat"].to_numpy(),
        mars["lon"].to_numpy(), mars["lat"].to_numpy(),
    )
    minimum_distance = float(protocol["eligibility"]["minimum_distance_from_any_mars_emitter_site_km"])
    eligible = eligible[eligible["distance_to_mars_km"] >= minimum_distance].copy()

    seed = str(protocol["selection"]["seed_text"])
    hard_fraction = float(protocol["selection"]["per_partition_hard_fraction"])
    selected_parts = []
    role_map = {"train": "auxiliary_training", "validation": "development"}
    for split, role in role_map.items():
        target = int(protocol["sample_sizes"][split])
        local = select_partition(
            eligible[eligible["split_name"] == split], target, hard_fraction, f"{seed}|{split}"
        )
        local["research_role"] = role
        selected_parts.append(local)
    selected = pd.concat(selected_parts, ignore_index=True).sort_values("id_loc_image")
    if not (
        (selected["width"] == 200).all()
        and (selected["height"] == 200).all()
        and (selected["transform_a"] == 10.0).all()
        and (selected["transform_b"] == 0.0).all()
        and (selected["transform_d"] == 0.0).all()
        and (selected["transform_e"] == -10.0).all()
    ):
        raise ValueError("Selected producer grids differ from the frozen 200x200 10 m contract")
    records = []
    for row in selected.to_dict("records"):
        records.append(
            {
                "schema_version": 1,
                "sample_id": str(row["id_loc_image"]),
                "group_id": f"cloudsen12:{row['roi_id']}",
                "research_role": str(row["research_role"]),
                "source_name": "CloudSEN12+ clear-scene false-positive cohort",
                "sensor_family": "Sentinel-2",
                "target_product": str(row["tile"]),
                "background_product": str(row["background_image_tile"]),
                "tile_date": str(row["tile_date"]),
                "source_center": [float(row["lon"]), float(row["lat"])],
                "source_grid": {
                    "crs": str(row["crs"]),
                    "transform": [
                        float(row["transform_a"]),
                        float(row["transform_b"]),
                        float(row["transform_c"]),
                        float(row["transform_d"]),
                        float(row["transform_e"]),
                        float(row["transform_f"]),
                    ],
                    "width": int(row["width"]),
                    "height": int(row["height"]),
                    "provenance": "frozen CloudSEN12+ producer metadata",
                },
                "wind_u": float(row["wind_u"]),
                "wind_v": float(row["wind_v"]),
                "label_state": "NO_PLUME",
                "selection_stratum": str(row["selection_stratum"]),
                "distance_to_nearest_mars_site_km": round(float(row["distance_to_mars_km"]), 6),
                "sampling_country": str(row["country"]),
                "cloudsen12_split": str(row["split_name"]),
                "plume_geometries": [],
            }
        )
    output = (root / args.output).resolve()
    write_jsonl(output, records)
    role_counts = selected["research_role"].value_counts().to_dict()
    report = {
        "schema_version": 1,
        "status": "bounded nonsealed CloudSEN12 spatial pilot selected; pixels not acquired",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "eligible_rows": int(len(eligible)),
            "selected_rows": int(len(selected)),
            "by_role": {key: int(value) for key, value in sorted(role_counts.items())},
            "by_stratum": {
                key: int(value) for key, value in sorted(selected["selection_stratum"].value_counts().items())
            },
            "countries": int(selected["country"].nunique()),
            "groups": int(selected["roi_id"].nunique()),
            "minimum_distance_to_mars_km": float(selected["distance_to_mars_km"].min()),
            "sealed_test_rows_selected": 0,
        },
        "output": {
            "path": args.output,
            "bytes": output.stat().st_size,
            "sha256": sha256(output),
        },
        "source_hashes": expected,
        "protocol_sha256": sha256(paths["protocol"]),
        "script_sha256": sha256(Path(__file__).resolve()),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "country_used_as_model_input": False,
        "published_producer_grid_preserved": True,
        "cloudsen12_test_accessed": False,
        "paper_test_accessed": False,
    }
    output_json = (root / args.output_json).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown((root / args.output_markdown).resolve(), report)
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
