#!/usr/bin/env python3
"""Materialize the exact published CloudSEN12+ test cohort after model freeze."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from acquire_mars_metadata import repo_root, sha256


DEFAULT_PROTOCOL = Path("configs/cloudsen12_fresh_test_protocol.json")
DEFAULT_OUTPUT = Path(".research/cloudsen12_fresh_test/cohort.jsonl")
DEFAULT_JSON = Path("reports/acquisition/cloudsen12_fresh_test_cohort.json")
DEFAULT_MARKDOWN = Path("reports/acquisition/CLOUDSEN12_FRESH_TEST_COHORT.md")


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(
            json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=True)
            + "\n"
            for record in records
        ),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    summary = report["summary"]
    lines = [
        "# CloudSEN12+ fresh published-test cohort",
        "",
        f"- Exact published test rows: **{summary['rows']:,}**.",
        f"- Unique ROI groups: **{summary['groups']:,}**.",
        f"- Countries: **{summary['countries']:,}**.",
        f"- Rows with missing published wind: **{summary['missing_wind_rows']:,}**.",
        "- Every row is published no-plume truth on its producer 200x200 grid.",
        f"- All-clear scenes: **{summary['all_clear_rows']:,}**; scenes with published non-clear pixels: **{summary['nonclear_rows']:,}**.",
        f"- Published pixel composition: **{summary['clear_pixels']:,}** clear and **{summary['nonclear_pixels']:,}** non-clear.",
        "- Exact-product resolution and imagery acquisition occur only after this cohort receipt is committed.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL.as_posix())
    parser.add_argument("--output", default=DEFAULT_OUTPUT.as_posix())
    parser.add_argument("--output-json", default=DEFAULT_JSON.as_posix())
    parser.add_argument("--output-markdown", default=DEFAULT_MARKDOWN.as_posix())
    args = parser.parse_args()
    root = repo_root()
    protocol_path = (root / args.protocol).resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    metadata_path = (root / protocol["sources"]["metadata_path"]).resolve()
    stats_path = (root / protocol["sources"]["stats_path"]).resolve()
    if sha256(metadata_path) != protocol["sources"]["metadata_sha256"]:
        raise ValueError("CloudSEN metadata hash mismatch")
    if sha256(stats_path) != protocol["sources"]["stats_sha256"]:
        raise ValueError("CloudSEN statistics hash mismatch")
    authorization = (root / protocol["authorization"]["report_path"]).resolve()
    if sha256(authorization) != protocol["authorization"]["report_sha256"]:
        raise ValueError("Model-finalization authorization hash mismatch")
    authorization_report = json.loads(authorization.read_text(encoding="utf-8"))
    if not authorization_report["all_finalization_gates_pass"]:
        raise ValueError("Model finalization did not authorize fresh test access")

    metadata_columns = [
        "id_loc_image", "location_name", "roi_id", "split_name", "isplume",
        "satellite", "tile", "background_image_tile", "tile_date", "country",
        "lon", "lat", "wind_u", "wind_v", "crs", "transform_a", "transform_b",
        "transform_c", "transform_d", "transform_e", "transform_f", "width", "height",
    ]
    metadata = pd.read_csv(metadata_path, usecols=metadata_columns, low_memory=False)
    stats = pd.read_csv(
        stats_path,
        usecols=["id_loc_image", "cloudmask_0.0", "cloudmask_1.0"],
        low_memory=False,
    ).rename(columns={"id_loc_image": "location_name"})
    stats[["cloudmask_0.0", "cloudmask_1.0"]] = stats[
        ["cloudmask_0.0", "cloudmask_1.0"]
    ].fillna(0.0)
    joined = metadata.merge(stats, on="location_name", how="inner", validate="one_to_one")
    selected = joined[joined["split_name"] == "test"].copy()
    if len(selected) != int(protocol["expected_rows"]):
        raise ValueError("Published CloudSEN test row count changed")
    clear_pixels = int(selected["cloudmask_0.0"].sum())
    nonclear_pixels = int(selected["cloudmask_1.0"].sum())
    all_clear_rows = int((selected["cloudmask_0.0"] == 40000.0).sum())
    nonclear_rows = int((selected["cloudmask_1.0"] > 0.0).sum())
    truth = protocol["truth_contract"]
    if (
        selected["isplume"].astype(bool).any()
        or not ((selected["cloudmask_0.0"] + selected["cloudmask_1.0"]) == 40000.0).all()
        or clear_pixels != int(truth["clear_pixels"])
        or nonclear_pixels != int(truth["nonclear_pixels"])
        or all_clear_rows != int(truth["all_clear_rows"])
        or nonclear_rows != int(truth["nonclear_rows"])
        or not selected["satellite"].astype(str).str.startswith("S2").all()
    ):
        raise ValueError("CloudSEN published test label/sensor contract changed")
    if not (
        (selected["width"] == 200).all()
        and (selected["height"] == 200).all()
        and (selected["transform_a"] == 10.0).all()
        and (selected["transform_b"] == 0.0).all()
        and (selected["transform_d"] == 0.0).all()
        and (selected["transform_e"] == -10.0).all()
    ):
        raise ValueError("CloudSEN published test producer grid changed")

    records = []
    for row in selected.sort_values("id_loc_image").to_dict("records"):
        records.append(
            {
                "schema_version": 1,
                "sample_id": str(row["id_loc_image"]),
                "group_id": f"cloudsen12:{row['roi_id']}",
                "research_role": "fresh_external_test",
            "source_name": "CloudSEN12+ published no-plume test",
                "sensor_family": "Sentinel-2",
                "target_product": str(row["tile"]),
                "background_product": str(row["background_image_tile"]),
                "tile_date": str(row["tile_date"]),
                "source_center": [float(row["lon"]), float(row["lat"])],
                "source_grid": {
                    "crs": str(row["crs"]),
                    "transform": [
                        float(row["transform_a"]), float(row["transform_b"]),
                        float(row["transform_c"]), float(row["transform_d"]),
                        float(row["transform_e"]), float(row["transform_f"]),
                    ],
                    "width": int(row["width"]),
                    "height": int(row["height"]),
                    "provenance": "published CloudSEN12+ producer metadata",
                },
                "wind_u": float(row["wind_u"]),
                "wind_v": float(row["wind_v"]),
                "label_state": "NO_PLUME",
                "sampling_country": str(row["country"]),
                "cloudsen12_split": "test",
                "plume_geometries": [],
            }
        )
    output = (root / args.output).resolve()
    write_jsonl(output, records)
    missing_wind = selected[["wind_u", "wind_v"]].isna().any(axis=1)
    report = {
        "schema_version": 1,
        "status": "exact published CloudSEN test cohort materialized; pixels not resolved",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "rows": len(records),
            "groups": int(selected["roi_id"].nunique()),
            "countries": int(selected["country"].nunique()),
            "missing_wind_rows": int(missing_wind.sum()),
            "plume_rows": 0,
            "all_clear_rows": all_clear_rows,
            "nonclear_rows": nonclear_rows,
            "clear_pixels": clear_pixels,
            "nonclear_pixels": nonclear_pixels,
        },
        "output": {
            "path": args.output,
            "bytes": output.stat().st_size,
            "sha256": sha256(output),
        },
        "authorization": protocol["authorization"],
        "protocol_sha256": sha256(protocol_path),
        "script_sha256": sha256(Path(__file__).resolve()),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "paper_test_accessed": False,
    }
    output_json = (root / args.output_json).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown((root / args.output_markdown).resolve(), report)
    print(json.dumps(report["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
