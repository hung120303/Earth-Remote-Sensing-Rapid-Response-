#!/usr/bin/env python3
"""Build a resolved auxiliary-only UNEP refresh acquisition subset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COHORT = Path(
    ".research/unep_mars_post2024_refresh_20260801/eligible_manifest_reconciled.jsonl"
)
DEFAULT_ASSETS = Path(
    ".research/unep_mars_post2024_refresh_20260801/nonsealed_exact_assets_reconciled.jsonl"
)
DEFAULT_PRIOR = Path(".research/unep_mars_post2024/model_auxiliary_training.jsonl")
DEFAULT_OUTPUT_COHORT = Path(
    ".research/unep_mars_post2024_refresh_20260801/acquisition_subset.jsonl"
)
DEFAULT_OUTPUT_ASSETS = Path(
    ".research/unep_mars_post2024_refresh_20260801/acquisition_subset_assets.jsonl"
)
DEFAULT_JSON = Path(
    ".research/unep_mars_post2024_refresh_20260801/acquisition_subset.json"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    identities = [str(row["sample_id"]) for row in rows]
    if len(identities) != len(set(identities)):
        raise ValueError(f"Duplicate sample identity in {path}")
    return rows


def select_refresh_subset(
    cohort: list[dict[str, Any]],
    assets: list[dict[str, Any]],
    prior: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cohort_by_id = {str(row["sample_id"]): row for row in cohort}
    prior_ids = {str(row["sample_id"]) for row in prior}
    asset_by_id = {str(row["sample_id"]): row for row in assets}
    selected_ids = sorted(
        sample_id
        for sample_id, row in cohort_by_id.items()
        if row["research_role"] == "auxiliary_training"
        and sample_id not in prior_ids
        and sample_id in asset_by_id
        and asset_by_id[sample_id].get("status") == "resolved"
    )
    selected_cohort = [cohort_by_id[sample_id] for sample_id in selected_ids]
    selected_assets = [asset_by_id[sample_id] for sample_id in selected_ids]
    if {row["sample_id"] for row in selected_cohort} != {
        row["sample_id"] for row in selected_assets
    }:
        raise RuntimeError("Cohort and asset subset identities differ")
    if any(row["research_role"] != "auxiliary_training" for row in selected_cohort):
        raise RuntimeError("Acquisition subset contains a non-auxiliary row")
    if any(row["status"] != "resolved" for row in selected_assets):
        raise RuntimeError("Acquisition subset contains an unresolved asset")
    return selected_cohort, selected_assets


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", default=DEFAULT_COHORT.as_posix())
    parser.add_argument("--assets", default=DEFAULT_ASSETS.as_posix())
    parser.add_argument("--prior", default=DEFAULT_PRIOR.as_posix())
    parser.add_argument("--output-cohort", default=DEFAULT_OUTPUT_COHORT.as_posix())
    parser.add_argument("--output-assets", default=DEFAULT_OUTPUT_ASSETS.as_posix())
    parser.add_argument("--output-json", default=DEFAULT_JSON.as_posix())
    args = parser.parse_args()
    paths = {
        name: (ROOT / value).resolve()
        for name, value in {
            "cohort": args.cohort,
            "assets": args.assets,
            "prior": args.prior,
            "output_cohort": args.output_cohort,
            "output_assets": args.output_assets,
            "output_json": args.output_json,
        }.items()
    }
    cohort, assets = select_refresh_subset(
        read_jsonl(paths["cohort"]),
        read_jsonl(paths["assets"]),
        read_jsonl(paths["prior"]),
    )
    write_jsonl(paths["output_cohort"], cohort)
    write_jsonl(paths["output_assets"], assets)
    report = {
        "schema_version": 1,
        "status": "resolved_auxiliary_identity_diff_materialized",
        "inputs": {
            name: {"path": getattr(args, name), "sha256": sha256(paths[name])}
            for name in ("cohort", "assets", "prior")
        },
        "selection": {
            "rows": len(cohort),
            "groups": len({str(row["group_id"]) for row in cohort}),
            "by_sensor": dict(sorted(Counter(str(row["sensor_family"]) for row in cohort).items())),
            "rule": "reconciled auxiliary role, exact products resolved, identity absent from prior model auxiliary manifest",
        },
        "outputs": {
            "cohort": {"path": args.output_cohort, "sha256": sha256(paths["output_cohort"])},
            "assets": {"path": args.output_assets, "sha256": sha256(paths["output_assets"])},
        },
        "development_or_sealed_rows": 0,
    }
    paths["output_json"].parent.mkdir(parents=True, exist_ok=True)
    temporary = paths["output_json"].with_suffix(paths["output_json"].suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, paths["output_json"])
    print(json.dumps(report["selection"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
