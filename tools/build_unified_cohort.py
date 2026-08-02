#!/usr/bin/env python3
"""Audit and bind the authorized ERSRR v6 multi-cohort training boundary."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = Path("reports/acquisition/mars_v6_unified_cohort.json")
PATHS = {
    "mars_manifest": Path(
        "EarthRemoteSensingRapidResponse/Data Collection/s2_emit_pairs/publication-v1/"
        "external/MARS-S2L/paper_v3_development_samples.jsonl"
    ),
    "mars_folds": Path("configs/mars_paper_v3_group_folds.json"),
    "methanes2cm_auxiliary": Path(
        ".research/methanes2cm_mars_disjoint/model_auxiliary_training.jsonl"
    ),
    "methanes2cm_development": Path(
        ".research/methanes2cm_mars_disjoint/model_development.jsonl"
    ),
    "methanes2cm_receipt": Path(
        "reports/acquisition/methanes2cm_mars_disjoint_model_manifest.json"
    ),
    "unep_auxiliary": Path(
        ".research/unep_mars_post2024_refresh_20260801/model_auxiliary_combined.jsonl"
    ),
    "unep_receipt": Path(
        "reports/acquisition/unep_mars_post2024_refresh_20260801.json"
    ),
    "cloudsen_auxiliary": Path(
        ".research/cloudsen12_spatial_pilot/model_auxiliary_training.jsonl"
    ),
    "prithvi_receipt": Path("reports/acquisition/prithvi_eo_2_tiny_tl.json"),
}


def sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def iter_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def label(row: dict[str, Any]) -> int:
    if "label" in row:
        value = int(row["label"])
    else:
        value = int(str(row["label_state"]).upper() == "PLUME")
    if value not in (0, 1):
        raise ValueError("Unified cohort contains a non-binary label")
    return value


def identifier(row: dict[str, Any]) -> str:
    value = str(row.get("sample_id", row.get("id", ""))).strip()
    if not value:
        raise ValueError("Unified cohort row lacks a sample identity")
    return value


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ids = [identifier(row) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Source cohort contains duplicate sample identities")
    groups = [str(row["group_id"]) for row in rows]
    locations = [
        str(row.get("exact_location_id", row.get("physical_location_id", row["group_id"])))
        for row in rows
    ]
    labels = [label(row) for row in rows]
    sensors = Counter(str(row.get("sensor_family", "Sentinel-2")) for row in rows)
    def product(row: dict[str, Any]) -> str:
        if row.get("product_level"):
            return str(row["product_level"])
        if str(row.get("source_dataset", "")).startswith("H1deaki/MethaneS2CM"):
            return "L2A"
        if str(row.get("sensor_family", "Sentinel-2")) == "Landsat":
            return "Collection 2 TOA"
        return "L1C"

    products = Counter(product(row) for row in rows)
    return {
        "rows": len(rows),
        "positives": int(sum(labels)),
        "negatives": int(len(labels) - sum(labels)),
        "groups": len(set(groups)),
        "exact_locations": len(set(locations)),
        "sensor_counts": dict(sorted(sensors.items())),
        "product_counts": dict(sorted(products.items())),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=DEFAULT_OUTPUT.as_posix())
    args = parser.parse_args()
    paths = {name: (ROOT / path).resolve() for name, path in PATHS.items()}
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Required v6 input is unavailable: {name}={path}")

    fold_payload = json.loads(paths["mars_folds"].read_text(encoding="utf-8"))
    group_to_fold = {
        str(item["group_id"]): int(item["fold"])
        for item in fold_payload["assignments"]
    }
    all_mars = iter_jsonl(paths["mars_manifest"])
    if any(str(row["research_role"]) not in {"development_training", "development_validation"} for row in all_mars):
        raise ValueError("MARS development manifest contains a sealed or unsupported role")
    mars = [row for row in all_mars if group_to_fold[str(row["group_id"])] in {3, 4}]
    mars_folds = Counter(group_to_fold[str(row["group_id"])] for row in mars)

    methane_aux = iter_jsonl(paths["methanes2cm_auxiliary"])
    methane_dev = iter_jsonl(paths["methanes2cm_development"])
    if any(str(row["research_role"]) != "auxiliary_training" for row in methane_aux):
        raise ValueError("MethaneS2CM auxiliary role changed")
    if any(str(row["research_role"]) != "development" for row in methane_dev):
        raise ValueError("MethaneS2CM development role changed")
    if any(float(row["minimum_mars_test_distance_km"]) <= 25.0 for row in methane_aux + methane_dev):
        raise ValueError("MethaneS2CM spatial exclusion is no longer strict")
    auxiliary_groups = {str(row["group_id"]) for row in methane_aux}
    development_groups = {str(row["group_id"]) for row in methane_dev}
    if auxiliary_groups & development_groups:
        raise ValueError("MethaneS2CM auxiliary and development groups overlap")
    ordered_development_groups = sorted(
        development_groups,
        key=lambda value: hashlib.sha256(
            f"ersrr-v6-risk-calibration|{value}".encode("utf-8")
        ).hexdigest(),
    )
    calibration_groups = ordered_development_groups[:24]
    confirmation_groups = ordered_development_groups[24:]
    if len(calibration_groups) != 24 or len(confirmation_groups) != 24:
        raise ValueError("V6 requires exactly 48 MethaneS2CM held-development groups")

    unep = iter_jsonl(paths["unep_auxiliary"])
    cloudsen = iter_jsonl(paths["cloudsen_auxiliary"])
    if any(label(row) != 1 or str(row["research_role"]) != "auxiliary_training" for row in unep):
        raise ValueError("UNEP auxiliary must remain positive-only training data")
    if any(label(row) != 0 or str(row["research_role"]) != "auxiliary_training" for row in cloudsen):
        raise ValueError("CloudSEN auxiliary must remain negative-only training data")

    prefixed_ids: set[str] = set()
    for source, rows in (
        ("mars", mars),
        ("methanes2cm", methane_aux),
        ("unep", unep),
        ("cloudsen", cloudsen),
    ):
        local = {f"{source}:{identifier(row)}" for row in rows}
        if prefixed_ids & local:
            raise ValueError("Unified source-prefixed identities overlap")
        prefixed_ids |= local

    inputs = {
        name: {
            "path": path.relative_to(ROOT).as_posix(),
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
        }
        for name, path in paths.items()
    }
    cohorts = {
        "mars_folds_3_4": {**summarize(mars), "fold_rows": dict(sorted(mars_folds.items()))},
        "methanes2cm_auxiliary_training": summarize(methane_aux),
        "methanes2cm_held_development": summarize(methane_dev),
        "unep_auxiliary_positive": summarize(unep),
        "cloudsen_auxiliary_negative": summarize(cloudsen),
    }
    training_names = (
        "mars_folds_3_4",
        "methanes2cm_auxiliary_training",
        "unep_auxiliary_positive",
        "cloudsen_auxiliary_negative",
    )
    report = {
        "schema_version": 1,
        "scope": "identity-only v6 unified-cohort audit; no imagery or outcomes copied",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": inputs,
        "cohorts": cohorts,
        "authorized_training_rows": int(sum(cohorts[name]["rows"] for name in training_names)),
        "held_confirmation_rows": int(cohorts["methanes2cm_held_development"]["rows"]),
        "methanes2cm_confirmation_partition": {
            "algorithm": "sort SHA256('ersrr-v6-risk-calibration|' + group_id); first 24 calibration, remaining 24 confirmation",
            "risk_calibration_groups": calibration_groups,
            "confirmation_groups": confirmation_groups,
            "risk_calibration_rows": int(
                sum(str(row["group_id"]) in set(calibration_groups) for row in methane_dev)
            ),
            "confirmation_rows": int(
                sum(str(row["group_id"]) in set(confirmation_groups) for row in methane_dev)
            ),
        },
        "boundaries": {
            "training": list(training_names),
            "held_confirmation": ["methanes2cm_held_development"],
            "prohibited": [
                "MARS fold 2",
                "MARS folds 0/1",
                "official MARS test",
                "MethaneS2CM location test",
                "UNEP sealed positives",
                "CloudSEN fresh test",
                "EMIT V002 outcomes",
                "MethaneSET",
            ],
        },
        "invariants": [
            "MARS training endpoints cross-fit folds 3 and 4; a held fold never fits itself.",
            "MethaneS2CM auxiliary and held-development 25 km groups are disjoint.",
            "Every selected MethaneS2CM location is strictly farther than 25 km from official MARS test locations.",
            "UNEP contributes positives only and CloudSEN contributes negatives only.",
            "Bulk imagery, HDF5, manifests, and future checkpoints remain ignored.",
        ],
    }
    output = (ROOT / args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": output.relative_to(ROOT).as_posix(), **cohorts}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
