#!/usr/bin/env python3
"""Freeze site-block folds and leakage controls for MARS paper-v3 development."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "EarthRemoteSensingRapidResponse"
if str(MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_ROOT))

from mars_s2l_adapter import iter_development_manifest  # noqa: E402

from acquire_mars_metadata import DEFAULT_OUTPUT, repo_root, sha256  # noqa: E402

DEFAULT_MANIFEST = DEFAULT_OUTPUT / "paper_v3_development_samples.jsonl"
DEFAULT_COHORT_REPORT = Path(
    "reports/acquisition/mars_s2l_paper_v3_mixed_cohort.json"
)
DEFAULT_CONFIG = Path("configs/mars_paper_v3_group_folds.json")
DEFAULT_JSON = Path("reports/experiments/mars_paper_v3_development_protocol.json")
DEFAULT_MARKDOWN = Path("reports/experiments/MARS_PAPER_V3_DEVELOPMENT_PROTOCOL.md")
N_FOLDS = 5
RANDOM_SEED = 20260713


def assignment_hash(group_to_fold: dict[str, int]) -> str:
    payload = "".join(
        f"{group_id}\t{group_to_fold[group_id]}\n"
        for group_id in sorted(group_to_fold)
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def assign_groups(
    records: list[dict[str, Any]],
    *,
    n_folds: int = N_FOLDS,
    random_seed: int = RANDOM_SEED,
) -> dict[str, int]:
    if n_folds < 3:
        raise ValueError("Research protocol requires at least three site-block folds")
    labels = np.asarray(
        [f"{record['label_state']}|{record['sensor_family']}" for record in records]
    )
    groups = np.asarray([str(record["group_id"]) for record in records])
    if len(set(groups)) < n_folds:
        raise ValueError("Fewer physical-location groups than requested folds")
    splitter = StratifiedGroupKFold(
        n_splits=n_folds, shuffle=True, random_state=random_seed
    )
    group_to_fold: dict[str, int] = {}
    for fold, (_, held_indices) in enumerate(
        splitter.split(np.zeros(len(records), dtype=np.uint8), labels, groups)
    ):
        for index in held_indices:
            group_id = str(groups[index])
            previous = group_to_fold.setdefault(group_id, fold)
            if previous != fold:
                raise ValueError("A physical-location group spans multiple folds")
    expected = set(str(value) for value in groups)
    if set(group_to_fold) != expected:
        raise ValueError("Not every development group received exactly one fold")
    return group_to_fold


def fold_summary(
    records: list[dict[str, Any]], group_to_fold: dict[str, int]
) -> dict[str, dict[str, int]]:
    summaries: dict[str, dict[str, int]] = {}
    for fold in range(max(group_to_fold.values()) + 1):
        selected = [
            record
            for record in records
            if group_to_fold[str(record["group_id"])] == fold
        ]
        counts = Counter()
        for record in selected:
            counts["rows"] += 1
            counts["positive"] += record["label_state"] == "PLUME"
            counts["negative"] += record["label_state"] == "NO_PLUME"
            counts["sentinel2"] += record["sensor_family"] == "Sentinel-2"
            counts["landsat"] += record["sensor_family"] == "Landsat"
            counts[f"role:{record['research_role']}"] += 1
        counts["sites"] = len({str(record["group_id"]) for record in selected})
        summaries[str(fold)] = dict(sorted(counts.items()))
    return summaries


def build_config(
    records: list[dict[str, Any]], group_to_fold: dict[str, int], manifest_hash: str
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "name": "mars_s2l_paper_v3_site_block_crossfit",
        "development_manifest_sha256": manifest_hash,
        "random_seed": RANDOM_SEED,
        "n_folds": N_FOLDS,
        "assignment_sha256": assignment_hash(group_to_fold),
        "assignments": [
            {"group_id": group_id, "fold": group_to_fold[group_id]}
            for group_id in sorted(group_to_fold)
        ],
        "architecture_selection": {
            "primary_fold": 0,
            "confirmation_fold": 1,
            "rule": "architecture changes require concordant AP and pixel-IoU gains on both site-held folds; no sealed-test access",
        },
        "crossfit": {
            "models": N_FOLDS,
            "fit_rule": "model k fits every development site except fold k",
            "oof_rule": "each development site is predicted only by the model that excluded it",
            "calibration_rule": "for outer fold k, calibrator and operating threshold fit OOF predictions from the other four folds only",
            "final_rule": "after freezing, fit the deployable calibrator on all OOF predictions and ensemble all five fold models",
        },
        "promotion_gate": {
            "scene": "cross-fitted AP higher, recall higher, and FPR no worse than the released checkpoint under identical rows",
            "pixel": "cross-fitted positive-pixel IoU higher than the released checkpoint",
            "uncertainty": "paired site-block bootstrap 95% interval for AP and IoU deltas must be above zero",
            "sensors": "report Sentinel-2 and Landsat strata; neither sensor may show a material regression",
        },
        "paper_test": {
            "access": "one shot only after model/checkpoint/calibration/threshold/postprocessing hashes are frozen",
            "views": ["full 43,529-scene archive", "15,655-scene test-only-site"],
            "comparison": "paper Table S5/S6 and stronger archived per-scene reconstruction",
        },
        "counts": {
            "rows": len(records),
            "sites": len(group_to_fold),
            "folds": fold_summary(records, group_to_fold),
        },
    }


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    config = report["protocol"]
    lines = [
        "# MARS paper-v3 successor development protocol",
        "",
        f"- Development scenes: {config['counts']['rows']:,}",
        f"- Physical-location groups: {config['counts']['sites']:,}",
        f"- Assignment SHA-256: `{config['assignment_sha256']}`",
        f"- Development manifest SHA-256: `{config['development_manifest_sha256']}`",
        "",
        "| Fold | Sites | Scenes | Plume | No plume | Sentinel-2 | Landsat |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for fold, values in config["counts"]["folds"].items():
        lines.append(
            f"| {fold} | {values['sites']:,} | {values['rows']:,} | {values['positive']:,} | {values['negative']:,} | {values['sentinel2']:,} | {values['landsat']:,} |"
        )
    lines.extend(
        [
            "",
            "Architecture work uses site-held fold 0 and must confirm on fold 1. The five-model campaign then produces exactly one out-of-fold prediction per development site. Calibration and thresholds are cross-fitted across those predictions.",
            "",
            "The sealed paper-test manifest is not read by this script. It may be opened once only after the complete candidate artifact and evaluator hashes are frozen.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST.as_posix())
    parser.add_argument("--cohort-report", default=DEFAULT_COHORT_REPORT.as_posix())
    parser.add_argument("--config", default=DEFAULT_CONFIG.as_posix())
    parser.add_argument("--output-json", default=DEFAULT_JSON.as_posix())
    parser.add_argument("--output-markdown", default=DEFAULT_MARKDOWN.as_posix())
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    root = repo_root()
    manifest = (root / args.manifest).resolve()
    cohort_report_path = (root / args.cohort_report).resolve()
    cohort_report = json.loads(cohort_report_path.read_text(encoding="utf-8"))
    expected_hash = cohort_report["local_ignored_artifacts"][
        "development_manifest_sha256"
    ]
    observed_hash = sha256(manifest)
    if observed_hash != expected_hash:
        raise ValueError("Development manifest differs from the frozen cohort report")
    records = list(iter_development_manifest(manifest))
    if any("paper_baseline" in record for record in records):
        raise ValueError("Development manifest contains a sealed-test baseline field")
    group_to_fold = assign_groups(records)
    config = build_config(records, group_to_fold, observed_hash)
    config_path = (root / args.config).resolve()
    if args.verify_only:
        observed = json.loads(config_path.read_text(encoding="utf-8"))
        if observed != config:
            raise ValueError("Frozen development fold protocol changed")
        print(
            json.dumps(
                {
                    "ok": True,
                    "rows": len(records),
                    "sites": len(group_to_fold),
                    "assignment_sha256": config["assignment_sha256"],
                },
                sort_keys=True,
            )
        )
        return 0

    write_json(config_path, config)
    report = {
        "schema_version": 1,
        "scope": "sealed-test-blind mixed-sensor successor development",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": config,
        "provenance": {
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True
            ).strip(),
            "script": "tools/freeze_mars_paper_protocol.py",
            "script_sha256": sha256(Path(__file__)),
            "sklearn": sklearn.__version__,
            "cohort_report_sha256": sha256(cohort_report_path),
        },
    }
    output_json = (root / args.output_json).resolve()
    output_markdown = (root / args.output_markdown).resolve()
    write_json(output_json, report)
    write_markdown(output_markdown, report)
    print(
        json.dumps(
            {
                "ok": True,
                "rows": len(records),
                "sites": len(group_to_fold),
                "assignment_sha256": config["assignment_sha256"],
                "folds": config["counts"]["folds"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
