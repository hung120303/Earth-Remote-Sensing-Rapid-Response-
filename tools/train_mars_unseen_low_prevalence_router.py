#!/usr/bin/env python3
"""Select temporal routing for unseen, low-prevalence sites on development folds."""

from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "EarthRemoteSensingRapidResponse"
for path in (MODEL_ROOT, ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from acquire_mars_metadata import sha256  # noqa: E402
from train_mars_crossfold_bagged_scene_head import load_development  # noqa: E402
from train_mars_oof_scene_ensemble_v2 import ap_group_bootstrap  # noqa: E402
from train_mars_scene_ranker import comparison, metric_summary  # noqa: E402
from train_mars_temporal_site_prior import temporal_site_prior  # noqa: E402


DEFAULT_PROTOCOL = Path("configs/mars_unseen_low_prevalence_router_protocol.json")


def low_prevalence_mask(
    labels: np.ndarray, groups: np.ndarray, max_positive_rate: float
) -> np.ndarray:
    if not 0.0 <= max_positive_rate <= 1.0:
        raise ValueError("max_positive_rate must be in [0,1]")
    labels = np.asarray(labels, dtype=np.uint8)
    groups = np.asarray(groups).astype(str)
    selected = np.zeros(labels.size, dtype=bool)
    for group in np.unique(groups):
        rows = groups == group
        if float(np.mean(labels[rows])) <= max_positive_rate:
            selected[rows] = True
    return selected


def evaluate_rows(values: dict[str, Any], scores: np.ndarray, rows: np.ndarray) -> dict[str, Any]:
    current = metric_summary(values["labels"][rows], values["current"][rows], values["sensors"][rows])
    candidate = metric_summary(values["labels"][rows], scores[rows], values["sensors"][rows])
    return {"current": current, "candidate": candidate, "versus_current": comparison(candidate, current)}


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    selected = report["selection"]["selected"]
    low = report["confirmation"]["low_prevalence_combined"]["versus_current"]["delta"]
    lines = [
        "# Unseen low-prevalence site router",
        "",
        f"- Selected minimum history: **{selected['min_site_size']}**; top-k: **{selected['top_k']}**; weight: **{selected['weight']:.2f}**.",
        f"- Confirmation low-prevalence AP delta: **{low['average_precision']:+.5f}**; recall delta: **{low['recall_at_fpr_0_0713']:+.5f}**.",
        f"- All promotion gates pass: **{str(report['all_promotion_gates_pass']).lower()}**.",
        "",
        "Known training sites remain exactly on the frozen current v3 head; the label-free temporal route applies only to unseen sites at inference.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL.as_posix())
    args = parser.parse_args()
    protocol_path = (ROOT / args.protocol).resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if sha256(Path(__file__).resolve()) != protocol["trainer"]["sha256"]:
        raise ValueError("Unseen-site router trainer hash mismatch")
    paths: dict[str, Path] = {}
    for name, contract in protocol["inputs"].items():
        path = (ROOT / contract["path"]).resolve()
        if sha256(path) != contract["sha256"]:
            raise ValueError(f"Frozen unseen-router input hash mismatch: {name}")
        paths[name] = path
    values = load_development(
        {"inner": paths["inner"], "fold0": paths["fold0"], "fold1": paths["fold1"]},
        paths["scores"],
    )
    max_rate = float(protocol["target_domain"]["maximum_site_positive_rate"])
    low_rows = low_prevalence_mask(values["labels"], values["groups"], max_rate)
    selection_folds = list(map(int, protocol["folds"]["selection"]))
    confirmation_folds = list(map(int, protocol["folds"]["confirmation"]))
    selection_rows = np.isin(values["folds"], selection_folds)
    confirmation_rows = np.isin(values["folds"], confirmation_folds)
    selection_low = selection_rows & low_rows
    confirmation_low = confirmation_rows & low_rows
    candidates: list[dict[str, Any]] = []
    scores_by_key: dict[str, np.ndarray] = {}
    for min_size, top_k, weight in itertools.product(
        protocol["search"]["min_site_size"], protocol["search"]["top_k"], protocol["search"]["weights"]
    ):
        scores = temporal_site_prior(
            values["current"], values["groups"], int(top_k), float(weight), int(min_size)
        )
        key = f"min{min_size}_top{top_k}_weight{weight}"
        scores_by_key[key] = scores
        low_combined = evaluate_rows(values, scores, selection_low)
        whole_combined = evaluate_rows(values, scores, selection_rows)
        low_per_fold = {
            str(fold): evaluate_rows(values, scores, (values["folds"] == fold) & low_rows)
            for fold in selection_folds
        }
        bootstrap = ap_group_bootstrap(
            values["labels"][selection_low], values["current"][selection_low], scores[selection_low],
            values["groups"][selection_low], replicates=int(protocol["bootstrap"]["replicates"]),
            seed=int(protocol["bootstrap"]["selection_seed"]) + len(candidates),
        )
        low_ap = [result["versus_current"]["delta"]["average_precision"] for result in low_per_fold.values()]
        low_recall = [result["versus_current"]["delta"]["recall_at_fpr_0_0713"] for result in low_per_fold.values()]
        whole_delta = whole_combined["versus_current"]["delta"]
        stable = bool(
            low_combined["versus_current"]["delta"]["average_precision"] > 0.0
            and low_combined["versus_current"]["delta"]["recall_at_fpr_0_0713"] >= 0.0
            and bootstrap["lower"] > 0.0
            and min(low_ap) >= -float(protocol["gates"]["per_fold_low_ap_tolerance"])
            and min(low_recall) >= -float(protocol["gates"]["per_fold_low_recall_tolerance"])
            and whole_delta["average_precision"] >= -float(protocol["gates"]["whole_fold_ap_noninferiority"])
            and whole_delta["recall_at_fpr_0_0713"] >= -float(protocol["gates"]["whole_fold_recall_noninferiority"])
        )
        candidates.append({
            "key": key, "min_site_size": int(min_size), "top_k": int(top_k), "weight": float(weight),
            "low_prevalence_combined": low_combined, "whole_combined": whole_combined,
            "low_prevalence_per_fold": low_per_fold, "paired_site_bootstrap_low_ap_delta": bootstrap,
            "stable": stable,
            "rank": [int(stable), bootstrap["lower"], min(low_ap), low_combined["versus_current"]["delta"]["average_precision"]],
        })
    selected = max(candidates, key=lambda candidate: tuple(candidate["rank"]))
    scores = scores_by_key[selected["key"]]
    confirmation_low_combined = evaluate_rows(values, scores, confirmation_low)
    confirmation_whole = evaluate_rows(values, scores, confirmation_rows)
    confirmation_low_per_fold = {
        str(fold): evaluate_rows(values, scores, (values["folds"] == fold) & low_rows)
        for fold in confirmation_folds
    }
    confirmation_bootstrap = ap_group_bootstrap(
        values["labels"][confirmation_low], values["current"][confirmation_low], scores[confirmation_low],
        values["groups"][confirmation_low], replicates=int(protocol["bootstrap"]["replicates"]),
        seed=int(protocol["bootstrap"]["confirmation_seed"]),
    )
    low_ap = [result["versus_current"]["delta"]["average_precision"] for result in confirmation_low_per_fold.values()]
    low_recall = [result["versus_current"]["delta"]["recall_at_fpr_0_0713"] for result in confirmation_low_per_fold.values()]
    whole_delta = confirmation_whole["versus_current"]["delta"]
    checks = {
        "selection_stable": bool(selected["stable"]),
        "confirmation_low_ap_point_higher": confirmation_low_combined["versus_current"]["delta"]["average_precision"] > 0.0,
        "confirmation_low_recall_point_no_lower": confirmation_low_combined["versus_current"]["delta"]["recall_at_fpr_0_0713"] >= 0.0,
        "confirmation_low_ap_lower_positive": confirmation_bootstrap["lower"] > 0.0,
        "each_confirmation_low_fold_ap_within_tolerance": min(low_ap) >= -float(protocol["gates"]["per_fold_low_ap_tolerance"]),
        "each_confirmation_low_fold_recall_within_tolerance": min(low_recall) >= -float(protocol["gates"]["per_fold_low_recall_tolerance"]),
        "confirmation_whole_ap_noninferior": whole_delta["average_precision"] >= -float(protocol["gates"]["whole_fold_ap_noninferiority"]),
        "confirmation_whole_recall_noninferior": whole_delta["recall_at_fpr_0_0713"] >= -float(protocol["gates"]["whole_fold_recall_noninferiority"]),
    }
    passed = all(checks.values())
    thresholds = [
        evaluate_rows(values, scores, values["folds"] == fold)["candidate"]["operating_point"]["threshold"]
        for fold in confirmation_folds
    ]
    operational_threshold = max(float(protocol["base_architecture"]["operational_scene_threshold"]), *map(float, thresholds))
    artifact_record = None
    if passed:
        artifact_path = (ROOT / protocol["outputs"]["artifact"]).resolve()
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = artifact_path.with_suffix(artifact_path.suffix + ".tmp")
        joblib.dump({
            "schema_version": 1,
            "kind": "mars_unseen_low_prevalence_temporal_router",
            "known_training_sites": sorted(set(values["groups"].tolist())),
            "maximum_validation_site_positive_rate": max_rate,
            "min_site_size": selected["min_site_size"], "top_k": selected["top_k"], "weight": selected["weight"],
            "operational_scene_threshold": operational_threshold,
            "base_score": "frozen v3 stronger OOF ExtraTrees scene score",
            "protocol_sha256": sha256(protocol_path),
        }, temporary, compress=3)
        os.replace(temporary, artifact_path)
        artifact_record = {
            "path": protocol["outputs"]["artifact"], "bytes": artifact_path.stat().st_size,
            "sha256": sha256(artifact_path), "operational_scene_threshold": operational_threshold,
            "known_training_sites": len(set(values["groups"].tolist())), "tracked": False,
        }
    report = {
        "schema_version": 1,
        "scope": "development-only unseen low-prevalence routing; exact paper cache not loaded",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "target_domain": {
            "maximum_site_positive_rate": max_rate,
            "selection_rows": int(np.count_nonzero(selection_low)),
            "selection_positive": int(values["labels"][selection_low].sum()),
            "selection_sites": len(set(values["groups"][selection_low].tolist())),
            "confirmation_rows": int(np.count_nonzero(confirmation_low)),
            "confirmation_positive": int(values["labels"][confirmation_low].sum()),
            "confirmation_sites": len(set(values["groups"][confirmation_low].tolist())),
        },
        "selection": {"candidates": candidates, "selected": selected},
        "confirmation": {
            "low_prevalence_combined": confirmation_low_combined,
            "whole_combined": confirmation_whole,
            "low_prevalence_per_fold": confirmation_low_per_fold,
            "paired_site_bootstrap_low_ap_delta": confirmation_bootstrap,
        },
        "promotion_checks": checks, "all_promotion_gates_pass": passed,
        "operational_scene_threshold": operational_threshold, "artifact": artifact_record,
        "decision": "Freeze unseen low-prevalence routing for fresh safety evaluation." if passed else "Reject unseen low-prevalence routing before fresh or paper evaluation.",
        "provenance": {
            "protocol_sha256": sha256(protocol_path), "script_sha256": sha256(Path(__file__).resolve()),
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        },
    }
    output_json = (ROOT / protocol["outputs"]["json"]).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown((ROOT / protocol["outputs"]["markdown"]).resolve(), report)
    print(json.dumps({"ok": passed, "selected": selected["key"], "checks": checks, "artifact": artifact_record}, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
