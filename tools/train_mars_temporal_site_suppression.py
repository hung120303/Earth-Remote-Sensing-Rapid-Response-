#!/usr/bin/env python3
"""Select a one-sided label-free temporal site-suppression rule on development."""

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
from train_mars_temporal_site_prior import logit  # noqa: E402


DEFAULT_PROTOCOL = Path("configs/mars_temporal_site_suppression_protocol.json")


def temporal_site_suppression(
    scores: np.ndarray,
    groups: np.ndarray,
    top_k: int,
    cutoff: float,
    weight: float,
) -> np.ndarray:
    """Suppress, but never raise, scenes at sites below a temporal confidence cutoff."""
    if top_k < 1 or not 0.0 < cutoff < 1.0 or weight < 0.0:
        raise ValueError("Invalid temporal suppression parameters")
    scores = np.asarray(scores, dtype=np.float64)
    groups = np.asarray(groups).astype(str)
    if scores.ndim != 1 or groups.shape != scores.shape or not np.isfinite(scores).all():
        raise ValueError("Invalid temporal suppression inputs")
    logits = logit(scores)
    penalty = np.zeros_like(logits)
    cutoff_logit = float(logit(np.asarray([cutoff]))[0])
    for group in np.unique(groups):
        rows = np.flatnonzero(groups == group)
        count = min(top_k, rows.size)
        evidence = float(np.mean(np.partition(logits[rows], -count)[-count:]))
        penalty[rows] = float(weight) * min(0.0, evidence - cutoff_logit)
    candidate = 1.0 / (1.0 + np.exp(-np.clip(logits + penalty, -40.0, 40.0)))
    if np.any(candidate > scores + 1e-15):
        raise RuntimeError("One-sided site suppression raised a scene score")
    return candidate


def evaluate_rows(values: dict[str, Any], scores: np.ndarray, rows: np.ndarray) -> dict[str, Any]:
    current = metric_summary(values["labels"][rows], values["current"][rows], values["sensors"][rows])
    candidate = metric_summary(values["labels"][rows], scores[rows], values["sensors"][rows])
    return {"current": current, "candidate": candidate, "versus_current": comparison(candidate, current)}


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    selected = report["selection"]["selected"]
    lines = [
        "# One-sided temporal site suppression",
        "",
        f"- Selected top-k: **{selected['top_k']}**; confidence cutoff: **{selected['cutoff']:.2f}**; penalty weight: **{selected['weight']:.2f}**.",
        f"- Selection AP delta: **{selected['combined']['versus_current']['delta']['average_precision']:+.5f}**.",
        f"- Confirmation AP delta: **{report['confirmation']['combined']['versus_current']['delta']['average_precision']:+.5f}**.",
        f"- All promotion gates pass: **{str(report['all_promotion_gates_pass']).lower()}**.",
        "",
        "The rule is label-free at inference and is mathematically one-sided: candidate scores never exceed frozen current scores.",
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
        raise ValueError("Temporal suppression trainer hash mismatch")
    paths: dict[str, Path] = {}
    for name, contract in protocol["inputs"].items():
        path = (ROOT / contract["path"]).resolve()
        if sha256(path) != contract["sha256"]:
            raise ValueError(f"Frozen temporal suppression input hash mismatch: {name}")
        paths[name] = path
    values = load_development(
        {"inner": paths["inner"], "fold0": paths["fold0"], "fold1": paths["fold1"]},
        paths["scores"],
    )
    selection_folds = list(map(int, protocol["folds"]["selection"]))
    confirmation_folds = list(map(int, protocol["folds"]["confirmation"]))
    selection_rows = np.isin(values["folds"], selection_folds)
    confirmation_rows = np.isin(values["folds"], confirmation_folds)
    candidates: list[dict[str, Any]] = []
    score_by_key: dict[str, np.ndarray] = {}
    for top_k, cutoff, weight in itertools.product(
        protocol["search"]["top_k"],
        protocol["search"]["cutoffs"],
        protocol["search"]["weights"],
    ):
        scores = temporal_site_suppression(
            values["current"], values["groups"], int(top_k), float(cutoff), float(weight)
        )
        key = f"top{top_k}_cutoff{cutoff}_weight{weight}"
        score_by_key[key] = scores
        combined = evaluate_rows(values, scores, selection_rows)
        per_fold = {str(fold): evaluate_rows(values, scores, values["folds"] == fold) for fold in selection_folds}
        bootstrap = ap_group_bootstrap(
            values["labels"][selection_rows], values["current"][selection_rows], scores[selection_rows],
            values["groups"][selection_rows], replicates=int(protocol["bootstrap"]["replicates"]),
            seed=int(protocol["bootstrap"]["selection_seed"]) + len(candidates),
        )
        ap_deltas = [result["versus_current"]["delta"]["average_precision"] for result in per_fold.values()]
        recall_deltas = [result["versus_current"]["delta"]["recall_at_fpr_0_0713"] for result in per_fold.values()]
        stable = bool(
            combined["versus_current"]["delta"]["average_precision"] > 0.0
            and combined["versus_current"]["delta"]["recall_at_fpr_0_0713"] >= 0.0
            and bootstrap["lower"] > 0.0
            and min(ap_deltas) >= 0.0
            and min(recall_deltas) >= -float(protocol["gates"]["per_fold_recall_tolerance"])
        )
        candidates.append({
            "key": key, "top_k": int(top_k), "cutoff": float(cutoff), "weight": float(weight),
            "combined": combined, "per_fold": per_fold, "paired_site_bootstrap_ap_delta": bootstrap,
            "stable": stable,
            "rank": [int(stable), bootstrap["lower"], min(ap_deltas), combined["versus_current"]["delta"]["average_precision"]],
        })
    selected = max(candidates, key=lambda candidate: tuple(candidate["rank"]))
    scores = score_by_key[selected["key"]]
    confirmation_combined = evaluate_rows(values, scores, confirmation_rows)
    confirmation_per_fold = {str(fold): evaluate_rows(values, scores, values["folds"] == fold) for fold in confirmation_folds}
    confirmation_bootstrap = ap_group_bootstrap(
        values["labels"][confirmation_rows], values["current"][confirmation_rows], scores[confirmation_rows],
        values["groups"][confirmation_rows], replicates=int(protocol["bootstrap"]["replicates"]),
        seed=int(protocol["bootstrap"]["confirmation_seed"]),
    )
    confirmation_ap = [result["versus_current"]["delta"]["average_precision"] for result in confirmation_per_fold.values()]
    confirmation_recall = [result["versus_current"]["delta"]["recall_at_fpr_0_0713"] for result in confirmation_per_fold.values()]
    checks = {
        "selection_stable": bool(selected["stable"]),
        "confirmation_ap_point_higher": confirmation_combined["versus_current"]["delta"]["average_precision"] > 0.0,
        "confirmation_recall_point_no_lower": confirmation_combined["versus_current"]["delta"]["recall_at_fpr_0_0713"] >= 0.0,
        "confirmation_ap_lower_positive": confirmation_bootstrap["lower"] > 0.0,
        "each_confirmation_fold_ap_no_lower": min(confirmation_ap) >= 0.0,
        "each_confirmation_fold_recall_within_tolerance": min(confirmation_recall) >= -float(protocol["gates"]["per_fold_recall_tolerance"]),
        "all_scores_one_sided": bool(np.all(scores <= values["current"] + 1e-15)),
    }
    passed = all(checks.values())
    thresholds = [confirmation_per_fold[str(fold)]["candidate"]["operating_point"]["threshold"] for fold in confirmation_folds]
    operational_threshold = max(map(float, thresholds))
    artifact_record = None
    if passed:
        artifact_path = (ROOT / protocol["outputs"]["artifact"]).resolve()
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = artifact_path.with_suffix(artifact_path.suffix + ".tmp")
        joblib.dump({
            "schema_version": 1, "kind": "mars_one_sided_temporal_site_suppression",
            "top_k": selected["top_k"], "cutoff": selected["cutoff"], "weight": selected["weight"],
            "operational_scene_threshold": operational_threshold,
            "base_score": "frozen v3 stronger OOF ExtraTrees scene score",
            "protocol_sha256": sha256(protocol_path),
        }, temporary, compress=3)
        os.replace(temporary, artifact_path)
        artifact_record = {
            "path": protocol["outputs"]["artifact"], "bytes": artifact_path.stat().st_size,
            "sha256": sha256(artifact_path), "operational_scene_threshold": operational_threshold,
            "tracked": False,
        }
    report = {
        "schema_version": 1,
        "scope": "development-only one-sided temporal suppression; exact paper cache not loaded",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection": {"candidates": candidates, "selected": selected},
        "confirmation": {"combined": confirmation_combined, "per_fold": confirmation_per_fold, "paired_site_bootstrap_ap_delta": confirmation_bootstrap},
        "promotion_checks": checks, "all_promotion_gates_pass": passed,
        "operational_scene_threshold": operational_threshold, "artifact": artifact_record,
        "decision": "Freeze one-sided temporal suppression for fresh safety evaluation." if passed else "Reject one-sided temporal suppression before fresh or paper evaluation.",
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
