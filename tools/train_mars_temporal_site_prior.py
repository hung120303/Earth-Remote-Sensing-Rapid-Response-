#!/usr/bin/env python3
"""Select a label-free temporal site prior on authorized MARS development folds."""

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


DEFAULT_PROTOCOL = Path("configs/mars_temporal_site_prior_protocol.json")


def logit(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=np.float64), 1e-8, 1.0 - 1e-8)
    return np.log(clipped) - np.log1p(-clipped)


def temporal_site_prior(
    scores: np.ndarray,
    groups: np.ndarray,
    top_k: int,
    weight: float,
    min_site_size: int = 1,
) -> np.ndarray:
    """Add a label-free site's top-k temporal evidence to every scene logit."""
    if top_k < 1 or weight < 0.0 or min_site_size < 1:
        raise ValueError("top_k/min_site_size must be positive and weight non-negative")
    scores = np.asarray(scores, dtype=np.float64)
    groups = np.asarray(groups).astype(str)
    if scores.ndim != 1 or groups.shape != scores.shape or not np.isfinite(scores).all():
        raise ValueError("Invalid temporal site-prior inputs")
    logits = logit(scores)
    priors = np.zeros_like(logits)
    for group in np.unique(groups):
        rows = np.flatnonzero(groups == group)
        if rows.size < min_site_size:
            continue
        count = min(top_k, rows.size)
        priors[rows] = np.mean(np.partition(logits[rows], -count)[-count:])
    candidate_logits = logits + float(weight) * priors
    return 1.0 / (1.0 + np.exp(-np.clip(candidate_logits, -40.0, 40.0)))


def evaluate_rows(values: dict[str, Any], scores: np.ndarray, rows: np.ndarray) -> dict[str, Any]:
    current = metric_summary(
        values["labels"][rows], values["current"][rows], values["sensors"][rows]
    )
    candidate = metric_summary(
        values["labels"][rows], scores[rows], values["sensors"][rows]
    )
    return {
        "current": current,
        "candidate": candidate,
        "versus_current": comparison(candidate, current),
    }


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    selected = report["selection"]["selected"]
    lines = [
        "# Label-free temporal site-prior scene architecture",
        "",
        f"- Selected minimum site history: **{selected['min_site_size']}** scenes; top-k: **{selected['top_k']}**; logit prior weight: **{selected['weight']:.2f}**.",
        f"- Selection AP delta: **{selected['combined']['versus_current']['delta']['average_precision']:+.5f}**.",
        f"- Confirmation AP delta: **{report['confirmation']['combined']['versus_current']['delta']['average_precision']:+.5f}**.",
        f"- All promotion gates pass: **{str(report['all_promotion_gates_pass']).lower()}**.",
        "",
        "The prior uses no labels at inference: it is the mean of each physical site's top-k current scene logits, added to every scene at that site.",
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
        raise ValueError("Temporal site-prior trainer hash mismatch")
    paths: dict[str, Path] = {}
    for name, contract in protocol["inputs"].items():
        path = (ROOT / contract["path"]).resolve()
        if sha256(path) != contract["sha256"]:
            raise ValueError(f"Frozen development input hash mismatch: {name}")
        paths[name] = path
    values = load_development(
        {"inner": paths["inner"], "fold0": paths["fold0"], "fold1": paths["fold1"]},
        paths["scores"],
    )
    selection_rows = np.isin(values["folds"], protocol["folds"]["selection"])
    confirmation_rows = np.isin(values["folds"], protocol["folds"]["confirmation"])
    candidates: list[dict[str, Any]] = []
    score_by_key: dict[str, np.ndarray] = {}
    for min_site_size, top_k, weight in itertools.product(
        protocol["search"].get("min_site_size", [1]),
        protocol["search"]["top_k"],
        protocol["search"]["weights"],
    ):
            scores = temporal_site_prior(
                values["current"], values["groups"], int(top_k), float(weight), int(min_site_size)
            )
            key = f"min{min_site_size}_top{top_k}_weight{weight}"
            score_by_key[key] = scores
            combined = evaluate_rows(values, scores, selection_rows)
            per_fold = {
                str(fold): evaluate_rows(values, scores, values["folds"] == fold)
                for fold in protocol["folds"]["selection"]
            }
            bootstrap = ap_group_bootstrap(
                values["labels"][selection_rows],
                values["current"][selection_rows],
                scores[selection_rows],
                values["groups"][selection_rows],
                replicates=int(protocol["bootstrap"]["replicates"]),
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
            candidates.append(
                {
                    "key": key,
                    "min_site_size": int(min_site_size),
                    "top_k": int(top_k),
                    "weight": float(weight),
                    "combined": combined,
                    "per_fold": per_fold,
                    "paired_site_bootstrap_ap_delta": bootstrap,
                    "stable": stable,
                    "rank": [
                        int(stable),
                        bootstrap["lower"],
                        min(ap_deltas),
                        combined["versus_current"]["delta"]["average_precision"],
                    ],
                }
            )
    selected = max(candidates, key=lambda candidate: tuple(candidate["rank"]))
    selected_scores = score_by_key[selected["key"]]
    confirmation_combined = evaluate_rows(values, selected_scores, confirmation_rows)
    confirmation_per_fold = {
        str(fold): evaluate_rows(values, selected_scores, values["folds"] == fold)
        for fold in protocol["folds"]["confirmation"]
    }
    confirmation_bootstrap = ap_group_bootstrap(
        values["labels"][confirmation_rows],
        values["current"][confirmation_rows],
        selected_scores[confirmation_rows],
        values["groups"][confirmation_rows],
        replicates=int(protocol["bootstrap"]["replicates"]),
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
    }
    passed = all(checks.values())
    thresholds = [
        confirmation_per_fold[str(fold)]["candidate"]["operating_point"]["threshold"]
        for fold in protocol["folds"]["confirmation"]
    ]
    operational_threshold = max(map(float, thresholds))
    artifact_path = (ROOT / protocol["outputs"]["artifact"]).resolve()
    artifact_record = None
    if passed:
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = artifact_path.with_suffix(artifact_path.suffix + ".tmp")
        joblib.dump(
            {
                "schema_version": 1,
                "kind": "mars_label_free_temporal_site_prior",
                "top_k": selected["top_k"],
                "weight": selected["weight"],
                "min_site_size": selected["min_site_size"],
                "operational_scene_threshold": operational_threshold,
                "base_score": "frozen v3 stronger OOF ExtraTrees scene score",
                "protocol_sha256": sha256(protocol_path),
            },
            temporary,
            compress=3,
        )
        os.replace(temporary, artifact_path)
        artifact_record = {
            "path": protocol["outputs"]["artifact"],
            "bytes": artifact_path.stat().st_size,
            "sha256": sha256(artifact_path),
            "operational_scene_threshold": operational_threshold,
            "tracked": False,
        }
    report = {
        "schema_version": 1,
        "scope": "development-only temporal site-prior selection; exact paper cache not loaded",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection": {"candidates": candidates, "selected": selected},
        "confirmation": {
            "combined": confirmation_combined,
            "per_fold": confirmation_per_fold,
            "paired_site_bootstrap_ap_delta": confirmation_bootstrap,
        },
        "promotion_checks": checks,
        "all_promotion_gates_pass": passed,
        "operational_scene_threshold": operational_threshold,
        "artifact": artifact_record,
        "decision": (
            "Freeze the temporal site prior for fresh safety evaluation."
            if passed
            else "Reject the temporal site prior before fresh or paper evaluation."
        ),
        "provenance": {
            "protocol_sha256": sha256(protocol_path),
            "script_sha256": sha256(Path(__file__).resolve()),
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
