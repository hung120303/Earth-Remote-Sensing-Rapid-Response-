#!/usr/bin/env python3
"""Train a cross-site risk prior from label-free temporal scene-score aggregates."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "EarthRemoteSensingRapidResponse"
for path in (MODEL_ROOT, ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from acquire_mars_metadata import sha256  # noqa: E402
from train_mars_crossfold_bagged_scene_head import load_development  # noqa: E402
from train_mars_oof_scene_ensemble_v2 import ap_group_bootstrap  # noqa: E402
from train_mars_scene_ranker import comparison, metric_summary  # noqa: E402


DEFAULT_PROTOCOL = Path("configs/mars_site_risk_prior_protocol.json")


def logit(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=np.float64), 1e-8, 1.0 - 1e-8)
    return np.log(clipped) - np.log1p(-clipped)


def aggregate_score_features(values: np.ndarray, prefix: str) -> tuple[list[float], list[str]]:
    raw = np.asarray(values, dtype=np.float64)
    logits = logit(raw)
    ordered = np.sort(logits)
    features = [
        float(np.max(logits)),
        float(np.mean(ordered[-min(2, ordered.size) :])),
        float(np.mean(ordered[-min(3, ordered.size) :])),
        float(np.mean(ordered[-min(5, ordered.size) :])),
        float(np.quantile(logits, 0.95)),
        float(np.quantile(logits, 0.90)),
        float(np.quantile(logits, 0.75)),
        float(np.median(logits)),
        float(np.mean(logits)),
        float(np.std(logits)),
        float(np.min(logits)),
        *[float(np.mean(raw >= threshold)) for threshold in (0.01, 0.03, 0.1, 0.3)],
    ]
    names = [
        f"{prefix}_{name}"
        for name in (
            "logit_max", "logit_top2_mean", "logit_top3_mean", "logit_top5_mean",
            "logit_q95", "logit_q90", "logit_q75", "logit_median", "logit_mean",
            "logit_std", "logit_min", "fraction_ge_001", "fraction_ge_003",
            "fraction_ge_01", "fraction_ge_03",
        )
    ]
    return features, names


def build_site_table(
    current: np.ndarray,
    primary: np.ndarray,
    sensors: np.ndarray,
    groups: np.ndarray,
    *,
    labels: np.ndarray | None = None,
    folds: np.ndarray | None = None,
) -> dict[str, Any]:
    groups = np.asarray(groups).astype(str)
    group_names, inverse = np.unique(groups, return_inverse=True)
    rows: list[list[float]] = []
    site_labels: list[int] = []
    site_folds: list[int] = []
    feature_names: list[str] | None = None
    for site_index, group in enumerate(group_names):
        selected = inverse == site_index
        current_values, current_names = aggregate_score_features(current[selected], "current")
        primary_values, primary_names = aggregate_score_features(primary[selected], "primary")
        local_names = [*current_names, *primary_names, "log1p_scene_count", "sentinel2_fraction"]
        if feature_names is None:
            feature_names = local_names
        elif feature_names != local_names:
            raise RuntimeError("Site feature schema changed")
        rows.append(
            [
                *current_values,
                *primary_values,
                float(np.log1p(np.count_nonzero(selected))),
                float(np.mean(np.asarray(sensors)[selected] == 0)),
            ]
        )
        if labels is not None:
            site_labels.append(int(np.any(np.asarray(labels)[selected] == 1)))
        if folds is not None:
            local_folds = np.unique(np.asarray(folds)[selected])
            if local_folds.size != 1:
                raise ValueError(f"Physical site spans development folds: {group}")
            site_folds.append(int(local_folds[0]))
    assert feature_names is not None
    return {
        "groups": group_names,
        "inverse": inverse,
        "features": np.asarray(rows, dtype=np.float64),
        "feature_names": feature_names,
        "labels": None if labels is None else np.asarray(site_labels, dtype=np.uint8),
        "folds": None if folds is None else np.asarray(site_folds, dtype=np.uint8),
    }


def build_model(spec: dict[str, Any], seed: int) -> Any:
    if spec["kind"] == "logistic":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=float(spec["C"]),
                class_weight="balanced",
                max_iter=2000,
                random_state=seed,
            ),
        )
    if spec["kind"] == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=500,
            max_depth=int(spec["max_depth"]),
            min_samples_leaf=int(spec["min_samples_leaf"]),
            max_features=float(spec["max_features"]),
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        )
    raise ValueError(f"Unknown site model kind: {spec['kind']}")


def site_prior_scores(
    current: np.ndarray, site_risk: np.ndarray, inverse: np.ndarray, weight: float
) -> np.ndarray:
    risk = np.clip(np.asarray(site_risk, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    candidate_logits = logit(current) + float(weight) * logit(risk)[inverse]
    return 1.0 / (1.0 + np.exp(-np.clip(candidate_logits, -40.0, 40.0)))


def evaluate_rows(values: dict[str, Any], scores: np.ndarray, rows: np.ndarray) -> dict[str, Any]:
    current = metric_summary(values["labels"][rows], values["current"][rows], values["sensors"][rows])
    candidate = metric_summary(values["labels"][rows], scores[rows], values["sensors"][rows])
    return {"current": current, "candidate": candidate, "versus_current": comparison(candidate, current)}


def crossfit_site_risk(
    table: dict[str, Any], selection_folds: list[int], spec: dict[str, Any], seed: int
) -> np.ndarray:
    risk = np.full(table["groups"].size, np.nan, dtype=np.float64)
    for holdout in selection_folds:
        fit = np.isin(table["folds"], selection_folds) & (table["folds"] != holdout)
        held = table["folds"] == holdout
        model = build_model(spec, seed + holdout)
        model.fit(table["features"][fit], table["labels"][fit])
        risk[held] = model.predict_proba(table["features"][held])[:, 1]
    if not np.isfinite(risk[np.isin(table["folds"], selection_folds)]).all():
        raise RuntimeError("Cross-fitted site risks are incomplete")
    return risk


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    selected = report["selection"]["selected"]
    lines = [
        "# Cross-site temporal risk prior",
        "",
        f"- Selected site model: `{selected['spec']['name']}`; scene-logit weight: **{selected['weight']:.2f}**.",
        f"- Selection AP delta: **{selected['combined']['versus_current']['delta']['average_precision']:+.5f}**.",
        f"- Confirmation AP delta: **{report['confirmation']['combined']['versus_current']['delta']['average_precision']:+.5f}**.",
        f"- All promotion gates pass: **{str(report['all_promotion_gates_pass']).lower()}**.",
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
        raise ValueError("Site-risk trainer hash mismatch")
    paths: dict[str, Path] = {}
    for name, contract in protocol["inputs"].items():
        path = (ROOT / contract["path"]).resolve()
        if sha256(path) != contract["sha256"]:
            raise ValueError(f"Frozen site-risk input hash mismatch: {name}")
        paths[name] = path
    values = load_development(
        {"inner": paths["inner"], "fold0": paths["fold0"], "fold1": paths["fold1"]},
        paths["scores"],
    )
    table = build_site_table(
        values["current"], values["primary"], values["sensors"], values["groups"],
        labels=values["labels"], folds=values["folds"],
    )
    selection_folds = list(map(int, protocol["folds"]["selection"]))
    confirmation_folds = list(map(int, protocol["folds"]["confirmation"]))
    selection_rows = np.isin(values["folds"], selection_folds)
    confirmation_rows = np.isin(values["folds"], confirmation_folds)
    candidates: list[dict[str, Any]] = []
    score_by_key: dict[str, np.ndarray] = {}
    for spec_index, spec in enumerate(protocol["search"]["models"]):
        site_risk = crossfit_site_risk(table, selection_folds, spec, int(protocol["seeds"]["model"]) + 100 * spec_index)
        for weight in protocol["search"]["weights"]:
            scores = site_prior_scores(values["current"], site_risk, table["inverse"], float(weight))
            key = f"{spec['name']}_weight{weight}"
            score_by_key[key] = scores
            combined = evaluate_rows(values, scores, selection_rows)
            per_fold = {str(fold): evaluate_rows(values, scores, values["folds"] == fold) for fold in selection_folds}
            bootstrap = ap_group_bootstrap(
                values["labels"][selection_rows], values["current"][selection_rows], scores[selection_rows],
                values["groups"][selection_rows], replicates=int(protocol["bootstrap"]["replicates"]),
                seed=int(protocol["seeds"]["selection_bootstrap"]) + len(candidates),
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
                "key": key, "spec": spec, "weight": float(weight), "combined": combined,
                "per_fold": per_fold, "paired_site_bootstrap_ap_delta": bootstrap, "stable": stable,
                "rank": [int(stable), bootstrap["lower"], min(ap_deltas), combined["versus_current"]["delta"]["average_precision"]],
            })
    selected = max(candidates, key=lambda candidate: tuple(candidate["rank"]))
    fit_sites = np.isin(table["folds"], selection_folds)
    selection_model = build_model(selected["spec"], int(protocol["seeds"]["confirmation_model"]))
    selection_model.fit(table["features"][fit_sites], table["labels"][fit_sites])
    confirmation_sites = np.isin(table["folds"], confirmation_folds)
    confirmation_risk = np.full(table["groups"].size, 0.5, dtype=np.float64)
    confirmation_risk[confirmation_sites] = selection_model.predict_proba(table["features"][confirmation_sites])[:, 1]
    confirmation_scores = site_prior_scores(values["current"], confirmation_risk, table["inverse"], selected["weight"])
    confirmation_combined = evaluate_rows(values, confirmation_scores, confirmation_rows)
    confirmation_per_fold = {str(fold): evaluate_rows(values, confirmation_scores, values["folds"] == fold) for fold in confirmation_folds}
    confirmation_bootstrap = ap_group_bootstrap(
        values["labels"][confirmation_rows], values["current"][confirmation_rows], confirmation_scores[confirmation_rows],
        values["groups"][confirmation_rows], replicates=int(protocol["bootstrap"]["replicates"]),
        seed=int(protocol["seeds"]["confirmation_bootstrap"]),
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
    thresholds = [confirmation_per_fold[str(fold)]["candidate"]["operating_point"]["threshold"] for fold in confirmation_folds]
    operational_threshold = max(map(float, thresholds))
    artifact_record = None
    if passed:
        final_model = build_model(selected["spec"], int(protocol["seeds"]["final_model"]))
        final_model.fit(table["features"], table["labels"])
        artifact_path = (ROOT / protocol["outputs"]["artifact"]).resolve()
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = artifact_path.with_suffix(artifact_path.suffix + ".tmp")
        joblib.dump({
            "schema_version": 1,
            "kind": "mars_cross_site_risk_prior",
            "model": final_model,
            "spec": selected["spec"],
            "weight": selected["weight"],
            "site_feature_names": table["feature_names"],
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
        "scope": "development-only cross-site risk prior; exact paper cache not loaded",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "site_summary": {
            "sites": int(table["groups"].size),
            "positive_sites": int(table["labels"].sum()),
            "features": len(table["feature_names"]),
        },
        "selection": {"candidates": candidates, "selected": selected},
        "confirmation": {"combined": confirmation_combined, "per_fold": confirmation_per_fold, "paired_site_bootstrap_ap_delta": confirmation_bootstrap},
        "promotion_checks": checks,
        "all_promotion_gates_pass": passed,
        "operational_scene_threshold": operational_threshold,
        "artifact": artifact_record,
        "decision": "Freeze the cross-site risk prior for fresh safety evaluation." if passed else "Reject the cross-site risk prior before fresh or paper evaluation.",
        "provenance": {
            "protocol_sha256": sha256(protocol_path), "script_sha256": sha256(Path(__file__).resolve()),
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "sklearn": sklearn.__version__, "numpy": np.__version__,
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
