#!/usr/bin/env python3
"""Train a set-context scene head for unseen low-prevalence MARS sites."""

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
import scipy
import sklearn
import xgboost
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "EarthRemoteSensingRapidResponse"
for path in (MODEL_ROOT, ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from acquire_mars_metadata import sha256  # noqa: E402
from train_mars_context_scene_ranker import leave_one_out_max  # noqa: E402
from train_mars_crossfold_bagged_scene_head import load_development  # noqa: E402
from train_mars_oof_scene_ensemble_v2 import ap_group_bootstrap  # noqa: E402
from train_mars_scene_ranker import (  # noqa: E402
    blend_scores,
    comparison,
    metric_summary,
    site_cell_weights,
)
from train_mars_unseen_low_prevalence_router import low_prevalence_mask  # noqa: E402
from train_mars_xgboost_scene_head import build_model  # noqa: E402


DEFAULT_PROTOCOL = Path("configs/mars_low_prevalence_set_context_head_protocol.json")


def logit(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=np.float64), 1e-7, 1.0 - 1e-7)
    return np.log(clipped) - np.log1p(-clipped)


def build_current_site_context(
    current: np.ndarray, groups: np.ndarray
) -> tuple[np.ndarray, list[str]]:
    current = np.asarray(current, dtype=np.float64)
    groups = np.asarray(groups).astype(str)
    logits = logit(current)
    context = np.empty((current.size, 9), dtype=np.float64)
    for group in np.unique(groups):
        rows = np.flatnonzero(groups == group)
        local = logits[rows]
        loo = leave_one_out_max(local[:, None]).ravel()
        context[rows] = np.column_stack(
            [
                current[rows],
                local,
                np.full(rows.size, np.mean(local)),
                np.full(rows.size, np.std(local)),
                np.full(rows.size, np.max(local)),
                np.full(rows.size, np.quantile(local, 0.9)),
                loo,
                rankdata(local, method="average") / rows.size,
                np.full(rows.size, np.log1p(rows.size)),
            ]
        )
    names = [
        "set_current_probability",
        "set_current_logit",
        "set_current_logit_mean",
        "set_current_logit_std",
        "set_current_logit_max",
        "set_current_logit_q90",
        "set_current_logit_leave_one_out_max",
        "set_current_within_site_rank",
        "set_log1p_site_size",
    ]
    if not np.isfinite(context).all():
        raise RuntimeError("Set-context features contain non-finite values")
    return context, names


def prepare_values(values: dict[str, Any]) -> dict[str, Any]:
    context, names = build_current_site_context(values["current"], values["groups"])
    output = dict(values)
    output["features"] = np.concatenate([values["features"], context], axis=1)
    output["model_feature_names"] = [*values["augmented_feature_names"], *names]
    return output


def evaluate_rows(
    values: dict[str, Any], scores: np.ndarray, rows: np.ndarray
) -> dict[str, Any]:
    current = metric_summary(values["labels"][rows], values["current"][rows], values["sensors"][rows])
    candidate = metric_summary(values["labels"][rows], scores[rows], values["sensors"][rows])
    return {"current": current, "candidate": candidate, "versus_current": comparison(candidate, current)}


def fit_model(
    spec: dict[str, Any], features: np.ndarray, labels: np.ndarray,
    groups: np.ndarray, sensors: np.ndarray, seed: int,
) -> Any:
    model = build_model(spec, seed=seed)
    weights = site_cell_weights(groups, labels, sensors)
    model.fit(features, labels, sample_weight=weights, verbose=False)
    return model


def crossfit_raw(
    values: dict[str, Any], low_rows: np.ndarray, folds: list[int],
    spec: dict[str, Any], seed: int,
) -> np.ndarray:
    raw = np.full(values["labels"].size, np.nan, dtype=np.float64)
    for holdout in folds:
        fit = np.isin(values["folds"], folds) & (values["folds"] != holdout) & low_rows
        held = values["folds"] == holdout
        model = fit_model(
            spec, values["features"][fit], values["labels"][fit],
            values["groups"][fit], values["sensors"][fit], seed + holdout,
        )
        raw[held] = model.predict_proba(values["features"][held])[:, 1]
    selected = np.isin(values["folds"], folds)
    if not np.isfinite(raw[selected]).all():
        raise RuntimeError("Cross-fitted set-context scores are incomplete")
    return raw


def evaluate_candidate(
    values: dict[str, Any], raw: np.ndarray, blend: float,
    low_rows: np.ndarray, folds: list[int],
) -> tuple[np.ndarray, dict[str, Any]]:
    selected = np.isin(values["folds"], folds)
    scores = values["current"].copy()
    scores[selected] = blend_scores(values["current"][selected], raw[selected], blend)
    low_selected = selected & low_rows
    low_combined = evaluate_rows(values, scores, low_selected)
    whole_combined = evaluate_rows(values, scores, selected)
    low_per_fold = {
        str(fold): evaluate_rows(values, scores, (values["folds"] == fold) & low_rows)
        for fold in folds
    }
    low_ap = [value["versus_current"]["delta"]["average_precision"] for value in low_per_fold.values()]
    low_recall = [value["versus_current"]["delta"]["recall_at_fpr_0_0713"] for value in low_per_fold.values()]
    result = {
        "blend": float(blend),
        "low_prevalence_combined": low_combined,
        "whole_combined": whole_combined,
        "low_prevalence_per_fold": low_per_fold,
        "rank": [
            min(low_ap),
            low_combined["versus_current"]["delta"]["average_precision"],
            min(low_recall),
            low_combined["versus_current"]["delta"]["recall_at_fpr_0_0713"],
            -float(blend),
        ],
    }
    return scores, result


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    selected = report["selection"]["selected"]
    confirmation = report["confirmation"]["low_prevalence_combined"]["versus_current"]["delta"]
    lines = [
        "# Low-prevalence set-context scene head",
        "",
        f"- Selected model: `{selected['spec']['name']}`; blend: **{selected['blend']:.2f}**.",
        f"- Selection low-prevalence AP delta: **{selected['low_prevalence_combined']['versus_current']['delta']['average_precision']:+.5f}**.",
        f"- Confirmation low-prevalence AP delta: **{confirmation['average_precision']:+.5f}**; recall delta: **{confirmation['recall_at_fpr_0_0713']:+.5f}**.",
        f"- All promotion gates pass: **{str(report['all_promotion_gates_pass']).lower()}**.",
        "",
        "The deployed feature transform is permutation-equivariant within physical sites and uses no inference-time labels.",
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
        raise ValueError("Set-context trainer hash mismatch")
    paths: dict[str, Path] = {}
    for name, contract in protocol["inputs"].items():
        path = (ROOT / contract["path"]).resolve()
        if sha256(path) != contract["sha256"]:
            raise ValueError(f"Frozen set-context input hash mismatch: {name}")
        paths[name] = path
    values = prepare_values(load_development(
        {"inner": paths["inner"], "fold0": paths["fold0"], "fold1": paths["fold1"]},
        paths["scores"],
    ))
    maximum_rate = float(protocol["target_domain"]["maximum_site_positive_rate"])
    low_rows = low_prevalence_mask(values["labels"], values["groups"], maximum_rate)
    selection_folds = list(map(int, protocol["folds"]["selection"]))
    confirmation_folds = list(map(int, protocol["folds"]["confirmation"]))
    candidates: list[dict[str, Any]] = []
    scores_by_key: dict[str, np.ndarray] = {}
    for spec_index, spec in enumerate(protocol["search"]["models"]):
        raw = crossfit_raw(
            values, low_rows, selection_folds, spec,
            int(protocol["seeds"]["selection_model"]) + 100 * spec_index,
        )
        for blend in protocol["search"]["blends"]:
            scores, candidate = evaluate_candidate(
                values, raw, float(blend), low_rows, selection_folds
            )
            candidate["spec"] = spec
            candidate["key"] = f"{spec['name']}_blend{blend}"
            candidates.append(candidate)
            scores_by_key[candidate["key"]] = scores
        print(json.dumps({"completed_selection_model": spec["name"]}), flush=True)
    selected = max(candidates, key=lambda value: tuple(value["rank"]))
    selected_scores = scores_by_key[selected["key"]]
    selection_rows = np.isin(values["folds"], selection_folds) & low_rows
    selection_bootstrap = ap_group_bootstrap(
        values["labels"][selection_rows], values["current"][selection_rows],
        selected_scores[selection_rows], values["groups"][selection_rows],
        replicates=int(protocol["bootstrap"]["replicates"]),
        seed=int(protocol["bootstrap"]["selection_seed"]),
    )
    selection_low_ap = [
        value["versus_current"]["delta"]["average_precision"]
        for value in selected["low_prevalence_per_fold"].values()
    ]
    selection_low_recall = [
        value["versus_current"]["delta"]["recall_at_fpr_0_0713"]
        for value in selected["low_prevalence_per_fold"].values()
    ]
    selection_whole = selected["whole_combined"]["versus_current"]["delta"]
    gates = protocol["gates"]
    selection_checks = {
        "low_ap_point_higher": selected["low_prevalence_combined"]["versus_current"]["delta"]["average_precision"] > 0.0,
        "low_recall_point_no_lower": selected["low_prevalence_combined"]["versus_current"]["delta"]["recall_at_fpr_0_0713"] >= 0.0,
        "low_ap_bootstrap_lower_positive": selection_bootstrap["lower"] > 0.0,
        "each_low_fold_ap_within_tolerance": min(selection_low_ap) >= -float(gates["per_fold_low_ap_tolerance"]),
        "each_low_fold_recall_within_tolerance": min(selection_low_recall) >= -float(gates["per_fold_low_recall_tolerance"]),
        "whole_ap_noninferior": selection_whole["average_precision"] >= -float(gates["whole_fold_ap_noninferiority"]),
        "whole_recall_noninferior": selection_whole["recall_at_fpr_0_0713"] >= -float(gates["whole_fold_recall_noninferiority"]),
    }

    fit = np.isin(values["folds"], selection_folds) & low_rows
    confirmation_rows = np.isin(values["folds"], confirmation_folds)
    model = fit_model(
        selected["spec"], values["features"][fit], values["labels"][fit],
        values["groups"][fit], values["sensors"][fit], int(protocol["seeds"]["confirmation_model"]),
    )
    raw_confirmation = model.predict_proba(values["features"][confirmation_rows])[:, 1]
    confirmation_scores = values["current"].copy()
    confirmation_scores[confirmation_rows] = blend_scores(
        values["current"][confirmation_rows], raw_confirmation, float(selected["blend"])
    )
    confirmation_low_rows = confirmation_rows & low_rows
    confirmation_low = evaluate_rows(values, confirmation_scores, confirmation_low_rows)
    confirmation_whole = evaluate_rows(values, confirmation_scores, confirmation_rows)
    confirmation_per_fold = {
        str(fold): evaluate_rows(values, confirmation_scores, (values["folds"] == fold) & low_rows)
        for fold in confirmation_folds
    }
    confirmation_bootstrap = ap_group_bootstrap(
        values["labels"][confirmation_low_rows], values["current"][confirmation_low_rows],
        confirmation_scores[confirmation_low_rows], values["groups"][confirmation_low_rows],
        replicates=int(protocol["bootstrap"]["replicates"]),
        seed=int(protocol["bootstrap"]["confirmation_seed"]),
    )
    confirmation_low_ap = [v["versus_current"]["delta"]["average_precision"] for v in confirmation_per_fold.values()]
    confirmation_low_recall = [v["versus_current"]["delta"]["recall_at_fpr_0_0713"] for v in confirmation_per_fold.values()]
    confirmation_delta = confirmation_low["versus_current"]["delta"]
    confirmation_whole_delta = confirmation_whole["versus_current"]["delta"]
    confirmation_checks = {
        "low_ap_point_higher": confirmation_delta["average_precision"] > 0.0,
        "low_recall_point_no_lower": confirmation_delta["recall_at_fpr_0_0713"] >= 0.0,
        "low_ap_bootstrap_lower_positive": confirmation_bootstrap["lower"] > 0.0,
        "each_low_fold_ap_within_tolerance": min(confirmation_low_ap) >= -float(gates["per_fold_low_ap_tolerance"]),
        "each_low_fold_recall_within_tolerance": min(confirmation_low_recall) >= -float(gates["per_fold_low_recall_tolerance"]),
        "whole_ap_noninferior": confirmation_whole_delta["average_precision"] >= -float(gates["whole_fold_ap_noninferiority"]),
        "whole_recall_noninferior": confirmation_whole_delta["recall_at_fpr_0_0713"] >= -float(gates["whole_fold_recall_noninferiority"]),
    }
    passed = all(selection_checks.values()) and all(confirmation_checks.values())
    artifact_record = None
    if passed:
        final_fit = low_rows
        final_model = fit_model(
            selected["spec"], values["features"][final_fit], values["labels"][final_fit],
            values["groups"][final_fit], values["sensors"][final_fit], int(protocol["seeds"]["final_model"]),
        )
        artifact_path = (ROOT / protocol["outputs"]["artifact"]).resolve()
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = artifact_path.with_suffix(artifact_path.suffix + ".tmp")
        joblib.dump({
            "schema_version": 1,
            "kind": "mars_low_prevalence_set_context_head",
            "model": final_model,
            "spec": selected["spec"],
            "blend": selected["blend"],
            "feature_names": values["model_feature_names"],
            "known_training_sites": sorted(set(values["groups"].tolist())),
            "maximum_training_site_positive_rate": maximum_rate,
            "operational_scene_threshold": float(protocol["base_architecture"]["operational_scene_threshold"]),
            "base_score": "frozen v3 stronger OOF ExtraTrees scene score",
            "protocol_sha256": sha256(protocol_path),
        }, temporary, compress=3)
        os.replace(temporary, artifact_path)
        artifact_record = {
            "path": protocol["outputs"]["artifact"], "bytes": artifact_path.stat().st_size,
            "sha256": sha256(artifact_path), "tracked": False,
            "known_training_sites": len(set(values["groups"].tolist())),
        }
    report = {
        "schema_version": 1,
        "scope": "development-only low-prevalence set-context head; exact paper cache not loaded",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "feature_count": int(values["features"].shape[1]),
        "target_domain": {
            "maximum_site_positive_rate": maximum_rate,
            "selection_rows": int(selection_rows.sum()), "selection_positive": int(values["labels"][selection_rows].sum()),
            "confirmation_rows": int(confirmation_low_rows.sum()), "confirmation_positive": int(values["labels"][confirmation_low_rows].sum()),
        },
        "selection": {
            "candidates": candidates, "selected": selected,
            "paired_site_bootstrap_low_ap_delta": selection_bootstrap, "checks": selection_checks,
        },
        "confirmation": {
            "low_prevalence_combined": confirmation_low, "whole_combined": confirmation_whole,
            "low_prevalence_per_fold": confirmation_per_fold,
            "paired_site_bootstrap_low_ap_delta": confirmation_bootstrap, "checks": confirmation_checks,
        },
        "all_promotion_gates_pass": passed,
        "artifact": artifact_record,
        "decision": "Freeze the set-context head for fresh safety evaluation." if passed else "Reject the set-context head before fresh or paper evaluation.",
        "provenance": {
            "protocol_sha256": sha256(protocol_path), "script_sha256": sha256(Path(__file__).resolve()),
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "numpy": np.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__,
            "xgboost": xgboost.__version__, "joblib": joblib.__version__,
        },
    }
    output_json = (ROOT / protocol["outputs"]["json"]).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown((ROOT / protocol["outputs"]["markdown"]).resolve(), report)
    print(json.dumps({"ok": passed, "selected": selected["key"], "selection": selection_checks, "confirmation": confirmation_checks, "artifact": artifact_record}, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
