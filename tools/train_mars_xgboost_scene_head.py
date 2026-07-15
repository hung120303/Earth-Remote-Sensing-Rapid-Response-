#!/usr/bin/env python3
"""Cross-fit regularized XGBoost scene heads on the five MARS development folds."""

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
import xgboost
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "EarthRemoteSensingRapidResponse"
for path in (MODEL_ROOT, ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from acquire_mars_metadata import repo_root, sha256  # noqa: E402
from train_mars_crossfold_bagged_scene_head import (  # noqa: E402
    FOLDS,
    DEFAULT_FOLD0_CACHE,
    DEFAULT_FOLD0_SHA256,
    DEFAULT_FOLD1_CACHE,
    DEFAULT_FOLD1_SHA256,
    DEFAULT_INNER_CACHE,
    DEFAULT_INNER_SHA256,
    DEFAULT_SCORE_CACHE,
    DEFAULT_SCORE_SHA256,
    load_development,
)
from train_mars_oof_scene_ensemble_v2 import ap_group_bootstrap  # noqa: E402
from train_mars_scene_ranker import blend_scores, comparison, metric_summary  # noqa: E402

DEFAULT_ARTIFACT = Path(
    "EarthRemoteSensingRapidResponse/artifacts/mars_xgboost_scene_head.joblib"
)
DEFAULT_JSON = Path("reports/experiments/mars_xgboost_scene_head.json")
DEFAULT_MARKDOWN = Path("reports/experiments/MARS_XGBOOST_SCENE_HEAD.md")

# Fixed before execution. These span conservative depth/learning-rate tradeoffs
# while keeping strong row/feature subsampling and regularization throughout.
MODEL_SPECS: tuple[dict[str, Any], ...] = (
    {
        "name": "depth3_lr004",
        "n_estimators": 600,
        "max_depth": 3,
        "learning_rate": 0.04,
        "min_child_weight": 10.0,
    },
    {
        "name": "depth4_lr003",
        "n_estimators": 800,
        "max_depth": 4,
        "learning_rate": 0.03,
        "min_child_weight": 10.0,
    },
    {
        "name": "depth5_lr0025",
        "n_estimators": 1000,
        "max_depth": 5,
        "learning_rate": 0.025,
        "min_child_weight": 20.0,
    },
)
BLENDS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.625, 0.75, 0.875, 1.0)
MINIMUM_POOLED_AP_GAIN = 0.002
SEED = 20261400


def build_model(spec: dict[str, Any], *, seed: int = SEED) -> XGBClassifier:
    """Construct a fixed, regularized CPU-histogram XGBoost classifier."""
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        device="cpu",
        n_estimators=int(spec["n_estimators"]),
        max_depth=int(spec["max_depth"]),
        learning_rate=float(spec["learning_rate"]),
        min_child_weight=float(spec["min_child_weight"]),
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=0.1,
        reg_lambda=10.0,
        gamma=0.0,
        max_bin=256,
        random_state=seed,
        n_jobs=12,
        verbosity=0,
    )


def oof_scores(values: dict[str, Any], spec: dict[str, Any]) -> np.ndarray:
    scores = np.empty(values["labels"].shape, dtype=np.float64)
    for holdout in FOLDS:
        fit = values["folds"] != holdout
        held = values["folds"] == holdout
        model = build_model(spec, seed=SEED + holdout)
        # No held labels, held metrics, or early-stopping callback enter fitting.
        model.fit(values["features"][fit], values["labels"][fit], verbose=False)
        scores[held] = model.predict_proba(values["features"][held])[:, 1]
        print(
            json.dumps(
                {
                    "model_spec": spec["name"],
                    "completed_holdout": holdout,
                    "training_rows": int(fit.sum()),
                    "held_rows": int(held.sum()),
                }
            ),
            flush=True,
        )
    if not np.isfinite(scores).all():
        raise RuntimeError("XGBoost OOF scores contain non-finite values")
    return scores


def evaluate_candidate(
    values: dict[str, Any], raw: np.ndarray, spec: dict[str, Any], blend: float
) -> dict[str, Any]:
    scores = blend_scores(values["current"], raw, blend)
    candidate = metric_summary(values["labels"], scores, values["sensors"])
    current = metric_summary(values["labels"], values["current"], values["sensors"])
    primary = metric_summary(values["labels"], values["primary"], values["sensors"])
    versus_current = comparison(candidate, current)
    versus_primary = comparison(candidate, primary)
    per_fold = {}
    for fold in FOLDS:
        rows = values["folds"] == fold
        local = metric_summary(values["labels"][rows], scores[rows], values["sensors"][rows])
        per_fold[str(fold)] = {
            "versus_current": comparison(
                local,
                metric_summary(
                    values["labels"][rows],
                    values["current"][rows],
                    values["sensors"][rows],
                ),
            ),
            "versus_primary": comparison(
                local,
                metric_summary(
                    values["labels"][rows],
                    values["primary"][rows],
                    values["sensors"][rows],
                ),
            ),
        }
    fold_ap = [
        value["versus_current"]["delta"]["average_precision"]
        for value in per_fold.values()
    ]
    fold_recall = [
        value["versus_current"]["delta"]["recall_at_fpr_0_0713"]
        for value in per_fold.values()
    ]
    stable = (
        versus_current["delta"]["average_precision"] >= MINIMUM_POOLED_AP_GAIN
        and versus_current["delta"]["recall_at_fpr_0_0713"] > 0.0
        and min(fold_ap) > 0.0
        and min(fold_recall) >= 0.0
        and min(versus_current["delta"]["sensor_average_precision"].values()) >= 0.0
        and versus_primary["delta"]["average_precision"] > 0.0
        and versus_primary["delta"]["recall_at_fpr_0_0713"] > 0.0
    )
    return {
        "model_spec": spec,
        "model_spec_name": spec["name"],
        "xgboost_blend": blend,
        "stable": bool(stable),
        "versus_current": versus_current,
        "versus_primary": versus_primary,
        "per_fold": per_fold,
        "rank": [
            int(stable),
            min(fold_recall),
            min(fold_ap),
            versus_current["delta"]["average_precision"],
            -blend,
        ],
    }


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    selected = report["selected"]
    delta = selected["versus_current"]["delta"]
    interval = selected["paired_group_bootstrap_ap_delta_vs_current"]
    lines = [
        "# Regularized XGBoost MARS scene head",
        "",
        "Every prediction is cross-fitted by complete physical-site fold; no held-label early stopping is used.",
        "",
        f"- Model specification: `{selected['model_spec_name']}`",
        f"- Current/XGBoost logit blend: {selected['xgboost_blend']:.3f}",
        f"- AP delta vs current: {delta['average_precision']:+.5f}",
        f"- Recall delta vs current: {delta['recall_at_fpr_0_0713']:+.5f}",
        f"- Paired-site AP interval vs current: [{interval['lower']:+.5f}, {interval['upper']:+.5f}]",
        "",
        "| Fold | AP delta vs current | Recall delta vs current |",
        "|---|---:|---:|",
    ]
    for fold, value in selected["per_fold"].items():
        local = value["versus_current"]["delta"]
        lines.append(
            f"| {fold} | {local['average_precision']:+.5f} | "
            f"{local['recall_at_fpr_0_0713']:+.5f} |"
        )
    lines.extend(["", report["decision"]])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inner-cache", default=DEFAULT_INNER_CACHE.as_posix())
    parser.add_argument("--inner-sha256", default=DEFAULT_INNER_SHA256)
    parser.add_argument("--fold0-cache", default=DEFAULT_FOLD0_CACHE.as_posix())
    parser.add_argument("--fold0-sha256", default=DEFAULT_FOLD0_SHA256)
    parser.add_argument("--fold1-cache", default=DEFAULT_FOLD1_CACHE.as_posix())
    parser.add_argument("--fold1-sha256", default=DEFAULT_FOLD1_SHA256)
    parser.add_argument("--score-cache", default=DEFAULT_SCORE_CACHE.as_posix())
    parser.add_argument("--score-sha256", default=DEFAULT_SCORE_SHA256)
    parser.add_argument("--artifact", default=DEFAULT_ARTIFACT.as_posix())
    parser.add_argument("--output-json", default=DEFAULT_JSON.as_posix())
    parser.add_argument("--output-markdown", default=DEFAULT_MARKDOWN.as_posix())
    args = parser.parse_args()
    root = repo_root()
    paths = {
        "inner": (root / args.inner_cache).resolve(),
        "fold0": (root / args.fold0_cache).resolve(),
        "fold1": (root / args.fold1_cache).resolve(),
        "score": (root / args.score_cache).resolve(),
    }
    expected = {
        "inner": args.inner_sha256,
        "fold0": args.fold0_sha256,
        "fold1": args.fold1_sha256,
        "score": args.score_sha256,
    }
    for name, digest in expected.items():
        if sha256(paths[name]) != digest:
            raise ValueError(f"Frozen {name} cache hash mismatch")
    values = load_development(
        {name: paths[name] for name in ("inner", "fold0", "fold1")}, paths["score"]
    )

    raw_store = {}
    candidates = []
    for spec in MODEL_SPECS:
        raw = oof_scores(values, spec)
        raw_store[spec["name"]] = raw
        candidates.extend(
            evaluate_candidate(values, raw, spec, blend) for blend in BLENDS
        )
    selected = max(candidates, key=lambda value: tuple(value["rank"]))
    selected_scores = blend_scores(
        values["current"],
        raw_store[selected["model_spec_name"]],
        float(selected["xgboost_blend"]),
    )
    selected["paired_group_bootstrap_ap_delta_vs_primary"] = ap_group_bootstrap(
        values["labels"], values["primary"], selected_scores, values["groups"],
        replicates=10_000, seed=20261420,
    )
    selected["paired_group_bootstrap_ap_delta_vs_current"] = ap_group_bootstrap(
        values["labels"], values["current"], selected_scores, values["groups"],
        replicates=10_000, seed=20261421,
    )
    passed = bool(
        selected["stable"]
        and selected["paired_group_bootstrap_ap_delta_vs_primary"]["lower"] > 0.0
        and selected["paired_group_bootstrap_ap_delta_vs_current"]["lower"] > 0.0
    )

    artifact_path = (root / args.artifact).resolve()
    artifact_digest = None
    if passed:
        final_model = build_model(selected["model_spec"], seed=SEED + 100)
        final_model.fit(values["features"], values["labels"], verbose=False)
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = artifact_path.with_suffix(artifact_path.suffix + ".tmp")
        joblib.dump(
            {
                "schema_version": 1,
                "kind": "mars_regularized_xgboost_scene_head",
                "model": final_model,
                "model_spec": selected["model_spec"],
                "xgboost_blend": float(selected["xgboost_blend"]),
                "base_score": "frozen v3 stronger scene head",
                "feature_names": values["augmented_feature_names"],
            },
            temporary,
            compress=3,
        )
        os.replace(temporary, artifact_path)
        artifact_digest = sha256(artifact_path)

    report = {
        "schema_version": 1,
        "scope": "five-fold development-only XGBoost scene-head selection; paper cache not loaded",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_specs": list(MODEL_SPECS),
        "blends": list(BLENDS),
        "minimum_pooled_ap_gain": MINIMUM_POOLED_AP_GAIN,
        "candidate_summaries": [
            {
                "model_spec_name": value["model_spec_name"],
                "xgboost_blend": value["xgboost_blend"],
                "stable": value["stable"],
                "ap_delta_vs_current": value["versus_current"]["delta"]["average_precision"],
                "recall_delta_vs_current": value["versus_current"]["delta"]["recall_at_fpr_0_0713"],
                "worst_fold_ap_delta_vs_current": min(
                    fold["versus_current"]["delta"]["average_precision"]
                    for fold in value["per_fold"].values()
                ),
                "worst_fold_recall_delta_vs_current": min(
                    fold["versus_current"]["delta"]["recall_at_fpr_0_0713"]
                    for fold in value["per_fold"].values()
                ),
            }
            for value in candidates
        ],
        "selected": selected,
        "all_promotion_gates_pass": passed,
        "decision": (
            "Freeze the XGBoost head and its artifact for one exact-paper replay."
            if passed else "Reject the XGBoost head before paper scoring."
        ),
        "provenance": {
            **{f"{name}_cache_sha256": digest for name, digest in expected.items()},
            "artifact_sha256": artifact_digest,
            "script_sha256": sha256(Path(__file__).resolve()),
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True
            ).strip(),
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "joblib": joblib.__version__,
        },
    }
    write_json((root / args.output_json).resolve(), report)
    write_markdown((root / args.output_markdown).resolve(), report)
    print(
        json.dumps(
            {
                "ok": passed,
                "model_spec_name": selected["model_spec_name"],
                "xgboost_blend": selected["xgboost_blend"],
                "ap_delta_vs_current": selected["versus_current"]["delta"]["average_precision"],
                "recall_delta_vs_current": selected["versus_current"]["delta"]["recall_at_fpr_0_0713"],
                "ap_lower_vs_current": selected["paired_group_bootstrap_ap_delta_vs_current"]["lower"],
                "artifact_sha256": artifact_digest,
            },
            indent=2,
        )
    )
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
