#!/usr/bin/env python3
"""Train target-density-weighted ExtraTrees heads on MARS development folds."""

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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "EarthRemoteSensingRapidResponse"
for path in (MODEL_ROOT, ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from acquire_mars_metadata import repo_root, sha256  # noqa: E402
from train_mars_crossfold_bagged_scene_head import (  # noqa: E402
    FOLDS,
    SPEC,
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
from train_mars_oof_scene_ensemble_v2 import ap_group_bootstrap, fit_model  # noqa: E402
from train_mars_scene_ranker import blend_scores, comparison, metric_summary  # noqa: E402

DEFAULT_ARTIFACT = Path(
    "EarthRemoteSensingRapidResponse/artifacts/mars_target_weighted_scene_head.joblib"
)
DEFAULT_JSON = Path("reports/experiments/mars_target_weighted_scene_head.json")
DEFAULT_MARKDOWN = Path("reports/experiments/MARS_TARGET_WEIGHTED_SCENE_HEAD.md")
WEIGHT_SPECS = (
    {"gamma": 0.5, "clip_lower": 0.25, "clip_upper": 4.0},
    {"gamma": 1.0, "clip_lower": 0.25, "clip_upper": 4.0},
    {"gamma": 0.5, "clip_lower": 0.1, "clip_upper": 10.0},
)
BLENDS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.625, 0.75, 0.875, 1.0)
MINIMUM_POOLED_AP_GAIN = 0.002


def domain_matrix(features: np.ndarray, sensors: np.ndarray) -> np.ndarray:
    return np.concatenate((features, sensors.astype(np.float64)[:, None]), axis=1)


def density_ratio_weights(
    source: np.ndarray,
    target: np.ndarray,
    *,
    gamma: float,
    clip_lower: float,
    clip_upper: float,
) -> tuple[np.ndarray, dict[str, float]]:
    if source.ndim != 2 or target.ndim != 2 or source.shape[1] != target.shape[1]:
        raise ValueError("Source and target domain features do not align")
    scaler = StandardScaler().fit(np.concatenate((source, target)))
    combined = scaler.transform(np.concatenate((source, target)))
    domain = np.concatenate(
        (np.zeros(source.shape[0], dtype=np.uint8), np.ones(target.shape[0], dtype=np.uint8))
    )
    classifier = LogisticRegression(
        C=0.1,
        max_iter=500,
        class_weight="balanced",
        solver="lbfgs",
        random_state=20261300,
    ).fit(combined, domain)
    probabilities = classifier.predict_proba(combined)[:, 1]
    source_probability = np.clip(probabilities[: source.shape[0]], 1e-4, 1.0 - 1e-4)
    ratio = source_probability / (1.0 - source_probability)
    ratio = np.clip(ratio, clip_lower, clip_upper) ** gamma
    weights = ratio / ratio.mean()
    if not np.isfinite(weights).all() or np.any(weights <= 0.0):
        raise RuntimeError("Density-ratio weights are invalid")
    return weights, {
        "domain_auc": float(roc_auc_score(domain, probabilities)),
        "weight_min": float(weights.min()),
        "weight_mean": float(weights.mean()),
        "weight_max": float(weights.max()),
        "effective_sample_size": float(weights.sum() ** 2 / np.square(weights).sum()),
    }


def spec_key(spec: dict[str, float]) -> str:
    return "_".join(f"{key}-{spec[key]}" for key in sorted(spec))


def oof_scores(
    values: dict[str, Any], weight_spec: dict[str, float]
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    scores = np.empty(values["labels"].shape, dtype=np.float64)
    audits = []
    all_domain = domain_matrix(values["features"], values["sensors"])
    for holdout in FOLDS:
        fit = values["folds"] != holdout
        held = values["folds"] == holdout
        weights, audit = density_ratio_weights(
            all_domain[fit], all_domain[held], **weight_spec
        )
        model = fit_model(
            SPEC, values["features"][fit], values["labels"][fit], weights
        )
        scores[held] = model.predict_proba(values["features"][held])[:, 1]
        audits.append(
            {
                "holdout": holdout,
                "training_rows": int(fit.sum()),
                "target_rows": int(held.sum()),
                **audit,
            }
        )
        print(
            json.dumps(
                {
                    "weight_spec": spec_key(weight_spec),
                    "completed_holdout": holdout,
                    **audit,
                }
            ),
            flush=True,
        )
    return scores, audits


def evaluate_candidate(
    values: dict[str, Any], raw: np.ndarray, weight_spec: dict[str, float], blend: float
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
                    values["labels"][rows], values["current"][rows], values["sensors"][rows]
                ),
            ),
            "versus_primary": comparison(
                local,
                metric_summary(
                    values["labels"][rows], values["primary"][rows], values["sensors"][rows]
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
        "weight_spec": weight_spec,
        "weight_spec_key": spec_key(weight_spec),
        "target_weighted_blend": blend,
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
        "# Unsupervised target-weighted MARS scene head",
        "",
        "Every held fold supplies features and sensor identity but no labels to its density-ratio estimator.",
        "",
        f"- Weight specification: `{selected['weight_spec_key']}`",
        f"- Target-weighted head blend: {selected['target_weighted_blend']:.3f}",
        f"- AP delta vs current: {delta['average_precision']:+.5f}",
        f"- Recall delta vs current: {delta['recall_at_fpr_0_0713']:+.5f}",
        f"- Paired-site AP interval vs current: [{interval['lower']:+.5f}, {interval['upper']:+.5f}]",
        "",
        report["decision"],
    ]
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
    audits = {}
    candidates = []
    for weight_spec in WEIGHT_SPECS:
        raw, local_audits = oof_scores(values, weight_spec)
        key = spec_key(weight_spec)
        raw_store[key] = raw
        audits[key] = local_audits
        candidates.extend(
            evaluate_candidate(values, raw, weight_spec, blend) for blend in BLENDS
        )
    selected = max(candidates, key=lambda value: tuple(value["rank"]))
    selected_scores = blend_scores(
        values["current"],
        raw_store[selected["weight_spec_key"]],
        float(selected["target_weighted_blend"]),
    )
    selected["paired_group_bootstrap_ap_delta_vs_primary"] = ap_group_bootstrap(
        values["labels"], values["primary"], selected_scores, values["groups"],
        replicates=10_000, seed=20261320,
    )
    selected["paired_group_bootstrap_ap_delta_vs_current"] = ap_group_bootstrap(
        values["labels"], values["current"], selected_scores, values["groups"],
        replicates=10_000, seed=20261321,
    )
    passed = bool(
        selected["stable"]
        and selected["paired_group_bootstrap_ap_delta_vs_primary"]["lower"] > 0.0
        and selected["paired_group_bootstrap_ap_delta_vs_current"]["lower"] > 0.0
    )
    thresholds = [
        value["versus_primary"]["metrics"]["operating_point"]["threshold"]
        for value in selected["per_fold"].values()
    ]
    artifact_path = (root / args.artifact).resolve()
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = artifact_path.with_suffix(artifact_path.suffix + ".tmp")
    joblib.dump(
        {
            "schema_version": 1,
            "kind": "mars_unsupervised_target_weighted_scene_control",
            "weight_spec": selected["weight_spec"],
            "target_weighted_blend": float(selected["target_weighted_blend"]),
            "base_score": "frozen v3 stronger scene head",
            "operational_scene_threshold": max(thresholds),
            "domain_features": "169 scene/context features plus sensor identity",
            "label_contract": "target labels are never used for density weighting or fitting",
        },
        temporary,
        compress=3,
    )
    os.replace(temporary, artifact_path)
    report = {
        "schema_version": 1,
        "scope": "five-fold unsupervised target-weight simulation; paper cache not loaded",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "weight_specs": list(WEIGHT_SPECS),
        "blends": list(BLENDS),
        "minimum_pooled_ap_gain": MINIMUM_POOLED_AP_GAIN,
        "domain_audits": audits,
        "candidate_summaries": [
            {
                "weight_spec_key": value["weight_spec_key"],
                "target_weighted_blend": value["target_weighted_blend"],
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
        "operational_scene_threshold": max(thresholds),
        "all_promotion_gates_pass": passed,
        "decision": (
            "Freeze unsupervised target weighting for one label-free paper adaptation and replay."
            if passed else "Reject unsupervised target weighting before paper adaptation."
        ),
        "provenance": {
            **{f"{name}_cache_sha256": digest for name, digest in expected.items()},
            "artifact_sha256": sha256(artifact_path),
            "script_sha256": sha256(Path(__file__).resolve()),
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True
            ).strip(),
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
    }
    write_json((root / args.output_json).resolve(), report)
    write_markdown((root / args.output_markdown).resolve(), report)
    print(json.dumps({
        "ok": passed,
        "weight_spec": selected["weight_spec"],
        "target_weighted_blend": selected["target_weighted_blend"],
        "ap_delta_vs_current": selected["versus_current"]["delta"]["average_precision"],
        "recall_delta_vs_current": selected["versus_current"]["delta"]["recall_at_fpr_0_0713"],
        "ap_lower_vs_current": selected["paired_group_bootstrap_ap_delta_vs_current"]["lower"],
        "artifact_sha256": report["provenance"]["artifact_sha256"],
    }, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
