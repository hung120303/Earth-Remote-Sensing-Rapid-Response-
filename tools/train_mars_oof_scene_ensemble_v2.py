#!/usr/bin/env python3
"""Search stronger cross-fitted MARS scene heads with site-bootstrap AP selection."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "EarthRemoteSensingRapidResponse"
for path in (MODEL_ROOT, ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from acquire_mars_metadata import repo_root, sha256  # noqa: E402
from evaluate_mars_successor_paper_test import (  # noqa: E402
    average_precision_from_cumulative,
    interval,
    plan_cumulative,
    score_plan,
)
from train_mars_context_scene_ranker import augment_site_context  # noqa: E402
from train_mars_scene_ranker import (  # noqa: E402
    blend_scores,
    comparison,
    metric_summary,
    site_cell_weights,
)

DEFAULT_CACHE = Path("outputs/mars_scene_features_folds234.npz")
DEFAULT_CACHE_SHA256 = "01d8587e283c1179d61a7c789eb514b3f699d3e7a75bf8c50e4baff3f1698b89"
DEFAULT_ARTIFACT = Path("EarthRemoteSensingRapidResponse/artifacts/mars_oof_scene_ensemble_v2.joblib")
DEFAULT_JSON = Path("reports/experiments/mars_oof_scene_ensemble_v2_folds234.json")
DEFAULT_MARKDOWN = Path("reports/experiments/MARS_OOF_SCENE_ENSEMBLE_V2_FOLDS234.md")
FOLDS = (2, 3, 4)
BLENDS = (0.25, 0.5, 0.625, 0.75, 0.875, 1.0)


def model_specs() -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for weighting in ("site_cell", "group", "uniform"):
        specs.extend(
            {
                "family": "hist_gradient_boosting",
                "weighting": weighting,
                "max_leaf_nodes": leaves,
                "min_samples_leaf": minimum,
                "l2_regularization": regularization,
            }
            for leaves, minimum, regularization in ((31, 20, 10.0), (63, 20, 10.0))
        )
        specs.extend(
            {
                "family": "extra_trees",
                "weighting": weighting,
                "n_estimators": 400,
                "min_samples_leaf": minimum,
                "max_features": maximum,
            }
            for minimum, maximum in ((5, 0.5),)
        )
    return specs


def spec_key(spec: dict[str, Any]) -> str:
    return "_".join(f"{key}-{spec[key]}" for key in sorted(spec))


def sample_weights(
    mode: str, groups: np.ndarray, labels: np.ndarray, sensors: np.ndarray
) -> np.ndarray:
    if mode == "site_cell":
        return site_cell_weights(groups, labels, sensors)
    if mode == "uniform":
        return np.ones(labels.shape, dtype=np.float64)
    if mode == "group":
        counts = Counter(groups.tolist())
        values = np.asarray([1.0 / counts[group] for group in groups], dtype=np.float64)
        return values / values.mean()
    raise ValueError(f"Unknown weighting mode: {mode}")


def fit_model(
    spec: dict[str, Any], features: np.ndarray, labels: np.ndarray, weights: np.ndarray
) -> Any:
    if spec["family"] == "hist_gradient_boosting":
        model = HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=250,
            max_leaf_nodes=int(spec["max_leaf_nodes"]),
            min_samples_leaf=int(spec["min_samples_leaf"]),
            l2_regularization=float(spec["l2_regularization"]),
            early_stopping=False,
            random_state=20260715,
        )
    elif spec["family"] == "extra_trees":
        model = ExtraTreesClassifier(
            n_estimators=int(spec["n_estimators"]),
            min_samples_leaf=int(spec["min_samples_leaf"]),
            max_features=float(spec["max_features"]),
            criterion="entropy",
            n_jobs=-1,
            random_state=20260715,
        )
    else:
        raise ValueError(f"Unknown model family: {spec['family']}")
    model.fit(features, labels, sample_weight=weights)
    return model


def ap_group_bootstrap(
    labels: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    groups: np.ndarray,
    *,
    replicates: int,
    seed: int,
    confidence: float = 0.95,
    batch_size: int = 64,
) -> dict[str, float | int]:
    _, group_index = np.unique(groups.astype(str), return_inverse=True)
    group_count = int(group_index.max()) + 1
    baseline_plan = score_plan(labels, baseline, group_index)
    candidate_plan = score_plan(labels, candidate, group_index)
    rng = np.random.default_rng(seed)
    probabilities = np.full(group_count, 1.0 / group_count)
    parts: list[np.ndarray] = []
    for start in range(0, replicates, batch_size):
        size = min(batch_size, replicates - start)
        draws = rng.multinomial(group_count, probabilities, size=size).astype(np.int32)
        base_tp, base_fp, base_positive = plan_cumulative(draws, baseline_plan)
        cand_tp, cand_fp, cand_positive = plan_cumulative(draws, candidate_plan)
        base_ap = average_precision_from_cumulative(base_tp, base_fp, base_positive)
        cand_ap = average_precision_from_cumulative(cand_tp, cand_fp, cand_positive)
        parts.append(cand_ap - base_ap)
    values = np.concatenate(parts)
    return {
        "replicates": replicates,
        "groups": group_count,
        "confidence": confidence,
        **interval(values, confidence),
    }


def stable(candidate: dict[str, Any]) -> bool:
    fold_ap = [value["delta"]["average_precision"] for value in candidate["per_fold"].values()]
    fold_recall = [
        value["delta"]["recall_at_fpr_0_0713"] for value in candidate["per_fold"].values()
    ]
    sensor_ap = candidate["delta"]["sensor_average_precision"].values()
    return (
        candidate["delta"]["average_precision"] > 0.0
        and candidate["delta"]["recall_at_fpr_0_0713"] > 0.0
        and min(fold_ap) > 0.0
        and min(fold_recall) >= 0.0
        and min(sensor_ap) >= -0.005
    )


def screen_rank(candidate: dict[str, Any]) -> tuple[bool, float, float, float]:
    fold_ap = min(value["delta"]["average_precision"] for value in candidate["per_fold"].values())
    fold_recall = min(
        value["delta"]["recall_at_fpr_0_0713"] for value in candidate["per_fold"].values()
    )
    return stable(candidate), fold_ap, candidate["delta"]["average_precision"], fold_recall


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    selected = report["selected"]
    bootstrap = selected["paired_group_bootstrap_ap_delta"]
    lines = [
        "# Stronger three-fold OOF MARS scene-head search",
        "",
        "Every fold-2/3/4 score is produced by a head trained on the other two folds. Folds 0/1 and the paper test were not loaded.",
        "",
        f"- Model: `{spec_key(selected['spec'])}`",
        f"- Blend: {selected['blend_lambda']:.3f}",
        f"- Pooled AP delta: {selected['delta']['average_precision']:+.5f}",
        f"- Worst-fold AP delta: {min(value['delta']['average_precision'] for value in selected['per_fold'].values()):+.5f}",
        f"- Pooled recall delta: {selected['delta']['recall_at_fpr_0_0713']:+.5f}",
        f"- Paired site-bootstrap AP delta: [{bootstrap['lower']:+.5f}, {bootstrap['upper']:+.5f}]",
        "",
        report["decision"],
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", default=DEFAULT_CACHE.as_posix())
    parser.add_argument("--cache-sha256", default=DEFAULT_CACHE_SHA256)
    parser.add_argument("--screen-bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--final-bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--artifact", default=DEFAULT_ARTIFACT.as_posix())
    parser.add_argument("--output-json", default=DEFAULT_JSON.as_posix())
    parser.add_argument("--output-markdown", default=DEFAULT_MARKDOWN.as_posix())
    args = parser.parse_args()
    root = repo_root()
    cache_path = (root / args.cache).resolve()
    if sha256(cache_path) != args.cache_sha256:
        raise ValueError("Scene feature cache hash mismatch")
    with np.load(cache_path, allow_pickle=False) as cache:
        base_features = cache["features"].astype(np.float64)
        feature_names = cache["feature_names"].astype(str)
        labels = cache["labels"].astype(np.uint8)
        sensors = cache["sensors"].astype(np.uint8)
        groups = cache["groups"].astype(str)
        folds = cache["folds"].astype(np.uint8)
        source_provenance = {
            name: str(cache[name].item())
            for name in ("artifact_sha256", "manifest_sha256", "protocol_sha256")
        }
    if set(np.unique(folds).tolist()) != set(FOLDS):
        raise ValueError("Cache must contain only folds 2, 3, and 4")
    features, augmented_names = augment_site_context(base_features, feature_names, groups)
    primary_index = int(np.flatnonzero(feature_names == "primary_connected_score")[0])
    primary = base_features[:, primary_index]
    baseline = metric_summary(labels, primary, sensors)
    baseline_by_fold = {
        str(fold): metric_summary(labels[folds == fold], primary[folds == fold], sensors[folds == fold])
        for fold in FOLDS
    }
    candidates: list[dict[str, Any]] = []
    score_store: list[np.ndarray] = []
    for spec_index, spec in enumerate(model_specs()):
        oof_head = np.empty(labels.shape, dtype=np.float64)
        for holdout in FOLDS:
            fit_rows = folds != holdout
            held_rows = folds == holdout
            weights = sample_weights(
                spec["weighting"], groups[fit_rows], labels[fit_rows], sensors[fit_rows]
            )
            fitted = fit_model(spec, features[fit_rows], labels[fit_rows], weights)
            oof_head[held_rows] = fitted.predict_proba(features[held_rows])[:, 1]
        for blend in BLENDS:
            scores = blend_scores(primary, oof_head, blend)
            pooled = comparison(metric_summary(labels, scores, sensors), baseline)
            per_fold = {
                str(fold): comparison(
                    metric_summary(labels[folds == fold], scores[folds == fold], sensors[folds == fold]),
                    baseline_by_fold[str(fold)],
                )
                for fold in FOLDS
            }
            candidate = {
                **pooled,
                "spec": spec,
                "blend_lambda": blend,
                "per_fold": per_fold,
                "stable_screen": False,
            }
            candidate["stable_screen"] = stable(candidate)
            candidates.append(candidate)
            score_store.append(scores)
        print(json.dumps({"completed": spec_index + 1, "total": len(model_specs()), "spec": spec_key(spec)}), flush=True)

    ranked_indices = sorted(range(len(candidates)), key=lambda index: screen_rank(candidates[index]), reverse=True)
    bootstrap_indices = ranked_indices[:12]
    for rank_index, candidate_index in enumerate(bootstrap_indices):
        candidates[candidate_index]["paired_group_bootstrap_ap_delta"] = ap_group_bootstrap(
            labels,
            primary,
            score_store[candidate_index],
            groups,
            replicates=args.screen_bootstrap_replicates,
            seed=20260715 + rank_index,
        )
    bootstrapped = [
        index for index in bootstrap_indices
        if candidates[index]["stable_screen"]
        and candidates[index]["paired_group_bootstrap_ap_delta"]["lower"] > 0.0
    ]
    selection_pool = bootstrapped or bootstrap_indices
    selected_index = max(
        selection_pool,
        key=lambda index: (
            candidates[index]["paired_group_bootstrap_ap_delta"]["lower"],
            *screen_rank(candidates[index]),
        ),
    )
    selected = candidates[selected_index]
    selected["paired_group_bootstrap_ap_delta"] = ap_group_bootstrap(
        labels,
        primary,
        score_store[selected_index],
        groups,
        replicates=args.final_bootstrap_replicates,
        seed=20261715,
    )
    passed = selected["stable_screen"] and selected["paired_group_bootstrap_ap_delta"]["lower"] > 0.0
    artifact_path = (root / args.artifact).resolve()
    artifact_hash = None
    if passed:
        weights = sample_weights(selected["spec"]["weighting"], groups, labels, sensors)
        fitted = fit_model(selected["spec"], features, labels, weights)
        payload = {
            "schema_version": 1,
            "architecture": "mars_oof_scene_ensemble_v2",
            "spec": selected["spec"],
            "blend_lambda": selected["blend_lambda"],
            "feature_names": feature_names.tolist(),
            "augmented_feature_names": augmented_names,
            "primary_feature": "primary_connected_score",
            "training_folds": list(FOLDS),
            "cache_sha256": args.cache_sha256,
            "source_provenance": source_provenance,
            "fitted": fitted,
        }
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = artifact_path.with_suffix(artifact_path.suffix + ".tmp")
        joblib.dump(payload, temporary, compress=3)
        os.replace(temporary, artifact_path)
        artifact_hash = sha256(artifact_path)
    report = {
        "schema_version": 1,
        "scope": "three-fold OOF selection on folds 2/3/4; folds 0/1 and paper test not loaded",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows": int(labels.size),
        "positive": int(labels.sum()),
        "folds": list(FOLDS),
        "model_specs": model_specs(),
        "blend_lambdas": list(BLENDS),
        "baseline": baseline,
        "baseline_by_fold": baseline_by_fold,
        "candidates": candidates,
        "selected": selected,
        "passed": passed,
        "decision": (
            "Refit the selected head on folds 2-4 and advance it to untouched fold-0 evaluation."
            if passed
            else "Reject this scene-head search before folds 0/1 or paper-test evaluation."
        ),
        "artifact": None if artifact_hash is None else {
            "path": args.artifact,
            "sha256": artifact_hash,
            "bytes": artifact_path.stat().st_size,
            "tracked": False,
        },
        "provenance": {
            "cache_path": args.cache,
            "cache_sha256": args.cache_sha256,
            **source_provenance,
            "script_sha256": sha256(Path(__file__).resolve()),
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
    }
    write_json((root / args.output_json).resolve(), report)
    write_markdown((root / args.output_markdown).resolve(), report)
    print(json.dumps({
        "ok": passed,
        "selected": {"spec": selected["spec"], "blend_lambda": selected["blend_lambda"]},
        "delta": selected["delta"],
        "worst_fold_ap_delta": min(value["delta"]["average_precision"] for value in selected["per_fold"].values()),
        "bootstrap": selected["paired_group_bootstrap_ap_delta"],
        "artifact_sha256": artifact_hash,
    }, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
