#!/usr/bin/env python3
"""Select a scene ranker augmented with label-free site-sequence context."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import scipy
import sklearn
from scipy.stats import rankdata

from acquire_mars_metadata import repo_root, sha256
from train_mars_hard_scene_ranker import MINIMUM_EXTRA_TRUE_POSITIVES, robust_checks, select_robust_candidate
from train_mars_scene_ranker import (
    BLEND_LAMBDAS,
    blend_scores,
    comparison,
    fit_model,
    metric_summary,
    predict_model,
    site_cell_weights,
    spec_key,
)

DEFAULT_CACHE = Path("outputs/mars_scene_features_folds234.npz")
DEFAULT_CACHE_SHA256 = "01d8587e283c1179d61a7c789eb514b3f699d3e7a75bf8c50e4baff3f1698b89"
DEFAULT_ARTIFACT = Path("EarthRemoteSensingRapidResponse/artifacts/mars_context_scene_ranker_folds234.joblib")
DEFAULT_JSON = Path("reports/experiments/mars_context_scene_ranker_inner_fold2.json")
DEFAULT_MARKDOWN = Path("reports/experiments/MARS_CONTEXT_SCENE_RANKER_INNER_FOLD2.md")
CONTEXT_BASE_FEATURES = (
    "primary_connected_score",
    "released_connected_score",
    "primary_top_100_mean",
    "primary_top_500_mean",
    "primary_area_above_0.3",
    "primary_area_above_0.5",
    "input_0_mean",
    "input_0_top_100_mean",
    "logit_delta_valid_mean",
    "clear_fraction",
)
CONTEXT_STATISTICS = ("group_mean", "group_std", "group_max", "group_q90", "leave_one_out_max", "within_group_rank")
CONTEXT_BLEND_LAMBDAS = (0.125, 0.25, 0.375, 0.5, 0.625)


def context_specs() -> list[dict[str, Any]]:
    specs = [{"family": "logistic", "C": value} for value in (0.01, 0.03, 0.1, 0.3)]
    specs.extend(
        {
            "family": "hist_gradient_boosting",
            "max_leaf_nodes": leaves,
            "min_samples_leaf": minimum,
            "l2_regularization": 10.0,
        }
        for leaves in (15, 31)
        for minimum in (20, 50)
    )
    return specs


def context_feature_names() -> list[str]:
    names = [
        f"context_{feature}_{statistic}"
        for feature in CONTEXT_BASE_FEATURES
        for statistic in CONTEXT_STATISTICS
    ]
    return [*names, "context_log_group_size"]


def leave_one_out_max(values: np.ndarray) -> np.ndarray:
    if values.shape[0] <= 1:
        return values.copy()
    maximum = np.max(values, axis=0)
    maximum_count = np.sum(values == maximum, axis=0)
    second = np.partition(values, -2, axis=0)[-2]
    return np.where(
        (values == maximum) & (maximum_count == 1),
        second,
        maximum,
    )


def augment_site_context(
    features: np.ndarray, feature_names: np.ndarray, groups: np.ndarray
) -> tuple[np.ndarray, list[str]]:
    indices = [int(np.flatnonzero(feature_names == name)[0]) for name in CONTEXT_BASE_FEATURES]
    selected = features[:, indices]
    context = np.empty((features.shape[0], len(context_feature_names())), dtype=np.float64)
    for group in np.unique(groups):
        rows = np.flatnonzero(groups == group)
        values = selected[rows]
        statistics = [
            np.broadcast_to(values.mean(axis=0), values.shape),
            np.broadcast_to(values.std(axis=0), values.shape),
            np.broadcast_to(values.max(axis=0), values.shape),
            np.broadcast_to(np.quantile(values, 0.9, axis=0), values.shape),
            leave_one_out_max(values),
            np.column_stack(
                [rankdata(values[:, column], method="average") / len(rows) for column in range(values.shape[1])]
            ),
        ]
        # Names are feature-major while statistics above are statistic-major.
        context[rows, :-1] = np.stack(statistics, axis=2).reshape(len(rows), -1)
        context[rows, -1] = np.log1p(len(rows))
    augmented = np.concatenate((features, context), axis=1)
    if not np.isfinite(augmented).all():
        raise RuntimeError("Site-context features contain non-finite values")
    return augmented, [*feature_names.tolist(), *context_feature_names()]


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    selected = report["selected"]
    lines = [
        "# Site-context MARS scene ranker inner selection",
        "",
        "Train folds 3-4, validate fold 2; folds 0/1 and the paper test were not loaded.",
        "",
        f"- Selected model: `{spec_key(selected['spec'])}`",
        f"- Head blend weight: {selected['blend_lambda']}",
        f"- AP delta: {selected['delta']['average_precision']:+.5f}",
        f"- Recall delta: {selected['delta']['recall_at_fpr_0_0713']:+.5f}",
        f"- Robust three-TP gate: {'pass' if all(selected['robust_checks'].values()) else 'fail'}",
        "",
        report["decision"],
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", default=DEFAULT_CACHE.as_posix())
    parser.add_argument("--cache-sha256", default=DEFAULT_CACHE_SHA256)
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
        base_names = cache["feature_names"].astype(str)
        labels = cache["labels"].astype(np.uint8)
        sensors = cache["sensors"].astype(np.uint8)
        groups = cache["groups"].astype(str)
        folds = cache["folds"].astype(np.uint8)
        source_provenance = {
            key: str(cache[key].item())
            for key in ("artifact_sha256", "manifest_sha256", "protocol_sha256")
        }
    if set(np.unique(folds).tolist()) != {2, 3, 4}:
        raise ValueError("Context-ranker cache must contain only folds 2, 3, and 4")
    features, augmented_names = augment_site_context(base_features, base_names, groups)
    primary_index = int(np.flatnonzero(base_names == "primary_connected_score")[0])
    train = np.isin(folds, (3, 4))
    validation = folds == 2
    weights = site_cell_weights(groups[train], labels[train], sensors[train])
    baseline = metric_summary(labels[validation], base_features[validation, primary_index], sensors[validation])
    positive_count = int(np.count_nonzero(labels[validation] == 1))
    candidates: list[dict[str, Any]] = []
    for spec in context_specs():
        fitted = fit_model(spec, features[train], labels[train], weights)
        head = predict_model(fitted, features[validation])
        for blend_lambda in CONTEXT_BLEND_LAMBDAS:
            scores = blend_scores(base_features[validation, primary_index], head, blend_lambda)
            candidate = comparison(metric_summary(labels[validation], scores, sensors[validation]), baseline)
            candidate.update({"spec": spec, "blend_lambda": blend_lambda})
            candidate["robust_checks"] = robust_checks(candidate, positive_count)
            candidates.append(candidate)
        print(json.dumps({"completed": spec_key(spec)}), flush=True)
    selected = select_robust_candidate(candidates)
    passed = all(selected["robust_checks"].values())
    decision = (
        "Refit the context head on folds 2-4 and freeze its fold-0 evaluation."
        if passed else "Reject site-context scene ranking before another fold-0 evaluation."
    )
    artifact_path = (root / args.artifact).resolve()
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_hash: str | None = None
    if passed:
        fitted = fit_model(selected["spec"], features, labels, site_cell_weights(groups, labels, sensors))
        payload = {
            "schema_version": 1,
            "architecture": "site_context_scene_ranker_v1",
            "spec": selected["spec"],
            "blend_lambda": selected["blend_lambda"],
            "minimum_extra_true_positives_inner": MINIMUM_EXTRA_TRUE_POSITIVES,
            "base_feature_names": base_names.tolist(),
            "augmented_feature_names": augmented_names,
            "primary_feature": "primary_connected_score",
            "context_base_features": list(CONTEXT_BASE_FEATURES),
            "context_statistics": list(CONTEXT_STATISTICS),
            "fitted": fitted,
            "training_folds": [2, 3, 4],
            "cache_sha256": args.cache_sha256,
            "source_provenance": source_provenance,
        }
        temporary = artifact_path.with_suffix(artifact_path.suffix + ".tmp")
        joblib.dump(payload, temporary, compress=3)
        os.replace(temporary, artifact_path)
        artifact_hash = sha256(artifact_path)
    report = {
        "schema_version": 1,
        "scope": "site-context inner selection: train folds 3-4, validate fold 2",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows": {"train": int(train.sum()), "validation": int(validation.sum())},
        "validation_positive": positive_count,
        "base_feature_count": int(base_features.shape[1]),
        "augmented_feature_count": int(features.shape[1]),
        "context_base_features": list(CONTEXT_BASE_FEATURES),
        "context_statistics": list(CONTEXT_STATISTICS),
        "minimum_extra_true_positives": MINIMUM_EXTRA_TRUE_POSITIVES,
        "model_specs": context_specs(),
        "blend_lambdas": list(CONTEXT_BLEND_LAMBDAS),
        "baseline": baseline,
        "candidates": candidates,
        "selected": selected,
        "decision": decision,
        "artifact": None if artifact_hash is None else {
            "path": args.artifact, "sha256": artifact_hash,
            "bytes": artifact_path.stat().st_size, "tracked": False,
        },
        "provenance": {
            "cache_path": args.cache, "cache_sha256": args.cache_sha256,
            **source_provenance,
            "script_sha256": sha256(Path(__file__).resolve()),
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
            "numpy": np.__version__, "scipy": scipy.__version__,
            "sklearn": sklearn.__version__, "joblib": joblib.__version__,
        },
    }
    output_json = (root / args.output_json).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown((root / args.output_markdown).resolve(), report)
    print(json.dumps({
        "ok": passed, "selected": {"spec": selected["spec"], "blend_lambda": selected["blend_lambda"]},
        "deltas": selected["delta"], "robust_checks": selected["robust_checks"],
        "artifact_sha256": artifact_hash, "decision": decision,
    }, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
