#!/usr/bin/env python3
"""Fit a hard-example scene ranker with a preregistered recall margin."""

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

from acquire_mars_metadata import repo_root, sha256
from train_mars_paper_residual import choose_threshold_at_fpr
from train_mars_scene_ranker import (
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
DEFAULT_ARTIFACT = Path("EarthRemoteSensingRapidResponse/artifacts/mars_hard_scene_ranker_folds234.joblib")
DEFAULT_JSON = Path("reports/experiments/mars_hard_scene_ranker_inner_fold2.json")
DEFAULT_MARKDOWN = Path("reports/experiments/MARS_HARD_SCENE_RANKER_INNER_FOLD2.md")
TARGET_FPR = 0.0713
CS = (0.03, 0.1, 0.3)
HARD_POSITIVE_MULTIPLIERS = (2.0, 4.0, 8.0, 16.0)
HARD_NEGATIVE_MULTIPLIERS = (1.0, 2.0)
BLEND_LAMBDAS = (0.125, 0.25, 0.375, 0.5, 0.625)
MINIMUM_EXTRA_TRUE_POSITIVES = 3


def hard_example_masks(labels: np.ndarray, primary_scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    operating = choose_threshold_at_fpr(labels, primary_scores, TARGET_FPR)
    threshold = float(operating["threshold"])
    return (
        (labels == 1) & (primary_scores < threshold),
        (labels == 0) & (primary_scores >= threshold),
        threshold,
    )


def hard_example_weights(
    groups: np.ndarray,
    labels: np.ndarray,
    sensors: np.ndarray,
    primary_scores: np.ndarray,
    positive_multiplier: float,
    negative_multiplier: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    weights = site_cell_weights(groups, labels, sensors)
    hard_positive, hard_negative, threshold = hard_example_masks(labels, primary_scores)
    weights = weights.copy()
    weights[hard_positive] *= positive_multiplier
    weights[hard_negative] *= negative_multiplier
    weights /= weights.mean()
    return weights, {
        "threshold": threshold,
        "hard_positive": int(hard_positive.sum()),
        "hard_negative": int(hard_negative.sum()),
    }


def robust_checks(candidate: dict[str, Any], positive_count: int) -> dict[str, bool]:
    checks = dict(candidate["checks"])
    checks["minimum_three_tp_recall_margin"] = (
        candidate["delta"]["recall_at_fpr_0_0713"]
        >= MINIMUM_EXTRA_TRUE_POSITIVES / positive_count
    )
    return checks


def select_robust_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    passing = [candidate for candidate in candidates if all(candidate["robust_checks"].values())]
    return max(passing or candidates, key=lambda candidate: tuple(candidate["rank"]))


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    selected = report["selected"]
    lines = [
        "# Hard-example MARS scene ranker inner selection",
        "",
        "Train folds 3-4, validate fold 2; folds 0/1 and the paper test were not loaded.",
        "",
        f"- Logistic C: {selected['spec']['C']}",
        f"- Hard-positive multiplier: {selected['hard_positive_multiplier']}",
        f"- Hard-negative multiplier: {selected['hard_negative_multiplier']}",
        f"- Head blend weight: {selected['blend_lambda']}",
        f"- AP delta: {selected['delta']['average_precision']:+.5f}",
        f"- Recall delta: {selected['delta']['recall_at_fpr_0_0713']:+.5f}",
        f"- Robust inner gate: {'pass' if all(selected['robust_checks'].values()) else 'fail'}",
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
        features = cache["features"].astype(np.float64)
        feature_names = cache["feature_names"].astype(str)
        labels = cache["labels"].astype(np.uint8)
        sensors = cache["sensors"].astype(np.uint8)
        groups = cache["groups"].astype(str)
        folds = cache["folds"].astype(np.uint8)
        source_provenance = {
            key: str(cache[key].item())
            for key in ("artifact_sha256", "manifest_sha256", "protocol_sha256")
        }
    if set(np.unique(folds).tolist()) != {2, 3, 4}:
        raise ValueError("Hard-ranker cache must contain only folds 2, 3, and 4")
    primary_index = int(np.flatnonzero(feature_names == "primary_connected_score")[0])
    train = np.isin(folds, (3, 4))
    validation = folds == 2
    baseline = metric_summary(labels[validation], features[validation, primary_index], sensors[validation])
    positive_count = int(np.count_nonzero(labels[validation] == 1))
    candidates: list[dict[str, Any]] = []
    for c in CS:
        spec = {"family": "logistic", "C": c}
        for positive_multiplier in HARD_POSITIVE_MULTIPLIERS:
            for negative_multiplier in HARD_NEGATIVE_MULTIPLIERS:
                weights, hard_counts = hard_example_weights(
                    groups[train], labels[train], sensors[train],
                    features[train, primary_index], positive_multiplier, negative_multiplier,
                )
                fitted = fit_model(spec, features[train], labels[train], weights)
                head = predict_model(fitted, features[validation])
                for blend_lambda in BLEND_LAMBDAS:
                    scores = blend_scores(features[validation, primary_index], head, blend_lambda)
                    candidate = comparison(
                        metric_summary(labels[validation], scores, sensors[validation]), baseline
                    )
                    candidate.update({
                        "spec": spec,
                        "hard_positive_multiplier": positive_multiplier,
                        "hard_negative_multiplier": negative_multiplier,
                        "blend_lambda": blend_lambda,
                        "training_hard_examples": hard_counts,
                    })
                    candidate["robust_checks"] = robust_checks(candidate, positive_count)
                    candidates.append(candidate)
                print(json.dumps({
                    "completed": spec_key(spec),
                    "hard_positive_multiplier": positive_multiplier,
                    "hard_negative_multiplier": negative_multiplier,
                }), flush=True)
    selected = select_robust_candidate(candidates)
    passed = all(selected["robust_checks"].values())
    decision = (
        "Refit the robust head on folds 2-4 and freeze a new fold-0 evaluation."
        if passed else "Reject hard-example scene ranking before another fold-0 evaluation."
    )
    artifact_path = (root / args.artifact).resolve()
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_hash: str | None = None
    full_hard_counts: dict[str, Any] | None = None
    if passed:
        weights, full_hard_counts = hard_example_weights(
            groups, labels, sensors, features[:, primary_index],
            selected["hard_positive_multiplier"], selected["hard_negative_multiplier"],
        )
        fitted = fit_model(selected["spec"], features, labels, weights)
        payload = {
            "schema_version": 1,
            "architecture": "hard_example_scene_ranker_v1",
            "spec": selected["spec"],
            "hard_positive_multiplier": selected["hard_positive_multiplier"],
            "hard_negative_multiplier": selected["hard_negative_multiplier"],
            "blend_lambda": selected["blend_lambda"],
            "minimum_extra_true_positives_inner": MINIMUM_EXTRA_TRUE_POSITIVES,
            "feature_names": feature_names.tolist(),
            "primary_feature": "primary_connected_score",
            "fitted": fitted,
            "training_folds": [2, 3, 4],
            "cache_sha256": args.cache_sha256,
            "source_provenance": source_provenance,
            "full_hard_examples": full_hard_counts,
        }
        temporary = artifact_path.with_suffix(artifact_path.suffix + ".tmp")
        joblib.dump(payload, temporary, compress=3)
        os.replace(temporary, artifact_path)
        artifact_hash = sha256(artifact_path)
    report = {
        "schema_version": 1,
        "scope": "hard-example inner selection: train folds 3-4, validate fold 2",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows": {"train": int(train.sum()), "validation": int(validation.sum())},
        "validation_positive": positive_count,
        "minimum_extra_true_positives": MINIMUM_EXTRA_TRUE_POSITIVES,
        "baseline": baseline,
        "grid": {
            "C": list(CS),
            "hard_positive_multiplier": list(HARD_POSITIVE_MULTIPLIERS),
            "hard_negative_multiplier": list(HARD_NEGATIVE_MULTIPLIERS),
            "blend_lambda": list(BLEND_LAMBDAS),
        },
        "candidates": candidates,
        "selected": selected,
        "decision": decision,
        "artifact": None if artifact_hash is None else {
            "path": args.artifact, "sha256": artifact_hash,
            "bytes": artifact_path.stat().st_size, "tracked": False,
            "full_hard_examples": full_hard_counts,
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
        "ok": passed,
        "selected": {
            "C": selected["spec"]["C"],
            "hard_positive_multiplier": selected["hard_positive_multiplier"],
            "hard_negative_multiplier": selected["hard_negative_multiplier"],
            "blend_lambda": selected["blend_lambda"],
        },
        "deltas": selected["delta"],
        "robust_checks": selected["robust_checks"],
        "artifact_sha256": artifact_hash,
        "decision": decision,
    }, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
