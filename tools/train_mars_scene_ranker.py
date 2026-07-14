#!/usr/bin/env python3
"""Select and fit a leakage-controlled scene ranker above frozen MARS masks."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import scipy
import sklearn
from scipy.special import expit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from acquire_mars_metadata import repo_root, sha256
from evaluate_released_marss2l import scene_metrics
from train_mars_paper_residual import SENSOR_NAMES, choose_threshold_at_fpr

DEFAULT_CACHE = Path("outputs/mars_scene_features_folds234.npz")
DEFAULT_CACHE_SHA256 = (
    "01d8587e283c1179d61a7c789eb514b3f699d3e7a75bf8c50e4baff3f1698b89"
)
DEFAULT_ARTIFACT = Path(
    "EarthRemoteSensingRapidResponse/artifacts/mars_scene_ranker_folds234.joblib"
)
DEFAULT_JSON = Path("reports/experiments/mars_scene_ranker_inner_fold2.json")
DEFAULT_MARKDOWN = Path("reports/experiments/MARS_SCENE_RANKER_INNER_FOLD2.md")
TARGET_FPR = 0.0713
BLEND_LAMBDAS = (0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0)


def model_specs() -> list[dict[str, Any]]:
    specs = [
        {"family": "logistic", "C": value}
        for value in (0.01, 0.03, 0.1, 0.3, 1.0)
    ]
    specs.extend(
        {
            "family": "hist_gradient_boosting",
            "max_leaf_nodes": leaves,
            "min_samples_leaf": minimum,
            "l2_regularization": regularization,
        }
        for leaves in (15, 31)
        for minimum in (20, 50)
        for regularization in (1.0, 10.0)
    )
    return specs


def spec_key(spec: dict[str, Any]) -> str:
    return "_".join(f"{key}-{spec[key]}" for key in sorted(spec))


def site_cell_weights(
    groups: np.ndarray, labels: np.ndarray, sensors: np.ndarray
) -> np.ndarray:
    keys = [f"{group}|{int(label)}|{int(sensor)}" for group, label, sensor in zip(groups, labels, sensors)]
    counts = Counter(keys)
    weights = np.asarray([1.0 / counts[key] for key in keys], dtype=np.float64)
    return weights / weights.mean()


def safe_logit(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=np.float64), 1e-5, 1.0 - 1e-5)
    return np.log(clipped) - np.log1p(-clipped)


def blend_scores(
    primary_scores: np.ndarray, head_probabilities: np.ndarray, weight: float
) -> np.ndarray:
    if not 0.0 <= weight <= 1.0:
        raise ValueError("blend weight must be in [0,1]")
    if weight == 0.0:
        return np.asarray(primary_scores, dtype=np.float64)
    if weight == 1.0:
        return np.asarray(head_probabilities, dtype=np.float64)
    return expit(
        (1.0 - weight) * safe_logit(primary_scores)
        + weight * safe_logit(head_probabilities)
    )


def fit_model(
    spec: dict[str, Any], features: np.ndarray, labels: np.ndarray, weights: np.ndarray
) -> dict[str, Any]:
    if spec["family"] == "logistic":
        scaler = StandardScaler().fit(features, sample_weight=weights)
        model = LogisticRegression(
            C=float(spec["C"]), solver="lbfgs", max_iter=1000, random_state=909
        )
        model.fit(scaler.transform(features), labels, sample_weight=weights)
        return {"scaler": scaler, "model": model}
    if spec["family"] == "hist_gradient_boosting":
        model = HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=250,
            max_leaf_nodes=int(spec["max_leaf_nodes"]),
            min_samples_leaf=int(spec["min_samples_leaf"]),
            l2_regularization=float(spec["l2_regularization"]),
            early_stopping=False,
            random_state=909,
        )
        model.fit(features, labels, sample_weight=weights)
        return {"scaler": None, "model": model}
    raise ValueError(f"Unknown model family: {spec['family']}")


def predict_model(fitted: dict[str, Any], features: np.ndarray) -> np.ndarray:
    scaler = fitted["scaler"]
    transformed = features if scaler is None else scaler.transform(features)
    return fitted["model"].predict_proba(transformed)[:, 1]


def metric_summary(
    labels: np.ndarray, scores: np.ndarray, sensors: np.ndarray
) -> dict[str, Any]:
    operating = choose_threshold_at_fpr(labels, scores, TARGET_FPR)
    threshold = float(operating["threshold"])
    summary = scene_metrics(labels, scores >= threshold, scores)
    summary["operating_point"] = operating
    summary["sensor_average_precision"] = {}
    for sensor_index, sensor_name in enumerate(SENSOR_NAMES):
        selection = sensors == sensor_index
        local = scene_metrics(
            labels[selection], scores[selection] >= threshold, scores[selection]
        )
        summary["sensor_average_precision"][sensor_name] = local["average_precision"]
    return summary


def comparison(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    ap_delta = float(candidate["average_precision"] - baseline["average_precision"])
    recall_delta = float(
        candidate["operating_point"]["recall"]
        - baseline["operating_point"]["recall"]
    )
    sensor_deltas = {
        name: float(
            candidate["sensor_average_precision"][name]
            - baseline["sensor_average_precision"][name]
        )
        for name in SENSOR_NAMES
    }
    checks = {
        "ap_higher": ap_delta > 0,
        "recall_at_fpr_0_0713_higher": recall_delta > 0,
        "no_material_sensor_ap_regression": all(value >= -0.01 for value in sensor_deltas.values()),
    }
    return {
        "metrics": candidate,
        "delta": {
            "average_precision": ap_delta,
            "recall_at_fpr_0_0713": recall_delta,
            "sensor_average_precision": sensor_deltas,
        },
        "checks": checks,
        "rank": [min(ap_delta, recall_delta), ap_delta + recall_delta, ap_delta],
    }


def select_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    passing = [candidate for candidate in candidates if all(candidate["checks"].values())]
    return max(passing or candidates, key=lambda candidate: tuple(candidate["rank"]))


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    selected = report["selected"]
    delta = selected["delta"]
    lines = [
        "# MARS scene ranker inner selection",
        "",
        "The ranker was trained on folds 3-4 and selected on fold 2. Folds 0, 1, and the paper test were not loaded.",
        "",
        f"- Selected model: `{spec_key(selected['spec'])}`",
        f"- Selected head weight: {selected['blend_lambda']:.3f}",
        f"- Fold-2 AP delta: {delta['average_precision']:+.5f}",
        f"- Fold-2 recall delta at <=7.13% FPR: {delta['recall_at_fpr_0_0713']:+.5f}",
        f"- All inner gates pass: {'yes' if all(selected['checks'].values()) else 'no'}",
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
        raise ValueError("Inner scene-ranker cache must contain only folds 2, 3, and 4")
    primary_index = int(np.flatnonzero(feature_names == "primary_connected_score")[0])
    train_selection = np.isin(folds, (3, 4))
    validation_selection = folds == 2
    weights = site_cell_weights(
        groups[train_selection], labels[train_selection], sensors[train_selection]
    )
    baseline = metric_summary(
        labels[validation_selection],
        features[validation_selection, primary_index],
        sensors[validation_selection],
    )
    candidates: list[dict[str, Any]] = []
    for spec in model_specs():
        fitted = fit_model(
            spec, features[train_selection], labels[train_selection], weights
        )
        head = predict_model(fitted, features[validation_selection])
        for blend_lambda in BLEND_LAMBDAS:
            scores = blend_scores(
                features[validation_selection, primary_index], head, blend_lambda
            )
            result = comparison(
                metric_summary(labels[validation_selection], scores, sensors[validation_selection]),
                baseline,
            )
            result["spec"] = spec
            result["blend_lambda"] = blend_lambda
            candidates.append(result)
        print(json.dumps({"completed": spec_key(spec)}), flush=True)
    selected = select_candidate(candidates)
    passed = all(selected["checks"].values())
    decision = (
        "Freeze this scene head, refit it on folds 2-4, then evaluate fold 0 once."
        if passed else "Reject scene-ranker family before fold-0 evaluation."
    )

    artifact_path = (root / args.artifact).resolve()
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_hash: str | None = None
    if passed:
        full_weights = site_cell_weights(groups, labels, sensors)
        fitted = fit_model(selected["spec"], features, labels, full_weights)
        payload = {
            "schema_version": 1,
            "spec": selected["spec"],
            "blend_lambda": selected["blend_lambda"],
            "feature_names": feature_names.tolist(),
            "primary_feature": "primary_connected_score",
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
        "scope": "inner selection: train folds 3-4, validate fold 2; folds 0/1 and paper test not loaded",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows": {"train": int(train_selection.sum()), "validation": int(validation_selection.sum())},
        "feature_count": int(features.shape[1]),
        "baseline": baseline,
        "model_specs": model_specs(),
        "blend_lambdas": list(BLEND_LAMBDAS),
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
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True
            ).strip(),
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
        "checks": selected["checks"], "artifact_sha256": artifact_hash, "decision": decision,
    }, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
