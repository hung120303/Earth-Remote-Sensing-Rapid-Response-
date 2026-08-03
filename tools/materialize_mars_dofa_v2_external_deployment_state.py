#!/usr/bin/env python3
"""Materialize the selected folds-3/4 DOFA-v2 probe for external inference.

This script opens only the already-authorized folds-3/4 DOFA feature cache. It
performs no candidate search and opens no fold-0/1/2, official-test, or external
cohort data. For each of five fixed projection seeds it fits the two opposite-
fold endpoints used by cross-fitting and serializes label-free inference state.
The preregistered external rule is an equal mean in log-odds space across all
ten endpoints.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import scipy.sparse
import sklearn
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.random_projection import SparseRandomProjection
from sklearn.utils.extmath import safe_sparse_dot

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "EarthRemoteSensingRapidResponse", ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from acquire_mars_metadata import sha256  # noqa: E402
from confirm_mars_dofa_v2_projection_ensemble import (  # noqa: E402
    mean_logit_probabilities,
)
from train_mars_dofa_v2_scene_probe import (  # noqa: E402
    FEATURE_WIDTH,
    PROJECTION_DIM,
    select_features,
)

DEFAULT_PROTOCOL = Path("configs/mars_dofa_v2_external_deployment_state_protocol.json")
EPSILON = 1e-4
ALLOWED_FOLDS = (3, 4)


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape)).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def fit_normalization(source: np.ndarray, epsilon: float = EPSILON) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(source, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("Normalization source must be a nonempty matrix")
    center = values.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = np.maximum(
        values.std(axis=0, dtype=np.float64), float(epsilon)
    ).astype(np.float32)
    if not np.isfinite(center).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise RuntimeError("Fitted normalization statistics are invalid")
    return center, scale


def apply_normalization(values: np.ndarray, center: np.ndarray, scale: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[1:] != center.shape:
        raise ValueError("Feature matrix and normalization state do not align")
    result = ((matrix - center) / scale).astype(np.float32)
    if not np.isfinite(result).all():
        raise RuntimeError("Normalized features are not finite")
    return result


def endpoint_probability(endpoint: dict[str, Any], features: np.ndarray) -> np.ndarray:
    raw = apply_normalization(features, endpoint["raw_center"], endpoint["raw_scale"])
    components = endpoint["projection_components"]
    if not scipy.sparse.issparse(components):
        raise ValueError("Projection components must remain sparse")
    projection_model = endpoint.get("projection_model")
    if projection_model is not None:
        projected = projection_model.transform(raw).astype(np.float32)
    else:
        projected = safe_sparse_dot(raw, components.T, dense_output=True).astype(np.float32)
    projected = apply_normalization(
        projected,
        endpoint["projected_center"],
        endpoint["projected_scale"],
    )
    coef = np.asarray(endpoint["classifier_coef"], dtype=np.float64)
    intercept = float(endpoint["classifier_intercept"])
    if coef.shape != (projected.shape[1],):
        raise ValueError("Classifier coefficient width changed")
    classifier_model = endpoint.get("classifier_model")
    if classifier_model is not None:
        probability = classifier_model.predict_proba(projected)[:, 1]
    else:
        decision = safe_sparse_dot(
            projected,
            coef.reshape(-1, 1),
            dense_output=True,
        ).reshape(-1)
        probability = expit(decision + intercept)
    if probability.shape != (features.shape[0],) or not np.isfinite(probability).all():
        raise RuntimeError("Endpoint probabilities are invalid")
    return probability.astype(np.float64)


def external_mean_logit_score(payload: dict[str, Any], features: np.ndarray) -> np.ndarray:
    endpoints = payload.get("endpoints")
    if not isinstance(endpoints, list) or len(endpoints) != 10:
        raise ValueError("External deployment requires exactly ten DOFA endpoints")
    probabilities = [endpoint_probability(endpoint, features) for endpoint in endpoints]
    return mean_logit_probabilities(probabilities)


def fit_endpoint(
    source_features: np.ndarray,
    source_labels: np.ndarray,
    *,
    projection_seed: int,
    held_fold: int,
    c_value: float,
) -> tuple[dict[str, Any], np.ndarray]:
    raw_center, raw_scale = fit_normalization(source_features)
    raw_source = apply_normalization(source_features, raw_center, raw_scale)
    projection = SparseRandomProjection(
        n_components=PROJECTION_DIM,
        density="auto",
        dense_output=True,
        random_state=int(projection_seed),
    )
    projected_source = projection.fit_transform(raw_source).astype(np.float32)
    projected_center, projected_scale = fit_normalization(projected_source)
    normalized_projected = apply_normalization(
        projected_source, projected_center, projected_scale
    )
    positives = int(np.count_nonzero(source_labels == 1))
    negatives = int(np.count_nonzero(source_labels == 0))
    if positives == 0 or negatives == 0:
        raise ValueError("Each fitted endpoint requires both development classes")
    positive_weight = float(np.sqrt(negatives / positives))
    sample_weight = np.where(source_labels == 1, positive_weight, 1.0)
    classifier = LogisticRegression(
        C=float(c_value),
        max_iter=500,
        solver="lbfgs",
        random_state=20260750 + int(held_fold),
    ).fit(normalized_projected, source_labels, sample_weight=sample_weight)
    if classifier.classes_.tolist() != [0, 1] or classifier.coef_.shape != (1, PROJECTION_DIM):
        raise RuntimeError("Unexpected fitted logistic endpoint schema")
    components = projection.components_.tocsr()
    endpoint = {
        "projection_seed": int(projection_seed),
        "held_fold": int(held_fold),
        "fit_fold": int(7 - held_fold),
        "fit_rows": int(source_labels.size),
        "fit_positive_rows": positives,
        "fit_negative_rows": negatives,
        "raw_center": raw_center,
        "raw_scale": raw_scale,
        "projection_components": components,
        "projection_model": projection,
        "projected_center": projected_center,
        "projected_scale": projected_scale,
        "classifier_coef": classifier.coef_[0].astype(np.float64),
        "classifier_intercept": float(classifier.intercept_[0]),
        "classifier_model": classifier,
        "classifier_C": float(c_value),
        "classifier_positive_weight": positive_weight,
        "classifier_random_state": 20260750 + int(held_fold),
    }
    fitted_probability = classifier.predict_proba(normalized_projected)[:, 1]
    replay_probability = endpoint_probability(endpoint, source_features)
    maximum_replay_error = float(np.max(np.abs(fitted_probability - replay_probability)))
    if maximum_replay_error > 1e-7:
        raise RuntimeError(
            "Serialized endpoint exceeds the frozen 1e-7 probability replay tolerance: "
            f"{maximum_replay_error:.3e}"
        )
    endpoint["maximum_fit_probability_replay_error"] = maximum_replay_error
    return endpoint, fitted_probability.astype(np.float64)


def build_deployment_state(
    features: np.ndarray,
    labels: np.ndarray,
    folds: np.ndarray,
    feature_names: np.ndarray,
    *,
    projection_seeds: list[int],
    c_value: float,
    source_bindings: dict[str, Any],
) -> tuple[dict[str, Any], np.ndarray]:
    if set(np.unique(folds).tolist()) != set(ALLOWED_FOLDS):
        raise ValueError("Materialization cache must contain folds 3 and 4 only")
    if features.ndim != 2 or features.shape[0] != labels.size or folds.shape != labels.shape:
        raise ValueError("Development feature rows do not align")
    if len(projection_seeds) != 5 or len(set(projection_seeds)) != 5:
        raise ValueError("Exactly five unique projection seeds are required")
    endpoints: list[dict[str, Any]] = []
    seed_crossfit: list[np.ndarray] = []
    for seed in projection_seeds:
        crossfit = np.empty(labels.shape, dtype=np.float64)
        for held_fold in ALLOWED_FOLDS:
            fit = folds != held_fold
            held = folds == held_fold
            endpoint, _ = fit_endpoint(
                features[fit],
                labels[fit],
                projection_seed=int(seed),
                held_fold=int(held_fold),
                c_value=float(c_value),
            )
            crossfit[held] = endpoint_probability(endpoint, features[held])
            endpoints.append(endpoint)
        if not np.isfinite(crossfit).all():
            raise RuntimeError("Cross-fitted endpoint replay is not finite")
        seed_crossfit.append(crossfit)
    aggregate_crossfit = mean_logit_probabilities(seed_crossfit)
    payload = {
        "schema_version": 1,
        "scope": "development_only_dofa_v2_folds34_external_deployment_state",
        "research_only": True,
        "outcome_blind_external_inference": True,
        "source_folds": list(ALLOWED_FOLDS),
        "feature_set": "change_extreme",
        "feature_width": int(features.shape[1]),
        "feature_names": feature_names.astype(str).tolist(),
        "normalization_mode": "global_train_fitted",
        "normalization_epsilon": EPSILON,
        "projection_dimension": PROJECTION_DIM,
        "projection_seeds": [int(value) for value in projection_seeds],
        "regularization_C": float(c_value),
        "endpoint_count": len(endpoints),
        "external_aggregation": (
            "equal arithmetic mean of all ten endpoint logits, then sigmoid; "
            "five fixed projection seeds x two opposite-fold fits"
        ),
        "endpoints": endpoints,
        "source_bindings": source_bindings,
        "runtime": {
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }
    return payload, aggregate_crossfit


def validate_protocol(protocol: dict[str, Any]) -> None:
    expected = {
        "feature_set": "change_extreme",
        "C": 0.01,
        "projection_dimension": 2048,
        "projection_seeds": [20260780, 20260781, 20260782, 20260783, 20260784],
        "normalization_mode": "global_train_fitted",
        "folds": [3, 4],
        "endpoint_aggregation": "equal_mean_logit_all_10_endpoints",
    }
    if protocol.get("deployment_rule") != expected:
        raise ValueError("External DOFA deployment rule differs from frozen protocol")
    if protocol.get("forbidden_access") != {
        "folds": [0, 1, 2],
        "official_test": True,
        "external_cohort": True,
    }:
        raise ValueError("Forbidden-access contract changed")


def write_joblib(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    joblib.dump(payload, temporary, compress=3)
    os.replace(temporary, path)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL.as_posix())
    args = parser.parse_args()
    protocol_path = (ROOT / args.protocol).resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    validate_protocol(protocol)
    script_path = Path(__file__).resolve()
    if sha256(script_path) != protocol["materializer"]["sha256"]:
        raise ValueError("Deployment-state materializer hash mismatch")
    resolved_inputs: dict[str, Path] = {}
    for name, binding in protocol["inputs"].items():
        path = (ROOT / binding["path"]).resolve()
        if sha256(path) != binding["sha256"]:
            raise ValueError(f"Frozen deployment-state input hash mismatch: {name}")
        resolved_inputs[name] = path
    selected_result = json.loads(
        resolved_inputs["selected_result"].read_text(encoding="utf-8")
    )
    if not selected_result["all_promotion_gates_pass"]:
        raise ValueError("Selected train-fitted DOFA result did not pass")
    if selected_result["selected"]["normalization_mode"] != "global_train_fitted":
        raise ValueError("Selected normalization mode changed")
    fixed = selected_result["fixed_candidate"]
    rule = protocol["deployment_rule"]
    if any(
        (
            fixed["feature_set"] != rule["feature_set"],
            float(fixed["C"]) != float(rule["C"]),
            int(fixed["projection_dimension"]) != int(rule["projection_dimension"]),
            list(map(int, fixed["projection_seeds"])) != list(map(int, rule["projection_seeds"])),
        )
    ):
        raise ValueError("Selected result differs from deployment rule")
    with np.load(resolved_inputs["dofa_folds34_features"], allow_pickle=False) as cache:
        encoded = cache["features"].astype(np.float32)
        names = cache["feature_names"].astype(str)
        labels = cache["labels"].astype(np.uint8)
        folds = cache["folds"].astype(np.uint8)
        sensors = cache["sensors"].astype(np.uint8)
        sample_ids = cache["sample_ids"].astype(str)
    if encoded.ndim != 2 or encoded.shape[1] != FEATURE_WIDTH:
        raise ValueError("DOFA feature cache schema changed")
    if set(np.unique(folds).tolist()) != set(ALLOWED_FOLDS):
        raise ValueError("DOFA deployment materialization may use only folds 3/4")
    if len(set(sample_ids.tolist())) != sample_ids.size:
        raise ValueError("Development feature IDs are not unique")
    if set(np.unique(sensors).tolist()) != {0, 1}:
        raise ValueError("Expected the frozen two-sensor development contract")
    selected_features, selected_names = select_features(encoded, names, str(rule["feature_set"]))
    source_bindings = {
        name: {"path": binding["path"], "sha256": binding["sha256"]}
        for name, binding in protocol["inputs"].items()
    }
    payload, crossfit = build_deployment_state(
        selected_features,
        labels,
        folds,
        selected_names,
        projection_seeds=list(map(int, rule["projection_seeds"])),
        c_value=float(rule["C"]),
        source_bindings=source_bindings,
    )
    output_path = (ROOT / protocol["outputs"]["state"]).resolve()
    write_joblib(output_path, payload)
    loaded = joblib.load(output_path)
    external_probe = external_mean_logit_score(loaded, selected_features[:3])
    if external_probe.shape != (3,) or not np.isfinite(external_probe).all():
        raise RuntimeError("Saved external deployment state failed replay")
    report = {
        "schema_version": 1,
        "status": "materialized_before_external_or_stanford_access",
        "scope": payload["scope"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": protocol_path.relative_to(ROOT).as_posix(),
        "protocol_sha256": sha256(protocol_path),
        "script_sha256": sha256(script_path),
        "state": {
            "path": output_path.relative_to(ROOT).as_posix(),
            "bytes": output_path.stat().st_size,
            "sha256": sha256(output_path),
            "tracked": False,
            "endpoint_count": len(payload["endpoints"]),
        },
        "deployment_rule": rule,
        "verification": {
            "source_rows": int(labels.size),
            "fold_counts": {
                str(fold): int(np.count_nonzero(folds == fold)) for fold in ALLOWED_FOLDS
            },
            "crossfit_probability_sha256": array_sha256(crossfit.astype(np.float64)),
            "serialized_endpoint_replay": True,
            "external_ten_endpoint_smoke": True,
        },
        "access": {
            "opened_folds": [3, 4],
            "folds_0_1_2_accessed": False,
            "official_test_accessed": False,
            "external_or_stanford_features_accessed": False,
            "external_or_stanford_outcomes_accessed": False,
        },
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
    }
    report_path = (ROOT / protocol["outputs"]["json"]).resolve()
    write_json(report_path, report)
    print(
        json.dumps(
            {
                "ok": True,
                "state": report["state"],
                "opened_folds": report["access"]["opened_folds"],
                "external_or_stanford_accessed": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
