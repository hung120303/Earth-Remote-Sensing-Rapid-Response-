#!/usr/bin/env python3
"""Train and confirm a cross-fitted meta-model over frozen MARS scene heads."""

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
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "EarthRemoteSensingRapidResponse"
for path in (MODEL_ROOT, ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from acquire_mars_metadata import repo_root, sha256  # noqa: E402
from train_mars_oof_scene_ensemble_v2 import (  # noqa: E402
    ap_group_bootstrap,
    sample_weights,
)
from train_mars_scene_ranker import (  # noqa: E402
    comparison,
    metric_summary,
    safe_logit,
)

DEFAULT_CACHE = Path("outputs/mars_scene_domain_routing_development_scores.npz")
DEFAULT_CACHE_SHA256 = "fd955b78b26a3b2a5165b4abab02180ccf4dad433511bf4da7afbff44275c1c7"
DEFAULT_ARTIFACT = Path("EarthRemoteSensingRapidResponse/artifacts/mars_scene_stacker_v3.joblib")
DEFAULT_JSON = Path("reports/experiments/mars_scene_stacker_v3.json")
DEFAULT_MARKDOWN = Path("reports/experiments/MARS_SCENE_STACKER_V3.md")
INNER_FOLDS = (2, 3, 4)


def stack_features(
    primary: np.ndarray,
    legacy: np.ndarray,
    new: np.ndarray,
    sensors: np.ndarray,
    offshore: np.ndarray,
    feature_set: str,
) -> tuple[np.ndarray, list[str]]:
    arrays = (primary, legacy, new, sensors, offshore)
    if any(value.shape != primary.shape for value in arrays):
        raise ValueError("stacker arrays must align")
    logits = np.column_stack([safe_logit(primary), safe_logit(legacy), safe_logit(new)])
    names = ["primary_logit", "legacy_logit", "new_logit"]
    if feature_set == "scores":
        return logits, names
    landsat = (sensors == 1).astype(np.float64)[:, None]
    values = [logits, landsat, landsat * logits]
    names.extend(["is_landsat", *[f"landsat_x_{name}" for name in names[:3]]])
    if feature_set == "sensor":
        return np.column_stack(values), names
    if feature_set != "sensor_domain":
        raise ValueError(f"Unknown stacker feature set: {feature_set}")
    domain = offshore.astype(np.float64)[:, None]
    values.extend([domain, domain * logits, domain * landsat])
    names.extend(
        ["is_offshore", *[f"offshore_x_{name}" for name in names[:3]], "offshore_x_landsat"]
    )
    return np.column_stack(values), names


def candidate_specs() -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for feature_set in ("scores", "sensor", "sensor_domain"):
        for weighting in ("uniform", "group", "site_cell"):
            for regularization in (0.01, 0.1, 1.0, 10.0):
                specs.append(
                    {
                        "family": "logistic",
                        "feature_set": feature_set,
                        "weighting": weighting,
                        "C": regularization,
                    }
                )
    for weighting in ("uniform", "group", "site_cell"):
        for leaves, minimum, regularization in (
            (3, 100, 1.0),
            (7, 100, 1.0),
            (7, 250, 10.0),
            (15, 250, 10.0),
        ):
            specs.append(
                {
                    "family": "hist_gradient_boosting",
                    "feature_set": "sensor_domain",
                    "weighting": weighting,
                    "max_leaf_nodes": leaves,
                    "min_samples_leaf": minimum,
                    "l2_regularization": regularization,
                }
            )
    return specs


def spec_key(spec: dict[str, Any]) -> str:
    return "_".join(f"{key}-{spec[key]}" for key in sorted(spec))


def fit_model(
    spec: dict[str, Any], features: np.ndarray, labels: np.ndarray, weights: np.ndarray
) -> dict[str, Any]:
    if spec["family"] == "logistic":
        scaler = StandardScaler().fit(features, sample_weight=weights)
        model = LogisticRegression(
            C=float(spec["C"]),
            solver="lbfgs",
            max_iter=2000,
            random_state=20260715,
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
            random_state=20260715,
        )
        model.fit(features, labels, sample_weight=weights)
        return {"scaler": None, "model": model}
    raise ValueError(f"Unknown stacker family: {spec['family']}")


def predict_model(fitted: dict[str, Any], features: np.ndarray) -> np.ndarray:
    scaler = fitted["scaler"]
    transformed = features if scaler is None else scaler.transform(features)
    return fitted["model"].predict_proba(transformed)[:, 1]


def load_partition(cache: Any, prefix: str) -> dict[str, np.ndarray]:
    return {
        "labels": cache[f"{prefix}_labels"].astype(np.uint8),
        "sensors": cache[f"{prefix}_sensors"].astype(np.uint8),
        "groups": cache[f"{prefix}_groups"].astype(str),
        "offshore": cache[f"{prefix}_offshore"].astype(bool),
        "primary": cache[f"{prefix}_primary"].astype(np.float64),
        "legacy": cache[f"{prefix}_legacy"].astype(np.float64),
        "new": cache[f"{prefix}_new"].astype(np.float64),
    }


def metric_comparison(
    partition: dict[str, np.ndarray], scores: np.ndarray, reference_name: str
) -> dict[str, Any]:
    candidate = metric_summary(partition["labels"], scores, partition["sensors"])
    reference = metric_summary(
        partition["labels"], partition[reference_name], partition["sensors"]
    )
    return comparison(candidate, reference)


def crossfit_candidate(
    spec: dict[str, Any], partition: dict[str, np.ndarray]
) -> tuple[np.ndarray, list[str]]:
    features, names = stack_features(
        partition["primary"],
        partition["legacy"],
        partition["new"],
        partition["sensors"],
        partition["offshore"],
        str(spec["feature_set"]),
    )
    folds = partition["folds"]
    scores = np.empty(partition["labels"].shape, dtype=np.float64)
    for holdout in INNER_FOLDS:
        fit_rows = folds != holdout
        held_rows = folds == holdout
        weights = sample_weights(
            str(spec["weighting"]),
            partition["groups"][fit_rows],
            partition["labels"][fit_rows],
            partition["sensors"][fit_rows],
        )
        fitted = fit_model(spec, features[fit_rows], partition["labels"][fit_rows], weights)
        scores[held_rows] = predict_model(fitted, features[held_rows])
    return scores, names


def screen_candidate(
    spec: dict[str, Any], partition: dict[str, np.ndarray], scores: np.ndarray
) -> dict[str, Any]:
    versus_primary = metric_comparison(partition, scores, "primary")
    versus_new = metric_comparison(partition, scores, "new")
    per_fold: dict[str, Any] = {}
    for fold in INNER_FOLDS:
        rows = partition["folds"] == fold
        local = {name: value[rows] for name, value in partition.items() if name != "folds"}
        per_fold[str(fold)] = {
            "versus_primary": metric_comparison(local, scores[rows], "primary"),
            "versus_new": metric_comparison(local, scores[rows], "new"),
        }
    primary_fold_ap = [
        value["versus_primary"]["delta"]["average_precision"] for value in per_fold.values()
    ]
    new_fold_ap = [
        value["versus_new"]["delta"]["average_precision"] for value in per_fold.values()
    ]
    stable = (
        versus_primary["delta"]["average_precision"] > 0.0
        and versus_primary["delta"]["recall_at_fpr_0_0713"] >= 0.0
        and min(primary_fold_ap) > 0.0
        and min(versus_primary["delta"]["sensor_average_precision"].values()) >= -0.005
        and versus_new["delta"]["average_precision"] > 0.0
        and versus_new["delta"]["recall_at_fpr_0_0713"] >= -0.0025
        and min(new_fold_ap) >= -0.005
        and min(versus_new["delta"]["sensor_average_precision"].values()) >= -0.005
    )
    rank = [
        int(stable),
        min(new_fold_ap),
        versus_new["delta"]["average_precision"],
        versus_new["delta"]["recall_at_fpr_0_0713"],
    ]
    return {
        "spec": spec,
        "spec_key": spec_key(spec),
        "stable": stable,
        "versus_primary": versus_primary,
        "versus_new": versus_new,
        "per_fold": per_fold,
        "rank": rank,
    }


def confirm_partition(
    partition: dict[str, np.ndarray], scores: np.ndarray, *, seed: int
) -> dict[str, Any]:
    versus_primary = metric_comparison(partition, scores, "primary")
    versus_new = metric_comparison(partition, scores, "new")
    bootstrap_primary = ap_group_bootstrap(
        partition["labels"],
        partition["primary"],
        scores,
        partition["groups"],
        replicates=10000,
        seed=seed,
    )
    bootstrap_new = ap_group_bootstrap(
        partition["labels"],
        partition["new"],
        scores,
        partition["groups"],
        replicates=10000,
        seed=seed + 100,
    )
    checks = {
        "ap_higher_than_primary": versus_primary["delta"]["average_precision"] > 0.0,
        "recall_higher_than_primary": versus_primary["delta"]["recall_at_fpr_0_0713"] > 0.0,
        "ap_ci_lower_positive_vs_primary": bootstrap_primary["lower"] > 0.0,
        "no_material_sensor_regression_vs_primary": min(
            versus_primary["delta"]["sensor_average_precision"].values()
        )
        >= -0.005,
        "no_material_ap_regression_vs_new": versus_new["delta"]["average_precision"]
        >= -0.005,
    }
    return {
        "rows": int(partition["labels"].size),
        "positive": int(partition["labels"].sum()),
        "sites": len(set(partition["groups"].tolist())),
        "versus_primary": versus_primary,
        "versus_new": versus_new,
        "paired_group_bootstrap_ap_delta_vs_primary": bootstrap_primary,
        "paired_group_bootstrap_ap_delta_vs_new": bootstrap_new,
        "checks": checks,
        "passed": all(checks.values()),
    }


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    selected = report["selected"]
    lines = [
        "# Cross-fitted MARS scene stacker v3",
        "",
        "The stacker was selected only from OOF folds 2/3/4, then frozen before folds 0/1 were scored.",
        "",
        f"- Selected model: `{selected['spec_key']}`",
        f"- Inner AP delta vs primary: {selected['versus_primary']['delta']['average_precision']:+.5f}",
        f"- Inner AP delta vs stronger head: {selected['versus_new']['delta']['average_precision']:+.5f}",
        f"- Inner AP interval vs stronger head: [{selected['paired_group_bootstrap_ap_delta_vs_new']['lower']:+.5f}, {selected['paired_group_bootstrap_ap_delta_vs_new']['upper']:+.5f}]",
        "",
        "| Partition | AP delta vs primary | Recall delta | AP 95% CI | AP delta vs new | Gates |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for name, value in report["confirmation"].items():
        primary = value["versus_primary"]["delta"]
        ci = value["paired_group_bootstrap_ap_delta_vs_primary"]
        lines.append(
            f"| {name} | {primary['average_precision']:+.5f} | "
            f"{primary['recall_at_fpr_0_0713']:+.5f} | [{ci['lower']:+.5f}, {ci['upper']:+.5f}] | "
            f"{value['versus_new']['delta']['average_precision']:+.5f} | "
            f"{'PASS' if value['passed'] else 'FAIL'} |"
        )
    lines.extend(["", report["decision"]])
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
        raise ValueError("Frozen development score-cache hash mismatch")
    with np.load(cache_path, allow_pickle=False) as cache:
        inner = load_partition(cache, "inner")
        inner["folds"] = cache["inner_folds"].astype(np.uint8)
        held = {name: load_partition(cache, name) for name in ("fold0", "fold1")}
    if set(np.unique(inner["folds"]).tolist()) != set(INNER_FOLDS):
        raise ValueError("Inner score cache must contain only folds 2, 3, and 4")

    candidates: list[dict[str, Any]] = []
    score_by_key: dict[str, np.ndarray] = {}
    feature_names_by_key: dict[str, list[str]] = {}
    for spec in candidate_specs():
        scores, names = crossfit_candidate(spec, inner)
        result = screen_candidate(spec, inner, scores)
        candidates.append(result)
        score_by_key[result["spec_key"]] = scores
        feature_names_by_key[result["spec_key"]] = names
    screened = sorted(candidates, key=lambda value: tuple(value["rank"]), reverse=True)[:5]
    for index, candidate in enumerate(screened):
        scores = score_by_key[candidate["spec_key"]]
        candidate["paired_group_bootstrap_ap_delta_vs_new"] = ap_group_bootstrap(
            inner["labels"],
            inner["new"],
            scores,
            inner["groups"],
            replicates=3000,
            seed=20260720 + index,
        )
    selected = max(
        screened,
        key=lambda value: (
            value["stable"],
            value["paired_group_bootstrap_ap_delta_vs_new"]["lower"],
            *value["rank"][1:],
        ),
    )
    selected_scores = score_by_key[selected["spec_key"]]
    selected["paired_group_bootstrap_ap_delta_vs_primary"] = ap_group_bootstrap(
        inner["labels"],
        inner["primary"],
        selected_scores,
        inner["groups"],
        replicates=10000,
        seed=20260725,
    )
    selected["paired_group_bootstrap_ap_delta_vs_new"] = ap_group_bootstrap(
        inner["labels"],
        inner["new"],
        selected_scores,
        inner["groups"],
        replicates=10000,
        seed=20260726,
    )
    selected["inner_passed"] = bool(
        selected["stable"]
        and selected["paired_group_bootstrap_ap_delta_vs_primary"]["lower"] > 0.0
        and selected["paired_group_bootstrap_ap_delta_vs_new"]["lower"] > -0.0025
    )

    selected_spec = selected["spec"]
    inner_features, feature_names = stack_features(
        inner["primary"],
        inner["legacy"],
        inner["new"],
        inner["sensors"],
        inner["offshore"],
        str(selected_spec["feature_set"]),
    )
    final_weights = sample_weights(
        str(selected_spec["weighting"]),
        inner["groups"],
        inner["labels"],
        inner["sensors"],
    )
    fitted = fit_model(selected_spec, inner_features, inner["labels"], final_weights)
    confirmation: dict[str, Any] = {}
    thresholds: list[float] = []
    for index, (name, partition) in enumerate(held.items()):
        features, local_names = stack_features(
            partition["primary"],
            partition["legacy"],
            partition["new"],
            partition["sensors"],
            partition["offshore"],
            str(selected_spec["feature_set"]),
        )
        if local_names != feature_names:
            raise ValueError("Held stacker feature schema mismatch")
        scores = predict_model(fitted, features)
        confirmation[name] = confirm_partition(partition, scores, seed=20260730 + index)
        thresholds.append(
            confirmation[name]["versus_primary"]["metrics"]["operating_point"]["threshold"]
        )

    artifact_path = (root / args.artifact).resolve()
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = artifact_path.with_suffix(artifact_path.suffix + ".tmp")
    payload = {
        "schema_version": 1,
        "kind": "mars_cross_fitted_scene_stacker_v3",
        "spec": selected_spec,
        "feature_names": feature_names,
        "fitted": fitted,
        "source_score_cache_sha256": args.cache_sha256,
        "operational_scene_threshold": max(thresholds),
    }
    joblib.dump(payload, temporary, compress=3)
    os.replace(temporary, artifact_path)
    passed = selected["inner_passed"] and all(value["passed"] for value in confirmation.values())
    report = {
        "schema_version": 1,
        "scope": "development only; paper-test labels and scores are not loaded",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_count": len(candidates),
        "selection_partition": "cross-fitted folds 2/3/4",
        "selected": selected,
        "top_screened": screened,
        "confirmation": confirmation,
        "operational_scene_threshold": max(thresholds),
        "all_promotion_gates_pass": passed,
        "decision": (
            "Freeze the cross-fitted stacker for a transparent post-test paper benchmark."
            if passed
            else "Reject the stacker before any paper-test evaluation."
        ),
        "provenance": {
            "development_score_cache_sha256": args.cache_sha256,
            "artifact_sha256": sha256(artifact_path),
            "script_sha256": sha256(Path(__file__).resolve()),
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True
            ).strip(),
            "sklearn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
    }
    write_json((root / args.output_json).resolve(), report)
    write_markdown((root / args.output_markdown).resolve(), report)
    print(
        json.dumps(
            {
                "ok": passed,
                "selected": selected["spec_key"],
                "inner_ap_delta_vs_primary": selected["versus_primary"]["delta"][
                    "average_precision"
                ],
                "inner_ap_delta_vs_new": selected["versus_new"]["delta"][
                    "average_precision"
                ],
                "confirmations": {
                    name: {
                        "passed": value["passed"],
                        "ap_delta_vs_primary": value["versus_primary"]["delta"][
                            "average_precision"
                        ],
                        "ap_delta_vs_new": value["versus_new"]["delta"]["average_precision"],
                    }
                    for name, value in confirmation.items()
                },
            },
            indent=2,
        )
    )
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
