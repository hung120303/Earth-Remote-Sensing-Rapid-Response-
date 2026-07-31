#!/usr/bin/env python3
"""Cross-fit a site-balanced sparse linear ranker on invariant dense features."""

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

import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "EarthRemoteSensingRapidResponse"
for path in (MODEL_ROOT, ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from acquire_mars_metadata import sha256  # noqa: E402
from explore_mars_dense_invariant_univariate import (  # noqa: E402
    ap_views,
    rank_normalize_by_domain,
)
from explore_mars_dense_sparse_invariant_ensemble import (  # noqa: E402
    evaluate_views,
    point_gates,
)
from train_mars_dense_ap_residual_ranker import (  # noqa: E402
    load_cache,
    logit,
    sigmoid,
)
from train_mars_oof_scene_ensemble_v2 import ap_group_bootstrap  # noqa: E402
from train_mars_scene_ranker import metric_summary  # noqa: E402
from train_mars_unseen_low_prevalence_router import (  # noqa: E402
    low_prevalence_mask,
)


DEFAULT_PROTOCOL = Path(
    "configs/mars_dense_invariant_site_linear_ranker_protocol.json"
)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def site_balanced_weights(
    labels: np.ndarray,
    sensors: np.ndarray,
    groups: np.ndarray,
    mode: str,
) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.uint8)
    sensors = np.asarray(sensors, dtype=np.uint8)
    groups = np.asarray(groups).astype(str)
    weights = np.zeros(labels.size, dtype=np.float64)
    for group in np.unique(groups):
        rows = np.flatnonzero(groups == group)
        if mode == "site_equal":
            weights[rows] = 1.0 / rows.size
            continue
        if mode == "site_label_equal":
            keys = [str(int(labels[index])) for index in rows]
        elif mode == "site_sensor_label_equal":
            keys = [
                f"{int(sensors[index])}|{int(labels[index])}"
                for index in rows
            ]
        else:
            raise ValueError(f"Unknown site weighting mode: {mode}")
        counts = Counter(keys)
        cell_count = len(counts)
        for index, key in zip(rows, keys):
            weights[index] = 1.0 / (cell_count * counts[key])
    if np.any(weights <= 0.0) or not np.isfinite(weights).all():
        raise ValueError("Site-balanced weights are not finite and positive")
    return weights / weights.mean()


def model_specs(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "C": float(c_value),
            "weighting": str(weighting),
        }
        for weighting in protocol["models"]["weighting"]
        for c_value in protocol["models"]["C"]
    ]


def model_key(spec: dict[str, Any]) -> str:
    return f"{spec['weighting']}_C{float(spec['C']):g}"


def fit_crossfold_head(
    features: np.ndarray,
    labels: np.ndarray,
    sensors: np.ndarray,
    groups: np.ndarray,
    folds: np.ndarray,
    spec: dict[str, Any],
    protocol: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    unique_folds = sorted(map(int, np.unique(folds)))
    if len(unique_folds) != 2:
        raise ValueError("Invariant site-linear ranker requires exactly two folds")
    logits = np.full(labels.size, np.nan, dtype=np.float64)
    coefficients: dict[int, np.ndarray] = {}
    fits: dict[str, Any] = {}
    for held in unique_folds:
        fit = unique_folds[1] if held == unique_folds[0] else unique_folds[0]
        fit_rows = folds == fit
        held_rows = folds == held
        weights = site_balanced_weights(
            labels[fit_rows],
            sensors[fit_rows],
            groups[fit_rows],
            str(spec["weighting"]),
        )
        model = LogisticRegression(
            l1_ratio=1.0,
            C=float(spec["C"]),
            solver="liblinear",
            fit_intercept=True,
            intercept_scaling=float(protocol["models"]["intercept_scaling"]),
            max_iter=int(protocol["models"]["max_iter"]),
            tol=float(protocol["models"]["tolerance"]),
            random_state=int(protocol["models"]["seed"]) + held,
        )
        model.fit(
            features[fit_rows],
            labels[fit_rows],
            sample_weight=weights,
        )
        logits[held_rows] = model.decision_function(features[held_rows])
        coefficient = np.asarray(model.coef_[0], dtype=np.float64)
        coefficients[held] = coefficient
        fits[str(held)] = {
            "fit_fold": fit,
            "held_fold": held,
            "fit_rows": int(np.count_nonzero(fit_rows)),
            "fit_positive": int(labels[fit_rows].sum()),
            "fit_sites": int(len(np.unique(groups[fit_rows]))),
            "held_rows": int(np.count_nonzero(held_rows)),
            "nonzero_coefficients": int(np.count_nonzero(coefficient)),
            "iterations": int(model.n_iter_[0]),
            "intercept": float(model.intercept_[0]),
        }
    if not np.isfinite(logits).all():
        raise ValueError("Cross-fitted invariant head logits are incomplete")
    normalized_logits = rank_normalize_by_domain(logits, folds, sensors)
    first = coefficients[unique_folds[0]]
    second = coefficients[unique_folds[1]]
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    cosine = (
        0.0
        if denominator == 0.0
        else float(np.dot(first, second) / denominator)
    )
    nonzero_union = (first != 0.0) | (second != 0.0)
    sign_agreement = (
        0.0
        if not np.any(nonzero_union)
        else float(
            np.mean(
                np.sign(first[nonzero_union])
                == np.sign(second[nonzero_union])
            )
        )
    )
    diagnostics = {
        "fits": fits,
        "coefficient_cosine": cosine,
        "nonzero_union": int(np.count_nonzero(nonzero_union)),
        "nonzero_intersection": int(
            np.count_nonzero((first != 0.0) & (second != 0.0))
        ),
        "union_sign_agreement": sign_agreement,
        "coefficients_by_held_fold": coefficients,
    }
    return normalized_logits, diagnostics


def compact_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "key": candidate["key"],
        "model_key": candidate["model_key"],
        "spec": candidate["spec"],
        "strength": candidate["strength"],
        "whole_average_precision_delta": candidate["whole"][
            "average_precision_delta"
        ],
        "whole_matched_fpr_recall_delta": candidate["whole"][
            "matched_fpr_recall_delta"
        ],
        "low_prevalence_average_precision_delta": candidate[
            "low_prevalence"
        ]["average_precision_delta"],
        "low_prevalence_matched_fpr_recall_delta": candidate[
            "low_prevalence"
        ]["matched_fpr_recall_delta"],
        "point_checks": candidate["point_checks"],
        "passed_point_gates": candidate["passed_point_gates"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL.as_posix())
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    protocol_path = (ROOT / args.protocol).resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if sha256(Path(__file__).resolve()) != protocol["trainer"]["sha256"]:
        raise ValueError("Frozen invariant site-linear trainer hash mismatch")
    for dependency in protocol["code_dependencies"]:
        path = (ROOT / dependency["path"]).resolve()
        if sha256(path) != dependency["sha256"]:
            raise ValueError(f"Frozen dependency mismatch: {dependency['path']}")
    paths: dict[str, Path] = {}
    for name, contract in protocol["inputs"].items():
        path = (ROOT / contract["path"]).resolve()
        if sha256(path) != contract["sha256"]:
            raise ValueError(f"Frozen input mismatch: {name}")
        paths[name] = path

    matrix, metadata, _columns = load_cache(paths["features"], paths["metadata"])
    labels = metadata["labels"].astype(np.uint8)
    sensors = metadata["sensors"].astype(np.uint8)
    groups = metadata["groups"].astype(str)
    folds = metadata["folds"].astype(np.uint8)
    base_scores = metadata["exact_base_scores"].astype(np.float64)
    names = metadata["feature_names"].astype(str)
    expected_folds = sorted(map(int, protocol["folds"]))
    if sorted(map(int, np.unique(folds))) != expected_folds:
        raise ValueError("Invariant site-linear cache contains unexpected folds")
    feature_indices = list(
        range(
            int(protocol["feature_range"]["start_inclusive"]),
            int(protocol["feature_range"]["end_exclusive"]),
        )
    )
    if args.smoke:
        feature_indices = feature_indices[:32]
    normalized = np.empty(
        (labels.size, len(feature_indices)),
        dtype=np.float32,
    )
    for output_index, feature_index in enumerate(feature_indices):
        normalized[:, output_index] = rank_normalize_by_domain(
            matrix[:, feature_index],
            folds,
            sensors,
        ).astype(np.float32)
    if not np.isfinite(normalized).all():
        raise ValueError("Invariant site-linear feature matrix is non-finite")

    maximum_rate = float(
        protocol["target_domain"]["maximum_site_positive_rate"]
    )
    low_rows = low_prevalence_mask(labels, groups, maximum_rate)
    all_rows = np.ones(labels.size, dtype=bool)
    base_logit = logit(base_scores)
    specs = model_specs(protocol)
    strengths = [float(value) for value in protocol["strengths"]]
    if args.smoke:
        specs = specs[:1]
        strengths = strengths[:2]

    model_diagnostics: dict[str, dict[str, Any]] = {}
    candidates: list[dict[str, Any]] = []
    scores_by_key: dict[str, np.ndarray] = {}
    for spec in specs:
        key = model_key(spec)
        head, diagnostics = fit_crossfold_head(
            normalized,
            labels,
            sensors,
            groups,
            folds,
            spec,
            protocol,
        )
        coefficients = diagnostics.pop("coefficients_by_held_fold")
        mean_absolute = np.mean(
            np.column_stack(
                [np.abs(value) for value in coefficients.values()]
            ),
            axis=1,
        )
        top = np.argsort(mean_absolute)[::-1][:20]
        diagnostics["top_absolute_features"] = [
            {
                "matrix_index": int(feature_indices[index]),
                "feature_name": str(names[feature_indices[index]]),
                "mean_absolute_coefficient": float(mean_absolute[index]),
                "coefficients_by_held_fold": {
                    str(held): float(values[index])
                    for held, values in coefficients.items()
                },
            }
            for index in top
            if mean_absolute[index] > 0.0
        ]
        model_diagnostics[key] = diagnostics
        for strength in strengths:
            scores = sigmoid(base_logit + strength * head)
            candidate_key = f"{key}_strength{strength:g}"
            scores_by_key[candidate_key] = scores
            whole = evaluate_views(
                labels,
                base_scores,
                scores,
                folds,
                sensors,
                all_rows,
            )
            low = evaluate_views(
                labels,
                base_scores,
                scores,
                folds,
                sensors,
                low_rows,
            )
            checks = point_gates(whole, low, protocol)
            candidates.append(
                {
                    "key": candidate_key,
                    "model_key": key,
                    "spec": spec,
                    "strength": strength,
                    "whole": whole,
                    "low_prevalence": low,
                    "point_checks": checks,
                    "passed_point_gates": all(checks.values()),
                    "rank": [
                        int(all(checks.values())),
                        min(
                            low["average_precision_delta"]["folds"].values()
                        ),
                        min(
                            low["average_precision_delta"]["sensors"].values()
                        ),
                        low["average_precision_delta"]["pooled"],
                        min(
                            whole["average_precision_delta"]["folds"].values()
                        ),
                        whole["average_precision_delta"]["pooled"],
                        diagnostics["coefficient_cosine"],
                        -strength,
                    ],
                }
            )

    candidates.sort(key=lambda row: tuple(row["rank"]), reverse=True)
    point_survivors = [
        row for row in candidates if row["passed_point_gates"]
    ]
    bootstrap_pool = point_survivors
    if args.smoke and not bootstrap_pool:
        bootstrap_pool = candidates
    bootstrap_top = min(
        len(bootstrap_pool),
        2 if args.smoke else int(protocol["bootstrap"]["top_k"]),
    )
    evaluated: list[dict[str, Any]] = []
    for rank_index, candidate in enumerate(bootstrap_pool[:bootstrap_top]):
        scores = scores_by_key[str(candidate["key"])]
        replicates = (
            100
            if args.smoke
            else int(protocol["bootstrap"]["replicates"])
        )
        whole_interval = ap_group_bootstrap(
            labels,
            base_scores,
            scores,
            groups,
            replicates=replicates,
            seed=int(protocol["bootstrap"]["whole_seed"]) + rank_index,
        )
        low_interval = ap_group_bootstrap(
            labels[low_rows],
            base_scores[low_rows],
            scores[low_rows],
            groups[low_rows],
            replicates=replicates,
            seed=int(protocol["bootstrap"]["low_prevalence_seed"])
            + rank_index,
        )
        bootstrap_checks = {
            "whole_paired_site_ap_lower_strictly_positive": bool(
                whole_interval["lower"] > 0.0
            ),
            "low_prevalence_paired_site_ap_lower_strictly_positive": bool(
                low_interval["lower"] > 0.0
            ),
        }
        evaluated.append(
            {
                **candidate,
                "paired_site_ap_delta": {
                    "whole": whole_interval,
                    "low_prevalence": low_interval,
                },
                "bootstrap_checks": bootstrap_checks,
                "passed": bool(
                    candidate["passed_point_gates"]
                    and all(bootstrap_checks.values())
                ),
            }
        )

    passed = [row for row in evaluated if row["passed"]]
    selected = (
        max(passed, key=lambda row: tuple(row["rank"])) if passed else None
    )
    summary = {
        "features": len(feature_indices),
        "models": len(specs),
        "candidate_count": len(candidates),
        "point_gate_candidates": len(point_survivors),
        "bootstrapped_candidates": len(evaluated),
        "passing_candidates": len(passed),
        "selected": (
            None
            if selected is None
            else {
                key: value
                for key, value in selected.items()
                if key != "rank"
            }
        ),
    }
    if args.smoke:
        print(json.dumps({"ok": True, **summary}, indent=2))
        return 0

    baseline_ap = ap_views(labels, base_scores, folds, sensors)
    baseline_metrics = metric_summary(labels, base_scores, sensors)
    low_ap = ap_views(
        labels[low_rows],
        base_scores[low_rows],
        folds[low_rows],
        sensors[low_rows],
    )
    low_metrics = metric_summary(
        labels[low_rows],
        base_scores[low_rows],
        sensors[low_rows],
    )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    report = {
        "schema_version": 1,
        "scope": (
            "exploratory cross-fitted invariant site-balanced sparse linear "
            "ranker on honest dense-Prithvi development features"
        ),
        "status": (
            "candidate_found_for_preregistered_fold2_test"
            if selected is not None
            else "rejected_before_fold2"
        ),
        "decision": (
            "Freeze the selected invariant linear ranker for fold-2 confirmation."
            if selected is not None
            else (
                "Retire this invariant site-linear ranker family before fold 2 "
                "or external access."
            )
        ),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "provenance": {
            "git_commit": commit,
            "protocol_sha256": sha256(protocol_path),
            "trainer_sha256": sha256(Path(__file__).resolve()),
            "features_sha256": sha256(paths["features"]),
            "metadata_sha256": sha256(paths["metadata"]),
            "sparse_ensemble_report_sha256": sha256(
                paths["sparse_ensemble_report"]
            ),
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
        },
        "baseline": {
            "whole": {
                "average_precision": baseline_ap,
                "metrics": baseline_metrics,
            },
            "low_prevalence": {
                "average_precision": low_ap,
                "metrics": low_metrics,
            },
        },
        "target_domain": {
            "definition": protocol["target_domain"],
            "rows": int(np.count_nonzero(low_rows)),
            "positive": int(labels[low_rows].sum()),
            "sites": int(len(np.unique(groups[low_rows]))),
        },
        "model_diagnostics": model_diagnostics,
        "screening_candidates": [
            compact_candidate(row)
            for row in candidates[
                : int(protocol["reporting"]["top_screening_candidates"])
            ]
        ],
        "summary": summary,
        "evaluated_candidates": [
            {
                key: value
                for key, value in row.items()
                if key != "rank"
            }
            for row in evaluated
        ],
        "fold2_accessed": False,
        "fresh_inputs_accessed": False,
        "exact_paper_inputs_accessed": False,
    }
    write_json((ROOT / protocol["output"]).resolve(), report)
    print(json.dumps({"ok": selected is not None, **summary}, indent=2))
    return 0 if selected is not None else 2


if __name__ == "__main__":
    raise SystemExit(main())
