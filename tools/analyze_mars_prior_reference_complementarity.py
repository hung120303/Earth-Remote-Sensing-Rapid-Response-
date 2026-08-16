#!/usr/bin/env python3
"""Analyze whether frozen prior references add MARS folds-3/4 signal."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "tools", ROOT / "EarthRemoteSensingRapidResponse"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from train_mars_oof_scene_ensemble_v2 import ap_group_bootstrap  # noqa: E402
from train_mars_scene_ranker import metric_summary  # noqa: E402

DEFAULT_PROTOCOL = Path("configs/mars_prior_reference_complementarity_protocol.json")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repo_path(value: str) -> Path:
    path = (ROOT / value).resolve()
    path.relative_to(ROOT)
    return path


def fixed_aggregations(
    view_scores: np.ndarray,
    view_mask: np.ndarray,
    selected_distances: np.ndarray,
    *,
    softmax_temperature: float,
) -> dict[str, np.ndarray]:
    """Return outcome-independent prior-reference score aggregations."""
    scores = np.asarray(view_scores, dtype=np.float64)
    mask = np.asarray(view_mask, dtype=bool)
    distances = np.asarray(selected_distances, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[1] != 6 or mask.shape != scores.shape:
        raise ValueError(
            "Reference score and mask shapes differ from the six-view contract"
        )
    if distances.shape != (scores.shape[0], 5):
        raise ValueError("Selected distances differ from the five-reference contract")
    if softmax_temperature <= 0.0:
        raise ValueError("Similarity temperature must be positive")
    if not mask[:, 0].all():
        raise ValueError("Every row must have an original-reference view")

    original = scores[:, 0].copy()
    alt_scores = scores[:, 1:]
    alt_mask = mask[:, 1:]
    result = {
        "original": original,
        "nearest": original.copy(),
        "median": original.copy(),
        "top2_mean": original.copy(),
        "similarity_weighted": original.copy(),
        "maximum": original.copy(),
    }
    for row in range(scores.shape[0]):
        valid = alt_mask[row]
        if not np.any(valid):
            continue
        values = alt_scores[row, valid]
        local_distances = distances[row, valid]
        if not np.isfinite(values).all() or not np.isfinite(local_distances).all():
            raise ValueError("Valid prior-reference values must be finite")
        result["nearest"][row] = values[0]
        result["median"][row] = float(np.median(values))
        result["maximum"][row] = float(np.max(values))
        top = np.sort(values)[-min(2, values.size) :]
        result["top2_mean"][row] = float(np.mean(top))
        logits = -local_distances / softmax_temperature
        weights = np.exp(logits - np.max(logits))
        result["similarity_weighted"][row] = float(
            np.sum(weights * values) / np.sum(weights)
        )
    return result


def reference_set_features(
    view_features: np.ndarray,
    view_mask: np.ndarray,
    selected_distances: np.ndarray,
    feature_names: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build outcome-independent row summaries for TP/FP separability checks."""
    values = np.asarray(view_features, dtype=np.float64)
    mask = np.asarray(view_mask, dtype=bool)
    distances = np.asarray(selected_distances, dtype=np.float64)
    names = np.asarray(feature_names).astype(str)
    if values.shape[:2] != mask.shape or values.shape[2] != names.size:
        raise ValueError("Reference feature schema is inconsistent")
    rows: list[np.ndarray] = []
    output_names: list[str] = []
    alt_mask = mask[:, 1:]
    for feature_index, name in enumerate(names.tolist()):
        original = values[:, 0, feature_index]
        local = values[:, 1:, feature_index]
        maximum = np.empty(values.shape[0], dtype=np.float64)
        mean = np.empty_like(maximum)
        std = np.empty_like(maximum)
        for row in range(values.shape[0]):
            selected = local[row, alt_mask[row]]
            if selected.size:
                maximum[row] = np.max(selected)
                mean[row] = np.mean(selected)
                std[row] = np.std(selected)
            else:
                maximum[row] = mean[row] = original[row]
                std[row] = 0.0
        rows.extend((maximum, mean, std, maximum - original))
        output_names.extend(
            (
                f"alt_max_{name}",
                f"alt_mean_{name}",
                f"alt_std_{name}",
                f"alt_max_minus_original_{name}",
            )
        )
    selected_count = alt_mask.sum(axis=1).astype(np.float64)
    distance_min = np.zeros(values.shape[0], dtype=np.float64)
    distance_mean = np.zeros_like(distance_min)
    distance_std = np.zeros_like(distance_min)
    for row in range(values.shape[0]):
        selected = distances[row, alt_mask[row]]
        if selected.size:
            distance_min[row] = np.min(selected)
            distance_mean[row] = np.mean(selected)
            distance_std[row] = np.std(selected)
    rows.extend((selected_count, distance_min, distance_mean, distance_std))
    output_names.extend(
        ("selected_count", "distance_min", "distance_mean", "distance_std")
    )
    output = np.stack(rows, axis=1)
    if not np.isfinite(output).all():
        raise ValueError("Reference-set summaries contain non-finite values")
    return output, np.asarray(output_names)


def standardized_contrasts(
    features: np.ndarray,
    names: np.ndarray,
    positive_selection: np.ndarray,
    negative_selection: np.ndarray,
    maximum: int,
) -> list[dict[str, Any]]:
    positive = features[positive_selection]
    negative = features[negative_selection]
    if positive.shape[0] < 2 or negative.shape[0] < 2:
        return []
    variance = (
        positive.var(axis=0, ddof=1) * (positive.shape[0] - 1)
        + negative.var(axis=0, ddof=1) * (negative.shape[0] - 1)
    ) / max(positive.shape[0] + negative.shape[0] - 2, 1)
    effects = (positive.mean(axis=0) - negative.mean(axis=0)) / np.sqrt(
        np.maximum(variance, 1e-12)
    )
    order = np.argsort(-np.abs(effects), kind="stable")[:maximum]
    return [
        {
            "feature": str(names[index]),
            "standardized_mean_difference": float(effects[index]),
            "positive_mean": float(positive[:, index].mean()),
            "negative_mean": float(negative[:, index].mean()),
        }
        for index in order
        if np.isfinite(effects[index])
    ]


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    return float(
        average_precision_score(labels.astype(np.uint8), scores.astype(np.float64))
    )


def score_summary(
    labels: np.ndarray,
    scores: np.ndarray,
    sensors: np.ndarray,
    folds: np.ndarray,
) -> dict[str, Any]:
    matched = metric_summary(labels, scores, sensors)
    return {
        "average_precision": average_precision(labels, scores),
        "matched_fpr": {
            key: float(matched["operating_point"][key])
            for key in ("threshold", "recall", "false_positive_rate")
        },
        "fold_average_precision": {
            str(fold): average_precision(labels[folds == fold], scores[folds == fold])
            for fold in (3, 4)
        },
        "sensor_average_precision": {
            str(sensor): average_precision(
                labels[sensors == sensor], scores[sensors == sensor]
            )
            for sensor in (0, 1)
        },
    }


def rescue_cell(
    selection: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    folds: np.ndarray,
) -> dict[str, Any]:
    positives = selection & (labels == 1)
    negatives = selection & (labels == 0)
    rows = int(selection.sum())
    positive_count = int(positives.sum())
    return {
        "rows": rows,
        "positives": positive_count,
        "negatives": int(negatives.sum()),
        "precision": None if rows == 0 else float(positive_count / rows),
        "positive_groups": int(np.unique(groups[positives]).size),
        "positive_fold_counts": {
            str(fold): int(np.sum(positives & (folds == fold))) for fold in (3, 4)
        },
    }


def metric_delta(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    return {
        "average_precision": float(
            candidate["average_precision"] - baseline["average_precision"]
        ),
        "matched_fpr_recall": float(
            candidate["matched_fpr"]["recall"] - baseline["matched_fpr"]["recall"]
        ),
        "fold_average_precision": {
            fold: float(
                candidate["fold_average_precision"][fold]
                - baseline["fold_average_precision"][fold]
            )
            for fold in ("3", "4")
        },
        "sensor_average_precision": {
            sensor: float(
                candidate["sensor_average_precision"][sensor]
                - baseline["sensor_average_precision"][sensor]
            )
            for sensor in ("0", "1")
        },
    }


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# MARS prior-reference complementarity diagnostic",
        "",
        "Authorized folds-3/4 outcome diagnostic; no model was fitted or promoted.",
        "",
        "| Fixed aggregation | AP | AP delta vs original released view | "
        "Recall delta at matched FPR | Paired-group AP interval | Signal gate |",
        "|---|---:|---:|---:|---:|:---:|",
    ]
    for name, value in report["fixed_aggregations"].items():
        interval = value["paired_group_ap_delta"]
        lines.append(
            f"| {name} | {value['metrics']['average_precision']:.6f} | "
            f"{value['delta']['average_precision']:+.6f} | "
            f"{value['delta']['matched_fpr_recall']:+.6f} | "
            f"[{interval['lower']:+.6f}, {interval['upper']:+.6f}] | "
            f"{'yes' if value['fixed_signal_pass'] else 'no'} |"
        )
    route = report["any_prior_reference_paper_rule"]
    lines.extend(
        [
            "",
            "## New paper-rule rescues",
            "",
            f"The max-over-prior rule adds {route['cell']['positives']} champion-missed "
            f"positives and {route['cell']['negatives']} negatives after excluding rows "
            "already above the original released paper rule.",
            "",
            f"**Decision:** {report['decision']}",
            "",
            "This result can authorize a cross-fitted reference-set model, but cannot "
            "promote a score, open another cohort, or support a superiority claim.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL.as_posix())
    args = parser.parse_args()
    protocol_path = repo_path(args.protocol)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if (
        sha256(Path(__file__).resolve())
        != protocol["implementation"]["script"]["sha256"]
    ):
        raise ValueError("Frozen diagnostic script hash mismatch")
    for dependency in protocol["implementation"]["code_dependencies"]:
        if sha256(repo_path(dependency["path"])) != dependency["sha256"]:
            raise ValueError(
                f"Frozen diagnostic dependency mismatch: {dependency['path']}"
            )
    inputs = {
        name: repo_path(contract["path"])
        for name, contract in protocol["inputs"].items()
    }
    for name, contract in protocol["inputs"].items():
        if sha256(inputs[name]) != contract["sha256"]:
            raise ValueError(f"Frozen diagnostic input mismatch: {name}")

    with np.load(inputs["prior_reference_scores"], allow_pickle=False) as cache:
        sample_ids = cache["sample_ids"].astype(str)
        folds = cache["folds"].astype(np.uint8)
        sensor_names = cache["sensors"].astype(str)
        view_mask = cache["view_mask"].astype(bool)
        names = cache["view_feature_names"].astype(str)
        view_features = cache["view_features"].astype(np.float64)
        distances = cache["selected_distances"].astype(np.float64)
    connected_index = int(np.flatnonzero(names == "connected_score")[0])
    aggregations = fixed_aggregations(
        view_features[:, :, connected_index],
        view_mask,
        distances,
        softmax_temperature=float(protocol["aggregations"]["similarity_temperature"]),
    )

    with np.load(inputs["champion_scores"], allow_pickle=False) as champion:
        champion_ids = champion["sample_ids"].astype(str)
        labels = champion["labels"].astype(np.uint8)
        sensors = champion["sensors"].astype(np.uint8)
        groups = champion["groups"].astype(str)
        champion_folds = champion["folds"].astype(np.uint8)
        champion_scores = champion["champion_scores"].astype(np.float64)
    if not (
        np.array_equal(sample_ids, champion_ids)
        and np.array_equal(folds, champion_folds)
        and np.array_equal(sensor_names == "Landsat", sensors == 1)
        and sample_ids.size == int(protocol["cohort"]["rows"])
    ):
        raise ValueError("Prior-reference and champion row contracts differ")
    if set(np.unique(folds).tolist()) != {3, 4}:
        raise ValueError("Diagnostic escaped authorized folds 3/4")

    baseline_metrics = score_summary(labels, aggregations["original"], sensors, folds)
    champion_metrics = score_summary(labels, champion_scores, sensors, folds)
    champion_threshold = float(champion_metrics["matched_fpr"]["threshold"])
    champion_decision = champion_scores >= champion_threshold
    bootstrap = protocol["bootstrap"]
    gates = protocol["gates"]
    results: dict[str, Any] = {}
    for candidate_index, name in enumerate(protocol["aggregations"]["fixed"]):
        scores = aggregations[name]
        metrics = score_summary(labels, scores, sensors, folds)
        delta = metric_delta(metrics, baseline_metrics)
        interval = ap_group_bootstrap(
            labels,
            aggregations["original"],
            scores,
            groups,
            replicates=int(bootstrap["replicates"]),
            seed=int(bootstrap["seed"]) + candidate_index,
            confidence=float(bootstrap["confidence"]),
        )
        new_rescue = (
            (~champion_decision)
            & (aggregations["original"] <= float(gates["released_paper_threshold"]))
            & (scores > float(gates["released_paper_threshold"]))
        )
        cell = rescue_cell(new_rescue, labels, groups, folds)
        checks = {
            "average_precision_delta": delta["average_precision"]
            >= float(gates["minimum_fixed_ap_delta"]),
            "matched_fpr_recall_nonnegative": delta["matched_fpr_recall"]
            >= float(gates["fixed_matched_fpr_recall_delta_minimum"]),
            "each_fold_ap_nonnegative": min(delta["fold_average_precision"].values())
            >= float(gates["fixed_each_fold_ap_delta_minimum"]),
            "each_sensor_ap_nonnegative": min(
                delta["sensor_average_precision"].values()
            )
            >= float(gates["fixed_each_sensor_ap_delta_minimum"]) - 1e-12,
            "paired_group_ap_lower_positive": float(interval["lower"]) > 0.0,
        }
        results[name] = {
            "metrics": metrics,
            "delta": delta,
            "paired_group_ap_delta": interval,
            "new_rescue": cell,
            "checks": checks,
            "fixed_signal_pass": all(checks.values()),
        }

    any_prior = (
        view_mask[:, 1:].any(axis=1)
        & (aggregations["maximum"] > float(gates["released_paper_threshold"]))
        & (aggregations["original"] <= float(gates["released_paper_threshold"]))
        & (~champion_decision)
    )
    any_cell = rescue_cell(any_prior, labels, groups, folds)
    set_features, set_feature_names = reference_set_features(
        view_features, view_mask, distances, names
    )
    contrasts = standardized_contrasts(
        set_features,
        set_feature_names,
        any_prior & (labels == 1),
        any_prior & (labels == 0),
        int(protocol["feature_contrast"]["maximum_reported_features"]),
    )
    maximum_contrast = max(
        (abs(value["standardized_mean_difference"]) for value in contrasts),
        default=0.0,
    )
    route_checks = {
        "minimum_new_positive_rescues": any_cell["positives"]
        >= int(gates["minimum_new_positive_rescues"]),
        "minimum_route_precision": any_cell["precision"] is not None
        and any_cell["precision"] >= float(gates["minimum_route_precision"]),
        "minimum_positive_groups": any_cell["positive_groups"]
        >= int(gates["minimum_positive_groups"]),
        "positive_rescue_in_each_fold": min(any_cell["positive_fold_counts"].values())
        >= int(gates["minimum_positive_rescues_per_fold"]),
        "minimum_feature_contrast": maximum_contrast
        >= float(gates["minimum_absolute_feature_contrast"]),
        "some_fixed_aggregation_improves_ap": max(
            value["delta"]["average_precision"] for value in results.values()
        )
        > 0.0,
    }
    fixed_pass = any(value["fixed_signal_pass"] for value in results.values())
    route_pass = all(route_checks.values())
    passed = fixed_pass or route_pass
    decision = (
        "PASS: authorize a separately preregistered cross-fitted reference-set temporal "
        "model on folds 3/4; no fixed aggregation is promoted by this diagnostic."
        if passed
        else "FAIL: retire the prior-reference architecture before model fitting."
    )
    report = {
        "schema_version": 1,
        "status": "complete_folds34_prior_reference_complementarity_diagnostic",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "cohort": {
            "rows": int(labels.size),
            "positives": int(labels.sum()),
            "negatives": int((labels == 0).sum()),
            "groups": int(np.unique(groups).size),
            "folds": [3, 4],
        },
        "baseline_original_released_view": baseline_metrics,
        "gaussian_dofa_champion": champion_metrics,
        "fixed_aggregations": results,
        "any_prior_reference_paper_rule": {
            "cell": any_cell,
            "feature_contrasts": contrasts,
            "maximum_absolute_feature_contrast": maximum_contrast,
            "checks": route_checks,
            "pass": route_pass,
        },
        "fixed_aggregation_signal_pass": fixed_pass,
        "all_continuation_gates_pass": passed,
        "decision": decision,
        "outcome_access": {
            "opened": "previously authorized MARS folds 3/4 labels/groups/sensors from the frozen champion cache",
            "folds_0_1_2_accessed": False,
            "external_outcomes_accessed": False,
            "official_test_accessed": False,
            "model_fitted": False,
            "score_promoted": False,
        },
        "provenance": {
            "protocol": {
                "path": protocol_path.relative_to(ROOT).as_posix(),
                "sha256": sha256(protocol_path),
            },
            "script": {
                "path": Path(__file__).resolve().relative_to(ROOT).as_posix(),
                "sha256": sha256(Path(__file__).resolve()),
            },
            "inputs": {
                name: {
                    "path": path.relative_to(ROOT).as_posix(),
                    "sha256": sha256(path),
                }
                for name, path in inputs.items()
            },
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
        },
        "claim_boundary": protocol["claim_boundary"],
    }
    output_json = repo_path(protocol["outputs"]["json"])
    output_markdown = repo_path(protocol["outputs"]["markdown"])
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_markdown(output_markdown, report)
    print(json.dumps({"ok": True, "pass": passed, "decision": decision}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
