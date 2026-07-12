#!/usr/bin/env python3
"""Aggregate the five frozen ERSRR seeds against released MARS-S2L.

This script consumes only checksum-bound compact scene caches produced during
the once-only strict campaign. It never opens a raster and never selects a
threshold, architecture, seed, or operating rule from strict-test behavior.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
from sklearn.metrics import average_precision_score, roc_auc_score

from acquire_mars_metadata import repo_root, sha256

FIXED_SEEDS = (101, 202, 303, 404, 505)
BOOTSTRAP_REPLICATES = 2_000
BOOTSTRAP_SEED = 20_260_712
DEFAULT_BASELINE = Path(
    "reports/experiments/mars_released_model_full_strict_baseline.json"
)
DEFAULT_JSON = Path("reports/experiments/mars_v3_strict_campaign.json")
DEFAULT_MARKDOWN = Path("reports/experiments/MARS_V3_STRICT_CAMPAIGN.md")
PAPER_TARGETS = {
    "source": "arXiv:2511.21777v3, Tables S1 and S3-S7 and S11",
    "full_official_test": {
        "images": 43_529,
        "plume_images": 1_813,
        "sites": 1_289,
        "average_precision": 0.6408,
        "recall": 0.7915,
        "false_positive_rate": 0.0713,
        "pixel_intersection_over_union": 0.3224,
        "probability_threshold": 0.5,
    },
    "test_only_sites": {
        "images": 15_655,
        "plume_images": 227,
        "sites": 697,
        "average_precision": 0.4496,
        "recall": 0.7753,
        "false_positive_rate": 0.0763,
        "probability_threshold": 0.5,
    },
    "full_official_test_strict_threshold": {
        "recall": 0.5836,
        "false_positive_rate": 0.0116,
        "probability_threshold": 0.98,
    },
}


def safe_path(root: Path, value: str | Path) -> Path:
    result = (root / value).resolve()
    if root not in result.parents:
        raise ValueError("Path must resolve beneath the repository root")
    return result


def tracked_dirty(root: Path) -> bool:
    status = subprocess.check_output(
        [
            "git",
            "-c",
            "core.autocrlf=true",
            "status",
            "--porcelain",
            "--untracked-files=no",
        ],
        cwd=root,
        text=True,
    )
    return bool(status.strip())


def ratio(numerator: float, denominator: float) -> float | None:
    return None if denominator == 0 else float(numerator / denominator)


def scene_metrics(
    labels: np.ndarray, predictions: np.ndarray, scores: np.ndarray
) -> dict[str, float | None]:
    y = np.asarray(labels, dtype=np.uint8)
    predicted = np.asarray(predictions, dtype=np.uint8)
    probability = np.asarray(scores, dtype=np.float64)
    if y.ndim != 1 or predicted.shape != y.shape or probability.shape != y.shape:
        raise ValueError("Scene metrics require matching one-dimensional arrays")
    tp = float(np.sum((y == 1) & (predicted == 1)))
    fn = float(np.sum((y == 1) & (predicted == 0)))
    fp = float(np.sum((y == 0) & (predicted == 1)))
    tn = float(np.sum((y == 0) & (predicted == 0)))
    both_classes = np.unique(y).size == 2
    return {
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "recall": ratio(tp, tp + fn),
        "specificity": ratio(tn, tn + fp),
        "false_positive_rate": ratio(fp, fp + tn),
        "precision": ratio(tp, tp + fp),
        "negative_predictive_value": ratio(tn, tn + fn),
        "accuracy": ratio(tp + tn, tp + fn + fp + tn),
        "auroc": float(roc_auc_score(y, probability)) if both_classes else None,
        "average_precision": (
            float(average_precision_score(y, probability)) if both_classes else None
        ),
    }


def summary(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("Bootstrap summary requires finite values")
    return {
        "mean": float(np.mean(array)),
        "standard_deviation": float(np.std(array, ddof=1)) if array.size > 1 else 0.0,
        "95ci": np.quantile(array, [0.025, 0.975]).astype(float).tolist(),
    }


def load_report_and_cache(
    root: Path, report_path: Path, *, kind: str
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    expected_scope = (
        "frozen_v3_full_strict_spatial_evaluation"
        if kind == "v3"
        else "released_mars-s2l_on_frozen_full_strict_spatial_cohort"
    )
    if report.get("scope") != expected_scope:
        raise ValueError(f"Unexpected {kind} report scope in {report_path}")
    artifact = report.get("scene_prediction_cache")
    if not isinstance(artifact, dict):
        raise ValueError(f"{report_path} lacks a bound scene prediction cache")
    cache_path = safe_path(root, artifact["path"])
    if cache_path.stat().st_size != int(artifact["bytes"]):
        raise ValueError(f"Prediction cache size mismatch for {report_path}")
    if sha256(cache_path) != artifact["sha256"]:
        raise ValueError(f"Prediction cache checksum mismatch for {report_path}")
    with np.load(cache_path, allow_pickle=False) as archive:
        cache = {name: archive[name].copy() for name in archive.files}
    if int(cache["schema_version"][0]) != 1:
        raise ValueError("Unsupported scene prediction cache schema")
    strict_identity = str(cache["strict_manifest_sha256"][0])
    report_identity = (
        report["source"]["strict_manifest_sha256"]
        if kind == "v3"
        else report["source"]["evaluation_manifest_sha256"]
    )
    if strict_identity != report_identity:
        raise ValueError("Cache/report strict-manifest identity mismatch")
    required = (
        ("sample_ids", "groups", "labels", "primary_scores", "primary_predictions")
        if kind == "v3"
        else ("sample_ids", "groups", "labels", "scores", "predictions")
    )
    arrays = [np.asarray(cache[name]) for name in required]
    if any(array.ndim != 1 for array in arrays) or len({array.shape for array in arrays}) != 1:
        raise ValueError("Scene cache arrays do not share a one-dimensional shape")
    if len(set(arrays[0].astype(str))) != arrays[0].size:
        raise ValueError("Scene cache contains duplicate sample IDs")
    return report, cache


def aligned_cache(
    cache: dict[str, np.ndarray], sample_ids: np.ndarray, *, kind: str
) -> dict[str, np.ndarray]:
    source_ids = cache["sample_ids"].astype(str)
    destination = np.asarray(sample_ids).astype(str)
    if set(source_ids) != set(destination):
        raise ValueError("Strict scene caches contain different sample-ID sets")
    positions = {sample_id: index for index, sample_id in enumerate(source_ids)}
    order = np.asarray([positions[sample_id] for sample_id in destination], dtype=np.int64)
    score_key = "primary_scores" if kind == "v3" else "scores"
    prediction_key = "primary_predictions" if kind == "v3" else "predictions"
    return {
        "sample_ids": source_ids[order],
        "groups": cache["groups"].astype(str)[order],
        "labels": cache["labels"].astype(np.uint8)[order],
        "scores": cache[score_key].astype(np.float64)[order],
        "predictions": cache[prediction_key].astype(np.uint8)[order],
    }


def verify_report_metrics(report: dict[str, Any], computed: dict[str, Any]) -> None:
    recorded = report["strict_spatial_test"]["scene_unweighted"]
    for name in (
        "tp",
        "fn",
        "fp",
        "tn",
        "recall",
        "specificity",
        "false_positive_rate",
        "auroc",
        "average_precision",
    ):
        expected = recorded[name]
        observed = computed[name]
        if expected is None or observed is None:
            if expected is not observed:
                raise ValueError(f"Report/cache metric mismatch for {name}")
        elif not np.isclose(float(expected), float(observed), atol=1e-6, rtol=1e-6):
            raise ValueError(f"Report/cache metric mismatch for {name}")


def mean_metric(rows: list[dict[str, Any]], name: str) -> float:
    values = [float(row[name]) for row in rows if row[name] is not None]
    if len(values) != len(rows):
        raise ValueError(f"Metric {name} is undefined for at least one seed")
    return float(np.mean(values))


def paired_campaign_bootstrap(
    labels: np.ndarray,
    groups: np.ndarray,
    candidates: list[dict[str, np.ndarray]],
    baseline: dict[str, np.ndarray],
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Jointly resample 25 km groups and the five training seeds."""
    if replicates < 100:
        raise ValueError("At least 100 bootstrap replicates are required")
    unique_groups = np.unique(groups.astype(str))
    group_rows = {
        group: np.flatnonzero(groups.astype(str) == group) for group in unique_groups
    }
    rng = np.random.default_rng(seed)
    values: dict[str, list[float]] = {
        "candidate_recall": [],
        "candidate_false_positive_rate": [],
        "candidate_average_precision": [],
        "candidate_auroc": [],
        "recall_delta": [],
        "false_positive_rate_delta": [],
        "relative_false_positive_rate_reduction": [],
        "average_precision_delta": [],
        "auroc_delta": [],
    }
    accepted = 0
    attempts = 0
    while accepted < replicates:
        attempts += 1
        if attempts > replicates * 10:
            raise RuntimeError("Unable to obtain valid paired bootstrap replicates")
        sampled_groups = rng.choice(unique_groups, size=unique_groups.size, replace=True)
        rows = np.concatenate([group_rows[group] for group in sampled_groups])
        if np.unique(labels[rows]).size != 2:
            continue
        sampled_seeds = rng.integers(0, len(candidates), size=len(candidates))
        candidate_metrics = [
            scene_metrics(
                labels[rows],
                candidates[index]["predictions"][rows],
                candidates[index]["scores"][rows],
            )
            for index in sampled_seeds
        ]
        baseline_metrics = scene_metrics(
            labels[rows], baseline["predictions"][rows], baseline["scores"][rows]
        )
        candidate_recall = mean_metric(candidate_metrics, "recall")
        candidate_fpr = mean_metric(candidate_metrics, "false_positive_rate")
        candidate_ap = mean_metric(candidate_metrics, "average_precision")
        candidate_auroc = mean_metric(candidate_metrics, "auroc")
        baseline_recall = float(baseline_metrics["recall"])
        baseline_fpr = float(baseline_metrics["false_positive_rate"])
        baseline_ap = float(baseline_metrics["average_precision"])
        baseline_auroc = float(baseline_metrics["auroc"])
        if baseline_fpr <= 0:
            continue
        values["candidate_recall"].append(candidate_recall)
        values["candidate_false_positive_rate"].append(candidate_fpr)
        values["candidate_average_precision"].append(candidate_ap)
        values["candidate_auroc"].append(candidate_auroc)
        values["recall_delta"].append(candidate_recall - baseline_recall)
        values["false_positive_rate_delta"].append(candidate_fpr - baseline_fpr)
        values["relative_false_positive_rate_reduction"].append(
            1.0 - candidate_fpr / baseline_fpr
        )
        values["average_precision_delta"].append(candidate_ap - baseline_ap)
        values["auroc_delta"].append(candidate_auroc - baseline_auroc)
        accepted += 1
    return {
        "method": (
            "paired nonparametric bootstrap: resample frozen 25 km groups with replacement and "
            "resample the five fixed training seeds with replacement inside each replicate"
        ),
        "replicates": replicates,
        "random_seed": seed,
        "group_count": int(unique_groups.size),
        **{name: summary(samples) for name, samples in values.items()},
    }


def pixel_metrics(report: dict[str, Any], *, kind: str) -> dict[str, float]:
    source = (
        report["strict_spatial_test"]["segmentation"]["pixel"]
        if kind == "v3"
        else report["strict_spatial_test"]["pixel_validity_aware"]
    )
    return {
        "average_precision": float(source["average_precision"]),
        "intersection_over_union": float(source["intersection_over_union"]),
        "dice": float(source["dice"]),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    point = report["same_cohort_comparison"]
    paired = report["paired_seed_group_bootstrap"]
    gate = report["promotion_gate"]
    candidate = point["ersrr_seed_mean"]
    baseline = point["released_mars_s2l"]
    lines = [
        "# ERSRR v3 five-seed strict campaign",
        "",
        "All checkpoints, thresholds, and proposal blends were frozen before this campaign.",
        "",
        f"- Cohort: {report['cohort']['samples']} scenes / {report['cohort']['groups']} frozen 25 km groups",
        f"- ERSRR seed-mean recall / FPR: {candidate['recall']:.3f} / {candidate['false_positive_rate']:.3f}",
        f"- Released MARS-S2L recall / FPR: {baseline['recall']:.3f} / {baseline['false_positive_rate']:.3f}",
        f"- Recall delta: {point['delta']['recall']:+.3f}",
        f"- Relative FPR reduction: {point['delta']['relative_false_positive_rate_reduction']:.1%}",
        f"- Paired recall-delta 95% CI: {paired['recall_delta']['95ci'][0]:+.3f} to {paired['recall_delta']['95ci'][1]:+.3f}",
        f"- Paired relative-FPR-reduction 95% CI: {paired['relative_false_positive_rate_reduction']['95ci'][0]:.1%} to {paired['relative_false_positive_rate_reduction']['95ci'][1]:.1%}",
        f"- Promotion gate: {'PASS' if gate['passed'] else 'FAIL'}",
        "",
        "## Decision",
        "",
        report["decision"],
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v3-report", action="append", dest="v3_reports")
    parser.add_argument("--baseline-report", default=DEFAULT_BASELINE.as_posix())
    parser.add_argument("--replicates", type=int, default=BOOTSTRAP_REPLICATES)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument("--output-json", default=DEFAULT_JSON.as_posix())
    parser.add_argument("--output-markdown", default=DEFAULT_MARKDOWN.as_posix())
    args = parser.parse_args()

    root = repo_root()
    report_values = args.v3_reports or [
        f"reports/experiments/mars_v3_seed{seed}_strict_evaluation.json"
        for seed in FIXED_SEEDS
    ]
    if len(report_values) != len(FIXED_SEEDS):
        raise ValueError("Exactly five v3 strict reports are required")
    report_paths = [safe_path(root, value) for value in report_values]
    baseline_path = safe_path(root, args.baseline_report)
    baseline_report, raw_baseline = load_report_and_cache(
        root, baseline_path, kind="baseline"
    )
    baseline = aligned_cache(raw_baseline, raw_baseline["sample_ids"], kind="baseline")
    sample_ids = baseline["sample_ids"]
    labels = baseline["labels"]
    groups = baseline["groups"]
    candidate_reports: list[dict[str, Any]] = []
    candidates: list[dict[str, np.ndarray]] = []
    seeds: list[int] = []
    strict_identity = baseline_report["source"]["evaluation_manifest_sha256"]
    for path in report_paths:
        report, raw = load_report_and_cache(root, path, kind="v3")
        seed = int(report["artifact"].get("seed", report.get("training", {}).get("seed", -1)))
        if seed < 0:
            # Training seed is retained in the validation-linked operating report.
            seed = int(np.asarray(raw["seed"])[0])
        aligned = aligned_cache(raw, sample_ids, kind="v3")
        if report["source"]["strict_manifest_sha256"] != strict_identity:
            raise ValueError("V3 and baseline reports use different strict manifests")
        if not np.array_equal(aligned["labels"], labels):
            raise ValueError("V3 and baseline caches disagree on scene labels")
        if not np.array_equal(aligned["groups"], groups):
            raise ValueError("V3 and baseline caches disagree on frozen groups")
        computed = scene_metrics(labels, aligned["predictions"], aligned["scores"])
        verify_report_metrics(report, computed)
        candidate_reports.append(report)
        candidates.append(aligned)
        seeds.append(seed)
    if tuple(sorted(seeds)) != FIXED_SEEDS:
        raise ValueError(f"Strict reports do not cover fixed seeds {FIXED_SEEDS}: {seeds}")
    order = np.argsort(seeds)
    seeds = [seeds[index] for index in order]
    candidates = [candidates[index] for index in order]
    candidate_reports = [candidate_reports[index] for index in order]
    report_paths = [report_paths[index] for index in order]
    baseline_metrics = scene_metrics(labels, baseline["predictions"], baseline["scores"])
    verify_report_metrics(baseline_report, baseline_metrics)
    seed_metrics = [
        scene_metrics(labels, item["predictions"], item["scores"]) for item in candidates
    ]
    metric_names = (
        "recall",
        "specificity",
        "false_positive_rate",
        "precision",
        "negative_predictive_value",
        "accuracy",
        "auroc",
        "average_precision",
    )
    candidate_mean = {name: mean_metric(seed_metrics, name) for name in metric_names}
    candidate_sd = {
        name: float(np.std([float(row[name]) for row in seed_metrics], ddof=1))
        for name in metric_names
    }
    baseline_fpr = float(baseline_metrics["false_positive_rate"])
    candidate_fpr = float(candidate_mean["false_positive_rate"])
    delta = {
        name: float(candidate_mean[name] - float(baseline_metrics[name]))
        for name in ("recall", "false_positive_rate", "auroc", "average_precision")
    }
    delta["relative_false_positive_rate_reduction"] = (
        1.0 - candidate_fpr / baseline_fpr if baseline_fpr > 0 else None
    )
    paired = paired_campaign_bootstrap(
        labels,
        groups,
        candidates,
        baseline,
        replicates=args.replicates,
        seed=args.bootstrap_seed,
    )
    seed_pixels = [pixel_metrics(report, kind="v3") for report in candidate_reports]
    baseline_pixel = pixel_metrics(baseline_report, kind="baseline")
    pixel_names = ("average_precision", "intersection_over_union", "dice")
    pixel_mean = {name: mean_metric(seed_pixels, name) for name in pixel_names}
    pixel_sd = {
        name: float(np.std([row[name] for row in seed_pixels], ddof=1))
        for name in pixel_names
    }
    gates = {
        "candidate_recall_lower_95ci_at_least_0_75": paired["candidate_recall"]["95ci"][0]
        >= 0.75,
        "candidate_mean_fpr_at_most_0_05": candidate_fpr <= 0.05,
        "candidate_mean_specificity_at_least_0_95": candidate_mean["specificity"] >= 0.95,
        "recall_noninferiority_lower_95ci_at_least_0": paired["recall_delta"]["95ci"][0]
        >= 0.0,
        "relative_fpr_reduction_lower_95ci_at_least_0_25": paired[
            "relative_false_positive_rate_reduction"
        ]["95ci"][0]
        >= 0.25,
    }
    report = {
        "schema_version": 1,
        "scope": "frozen_v3_five_seed_full_strict_campaign",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "cohort": {
            "samples": int(labels.size),
            "positives": int(np.sum(labels)),
            "negatives": int(np.sum(labels == 0)),
            "groups": int(np.unique(groups).size),
            "strict_manifest_sha256": strict_identity,
        },
        "fixed_seeds": seeds,
        "same_cohort_comparison": {
            "ersrr_per_seed": [
                {"seed": seed, **metrics_row}
                for seed, metrics_row in zip(seeds, seed_metrics)
            ],
            "ersrr_seed_mean": candidate_mean,
            "ersrr_seed_standard_deviation": candidate_sd,
            "released_mars_s2l": baseline_metrics,
            "delta": delta,
        },
        "paired_seed_group_bootstrap": paired,
        "segmentation": {
            "ersrr_per_seed": [
                {"seed": seed, **metrics_row}
                for seed, metrics_row in zip(seeds, seed_pixels)
            ],
            "ersrr_seed_mean": pixel_mean,
            "ersrr_seed_standard_deviation": pixel_sd,
            "released_mars_s2l": baseline_pixel,
            "delta": {
                name: pixel_mean[name] - baseline_pixel[name] for name in pixel_names
            },
        },
        "official_mars_s2l_paper_targets_not_same_cohort": PAPER_TARGETS,
        "promotion_gate": {"criteria": gates, "passed": all(gates.values())},
        "decision": (
            "ERSRR v3 outperforms the released MARS-S2L checkpoint on the paired full strict cohort and clears the frozen MARS gate; untouched EMIT confirmation remains required."
            if all(gates.values())
            else "ERSRR v3 does not clear every frozen same-cohort MARS-S2L promotion criterion; preserve the result and do not retune from strict-test behavior."
        ),
        "inputs": {
            "v3_reports": [
                {
                    "path": path.relative_to(root).as_posix(),
                    "sha256": sha256(path),
                    "scene_cache": report["scene_prediction_cache"],
                }
                for path, report in zip(report_paths, candidate_reports)
            ],
            "released_baseline_report": {
                "path": baseline_path.relative_to(root).as_posix(),
                "sha256": sha256(baseline_path),
                "scene_cache": baseline_report["scene_prediction_cache"],
            },
        },
        "runtime": {"numpy": np.__version__, "sklearn": sklearn.__version__},
        "provenance": {
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True
            ).strip(),
            "git_tracked_worktree_dirty_at_start": tracked_dirty(root),
            "script": "tools/aggregate_mars_v3_strict.py",
            "script_sha256": sha256(Path(__file__)),
        },
    }
    output_json = safe_path(root, args.output_json)
    output_markdown = safe_path(root, args.output_markdown)
    write_json(output_json, report)
    write_markdown(output_markdown, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
