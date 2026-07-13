#!/usr/bin/env python3
"""Freeze the predeclared three-seed MethaneS2CM v5.1 ensemble."""

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

import h5py
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "EarthRemoteSensingRapidResponse"
if str(MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_ROOT))

from acquire_mars_metadata import repo_root, sha256  # noqa: E402
from train_mars_v3 import safe_output, tracked_dirty, write_json  # noqa: E402
from train_methanes2cm_v5 import (  # noqa: E402
    DEFAULT_PACKED,
    PIXEL_THRESHOLDS,
    TARGET_FPRS,
    choose_threshold_at_fpr,
)

DEFAULT_PROTOCOL = Path("reports/experiments/methanes2cm_v5_1_campaign_protocol.json")
DEFAULT_REPORTS = (
    Path("reports/experiments/methanes2cm_v5_1_seed1101_validation.json"),
    Path("reports/experiments/methanes2cm_v5_1_seed2202_validation.json"),
    Path("reports/experiments/methanes2cm_v5_1_seed3303_validation.json"),
)
DEFAULT_V5_BASELINE = Path("reports/experiments/methanes2cm_v5_seed1101_validation.json")
DEFAULT_REPORT = Path("reports/experiments/methanes2cm_v5_1_ensemble_validation.json")
DEFAULT_MARKDOWN = Path("reports/experiments/METHANES2CM_V5_1_ENSEMBLE_VALIDATION.md")
DEFAULT_CACHE = Path(
    "EarthRemoteSensingRapidResponse/Data Collection/s2_emit_pairs/publication-v1/"
    "external/MethaneS2CM/l2a_location_split_32x32/v5_1_ensemble_calibration.npz"
)
BOOTSTRAP_SEED = 20260713
BOOTSTRAP_REPLICATES = 2000
CALIBRATION_FOLDS = 5


def empirical_percentile(training: np.ndarray, evaluation: np.ndarray) -> np.ndarray:
    reference = np.sort(np.asarray(training, dtype=np.float64))
    values = np.asarray(evaluation, dtype=np.float64)
    if reference.size == 0:
        raise ValueError("Empirical calibration requires training scores")
    return np.searchsorted(reference, values, side="right") / reference.size


def stable_group_folds(groups: np.ndarray, folds: int = CALIBRATION_FOLDS) -> np.ndarray:
    unique = sorted(set(str(value) for value in groups))
    mapping = {
        group: int(hashlib.sha256(group.encode("utf-8")).hexdigest()[:16], 16) % folds
        for group in unique
    }
    assignments = np.asarray([mapping[str(group)] for group in groups], dtype=np.int8)
    if len(unique) >= folds and set(assignments.tolist()) != set(range(folds)):
        raise ValueError("Stable group hash did not populate every calibration fold")
    return assignments


def binary_metrics(labels: np.ndarray, decisions: np.ndarray) -> dict[str, float | int]:
    truth = np.asarray(labels, dtype=bool)
    prediction = np.asarray(decisions, dtype=bool)
    true_positive = int(np.count_nonzero(prediction & truth))
    false_positive = int(np.count_nonzero(prediction & ~truth))
    true_negative = int(np.count_nonzero(~prediction & ~truth))
    false_negative = int(np.count_nonzero(~prediction & truth))
    return {
        "recall": true_positive / max(true_positive + false_negative, 1),
        "false_positive_rate": false_positive / max(false_positive + true_negative, 1),
        "precision": true_positive / max(true_positive + false_positive, 1),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "true_negative": true_negative,
        "false_negative": false_negative,
    }


def load_aligned_caches(
    root: Path, reports: list[dict[str, Any]]
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    loaded: list[dict[str, np.ndarray]] = []
    for report in reports:
        cache_path = root / report["prediction_cache"]["path"]
        if sha256(cache_path) != report["prediction_cache"]["sha256"]:
            raise ValueError(f"Prediction cache hash mismatch: {cache_path}")
        with np.load(cache_path, allow_pickle=False) as source:
            loaded.append({name: source[name] for name in source.files})
    reference_ids = loaded[0]["sample_id"].astype(np.int64)
    aligned: list[dict[str, np.ndarray]] = [loaded[0]]
    for cache in loaded[1:]:
        index = {int(identifier): row for row, identifier in enumerate(cache["sample_id"])}
        try:
            order = np.asarray([index[int(identifier)] for identifier in reference_ids])
        except KeyError as exc:
            raise ValueError("Seed caches do not contain identical sample ids") from exc
        aligned.append({name: values[order] for name, values in cache.items()})
    for name in ("label", "packed_index", "group_id", "exact_location_id", "observable"):
        for cache in aligned[1:]:
            if not np.array_equal(aligned[0][name], cache[name]):
                raise ValueError(f"Aligned seed caches disagree on {name}")
    raw_scores = np.stack(
        [cache["scene_score"].astype(np.float32) for cache in aligned], axis=0
    )
    shared = {
        name: aligned[0][name]
        for name in (
            "sample_id",
            "packed_index",
            "label",
            "group_id",
            "exact_location_id",
            "observable",
        )
    }
    return shared, raw_scores


def read_truth(packed_path: Path, packed_indices: np.ndarray) -> np.ndarray:
    indices = np.asarray(packed_indices, dtype=np.int64)
    order = np.argsort(indices)
    if len(np.unique(indices)) != len(indices):
        raise ValueError("Development packed indices are not unique")
    with h5py.File(packed_path, "r") as source:
        sorted_truth = source["mask"][indices[order]].astype(bool)
    result = np.empty_like(sorted_truth)
    result[order] = sorted_truth
    return result


def pixel_statistics(
    probability: np.ndarray,
    truth: np.ndarray,
    observable: np.ndarray,
) -> dict[str, Any]:
    valid = np.asarray(observable, dtype=bool)
    target = np.asarray(truth, dtype=bool) & valid
    scores = np.asarray(probability, dtype=np.float32)
    truth_area = target.reshape(target.shape[0], -1).sum(axis=1).astype(np.int64)
    intersections = np.zeros((len(PIXEL_THRESHOLDS), target.shape[0]), dtype=np.int32)
    predicted = np.zeros_like(intersections)
    for threshold_index, threshold in enumerate(PIXEL_THRESHOLDS):
        prediction = (scores >= threshold) & valid
        intersections[threshold_index] = (prediction & target).reshape(
            target.shape[0], -1
        ).sum(axis=1)
        predicted[threshold_index] = prediction.reshape(target.shape[0], -1).sum(axis=1)
    total_intersection = intersections.sum(axis=1, dtype=np.int64)
    total_predicted = predicted.sum(axis=1, dtype=np.int64)
    total_truth = int(truth_area.sum())
    dice = 2.0 * total_intersection / np.maximum(total_predicted + total_truth, 1)
    union = total_predicted + total_truth - total_intersection
    iou = total_intersection / np.maximum(union, 1)
    selected = int(np.argmax(dice))
    return {
        "average_precision": float(average_precision_score(target[valid], scores[valid])),
        "observable_pixels": int(np.count_nonzero(valid)),
        "truth_pixels": total_truth,
        "intersections_by_threshold_scene": intersections,
        "predicted_by_threshold_scene": predicted,
        "truth_by_scene": truth_area,
        "grid": [
            {
                "threshold": threshold,
                "dice": float(dice[index]),
                "intersection_over_union": float(iou[index]),
                "intersection_pixels": int(total_intersection[index]),
                "predicted_positive_pixels": int(total_predicted[index]),
            }
            for index, threshold in enumerate(PIXEL_THRESHOLDS)
        ],
        "selected_index": selected,
        "selected": {
            "threshold": PIXEL_THRESHOLDS[selected],
            "dice": float(dice[selected]),
            "intersection_over_union": float(iou[selected]),
            "intersection_pixels": int(total_intersection[selected]),
            "predicted_positive_pixels": int(total_predicted[selected]),
        },
    }


def group_held_audit(
    labels: np.ndarray,
    groups: np.ndarray,
    raw_scores: np.ndarray,
    pixel: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    folds = stable_group_folds(groups)
    held_scores = np.zeros(len(labels), dtype=np.float64)
    held_decisions = {
        str(target): np.zeros(len(labels), dtype=bool) for target in TARGET_FPRS
    }
    held_pixel_intersection = np.zeros(len(labels), dtype=np.int32)
    held_pixel_predicted = np.zeros(len(labels), dtype=np.int32)
    fold_reports: list[dict[str, Any]] = []
    for fold in range(CALIBRATION_FOLDS):
        training = folds != fold
        held = folds == fold
        training_seed_scores = np.stack(
            [empirical_percentile(seed[training], seed[training]) for seed in raw_scores]
        )
        held_seed_scores = np.stack(
            [empirical_percentile(seed[training], seed[held]) for seed in raw_scores]
        )
        training_ensemble = training_seed_scores.mean(axis=0)
        held_ensemble = held_seed_scores.mean(axis=0)
        held_scores[held] = held_ensemble
        thresholds: dict[str, Any] = {}
        for target in TARGET_FPRS:
            selection = choose_threshold_at_fpr(
                labels[training], training_ensemble, target
            )
            held_decisions[str(target)][held] = held_ensemble >= selection["threshold"]
            thresholds[str(target)] = selection
        intersections = pixel["intersections_by_threshold_scene"]
        predicted = pixel["predicted_by_threshold_scene"]
        truth_area = pixel["truth_by_scene"]
        training_intersection = intersections[:, training].sum(axis=1, dtype=np.int64)
        training_predicted = predicted[:, training].sum(axis=1, dtype=np.int64)
        training_truth = int(truth_area[training].sum())
        dice = 2.0 * training_intersection / np.maximum(
            training_predicted + training_truth, 1
        )
        selected_pixel = int(np.argmax(dice))
        held_pixel_intersection[held] = intersections[selected_pixel, held]
        held_pixel_predicted[held] = predicted[selected_pixel, held]
        fold_reports.append(
            {
                "fold": fold,
                "training_groups": len(set(groups[training])),
                "held_groups": len(set(groups[held])),
                "training_scenes": int(np.count_nonzero(training)),
                "held_scenes": int(np.count_nonzero(held)),
                "scene_thresholds_selected_on_training_groups": thresholds,
                "pixel_threshold_selected_on_training_groups": PIXEL_THRESHOLDS[
                    selected_pixel
                ],
            }
        )
    total_intersection = int(held_pixel_intersection.sum())
    total_predicted = int(held_pixel_predicted.sum())
    total_truth = int(pixel["truth_by_scene"].sum())
    union = total_predicted + total_truth - total_intersection
    metrics = {
        "method": (
            "five stable-hash folds of frozen 25 km groups; calibrate each seed and select "
            "scene/pixel thresholds on four folds, then apply to the held fold"
        ),
        "ranking": {
            "average_precision": float(average_precision_score(labels, held_scores)),
            "auroc": float(roc_auc_score(labels, held_scores)),
        },
        "operating_points": {
            target: binary_metrics(labels, decision)
            for target, decision in held_decisions.items()
        },
        "segmentation": {
            "dice": 2.0 * total_intersection / max(total_predicted + total_truth, 1),
            "intersection_over_union": total_intersection / max(union, 1),
            "intersection_pixels": total_intersection,
            "predicted_positive_pixels": total_predicted,
            "truth_positive_pixels": total_truth,
        },
        "folds": fold_reports,
    }
    values = {
        "scene_score": held_scores,
        "decision_0.05": held_decisions["0.05"],
        "pixel_intersection": held_pixel_intersection,
        "pixel_predicted": held_pixel_predicted,
        "pixel_truth": pixel["truth_by_scene"],
        "fold": folds,
    }
    return metrics, values


def bootstrap_groups(
    labels: np.ndarray,
    groups: np.ndarray,
    audit_values: dict[str, np.ndarray],
    point: dict[str, Any],
) -> dict[str, Any]:
    unique = np.asarray(sorted(set(str(value) for value in groups)))
    by_group = {
        group: np.flatnonzero(groups.astype(str) == group) for group in unique
    }
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples: dict[str, list[float]] = {
        "average_precision": [],
        "auroc": [],
        "recall": [],
        "false_positive_rate": [],
        "precision": [],
        "pixel_dice": [],
        "pixel_intersection_over_union": [],
    }
    attempts = 0
    while len(samples["average_precision"]) < BOOTSTRAP_REPLICATES:
        attempts += 1
        selected_groups = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([by_group[group] for group in selected_groups])
        local_labels = labels[indices]
        if len(np.unique(local_labels)) != 2:
            if attempts > BOOTSTRAP_REPLICATES * 10:
                raise RuntimeError("Too many class-degenerate group bootstrap samples")
            continue
        score = audit_values["scene_score"][indices]
        binary = binary_metrics(local_labels, audit_values["decision_0.05"][indices])
        intersection = int(audit_values["pixel_intersection"][indices].sum())
        predicted = int(audit_values["pixel_predicted"][indices].sum())
        truth_area = int(audit_values["pixel_truth"][indices].sum())
        union = predicted + truth_area - intersection
        samples["average_precision"].append(float(average_precision_score(local_labels, score)))
        samples["auroc"].append(float(roc_auc_score(local_labels, score)))
        for name in ("recall", "false_positive_rate", "precision"):
            samples[name].append(float(binary[name]))
        samples["pixel_dice"].append(2.0 * intersection / max(predicted + truth_area, 1))
        samples["pixel_intersection_over_union"].append(
            intersection / max(union, 1)
        )
    point_values = {
        "average_precision": point["ranking"]["average_precision"],
        "auroc": point["ranking"]["auroc"],
        "recall": point["operating_points"]["0.05"]["recall"],
        "false_positive_rate": point["operating_points"]["0.05"][
            "false_positive_rate"
        ],
        "precision": point["operating_points"]["0.05"]["precision"],
        "pixel_dice": point["segmentation"]["dice"],
        "pixel_intersection_over_union": point["segmentation"][
            "intersection_over_union"
        ],
    }
    result: dict[str, Any] = {
        "method": "2000 nonparametric resamples of the 64 frozen 25 km groups with replacement",
        "seed": BOOTSTRAP_SEED,
        "replicates": BOOTSTRAP_REPLICATES,
    }
    for name, values in samples.items():
        array = np.asarray(values)
        result[name] = {
            "point_estimate": float(point_values[name]),
            "bootstrap_mean": float(np.mean(array)),
            "bootstrap_standard_deviation": float(np.std(array, ddof=1)),
            "95ci": [float(value) for value in np.quantile(array, (0.025, 0.975))],
        }
    return result


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    held = report["group_held_calibration_audit"]
    final = report["final_all_development_rule"]
    lines = [
        "# MethaneS2CM v5.1 three-seed ensemble validation",
        "",
        "Frozen internal-development result. Location-test imagery remains sealed.",
        "",
        "| Evaluation | Scene AP | AUROC | Recall @ 5% FPR target | Realized FPR | Pixel Dice | Pixel IoU |",
        "|---|---:|---:|---:|---:|---:|---:|",
        (
            f"| Five-fold 25 km group-held calibration | {held['ranking']['average_precision']:.4f} | "
            f"{held['ranking']['auroc']:.4f} | {held['operating_points']['0.05']['recall']:.4f} | "
            f"{held['operating_points']['0.05']['false_positive_rate']:.4f} | "
            f"{held['segmentation']['dice']:.4f} | {held['segmentation']['intersection_over_union']:.4f} |"
        ),
        (
            f"| Final all-development frozen rule | {final['ranking']['average_precision']:.4f} | "
            f"{final['ranking']['auroc']:.4f} | {final['operating_points']['0.05']['recall']:.4f} | "
            f"{final['operating_points']['0.05']['false_positive_rate']:.4f} | "
            f"{final['segmentation']['dice']:.4f} | {final['segmentation']['intersection_over_union']:.4f} |"
        ),
        "",
        f"- Seeds: {', '.join(str(seed['seed']) for seed in report['seeds'])}",
        f"- Frozen final scene threshold at 5% target FPR: {final['operating_points']['0.05']['threshold']:.6f}",
        f"- Frozen pixel threshold: {final['segmentation']['threshold']:.2f}",
        f"- Ensemble pixel AP: {final['segmentation']['average_precision']:.4f}",
        "- Bootstrap intervals resample the 64 frozen 25 km groups.",
        "- Checkpoint selection still used this development cohort; the sealed location test is the external estimate.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL.as_posix())
    parser.add_argument("--reports", nargs=3, default=[path.as_posix() for path in DEFAULT_REPORTS])
    parser.add_argument("--packed", default=DEFAULT_PACKED.as_posix())
    parser.add_argument("--v5-baseline", default=DEFAULT_V5_BASELINE.as_posix())
    parser.add_argument("--report", default=DEFAULT_REPORT.as_posix())
    parser.add_argument("--markdown", default=DEFAULT_MARKDOWN.as_posix())
    parser.add_argument("--calibration-cache", default=DEFAULT_CACHE.as_posix())
    args = parser.parse_args()

    root = repo_root()
    if tracked_dirty(root):
        raise RuntimeError("Refusing v5.1 aggregation from a dirty tracked worktree")
    protocol_path = (root / args.protocol).resolve()
    packed_path = (root / args.packed).resolve()
    baseline_path = (root / args.v5_baseline).resolve()
    report_path = safe_output(root, args.report)
    markdown_path = safe_output(root, args.markdown)
    calibration_path = safe_output(root, args.calibration_cache)
    if (
        subprocess.run(
            ["git", "check-ignore", "--quiet", "--", calibration_path.relative_to(root)],
            cwd=root,
            check=False,
        ).returncode
        != 0
    ):
        raise ValueError("Ensemble calibration cache must be ignored by Git")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if sha256(packed_path) != protocol["source"]["packed_train_sha256"]:
        raise ValueError("Packed training/development data violates the frozen source hash")
    report_paths = [(root / value).resolve() for value in args.reports]
    reports: list[dict[str, Any]] = []
    for path in report_paths:
        value = json.loads(path.read_text(encoding="utf-8"))
        value["_path"] = path.relative_to(root).as_posix()
        reports.append(value)
    expected_seeds = [int(value) for value in protocol["training"]["seeds"]]
    if [int(report["seed"]) for report in reports] != expected_seeds:
        raise ValueError("Reports do not match the frozen campaign seed order")
    for report in reports:
        model = report["model"]
        if (
            model["model_name"] != protocol["architecture"]["name"]
            or float(model["context_scene_weight"])
            != float(protocol["architecture"]["context_scene_weight"])
        ):
            raise ValueError("Seed report violates the frozen v5.1 architecture")
        if report["seal"]["packed_sha256"] != protocol["source"]["packed_train_sha256"]:
            raise ValueError("Seed report violates the frozen packed data identity")
        checkpoint_path = root / report["checkpoint"]["path"]
        if sha256(checkpoint_path) != report["checkpoint"]["sha256"]:
            raise ValueError(f"Checkpoint hash mismatch: {checkpoint_path}")

    shared, raw_scores = load_aligned_caches(root, reports)
    labels = shared["label"].astype(np.uint8)
    groups = shared["group_id"].astype(str)
    aligned_probabilities: list[np.ndarray] = []
    for report in reports:
        cache_path = root / report["prediction_cache"]["path"]
        with np.load(cache_path, allow_pickle=False) as source:
            identifiers = source["sample_id"].astype(np.int64)
            index = {int(identifier): row for row, identifier in enumerate(identifiers)}
            order = np.asarray([index[int(value)] for value in shared["sample_id"]])
            aligned_probabilities.append(
                source["segmentation_probability"][order].astype(np.float32)
            )
    ensemble_probability = np.mean(np.stack(aligned_probabilities), axis=0)
    truth = read_truth(packed_path, shared["packed_index"])
    observable = shared["observable"].astype(bool)
    if not np.array_equal(np.any(truth & observable, axis=(1, 2)).astype(np.uint8), labels):
        raise ValueError("Packed truth disagrees with ensemble scene labels")
    pixel = pixel_statistics(ensemble_probability, truth, observable)

    calibrated = np.stack(
        [empirical_percentile(seed_scores, seed_scores) for seed_scores in raw_scores]
    )
    final_scene_scores = calibrated.mean(axis=0)
    final_points = {
        str(target): choose_threshold_at_fpr(labels, final_scene_scores, target)
        for target in TARGET_FPRS
    }
    final_rule = {
        "scene_calibration": "equal mean of three per-seed empirical-CDF percentiles fitted on all internal development scores",
        "ranking": {
            "average_precision": float(average_precision_score(labels, final_scene_scores)),
            "auroc": float(roc_auc_score(labels, final_scene_scores)),
        },
        "operating_points": final_points,
        "segmentation": {
            "method": "equal arithmetic mean of three seed probability maps",
            "average_precision": pixel["average_precision"],
            "threshold": pixel["selected"]["threshold"],
            "dice": pixel["selected"]["dice"],
            "intersection_over_union": pixel["selected"]["intersection_over_union"],
            "intersection_pixels": pixel["selected"]["intersection_pixels"],
            "predicted_positive_pixels": pixel["selected"]["predicted_positive_pixels"],
            "truth_positive_pixels": pixel["truth_pixels"],
            "observable_pixels": pixel["observable_pixels"],
            "threshold_grid": pixel["grid"],
        },
    }
    held_audit, held_values = group_held_audit(labels, groups, raw_scores, pixel)
    uncertainty = bootstrap_groups(labels, groups, held_values, held_audit)

    calibration_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = calibration_path.with_suffix(calibration_path.suffix + ".tmp")
    with temporary.open("wb") as destination:
        np.savez_compressed(
            destination,
            seeds=np.asarray(expected_seeds, dtype=np.int64),
            seed_sorted_development_scores=np.sort(raw_scores, axis=1).astype(np.float32),
            target_fprs=np.asarray(TARGET_FPRS, dtype=np.float64),
            scene_thresholds=np.asarray(
                [final_points[str(target)]["threshold"] for target in TARGET_FPRS],
                dtype=np.float64,
            ),
            pixel_threshold=np.asarray([pixel["selected"]["threshold"]], dtype=np.float64),
            sample_id=shared["sample_id"],
            label=labels,
            group_id=groups,
            final_development_scene_score=final_scene_scores.astype(np.float32),
            crossfit_development_scene_score=held_values["scene_score"].astype(np.float32),
            ensemble_development_probability=ensemble_probability.astype(np.float16),
        )
    os.replace(temporary, calibration_path)

    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    seed_summaries = [
        {
            "seed": int(report["seed"]),
            "best_epoch": int(report["training"]["best_epoch"]),
            "scene_average_precision": report["validation"]["average_precision"],
            "scene_auroc": report["validation"]["auroc"],
            "recall_at_fpr_le_0.05": report["validation"]["operating_points"]["0.05"][
                "recall"
            ],
            "pixel_average_precision": report["validation"]["segmentation"][
                "average_precision_all_observable_pixels"
            ],
            "pixel_dice": report["validation"]["segmentation"]["dice"],
            "pixel_intersection_over_union": report["validation"]["segmentation"][
                "intersection_over_union"
            ],
            "checkpoint": report["checkpoint"],
            "prediction_cache": report["prediction_cache"],
            "report_path": report["_path"],
            "report_sha256": sha256(root / report["_path"]),
        }
        for report in reports
    ]
    report = {
        "schema_version": 1,
        "scope": "methanes2cm_v5_1_three_seed_internal_development_ensemble",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "campaign_protocol": {
            "path": protocol_path.relative_to(root).as_posix(),
            "sha256": sha256(protocol_path),
        },
        "cohort": {
            "scenes": int(len(labels)),
            "positives": int(np.count_nonzero(labels == 1)),
            "negatives": int(np.count_nonzero(labels == 0)),
            "geographic_groups": len(set(groups)),
            "location_test_images_opened": False,
        },
        "seeds": seed_summaries,
        "seed_mean": {
            name: float(np.mean([seed[name] for seed in seed_summaries]))
            for name in (
                "scene_average_precision",
                "scene_auroc",
                "recall_at_fpr_le_0.05",
                "pixel_average_precision",
                "pixel_dice",
                "pixel_intersection_over_union",
            )
        },
        "group_held_calibration_audit": held_audit,
        "group_bootstrap": uncertainty,
        "final_all_development_rule": final_rule,
        "controlled_v5_seed1101_reference": {
            "scene_average_precision": baseline["validation"]["average_precision"],
            "scene_auroc": baseline["validation"]["auroc"],
            "recall_at_fpr_le_0.05": baseline["validation"]["operating_points"]["0.05"][
                "recall"
            ],
            "pixel_average_precision": baseline["validation"]["segmentation"][
                "average_precision_all_observable_pixels"
            ],
            "pixel_dice": baseline["validation"]["segmentation"]["dice"],
            "pixel_intersection_over_union": baseline["validation"]["segmentation"][
                "intersection_over_union"
            ],
            "report_sha256": sha256(baseline_path),
        },
        "calibration_cache": {
            "path": calibration_path.relative_to(root).as_posix(),
            "bytes": calibration_path.stat().st_size,
            "sha256": sha256(calibration_path),
            "tracked": False,
        },
        "freeze": {
            "architecture_frozen": True,
            "checkpoints_frozen": True,
            "scene_calibrators_frozen": True,
            "scene_thresholds_frozen": True,
            "pixel_threshold_frozen": True,
            "location_test_still_sealed": True,
            "remaining_before_unlock": "commit the one-shot test acquisition and comparison evaluator against these hashes",
        },
        "interpretation": (
            "Internal development confirmation. Group-held calibration reduces calibration-rule "
            "optimism, but checkpoints were selected on this cohort; external performance is "
            "unknown until the one-shot location test."
        ),
        "provenance": {
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True
            ).strip(),
            "script": Path(__file__).resolve().relative_to(root).as_posix(),
            "script_sha256": sha256(Path(__file__).resolve()),
            "tracked_worktree_dirty_at_start": False,
        },
    }
    write_json(report_path, report)
    write_markdown(markdown_path, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
