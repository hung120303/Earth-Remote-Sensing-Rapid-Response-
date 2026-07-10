#!/usr/bin/env python3
"""Benchmark compact residual U-Nets on the quality-filtered ERSRR cohort.

This is a CPU-friendly, segmentation-first experiment.  It uses group-held-out
outer folds, a disjoint group-held-out inner validation set for early stopping,
train-fold-only channel statistics, and explicit target-validity masking.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import keras
import numpy as np
import rasterio
import sklearn
import tensorflow as tf
from rasterio.enums import Resampling
from sklearn.model_selection import GroupKFold

MODEL_DIR = Path(__file__).resolve().parents[1] / "EarthRemoteSensingRapidResponse"
sys.path.insert(0, str(MODEL_DIR))

from ersrr_core import (  # noqa: E402
    MaskedBCEDice,
    build_compact_resunet,
    masked_dice,
    masked_iou,
    normalize_features,
    transform_sentinel2,
)
from audit_dataset import parse_filename
from run_research_baselines import (
    CALIBRATION_THRESHOLDS,
    classification_metrics,
    json_safe,
    leakage_safe_scene_groups,
    sample_indices,
)


def repo_root() -> Path:
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        check=True,
        capture_output=True,
        text=True,
    )
    return Path(result.stdout.strip()).resolve()


def git_value(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def cohort_sha256(root: Path, metadata: list[dict[str, Any]]) -> str:
    """Hash ordered relative paths and bytes for every selected cohort file."""
    digest = hashlib.sha256()
    digest.update(b"ersrr-quality-cohort-v1\0")
    for item in sorted(metadata, key=lambda value: value["file"]):
        relative = str(item["file"]).replace("\\", "/")
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        with (root / relative).open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def load_scenes(
    dataset_dir: Path,
    *,
    image_size: int,
    threshold: float,
    min_valid_pct: float,
    max_abs_gap_days: float,
) -> dict[str, Any]:
    images: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    valid_masks: list[np.ndarray] = []
    groups: list[str] = []
    names: list[str] = []
    metadata: list[dict[str, Any]] = []
    exclusions: dict[str, int] = {}

    def exclude(reason: str) -> None:
        exclusions[reason] = exclusions.get(reason, 0) + 1

    for path in sorted(dataset_dir.glob("*.tif")):
        info = parse_filename(path)
        if not info.get("emit_id") or not info.get("s2_tile"):
            exclude("missing_provenance")
            continue
        gap = info.get("days_between_s2_emit")
        if gap is None or abs(float(gap)) > max_abs_gap_days:
            exclude("temporal_gap")
            continue
        with rasterio.open(path) as source:
            if source.count != 6:
                exclude("band_contract")
                continue
            bands = source.read(
                indexes=(1, 2, 3, 4, 5),
                out_shape=(5, image_size, image_size),
                resampling=Resampling.bilinear,
            ).astype("float32")
            target = source.read(
                6,
                out_shape=(image_size, image_size),
                resampling=Resampling.nearest,
            ).astype("float32")
        image = np.moveaxis(bands, 0, -1)
        valid = np.isfinite(target) & (target != -9999.0) & np.all(np.isfinite(image), axis=-1)
        valid_pct = float(100.0 * valid.mean())
        if valid_pct < min_valid_pct:
            exclude("valid_coverage")
            continue
        images.append(image)
        labels.append((target > threshold).astype("float32"))
        valid_masks.append(valid.astype("float32"))
        groups.append(str(info["s2_tile"]))
        names.append(path.name)
        metadata.append(
            {
                "file": str(path.relative_to(repo_root())),
                "s2_tile": info["s2_tile"],
                "s2_scene": f"{info['s2_start']}_{info['s2_end']}",
                "emit_id": info["emit_id"],
                "gap_days": float(gap),
                "valid_pct": valid_pct,
                "positive_pct_of_valid": float(100.0 * np.count_nonzero((target > threshold) & valid) / max(valid.sum(), 1)),
            }
        )
    if not images:
        raise RuntimeError(f"No qualifying images found under {dataset_dir}")
    group_by_file = leakage_safe_scene_groups(metadata)
    groups = [group_by_file[item["file"]] for item in metadata]
    for item, group in zip(metadata, groups):
        item["leakage_group"] = group
    return {
        "images": np.stack(images),
        "labels": np.stack(labels)[..., None],
        "valid": np.stack(valid_masks)[..., None],
        "groups": np.asarray(groups, dtype=object),
        "names": np.asarray(names, dtype=object),
        "metadata": metadata,
        "exclusions": exclusions,
    }


def transform_images(images: np.ndarray, architecture: str) -> np.ndarray:
    if architecture == "raw_resunet":
        return transform_sentinel2(images, physics=False)
    if architecture == "physics_resunet":
        return transform_sentinel2(images, physics=True)
    raise ValueError(f"Unknown architecture: {architecture}")


def inner_group_split(indices: np.ndarray, groups: np.ndarray, seed: int, val_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    unique = np.unique(groups[indices])
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    val_count = max(1, int(round(unique.size * val_fraction)))
    val_groups = set(unique[:val_count])
    is_val = np.asarray([group in val_groups for group in groups[indices]])
    return indices[~is_val], indices[is_val]


def augment_quarter_turns(images: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    image_parts = [np.rot90(images, k=k, axes=(1, 2)) for k in range(4)]
    target_parts = [np.rot90(targets, k=k, axes=(1, 2)) for k in range(4)]
    return np.concatenate(image_parts), np.concatenate(target_parts)


def normalization_stats(features: np.ndarray, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    selected = features[indices]
    mean = selected.mean(axis=(0, 1, 2), dtype="float64").astype("float32")
    std = selected.std(axis=(0, 1, 2), dtype="float64").astype("float32")
    std = np.where(std < 1e-6, 1.0, std)
    return mean, std


def evaluate_predictions(
    labels: np.ndarray,
    valid: np.ndarray,
    probabilities: np.ndarray,
    names: np.ndarray,
    *,
    pixels_per_scene: int,
    seed: int,
    decision_threshold: float,
) -> dict[str, Any]:
    full_rows = []
    sampled_rows = []
    full_truth_parts = []
    full_score_parts = []
    sampled_truth_parts = []
    sampled_score_parts = []
    for index, name in enumerate(names):
        is_valid = valid[index, ..., 0].astype(bool)
        truth = labels[index, ..., 0][is_valid].astype("uint8")
        scores = probabilities[index, ..., 0][is_valid]
        full_rows.append(classification_metrics(truth, scores, threshold=decision_threshold))
        full_truth_parts.append(truth)
        full_score_parts.append(scores)

        sample = sample_indices(is_valid, pixels_per_scene, name=str(name), seed=seed)
        truth_flat = labels[index, ..., 0].reshape(-1)[sample].astype("uint8")
        scores_flat = probabilities[index, ..., 0].reshape(-1)[sample]
        sampled_rows.append(classification_metrics(truth_flat, scores_flat, threshold=decision_threshold))
        sampled_truth_parts.append(truth_flat)
        sampled_score_parts.append(scores_flat)

    def summarize(rows: list[dict[str, float]]) -> dict[str, float]:
        result = {}
        for key in rows[0]:
            values = np.asarray([row[key] for row in rows], dtype="float64")
            result[f"{key}_mean"] = float(np.nanmean(values))
            result[f"{key}_std"] = float(np.nanstd(values))
        return result

    return {
        "full_pixel": classification_metrics(
            np.concatenate(full_truth_parts),
            np.concatenate(full_score_parts),
            threshold=decision_threshold,
        ),
        "sampled_pixel": classification_metrics(
            np.concatenate(sampled_truth_parts),
            np.concatenate(sampled_score_parts),
            threshold=decision_threshold,
        ),
        "full_scene_macro": summarize(full_rows),
        "sampled_scene_macro": summarize(sampled_rows),
    }


def calibrate_threshold(labels: np.ndarray, valid: np.ndarray, probabilities: np.ndarray) -> float:
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in CALIBRATION_THRESHOLDS:
        rows = []
        for index in range(labels.shape[0]):
            is_valid = valid[index, ..., 0].astype(bool)
            truth = labels[index, ..., 0][is_valid].astype("uint8")
            scores = probabilities[index, ..., 0][is_valid]
            rows.append(classification_metrics(truth, scores, threshold=float(threshold)))
        macro_f1 = float(np.nanmean([row["f1"] for row in rows]))
        if macro_f1 > best_f1 + 1e-12 or (
            abs(macro_f1 - best_f1) <= 1e-12 and abs(float(threshold) - 0.5) < abs(best_threshold - 0.5)
        ):
            best_f1 = macro_f1
            best_threshold = float(threshold)
    return best_threshold


def train_architecture(
    architecture: str,
    dataset: dict[str, Any],
    *,
    folds: int,
    seed: int,
    epochs: int,
    batch_size: int,
    base_filters: int,
    learning_rate: float,
    inner_val_fraction: float,
    sampled_pixels: int,
    positive_weight_override: float | None,
) -> dict[str, Any]:
    features = transform_images(dataset["images"], architecture)
    labels = dataset["labels"]
    valid = dataset["valid"]
    targets = np.concatenate([labels, valid], axis=-1)
    groups = dataset["groups"]
    splitter = GroupKFold(n_splits=folds)
    fold_rows = []
    started = time.perf_counter()

    for fold, (outer_train, outer_test) in enumerate(splitter.split(features, groups=groups), start=1):
        fold_started = time.perf_counter()
        fit_indices, inner_val = inner_group_split(
            outer_train,
            groups,
            seed=seed + fold,
            val_fraction=inner_val_fraction,
        )
        mean, std = normalization_stats(features, fit_indices)
        normalized = normalize_features(features, mean, std)
        x_fit, y_fit = augment_quarter_turns(normalized[fit_indices], targets[fit_indices])
        fit_valid = valid[fit_indices]
        fit_positive = float(np.sum(labels[fit_indices] * fit_valid))
        fit_negative = float(np.sum((1.0 - labels[fit_indices]) * fit_valid))
        positive_weight = (
            float(positive_weight_override)
            if positive_weight_override is not None
            else float(np.clip(fit_negative / max(fit_positive, 1.0), 1.0, 25.0))
        )

        keras.backend.clear_session()
        keras.utils.set_random_seed(seed + fold)
        model = build_compact_resunet(
            input_channels=normalized.shape[-1],
            base_filters=base_filters,
            image_size=normalized.shape[1],
        )
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0),
            loss=MaskedBCEDice(positive_weight),
            metrics=[masked_dice, masked_iou],
        )
        print(
            f"{architecture} fold {fold}/{folds}: fit={fit_indices.size}, "
            f"val={inner_val.size}, test={outer_test.size}, params={model.count_params()}",
            flush=True,
        )
        history = model.fit(
            x_fit,
            y_fit,
            validation_data=(normalized[inner_val], targets[inner_val]),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=3,
                    min_delta=1e-4,
                    restore_best_weights=True,
                )
            ],
            verbose=0,
        )
        validation_probabilities = model.predict(normalized[inner_val], batch_size=batch_size, verbose=0)
        decision_threshold = calibrate_threshold(
            labels[inner_val],
            valid[inner_val],
            validation_probabilities,
        )
        probabilities = model.predict(normalized[outer_test], batch_size=batch_size, verbose=0)
        metrics = evaluate_predictions(
            labels[outer_test],
            valid[outer_test],
            probabilities,
            dataset["names"][outer_test],
            pixels_per_scene=sampled_pixels,
            seed=seed,
            decision_threshold=decision_threshold,
        )
        fold_rows.append(
            {
                "fold": fold,
                "fit_scenes": int(fit_indices.size),
                "validation_scenes": int(inner_val.size),
                "test_scenes": int(outer_test.size),
                "fit_groups": int(np.unique(groups[fit_indices]).size),
                "validation_groups": int(np.unique(groups[inner_val]).size),
                "test_groups": int(np.unique(groups[outer_test]).size),
                "epochs_ran": len(history.history["loss"]),
                "best_val_loss": float(np.min(history.history["val_loss"])),
                "positive_weight": positive_weight,
                "decision_threshold": decision_threshold,
                "normalization_mean": mean.tolist(),
                "normalization_std": std.tolist(),
                "metrics": metrics,
                "runtime_seconds": round(time.perf_counter() - fold_started, 3),
            }
        )
        print(
            f"{architecture} fold {fold}: sampled scene F1="
            f"{metrics['sampled_scene_macro']['f1_mean']:.4f}, IoU="
            f"{metrics['sampled_scene_macro']['iou_mean']:.4f}, threshold={decision_threshold:.3f}",
            flush=True,
        )

    summary: dict[str, dict[str, float]] = {}
    for scope in ("full_pixel", "sampled_pixel", "full_scene_macro", "sampled_scene_macro"):
        summary[scope] = {}
        keys = fold_rows[0]["metrics"][scope]
        for key in keys:
            values = np.asarray([row["metrics"][scope][key] for row in fold_rows], dtype="float64")
            if scope.endswith("scene_macro"):
                if not key.endswith("_mean"):
                    continue
                summary[scope][key] = float(np.nanmean(values))
                summary[scope][key.replace("_mean", "_fold_std")] = float(np.nanstd(values))
            else:
                summary[scope][f"{key}_mean"] = float(np.nanmean(values))
                summary[scope][f"{key}_fold_std"] = float(np.nanstd(values))
    return {
        "architecture": architecture,
        "input_channels": int(features.shape[-1]),
        "parameters": int(
            build_compact_resunet(
                input_channels=features.shape[-1],
                base_filters=base_filters,
                image_size=features.shape[1],
            ).count_params()
        ),
        "runtime_seconds": round(time.perf_counter() - started, 3),
        "summary": summary,
        "folds": fold_rows,
    }


def write_markdown(path: Path, results: dict[str, Any]) -> None:
    lines = [
        "# ERSRR compact residual U-Net experiment",
        "",
        f"Generated: `{results['generated_at_utc']}`",
        "",
        f"Cohort: {results['cohort']['selected_scenes']} scenes / {results['cohort']['groups']} "
        "leakage-safe scene components; "
        f"target > {results['target_threshold_ppm_m']:g} ppm·m.",
        "",
        "| Architecture | Channels | Parameters | Sampled AUPRC | Sampled AUROC | Sampled F1 | Sampled IoU | Runtime s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model in results["models"]:
        ranking = model["summary"]["sampled_pixel"]
        scene = model["summary"]["sampled_scene_macro"]
        lines.append(
            f"| {model['architecture']} | {model['input_channels']} | {model['parameters']:,} | "
            f"{ranking['auprc_mean']:.4f} | {ranking['roc_auc_mean']:.4f} | "
            f"{scene['f1_mean']:.4f} | {scene['iou_mean']:.4f} | {model['runtime_seconds']:.1f} |"
        )
    lines.extend(
        [
            "",
            "Outer test groups are never used for normalization or early stopping. Each outer training fold "
            "contains a disjoint group-held-out inner validation subset, which also calibrates the decision threshold.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default="EarthRemoteSensingRapidResponse/Dataset/train_test")
    parser.add_argument("--output", default="reports/experiments/unet_results.json")
    parser.add_argument("--markdown", default="reports/experiments/UNET_RESULTS.md")
    parser.add_argument("--architectures", nargs="+", choices=("raw_resunet", "physics_resunet"), default=["raw_resunet", "physics_resunet"])
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--threshold", type=float, default=300.0)
    parser.add_argument("--min-valid-pct", type=float, default=10.0)
    parser.add_argument("--max-abs-gap-days", type=float, default=7.0)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--base-filters", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument(
        "--positive-weight",
        type=float,
        help="Fixed positive-class weight; omit to estimate it from each fit partition",
    )
    parser.add_argument("--inner-val-fraction", type=float, default=0.2)
    parser.add_argument("--sampled-pixels", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.positive_weight is not None and args.positive_weight <= 0.0:
        parser.error("positive-weight must be positive")
    root = repo_root()
    started = time.perf_counter()
    try:
        tf.config.experimental.enable_op_determinism()
        dataset = load_scenes(
            root / args.dataset_dir,
            image_size=args.image_size,
            threshold=args.threshold,
            min_valid_pct=args.min_valid_pct,
            max_abs_gap_days=args.max_abs_gap_days,
        )
        models = [
            train_architecture(
                architecture,
                dataset,
                folds=args.folds,
                seed=args.seed,
                epochs=args.epochs,
                batch_size=args.batch_size,
                base_filters=args.base_filters,
                learning_rate=args.learning_rate,
                inner_val_fraction=args.inner_val_fraction,
                sampled_pixels=args.sampled_pixels,
                positive_weight_override=args.positive_weight,
            )
            for architecture in args.architectures
        ]
        results = {
            "schema_version": 1,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "tensorflow_version": tf.__version__,
            "keras_version": keras.__version__,
            "device": "CPU" if not tf.config.list_physical_devices("GPU") else "GPU",
            "image_size": args.image_size,
            "target_threshold_ppm_m": args.threshold,
            "folds": args.folds,
            "epochs_max": args.epochs,
            "batch_size": args.batch_size,
            "base_filters": args.base_filters,
            "positive_weight": args.positive_weight if args.positive_weight is not None else "fit_partition_ratio",
            "seed": args.seed,
            "provenance": {
                "git_commit": git_value(root, "rev-parse", "HEAD"),
                "git_tracked_worktree_dirty": bool(
                    git_value(root, "status", "--porcelain", "--untracked-files=no")
                ),
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "cohort_sha256": cohort_sha256(root, dataset["metadata"]),
                "python": sys.version.split()[0],
                "tensorflow": tf.__version__,
                "keras": keras.__version__,
                "numpy": np.__version__,
                "rasterio": rasterio.__version__,
                "scikit_learn": sklearn.__version__,
            },
            "cohort": {
                "selected_scenes": int(dataset["images"].shape[0]),
                "groups": int(np.unique(dataset["groups"]).size),
                "exclusions": dataset["exclusions"],
                "files": dataset["metadata"],
            },
            "models": models,
            "total_runtime_seconds": round(time.perf_counter() - started, 3),
        }
        safe = json_safe(results)
        output_path = root / args.output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(safe, indent=2, sort_keys=True), encoding="utf-8")
        write_markdown(root / args.markdown, safe)
        print(json.dumps({"ok": True, "output": args.output, "markdown": args.markdown, "runtime_seconds": safe["total_runtime_seconds"]}, indent=2))
        return 0
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc), "error_type": type(exc).__name__}, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
