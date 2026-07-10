#!/usr/bin/env python3
"""Train a selected research-only ERSRR compact ResUNet artifact.

The fit and calibration partitions are disjoint by Sentinel-2 MGRS tile.  Only
fit groups contribute normalization statistics or gradient updates; the held-
out groups select early stopping and the final probability threshold.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import keras
import numpy as np
import tensorflow as tf

MODEL_DIR = Path(__file__).resolve().parents[1] / "EarthRemoteSensingRapidResponse"
sys.path.insert(0, str(MODEL_DIR))

from ersrr_core import (  # noqa: E402
    ARTIFACT_SCHEMA_VERSION,
    BAND_ORDER,
    FEATURE_NAMES,
    MODEL_NAME,
    RAW_FEATURE_NAMES,
    MaskedBCEDice,
    build_compact_resunet,
    load_artifact,
    masked_dice,
    masked_iou,
    normalize_features,
    predict_tile,
    sha256,
    transform_sentinel2,
)
from run_unet_experiment import (  # noqa: E402
    augment_quarter_turns,
    calibrate_threshold,
    cohort_sha256,
    evaluate_predictions,
    inner_group_split,
    json_safe,
    load_scenes,
    normalization_stats,
    repo_root,
)


def git_value(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default="EarthRemoteSensingRapidResponse/Dataset/train_test")
    parser.add_argument(
        "--artifact-dir",
        default="EarthRemoteSensingRapidResponse/artifacts/compact_resunet_v1",
    )
    parser.add_argument(
        "--architecture",
        choices=("raw_resunet", "physics_resunet"),
        default="raw_resunet",
    )
    parser.add_argument("--image-size", type=int, choices=(128,), default=128)
    parser.add_argument("--threshold", type=float, default=300.0)
    parser.add_argument("--min-valid-pct", type=float, default=10.0)
    parser.add_argument("--max-abs-gap-days", type=float, default=7.0)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--base-filters", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--positive-weight", type=float, default=1.0)
    parser.add_argument("--sampled-pixels", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.image_size != 128:
        parser.error("the legacy 10 m cohort must be trained at 128 pixels for the 20 m serving contract")
    if args.epochs <= 0 or args.patience < 0 or args.batch_size <= 0:
        parser.error("epochs and batch-size must be positive; patience cannot be negative")
    if not 0.0 < args.validation_fraction < 1.0:
        parser.error("validation-fraction must be between 0 and 1")
    if args.positive_weight <= 0.0:
        parser.error("positive-weight must be positive")


def train(args: argparse.Namespace) -> dict[str, Any]:
    root = repo_root()
    started = time.perf_counter()
    source_commit = git_value(root, "rev-parse", "HEAD")
    source_dirty = bool(git_value(root, "status", "--porcelain", "--untracked-files=no"))

    dataset = load_scenes(
        root / args.dataset_dir,
        image_size=args.image_size,
        threshold=args.threshold,
        min_valid_pct=args.min_valid_pct,
        max_abs_gap_days=args.max_abs_gap_days,
    )
    all_indices = np.arange(dataset["images"].shape[0])
    fit_indices, calibration_indices = inner_group_split(
        all_indices,
        dataset["groups"],
        seed=args.seed,
        val_fraction=args.validation_fraction,
    )
    if not fit_indices.size or not calibration_indices.size:
        raise RuntimeError("Group split produced an empty fit or calibration partition")
    fit_groups = set(dataset["groups"][fit_indices])
    calibration_groups = set(dataset["groups"][calibration_indices])
    if fit_groups & calibration_groups:
        raise RuntimeError("Fit and calibration groups overlap")

    use_physics = args.architecture == "physics_resunet"
    feature_names = FEATURE_NAMES if use_physics else RAW_FEATURE_NAMES
    features = transform_sentinel2(dataset["images"], physics=use_physics)
    mean, std = normalization_stats(features, fit_indices)
    normalized = normalize_features(features, mean, std)
    targets = np.concatenate([dataset["labels"], dataset["valid"]], axis=-1)
    x_fit, y_fit = augment_quarter_turns(normalized[fit_indices], targets[fit_indices])

    keras.backend.clear_session()
    keras.utils.set_random_seed(args.seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except (AttributeError, RuntimeError):
        pass
    model = build_compact_resunet(
        input_channels=len(feature_names),
        base_filters=args.base_filters,
        image_size=args.image_size,
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.learning_rate, clipnorm=1.0),
        loss=MaskedBCEDice(args.positive_weight),
        metrics=[masked_dice, masked_iou],
    )
    history = model.fit(
        x_fit,
        y_fit,
        validation_data=(normalized[calibration_indices], targets[calibration_indices]),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=args.patience,
                min_delta=1e-4,
                restore_best_weights=True,
            )
        ],
        verbose=2,
    )
    calibration_probabilities = model.predict(
        normalized[calibration_indices],
        batch_size=args.batch_size,
        verbose=0,
    )
    decision_threshold = calibrate_threshold(
        dataset["labels"][calibration_indices],
        dataset["valid"][calibration_indices],
        calibration_probabilities,
    )
    calibration_metrics = evaluate_predictions(
        dataset["labels"][calibration_indices],
        dataset["valid"][calibration_indices],
        calibration_probabilities,
        dataset["names"][calibration_indices],
        pixels_per_scene=args.sampled_pixels,
        seed=args.seed,
        decision_threshold=decision_threshold,
    )

    artifact_dir = (root / args.artifact_dir).resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    pending_model_path = artifact_dir / "model.pending.keras"
    model_path = artifact_dir / "model.keras"
    model.save(pending_model_path, overwrite=True)
    pending_model_path.replace(model_path)

    valid_pixels = float(dataset["valid"].sum())
    positive_pixels = float((dataset["labels"] * dataset["valid"]).sum())
    selected_files = sorted(str(item["file"]).replace("\\", "/") for item in dataset["metadata"])
    config: dict[str, Any] = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "research_only",
        "task": "plume_segmentation",
        "architecture": args.architecture,
        "model_name": MODEL_NAME,
        "model_file": model_path.name,
        "model_sha256": sha256(model_path),
        "model_parameters": int(model.count_params()),
        "band_order": list(BAND_ORDER),
        "inference_tile_size": args.image_size,
        "input_product": "COPERNICUS/S2_HARMONIZED",
        "product_level": "L1C_TOA",
        "input_resolution_m": 20.0,
        "input_radiometric_scale": 1.0,
        "input_radiometric_offset": 0.0,
        "input_resampling": "bilinear",
        "input_nodata_value": 0,
        "physics_features": use_physics,
        "feature_names": list(feature_names),
        "normalization": {
            "method": "per_channel_zscore",
            "fit_partition": "group_disjoint_fit_only",
            "mean": mean.tolist(),
            "std": std.tolist(),
        },
        "decision_threshold": decision_threshold,
        "target_threshold_ppm_m": args.threshold,
        "target_definition": f"valid EMIT_CH4 > {args.threshold:g} ppm*m",
        "target_nodata": -9999.0,
        "training": {
            "seed": args.seed,
            "training_image_size": args.image_size,
            "base_filters": args.base_filters,
            "learning_rate": args.learning_rate,
            "positive_weight": args.positive_weight,
            "batch_size": args.batch_size,
            "epochs_requested": args.epochs,
            "epochs_ran": len(history.history["loss"]),
            "best_validation_loss": float(np.min(history.history["val_loss"])),
            "augmentation": "quarter_turn_rotations",
            "loss": "validity_masked_bce_plus_soft_dice",
        },
        "cohort": {
            "dataset_dir": args.dataset_dir.replace("\\", "/"),
            "selection": {
                "min_valid_pct": args.min_valid_pct,
                "max_abs_gap_days": args.max_abs_gap_days,
            },
            "selected_scenes": int(all_indices.size),
            "selected_groups": int(np.unique(dataset["groups"]).size),
            "fit_scenes": int(fit_indices.size),
            "fit_groups": sorted(str(value) for value in fit_groups),
            "calibration_scenes": int(calibration_indices.size),
            "calibration_groups": sorted(str(value) for value in calibration_groups),
            "valid_pixels": int(valid_pixels),
            "positive_fraction_of_valid": positive_pixels / max(valid_pixels, 1.0),
            "exclusions": dataset["exclusions"],
            "files": selected_files,
            "sha256": cohort_sha256(root, dataset["metadata"]),
        },
        "calibration_metrics": json_safe(calibration_metrics),
        "provenance": {
            "git_commit": source_commit,
            "git_tracked_worktree_dirty_at_start": source_dirty,
            "python": sys.version.split()[0],
            "tensorflow": tf.__version__,
            "keras": keras.__version__,
            "numpy": np.__version__,
            "trainer": "tools/train_compact_model.py",
        },
        "limitations": [
            "Research-only artifact; it is not validated for operational methane alerts.",
            "The legacy cohort is small, positive-centered, and contains no verified plume-free negative scenes.",
            "Targets have substantial nodata coverage and historical nodata-metadata inconsistencies.",
            "Sentinel-2 and EMIT acquisitions may differ by up to seven days in this selected cohort.",
            "The artifact expects legacy Sentinel-2 L1C/TOA inputs and must not silently consume L2A/SR imagery.",
            "Output is a plume probability mask, not methane concentration in physical units.",
        ],
    }
    pending_config_path = artifact_dir / "config.pending.json"
    config_path = artifact_dir / "config.json"
    pending_config_path.write_text(
        json.dumps(json_safe(config), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    pending_config_path.replace(config_path)

    loaded_model, loaded_config = load_artifact(artifact_dir)
    verification = predict_tile(
        loaded_model,
        loaded_config,
        dataset["images"][calibration_indices[0]],
    )
    if verification.shape != (args.image_size, args.image_size) or not np.all(np.isfinite(verification)):
        raise RuntimeError(f"Saved artifact verification failed with output shape {verification.shape}")

    return {
        "ok": True,
        "architecture": args.architecture,
        "artifact_dir": str(artifact_dir.relative_to(root)).replace("\\", "/"),
        "model_sha256": config["model_sha256"],
        "cohort_sha256": config["cohort"]["sha256"],
        "selected_scenes": int(all_indices.size),
        "fit_scenes": int(fit_indices.size),
        "calibration_scenes": int(calibration_indices.size),
        "fit_groups": len(fit_groups),
        "calibration_groups": len(calibration_groups),
        "epochs_ran": len(history.history["loss"]),
        "decision_threshold": decision_threshold,
        "calibration_sampled_scene_macro": calibration_metrics["sampled_scene_macro"],
        "parameters": int(model.count_params()),
        "verification_shape": list(verification.shape),
        "runtime_seconds": round(time.perf_counter() - started, 3),
    }


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(parser, args)
    try:
        print(json.dumps(json_safe(train(args)), indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc), "error_type": type(exc).__name__}, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
