#!/usr/bin/env python3
"""Run validity-aware MARS-S2L pilot baselines without touching the locked corpus.

The 18-sample pilot is a contract smoke test, not an accuracy benchmark. Models
fit only the six train samples, thresholds and minimum component sizes are
selected only on the six validation samples, and the six test-only-location
samples are evaluated once with the frozen operating rule.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import rasterio
import sklearn
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = REPOSITORY_ROOT / "EarthRemoteSensingRapidResponse"
if str(MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_ROOT))

from mars_s2l_adapter import (  # noqa: E402
    ENHANCEMENT_UNIT_STATUS,
    MarsS2Sample,
    iter_manifest,
    load_samples,
)

from acquire_mars_metadata import DEFAULT_OUTPUT, REVISION, checked_output_dir, repo_root, sha256  # noqa: E402
from acquire_mars_pilot import PILOT_MANIFEST, verify_manifest  # noqa: E402

DEFAULT_JSON = Path("reports/experiments/mars_pilot_baselines.json")
DEFAULT_MARKDOWN = Path("reports/experiments/MARS_PILOT_BASELINES.md")
SEED = 42
MAX_NEGATIVE_PIXELS_PER_TRAIN_SCENE = 5_000
MIN_COMPONENT_CANDIDATES = (1, 25, 50, 100, 200)
UPSTREAM_IMPLEMENTATION_COMMIT = "f7d264c2c845dfba1cb27f76ef6026275f8d8758"


def git_commit(root: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


def tracked_dirty(root: Path) -> bool:
    output = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=root, text=True
    )
    generated = {
        DEFAULT_JSON.as_posix(),
        DEFAULT_MARKDOWN.as_posix(),
    }
    changed = {
        line[3:].strip().replace("\\", "/")
        for line in output.splitlines()
        if len(line) >= 4
    }
    return bool(changed - generated)


def safe_output(root: Path, value: str) -> Path:
    result = (root / value).resolve()
    if root not in result.parents:
        raise ValueError("Experiment output must resolve beneath the repository root")
    return result


def split_samples(samples: list[MarsS2Sample]) -> dict[str, list[MarsS2Sample]]:
    result = {split: [sample for sample in samples if sample.split == split] for split in ("train", "val", "test")}
    if any(len(result[split]) != 6 for split in result):
        raise ValueError(f"Expected six pilot samples per split, got { {key: len(value) for key, value in result.items()} }")
    if any(sum(sample.presence for sample in result[split]) != 3 for split in result):
        raise ValueError("Expected three plume and three no-plume pilot samples per split")
    return result


def pixel_features(sample: MarsS2Sample) -> np.ndarray:
    epsilon = np.float32(1e-4)
    normalized_change = (sample.target - sample.reference) / (
        sample.target + sample.reference + epsilon
    )
    log_change = np.log1p(sample.target) - np.log1p(sample.reference)
    mbmp = (1.0 - sample.mbmp_valid_aware)[None, ...]
    return np.concatenate([mbmp, normalized_change, log_change], axis=0).astype(np.float32)


def fit_pixel_logistic(train: list[MarsS2Sample]) -> Any:
    rng = np.random.default_rng(SEED)
    feature_rows: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for sample in train:
        features = pixel_features(sample)
        observable = sample.observable_mask
        plume = sample.plume_mask & observable
        positive_indices = np.flatnonzero(plume.ravel())
        negative_indices = np.flatnonzero((observable & ~sample.plume_mask).ravel())
        if negative_indices.size > MAX_NEGATIVE_PIXELS_PER_TRAIN_SCENE:
            negative_indices = rng.choice(
                negative_indices, MAX_NEGATIVE_PIXELS_PER_TRAIN_SCENE, replace=False
            )
        selected = np.concatenate([positive_indices, negative_indices])
        flat = np.moveaxis(features, 0, -1).reshape(-1, features.shape[0])
        feature_rows.append(flat[selected])
        labels.append(
            np.concatenate(
                [
                    np.ones(positive_indices.size, dtype=np.uint8),
                    np.zeros(negative_indices.size, dtype=np.uint8),
                ]
            )
        )
    x_train = np.concatenate(feature_rows)
    y_train = np.concatenate(labels)
    if len(np.unique(y_train)) != 2:
        raise ValueError("Pixel training set must contain both classes")
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=2_000,
            random_state=SEED,
            solver="lbfgs",
        ),
    )
    model.fit(x_train, y_train)
    return model


def score_logistic(model: Any, sample: MarsS2Sample) -> np.ndarray:
    features = pixel_features(sample)
    flat = np.moveaxis(features, 0, -1).reshape(-1, features.shape[0])
    scores = model.predict_proba(flat)[:, 1].reshape(sample.plume_mask.shape)
    scores[~sample.observable_mask] = 0.0
    return scores.astype(np.float32)


def score_mbmp_valid(sample: MarsS2Sample) -> np.ndarray:
    result = (1.0 - sample.mbmp_valid_aware).astype(np.float32)
    result[~sample.observable_mask] = 0.0
    return result


def score_mbmp_release(sample: MarsS2Sample) -> np.ndarray:
    result = (1.0 - sample.mbmp_release_compatible).astype(np.float32)
    result[~sample.observable_mask] = 0.0
    return result


def remove_small_components(mask: np.ndarray, minimum_pixels: int) -> np.ndarray:
    labels, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    if count == 0:
        return np.zeros(mask.shape, dtype=bool)
    sizes = np.bincount(labels.ravel())
    keep = sizes >= minimum_pixels
    keep[0] = False
    return keep[labels]


def safe_ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def scene_metrics(truth: list[int], predicted: list[int], scores: list[float]) -> dict[str, Any]:
    y_true = np.asarray(truth, dtype=np.uint8)
    y_pred = np.asarray(predicted, dtype=np.uint8)
    y_score = np.asarray(scores, dtype=np.float64)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return {
        "n": int(y_true.size),
        "positive": int(y_true.sum()),
        "negative": int((1 - y_true).sum()),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": safe_ratio(tp, tp + fp),
        "recall": safe_ratio(tp, tp + fn),
        "specificity": safe_ratio(tn, tn + fp),
        "false_positive_rate": safe_ratio(fp, fp + tn),
        "accuracy": safe_ratio(tp + tn, len(y_true)),
        "balanced_accuracy": 0.5
        * ((safe_ratio(tp, tp + fn) or 0.0) + (safe_ratio(tn, tn + fp) or 0.0)),
        "average_precision": float(average_precision_score(y_true, y_score)),
        "auroc": float(roc_auc_score(y_true, y_score)),
        "brier": float(brier_score_loss(y_true, np.clip(y_score, 0.0, 1.0))),
    }


def evaluate_rule(
    samples: list[MarsS2Sample],
    score_maps: dict[str, np.ndarray],
    threshold: float,
    minimum_pixels: int,
) -> dict[str, Any]:
    truth: list[int] = []
    predicted: list[int] = []
    scene_scores: list[float] = []
    pixel_truth: list[np.ndarray] = []
    pixel_scores: list[np.ndarray] = []
    intersection = predicted_area = truth_area = 0
    per_sample: list[dict[str, Any]] = []
    for sample in samples:
        scores = score_maps[sample.sample_id]
        candidate = (scores >= threshold) & sample.observable_mask
        prediction = remove_small_components(candidate, minimum_pixels)
        observed_scores = scores[sample.observable_mask]
        scene_score = float(np.quantile(observed_scores, 0.99)) if observed_scores.size else 0.0
        scene_prediction = int(np.any(prediction))
        truth.append(sample.presence)
        predicted.append(scene_prediction)
        scene_scores.append(scene_score)
        local_truth = sample.plume_mask[sample.observable_mask]
        local_prediction = prediction[sample.observable_mask]
        pixel_truth.append(local_truth.astype(np.uint8))
        pixel_scores.append(observed_scores.astype(np.float32))
        intersection += int(np.count_nonzero(local_truth & local_prediction))
        predicted_area += int(np.count_nonzero(local_prediction))
        truth_area += int(np.count_nonzero(local_truth))
        per_sample.append(
            {
                "sample_id": sample.sample_id,
                "label": sample.presence,
                "prediction": scene_prediction,
                "scene_score_q99": scene_score,
                "predicted_pixels": int(np.count_nonzero(local_prediction)),
                "truth_pixels": int(np.count_nonzero(local_truth)),
            }
        )
    all_pixel_truth = np.concatenate(pixel_truth)
    all_pixel_scores = np.concatenate(pixel_scores)
    union = predicted_area + truth_area - intersection
    dice_denominator = predicted_area + truth_area
    metrics = scene_metrics(truth, predicted, scene_scores)
    metrics["pixel"] = {
        "observable_pixels": int(all_pixel_truth.size),
        "positive_pixels": int(all_pixel_truth.sum()),
        "average_precision": float(average_precision_score(all_pixel_truth, all_pixel_scores)),
        "intersection_over_union": safe_ratio(intersection, union),
        "dice": safe_ratio(2 * intersection, dice_denominator),
        "predicted_positive_pixels": predicted_area,
        "truth_positive_pixels": truth_area,
    }
    metrics["samples"] = per_sample
    return metrics


def evaluate_scene_rule(
    samples: list[MarsS2Sample],
    score_maps: dict[str, np.ndarray],
    threshold: float,
    minimum_pixels: int,
) -> dict[str, Any]:
    """Evaluate scene decisions cheaply while searching validation rules."""
    truth: list[int] = []
    predicted: list[int] = []
    scene_scores: list[float] = []
    for sample in samples:
        scores = score_maps[sample.sample_id]
        candidate = (scores >= threshold) & sample.observable_mask
        prediction = remove_small_components(candidate, minimum_pixels)
        observed_scores = scores[sample.observable_mask]
        truth.append(sample.presence)
        predicted.append(int(np.any(prediction)))
        scene_scores.append(
            float(np.quantile(observed_scores, 0.99)) if observed_scores.size else 0.0
        )
    return scene_metrics(truth, predicted, scene_scores)


def threshold_candidates(score_maps: dict[str, np.ndarray], samples: list[MarsS2Sample]) -> list[float]:
    observed = np.concatenate(
        [score_maps[sample.sample_id][sample.observable_mask] for sample in samples]
    )
    quantiles = np.linspace(0.80, 0.9999, 80)
    candidates = set(float(value) for value in np.quantile(observed, quantiles))
    candidates.update(float(value) for value in np.linspace(0.0, 0.25, 51))
    return sorted(candidates)


def select_operating_rule(
    validation: list[MarsS2Sample], score_maps: dict[str, np.ndarray]
) -> tuple[dict[str, Any], dict[str, Any]]:
    candidates: list[tuple[tuple[float, ...], float, int, dict[str, Any]]] = []
    for threshold in threshold_candidates(score_maps, validation):
        for minimum_pixels in MIN_COMPONENT_CANDIDATES:
            metrics = evaluate_scene_rule(
                validation, score_maps, threshold=threshold, minimum_pixels=minimum_pixels
            )
            # On three negative scenes, the only meaningful <=5% FPR setting is zero FPs.
            fpr = float(metrics["false_positive_rate"] or 0.0)
            recall = float(metrics["recall"] or 0.0)
            balanced = float(metrics["balanced_accuracy"] or 0.0)
            constrained = 1.0 if fpr == 0.0 else 0.0
            # Prefer the most sensitive zero-FP rule, then the smaller component
            # requirement and lower pixel threshold to avoid arbitrary conservatism.
            rank = (
                constrained,
                recall if constrained else balanced,
                -fpr,
                -minimum_pixels,
                -threshold,
            )
            candidates.append((rank, threshold, minimum_pixels, metrics))
    _, threshold, minimum_pixels, metrics = max(candidates, key=lambda item: item[0])
    rule = {
        "pixel_score_threshold": threshold,
        "minimum_connected_pixels": minimum_pixels,
        "selection_split": "val",
        "selection_objective": "maximize recall at zero observed validation false positives; then prefer the less restrictive connected-area rule",
        "validation_negative_resolution": "one false positive equals 0.333 FPR; pilot cannot estimate a 0.05 target",
    }
    detailed_metrics = evaluate_rule(
        validation,
        score_maps,
        threshold=threshold,
        minimum_pixels=minimum_pixels,
    )
    return rule, detailed_metrics


def run_model(
    name: str,
    scorer: Callable[[MarsS2Sample], np.ndarray],
    splits: dict[str, list[MarsS2Sample]],
) -> dict[str, Any]:
    score_maps = {
        sample.sample_id: scorer(sample)
        for split in ("val", "test")
        for sample in splits[split]
    }
    rule, validation = select_operating_rule(splits["val"], score_maps)
    test = evaluate_rule(
        splits["test"],
        score_maps,
        threshold=float(rule["pixel_score_threshold"]),
        minimum_pixels=int(rule["minimum_connected_pixels"]),
    )
    return {"name": name, "operating_rule": rule, "validation": validation, "test": test}


def trivial_scene_baselines(samples: list[MarsS2Sample]) -> dict[str, Any]:
    truth = [sample.presence for sample in samples]
    return {
        "all_no_plume": scene_metrics(truth, [0] * len(samples), [0.0] * len(samples)),
        "all_plume": scene_metrics(truth, [1] * len(samples), [1.0] * len(samples)),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# MARS-S2L contract-pilot baselines",
        "",
        "These results are an end-to-end adapter and evaluation smoke test on 18 deliberately balanced samples. They are not an accuracy estimate and are not suitable for a paper claim.",
        "",
        f"- Source revision: `{REVISION}`",
        f"- Pilot identity: `{report['input']['pilot_identity_sha256']}`",
        "- Split: 6 train / 6 validation / 6 test; each has 3 plume and 3 no-plume scenes.",
        "- Test scenes: deterministic samples from locations absent from train and validation.",
        "- Thresholds/component sizes: validation only; test never used for selection.",
        "",
        "| Model | Val recall | Val FPR | Test recall | Test specificity | Test pixel AP | Test IoU |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model in report["models"]:
        validation = model["validation"]
        test = model["test"]
        lines.append(
            f"| {model['name']} | {fmt(validation['recall'])} | "
            f"{fmt(validation['false_positive_rate'])} | {fmt(test['recall'])} | "
            f"{fmt(test['specificity'])} | {fmt(test['pixel']['average_precision'])} | "
            f"{fmt(test['pixel']['intersection_over_union'])} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Passing this test means the native target/reference adapter, zero-mask negatives, explicit observability mask, validation-only operating rule, and test path execute coherently.",
            "- With only three test positives and three test negatives, one mistake changes recall or FPR by 0.333. Model rankings are therefore unstable and must not drive an architecture promotion.",
            "- The logistic model is a low-capacity pipeline check. The full cohort and five-seed, site-blocked protocol remain required before comparing architectures.",
            f"- Enhancement units are unresolved: {ENHANCEMENT_UNIT_STATUS}.",
            "",
            "The next experiment should reproduce the released MARS-S2L/CH4Net baselines on the frozen cohort, then compare the dual-temporal selective architecture under the same test and calibration contract.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-dir", default=DEFAULT_OUTPUT.as_posix())
    parser.add_argument("--output-json", default=DEFAULT_JSON.as_posix())
    parser.add_argument("--output-markdown", default=DEFAULT_MARKDOWN.as_posix())
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args()
    root = repo_root()
    try:
        metadata_dir = checked_output_dir(root, args.metadata_dir)
        pilot = verify_manifest(metadata_dir)
        manifest_path = metadata_dir / PILOT_MANIFEST
        output_json = safe_output(root, args.output_json)
        output_markdown = safe_output(root, args.output_markdown)
        samples = load_samples(metadata_dir, iter_manifest(manifest_path))
        splits = split_samples(samples)
        logistic = fit_pixel_logistic(splits["train"])
        models = [
            run_model("MBMP release-compatible", score_mbmp_release, splits),
            run_model("MBMP validity-aware", score_mbmp_valid, splits),
            run_model("Pixel logistic (13 features)", lambda sample: score_logistic(logistic, sample), splits),
        ]
        report = {
            "schema_version": 1,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "scope": "contract_smoke_test_not_accuracy_claim",
            "input": {
                "repository": "UNEP-IMEO/MARS-S2L",
                "revision": REVISION,
                "pilot_manifest_sha256": sha256(manifest_path),
                "pilot_identity_sha256": pilot["integrity"]["pilot_identity_sha256"],
                "sample_count": len(samples),
                "split_counts": {split: len(values) for split, values in splits.items()},
                "test_locations": "absent from train and validation by acquisition selection",
            },
            "adapter_contract": {
                "reflectance": "uint16 / 5000, clipped to [0,2], matching released loader",
                "cloud_classes": "0 clear; 1 thick cloud; 2 thin cloud; 3 cloud shadow",
                "observable": "all 12 radiometric bands nonzero and cloud class 0",
                "negative_target": "zero mask synthesized in memory",
                "enhancement_unit_status": ENHANCEMENT_UNIT_STATUS,
            },
            "upstream_reference": {
                "repository": "https://github.com/UNEP-IMEO-MARS/marss2l",
                "inspected_commit": UPSTREAM_IMPLEMENTATION_COMMIT,
                "mbmp_definition": "marss2l/mbmp_torch.py",
                "loader_definition": "marss2l/loaders.py",
            },
            "training": {
                "seed": SEED,
                "pixel_logistic_negative_cap_per_train_scene": MAX_NEGATIVE_PIXELS_PER_TRAIN_SCENE,
                "pixel_logistic_features": [
                    "1-minus-valid-aware-MBMP",
                    *[f"normalized_change_{index}" for index in range(6)],
                    *[f"log_change_{index}" for index in range(6)],
                ],
            },
            "trivial_test_baselines": trivial_scene_baselines(splits["test"]),
            "models": models,
            "limitations": [
                "Only six scenes per split and three examples per class.",
                "The pilot was selected to verify contracts, not represent the deployment distribution.",
                "No confidence interval or architecture promotion is valid from this sample.",
                "The enhancement raster unit conflict blocks quantitative regression claims.",
            ],
            "provenance": {
                "git_commit": git_commit(root),
                "git_tracked_worktree_dirty_at_start": tracked_dirty(root),
                "script": "tools/run_mars_pilot_baselines.py",
                "script_sha256": sha256(Path(__file__)),
                "adapter": "EarthRemoteSensingRapidResponse/mars_s2l_adapter.py",
                "adapter_sha256": sha256(MODEL_ROOT / "mars_s2l_adapter.py"),
                "python": sys.version.split()[0],
                "numpy": np.__version__,
                "rasterio": rasterio.__version__,
                "sklearn": sklearn.__version__,
            },
        }
        write_json(output_json, report)
        write_markdown(output_markdown, report)
    except (FileNotFoundError, OSError, RuntimeError, ValueError, rasterio.errors.RasterioError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, indent=None if args.compact else 2))
        return 2
    payload = {
        "ok": True,
        "scope": report["scope"],
        "sample_count": report["input"]["sample_count"],
        "models": [
            {
                "name": model["name"],
                "test_recall": model["test"]["recall"],
                "test_specificity": model["test"]["specificity"],
                "test_pixel_ap": model["test"]["pixel"]["average_precision"],
            }
            for model in models
        ],
        "output_json": output_json.relative_to(root).as_posix(),
        "output_markdown": output_markdown.relative_to(root).as_posix(),
    }
    print(json.dumps(payload, indent=None if args.compact else 2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
