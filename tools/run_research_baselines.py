#!/usr/bin/env python3
"""Run leakage-resistant spectral baselines for ERSRR research decisions.

Two cohorts are evaluated independently:

1. Quality-filtered legacy six-band tiles, grouped by Sentinel-2 MGRS tile.
2. New EMIT V002 CMR plume masks with bracketing Sentinel-2 L2A stacks,
   grouped by EMIT granule.

The script deliberately uses small classical models.  Its purpose is to test
whether explicit SWIR and temporal features carry signal before spending GPU
time on a larger segmentation network.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import rasterio
import sklearn
from rasterio.enums import Resampling
from rasterio.warp import transform_geom
from scipy.ndimage import uniform_filter
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from audit_dataset import parse_filename

LEGACY_THRESHOLDS = (300.0, 1000.0)
CALIBRATION_THRESHOLDS = tuple([0.01, 0.025, 0.05, 0.075, *np.linspace(0.1, 0.9, 33)])
EPSILON = 1.0


def leakage_safe_scene_groups(records: list[dict[str, Any]]) -> dict[str, str]:
    """Connect scenes sharing an MGRS tile, S2 acquisition, or EMIT granule."""
    parent = list(range(len(records)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for field in ("s2_tile", "s2_scene", "emit_id"):
        first_seen: dict[str, int] = {}
        for index, record in enumerate(records):
            value = str(record.get(field) or "")
            if not value:
                continue
            if value in first_seen:
                union(index, first_seen[value])
            else:
                first_seen[value] = index

    components: dict[int, list[str]] = {}
    for index, record in enumerate(records):
        components.setdefault(find(index), []).append(str(record["file"]))
    labels = {
        root: "scene-component-" + hashlib.sha256("\n".join(sorted(files)).encode("utf-8")).hexdigest()[:12]
        for root, files in components.items()
    }
    return {str(record["file"]): labels[find(index)] for index, record in enumerate(records)}


def geometry_bounds(geometry: dict[str, Any]) -> tuple[float, float, float, float]:
    points: list[tuple[float, float]] = []

    def collect(value: Any) -> None:
        if (
            isinstance(value, (list, tuple))
            and len(value) >= 2
            and isinstance(value[0], (int, float))
            and isinstance(value[1], (int, float))
        ):
            points.append((float(value[0]), float(value[1])))
            return
        if isinstance(value, (list, tuple)):
            for child in value:
                collect(child)

    collect(geometry.get("coordinates", []))
    if not points:
        raise ValueError("Plume geometry contains no coordinates")
    xs, ys = zip(*points)
    return min(xs), min(ys), max(xs), max(ys)


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


def file_set_sha256(root: Path, paths: list[Path], namespace: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(namespace + b"\0")
    for path in sorted({item.resolve() for item in paths}):
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8") + b"\0")
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def v002_input_paths(root: Path, batch_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for manifest_path in sorted(batch_dir.glob("*/manifest.json")):
        paths.append(manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        plume_path = manifest.get("plume_geojson")
        if isinstance(plume_path, str):
            paths.append(root / plume_path)
        for scene in manifest.get("scenes", []):
            for field in ("stack", "mask"):
                value = scene.get(field)
                if isinstance(value, str):
                    paths.append(root / value)
    return paths


def stable_rng(name: str, seed: int) -> np.random.Generator:
    digest = hashlib.sha256(name.encode("utf-8")).hexdigest()
    return np.random.default_rng(int(digest[:8], 16) ^ seed)


def sample_indices(valid: np.ndarray, count: int, *, name: str, seed: int) -> np.ndarray:
    indices = np.flatnonzero(valid.reshape(-1))
    if not indices.size:
        return indices
    if indices.size <= count:
        return indices
    return np.sort(stable_rng(name, seed).choice(indices, size=count, replace=False))


def raw_log_features(bands: np.ndarray) -> np.ndarray:
    return np.log1p(np.clip(bands.astype("float32", copy=False), 0.0, None))


def physics_only_features(bands: np.ndarray) -> np.ndarray:
    values = np.clip(bands.astype("float32", copy=False), 0.0, None)
    b2, b3, b4, b11, b12 = (values[:, index] for index in range(5))
    log_values = np.log1p(values)
    swir_log_ratio = log_values[:, 4] - log_values[:, 3]
    swir_normalized_difference = (b12 - b11) / (b12 + b11 + EPSILON)
    b11_red_log_ratio = log_values[:, 3] - log_values[:, 2]
    b12_red_log_ratio = log_values[:, 4] - log_values[:, 2]
    swir_visible_log_ratio = np.log1p(b11 + b12) - np.log1p(b2 + b3 + b4)
    visible_normalized_difference = (b4 - b2) / (b4 + b2 + EPSILON)
    return np.column_stack(
        [
            swir_log_ratio,
            swir_normalized_difference,
            b11_red_log_ratio,
            b12_red_log_ratio,
            swir_visible_log_ratio,
            visible_normalized_difference,
        ]
    ).astype("float32")


def physics_features(bands: np.ndarray) -> np.ndarray:
    return np.column_stack([raw_log_features(bands), physics_only_features(bands)]).astype("float32")


def bitemporal_features(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    before_log = raw_log_features(before)
    after_log = raw_log_features(after)
    before_physics = physics_only_features(before)
    after_physics = physics_only_features(after)
    return np.column_stack(
        [
            before_log,
            after_log,
            after_log - before_log,
            before_physics,
            after_physics,
            after_physics - before_physics,
        ]
    ).astype("float32")


def spatial_context_image(bands: np.ndarray) -> np.ndarray:
    """Return local means and local contrast without leaking coordinates."""
    height, width, channels = bands.shape
    log_bands = raw_log_features(bands.reshape(-1, channels)).reshape(height, width, channels)
    mean_3 = uniform_filter(log_bands, size=(3, 3, 1), mode="reflect")
    mean_9 = uniform_filter(log_bands, size=(9, 9, 1), mode="reflect")
    return np.concatenate([mean_3, mean_9, log_bands - mean_9], axis=-1).astype("float32")


def load_legacy_samples(
    dataset_dir: Path,
    *,
    image_size: int,
    pixels_per_scene: int,
    seed: int,
    min_valid_pct: float,
    max_abs_gap_days: float,
) -> dict[str, Any]:
    root = repo_root()
    raw_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    context_parts: list[np.ndarray] = []
    group_parts: list[np.ndarray] = []
    scene_parts: list[np.ndarray] = []
    selected: list[dict[str, Any]] = []
    exclusions: Counter[str] = Counter()

    for path in sorted(dataset_dir.glob("*.tif")):
        info = parse_filename(path)
        if not info.get("emit_id") or not info.get("s2_tile"):
            exclusions["missing_provenance"] += 1
            continue
        gap = info.get("days_between_s2_emit")
        if gap is None or abs(float(gap)) > max_abs_gap_days:
            exclusions["temporal_gap"] += 1
            continue
        with rasterio.open(path) as source:
            if source.count != 6 or source.width != 256 or source.height != 256:
                exclusions["shape_or_band_contract"] += 1
                continue
            band_data = source.read(
                indexes=(1, 2, 3, 4, 5),
                out_shape=(5, image_size, image_size),
                resampling=Resampling.bilinear,
            ).astype("float32")
            target = source.read(
                6,
                out_shape=(image_size, image_size),
                resampling=Resampling.nearest,
            ).astype("float32")
        bands = np.moveaxis(band_data, 0, -1)
        valid = np.isfinite(target) & (target != -9999.0) & np.all(np.isfinite(bands), axis=-1)
        valid_pct = float(100.0 * valid.mean())
        if valid_pct < min_valid_pct:
            exclusions["valid_coverage"] += 1
            continue
        indices = sample_indices(valid, pixels_per_scene, name=path.name, seed=seed)
        if not indices.size:
            exclusions["no_valid_pixels"] += 1
            continue
        flat_bands = bands.reshape(-1, 5)[indices]
        flat_target = target.reshape(-1)[indices]
        flat_context = spatial_context_image(bands).reshape(-1, 15)[indices]
        raw_parts.append(flat_bands)
        target_parts.append(flat_target)
        context_parts.append(flat_context)
        scene_parts.append(np.full(indices.size, path.name, dtype=object))
        selected.append(
            {
                "file": str(path.relative_to(root)).replace("\\", "/"),
                "s2_tile": info["s2_tile"],
                "s2_scene": f"{info['s2_start']}_{info['s2_end']}",
                "emit_id": info["emit_id"],
                "gap_days": float(gap),
                "valid_pct": valid_pct,
                "sampled_pixels": int(indices.size),
            }
        )

    if not raw_parts:
        raise RuntimeError(f"No qualifying legacy scenes under {dataset_dir}")
    group_by_file = leakage_safe_scene_groups(selected)
    group_parts = []
    for item in selected:
        item["leakage_group"] = group_by_file[item["file"]]
        group_parts.append(np.full(item["sampled_pixels"], item["leakage_group"], dtype=object))
    return {
        "bands": np.concatenate(raw_parts),
        "target": np.concatenate(target_parts),
        "context": np.concatenate(context_parts),
        "groups": np.concatenate(group_parts),
        "scenes": np.concatenate(scene_parts),
        "selected": selected,
        "exclusions": dict(exclusions),
    }


def load_v002_samples(
    batch_dir: Path,
    *,
    pixels_per_scene: int,
    seed: int,
    min_roi_clear_pct: float,
    min_mask_positive_pct: float,
    max_mask_positive_pct: float,
) -> dict[str, Any]:
    root = repo_root()
    nearest_parts: list[np.ndarray] = []
    before_parts: list[np.ndarray] = []
    after_parts: list[np.ndarray] = []
    nearest_context_parts: list[np.ndarray] = []
    before_context_parts: list[np.ndarray] = []
    after_context_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    group_parts: list[np.ndarray] = []
    scene_parts: list[np.ndarray] = []
    selected: list[dict[str, Any]] = []
    exclusions: Counter[str] = Counter()

    for manifest_path in sorted(batch_dir.glob("*/manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        scenes = sorted(manifest.get("scenes", []), key=lambda item: float(item["offset_hours"]))
        before_candidates = [item for item in scenes if float(item["offset_hours"]) <= 0]
        after_candidates = [item for item in scenes if float(item["offset_hours"]) >= 0]
        if not before_candidates or not after_candidates:
            exclusions["not_bracketed"] += 1
            continue
        before_info = min(before_candidates, key=lambda item: abs(float(item["offset_hours"])))
        after_info = min(after_candidates, key=lambda item: abs(float(item["offset_hours"])))
        if min(float(before_info["roi_clear_pct"]), float(after_info["roi_clear_pct"])) < min_roi_clear_pct:
            exclusions["roi_cloud"] += 1
            continue

        before_path = root / before_info["stack"]
        after_path = root / after_info["stack"]
        mask_path = root / before_info["mask"]
        with rasterio.open(before_path) as source:
            before = np.moveaxis(source.read().astype("float32"), 0, -1)
            before_contract = (source.crs, source.transform, source.shape)
        with rasterio.open(after_path) as source:
            after = np.moveaxis(source.read().astype("float32"), 0, -1)
            after_contract = (source.crs, source.transform, source.shape)
        with rasterio.open(mask_path) as source:
            mask = source.read(1) > 0
            mask_contract = (source.crs, source.transform, source.shape)
            mask_bounds = source.bounds
            mask_crs = source.crs
        if before_contract != after_contract or before_contract != mask_contract:
            exclusions["grid_contract"] += 1
            continue
        if before.shape[-1] != 5 or after.shape[-1] != 5:
            exclusions["band_contract"] += 1
            continue

        try:
            plume_path = root / manifest["plume_geojson"]
            plume_feature = json.loads(plume_path.read_text(encoding="utf-8"))
            projected_plume = transform_geom("EPSG:4326", mask_crs, plume_feature["geometry"], precision=3)
            plume_bounds = geometry_bounds(projected_plume)
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
            exclusions["geometry_contract"] += 1
            continue
        plume_fully_contained = (
            mask_bounds.left <= plume_bounds[0]
            and mask_bounds.bottom <= plume_bounds[1]
            and mask_bounds.right >= plume_bounds[2]
            and mask_bounds.top >= plume_bounds[3]
        )
        if not plume_fully_contained:
            exclusions["plume_clipped"] += 1
            continue

        mask_positive_pct = float(100.0 * mask.mean())
        if not (min_mask_positive_pct <= mask_positive_pct <= max_mask_positive_pct):
            exclusions["mask_coverage"] += 1
            continue

        valid = np.all(before > 0, axis=-1) & np.all(after > 0, axis=-1)
        indices = sample_indices(valid, pixels_per_scene, name=manifest["granule_id"], seed=seed)
        if not indices.size:
            exclusions["no_valid_pixels"] += 1
            continue
        before_flat = before.reshape(-1, 5)[indices]
        after_flat = after.reshape(-1, 5)[indices]
        before_context = spatial_context_image(before).reshape(-1, 15)[indices]
        after_context = spatial_context_image(after).reshape(-1, 15)[indices]
        labels = mask.reshape(-1)[indices]
        nearest_flat = (
            before_flat
            if abs(float(before_info["offset_hours"])) <= abs(float(after_info["offset_hours"]))
            else after_flat
        )
        nearest_context = (
            before_context
            if abs(float(before_info["offset_hours"])) <= abs(float(after_info["offset_hours"]))
            else after_context
        )
        granule_id = manifest["granule_id"]
        nearest_parts.append(nearest_flat)
        before_parts.append(before_flat)
        after_parts.append(after_flat)
        nearest_context_parts.append(nearest_context)
        before_context_parts.append(before_context)
        after_context_parts.append(after_context)
        label_parts.append(labels.astype("uint8"))
        group_parts.append(np.full(indices.size, granule_id, dtype=object))
        scene_parts.append(np.full(indices.size, granule_id, dtype=object))
        selected.append(
            {
                "granule_id": granule_id,
                "before_scene": before_info["scene_id"],
                "before_offset_hours": float(before_info["offset_hours"]),
                "before_roi_clear_pct": float(before_info["roi_clear_pct"]),
                "after_scene": after_info["scene_id"],
                "after_offset_hours": float(after_info["offset_hours"]),
                "after_roi_clear_pct": float(after_info["roi_clear_pct"]),
                "mask_positive_pct": mask_positive_pct,
                "plume_fully_contained": True,
                "sampled_pixels": int(indices.size),
            }
        )

    if not nearest_parts:
        raise RuntimeError(f"No qualifying V002 pilot pairs under {batch_dir}")
    return {
        "nearest": np.concatenate(nearest_parts),
        "before": np.concatenate(before_parts),
        "after": np.concatenate(after_parts),
        "nearest_context": np.concatenate(nearest_context_parts),
        "before_context": np.concatenate(before_context_parts),
        "after_context": np.concatenate(after_context_parts),
        "labels": np.concatenate(label_parts),
        "groups": np.concatenate(group_parts),
        "scenes": np.concatenate(scene_parts),
        "selected": selected,
        "exclusions": dict(exclusions),
    }


def estimator(kind: str, seed: int) -> Any:
    if kind == "dummy":
        return DummyClassifier(strategy="prior")
    if kind == "logistic":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=seed,
                solver="lbfgs",
            ),
        )
    if kind == "histgb":
        return HistGradientBoostingClassifier(
            class_weight="balanced",
            learning_rate=0.07,
            l2_regularization=1.0,
            max_iter=120,
            max_leaf_nodes=15,
            min_samples_leaf=40,
            random_state=seed,
        )
    raise ValueError(f"Unknown estimator kind: {kind}")


def roc_auc(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    positives = int(y_true.sum())
    negatives = int(y_true.size - positives)
    if positives == 0 or negatives == 0:
        return float("nan")
    return float(roc_auc_score(y_true, probabilities))


def average_precision(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    positives = int(y_true.sum())
    if positives == 0:
        return float("nan")
    return float(average_precision_score(y_true, probabilities))


def classification_metrics(y_true: np.ndarray, probabilities: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    truth = y_true.astype(bool)
    predicted = probabilities >= threshold
    tp = int(np.count_nonzero(truth & predicted))
    fp = int(np.count_nonzero(~truth & predicted))
    fn = int(np.count_nonzero(truth & ~predicted))
    tn = int(np.count_nonzero(~truth & ~predicted))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 1.0
    return {
        "auprc": average_precision(truth.astype("uint8"), probabilities),
        "roc_auc": roc_auc(truth.astype("uint8"), probabilities),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "iou": float(iou),
        "specificity": float(tn / (tn + fp)) if tn + fp else float("nan"),
    }


def mean_metric(rows: list[dict[str, float]], key: str) -> float:
    values = np.asarray([row[key] for row in rows], dtype="float64")
    return float(np.nanmean(values)) if np.any(np.isfinite(values)) else float("nan")


def inner_calibration_split(train_index: np.ndarray, groups: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    unique = np.unique(groups[train_index])
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    calibration_count = max(1, int(round(unique.size * 0.2)))
    calibration_groups = set(unique[:calibration_count])
    is_calibration = np.asarray([group in calibration_groups for group in groups[train_index]])
    return train_index[~is_calibration], train_index[is_calibration]


def calibrate_threshold(labels: np.ndarray, probabilities: np.ndarray, scenes: np.ndarray) -> float:
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in CALIBRATION_THRESHOLDS:
        rows = []
        for scene in np.unique(scenes):
            mask = scenes == scene
            rows.append(classification_metrics(labels[mask], probabilities[mask], threshold=float(threshold)))
        macro_f1 = float(np.nanmean([row["f1"] for row in rows]))
        if macro_f1 > best_f1 + 1e-12 or (
            abs(macro_f1 - best_f1) <= 1e-12 and abs(float(threshold) - 0.5) < abs(best_threshold - 0.5)
        ):
            best_f1 = macro_f1
            best_threshold = float(threshold)
    return best_threshold


def evaluate_model(
    name: str,
    features: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    scenes: np.ndarray,
    *,
    model_kind: str,
    folds: int,
    seed: int,
) -> dict[str, Any]:
    unique_groups = np.unique(groups)
    if unique_groups.size < folds:
        raise ValueError(f"{name}: {unique_groups.size} groups cannot support {folds} folds")
    splitter = GroupKFold(n_splits=folds)
    fold_rows = []
    started = time.perf_counter()
    for fold_index, (train_index, test_index) in enumerate(splitter.split(features, labels, groups), start=1):
        fit_index, calibration_index = inner_calibration_split(train_index, groups, seed + fold_index)
        model = estimator(model_kind, seed + fold_index)
        model.fit(features[fit_index], labels[fit_index])
        calibration_probabilities = model.predict_proba(features[calibration_index])[:, 1]
        decision_threshold = calibrate_threshold(
            labels[calibration_index],
            calibration_probabilities,
            scenes[calibration_index],
        )
        probabilities = model.predict_proba(features[test_index])[:, 1]
        pixel = classification_metrics(labels[test_index], probabilities, threshold=decision_threshold)
        scene_rows = []
        test_scenes = scenes[test_index]
        for scene in np.unique(test_scenes):
            mask = test_scenes == scene
            scene_rows.append(
                classification_metrics(labels[test_index][mask], probabilities[mask], threshold=decision_threshold)
            )
        fold_rows.append(
            {
                "fold": fold_index,
                "fit_samples": int(fit_index.size),
                "calibration_samples": int(calibration_index.size),
                "test_samples": int(test_index.size),
                "fit_groups": int(np.unique(groups[fit_index]).size),
                "calibration_groups": int(np.unique(groups[calibration_index]).size),
                "test_groups": int(np.unique(groups[test_index]).size),
                "decision_threshold": decision_threshold,
                "pixel": pixel,
                "scene_macro": {key: mean_metric(scene_rows, key) for key in pixel},
            }
        )

    summary: dict[str, dict[str, float]] = {}
    for scope in ("pixel", "scene_macro"):
        summary[scope] = {}
        for metric in fold_rows[0][scope]:
            values = np.asarray([row[scope][metric] for row in fold_rows], dtype="float64")
            summary[scope][f"{metric}_mean"] = float(np.nanmean(values))
            summary[scope][f"{metric}_std"] = float(np.nanstd(values))
    return {
        "name": name,
        "model_kind": model_kind,
        "feature_count": int(features.shape[1]),
        "samples": int(features.shape[0]),
        "positive_fraction": float(labels.mean()),
        "groups": int(unique_groups.size),
        "threshold": "inner-group-calibrated",
        "runtime_seconds": round(time.perf_counter() - started, 3),
        "summary": summary,
        "folds": fold_rows,
    }


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_markdown(path: Path, results: dict[str, Any]) -> None:
    lines = [
        "# ERSRR grouped baseline results",
        "",
        f"Generated: `{results['generated_at_utc']}`",
        "",
        f"All metrics are {results['folds']}-fold group-held-out results with thresholds calibrated on "
        "disjoint inner groups. "
        "Legacy imagery and targets are evaluated on the same 128-pixel (~20 m) grid as the U-Net. "
        "Groups are connected components sharing an MGRS tile, Sentinel-2 acquisition, or EMIT granule. "
        "AUPRC and AUROC are threshold-free; F1/IoU are also reported as macro per-scene means.",
        "",
    ]
    for cohort_name, cohort in results["cohorts"].items():
        lines.extend([f"## {cohort_name}", ""])
        for experiment_set in cohort["experiment_sets"]:
            if "target_threshold_ppm_m" in experiment_set:
                lines.extend([f"### Target > {experiment_set['target_threshold_ppm_m']:g} ppm·m", ""])
            lines.extend(
                [
                    "| Model | Features | Pixel AUPRC | Pixel AUROC | Scene F1 | Scene IoU | Runtime s |",
                    "|---|---:|---:|---:|---:|---:|---:|",
                ]
            )
            for model in experiment_set["models"]:
                pixel = model["summary"]["pixel"]
                scene = model["summary"]["scene_macro"]
                lines.append(
                    f"| {model['name']} | {model['feature_count']} | "
                    f"{pixel['auprc_mean']:.4f} | {pixel['roc_auc_mean']:.4f} | "
                    f"{scene['f1_mean']:.4f} | {scene['iou_mean']:.4f} | {model['runtime_seconds']:.2f} |"
                )
            lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = repo_root()
    legacy_dir = root / args.legacy_dir
    v002_batch_dir = root / args.v002_batch_dir
    legacy = load_legacy_samples(
        legacy_dir,
        image_size=args.image_size,
        pixels_per_scene=args.pixels_per_scene,
        seed=args.seed,
        min_valid_pct=args.min_valid_pct,
        max_abs_gap_days=args.max_abs_gap_days,
    )
    legacy_raw = raw_log_features(legacy["bands"])
    legacy_physics = physics_features(legacy["bands"])
    legacy_context = np.column_stack([legacy_physics, legacy["context"]]).astype("float32")
    legacy_sets = []
    for threshold in args.thresholds:
        labels = (legacy["target"] > threshold).astype("uint8")
        specifications: list[tuple[str, np.ndarray, str]] = [
            ("prior_dummy", legacy_raw, "dummy"),
            ("raw_logistic", legacy_raw, "logistic"),
            ("physics_logistic", legacy_physics, "logistic"),
            ("physics_histgb", legacy_physics, "histgb"),
        ]
        models = [
            evaluate_model(
                name,
                features,
                labels,
                legacy["groups"],
                legacy["scenes"],
                model_kind=kind,
                folds=args.folds,
                seed=args.seed,
            )
            for name, features, kind in specifications
        ]
        legacy_sets.append({"target_threshold_ppm_m": threshold, "models": models})

    v002 = load_v002_samples(
        v002_batch_dir,
        pixels_per_scene=args.pixels_per_scene,
        seed=args.seed,
        min_roi_clear_pct=args.min_roi_clear_pct,
        min_mask_positive_pct=args.min_mask_positive_pct,
        max_mask_positive_pct=args.max_mask_positive_pct,
    )
    v002_raw = raw_log_features(v002["nearest"])
    v002_physics = physics_features(v002["nearest"])
    v002_bitemporal = bitemporal_features(v002["before"], v002["after"])
    v002_single_context = np.column_stack([v002_physics, v002["nearest_context"]]).astype("float32")
    v002_bitemporal_context = np.column_stack(
        [
            v002_bitemporal,
            v002["before_context"],
            v002["after_context"],
            v002["after_context"] - v002["before_context"],
        ]
    ).astype("float32")
    v002_specs: list[tuple[str, np.ndarray, str]] = [
        ("prior_dummy", v002_raw, "dummy"),
        ("raw_single_logistic", v002_raw, "logistic"),
        ("physics_single_logistic", v002_physics, "logistic"),
        ("physics_single_histgb", v002_physics, "histgb"),
        ("bitemporal_logistic", v002_bitemporal, "logistic"),
        ("bitemporal_histgb", v002_bitemporal, "histgb"),
    ]
    v002_models = [
        evaluate_model(
            name,
            features,
            v002["labels"],
            v002["groups"],
            v002["scenes"],
            model_kind=kind,
            folds=args.folds,
            seed=args.seed,
        )
        for name, features, kind in v002_specs
    ]

    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "folds": args.folds,
        "pixels_per_scene": args.pixels_per_scene,
        "legacy_image_size": args.image_size,
        "legacy_effective_resolution_m": 20.0,
        "provenance": {
            "git_commit": git_value(root, "rev-parse", "HEAD"),
            "git_tracked_worktree_dirty": bool(
                git_value(root, "status", "--porcelain", "--untracked-files=no")
            ),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "legacy_input_sha256": file_set_sha256(
                root,
                list(legacy_dir.glob("*.tif")),
                b"ersrr-legacy-baseline-input-v1",
            ),
            "v002_input_sha256": file_set_sha256(
                root,
                v002_input_paths(root, v002_batch_dir),
                b"ersrr-v002-baseline-input-v1",
            ),
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "rasterio": rasterio.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "cohorts": {
            "legacy_quality_cohort": {
                "selection": {
                    "min_valid_pct": args.min_valid_pct,
                    "max_abs_gap_days": args.max_abs_gap_days,
                    "selected_scenes": len(legacy["selected"]),
                    "groups": int(np.unique(legacy["groups"]).size),
                    "exclusions": legacy["exclusions"],
                    "files": legacy["selected"],
                },
                "experiment_sets": legacy_sets,
            },
            "v002_physical_mask_holdout": {
                "selection": {
                    "min_roi_clear_pct": args.min_roi_clear_pct,
                    "min_mask_positive_pct": args.min_mask_positive_pct,
                    "max_mask_positive_pct": args.max_mask_positive_pct,
                    "selected_granules": len(v002["selected"]),
                    "groups": int(np.unique(v002["groups"]).size),
                    "exclusions": v002["exclusions"],
                    "granules": v002["selected"],
                },
                "experiment_sets": [{"models": v002_models}],
            },
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--legacy-dir",
        default="EarthRemoteSensingRapidResponse/Dataset/train_test",
    )
    parser.add_argument(
        "--v002-batch-dir",
        default=(
            "EarthRemoteSensingRapidResponse/Data Collection/s2_emit_pairs/"
            "emit-v002-2026-07"
        ),
    )
    parser.add_argument("--output", default="reports/experiments/baseline_results.json")
    parser.add_argument("--markdown", default="reports/experiments/BASELINE_RESULTS.md")
    parser.add_argument("--pixels-per-scene", type=int, default=2048)
    parser.add_argument("--image-size", type=int, choices=(128,), default=128)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--thresholds", type=float, nargs="+", default=list(LEGACY_THRESHOLDS))
    parser.add_argument("--min-valid-pct", type=float, default=10.0)
    parser.add_argument("--max-abs-gap-days", type=float, default=7.0)
    parser.add_argument("--min-roi-clear-pct", type=float, default=70.0)
    parser.add_argument("--min-mask-positive-pct", type=float, default=1.0)
    parser.add_argument("--max-mask-positive-pct", type=float, default=50.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.pixels_per_scene <= 0 or args.folds < 2:
        parser.error("pixels-per-scene must be positive and folds must be at least two")
    started = time.perf_counter()
    try:
        results = run(args)
        results["total_runtime_seconds"] = round(time.perf_counter() - started, 3)
        safe_results = json_safe(results)
        root = repo_root()
        output_path = root / args.output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(safe_results, indent=2, sort_keys=True), encoding="utf-8")
        write_markdown(root / args.markdown, safe_results)
        print(
            json.dumps(
                {
                    "ok": True,
                    "output": str(output_path.relative_to(root)),
                    "markdown": args.markdown,
                    "runtime_seconds": results["total_runtime_seconds"],
                    "legacy_scenes": safe_results["cohorts"]["legacy_quality_cohort"]["selection"]["selected_scenes"],
                    "v002_granules": safe_results["cohorts"]["v002_physical_mask_holdout"]["selection"]["selected_granules"],
                },
                indent=2,
            )
        )
        return 0
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc), "error_type": type(exc).__name__}, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
