#!/usr/bin/env python3
"""Train/evaluate the preregistered standalone MARS sensor-aware ordinal pilot."""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence

import numpy as np
import torch
from scipy import ndimage
from sklearn.metrics import average_precision_score
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "EarthRemoteSensingRapidResponse"
for module_path in (MODEL_ROOT, ROOT / "tools"):
    if str(module_path) not in sys.path:
        sys.path.insert(0, str(module_path))

from mars_s2l_adapter import compute_mbmp, iter_development_manifest, load_sample  # noqa: E402
from mars_sensor_ordinal import MarsSensorOrdinalUNet, pixel_loss  # noqa: E402

DEFAULT_PROTOCOL = Path("configs/mars_sensor_ordinal_protocol.json")
ALLOWED_FOLDS = frozenset({3, 4})
SEED = 26082917
SENSOR_INDEX = {"Sentinel-2": 0, "Landsat": 1}
DENSE_THRESHOLDS = (0.8, 0.7)
DENSE_SCENE_GATE = 0.75
MINIMUM_CONNECTED_PIXELS = 100
GAUSSIAN_DENSE_STRENGTH = 0.1
REQUIRED_RUNTIME_ENV = {
    "PYTORCH_ALLOC_CONF": "expandable_segments:True",
    "CUDA_MODULE_LOADING": "LAZY",
}
REQUIRED_NATIVE_WINDOWS_RUNTIME = {
    "platform": "Windows",
    "gpu": "NVIDIA GeForce RTX 5070",
    "nvidia_driver": "595.79",
    "torch": "2.11.0+cu128",
    "numpy": "2.4.4",
    "rasterio": "1.4.4",
    "scikit-learn": "1.9.0",
    "scipy": "1.17.1",
}
RECOVERY_SCHEMA_VERSION = 1
REQUIRED_RECOVERY_PAYLOAD_KEYS = frozenset({
    "schema_version", "identity", "live_model_state", "best_model_state",
    "pixel_optimizer_state", "scene_optimizer_state",
    "pixel_optimizer_param_groups", "scene_optimizer_param_groups",
    "pixel_optimizer_lrs", "scene_optimizer_lrs", "best_rank", "best_epoch",
    "history", "cutpoints", "completed_epoch", "next_epoch", "held_fold",
    "fit_fold", "rng_state", "access_ledger",
})


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def runtime_signature() -> dict[str, Any]:
    """Return the exact native-Windows production runtime identity."""
    driver = "unavailable"
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
        driver = completed.stdout.strip().splitlines()[0].strip()
    except (OSError, subprocess.SubprocessError, IndexError):
        pass
    versions: dict[str, str] = {}
    for distribution in ("numpy", "rasterio", "scikit-learn", "scipy"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = "unavailable"
    return {
        "platform": platform.system(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "unavailable",
        "nvidia_driver": driver,
        "torch": torch.__version__,
        **versions,
        "environment": {name: os.environ.get(name) for name in REQUIRED_RUNTIME_ENV},
    }


def verify_runtime_environment(*, require_native_windows: bool = False) -> dict[str, Any]:
    """Require exact env vars and, for production modes, the frozen Windows stack."""
    mismatches = {
        name: {"required": required, "actual": os.environ.get(name)}
        for name, required in REQUIRED_RUNTIME_ENV.items()
        if os.environ.get(name) != required
    }
    if mismatches:
        raise RuntimeError(f"Runtime compatibility environment mismatch: {mismatches}")
    signature = runtime_signature()
    if require_native_windows:
        runtime_mismatches = {
            name: {"required": required, "actual": signature.get(name)}
            for name, required in REQUIRED_NATIVE_WINDOWS_RUNTIME.items()
            if signature.get(name) != required
        }
        if runtime_mismatches:
            raise RuntimeError(f"Native-Windows production runtime mismatch: {runtime_mismatches}")
    return signature


def validate_requested_folds(folds: Iterable[int]) -> tuple[int, ...]:
    result = tuple(sorted(set(map(int, folds))))
    if not result or not set(result).issubset(ALLOWED_FOLDS):
        raise ValueError(f"Protected fold request rejected: {result}; allowed folds are [3, 4]")
    return result


def fold_lookup(path: Path) -> dict[str, int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    result = {str(row["group_id"]): int(row["fold"]) for row in payload["assignments"]}
    if set(result.values()) != {0, 1, 2, 3, 4}:
        raise ValueError("Canonical fold protocol is incomplete")
    return result


def records_for_folds(manifest: Path, groups: dict[str, int], folds: Iterable[int]) -> list[dict[str, Any]]:
    selected = set(validate_requested_folds(folds))
    rows = [row for row in iter_development_manifest(manifest) if groups[str(row["group_id"])] in selected]
    if not rows:
        raise ValueError("No rows found for requested folds")
    return rows


def deterministic_inner_split(records: Sequence[dict[str, Any]], seed: int = SEED) -> tuple[set[str], set[str]]:
    """Hash-stratified canonical-group split with missing scene-class promotion."""
    labels: dict[str, bool] = {}
    scene_labels: dict[str, set[int]] = {}
    for row in records:
        group = str(row["group_id"])
        positive = str(row["label_state"]) == "PLUME"
        labels[group] = labels.get(group, False) or positive
        scene_labels.setdefault(group, set()).add(int(positive))
    validation: set[str] = set()
    training: set[str] = set(labels)
    for positive in (False, True):
        ordered = sorted(
            (group for group, value in labels.items() if value is positive),
            key=lambda group: hashlib.sha1(f"{seed}:{group}".encode()).hexdigest(),
        )
        chosen = set(ordered[::5])
        validation.update(chosen)
        training.difference_update(chosen)
    validation_scene_classes = set().union(*(scene_labels[g] for g in validation)) if validation else set()
    for missing in sorted({0, 1} - validation_scene_classes):
        eligible = sorted(
            (group for group in training if missing in scene_labels[group]),
            key=lambda group: hashlib.sha1(f"{seed}:{group}".encode()).hexdigest(),
        )
        if not eligible:
            raise ValueError(f"Cannot promote missing inner-validation scene class {missing}")
        validation.add(eligible[0])
        training.remove(eligible[0])
    if not training or training & validation:
        raise ValueError("Invalid deterministic inner split")
    return training, validation


def fit_ordinal_cutpoints(metadata_root: Path, records: Sequence[dict[str, Any]]) -> np.ndarray:
    """Fit producer-order cutpoints from inner-training records and nowhere else."""
    values: list[np.ndarray] = []
    for row in records:
        if str(row["label_state"]) != "PLUME" or not bool(row.get("pixel_truth_available", True)):
            continue
        sample = load_sample(metadata_root, row, allow_empty_positive_mask=True)
        enhancement = sample.methane_enhancement_raw
        if enhancement is None:
            continue
        supported = sample.plume_mask & sample.observable_mask & np.isfinite(enhancement)
        if np.any(supported):
            values.append(np.asarray(enhancement[supported], dtype=np.float64))
    if not values:
        raise ValueError("Inner-training records have no finite observable positive enhancement pixels")
    result = np.quantile(np.concatenate(values), (0.25, 0.50, 0.75), method="linear")
    if not np.isfinite(result).all() or np.any(result[1:] < result[:-1]):
        raise ValueError("Invalid ordinal cutpoints")
    return result


def prepare_endpoint_data(
    metadata_root: Path, fit_records: Sequence[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], set[str], set[str], np.ndarray]:
    """Freeze the inner split, then fit ordinal cutpoints on inner training only."""
    training_groups, validation_groups = deterministic_inner_split(fit_records)
    inner_training = [row for row in fit_records if str(row["group_id"]) in training_groups]
    inner_validation = [row for row in fit_records if str(row["group_id"]) in validation_groups]
    cutpoints = fit_ordinal_cutpoints(metadata_root, inner_training)
    return inner_training, inner_validation, training_groups, validation_groups, cutpoints


def ordinal_levels(enhancement: np.ndarray | None, plume: np.ndarray, observable: np.ndarray, cutpoints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Assign ties low; unsupported positive pixels receive no ordinal weight."""
    levels = np.zeros(plume.shape, dtype=np.int64)
    support = np.asarray(observable, bool) & ~np.asarray(plume, bool)
    if enhancement is not None:
        positive = np.asarray(plume, bool) & np.asarray(observable, bool) & np.isfinite(enhancement)
        levels[positive] = np.searchsorted(np.asarray(cutpoints), enhancement[positive], side="left") + 1
        support |= positive
    return levels, support


def model_input(sample: Any) -> np.ndarray:
    # The adapter exposes raw DN / 5000. MARS physical reflectance is DN / 10000,
    # so apply the fixed 0.5 conversion before the frozen clamp and scaling.
    reflectance = np.asarray(sample.reflectance_pair, np.float32) * 0.5
    reflectance = np.clip(reflectance, 0.0, 1.5)
    reflectance = reflectance * (2.0 / 1.5) - 1.0
    support = np.stack((sample.radiometric_valid_mask, ~sample.clear_mask)).astype(np.float32)
    result = np.concatenate((reflectance, support)).astype(np.float32)
    if result.shape[0] != 14 or not np.isfinite(result).all():
        raise ValueError("Invalid 14-channel sensor-aware input")
    return result


def _crop(array: np.ndarray, top: int, left: int, size: int, *, value: float = 0.0) -> np.ndarray:
    height, width = array.shape[-2:]
    result = np.full((*array.shape[:-2], size, size), value, dtype=array.dtype)
    src_top, src_left = max(top, 0), max(left, 0)
    src_bottom, src_right = min(top + size, height), min(left + size, width)
    if src_bottom > src_top and src_right > src_left:
        result[..., src_top - top : src_bottom - top, src_left - left : src_right - left] = array[..., src_top:src_bottom, src_left:src_right]
    return result


def _random_valid_crop_origin(
    observable: np.ndarray,
    *,
    size: int,
    minimum_valid_fraction: float,
    rng: np.random.Generator,
) -> tuple[int, int]:
    """Uniformly choose an integer crop origin meeting source-support coverage.

    Coverage is computed over source pixels intersected by the crop. Padding is
    outside the sensor raster and therefore is not counted as invalid support.
    A random permutation makes the first passing origin uniform over all passing
    origins while normally terminating after only one or two tests.
    """
    height, width = observable.shape
    tops = np.arange(min(0, height - size), max(height - size, 0) + 1)
    lefts = np.arange(min(0, width - size), max(width - size, 0) + 1)
    positions = np.asarray([(int(top), int(left)) for top in tops for left in lefts])
    for index in rng.permutation(len(positions)):
        top, left = map(int, positions[index])
        view = observable[
            max(top, 0) : min(top + size, height),
            max(left, 0) : min(left + size, width),
        ]
        if view.size and float(view.mean()) >= minimum_valid_fraction:
            return top, left
    raise ValueError(
        f"Scene has no {size}x{size} crop with source valid fraction "
        f">= {minimum_valid_fraction:.2f}"
    )


def apply_shared_spatial_transform(
    values: dict[str, Any], *, turns: int, horizontal: bool, vertical: bool
) -> dict[str, Any]:
    """Apply one sampled transform identically to every spatial tensor."""
    spatial = ("inputs", "observable", "plume", "ordinal_level", "ordinal_support")
    for key in spatial:
        array = np.rot90(values[key], int(turns), axes=(-2, -1))
        if horizontal:
            array = np.flip(array, axis=-1)
        if vertical:
            array = np.flip(array, axis=-2)
        values[key] = np.ascontiguousarray(array)
    return values


def _load_training_sample(metadata_root: Path, row: dict[str, Any]) -> Any:
    return load_sample(
        metadata_root,
        row,
        require_enhancement=False,
        allow_empty_positive_mask=True,
        allow_missing_positive_mask=True,
    )


def make_crop(sample: Any, cutpoints: np.ndarray, *, size: int, rng: np.random.Generator, augment: bool) -> dict[str, Any]:
    inputs = model_input(sample)
    observable = np.asarray(sample.observable_mask, bool)
    plume = np.asarray(sample.plume_mask, bool)
    if sample.presence and np.any(plume & observable):
        candidates = np.argwhere(plume & observable)
        center = candidates[int(rng.integers(len(candidates)))]
        top, left = int(center[0] - size // 2), int(center[1] - size // 2)
    else:
        top, left = _random_valid_crop_origin(
            observable,
            size=size,
            minimum_valid_fraction=0.70,
            rng=rng,
        )
    levels, ordinal_support = ordinal_levels(sample.methane_enhancement_raw, plume, observable, cutpoints)
    values: dict[str, Any] = {
        "inputs": _crop(inputs, top, left, size),
        "observable": _crop(observable, top, left, size),
        "plume": _crop(plume, top, left, size),
        "ordinal_level": _crop(levels, top, left, size),
        "ordinal_support": _crop(ordinal_support, top, left, size),
        "presence": float(sample.presence),
        "sensor_index": SENSOR_INDEX[sample.sensor_family],
        "sample_id": sample.sample_id,
    }
    if augment:
        apply_shared_spatial_transform(
            values,
            turns=int(rng.integers(4)),
            horizontal=bool(rng.integers(2)),
            vertical=bool(rng.integers(2)),
        )
    return values


def make_full(sample: Any, cutpoints: np.ndarray) -> dict[str, Any]:
    plume = np.asarray(sample.plume_mask, bool)
    observable = np.asarray(sample.observable_mask, bool)
    levels, ordinal_support = ordinal_levels(sample.methane_enhancement_raw, plume, observable, cutpoints)
    return {
        "inputs": model_input(sample), "observable": observable, "plume": plume,
        "ordinal_level": levels, "ordinal_support": ordinal_support,
        "presence": float(sample.presence), "sensor_index": SENSOR_INDEX[sample.sensor_family],
        "sample_id": sample.sample_id,
    }


def collate(rows: Sequence[dict[str, Any]], device: torch.device) -> dict[str, Any]:
    """Pad native full scenes (or stack crops) to one multiple-of-eight batch."""
    height = max(row["inputs"].shape[-2] for row in rows)
    width = max(row["inputs"].shape[-1] for row in rows)
    height, width = int(math.ceil(height / 8) * 8), int(math.ceil(width / 8) * 8)
    padded: list[dict[str, Any]] = []
    for row in rows:
        local = dict(row)
        for key in ("inputs", "observable", "plume", "ordinal_level", "ordinal_support"):
            array = row[key]
            result = np.zeros((*array.shape[:-2], height, width), dtype=array.dtype)
            result[..., : array.shape[-2], : array.shape[-1]] = array
            local[key] = result
        padded.append(local)
    result: dict[str, Any] = {"sample_id": [row["sample_id"] for row in rows]}
    if all("group_id" in row for row in rows):
        result["group_id"] = [str(row["group_id"]) for row in rows]
    fields = (("inputs", torch.float32), ("observable", torch.bool), ("plume", torch.float32),
              ("ordinal_level", torch.long), ("ordinal_support", torch.bool),
              ("presence", torch.float32), ("sensor_index", torch.long))
    for key, dtype in fields:
        tensor = torch.as_tensor(np.stack([row[key] for row in padded]), dtype=dtype, device=device)
        if key == "observable":
            tensor = tensor[:, None]
        result[key] = tensor
    return result


@dataclass
class SiteBalancedBatcher:
    metadata_root: Path
    records: Sequence[dict[str, Any]]
    cutpoints: np.ndarray
    rng: np.random.Generator

    def __post_init__(self) -> None:
        self.by_label: dict[int, dict[str, list[dict[str, Any]]]] = {0: {}, 1: {}}
        for row in self.records:
            label = int(str(row["label_state"]) == "PLUME")
            group = str(row["group_id"])
            self.by_label[label].setdefault(group, []).append(row)
        if not self.by_label[0] or not self.by_label[1]:
            raise ValueError("Site-balanced training requires positive and negative groups")

    def rows(self, positive: int, negative: int) -> list[dict[str, Any]]:
        selected: list[dict[str, Any]] = []
        used_groups: set[str] = set()
        for label, count in ((1, positive), (0, negative)):
            groups = sorted(set(self.by_label[label]) - used_groups)
            if len(groups) < count:
                raise ValueError(
                    f"Need {count} distinct label-{label} sites, found {len(groups)}"
                )
            chosen = self.rng.choice(groups, size=count, replace=False)
            for group in chosen:
                options = self.by_label[label][str(group)]
                selected.append(options[int(self.rng.integers(len(options)))])
                used_groups.add(str(group))
        order = self.rng.permutation(len(selected))
        return [selected[int(index)] for index in order]

    def dense_batch(self, device: torch.device, size: int = 256) -> dict[str, Any]:
        crops: list[dict[str, Any]] = []
        used_groups: set[str] = set()
        for label, count in ((1, 8), (0, 8)):
            groups = np.asarray(sorted(set(self.by_label[label]) - used_groups))
            accepted = 0
            for group_index in self.rng.permutation(len(groups)):
                group = str(groups[int(group_index)])
                options = self.by_label[label][group]
                for option_index in self.rng.permutation(len(options)):
                    sample = _load_training_sample(
                        self.metadata_root, options[int(option_index)]
                    )
                    try:
                        crop = make_crop(
                            sample,
                            self.cutpoints,
                            size=size,
                            rng=self.rng,
                            augment=True,
                        )
                    except ValueError as error:
                        if "source valid fraction" not in str(error):
                            raise
                        continue
                    crop["group_id"] = group
                    crops.append(crop)
                    used_groups.add(group)
                    accepted += 1
                    break
                if accepted == count:
                    break
            if accepted != count:
                raise ValueError(
                    f"Could not form {count} distinct valid label-{label} site crops; "
                    f"formed {accepted}"
                )
        order = self.rng.permutation(len(crops))
        return collate([crops[int(index)] for index in order], device)

    def scene_batch(self, device: torch.device) -> dict[str, Any]:
        rows = self.rows(2, 2)
        samples = [_load_training_sample(self.metadata_root, row) for row in rows]
        return collate([make_full(sample, self.cutpoints) for sample in samples], device)


def pixel_step(model: MarsSensorOrdinalUNet, batch: dict[str, Any], optimizer: torch.optim.Optimizer) -> dict[str, float]:
    optimizer.zero_grad(set_to_none=True)
    output = model(batch["inputs"], batch["sensor_index"], batch["observable"])
    losses = pixel_loss(output, batch["plume"], batch["observable"].squeeze(1), batch["ordinal_level"], batch["ordinal_support"])
    if not torch.isfinite(losses["loss"]):
        raise FloatingPointError("Non-finite pixel loss")
    losses["loss"].backward()
    # The foreach CUDA path may request an additional TensorList workspace at
    # peak backward memory. Keep the frozen global L2 clip while forcing the
    # scalar implementation, whose temporary allocation is bounded per tensor.
    gradient = torch.nn.utils.clip_grad_norm_(model.pixel_parameters(), 2.0, foreach=False)
    if not torch.isfinite(gradient):
        raise FloatingPointError("Non-finite pixel gradient")
    optimizer.step()
    return {"pixel_loss": float(losses["loss"].detach()), "pixel_gradient_norm": float(gradient)}


def scene_step(model: MarsSensorOrdinalUNet, batch: dict[str, Any], optimizer: torch.optim.Optimizer) -> dict[str, float]:
    for parameter in model.parameters():
        parameter.grad = None
    optimizer.zero_grad(set_to_none=True)
    output = model(batch["inputs"], batch["sensor_index"], batch["observable"])
    loss = F.binary_cross_entropy_with_logits(output["scene_logit"], batch["presence"])
    if not torch.isfinite(loss):
        raise FloatingPointError("Non-finite scene loss")
    loss.backward()
    leaking = [name for name, parameter in model.named_parameters()
               if not name.startswith(("scene_projection.", "scene_mlp."))
               and parameter.grad is not None and torch.count_nonzero(parameter.grad).item()]
    if leaking:
        raise RuntimeError(f"Scene gradient leaked into pixel model: {leaking[:3]}")
    gradient = torch.nn.utils.clip_grad_norm_(model.scene_parameters(), 2.0, foreach=False)
    if not torch.isfinite(gradient):
        raise FloatingPointError("Non-finite scene gradient")
    optimizer.step()
    return {"scene_loss": float(loss.detach()), "scene_gradient_norm": float(gradient)}


def finite_gradient_step(model: MarsSensorOrdinalUNet, batch: dict[str, Any], pixel_optimizer: torch.optim.Optimizer, scene_optimizer: torch.optim.Optimizer) -> dict[str, float]:
    """Smoke helper: one independent pixel update and one independent scene update."""
    return {**pixel_step(model, batch, pixel_optimizer), **scene_step(model, batch, scene_optimizer)}


def dense_counts(truth: np.ndarray, prediction: np.ndarray, valid: np.ndarray) -> np.ndarray:
    truth, prediction, valid = map(lambda value: np.asarray(value, bool), (truth, prediction, valid))
    return np.asarray([np.count_nonzero(valid & truth & prediction), np.count_nonzero(valid & ~truth & prediction), np.count_nonzero(valid & truth & ~prediction)], dtype=np.int64)


def aggregate_iou(counts: np.ndarray) -> float:
    total = np.asarray(counts, np.int64).sum(axis=0)
    return float(total[0] / max(int(total.sum()), 1))


def _safe_ap(labels: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(labels)) != 2:
        raise ValueError("Average precision requires both scene classes")
    return float(average_precision_score(labels, scores))


@torch.no_grad()
def evaluate_candidate(model: MarsSensorOrdinalUNet, metadata_root: Path, records: Sequence[dict[str, Any]], cutpoints: np.ndarray, device: torch.device, *, fold: int) -> dict[str, np.ndarray]:
    """Run one native-resolution candidate inference pass in manifest order."""
    model.eval()
    result: dict[str, list[Any]] = {key: [] for key in ("sample_ids", "labels", "sensors", "groups", "folds", "scores", "dense_counts")}
    for row in records:
        sample = _load_training_sample(metadata_root, row)
        batch = collate([make_full(sample, cutpoints)], device)
        output = model(batch["inputs"], batch["sensor_index"], batch["observable"])
        score = float(torch.sigmoid(output["scene_logit"])[0].cpu())
        probability = torch.sigmoid(output["binary_logit"])[0, 0].cpu().numpy()
        height, width = sample.observable_mask.shape
        prediction = probability[:height, :width] >= 0.40
        pixel_valid = sample.observable_mask & bool(row.get("pixel_truth_available", True))
        result["sample_ids"].append(sample.sample_id)
        result["labels"].append(sample.presence)
        result["sensors"].append(SENSOR_INDEX[sample.sensor_family])
        result["groups"].append(str(row["group_id"]))
        result["folds"].append(fold)
        result["scores"].append(score)
        result["dense_counts"].append(dense_counts(sample.plume_mask, prediction, pixel_valid))
    return {
        "sample_ids": np.asarray(result["sample_ids"], dtype=str),
        "labels": np.asarray(result["labels"], dtype=np.uint8),
        "sensors": np.asarray(result["sensors"], dtype=np.uint8),
        "groups": np.asarray(result["groups"], dtype=str),
        "folds": np.asarray(result["folds"], dtype=np.uint8),
        "scores": np.asarray(result["scores"], dtype=np.float64),
        "dense_counts": np.asarray(result["dense_counts"], dtype=np.int64),
    }


def checkpoint_metrics(values: dict[str, np.ndarray]) -> dict[str, float]:
    labels, scores = values["labels"], values["scores"]
    clipped = np.clip(scores, 1e-7, 1 - 1e-7)
    bce = float(np.mean(-(labels * np.log(clipped) + (1 - labels) * np.log1p(-clipped))))
    return {"scene_ap": _safe_ap(labels, scores), "dense_iou": aggregate_iou(values["dense_counts"]), "scene_bce": bce}


def checkpoint_rank(metrics: dict[str, float]) -> tuple[float, float, float]:
    return (metrics["scene_ap"], metrics["dense_iou"], -metrics["scene_bce"])


def pixel_learning_rate(epoch: int, epochs: int, base: float = 3e-4, minimum: float = 3e-5) -> float:
    if epoch <= 2:
        return base * epoch / 2.0
    progress = (epoch - 2) / max(epochs - 2, 1)
    return minimum + 0.5 * (base - minimum) * (1.0 + math.cos(math.pi * progress))


def scene_learning_rate(
    epoch: int,
    epochs: int,
    *,
    warmup_step: int | None = None,
    warmup_steps: int = 150,
    base: float = 1e-3,
    minimum: float = 1e-4,
) -> float:
    if epoch < 5:
        return 0.0
    if epoch == 5 and warmup_step is not None:
        if not 1 <= warmup_step <= warmup_steps:
            raise ValueError("Scene warmup step is outside the frozen epoch-5 schedule")
        return base * warmup_step / warmup_steps
    progress = (epoch - 5) / max(epochs - 5, 1)
    return minimum + 0.5 * (base - minimum) * (1.0 + math.cos(math.pi * progress))


def set_learning_rate(optimizer: torch.optim.Optimizer, value: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = value


def _cpu_state(model: MarsSensorOrdinalUNet) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def train_endpoint(
    protocol: dict[str, Any],
    paths: dict[str, Path],
    fit_records: Sequence[dict[str, Any]],
    held_fold: int,
    device: torch.device,
    *,
    protocol_path: Path | None = None,
    runtime: dict[str, Any] | None = None,
    ledger: AccessLedger | None = None,
) -> tuple[dict[str, Any], dict[str, torch.Tensor], np.ndarray]:
    """Train one endpoint with an exact, post-validation epoch commit boundary."""
    inner_training, inner_validation, training_groups, validation_groups, cutpoints = prepare_endpoint_data(
        paths["metadata_root"], fit_records
    )
    protocol_path = protocol_path or (ROOT / DEFAULT_PROTOCOL).resolve()
    runtime = runtime or runtime_signature()
    ledger = ledger or AccessLedger(comparator_integrity_bytes_hashed=True)
    identity = checkpoint_identity(
        protocol, protocol_path, paths, fit_records, inner_training, inner_validation,
        held_fold, runtime, ledger,
    )
    model = MarsSensorOrdinalUNet().to(device)
    spec = protocol["training"]
    pixel_optimizer = torch.optim.AdamW(model.pixel_parameters(), lr=0.0, betas=(0.9, 0.999), weight_decay=1e-4)
    scene_optimizer = torch.optim.AdamW(model.scene_parameters(), lr=0.0, betas=(0.9, 0.999), weight_decay=1e-4)
    batcher = SiteBalancedBatcher(paths["metadata_root"], inner_training, cutpoints, np.random.default_rng(SEED))
    recovery_root = (ROOT / protocol["outputs"]["candidate_predictions"]).resolve().parent / "recovery" / f"held-{held_fold}"
    store = RecoveryStore(recovery_root, identity, device)
    best_state: dict[str, torch.Tensor] | None = None
    best_rank: tuple[float, float, float] | None = None
    best_epoch = 0
    history: list[dict[str, Any]] = []
    start_epoch = 1
    recovered = store.load()
    if recovered is not None:
        model.load_state_dict(recovered["live_model_state"], strict=True)
        pixel_optimizer.load_state_dict(recovered["pixel_optimizer_state"])
        scene_optimizer.load_state_dict(recovered["scene_optimizer_state"])
        best_state = recovered["best_model_state"]
        best_rank = tuple(map(float, recovered["best_rank"]))
        best_epoch = int(recovered["best_epoch"])
        history = copy.deepcopy(recovered["history"])
        start_epoch = int(recovered["next_epoch"])
        if (
            int(recovered["held_fold"]) != held_fold
            or int(recovered["fit_fold"]) != 7 - held_fold
            or not np.array_equal(np.asarray(recovered["cutpoints"]), cutpoints)
            or int(recovered["completed_epoch"]) + 1 != start_epoch
            or len(history) != int(recovered["completed_epoch"])
        ):
            raise RuntimeError("Recovery endpoint/cutpoint/history boundary mismatch")
        if (
            recovered.get("pixel_optimizer_param_groups") != recovered["pixel_optimizer_state"].get("param_groups")
            or recovered.get("scene_optimizer_param_groups") != recovered["scene_optimizer_state"].get("param_groups")
            or recovered.get("pixel_optimizer_lrs") != [float(group["lr"]) for group in pixel_optimizer.param_groups]
            or recovered.get("scene_optimizer_lrs") != [float(group["lr"]) for group in scene_optimizer.param_groups]
        ):
            raise RuntimeError("Recovery optimizer groups/learning rates mismatch")
        # This is deliberately last: construction and all state loading may consume RNG.
        restore_rng_state(recovered["rng_state"], batcher)
    epochs = int(spec["epochs"])
    for epoch in range(start_epoch, epochs + 1):
        pixel_lr = pixel_learning_rate(epoch, epochs)
        scene_lr = scene_learning_rate(epoch, epochs)
        set_learning_rate(pixel_optimizer, pixel_lr)
        set_learning_rate(scene_optimizer, scene_lr)
        model.train()
        pixel_losses = [pixel_step(model, batcher.dense_batch(device, int(spec["crop_size"])), pixel_optimizer)["pixel_loss"] for _ in range(int(spec["dense_steps_per_epoch"]))]
        scene_losses: list[float] = []
        if epoch >= 5:
            scene_steps = int(spec["scene_steps_per_epoch_after_epoch_4"])
            for step in range(1, scene_steps + 1):
                if epoch == 5:
                    set_learning_rate(
                        scene_optimizer,
                        scene_learning_rate(
                            epoch,
                            epochs,
                            warmup_step=step,
                            warmup_steps=scene_steps,
                        ),
                    )
                scene_losses.append(
                    scene_step(model, batcher.scene_batch(device), scene_optimizer)["scene_loss"]
                )
        validation = evaluate_candidate(model, paths["metadata_root"], inner_validation, cutpoints, device, fold=held_fold)
        ledger.inner_validation_outcomes_opened = True
        metrics = checkpoint_metrics(validation)
        rank = checkpoint_rank(metrics)
        if best_rank is None or rank > best_rank:
            best_rank, best_state, best_epoch = rank, _cpu_state(model), epoch
        row = {"epoch": epoch, "pixel_lr": pixel_lr, "scene_lr": scene_lr,
               "pixel_loss_mean": float(np.mean(pixel_losses)),
               "scene_loss_mean": None if not scene_losses else float(np.mean(scene_losses)), **metrics,
               "selected_so_far": best_epoch}
        history.append(row)
        if best_state is None or best_rank is None:
            raise RuntimeError("Checkpoint selection produced no endpoint")
        # The generation is committed only after updates, validation, rank, and history.
        # A crash anywhere earlier therefore discards the entire partial epoch.
        store.save({
            "schema_version": RECOVERY_SCHEMA_VERSION,
            "identity": identity,
            "live_model_state": _cpu_state(model),
            "best_model_state": best_state,
            "pixel_optimizer_state": pixel_optimizer.state_dict(),
            "scene_optimizer_state": scene_optimizer.state_dict(),
            "pixel_optimizer_param_groups": copy.deepcopy(pixel_optimizer.state_dict()["param_groups"]),
            "scene_optimizer_param_groups": copy.deepcopy(scene_optimizer.state_dict()["param_groups"]),
            "pixel_optimizer_lrs": [float(group["lr"]) for group in pixel_optimizer.param_groups],
            "scene_optimizer_lrs": [float(group["lr"]) for group in scene_optimizer.param_groups],
            "best_rank": list(best_rank),
            "best_epoch": best_epoch,
            "history": copy.deepcopy(history),
            "cutpoints": cutpoints.copy(),
            "completed_epoch": epoch,
            "next_epoch": epoch + 1,
            "held_fold": held_fold,
            "fit_fold": 7 - held_fold,
            "rng_state": capture_rng_state(batcher),
            "access_ledger": ledger.snapshot(),
        })
        print(json.dumps({"progress": "endpoint_epoch", "held_fold": held_fold, **row}), flush=True)
    if best_state is None or best_rank is None:
        raise RuntimeError("Checkpoint selection produced no endpoint")
    model.load_state_dict(best_state, strict=True)
    metadata = {
        "held_fold": held_fold, "fit_fold": 7 - held_fold,
        "inner_training_groups": len(training_groups), "inner_validation_groups": len(validation_groups),
        "inner_training_rows": len(inner_training), "inner_validation_rows": len(inner_validation),
        "cutpoints": cutpoints.tolist(), "cutpoint_source": "inner_training_groups_only",
        "selected_epoch": best_epoch, "selected_rank": list(best_rank), "history": history,
        "recovery_identity_sha256": canonical_json_hash(identity),
    }
    # Durable endpoint state/metadata is sealed before run_full is allowed to open held data.
    endpoint_path = recovery_root / "endpoint.pt"
    endpoint_descriptor_path = recovery_root / "endpoint.json"
    endpoint_value = {
        "schema_version": RECOVERY_SCHEMA_VERSION,
        "identity": identity,
        "metadata": metadata,
        "best_model_state": best_state,
        "cutpoints": cutpoints.copy(),
    }
    if endpoint_descriptor_path.exists() and not endpoint_path.exists():
        raise RuntimeError("Durable endpoint descriptor exists without endpoint state")
    if endpoint_path.exists():
        if not endpoint_path.is_file() or endpoint_path.stat().st_mode & 0o222:
            raise RuntimeError("Durable endpoint state is not an immutable file")
        existing_endpoint = torch.load(endpoint_path, map_location="cpu", weights_only=False)
        if (
            existing_endpoint.get("identity") != identity
            or existing_endpoint.get("metadata") != metadata
            or not np.array_equal(np.asarray(existing_endpoint.get("cutpoints")), cutpoints)
            or not _nested_exact_equal(existing_endpoint.get("best_model_state"), best_state)
        ):
            raise RuntimeError("Durable endpoint artifact contents mismatch")
    else:
        temporary = endpoint_path.with_name(f".{endpoint_path.name}.{uuid.uuid4().hex}.tmp")
        with temporary.open("wb") as stream:
            torch.save(endpoint_value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o444)
        os.replace(temporary, endpoint_path)
        _fsync_directory(endpoint_path.parent)
    _seal_or_validate_immutable_json(endpoint_descriptor_path, {
        "schema_version": RECOVERY_SCHEMA_VERSION,
        "identity_sha256": canonical_json_hash(identity),
        "sha256": sha256(endpoint_path),
        "selected_epoch": best_epoch,
    })
    return metadata, best_state, cutpoints


def matched_fpr_recall(labels: np.ndarray, scores: np.ndarray, fpr: float) -> float:
    labels, scores = np.asarray(labels, int), np.asarray(scores, float)
    order = np.argsort(-scores, kind="stable")
    sorted_labels = labels[order]
    ends = np.flatnonzero(np.r_[scores[order][1:] != scores[order][:-1], True])
    tp = np.cumsum(sorted_labels)[ends]
    fp = np.cumsum(1 - sorted_labels)[ends]
    allowed = fp / max(np.count_nonzero(labels == 0), 1) <= float(fpr) + 1e-12
    return 0.0 if not np.any(allowed) else float(np.max(tp[allowed]) / max(labels.sum(), 1))


def _bootstrap_group_weights(groups: np.ndarray, folds: np.ndarray, replicates: int, seed: int) -> Iterator[np.ndarray]:
    """Yield per-row multiplicities for paired group draws without quadratic scans."""
    unique_groups, row_group = np.unique(groups, return_inverse=True)
    group_fold = np.full(len(unique_groups), -1, dtype=np.int8)
    for index in range(len(unique_groups)):
        local_folds = np.unique(folds[row_group == index])
        if len(local_folds) != 1:
            raise ValueError("A canonical group crosses outer folds")
        group_fold[index] = int(local_folds[0])
    strata = [np.flatnonzero(group_fold == fold) for fold in (3, 4)]
    if any(not len(values) for values in strata):
        raise ValueError("Bootstrap requires groups in both outer folds")
    rng = np.random.default_rng(seed)
    for _ in range(replicates):
        group_weights = np.zeros(len(unique_groups), dtype=np.int64)
        for values in strata:
            selected = rng.choice(values, size=len(values), replace=True)
            group_weights += np.bincount(selected, minlength=len(unique_groups))
        yield group_weights[row_group]


def metric_gates(labels: np.ndarray, candidate: np.ndarray, comparator: np.ndarray, folds: np.ndarray, sensors: np.ndarray, groups: np.ndarray, candidate_dense: np.ndarray, comparator_dense: np.ndarray, *, replicates: int, ap_seed: int, dense_seed: int) -> dict[str, Any]:
    labels = np.asarray(labels, int); candidate = np.asarray(candidate, float); comparator = np.asarray(comparator, float)
    folds = np.asarray(folds, int); sensors = np.asarray(sensors); groups = np.asarray(groups).astype(str)
    candidate_dense = np.asarray(candidate_dense, np.int64); comparator_dense = np.asarray(comparator_dense, np.int64)
    size = len(labels)
    if any(len(values) != size for values in (candidate, comparator, folds, sensors, groups, candidate_dense, comparator_dense)):
        raise ValueError("Metric vectors are not aligned")
    def ap_delta(rows: np.ndarray) -> float:
        return _safe_ap(labels[rows], candidate[rows]) - _safe_ap(labels[rows], comparator[rows])
    pooled = ap_delta(np.ones(size, bool))
    by_fold = {str(value): ap_delta(folds == value) for value in (3, 4)}
    by_sensor = {str(value): ap_delta(sensors == value) for value in np.unique(sensors)}
    fprs = (0.005, 0.01, 0.02, 0.05, 0.10)
    recall = float(np.mean([matched_fpr_recall(labels, candidate, value) - matched_fpr_recall(labels, comparator, value) for value in fprs]))
    ap_bootstrap: list[float] = []
    for weights in _bootstrap_group_weights(groups, folds, replicates, ap_seed):
        ap_bootstrap.append(float(
            average_precision_score(labels, candidate, sample_weight=weights)
            - average_precision_score(labels, comparator, sample_weight=weights)
        ))
    dense_bootstrap: list[float] = []
    for weights in _bootstrap_group_weights(groups, folds, replicates, dense_seed):
        dense_bootstrap.append(aggregate_iou(candidate_dense * weights[:, None]) - aggregate_iou(comparator_dense * weights[:, None]))
    dense_delta = aggregate_iou(candidate_dense) - aggregate_iou(comparator_dense)
    ap_lower = float(np.quantile(ap_bootstrap, 0.025, method="linear"))
    dense_lower = float(np.quantile(dense_bootstrap, 0.025, method="linear"))
    checks = {
        "pooled_ap_delta_gte_0_003": pooled >= 0.003,
        "each_fold_ap_positive": min(by_fold.values()) > 0,
        "each_sensor_ap_positive": min(by_sensor.values()) > 0,
        "matched_fpr_recall_nonnegative": recall >= 0,
        "ap_bootstrap_lower_positive": ap_lower > 0,
        "dense_iou_delta_positive": dense_delta > 0,
        "dense_bootstrap_lower_positive": dense_lower > 0,
    }
    return {"bootstrap_replicates": replicates, "pooled_ap_delta": pooled, "fold_ap_delta": by_fold,
            "sensor_ap_delta": by_sensor, "matched_fpr_recall_delta": recall,
            "ap_bootstrap_lower": ap_lower, "dense_iou_delta": dense_delta,
            "dense_bootstrap_lower": dense_lower, "checks": checks, "passed": all(checks.values())}


def align_comparator(candidate: dict[str, np.ndarray], comparator_path: Path) -> dict[str, np.ndarray]:
    """Align frozen scene evidence only after immutable candidate predictions exist."""
    with np.load(comparator_path, allow_pickle=False) as source:
        comparator = {key: source[key].copy() for key in source.files}
    required = ("sample_ids", "folds", "sensors", "groups")
    if any(key not in candidate or key not in comparator for key in required):
        raise ValueError("Candidate/comparator identity schema incomplete")
    lookup = {str(value): index for index, value in enumerate(comparator["sample_ids"])}
    if len(lookup) != len(comparator["sample_ids"]) or set(map(str, candidate["sample_ids"])) != set(lookup):
        raise ValueError("Candidate/comparator identities differ")
    order = np.asarray([lookup[str(value)] for value in candidate["sample_ids"]])
    for key in required[1:]:
        if not np.array_equal(np.asarray(candidate[key]).astype(str), np.asarray(comparator[key])[order].astype(str)):
            raise ValueError(f"Candidate/comparator {key} differ")
    return {key: values[order] if np.asarray(values).shape[:1] == (len(order),) else values for key, values in comparator.items()}


def component_mask_at(probability: np.ndarray, threshold: float) -> np.ndarray:
    labels, count = ndimage.label(probability > threshold, structure=np.ones((3, 3), dtype=np.uint8))
    if count == 0:
        return np.zeros(probability.shape, dtype=bool)
    sizes = np.bincount(labels.ravel())
    keep = sizes >= MINIMUM_CONNECTED_PIXELS
    keep[0] = False
    return keep[labels]


def gaussian_input(sample: Any, row: dict[str, Any]) -> np.ndarray:
    spectral = sample.reflectance_pair.copy()
    mbmp = compute_mbmp(spectral[:6], spectral[6:])
    height, width = mbmp.shape
    wind = np.broadcast_to(np.asarray((float(row["wind_u"]), float(row["wind_v"])), np.float32)[:, None, None] / 8.0, (2, height, width)).copy()
    cloud = (sample.cloud_classes > 0).astype(np.float32)[None]
    return np.concatenate((mbmp[None], spectral, wind, cloud)).astype(np.float32)


@torch.no_grad()
def reconstruct_dense_comparator(candidate: dict[str, np.ndarray], records: Sequence[dict[str, Any]], paths: dict[str, Path], device: torch.device, scene_comparator: dict[str, np.ndarray]) -> np.ndarray:
    """Reconstruct the frozen strength-0.1 Gaussian dense endpoint and mask rule."""
    from mars_paper_model import ReleasedMarsUNet, released_state
    from train_mars_gaussian_contrast_crossfit import TransferGaussianContrastViTUNet

    protocol = json.loads(paths["gaussian_protocol"].read_text(encoding="utf-8"))
    payload = torch.load(paths["gaussian_dense_state"], map_location="cpu", weights_only=False)
    if payload.get("protocol_sha256") != sha256(paths["gaussian_protocol"]):
        raise ValueError("Gaussian dense state protocol binding differs")
    states = payload.get("states_by_held_fold", {})
    if set(states) != {"3", "4"}:
        raise ValueError("Gaussian dense state must contain exactly held-fold 3 and 4 endpoints")
    teacher = ReleasedMarsUNet().to(device)
    incompatible = teacher.load_state_dict(released_state(paths["released_dense_checkpoint"]), strict=False)
    if incompatible.missing_keys or any(not key.startswith("out_mlp.") for key in incompatible.unexpected_keys):
        raise ValueError(f"Released dense checkpoint compatibility differs: {incompatible}")
    teacher.eval()
    record_lookup = {str(row["sample_id"]): row for row in records}
    spatial_base = np.asarray(scene_comparator["spatial_prithvi_scores"], dtype=np.float64)
    output_counts = np.zeros((len(candidate["sample_ids"]), 3), dtype=np.int64)
    for held_fold in (3, 4):
        model = TransferGaussianContrastViTUNet(**protocol["architecture"]["model"]).to(device)
        model.load_state_dict(states[str(held_fold)], strict=True)
        model.eval()
        indices = np.flatnonzero(candidate["folds"] == held_fold)
        for index in indices:
            row = record_lookup[str(candidate["sample_ids"][index])]
            sample = _load_training_sample(paths["metadata_root"], row)
            inputs = torch.as_tensor(gaussian_input(sample, row)[None], dtype=torch.float32, device=device)
            observable = torch.as_tensor(sample.observable_mask[None, None], dtype=torch.float32, device=device)
            sensor = torch.tensor([SENSOR_INDEX[sample.sensor_family]], dtype=torch.long, device=device)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                baseline_logit = teacher(inputs)
                gaussian = model(inputs, observable, sensor)
            bounded_dense = 2.0 * torch.tanh(gaussian["segmentation_logits"].float() / 2.0)
            probability = torch.sigmoid(
                baseline_logit[0, 0].float()
                + GAUSSIAN_DENSE_STRENGTH * bounded_dense[0, 0]
            ).cpu().numpy()
            probability[~sample.clear_mask] = 0.0
            prediction = component_mask_at(probability, DENSE_THRESHOLDS[int(sensor.item())])
            base = torch.tensor([spatial_base[index]], dtype=torch.float32, device=device)
            dense_scene_score = float(
                model.fuse_scene_score(
                    base,
                    gaussian["scene_logit"].float(),
                    GAUSSIAN_DENSE_STRENGTH,
                )[0].cpu()
            )
            if dense_scene_score < DENSE_SCENE_GATE:
                prediction[:] = False
            valid = sample.observable_mask & bool(row.get("pixel_truth_available", True))
            output_counts[index] = dense_counts(sample.plume_mask, prediction, valid)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return output_counts


def atomic_npz(path: Path, **values: Any) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite one-shot output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp.npz")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **values)
        stream.flush()
        os.fsync(stream.fileno())
    os.chmod(temporary, 0o444)
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def atomic_torch(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite one-shot output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("wb") as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.chmod(temporary, 0o444)
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def atomic_text(path: Path, text: str) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite one-shot output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    os.chmod(temporary, 0o444)
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def protocol_identity(protocol: dict[str, Any]) -> str:
    """Hash canonical protocol content with only the self-hash field omitted."""
    canonical = dict(protocol)
    canonical.pop("protocol_sha256_self_excluding_field", None)
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def canonical_string_set_hash(values: Iterable[str]) -> str:
    payload = "\n".join(sorted(map(str, values))) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def canonical_json_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def ordered_records_hash(records: Sequence[dict[str, Any]]) -> str:
    """Hash complete records in manifest order (ordering is intentionally significant)."""
    return canonical_json_hash(list(records))


@dataclass
class AccessLedger:
    """Machine-checked record of evidence opened during the one-shot run."""

    comparator_integrity_bytes_hashed: bool = False
    comparator_values_decoded: bool = False
    inner_validation_outcomes_opened: bool = False
    held_folds_opened: tuple[int, ...] = ()
    folds_0_1_2_opened: bool = False
    external_or_official_evidence_opened: bool = False

    def snapshot(self) -> dict[str, Any]:
        return {
            "comparator_integrity_bytes_hashed": self.comparator_integrity_bytes_hashed,
            "comparator_values_decoded": self.comparator_values_decoded,
            "inner_validation_outcomes_opened": self.inner_validation_outcomes_opened,
            "held_folds_opened": list(self.held_folds_opened),
            "folds_0_1_2_opened": self.folds_0_1_2_opened,
            "external_or_official_evidence_opened": self.external_or_official_evidence_opened,
        }

    def open_held_fold(self, fold: int) -> None:
        validate_requested_folds((fold,))
        self.held_folds_opened = tuple(sorted(set(self.held_folds_opened) | {int(fold)}))

    def assert_recovery_safe(self, current_held_fold: int | None = None) -> None:
        if (
            self.comparator_values_decoded
            or self.folds_0_1_2_opened
            or self.external_or_official_evidence_opened
        ):
            raise RuntimeError(f"Recovery access ledger is not pre-held: {self.snapshot()}")
        expected_prior = () if current_held_fold in (None, 3) else (3,)
        if self.held_folds_opened != expected_prior:
            raise RuntimeError(
                f"Recovery access ledger does not match prior immutable held parts: {self.snapshot()}"
            )


def preflight_comparator_hashes(paths: dict[str, Path], ledger: AccessLedger) -> dict[str, str]:
    """Authenticate comparator bytes without decoding any comparator value."""
    ledger.assert_recovery_safe()
    names = ("champion_scene_cache", "gaussian_dense_state", "gaussian_protocol", "released_dense_checkpoint")
    result = {name: sha256(paths[name]) for name in names}
    ledger.comparator_integrity_bytes_hashed = True
    ledger.comparator_values_decoded = False
    return result


def scientific_digest(protocol: dict[str, Any]) -> str:
    """Bind settings that can affect scientific outcomes, excluding durability mechanics."""
    keys = (
        "outer_folds", "seed", "architecture", "ordinal_targets", "inner_split", "training",
        "evaluation", "bootstrap", "gates",
    )
    settings = {key: protocol[key] for key in keys}
    settings["implemented_constants"] = {
        "seed": SEED,
        "dense_thresholds": DENSE_THRESHOLDS,
        "dense_scene_gate": DENSE_SCENE_GATE,
        "minimum_connected_pixels": MINIMUM_CONNECTED_PIXELS,
        "gaussian_dense_strength": GAUSSIAN_DENSE_STRENGTH,
    }
    return canonical_json_hash(settings)


def schedule_digest(protocol: dict[str, Any]) -> str:
    return canonical_json_hash(protocol["training"])


def checkpoint_identity(
    protocol: dict[str, Any],
    protocol_path: Path,
    paths: dict[str, Path],
    fit_records: Sequence[dict[str, Any]],
    inner_training: Sequence[dict[str, Any]],
    inner_validation: Sequence[dict[str, Any]],
    held_fold: int,
    runtime: dict[str, Any],
    ledger: AccessLedger,
) -> dict[str, Any]:
    """Construct the complete frozen identity required for exact epoch recovery."""
    ledger.assert_recovery_safe(held_fold)
    dependencies = {
        name: {
            "path": str(path),
            "protocol_sha256": protocol["dependencies"][name]["sha256"],
            "actual_sha256": "directory" if path.is_dir() else sha256(path),
        }
        for name, path in sorted(paths.items())
    }
    code_dependencies = [
        {
            "path": str((ROOT / binding["path"]).resolve()),
            "protocol_sha256": binding["sha256"],
            "actual_sha256": sha256((ROOT / binding["path"]).resolve()),
        }
        for binding in protocol.get("code_dependencies", [])
    ]
    return {
        "schema_version": RECOVERY_SCHEMA_VERSION,
        "protocol_path": str(protocol_path),
        "protocol_file_sha256": sha256(protocol_path),
        "protocol_identity": protocol_identity(protocol),
        "scientific_digest": scientific_digest(protocol),
        "trainer": {
            "path": str(Path(__file__).resolve()),
            "actual_sha256": sha256(Path(__file__).resolve()),
            "protocol_sha256": protocol["trainer"]["sha256"],
        },
        "model": {
            "path": str((ROOT / protocol["model"]["path"]).resolve()),
            "actual_sha256": sha256((ROOT / protocol["model"]["path"]).resolve()),
            "protocol_sha256": protocol["model"]["sha256"],
        },
        "dependencies": dependencies,
        "code_dependencies": code_dependencies,
        "manifest_sha256": sha256(paths["manifest"]),
        "fold_protocol_sha256": sha256(paths["fold_protocol"]),
        "held_fold": int(held_fold),
        "fit_fold": int(7 - held_fold),
        "ordered_fit_records_sha256": ordered_records_hash(fit_records),
        "ordered_inner_training_records_sha256": ordered_records_hash(inner_training),
        "ordered_inner_validation_records_sha256": ordered_records_hash(inner_validation),
        "inner_training_groups_sha256": canonical_string_set_hash({str(row["group_id"]) for row in inner_training}),
        "inner_validation_groups_sha256": canonical_string_set_hash({str(row["group_id"]) for row in inner_validation}),
        "seed": SEED,
        "schedule_digest": schedule_digest(protocol),
        "runtime_signature": runtime,
        "access_ledger": ledger.snapshot(),
    }


def capture_rng_state(batcher: SiteBalancedBatcher) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy_legacy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda_all": [value.clone() for value in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else [],
        "site_balanced_batcher_bit_generator": copy.deepcopy(batcher.rng.bit_generator.state),
    }


def _cpu_byte_rng_state(value: Any, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise RuntimeError(f"Checkpoint {name} RNG state must be a tensor")
    if value.dtype != torch.uint8:
        raise RuntimeError(f"Checkpoint {name} RNG state must have dtype uint8")
    if value.ndim != 1:
        raise RuntimeError(f"Checkpoint {name} RNG state must be one-dimensional")
    try:
        result = value.detach().to(device="cpu", dtype=torch.uint8).contiguous()
    except (RuntimeError, TypeError) as error:
        raise RuntimeError(f"Checkpoint {name} RNG state cannot be restored byte-exactly on CPU") from error
    if result.device.type != "cpu" or result.dtype != torch.uint8 or result.ndim != 1:
        raise RuntimeError(f"Checkpoint {name} RNG state normalization failed")
    return result


def restore_rng_state(state: dict[str, Any], batcher: SiteBalancedBatcher) -> None:
    """Restore all RNGs; callers must invoke this after every construction/load."""
    required = {"python", "numpy_legacy", "torch_cpu", "torch_cuda_all", "site_balanced_batcher_bit_generator"}
    if not isinstance(state, dict) or set(state) != required:
        raise RuntimeError("Checkpoint RNG state coverage mismatch")
    torch_cpu_state = _cpu_byte_rng_state(state["torch_cpu"], "torch_cpu")
    cuda_values = state["torch_cuda_all"]
    if not isinstance(cuda_values, (list, tuple)):
        raise RuntimeError("Checkpoint torch_cuda_all RNG state must be a sequence")
    cuda_states = [
        _cpu_byte_rng_state(value, f"torch_cuda_all[{index}]")
        for index, value in enumerate(cuda_values)
    ]
    if torch.cuda.is_available() and len(cuda_states) != torch.cuda.device_count():
        raise RuntimeError("Checkpoint CUDA RNG device count differs from runtime")
    random.setstate(state["python"])
    np.random.set_state(state["numpy_legacy"])
    torch.set_rng_state(torch_cpu_state)
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_states)
    batcher.rng.bit_generator.state = copy.deepcopy(state["site_balanced_batcher_bit_generator"])


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_replace_json(path: Path, value: Any) -> None:
    """Atomically replace mutable coordination metadata, with durable flushes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, sort_keys=True, separators=(",", ":"), default=str)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)


class RecoveryStore:
    """Atomic unique checkpoint generations with latest/previous validation fallback."""

    def __init__(self, root: Path, identity: dict[str, Any], device: torch.device):
        self.root = root
        self.identity = identity
        self.identity_sha256 = canonical_json_hash(identity)
        self.device = device
        self.pointer = root / "latest.json"

    def _pointer_descriptors(self) -> list[dict[str, Any]]:
        if not self.pointer.exists():
            return []
        try:
            value = json.loads(self.pointer.read_text(encoding="utf-8"))
            descriptors = value["generations"]
            if value.get("schema_version") != RECOVERY_SCHEMA_VERSION or not isinstance(descriptors, list):
                raise ValueError("invalid pointer schema")
            return descriptors[:2]
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            manifests = sorted(
                self.root.glob("generation-*.json"),
                key=lambda path: path.stat().st_mtime_ns,
                reverse=True,
            )[:2]
            result = []
            for path in manifests:
                try:
                    result.append(json.loads(path.read_text(encoding="utf-8")))
                except (OSError, json.JSONDecodeError):
                    continue
            return result

    def _validate_payload(self, payload: dict[str, Any]) -> None:
        missing = REQUIRED_RECOVERY_PAYLOAD_KEYS - set(payload)
        if missing:
            raise RuntimeError(f"Recovery checkpoint is missing required state: {sorted(missing)}")
        if payload.get("schema_version") != RECOVERY_SCHEMA_VERSION or payload.get("identity") != self.identity:
            raise RuntimeError("Recovery checkpoint identity or schema mismatch")
        if int(payload["next_epoch"]) != int(payload["completed_epoch"]) + 1:
            raise RuntimeError("Recovery checkpoint is not at a complete epoch boundary")
        rng_keys = {"python", "numpy_legacy", "torch_cpu", "torch_cuda_all", "site_balanced_batcher_bit_generator"}
        if set(payload["rng_state"]) != rng_keys:
            raise RuntimeError("Recovery checkpoint RNG state coverage mismatch")
        for name in ("pixel", "scene"):
            optimizer = payload[f"{name}_optimizer_state"]
            if optimizer.get("param_groups") != payload[f"{name}_optimizer_param_groups"]:
                raise RuntimeError(f"Recovery {name} optimizer groups mismatch")
            if [float(group["lr"]) for group in optimizer.get("param_groups", [])] != payload[f"{name}_optimizer_lrs"]:
                raise RuntimeError(f"Recovery {name} optimizer learning rates mismatch")
        access = payload["access_ledger"]
        expected_access = self.identity.get("access_ledger", {})
        if (
            access.get("comparator_values_decoded")
            or tuple(access.get("held_folds_opened", ()))
            != tuple(expected_access.get("held_folds_opened", ()))
            or access.get("folds_0_1_2_opened")
            or access.get("external_or_official_evidence_opened")
        ):
            raise RuntimeError("Recovery checkpoint contains forbidden evidence access or an access-ledger mismatch")

    def load(self) -> dict[str, Any] | None:
        errors: list[str] = []
        for descriptor in self._pointer_descriptors():
            if descriptor.get("identity_sha256") != self.identity_sha256:
                raise RuntimeError("Recovery identity/runtime/access mismatch")
            checkpoint = self.root / str(descriptor.get("checkpoint", ""))
            if not checkpoint.is_file() or sha256(checkpoint) != descriptor.get("sha256"):
                errors.append(f"invalid generation {checkpoint.name}")
                continue
            try:
                # Load the complete checkpoint on CPU. Model.load_state_dict copies live/best
                # tensors to the model device, and Optimizer.load_state_dict casts optimizer
                # tensors according to their parameter policy. Keeping serialized RNG bytes on
                # CPU prevents map_location from turning CPU RNG state into a CUDA ByteTensor.
                payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            except Exception as error:
                errors.append(f"unreadable generation {checkpoint.name}: {error}")
                continue
            self._validate_payload(payload)
            if (
                payload.get("schema_version") != RECOVERY_SCHEMA_VERSION
                or payload.get("identity") != self.identity
                or payload.get("completed_epoch") != descriptor.get("completed_epoch")
                or payload.get("next_epoch") != descriptor.get("next_epoch")
            ):
                raise RuntimeError("Recovery checkpoint identity or epoch metadata mismatch")
            expected_access = self.identity.get("access_ledger", {})
            access = payload.get("access_ledger")
            if access is not None and (
                bool(access.get("comparator_integrity_bytes_hashed"))
                != bool(expected_access.get("comparator_integrity_bytes_hashed", False))
                or access.get("comparator_values_decoded")
                or tuple(access.get("held_folds_opened", ()))
                != tuple(expected_access.get("held_folds_opened", ()))
                or access.get("folds_0_1_2_opened")
                or access.get("external_or_official_evidence_opened")
            ):
                raise RuntimeError("Recovery checkpoint access-ledger mismatch")
            return payload
        if errors:
            raise RuntimeError("No valid checkpoint in two-generation recovery window: " + "; ".join(errors))
        return None

    def save(self, payload: dict[str, Any]) -> Path:
        self._validate_payload(payload)
        self.root.mkdir(parents=True, exist_ok=True)
        generation = f"generation-{int(payload['completed_epoch']):04d}-{time.time_ns()}-{uuid.uuid4().hex}"
        checkpoint = self.root / f"{generation}.pt"
        temporary = self.root / f".{generation}.tmp"
        with temporary.open("wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, checkpoint)
        descriptor = {
            "schema_version": RECOVERY_SCHEMA_VERSION,
            "checkpoint": checkpoint.name,
            "sha256": sha256(checkpoint),
            "identity_sha256": self.identity_sha256,
            "completed_epoch": int(payload["completed_epoch"]),
            "next_epoch": int(payload["next_epoch"]),
        }
        manifest = self.root / f"{generation}.json"
        atomic_replace_json(manifest, descriptor)
        previous = self._pointer_descriptors()
        generations = [descriptor] + [item for item in previous if item.get("checkpoint") != checkpoint.name]
        atomic_replace_json(
            self.pointer,
            {"schema_version": RECOVERY_SCHEMA_VERSION, "generations": generations[:2]},
        )
        _fsync_directory(self.root)
        return checkpoint


def _verify_recovery_output_phase(protocol: dict[str, Any]) -> None:
    """Admit only an immutable prefix of the finalization state machine."""
    outputs = protocol["outputs"]
    ordered_names = ("candidate_predictions", "endpoint_states", "json", "markdown")
    ordered_paths = [(ROOT / outputs[name]).resolve() for name in ordered_names]
    present = [path.exists() for path in ordered_paths]
    if present != sorted(present, reverse=True):
        raise RuntimeError("Recovery outputs are an unrelated or partial finalization phase")
    for path, exists in zip(ordered_paths, present):
        if exists and (not path.is_file() or path.stat().st_mode & 0o222):
            raise RuntimeError(f"Recovery output is non-file or mutable: {path}")

    candidate_path = ordered_paths[0]
    expected_protocol_identity = protocol_identity(protocol)
    expected_scientific_digest = scientific_digest(protocol)
    part_states: list[str] = []
    complete_parts: list[dict[str, np.ndarray]] = []
    complete_part_paths: list[Path] = []
    complete_bindings: list[dict[str, Any]] = []
    for held_fold in (3, 4):
        part_path = candidate_path.with_name(
            f"{candidate_path.stem}.held-{held_fold}.part.json"
        )
        start_path, completion_path = _receipt_paths(part_path)
        start, part, completion = start_path.exists(), part_path.exists(), completion_path.exists()
        if part and not start or completion and not (start and part):
            raise RuntimeError("Held recovery receipts are unrelated or partial")
        for path in (start_path, part_path, completion_path):
            if path.exists() and (not path.is_file() or path.stat().st_mode & 0o222):
                raise RuntimeError(f"Held recovery artifact is non-file or mutable: {path}")
        if start:
            try:
                start_payload = json.loads(start_path.read_text(encoding="utf-8"))
                binding = start_payload["binding"]
            except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
                raise RuntimeError("Held access-start receipt is unreadable") from error
            _seal_or_validate_immutable_json(start_path, _held_access_start_receipt(binding))
            if (
                not isinstance(binding, dict)
                or int(binding.get("held_fold", -1)) != held_fold
                or binding.get("protocol_identity") != expected_protocol_identity
                or binding.get("scientific_digest") != expected_scientific_digest
            ):
                raise RuntimeError("Held access-start receipt fold binding mismatch")
            if part:
                decoded = seal_or_reuse_prediction_part(part_path, None, binding)
                if completion:
                    _seal_or_validate_immutable_json(
                        completion_path,
                        _held_completion_receipt(part_path, start_path, binding),
                    )
                    complete_parts.append(decoded)
                    complete_part_paths.append(part_path)
                    complete_bindings.append(binding)
        part_states.append("complete" if completion else "part" if part else "started" if start else "none")
    if part_states[1] != "none" and part_states[0] != "complete":
        raise RuntimeError("Held recovery parts do not follow frozen fold order")
    if present[0] and part_states != ["complete", "complete"]:
        raise RuntimeError("Final candidate exists without both complete immutable held parts")
    if present[0]:
        seal_or_validate_final_candidate(
            candidate_path,
            merge_predictions(complete_parts),
            complete_part_paths,
            complete_bindings,
        )
    state_path, json_path, markdown_path = ordered_paths[1:]
    if present[1]:
        try:
            state = torch.load(state_path, map_location="cpu", weights_only=False)
        except Exception as error:
            raise RuntimeError("Recovery endpoint-states output is unreadable") from error
        if (
            not isinstance(state, dict)
            or state.get("schema_version") != RECOVERY_SCHEMA_VERSION
            or state.get("seed") != SEED
            or state.get("protocol_identity") != expected_protocol_identity
            or state.get("scientific_digest") != expected_scientific_digest
            or state.get("candidate_sha256") != sha256(candidate_path)
            or set(state.get("states_by_held_fold", {})) != {"3", "4"}
        ):
            raise RuntimeError("Recovery endpoint-states output schema mismatch")
    report: dict[str, Any] | None = None
    if present[2]:
        try:
            report = json.loads(json_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError("Recovery JSON output is unreadable") from error
        if (
            not isinstance(report, dict)
            or report.get("protocol_identity") != expected_protocol_identity
            or report.get("scientific_digest") != expected_scientific_digest
            or report.get("candidate_predictions", {}).get("sha256") != sha256(candidate_path)
            or report.get("endpoint_states", {}).get("sha256") != sha256(state_path)
        ):
            raise RuntimeError("Recovery JSON output artifact hash mismatch")
    if present[3]:
        assert report is not None
        seal_or_validate_text_artifact(markdown_path, markdown_report(report))


def verify_protocol(protocol: dict[str, Any], protocol_path: Path, *, smoke: bool) -> dict[str, Path]:
    frozen = protocol.get("status") == "frozen_before_held_outcomes"
    if not smoke and not frozen:
        raise ValueError("Held-fold run requires a frozen protocol")
    if tuple(protocol.get("outer_folds", ())) != (3, 4):
        raise ValueError("Protocol must bind exactly outer folds [3, 4]")
    validate_requested_folds(protocol["outer_folds"])
    if protocol_identity(protocol) != protocol.get("protocol_sha256_self_excluding_field"):
        raise ValueError("Frozen protocol self-hash mismatch")
    paths = {name: (ROOT / binding["path"]).resolve() for name, binding in protocol["dependencies"].items()}
    if frozen:
        bindings = [("trainer", protocol["trainer"]), ("model", protocol["model"]),
                    *[(f"code_dependency_{index}", binding) for index, binding in enumerate(protocol.get("code_dependencies", []))]]
        allowed_data_dependencies = {"manifest", "fold_protocol"} if smoke else set(protocol["dependencies"])
        bindings.extend(
            (name, binding)
            for name, binding in protocol["dependencies"].items()
            if name in allowed_data_dependencies and binding["sha256"] != "directory"
        )
        for name, binding in bindings:
            path = (ROOT / binding["path"]).resolve()
            if not path.is_file() or sha256(path) != binding["sha256"]:
                raise ValueError(f"Frozen dependency hash mismatch: {name}")
    required_paths = {"metadata_root", "manifest", "fold_protocol"} if smoke else set(paths)
    for name, path in paths.items():
        if name not in required_paths:
            continue
        if not path.exists():
            raise FileNotFoundError(f"Missing dependency: {name}")
    if not smoke:
        _verify_recovery_output_phase(protocol)
    return paths


def smoke(protocol: dict[str, Any], paths: dict[str, Path]) -> dict[str, Any]:
    """One pixel and one isolated-scene update using fold-3 fitting rows only."""
    groups = fold_lookup(paths["fold_protocol"])
    records = records_for_folds(paths["manifest"], groups, [3])
    positives = [row for row in records if row["label_state"] == "PLUME" and bool(row.get("pixel_truth_available", True))]
    negatives = [row for row in records if row["label_state"] == "NO_PLUME"]
    if not positives or not negatives:
        raise ValueError("Fold-3 smoke needs pixel-truth positive and negative rows")
    positive_sample = _load_training_sample(paths["metadata_root"], positives[0])
    enhancement = positive_sample.methane_enhancement_raw
    supported = positive_sample.plume_mask & positive_sample.observable_mask & np.isfinite(enhancement)
    cutpoints = np.quantile(enhancement[supported], (0.25, 0.5, 0.75), method="linear")
    negative_sample = _load_training_sample(paths["metadata_root"], negatives[0])
    rng = np.random.default_rng(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch = collate([make_crop(positive_sample, cutpoints, size=256, rng=rng, augment=True), make_crop(negative_sample, cutpoints, size=256, rng=rng, augment=True)], device)
    model = MarsSensorOrdinalUNet().to(device)
    pixel_optimizer = torch.optim.AdamW(model.pixel_parameters(), lr=3e-4, betas=(0.9, 0.999), weight_decay=1e-4)
    scene_optimizer = torch.optim.AdamW(model.scene_parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=1e-4)
    result = finite_gradient_step(model, batch, pixel_optimizer, scene_optimizer)
    result.update({"ok": all(math.isfinite(value) for value in result.values()), "scope": "fold_3_fitting_data_only", "rows": 2,
                   "sample_ids": batch["sample_id"], "input_shape": list(batch["inputs"].shape),
                   "model": model.artifact_metadata(), "device": str(device), "held_fold_outcome_opened": False,
                   "folds_0_1_2_opened": False})
    if torch.cuda.is_available():
        result["peak_cuda_bytes"] = torch.cuda.max_memory_allocated()
    return result


def _nested_exact_equal(left: Any, right: Any) -> bool:
    if torch.is_tensor(left) or torch.is_tensor(right):
        return torch.is_tensor(left) and torch.is_tensor(right) and torch.equal(left.cpu(), right.cpu())
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return isinstance(left, np.ndarray) and isinstance(right, np.ndarray) and np.array_equal(left, right)
    if isinstance(left, dict) or isinstance(right, dict):
        return isinstance(left, dict) and isinstance(right, dict) and left.keys() == right.keys() and all(
            _nested_exact_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return type(left) is type(right) and len(left) == len(right) and all(
            _nested_exact_equal(a, b) for a, b in zip(left, right)
        )
    return bool(left == right)


def checkpoint_roundtrip_smoke(protocol: dict[str, Any], protocol_path: Path, paths: dict[str, Path]) -> dict[str, Any]:
    """Attest exact next-batch/update recovery using fitting evidence only."""
    if not torch.cuda.is_available():
        raise RuntimeError("--checkpoint-roundtrip-smoke requires CUDA")
    groups = fold_lookup(paths["fold_protocol"])
    fit_records = records_for_folds(paths["manifest"], groups, [3])
    inner_training, inner_validation, training_groups, validation_groups, cutpoints = prepare_endpoint_data(
        paths["metadata_root"], fit_records
    )
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    batcher = SiteBalancedBatcher(
        paths["metadata_root"], inner_training, cutpoints, np.random.default_rng(SEED)
    )
    first_batch = batcher.dense_batch(device, int(protocol["training"]["crop_size"]))
    expected_batch = int(protocol["training"]["dense_batch_size"])
    if first_batch["inputs"].shape[0] != expected_batch:
        raise RuntimeError("Roundtrip smoke did not form the frozen dense batch size")
    if len(set(first_batch["group_id"])) != expected_batch:
        raise RuntimeError("Roundtrip smoke dense batch repeated a canonical group")
    if int(first_batch["presence"].sum().item()) != expected_batch // 2:
        raise RuntimeError("Roundtrip smoke dense batch does not have the frozen 8/8 balance")
    model = MarsSensorOrdinalUNet().to(device).train()
    optimizer_spec = protocol["training"]["pixel_optimizer"]
    optimizer = torch.optim.AdamW(
        model.pixel_parameters(),
        lr=pixel_learning_rate(1, int(protocol["training"]["epochs"])),
        betas=(float(optimizer_spec["betas"][0]), float(optimizer_spec["betas"][1])),
        weight_decay=float(optimizer_spec["weight_decay"]),
    )
    scene_optimizer_spec = protocol["training"]["scene_optimizer"]
    scene_optimizer = torch.optim.AdamW(
        model.scene_parameters(),
        lr=0.0,
        betas=(float(scene_optimizer_spec["betas"][0]), float(scene_optimizer_spec["betas"][1])),
        weight_decay=float(scene_optimizer_spec["weight_decay"]),
    )
    first_result = pixel_step(model, first_batch, optimizer)
    first_ids = list(first_batch["sample_id"])
    first_groups = list(first_batch["group_id"])
    checkpoint_rng = capture_rng_state(batcher)
    smoke_identity = {
        "mode": "checkpoint_roundtrip_smoke",
        "protocol_sha256": sha256(protocol_path),
        "protocol_identity": protocol_identity(protocol),
        "scientific_digest": scientific_digest(protocol),
        "runtime_signature": runtime_signature(),
        "fit_fold": 3,
        "held_fold": 4,
        "ordered_fit_records_sha256": ordered_records_hash(fit_records),
        "ordered_inner_training_records_sha256": ordered_records_hash(inner_training),
        "seed": SEED,
        "schedule_digest": schedule_digest(protocol),
        "access_ledger": AccessLedger(comparator_integrity_bytes_hashed=False).snapshot(),
    }
    checkpoint_payload = {
        "schema_version": RECOVERY_SCHEMA_VERSION,
        "identity": smoke_identity,
        "live_model_state": _cpu_state(model),
        "best_model_state": _cpu_state(model),
        "pixel_optimizer_state": optimizer.state_dict(),
        "scene_optimizer_state": scene_optimizer.state_dict(),
        "pixel_optimizer_param_groups": copy.deepcopy(optimizer.state_dict()["param_groups"]),
        "scene_optimizer_param_groups": copy.deepcopy(scene_optimizer.state_dict()["param_groups"]),
        "pixel_optimizer_lrs": [float(group["lr"]) for group in optimizer.param_groups],
        "scene_optimizer_lrs": [float(group["lr"]) for group in scene_optimizer.param_groups],
        "best_rank": [0.0, 0.0, 0.0],
        "best_epoch": 0,
        "history": [],
        "cutpoints": cutpoints.copy(),
        "rng_state": checkpoint_rng,
        "completed_epoch": 0,
        "next_epoch": 1,
        "held_fold": 4,
        "fit_fold": 3,
        "access_ledger": smoke_identity["access_ledger"],
    }
    with tempfile.TemporaryDirectory(prefix="mars-ordinal-roundtrip-") as temporary_directory:
        store = RecoveryStore(Path(temporary_directory), smoke_identity, device)
        store.save(checkpoint_payload)
        next_expected = batcher.dense_batch(device, int(protocol["training"]["crop_size"]))
        expected_ids = list(next_expected["sample_id"])
        expected_groups = list(next_expected["group_id"])
        expected_result = pixel_step(model, next_expected, optimizer)
        expected_model_state = _cpu_state(model)
        expected_optimizer_state = copy.deepcopy(optimizer.state_dict())
        del model, optimizer, scene_optimizer, first_batch, next_expected
        torch.cuda.empty_cache()
        recovered_model = MarsSensorOrdinalUNet().to(device).train()
        recovered_optimizer = torch.optim.AdamW(
            recovered_model.pixel_parameters(),
            lr=0.0,
            betas=(float(optimizer_spec["betas"][0]), float(optimizer_spec["betas"][1])),
            weight_decay=float(optimizer_spec["weight_decay"]),
        )
        recovered_scene_optimizer = torch.optim.AdamW(
            recovered_model.scene_parameters(),
            lr=0.0,
            betas=(float(scene_optimizer_spec["betas"][0]), float(scene_optimizer_spec["betas"][1])),
            weight_decay=float(scene_optimizer_spec["weight_decay"]),
        )
        recovered_batcher = SiteBalancedBatcher(
            paths["metadata_root"], inner_training, cutpoints, np.random.default_rng(SEED + 1)
        )
        recovered = store.load()
        if recovered is None:
            raise RuntimeError("Roundtrip smoke checkpoint disappeared")
        recovered_model.load_state_dict(recovered["live_model_state"], strict=True)
        recovered_optimizer.load_state_dict(recovered["pixel_optimizer_state"])
        recovered_scene_optimizer.load_state_dict(recovered["scene_optimizer_state"])
        # Restore RNG last, after model/optimizer/batcher construction and state loading.
        restore_rng_state(recovered["rng_state"], recovered_batcher)
        next_recovered = recovered_batcher.dense_batch(device, int(protocol["training"]["crop_size"]))
        identities_equal = (
            expected_ids == list(next_recovered["sample_id"])
            and expected_groups == list(next_recovered["group_id"])
        )
        recovered_result = pixel_step(recovered_model, next_recovered, recovered_optimizer)
        next_step_equal = (
            identities_equal
            and expected_result == recovered_result
            and _nested_exact_equal(expected_model_state, _cpu_state(recovered_model))
            and _nested_exact_equal(expected_optimizer_state, recovered_optimizer.state_dict())
        )
    result: dict[str, Any] = dict(first_result)
    result.update({
        "ok": all(math.isfinite(value) for value in first_result.values()) and next_step_equal,
        "mode": "checkpoint_roundtrip_smoke",
        "scope": "fold_3_fitting_data_only",
        "endpoint_role": "fit_fold_3_for_held_fold_4",
        "protocol": str(protocol_path.relative_to(ROOT)),
        "protocol_sha256": sha256(protocol_path),
        "protocol_identity": protocol_identity(protocol),
        "runtime_environment": dict(REQUIRED_RUNTIME_ENV),
        "runtime_signature": runtime_signature(),
        "seed": SEED,
        "rows": expected_batch,
        "distinct_groups": len(set(first_groups)),
        "first_sample_ids_sha256": canonical_json_hash(first_ids),
        "first_group_ids_sha256": canonical_json_hash(first_groups),
        "next_sample_ids_sha256": canonical_json_hash(expected_ids),
        "next_group_ids_sha256": canonical_json_hash(expected_groups),
        "next_sample_and_group_identities_equal": identities_equal,
        "next_step_model_optimizer_exactly_equal": next_step_equal,
        "input_shape": [expected_batch, 14, int(protocol["training"]["crop_size"]), int(protocol["training"]["crop_size"])],
        "pixel_learning_rate": pixel_learning_rate(1, int(protocol["training"]["epochs"])),
        "inner_training_rows": len(inner_training),
        "inner_training_groups": len(training_groups),
        "inner_training_groups_sha256": canonical_string_set_hash(training_groups),
        "inner_validation_rows": len(inner_validation),
        "inner_validation_groups": len(validation_groups),
        "cutpoints": cutpoints.tolist(),
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "inner_validation_outcome_opened": False,
        "held_outcome_opened": False,
        "comparator_integrity_bytes_hashed": False,
        "comparator_values_decoded": False,
        "folds_0_1_2_opened": False,
        "external_or_official_evidence_opened": False,
    })
    return result


def runtime_smoke(protocol: dict[str, Any], protocol_path: Path, paths: dict[str, Path]) -> dict[str, Any]:
    """Backward-compatible name for the frozen checkpoint-roundtrip smoke."""
    return checkpoint_roundtrip_smoke(protocol, protocol_path, paths)


def prediction_part_payload(values: dict[str, np.ndarray], binding: dict[str, Any]) -> dict[str, Any]:
    arrays = {
        key: {"dtype": str(np.asarray(value).dtype), "shape": list(np.asarray(value).shape), "values": np.asarray(value).tolist()}
        for key, value in values.items()
    }
    return {"schema_version": RECOVERY_SCHEMA_VERSION, "binding": binding, "arrays": arrays}


def decode_prediction_part(payload: dict[str, Any], expected_binding: dict[str, Any]) -> dict[str, np.ndarray]:
    if payload.get("schema_version") != RECOVERY_SCHEMA_VERSION or payload.get("binding") != expected_binding:
        raise RuntimeError("Held prediction part identity mismatch")
    result: dict[str, np.ndarray] = {}
    for key, encoded in payload.get("arrays", {}).items():
        value = np.asarray(encoded["values"], dtype=np.dtype(encoded["dtype"]))
        if list(value.shape) != encoded["shape"]:
            raise RuntimeError(f"Held prediction part shape mismatch: {key}")
        result[key] = value
    required = {"sample_ids", "labels", "sensors", "groups", "folds", "scores", "dense_counts"}
    if set(result) != required:
        raise RuntimeError("Held prediction part array schema mismatch")
    size = len(result["sample_ids"])
    if any(len(value) != size for value in result.values()) or not np.all(result["folds"] == expected_binding["held_fold"]):
        raise RuntimeError("Held prediction part row/fold mismatch")
    if canonical_json_hash(result["sample_ids"].astype(str).tolist()) != expected_binding["ordered_sample_ids_sha256"]:
        raise RuntimeError("Held prediction part sample order mismatch")
    return result


def seal_or_reuse_prediction_part(
    path: Path,
    values: dict[str, np.ndarray] | None,
    binding: dict[str, Any],
) -> dict[str, np.ndarray]:
    """Seal a JSON-held part, or reuse it only after full identity/schema validation."""
    if path.exists():
        if path.stat().st_mode & 0o222:
            raise RuntimeError(f"Existing held prediction part is not read-only: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        return decode_prediction_part(payload, binding)
    if values is None:
        raise FileNotFoundError(path)
    payload = prediction_part_payload(values, binding)
    atomic_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return decode_prediction_part(payload, binding)


def _receipt_paths(part_path: Path) -> tuple[Path, Path]:
    return (
        part_path.with_name(f"{part_path.stem}.access-start.json"),
        part_path.with_name(f"{part_path.stem}.completion.json"),
    )


def _seal_or_validate_immutable_json(path: Path, expected: dict[str, Any]) -> None:
    """Create an immutable receipt, or require the existing receipt to be exact."""
    if path.exists():
        if not path.is_file() or path.stat().st_mode & 0o222:
            raise RuntimeError(f"Recovery receipt is missing, non-file, or mutable: {path}")
        try:
            actual = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(f"Recovery receipt is unreadable: {path}") from error
        if actual != expected:
            raise RuntimeError(f"Recovery receipt binding mismatch: {path}")
        return
    atomic_text(path, json.dumps(expected, indent=2, sort_keys=True) + "\n")


def _held_access_start_receipt(binding: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": RECOVERY_SCHEMA_VERSION,
        "kind": "held_access_start",
        "binding": binding,
    }


def _held_completion_receipt(
    part_path: Path,
    start_path: Path,
    binding: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": RECOVERY_SCHEMA_VERSION,
        "kind": "held_prediction_part_complete",
        "binding_sha256": canonical_json_hash(binding),
        "start_receipt_sha256": sha256(start_path),
        "part_sha256": sha256(part_path),
    }


def evaluate_or_reuse_prediction_part(
    path: Path,
    binding: dict[str, Any],
    evaluate: Callable[[], dict[str, np.ndarray]],
) -> tuple[dict[str, np.ndarray], bool]:
    """Durably authorize held access and never repeat an uncertain evaluation.

    The start receipt is fsynced before the evaluator is entered.  Therefore a
    start receipt without a sealed part is an irrecoverable one-shot boundary,
    not permission to try the evaluator again.  A crash after the part seal but
    before its completion receipt is recoverable without reopening held data.
    """
    start_path, completion_path = _receipt_paths(path)
    start_exists, part_exists, completion_exists = (
        start_path.exists(), path.exists(), completion_path.exists()
    )
    if completion_exists and not (start_exists and part_exists):
        raise RuntimeError("Held completion receipt exists without its start receipt and part")
    if part_exists and not start_exists:
        raise RuntimeError("Held prediction part exists without a durable access-start receipt")
    if start_exists:
        _seal_or_validate_immutable_json(start_path, _held_access_start_receipt(binding))
        if not part_exists:
            raise RuntimeError(
                "Held access already started without a sealed prediction part; refusing repeat evaluation"
            )
        part = seal_or_reuse_prediction_part(path, None, binding)
        completion = _held_completion_receipt(path, start_path, binding)
        _seal_or_validate_immutable_json(completion_path, completion)
        return part, True

    if completion_exists:
        raise RuntimeError("Held completion receipt exists without a durable access-start receipt")
    _seal_or_validate_immutable_json(start_path, _held_access_start_receipt(binding))
    values = evaluate()
    part = seal_or_reuse_prediction_part(path, values, binding)
    _seal_or_validate_immutable_json(
        completion_path, _held_completion_receipt(path, start_path, binding)
    )
    return part, False


def authorize_comparator_decode(
    candidate_path: Path,
    part_paths: Sequence[Path],
    ledger: AccessLedger,
) -> None:
    """Open semantic comparator access only at the frozen immutable boundary."""
    if len(part_paths) != 2 or len({path.resolve() for path in part_paths}) != 2:
        raise RuntimeError("Comparator decode requires exactly two distinct held parts")
    for path in (*part_paths, candidate_path):
        if not path.is_file() or path.stat().st_mode & 0o222:
            raise RuntimeError(f"Comparator decode blocked by mutable or missing candidate artifact: {path}")
    if set(ledger.held_folds_opened) != {3, 4}:
        raise RuntimeError("Comparator decode requires both held folds to be durably represented")
    if not ledger.comparator_integrity_bytes_hashed or ledger.comparator_values_decoded:
        raise RuntimeError("Comparator access ledger is not at the frozen pre-decode boundary")
    ledger.comparator_values_decoded = True


def merge_predictions(parts: Sequence[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    result = {key: np.concatenate([part[key] for part in parts], axis=0) for key in parts[0]}
    if len(set(result["sample_ids"].tolist())) != len(result["sample_ids"]):
        raise ValueError("Candidate sample identities are not unique")
    return result


def _candidate_recovery_binding(
    part_paths: Sequence[Path], bindings: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    if len(part_paths) != 2 or len(bindings) != 2:
        raise RuntimeError("Final candidate requires exactly two held parts and bindings")
    parts: list[dict[str, Any]] = []
    for path, binding in zip(part_paths, bindings):
        start_path, completion_path = _receipt_paths(path)
        if any(
            not item.is_file() or item.stat().st_mode & 0o222
            for item in (start_path, path, completion_path)
        ):
            raise RuntimeError("Final candidate requires complete immutable held-part receipts")
        _seal_or_validate_immutable_json(start_path, _held_access_start_receipt(binding))
        _seal_or_validate_immutable_json(
            completion_path, _held_completion_receipt(path, start_path, binding)
        )
        parts.append({
            "held_fold": int(binding["held_fold"]),
            "binding_sha256": canonical_json_hash(binding),
            "part_sha256": sha256(path),
            "start_receipt_sha256": sha256(start_path),
            "completion_receipt_sha256": sha256(completion_path),
        })
    if [part["held_fold"] for part in parts] != [3, 4]:
        raise RuntimeError("Final candidate held-part order must be exactly [3, 4]")
    return {"schema_version": RECOVERY_SCHEMA_VERSION, "held_parts": parts}


def seal_or_validate_final_candidate(
    path: Path,
    candidate: dict[str, np.ndarray],
    part_paths: Sequence[Path],
    bindings: Sequence[dict[str, Any]],
) -> bool:
    """Seal the merged candidate once, or validate exact recovery equivalence."""
    recovery_binding = _candidate_recovery_binding(part_paths, bindings)
    encoded_binding = json.dumps(
        recovery_binding, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    expected_keys = {"schema_version", "recovery_binding_json", *candidate.keys()}
    if path.exists():
        if not path.is_file() or path.stat().st_mode & 0o222:
            raise RuntimeError("Existing final candidate is missing, non-file, or mutable")
        try:
            with np.load(path, allow_pickle=False) as source:
                if set(source.files) != expected_keys:
                    raise RuntimeError("Existing final candidate array schema mismatch")
                if int(np.asarray(source["schema_version"]).item()) != RECOVERY_SCHEMA_VERSION:
                    raise RuntimeError("Existing final candidate schema mismatch")
                if str(np.asarray(source["recovery_binding_json"]).item()) != encoded_binding:
                    raise RuntimeError("Existing final candidate held-part binding/hash mismatch")
                for key, expected in candidate.items():
                    expected_array = np.asarray(expected)
                    if source[key].dtype != expected_array.dtype or not np.array_equal(source[key], expected_array):
                        raise RuntimeError(f"Existing final candidate array mismatch: {key}")
        except (OSError, ValueError) as error:
            raise RuntimeError("Existing final candidate is unreadable") from error
        return True
    atomic_npz(
        path,
        schema_version=np.asarray(RECOVERY_SCHEMA_VERSION, dtype=np.uint8),
        recovery_binding_json=np.asarray(encoded_binding),
        **candidate,
    )
    return False


def seal_or_validate_torch_artifact(path: Path, value: Any) -> bool:
    """Seal a deterministic torch output, or require exact recovery contents."""
    if path.exists():
        if not path.is_file() or path.stat().st_mode & 0o222:
            raise RuntimeError(f"Existing torch recovery output is missing or mutable: {path}")
        try:
            actual = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as error:
            raise RuntimeError(f"Existing torch recovery output is unreadable: {path}") from error
        if not _nested_exact_equal(actual, value):
            raise RuntimeError(f"Existing torch recovery output contents mismatch: {path}")
        return True
    atomic_torch(path, value)
    return False


def seal_or_validate_text_artifact(path: Path, text: str) -> bool:
    """Seal JSON/Markdown once; recovery may only reuse exactly consistent output."""
    if path.exists():
        if not path.is_file() or path.stat().st_mode & 0o222:
            raise RuntimeError(f"Existing text recovery output is missing or mutable: {path}")
        if path.read_text(encoding="utf-8") != text:
            raise RuntimeError(f"Existing text recovery output contents mismatch: {path}")
        return True
    atomic_text(path, text)
    return False


def markdown_report(report: dict[str, Any]) -> str:
    metrics = report["metrics"]
    lines = ["# MARS sensor-aware ordinal folds 3/4", "",
             f"- Promotion gates pass: **{metrics['passed']}**",
             f"- Pooled AP delta: **{metrics['pooled_ap_delta']:+.6f}**",
             f"- AP bootstrap lower bound: **{metrics['ap_bootstrap_lower']:+.6f}**",
             f"- Matched-FPR recall delta: **{metrics['matched_fpr_recall_delta']:+.6f}**",
             f"- Dense IoU delta: **{metrics['dense_iou_delta']:+.6f}**",
             f"- Dense bootstrap lower bound: **{metrics['dense_bootstrap_lower']:+.6f}**", "",
             "## Endpoint checkpoints", ""]
    for endpoint in report["endpoints"]:
        lines.append(f"- Held fold {endpoint['held_fold']}: epoch {endpoint['selected_epoch']}, inner-training cutpoints {endpoint['cutpoints']}")
    lines.extend(["", "## Gate checks", ""])
    lines.extend(f"- `{name}`: **{value}**" for name, value in metrics["checks"].items())
    lines.extend(["", "Enhancement values are used only as producer-supplied ordinal ordering; no physical-unit claim is made.", ""])
    return "\n".join(lines)


def run_full(
    protocol: dict[str, Any],
    protocol_path: Path,
    paths: dict[str, Path],
    *,
    runtime: dict[str, Any] | None = None,
) -> dict[str, Any]:
    groups = fold_lookup(paths["fold_protocol"])
    all_records = records_for_folds(paths["manifest"], groups, [3, 4])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runtime = runtime or runtime_signature()
    ledger = AccessLedger()
    comparator_preflight = preflight_comparator_hashes(paths, ledger)
    endpoint_metadata: list[dict[str, Any]] = []
    endpoint_states: dict[str, Any] = {}
    candidate_path = (ROOT / protocol["outputs"]["candidate_predictions"]).resolve()
    prediction_parts: list[dict[str, np.ndarray]] = []
    part_paths: list[Path] = []
    part_bindings: list[dict[str, Any]] = []
    part_receipts: dict[str, Any] = {}
    # Preserve the frozen endpoint->held->endpoint->held execution order. On
    # restart, train_endpoint restores its complete boundary, and constructing
    # the held model on both paths preserves the exact global RNG progression.
    for held_fold in (3, 4):
        fit_records = [row for row in all_records if groups[str(row["group_id"])] == 7 - held_fold]
        held_records = [row for row in all_records if groups[str(row["group_id"])] == held_fold]
        metadata, state, cutpoints = train_endpoint(
            protocol,
            paths,
            fit_records,
            held_fold,
            device,
            protocol_path=protocol_path,
            runtime=runtime,
            ledger=ledger,
        )
        ledger.inner_validation_outcomes_opened = True
        endpoint_metadata.append(metadata)
        endpoint_states[str(held_fold)] = {
            "state_dict": state,
            "selected_epoch": metadata["selected_epoch"],
            "cutpoints": cutpoints,
            "recovery_identity_sha256": metadata["recovery_identity_sha256"],
        }
        binding = {
            "protocol_identity": protocol_identity(protocol),
            "scientific_digest": scientific_digest(protocol),
            "held_fold": held_fold,
            "fit_fold": 7 - held_fold,
            "endpoint_recovery_identity_sha256": metadata["recovery_identity_sha256"],
            "ordered_held_records_sha256": ordered_records_hash(held_records),
            "ordered_sample_ids_sha256": canonical_json_hash([str(row["sample_id"]) for row in held_records]),
            "access_before_open": {
                "comparator_integrity_bytes_hashed": True,
                "comparator_values_decoded": False,
                "previous_immutable_held_folds": list(ledger.held_folds_opened),
                "folds_0_1_2_opened": False,
                "external_or_official_evidence_opened": False,
            },
        }
        part_path = candidate_path.with_name(f"{candidate_path.stem}.held-{held_fold}.part.json")
        # Construction and state loading are intentionally performed even when
        # reusing a part: they consume the same global RNG as the original path,
        # while the held evaluation itself is deterministic and is not repeated.
        model = MarsSensorOrdinalUNet().to(device)
        model.load_state_dict(state, strict=True)
        def evaluate_once() -> dict[str, np.ndarray]:
            ledger.open_held_fold(held_fold)
            return evaluate_candidate(
                model, paths["metadata_root"], held_records, cutpoints, device, fold=held_fold
            )
        part, _reused = evaluate_or_reuse_prediction_part(part_path, binding, evaluate_once)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        ledger.open_held_fold(held_fold)
        prediction_parts.append(part)
        part_paths.append(part_path)
        part_bindings.append(binding)
        start_path, completion_path = _receipt_paths(part_path)
        part_receipts[str(held_fold)] = {
            "path": str(part_path.relative_to(ROOT)),
            "sha256": sha256(part_path),
            "immutable_mode": "0444",
            "access_start_receipt": {
                "path": str(start_path.relative_to(ROOT)), "sha256": sha256(start_path)
            },
            "completion_receipt": {
                "path": str(completion_path.relative_to(ROOT)), "sha256": sha256(completion_path)
            },
        }
    candidate = merge_predictions(prediction_parts)
    seal_or_validate_final_candidate(candidate_path, candidate, part_paths, part_bindings)
    candidate_hash = sha256(candidate_path)
    if candidate_path.stat().st_mode & 0o222:
        raise RuntimeError("Final candidate artifact is not immutable")
    # Only now, after both immutable parts have been merged and the final candidate
    # is immutable, may comparator containers be semantically decoded.
    authorize_comparator_decode(candidate_path, part_paths, ledger)
    scene_comparator = align_comparator(candidate, paths["champion_scene_cache"])
    comparator_dense = reconstruct_dense_comparator(candidate, all_records, paths, device, scene_comparator)
    bootstrap = protocol["bootstrap"]
    metrics = metric_gates(candidate["labels"], candidate["scores"], scene_comparator["champion_scores"],
                           candidate["folds"], candidate["sensors"], candidate["groups"],
                           candidate["dense_counts"], comparator_dense,
                           replicates=int(bootstrap["replicates"]), ap_seed=int(bootstrap["ap_seed"]), dense_seed=int(bootstrap["dense_seed"]))
    state_path = (ROOT / protocol["outputs"]["endpoint_states"]).resolve()
    state_value = {
        "schema_version": RECOVERY_SCHEMA_VERSION,
        "protocol_identity": protocol_identity(protocol),
        "scientific_digest": scientific_digest(protocol),
        "candidate_sha256": candidate_hash,
        "seed": SEED,
        "states_by_held_fold": endpoint_states,
    }
    seal_or_validate_torch_artifact(state_path, state_value)
    report = {"schema_version": 1, "protocol": str(protocol_path.relative_to(ROOT)), "protocol_sha256": sha256(protocol_path),
              "protocol_identity": protocol_identity(protocol), "scientific_digest": scientific_digest(protocol),
              "seed": SEED, "scope": "development folds 3/4 only", "endpoints": endpoint_metadata,
              "candidate_predictions": {"path": str(candidate_path.relative_to(ROOT)), "sha256": candidate_hash, "immutable_mode": "0444",
                                        "held_parts": part_receipts},
              "endpoint_states": {"path": str(state_path.relative_to(ROOT)), "sha256": sha256(state_path), "exactly_one_checkpoint_per_endpoint": True},
              "comparators": {"preflight_raw_sha256": comparator_preflight,
                              "scene": {"path": str(paths["champion_scene_cache"].relative_to(ROOT)), "sha256": sha256(paths["champion_scene_cache"])},
                              "dense": {"state_path": str(paths["gaussian_dense_state"].relative_to(ROOT)), "state_sha256": sha256(paths["gaussian_dense_state"]),
                                        "strength": GAUSSIAN_DENSE_STRENGTH, "thresholds": list(DENSE_THRESHOLDS), "scene_gate": DENSE_SCENE_GATE,
                                        "minimum_connected_pixels": MINIMUM_CONNECTED_PIXELS}},
              "access_ledger": ledger.snapshot(),
              "metrics": metrics, "decision": "stop_for_codex_review" if metrics["passed"] else "stop_no_second_seed"}
    json_path = (ROOT / protocol["outputs"]["json"]).resolve()
    markdown_path = (ROOT / protocol["outputs"]["markdown"]).resolve()
    seal_or_validate_text_artifact(json_path, json.dumps(report, indent=2, sort_keys=True) + "\n")
    seal_or_validate_text_artifact(markdown_path, markdown_report(report))
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL.as_posix())
    parser.add_argument("--smoke", action="store_true", help="Authorized fold-3 fitting-data gradient smoke only")
    parser.add_argument("--runtime-smoke", action="store_true", help="Alias for the native-Windows checkpoint roundtrip smoke")
    parser.add_argument("--checkpoint-roundtrip-smoke", action="store_true", help="Native-Windows fitting-only production batch-16 recovery attestation")
    parser.add_argument("--run-held-folds", action="store_true", help="Explicit one-shot held-fold authorization")
    args = parser.parse_args(argv)
    if sum(map(int, (args.smoke, args.runtime_smoke, args.checkpoint_roundtrip_smoke, args.run_held_folds))) > 1:
        parser.error("--smoke, --runtime-smoke, --checkpoint-roundtrip-smoke, and --run-held-folds are mutually exclusive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    protocol_path = (ROOT / args.protocol).resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if not args.smoke and not args.runtime_smoke and not args.checkpoint_roundtrip_smoke and not args.run_held_folds:
        raise RuntimeError("Refusing held outcomes without explicit --run-held-folds authorization")
    runtime: dict[str, Any] | None = None
    if args.runtime_smoke or args.checkpoint_roundtrip_smoke or args.run_held_folds:
        runtime = verify_runtime_environment(require_native_windows=True)
    paths = verify_protocol(
        protocol,
        protocol_path,
        smoke=args.smoke or args.runtime_smoke or args.checkpoint_roundtrip_smoke,
    )
    seed_everything()
    if args.smoke:
        result = smoke(protocol, paths)
        print(json.dumps(result, sort_keys=True))
        return 0 if result["ok"] else 1
    if args.runtime_smoke or args.checkpoint_roundtrip_smoke:
        result = checkpoint_roundtrip_smoke(protocol, protocol_path, paths)
        print(json.dumps(result, sort_keys=True))
        return 0 if result["ok"] else 1
    report = run_full(protocol, protocol_path, paths, runtime=runtime)
    print(json.dumps({"passed": report["metrics"]["passed"], "decision": report["decision"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
