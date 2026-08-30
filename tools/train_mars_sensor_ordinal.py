#!/usr/bin/env python3
"""Train/evaluate the preregistered standalone MARS sensor-aware ordinal pilot."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

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
    reflectance = np.clip(np.asarray(sample.reflectance_pair, np.float32), 0.0, 1.5)
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
        by_group: dict[str, list[dict[str, Any]]] = {}
        for row in self.records:
            by_group.setdefault(str(row["group_id"]), []).append(row)
        for group, rows in by_group.items():
            positive_rows = [row for row in rows if str(row["label_state"]) == "PLUME"]
            label = int(bool(positive_rows))
            self.by_label[label][group] = positive_rows if label else rows
        if not self.by_label[0] or not self.by_label[1]:
            raise ValueError("Site-balanced training requires positive and negative groups")
        if set(self.by_label[0]) & set(self.by_label[1]):
            raise ValueError("A site cannot enter both label pools")

    def rows(self, positive: int, negative: int) -> list[dict[str, Any]]:
        selected: list[dict[str, Any]] = []
        for label, count in ((1, positive), (0, negative)):
            groups = sorted(self.by_label[label])
            if len(groups) < count:
                raise ValueError(
                    f"Need {count} distinct label-{label} sites, found {len(groups)}"
                )
            chosen = self.rng.choice(groups, size=count, replace=False)
            for group in chosen:
                options = self.by_label[label][str(group)]
                selected.append(options[int(self.rng.integers(len(options)))])
        order = self.rng.permutation(len(selected))
        return [selected[int(index)] for index in order]

    def dense_batch(self, device: torch.device, size: int = 256) -> dict[str, Any]:
        crops: list[dict[str, Any]] = []
        for label, count in ((1, 8), (0, 8)):
            groups = np.asarray(sorted(self.by_label[label]))
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
                    crops.append(crop)
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
    gradient = torch.nn.utils.clip_grad_norm_(model.pixel_parameters(), 2.0)
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
    gradient = torch.nn.utils.clip_grad_norm_(model.scene_parameters(), 2.0)
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


def train_endpoint(protocol: dict[str, Any], paths: dict[str, Path], fit_records: Sequence[dict[str, Any]], held_fold: int, device: torch.device) -> tuple[dict[str, Any], dict[str, torch.Tensor], np.ndarray]:
    training_groups, validation_groups = deterministic_inner_split(fit_records)
    inner_training = [row for row in fit_records if str(row["group_id"]) in training_groups]
    inner_validation = [row for row in fit_records if str(row["group_id"]) in validation_groups]
    cutpoints = fit_ordinal_cutpoints(paths["metadata_root"], fit_records)
    model = MarsSensorOrdinalUNet().to(device)
    spec = protocol["training"]
    pixel_optimizer = torch.optim.AdamW(model.pixel_parameters(), lr=0.0, betas=(0.9, 0.999), weight_decay=1e-4)
    scene_optimizer = torch.optim.AdamW(model.scene_parameters(), lr=0.0, betas=(0.9, 0.999), weight_decay=1e-4)
    batcher = SiteBalancedBatcher(paths["metadata_root"], inner_training, cutpoints, np.random.default_rng(SEED))
    best_state: dict[str, torch.Tensor] | None = None
    best_rank: tuple[float, float, float] | None = None
    best_epoch = 0
    history: list[dict[str, Any]] = []
    epochs = int(spec["epochs"])
    for epoch in range(1, epochs + 1):
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
        metrics = checkpoint_metrics(validation)
        rank = checkpoint_rank(metrics)
        if best_rank is None or rank > best_rank:
            best_rank, best_state, best_epoch = rank, _cpu_state(model), epoch
        row = {"epoch": epoch, "pixel_lr": pixel_lr, "scene_lr": scene_lr,
               "pixel_loss_mean": float(np.mean(pixel_losses)),
               "scene_loss_mean": None if not scene_losses else float(np.mean(scene_losses)), **metrics,
               "selected_so_far": best_epoch}
        history.append(row)
        print(json.dumps({"progress": "endpoint_epoch", "held_fold": held_fold, **row}), flush=True)
    if best_state is None:
        raise RuntimeError("Checkpoint selection produced no endpoint")
    model.load_state_dict(best_state, strict=True)
    metadata = {
        "held_fold": held_fold, "fit_fold": 7 - held_fold,
        "inner_training_groups": len(training_groups), "inner_validation_groups": len(validation_groups),
        "inner_training_rows": len(inner_training), "inner_validation_rows": len(inner_validation),
        "cutpoints": cutpoints.tolist(), "cutpoint_source": "outer_fitting_fold_only",
        "selected_epoch": best_epoch, "selected_rank": list(best_rank), "history": history,
    }
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
    temporary = path.with_suffix(path.suffix + ".tmp.npz")
    np.savez_compressed(temporary, **values)
    os.replace(temporary, path)


def atomic_torch(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite one-shot output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def atomic_text(path: Path, text: str) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite one-shot output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def verify_protocol(protocol: dict[str, Any], protocol_path: Path, *, smoke: bool) -> dict[str, Path]:
    frozen = protocol.get("status") == "frozen_before_held_outcomes"
    if not smoke and not frozen:
        raise ValueError("Held-fold run requires a frozen protocol")
    if tuple(protocol.get("outer_folds", ())) != (3, 4):
        raise ValueError("Protocol must bind exactly outer folds [3, 4]")
    validate_requested_folds(protocol["outer_folds"])
    paths = {name: (ROOT / binding["path"]).resolve() for name, binding in protocol["dependencies"].items()}
    if frozen:
        bindings = [("trainer", protocol["trainer"]), ("model", protocol["model"]),
                    *[(f"code_dependency_{index}", binding) for index, binding in enumerate(protocol.get("code_dependencies", []))],
                    *[(name, binding) for name, binding in protocol["dependencies"].items() if binding["sha256"] != "directory"]]
        for name, binding in bindings:
            path = (ROOT / binding["path"]).resolve()
            if not path.is_file() or sha256(path) != binding["sha256"]:
                raise ValueError(f"Frozen dependency hash mismatch: {name}")
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing dependency: {name}")
    if not smoke:
        for path in ((ROOT / value).resolve() for value in protocol["outputs"].values()):
            if path.exists():
                raise FileExistsError(f"Refusing to overwrite one-shot output: {path}")
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


def merge_predictions(parts: Sequence[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    result = {key: np.concatenate([part[key] for part in parts], axis=0) for key in parts[0]}
    if len(set(result["sample_ids"].tolist())) != len(result["sample_ids"]):
        raise ValueError("Candidate sample identities are not unique")
    return result


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


def run_full(protocol: dict[str, Any], protocol_path: Path, paths: dict[str, Path]) -> dict[str, Any]:
    groups = fold_lookup(paths["fold_protocol"])
    all_records = records_for_folds(paths["manifest"], groups, [3, 4])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    endpoint_metadata: list[dict[str, Any]] = []
    endpoint_states: dict[str, Any] = {}
    prediction_parts: list[dict[str, np.ndarray]] = []
    for held_fold in (3, 4):
        fit_records = [row for row in all_records if groups[str(row["group_id"])] == 7 - held_fold]
        held_records = [row for row in all_records if groups[str(row["group_id"])] == held_fold]
        metadata, state, cutpoints = train_endpoint(protocol, paths, fit_records, held_fold, device)
        endpoint_metadata.append(metadata)
        endpoint_states[str(held_fold)] = {"state_dict": state, "selected_epoch": metadata["selected_epoch"], "cutpoints": cutpoints}
        model = MarsSensorOrdinalUNet().to(device)
        model.load_state_dict(state, strict=True)
        prediction_parts.append(evaluate_candidate(model, paths["metadata_root"], held_records, cutpoints, device, fold=held_fold))
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    candidate = merge_predictions(prediction_parts)
    candidate_path = (ROOT / protocol["outputs"]["candidate_predictions"]).resolve()
    atomic_npz(candidate_path, schema_version=np.uint8(1), **candidate)
    os.chmod(candidate_path, 0o444)
    candidate_hash = sha256(candidate_path)
    # Comparator evidence is opened/aligned only after candidate predictions are immutable.
    scene_comparator = align_comparator(candidate, paths["champion_scene_cache"])
    comparator_dense = reconstruct_dense_comparator(candidate, all_records, paths, device, scene_comparator)
    bootstrap = protocol["bootstrap"]
    metrics = metric_gates(candidate["labels"], candidate["scores"], scene_comparator["champion_scores"],
                           candidate["folds"], candidate["sensors"], candidate["groups"],
                           candidate["dense_counts"], comparator_dense,
                           replicates=int(bootstrap["replicates"]), ap_seed=int(bootstrap["ap_seed"]), dense_seed=int(bootstrap["dense_seed"]))
    state_path = (ROOT / protocol["outputs"]["endpoint_states"]).resolve()
    atomic_torch(state_path, {"schema_version": 1, "seed": SEED, "states_by_held_fold": endpoint_states})
    report = {"schema_version": 1, "protocol": str(protocol_path.relative_to(ROOT)), "protocol_sha256": sha256(protocol_path),
              "seed": SEED, "scope": "development folds 3/4 only", "endpoints": endpoint_metadata,
              "candidate_predictions": {"path": str(candidate_path.relative_to(ROOT)), "sha256": candidate_hash, "immutable_mode": "0444"},
              "endpoint_states": {"path": str(state_path.relative_to(ROOT)), "sha256": sha256(state_path), "exactly_one_checkpoint_per_endpoint": True},
              "comparators": {"scene": {"path": str(paths["champion_scene_cache"].relative_to(ROOT)), "sha256": sha256(paths["champion_scene_cache"])},
                              "dense": {"state_path": str(paths["gaussian_dense_state"].relative_to(ROOT)), "state_sha256": sha256(paths["gaussian_dense_state"]),
                                        "strength": GAUSSIAN_DENSE_STRENGTH, "thresholds": list(DENSE_THRESHOLDS), "scene_gate": DENSE_SCENE_GATE,
                                        "minimum_connected_pixels": MINIMUM_CONNECTED_PIXELS}},
              "metrics": metrics, "decision": "stop_for_codex_review" if metrics["passed"] else "stop_no_second_seed"}
    json_path = (ROOT / protocol["outputs"]["json"]).resolve()
    markdown_path = (ROOT / protocol["outputs"]["markdown"]).resolve()
    atomic_text(json_path, json.dumps(report, indent=2, sort_keys=True) + "\n")
    atomic_text(markdown_path, markdown_report(report))
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL.as_posix())
    parser.add_argument("--smoke", action="store_true", help="Authorized fold-3 fitting-data gradient smoke only")
    parser.add_argument("--run-held-folds", action="store_true", help="Explicit one-shot held-fold authorization")
    args = parser.parse_args(argv)
    if args.smoke and args.run_held_folds:
        parser.error("--smoke and --run-held-folds are mutually exclusive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    protocol_path = (ROOT / args.protocol).resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if not args.smoke and not args.run_held_folds:
        raise RuntimeError("Refusing held outcomes without explicit --run-held-folds authorization")
    paths = verify_protocol(protocol, protocol_path, smoke=args.smoke)
    seed_everything()
    if args.smoke:
        result = smoke(protocol, paths)
        print(json.dumps(result, sort_keys=True))
        return 0 if result["ok"] else 1
    report = run_full(protocol, protocol_path, paths)
    print(json.dumps({"passed": report["metrics"]["passed"], "decision": report["decision"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
