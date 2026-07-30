#!/usr/bin/env python3
"""Train an identity-safe dense Prithvi/U-Net fusion pilot on development fold 2."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "EarthRemoteSensingRapidResponse"
for path in (MODEL_ROOT, ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from acquire_mars_metadata import sha256  # noqa: E402
from analyze_mars_mask_routing import paired_group_bootstrap as pixel_bootstrap  # noqa: E402
from analyze_mars_mask_thresholds import component_mask_at  # noqa: E402
from mars_dense_prithvi_teacher import (  # noqa: E402
    DensePrithviTeacherAdapter,
    PRITHVI_CHANNELS,
    PRITHVI_GRID_SIZE,
)
from mars_paper_model import SENSOR_NAMES, released_state  # noqa: E402
from train_mars_oof_scene_ensemble_v2 import ap_group_bootstrap  # noqa: E402
from train_mars_paper_residual import (  # noqa: E402
    MarsPaperDataset,
    iter_development_manifest,
    move_batch,
    verify_acquisition_receipt,
)
from train_mars_physics_guided_teacher_balanced_pilot import (  # noqa: E402
    balanced_request_weights,
)
from train_mars_physics_guided_teacher_pilot import seed_everything  # noqa: E402
from train_mars_scene_ranker import comparison, metric_summary  # noqa: E402


DEFAULT_PROTOCOL = Path("configs/mars_dense_prithvi_teacher_pilot_protocol.json")
MINIMUM_CONNECTED_PIXELS = 100
MASK_THRESHOLDS = (0.8, 0.7)
SCENE_GATE = 0.75


def load_feature_contract(
    features_path: Path,
    metadata_path: Path,
    scores_path: Path,
    expected_features_sha256: str,
) -> tuple[np.ndarray, dict[str, int], np.ndarray, dict[str, Any]]:
    features = np.load(features_path, mmap_mode="r", allow_pickle=False)
    with np.load(metadata_path, allow_pickle=False) as values:
        metadata = {name: values[name] for name in values.files}
    if features.shape != (
        metadata["sample_ids"].size,
        PRITHVI_CHANNELS,
        PRITHVI_GRID_SIZE,
        PRITHVI_GRID_SIZE,
    ):
        raise ValueError("Dense Prithvi feature cache geometry differs from the frozen schema")
    if str(metadata["features_sha256"].item()) != expected_features_sha256:
        raise ValueError("Dense Prithvi feature cache differs from its metadata receipt")
    sample_ids = metadata["sample_ids"].astype(str)
    if np.unique(sample_ids).size != sample_ids.size:
        raise ValueError("Dense Prithvi metadata contains duplicate sample identifiers")
    row_by_id = {sample_id: index for index, sample_id in enumerate(sample_ids)}

    base_scores = np.full(sample_ids.size, np.nan, dtype=np.float64)
    with np.load(scores_path, allow_pickle=False) as values:
        for prefix, rows in (
            ("fold0", metadata["folds"] == 0),
            ("fold1", metadata["folds"] == 1),
            ("inner", metadata["folds"] >= 2),
        ):
            if not np.array_equal(metadata["labels"][rows], values[f"{prefix}_labels"]):
                raise ValueError(f"{prefix} score labels do not align to Prithvi rows")
            if not np.array_equal(metadata["sensors"][rows], values[f"{prefix}_sensors"]):
                raise ValueError(f"{prefix} score sensors do not align to Prithvi rows")
            if not np.array_equal(metadata["groups"][rows].astype(str), values[f"{prefix}_groups"].astype(str)):
                raise ValueError(f"{prefix} score groups do not align to Prithvi rows")
            if prefix == "inner" and not np.array_equal(
                metadata["folds"][rows], values["inner_folds"]
            ):
                raise ValueError("Inner score folds do not align to Prithvi rows")
            base_scores[rows] = values[f"{prefix}_new"]
    if not np.isfinite(base_scores).all() or np.any((base_scores < 0) | (base_scores > 1)):
        raise ValueError("Cross-fitted base scores are incomplete or outside [0,1]")
    identity = {
        "rows": int(sample_ids.size),
        "sample_id_sha256": hashlib.sha256("\n".join(sample_ids).encode()).hexdigest(),
        "feature_shape": list(features.shape),
        "feature_dtype": str(features.dtype),
        "fold_counts": {
            str(int(fold)): int(np.count_nonzero(metadata["folds"] == fold))
            for fold in np.unique(metadata["folds"])
        },
        "foundation_checkpoint_sha256": str(metadata["checkpoint_sha256"].item()),
        "manifest_sha256": str(metadata["manifest_sha256"].item()),
        "fold_protocol_sha256": str(metadata["protocol_sha256"].item()),
    }
    return features, row_by_id, base_scores, identity


class DensePrithviDataset(MarsPaperDataset):
    """Attach aligned frozen token maps and joint geometric augmentation."""

    def __init__(
        self,
        metadata_dir: Path,
        records: list[dict[str, Any]],
        *,
        features: np.ndarray,
        row_by_id: dict[str, int],
        base_scores: np.ndarray,
        augment: bool,
        seed: int,
    ) -> None:
        super().__init__(metadata_dir, records, augment=False, seed=seed)
        self.features = features
        self.row_by_id = row_by_id
        self.base_scores = base_scores
        self.joint_augment = augment
        missing = [str(row["sample_id"]) for row in records if str(row["sample_id"]) not in row_by_id]
        if missing:
            raise ValueError(f"Dense Prithvi cache lacks {len(missing)} requested samples")

    @staticmethod
    def _transform(
        values: torch.Tensor, turns: int, horizontal: bool, vertical: bool
    ) -> torch.Tensor:
        if turns:
            values = torch.rot90(values, turns, dims=(-2, -1))
        if horizontal:
            values = torch.flip(values, dims=(-1,))
        if vertical:
            values = torch.flip(values, dims=(-2,))
        return values.contiguous()

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = super().__getitem__(index)
        row = self.row_by_id[str(item["sample_id"])]
        tokens = torch.from_numpy(
            np.array(self.features[row], dtype=np.float32, copy=True)
        )
        if self.joint_augment:
            rng = self.rng()
            turns = int(rng.integers(0, 4))
            horizontal = bool(rng.integers(0, 2))
            vertical = bool(rng.integers(0, 2))
            wind = (
                float(item["inputs"][13, 0, 0]) * 8.0,
                float(item["inputs"][14, 0, 0]) * 8.0,
            )
            for _ in range(turns):
                wind = (-wind[1], wind[0])
            if horizontal:
                wind = (-wind[0], wind[1])
            if vertical:
                wind = (wind[0], -wind[1])
            for name in ("inputs", "observable", "clear", "mask"):
                item[name] = self._transform(
                    item[name], turns, horizontal, vertical
                )
            tokens = self._transform(tokens, turns, horizontal, vertical)
            item["inputs"][13].fill_(wind[0] / 8.0)
            item["inputs"][14].fill_(wind[1] / 8.0)
        item["prithvi_tokens"] = tokens
        item["base_scene_score"] = torch.tensor(
            float(self.base_scores[row]), dtype=torch.float32
        )
        return item


def weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return (values * weights).sum() / weights.sum().clamp_min(1e-6)


def fusion_loss(
    output: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    spec: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    logits = output["segmentation_logits"]
    target = batch["mask"]
    observable = batch["observable"]
    pixel_truth = batch["pixel_truth_available"].bool()
    supervised = (batch["presence"] < 0.5) | pixel_truth
    row_weights = torch.where(
        supervised, torch.ones_like(batch["presence"]), torch.zeros_like(batch["presence"])
    )

    positive_weight = torch.tensor(
        float(spec["positive_pixel_weight"]), device=logits.device, dtype=logits.dtype
    )
    bce = F.binary_cross_entropy_with_logits(
        logits, target, reduction="none", pos_weight=positive_weight
    )
    probability = torch.sigmoid(logits)
    pt = torch.where(target > 0.5, probability, 1.0 - probability)
    focal_pixels = ((1.0 - pt) ** float(spec["focal_gamma"])) * bce * observable
    focal_rows = focal_pixels.flatten(1).sum(dim=1)
    focal_rows /= observable.flatten(1).sum(dim=1).clamp_min(1.0)
    focal = weighted_mean(focal_rows, row_weights)

    flat_probability = (probability * observable).flatten(1)
    flat_target = (target * observable).flatten(1)
    positive_rows = (flat_target.sum(dim=1) > 0) & supervised
    if bool(positive_rows.any()):
        intersection = (flat_probability[positive_rows] * flat_target[positive_rows]).sum(dim=1)
        denominator = (
            flat_probability[positive_rows].sum(dim=1)
            + flat_target[positive_rows].sum(dim=1)
        )
        dice = (
            1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)
        ).mean()
    else:
        dice = logits.sum() * 0.0

    patch_target = F.adaptive_max_pool2d(target, (PRITHVI_GRID_SIZE, PRITHVI_GRID_SIZE))
    patch_visible = (
        F.adaptive_avg_pool2d(observable, (PRITHVI_GRID_SIZE, PRITHVI_GRID_SIZE))
        >= 0.9
    ).to(logits.dtype)
    patch_valid = patch_visible * supervised[:, None, None, None].to(logits.dtype)
    patch_bce = F.binary_cross_entropy_with_logits(
        output["patch_logits"],
        patch_target,
        reduction="none",
        pos_weight=torch.tensor(
            float(spec["patch_positive_weight"]),
            device=logits.device,
            dtype=logits.dtype,
        ),
    )
    patch_rows = (patch_bce * patch_valid).flatten(1).sum(dim=1)
    patch_rows /= patch_valid.flatten(1).sum(dim=1).clamp_min(1.0)
    patch = weighted_mean(patch_rows, row_weights)

    scene = F.binary_cross_entropy_with_logits(
        output["scene_logit"], batch["presence"]
    )
    positive = output["scene_logit"][batch["presence"] > 0.5]
    negative = output["scene_logit"][batch["presence"] < 0.5]
    pair = (
        F.softplus(
            float(spec["pair_margin"]) - positive[:, None] + negative[None, :]
        ).mean()
        if positive.numel() and negative.numel()
        else logits.sum() * 0.0
    )

    negative_pixels = (
        (batch["presence"] < 0.5)[:, None, None, None] & (observable > 0.5)
    )
    positive_pixels = (
        supervised[:, None, None, None]
        & (target > 0.5)
        & (observable > 0.5)
    )
    upward = F.relu(logits - output["baseline_logits"].detach())
    downward = F.relu(output["baseline_logits"].detach() - logits)
    upward_penalty = (
        (upward * negative_pixels).sum() / negative_pixels.sum().clamp_min(1)
    )
    downward_penalty = (
        (downward * positive_pixels).sum() / positive_pixels.sum().clamp_min(1)
    )
    correction = output["correction_logits"].square().mean()
    scene_delta = output["scene_delta_logit"].square().mean()
    total = (
        focal
        + float(spec["dice_weight"]) * dice
        + float(spec["patch_weight"]) * patch
        + float(spec["scene_weight"]) * scene
        + float(spec["pair_weight"]) * pair
        + float(spec["negative_upward_weight"]) * upward_penalty
        + float(spec["positive_downward_weight"]) * downward_penalty
        + float(spec["correction_l2_weight"]) * correction
        + float(spec["scene_delta_l2_weight"]) * scene_delta
    )
    return total, {
        "loss": float(total.detach()),
        "focal": float(focal.detach()),
        "dice": float(dice.detach()),
        "patch": float(patch.detach()),
        "scene": float(scene.detach()),
        "pair": float(pair.detach()),
        "negative_upward": float(upward_penalty.detach()),
        "positive_downward": float(downward_penalty.detach()),
        "correction_l2": float(correction.detach()),
        "scene_delta_l2": float(scene_delta.detach()),
    }


def train_endpoint(
    model: DensePrithviTeacherAdapter,
    loader: DataLoader[dict[str, Any]],
    spec: dict[str, Any],
    device: torch.device,
    epochs: int,
) -> list[dict[str, float]]:
    seed_everything(int(spec["seed"]))
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(spec["learning_rate"]),
        weight_decay=float(spec["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda")
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        sums: dict[str, float] = {}
        batches = 0
        for batch in loader:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                output = model(
                    batch["inputs"],
                    batch["observable"],
                    batch["sensor_index"],
                    batch["prithvi_tokens"],
                    batch["base_scene_score"],
                )
                loss, parts = fusion_loss(output, batch, spec)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(parameters, float(spec["gradient_clip"]))
            scaler.step(optimizer)
            scaler.update()
            batches += 1
            for key, value in parts.items():
                sums[key] = sums.get(key, 0.0) + value
        scheduler.step()
        row = {"epoch": epoch, **{key: value / batches for key, value in sums.items()}}
        history.append(row)
        print(json.dumps(row), flush=True)
    return history


def pixel_counts(
    prediction: np.ndarray, truth: np.ndarray, observable: np.ndarray
) -> np.ndarray:
    return np.asarray(
        (
            np.count_nonzero(prediction & truth),
            np.count_nonzero(prediction & observable & ~truth),
            np.count_nonzero(truth & ~prediction),
        ),
        dtype=np.int64,
    )


def pixel_summary(counts: np.ndarray) -> dict[str, float | int]:
    total = counts.sum(axis=0, dtype=np.int64)
    tp, fp, fn = (int(value) for value in total)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "intersection_over_union": tp / max(tp + fp + fn, 1),
        "recall": tp / max(tp + fn, 1),
        "precision": tp / max(tp + fp, 1),
    }


@torch.no_grad()
def evaluate(
    model: DensePrithviTeacherAdapter,
    loader: DataLoader[dict[str, Any]],
    strengths: list[float],
    device: torch.device,
    bootstrap: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    model.eval()
    labels: list[int] = []
    sensors: list[int] = []
    groups: list[str] = []
    sample_ids: list[str] = []
    base_scores: list[float] = []
    corrections: list[float] = []
    baseline_pixels: list[np.ndarray] = []
    candidate_pixels: dict[float, list[np.ndarray]] = {value: [] for value in strengths}
    for batch in loader:
        local_ids = [str(value) for value in batch["sample_id"]]
        local_groups = [str(value) for value in batch["group_id"]]
        batch = move_batch(batch, device)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            output = model(
                batch["inputs"],
                batch["observable"],
                batch["sensor_index"],
                batch["prithvi_tokens"],
                batch["base_scene_score"],
            )
        baseline_probability = torch.sigmoid(output["baseline_logits"]).float()
        base_logit = output["base_scene_logit"].float()
        scene_delta = output["scene_delta_logit"].float()
        pixel_delta = output["correction_logits"].float()
        for index in range(baseline_probability.shape[0]):
            sensor = int(batch["sensor_index"][index])
            threshold = MASK_THRESHOLDS[sensor]
            observable = batch["observable"][index, 0].cpu().numpy() > 0.5
            clear = batch["clear"][index, 0].cpu().numpy() > 0.5
            truth = (batch["mask"][index, 0].cpu().numpy() > 0.5) & observable
            base_score = float(batch["base_scene_score"][index])
            base_map = baseline_probability[index, 0].cpu().numpy()
            base_map[~clear] = 0.0
            base_prediction = component_mask_at(
                base_map, threshold, MINIMUM_CONNECTED_PIXELS
            )
            if base_score < SCENE_GATE:
                base_prediction[:] = False
            baseline_pixels.append(pixel_counts(base_prediction, truth, observable))
            for strength in strengths:
                probability = torch.sigmoid(
                    output["baseline_logits"][index, 0].float()
                    + float(strength) * pixel_delta[index, 0]
                ).cpu().numpy()
                probability[~clear] = 0.0
                score = float(
                    torch.sigmoid(
                        base_logit[index] + float(strength) * scene_delta[index]
                    )
                )
                prediction = component_mask_at(
                    probability, threshold, MINIMUM_CONNECTED_PIXELS
                )
                if score < SCENE_GATE:
                    prediction[:] = False
                candidate_pixels[strength].append(
                    pixel_counts(prediction, truth, observable)
                )
        labels.extend(int(value) for value in batch["presence"].cpu().numpy())
        sensors.extend(int(value) for value in batch["sensor_index"].cpu().numpy())
        groups.extend(local_groups)
        sample_ids.extend(local_ids)
        base_scores.extend(float(value) for value in batch["base_scene_score"].cpu().numpy())
        corrections.extend(float(value) for value in scene_delta.cpu().numpy())

    y = np.asarray(labels, dtype=np.uint8)
    sensor_array = np.asarray(sensors, dtype=np.uint8)
    group_array = np.asarray(groups)
    base = np.asarray(base_scores, dtype=np.float64)
    delta = np.asarray(corrections, dtype=np.float64)
    base_logits = np.log(np.clip(base, 1e-6, 1 - 1e-6) / np.clip(1 - base, 1e-6, 1))
    base_metrics = metric_summary(y, base, sensor_array)
    base_pixel_array = np.asarray(baseline_pixels, dtype=np.int64)
    base_pixel_summary = pixel_summary(base_pixel_array)
    candidates: list[dict[str, Any]] = []
    for strength in strengths:
        scores = 1.0 / (1.0 + np.exp(-(base_logits + strength * delta)))
        metrics = metric_summary(y, scores, sensor_array)
        versus = comparison(metrics, base_metrics)
        ap_interval = ap_group_bootstrap(
            y,
            base,
            scores,
            group_array,
            replicates=int(bootstrap["replicates"]),
            seed=int(bootstrap["seed"]),
        )
        local_pixels = np.asarray(candidate_pixels[strength], dtype=np.int64)
        pixel_metrics = pixel_summary(local_pixels)
        iou_delta = float(
            pixel_metrics["intersection_over_union"]
            - base_pixel_summary["intersection_over_union"]
        )
        pixel_interval = pixel_bootstrap(
            base_pixel_array,
            local_pixels,
            group_array,
            replicates=int(bootstrap["replicates"]),
            seed=int(bootstrap["seed"]) + 1,
            confidence=float(bootstrap["confidence"]),
        )
        sensor_delta = versus["delta"]["sensor_average_precision"]
        passed = bool(
            versus["delta"]["average_precision"] >= 0.001
            and versus["delta"]["recall_at_fpr_0_0713"] >= 0.0
            and min(sensor_delta.values()) >= 0.0
            and ap_interval["lower"] > 0.0
            and iou_delta > 0.0
        )
        candidates.append(
            {
                "strength": strength,
                "metrics": metrics,
                "versus_current": versus,
                "paired_site_ap_delta": ap_interval,
                "current_pixel_rule": base_pixel_summary,
                "candidate_pixel_rule": pixel_metrics,
                "pixel_iou_delta": iou_delta,
                "paired_site_pixel_iou_delta": pixel_interval,
                "passed": passed,
                "rank": [
                    int(passed),
                    min(sensor_delta.values()),
                    ap_interval["lower"],
                    versus["delta"]["average_precision"],
                    versus["delta"]["recall_at_fpr_0_0713"],
                    iou_delta,
                    -strength,
                ],
            }
        )
    identity = {
        "rows": len(y),
        "sample_id_sha256": hashlib.sha256("\n".join(sample_ids).encode()).hexdigest(),
        "current_metrics": base_metrics,
        "current_pixel_rule": base_pixel_summary,
    }
    return candidates, identity


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    selected = report["selected"]
    scene = selected["versus_current"]["delta"]
    ap_ci = selected["paired_site_ap_delta"]
    pixel_ci = selected["paired_site_pixel_iou_delta"]
    lines = [
        "# Dense Prithvi / released-U-Net fusion pilot",
        "",
        "Development fold 2 only; no exact-paper or fresh external inputs were loaded.",
        "",
        f"- Selected residual strength: {selected['strength']:.2f}",
        f"- AP delta versus current cross-fitted ranker: {scene['average_precision']:+.6f}",
        f"- Matched-FPR recall delta: {scene['recall_at_fpr_0_0713']:+.6f}",
        f"- Paired-site AP interval: [{ap_ci['lower']:+.6f}, {ap_ci['upper']:+.6f}]",
        f"- Dense-mask IoU delta versus current rule: {selected['pixel_iou_delta']:+.6f}",
        f"- Paired-site IoU interval: [{pixel_ci['lower']:+.6f}, {pixel_ci['upper']:+.6f}]",
        "",
        report["decision"],
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL.as_posix())
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    protocol_path = (ROOT / args.protocol).resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if sha256(Path(__file__).resolve()) != protocol["trainer"]["sha256"]:
        raise ValueError("Frozen dense-fusion trainer hash mismatch")
    for dependency in protocol["code_dependencies"]:
        path = (ROOT / dependency["path"]).resolve()
        if sha256(path) != dependency["sha256"]:
            raise ValueError(f"Frozen dependency mismatch: {dependency['path']}")
    paths: dict[str, Path] = {}
    for name, contract in protocol["inputs"].items():
        path = (ROOT / contract["path"]).resolve()
        if path.is_file() and sha256(path) != contract["sha256"]:
            raise ValueError(f"Frozen input mismatch: {name}")
        paths[name] = path
    if not paths["metadata_root"].is_dir():
        raise ValueError("MARS metadata root is unavailable")
    verify_acquisition_receipt(paths["acquisition_receipt"], sha256(paths["manifest"]))
    features, row_by_id, base_scores, feature_identity = load_feature_contract(
        paths["features"],
        paths["feature_metadata"],
        paths["score_cache"],
        str(protocol["inputs"]["features"]["sha256"]),
    )
    fold_protocol = json.loads(paths["fold_protocol"].read_text(encoding="utf-8"))
    group_to_fold = {
        str(row["group_id"]): int(row["fold"]) for row in fold_protocol["assignments"]
    }
    records = list(iter_development_manifest(paths["manifest"]))
    fit_folds = set(map(int, protocol["folds"]["fit"]))
    held_fold = int(protocol["folds"]["held"])
    fit_records = [
        row for row in records if group_to_fold[str(row["group_id"])] in fit_folds
    ]
    held_records = [
        row for row in records if group_to_fold[str(row["group_id"])] == held_fold
    ]
    spec = protocol["training"]
    if args.smoke:
        selected: list[dict[str, Any]] = []
        counts: Counter[tuple[str, str]] = Counter()
        for row in fit_records:
            key = (str(row["label_state"]), str(row["sensor_family"]))
            if counts[key] < 4:
                selected.append(row)
                counts[key] += 1
            if len(counts) == 4 and min(counts.values()) >= 4:
                break
        fit_records = selected
        held_records = selected
    weights, request_mass = balanced_request_weights(fit_records)
    positive_mass = sum(
        value for key, value in request_mass.items() if key.startswith("PLUME|")
    )
    if not np.isclose(positive_mass, 0.5, atol=1e-12):
        raise ValueError("Balanced sampler does not assign 0.5 request mass to plumes")
    train_dataset = DensePrithviDataset(
        paths["metadata_root"],
        fit_records,
        features=features,
        row_by_id=row_by_id,
        base_scores=base_scores,
        augment=True,
        seed=int(spec["seed"]),
    )
    evaluation_dataset = DensePrithviDataset(
        paths["metadata_root"],
        held_records,
        features=features,
        row_by_id=row_by_id,
        base_scores=base_scores,
        augment=False,
        seed=int(spec["seed"]),
    )
    samples = 32 if args.smoke else int(spec["samples_per_epoch"])
    workers = 0 if args.smoke else int(spec["loader_workers"])
    # Exercise the exact full-run batch geometry during the smoke so memory
    # feasibility is established before the protocol is frozen.
    batch_size = int(spec["batch_size"])
    sampler = WeightedRandomSampler(
        weights,
        num_samples=samples,
        replacement=True,
        generator=torch.Generator().manual_seed(int(spec["seed"])),
    )
    options = {
        "batch_size": batch_size,
        "num_workers": workers,
        "pin_memory": True,
        "persistent_workers": workers > 0,
    }
    train_loader = DataLoader(train_dataset, sampler=sampler, **options)
    evaluation_loader = DataLoader(
        evaluation_dataset, shuffle=False, **options
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Dense Prithvi fusion pilot requires CUDA")
    model = DensePrithviTeacherAdapter().to(device)
    model.load_released_checkpoint(released_state(paths["released_checkpoint"]))
    with torch.no_grad():
        first = move_batch(next(iter(evaluation_loader)), device)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            initial = model(
                first["inputs"],
                first["observable"],
                first["sensor_index"],
                first["prithvi_tokens"],
                first["base_scene_score"],
            )
        identity_pixel = float(initial["correction_logits"].abs().max())
        identity_scene = float(initial["scene_delta_logit"].abs().max())
    if identity_pixel != 0.0 or identity_scene != 0.0:
        raise ValueError(
            f"Fusion initialization is not exact identity: pixel={identity_pixel}, scene={identity_scene}"
        )
    history = train_endpoint(
        model,
        train_loader,
        spec,
        device,
        1 if args.smoke else int(spec["epochs"]),
    )
    if args.smoke:
        finite = all(
            torch.isfinite(value).all() for value in model.trainable_state().values()
        )
        print(
            json.dumps(
                {
                    "ok": finite,
                    "identity_pixel_max_abs": identity_pixel,
                    "identity_scene_max_abs": identity_scene,
                    "request_mass": request_mass,
                    "feature_identity": feature_identity,
                    "trainable_parameters": model.trainable_parameter_count(),
                    "history": history,
                }
            )
        )
        return 0 if finite else 1

    candidates, evaluation_identity = evaluate(
        model,
        evaluation_loader,
        [float(value) for value in protocol["search"]["strengths"]],
        device,
        protocol["bootstrap"],
    )
    selected = max(candidates, key=lambda row: tuple(row["rank"]))
    passed = bool(selected["passed"])
    artifact = None
    if passed:
        artifact_path = (ROOT / protocol["outputs"]["artifact"]).resolve()
        if artifact_path.exists():
            raise FileExistsError(f"Refusing to overwrite existing artifact: {artifact_path}")
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "schema_version": 1,
                "adapter_state": model.trainable_state(),
                "selected_strength": selected["strength"],
                "protocol_sha256": sha256(protocol_path),
                "feature_identity": feature_identity,
            },
            artifact_path,
        )
        artifact = {
            "path": artifact_path.relative_to(ROOT).as_posix(),
            "bytes": artifact_path.stat().st_size,
            "sha256": sha256(artifact_path),
        }
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    report = {
        "schema_version": 1,
        "scope": "dense frozen-Prithvi token fusion into the released MARS-S2L U-Net",
        "all_promotion_gates_pass": passed,
        "decision": (
            "Authorize preregistered multi-seed full-development cross-fit."
            if passed
            else "Reject dense token fusion pilot before external scoring."
        ),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "provenance": {
            "git_commit": commit,
            "protocol_sha256": sha256(protocol_path),
            "trainer_sha256": sha256(Path(__file__).resolve()),
            "device": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
        "fit_folds": sorted(fit_folds),
        "held_fold": held_fold,
        "fit_rows": len(fit_records),
        "held_rows": len(held_records),
        "request_mass": request_mass,
        "feature_identity": feature_identity,
        "trainable_parameters": model.trainable_parameter_count(),
        "identity_pixel_max_abs": identity_pixel,
        "identity_scene_max_abs": identity_scene,
        "history": history,
        "evaluation_identity": evaluation_identity,
        "candidates": candidates,
        "selected": selected,
        "artifact": artifact,
        "external_inputs_accessed": False,
    }
    output_json = (ROOT / protocol["outputs"]["json"]).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_json.with_suffix(output_json.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, output_json)
    write_markdown((ROOT / protocol["outputs"]["markdown"]).resolve(), report)
    print(
        json.dumps(
            {
                "ok": passed,
                "strength": selected["strength"],
                "ap_delta": selected["versus_current"]["delta"]["average_precision"],
                "ap_lower": selected["paired_site_ap_delta"]["lower"],
                "recall_delta": selected["versus_current"]["delta"][
                    "recall_at_fpr_0_0713"
                ],
                "iou_delta": selected["pixel_iou_delta"],
                "iou_lower": selected["paired_site_pixel_iou_delta"]["lower"],
            },
            indent=2,
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
