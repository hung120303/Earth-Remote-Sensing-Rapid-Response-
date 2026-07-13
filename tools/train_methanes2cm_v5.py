#!/usr/bin/env python3
"""Train ERSRR v5 on the sealed MethaneS2CM fitting/development protocol."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import sklearn
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, get_worker_info

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "EarthRemoteSensingRapidResponse"
if str(MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_ROOT))

from mars_s2l_adapter import compute_mbmp  # noqa: E402
from methanes2cm_adapter import (  # noqa: E402
    MODEL_BAND_INDICES,
    REFLECTANCE_DIVISOR,
    REFLECTANCE_MAX,
    V5_INPUT_CHANNELS,
)
from methanes2cm_v5_model import MethaneS2CMV5Model  # noqa: E402

from acquire_mars_metadata import repo_root, sha256  # noqa: E402
from train_mars_v3 import safe_output, seed_everything, tracked_dirty, write_json  # noqa: E402

DATA_ROOT = Path(
    "EarthRemoteSensingRapidResponse/Data Collection/s2_emit_pairs/publication-v1/"
    "external/MethaneS2CM/l2a_location_split_32x32"
)
DEFAULT_PACKED = DATA_ROOT / "v5_train_packed.h5"
DEFAULT_MANIFEST = DATA_ROOT / "v5_train_development_manifest.jsonl"
DEFAULT_PROTOCOL = Path("reports/experiments/methanes2cm_v5_protocol.json")
DEFAULT_ACQUISITION = Path("reports/experiments/methanes2cm_v5_train_acquisition.json")
DEFAULT_SEED = 1101
DEFAULT_CHECKPOINT = Path(
    "EarthRemoteSensingRapidResponse/artifacts/methanes2cm_v5_seed1101.pt"
)
DEFAULT_REPORT = Path("reports/experiments/methanes2cm_v5_seed1101_validation.json")
DEFAULT_MARKDOWN = Path("reports/experiments/METHANES2CM_V5_SEED1101_VALIDATION.md")
DEFAULT_CACHE = DATA_ROOT / "v5_validation_seed1101.npz"
TARGET_FPRS = (0.02, 0.05, 0.08, 0.095)
PIXEL_THRESHOLDS = tuple(float(value) for value in np.linspace(0.05, 0.95, 19))
SOURCE_REVISION = "ee9a96d4994ca6bc45725c1e92d7a06258131eaf"


def read_manifest(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid v5 manifest line {line_number}") from exc
            if row.get("research_role") not in {"internal_fitting", "internal_development"}:
                raise ValueError(f"Unexpected v5 research role on line {line_number}")
            if str(row.get("source_revision")) != SOURCE_REVISION:
                raise ValueError(f"Unexpected source revision on line {line_number}")
            rows.append(row)
    rows.sort(key=lambda row: int(row["id"]))
    if len({str(row["id"]) for row in rows}) != len(rows):
        raise ValueError("Duplicate sample id in v5 manifest")
    return rows


def smoke_records(
    rows: list[dict[str, Any]], count: int, seed: int
) -> list[dict[str, Any]]:
    if count < 2:
        raise ValueError("Smoke cohort must contain at least two samples")
    rng = np.random.default_rng(seed)
    selected: list[dict[str, Any]] = []
    for label in (0, 1):
        candidates = [row for row in rows if int(row["label"]) == label]
        take = min(len(candidates), count // 2)
        indices = rng.choice(len(candidates), size=take, replace=False)
        selected.extend(candidates[int(index)] for index in indices)
    selected.sort(key=lambda row: int(row["id"]))
    return selected


class PackedMethaneS2CMDataset(Dataset[dict[str, Any]]):
    """Lazy per-worker reader for the ignored, checksum-bound packed HDF5."""

    def __init__(
        self,
        packed_path: Path,
        records: list[dict[str, Any]],
        *,
        augment: bool,
        seed: int,
    ) -> None:
        self.packed_path = packed_path
        self.records = records
        self.augment = augment
        self.seed = seed
        self._packed: h5py.File | None = None
        self._rng: np.random.Generator | None = None
        with h5py.File(packed_path, "r") as source:
            if str(source.attrs.get("source_revision")) != SOURCE_REVISION:
                raise ValueError("Packed v5 source revision does not match the sealed protocol")
            identifiers = source["sample_id"][:].astype(np.int64)
            labels = source["label"][:].astype(np.uint8)
        index_by_id = {int(identifier): index for index, identifier in enumerate(identifiers)}
        if len(index_by_id) != len(identifiers):
            raise ValueError("Packed v5 sample ids are not unique")
        self.packed_indices: list[int] = []
        for row in records:
            identifier = int(row["id"])
            if identifier not in index_by_id:
                raise ValueError(f"Packed v5 data lacks manifest sample {identifier}")
            packed_index = index_by_id[identifier]
            if int(labels[packed_index]) != int(row["label"]):
                raise ValueError(f"Packed label disagrees for sample {identifier}")
            self.packed_indices.append(packed_index)

    def __len__(self) -> int:
        return len(self.records)

    def packed(self) -> h5py.File:
        if self._packed is None:
            self._packed = h5py.File(self.packed_path, "r", rdcc_nbytes=32 * 1024 * 1024)
        return self._packed

    def rng(self) -> np.random.Generator:
        if self._rng is None:
            worker = get_worker_info()
            worker_seed = self.seed if worker is None else int(worker.seed) + self.seed
            self._rng = np.random.default_rng(worker_seed % (2**63 - 1))
        return self._rng

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_packed"] = None
        state["_rng"] = None
        return state

    def __del__(self) -> None:
        if self._packed is not None:
            self._packed.close()

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.records[index]
        packed_index = self.packed_indices[index]
        source = self.packed()
        target_raw = source["target"][packed_index]
        reference90_raw = source["reference90"][packed_index]
        reference365_raw = source["reference365"][packed_index]
        mask = source["mask"][packed_index].astype(np.float32)
        selected_raw = [
            values[np.asarray(MODEL_BAND_INDICES)]
            for values in (target_raw, reference90_raw, reference365_raw)
        ]
        observable = np.all(np.concatenate(selected_raw, axis=0) != 0, axis=0)
        spectral = [
            np.clip(
                values.astype(np.float32) / REFLECTANCE_DIVISOR,
                0.0,
                REFLECTANCE_MAX,
            )
            for values in selected_raw
        ]
        mbmp90 = compute_mbmp(spectral[0], spectral[1], valid_mask=observable)
        mbmp365 = compute_mbmp(spectral[0], spectral[2], valid_mask=observable)
        inputs = np.concatenate(
            [mbmp90[None], mbmp365[None], *spectral], axis=0
        ).astype(np.float32)
        inputs[:, ~observable] = 0.0
        inputs[:2, ~observable] = 1.0

        if self.augment:
            rng = self.rng()
            turns = int(rng.integers(0, 4))
            if turns:
                inputs = np.rot90(inputs, turns, axes=(1, 2)).copy()
                observable = np.rot90(observable, turns).copy()
                mask = np.rot90(mask, turns).copy()
            if bool(rng.integers(0, 2)):
                inputs = inputs[:, :, ::-1].copy()
                observable = observable[:, ::-1].copy()
                mask = mask[:, ::-1].copy()
            if bool(rng.integers(0, 2)):
                inputs = inputs[:, ::-1, :].copy()
                observable = observable[::-1, :].copy()
                mask = mask[::-1, :].copy()
        if inputs.shape != (len(V5_INPUT_CHANNELS), 32, 32):
            raise ValueError("Constructed packed v5 input violates its channel contract")
        presence = float(int(row["label"]))
        if bool(presence) != bool(np.any(mask > 0.5)):
            raise ValueError(f"Packed mask/manifest label mismatch for sample {row['id']}")
        return {
            "inputs": torch.from_numpy(inputs),
            "observable": torch.from_numpy(observable[None].astype(np.float32)),
            "mask": torch.from_numpy(mask[None]),
            "presence": torch.tensor(presence, dtype=torch.float32),
            "sample_id": int(row["id"]),
            "packed_index": packed_index,
            "group_id": str(row["group_id"]),
            "exact_location_id": str(row["exact_location_id"]),
        }


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def choose_threshold_at_fpr(
    labels: np.ndarray, scores: np.ndarray, target_fpr: float
) -> dict[str, float]:
    truth = np.asarray(labels, dtype=np.uint8)
    probability = np.asarray(scores, dtype=np.float64)
    candidates = np.concatenate(
        [[np.nextafter(float(np.max(probability)), math.inf)], np.unique(probability)[::-1]]
    )
    best: tuple[float, float, float] | None = None
    result: dict[str, float] | None = None
    for threshold in candidates:
        prediction = probability >= threshold
        negatives = truth == 0
        positives = truth == 1
        false_positive_rate = float(np.mean(prediction[negatives]))
        if false_positive_rate > target_fpr + 1e-12:
            continue
        recall = float(np.mean(prediction[positives]))
        precision = float(np.sum(prediction & positives) / max(np.sum(prediction), 1))
        rank = (recall, false_positive_rate, -float(threshold))
        if best is None or rank > best:
            best = rank
            result = {
                "threshold": float(threshold),
                "recall": recall,
                "false_positive_rate": false_positive_rate,
                "precision": precision,
            }
    if result is None:
        raise ValueError("No scene threshold satisfies the requested FPR")
    return result


def segmentation_first_loss(
    output: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    hard_negative_fraction: float = 0.02,
    scene_weight: float = 0.5,
) -> tuple[torch.Tensor, dict[str, float]]:
    logits = output["segmentation_logits"]
    target = batch["mask"]
    observable = batch["observable"]
    valid = observable > 0.5
    positive = (target > 0.5) & valid
    negative = (~positive) & valid
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    positive_count = positive.flatten(1).sum(dim=1)
    positive_loss_by_scene = (bce * positive).flatten(1).sum(dim=1) / positive_count.clamp_min(1)
    positive_bce = (
        positive_loss_by_scene[positive_count > 0].mean()
        if torch.any(positive_count > 0)
        else logits.sum() * 0.0
    )
    negative_bce = bce.masked_fill(~negative, -1e4).flatten(1)
    hard_count = max(1, int(negative_bce.shape[1] * hard_negative_fraction))
    hard_negative_bce = torch.topk(negative_bce, k=hard_count, dim=1).values.mean()
    probability = torch.sigmoid(logits) * observable
    truth = target * observable
    intersection = (probability * truth).sum(dim=(-2, -1))
    denominator = probability.sum(dim=(-2, -1)) + truth.sum(dim=(-2, -1))
    positive_scenes = truth.sum(dim=(-2, -1)) > 0
    dice = (2.0 * intersection + 1.0) / (denominator + 1.0)
    dice_loss = (
        1.0 - dice[positive_scenes].mean()
        if torch.any(positive_scenes)
        else logits.sum() * 0.0
    )
    scene_bce = F.binary_cross_entropy_with_logits(
        output["scene_logit"], batch["presence"]
    )
    total = positive_bce + hard_negative_bce + 0.5 * dice_loss + scene_weight * scene_bce
    return total, {
        "total": float(total.detach()),
        "positive_bce": float(positive_bce.detach()),
        "hard_negative_bce": float(hard_negative_bce.detach()),
        "positive_dice_loss": float(dice_loss.detach()),
        "scene_bce": float(scene_bce.detach()),
    }


@torch.no_grad()
def validation_summary(
    model: MethaneS2CMV5Model,
    loader: DataLoader[dict[str, Any]],
    device: torch.device,
    *,
    retain_predictions: bool = False,
) -> tuple[dict[str, Any], dict[str, np.ndarray] | None]:
    model.eval()
    labels: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    sample_ids: list[np.ndarray] = []
    packed_indices: list[np.ndarray] = []
    groups: list[str] = []
    exact_locations: list[str] = []
    pixel_truth: list[np.ndarray] = []
    pixel_scores: list[np.ndarray] = []
    probability_maps: list[np.ndarray] = []
    observable_maps: list[np.ndarray] = []
    intersections = np.zeros(len(PIXEL_THRESHOLDS), dtype=np.float64)
    predicted = np.zeros(len(PIXEL_THRESHOLDS), dtype=np.float64)
    truth_pixels = 0.0
    observable_pixels = 0
    for batch in loader:
        batch = move_batch(batch, device)
        with torch.amp.autocast(
            "cuda", dtype=torch.float16, enabled=device.type == "cuda"
        ):
            output = model(batch["inputs"], batch["observable"])
        probability = torch.sigmoid(output["segmentation_logits"]).float()
        observable = batch["observable"] > 0.5
        truth = (batch["mask"] > 0.5) & observable
        labels.append(batch["presence"].cpu().numpy())
        scores.append(torch.sigmoid(output["scene_logit"]).float().cpu().numpy())
        sample_ids.append(batch["sample_id"].cpu().numpy())
        packed_indices.append(batch["packed_index"].cpu().numpy())
        groups.extend(str(value) for value in batch["group_id"])
        exact_locations.extend(str(value) for value in batch["exact_location_id"])
        truth_pixels += float(truth.sum())
        observable_pixels += int(observable.sum())
        local_probability = probability[observable].cpu().numpy().astype(np.float32)
        local_truth = truth[observable].cpu().numpy().astype(np.uint8)
        pixel_scores.append(local_probability)
        pixel_truth.append(local_truth)
        for threshold_index, threshold in enumerate(PIXEL_THRESHOLDS):
            prediction = (probability >= threshold) & observable
            intersections[threshold_index] += float(torch.sum(prediction & truth))
            predicted[threshold_index] += float(torch.sum(prediction))
        if retain_predictions:
            probability_maps.append(probability[:, 0].cpu().numpy().astype(np.float16))
            observable_maps.append(observable[:, 0].cpu().numpy().astype(np.uint8))

    y = np.concatenate(labels).astype(np.uint8)
    p = np.concatenate(scores).astype(np.float32)
    truth_values = np.concatenate(pixel_truth)
    score_values = np.concatenate(pixel_scores)
    dice = 2.0 * intersections / np.maximum(predicted + truth_pixels, 1.0)
    union = predicted + truth_pixels - intersections
    iou = intersections / np.maximum(union, 1.0)
    best_pixel = int(np.argmax(dice))
    summary = {
        "scenes": int(y.size),
        "positives": int(np.count_nonzero(y == 1)),
        "negatives": int(np.count_nonzero(y == 0)),
        "geographic_groups": len(set(groups)),
        "exact_locations": len(set(exact_locations)),
        "average_precision": float(average_precision_score(y, p)),
        "auroc": float(roc_auc_score(y, p)),
        "operating_points": {
            str(target): choose_threshold_at_fpr(y, p, target) for target in TARGET_FPRS
        },
        "segmentation": {
            "average_precision_all_observable_pixels": float(
                average_precision_score(truth_values, score_values)
            ),
            "threshold": PIXEL_THRESHOLDS[best_pixel],
            "dice": float(dice[best_pixel]),
            "intersection_over_union": float(iou[best_pixel]),
            "truth_positive_pixels": int(truth_pixels),
            "predicted_positive_pixels": int(predicted[best_pixel]),
            "observable_pixels": observable_pixels,
            "threshold_grid": [
                {
                    "threshold": threshold,
                    "dice": float(dice[index]),
                    "intersection_over_union": float(iou[index]),
                    "intersection_pixels": int(intersections[index]),
                    "predicted_positive_pixels": int(predicted[index]),
                }
                for index, threshold in enumerate(PIXEL_THRESHOLDS)
            ],
        },
    }
    if not retain_predictions:
        return summary, None
    predictions = {
        "sample_id": np.concatenate(sample_ids).astype(np.int64),
        "packed_index": np.concatenate(packed_indices).astype(np.int64),
        "label": y,
        "group_id": np.asarray(groups),
        "exact_location_id": np.asarray(exact_locations),
        "scene_score": p,
        "segmentation_probability": np.concatenate(probability_maps),
        "observable": np.concatenate(observable_maps),
    }
    return summary, predictions


def train(
    model: MethaneS2CMV5Model,
    train_loader: DataLoader[dict[str, Any]],
    validation_loader: DataLoader[dict[str, Any]],
    device: torch.device,
    checkpoint: Path,
    *,
    epochs: int,
    learning_rate: float,
    patience: int,
    seed: int,
) -> tuple[list[dict[str, Any]], int]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = []
    best_rank = (-math.inf, -math.inf, -math.inf, -math.inf)
    best_epoch = -1
    stale = 0
    for epoch in range(1, epochs + 1):
        model.train()
        parts: list[dict[str, float]] = []
        started = time.perf_counter()
        for batch in train_loader:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                "cuda", dtype=torch.float16, enabled=device.type == "cuda"
            ):
                output = model(batch["inputs"], batch["observable"])
                loss, loss_parts = segmentation_first_loss(output, batch)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            parts.append(loss_parts)
        scheduler.step()
        validation, _ = validation_summary(model, validation_loader, device)
        record = {
            "epoch": epoch,
            "seconds": round(time.perf_counter() - started, 3),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "training": {
                key: float(np.mean([item[key] for item in parts])) for key in parts[0]
            },
            "validation": validation,
        }
        history.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
        rank = (
            float(validation["average_precision"]),
            float(validation["auroc"]),
            float(validation["operating_points"]["0.05"]["recall"]),
            float(validation["segmentation"]["dice"]),
        )
        if rank > best_rank:
            best_rank = rank
            best_epoch = epoch
            stale = 0
            temporary = checkpoint.with_suffix(checkpoint.suffix + ".tmp")
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "model_metadata": model.artifact_metadata(),
                    "epoch": epoch,
                    "seed": seed,
                    "validation": validation,
                },
                temporary,
            )
            os.replace(temporary, checkpoint)
        else:
            stale += 1
            if stale >= patience:
                break
    if best_epoch < 0:
        raise RuntimeError("V5 training produced no validation-selected checkpoint")
    payload = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(payload["state_dict"], strict=True)
    return history, best_epoch


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    validation = report["validation"]
    op = validation["operating_points"]["0.05"]
    segmentation = validation["segmentation"]
    lines = [
        f"# MethaneS2CM v5 seed {report['seed']} validation",
        "",
        "Pipeline smoke test only; not an accuracy result."
        if report["smoke_test"]
        else "Development result on frozen, 25 km-disjoint MethaneS2CM groups; location-test imagery remains sealed.",
        "",
        f"- Model: `{report['model']['model_name']}` ({report['model']['parameter_count']:,} parameters)",
        f"- Cohort: {report['cohort']['fitting_samples']:,} fit / {report['cohort']['development_samples']:,} development crops",
        f"- Best epoch: {report['training']['best_epoch']} / {len(report['training']['history'])}",
        f"- Scene AP / AUROC: {validation['average_precision']:.4f} / {validation['auroc']:.4f}",
        f"- Recall at <=5% FPR: {op['recall']:.4f} (FPR {op['false_positive_rate']:.4f})",
        f"- Pixel AP / Dice / IoU: {segmentation['average_precision_all_observable_pixels']:.4f} / {segmentation['dice']:.4f} / {segmentation['intersection_over_union']:.4f}",
        "- The released location-test image split was not extracted or opened.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packed", default=DEFAULT_PACKED.as_posix())
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST.as_posix())
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL.as_posix())
    parser.add_argument("--acquisition", default=DEFAULT_ACQUISITION.as_posix())
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT.as_posix())
    parser.add_argument("--report", default=DEFAULT_REPORT.as_posix())
    parser.add_argument("--markdown", default=DEFAULT_MARKDOWN.as_posix())
    parser.add_argument("--prediction-cache", default=DEFAULT_CACHE.as_posix())
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    root = repo_root()
    if tracked_dirty(root):
        raise RuntimeError("Refusing v5 training from a dirty tracked worktree")
    packed_path = (root / args.packed).resolve()
    manifest_path = (root / args.manifest).resolve()
    protocol_path = (root / args.protocol).resolve()
    acquisition_path = (root / args.acquisition).resolve()
    for required in (packed_path, manifest_path, protocol_path, acquisition_path):
        if not required.is_file():
            raise FileNotFoundError(required)
    checkpoint = safe_output(root, args.checkpoint)
    report_path = safe_output(root, args.report)
    markdown_path = safe_output(root, args.markdown)
    cache_path = safe_output(root, args.prediction_cache)
    if (
        subprocess.run(
            ["git", "check-ignore", "--quiet", "--", cache_path.relative_to(root)],
            cwd=root,
            check=False,
        ).returncode
        != 0
    ):
        raise ValueError("V5 prediction cache must be ignored by Git")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    seed_everything(args.seed)

    rows = read_manifest(manifest_path)
    by_role = {
        role: [row for row in rows if row["research_role"] == role]
        for role in ("internal_fitting", "internal_development")
    }
    group_overlap = set(row["group_id"] for row in by_role["internal_fitting"]) & set(
        row["group_id"] for row in by_role["internal_development"]
    )
    if group_overlap:
        raise ValueError("Frozen v5 fitting/development groups overlap")
    fitting_records = by_role["internal_fitting"]
    development_records = by_role["internal_development"]
    if args.smoke:
        fitting_records = smoke_records(fitting_records, 256, args.seed)
        development_records = smoke_records(development_records, 128, args.seed + 1)

    train_dataset = PackedMethaneS2CMDataset(
        packed_path, fitting_records, augment=True, seed=args.seed
    )
    development_dataset = PackedMethaneS2CMDataset(
        packed_path, development_records, augment=False, seed=args.seed + 1
    )
    loader_options: dict[str, Any] = {
        "batch_size": min(args.batch_size, len(train_dataset)),
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.workers > 0,
        "generator": torch.Generator().manual_seed(args.seed),
    }
    if args.workers > 0:
        loader_options["prefetch_factor"] = 2
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_options)
    validation_options = dict(loader_options)
    validation_options["batch_size"] = min(args.batch_size, len(development_dataset))
    validation_options["generator"] = torch.Generator().manual_seed(args.seed + 1)
    validation_loader = DataLoader(development_dataset, shuffle=False, **validation_options)

    model = MethaneS2CMV5Model().to(device)
    epochs = 1 if args.smoke else args.epochs
    history, best_epoch = train(
        model,
        train_loader,
        validation_loader,
        device,
        checkpoint,
        epochs=epochs,
        learning_rate=args.learning_rate,
        patience=args.patience,
        seed=args.seed,
    )
    validation, predictions = validation_summary(
        model, validation_loader, device, retain_predictions=True
    )
    if predictions is None:
        raise RuntimeError("Final v5 validation did not retain predictions")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_cache = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with temporary_cache.open("wb") as destination:
        np.savez_compressed(destination, **predictions)
    os.replace(temporary_cache, cache_path)

    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    acquisition = json.loads(acquisition_path.read_text(encoding="utf-8"))
    report = {
        "schema_version": 1,
        "scope": "methanes2cm_v5_internal_development",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "smoke_test": args.smoke,
        "seed": args.seed,
        "source": protocol["source"],
        "seal": {
            "protocol_path": protocol_path.relative_to(root).as_posix(),
            "protocol_sha256": sha256(protocol_path),
            "manifest_path": manifest_path.relative_to(root).as_posix(),
            "manifest_sha256": sha256(manifest_path),
            "packed_path": packed_path.relative_to(root).as_posix(),
            "packed_sha256": acquisition["extraction"]["packed_sha256"],
            "location_test_images_opened": False,
        },
        "cohort": {
            "fitting_samples": len(fitting_records),
            "fitting_positives": sum(int(row["label"]) for row in fitting_records),
            "fitting_groups": len(set(row["group_id"] for row in fitting_records)),
            "fitting_exact_locations": len(
                set(row["exact_location_id"] for row in fitting_records)
            ),
            "development_samples": len(development_records),
            "development_positives": sum(int(row["label"]) for row in development_records),
            "development_groups": len(
                set(row["group_id"] for row in development_records)
            ),
            "development_exact_locations": len(
                set(row["exact_location_id"] for row in development_records)
            ),
            "geographic_group_overlap": 0,
        },
        "model": model.artifact_metadata(),
        "training": {
            "epochs_requested": epochs,
            "best_epoch": best_epoch,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "optimizer": "AdamW(weight_decay=1e-4)",
            "scheduler": "CosineAnnealingLR",
            "objective": (
                "positive-pixel BCE + top-2%-negative BCE + 0.5 positive soft-Dice + "
                "0.5 BCE on segmentation-derived scene logit"
            ),
            "augmentation": "random right-angle rotations and horizontal/vertical flips",
            "selection_rank": [
                "scene_average_precision",
                "scene_auroc",
                "recall_at_fpr_le_0.05",
                "pixel_dice",
            ],
            "history": history,
        },
        "validation": validation,
        "checkpoint": {
            "path": checkpoint.relative_to(root).as_posix(),
            "bytes": checkpoint.stat().st_size,
            "sha256": sha256(checkpoint),
            "tracked": False,
        },
        "prediction_cache": {
            "path": cache_path.relative_to(root).as_posix(),
            "bytes": cache_path.stat().st_size,
            "sha256": sha256(cache_path),
            "tracked": False,
        },
        "reproducibility": {
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True
            ).strip(),
            "script": Path(__file__).resolve().relative_to(root).as_posix(),
            "script_sha256": sha256(Path(__file__).resolve()),
            "python": sys.version,
            "torch": torch.__version__,
            "sklearn": sklearn.__version__,
            "device": str(device),
            "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "tracked_worktree_dirty_at_start": False,
            "deterministic_algorithms_enforced": False,
        },
        "decision": (
            "Smoke contract passes; do not interpret metrics."
            if args.smoke
            else "Internal development candidate only. Keep the location-test imagery sealed until the multi-seed rule is frozen."
        ),
    }
    write_json(report_path, report)
    write_markdown(markdown_path, report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
