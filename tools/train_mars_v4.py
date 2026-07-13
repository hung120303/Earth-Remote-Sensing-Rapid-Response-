#!/usr/bin/env python3
"""Train the simulation-assisted ERSRR v4 temporal-Siamese detector."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import sklearn
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler, get_worker_info

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "EarthRemoteSensingRapidResponse"
if str(MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_ROOT))

from mars_s2l_adapter import (  # noqa: E402
    REFLECTANCE_DIVISOR,
    REFLECTANCE_MAX,
    compute_mbmp,
    iter_manifest,
    load_sample,
)
from mars_v4_model import INPUT_CHANNELS, MarsV4Model  # noqa: E402
from mars_v4_simulation import MarsPlumeSimulator  # noqa: E402

from acquire_mars_metadata import (  # noqa: E402
    DEFAULT_OUTPUT,
    REVISION,
    checked_output_dir,
    repo_root,
    sha256,
)
from build_mars_dev_cohort import DEV_SAMPLES  # noqa: E402
from build_mars_v3_training_cohort import V3_SAMPLES  # noqa: E402
from train_mars_v3 import (  # noqa: E402
    DEFAULT_METADATA_CSV,
    rotate_wind,
    safe_output,
    seed_everything,
    stratified_smoke,
    tracked_dirty,
    training_weights,
    write_json,
)

DEFAULT_SEED = 606
DEFAULT_LUT = Path("configs/mars_s2_integrated_transmittances.json")
DEFAULT_CHECKPOINT = Path("EarthRemoteSensingRapidResponse/artifacts/mars_v4_seed606.pt")
DEFAULT_JSON = Path("reports/experiments/mars_v4_seed606_validation.json")
DEFAULT_MARKDOWN = Path("reports/experiments/MARS_V4_SEED606_VALIDATION.md")
SMOKE_CHECKPOINT = Path("EarthRemoteSensingRapidResponse/artifacts/mars_v4_smoke_seed606.pt")
SMOKE_JSON = Path("reports/experiments/mars_v4_smoke.json")
SMOKE_MARKDOWN = Path("reports/experiments/MARS_V4_SMOKE.md")
TARGET_FPRS = (0.05, 0.08, 0.095)
PIXEL_THRESHOLDS = tuple(float(value) for value in np.linspace(0.1, 0.9, 9))


def finite_float(value: Any, default: float = math.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def safe_data_asset(metadata_dir: Path, value: str) -> Path:
    path = (metadata_dir / value).resolve()
    if metadata_dir.resolve() not in path.parents:
        raise ValueError("MARS asset path escapes the pinned dataset directory")
    return path


def metadata_and_plume_library(
    metadata_dir: Path,
    metadata_csv: Path,
    required_ids: set[str],
    fit_positive_ids: set[str],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    scenes: dict[str, dict[str, Any]] = {}
    library: list[dict[str, Any]] = []
    with metadata_csv.open("r", encoding="utf-8-sig", newline="") as source:
        for row in csv.DictReader(source):
            sample_id = str(row["id_loc_image"])
            if sample_id in required_ids:
                wind_u = finite_float(row.get("wind_u"), 4.0)
                wind_v = finite_float(row.get("wind_v"), 4.0)
                scenes[sample_id] = {
                    "wind": (float(np.clip(wind_u, -20, 20)), float(np.clip(wind_v, -20, 20))),
                    "sza": finite_float(row.get("sza"), 30.0),
                    "vza": finite_float(row.get("vza"), 5.0),
                    "satellite": str(row.get("satellite") or "S2A"),
                    "offshore": truthy(row.get("offshore")),
                }
            if sample_id not in fit_positive_ids:
                continue
            ch4_relative = str(row.get("ch4path") or "")
            mask_relative = str(row.get("plumepath") or "")
            if not ch4_relative or not mask_relative:
                continue
            ch4_path = safe_data_asset(metadata_dir, ch4_relative)
            mask_path = safe_data_asset(metadata_dir, mask_relative)
            if not ch4_path.is_file() or not mask_path.is_file():
                continue
            wind_u = finite_float(row.get("wind_u"))
            wind_v = finite_float(row.get("wind_v"))
            if not math.isfinite(wind_u) or not math.isfinite(wind_v):
                continue
            library.append(
                {
                    "sample_id": sample_id,
                    "ch4_path": ch4_path,
                    "mask_path": mask_path,
                    "wind": (wind_u, wind_v),
                    "wind_speed": float(np.hypot(wind_u, wind_v)),
                    "row": int(finite_float(row.get("window_row_off"), 0)),
                    "col": int(finite_float(row.get("window_col_off"), 0)),
                    "height": int(finite_float(row.get("window_height"), 0)),
                    "width": int(finite_float(row.get("window_width"), 0)),
                }
            )
    missing = required_ids - set(scenes)
    if missing:
        raise ValueError(f"Metadata CSV is missing geometry/wind for {len(missing)} selected scenes")
    if not library:
        raise ValueError("No fit-split CH4 rasters are available for plume simulation")
    library.sort(key=lambda item: item["sample_id"])
    return scenes, library


class MarsV4Dataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        metadata_dir: Path,
        records: list[dict[str, Any]],
        scene_metadata: dict[str, dict[str, Any]],
        *,
        lut_path: Path,
        plume_library: list[dict[str, Any]],
        augment: bool,
        simulation_fraction: float,
        seed: int,
    ) -> None:
        if not 0.0 <= simulation_fraction <= 1.0:
            raise ValueError("Simulation fraction must be in [0,1]")
        self.metadata_dir = metadata_dir
        self.records = records
        self.scene_metadata = scene_metadata
        self.lut_path = lut_path
        self.plume_library = plume_library
        self.augment = augment
        self.simulation_fraction = simulation_fraction
        self.seed = seed
        self.eligible_negative_indices = [
            index
            for index, record in enumerate(records)
            if record["label_state"] == "NO_PLUME"
            and not scene_metadata[str(record["sample_id"])]["offshore"]
            and np.linalg.norm(scene_metadata[str(record["sample_id"])]["wind"]) <= 9.0
            and str(record.get("observability", "")).lower() == "clear"
        ]
        if augment and simulation_fraction > 0 and not self.eligible_negative_indices:
            raise ValueError("Simulation requires clear, onshore no-plume fit scenes")
        self._rng: np.random.Generator | None = None
        self._simulator: MarsPlumeSimulator | None = None
        self._plume_cache: OrderedDict[str, tuple[np.ndarray, np.ndarray]] = OrderedDict()
        self._library_speeds = np.asarray(
            [item["wind_speed"] for item in plume_library], dtype=np.float32
        )

    def __len__(self) -> int:
        return len(self.records)

    def rng(self) -> np.random.Generator:
        if self._rng is None:
            worker = get_worker_info()
            worker_seed = self.seed if worker is None else int(worker.seed)
            self._rng = np.random.default_rng(worker_seed)
        return self._rng

    def simulator(self) -> MarsPlumeSimulator:
        if self._simulator is None:
            self._simulator = MarsPlumeSimulator(self.lut_path)
        return self._simulator

    def plume_arrays(self, item: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        key = str(item["sample_id"])
        if key in self._plume_cache:
            result = self._plume_cache.pop(key)
            self._plume_cache[key] = result
            return result
        with rasterio.open(item["ch4_path"]) as source:
            ch4 = source.read(1).astype(np.float32)
        with rasterio.open(item["mask_path"]) as source:
            mask = source.read(1).astype(bool)
        row, col = int(item["row"]), int(item["col"])
        height, width = int(item["height"]), int(item["width"])
        if height > 0 and width > 0:
            row_end = min(mask.shape[0], row + height)
            col_end = min(mask.shape[1], col + width)
            ch4 = ch4[max(0, row) : row_end, max(0, col) : col_end]
            mask = mask[max(0, row) : row_end, max(0, col) : col_end]
        if not np.any(mask):
            raise ValueError(f"Simulation plume {key} has an empty cropped mask")
        result = (ch4, mask)
        self._plume_cache[key] = result
        while len(self._plume_cache) > 16:
            self._plume_cache.popitem(last=False)
        return result

    def simulated_positive(
        self, requested_record: dict[str, Any]
    ) -> tuple[dict[str, Any], Any, np.ndarray, np.ndarray, tuple[float, float], str] | None:
        rng = self.rng()
        target_record = self.records[int(rng.choice(self.eligible_negative_indices))]
        target_sample = load_sample(self.metadata_dir, target_record, require_enhancement=False)
        metadata = self.scene_metadata[target_sample.sample_id]
        target_speed = float(np.linalg.norm(metadata["wind"]))
        distance = np.abs(self._library_speeds - target_speed)
        tolerance = max(1.5, float(np.min(distance)))
        candidates = np.flatnonzero(distance <= tolerance)
        source = self.plume_library[int(rng.choice(candidates))]
        ch4, source_mask = self.plume_arrays(source)
        result = self.simulator().simulate(
            target_sample.raw_pair[:6],
            ch4,
            source_mask,
            source_wind=source["wind"],
            target_wind=metadata["wind"],
            satellite=metadata["satellite"],
            solar_zenith_degrees=metadata["sza"],
            view_zenith_degrees=metadata["vza"],
            rng=rng,
        )
        visible = result.mask & target_sample.observable_mask
        if np.count_nonzero(visible) / max(np.count_nonzero(result.mask), 1) < 0.5:
            return None
        raw_pair = np.concatenate([result.target, target_sample.raw_pair[6:]], axis=0)
        reflectance = np.clip(
            raw_pair.astype(np.float32) / REFLECTANCE_DIVISOR, 0.0, REFLECTANCE_MAX
        )
        identifier = f"sim:{target_sample.sample_id}:{source['sample_id']}"
        return (
            target_record,
            target_sample,
            reflectance,
            result.mask.astype(np.float32),
            metadata["wind"],
            identifier,
        )

    def __getitem__(self, index: int) -> dict[str, Any]:
        requested = self.records[index]
        simulated = False
        simulation = None
        if (
            self.augment
            and requested["label_state"] == "PLUME"
            and self.plume_library
            and self.rng().random() < self.simulation_fraction
        ):
            try:
                simulation = self.simulated_positive(requested)
            except (OSError, ValueError, FloatingPointError):
                simulation = None
        if simulation is None:
            record = requested
            sample = load_sample(self.metadata_dir, record, require_enhancement=False)
            spectral = sample.reflectance_pair.copy()
            mask = sample.plume_mask.astype(np.float32)
            wind = self.scene_metadata[sample.sample_id]["wind"]
            sample_id = sample.sample_id
        else:
            record, sample, spectral, mask, wind, sample_id = simulation
            simulated = True
        cloud = (sample.cloud_classes > 0).astype(np.float32)
        observable = sample.observable_mask.astype(np.float32)
        mbmp = compute_mbmp(spectral[:6], spectral[6:])
        if self.augment:
            rng = self.rng()
            turns = int(rng.integers(0, 4))
            if turns:
                spectral = np.rot90(spectral, turns, axes=(1, 2)).copy()
                mbmp = np.rot90(mbmp, turns).copy()
                cloud = np.rot90(cloud, turns).copy()
                observable = np.rot90(observable, turns).copy()
                mask = np.rot90(mask, turns).copy()
                wind = rotate_wind(wind, turns)
            if bool(rng.integers(0, 2)):
                spectral = spectral[:, :, ::-1].copy()
                mbmp = mbmp[:, ::-1].copy()
                cloud = cloud[:, ::-1].copy()
                observable = observable[:, ::-1].copy()
                mask = mask[:, ::-1].copy()
                wind = (-wind[0], wind[1])
            if bool(rng.integers(0, 2)):
                spectral = spectral[:, ::-1, :].copy()
                mbmp = mbmp[::-1, :].copy()
                cloud = cloud[::-1, :].copy()
                observable = observable[::-1, :].copy()
                mask = mask[::-1, :].copy()
                wind = (wind[0], -wind[1])
        height, width = mbmp.shape
        wind_channels = np.broadcast_to(
            np.asarray(wind, dtype=np.float32)[:, None, None] / 8.0,
            (2, height, width),
        ).copy()
        inputs = np.concatenate(
            [mbmp[None], spectral, wind_channels, cloud[None]], axis=0
        ).astype(np.float32)
        if inputs.shape[0] != len(INPUT_CHANNELS):
            raise ValueError("Constructed input violates the v4 channel contract")
        presence = float(np.any(mask > 0.5))
        return {
            "inputs": torch.from_numpy(inputs),
            "observable": torch.from_numpy(observable[None]),
            "mask": torch.from_numpy(mask[None].astype(np.float32)),
            "presence": torch.tensor(presence, dtype=torch.float32),
            "simulated": torch.tensor(float(simulated), dtype=torch.float32),
            "sample_id": sample_id,
            "group_id": str(record["group_id"]),
        }


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def hard_negative_segmentation_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    observable: torch.Tensor,
    *,
    hard_negative_fraction: float = 0.02,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if not 0.0 < hard_negative_fraction <= 1.0:
        raise ValueError("Hard-negative fraction must be in (0,1]")
    valid = observable > 0.5
    positive = (target > 0.5) & valid
    negative = (~positive) & valid
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    positive_count = positive.flatten(1).sum(dim=1)
    positive_loss = (bce * positive).flatten(1).sum(dim=1) / positive_count.clamp_min(1)
    positive_loss = positive_loss[positive_count > 0].mean() if torch.any(positive_count > 0) else logits.sum() * 0

    negative_bce = bce.masked_fill(~negative, -1e4).flatten(1)
    hard_count = max(1, int(negative_bce.shape[1] * hard_negative_fraction))
    hard_negative = torch.topk(negative_bce, k=hard_count, dim=1).values.mean()
    probability = torch.sigmoid(logits) * observable
    truth = target * observable
    intersection = (probability * truth).sum(dim=(-2, -1))
    denominator = probability.sum(dim=(-2, -1)) + truth.sum(dim=(-2, -1))
    positive_scenes = truth.sum(dim=(-2, -1)) > 0
    dice = (2.0 * intersection + 1.0) / (denominator + 1.0)
    dice_loss = (
        1.0 - dice[positive_scenes].mean()
        if torch.any(positive_scenes)
        else logits.sum() * 0
    )
    total = positive_loss + hard_negative + 0.5 * dice_loss
    return total, {
        "positive_bce": positive_loss,
        "hard_negative_bce": hard_negative,
        "positive_dice_loss": dice_loss,
    }


def total_loss(
    output: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, dict[str, float]]:
    segmentation, parts = hard_negative_segmentation_loss(
        output["segmentation_logits"], batch["mask"], batch["observable"]
    )
    scene = F.binary_cross_entropy_with_logits(output["scene_logit"], batch["presence"])
    total = segmentation + 0.5 * scene
    return total, {
        "total": float(total.detach()),
        "positive_bce": float(parts["positive_bce"].detach()),
        "hard_negative_bce": float(parts["hard_negative_bce"].detach()),
        "positive_dice_loss": float(parts["positive_dice_loss"].detach()),
        "scene_bce": float(scene.detach()),
        "simulated_fraction": float(batch["simulated"].mean().detach()),
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
        fpr = float(np.mean(prediction[negatives]))
        if fpr > target_fpr + 1e-12:
            continue
        recall = float(np.mean(prediction[positives]))
        precision = float(np.sum(prediction & positives) / max(np.sum(prediction), 1))
        rank = (recall, fpr, -float(threshold))
        if best is None or rank > best:
            best = rank
            result = {
                "target_fpr": target_fpr,
                "threshold": float(threshold),
                "recall": recall,
                "false_positive_rate": fpr,
                "precision": precision,
            }
    if result is None:
        raise ValueError("No threshold satisfies the requested FPR")
    return result


@torch.no_grad()
def validation_summary(
    model: MarsV4Model,
    loader: DataLoader[dict[str, Any]],
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    labels: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    intersections = np.zeros(len(PIXEL_THRESHOLDS), dtype=np.float64)
    predicted = np.zeros(len(PIXEL_THRESHOLDS), dtype=np.float64)
    truth_pixels = 0.0
    soft_dice: list[np.ndarray] = []
    for batch in loader:
        batch = move_batch(batch, device)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            output = model(batch["inputs"], batch["observable"])
        probability = torch.sigmoid(output["segmentation_logits"]).float()
        observable = batch["observable"]
        truth = batch["mask"] * observable
        labels.append(batch["presence"].cpu().numpy())
        scores.append(torch.sigmoid(output["scene_logit"]).float().cpu().numpy())
        positive = batch["presence"] > 0.5
        if not torch.any(positive):
            continue
        local_probability = probability[positive] * observable[positive]
        local_truth = truth[positive] > 0.5
        intersection = (local_probability * local_truth).sum(dim=(-2, -1))
        denominator = local_probability.sum(dim=(-2, -1)) + local_truth.sum(dim=(-2, -1))
        soft_dice.append(((2 * intersection + 1) / (denominator + 1))[:, 0].cpu().numpy())
        truth_pixels += float(local_truth.sum())
        for index, threshold in enumerate(PIXEL_THRESHOLDS):
            prediction = local_probability >= threshold
            intersections[index] += float(torch.sum(prediction & local_truth))
            predicted[index] += float(torch.sum(prediction))
    y = np.concatenate(labels).astype(np.uint8)
    p = np.concatenate(scores)
    hard_dice = (2 * intersections + 1) / (predicted + truth_pixels + 1)
    best_pixel = int(np.argmax(hard_dice))
    return {
        "scenes": int(y.size),
        "positives": int(np.count_nonzero(y == 1)),
        "negatives": int(np.count_nonzero(y == 0)),
        "average_precision": float(average_precision_score(y, p)),
        "auroc": float(roc_auc_score(y, p)),
        "operating_points": {
            str(target): choose_threshold_at_fpr(y, p, target) for target in TARGET_FPRS
        },
        "positive_soft_dice": float(np.mean(np.concatenate(soft_dice))),
        "segmentation_threshold": PIXEL_THRESHOLDS[best_pixel],
        "positive_pixel_dice": float(hard_dice[best_pixel]),
    }


def train(
    model: MarsV4Model,
    train_loader: DataLoader[dict[str, Any]],
    validation_loader: DataLoader[dict[str, Any]],
    device: torch.device,
    checkpoint: Path,
    *,
    epochs: int,
    learning_rate: float,
    patience: int,
    validation_every: int,
    seed: int,
) -> tuple[list[dict[str, Any]], int]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = []
    best_rank = (-math.inf, -math.inf, -math.inf)
    best_epoch = -1
    stale = 0
    for epoch in range(1, epochs + 1):
        model.train()
        parts: list[dict[str, float]] = []
        started = time.perf_counter()
        for batch in train_loader:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                output = model(batch["inputs"], batch["observable"])
                loss, loss_parts = total_loss(output, batch)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            parts.append(loss_parts)
        scheduler.step()
        validation = (
            validation_summary(model, validation_loader, device)
            if epoch % validation_every == 0 or epoch == epochs
            else None
        )
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
        if validation is None:
            continue
        rank = (
            float(validation["average_precision"]),
            float(validation["operating_points"]["0.08"]["recall"]),
            float(validation["positive_pixel_dice"]),
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
        raise RuntimeError("V4 training produced no validation-selected checkpoint")
    payload = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(payload["state_dict"], strict=True)
    return history, best_epoch


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    validation = report["validation"]
    lines = [
        "# ERSRR MARS v4 simulation-first validation",
        "",
        "Pipeline smoke test only; not an accuracy result."
        if report["smoke_test"]
        else "Development result on spatially isolated internal groups; the opened strict cohort is not used for selection.",
        "",
        f"- Model: `{report['model']['model_name']}` / {report['model']['parameter_count']:,} parameters",
        f"- Samples: {report['cohort']['training']:,} fit / {report['cohort']['validation']:,} validation",
        f"- Simulation library: {report['simulation']['fit_split_plumes_with_ch4']:,} real enhancement rasters",
        f"- Best epoch: {report['training']['best_epoch']} / {len(report['training']['history'])}",
        f"- Validation AP / AUROC: {validation['average_precision']:.3f} / {validation['auroc']:.3f}",
        f"- Recall at <=8% FPR: {validation['operating_points']['0.08']['recall']:.3f} (FPR {validation['operating_points']['0.08']['false_positive_rate']:.3f})",
        f"- Positive pixel Dice: {validation['positive_pixel_dice']:.3f}",
        "",
        "## Decision",
        "",
        report["decision"],
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-dir", default=DEFAULT_OUTPUT.as_posix())
    parser.add_argument("--metadata-csv", default=DEFAULT_METADATA_CSV.as_posix())
    parser.add_argument("--manifest-file")
    parser.add_argument("--lut", default=DEFAULT_LUT.as_posix())
    parser.add_argument("--checkpoint")
    parser.add_argument("--output-json")
    parser.add_argument("--output-markdown")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--validation-every", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--samples-per-epoch", type=int, default=32768)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--simulation-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    root = repo_root()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for v4 training")
    if args.epochs < 1 or args.validation_every < 1 or args.samples_per_epoch < 1:
        raise ValueError("Epoch, validation, and sample counts must be positive")
    seed_everything(args.seed)
    metadata_dir = checked_output_dir(root, args.metadata_dir)
    metadata_csv = (root / args.metadata_csv).resolve()
    lut_path = (root / args.lut).resolve()
    manifest_name = args.manifest_file or (DEV_SAMPLES if args.smoke else V3_SAMPLES)
    manifest = metadata_dir / manifest_name
    records = list(iter_manifest(manifest))
    by_role = {
        role: [record for record in records if record["research_role"] == role]
        for role in ("internal_training", "internal_validation")
    }
    all_fit_positive_ids = {
        str(record["sample_id"])
        for record in by_role["internal_training"]
        if record["label_state"] == "PLUME"
    }
    if args.smoke:
        by_role["internal_training"] = stratified_smoke(
            by_role["internal_training"], 32, args.seed
        )
        by_role["internal_validation"] = stratified_smoke(
            by_role["internal_validation"], 16, args.seed + 1
        )
        args.epochs = 1
        args.patience = 1
        args.validation_every = 1
        args.batch_size = min(args.batch_size, 2)
        args.workers = 0
        args.samples_per_epoch = 16
    if not by_role["internal_training"] or not by_role["internal_validation"]:
        raise ValueError("Manifest lacks internal training or validation records")
    training_groups = {record["group_id"] for record in by_role["internal_training"]}
    validation_groups = {record["group_id"] for record in by_role["internal_validation"]}
    if training_groups & validation_groups:
        raise ValueError("Training and validation groups overlap")
    required_ids = {
        str(record["sample_id"])
        for role_records in by_role.values()
        for record in role_records
    }
    scene_metadata, plume_library = metadata_and_plume_library(
        metadata_dir, metadata_csv, required_ids, all_fit_positive_ids
    )
    train_dataset = MarsV4Dataset(
        metadata_dir,
        by_role["internal_training"],
        scene_metadata,
        lut_path=lut_path,
        plume_library=plume_library,
        augment=True,
        simulation_fraction=args.simulation_fraction,
        seed=args.seed,
    )
    validation_dataset = MarsV4Dataset(
        metadata_dir,
        by_role["internal_validation"],
        scene_metadata,
        lut_path=lut_path,
        plume_library=[],
        augment=False,
        simulation_fraction=0.0,
        seed=args.seed,
    )
    weights = training_weights(by_role["internal_training"])
    sampler = WeightedRandomSampler(
        weights,
        num_samples=args.samples_per_epoch,
        replacement=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    loader_options: dict[str, Any] = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": True,
        "persistent_workers": args.workers > 0,
    }
    train_loader = DataLoader(train_dataset, sampler=sampler, **loader_options)
    validation_loader = DataLoader(validation_dataset, shuffle=False, **loader_options)
    checkpoint = safe_output(
        root,
        args.checkpoint
        or (SMOKE_CHECKPOINT.as_posix() if args.smoke else DEFAULT_CHECKPOINT.as_posix()),
    )
    output_json = safe_output(
        root,
        args.output_json or (SMOKE_JSON.as_posix() if args.smoke else DEFAULT_JSON.as_posix()),
    )
    output_markdown = safe_output(
        root,
        args.output_markdown
        or (SMOKE_MARKDOWN.as_posix() if args.smoke else DEFAULT_MARKDOWN.as_posix()),
    )
    model = MarsV4Model().to(torch.device("cuda"))
    history, best_epoch = train(
        model,
        train_loader,
        validation_loader,
        torch.device("cuda"),
        checkpoint,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        patience=args.patience,
        validation_every=args.validation_every,
        seed=args.seed,
    )
    validation = validation_summary(model, validation_loader, torch.device("cuda"))
    report = {
        "schema_version": 1,
        "scope": "v4_pipeline_smoke" if args.smoke else "v4_internal_validation_selection",
        "smoke_test": args.smoke,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "dataset": "UNEP-IMEO/MARS-S2L",
            "revision": REVISION,
            "manifest": manifest.relative_to(root).as_posix(),
            "manifest_sha256": sha256(manifest),
        },
        "cohort": {
            "training": len(by_role["internal_training"]),
            "validation": len(by_role["internal_validation"]),
            "training_groups": len(training_groups),
            "validation_groups": len(validation_groups),
            "group_overlap": 0,
            "strict_spatial_test_loaded": False,
        },
        "model": model.artifact_metadata(),
        "simulation": {
            "method": "MARS-S2L integrated-transmittance LUT; real fit-split CH4 fields rotated to target wind and scaled uniformly 0.5-1.5x",
            "requested_fraction_of_positive_draws": args.simulation_fraction,
            "fit_split_plumes_with_ch4": len(plume_library),
            "lut_path": lut_path.relative_to(root).as_posix(),
            "lut_sha256": sha256(lut_path),
            "validation_simulation": False,
            "cross_split_plume_use": False,
        },
        "training": {
            "seed": args.seed,
            "best_epoch": best_epoch,
            "epochs_requested": args.epochs,
            "batch_size": args.batch_size,
            "samples_per_epoch": args.samples_per_epoch,
            "learning_rate": args.learning_rate,
            "validation_every": args.validation_every,
            "history": history,
            "loss": "positive BCE + top-2%-hard-negative BCE + 0.5 Dice + 0.5 segmentation-derived scene BCE",
        },
        "validation": validation,
        "operating_rule": {
            "selected_on": "internal_validation_only",
            "scene_thresholds": validation["operating_points"],
            "segmentation_threshold": validation["segmentation_threshold"],
        },
        "artifact": {
            "path": checkpoint.relative_to(root).as_posix(),
            "bytes": checkpoint.stat().st_size,
            "sha256": sha256(checkpoint),
            "tracked": False,
        },
        "runtime": {
            "device": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
            "rasterio": rasterio.__version__,
        },
        "provenance": {
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True
            ).strip(),
            "git_tracked_worktree_dirty_at_start": tracked_dirty(root),
            "script": "tools/train_mars_v4.py",
            "script_sha256": sha256(Path(__file__)),
            "model_source": "EarthRemoteSensingRapidResponse/mars_v4_model.py",
            "model_source_sha256": sha256(MODEL_ROOT / "mars_v4_model.py"),
            "simulation_source": "EarthRemoteSensingRapidResponse/mars_v4_simulation.py",
            "simulation_source_sha256": sha256(MODEL_ROOT / "mars_v4_simulation.py"),
        },
        "decision": (
            "Pipeline contract passes. Do not interpret smoke metrics."
            if args.smoke
            else "Freeze this validation-selected checkpoint before one development evaluation on the already-opened strict cohort."
        ),
    }
    write_json(output_json, report)
    write_markdown(output_markdown, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
