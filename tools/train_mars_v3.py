#!/usr/bin/env python3
"""Train ERSRR MARS v3 from scratch under the frozen group protocol."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from collections import Counter
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

from mars_s2l_adapter import iter_manifest, load_sample  # noqa: E402
from mars_v3_model import INPUT_CHANNELS, MarsV3Model  # noqa: E402

from acquire_mars_metadata import DEFAULT_OUTPUT, REVISION, checked_output_dir, repo_root, sha256  # noqa: E402
from build_mars_dev_cohort import DEV_SAMPLES  # noqa: E402
from build_mars_v3_training_cohort import V3_SAMPLES  # noqa: E402
from run_mars_dev_scene_baselines import (  # noqa: E402
    choose_lower_threshold,
    choose_upper_threshold,
    role_weights,
)
from train_mars_joint_model import (  # noqa: E402
    choose_quality_threshold,
    select_segmentation_by_dice,
)

DEFAULT_METADATA_CSV = DEFAULT_OUTPUT / "validated_images_all.csv"
DEFAULT_CHECKPOINT = Path("EarthRemoteSensingRapidResponse/artifacts/mars_v3_seed303.pt")
DEFAULT_JSON = Path("reports/experiments/mars_v3_validation.json")
DEFAULT_MARKDOWN = Path("reports/experiments/MARS_V3_VALIDATION.md")
SMOKE_CHECKPOINT = Path("EarthRemoteSensingRapidResponse/artifacts/mars_v3_smoke_seed303.pt")
SMOKE_JSON = Path("reports/experiments/mars_v3_smoke.json")
SMOKE_MARKDOWN = Path("reports/experiments/MARS_V3_SMOKE.md")
DEFAULT_SEED = 303
SEGMENTATION_POSITIVE_WEIGHT = 20.0
SEGMENTATION_DICE_WEIGHT = 0.5
PRESENCE_LOSS_WEIGHT = 1.5
QUALITY_LOSS_WEIGHT = 0.05


def safe_output(root: Path, value: str) -> Path:
    result = (root / value).resolve()
    if root not in result.parents:
        raise ValueError("Output must resolve beneath the repository root")
    return result


def tracked_dirty(root: Path) -> bool:
    status = subprocess.check_output(
        [
            "git",
            "-c",
            "core.autocrlf=true",
            "status",
            "--porcelain",
            "--untracked-files=no",
        ],
        cwd=root,
        text=True,
    )
    return bool(status.strip())


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def wind_lookup(path: Path, required: set[str]) -> dict[str, tuple[float, float]]:
    result: dict[str, tuple[float, float]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        for row in csv.DictReader(source):
            sample_id = str(row["id_loc_image"])
            if sample_id not in required:
                continue
            values: list[float] = []
            for key in ("wind_u", "wind_v"):
                try:
                    value = float(row[key])
                except (TypeError, ValueError):
                    value = math.nan
                values.append(4.0 if not math.isfinite(value) else float(np.clip(value, -20, 20)))
            result[sample_id] = (values[0], values[1])
    missing = required - set(result)
    if missing:
        raise ValueError(f"Metadata CSV is missing wind for {len(missing)} selected samples")
    return result


def rotate_wind(wind: tuple[float, float], quarter_turns: int) -> tuple[float, float]:
    u, v = wind
    for _ in range(quarter_turns % 4):
        u, v = -v, u
    return u, v


class MarsV3Dataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        metadata_dir: Path,
        records: list[dict[str, Any]],
        winds: dict[str, tuple[float, float]],
        *,
        augment: bool,
        seed: int,
    ) -> None:
        self.metadata_dir = metadata_dir
        self.records = records
        self.winds = winds
        self.augment = augment
        self.seed = seed
        self._augmentation_rng: np.random.Generator | None = None

    def augmentation_rng(self) -> np.random.Generator:
        """Return one deterministic stream per persistent data-loader worker."""
        if self._augmentation_rng is None:
            worker = get_worker_info()
            worker_seed = self.seed if worker is None else int(worker.seed)
            self._augmentation_rng = np.random.default_rng(worker_seed)
        return self._augmentation_rng

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        sample = load_sample(self.metadata_dir, record, require_enhancement=False)
        spectral = sample.reflectance_pair.copy()
        mbmp = sample.mbmp_release_compatible.copy()
        cloud = (sample.cloud_classes > 0).astype(np.float32)
        observable = sample.observable_mask.astype(np.float32)
        mask = sample.plume_mask.astype(np.float32)
        wind = self.winds[sample.sample_id]
        if self.augment:
            # The stream advances across repeated samples and epochs. The old
            # index-derived seed made each sample/worker transform effectively
            # static despite weighted resampling.
            rng = self.augmentation_rng()
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
            [mbmp[None, ...], spectral, wind_channels, cloud[None, ...]], axis=0
        ).astype(np.float32)
        if inputs.shape[0] != len(INPUT_CHANNELS):
            raise ValueError("Constructed v3 input does not match the frozen channel contract")
        truth_area = float(np.count_nonzero((mask > 0.5) & (observable > 0.5)))
        return {
            "inputs": torch.from_numpy(inputs),
            "observable": torch.from_numpy(observable[None, ...]),
            "mask": torch.from_numpy(mask[None, ...]),
            "presence": torch.tensor(sample.presence, dtype=torch.float32),
            "quality": torch.tensor(float(np.mean(observable)) >= 0.99, dtype=torch.float32),
            "truth_area": torch.tensor(truth_area, dtype=torch.float32),
            "sample_id": sample.sample_id,
            "group_id": str(record["group_id"]),
        }


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def masked_segmentation_loss(
    logits: torch.Tensor, target: torch.Tensor, observable: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    positive_weight = torch.tensor(
        SEGMENTATION_POSITIVE_WEIGHT, device=logits.device, dtype=logits.dtype
    )
    bce = F.binary_cross_entropy_with_logits(
        logits, target, reduction="none", pos_weight=positive_weight
    )
    bce = (bce * observable).sum() / observable.sum().clamp_min(1.0)
    probability = torch.sigmoid(logits) * observable
    truth = target * observable
    intersection = (probability * truth).sum(dim=(-2, -1))
    denominator = probability.sum(dim=(-2, -1)) + truth.sum(dim=(-2, -1))
    dice_loss = 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()
    return bce + SEGMENTATION_DICE_WEIGHT * dice_loss, bce, dice_loss


def presence_loss(
    logits: torch.Tensor, target: torch.Tensor, truth_area: torch.Tensor
) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    probability = torch.sigmoid(logits)
    correct = probability * target + (1.0 - probability) * (1.0 - target)
    small_plume_weight = torch.sqrt(1000.0 / (truth_area + 1.0)).clamp(0.75, 3.0)
    sample_weight = torch.where(target > 0.5, small_plume_weight, torch.ones_like(target))
    return (bce * (1.0 - correct) * sample_weight).mean()


def total_loss(
    output: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, dict[str, float]]:
    segmentation, bce, dice = masked_segmentation_loss(
        output["segmentation_logits"], batch["mask"], batch["observable"]
    )
    presence = presence_loss(
        output["presence_logit"], batch["presence"], batch["truth_area"]
    )
    quality = F.binary_cross_entropy_with_logits(output["quality_logit"], batch["quality"])
    total = segmentation + PRESENCE_LOSS_WEIGHT * presence + QUALITY_LOSS_WEIGHT * quality
    return total, {
        "total": float(total.detach()),
        "segmentation_bce": float(bce.detach()),
        "dice_loss": float(dice.detach()),
        "presence_focal": float(presence.detach()),
        "quality_bce": float(quality.detach()),
    }


@torch.no_grad()
def validation_summary(
    model: MarsV3Model, loader: DataLoader[dict[str, Any]], device: torch.device
) -> dict[str, Any]:
    model.eval()
    labels: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    positive_dice: list[np.ndarray] = []
    for batch in loader:
        batch = move_batch(batch, device)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            output = model(batch["inputs"], batch["observable"])
        labels.append(batch["presence"].cpu().numpy())
        scores.append(torch.sigmoid(output["presence_logit"]).float().cpu().numpy())
        probability = torch.sigmoid(output["segmentation_logits"]).float()
        truth = batch["mask"] * batch["observable"]
        prediction = probability * batch["observable"]
        intersection = (prediction * truth).sum(dim=(-2, -1))
        denominator = prediction.sum(dim=(-2, -1)) + truth.sum(dim=(-2, -1))
        dice = ((2.0 * intersection + 1.0) / (denominator + 1.0))[:, 0]
        positive = batch["presence"] > 0.5
        if torch.any(positive):
            positive_dice.append(dice[positive].cpu().numpy())
    y = np.concatenate(labels).astype(np.uint8)
    p = np.concatenate(scores)
    threshold, operating = choose_upper_threshold(y, p)
    return {
        "presence_threshold": threshold,
        "recall_at_fpr5": operating["recall"],
        "observed_fpr": operating["false_positive_rate"],
        "presence_auroc": float(roc_auc_score(y, p)),
        "presence_average_precision": float(average_precision_score(y, p)),
        "positive_soft_dice": float(np.mean(np.concatenate(positive_dice))),
    }


@torch.no_grad()
def collect_predictions(
    model: MarsV3Model, loader: DataLoader[dict[str, Any]], device: torch.device
) -> dict[str, np.ndarray]:
    model.eval()
    values: dict[str, list[Any]] = {
        "presence": [],
        "quality": [],
        "labels": [],
        "quality_labels": [],
        "segmentation": [],
        "observable": [],
        "truth": [],
        "groups": [],
        "sample_ids": [],
    }
    for batch in loader:
        batch = move_batch(batch, device)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            output = model(batch["inputs"], batch["observable"])
        values["presence"].append(
            torch.sigmoid(output["presence_logit"]).float().cpu().numpy()
        )
        values["quality"].append(
            torch.sigmoid(output["quality_logit"]).float().cpu().numpy()
        )
        values["labels"].append(batch["presence"].cpu().numpy())
        values["quality_labels"].append(batch["quality"].cpu().numpy())
        values["segmentation"].append(
            torch.sigmoid(output["segmentation_logits"])
            .float()
            .cpu()
            .numpy()[:, 0]
            .astype(np.float16)
        )
        values["observable"].extend(
            np.packbits(item[0].cpu().numpy().astype(bool).ravel())
            for item in batch["observable"]
        )
        values["truth"].extend(
            np.packbits(item[0].cpu().numpy().astype(bool).ravel()) for item in batch["mask"]
        )
        values["groups"].extend(batch["group_id"])
        values["sample_ids"].extend(batch["sample_id"])
    return {
        "presence": np.concatenate(values["presence"]),
        "quality": np.concatenate(values["quality"]),
        "labels": np.concatenate(values["labels"]).astype(np.uint8),
        "quality_labels": np.concatenate(values["quality_labels"]).astype(np.uint8),
        "segmentation": np.concatenate(values["segmentation"]),
        "observable": np.stack(values["observable"]),
        "truth": np.stack(values["truth"]),
        "groups": np.asarray(values["groups"]),
        "sample_ids": np.asarray(values["sample_ids"]),
    }


def training_weights(records: list[dict[str, Any]]) -> list[float]:
    label_counts = Counter(str(record["label_state"]) for record in records)
    group_counts = Counter(str(record["group_id"]) for record in records)
    return [
        (0.5 / label_counts[str(record["label_state"])])
        / math.sqrt(group_counts[str(record["group_id"])])
        for record in records
    ]


def stratified_smoke(records: list[dict[str, Any]], size: int, seed: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for label in ("PLUME", "NO_PLUME"):
        candidates = [record for record in records if record["label_state"] == label]
        candidates.sort(
            key=lambda record: hashlib.sha256(
                f"{seed}:{record['group_id']}:{record['sample_id']}".encode()
            ).hexdigest()
        )
        selected.extend(candidates[: size // 2])
    selected.sort(key=lambda record: str(record["sample_id"]))
    if len(selected) != size:
        raise ValueError(f"Could not build balanced smoke subset of {size} samples")
    return selected


def train(
    model: MarsV3Model,
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
        validation = validation_summary(model, validation_loader, device)
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
            float(validation["recall_at_fpr5"] or 0.0),
            float(validation["presence_average_precision"]),
            float(validation["positive_soft_dice"]),
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
        raise RuntimeError("V3 training produced no checkpoint")
    payload = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(payload["state_dict"], strict=True)
    return history, best_epoch


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    validation = report["validation"]["scene"]
    lines = [
        "# ERSRR MARS v3 validation result",
        "",
        "Pipeline smoke test only; not an accuracy result."
        if report["smoke_test"]
        else "Validation-selected result on the frozen internal groups; strict test remains untouched.",
        "",
        f"- Model: `{report['model']['model_name']}` / {report['model']['parameter_count']:,} parameters",
        f"- Samples: {report['cohort']['training']} train / {report['cohort']['validation']} validation",
        f"- Best epoch: {report['training']['best_epoch']} / {len(report['training']['history'])}",
        f"- Validation recall at FPR <= 0.05: {validation['recall']:.3f} (FPR {validation['false_positive_rate']:.3f})",
        f"- Validation AUROC / AP: {validation['auroc']:.3f} / {validation['average_precision']:.3f}",
        f"- Validation mask Dice: {report['validation']['segmentation']['pixel_dice']:.3f}",
        f"- Checkpoint SHA-256: `{report['artifact']['sha256']}`",
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
    parser.add_argument("--checkpoint")
    parser.add_argument("--output-json")
    parser.add_argument("--output-markdown")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    root = repo_root()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for v3 training")
    seed_everything(args.seed)
    metadata_dir = checked_output_dir(root, args.metadata_dir)
    metadata_csv = (root / args.metadata_csv).resolve()
    manifest_name = args.manifest_file or (DEV_SAMPLES if args.smoke else V3_SAMPLES)
    manifest = metadata_dir / manifest_name
    records = list(iter_manifest(manifest))
    by_role = {
        role: [record for record in records if record["research_role"] == role]
        for role in ("internal_training", "internal_validation")
    }
    if args.smoke:
        by_role["internal_training"] = stratified_smoke(
            by_role["internal_training"], 64, args.seed
        )
        by_role["internal_validation"] = stratified_smoke(
            by_role["internal_validation"], 32, args.seed + 1
        )
        args.epochs = 1
        args.patience = 1
        args.batch_size = min(args.batch_size, 4)
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
    winds = wind_lookup(metadata_csv, required_ids)
    train_dataset = MarsV3Dataset(
        metadata_dir, by_role["internal_training"], winds, augment=True, seed=args.seed
    )
    validation_dataset = MarsV3Dataset(
        metadata_dir, by_role["internal_validation"], winds, augment=False, seed=args.seed
    )
    weights = training_weights(by_role["internal_training"])
    sampler = WeightedRandomSampler(
        weights,
        num_samples=len(weights),
        replacement=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    loader_options = {
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
        root, args.output_json or (SMOKE_JSON.as_posix() if args.smoke else DEFAULT_JSON.as_posix())
    )
    output_markdown = safe_output(
        root,
        args.output_markdown
        or (SMOKE_MARKDOWN.as_posix() if args.smoke else DEFAULT_MARKDOWN.as_posix()),
    )
    device = torch.device("cuda")
    model = MarsV3Model().to(device)
    history, best_epoch = train(
        model,
        train_loader,
        validation_loader,
        device,
        checkpoint,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        patience=args.patience,
        seed=args.seed,
    )
    validation = collect_predictions(model, validation_loader, device)
    upper, validation_scene = choose_upper_threshold(
        validation["labels"], validation["presence"]
    )
    validation_weights = role_weights(validation["labels"], "internal_validation")
    lower, lower_selection = choose_lower_threshold(
        validation["labels"], validation["presence"], upper, validation_weights
    )
    quality_threshold, quality_selection = choose_quality_threshold(
        validation["quality_labels"], validation["quality"]
    )
    segmentation_rule, validation_segmentation = select_segmentation_by_dice(
        validation["segmentation"],
        validation["observable"],
        validation["truth"],
        validation["labels"],
    )
    report = {
        "schema_version": 1,
        "scope": "v3_pipeline_smoke" if args.smoke else "v3_internal_validation_selection",
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
        "training": {
            "seed": args.seed,
            "best_epoch": best_epoch,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "history": history,
            "small_plume_presence_weighting": "sqrt(1000/(valid_truth_pixels+1)), clipped [0.75,3]",
        },
        "operating_rule": {
            "selected_on": "internal_validation_only",
            "upper_plume_threshold": upper,
            "lower_no_plume_threshold": lower,
            "lower_selection": lower_selection,
            "quality_threshold": quality_threshold,
            "quality_selection": quality_selection,
            "segmentation": segmentation_rule,
        },
        "validation": {
            "scene": validation_scene,
            "segmentation": validation_segmentation,
            "presence_auroc": float(
                roc_auc_score(validation["labels"], validation["presence"])
            ),
            "presence_average_precision": float(
                average_precision_score(validation["labels"], validation["presence"])
            ),
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
            "script": "tools/train_mars_v3.py",
            "script_sha256": sha256(Path(__file__)),
            "model_source": "EarthRemoteSensingRapidResponse/mars_v3_model.py",
            "model_source_sha256": sha256(MODEL_ROOT / "mars_v3_model.py"),
        },
        "decision": (
            "Pipeline contract passes. Do not interpret smoke metrics; acquire the frozen full fit/validation corpus before architecture selection."
            if args.smoke
            else "Freeze this validation-selected checkpoint before any strict-spatial evaluation."
        ),
    }
    write_json(output_json, report)
    write_markdown(output_markdown, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
