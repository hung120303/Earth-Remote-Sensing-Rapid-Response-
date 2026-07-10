#!/usr/bin/env python3
"""Train and evaluate the frozen ERSRR MARS joint model on development roles."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import sklearn
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "EarthRemoteSensingRapidResponse"
if str(MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_ROOT))

from mars_joint_model import MODEL_NAME, MarsJointModel  # noqa: E402
from mars_s2l_adapter import iter_manifest, load_sample  # noqa: E402

from acquire_mars_metadata import DEFAULT_OUTPUT, REVISION, checked_output_dir, repo_root, sha256  # noqa: E402
from build_mars_dev_cohort import DEV_SAMPLES, DEFAULT_JSON as DEV_REPORT_JSON  # noqa: E402
from run_mars_dev_pixel_baselines import evaluate_rule, select_rule  # noqa: E402
from run_mars_dev_scene_baselines import (  # noqa: E402
    bootstrap_ci,
    choose_lower_threshold,
    choose_upper_threshold,
    metrics,
    role_weights,
)

DEFAULT_JSON = Path("reports/experiments/mars_joint_development.json")
DEFAULT_MARKDOWN = Path("reports/experiments/MARS_JOINT_DEVELOPMENT.md")
DEFAULT_CHECKPOINT = Path(
    "EarthRemoteSensingRapidResponse/artifacts/mars_joint_development_v1_seed101.pt"
)
DEFAULT_SEED = 101
QUALITY_OBSERVABLE_FRACTION = 0.99
SEGMENTATION_POSITIVE_WEIGHT = 20.0
PRESENCE_LOSS_WEIGHT = 0.75
QUALITY_LOSS_WEIGHT = 0.15
DICE_LOSS_WEIGHT = 0.5


def git_commit(root: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


def tracked_dirty(root: Path) -> bool:
    output = subprocess.check_output(
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
    return bool(output.strip())


def safe_output(root: Path, value: str) -> Path:
    result = (root / value).resolve()
    if root not in result.parents:
        raise ValueError("Output must resolve beneath the repository root")
    return result


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class MarsJointDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        metadata_dir: Path,
        records: list[dict[str, Any]],
        *,
        augment: bool,
        seed: int,
    ) -> None:
        self.metadata_dir = metadata_dir
        self.records = records
        self.augment = augment
        self.seed = seed

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        sample = load_sample(self.metadata_dir, record)
        target = sample.target.copy()
        reference = sample.reference.copy()
        mbmp = np.clip(1.0 - sample.mbmp_valid_aware, -0.5, 0.5)[None, ...].copy()
        observable = sample.observable_mask[None, ...].astype(np.float32)
        mask = sample.plume_mask[None, ...].astype(np.float32)
        if self.augment:
            rng = np.random.default_rng(self.seed + index + int(torch.initial_seed() % 1_000_003))
            rotation = int(rng.integers(0, 4))
            if rotation:
                target = np.rot90(target, rotation, axes=(1, 2)).copy()
                reference = np.rot90(reference, rotation, axes=(1, 2)).copy()
                mbmp = np.rot90(mbmp, rotation, axes=(1, 2)).copy()
                observable = np.rot90(observable, rotation, axes=(1, 2)).copy()
                mask = np.rot90(mask, rotation, axes=(1, 2)).copy()
            if bool(rng.integers(0, 2)):
                target = target[:, :, ::-1].copy()
                reference = reference[:, :, ::-1].copy()
                mbmp = mbmp[:, :, ::-1].copy()
                observable = observable[:, :, ::-1].copy()
                mask = mask[:, :, ::-1].copy()
            if bool(rng.integers(0, 2)):
                target = target[:, ::-1, :].copy()
                reference = reference[:, ::-1, :].copy()
                mbmp = mbmp[:, ::-1, :].copy()
                observable = observable[:, ::-1, :].copy()
                mask = mask[:, ::-1, :].copy()
        observed_fraction = float(np.mean(observable))
        return {
            "target": torch.from_numpy(target),
            "reference": torch.from_numpy(reference),
            "mbmp": torch.from_numpy(mbmp),
            "observable": torch.from_numpy(observable),
            "mask": torch.from_numpy(mask),
            "presence": torch.tensor(sample.presence, dtype=torch.float32),
            "quality": torch.tensor(
                observed_fraction >= QUALITY_OBSERVABLE_FRACTION, dtype=torch.float32
            ),
            "sample_id": sample.sample_id,
            "group_id": record["group_id"],
        }


def focal_presence_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probabilities = torch.sigmoid(logits)
    correct_probability = probabilities * targets + (1.0 - probabilities) * (1.0 - targets)
    return ((1.0 - correct_probability).pow(2.0) * bce).mean()


def masked_segmentation_loss(
    logits: torch.Tensor, targets: torch.Tensor, observable: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    positive_weight = torch.tensor(
        SEGMENTATION_POSITIVE_WEIGHT, dtype=logits.dtype, device=logits.device
    )
    bce = F.binary_cross_entropy_with_logits(
        logits, targets, reduction="none", pos_weight=positive_weight
    )
    bce = (bce * observable).sum() / observable.sum().clamp_min(1.0)
    probabilities = torch.sigmoid(logits) * observable
    masked_targets = targets * observable
    intersection = (probabilities * masked_targets).sum(dim=(-2, -1))
    denominator = probabilities.sum(dim=(-2, -1)) + masked_targets.sum(dim=(-2, -1))
    dice = (2.0 * intersection + 1.0) / (denominator + 1.0)
    dice_loss = 1.0 - dice.mean()
    return bce + DICE_LOSS_WEIGHT * dice_loss, bce, dice_loss


def quality_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    positive = targets.sum().clamp_min(1.0)
    negative = (1.0 - targets).sum().clamp_min(1.0)
    weights = torch.where(targets > 0.5, 0.5 / positive, 0.5 / negative) * targets.numel()
    return (
        F.binary_cross_entropy_with_logits(logits, targets, reduction="none") * weights
    ).mean()


def total_loss(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
    segmentation, bce, dice = masked_segmentation_loss(
        outputs["segmentation_logits"], batch["mask"], batch["observable"]
    )
    presence = focal_presence_loss(outputs["presence_logit"], batch["presence"])
    quality = quality_loss(outputs["quality_logit"], batch["quality"])
    total = segmentation + PRESENCE_LOSS_WEIGHT * presence + QUALITY_LOSS_WEIGHT * quality
    return total, {
        "total": float(total.detach()),
        "segmentation_bce": float(bce.detach()),
        "dice_loss": float(dice.detach()),
        "presence_focal": float(presence.detach()),
        "quality_bce": float(quality.detach()),
    }


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def forward_batch(model: nn.Module, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
    return model(batch["target"], batch["reference"], batch["mbmp"], batch["observable"])


@torch.no_grad()
def validation_summary(
    model: nn.Module, loader: DataLoader[dict[str, Any]], device: torch.device
) -> dict[str, Any]:
    model.eval()
    truth: list[np.ndarray] = []
    presence_scores: list[np.ndarray] = []
    quality_truth: list[np.ndarray] = []
    quality_scores: list[np.ndarray] = []
    losses: list[float] = []
    for batch in loader:
        batch = move_batch(batch, device)
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            outputs = forward_batch(model, batch)
            loss, _ = total_loss(outputs, batch)
        losses.append(float(loss))
        truth.append(batch["presence"].cpu().numpy())
        presence_scores.append(torch.sigmoid(outputs["presence_logit"]).float().cpu().numpy())
        quality_truth.append(batch["quality"].cpu().numpy())
        quality_scores.append(torch.sigmoid(outputs["quality_logit"]).float().cpu().numpy())
    y = np.concatenate(truth)
    p = np.concatenate(presence_scores)
    qy = np.concatenate(quality_truth)
    qp = np.concatenate(quality_scores)
    return {
        "loss": float(np.mean(losses)),
        "presence_average_precision": float(average_precision_score(y, p)),
        "presence_auroc": float(roc_auc_score(y, p)),
        "quality_auroc": float(roc_auc_score(qy, qp)),
    }


def train_model(
    model: MarsJointModel,
    train_loader: DataLoader[dict[str, Any]],
    val_loader: DataLoader[dict[str, Any]],
    device: torch.device,
    checkpoint: Path,
    *,
    epochs: int,
    learning_rate: float,
    seed: int,
    patience: int,
) -> tuple[list[dict[str, Any]], int]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history: list[dict[str, Any]] = []
    best_epoch = -1
    best_score = -math.inf
    stale = 0
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses: list[dict[str, float]] = []
        started = time.perf_counter()
        for batch in train_loader:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"
            ):
                outputs = forward_batch(model, batch)
                loss, parts = total_loss(outputs, batch)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            epoch_losses.append(parts)
        scheduler.step()
        validation = validation_summary(model, val_loader, device)
        training = {
            key: float(np.mean([item[key] for item in epoch_losses]))
            for key in epoch_losses[0]
        }
        record = {
            "epoch": epoch,
            "seconds": round(time.perf_counter() - started, 3),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "training": training,
            "validation": validation,
        }
        history.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
        score = validation["presence_average_precision"]
        if score > best_score + 1e-5:
            best_score = score
            best_epoch = epoch
            stale = 0
            temporary = checkpoint.with_suffix(checkpoint.suffix + ".tmp")
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "model_metadata": model.artifact_metadata(),
                    "seed": seed,
                    "epoch": epoch,
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
        raise RuntimeError("Training did not produce a checkpoint")
    checkpoint_payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint_payload["state_dict"])
    return history, best_epoch


@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader: DataLoader[dict[str, Any]],
    device: torch.device,
) -> dict[str, np.ndarray]:
    model.eval()
    presence: list[np.ndarray] = []
    quality: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    quality_labels: list[np.ndarray] = []
    segmentation: list[np.ndarray] = []
    observable: list[np.ndarray] = []
    truth: list[np.ndarray] = []
    groups: list[str] = []
    sample_ids: list[str] = []
    for batch in loader:
        batch = move_batch(batch, device)
        with torch.amp.autocast(
            device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"
        ):
            outputs = forward_batch(model, batch)
        presence.append(torch.sigmoid(outputs["presence_logit"]).float().cpu().numpy())
        quality.append(torch.sigmoid(outputs["quality_logit"]).float().cpu().numpy())
        labels.append(batch["presence"].cpu().numpy())
        quality_labels.append(batch["quality"].cpu().numpy())
        segmentation.append(
            torch.sigmoid(outputs["segmentation_logits"]).float().cpu().numpy()[:, 0].astype(np.float16)
        )
        observable.extend(
            np.packbits(value[0].cpu().numpy().astype(bool).ravel())
            for value in batch["observable"]
        )
        truth.extend(
            np.packbits(value[0].cpu().numpy().astype(bool).ravel()) for value in batch["mask"]
        )
        groups.extend(batch["group_id"])
        sample_ids.extend(batch["sample_id"])
    return {
        "presence": np.concatenate(presence),
        "quality": np.concatenate(quality),
        "labels": np.concatenate(labels).astype(np.uint8),
        "quality_labels": np.concatenate(quality_labels).astype(np.uint8),
        "segmentation": np.concatenate(segmentation),
        "observable": np.stack(observable),
        "truth": np.stack(truth),
        "groups": np.asarray(groups),
        "sample_ids": np.asarray(sample_ids),
    }


def choose_quality_threshold(labels: np.ndarray, scores: np.ndarray) -> tuple[float, dict[str, Any]]:
    candidates = np.unique(scores)
    best: tuple[tuple[float, ...], float, dict[str, Any]] | None = None
    for threshold in candidates:
        prediction = scores >= threshold
        tp = int(np.sum((labels == 1) & prediction))
        fn = int(np.sum((labels == 1) & ~prediction))
        tn = int(np.sum((labels == 0) & ~prediction))
        fp = int(np.sum((labels == 0) & prediction))
        recall = tp / (tp + fn) if tp + fn else 0.0
        specificity = tn / (tn + fp) if tn + fp else 0.0
        feasible = 1.0 if recall >= 0.95 else 0.0
        candidate = (
            (feasible, specificity if feasible else recall, recall, threshold),
            float(threshold),
            {
                "recall_high_quality": recall,
                "specificity_low_quality": specificity,
                "tp": tp,
                "tn": tn,
                "fp": fp,
                "fn": fn,
            },
        )
        if best is None or candidate[0] > best[0]:
            best = candidate
    if best is None:
        raise RuntimeError("Could not select a quality threshold")
    return best[1], best[2]


def selective_quality_metrics(
    labels: np.ndarray,
    presence_scores: np.ndarray,
    quality_scores: np.ndarray,
    lower: float | None,
    upper: float,
    quality_threshold: float,
    weights: np.ndarray,
) -> dict[str, Any]:
    quality_ok = quality_scores >= quality_threshold
    plume = (presence_scores >= upper) & quality_ok
    no_plume = (
        np.zeros(labels.shape, dtype=bool)
        if lower is None
        else (presence_scores <= lower) & quality_ok
    )
    accepted = plume | no_plume
    tp = float(np.sum(weights[(labels == 1) & plume]))
    fp = float(np.sum(weights[(labels == 0) & plume]))
    tn = float(np.sum(weights[(labels == 0) & no_plume]))
    fn = float(np.sum(weights[(labels == 1) & no_plume]))
    total = float(np.sum(weights))
    return {
        "weighted_coverage": float(np.sum(weights[accepted]) / total),
        "sample_coverage": float(np.mean(accepted)),
        "abstention_rate": float(1.0 - np.mean(accepted)),
        "accepted_samples": int(np.sum(accepted)),
        "accepted_plume": int(np.sum(plume)),
        "accepted_no_plume": int(np.sum(no_plume)),
        "accepted_precision": None if tp + fp == 0 else tp / (tp + fp),
        "accepted_no_plume_npv": None if tn + fn == 0 else tn / (tn + fn),
        "accepted_error_rate": None if tp + tn + fp + fn == 0 else (fp + fn) / (tp + tn + fp + fn),
        "weighted_confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    scene = report["test"]["scene_unweighted"]
    ci = report["test"]["group_bootstrap"]
    segmentation = report["test"]["segmentation"]
    selective = report["test"]["selective_with_quality"]
    lines = [
        "# ERSRR MARS joint-model development result",
        "",
        "Single-seed development experiment under the frozen group-disjoint protocol; not a final paper claim.",
        "",
        f"- Model: `{report['model']['model_name']}` / {report['model']['parameter_count']:,} parameters",
        f"- Best epoch: {report['training']['best_epoch']} / {len(report['training']['history'])}",
        f"- Checkpoint SHA-256: `{report['artifact']['sha256']}`",
        f"- Validation-selected presence thresholds: no-plume `{report['operating_rule']['lower_no_plume_threshold']}`, plume `{report['operating_rule']['upper_plume_threshold']:.6f}`",
        f"- Strict-spatial scene recall: {fmt(scene['recall'])}; specificity: {fmt(scene['specificity'])}; FPR: {fmt(scene['false_positive_rate'])}",
        f"- Group-bootstrap recall 95% CI: {ci['recall_95ci'][0]:.3f}-{ci['recall_95ci'][1]:.3f}",
        f"- Segmentation pixel AP: {segmentation['pixel']['average_precision']:.4f}; IoU: {segmentation['pixel']['intersection_over_union']:.4f}",
        f"- Selective weighted coverage: {selective['weighted_coverage']:.3f}; accepted no-plume NPV: {fmt(selective['accepted_no_plume_npv'])}",
        "",
        "## Decision",
        "",
        report["decision"],
        "",
        "The quality head is trained on the predeclared >=99% observable label within the clear-S2 tranche. Non-clear/unobservable scenes are still required before treating learned quality as an operational abstention guarantee.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-dir", default=DEFAULT_OUTPUT.as_posix())
    parser.add_argument("--output-json", default=DEFAULT_JSON.as_posix())
    parser.add_argument("--output-markdown", default=DEFAULT_MARKDOWN.as_posix())
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT.as_posix())
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args()
    root = repo_root()
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the development joint-model experiment")
        seed_everything(args.seed)
        metadata_dir = checked_output_dir(root, args.metadata_dir)
        manifest_path = metadata_dir / DEV_SAMPLES
        dev_report = json.loads((root / DEV_REPORT_JSON).read_text(encoding="utf-8"))
        if sha256(manifest_path) != dev_report["identities"]["sample_manifest_sha256"]:
            raise ValueError("Development sample manifest identity mismatch")
        records = list(iter_manifest(manifest_path))
        by_role = {
            role: [record for record in records if record["research_role"] == role]
            for role in ("internal_training", "internal_validation", "strict_spatial_test")
        }
        train_dataset = MarsJointDataset(
            metadata_dir, by_role["internal_training"], augment=True, seed=args.seed
        )
        val_dataset = MarsJointDataset(
            metadata_dir, by_role["internal_validation"], augment=False, seed=args.seed
        )
        test_dataset = MarsJointDataset(
            metadata_dir, by_role["strict_spatial_test"], augment=False, seed=args.seed
        )
        sample_weights = [
            2.0 if record["label_state"] == "PLUME" else 1.0
            for record in by_role["internal_training"]
        ]
        generator = torch.Generator().manual_seed(args.seed)
        sampler = WeightedRandomSampler(
            sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
            generator=generator,
        )
        loader_options = {
            "batch_size": args.batch_size,
            "num_workers": args.workers,
            "pin_memory": True,
            "persistent_workers": args.workers > 0,
        }
        train_loader = DataLoader(train_dataset, sampler=sampler, **loader_options)
        val_loader = DataLoader(val_dataset, shuffle=False, **loader_options)
        test_loader = DataLoader(test_dataset, shuffle=False, **loader_options)
        device = torch.device("cuda")
        model = MarsJointModel().to(device)
        checkpoint = safe_output(root, args.checkpoint)
        history, best_epoch = train_model(
            model,
            train_loader,
            val_loader,
            device,
            checkpoint,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            seed=args.seed,
            patience=args.patience,
        )
        validation = collect_predictions(model, val_loader, device)
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
        segmentation_rule, validation_segmentation = select_rule(
            validation["segmentation"],
            validation["observable"],
            validation["truth"],
            validation["labels"],
        )
        # The strict spatial benchmark is evaluated only after all validation rules are frozen.
        test = collect_predictions(model, test_loader, device)
        test_weights = role_weights(test["labels"], "strict_spatial_test")
        test_scene_unweighted = metrics(test["labels"], test["presence"], upper)
        test_scene_weighted = metrics(
            test["labels"], test["presence"], upper, weights=test_weights
        )
        test_segmentation = evaluate_rule(
            test["segmentation"],
            test["observable"],
            test["truth"],
            test["labels"],
            test["groups"].astype(str),
            segmentation_rule,
        )
        group_ci = bootstrap_ci(
            test["labels"], test["presence"], test["groups"].astype(str), upper, args.seed
        )
        test_quality = {
            "auroc": float(roc_auc_score(test["quality_labels"], test["quality"])),
            "threshold": quality_threshold,
        }
        selective = selective_quality_metrics(
            test["labels"],
            test["presence"],
            test["quality"],
            lower,
            upper,
            quality_threshold,
            test_weights,
        )
        gate = (
            float(group_ci["recall_95ci"][0]) >= 0.75
            and float(test_scene_unweighted["false_positive_rate"] or 1.0) <= 0.05
            and float(test_scene_unweighted["specificity"] or 0.0) >= 0.95
        )
        decision = (
            "The joint model clears the provisional development gate; run all five fixed seeds and reproduce released baselines before promotion."
            if gate
            else "The joint model does not yet clear the promotion gate. Use validation-only error analysis and hard-negative mining before any backbone expansion; do not tune on the strict benchmark."
        )
        artifact_hash = sha256(checkpoint)
        output_json = safe_output(root, args.output_json)
        output_markdown = safe_output(root, args.output_markdown)
        report = {
            "schema_version": 1,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "scope": "single_seed_joint_model_development_not_final_paper_claim",
            "source": {
                "repository": "UNEP-IMEO/MARS-S2L",
                "revision": REVISION,
                "development_manifest_sha256": sha256(manifest_path),
            },
            "runtime": {
                "torch": torch.__version__,
                "cuda_runtime": torch.version.cuda,
                "device": torch.cuda.get_device_name(0),
                "compute_capability": list(torch.cuda.get_device_capability(0)),
            },
            "model": model.artifact_metadata(),
            "training": {
                "seed": args.seed,
                "epochs_requested": args.epochs,
                "best_epoch": best_epoch,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "early_stopping_patience": args.patience,
                "segmentation_positive_weight": SEGMENTATION_POSITIVE_WEIGHT,
                "loss_weights": {
                    "presence": PRESENCE_LOSS_WEIGHT,
                    "quality": QUALITY_LOSS_WEIGHT,
                    "dice": DICE_LOSS_WEIGHT,
                },
                "history": history,
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
            },
            "test": {
                "scene_unweighted": test_scene_unweighted,
                "scene_representative_weighted": test_scene_weighted,
                "segmentation": test_segmentation,
                "quality": test_quality,
                "selective_with_quality": selective,
                "group_bootstrap": group_ci,
            },
            "artifact": {
                "path": checkpoint.relative_to(root).as_posix(),
                "bytes": checkpoint.stat().st_size,
                "sha256": artifact_hash,
                "tracked": False,
            },
            "promotion_gate_passed_on_development_tranche": gate,
            "decision": decision,
            "limitations": [
                "Single training seed; final protocol requires five fixed seeds.",
                "Class-enriched development tranche, not final prevalence.",
                "Quality head has only within-clear-tranche supervision; non-clear scenes remain required.",
                "Released MARS-S2L and CH4Net reproduction remains outstanding.",
            ],
            "provenance": {
                "git_commit": git_commit(root),
                "git_tracked_worktree_dirty_at_start": tracked_dirty(root),
                "script": "tools/train_mars_joint_model.py",
                "script_sha256": sha256(Path(__file__)),
                "model_source": "EarthRemoteSensingRapidResponse/mars_joint_model.py",
                "model_source_sha256": sha256(MODEL_ROOT / "mars_joint_model.py"),
                "python": sys.version.split()[0],
                "numpy": np.__version__,
                "rasterio": rasterio.__version__,
                "sklearn": sklearn.__version__,
            },
        }
        write_json(output_json, report)
        write_markdown(output_markdown, report)
    except (
        FileNotFoundError,
        OSError,
        RuntimeError,
        ValueError,
        rasterio.errors.RasterioError,
    ) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, indent=None if args.compact else 2))
        return 2
    payload = {
        "ok": True,
        "gate_passed": gate,
        "best_epoch": best_epoch,
        "test_recall": test_scene_unweighted["recall"],
        "test_specificity": test_scene_unweighted["specificity"],
        "test_pixel_ap": test_segmentation["pixel"]["average_precision"],
        "checkpoint_sha256": artifact_hash,
        "output_json": output_json.relative_to(root).as_posix(),
        "output_markdown": output_markdown.relative_to(root).as_posix(),
    }
    print(json.dumps(payload, indent=None if args.compact else 2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
