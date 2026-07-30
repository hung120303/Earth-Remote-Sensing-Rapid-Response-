#!/usr/bin/env python3
"""Run the frozen instance-aware physics-guided teacher pilot."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy import ndimage
from torch.nn import functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "EarthRemoteSensingRapidResponse"
for path in (MODEL_ROOT, ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from acquire_mars_metadata import sha256  # noqa: E402
from mars_instance_guided_teacher import InstanceGuidedTeacherAdapter  # noqa: E402
from mars_paper_model import released_state  # noqa: E402
from train_mars_paper_residual import (  # noqa: E402
    MarsPaperDataset,
    iter_development_manifest,
    move_batch,
    verify_acquisition_receipt,
)
from train_mars_physics_guided_teacher_balanced_pilot import (  # noqa: E402
    balanced_request_weights,
)
from train_mars_physics_guided_teacher_pilot import evaluate, seed_everything  # noqa: E402
from train_mars_source_aligned_residual import (  # noqa: E402
    MarsSourceAlignedDataset,
    offshore_flags,
)


DEFAULT_PROTOCOL = Path("configs/mars_instance_guided_teacher_pilot_protocol.json")


class InstanceTargetDataset(MarsSourceAlignedDataset):
    """Attach component-center heatmaps after all geometric augmentation."""

    @staticmethod
    def center_heatmap(mask: np.ndarray) -> np.ndarray:
        labels, count = ndimage.label(mask > 0.5, structure=np.ones((3, 3), dtype=np.uint8))
        impulses = np.zeros(mask.shape, dtype=np.float32)
        for component_id in range(1, count + 1):
            component = labels == component_id
            distance = ndimage.distance_transform_edt(component)
            if distance.max() <= 0:
                continue
            row, col = np.unravel_index(int(np.argmax(distance)), distance.shape)
            impulses[row, col] = 1.0
        if not np.any(impulses):
            return impulses
        heatmap = ndimage.gaussian_filter(impulses, sigma=2.0, mode="constant")
        return (heatmap / max(float(heatmap.max()), 1e-6)).astype(np.float32)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = super().__getitem__(index)
        mask = item["mask"][0].numpy()
        centers = self.center_heatmap(mask)
        item["component_center"] = torch.from_numpy(centers[None])
        item["component_count"] = torch.tensor(
            int(ndimage.label(mask > 0.5, structure=np.ones((3, 3), dtype=np.uint8))[1])
        )
        return item


def weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return (values * weights).sum() / weights.sum().clamp_min(1e-6)


def instance_loss(
    output: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    spec: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    logits = output["segmentation_logits"]
    target = batch["mask"]
    observable = batch["observable"]
    simulated = batch["simulated"].float()
    sample_weight = torch.where(
        simulated > 0.5,
        torch.full_like(simulated, float(spec["synthetic_pixel_weight"])),
        torch.ones_like(simulated),
    )

    positive_weight = torch.tensor(
        float(spec["positive_pixel_weight"]), device=logits.device, dtype=logits.dtype
    )
    bce = F.binary_cross_entropy_with_logits(
        logits, target, reduction="none", pos_weight=positive_weight
    )
    probability = torch.sigmoid(logits)
    pt = torch.where(target > 0.5, probability, 1.0 - probability)
    enhancement_weight = 1.0 + target * torch.clamp(batch["enhancement"], 0.0, 2000.0) / 1000.0
    focal_pixels = ((1.0 - pt) ** float(spec["focal_gamma"])) * bce * enhancement_weight
    focal_rows = (focal_pixels * observable).flatten(1).sum(dim=1)
    focal_rows /= observable.flatten(1).sum(dim=1).clamp_min(1.0)
    focal = weighted_mean(focal_rows, sample_weight)

    flat_probability = (probability * observable).flatten(1)
    flat_target = (target * observable).flatten(1)
    positive_rows = flat_target.sum(dim=1) > 0
    if bool(positive_rows.any()):
        intersection = (flat_probability[positive_rows] * flat_target[positive_rows]).sum(dim=1)
        denominator = flat_probability[positive_rows].sum(dim=1) + flat_target[positive_rows].sum(dim=1)
        dice_rows = 1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)
        dice = weighted_mean(dice_rows, sample_weight[positive_rows])
    else:
        dice = logits.sum() * 0.0

    object_logits = output["object_logits"]
    coarse_target = F.max_pool2d(target, kernel_size=2, stride=2)
    coarse_center = F.max_pool2d(batch["component_center"], kernel_size=2, stride=2)
    coarse_observable = (
        F.avg_pool2d(observable, kernel_size=2, stride=2) >= 0.9
    ).to(logits.dtype)
    occupancy_bce = F.binary_cross_entropy_with_logits(
        object_logits[:, 0:1],
        coarse_target,
        reduction="none",
        pos_weight=torch.tensor(
            float(spec["object_positive_weight"]), device=logits.device, dtype=logits.dtype
        ),
    )
    occupancy_rows = (occupancy_bce * coarse_observable).flatten(1).sum(dim=1)
    occupancy_rows /= coarse_observable.flatten(1).sum(dim=1).clamp_min(1.0)
    occupancy = weighted_mean(occupancy_rows, sample_weight)

    center_bce = F.binary_cross_entropy_with_logits(
        object_logits[:, 1:2],
        coarse_center,
        reduction="none",
        pos_weight=torch.tensor(
            float(spec["center_positive_weight"]), device=logits.device, dtype=logits.dtype
        ),
    )
    center_rows = (center_bce * coarse_observable).flatten(1).sum(dim=1)
    center_rows /= coarse_observable.flatten(1).sum(dim=1).clamp_min(1.0)
    center = weighted_mean(center_rows, sample_weight)

    real = simulated < 0.5
    if bool(real.any()):
        scene = F.binary_cross_entropy_with_logits(
            output["scene_logit"][real], batch["presence"][real]
        )
        real_logits = output["scene_logit"][real]
        real_presence = batch["presence"][real]
        positive = real_logits[real_presence > 0.5]
        negative = real_logits[real_presence < 0.5]
        pair = (
            F.softplus(float(spec["pair_margin"]) - positive[:, None] + negative[None, :]).mean()
            if positive.numel() and negative.numel()
            else logits.sum() * 0.0
        )
    else:
        scene = logits.sum() * 0.0
        pair = logits.sum() * 0.0

    real_negative = (
        (batch["presence"] < 0.5) & real
    )[:, None, None, None] & (observable > 0.5)
    real_positive = (
        (batch["presence"] > 0.5) & real
    )[:, None, None, None] & (target > 0.5) & (observable > 0.5)
    upward = F.relu(logits - output["baseline_logits"].detach())
    downward = F.relu(output["baseline_logits"].detach() - logits)
    upward_penalty = (upward * real_negative).sum() / real_negative.sum().clamp_min(1)
    downward_penalty = (downward * real_positive).sum() / real_positive.sum().clamp_min(1)
    correction = output["correction_logits"].square().mean()
    gate_sparsity = output["object_gate"].mean()

    total = (
        focal
        + float(spec["dice_weight"]) * dice
        + float(spec["object_weight"]) * occupancy
        + float(spec["center_weight"]) * center
        + float(spec["scene_weight"]) * scene
        + float(spec["pair_weight"]) * pair
        + float(spec["negative_upward_weight"]) * upward_penalty
        + float(spec["positive_downward_weight"]) * downward_penalty
        + float(spec["correction_l2_weight"]) * correction
        + float(spec["gate_sparsity_weight"]) * gate_sparsity
    )
    return total, {
        "loss": float(total.detach()),
        "focal": float(focal.detach()),
        "dice": float(dice.detach()),
        "object": float(occupancy.detach()),
        "center": float(center.detach()),
        "scene_real": float(scene.detach()),
        "pair_real": float(pair.detach()),
        "negative_upward_real": float(upward_penalty.detach()),
        "positive_downward_real": float(downward_penalty.detach()),
        "correction_l2": float(correction.detach()),
        "gate_mean": float(gate_sparsity.detach()),
        "simulated_fraction": float(simulated.mean().detach()),
    }


def train_endpoint(
    model: InstanceGuidedTeacherAdapter,
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
                output = model(batch["inputs"], batch["observable"], batch["sensor_index"])
                loss, parts = instance_loss(output, batch, spec)
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL.as_posix())
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    protocol_path = (ROOT / args.protocol).resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if sha256(Path(__file__).resolve()) != protocol["trainer"]["sha256"]:
        raise ValueError("Frozen instance-pilot trainer hash mismatch")
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
        selected = []
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
    flags = offshore_flags(paths["metadata_csv"], {str(row["sample_id"]) for row in fit_records})
    dataset = InstanceTargetDataset(
        paths["metadata_root"],
        fit_records,
        flags,
        lut_path=paths["lut"],
        augment=True,
        simulation_fraction=float(spec["simulation_fraction"]),
        crop_size=int(spec["crop_size"]),
        seed=int(spec["seed"]),
    )
    samples = 64 if args.smoke else int(spec["samples_per_epoch"])
    workers = 2 if args.smoke else int(spec["loader_workers"])
    batch_size = 4 if args.smoke else int(spec["batch_size"])
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
    train_loader = DataLoader(dataset, sampler=sampler, **options)
    evaluation_loader = DataLoader(
        MarsPaperDataset(
            paths["metadata_root"], held_records, augment=False, seed=int(spec["seed"])
        ),
        shuffle=False,
        **options,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Instance-guided teacher pilot requires CUDA")
    model = InstanceGuidedTeacherAdapter().to(device)
    model.load_released_checkpoint(released_state(paths["released_checkpoint"]))
    with torch.no_grad():
        first = move_batch(next(iter(evaluation_loader)), device)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            initial = model(first["inputs"], first["observable"], first["sensor_index"])
        identity_max = float(initial["correction_logits"].abs().max())
    if identity_max != 0.0:
        raise ValueError(f"Adapter initialization is not exact identity: {identity_max}")

    history = train_endpoint(
        model, train_loader, spec, device, 1 if args.smoke else int(spec["epochs"])
    )
    if args.smoke:
        finite = all(torch.isfinite(value).all() for value in model.trainable_state().values())
        center_rows = sum(
            int(InstanceTargetDataset.center_heatmap(row["mask"][0].numpy()).max() > 0)
            for row in [dataset[index] for index in range(min(16, len(dataset)))]
        )
        print(
            json.dumps(
                {
                    "ok": finite and center_rows > 0,
                    "identity_max_abs": identity_max,
                    "request_mass": request_mass,
                    "trainable_parameters": model.trainable_parameter_count(),
                    "center_target_rows": center_rows,
                    "history": history,
                }
            )
        )
        return 0 if finite and center_rows > 0 else 1

    candidates, identity = evaluate(
        model,
        evaluation_loader,
        [float(value) for value in protocol["search"]["strengths"]],
        device,
        protocol["bootstrap"],
    )
    selected = max(candidates, key=lambda row: row["rank"])
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
        "scope": "instance-aware physics-guided released-U-Net adapter pilot",
        "all_promotion_gates_pass": passed,
        "decision": (
            "Authorize preregistered multi-seed cross-fit."
            if passed
            else "Reject instance-aware adapter before full cross-fit or external scoring."
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
        "trainable_parameters": model.trainable_parameter_count(),
        "identity_max_abs": identity_max,
        "history": history,
        "released_identity": identity,
        "candidates": candidates,
        "selected": selected,
        "artifact": artifact,
        "external_inputs_accessed": False,
    }
    output = (ROOT / protocol["outputs"]["json"]).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "ok": passed,
                "strength": selected["strength"],
                "ap_delta": selected["versus_released"]["delta"]["average_precision"],
                "ap_lower": selected["bootstrap"]["lower"],
                "recall_delta": selected["versus_released"]["delta"][
                    "recall_at_fpr_0_0713"
                ],
                "iou_delta": selected["pixel_iou_delta"],
                "realized_simulation": [
                    row["simulated_fraction"] for row in history
                ],
            },
            indent=2,
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

