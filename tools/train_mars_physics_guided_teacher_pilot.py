#!/usr/bin/env python3
"""Run the frozen physics-guided released-U-Net adapter pilot on development fold 2."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
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
from evaluate_released_marss2l import component_mask, connected_scene_score  # noqa: E402
from mars_paper_model import SENSOR_NAMES, released_state  # noqa: E402
from mars_physics_guided_teacher import PhysicsGuidedTeacherAdapter  # noqa: E402
from train_mars_oof_scene_ensemble_v2 import ap_group_bootstrap  # noqa: E402
from train_mars_paper_residual import (  # noqa: E402
    MarsPaperDataset,
    add_pixels,
    finish_pixels,
    iter_development_manifest,
    move_batch,
    verify_acquisition_receipt,
)
from train_mars_scene_ranker import comparison, metric_summary  # noqa: E402
from train_mars_source_aligned_residual import (  # noqa: E402
    MarsSourceAlignedDataset,
    offshore_flags,
    source_aligned_sampling_weights,
)


DEFAULT_PROTOCOL = Path("configs/mars_physics_guided_teacher_pilot_protocol.json")


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def loss_function(
    output: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], spec: dict[str, Any]
) -> tuple[torch.Tensor, dict[str, float]]:
    logits = output["segmentation_logits"]
    target = batch["mask"]
    observable = batch["observable"]
    pos_weight = torch.tensor(float(spec["positive_pixel_weight"]), device=logits.device)
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none", pos_weight=pos_weight)
    probability = torch.sigmoid(logits)
    pt = torch.where(target > 0.5, probability, 1.0 - probability)
    enhancement_weight = 1.0 + target * torch.clamp(batch["enhancement"], 0.0, 2000.0) / 1000.0
    focal = (((1.0 - pt) ** float(spec["focal_gamma"])) * bce * enhancement_weight * observable).sum()
    focal = focal / observable.sum().clamp_min(1.0)

    flat_probability = (probability * observable).flatten(1)
    flat_target = (target * observable).flatten(1)
    positive_rows = flat_target.sum(dim=1) > 0
    if bool(positive_rows.any()):
        intersection = (flat_probability[positive_rows] * flat_target[positive_rows]).sum(dim=1)
        denominator = flat_probability[positive_rows].sum(dim=1) + flat_target[positive_rows].sum(dim=1)
        dice = (1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)).mean()
    else:
        dice = logits.sum() * 0.0

    scene = F.binary_cross_entropy_with_logits(output["scene_logit"], batch["presence"])
    positive = output["scene_logit"][batch["presence"] > 0.5]
    negative = output["scene_logit"][batch["presence"] < 0.5]
    pair = (
        F.softplus(float(spec["pair_margin"]) - positive[:, None] + negative[None, :]).mean()
        if positive.numel() and negative.numel()
        else logits.sum() * 0.0
    )
    negative_pixels = (batch["presence"] < 0.5)[:, None, None, None] & (observable > 0.5)
    positive_pixels = (target > 0.5) & (observable > 0.5)
    upward = F.relu(logits - output["baseline_logits"].detach())
    downward = F.relu(output["baseline_logits"].detach() - logits)
    upward_penalty = (upward * negative_pixels).sum() / negative_pixels.sum().clamp_min(1)
    downward_penalty = (downward * positive_pixels).sum() / positive_pixels.sum().clamp_min(1)
    correction = output["correction_logits"].square().mean()
    total = (
        focal
        + float(spec["dice_weight"]) * dice
        + float(spec["scene_weight"]) * scene
        + float(spec["pair_weight"]) * pair
        + float(spec["negative_upward_weight"]) * upward_penalty
        + float(spec["positive_downward_weight"]) * downward_penalty
        + float(spec["correction_l2_weight"]) * correction
    )
    values = {
        "loss": float(total.detach()),
        "focal": float(focal.detach()),
        "dice": float(dice.detach()),
        "scene": float(scene.detach()),
        "pair": float(pair.detach()),
        "negative_upward": float(upward_penalty.detach()),
        "positive_downward": float(downward_penalty.detach()),
        "correction_l2": float(correction.detach()),
        "simulated_fraction": float(batch["simulated"].float().mean().detach()),
    }
    return total, values


def train_endpoint(
    model: PhysicsGuidedTeacherAdapter,
    loader: DataLoader[dict[str, Any]],
    spec: dict[str, Any],
    device: torch.device,
    seed: int,
    epochs: int,
) -> list[dict[str, float]]:
    seed_everything(seed)
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
                loss, parts = loss_function(output, batch, spec)
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


@torch.no_grad()
def evaluate(
    model: PhysicsGuidedTeacherAdapter,
    loader: DataLoader[dict[str, Any]],
    strengths: list[float],
    device: torch.device,
    bootstrap: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    model.eval()
    all_strengths = [0.0, *strengths]
    scores = {strength: [] for strength in all_strengths}
    pixels = {strength: {"intersection": 0.0, "predicted": 0.0, "truth": 0.0} for strength in all_strengths}
    labels: list[int] = []
    groups: list[str] = []
    sensors: list[int] = []
    sample_ids: list[str] = []
    for batch in loader:
        local_ids = [str(value) for value in batch["sample_id"]]
        local_groups = [str(value) for value in batch["group_id"]]
        batch = move_batch(batch, device)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            output = model(batch["inputs"], batch["observable"], batch["sensor_index"])
        baseline = output["baseline_logits"]
        correction = output["correction_logits"]
        clear = batch["clear"] > 0.5
        for strength in all_strengths:
            probability = torch.sigmoid(baseline + float(strength) * correction).float()
            probability = probability.masked_fill(~clear, 0.0)
            for index in range(probability.shape[0]):
                score = probability[index, 0].cpu().numpy()
                observable = batch["observable"][index, 0].cpu().numpy() > 0.5
                truth = (batch["mask"][index, 0].cpu().numpy() > 0.5) & observable
                scores[strength].append(connected_scene_score(score))
                add_pixels(pixels[strength], score, truth, observable)
        labels.extend(int(value) for value in batch["presence"].cpu().numpy())
        sensors.extend(int(value) for value in batch["sensor_index"].cpu().numpy())
        groups.extend(local_groups)
        sample_ids.extend(local_ids)
    y = np.asarray(labels, dtype=np.uint8)
    sensor_array = np.asarray(sensors, dtype=np.uint8)
    group_array = np.asarray(groups)
    baseline_scores = np.asarray(scores[0.0], dtype=np.float64)
    baseline_metrics = metric_summary(y, baseline_scores, sensor_array)
    baseline_pixels = finish_pixels(pixels[0.0])
    candidates = []
    for strength in strengths:
        local_scores = np.asarray(scores[strength], dtype=np.float64)
        metrics = metric_summary(y, local_scores, sensor_array)
        versus = comparison(metrics, baseline_metrics)
        local_pixels = finish_pixels(pixels[strength])
        iou_delta = float(local_pixels["intersection_over_union"] - baseline_pixels["intersection_over_union"])
        interval = ap_group_bootstrap(
            y,
            baseline_scores,
            local_scores,
            group_array,
            replicates=int(bootstrap["replicates"]),
            seed=int(bootstrap["seed"]),
        )
        sensor_delta = versus["delta"]["sensor_average_precision"]
        passed = bool(
            versus["delta"]["average_precision"] >= 0.003
            and versus["delta"]["recall_at_fpr_0_0713"] >= 0.0
            and iou_delta > 0.0
            and min(sensor_delta.values()) >= 0.0
            and interval["lower"] > 0.0
        )
        candidates.append(
            {
                "strength": strength,
                "metrics": metrics,
                "versus_released": versus,
                "pixel_fixed_0_5": local_pixels,
                "pixel_iou_delta": iou_delta,
                "bootstrap": interval,
                "passed": passed,
                "rank": [
                    int(passed),
                    min(sensor_delta.values()),
                    versus["delta"]["average_precision"],
                    versus["delta"]["recall_at_fpr_0_0713"],
                    iou_delta,
                    -strength,
                ],
            }
        )
    identity = {
        "rows": len(y),
        "sample_id_sha256": __import__("hashlib").sha256("\n".join(sample_ids).encode()).hexdigest(),
        "released_metrics": baseline_metrics,
        "released_pixel_fixed_0_5": baseline_pixels,
    }
    return candidates, identity


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL.as_posix())
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    protocol_path = (ROOT / args.protocol).resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if sha256(Path(__file__).resolve()) != protocol["trainer"]["sha256"]:
        raise ValueError("Frozen trainer hash mismatch")
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
    group_to_fold = {str(row["group_id"]): int(row["fold"]) for row in fold_protocol["assignments"]}
    records = list(iter_development_manifest(paths["manifest"]))
    fit_folds = set(map(int, protocol["folds"]["fit"]))
    held_fold = int(protocol["folds"]["held"])
    fit_records = [row for row in records if group_to_fold[str(row["group_id"])] in fit_folds]
    held_records = [row for row in records if group_to_fold[str(row["group_id"])] == held_fold]
    spec = protocol["training"]
    if args.smoke:
        selected = []
        strata: dict[tuple[str, str], int] = {}
        for row in fit_records:
            key = (str(row["label_state"]), str(row["sensor_family"]))
            if strata.get(key, 0) < 4:
                selected.append(row)
                strata[key] = strata.get(key, 0) + 1
            if len(strata) == 4 and min(strata.values()) >= 4:
                break
        fit_records = selected
        held_records = selected
    flags = offshore_flags(paths["metadata_csv"], {str(row["sample_id"]) for row in fit_records})
    dataset = MarsSourceAlignedDataset(
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
        source_aligned_sampling_weights(fit_records),
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
        MarsPaperDataset(paths["metadata_root"], held_records, augment=False, seed=int(spec["seed"])),
        shuffle=False,
        **options,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Physics-guided teacher pilot requires CUDA")
    model = PhysicsGuidedTeacherAdapter().to(device)
    model.load_released_checkpoint(released_state(paths["released_checkpoint"]))
    with torch.no_grad():
        first = move_batch(next(iter(evaluation_loader)), device)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            initial = model(first["inputs"], first["observable"], first["sensor_index"])
        identity_max = float(initial["correction_logits"].abs().max())
    if identity_max != 0.0:
        raise ValueError(f"Adapter initialization is not exact identity: {identity_max}")
    epochs = 1 if args.smoke else int(spec["epochs"])
    history = train_endpoint(model, train_loader, spec, device, int(spec["seed"]), epochs)
    if args.smoke:
        with torch.no_grad():
            finite = all(torch.isfinite(value).all() for value in model.trainable_state().values())
        print(json.dumps({"ok": finite, "identity_max_abs": identity_max, "rows": len(fit_records), "trainable_parameters": model.trainable_parameter_count(), "history": history}))
        return 0 if finite else 1
    candidates, identity = evaluate(
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
            },
            artifact_path,
        )
        artifact = {"path": artifact_path.relative_to(ROOT).as_posix(), "bytes": artifact_path.stat().st_size, "sha256": sha256(artifact_path)}
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
    report = {
        "schema_version": 1,
        "scope": "development fold-2 physics-guided released-U-Net adapter pilot",
        "all_promotion_gates_pass": passed,
        "decision": "Authorize full cross-fit protocol." if passed else "Reject adapter pilot before full cross-fit or external scoring.",
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
    print(json.dumps({"ok": passed, "strength": selected["strength"], "ap_delta": selected["versus_released"]["delta"]["average_precision"], "ap_lower": selected["bootstrap"]["lower"], "recall_delta": selected["versus_released"]["delta"]["recall_at_fpr_0_0713"], "iou_delta": selected["pixel_iou_delta"]}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
