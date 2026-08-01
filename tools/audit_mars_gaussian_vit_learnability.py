#!/usr/bin/env python3
"""Audit whether the exact Gaussian ViT can learn its synthetic task."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "EarthRemoteSensingRapidResponse"
for path in (MODEL_ROOT, ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from acquire_mars_metadata import sha256  # noqa: E402
from mars_gaussian_vit import GaussianPretrainedViTUNet  # noqa: E402
from train_mars_dinov3_methane_fusion_pilot import partial_auc_pair_loss  # noqa: E402
from train_mars_gaussian_vit_pilot import GaussianSyntheticSceneDataset  # noqa: E402
from train_mars_paper_residual import iter_development_manifest, move_batch  # noqa: E402
from train_mars_physics_guided_teacher_pilot import seed_everything  # noqa: E402
from train_methanes2cm_v5 import segmentation_first_loss  # noqa: E402


DEFAULT_PROTOCOL = Path("configs/mars_gaussian_vit_learnability_audit.json")


class CachedDataset(Dataset[dict[str, Any]]):
    """Materialize the small deterministic bank once to isolate optimization."""

    def __init__(self, source: Dataset[dict[str, Any]], name: str) -> None:
        self.rows: list[dict[str, Any]] = []
        for index in range(len(source)):
            row = source[index]
            for key, value in row.items():
                if torch.is_tensor(value) and not torch.isfinite(value).all():
                    raise ValueError(f"Non-finite {name} tensor {key} at index {index}")
            self.rows.append(row)
            if (index + 1) % 64 == 0:
                print(json.dumps({"progress": "cache", "split": name, "rows": index + 1}), flush=True)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


def verify_protocol(protocol_path: Path, protocol: dict[str, Any]) -> dict[str, Path]:
    if protocol["status"] != "frozen_before_outcomes":
        raise ValueError("Learnability audit must be frozen before outcomes")
    if sha256(Path(__file__).resolve()) != protocol["trainer_sha256"]:
        raise ValueError("Frozen learnability trainer hash mismatch")
    paths: dict[str, Path] = {}
    for name, contract in protocol["inputs"].items():
        path = (ROOT / contract["path"]).resolve()
        if not path.exists():
            raise ValueError(f"Missing input: {name}")
        if path.is_file() and sha256(path) != contract["sha256"]:
            raise ValueError(f"Input hash mismatch: {name}")
        paths[name] = path
    return paths


def finite_batch(batch: dict[str, Any], *, phase: str) -> None:
    for key, value in batch.items():
        if torch.is_tensor(value) and not torch.isfinite(value).all():
            ids = list(batch.get("sample_id", []))
            raise FloatingPointError(f"Non-finite {phase} batch key={key} samples={ids}")


@torch.no_grad()
def evaluate(
    model: GaussianPretrainedViTUNet,
    dataset: Dataset[dict[str, Any]],
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    labels: list[float] = []
    scores: list[float] = []
    mbmp_scores: list[float] = []
    intersection = 0
    union = 0
    true_pixels = 0
    predicted_pixels = 0
    for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0):
        finite_batch(batch, phase="evaluation_input")
        batch = move_batch(batch, device)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            output = model(batch["inputs"], batch["observable"], batch["sensor_index"])
        finite_batch(output, phase="evaluation_output")
        probability = torch.sigmoid(output["segmentation_logits"].float())
        prediction = (probability >= 0.5) & (batch["observable"] > 0.5)
        truth = (batch["mask"] > 0.5) & (batch["observable"] > 0.5)
        positives = batch["presence"] > 0.5
        intersection += int((prediction[positives] & truth[positives]).sum().item())
        union += int((prediction[positives] | truth[positives]).sum().item())
        true_pixels += int(truth[positives].sum().item())
        predicted_pixels += int(prediction[positives].sum().item())
        labels.extend(float(value) for value in batch["presence"].cpu())
        scores.extend(float(value) for value in output["scene_logit"].float().cpu())
        mbmp = batch["inputs"][:, 0].flatten(1).float()
        top_count = max(1, int(np.ceil(mbmp.shape[1] * 0.01)))
        mbmp_scores.extend(float(value) for value in torch.topk(mbmp, top_count, dim=1).values.mean(1).cpu())
    labels_array = np.asarray(labels, dtype=np.int64)
    scores_array = np.asarray(scores, dtype=np.float64)
    mbmp_array = np.asarray(mbmp_scores, dtype=np.float64)
    positive_scores = scores_array[labels_array == 1]
    negative_scores = scores_array[labels_array == 0]
    return {
        "rows": int(len(labels_array)),
        "scene_average_precision": float(average_precision_score(labels_array, scores_array)),
        "scene_auroc": float(roc_auc_score(labels_array, scores_array)),
        "scene_positive_logit_mean": float(positive_scores.mean()),
        "scene_negative_logit_mean": float(negative_scores.mean()),
        "scene_logit_separation": float(positive_scores.mean() - negative_scores.mean()),
        "mbmp_top1pct_average_precision": float(average_precision_score(labels_array, mbmp_array)),
        "pixel_iou_at_0_5": float(intersection / max(union, 1)),
        "pixel_recall_at_0_5": float(intersection / max(true_pixels, 1)),
        "positive_predicted_pixels": int(predicted_pixels),
        "positive_true_pixels": int(true_pixels),
    }


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    initial = report["checkpoints"][0]
    final = report["checkpoints"][-1]
    lines = [
        "# Gaussian-ViT synthetic learnability audit",
        "",
        f"- Memorization gate: **{report['gates']['memorization']}**",
        f"- Disjoint-validation gate: **{report['gates']['validation_transfer']}**",
        f"- Train scene AP: **{initial['train']['scene_average_precision']:.4f} -> {final['train']['scene_average_precision']:.4f}**",
        f"- Validation scene AP: **{initial['validation']['scene_average_precision']:.4f} -> {final['validation']['scene_average_precision']:.4f}**",
        f"- Train pixel IoU: **{initial['train']['pixel_iou_at_0_5']:.4f} -> {final['train']['pixel_iou_at_0_5']:.4f}**",
        f"- Validation pixel IoU: **{initial['validation']['pixel_iou_at_0_5']:.4f} -> {final['validation']['pixel_iou_at_0_5']:.4f}**",
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
    args = parser.parse_args()
    protocol_path = (ROOT / args.protocol).resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    paths = verify_protocol(protocol_path, protocol)
    fold_protocol = json.loads(paths["fold_protocol"].read_text(encoding="utf-8"))
    group_to_fold = {str(row["group_id"]): int(row["fold"]) for row in fold_protocol["assignments"]}
    records = [
        row
        for row in iter_development_manifest(paths["manifest"])
        if group_to_fold[str(row["group_id"])] in set(protocol["authorized_folds"])
    ]
    negatives = [
        row
        for row in records
        if row["label_state"] == "NO_PLUME"
        and str(row.get("observability", "")).lower() == "clear"
        and 0.5 <= np.hypot(float(row["wind_u"]), float(row["wind_v"])) <= 15.0
    ]
    spec = protocol["training"]
    seed_everything(int(spec["seed"]))
    train_bank = CachedDataset(
        GaussianSyntheticSceneDataset(
            paths["metadata_root"], negatives, paths["transmittance_lut"],
            template_start=int(protocol["bank"]["train_start"]),
            template_count=int(protocol["bank"]["template_count"]),
            seed=int(spec["data_seed"]), crop_size=int(spec["crop_size"]), augment=False,
        ),
        "train",
    )
    validation_bank = CachedDataset(
        GaussianSyntheticSceneDataset(
            paths["metadata_root"], negatives, paths["transmittance_lut"],
            template_start=int(protocol["bank"]["validation_start"]),
            template_count=int(protocol["bank"]["template_count"]),
            seed=int(spec["data_seed"]), crop_size=int(spec["crop_size"]), augment=False,
        ),
        "validation",
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Gaussian-ViT learnability audit requires CUDA")
    torch.cuda.reset_peak_memory_stats()
    model = GaussianPretrainedViTUNet(**protocol["model"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(spec["learning_rate"]), weight_decay=float(spec["weight_decay"])
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        init_scale=float(spec["amp_initial_scale"]),
        growth_interval=int(spec["amp_growth_interval"]),
    )
    generator = torch.Generator().manual_seed(int(spec["loader_seed"]))
    loader = DataLoader(
        train_bank, batch_size=int(spec["batch_size"]), shuffle=True,
        generator=generator, num_workers=0, pin_memory=True,
    )
    checkpoints: list[dict[str, Any]] = []

    def snapshot(epoch: int, history: dict[str, float] | None = None) -> None:
        checkpoints.append({
            "epoch": epoch,
            "history": history,
            "train": evaluate(model, train_bank, int(spec["evaluation_batch_size"]), device),
            "validation": evaluate(model, validation_bank, int(spec["evaluation_batch_size"]), device),
        })
        print(json.dumps({"progress": "checkpoint", "epoch": epoch, "train_ap": checkpoints[-1]["train"]["scene_average_precision"], "validation_ap": checkpoints[-1]["validation"]["scene_average_precision"]}), flush=True)

    snapshot(0)
    checkpoint_epochs = set(map(int, spec["checkpoint_epochs"]))
    for epoch in range(1, int(spec["epochs"]) + 1):
        model.train()
        started = time.perf_counter()
        sums = {"loss": 0.0, "base": 0.0, "partial_auc": 0.0}
        batches = 0
        for batch in loader:
            finite_batch(batch, phase=f"train_epoch_{epoch}_input")
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                output = model(batch["inputs"], batch["observable"], batch["sensor_index"])
                finite_batch(output, phase=f"train_epoch_{epoch}_output")
                base, _ = segmentation_first_loss(
                    output, batch,
                    hard_negative_fraction=float(spec["hard_negative_fraction"]),
                    scene_weight=float(spec["scene_weight"]),
                )
                partial = partial_auc_pair_loss(
                    output["scene_logit"], batch["presence"],
                    negative_fraction=float(spec["partial_auc_negative_fraction"]),
                    margin=float(spec["partial_auc_margin"]),
                )
                loss = base + float(spec["partial_auc_weight"]) * partial
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch}: {list(batch['sample_id'])}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(spec["gradient_clip"]))
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError(f"Non-finite gradient at epoch {epoch}: {list(batch['sample_id'])}")
            scaler.step(optimizer)
            scaler.update()
            sums["loss"] += float(loss.detach())
            sums["base"] += float(base.detach())
            sums["partial_auc"] += float(partial.detach())
            batches += 1
        history = {
            "seconds": time.perf_counter() - started,
            "amp_scale": float(scaler.get_scale()),
            **{key: value / max(batches, 1) for key, value in sums.items()},
        }
        print(json.dumps({"progress": "epoch", "epoch": epoch, **history}), flush=True)
        if epoch in checkpoint_epochs:
            snapshot(epoch, history)

    initial, final = checkpoints[0], checkpoints[-1]
    thresholds = protocol["gates"]
    gates = {
        "memorization": bool(
            final["train"]["scene_average_precision"] >= float(thresholds["train_scene_ap_min"])
            and final["train"]["pixel_iou_at_0_5"] >= float(thresholds["train_pixel_iou_min"])
        ),
        "validation_transfer": bool(
            final["validation"]["scene_average_precision"] >= float(thresholds["validation_scene_ap_min"])
            and final["validation"]["pixel_iou_at_0_5"] - initial["validation"]["pixel_iou_at_0_5"]
            >= float(thresholds["validation_pixel_iou_gain_min"])
        ),
    }
    if not gates["memorization"]:
        decision = "Stop: architecture/loss path cannot memorize the bounded synthetic bank."
    elif not gates["validation_transfer"]:
        decision = "Memorization passes but disjoint-template transfer does not; increase diversity before exposure."
    else:
        decision = "Both gates pass; a separately frozen exposure-scaling experiment is justified."
    report = {
        "schema_version": 1,
        "status": "completed",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": protocol["scope"],
        "protocol": protocol_path.relative_to(ROOT).as_posix(),
        "protocol_sha256": sha256(protocol_path),
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip(),
        "device": torch.cuda.get_device_name(device),
        "peak_cuda_bytes": int(torch.cuda.max_memory_allocated()),
        "eligible_backgrounds": len(negatives),
        "checkpoints": checkpoints,
        "gates": gates,
        "decision": decision,
        "artifact": None,
        "closed_data": protocol["closed_data"],
    }
    json_path = (ROOT / protocol["outputs"]["json"]).resolve()
    atomic_json(json_path, report)
    write_markdown((ROOT / protocol["outputs"]["markdown"]).resolve(), report)
    print(json.dumps({"ok": True, "gates": gates, "decision": decision}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
