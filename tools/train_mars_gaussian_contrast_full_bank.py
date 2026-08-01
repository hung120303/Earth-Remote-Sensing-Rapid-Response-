#!/usr/bin/env python3
"""Train and validate the physics-contrast ViT on the full Gaussian bank."""

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
from torch.utils.data import DataLoader, Sampler

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "EarthRemoteSensingRapidResponse"
for path in (MODEL_ROOT, ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from acquire_mars_metadata import sha256  # noqa: E402
from mars_gaussian_contrast_vit import GaussianContrastViTUNet  # noqa: E402
from train_mars_gaussian_vit_pilot import GaussianSyntheticSceneDataset  # noqa: E402
from train_mars_paper_residual import iter_development_manifest, move_batch  # noqa: E402
from train_mars_physics_guided_teacher_pilot import seed_everything  # noqa: E402
from train_methanes2cm_v5 import segmentation_first_loss  # noqa: E402


DEFAULT_PROTOCOL = Path("configs/mars_gaussian_contrast_full_bank_protocol.json")


class PairShuffleSampler(Sampler[int]):
    """Shuffle templates while keeping each positive/unchanged twin adjacent."""

    def __init__(self, template_count: int, seed: int) -> None:
        self.template_count = int(template_count)
        self.seed = int(seed)
        self.epoch = 0

    def __iter__(self):
        order = np.random.default_rng(self.seed + self.epoch).permutation(self.template_count)
        self.epoch += 1
        for template_index in order:
            yield int(template_index) * 2
            yield int(template_index) * 2 + 1

    def __len__(self) -> int:
        return self.template_count * 2


def verify_protocol(protocol: dict[str, Any]) -> dict[str, Path]:
    if protocol["status"] != "frozen_before_outcomes":
        raise ValueError("Full-bank protocol must be frozen before outcomes")
    if sha256(Path(__file__).resolve()) != protocol["trainer_sha256"]:
        raise ValueError("Frozen full-bank trainer hash mismatch")
    for dependency in protocol["code_dependencies"]:
        path = (ROOT / dependency["path"]).resolve()
        if not path.is_file() or sha256(path) != dependency["sha256"]:
            raise ValueError(f"Code dependency hash mismatch: {dependency['path']}")
    paths: dict[str, Path] = {}
    for name, contract in protocol["inputs"].items():
        path = (ROOT / contract["path"]).resolve()
        if not path.exists():
            raise ValueError(f"Missing input: {name}")
        if path.is_file() and sha256(path) != contract["sha256"]:
            raise ValueError(f"Input hash mismatch: {name}")
        paths[name] = path
    return paths


def finite_tensors(values: dict[str, Any], *, phase: str) -> None:
    for key, value in values.items():
        if torch.is_tensor(value) and not torch.isfinite(value).all():
            raise FloatingPointError(
                f"Non-finite {phase} tensor key={key} samples={list(values.get('sample_id', []))}"
            )


def make_loader(
    dataset: GaussianSyntheticSceneDataset,
    *,
    batch_size: int,
    workers: int,
    shuffle: bool,
    seed: int,
    pair_shuffle: bool = False,
) -> DataLoader[dict[str, Any]]:
    sampler = PairShuffleSampler(dataset.template_count, seed) if pair_shuffle else None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        generator=torch.Generator().manual_seed(seed),
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
    )


@torch.no_grad()
def evaluate_dense(
    model: GaussianContrastViTUNet,
    loader: DataLoader[dict[str, Any]],
    device: torch.device,
    *,
    epoch: int,
) -> dict[str, float]:
    model.eval()
    labels: list[float] = []
    scores: list[float] = []
    intersection = 0
    union = 0
    true_pixels = 0
    predicted_pixels = 0
    started = time.perf_counter()
    for batch_index, batch in enumerate(loader, start=1):
        finite_tensors(batch, phase="validation_input")
        batch = move_batch(batch, device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            output = model(batch["inputs"], batch["observable"], batch["sensor_index"])
        finite_tensors(output, phase="validation_output")
        prediction = (torch.sigmoid(output["segmentation_logits"].float()) >= 0.5) & (
            batch["observable"] > 0.5
        )
        truth = (batch["mask"] > 0.5) & (batch["observable"] > 0.5)
        positives = batch["presence"] > 0.5
        intersection += int((prediction[positives] & truth[positives]).sum().item())
        union += int((prediction[positives] | truth[positives]).sum().item())
        true_pixels += int(truth[positives].sum().item())
        predicted_pixels += int(prediction[positives].sum().item())
        labels.extend(float(value) for value in batch["presence"].cpu())
        scores.extend(float(value) for value in output["top_evidence"].float().cpu())
        if batch_index % 64 == 0:
            print(
                json.dumps(
                    {"progress": "validation_batch", "epoch": epoch, "batch": batch_index}
                ),
                flush=True,
            )
    labels_array = np.asarray(labels, dtype=np.int64)
    scores_array = np.asarray(scores, dtype=np.float64)
    return {
        "rows": int(len(labels_array)),
        "seconds": time.perf_counter() - started,
        "dense_top1pct_average_precision": float(
            average_precision_score(labels_array, scores_array)
        ),
        "dense_top1pct_auroc": float(roc_auc_score(labels_array, scores_array)),
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
    selected = report["selected_epoch"]
    validation = selected["validation"]
    lines = [
        "# Full-bank physics-contrast Gaussian-ViT audit",
        "",
        f"- Full-bank gate: **{report['passed']}**",
        f"- Selected epoch: **{selected['epoch']}**",
        f"- Validation dense-evidence AP: **{validation['dense_top1pct_average_precision']:.4f}**",
        f"- Validation dense-evidence AUROC: **{validation['dense_top1pct_auroc']:.4f}**",
        f"- Validation mask IoU: **{validation['pixel_iou_at_0_5']:.4f}**",
        f"- Validation pixel recall: **{validation['pixel_recall_at_0_5']:.4f}**",
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
    paths = verify_protocol(protocol)
    fold_protocol = json.loads(paths["fold_protocol"].read_text(encoding="utf-8"))
    group_to_fold = {
        str(row["group_id"]): int(row["fold"]) for row in fold_protocol["assignments"]
    }
    authorized_folds = set(map(int, protocol["authorized_folds"]))
    records = [
        row
        for row in iter_development_manifest(paths["manifest"])
        if group_to_fold[str(row["group_id"])] in authorized_folds
    ]
    negatives = [
        row
        for row in records
        if row["label_state"] == "NO_PLUME"
        and str(row.get("observability", "")).lower() == "clear"
        and 0.5 <= np.hypot(float(row["wind_u"]), float(row["wind_v"])) <= 15.0
    ]
    spec = protocol["training"]
    bank = protocol["bank"]
    train_templates = 8 if args.smoke else int(bank["train_template_count"])
    validation_templates = 8 if args.smoke else int(bank["validation_template_count"])
    epochs = 1 if args.smoke else int(spec["epochs"])
    seed_everything(int(spec["seed"]))
    train_dataset = GaussianSyntheticSceneDataset(
        paths["metadata_root"],
        negatives,
        paths["transmittance_lut"],
        template_start=int(bank["train_start"]),
        template_count=train_templates,
        seed=int(spec["data_seed"]),
        crop_size=int(spec["crop_size"]),
        augment=bool(spec["augment"]),
    )
    validation_dataset = GaussianSyntheticSceneDataset(
        paths["metadata_root"],
        negatives,
        paths["transmittance_lut"],
        template_start=int(bank["validation_start"]),
        template_count=validation_templates,
        seed=int(spec["data_seed"]),
        crop_size=int(spec["crop_size"]),
        augment=False,
    )
    train_loader = make_loader(
        train_dataset,
        batch_size=int(spec["batch_size"]),
        workers=0 if args.smoke else int(spec["loader_workers"]),
        shuffle=True,
        seed=int(spec["loader_seed"]),
        pair_shuffle=True,
    )
    validation_loader = make_loader(
        validation_dataset,
        batch_size=int(spec["evaluation_batch_size"]),
        workers=0 if args.smoke else int(spec["loader_workers"]),
        shuffle=False,
        seed=int(spec["loader_seed"]) + 1,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Full-bank Gaussian audit requires CUDA")
    torch.cuda.reset_peak_memory_stats()
    model = GaussianContrastViTUNet(**protocol["model"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(spec["learning_rate"]),
        weight_decay=float(spec["weight_decay"]),
    )
    epoch_rows: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        started = time.perf_counter()
        sums = {"loss": 0.0, "positive_bce": 0.0, "hard_negative_bce": 0.0, "positive_dice_loss": 0.0}
        batches = 0
        for batch_index, batch in enumerate(train_loader, start=1):
            finite_tensors(batch, phase="training_input")
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                output = model(batch["inputs"], batch["observable"], batch["sensor_index"])
                finite_tensors(output, phase="training_output")
                loss, parts = segmentation_first_loss(
                    output,
                    batch,
                    hard_negative_fraction=float(spec["hard_negative_fraction"]),
                    scene_weight=0.0,
                )
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite training loss epoch={epoch} samples={list(batch['sample_id'])}"
                )
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(spec["gradient_clip"])
            )
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError(
                    f"Non-finite training gradient epoch={epoch} samples={list(batch['sample_id'])}"
                )
            optimizer.step()
            sums["loss"] += float(loss.detach())
            for key in ("positive_bce", "hard_negative_bce", "positive_dice_loss"):
                sums[key] += float(parts[key])
            batches += 1
            if batch_index % int(spec["progress_every_batches"]) == 0:
                print(
                    json.dumps(
                        {
                            "progress": "training_batch",
                            "epoch": epoch,
                            "batch": batch_index,
                            "elapsed_seconds": time.perf_counter() - started,
                        }
                    ),
                    flush=True,
                )
        training = {
            "seconds": time.perf_counter() - started,
            "batches": batches,
            "requests": len(train_dataset),
            **{key: value / max(batches, 1) for key, value in sums.items()},
        }
        validation = evaluate_dense(model, validation_loader, device, epoch=epoch)
        row = {"epoch": epoch, "training": training, "validation": validation}
        epoch_rows.append(row)
        print(
            json.dumps(
                {
                    "progress": "epoch",
                    "epoch": epoch,
                    "loss": training["loss"],
                    "validation_ap": validation["dense_top1pct_average_precision"],
                    "validation_iou": validation["pixel_iou_at_0_5"],
                }
            ),
            flush=True,
        )
    if args.smoke:
        print(
            json.dumps(
                {
                    "ok": True,
                    "smoke": True,
                    "model": model.artifact_metadata(),
                    "peak_cuda_bytes": int(torch.cuda.max_memory_allocated()),
                    "epoch": epoch_rows[-1],
                },
                sort_keys=True,
            )
        )
        return 0
    selected = max(
        epoch_rows,
        key=lambda row: (
            row["validation"]["pixel_iou_at_0_5"],
            row["validation"]["dense_top1pct_average_precision"],
            -int(row["epoch"]),
        ),
    )
    gates = protocol["gates"]
    passed = bool(
        selected["validation"]["dense_top1pct_average_precision"]
        >= float(gates["validation_dense_ap_min"])
        and selected["validation"]["pixel_iou_at_0_5"]
        >= float(gates["validation_pixel_iou_min"])
        and selected["validation"]["pixel_recall_at_0_5"]
        >= float(gates["validation_pixel_recall_min"])
    )
    decision = (
        "Freeze the full-bank schedule for separately cross-fitted real MARS endpoints."
        if passed
        else "Reject full-bank scaling before any real held-fold outcome."
    )
    report = {
        "schema_version": 1,
        "status": "passed" if passed else "rejected",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": protocol["scope"],
        "protocol": protocol_path.relative_to(ROOT).as_posix(),
        "protocol_sha256": sha256(protocol_path),
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
        ).stdout.strip(),
        "device": torch.cuda.get_device_name(device),
        "peak_cuda_bytes": int(torch.cuda.max_memory_allocated()),
        "eligible_backgrounds": len(negatives),
        "train_templates": train_templates,
        "validation_templates": validation_templates,
        "total_training_requests": int(sum(row["training"]["requests"] for row in epoch_rows)),
        "model": model.artifact_metadata(),
        "epochs": epoch_rows,
        "selection_rule": protocol["selection_rule"],
        "selected_epoch": selected,
        "passed": passed,
        "decision": decision,
        "artifact": None,
        "closed_data": protocol["closed_data"],
    }
    json_path = (ROOT / protocol["outputs"]["json"]).resolve()
    atomic_json(json_path, report)
    write_markdown((ROOT / protocol["outputs"]["markdown"]).resolve(), report)
    print(json.dumps({"ok": True, "passed": passed, "selected_epoch": selected["epoch"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
