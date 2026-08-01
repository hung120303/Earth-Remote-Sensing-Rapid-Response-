#!/usr/bin/env python3
"""Cross-fit the full-bank physics-contrast Gaussian ViT on MARS folds 3/4."""

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

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Sampler, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "EarthRemoteSensingRapidResponse"
for path in (MODEL_ROOT, ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from acquire_mars_metadata import sha256  # noqa: E402
from mars_gaussian_contrast_vit import GaussianContrastViTUNet  # noqa: E402
from train_mars_dinov3_methane_fusion_pilot import (  # noqa: E402
    load_base_score_contract,
    partial_auc_pair_loss,
)
from train_mars_gaussian_contrast_full_bank import PairShuffleSampler  # noqa: E402
from train_mars_gaussian_vit_pilot import (  # noqa: E402
    GaussianSyntheticSceneDataset,
    RealCropDataset,
    _artifact_state,
    _load_released_teacher,
    collect_predictions,
)
from train_mars_ndmi_bitemporal_fusion_pilot import (  # noqa: E402
    merge_predictions,
    summarize_predictions,
)
from train_mars_paper_residual import (  # noqa: E402
    MarsPaperDataset,
    iter_development_manifest,
    move_batch,
    smoke_subset,
    verify_acquisition_receipt,
)
from train_mars_physics_guided_teacher_balanced_pilot import (  # noqa: E402
    balanced_request_weights,
)
from train_mars_physics_guided_teacher_pilot import seed_everything  # noqa: E402
from train_mars_scene_ranker import comparison, metric_summary  # noqa: E402
from train_methanes2cm_v5 import segmentation_first_loss  # noqa: E402


DEFAULT_PROTOCOL = Path("configs/mars_gaussian_contrast_crossfit_protocol.json")


class TransferGaussianContrastViTUNet(GaussianContrastViTUNet):
    """Contrast model with the frozen bounded-residual scene-score contract."""

    def __init__(self, *, scene_protection_gate: float = 0.0, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if not 0.0 <= float(scene_protection_gate) < 1.0:
            raise ValueError("Scene protection gate must lie in [0, 1)")
        self.scene_protection_gate = float(scene_protection_gate)

    def fuse_scene_score(
        self,
        baseline_score: torch.Tensor,
        scene_logit: torch.Tensor,
        strength: float,
    ) -> torch.Tensor:
        baseline = torch.logit(baseline_score.clamp(1e-6, 1 - 1e-6))
        correction = 2.0 * torch.tanh(scene_logit / 2.0)
        candidate = torch.sigmoid(baseline + float(strength) * correction)
        return torch.where(
            baseline_score >= self.scene_protection_gate,
            candidate,
            baseline_score,
        )

    def artifact_metadata(self) -> dict[str, Any]:
        return {
            **super().artifact_metadata(),
            "scene_protection_gate": self.scene_protection_gate,
        }


def finite_tensors(values: dict[str, Any], *, phase: str) -> None:
    """Fail immediately on corrupt inputs or unstable model outputs."""

    for key, value in values.items():
        if torch.is_tensor(value) and not torch.isfinite(value).all():
            raise FloatingPointError(
                f"Non-finite {phase} tensor key={key} samples={list(values.get('sample_id', []))}"
            )


def make_loader(
    dataset: Dataset[dict[str, Any]],
    *,
    batch_size: int,
    workers: int,
    seed: int,
    sampler: Sampler[int] | None = None,
) -> DataLoader[dict[str, Any]]:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        generator=torch.Generator().manual_seed(seed),
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
    )


@torch.no_grad()
def collect_scene_logits(
    model: TransferGaussianContrastViTUNet,
    loader: DataLoader[dict[str, Any]],
    base_scores: dict[str, float],
    device: torch.device,
    fold: int,
) -> dict[str, Any]:
    """Collect compact raw scene evidence without dense candidate accounting."""
    model.eval()
    rows: dict[str, list[Any]] = {
        "labels": [],
        "sensors": [],
        "groups": [],
        "sample_ids": [],
        "folds": [],
        "base_scores": [],
        "raw_scene_logits": [],
    }
    for batch_index, batch in enumerate(loader, start=1):
        sample_ids = [str(value) for value in batch["sample_id"]]
        local_base = np.asarray([base_scores[value] for value in sample_ids], dtype=np.float64)
        batch = move_batch(batch, device)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            output = model(batch["inputs"], batch["observable"], batch["sensor_index"])
        finite_tensors(output, phase="scene_cache_inference")
        rows["labels"].extend(int(value) for value in batch["presence"].cpu().numpy())
        rows["sensors"].extend(int(value) for value in batch["sensor_index"].cpu().numpy())
        rows["groups"].extend(str(value) for value in batch["group_id"])
        rows["sample_ids"].extend(sample_ids)
        rows["folds"].extend([fold] * len(sample_ids))
        rows["base_scores"].extend(float(value) for value in local_base)
        rows["raw_scene_logits"].extend(
            float(value) for value in output["scene_logit"].float().cpu().numpy()
        )
        if batch_index % 64 == 0:
            print(
                json.dumps(
                    {"progress": "scene_cache_inference", "fold": fold, "batch": batch_index}
                ),
                flush=True,
            )
    return rows


def merge_scene_logits(parts: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    if not parts:
        raise ValueError("At least one scene-logit part is required")
    merged = {
        "labels": np.concatenate([np.asarray(part["labels"], dtype=np.uint8) for part in parts]),
        "sensors": np.concatenate([np.asarray(part["sensors"], dtype=np.uint8) for part in parts]),
        "groups": np.concatenate([np.asarray(part["groups"], dtype=str) for part in parts]),
        "sample_ids": np.concatenate([np.asarray(part["sample_ids"], dtype=str) for part in parts]),
        "folds": np.concatenate([np.asarray(part["folds"], dtype=np.uint8) for part in parts]),
        "base_scores": np.concatenate(
            [np.asarray(part["base_scores"], dtype=np.float64) for part in parts]
        ),
        "raw_scene_logits": np.concatenate(
            [np.asarray(part["raw_scene_logits"], dtype=np.float32) for part in parts]
        ),
    }
    size = merged["labels"].size
    if any(values.size != size for values in merged.values()):
        raise ValueError("Scene-logit cache vectors do not align")
    if len(set(merged["sample_ids"].tolist())) != size:
        raise ValueError("Scene-logit cache identities are not unique")
    if not np.isfinite(merged["base_scores"]).all() or not np.isfinite(
        merged["raw_scene_logits"]
    ).all():
        raise ValueError("Scene-logit cache contains non-finite values")
    return merged


def replay_scene_scores(
    base_scores: np.ndarray,
    raw_scene_logits: np.ndarray,
    *,
    strength: float,
    gate: float,
) -> np.ndarray:
    baseline = np.asarray(base_scores, dtype=np.float64)
    correction = 2.0 * np.tanh(np.asarray(raw_scene_logits, dtype=np.float64) / 2.0)
    clipped = np.clip(baseline, 1e-6, 1.0 - 1e-6)
    logits = np.log(clipped / (1.0 - clipped)) + float(strength) * correction
    candidate = np.where(
        logits >= 0.0,
        1.0 / (1.0 + np.exp(-logits)),
        np.exp(logits) / (1.0 + np.exp(logits)),
    )
    return np.where(baseline >= float(gate), candidate, baseline)


def atomic_npz(path: Path, **values: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **values)
    os.replace(temporary, path)


def train_phase(
    model: TransferGaussianContrastViTUNet,
    loader: DataLoader[dict[str, Any]],
    optimizer: torch.optim.Optimizer,
    spec: dict[str, Any],
    device: torch.device,
    *,
    phase: str,
    epochs: int,
    dense_only: bool,
) -> list[dict[str, float]]:
    """Train with BF16 and a dense-only synthetic warm start when requested."""

    parameters = [value for value in model.parameters() if value.requires_grad]
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        sums: dict[str, float] = {}
        batches = 0
        started = time.perf_counter()
        for batch_index, batch in enumerate(loader, start=1):
            finite_tensors(batch, phase=f"{phase}_input")
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                output = model(
                    batch["inputs"], batch["observable"], batch["sensor_index"]
                )
                finite_tensors(output, phase=f"{phase}_output")
                base_loss, parts = segmentation_first_loss(
                    output,
                    batch,
                    hard_negative_fraction=float(spec["hard_negative_fraction"]),
                    scene_weight=0.0 if dense_only else float(spec["scene_weight"]),
                )
                zero = output["scene_logit"].sum() * 0.0
                partial_auc = zero
                residual_scene = zero
                direction = zero
                if not dense_only:
                    partial_auc = partial_auc_pair_loss(
                        output["scene_logit"],
                        batch["presence"],
                        negative_fraction=float(spec["partial_auc_negative_fraction"]),
                        margin=float(spec["partial_auc_margin"]),
                    )
                    real = batch["is_synthetic"] < 0.5
                    if torch.any(real):
                        baseline_logit = torch.logit(
                            batch["base_scene_score"][real].clamp(1e-6, 1 - 1e-6)
                        )
                        fused_logit = baseline_logit + 2.0 * torch.tanh(
                            output["scene_logit"][real] / 2.0
                        )
                        residual_scene = F.binary_cross_entropy_with_logits(
                            fused_logit, batch["presence"][real]
                        )
                        real_positive = real & (batch["presence"] > 0.5)
                        real_negative = real & (batch["presence"] < 0.5)
                        penalties: list[torch.Tensor] = []
                        if torch.any(real_positive):
                            penalties.append(
                                F.softplus(-output["scene_logit"][real_positive]).mean()
                            )
                        if torch.any(real_negative):
                            penalties.append(
                                F.softplus(output["scene_logit"][real_negative]).mean()
                            )
                        if penalties:
                            direction = torch.stack(penalties).mean()
                loss = (
                    base_loss
                    + (0.0 if dense_only else float(spec["partial_auc_weight"])) * partial_auc
                    + (0.0 if dense_only else float(spec["residual_scene_weight"]))
                    * residual_scene
                    + (0.0 if dense_only else float(spec["direction_weight"])) * direction
                )
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite {phase} loss epoch={epoch} samples={list(batch['sample_id'])}"
                )
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                parameters, float(spec["gradient_clip"])
            )
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError(
                    f"Non-finite {phase} gradient epoch={epoch} samples={list(batch['sample_id'])}"
                )
            optimizer.step()
            row = {
                **{key: float(value) for key, value in parts.items()},
                "loss": float(loss.detach()),
                "partial_auc": float(partial_auc.detach()),
                "residual_scene_bce": float(residual_scene.detach()),
                "direction_loss": float(direction.detach()),
            }
            batches += 1
            for key, value in row.items():
                sums[key] = sums.get(key, 0.0) + value
            if batch_index % int(spec["progress_every_batches"]) == 0:
                print(
                    json.dumps(
                        {
                            "progress": "training_batch",
                            "phase": phase,
                            "epoch": epoch,
                            "batch": batch_index,
                            "elapsed_seconds": time.perf_counter() - started,
                        }
                    ),
                    flush=True,
                )
        result = {
            "phase": phase,
            "epoch": epoch,
            "seconds": time.perf_counter() - started,
            "batches": batches,
            "requests": len(loader.sampler) if loader.sampler is not None else len(loader.dataset),
            **{key: value / max(batches, 1) for key, value in sums.items()},
        }
        history.append(result)
        print(json.dumps(result), flush=True)
    return history


def verify_protocol(
    protocol: dict[str, Any], *, protocol_path: Path, smoke: bool
) -> dict[str, Path]:
    frozen = protocol["status"] == "frozen_before_held_outcomes"
    if not smoke and not frozen:
        raise ValueError("Held-fold scoring requires a frozen protocol")
    if frozen and sha256(Path(__file__).resolve()) != protocol["trainer"]["sha256"]:
        raise ValueError("Frozen contrast cross-fit trainer hash mismatch")
    if frozen:
        for dependency in protocol["code_dependencies"]:
            path = (ROOT / dependency["path"]).resolve()
            if not path.is_file() or sha256(path) != dependency["sha256"]:
                raise ValueError(f"Frozen dependency mismatch: {dependency['path']}")
    paths: dict[str, Path] = {}
    for name, contract in protocol["inputs"].items():
        path = (ROOT / contract["path"]).resolve()
        if not path.exists():
            raise ValueError(f"Missing input: {name}")
        expected = contract["sha256"]
        if path.is_file() and expected != "pending_smoke" and sha256(path) != expected:
            raise ValueError(f"Frozen input mismatch: {name}")
        paths[name] = path
    verify_acquisition_receipt(paths["acquisition_receipt"], sha256(paths["manifest"]))
    if frozen and sha256(protocol_path) != protocol["protocol_sha256_self_excluding_field"]:
        # The self hash is informational because embedding a file's exact hash in itself is impossible.
        if protocol["protocol_sha256_self_excluding_field"] != "see_git_commit":
            raise ValueError("Protocol self-hash marker is invalid")
    return paths


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    selected = report["selected"]
    delta = selected["versus_current"]["delta"]
    ap = selected["paired_site_ap_delta"]
    iou = selected["paired_site_pixel_iou_delta"]
    lines = [
        "# Physics-contrast Gaussian ViT real-development cross-fit",
        "",
        f"- Promotion gates pass: **{report['all_promotion_gates_pass']}**",
        f"- Selected fusion strength: **{selected['strength']}**",
        f"- AP delta versus current spatial-Prithvi: **{delta['average_precision']:+.6f}**",
        f"- Matched-FPR recall delta: **{delta['recall_at_fpr_0_0713']:+.6f}**",
        f"- Paired-site AP interval: **[{ap['lower']:+.6f}, {ap['upper']:+.6f}]**",
        f"- Pixel IoU delta: **{selected['pixel_iou_delta']:+.6f}**",
        f"- Paired-site IoU interval: **[{iou['lower']:+.6f}, {iou['upper']:+.6f}]**",
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
    scene_cache_replay = bool(protocol.get("scene_cache_replay", False))
    paths = verify_protocol(protocol, protocol_path=protocol_path, smoke=args.smoke)
    fold_protocol = json.loads(paths["fold_protocol"].read_text(encoding="utf-8"))
    group_to_fold = {
        str(row["group_id"]): int(row["fold"]) for row in fold_protocol["assignments"]
    }
    all_records = list(iter_development_manifest(paths["manifest"]))
    selected_folds = set(map(int, protocol["folds"]))
    records = [
        row for row in all_records if group_to_fold[str(row["group_id"])] in selected_folds
    ]
    base_scores, score_identity = load_base_score_contract(
        all_records, group_to_fold, paths["score_cache"]
    )
    spec = protocol["training"]
    strengths = [float(value) for value in protocol["search"]["strengths"]]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Physics-contrast cross-fit requires CUDA")
    torch.cuda.reset_peak_memory_stats()

    if args.smoke:
        seed = int(spec["seed"])
        seed_everything(seed)
        smoke_records = smoke_subset(records, 2)
        negatives = [
            row
            for row in records
            if row["label_state"] == "NO_PLUME"
            and str(row.get("observability", "")).lower() == "clear"
            and 0.5 <= np.hypot(float(row["wind_u"]), float(row["wind_v"])) <= 15.0
        ]
        synthetic = GaussianSyntheticSceneDataset(
            paths["metadata_root"],
            negatives,
            paths["transmittance_lut"],
            template_start=int(protocol["synthetic_bank"]["train_start"]),
            template_count=4,
            seed=int(spec["synthetic_data_seed"]),
            crop_size=int(spec["crop_size"]),
            augment=True,
        )
        model = TransferGaussianContrastViTUNet(**protocol["architecture"]["model"]).to(device)
        pretrain_optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(spec["pretrain_learning_rate"]),
            weight_decay=float(spec["weight_decay"]),
        )
        pretrain_loader = make_loader(
            synthetic,
            batch_size=int(spec["batch_size"]),
            workers=0,
            seed=int(spec["pretrain_loader_seed"]),
            sampler=PairShuffleSampler(4, int(spec["pretrain_loader_seed"])),
        )
        scene_head_before = {
            name: value.detach().clone()
            for name, value in model.scene_head.named_parameters()
        }
        history = train_phase(
            model,
            pretrain_loader,
            pretrain_optimizer,
            spec,
            device,
            phase=(
                "smoke_scene_aligned_pretrain"
                if bool(spec.get("synthetic_scene_supervision", False))
                else "smoke_dense_pretrain"
            ),
            epochs=1,
            dense_only=not bool(spec.get("synthetic_scene_supervision", False)),
        )
        scene_head_update_max_abs = max(
            float((value.detach() - scene_head_before[name]).abs().max())
            for name, value in model.scene_head.named_parameters()
        )
        real = RealCropDataset(
            MarsPaperDataset(paths["metadata_root"], smoke_records, augment=True, seed=seed),
            base_scores,
            seed=seed,
            crop_size=int(spec["crop_size"]),
            plume_center_probability=float(spec["plume_center_probability"]),
        )
        joint_loader = make_loader(
            real, batch_size=2, workers=0, seed=seed + 20
        )
        joint_optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(spec["joint_learning_rate"]),
            weight_decay=float(spec["weight_decay"]),
        )
        history.extend(
            train_phase(
                model,
                joint_loader,
                joint_optimizer,
                spec,
                device,
                phase="smoke_real_joint",
                epochs=1,
                dense_only=False,
            )
        )
        full_loader = make_loader(
            MarsPaperDataset(paths["metadata_root"], smoke_records, augment=False, seed=seed),
            batch_size=2,
            workers=0,
            seed=seed + 30,
        )
        full = move_batch(next(iter(full_loader)), device)
        model.eval()
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            output = model(full["inputs"], full["observable"], full["sensor_index"])
        finite_tensors(output, phase="smoke_full_scene_output")
        finite = all(
            math.isfinite(float(value))
            for row in history
            for value in row.values()
            if isinstance(value, (float, int))
        )
        print(
            json.dumps(
                {
                    "ok": finite,
                    "model": model.artifact_metadata(),
                    "full_scene_mask_shape": list(output["segmentation_logits"].shape),
                    "full_scene_logit_shape": list(output["scene_logit"].shape),
                    "scene_head_update_max_abs": scene_head_update_max_abs,
                    "peak_cuda_bytes": int(torch.cuda.max_memory_allocated()),
                    "device": torch.cuda.get_device_name(device),
                    "history": history,
                },
                sort_keys=True,
            )
        )
        return 0 if finite else 1

    if scene_cache_replay:
        for key in ("scene_cache", "endpoint_state_cache", "json", "markdown"):
            output = (ROOT / protocol["outputs"][key]).resolve()
            if output.exists():
                raise FileExistsError(f"Refusing to overwrite scene-cache replay output: {key}")
    teacher = (
        None
        if scene_cache_replay
        else _load_released_teacher(paths["released_checkpoint"], device)
    )
    prediction_parts: list[dict[str, Any]] = []
    endpoints: list[dict[str, Any]] = []
    endpoint_states: dict[str, Any] = {}
    bank = protocol["synthetic_bank"]
    template_count = int(bank["train_template_count"])
    for held_fold in sorted(selected_folds):
        fit_folds = selected_folds - {held_fold}
        fit_records = [
            row for row in records if group_to_fold[str(row["group_id"])] in fit_folds
        ]
        held_records = [
            row for row in records if group_to_fold[str(row["group_id"])] == held_fold
        ]
        negative_records = [
            row
            for row in fit_records
            if row["label_state"] == "NO_PLUME"
            and str(row.get("observability", "")).lower() == "clear"
            and 0.5 <= np.hypot(float(row["wind_u"]), float(row["wind_v"])) <= 15.0
        ]
        seed = int(spec["seed"]) + held_fold
        seed_everything(seed)
        model = TransferGaussianContrastViTUNet(**protocol["architecture"]["model"]).to(device)
        synthetic = GaussianSyntheticSceneDataset(
            paths["metadata_root"],
            negative_records,
            paths["transmittance_lut"],
            template_start=int(bank["train_start"]),
            template_count=template_count,
            seed=int(spec["synthetic_data_seed"]),
            crop_size=int(spec["crop_size"]),
            augment=True,
        )
        pretrain_loader = make_loader(
            synthetic,
            batch_size=int(spec["batch_size"]),
            workers=int(spec["loader_workers"]),
            seed=int(spec["pretrain_loader_seed"]),
            sampler=PairShuffleSampler(template_count, int(spec["pretrain_loader_seed"])),
        )
        pretrain_optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(spec["pretrain_learning_rate"]),
            weight_decay=float(spec["weight_decay"]),
        )
        history = train_phase(
            model,
            pretrain_loader,
            pretrain_optimizer,
            spec,
            device,
            phase=(
                "gaussian_scene_aligned_pretrain"
                if bool(spec.get("synthetic_scene_supervision", False))
                else "gaussian_contrast_pretrain"
            ),
            epochs=int(spec["synthetic_pretrain_epochs"]),
            dense_only=not bool(spec.get("synthetic_scene_supervision", False)),
        )
        real = RealCropDataset(
            MarsPaperDataset(paths["metadata_root"], fit_records, augment=True, seed=seed),
            base_scores,
            seed=seed,
            crop_size=int(spec["crop_size"]),
            plume_center_probability=float(spec["plume_center_probability"]),
        )
        real_weights, real_mass_detail = balanced_request_weights(fit_records)
        synthetic_weights = torch.full(
            (len(synthetic),), 1.0 / len(synthetic), dtype=torch.double
        )
        real_mass = float(protocol["sampling"]["real_mass"])
        joint_weights = torch.cat(
            (real_weights * real_mass, synthetic_weights * (1.0 - real_mass))
        )
        joint_weights /= joint_weights.sum()
        joint_sampler = WeightedRandomSampler(
            joint_weights,
            num_samples=int(spec["joint_samples_per_epoch"]),
            replacement=True,
            generator=torch.Generator().manual_seed(seed + 20),
        )
        joint_loader = make_loader(
            ConcatDataset((real, synthetic)),
            batch_size=int(spec["batch_size"]),
            workers=int(spec["loader_workers"]),
            seed=seed + 20,
            sampler=joint_sampler,
        )
        joint_optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(spec["joint_learning_rate"]),
            weight_decay=float(spec["weight_decay"]),
        )
        history.extend(
            train_phase(
                model,
                joint_loader,
                joint_optimizer,
                spec,
                device,
                phase="real_synthetic_joint",
                epochs=int(spec["joint_epochs"]),
                dense_only=False,
            )
        )
        held_loader = make_loader(
            MarsPaperDataset(paths["metadata_root"], held_records, augment=False, seed=seed),
            batch_size=int(spec["evaluation_batch_size"]),
            workers=int(spec["loader_workers"]),
            seed=seed + 30,
        )
        if scene_cache_replay:
            prediction_parts.append(
                collect_scene_logits(model, held_loader, base_scores, device, held_fold)
            )
        else:
            if teacher is None:
                raise RuntimeError("Dense evaluation requires the released teacher")
            prediction_parts.append(
                collect_predictions(
                    model, teacher, held_loader, base_scores, strengths, device, held_fold
                )
            )
        endpoints.append(
            {
                "held_fold": held_fold,
                "fit_folds": sorted(fit_folds),
                "fit_rows": len(fit_records),
                "held_rows": len(held_records),
                "eligible_synthetic_backgrounds": len(negative_records),
                "seed": seed,
                "real_mass": real_mass,
                "real_request_mass": real_mass_detail,
                "history": history,
            }
        )
        endpoint_states[str(held_fold)] = _artifact_state(model)
        del model, pretrain_optimizer, joint_optimizer, pretrain_loader, joint_loader, held_loader
        torch.cuda.empty_cache()

    if scene_cache_replay:
        raw_scene = merge_scene_logits(prediction_parts)
        reference = json.loads(paths["reference_result"].read_text(encoding="utf-8"))
        tolerance = float(protocol["replay_validation"]["absolute_tolerance"])
        replay_rows = []
        for strength in strengths:
            scores = replay_scene_scores(
                raw_scene["base_scores"],
                raw_scene["raw_scene_logits"],
                strength=strength,
                gate=float(protocol["architecture"]["model"]["scene_protection_gate"]),
            )
            observed = comparison(
                metric_summary(raw_scene["labels"], scores, raw_scene["sensors"]),
                metric_summary(
                    raw_scene["labels"], raw_scene["base_scores"], raw_scene["sensors"]
                ),
            )["delta"]
            expected_row = next(
                row for row in reference["candidates"] if float(row["strength"]) == strength
            )
            expected = expected_row["versus_current"]["delta"]
            checks = {
                "average_precision": abs(
                    float(observed["average_precision"])
                    - float(expected["average_precision"])
                )
                <= tolerance,
                "recall": abs(
                    float(observed["recall_at_fpr_0_0713"])
                    - float(expected["recall_at_fpr_0_0713"])
                )
                <= tolerance,
                "sensor_average_precision": all(
                    abs(
                        float(observed["sensor_average_precision"][sensor])
                        - float(expected["sensor_average_precision"][sensor])
                    )
                    <= tolerance
                    for sensor in expected["sensor_average_precision"]
                ),
            }
            if not all(checks.values()):
                raise RuntimeError(f"Scene-cache replay differs at strength {strength}: {checks}")
            replay_rows.append(
                {
                    "strength": strength,
                    "delta": observed,
                    "reference_delta": expected,
                    "checks": checks,
                }
            )
        cache_path = (ROOT / protocol["outputs"]["scene_cache"]).resolve()
        state_path = (ROOT / protocol["outputs"]["endpoint_state_cache"]).resolve()
        atomic_npz(
            cache_path,
            schema_version=np.asarray(1, dtype=np.int64),
            protocol_sha256=np.asarray(sha256(protocol_path)),
            **raw_scene,
        )
        state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_state = state_path.with_suffix(".tmp.pt")
        torch.save(
            {
                "schema_version": 1,
                "research_only": True,
                "protocol_sha256": sha256(protocol_path),
                "states_by_held_fold": endpoint_states,
            },
            temporary_state,
        )
        os.replace(temporary_state, state_path)
        report = {
            "schema_version": 1,
            "status": "completed_exact_scene_cache_replay",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "scope": protocol["scope"],
            "protocol": protocol_path.relative_to(ROOT).as_posix(),
            "protocol_sha256": sha256(protocol_path),
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "rows": int(raw_scene["labels"].size),
            "fold_counts": {
                str(fold): int(np.sum(raw_scene["folds"] == fold))
                for fold in sorted(set(map(int, raw_scene["folds"])))
            },
            "replay_validation": replay_rows,
            "all_replay_checks_pass": True,
            "scene_cache": {
                "path": protocol["outputs"]["scene_cache"],
                "bytes": cache_path.stat().st_size,
                "sha256": sha256(cache_path),
                "tracked": False,
            },
            "endpoint_state_cache": {
                "path": protocol["outputs"]["endpoint_state_cache"],
                "bytes": state_path.stat().st_size,
                "sha256": sha256(state_path),
                "tracked": False,
            },
            "endpoint_summaries": [
                {
                    "held_fold": endpoint["held_fold"],
                    "fit_folds": endpoint["fit_folds"],
                    "fit_rows": endpoint["fit_rows"],
                    "held_rows": endpoint["held_rows"],
                    "seed": endpoint["seed"],
                    "final_phase_losses": {
                        phase: next(
                            row for row in reversed(endpoint["history"]) if row["phase"] == phase
                        )["loss"]
                        for phase in ("gaussian_scene_aligned_pretrain", "real_synthetic_joint")
                    },
                }
                for endpoint in endpoints
            ],
            "external_or_official_test_accessed": False,
        }
        json_path = (ROOT / protocol["outputs"]["json"]).resolve()
        markdown_path = (ROOT / protocol["outputs"]["markdown"]).resolve()
        json_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_json = json_path.with_suffix(".tmp.json")
        temporary_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary_json, json_path)
        markdown_path.write_text(
            "# Gaussian scene-aligned cache replay\n\n"
            f"- Rows: **{report['rows']:,}**\n"
            f"- Exact replay checks pass: **{report['all_replay_checks_pass']}**\n"
            f"- Scene cache SHA-256: `{report['scene_cache']['sha256']}`\n"
            f"- Endpoint-state SHA-256: `{report['endpoint_state_cache']['sha256']}`\n",
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "ok": True,
                    "rows": report["rows"],
                    "scene_cache_sha256": report["scene_cache"]["sha256"],
                    "state_cache_sha256": report["endpoint_state_cache"]["sha256"],
                },
                sort_keys=True,
            )
        )
        return 0

    raw = merge_predictions(prediction_parts, strengths)
    candidates, identity = summarize_predictions(
        raw, strengths, protocol["bootstrap"], protocol["gates"]
    )
    selected = max(candidates, key=lambda row: tuple(row["rank"]))
    passed = bool(selected["passed"])
    artifact: dict[str, Any] | None = None
    if passed:
        artifact_path = (ROOT / protocol["outputs"]["artifact"]).resolve()
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "schema_version": 1,
                "model": TransferGaussianContrastViTUNet(
                    **protocol["architecture"]["model"]
                ).artifact_metadata(),
                "states_by_held_fold": endpoint_states,
                "selected_strength": selected["strength"],
                "protocol_sha256": sha256(protocol_path),
            },
            artifact_path,
        )
        artifact = {
            "path": protocol["outputs"]["artifact"],
            "bytes": artifact_path.stat().st_size,
            "sha256": sha256(artifact_path),
        }
    report = {
        "schema_version": 1,
        "status": "passed" if passed else "rejected",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": protocol["scope"],
        "protocol": protocol_path.relative_to(ROOT).as_posix(),
        "protocol_sha256": sha256(protocol_path),
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "score_contract": score_identity,
        "identity": identity,
        "endpoints": endpoints,
        "candidates": candidates,
        "selected": selected,
        "all_promotion_gates_pass": passed,
        "artifact": artifact,
        "peak_cuda_bytes": int(torch.cuda.max_memory_allocated()),
        "device": torch.cuda.get_device_name(device),
        "decision": (
            "Promote to the predeclared new-seed fold-2 confirmation; keep folds 0/1 and official test closed."
            if passed
            else "Reject without opening fold 2, folds 0/1, or the official test."
        ),
    }
    json_path = (ROOT / protocol["outputs"]["json"]).resolve()
    json_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = json_path.with_suffix(json_path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, json_path)
    write_markdown((ROOT / protocol["outputs"]["markdown"]).resolve(), report)
    print(json.dumps({"ok": True, "passed": passed, "selected": selected["strength"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
