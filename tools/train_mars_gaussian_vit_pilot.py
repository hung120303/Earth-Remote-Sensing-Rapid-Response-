#!/usr/bin/env python3
"""Cross-fit a Gaussian-pretrained full-scene ViT-U-Net on MARS folds 3/4."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset, WeightedRandomSampler, get_worker_info

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "EarthRemoteSensingRapidResponse"
for path in (MODEL_ROOT, ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from acquire_mars_metadata import sha256  # noqa: E402
from analyze_mars_mask_thresholds import component_mask_at  # noqa: E402
from mars_gaussian_plume import (  # noqa: E402
    MarsGaussianPlumeSimulator,
    sample_gaussian_parameters,
)
from mars_gaussian_vit import GaussianPretrainedViTUNet  # noqa: E402
from mars_paper_model import ReleasedMarsUNet, released_state  # noqa: E402
from mars_s2l_adapter import (  # noqa: E402
    REFLECTANCE_DIVISOR,
    REFLECTANCE_MAX,
    compute_mbmp,
    load_sample,
)
from train_mars_dense_prithvi_teacher_pilot import pixel_counts  # noqa: E402
from train_mars_dinov3_methane_fusion_pilot import (  # noqa: E402
    MASK_SCENE_GATE,
    MASK_THRESHOLDS,
    MINIMUM_CONNECTED_PIXELS,
    load_base_score_contract,
    partial_auc_pair_loss,
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
from train_mars_physical_patch_transfer_pilot import write_markdown  # noqa: E402
from train_mars_physics_guided_teacher_balanced_pilot import (  # noqa: E402
    balanced_request_weights,
)
from train_mars_physics_guided_teacher_pilot import seed_everything  # noqa: E402
from train_methanes2cm_v5 import segmentation_first_loss  # noqa: E402


DEFAULT_PROTOCOL = Path("configs/mars_gaussian_vit_pilot_protocol.json")


def rotate_wind(
    wind: tuple[float, float], quarter_turns: int
) -> tuple[float, float]:
    u, v = wind
    for _ in range(quarter_turns % 4):
        u, v = -v, u
    return float(u), float(v)


def build_input(
    raw_pair: np.ndarray,
    cloud: np.ndarray,
    wind: tuple[float, float],
) -> np.ndarray:
    spectral = np.clip(
        raw_pair.astype(np.float32) / REFLECTANCE_DIVISOR,
        0.0,
        REFLECTANCE_MAX,
    )
    mbmp = compute_mbmp(spectral[:6], spectral[6:])
    wind_channels = np.broadcast_to(
        np.asarray(wind, dtype=np.float32)[:, None, None] / 8.0,
        (2, *mbmp.shape),
    ).copy()
    return np.concatenate(
        [mbmp[None], spectral, wind_channels, cloud[None]], axis=0
    ).astype(np.float32)


def augment_arrays(
    raw_pair: np.ndarray,
    cloud: np.ndarray,
    clear: np.ndarray,
    observable: np.ndarray,
    mask: np.ndarray,
    wind: tuple[float, float],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[float, float]]:
    turns = int(rng.integers(0, 4))
    if turns:
        raw_pair = np.rot90(raw_pair, turns, axes=(1, 2)).copy()
        cloud = np.rot90(cloud, turns).copy()
        clear = np.rot90(clear, turns).copy()
        observable = np.rot90(observable, turns).copy()
        mask = np.rot90(mask, turns).copy()
        wind = rotate_wind(wind, turns)
    if bool(rng.integers(0, 2)):
        raw_pair = raw_pair[:, :, ::-1].copy()
        cloud = cloud[:, ::-1].copy()
        clear = clear[:, ::-1].copy()
        observable = observable[:, ::-1].copy()
        mask = mask[:, ::-1].copy()
        wind = (-wind[0], wind[1])
    if bool(rng.integers(0, 2)):
        raw_pair = raw_pair[:, ::-1, :].copy()
        cloud = cloud[::-1, :].copy()
        clear = clear[::-1, :].copy()
        observable = observable[::-1, :].copy()
        mask = mask[::-1, :].copy()
        wind = (wind[0], -wind[1])
    return raw_pair, cloud, clear, observable, mask, wind


class GaussianSyntheticSceneDataset(Dataset[dict[str, Any]]):
    """Index-disjoint synthetic positives paired with unchanged real negatives."""

    def __init__(
        self,
        metadata_root: Path,
        negative_records: list[dict[str, Any]],
        lut_path: Path,
        *,
        template_start: int,
        template_count: int,
        seed: int,
        crop_size: int,
        augment: bool,
    ) -> None:
        self.metadata_root = metadata_root
        self.lut_path = lut_path
        self.template_start = int(template_start)
        self.template_count = int(template_count)
        self.seed = int(seed)
        self.crop_size = int(crop_size)
        self.augment = bool(augment)
        self.pools = {
            0: [row for row in negative_records if row["sensor_family"] == "Sentinel-2"],
            1: [row for row in negative_records if row["sensor_family"] == "Landsat"],
        }
        if any(not values for values in self.pools.values()):
            raise ValueError("Synthetic training requires no-plume backgrounds from both sensors")
        self._simulator: MarsGaussianPlumeSimulator | None = None

    def __len__(self) -> int:
        return self.template_count * 2

    def simulator(self) -> MarsGaussianPlumeSimulator:
        if self._simulator is None:
            self._simulator = MarsGaussianPlumeSimulator(self.lut_path)
        return self._simulator

    def __getitem__(self, index: int) -> dict[str, Any]:
        template_index = self.template_start + index // 2
        positive = index % 2 == 0
        sensor_index = template_index % 2
        rng = np.random.default_rng(self.seed + template_index * 104729)
        pool = self.pools[sensor_index]
        record = pool[int(rng.integers(len(pool)))]
        sample = load_sample(self.metadata_root, record, require_enhancement=False)
        maximum_y = sample.raw_pair.shape[-2] - self.crop_size
        maximum_x = sample.raw_pair.shape[-1] - self.crop_size
        if maximum_y < 0 or maximum_x < 0:
            raise ValueError("Synthetic background is smaller than the training crop")
        y = int(rng.integers(0, maximum_y + 1))
        x = int(rng.integers(0, maximum_x + 1))
        region = np.s_[..., y : y + self.crop_size, x : x + self.crop_size]
        raw_pair = sample.raw_pair[region].copy()
        cloud = (sample.cloud_classes[y : y + self.crop_size, x : x + self.crop_size] > 0).astype(np.float32)
        clear = sample.clear_mask[y : y + self.crop_size, x : x + self.crop_size].astype(np.float32)
        observable = sample.observable_mask[y : y + self.crop_size, x : x + self.crop_size].astype(np.float32)
        mask = np.zeros((self.crop_size, self.crop_size), dtype=np.float32)
        wind = (float(record["wind_u"]), float(record["wind_v"]))
        if positive:
            parameters = sample_gaussian_parameters(
                (self.crop_size, self.crop_size), wind, rng
            )
            simulated = self.simulator().simulate(
                raw_pair[:6],
                satellite=str(record["satellite"]),
                solar_zenith_degrees=float(record["solar_zenith_angle"]),
                view_zenith_degrees=float(record["view_zenith_angle"]),
                resolution_m=10.0,
                parameters=parameters,
                rng=rng,
            )
            raw_pair[:6] = simulated.target
            mask = simulated.mask.astype(np.float32) * observable
            if not np.any(mask):
                raise ValueError("Synthetic plume has no observable pixels")
        if self.augment:
            raw_pair, cloud, clear, observable, mask, wind = augment_arrays(
                raw_pair, cloud, clear, observable, mask, wind, rng
            )
        return {
            "inputs": torch.from_numpy(build_input(raw_pair, cloud, wind)),
            "observable": torch.from_numpy(observable[None]),
            "clear": torch.from_numpy(clear[None]),
            "mask": torch.from_numpy(mask[None]),
            "presence": torch.tensor(float(positive), dtype=torch.float32),
            "sensor_index": torch.tensor(sensor_index, dtype=torch.long),
            "sample_id": f"gaussian:{template_index}:{int(positive)}:{sample.sample_id}",
            "group_id": f"gaussian:{template_index}",
            "pixel_truth_available": torch.tensor(True, dtype=torch.bool),
            "is_synthetic": torch.tensor(1.0, dtype=torch.float32),
            "base_scene_score": torch.tensor(0.5, dtype=torch.float32),
        }


class RealCropDataset(Dataset[dict[str, Any]]):
    """Plume-centered 160-pixel crops from genuine MARS fit-fold scenes."""

    def __init__(
        self,
        dataset: MarsPaperDataset,
        base_scores: dict[str, float],
        *,
        seed: int,
        crop_size: int,
        plume_center_probability: float,
    ) -> None:
        self.dataset = dataset
        self.base_scores = base_scores
        self.seed = int(seed)
        self.crop_size = int(crop_size)
        self.plume_center_probability = float(plume_center_probability)
        self._rng: np.random.Generator | None = None

    def __len__(self) -> int:
        return len(self.dataset)

    def rng(self) -> np.random.Generator:
        if self._rng is None:
            worker = get_worker_info()
            seed = self.seed if worker is None else self.seed + int(worker.seed)
            self._rng = np.random.default_rng(seed % (2**63 - 1))
        return self._rng

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.dataset[index]
        maximum_y = item["inputs"].shape[-2] - self.crop_size
        maximum_x = item["inputs"].shape[-1] - self.crop_size
        rng = self.rng()
        positive = torch.nonzero(item["mask"][0] > 0.5, as_tuple=False).numpy()
        if len(positive) and rng.random() < self.plume_center_probability:
            row, col = positive[int(rng.integers(len(positive)))]
            y = int(np.clip(int(row) - self.crop_size // 2 + rng.integers(-16, 17), 0, maximum_y))
            x = int(np.clip(int(col) - self.crop_size // 2 + rng.integers(-16, 17), 0, maximum_x))
        else:
            y = int(rng.integers(0, maximum_y + 1))
            x = int(rng.integers(0, maximum_x + 1))
        for key in ("inputs", "observable", "clear", "mask"):
            item[key] = item[key][..., y : y + self.crop_size, x : x + self.crop_size]
        item["presence"] = torch.tensor(
            float(torch.any((item["mask"] > 0.5) & (item["observable"] > 0.5))),
            dtype=torch.float32,
        )
        item["is_synthetic"] = torch.tensor(0.0, dtype=torch.float32)
        item["base_scene_score"] = torch.tensor(
            self.base_scores[str(item["sample_id"])], dtype=torch.float32
        )
        return item


def make_loader(
    dataset: Dataset[dict[str, Any]],
    *,
    batch_size: int,
    workers: int,
    sampler: WeightedRandomSampler | None = None,
) -> DataLoader[dict[str, Any]]:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
    )


def train_phase(
    model: GaussianPretrainedViTUNet,
    loader: DataLoader[dict[str, Any]],
    optimizer: torch.optim.Optimizer,
    spec: dict[str, Any],
    device: torch.device,
    *,
    phase: str,
    epochs: int,
) -> list[dict[str, float]]:
    scaler = torch.amp.GradScaler("cuda")
    parameters = [value for value in model.parameters() if value.requires_grad]
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        sums: dict[str, float] = {}
        batches = 0
        started = time.perf_counter()
        for batch_index, batch in enumerate(loader, start=1):
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                output = model(
                    batch["inputs"], batch["observable"], batch["sensor_index"]
                )
                base_loss, parts = segmentation_first_loss(
                    output,
                    batch,
                    hard_negative_fraction=float(spec["hard_negative_fraction"]),
                    scene_weight=float(spec["scene_weight"]),
                )
                partial_auc = partial_auc_pair_loss(
                    output["scene_logit"],
                    batch["presence"],
                    negative_fraction=float(spec["partial_auc_negative_fraction"]),
                    margin=float(spec["partial_auc_margin"]),
                )
                real = batch["is_synthetic"] < 0.5
                residual_scene = output["scene_logit"].sum() * 0.0
                direction = output["scene_logit"].sum() * 0.0
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
                        penalties.append(F.softplus(-output["scene_logit"][real_positive]).mean())
                    if torch.any(real_negative):
                        penalties.append(F.softplus(output["scene_logit"][real_negative]).mean())
                    if penalties:
                        direction = torch.stack(penalties).mean()
                loss = (
                    base_loss
                    + float(spec["partial_auc_weight"]) * partial_auc
                    + float(spec["residual_scene_weight"]) * residual_scene
                    + float(spec["direction_weight"]) * direction
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(parameters, float(spec["gradient_clip"]))
            scaler.step(optimizer)
            scaler.update()
            row = {
                **parts,
                "loss": float(loss.detach()),
                "partial_auc": float(partial_auc.detach()),
                "residual_scene_bce": float(residual_scene.detach()),
                "direction_loss": float(direction.detach()),
            }
            batches += 1
            for key, value in row.items():
                sums[key] = sums.get(key, 0.0) + float(value)
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
            **{key: value / max(batches, 1) for key, value in sums.items()},
        }
        history.append(result)
        print(json.dumps(result), flush=True)
    return history


def _load_released_teacher(path: Path, device: torch.device) -> ReleasedMarsUNet:
    teacher = ReleasedMarsUNet().to(device)
    incompatible = teacher.load_state_dict(released_state(path), strict=False)
    if incompatible.missing_keys:
        raise ValueError(f"Released checkpoint is missing keys: {incompatible.missing_keys}")
    if any(not key.startswith("out_mlp.") for key in incompatible.unexpected_keys):
        raise ValueError(f"Released checkpoint has unexpected keys: {incompatible.unexpected_keys}")
    teacher.eval()
    return teacher


@torch.no_grad()
def collect_predictions(
    model: GaussianPretrainedViTUNet,
    teacher: ReleasedMarsUNet,
    loader: DataLoader[dict[str, Any]],
    base_scores: dict[str, float],
    strengths: list[float],
    device: torch.device,
    fold: int,
) -> dict[str, Any]:
    model.eval()
    rows: dict[str, Any] = {
        "labels": [],
        "sensors": [],
        "groups": [],
        "sample_ids": [],
        "folds": [],
        "base_scores": [],
        "base_pixels": [],
        "candidate_scores": {str(value): [] for value in strengths},
        "candidate_pixels": {str(value): [] for value in strengths},
    }
    for batch_index, batch in enumerate(loader, start=1):
        sample_ids = [str(value) for value in batch["sample_id"]]
        group_ids = [str(value) for value in batch["group_id"]]
        local_base = torch.tensor(
            [base_scores[value] for value in sample_ids], dtype=torch.float32, device=device
        )
        batch = move_batch(batch, device)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            baseline_logits = teacher(batch["inputs"])
            output = model(
                batch["inputs"], batch["observable"], batch["sensor_index"]
            )
        baseline_probability = torch.sigmoid(baseline_logits.float())
        bounded_local = 2.0 * torch.tanh(output["segmentation_logits"].float() / 2.0)
        candidate_scores: dict[str, torch.Tensor] = {}
        for strength in strengths:
            scores = model.fuse_scene_score(local_base, output["scene_logit"].float(), strength)
            candidate_scores[str(strength)] = scores
            rows["candidate_scores"][str(strength)].extend(
                float(value) for value in scores.cpu().numpy()
            )
        for index in range(len(sample_ids)):
            sensor = int(batch["sensor_index"][index])
            threshold = MASK_THRESHOLDS[sensor]
            observable = batch["observable"][index, 0].cpu().numpy() > 0.5
            clear = batch["clear"][index, 0].cpu().numpy() > 0.5
            truth = (batch["mask"][index, 0].cpu().numpy() > 0.5) & observable
            base_score = float(local_base[index])
            base_map = baseline_probability[index, 0].cpu().numpy()
            base_map[~clear] = 0.0
            base_prediction = component_mask_at(base_map, threshold, MINIMUM_CONNECTED_PIXELS)
            if base_score < MASK_SCENE_GATE:
                base_prediction[:] = False
            rows["base_pixels"].append(pixel_counts(base_prediction, truth, observable))
            for strength in strengths:
                key = str(strength)
                probability = torch.sigmoid(
                    baseline_logits[index, 0].float()
                    + float(strength) * bounded_local[index, 0]
                ).cpu().numpy()
                probability[~clear] = 0.0
                prediction = component_mask_at(probability, threshold, MINIMUM_CONNECTED_PIXELS)
                if float(candidate_scores[key][index]) < MASK_SCENE_GATE:
                    prediction[:] = False
                rows["candidate_pixels"][key].append(
                    pixel_counts(prediction, truth, observable)
                )
        rows["labels"].extend(int(value) for value in batch["presence"].cpu().numpy())
        rows["sensors"].extend(int(value) for value in batch["sensor_index"].cpu().numpy())
        rows["groups"].extend(group_ids)
        rows["sample_ids"].extend(sample_ids)
        rows["folds"].extend([fold] * len(sample_ids))
        rows["base_scores"].extend(float(value) for value in local_base.cpu().numpy())
        if batch_index % 64 == 0:
            print(json.dumps({"progress": "vit_inference_batch", "fold": fold, "batch": batch_index}), flush=True)
    return rows


def verify_protocol(protocol: dict[str, Any], *, smoke: bool) -> dict[str, Path]:
    frozen = str(protocol["status"]).startswith("frozen")
    if not frozen and not smoke:
        raise ValueError("Outcome evaluation requires a frozen protocol")
    if frozen and sha256(Path(__file__).resolve()) != protocol["trainer"]["sha256"]:
        raise ValueError("Frozen Gaussian-ViT trainer hash mismatch")
    if frozen:
        for dependency in protocol["code_dependencies"]:
            path = (ROOT / dependency["path"]).resolve()
            if sha256(path) != dependency["sha256"]:
                raise ValueError(f"Frozen dependency mismatch: {dependency['path']}")
    paths: dict[str, Path] = {}
    for name, contract in protocol["inputs"].items():
        path = (ROOT / contract["path"]).resolve()
        if contract["sha256"] != "directory_verified_by_acquisition_receipt":
            if sha256(path) != contract["sha256"]:
                raise ValueError(f"Frozen input mismatch: {name}")
        paths[name] = path
    verify_acquisition_receipt(paths["acquisition_receipt"], sha256(paths["manifest"]))
    return paths


def _artifact_state(model: GaussianPretrainedViTUNet) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu() for name, value in model.state_dict().items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL.as_posix())
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    protocol_path = (ROOT / args.protocol).resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    paths = verify_protocol(protocol, smoke=args.smoke)
    fold_protocol = json.loads(paths["fold_protocol"].read_text(encoding="utf-8"))
    group_to_fold = {
        str(row["group_id"]): int(row["fold"]) for row in fold_protocol["assignments"]
    }
    all_records = list(iter_development_manifest(paths["manifest"]))
    selected_folds = set(map(int, protocol["folds"]))
    mars_records = [
        row for row in all_records if group_to_fold[str(row["group_id"])] in selected_folds
    ]
    base_scores, score_identity = load_base_score_contract(
        all_records, group_to_fold, paths["score_cache"]
    )
    spec = protocol["training"]
    strengths = [float(value) for value in protocol["search"]["strengths"]]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Gaussian-ViT pilot requires CUDA")
    torch.cuda.reset_peak_memory_stats()

    if args.smoke:
        seed = int(spec["seed"])
        seed_everything(seed)
        smoke_records = smoke_subset(mars_records, 2)
        negatives = [row for row in mars_records if row["label_state"] == "NO_PLUME"]
        synthetic = GaussianSyntheticSceneDataset(
            paths["metadata_root"],
            negatives,
            paths["transmittance_lut"],
            template_start=0,
            template_count=max(2, int(spec["batch_size"]) // 2),
            seed=seed,
            crop_size=int(spec["crop_size"]),
            augment=True,
        )
        loader = make_loader(
            synthetic, batch_size=int(spec["batch_size"]), workers=0
        )
        model = GaussianPretrainedViTUNet(**protocol["architecture"]["model"]).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(spec["learning_rate"]),
            weight_decay=float(spec["weight_decay"]),
        )
        history = train_phase(
            model, loader, optimizer, spec, device, phase="smoke_synthetic", epochs=1
        )
        real_smoke = RealCropDataset(
            MarsPaperDataset(
                paths["metadata_root"], smoke_records, augment=True, seed=seed + 1
            ),
            base_scores,
            seed=seed + 1,
            crop_size=int(spec["crop_size"]),
            plume_center_probability=float(spec["plume_center_probability"]),
        )
        real_loader = make_loader(real_smoke, batch_size=4, workers=0)
        history.extend(
            train_phase(
                model,
                real_loader,
                optimizer,
                spec,
                device,
                phase="smoke_real_residual",
                epochs=1,
            )
        )
        full_loader = make_loader(
            MarsPaperDataset(paths["metadata_root"], smoke_records, augment=False, seed=seed),
            batch_size=2,
            workers=0,
        )
        full = move_batch(next(iter(full_loader)), device)
        model.eval()
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
            output = model(full["inputs"], full["observable"], full["sensor_index"])
        finite = bool(
            torch.isfinite(output["segmentation_logits"]).all()
            and torch.isfinite(output["scene_logit"]).all()
            and all(
                math.isfinite(float(value))
                for value in history[-1].values()
                if isinstance(value, (float, int))
            )
        )
        print(
            json.dumps(
                {
                    "ok": finite,
                    "model": model.artifact_metadata(),
                    "full_scene_mask_shape": list(output["segmentation_logits"].shape),
                    "full_scene_logit_shape": list(output["scene_logit"].shape),
                    "peak_cuda_bytes": int(torch.cuda.max_memory_allocated()),
                    "device": torch.cuda.get_device_name(device),
                    "history": history,
                },
                sort_keys=True,
            )
        )
        return 0 if finite else 1

    teacher = _load_released_teacher(paths["released_checkpoint"], device)
    prediction_parts: list[dict[str, Any]] = []
    endpoints: list[dict[str, Any]] = []
    endpoint_states: dict[str, Any] = {}
    bank = protocol["synthetic_bank"]
    template_start, template_stop = map(int, bank["train_indices"])
    template_count = template_stop - template_start + 1
    for held_fold in sorted(selected_folds):
        fit_folds = selected_folds - {held_fold}
        fit_records = [
            row for row in mars_records if group_to_fold[str(row["group_id"])] in fit_folds
        ]
        held_records = [
            row for row in mars_records if group_to_fold[str(row["group_id"])] == held_fold
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
        model = GaussianPretrainedViTUNet(**protocol["architecture"]["model"]).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(spec["learning_rate"]),
            weight_decay=float(spec["weight_decay"]),
        )
        synthetic = GaussianSyntheticSceneDataset(
            paths["metadata_root"],
            negative_records,
            paths["transmittance_lut"],
            template_start=template_start,
            template_count=template_count,
            seed=seed + 1000,
            crop_size=int(spec["crop_size"]),
            augment=True,
        )
        synthetic_weights = torch.full((len(synthetic),), 1.0 / len(synthetic), dtype=torch.double)
        pretrain_sampler = WeightedRandomSampler(
            synthetic_weights,
            num_samples=int(spec["synthetic_samples_per_epoch"]),
            replacement=False,
            generator=torch.Generator().manual_seed(seed + 10),
        )
        pretrain_loader = make_loader(
            synthetic,
            batch_size=int(spec["batch_size"]),
            workers=int(spec["loader_workers"]),
            sampler=pretrain_sampler,
        )
        history = train_phase(
            model,
            pretrain_loader,
            optimizer,
            spec,
            device,
            phase="gaussian_pretrain",
            epochs=int(spec["synthetic_pretrain_epochs"]),
        )
        real = RealCropDataset(
            MarsPaperDataset(paths["metadata_root"], fit_records, augment=True, seed=seed),
            base_scores,
            seed=seed,
            crop_size=int(spec["crop_size"]),
            plume_center_probability=float(spec["plume_center_probability"]),
        )
        real_weights, real_mass_detail = balanced_request_weights(fit_records)
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
            sampler=joint_sampler,
        )
        history.extend(
            train_phase(
                model,
                joint_loader,
                optimizer,
                spec,
                device,
                phase="real_synthetic_joint",
                epochs=int(spec["joint_epochs"]),
            )
        )
        held_loader = make_loader(
            MarsPaperDataset(paths["metadata_root"], held_records, augment=False, seed=seed),
            batch_size=int(spec["evaluation_batch_size"]),
            workers=int(spec["loader_workers"]),
        )
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
        del model, optimizer, pretrain_loader, joint_loader, held_loader
        torch.cuda.empty_cache()

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
                "model": GaussianPretrainedViTUNet(
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
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
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
            "Promote to a new-seed fold-2 confirmation; keep folds 0/1 and official test closed."
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
