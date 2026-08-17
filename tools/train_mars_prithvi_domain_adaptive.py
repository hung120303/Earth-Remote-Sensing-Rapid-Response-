#!/usr/bin/env python3
"""Cross-fit spectral-temporal domain adaptation of Prithvi for MARS-S2L."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from collections import Counter
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import rasterio
import torch
from scipy.special import expit, logit
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "EarthRemoteSensingRapidResponse"
for path in (MODEL_ROOT, ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from acquire_mars_metadata import sha256
from extract_mars_prithvi_scene_features import (
    date_coordinate,
    reference_date_coordinate,
)
from mars_prithvi_domain_adaptation import (
    configure_extended_pretraining,
    merge_lora_inplace,
    patch_group_normalized_l1_loss,
)
from mars_prithvi_domain_residual import DomainAdaptiveResidualSceneModel
from mars_prithvi_lora_model import (
    load_trainable_state,
    trainable_parameter_count,
    trainable_state,
)
from mars_s2l_adapter import safe_asset_path, validate_image_band_order
from train_mars_oof_scene_ensemble_v2 import ap_group_bootstrap
from train_mars_paper_residual import (
    MarsPaperDataset,
    iter_development_manifest,
    move_batch,
    verify_acquisition_receipt,
)
from train_mars_prithvi_spatial_head import patch_supervision_loss
from train_mars_scene_ranker import comparison, metric_summary
from train_methanes2cm_v5 import (
    MODEL_BAND_INDICES,
    REFLECTANCE_DIVISOR,
    REFLECTANCE_MAX,
    SOURCE_REVISION,
)

DEFAULT_PROTOCOL = Path("configs/mars_prithvi_domain_adaptive_protocol.json")
INPUT_SIZE = 128


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def verify_protocol(
    protocol_path: Path, protocol: dict[str, Any], *, smoke: bool
) -> dict[str, Path]:
    frozen = str(protocol["status"]).startswith("frozen_")
    if not smoke and not frozen:
        raise ValueError(
            "Outcome training requires a frozen domain-adaptation protocol"
        )
    trainer = Path(__file__).resolve()
    if frozen and sha256(trainer) != protocol["trainer"]["sha256"]:
        raise ValueError("Frozen domain-adaptation trainer hash mismatch")
    if frozen:
        for item in protocol["code_dependencies"]:
            path = (ROOT / item["path"]).resolve()
            if sha256(path) != item["sha256"]:
                raise ValueError(f"Frozen dependency mismatch: {item['path']}")
    paths: dict[str, Path] = {}
    for name, contract in protocol["inputs"].items():
        path = (ROOT / contract["path"]).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Required input is unavailable: {name}={path}")
        expected = str(contract.get("sha256", ""))
        if path.is_file() and expected and sha256(path) != expected:
            raise ValueError(f"Input hash mismatch: {name}")
        paths[name] = path
    return paths


def build_foundation(
    paths: dict[str, Path], device: torch.device
) -> tuple[nn.Module, torch.Tensor, torch.Tensor, dict[str, Any]]:
    receipt = json.loads(paths["foundation_receipt"].read_text(encoding="utf-8"))
    foundation_dir = paths["foundation_config"].parent
    for item in receipt["files"]:
        path = (ROOT / item["path"]).resolve()
        if path.stat().st_size != int(item["bytes"]) or sha256(path) != item["sha256"]:
            raise ValueError(f"Prithvi foundation identity mismatch: {item['path']}")
    if str(foundation_dir) not in sys.path:
        sys.path.insert(0, str(foundation_dir))
    from prithvi_mae import PrithviMAE  # type: ignore

    config = json.loads(paths["foundation_config"].read_text(encoding="utf-8"))[
        "pretrained_cfg"
    ]
    model_config = dict(config)
    model_config.update(img_size=INPUT_SIZE, num_frames=2, in_chans=6)
    model = PrithviMAE(**model_config)
    state = torch.load(
        paths["foundation_checkpoint"], map_location="cpu", weights_only=True
    )
    state["encoder.pos_embed"] = model.encoder.pos_embed
    state["decoder.decoder_pos_embed"] = model.decoder.decoder_pos_embed
    model.load_state_dict(state, strict=True)
    mean = torch.tensor(config["mean"], dtype=torch.float32, device=device)[
        None, :, None, None, None
    ]
    std = torch.tensor(config["std"], dtype=torch.float32, device=device)[
        None, :, None, None, None
    ]
    return model.to(device), mean, std, config


class MethaneS2CMUnlabeledDataset(Dataset[dict[str, Any]]):
    """Read only reflectance arrays and identities from the packed auxiliary cohort."""

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
                raise ValueError("MethaneS2CM source revision differs")
            identifiers = source["sample_id"][:].astype(np.int64)
        index_by_id = {int(value): index for index, value in enumerate(identifiers)}
        if len(index_by_id) != len(identifiers):
            raise ValueError("Packed MethaneS2CM identifiers are not unique")
        self.packed_indices = []
        for record in records:
            identifier = int(record["id"])
            if identifier not in index_by_id:
                raise ValueError(f"Packed auxiliary data lacks sample {identifier}")
            self.packed_indices.append(index_by_id[identifier])

    def __len__(self) -> int:
        return len(self.records)

    def packed(self) -> h5py.File:
        if self._packed is None:
            self._packed = h5py.File(
                self.packed_path, "r", rdcc_nbytes=32 * 1024 * 1024
            )
        return self._packed

    def rng(self) -> np.random.Generator:
        if self._rng is None:
            worker = torch.utils.data.get_worker_info()
            value = self.seed if worker is None else int(worker.seed) + self.seed
            self._rng = np.random.default_rng(value % (2**63 - 1))
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
        source = self.packed()
        packed_index = self.packed_indices[index]
        raw = [
            source[name][packed_index][np.asarray(MODEL_BAND_INDICES)]
            for name in ("reference90", "reference365", "target")
        ]
        observable = np.all(np.concatenate(raw, axis=0) != 0, axis=0)
        frames = np.stack(
            [
                np.clip(
                    value.astype(np.float32) / REFLECTANCE_DIVISOR,
                    0.0,
                    REFLECTANCE_MAX,
                )
                for value in raw
            ],
            axis=0,
        )
        frames[:, :, ~observable] = 0.0
        if self.augment:
            rng = self.rng()
            turns = int(rng.integers(0, 4))
            if turns:
                frames = np.rot90(frames, turns, axes=(2, 3)).copy()
                observable = np.rot90(observable, turns).copy()
            if bool(rng.integers(0, 2)):
                frames = frames[:, :, :, ::-1].copy()
                observable = observable[:, ::-1].copy()
            if bool(rng.integers(0, 2)):
                frames = frames[:, :, ::-1, :].copy()
                observable = observable[::-1, :].copy()
        return {
            "frames": torch.from_numpy(frames),
            "observable": torch.from_numpy(observable[None].astype(np.float32)),
            "sample_id": str(self.records[index]["id"]),
        }


class MarsUnlabeledDataset(Dataset[dict[str, Any]]):
    """Read MARS reflectance/cloud assets without opening label or plume assets."""

    def __init__(
        self,
        metadata_root: Path,
        records: list[dict[str, Any]],
        *,
        augment: bool,
        seed: int,
    ) -> None:
        self.metadata_root = metadata_root
        self.records = records
        self.augment = augment
        self.seed = seed
        self._rng: np.random.Generator | None = None

    def __len__(self) -> int:
        return len(self.records)

    def rng(self) -> np.random.Generator:
        if self._rng is None:
            worker = torch.utils.data.get_worker_info()
            value = self.seed if worker is None else int(worker.seed) + self.seed
            self._rng = np.random.default_rng(value % (2**63 - 1))
        return self._rng

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        roles = {str(item["role"]): str(item["path"]) for item in record["assets"]}
        if not {"image", "cloud_mask"}.issubset(roles):
            raise ValueError(
                f"MARS unlabeled row lacks image/cloud: {record['sample_id']}"
            )
        image_path = safe_asset_path(self.metadata_root, roles["image"])
        cloud_path = safe_asset_path(self.metadata_root, roles["cloud_mask"])
        with rasterio.open(image_path) as source:
            if source.count != 12 or set(source.dtypes) != {"uint16"}:
                raise ValueError(f"Invalid MARS image raster: {record['sample_id']}")
            validate_image_band_order(record, tuple(source.descriptions))
            raw = source.read()
            shape = (source.height, source.width)
        with rasterio.open(cloud_path) as source:
            if source.count != 1 or (source.height, source.width) != shape:
                raise ValueError(f"Invalid MARS cloud raster: {record['sample_id']}")
            cloud = source.read(1)
        observable = np.all(raw != 0, axis=0) & (cloud == 0)
        pair = np.stack(
            (
                np.clip(raw[6:].astype(np.float32) / 10_000.0, 0.0, 2.0),
                np.clip(raw[:6].astype(np.float32) / 10_000.0, 0.0, 2.0),
            ),
            axis=0,
        )
        pair[:, :, ~observable] = 0.0
        if self.augment:
            rng = self.rng()
            turns = int(rng.integers(0, 4))
            if turns:
                pair = np.rot90(pair, turns, axes=(2, 3)).copy()
                observable = np.rot90(observable, turns).copy()
            if bool(rng.integers(0, 2)):
                pair = pair[:, :, :, ::-1].copy()
                observable = observable[:, ::-1].copy()
            if bool(rng.integers(0, 2)):
                pair = pair[:, :, ::-1, :].copy()
                observable = observable[::-1, :].copy()
        return {
            "pair": torch.from_numpy(pair),
            "observable": torch.from_numpy(observable[None].astype(np.float32)),
            "sample_id": str(record["sample_id"]),
        }


def cycle(loader: DataLoader[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    while True:
        yield from loader


def prithvi_input(
    pair: torch.Tensor,
    observable: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    """Convert physical Bx2x6 frames to normalized Bx6x2x128x128."""

    if pair.ndim != 5 or pair.shape[1:3] != (2, 6):
        raise ValueError("Physical pair must have shape Bx2x6xHxW")
    batch = pair.shape[0]
    resized = F.interpolate(
        pair.flatten(0, 1),
        size=(INPUT_SIZE, INPUT_SIZE),
        mode="bilinear",
        align_corners=False,
    ).reshape(batch, 2, 6, INPUT_SIZE, INPUT_SIZE)
    valid = F.interpolate(
        observable.float(), size=(INPUT_SIZE, INPUT_SIZE), mode="nearest"
    )[:, :, None]
    values = resized.permute(0, 2, 1, 3, 4)
    normalized = (values * 10_000.0 - mean) / std
    return normalized.masked_fill(valid <= 0.5, 0.0)


def prithvi_observable(observable: torch.Tensor) -> torch.Tensor:
    if observable.ndim != 4 or observable.shape[1] != 1:
        raise ValueError("Observable mask must have shape Bx1xHxW")
    return F.interpolate(
        observable.float(), size=(INPUT_SIZE, INPUT_SIZE), mode="nearest"
    )


def mars_pair(
    batch: dict[str, Any], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    inputs = batch["inputs"].to(device=device, dtype=torch.float32)
    observable = batch["observable"].to(device=device, dtype=torch.float32)
    # The released adapter divides producer reflectance by 5,000.  Restore the
    # physical 0..1 scale before applying Prithvi's 10,000-DN normalization.
    reference = inputs[:, 7:13] * 0.5
    target = inputs[:, 1:7] * 0.5
    return torch.stack((reference, target), dim=1), observable


def mars_coordinates(
    sample_ids: list[str], metadata: dict[str, dict[str, Any]], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    temporal = torch.tensor(
        [
            [
                reference_date_coordinate(metadata[value]),
                date_coordinate(str(metadata[value]["target_datetime"])),
            ]
            for value in sample_ids
        ],
        dtype=torch.float32,
        device=device,
    )
    # Geography is intentionally unavailable to the model; zero is a constant
    # sentinel accepted by the pretrained coordinate encoder.
    location = torch.zeros((len(sample_ids), 2), dtype=torch.float32, device=device)
    return temporal, location


def auxiliary_coordinates(
    count: int, reference_index: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    reference = (2020.0, 276.0) if reference_index == 0 else (2020.0, 1.0)
    temporal = torch.tensor(
        [[reference, (2021.0, 1.0)]] * count,
        dtype=torch.float32,
        device=device,
    )
    return temporal, torch.zeros((count, 2), dtype=torch.float32, device=device)


def loader_options(spec: dict[str, Any], *, workers_key: str) -> dict[str, Any]:
    workers = int(spec[workers_key])
    result: dict[str, Any] = {
        "num_workers": workers,
        "pin_memory": True,
        "persistent_workers": workers > 0,
    }
    if workers > 0:
        result["prefetch_factor"] = int(spec["prefetch_factor"])
    return result


def encoder_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu() for name, value in model.encoder.state_dict().items()
    }


def pretrain_encoder(
    records: list[dict[str, Any]],
    metadata: dict[str, dict[str, Any]],
    auxiliary: list[dict[str, Any]],
    paths: dict[str, Path],
    protocol: dict[str, Any],
    seed: int,
    held_fold: int,
    device: torch.device,
    *,
    smoke: bool,
) -> tuple[dict[str, torch.Tensor], list[dict[str, float]], dict[str, Any]]:
    spec = protocol["pretraining"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    foundation, mean, std, _ = build_foundation(paths, device)
    retained_decoder_blocks = int(spec["decoder_blocks"])
    if retained_decoder_blocks <= 0 or retained_decoder_blocks > len(
        foundation.decoder.decoder_blocks
    ):
        raise ValueError("Requested MAE decoder depth is outside the foundation depth")
    foundation.decoder.decoder_blocks = nn.ModuleList(
        list(foundation.decoder.decoder_blocks[:retained_decoder_blocks])
    )
    receipt = configure_extended_pretraining(
        foundation,
        rank=int(spec["rank"]),
        alpha=float(spec["alpha"]),
        dropout=float(spec["lora_dropout"]),
        fully_unfrozen_last_blocks=int(spec["fully_unfrozen_last_blocks"]),
    )
    receipt["retained_decoder_blocks"] = retained_decoder_blocks
    lora, encoder_full, decoder = [], [], []
    for name, parameter in foundation.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("decoder."):
            decoder.append(parameter)
        elif ".a." in name or ".b." in name:
            lora.append(parameter)
        else:
            encoder_full.append(parameter)
    optimizer = torch.optim.AdamW(
        [
            {"params": lora, "lr": float(spec["lora_learning_rate"])},
            {"params": encoder_full, "lr": float(spec["encoder_learning_rate"])},
            {"params": decoder, "lr": float(spec["decoder_learning_rate"])},
        ],
        weight_decay=float(spec["weight_decay"]),
    )
    per_source = int(spec["batch_size"]) // 2
    if per_source <= 0 or int(spec["batch_size"]) % 2:
        raise ValueError("Pretraining batch size must be positive and even")
    mars_loader = DataLoader(
        MarsUnlabeledDataset(paths["metadata_root"], records, augment=True, seed=seed),
        batch_size=per_source,
        shuffle=True,
        drop_last=True,
        generator=torch.Generator().manual_seed(seed),
        **loader_options(spec, workers_key="mars_workers"),
    )
    auxiliary_loader = DataLoader(
        MethaneS2CMUnlabeledDataset(
            paths["methanes2cm_packed"], auxiliary, augment=True, seed=seed + 1
        ),
        batch_size=per_source,
        shuffle=True,
        drop_last=True,
        generator=torch.Generator().manual_seed(seed + 1),
        **loader_options(spec, workers_key="auxiliary_workers"),
    )
    mars_batches = cycle(mars_loader)
    auxiliary_batches = cycle(auxiliary_loader)
    steps = int(spec["smoke_steps"] if smoke else spec["steps"])
    warmup = min(int(spec["warmup_steps"]), max(steps - 1, 0))
    history: list[dict[str, float]] = []
    foundation.train()
    sums = Counter()
    for step in range(steps):
        mars = next(mars_batches)
        aux = next(auxiliary_batches)
        mars_ids = [str(value) for value in mars["sample_id"]]
        mars_values = mars["pair"].to(device=device, dtype=torch.float32)
        mars_observable = mars["observable"].to(device=device, dtype=torch.float32)
        mars_temporal, mars_location = mars_coordinates(mars_ids, metadata, device)
        reference_index = step % 2
        aux_frames = aux["frames"].to(device=device, dtype=torch.float32)
        aux_pair = torch.stack(
            (aux_frames[:, reference_index], aux_frames[:, 2]), dim=1
        )
        aux_observable = aux["observable"].to(device=device, dtype=torch.float32)
        aux_temporal, aux_location = auxiliary_coordinates(
            aux_pair.shape[0], reference_index, device
        )
        values = torch.cat(
            (
                prithvi_input(mars_values, mars_observable, mean, std),
                prithvi_input(aux_pair, aux_observable, mean, std),
            ),
            dim=0,
        )
        observable = torch.cat(
            (
                prithvi_observable(mars_observable),
                prithvi_observable(aux_observable),
            ),
            dim=0,
        )
        temporal = torch.cat((mars_temporal, aux_temporal), dim=0)
        location = torch.cat((mars_location, aux_location), dim=0)
        if step < warmup:
            factor = float(step + 1) / float(max(warmup, 1))
        else:
            progress = float(step - warmup) / float(max(steps - warmup - 1, 1))
            factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        for group, base in zip(
            optimizer.param_groups,
            (
                float(spec["lora_learning_rate"]),
                float(spec["encoder_learning_rate"]),
                float(spec["decoder_learning_rate"]),
            ),
        ):
            group["lr"] = base * factor
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            latent, mask, ids_restore = foundation.encoder(
                values,
                temporal,
                location,
                float(spec["mask_ratio"]),
            )
            pred = foundation.decoder(
                latent, ids_restore, temporal, location, input_size=values.shape
            )
            losses = patch_group_normalized_l1_loss(
                foundation,
                values,
                pred,
                mask,
                observable,
                band_groups=tuple(tuple(group) for group in spec["band_groups"]),
                temporal_difference_weight=float(spec["temporal_difference_weight"]),
            )
            loss = losses["loss"]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = nn.utils.clip_grad_norm_(
            [value for value in foundation.parameters() if value.requires_grad],
            float(spec["gradient_clip"]),
        )
        if not torch.isfinite(norm):
            raise FloatingPointError("Non-finite extended-pretraining gradient")
        optimizer.step()
        for key in ("loss", "reconstruction", "temporal_difference"):
            sums[key] += float(losses[key].detach())
        report_every = int(spec["report_every_steps"])
        if (step + 1) % report_every == 0 or step + 1 == steps:
            divisor = (
                report_every
                if (step + 1) % report_every == 0
                else (step + 1) % report_every
            )
            row = {
                "step": float(step + 1),
                "loss": sums["loss"] / divisor,
                "reconstruction": sums["reconstruction"] / divisor,
                "temporal_difference": sums["temporal_difference"] / divisor,
                "learning_rate_factor": factor,
            }
            history.append(row)
            print(
                json.dumps(
                    {"phase": "pretrain", "seed": seed, "held_fold": held_fold, **row}
                ),
                flush=True,
            )
            sums.clear()
    merge_lora_inplace(foundation)
    state = encoder_state(foundation)
    del foundation
    torch.cuda.empty_cache()
    return state, history, receipt


def load_champion(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        required = {
            "sample_ids",
            "labels",
            "sensors",
            "groups",
            "folds",
            "champion_scores",
        }
        if not required.issubset(source.files):
            raise ValueError("Champion cache schema is incomplete")
        values = {name: source[name].copy() for name in required}
    if set(np.unique(values["folds"]).tolist()) != {3, 4}:
        raise ValueError("Champion cache must contain only folds 3/4")
    if len(set(values["sample_ids"].astype(str).tolist())) != len(values["sample_ids"]):
        raise ValueError("Champion cache sample identifiers are not unique")
    scores = values["champion_scores"].astype(np.float64)
    if np.any((scores <= 0.0) | (scores >= 1.0)) or not np.isfinite(scores).all():
        raise ValueError(
            "Champion probabilities must be finite and strictly inside (0,1)"
        )
    return values


def record_weights(
    records: list[dict[str, Any]], champion_by_id: dict[str, float]
) -> torch.Tensor:
    groups = Counter(str(value["group_id"]) for value in records)
    classes = Counter(str(value["label_state"]) for value in records)
    sensors = Counter(str(value["sensor_family"]) for value in records)
    weights = []
    for record in records:
        label = float(record["label_state"] == "PLUME")
        score = float(champion_by_id[str(record["sample_id"])])
        value = 1.0 / groups[str(record["group_id"])]
        value *= math.sqrt(len(records) / classes[str(record["label_state"])])
        value *= math.sqrt(len(records) / sensors[str(record["sensor_family"])])
        value *= 1.0 + 2.0 * abs(label - score)
        weights.append(value)
    result = torch.tensor(weights, dtype=torch.double)
    return result / result.mean()


def patch_targets(batch: dict[str, Any], grid: int) -> torch.Tensor:
    plume = F.adaptive_max_pool2d(batch["mask"] * batch["observable"], (grid, grid))
    visible = F.adaptive_avg_pool2d(batch["observable"], (grid, grid))
    return torch.cat((plume, visible), dim=1)


def build_scene_model(
    paths: dict[str, Path],
    encoder: dict[str, torch.Tensor],
    spec: dict[str, Any],
    device: torch.device,
    state: dict[str, torch.Tensor] | None = None,
) -> tuple[DomainAdaptiveResidualSceneModel, torch.Tensor, torch.Tensor]:
    foundation, mean, std, _ = build_foundation(paths, device)
    foundation.encoder.load_state_dict(encoder, strict=True)
    model = DomainAdaptiveResidualSceneModel(foundation, spec).to(device)
    if state is not None:
        load_trainable_state(model, state)
    return model, mean, std


def train_scene_model(
    records: list[dict[str, Any]],
    metadata: dict[str, dict[str, Any]],
    champion_by_id: dict[str, float],
    paths: dict[str, Path],
    encoder: dict[str, torch.Tensor],
    protocol: dict[str, Any],
    seed: int,
    held_fold: int,
    device: torch.device,
    *,
    smoke: bool,
) -> tuple[dict[str, torch.Tensor], list[dict[str, float]], int]:
    spec = protocol["scene_training"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    model, mean, std = build_scene_model(paths, encoder, spec, device)
    lora, head = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (lora if ".a." in name or ".b." in name else head).append(parameter)
    optimizer = torch.optim.AdamW(
        [
            {"params": lora, "lr": float(spec["lora_learning_rate"])},
            {"params": head, "lr": float(spec["head_learning_rate"])},
        ],
        weight_decay=float(spec["weight_decay"]),
    )
    sampler = WeightedRandomSampler(
        record_weights(records, champion_by_id),
        num_samples=len(records),
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )
    batch_size = (
        min(int(spec["batch_size"]), len(records)) if smoke else int(spec["batch_size"])
    )
    loader = DataLoader(
        MarsPaperDataset(paths["metadata_root"], records, augment=True, seed=seed),
        batch_size=batch_size,
        sampler=sampler,
        drop_last=not smoke,
        **loader_options(spec, workers_key="workers"),
    )
    epochs = int(spec["smoke_epochs"] if smoke else spec["epochs"])
    history: list[dict[str, float]] = []
    for epoch in range(epochs):
        model.train()
        sums = Counter()
        batches = 0
        for batch in loader:
            sample_ids = [str(value) for value in batch["sample_id"]]
            temporal, location = mars_coordinates(sample_ids, metadata, device)
            baseline = torch.tensor(
                [logit(champion_by_id[value]) for value in sample_ids],
                dtype=torch.float32,
                device=device,
            )
            batch = move_batch(batch, device)
            pair, observable = mars_pair(batch, device)
            values = prithvi_input(pair, observable, mean, std)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                output = model(values, temporal, location, batch["sensor_index"])
                candidate = (
                    baseline
                    + float(spec["training_strength"]) * output["bounded_correction"]
                )
                scene = F.binary_cross_entropy_with_logits(candidate, batch["presence"])
                target = patch_targets(batch, output["patch_logits"].shape[-1])
                patch_rows = patch_supervision_loss(
                    output["patch_logits"],
                    target,
                    batch["presence"],
                    batch["pixel_truth_available"],
                )
                patch = patch_rows.mean()
                positive = candidate[batch["presence"] > 0.5]
                negative = candidate[batch["presence"] < 0.5]
                pair_loss = (
                    F.softplus(
                        float(spec["pair_margin"])
                        - positive[:, None]
                        + negative[None, :]
                    ).mean()
                    if positive.numel() and negative.numel()
                    else candidate.sum() * 0.0
                )
                regularization = output["bounded_correction"].square().mean()
                loss = (
                    scene
                    + float(spec["patch_loss_weight"]) * patch
                    + float(spec["pair_loss_weight"]) * pair_loss
                    + float(spec["correction_l2_weight"]) * regularization
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = nn.utils.clip_grad_norm_(
                [value for value in model.parameters() if value.requires_grad],
                float(spec["gradient_clip"]),
            )
            if not torch.isfinite(norm):
                raise FloatingPointError("Non-finite scene-training gradient")
            optimizer.step()
            batches += 1
            for name, value in (
                ("loss", loss),
                ("scene", scene),
                ("patch", patch),
                ("pair", pair_loss),
                ("correction_l2", regularization),
            ):
                sums[name] += float(value.detach())
        row = {
            "epoch": float(epoch + 1),
            **{name: value / max(batches, 1) for name, value in sums.items()},
        }
        history.append(row)
        print(
            json.dumps({"phase": "scene", "seed": seed, "held_fold": held_fold, **row}),
            flush=True,
        )
    state = trainable_state(model)
    count = trainable_parameter_count(model)
    del model
    torch.cuda.empty_cache()
    return state, history, count


@torch.no_grad()
def score_scene_model(
    records: list[dict[str, Any]],
    metadata: dict[str, dict[str, Any]],
    paths: dict[str, Path],
    encoder: dict[str, torch.Tensor],
    scene_state: dict[str, torch.Tensor],
    protocol: dict[str, Any],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    spec = protocol["scene_training"]
    model, mean, std = build_scene_model(paths, encoder, spec, device, scene_state)
    model.eval()
    loader = DataLoader(
        MarsPaperDataset(paths["metadata_root"], records, augment=False, seed=0),
        batch_size=int(spec["evaluation_batch_size"]),
        shuffle=False,
        **loader_options(spec, workers_key="workers"),
    )
    identifiers: list[str] = []
    corrections: list[np.ndarray] = []
    for batch in loader:
        sample_ids = [str(value) for value in batch["sample_id"]]
        temporal, location = mars_coordinates(sample_ids, metadata, device)
        batch = move_batch(batch, device)
        pair, observable = mars_pair(batch, device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            output = model(
                prithvi_input(pair, observable, mean, std),
                temporal,
                location,
                batch["sensor_index"],
            )
        identifiers.extend(sample_ids)
        corrections.append(output["bounded_correction"].float().cpu().numpy())
    del model
    torch.cuda.empty_cache()
    result = np.concatenate(corrections).astype(np.float64)
    if not np.isfinite(result).all():
        raise FloatingPointError("Scene residual contains non-finite values")
    return np.asarray(identifiers), result


def corrected_scores(
    baseline: np.ndarray, correction: np.ndarray, strength: float
) -> np.ndarray:
    if not 0.0 < strength <= 1.0:
        raise ValueError("Correction strength must be in (0,1]")
    baseline = np.asarray(baseline, dtype=np.float64)
    correction = np.asarray(correction, dtype=np.float64)
    if baseline.shape != correction.shape:
        raise ValueError("Baseline and correction shapes differ")
    return expit(logit(np.clip(baseline, 1e-7, 1.0 - 1e-7)) + strength * correction)


def evaluate_candidate(
    values: dict[str, np.ndarray],
    correction: np.ndarray,
    strength: float,
    protocol: dict[str, Any],
    *,
    bootstrap_seed: int,
    final: bool,
) -> dict[str, Any]:
    baseline_scores = values["champion_scores"].astype(np.float64)
    scores = corrected_scores(baseline_scores, correction, strength)
    labels = values["labels"].astype(np.uint8)
    sensors = values["sensors"].astype(np.uint8)
    baseline = metric_summary(labels, baseline_scores, sensors)
    candidate = metric_summary(labels, scores, sensors)
    versus = comparison(candidate, baseline)
    per_fold = {}
    for fold in (3, 4):
        rows = values["folds"] == fold
        per_fold[str(fold)] = comparison(
            metric_summary(labels[rows], scores[rows], sensors[rows]),
            metric_summary(labels[rows], baseline_scores[rows], sensors[rows]),
        )
    bootstrap = ap_group_bootstrap(
        labels,
        baseline_scores,
        scores,
        values["groups"].astype(str),
        replicates=int(protocol["bootstrap"]["replicates"]),
        seed=bootstrap_seed,
    )
    gates = protocol["gates"]["final" if final else "pilot"]
    fold_ap = [value["delta"]["average_precision"] for value in per_fold.values()]
    fold_recall = [
        value["delta"]["recall_at_fpr_0_0713"] for value in per_fold.values()
    ]
    checks = {
        "pooled_ap": versus["delta"]["average_precision"]
        >= float(gates["average_precision_delta_minimum"]),
        "every_fold_ap": min(fold_ap) > 0.0,
        "every_sensor_ap": min(versus["delta"]["sensor_average_precision"].values())
        > 0.0,
        "pooled_recall": versus["delta"]["recall_at_fpr_0_0713"] >= 0.0,
        "every_fold_recall": min(fold_recall) >= 0.0,
        "paired_site_ap": bootstrap["lower"] > 0.0,
    }
    return {
        "strength": strength,
        "metrics": candidate,
        "versus_champion": versus,
        "per_fold": per_fold,
        "bootstrap": bootstrap,
        "checks": checks,
        "passed": all(checks.values()),
        "rank": [
            int(all(checks.values())),
            min(fold_ap),
            versus["delta"]["average_precision"],
            versus["delta"]["recall_at_fpr_0_0713"],
            -strength,
        ],
    }


def standalone_replication_checks(result: dict[str, Any]) -> dict[str, bool]:
    # Seed two is a genuine replication gate, not merely a contributor to the
    # two-seed mean. It must independently pass every frozen pilot check,
    # including sensor directionality, both-fold recall, and paired-site CI.
    return {f"seed_two_{name}": bool(value) for name, value in result["checks"].items()}


def save_torch_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def checkpoint_path(
    protocol: dict[str, Any], seed: int, held_fold: int, phase: str
) -> Path:
    directory = (ROOT / protocol["outputs"]["checkpoint_dir"]).resolve()
    return directory / f"seed_{seed}_held_{held_fold}_{phase}.pt"


def load_checkpoint(
    path: Path,
    *,
    protocol_hash: str,
    seed: int,
    held_fold: int,
    phase: str,
) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = torch.load(path, map_location="cpu", weights_only=True)
    expected = {
        "schema_version": 1,
        "protocol_sha256": protocol_hash,
        "seed": seed,
        "held_fold": held_fold,
        "phase": phase,
    }
    for name, value in expected.items():
        if payload.get(name) != value:
            raise ValueError(f"Stale or mismatched checkpoint {path}: {name}")
    return payload


def run_seed(
    seed: int,
    records: list[dict[str, Any]],
    metadata: dict[str, dict[str, Any]],
    group_to_fold: dict[str, int],
    auxiliary: list[dict[str, Any]],
    champion: dict[str, np.ndarray],
    paths: dict[str, Path],
    protocol: dict[str, Any],
    protocol_hash: str,
    device: torch.device,
    *,
    smoke: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    lookup = {
        value: index
        for index, value in enumerate(champion["sample_ids"].astype(str).tolist())
    }
    champion_by_id = {
        value: float(champion["champion_scores"][index])
        for value, index in lookup.items()
    }
    corrections = np.full(len(lookup), np.nan, dtype=np.float64)
    histories: dict[str, Any] = {}
    parameter_count: int | None = None
    for held_fold in (3, 4):
        fit_fold = 4 if held_fold == 3 else 3
        fit_records = [
            value
            for value in records
            if group_to_fold[str(value["group_id"])] == fit_fold
        ]
        pretrain_records = [
            value
            for value in fit_records
            if str(value.get("reference_scene_id", "")).strip()
        ]
        held_records = [
            value
            for value in records
            if group_to_fold[str(value["group_id"])] == held_fold
        ]
        encoder_checkpoint = checkpoint_path(protocol, seed, held_fold, "encoder")
        encoder_payload = (
            None
            if smoke
            else load_checkpoint(
                encoder_checkpoint,
                protocol_hash=protocol_hash,
                seed=seed,
                held_fold=held_fold,
                phase="encoder",
            )
        )
        if encoder_payload is None:
            encoder, pretrain_history, pretrain_receipt = pretrain_encoder(
                pretrain_records,
                metadata,
                auxiliary,
                paths,
                protocol,
                seed + held_fold,
                held_fold,
                device,
                smoke=smoke,
            )
            if not smoke:
                save_torch_atomic(
                    encoder_checkpoint,
                    {
                        "schema_version": 1,
                        "protocol_sha256": protocol_hash,
                        "seed": seed,
                        "held_fold": held_fold,
                        "phase": "encoder",
                        "encoder_state": encoder,
                        "history": pretrain_history,
                        "trainability_receipt": pretrain_receipt,
                    },
                )
        else:
            encoder = encoder_payload["encoder_state"]
            pretrain_history = encoder_payload["history"]
            pretrain_receipt = encoder_payload["trainability_receipt"]
        scene_checkpoint = checkpoint_path(protocol, seed, held_fold, "scene")
        scene_payload = (
            None
            if smoke
            else load_checkpoint(
                scene_checkpoint,
                protocol_hash=protocol_hash,
                seed=seed,
                held_fold=held_fold,
                phase="scene",
            )
        )
        if scene_payload is None:
            scene_state, scene_history, parameter_count = train_scene_model(
                fit_records,
                metadata,
                champion_by_id,
                paths,
                encoder,
                protocol,
                seed + 100 + held_fold,
                held_fold,
                device,
                smoke=smoke,
            )
            if not smoke:
                save_torch_atomic(
                    scene_checkpoint,
                    {
                        "schema_version": 1,
                        "protocol_sha256": protocol_hash,
                        "seed": seed,
                        "held_fold": held_fold,
                        "phase": "scene",
                        "trainable_state": scene_state,
                        "history": scene_history,
                        "trainable_parameters": parameter_count,
                    },
                )
        else:
            scene_state = scene_payload["trainable_state"]
            scene_history = scene_payload["history"]
            parameter_count = int(scene_payload["trainable_parameters"])
        identifiers, local = score_scene_model(
            held_records,
            metadata,
            paths,
            encoder,
            scene_state,
            protocol,
            device,
        )
        positions = np.asarray([lookup[value] for value in identifiers])
        if np.any(np.isfinite(corrections[positions])):
            raise RuntimeError("Cross-fit scene correction wrote a row twice")
        corrections[positions] = local
        histories[str(held_fold)] = {
            "fit_fold": fit_fold,
            "fit_rows": len(fit_records),
            "pretrain_rows": len(pretrain_records),
            "held_rows": len(held_records),
            "pretraining": pretrain_history,
            "scene": scene_history,
            "trainability_receipt": pretrain_receipt,
        }
        print(
            json.dumps(
                {
                    "completed_seed": seed,
                    "held_fold": held_fold,
                    "fit_rows": len(fit_records),
                    "held_rows": len(held_records),
                }
            ),
            flush=True,
        )
        if smoke:
            break
    if smoke:
        return corrections, {
            "histories": histories,
            "trainable_parameters": parameter_count,
        }
    if not np.isfinite(corrections).all():
        raise RuntimeError("Cross-fit scene corrections are incomplete")
    return corrections, {
        "histories": histories,
        "trainable_parameters": parameter_count,
    }


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    pilot = report.get("pilot_selected")
    final = report.get("final")
    lines = [
        "# Spectral-temporal domain-adaptive Prithvi",
        "",
        "This experiment continued Prithvi masked-autoencoder pretraining on only the opposite MARS development fold and a fixed external auxiliary cohort. It never loaded MARS folds 0/1/2 or the official test.",
        "",
        f"- Decision: **{report['decision']}**",
        f"- Pilot passed: {'yes' if report['pilot_passed'] else 'no'}",
    ]
    if pilot is not None:
        delta = pilot["versus_champion"]["delta"]
        lines.extend(
            (
                f"- Pilot strength: {pilot['strength']:.3f}",
                f"- Pilot AP delta: {delta['average_precision']:+.6f}",
                f"- Pilot recall delta at matched FPR: {delta['recall_at_fpr_0_0713']:+.6f}",
                f"- Pilot paired-site AP interval: [{pilot['bootstrap']['lower']:+.6f}, {pilot['bootstrap']['upper']:+.6f}]",
            )
        )
    if final is not None:
        delta = final["versus_champion"]["delta"]
        lines.extend(
            (
                f"- Two-seed AP delta: {delta['average_precision']:+.6f}",
                f"- Two-seed recall delta at matched FPR: {delta['recall_at_fpr_0_0713']:+.6f}",
                f"- Two-seed paired-site AP interval: [{final['bootstrap']['lower']:+.6f}, {final['bootstrap']['upper']:+.6f}]",
                f"- Final promotion gates passed: {'yes' if report['all_promotion_gates_pass'] else 'no'}",
            )
        )
    lines.extend(("", report["decision_detail"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL.as_posix())
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    protocol_path = (ROOT / args.protocol).resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    paths = verify_protocol(protocol_path, protocol, smoke=args.smoke)
    verify_acquisition_receipt(paths["acquisition_receipt"], sha256(paths["manifest"]))
    fold_protocol = json.loads(paths["fold_protocol"].read_text(encoding="utf-8"))
    group_to_fold = {
        str(item["group_id"]): int(item["fold"])
        for item in fold_protocol["assignments"]
    }
    all_records = list(iter_development_manifest(paths["manifest"]))
    records = [
        value
        for value in all_records
        if group_to_fold[str(value["group_id"])] in {3, 4}
    ]
    metadata = {str(value["sample_id"]): value for value in records}
    if len(metadata) != len(records):
        raise ValueError("MARS folds3/4 sample identifiers are not unique")
    auxiliary = read_jsonl(paths["methanes2cm_auxiliary"])
    champion = load_champion(paths["champion_cache"])
    if set(champion["sample_ids"].astype(str).tolist()) != set(metadata):
        raise ValueError("Champion cache and folds3/4 manifest identities differ")
    by_id = {str(value["sample_id"]): value for value in records}
    for index, identifier in enumerate(champion["sample_ids"].astype(str)):
        record = by_id[identifier]
        if int(champion["folds"][index]) != group_to_fold[str(record["group_id"])]:
            raise ValueError("Champion fold assignment differs from manifest")
        if int(champion["labels"][index]) != int(record["label_state"] == "PLUME"):
            raise ValueError("Champion label differs from manifest")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Domain-adaptive Prithvi training requires CUDA")
    protocol_hash = sha256(protocol_path)
    if args.smoke:
        smoke_records: list[dict[str, Any]] = []
        counts: Counter[tuple[int, str, str]] = Counter()
        for record in records:
            fold = group_to_fold[str(record["group_id"])]
            key = (fold, str(record["sensor_family"]), str(record["label_state"]))
            if counts[key] < 2:
                smoke_records.append(record)
                counts[key] += 1
        correction, details = run_seed(
            int(protocol["seeds"]["pilot"]),
            smoke_records,
            {str(value["sample_id"]): value for value in smoke_records},
            group_to_fold,
            auxiliary[:32],
            {
                name: (
                    np.asarray(
                        [
                            champion[name][
                                int(
                                    np.flatnonzero(
                                        champion["sample_ids"].astype(str)
                                        == str(record["sample_id"])
                                    )[0]
                                )
                            ]
                            for record in smoke_records
                        ]
                    )
                    if name != "sample_ids"
                    else np.asarray(
                        [str(value["sample_id"]) for value in smoke_records]
                    )
                )
                for name in champion
            },
            paths,
            protocol,
            protocol_hash,
            device,
            smoke=True,
        )
        ok = bool(np.isfinite(correction).any() and details["trainable_parameters"])
        print(json.dumps({"ok": ok, "smoke": details}, indent=2))
        return 0 if ok else 1

    pilot_seed = int(protocol["seeds"]["pilot"])
    pilot_correction, pilot_details = run_seed(
        pilot_seed,
        records,
        metadata,
        group_to_fold,
        auxiliary,
        champion,
        paths,
        protocol,
        protocol_hash,
        device,
        smoke=False,
    )
    pilot_candidates = [
        evaluate_candidate(
            champion,
            pilot_correction,
            float(strength),
            protocol,
            bootstrap_seed=int(protocol["bootstrap"]["pilot_seed"]) + index,
            final=False,
        )
        for index, strength in enumerate(protocol["search"]["strengths"])
    ]
    pilot_selected = max(pilot_candidates, key=lambda value: tuple(value["rank"]))
    pilot_passed = bool(pilot_selected["passed"])
    replicate_correction = None
    replicate_details = None
    replicate_standalone = None
    final = None
    final_scores = None
    passed = False
    if pilot_passed:
        replicate_seed = int(protocol["seeds"]["replication"])
        replicate_correction, replicate_details = run_seed(
            replicate_seed,
            records,
            metadata,
            group_to_fold,
            auxiliary,
            champion,
            paths,
            protocol,
            protocol_hash,
            device,
            smoke=False,
        )
        replicate_result = evaluate_candidate(
            champion,
            replicate_correction,
            float(pilot_selected["strength"]),
            protocol,
            bootstrap_seed=int(protocol["bootstrap"]["replication_seed"]),
            final=False,
        )
        replicate_standalone = standalone_replication_checks(replicate_result)
        mean_correction = 0.5 * (pilot_correction + replicate_correction)
        final = evaluate_candidate(
            champion,
            mean_correction,
            float(pilot_selected["strength"]),
            protocol,
            bootstrap_seed=int(protocol["bootstrap"]["final_seed"]),
            final=True,
        )
        passed = bool(final["passed"] and all(replicate_standalone.values()))
        final_scores = corrected_scores(
            champion["champion_scores"].astype(np.float64),
            mean_correction,
            float(pilot_selected["strength"]),
        )
    decision = "PROMOTE" if passed else "REJECT"
    detail = (
        "The two-seed development gate passed. Freeze the candidate before any external confirmation."
        if passed
        else (
            "The pilot failed its preregistered development gate; replication and all external/official evaluation were skipped."
            if not pilot_passed
            else "The pilot replicated, but the two-seed candidate failed at least one final promotion or seed-stability gate."
        )
    )
    artifact_record = None
    score_record = None
    if passed and replicate_correction is not None and final_scores is not None:
        artifact_path = (ROOT / protocol["outputs"]["artifact"]).resolve()
        encoder_scene_payloads = []
        for seed in (pilot_seed, int(protocol["seeds"]["replication"])):
            for held_fold in (3, 4):
                encoder_payload = load_checkpoint(
                    checkpoint_path(protocol, seed, held_fold, "encoder"),
                    protocol_hash=protocol_hash,
                    seed=seed,
                    held_fold=held_fold,
                    phase="encoder",
                )
                scene_payload = load_checkpoint(
                    checkpoint_path(protocol, seed, held_fold, "scene"),
                    protocol_hash=protocol_hash,
                    seed=seed,
                    held_fold=held_fold,
                    phase="scene",
                )
                assert encoder_payload is not None and scene_payload is not None
                encoder_scene_payloads.append(
                    {
                        "seed": seed,
                        "held_fold": held_fold,
                        "encoder_state": encoder_payload["encoder_state"],
                        "scene_state": scene_payload["trainable_state"],
                    }
                )
        save_torch_atomic(
            artifact_path,
            {
                "schema_version": 1,
                "kind": "mars_prithvi_domain_adaptive_residual",
                "protocol_sha256": protocol_hash,
                "strength": float(pilot_selected["strength"]),
                "external_member_aggregation": protocol["data_contract"][
                    "external_member_aggregation"
                ],
                "members": encoder_scene_payloads,
            },
        )
        artifact_record = {
            "path": protocol["outputs"]["artifact"],
            "bytes": artifact_path.stat().st_size,
            "sha256": sha256(artifact_path),
            "tracked": False,
        }
        score_path = (ROOT / protocol["outputs"]["development_scores"]).resolve()
        score_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = score_path.with_suffix(".tmp.npz")
        np.savez_compressed(
            temporary,
            sample_ids=champion["sample_ids"],
            labels=champion["labels"],
            sensors=champion["sensors"],
            groups=champion["groups"],
            folds=champion["folds"],
            champion_scores=champion["champion_scores"],
            pilot_correction=pilot_correction,
            replication_correction=replicate_correction,
            candidate_scores=final_scores,
        )
        os.replace(temporary, score_path)
        score_record = {
            "path": protocol["outputs"]["development_scores"],
            "bytes": score_path.stat().st_size,
            "sha256": sha256(score_path),
            "tracked": False,
        }
    report = {
        "schema_version": 1,
        "scope": "strict MARS folds3/4 cross-fit; no folds0/1/2 or official data",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "decision_detail": detail,
        "pilot_candidates": pilot_candidates,
        "pilot_selected": pilot_selected,
        "pilot_passed": pilot_passed,
        "replication_standalone_checks": replicate_standalone,
        "final": final,
        "all_promotion_gates_pass": passed,
        "training": {
            "pilot": pilot_details,
            "replication": replicate_details,
        },
        "artifact": artifact_record,
        "development_scores": score_record,
        "provenance": {
            "protocol_sha256": protocol_hash,
            "script_sha256": sha256(Path(__file__).resolve()),
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "device": torch.cuda.get_device_name(device),
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
    }
    output_json = (ROOT / protocol["outputs"]["json"]).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_markdown((ROOT / protocol["outputs"]["markdown"]).resolve(), report)
    print(
        json.dumps(
            {
                "ok": passed,
                "decision": decision,
                "pilot_ap_delta": pilot_selected["versus_champion"]["delta"][
                    "average_precision"
                ],
                "pilot_ap_lower": pilot_selected["bootstrap"]["lower"],
                "final_ap_delta": None
                if final is None
                else final["versus_champion"]["delta"]["average_precision"],
            },
            indent=2,
        )
    )
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
