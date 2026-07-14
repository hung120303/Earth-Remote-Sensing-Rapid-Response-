#!/usr/bin/env python3
"""Fit the MARS residual with CH4 weighting and wind-matched plume simulation."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import time
from collections import Counter, OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import scipy
import sklearn
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler, get_worker_info

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "EarthRemoteSensingRapidResponse"
if str(MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_ROOT))
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from acquire_mars_metadata import DEFAULT_OUTPUT, repo_root, sha256
from mars_s2l_adapter import (
    REFLECTANCE_DIVISOR,
    REFLECTANCE_MAX,
    compute_mbmp,
    load_sample,
)
from mars_v4_simulation import MarsPlumeSimulator
from train_mars_paper_residual import (
    DEFAULT_ACQUISITION_RECEIPT,
    DEFAULT_CHECKPOINT,
    DEFAULT_MANIFEST,
    DEFAULT_PROTOCOL,
    SENSOR_NAMES,
    MarsPaperDataset,
    MarsPaperResidualModel,
    available_smoke_subset,
    iter_development_manifest,
    move_batch,
    rotate_wind,
    seed_everything,
    validation_summary,
    verify_acquisition_receipt,
    write_json,
)

DEFAULT_PARENT_ARTIFACT = Path(
    "EarthRemoteSensingRapidResponse/artifacts/mars_paper_residual_fold0_seed606.pt"
)
DEFAULT_PARENT_SHA256 = (
    "b94880d858e1e7791591eeb5f7d0da9be84b99a324e980437ebe83cfae6c7d49"
)
DEFAULT_METADATA_CSV = DEFAULT_OUTPUT / "validated_images_all.csv"
DEFAULT_LUT = Path("configs/mars_s2_integrated_transmittances.json")
DEFAULT_ARTIFACT = Path(
    "EarthRemoteSensingRapidResponse/artifacts/mars_source_aligned_fold0_seed707.pt"
)
DEFAULT_JSON = Path("reports/experiments/mars_source_aligned_fold0_seed707.json")
DEFAULT_MARKDOWN = Path(
    "reports/experiments/MARS_SOURCE_ALIGNED_FOLD0_SEED707.md"
)


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def offshore_flags(path: Path, required_ids: set[str]) -> dict[str, bool]:
    result: dict[str, bool] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        for row in csv.DictReader(source):
            sample_id = str(row["id_loc_image"])
            if sample_id in required_ids:
                result[sample_id] = truthy(row.get("offshore"))
    missing = required_ids - set(result)
    if missing:
        raise ValueError(f"Metadata CSV lacks offshore flags for {len(missing)} rows")
    return result


def source_aligned_sampling_weights(records: list[dict[str, Any]]) -> torch.Tensor:
    cells = [
        (str(record["group_id"]), str(record["label_state"]), str(record["sensor_family"]))
        for record in records
    ]
    counts = Counter(cells)
    return torch.tensor([1.0 / counts[cell] for cell in cells], dtype=torch.double)


@torch.no_grad()
def contract_residual_strength(model: MarsPaperResidualModel, strength: float) -> None:
    """Represent released + strength*(trained-released) in the model parameters."""
    if not 0.0 <= strength <= 1.0:
        raise ValueError("Residual strength must be in [0,1]")
    model.correction.output.weight.mul_(strength)
    model.correction.output.bias.mul_(strength)
    old_scale = model.sensor_log_scale.exp()
    model.sensor_log_scale.copy_(torch.log1p(strength * (old_scale - 1.0)))
    model.sensor_bias.mul_(strength)


class MarsSourceAlignedDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        metadata_dir: Path,
        records: list[dict[str, Any]],
        offshore_by_id: dict[str, bool],
        *,
        lut_path: Path,
        augment: bool,
        simulation_fraction: float,
        crop_size: int,
        seed: int,
    ) -> None:
        if not 0.0 <= simulation_fraction <= 1.0:
            raise ValueError("Simulation fraction must be in [0,1]")
        if crop_size <= 0:
            raise ValueError("Crop size must be positive")
        self.metadata_dir = metadata_dir
        self.records = records
        self.offshore_by_id = offshore_by_id
        self.lut_path = lut_path
        self.augment = augment
        self.simulation_fraction = simulation_fraction
        self.crop_size = crop_size
        self.seed = seed
        self.positive_indices = [
            index
            for index, record in enumerate(records)
            if record["label_state"] == "PLUME"
            and str(record.get("observability", "")).lower() != "bad_retrieval"
        ]
        self.negative_indices = [
            index
            for index, record in enumerate(records)
            if record["label_state"] == "NO_PLUME"
            and not offshore_by_id[str(record["sample_id"])]
            and float(np.hypot(record["wind_u"], record["wind_v"])) <= 9.0
            and str(record.get("observability", "")).lower() == "clear"
        ]
        self.positive_speeds = np.asarray(
            [
                np.hypot(records[index]["wind_u"], records[index]["wind_v"])
                for index in self.positive_indices
            ],
            dtype=np.float32,
        )
        if augment and simulation_fraction > 0 and (
            not self.positive_indices or not self.negative_indices
        ):
            raise ValueError("Simulation requires fit-fold plume and clear no-plume rows")
        self._rng: np.random.Generator | None = None
        self._simulator: MarsPlumeSimulator | None = None
        self._source_cache: OrderedDict[str, Any] = OrderedDict()

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

    def source_sample(self, index: int) -> Any:
        record = self.records[index]
        key = str(record["sample_id"])
        if key in self._source_cache:
            sample = self._source_cache.pop(key)
            self._source_cache[key] = sample
            return sample
        sample = load_sample(
            self.metadata_dir,
            record,
            require_enhancement=True,
            allow_empty_positive_mask=False,
        )
        self._source_cache[key] = sample
        while len(self._source_cache) > 8:
            self._source_cache.popitem(last=False)
        return sample

    def simulate(self) -> tuple[dict[str, Any], Any, np.ndarray, np.ndarray, np.ndarray] | None:
        rng = self.rng()
        target_index = int(rng.choice(self.negative_indices))
        target_record = self.records[target_index]
        target_speed = float(np.hypot(target_record["wind_u"], target_record["wind_v"]))
        candidates = np.flatnonzero(np.abs(self.positive_speeds - target_speed) <= 1.5)
        if candidates.size == 0:
            return None
        source_index = self.positive_indices[int(rng.choice(candidates))]
        source_record = self.records[source_index]
        source = self.source_sample(source_index)
        target = load_sample(
            self.metadata_dir,
            target_record,
            require_enhancement=False,
            allow_empty_positive_mask=False,
        )
        if source.methane_enhancement_raw is None:
            return None
        result = self.simulator().simulate(
            target.raw_pair[:6],
            source.methane_enhancement_raw,
            source.plume_mask,
            source_wind=(float(source_record["wind_u"]), float(source_record["wind_v"])),
            target_wind=(float(target_record["wind_u"]), float(target_record["wind_v"])),
            satellite=str(target_record["satellite"]),
            solar_zenith_degrees=float(target_record["solar_zenith_angle"]),
            view_zenith_degrees=float(target_record["view_zenith_angle"]),
            rng=rng,
        )
        visible = result.mask & target.observable_mask
        if np.count_nonzero(visible) / max(np.count_nonzero(result.mask), 1) < 0.5:
            return None
        raw_pair = np.concatenate([result.target, target.raw_pair[6:]], axis=0)
        reflectance = np.clip(
            raw_pair.astype(np.float32) / REFLECTANCE_DIVISOR,
            0.0,
            REFLECTANCE_MAX,
        )
        return (
            target_record,
            target,
            reflectance,
            result.mask.astype(np.float32),
            result.delta_ch4.astype(np.float32),
        )

    def crop_start(self, mask: np.ndarray) -> tuple[int, int]:
        height, width = mask.shape
        if self.crop_size > height or self.crop_size > width:
            raise ValueError("Training crop exceeds a source image")
        row_limit = height - self.crop_size
        col_limit = width - self.crop_size
        if np.any(mask):
            rows, cols = np.nonzero(mask)
            row = int(np.clip(round((rows.min() + rows.max()) / 2) - self.crop_size // 2, 0, row_limit))
            col = int(np.clip(round((cols.min() + cols.max()) / 2) - self.crop_size // 2, 0, col_limit))
            return row, col
        rng = self.rng()
        return (
            int(rng.integers(0, row_limit + 1)),
            int(rng.integers(0, col_limit + 1)),
        )

    def __getitem__(self, index: int) -> dict[str, Any]:
        requested = self.records[index]
        simulation = None
        if (
            self.augment
            and requested["label_state"] == "PLUME"
            and self.rng().random() < self.simulation_fraction
        ):
            try:
                simulation = self.simulate()
            except (OSError, ValueError, FloatingPointError):
                simulation = None
        if simulation is None:
            record = requested
            sample = load_sample(
                self.metadata_dir,
                record,
                require_enhancement=record["label_state"] == "PLUME",
                allow_empty_positive_mask=True,
            )
            spectral = sample.reflectance_pair.copy()
            mask = sample.plume_mask.astype(np.float32)
            enhancement = (
                np.zeros_like(mask, dtype=np.float32)
                if sample.methane_enhancement_raw is None
                else sample.methane_enhancement_raw.astype(np.float32)
            )
            simulated = False
        else:
            record, sample, spectral, mask, enhancement = simulation
            simulated = True
        cloud = (sample.cloud_classes > 0).astype(np.float32)
        clear = sample.clear_mask.astype(np.float32)
        observable = sample.observable_mask.astype(np.float32)
        row, col = self.crop_start(mask)
        rows = slice(row, row + self.crop_size)
        cols = slice(col, col + self.crop_size)
        spectral = spectral[:, rows, cols].copy()
        mask = mask[rows, cols].copy()
        enhancement = enhancement[rows, cols].copy()
        cloud = cloud[rows, cols].copy()
        clear = clear[rows, cols].copy()
        observable = observable[rows, cols].copy()
        wind = (float(record["wind_u"]), float(record["wind_v"]))
        mbmp = compute_mbmp(spectral[:6], spectral[6:])
        if self.augment:
            rng = self.rng()
            turns = int(rng.integers(0, 4))
            if turns:
                spectral = np.rot90(spectral, turns, axes=(1, 2)).copy()
                mask = np.rot90(mask, turns).copy()
                enhancement = np.rot90(enhancement, turns).copy()
                cloud = np.rot90(cloud, turns).copy()
                clear = np.rot90(clear, turns).copy()
                observable = np.rot90(observable, turns).copy()
                mbmp = np.rot90(mbmp, turns).copy()
                wind = rotate_wind(wind, turns)
            if bool(rng.integers(0, 2)):
                spectral = spectral[:, :, ::-1].copy()
                mask = mask[:, ::-1].copy()
                enhancement = enhancement[:, ::-1].copy()
                cloud = cloud[:, ::-1].copy()
                clear = clear[:, ::-1].copy()
                observable = observable[:, ::-1].copy()
                mbmp = mbmp[:, ::-1].copy()
                wind = (-wind[0], wind[1])
            if bool(rng.integers(0, 2)):
                spectral = spectral[:, ::-1, :].copy()
                mask = mask[::-1, :].copy()
                enhancement = enhancement[::-1, :].copy()
                cloud = cloud[::-1, :].copy()
                clear = clear[::-1, :].copy()
                observable = observable[::-1, :].copy()
                mbmp = mbmp[::-1, :].copy()
                wind = (wind[0], -wind[1])
        wind_channels = np.broadcast_to(
            np.asarray(wind, dtype=np.float32)[:, None, None] / 8.0,
            (2, self.crop_size, self.crop_size),
        ).copy()
        inputs = np.concatenate(
            [mbmp[None], spectral, wind_channels, cloud[None]], axis=0
        ).astype(np.float32)
        return {
            "inputs": torch.from_numpy(inputs),
            "observable": torch.from_numpy(observable[None]),
            "clear": torch.from_numpy(clear[None]),
            "mask": torch.from_numpy(mask[None]),
            "enhancement": torch.from_numpy(enhancement[None]),
            "presence": torch.tensor(float(record["label_state"] == "PLUME" or simulated)),
            "sensor_index": torch.tensor(SENSOR_NAMES.index(sample.sensor_family)),
            "simulated": torch.tensor(float(simulated)),
            "sample_id": sample.sample_id,
            "group_id": str(record["group_id"]),
        }


def source_aligned_loss(
    output: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    scene_weight: float,
    negative_upward_weight: float,
    positive_downward_weight: float,
    correction_l2_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    target = batch["mask"]
    logits = output["segmentation_logits"]
    positive_weight = torch.tensor(10.0, device=logits.device, dtype=logits.dtype)
    bce = F.binary_cross_entropy_with_logits(
        logits, target, reduction="none", pos_weight=positive_weight
    )
    ch4_weight = torch.clamp(batch["enhancement"], 100.0, 2000.0) / 1000.0
    weights = ch4_weight * target + (1.0 - target)
    segmentation = (bce * weights).mean()
    scene = F.binary_cross_entropy_with_logits(output["scene_logit"], batch["presence"])
    negative = (batch["presence"] < 0.5)[:, None, None, None]
    valid_negative = negative & (batch["observable"] > 0.5)
    positive = (target > 0.5) & (batch["observable"] > 0.5)
    upward = F.relu(logits - output["baseline_logits"].detach())
    downward = F.relu(output["baseline_logits"].detach() - logits)
    upward_penalty = (upward * valid_negative).sum() / valid_negative.sum().clamp_min(1)
    downward_penalty = (downward * positive).sum() / positive.sum().clamp_min(1)
    correction_penalty = output["correction_logits"].square().mean()
    total = (
        segmentation
        + scene_weight * scene
        + negative_upward_weight * upward_penalty
        + positive_downward_weight * downward_penalty
        + correction_l2_weight * correction_penalty
    )
    return total, {
        "total": float(total.detach()),
        "ch4_weighted_bce": float(segmentation.detach()),
        "scene_bce": float(scene.detach()),
        "negative_upward_penalty": float(upward_penalty.detach()),
        "positive_downward_penalty": float(downward_penalty.detach()),
        "correction_l2": float(correction_penalty.detach()),
        "simulated_fraction": float(batch["simulated"].mean().detach()),
    }


def artifact_payload(
    model: MarsPaperResidualModel,
    *,
    fold: int,
    seed: int,
    epoch: int,
    protocol_hash: str,
    validation: dict[str, Any],
    configuration: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "model": model.artifact_metadata(),
        "fold": fold,
        "seed": seed,
        "epoch": epoch,
        "protocol_sha256": protocol_hash,
        "parent_artifact_sha256": DEFAULT_PARENT_SHA256,
        "correction_state_dict": model.correction.state_dict(),
        "sensor_log_scale": model.sensor_log_scale.detach().cpu(),
        "sensor_bias": model.sensor_bias.detach().cpu(),
        "validation": validation,
        "configuration": configuration,
    }


def epoch_snapshot_path(artifact: Path, epoch: int) -> Path:
    if epoch <= 0:
        raise ValueError("snapshot epoch must be positive")
    return artifact.with_name(f"{artifact.stem}_epoch{epoch}{artifact.suffix}")


def train(
    model: MarsPaperResidualModel,
    train_loader: DataLoader[dict[str, Any]],
    validation_loader: DataLoader[dict[str, Any]],
    device: torch.device,
    artifact: Path,
    *,
    fold: int,
    seed: int,
    epochs: int,
    learning_rate: float,
    patience: int,
    protocol_hash: str,
    configuration: dict[str, Any],
    snapshot_epochs: frozenset[int] = frozenset(),
) -> tuple[list[dict[str, Any]], int]:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history: list[dict[str, Any]] = []
    best_rank = (-math.inf, -math.inf, -math.inf)
    best_epoch = -1
    stale = 0
    baseline_reference: dict[str, Any] | None = None
    artifact.parent.mkdir(parents=True, exist_ok=True)
    loss_args = {
        key: configuration[key]
        for key in (
            "scene_weight",
            "negative_upward_weight",
            "positive_downward_weight",
            "correction_l2_weight",
        )
    }
    for epoch in range(1, epochs + 1):
        model.train()
        started = time.perf_counter()
        parts: list[dict[str, float]] = []
        for batch in train_loader:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                output = model(batch["inputs"], batch["observable"], batch["sensor_index"])
                loss, values = source_aligned_loss(output, batch, **loss_args)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(parameters, 2.0)
            scaler.step(optimizer)
            scaler.update()
            parts.append(values)
        scheduler.step()
        validation = validation_summary(
            model, validation_loader, device, baseline_reference=baseline_reference
        )
        if baseline_reference is None:
            baseline_reference = validation
        ap_delta = float(validation["delta"]["average_precision"])
        iou_delta = float(validation["delta"]["pixel_iou"])
        rank = (
            min(ap_delta, iou_delta),
            ap_delta + iou_delta,
            float(validation["delta"]["recall_at_fpr_0_0713"]),
        )
        record = {
            "epoch": epoch,
            "seconds": round(time.perf_counter() - started, 3),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "loss": {key: float(np.mean([part[key] for part in parts])) for key in parts[0]},
            "validation": validation,
        }
        history.append(record)
        print(json.dumps({"epoch": epoch, "rank": rank, "seconds": record["seconds"]}), flush=True)
        payload = artifact_payload(
            model,
            fold=fold,
            seed=seed,
            epoch=epoch,
            protocol_hash=protocol_hash,
            validation=validation,
            configuration=configuration,
        )
        if epoch in snapshot_epochs:
            snapshot = epoch_snapshot_path(artifact, epoch)
            torch.save(payload, snapshot)
            print(json.dumps({"snapshot": snapshot.as_posix(), "epoch": epoch}), flush=True)
        if rank > best_rank:
            best_rank = rank
            best_epoch = epoch
            stale = 0
            torch.save(
                payload,
                artifact,
            )
        else:
            stale += 1
            if stale >= patience:
                break
    return history, best_epoch


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    best = report["best_validation"]
    candidate = best["candidate"]
    baseline = best["released_baseline"]
    delta = best["delta"]
    lines = [
        "# Source-aligned MARS residual fold-0 validation",
        "",
        f"- Scope: {'smoke only' if report['experiment']['smoke'] else 'frozen site-held development'}",
        f"- Best epoch: {report['training']['best_epoch']}",
        "",
        "| Model | AP | Recall at <=7.13% FPR | Pixel IoU |",
        "|---|---:|---:|---:|",
        f"| Released MARS-S2L | {baseline['average_precision']:.5f} | {baseline['operating_points']['0.0713']['recall']:.5f} | {baseline['pixel_fixed_0_5']['intersection_over_union']:.5f} |",
        f"| Source-aligned residual | {candidate['average_precision']:.5f} | {candidate['operating_points']['0.0713']['recall']:.5f} | {candidate['pixel_fixed_0_5']['intersection_over_union']:.5f} |",
        "",
        f"Deltas: AP {delta['average_precision']:+.5f}, recall {delta['recall_at_fpr_0_0713']:+.5f}, IoU {delta['pixel_iou']:+.5f}.",
        "",
        report["decision"],
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-dir", default=DEFAULT_OUTPUT.as_posix())
    parser.add_argument("--metadata-csv", default=DEFAULT_METADATA_CSV.as_posix())
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST.as_posix())
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL.as_posix())
    parser.add_argument("--acquisition-receipt", default=DEFAULT_ACQUISITION_RECEIPT.as_posix())
    parser.add_argument("--released-checkpoint", default=DEFAULT_CHECKPOINT.as_posix())
    parser.add_argument("--parent-artifact", default=DEFAULT_PARENT_ARTIFACT.as_posix())
    parser.add_argument("--parent-sha256", default=DEFAULT_PARENT_SHA256)
    parser.add_argument("--lut", default=DEFAULT_LUT.as_posix())
    parser.add_argument("--artifact", default=DEFAULT_ARTIFACT.as_posix())
    parser.add_argument("--output-json", default=DEFAULT_JSON.as_posix())
    parser.add_argument("--output-markdown", default=DEFAULT_MARKDOWN.as_posix())
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=707)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--samples-per-epoch", type=int, default=32768)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--initial-strength", type=float, default=0.5)
    parser.add_argument("--simulation-fraction", type=float, default=0.5)
    parser.add_argument("--crop-size", type=int, default=192)
    parser.add_argument("--scene-weight", type=float, default=0.05)
    parser.add_argument("--negative-upward-weight", type=float, default=0.10)
    parser.add_argument("--positive-downward-weight", type=float, default=0.10)
    parser.add_argument("--correction-l2-weight", type=float, default=0.001)
    parser.add_argument(
        "--snapshot-epoch",
        type=int,
        action="append",
        default=[],
        help="Also save an ignored checkpoint after this epoch; may be repeated.",
    )
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.epochs <= 0 or args.samples_per_epoch <= 0 or args.batch_size <= 0:
        parser.error("epochs, samples-per-epoch, and batch-size must be positive")
    if not 0.0 <= args.simulation_fraction <= 1.0:
        parser.error("simulation fraction must be in [0,1]")
    if any(epoch <= 0 or epoch > args.epochs for epoch in args.snapshot_epoch):
        parser.error("snapshot epochs must be within the requested training range")
    root = repo_root()
    seed_everything(args.seed)
    metadata_dir = (root / args.metadata_dir).resolve()
    manifest = (root / args.manifest).resolve()
    protocol_path = (root / args.protocol).resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    manifest_hash = sha256(manifest)
    if manifest_hash != protocol["development_manifest_sha256"]:
        raise ValueError("Development manifest differs from frozen protocol")
    group_to_fold = {
        str(item["group_id"]): int(item["fold"])
        for item in protocol["assignments"]
    }
    records = list(iter_development_manifest(manifest))
    fit_records = [record for record in records if group_to_fold[str(record["group_id"])] != args.fold]
    validation_records = [record for record in records if group_to_fold[str(record["group_id"])] == args.fold]
    if args.smoke:
        fit_records = available_smoke_subset(metadata_dir, fit_records)
        validation_records = available_smoke_subset(metadata_dir, validation_records)
        args.epochs = 1
        args.samples_per_epoch = min(args.samples_per_epoch, 128)
        args.batch_size = min(args.batch_size, 4)
        args.workers = min(args.workers, 2)
    else:
        verify_acquisition_receipt((root / args.acquisition_receipt).resolve(), manifest_hash)
    required_ids = {str(record["sample_id"]) for record in fit_records}
    flags = offshore_flags((root / args.metadata_csv).resolve(), required_ids)
    train_dataset = MarsSourceAlignedDataset(
        metadata_dir,
        fit_records,
        flags,
        lut_path=(root / args.lut).resolve(),
        augment=True,
        simulation_fraction=args.simulation_fraction,
        crop_size=args.crop_size,
        seed=args.seed,
    )
    validation_dataset = MarsPaperDataset(metadata_dir, validation_records, augment=False, seed=args.seed)
    sampler = WeightedRandomSampler(
        source_aligned_sampling_weights(fit_records),
        num_samples=args.samples_per_epoch,
        replacement=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    options = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": True,
        "persistent_workers": args.workers > 0,
    }
    train_loader = DataLoader(train_dataset, sampler=sampler, **options)
    validation_loader = DataLoader(validation_dataset, shuffle=False, **options)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MarsPaperResidualModel().to(device)
    released_checkpoint = (root / args.released_checkpoint).resolve()
    model.load_released_checkpoint(released_checkpoint)
    parent_path = (root / args.parent_artifact).resolve()
    if sha256(parent_path) != args.parent_sha256:
        raise ValueError("Parent residual artifact hash mismatch")
    parent = torch.load(parent_path, map_location="cpu", weights_only=True)
    if int(parent["fold"]) != args.fold or parent["protocol_sha256"] != sha256(protocol_path):
        raise ValueError("Parent residual artifact covers a different experiment")
    model.correction.load_state_dict(parent["correction_state_dict"])
    with torch.no_grad():
        model.sensor_log_scale.copy_(parent["sensor_log_scale"].to(device))
        model.sensor_bias.copy_(parent["sensor_bias"].to(device))
    contract_residual_strength(model, args.initial_strength)
    configuration = {
        "initial_strength": args.initial_strength,
        "simulation_fraction": args.simulation_fraction,
        "wind_speed_tolerance_mps": 1.5,
        "maximum_target_wind_mps": 9.0,
        "crop_size": args.crop_size,
        "positive_bce_weight": 10.0,
        "ch4_weight_min_ppb": 100.0,
        "ch4_weight_max_ppb": 2000.0,
        "ch4_weight_scale_ppb": 1000.0,
        "scene_weight": args.scene_weight,
        "negative_upward_weight": args.negative_upward_weight,
        "positive_downward_weight": args.positive_downward_weight,
        "correction_l2_weight": args.correction_l2_weight,
    }
    artifact = (root / args.artifact).resolve()
    history, best_epoch = train(
        model,
        train_loader,
        validation_loader,
        device,
        artifact,
        fold=args.fold,
        seed=args.seed,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        patience=args.patience,
        protocol_hash=sha256(protocol_path),
        configuration=configuration,
        snapshot_epochs=frozenset(args.snapshot_epoch),
    )
    best = next(item["validation"] for item in history if item["epoch"] == best_epoch)
    checks = {
        "ap_higher": best["delta"]["average_precision"] > 0,
        "pixel_iou_higher": best["delta"]["pixel_iou"] > 0,
        "recall_at_fpr_0_0713_higher": best["delta"]["recall_at_fpr_0_0713"] > 0,
        "no_material_sensor_regression": all(
            value["eligible_for_promotion"]
            and value["delta"]["average_precision"] >= -0.01
            and value["delta"]["pixel_iou"] >= -0.01
            for value in best["sensor_strata"].values()
        ),
    }
    decision = (
        "Advance source-aligned residual to fold-1 confirmation."
        if not args.smoke and all(checks.values())
        else ("Smoke only; no promotion." if args.smoke else "Reject source-aligned residual on fold 0.")
    )
    report = {
        "schema_version": 1,
        "scope": "site-held development; fold 1 and paper test not loaded",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": {"held_out_fold": args.fold, "seed": args.seed, "smoke": args.smoke},
        "cohort": {
            "fit_rows": len(fit_records),
            "validation_rows": len(validation_records),
            "fit_sites": len({record["group_id"] for record in fit_records}),
            "validation_sites": len({record["group_id"] for record in validation_records}),
        },
        "configuration": configuration,
        "training": {
            "epochs_requested": args.epochs,
            "samples_per_epoch": args.samples_per_epoch,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "patience": args.patience,
            "best_epoch": best_epoch,
            "history": history,
        },
        "best_validation": best,
        "promotion_checks": checks,
        "decision": decision,
        "artifact": {
            "path": artifact.relative_to(root).as_posix(),
            "bytes": artifact.stat().st_size,
            "sha256": sha256(artifact),
            "tracked": False,
        },
        "provenance": {
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
            "script_sha256": sha256(Path(__file__).resolve()),
            "development_manifest_sha256": manifest_hash,
            "protocol_sha256": sha256(protocol_path),
            "parent_artifact_sha256": args.parent_sha256,
            "released_checkpoint_sha256": sha256(released_checkpoint),
            "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
            "torch": torch.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
            "rasterio": rasterio.__version__,
        },
    }
    write_json((root / args.output_json).resolve(), report)
    write_markdown((root / args.output_markdown).resolve(), report)
    print(json.dumps({"ok": True, "best_epoch": best_epoch, "checks": checks, "decision": decision}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
