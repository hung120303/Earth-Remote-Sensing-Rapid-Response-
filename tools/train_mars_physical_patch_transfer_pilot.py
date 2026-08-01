#!/usr/bin/env python3
"""Cross-fit the 640 m source-invariant plume detector on authorized data."""

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
from mars_paper_model import ReleasedMarsUNet, released_state  # noqa: E402
from mars_physical_patch_transfer import (  # noqa: E402
    MARS_TILE_PIXELS,
    PATCH_PIXELS,
    PhysicalPatchTransferDetector,
    mars_tile_to_canonical,
    physical_tile_starts,
)
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
from train_mars_physics_guided_teacher_balanced_pilot import (  # noqa: E402
    balanced_request_weights,
)
from train_mars_physics_guided_teacher_pilot import seed_everything  # noqa: E402
from train_methanes2cm_v5 import (  # noqa: E402
    PackedMethaneS2CMDataset,
    segmentation_first_loss,
)
from train_mars_dense_prithvi_teacher_pilot import pixel_counts  # noqa: E402


DEFAULT_PROTOCOL = Path("configs/mars_physical_patch_transfer_pilot_protocol.json")


def iter_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def group_label_balanced_weights(records: list[dict[str, Any]]) -> torch.Tensor:
    """Equal label mass, then equal physical-group mass within each label."""

    labels = [int(row["label"]) for row in records]
    if set(labels) != {0, 1}:
        raise ValueError("External auxiliary cohort must contain both labels")
    group_counts = Counter((int(row["label"]), str(row["group_id"])) for row in records)
    groups_by_label = Counter()
    for label, group in group_counts:
        del group
        groups_by_label[label] += 1
    weights = torch.tensor(
        [
            0.5
            / groups_by_label[int(row["label"])]
            / group_counts[(int(row["label"]), str(row["group_id"]))]
            for row in records
        ],
        dtype=torch.double,
    )
    return weights / weights.sum()


class MarsPhysicalPatchDataset(Dataset[dict[str, Any]]):
    """Sample a 640 m MARS crop and convert it to the shared 20 m grid."""

    def __init__(
        self,
        dataset: MarsPaperDataset,
        *,
        seed: int,
        plume_center_probability: float,
    ) -> None:
        self.dataset = dataset
        self.seed = int(seed)
        self.plume_center_probability = float(plume_center_probability)
        self._rng: np.random.Generator | None = None

    def __len__(self) -> int:
        return len(self.dataset)

    def rng(self) -> np.random.Generator:
        if self._rng is None:
            worker = get_worker_info()
            worker_seed = self.seed if worker is None else int(worker.seed) + self.seed
            self._rng = np.random.default_rng(worker_seed % (2**63 - 1))
        return self._rng

    def _origin(self, mask: torch.Tensor) -> tuple[int, int]:
        height, width = mask.shape[-2:]
        maximum_y = height - MARS_TILE_PIXELS
        maximum_x = width - MARS_TILE_PIXELS
        if maximum_y < 0 or maximum_x < 0:
            raise ValueError("MARS scene is smaller than the physical crop")
        rng = self.rng()
        positive = torch.nonzero(mask[0] > 0.5, as_tuple=False).cpu().numpy()
        if len(positive) and rng.random() < self.plume_center_probability:
            y, x = positive[int(rng.integers(0, len(positive)))]
            jitter_y = int(rng.integers(-12, 13))
            jitter_x = int(rng.integers(-12, 13))
            origin_y = int(np.clip(int(y) - MARS_TILE_PIXELS // 2 + jitter_y, 0, maximum_y))
            origin_x = int(np.clip(int(x) - MARS_TILE_PIXELS // 2 + jitter_x, 0, maximum_x))
            return origin_y, origin_x
        return (
            int(rng.integers(0, maximum_y + 1)),
            int(rng.integers(0, maximum_x + 1)),
        )

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.dataset[index]
        y, x = self._origin(item["mask"])
        region = np.s_[..., y : y + MARS_TILE_PIXELS, x : x + MARS_TILE_PIXELS]
        values = item["inputs"][region][None]
        observable = item["observable"][region][None]
        canonical, auxiliary, pooled_observable = mars_tile_to_canonical(values, observable)
        mask = F.max_pool2d(item["mask"][region][None], kernel_size=2, stride=2)
        clear = (
            F.avg_pool2d(item["clear"][region][None], kernel_size=2, stride=2)
            >= 1.0 - 1e-6
        ).float()
        mask = (mask > 0.5).float() * pooled_observable
        presence = float(torch.any(mask > 0.5))
        return {
            "inputs": canonical[0],
            "auxiliary": auxiliary[0],
            "observable": pooled_observable[0],
            "clear": clear[0],
            "mask": mask[0],
            "presence": torch.tensor(presence, dtype=torch.float32),
            "sensor_index": item["sensor_index"],
            "source_index": torch.tensor(0, dtype=torch.long),
            "sample_id": f"mars:{item['sample_id']}",
            "group_id": f"mars:{item['group_id']}",
            "pixel_truth_available": item["pixel_truth_available"],
        }


class MethaneS2CMPhysicalDataset(Dataset[dict[str, Any]]):
    """Map native 20 m MethaneS2CM crops into the shared training schema."""

    def __init__(self, dataset: PackedMethaneS2CMDataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.dataset[index]
        observable = item["observable"]
        auxiliary = torch.zeros(3, PATCH_PIXELS, PATCH_PIXELS, dtype=torch.float32)
        auxiliary[0:2] = 0.5
        return {
            "inputs": item["inputs"],
            "auxiliary": auxiliary,
            "observable": observable,
            "clear": observable.clone(),
            "mask": item["mask"],
            "presence": item["presence"],
            "sensor_index": torch.tensor(0, dtype=torch.long),
            "source_index": torch.tensor(1, dtype=torch.long),
            "sample_id": f"methanes2cm:{item['sample_id']}",
            "group_id": f"methanes2cm:{item['group_id']}",
            "pixel_truth_available": torch.tensor(True, dtype=torch.bool),
        }


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


def combined_sampling_weights(
    mars_records: list[dict[str, Any]],
    external_records: list[dict[str, Any]],
    mars_mass: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    mars, mars_detail = balanced_request_weights(mars_records)
    external = group_label_balanced_weights(external_records)
    weights = torch.cat((mars * mars_mass, external * (1.0 - mars_mass)))
    weights /= weights.sum()
    mass = {
        "mars": float(weights[: len(mars_records)].sum()),
        "methanes2cm": float(weights[len(mars_records) :].sum()),
    }
    for key, value in mars_detail.items():
        mass[f"mars:{key}"] = float(value) * mass["mars"]
    return weights, mass


def train_phase(
    model: PhysicalPatchTransferDetector,
    loader: DataLoader[dict[str, Any]],
    optimizer: torch.optim.Optimizer,
    spec: dict[str, Any],
    device: torch.device,
    *,
    phase: str,
    epochs: int,
) -> list[dict[str, float]]:
    scaler = torch.amp.GradScaler("cuda")
    history: list[dict[str, float]] = []
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
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
                    batch["inputs"],
                    batch["auxiliary"],
                    batch["observable"],
                    batch["sensor_index"],
                    grl_strength=float(spec["gradient_reversal_strength"]),
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
                sentinel = batch["sensor_index"] == 0
                domain = output["scene_logit"].sum() * 0.0
                if torch.any(sentinel) and torch.unique(batch["source_index"][sentinel]).numel() == 2:
                    domain = F.binary_cross_entropy_with_logits(
                        output["domain_logit"][sentinel],
                        batch["source_index"][sentinel].float(),
                    )
                loss = (
                    base_loss
                    + float(spec["partial_auc_weight"]) * partial_auc
                    + float(spec["domain_weight"]) * domain
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(parameters, float(spec["gradient_clip"]))
            scaler.step(optimizer)
            scaler.update()
            row_parts = {
                **parts,
                "loss": float(loss.detach()),
                "partial_auc": float(partial_auc.detach()),
                "domain_bce": float(domain.detach()),
            }
            batches += 1
            for key, value in row_parts.items():
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
    for parameter in teacher.parameters():
        parameter.requires_grad = False
    return teacher


@torch.no_grad()
def infer_full_scene(
    model: PhysicalPatchTransferDetector,
    values: torch.Tensor,
    observable: torch.Tensor,
    sensor_index: torch.Tensor,
    *,
    tile_batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tile Bx16x200x200 scenes, returning Bx1x200x200 evidence and B logits."""

    starts_y = physical_tile_starts(values.shape[-2])
    starts_x = physical_tile_starts(values.shape[-1])
    locations = [(y, x) for y in starts_y for x in starts_x]
    tiles = torch.stack(
        [values[..., y : y + MARS_TILE_PIXELS, x : x + MARS_TILE_PIXELS] for y, x in locations],
        dim=1,
    )
    visible = torch.stack(
        [observable[..., y : y + MARS_TILE_PIXELS, x : x + MARS_TILE_PIXELS] for y, x in locations],
        dim=1,
    )
    batch, tile_count = tiles.shape[:2]
    tiles = tiles.flatten(0, 1)
    visible = visible.flatten(0, 1)
    repeated_sensor = sensor_index[:, None].expand(-1, tile_count).reshape(-1)
    mask_parts: list[torch.Tensor] = []
    scene_parts: list[torch.Tensor] = []
    for start in range(0, len(tiles), tile_batch_size):
        stop = min(start + tile_batch_size, len(tiles))
        canonical, auxiliary, pooled_visible = mars_tile_to_canonical(
            tiles[start:stop], visible[start:stop]
        )
        with torch.amp.autocast("cuda", dtype=torch.float16):
            output = model(
                canonical,
                auxiliary,
                pooled_visible,
                repeated_sensor[start:stop],
            )
        mask_parts.append(output["segmentation_logits"].float())
        scene_parts.append(output["scene_logit"].float())
    patch_logits = torch.cat(mask_parts).reshape(batch, tile_count, 1, PATCH_PIXELS, PATCH_PIXELS)
    tile_scene = torch.cat(scene_parts).reshape(batch, tile_count)
    accumulator = torch.zeros(
        batch, 1, values.shape[-2], values.shape[-1], device=values.device
    )
    counts = torch.zeros_like(accumulator)
    for tile_index, (y, x) in enumerate(locations):
        upsampled = F.interpolate(
            patch_logits[:, tile_index],
            size=(MARS_TILE_PIXELS, MARS_TILE_PIXELS),
            mode="bilinear",
            align_corners=False,
        )
        accumulator[..., y : y + MARS_TILE_PIXELS, x : x + MARS_TILE_PIXELS] += upsampled
        counts[..., y : y + MARS_TILE_PIXELS, x : x + MARS_TILE_PIXELS] += 1.0
    if torch.any(counts == 0):
        raise ValueError("Physical tiling left uncovered scene pixels")
    return accumulator / counts, model.aggregate_tile_scene_logits(tile_scene)


@torch.no_grad()
def collect_predictions(
    model: PhysicalPatchTransferDetector,
    teacher: ReleasedMarsUNet,
    loader: DataLoader[dict[str, Any]],
    base_scores: dict[str, float],
    strengths: list[float],
    device: torch.device,
    fold: int,
    tile_batch_size: int,
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
        local_logits, scene_logit = infer_full_scene(
            model,
            batch["inputs"],
            batch["observable"],
            batch["sensor_index"],
            tile_batch_size=tile_batch_size,
        )
        baseline_probability = torch.sigmoid(baseline_logits.float())
        bounded_local = 2.0 * torch.tanh(local_logits / 2.0)
        for strength in strengths:
            scores = model.fuse_scene_score(
                local_base,
                scene_logit,
                batch["sensor_index"],
                strength,
                sentinel_only=True,
            )
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
                probability = torch.sigmoid(
                    baseline_logits[index, 0].float()
                    + float(strength) * bounded_local[index, 0]
                ).cpu().numpy()
                probability[~clear] = 0.0
                prediction = component_mask_at(probability, threshold, MINIMUM_CONNECTED_PIXELS)
                if base_score < MASK_SCENE_GATE:
                    prediction[:] = False
                rows["candidate_pixels"][str(strength)].append(
                    pixel_counts(prediction, truth, observable)
                )
        rows["labels"].extend(int(value) for value in batch["presence"].cpu().numpy())
        rows["sensors"].extend(int(value) for value in batch["sensor_index"].cpu().numpy())
        rows["groups"].extend(group_ids)
        rows["sample_ids"].extend(sample_ids)
        rows["folds"].extend([fold] * len(sample_ids))
        rows["base_scores"].extend(float(value) for value in local_base.cpu().numpy())
        if batch_index % 64 == 0:
            print(
                json.dumps(
                    {"progress": "physical_inference_batch", "fold": fold, "batch": batch_index}
                ),
                flush=True,
            )
    return rows


def verify_protocol(protocol: dict[str, Any], *, smoke: bool) -> dict[str, Path]:
    frozen = str(protocol["status"]).startswith("frozen")
    if not frozen and not smoke:
        raise ValueError("Outcome evaluation requires a frozen protocol")
    if frozen and sha256(Path(__file__).resolve()) != protocol["trainer"]["sha256"]:
        raise ValueError("Frozen physical-patch trainer hash mismatch")
    if frozen:
        for dependency in protocol["code_dependencies"]:
            path = (ROOT / dependency["path"]).resolve()
            if sha256(path) != dependency["sha256"]:
                raise ValueError(f"Frozen dependency mismatch: {dependency['path']}")
    paths: dict[str, Path] = {}
    for name, contract in protocol["inputs"].items():
        path = (ROOT / contract["path"]).resolve()
        if not path.exists():
            raise ValueError(f"Required input is unavailable: {name}")
        if frozen and path.is_file() and sha256(path) != contract["sha256"]:
            raise ValueError(f"Frozen input mismatch: {name}")
        paths[name] = path
    verify_acquisition_receipt(paths["acquisition_receipt"], sha256(paths["manifest"]))
    return paths


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    selected = report["selected"]
    delta = selected["versus_current"]["delta"]
    ap = selected["paired_site_ap_delta"]
    iou = selected["paired_site_pixel_iou_delta"]
    lines = [
        "# Physical-scale cross-domain plume detector pilot",
        "",
        f"- Promotion gates pass: **{report['all_promotion_gates_pass']}**",
        f"- Selected fusion strength: **{selected['strength']}**",
        f"- AP delta versus current spatial-Prithvi score: **{delta['average_precision']:+.6f}**",
        f"- Matched-FPR recall delta: **{delta['recall_at_fpr_0_0713']:+.6f}**",
        f"- Paired-site AP interval: **[{ap['lower']:+.6f}, {ap['upper']:+.6f}]**",
        f"- Dense-mask IoU delta: **{selected['pixel_iou_delta']:+.6f}**",
        f"- Paired-site IoU interval: **[{iou['lower']:+.6f}, {iou['upper']:+.6f}]**",
        "",
        report["decision"],
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    os.replace(temporary, path)


def _artifact_state(model: PhysicalPatchTransferDetector) -> dict[str, torch.Tensor]:
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
    external_records = iter_jsonl(paths["methanes2cm_auxiliary_manifest"])
    if any(
        str(row.get("research_role")) != "auxiliary_training"
        or str(row.get("source_research_role")) != "internal_fitting"
        for row in external_records
    ):
        raise ValueError(
            "MethaneS2CM auxiliary manifest must contain only frozen source-fitting rows"
        )
    base_scores, score_identity = load_base_score_contract(
        all_records, group_to_fold, paths["score_cache"]
    )
    spec = protocol["training"]
    strengths = [float(value) for value in protocol["search"]["strengths"]]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Physical-patch transfer pilot requires CUDA")
    torch.cuda.reset_peak_memory_stats()

    if args.smoke:
        seed = int(spec["seed"])
        seed_everything(seed)
        mars_smoke = smoke_subset(mars_records, 2)
        external_smoke: list[dict[str, Any]] = []
        for label in (0, 1):
            external_smoke.extend(
                [row for row in external_records if int(row["label"]) == label][:4]
            )
        mars_dataset = MarsPhysicalPatchDataset(
            MarsPaperDataset(paths["metadata_root"], mars_smoke, augment=True, seed=seed),
            seed=seed,
            plume_center_probability=float(spec["plume_center_probability"]),
        )
        external_dataset = MethaneS2CMPhysicalDataset(
            PackedMethaneS2CMDataset(
                paths["methanes2cm_packed"], external_smoke, augment=True, seed=seed + 1
            )
        )
        weights, request_mass = combined_sampling_weights(
            mars_smoke,
            external_smoke,
            float(protocol["sampling"]["mars_mass"]),
        )
        sampler = WeightedRandomSampler(
            weights,
            num_samples=int(spec["batch_size"]) * 2,
            replacement=True,
            generator=torch.Generator().manual_seed(seed),
        )
        loader = make_loader(
            ConcatDataset((mars_dataset, external_dataset)),
            batch_size=int(spec["batch_size"]),
            workers=0,
            sampler=sampler,
        )
        model = PhysicalPatchTransferDetector(
            context_scene_weight=float(protocol["architecture"]["context_scene_weight"])
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(spec["learning_rate"]),
            weight_decay=float(spec["weight_decay"]),
        )
        history = train_phase(
            model, loader, optimizer, spec, device, phase="smoke_joint", epochs=1
        )
        full_loader = make_loader(
            MarsPaperDataset(paths["metadata_root"], mars_smoke[:2], augment=False, seed=seed),
            batch_size=2,
            workers=0,
        )
        full_batch = move_batch(next(iter(full_loader)), device)
        teacher = _load_released_teacher(paths["released_checkpoint"], device)
        model.eval()
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
            baseline_logits = teacher(full_batch["inputs"])
        local_logits, local_scene = infer_full_scene(
            model,
            full_batch["inputs"],
            full_batch["observable"],
            full_batch["sensor_index"],
            tile_batch_size=int(spec["tile_inference_batch_size"]),
        )
        inference_finite = bool(
            torch.isfinite(baseline_logits).all()
            and torch.isfinite(local_logits).all()
            and torch.isfinite(local_scene).all()
        )
        inference_contract = {
            "scenes": int(len(full_batch["inputs"])),
            "scene_shape": list(full_batch["inputs"].shape[-2:]),
            "baseline_logit_shape": list(baseline_logits.shape),
            "stitched_local_logit_shape": list(local_logits.shape),
            "local_scene_shape": list(local_scene.shape),
            "finite": inference_finite,
        }
        finite = all(
            math.isfinite(float(value))
            for value in history[-1].values()
            if isinstance(value, (float, int))
        ) and inference_finite
        result = {
            "ok": finite,
            "finite_optimization": finite,
            "request_mass": request_mass,
            "model": model.artifact_metadata(),
            "full_scene_inference": inference_contract,
            "peak_cuda_bytes": int(torch.cuda.max_memory_allocated()),
            "device": torch.cuda.get_device_name(device),
            "history": history,
        }
        print(json.dumps(result, sort_keys=True))
        return 0 if finite else 1

    endpoint_results: list[dict[str, Any]] = []
    prediction_parts: list[dict[str, Any]] = []
    endpoint_states: dict[str, Any] = {}
    teacher = _load_released_teacher(paths["released_checkpoint"], device)
    for held_fold in sorted(selected_folds):
        fit_folds = selected_folds - {held_fold}
        fit_records = [
            row for row in mars_records if group_to_fold[str(row["group_id"])] in fit_folds
        ]
        held_records = [
            row for row in mars_records if group_to_fold[str(row["group_id"])] == held_fold
        ]
        seed = int(spec["seed"]) + held_fold
        seed_everything(seed)
        model = PhysicalPatchTransferDetector(
            context_scene_weight=float(protocol["architecture"]["context_scene_weight"])
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(spec["learning_rate"]),
            weight_decay=float(spec["weight_decay"]),
        )

        external_pretrain = MethaneS2CMPhysicalDataset(
            PackedMethaneS2CMDataset(
                paths["methanes2cm_packed"], external_records, augment=True, seed=seed + 100
            )
        )
        external_weights = group_label_balanced_weights(external_records)
        external_sampler = WeightedRandomSampler(
            external_weights,
            num_samples=int(spec["external_samples_per_epoch"]),
            replacement=True,
            generator=torch.Generator().manual_seed(seed + 10),
        )
        external_loader = make_loader(
            external_pretrain,
            batch_size=int(spec["batch_size"]),
            workers=int(spec["loader_workers"]),
            sampler=external_sampler,
        )
        history = train_phase(
            model,
            external_loader,
            optimizer,
            spec,
            device,
            phase="external_pretrain",
            epochs=int(spec["external_pretrain_epochs"]),
        )

        mars_fit = MarsPhysicalPatchDataset(
            MarsPaperDataset(paths["metadata_root"], fit_records, augment=True, seed=seed),
            seed=seed,
            plume_center_probability=float(spec["plume_center_probability"]),
        )
        external_joint = MethaneS2CMPhysicalDataset(
            PackedMethaneS2CMDataset(
                paths["methanes2cm_packed"], external_records, augment=True, seed=seed + 200
            )
        )
        weights, request_mass = combined_sampling_weights(
            fit_records,
            external_records,
            float(protocol["sampling"]["mars_mass"]),
        )
        joint_sampler = WeightedRandomSampler(
            weights,
            num_samples=int(spec["joint_samples_per_epoch"]),
            replacement=True,
            generator=torch.Generator().manual_seed(seed + 20),
        )
        joint_loader = make_loader(
            ConcatDataset((mars_fit, external_joint)),
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
                phase="joint_finetune",
                epochs=int(spec["joint_epochs"]),
            )
        )
        held_loader = make_loader(
            MarsPaperDataset(paths["metadata_root"], held_records, augment=False, seed=seed),
            batch_size=int(spec["evaluation_batch_size"]),
            workers=int(spec["loader_workers"]),
        )
        predictions = collect_predictions(
            model,
            teacher,
            held_loader,
            base_scores,
            strengths,
            device,
            held_fold,
            int(spec["tile_inference_batch_size"]),
        )
        prediction_parts.append(predictions)
        endpoint_results.append(
            {
                "held_fold": held_fold,
                "fit_folds": sorted(fit_folds),
                "fit_rows": len(fit_records),
                "held_rows": len(held_records),
                "external_rows": len(external_records),
                "seed": seed,
                "request_mass": request_mass,
                "history": history,
            }
        )
        endpoint_states[str(held_fold)] = _artifact_state(model)

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
                "model": PhysicalPatchTransferDetector(
                    context_scene_weight=float(
                        protocol["architecture"]["context_scene_weight"]
                    )
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
        "endpoints": endpoint_results,
        "candidates": candidates,
        "selected": selected,
        "all_promotion_gates_pass": passed,
        "artifact": artifact,
        "peak_cuda_bytes": int(torch.cuda.max_memory_allocated()),
        "device": torch.cuda.get_device_name(device),
        "decision": (
            "Promote to a new-seed source-disjoint confirmation; keep fold 2, external development, and paper test closed."
            if passed
            else "Reject this pilot without opening fold 2, external development, or the paper test."
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
