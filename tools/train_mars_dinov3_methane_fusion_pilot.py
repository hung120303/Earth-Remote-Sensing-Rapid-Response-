#!/usr/bin/env python3
"""Cross-fit an identity-safe DINOv3/methane spatial fusion pilot."""

from __future__ import annotations

import argparse
import hashlib
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
from torch.utils.data import DataLoader, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "EarthRemoteSensingRapidResponse"
for path in (MODEL_ROOT, ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from acquire_mars_metadata import sha256  # noqa: E402
from analyze_mars_mask_routing import paired_group_bootstrap as pixel_bootstrap  # noqa: E402
from analyze_mars_mask_thresholds import component_mask_at  # noqa: E402
from mars_dinov3_methane_fusion import (  # noqa: E402
    COUNTERFACTUAL_CHANNELS,
    DinoMethaneFusionAdapter,
    PATCH_GRID_SIZE,
    SCENE_PROTECTION_GATE,
)
from mars_paper_model import SENSOR_NAMES, released_state  # noqa: E402
from mars_s2l_adapter import label_state  # noqa: E402
from train_mars_dense_prithvi_teacher_pilot import (  # noqa: E402
    fusion_loss,
    pixel_counts,
    pixel_summary,
)
from train_mars_oof_scene_ensemble_v2 import ap_group_bootstrap  # noqa: E402
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


DEFAULT_PROTOCOL = Path("configs/mars_dinov3_methane_fusion_pilot_protocol.json")
MINIMUM_CONNECTED_PIXELS = 100
MASK_THRESHOLDS = (0.8, 0.7)
MASK_SCENE_GATE = 0.75


def load_counterfactual_contract(
    images_path: Path,
    metadata_path: Path,
    *,
    expected_images_sha256: str,
) -> tuple[np.ndarray, dict[str, int], dict[str, Any]]:
    images = np.load(images_path, mmap_mode="r", allow_pickle=False)
    with np.load(metadata_path, allow_pickle=False) as values:
        metadata = {name: values[name] for name in values.files}
    expected_shape = (
        metadata["sample_ids"].size,
        COUNTERFACTUAL_CHANNELS,
        64,
        64,
    )
    if images.shape != expected_shape or images.dtype != np.float16:
        raise ValueError("Counterfactual cache geometry differs from the frozen schema")
    if str(metadata["images_sha256"].item()) != expected_images_sha256:
        raise ValueError("Counterfactual cache differs from its metadata receipt")
    sample_ids = metadata["sample_ids"].astype(str)
    if np.unique(sample_ids).size != sample_ids.size:
        raise ValueError("Counterfactual metadata contains duplicate sample identifiers")
    identity = {
        "rows": int(sample_ids.size),
        "shape": list(images.shape),
        "dtype": str(images.dtype),
        "sample_id_sha256": hashlib.sha256("\n".join(sample_ids).encode()).hexdigest(),
        "channel_names": metadata["channel_names"].astype(str).tolist(),
        "fold_counts": {
            str(int(fold)): int(np.count_nonzero(metadata["folds"] == fold))
            for fold in np.unique(metadata["folds"])
        },
        "manifest_sha256": str(metadata["manifest_sha256"].item()),
        "fold_protocol_sha256": str(metadata["fold_protocol_sha256"].item()),
        "released_checkpoint_sha256": str(
            metadata["released_checkpoint_sha256"].item()
        ),
    }
    return images, {value: index for index, value in enumerate(sample_ids)}, identity


def load_base_score_contract(
    records: list[dict[str, Any]],
    group_to_fold: dict[str, int],
    scores_path: Path,
) -> tuple[dict[str, float], dict[str, Any]]:
    inner = [
        row
        for row in records
        if group_to_fold[str(row["group_id"])] >= 2
    ]
    with np.load(scores_path, allow_pickle=False) as values:
        labels = values["inner_labels"]
        sensors = values["inner_sensors"]
        groups = values["inner_groups"].astype(str)
        folds = values["inner_folds"]
        scores = values["inner_new"].astype(np.float64)
    expected_labels = np.asarray(
        [int(label_state(row) == "PLUME") for row in inner], dtype=np.uint8
    )
    expected_sensors = np.asarray(
        [SENSOR_NAMES.index(str(row["sensor_family"])) for row in inner], dtype=np.uint8
    )
    expected_groups = np.asarray([str(row["group_id"]) for row in inner])
    expected_folds = np.asarray(
        [group_to_fold[str(row["group_id"])] for row in inner], dtype=np.uint8
    )
    if not (
        np.array_equal(labels, expected_labels)
        and np.array_equal(sensors, expected_sensors)
        and np.array_equal(groups, expected_groups)
        and np.array_equal(folds, expected_folds)
    ):
        raise ValueError("Current cross-fitted scene scores do not align to the manifest")
    if not np.isfinite(scores).all() or np.any((scores < 0.0) | (scores > 1.0)):
        raise ValueError("Current cross-fitted scene scores are invalid")
    sample_ids = [str(row["sample_id"]) for row in inner]
    return dict(zip(sample_ids, scores.tolist())), {
        "rows": len(inner),
        "sample_id_sha256": hashlib.sha256("\n".join(sample_ids).encode()).hexdigest(),
        "fold_counts": dict(Counter(map(int, folds))),
    }


class DinoMethaneDataset(MarsPaperDataset):
    """Attach aligned counterfactual maps and apply one joint geometry transform."""

    def __init__(
        self,
        metadata_dir: Path,
        records: list[dict[str, Any]],
        *,
        images: np.ndarray,
        row_by_id: dict[str, int],
        base_scores: dict[str, float],
        augment: bool,
        seed: int,
    ) -> None:
        super().__init__(metadata_dir, records, augment=False, seed=seed)
        self.images = images
        self.row_by_id = row_by_id
        self.base_scores = base_scores
        self.joint_augment = augment
        missing = [
            str(row["sample_id"])
            for row in records
            if str(row["sample_id"]) not in row_by_id
            or str(row["sample_id"]) not in base_scores
        ]
        if missing:
            raise ValueError(f"Spatial pilot caches lack {len(missing)} requested samples")

    @staticmethod
    def _transform(
        values: torch.Tensor, turns: int, horizontal: bool, vertical: bool
    ) -> torch.Tensor:
        if turns:
            values = torch.rot90(values, turns, dims=(-2, -1))
        if horizontal:
            values = torch.flip(values, dims=(-1,))
        if vertical:
            values = torch.flip(values, dims=(-2,))
        return values.contiguous()

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = super().__getitem__(index)
        sample_id = str(item["sample_id"])
        maps = torch.from_numpy(
            np.array(self.images[self.row_by_id[sample_id]], dtype=np.float32, copy=True)
        )
        if self.joint_augment:
            rng = self.rng()
            turns = int(rng.integers(0, 4))
            horizontal = bool(rng.integers(0, 2))
            vertical = bool(rng.integers(0, 2))
            wind = (
                float(item["inputs"][13, 0, 0]) * 8.0,
                float(item["inputs"][14, 0, 0]) * 8.0,
            )
            for _ in range(turns):
                wind = (-wind[1], wind[0])
            if horizontal:
                wind = (-wind[0], wind[1])
            if vertical:
                wind = (wind[0], -wind[1])
            for name in ("inputs", "observable", "clear", "mask"):
                item[name] = self._transform(item[name], turns, horizontal, vertical)
            maps = self._transform(maps, turns, horizontal, vertical)
            item["inputs"][13].fill_(wind[0] / 8.0)
            item["inputs"][14].fill_(wind[1] / 8.0)
        # The inherited loss helper calls this key; its value is the documented
        # 28-channel counterfactual methane tensor, not a Prithvi token cache.
        item["prithvi_tokens"] = maps
        item["base_scene_score"] = torch.tensor(
            self.base_scores[sample_id], dtype=torch.float32
        )
        return item


def partial_auc_pair_loss(
    scene_logits: torch.Tensor,
    presence: torch.Tensor,
    *,
    negative_fraction: float,
    margin: float,
) -> torch.Tensor:
    positive = scene_logits[presence > 0.5]
    negative = scene_logits[presence < 0.5]
    if not positive.numel() or not negative.numel():
        return scene_logits.sum() * 0.0
    hard_count = max(1, int(math.ceil(negative.numel() * negative_fraction)))
    hard_negative = torch.topk(negative, k=hard_count).values
    return F.softplus(margin - positive[:, None] + hard_negative[None, :]).mean()


def train_endpoint(
    model: DinoMethaneFusionAdapter,
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
        started = time.perf_counter()
        for batch in loader:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                output = model(
                    batch["inputs"],
                    batch["observable"],
                    batch["sensor_index"],
                    batch["prithvi_tokens"],
                    batch["base_scene_score"],
                )
                base_loss, parts = fusion_loss(output, batch, spec)
                partial_pair = partial_auc_pair_loss(
                    output["scene_logit"],
                    batch["presence"],
                    negative_fraction=float(spec["partial_auc_negative_fraction"]),
                    margin=float(spec["partial_auc_margin"]),
                )
                loss = base_loss + float(spec["partial_auc_weight"]) * partial_pair
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(parameters, float(spec["gradient_clip"]))
            scaler.step(optimizer)
            scaler.update()
            parts["loss"] = float(loss.detach())
            parts["partial_auc_pair"] = float(partial_pair.detach())
            batches += 1
            for key, value in parts.items():
                sums[key] = sums.get(key, 0.0) + value
        scheduler.step()
        row = {
            "epoch": epoch,
            "seconds": time.perf_counter() - started,
            **{key: value / batches for key, value in sums.items()},
        }
        history.append(row)
        print(json.dumps(row), flush=True)
    return history


@torch.no_grad()
def collect_predictions(
    model: DinoMethaneFusionAdapter,
    loader: DataLoader[dict[str, Any]],
    strengths: list[float],
    device: torch.device,
    fold: int,
    scene_evidence_weight: float,
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
    for batch in loader:
        local_ids = [str(value) for value in batch["sample_id"]]
        local_groups = [str(value) for value in batch["group_id"]]
        batch = move_batch(batch, device)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            output = model(
                batch["inputs"],
                batch["observable"],
                batch["sensor_index"],
                batch["prithvi_tokens"],
                batch["base_scene_score"],
            )
        baseline_probability = torch.sigmoid(output["baseline_logits"]).float()
        pixel_delta = output["correction_logits"].float()
        baseline_surrogate = model.scene_surrogate(
            output["baseline_logits"].float(), batch["observable"].float()
        )
        for strength in strengths:
            candidate_logits = (
                output["baseline_logits"].float() + float(strength) * pixel_delta
            )
            candidate_surrogate = model.scene_surrogate(
                candidate_logits, batch["observable"].float()
            )
            surrogate_delta = 2.0 * torch.tanh(
                candidate_surrogate - baseline_surrogate
            )
            score = model.protected_scene_score(
                batch["base_scene_score"],
                surrogate_delta,
                scene_evidence_weight,
            )
            rows["candidate_scores"][str(strength)].extend(
                float(value) for value in score.cpu().numpy()
            )
        for index in range(baseline_probability.shape[0]):
            sensor = int(batch["sensor_index"][index])
            threshold = MASK_THRESHOLDS[sensor]
            observable = batch["observable"][index, 0].cpu().numpy() > 0.5
            clear = batch["clear"][index, 0].cpu().numpy() > 0.5
            truth = (batch["mask"][index, 0].cpu().numpy() > 0.5) & observable
            base_score = float(batch["base_scene_score"][index])
            base_map = baseline_probability[index, 0].cpu().numpy()
            base_map[~clear] = 0.0
            base_prediction = component_mask_at(
                base_map, threshold, MINIMUM_CONNECTED_PIXELS
            )
            if base_score < MASK_SCENE_GATE:
                base_prediction[:] = False
            rows["base_pixels"].append(pixel_counts(base_prediction, truth, observable))
            for strength in strengths:
                probability = torch.sigmoid(
                    output["baseline_logits"][index, 0].float()
                    + float(strength) * pixel_delta[index, 0]
                ).cpu().numpy()
                probability[~clear] = 0.0
                prediction = component_mask_at(
                    probability, threshold, MINIMUM_CONNECTED_PIXELS
                )
                if base_score < MASK_SCENE_GATE:
                    prediction[:] = False
                rows["candidate_pixels"][str(strength)].append(
                    pixel_counts(prediction, truth, observable)
                )
        rows["labels"].extend(int(value) for value in batch["presence"].cpu().numpy())
        rows["sensors"].extend(int(value) for value in batch["sensor_index"].cpu().numpy())
        rows["groups"].extend(local_groups)
        rows["sample_ids"].extend(local_ids)
        rows["folds"].extend([fold] * len(local_ids))
        rows["base_scores"].extend(
            float(value) for value in batch["base_scene_score"].cpu().numpy()
        )
    return rows


def merge_predictions(parts: list[dict[str, Any]], strengths: list[float]) -> dict[str, Any]:
    result = {
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
    for part in parts:
        for key in ("labels", "sensors", "groups", "sample_ids", "folds", "base_scores", "base_pixels"):
            result[key].extend(part[key])
        for strength in strengths:
            result["candidate_scores"][str(strength)].extend(
                part["candidate_scores"][str(strength)]
            )
            result["candidate_pixels"][str(strength)].extend(
                part["candidate_pixels"][str(strength)]
            )
    return result


def summarize_predictions(
    raw: dict[str, Any],
    strengths: list[float],
    bootstrap: dict[str, Any],
    gates: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    y = np.asarray(raw["labels"], dtype=np.uint8)
    sensors = np.asarray(raw["sensors"], dtype=np.uint8)
    groups = np.asarray(raw["groups"])
    folds = np.asarray(raw["folds"], dtype=np.uint8)
    base = np.asarray(raw["base_scores"], dtype=np.float64)
    base_pixels = np.asarray(raw["base_pixels"], dtype=np.int64)
    base_metrics = metric_summary(y, base, sensors)
    base_pixel_summary = pixel_summary(base_pixels)
    candidates: list[dict[str, Any]] = []
    for strength in strengths:
        key = str(strength)
        scores = np.asarray(raw["candidate_scores"][key], dtype=np.float64)
        local_pixels = np.asarray(raw["candidate_pixels"][key], dtype=np.int64)
        metrics = metric_summary(y, scores, sensors)
        versus = comparison(metrics, base_metrics)
        ap_interval = ap_group_bootstrap(
            y,
            base,
            scores,
            groups,
            replicates=int(bootstrap["replicates"]),
            seed=int(bootstrap["seed"]),
        )
        pixel_metrics = pixel_summary(local_pixels)
        iou_delta = float(
            pixel_metrics["intersection_over_union"]
            - base_pixel_summary["intersection_over_union"]
        )
        pixel_interval = pixel_bootstrap(
            base_pixels,
            local_pixels,
            groups,
            replicates=int(bootstrap["replicates"]),
            seed=int(bootstrap["seed"]) + 1,
            confidence=float(bootstrap["confidence"]),
        )
        by_fold: dict[str, Any] = {}
        for fold in sorted(np.unique(folds)):
            selected = folds == fold
            fold_base = metric_summary(y[selected], base[selected], sensors[selected])
            fold_metrics = metric_summary(y[selected], scores[selected], sensors[selected])
            fold_base_pixels = pixel_summary(base_pixels[selected])
            fold_pixels = pixel_summary(local_pixels[selected])
            by_fold[str(int(fold))] = {
                "rows": int(np.count_nonzero(selected)),
                "scene": comparison(fold_metrics, fold_base),
                "current_pixel_rule": fold_base_pixels,
                "candidate_pixel_rule": fold_pixels,
                "pixel_iou_delta": float(
                    fold_pixels["intersection_over_union"]
                    - fold_base_pixels["intersection_over_union"]
                ),
            }
        sensor_delta = versus["delta"]["sensor_average_precision"]
        fold_ap = [value["scene"]["delta"]["average_precision"] for value in by_fold.values()]
        fold_iou = [value["pixel_iou_delta"] for value in by_fold.values()]
        confusion_keys = ("tp", "fp", "tn", "fn")
        same_operating_counts = all(metrics[name] == base_metrics[name] for name in confusion_keys)
        checks = {
            "average_precision_delta": versus["delta"]["average_precision"]
            >= float(gates["average_precision_delta_minimum"]),
            "matched_fpr_recall": versus["delta"]["recall_at_fpr_0_0713"]
            >= float(gates["matched_fpr_recall_delta_minimum"]),
            "no_worse_fpr": metrics["false_positive_rate"]
            <= base_metrics["false_positive_rate"] + 1e-15,
            "each_sensor_ap": min(sensor_delta.values())
            >= float(gates["each_sensor_ap_delta_minimum"]),
            "each_fold_ap": min(fold_ap) >= float(gates["each_fold_ap_delta_minimum"]),
            "paired_site_ap": ap_interval["lower"] > 0.0,
            "pixel_iou": iou_delta > 0.0,
            "each_fold_pixel_iou": min(fold_iou) > 0.0,
            "paired_site_pixel_iou": pixel_interval["lower"] > 0.0,
            "operating_counts_preserved": same_operating_counts,
        }
        passed = all(checks.values())
        candidates.append(
            {
                "strength": strength,
                "metrics": metrics,
                "versus_current": versus,
                "paired_site_ap_delta": ap_interval,
                "current_pixel_rule": base_pixel_summary,
                "candidate_pixel_rule": pixel_metrics,
                "pixel_iou_delta": iou_delta,
                "paired_site_pixel_iou_delta": pixel_interval,
                "by_fold": by_fold,
                "checks": checks,
                "passed": passed,
                "rank": [
                    int(passed),
                    min(fold_ap),
                    min(sensor_delta.values()),
                    ap_interval["lower"],
                    pixel_interval["lower"],
                    versus["delta"]["average_precision"],
                    versus["delta"]["recall_at_fpr_0_0713"],
                    -strength,
                ],
            }
        )
    identity = {
        "rows": len(y),
        "fold_counts": dict(Counter(map(int, folds))),
        "sample_id_sha256": hashlib.sha256("\n".join(raw["sample_ids"]).encode()).hexdigest(),
        "current_metrics": base_metrics,
        "current_pixel_rule": base_pixel_summary,
    }
    return candidates, identity


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    selected = report["selected"]
    scene = selected["versus_current"]["delta"]
    ap_ci = selected["paired_site_ap_delta"]
    pixel_ci = selected["paired_site_pixel_iou_delta"]
    lines = [
        "# DINOv3 methane-gated spatial fusion pilot",
        "",
        f"- Promotion gates pass: {report['all_promotion_gates_pass']}",
        f"- Selected residual strength: {selected['strength']}",
        f"- AP delta versus current cross-fitted ranker: {scene['average_precision']:+.6f}",
        f"- Matched-FPR recall delta: {scene['recall_at_fpr_0_0713']:+.6f}",
        f"- Paired-site AP interval: [{ap_ci['lower']:+.6f}, {ap_ci['upper']:+.6f}]",
        f"- Dense-mask IoU delta: {selected['pixel_iou_delta']:+.6f}",
        f"- Paired-site IoU interval: [{pixel_ci['lower']:+.6f}, {pixel_ci['upper']:+.6f}]",
        "",
        report["decision"],
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    os.replace(temporary, path)


def verify_protocol(protocol_path: Path, protocol: dict[str, Any], *, smoke: bool) -> dict[str, Path]:
    frozen = str(protocol["status"]).startswith("frozen")
    if not frozen and not smoke:
        raise ValueError("Outcome evaluation requires a frozen protocol")
    if frozen and sha256(Path(__file__).resolve()) != protocol["trainer"]["sha256"]:
        raise ValueError("Frozen DINOv3-fusion trainer hash mismatch")
    if frozen:
        for dependency in protocol["code_dependencies"]:
            path = (ROOT / dependency["path"]).resolve()
            if sha256(path) != dependency["sha256"]:
                raise ValueError(f"Frozen dependency mismatch: {dependency['path']}")
    paths: dict[str, Path] = {}
    for name, contract in protocol["inputs"].items():
        path = (ROOT / contract["path"]).resolve()
        if frozen and path.is_file() and sha256(path) != contract["sha256"]:
            raise ValueError(f"Frozen input mismatch: {name}")
        paths[name] = path
    if not paths["metadata_root"].is_dir():
        raise ValueError("MARS metadata root is unavailable")
    verify_acquisition_receipt(paths["acquisition_receipt"], sha256(paths["manifest"]))
    return paths


def make_loader(
    dataset: DinoMethaneDataset,
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL.as_posix())
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    protocol_path = (ROOT / args.protocol).resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    paths = verify_protocol(protocol_path, protocol, smoke=args.smoke)
    fold_protocol = json.loads(paths["fold_protocol"].read_text(encoding="utf-8"))
    group_to_fold = {
        str(row["group_id"]): int(row["fold"])
        for row in fold_protocol["assignments"]
    }
    all_records = list(iter_development_manifest(paths["manifest"]))
    selected_folds = set(map(int, protocol["folds"]))
    records = [
        row
        for row in all_records
        if group_to_fold[str(row["group_id"])] in selected_folds
    ]
    images, row_by_id, counterfactual_identity = load_counterfactual_contract(
        paths["counterfactual_images"],
        paths["counterfactual_metadata"],
        expected_images_sha256=protocol["inputs"]["counterfactual_images"]["sha256"],
    )
    base_scores, score_identity = load_base_score_contract(
        all_records, group_to_fold, paths["score_cache"]
    )
    spec = protocol["training"]
    batch_size = int(spec["batch_size"])
    workers = int(spec["loader_workers"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("DINOv3 methane fusion requires CUDA")
    torch.cuda.reset_peak_memory_stats()

    if args.smoke:
        seed_everything(int(spec["seed"]))
        fit = smoke_subset(
            [row for row in records if group_to_fold[str(row["group_id"])] == 3], 2
        )
        dataset = DinoMethaneDataset(
            paths["metadata_root"],
            fit,
            images=images,
            row_by_id=row_by_id,
            base_scores=base_scores,
            augment=True,
            seed=int(spec["seed"]),
        )
        weights, request_mass = balanced_request_weights(fit)
        sampler = WeightedRandomSampler(
            weights,
            num_samples=max(batch_size * 2, len(fit)),
            replacement=True,
            generator=torch.Generator().manual_seed(int(spec["seed"])),
        )
        loader = make_loader(
            dataset, batch_size=batch_size, workers=0, sampler=sampler
        )
        model = DinoMethaneFusionAdapter(paths["dino_checkpoint"]).to(device)
        model.load_released_checkpoint(released_state(paths["released_checkpoint"]))
        with torch.no_grad():
            first = move_batch(next(iter(loader)), device)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                initial = model(
                    first["inputs"],
                    first["observable"],
                    first["sensor_index"],
                    first["prithvi_tokens"],
                    first["base_scene_score"],
                )
            identity_pixel = float(initial["correction_logits"].abs().max())
            identity_scene = float(initial["scene_delta_logit"].abs().max())
            identity_score = float(
                (initial["scene_score"] - first["base_scene_score"]).abs().max()
            )
        if identity_pixel != 0.0 or identity_scene != 0.0 or identity_score != 0.0:
            raise ValueError("DINOv3 fusion initialization is not exact identity")
        started = time.perf_counter()
        history = train_endpoint(model, loader, spec, device, 1)
        elapsed = time.perf_counter() - started
        finite = all(torch.isfinite(value).all() for value in model.trainable_state().values())
        print(
            json.dumps(
                {
                    "ok": bool(finite),
                    "identity_pixel_max_abs": identity_pixel,
                    "identity_scene_max_abs": identity_scene,
                    "identity_score_max_abs": identity_score,
                    "elapsed_seconds": elapsed,
                    "peak_cuda_bytes": int(torch.cuda.max_memory_allocated()),
                    "trainable_parameters": model.trainable_parameter_count(),
                    "request_mass": request_mass,
                    "history": history,
                },
                indent=2,
            )
        )
        return 0 if finite else 1

    strengths = [float(value) for value in protocol["search"]["strengths"]]
    raw_parts: list[dict[str, Any]] = []
    endpoint_states: dict[str, dict[str, torch.Tensor]] = {}
    endpoints: list[dict[str, Any]] = []
    trainable_parameters: int | None = None
    for held_fold in sorted(selected_folds):
        fit_records = [
            row
            for row in records
            if group_to_fold[str(row["group_id"])] != held_fold
        ]
        held_records = [
            row
            for row in records
            if group_to_fold[str(row["group_id"])] == held_fold
        ]
        weights, request_mass = balanced_request_weights(fit_records)
        endpoint_seed = int(spec["seed"]) + held_fold
        seed_everything(endpoint_seed)
        endpoint_spec = dict(spec)
        endpoint_spec["seed"] = endpoint_seed
        train_dataset = DinoMethaneDataset(
            paths["metadata_root"],
            fit_records,
            images=images,
            row_by_id=row_by_id,
            base_scores=base_scores,
            augment=True,
            seed=endpoint_seed,
        )
        held_dataset = DinoMethaneDataset(
            paths["metadata_root"],
            held_records,
            images=images,
            row_by_id=row_by_id,
            base_scores=base_scores,
            augment=False,
            seed=endpoint_seed,
        )
        sampler = WeightedRandomSampler(
            weights,
            num_samples=int(spec["samples_per_epoch"]),
            replacement=True,
            generator=torch.Generator().manual_seed(endpoint_seed),
        )
        train_loader = make_loader(
            train_dataset, batch_size=batch_size, workers=workers, sampler=sampler
        )
        held_loader = make_loader(
            held_dataset, batch_size=batch_size, workers=workers
        )
        model = DinoMethaneFusionAdapter(paths["dino_checkpoint"]).to(device)
        model.load_released_checkpoint(released_state(paths["released_checkpoint"]))
        local_parameter_count = model.trainable_parameter_count()
        if trainable_parameters is None:
            trainable_parameters = local_parameter_count
        elif trainable_parameters != local_parameter_count:
            raise RuntimeError("Cross-fit endpoints have different trainable parameter counts")
        with torch.no_grad():
            first = move_batch(next(iter(held_loader)), device)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                initial = model(
                    first["inputs"],
                    first["observable"],
                    first["sensor_index"],
                    first["prithvi_tokens"],
                    first["base_scene_score"],
                )
            identity = {
                "pixel_max_abs": float(initial["correction_logits"].abs().max()),
                "scene_delta_max_abs": float(initial["scene_delta_logit"].abs().max()),
                "scene_score_max_abs": float(
                    (initial["scene_score"] - first["base_scene_score"]).abs().max()
                ),
            }
        if any(value != 0.0 for value in identity.values()):
            raise ValueError(f"Endpoint {held_fold} is not exact identity: {identity}")
        history = train_endpoint(
            model, train_loader, endpoint_spec, device, int(spec["epochs"])
        )
        raw_parts.append(
            collect_predictions(
                model,
                held_loader,
                strengths,
                device,
                held_fold,
                float(protocol["search"]["scene_evidence_weight"]),
            )
        )
        endpoint_states[str(held_fold)] = model.trainable_state()
        endpoints.append(
            {
                "held_fold": held_fold,
                "fit_fold": next(iter(selected_folds - {held_fold})),
                "fit_rows": len(fit_records),
                "held_rows": len(held_records),
                "request_mass": request_mass,
                "identity": identity,
                "history": history,
            }
        )
        del model, train_loader, held_loader
        torch.cuda.empty_cache()

    candidates, evaluation_identity = summarize_predictions(
        merge_predictions(raw_parts, strengths),
        strengths,
        protocol["bootstrap"],
        protocol["gates"],
    )
    selected = max(candidates, key=lambda row: tuple(row["rank"]))
    passed = bool(selected["passed"])
    artifact = None
    if passed:
        artifact_path = (ROOT / protocol["outputs"]["artifact"]).resolve()
        if artifact_path.exists():
            raise FileExistsError(f"Refusing to overwrite artifact: {artifact_path}")
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "schema_version": 1,
                "endpoint_states": endpoint_states,
                "selected_strength": selected["strength"],
                "protocol_sha256": sha256(protocol_path),
                "counterfactual_identity": counterfactual_identity,
                "score_identity": score_identity,
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
        "scope": "two-fold DINOv3 semantic-change and counterfactual-methane spatial fusion pilot",
        "all_promotion_gates_pass": passed,
        "decision": (
            "Authorize a separately frozen multi-seed confirmation."
            if passed
            else "Reject this fixed DINOv3 fusion pilot without external scoring."
        ),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "provenance": {
            "git_commit": commit,
            "protocol_sha256": sha256(protocol_path),
            "trainer_sha256": sha256(Path(__file__).resolve()),
            "device": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "peak_cuda_bytes": int(torch.cuda.max_memory_allocated()),
        },
        "folds": sorted(selected_folds),
        "endpoints": endpoints,
        "counterfactual_identity": counterfactual_identity,
        "score_identity": score_identity,
        "trainable_parameters": trainable_parameters,
        "evaluation_identity": evaluation_identity,
        "candidates": candidates,
        "selected": selected,
        "artifact": artifact,
        "external_inputs_accessed": False,
    }
    output_json = (ROOT / protocol["outputs"]["json"]).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_json.with_suffix(output_json.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output_json)
    write_markdown((ROOT / protocol["outputs"]["markdown"]).resolve(), report)
    print(
        json.dumps(
            {
                "ok": passed,
                "strength": selected["strength"],
                "ap_delta": selected["versus_current"]["delta"]["average_precision"],
                "ap_lower": selected["paired_site_ap_delta"]["lower"],
                "recall_delta": selected["versus_current"]["delta"]["recall_at_fpr_0_0713"],
                "iou_delta": selected["pixel_iou_delta"],
                "iou_lower": selected["paired_site_pixel_iou_delta"]["lower"],
            },
            indent=2,
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
