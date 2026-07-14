#!/usr/bin/env python3
"""Train the baseline-preserving MARS paper successor on frozen site folds."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import scipy
import sklearn
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler, get_worker_info

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "EarthRemoteSensingRapidResponse"
if str(MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_ROOT))
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from mars_paper_model import (  # noqa: E402
    MarsPaperResidualModel,
    RELEASED_CHECKPOINT_SHA256,
    SENSOR_NAMES,
)
from mars_s2l_adapter import (  # noqa: E402
    compute_mbmp,
    iter_development_manifest,
    load_sample,
    safe_asset_path,
)

from acquire_mars_metadata import DEFAULT_OUTPUT, repo_root, sha256  # noqa: E402
from evaluate_released_marss2l import (  # noqa: E402
    MODEL_BASE,
    component_mask,
    connected_scene_score,
    scene_metrics,
)
from train_mars_v3 import rotate_wind  # noqa: E402
from train_mars_v4 import (  # noqa: E402
    choose_threshold_at_fpr,
    hard_negative_segmentation_loss,
)

DEFAULT_MANIFEST = DEFAULT_OUTPUT / "paper_v3_development_samples.jsonl"
DEFAULT_PROTOCOL = Path("configs/mars_paper_v3_group_folds.json")
DEFAULT_ACQUISITION_RECEIPT = Path(
    "reports/acquisition/mars_s2l_paper_v3_development_download.json"
)
DEFAULT_CHECKPOINT = MODEL_BASE / "MARSS2L_20250326/best_epoch"
DEFAULT_ARTIFACT = Path("EarthRemoteSensingRapidResponse/artifacts/mars_paper_residual.pt")
DEFAULT_JSON = Path("reports/experiments/mars_paper_residual_validation.json")
DEFAULT_MARKDOWN = Path("reports/experiments/MARS_PAPER_RESIDUAL_VALIDATION.md")
TARGET_FPRS = (0.05, 0.0713)
PIXEL_THRESHOLD = 0.5


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def complete_record(metadata_dir: Path, record: dict[str, Any]) -> bool:
    for asset in record["assets"]:
        path = safe_asset_path(metadata_dir, str(asset["path"]))
        try:
            size = path.stat().st_size
        except FileNotFoundError:
            return False
        if size != int(asset["size"]):
            return False
    return True


def smoke_subset(records: list[dict[str, Any]], limit_per_stratum: int = 16) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for record in records:
        key = f"{record['label_state']}|{record['sensor_family']}"
        if counts[key] >= limit_per_stratum:
            continue
        selected.append(record)
        counts[key] += 1
    if not selected or not {record["label_state"] for record in selected}.issuperset(
        {"PLUME", "NO_PLUME"}
    ):
        raise ValueError("Smoke subset requires available plume and no-plume scenes")
    return selected


def available_smoke_subset(
    metadata_dir: Path,
    records: list[dict[str, Any]],
    limit_per_stratum: int = 16,
    max_records_scanned: int = 4096,
) -> list[dict[str, Any]]:
    """Collect a tiny available subset without inventorying the full cohort."""
    selected: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for record in records[:max_records_scanned]:
        key = f"{record['label_state']}|{record['sensor_family']}"
        if counts[key] >= limit_per_stratum or not complete_record(metadata_dir, record):
            continue
        selected.append(record)
        counts[key] += 1
        if len(counts) == 4 and all(value >= limit_per_stratum for value in counts.values()):
            break
    return smoke_subset(selected, limit_per_stratum)


def verify_acquisition_receipt(path: Path, manifest_hash: str) -> None:
    receipt = json.loads(path.read_text(encoding="utf-8"))
    result = receipt.get("result", {})
    manifest_filter = result.get("manifest_filter", {})
    if not result.get("ok") or not result.get("all_selected_assets_verified", True):
        raise ValueError("Development acquisition receipt is not successful")
    if manifest_filter.get("sha256") != manifest_hash:
        raise ValueError("Development acquisition receipt covers a different manifest")


class MarsPaperDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        metadata_dir: Path,
        records: list[dict[str, Any]],
        *,
        augment: bool,
        seed: int,
    ) -> None:
        self.metadata_dir = metadata_dir
        self.records = records
        self.augment = augment
        self.seed = seed
        self._rng: np.random.Generator | None = None

    def __len__(self) -> int:
        return len(self.records)

    def rng(self) -> np.random.Generator:
        if self._rng is None:
            worker = get_worker_info()
            worker_seed = self.seed if worker is None else int(worker.seed)
            self._rng = np.random.default_rng(worker_seed)
        return self._rng

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        # The pinned public training split contains six isplume=1 rows from one
        # bad-retrieval site whose producer masks are empty. Upstream retains
        # their positive scene labels and zero pixel targets; match that
        # behavior explicitly rather than silently relabeling or dropping them.
        sample = load_sample(
            self.metadata_dir,
            record,
            require_enhancement=False,
            allow_empty_positive_mask=True,
        )
        spectral = sample.reflectance_pair.copy()
        cloud = (sample.cloud_classes > 0).astype(np.float32)
        clear = sample.clear_mask.astype(np.float32)
        observable = sample.observable_mask.astype(np.float32)
        mask = sample.plume_mask.astype(np.float32)
        wind = (float(record["wind_u"]), float(record["wind_v"]))
        mbmp = compute_mbmp(spectral[:6], spectral[6:])
        if self.augment:
            rng = self.rng()
            turns = int(rng.integers(0, 4))
            if turns:
                spectral = np.rot90(spectral, turns, axes=(1, 2)).copy()
                mbmp = np.rot90(mbmp, turns).copy()
                cloud = np.rot90(cloud, turns).copy()
                clear = np.rot90(clear, turns).copy()
                observable = np.rot90(observable, turns).copy()
                mask = np.rot90(mask, turns).copy()
                wind = rotate_wind(wind, turns)
            if bool(rng.integers(0, 2)):
                spectral = spectral[:, :, ::-1].copy()
                mbmp = mbmp[:, ::-1].copy()
                cloud = cloud[:, ::-1].copy()
                clear = clear[:, ::-1].copy()
                observable = observable[:, ::-1].copy()
                mask = mask[:, ::-1].copy()
                wind = (-wind[0], wind[1])
            if bool(rng.integers(0, 2)):
                spectral = spectral[:, ::-1, :].copy()
                mbmp = mbmp[::-1, :].copy()
                cloud = cloud[::-1, :].copy()
                clear = clear[::-1, :].copy()
                observable = observable[::-1, :].copy()
                mask = mask[::-1, :].copy()
                wind = (wind[0], -wind[1])
        height, width = mbmp.shape
        wind_channels = np.broadcast_to(
            np.asarray(wind, dtype=np.float32)[:, None, None] / 8.0,
            (2, height, width),
        ).copy()
        inputs = np.concatenate(
            [mbmp[None], spectral, wind_channels, cloud[None]], axis=0
        ).astype(np.float32)
        sensor_index = SENSOR_NAMES.index(sample.sensor_family)
        return {
            "inputs": torch.from_numpy(inputs),
            "observable": torch.from_numpy(observable[None]),
            "clear": torch.from_numpy(clear[None]),
            "mask": torch.from_numpy(mask[None]),
            "presence": torch.tensor(sample.presence, dtype=torch.float32),
            "sensor_index": torch.tensor(sensor_index, dtype=torch.long),
            "sample_id": sample.sample_id,
            "group_id": str(record["group_id"]),
        }


def sampling_weights(records: list[dict[str, Any]]) -> torch.Tensor:
    strata = [f"{record['label_state']}|{record['sensor_family']}" for record in records]
    counts = Counter(strata)
    return torch.tensor([1.0 / counts[value] for value in strata], dtype=torch.double)


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def successor_loss(
    output: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    scene_weight: float = 0.25,
    negative_upward_weight: float = 0.25,
    positive_downward_weight: float = 0.10,
    correction_l2_weight: float = 0.002,
) -> tuple[torch.Tensor, dict[str, float]]:
    segmentation, segmentation_parts = hard_negative_segmentation_loss(
        output["segmentation_logits"], batch["mask"], batch["observable"]
    )
    scene = F.binary_cross_entropy_with_logits(
        output["scene_logit"], batch["presence"]
    )
    negative_scene = (batch["presence"] < 0.5)[:, None, None, None]
    valid_negative = negative_scene & (batch["observable"] > 0.5)
    upward = F.relu(
        output["segmentation_logits"] - output["baseline_logits"].detach()
    )
    upward_penalty = (upward * valid_negative).sum() / valid_negative.sum().clamp_min(1)
    valid_positive = (batch["mask"] > 0.5) & (batch["observable"] > 0.5)
    downward = F.relu(
        output["baseline_logits"].detach() - output["segmentation_logits"]
    )
    downward_penalty = (downward * valid_positive).sum() / valid_positive.sum().clamp_min(
        1
    )
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
        "positive_bce": float(segmentation_parts["positive_bce"].detach()),
        "hard_negative_bce": float(
            segmentation_parts["hard_negative_bce"].detach()
        ),
        "positive_dice_loss": float(
            segmentation_parts["positive_dice_loss"].detach()
        ),
        "scene_bce": float(scene.detach()),
        "negative_upward_penalty": float(upward_penalty.detach()),
        "positive_downward_penalty": float(downward_penalty.detach()),
        "correction_l2": float(correction_penalty.detach()),
    }


def pixel_accumulator() -> dict[str, float]:
    return {"intersection": 0.0, "predicted": 0.0, "truth": 0.0}


def add_pixels(
    accumulator: dict[str, float],
    score: np.ndarray,
    truth: np.ndarray,
    observable: np.ndarray,
) -> None:
    prediction = component_mask(score)
    accumulator["intersection"] += int(np.count_nonzero(prediction & truth))
    accumulator["predicted"] += int(np.count_nonzero(prediction & observable))
    accumulator["truth"] += int(np.count_nonzero(truth))


def finish_pixels(accumulator: dict[str, float]) -> dict[str, float]:
    union = accumulator["predicted"] + accumulator["truth"] - accumulator["intersection"]
    total = accumulator["predicted"] + accumulator["truth"]
    return {
        "intersection_over_union": 0.0 if union == 0 else accumulator["intersection"] / union,
        "dice": 0.0 if total == 0 else 2.0 * accumulator["intersection"] / total,
        "intersection_pixels": int(accumulator["intersection"]),
        "predicted_positive_pixels": int(accumulator["predicted"]),
        "truth_positive_pixels": int(accumulator["truth"]),
    }


@torch.no_grad()
def validation_summary(
    model: MarsPaperResidualModel,
    loader: DataLoader[dict[str, Any]],
    device: torch.device,
    baseline_reference: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if baseline_reference is not None and getattr(model, "backbone_trainable", False):
        raise ValueError("Released-baseline caching requires a frozen backbone")
    model.eval()
    labels: list[int] = []
    groups: list[str] = []
    sample_ids: list[str] = []
    candidate_scores: list[float] = []
    baseline_scores: list[float] = []
    candidate_predictions: list[bool] = []
    baseline_predictions: list[bool] = []
    sensor_indices: list[int] = []
    candidate_pixels = pixel_accumulator()
    baseline_pixels = pixel_accumulator()
    candidate_pixels_by_sensor = {name: pixel_accumulator() for name in SENSOR_NAMES}
    baseline_pixels_by_sensor = {name: pixel_accumulator() for name in SENSOR_NAMES}
    for batch in loader:
        batch = move_batch(batch, device)
        with torch.amp.autocast(
            "cuda", dtype=torch.float16, enabled=device.type == "cuda"
        ):
            output = model(
                batch["inputs"], batch["observable"], batch["sensor_index"]
            )
        candidate_probability = torch.sigmoid(output["segmentation_logits"]).float()
        baseline_probability = (
            None
            if baseline_reference is not None
            else torch.sigmoid(output["baseline_logits"]).float()
        )
        clear = batch["clear"] > 0.5
        candidate_probability = candidate_probability.masked_fill(~clear, 0.0)
        if baseline_probability is not None:
            baseline_probability = baseline_probability.masked_fill(~clear, 0.0)
        for index in range(candidate_probability.shape[0]):
            candidate = candidate_probability[index, 0].cpu().numpy()
            baseline = (
                None
                if baseline_probability is None
                else baseline_probability[index, 0].cpu().numpy()
            )
            observable = batch["observable"][index, 0].cpu().numpy() > 0.5
            truth = (batch["mask"][index, 0].cpu().numpy() > 0.5) & observable
            candidate_component = component_mask(candidate)
            labels.append(int(batch["presence"][index].item()))
            groups.append(str(batch["group_id"][index]))
            sample_ids.append(str(batch["sample_id"][index]))
            candidate_scores.append(connected_scene_score(candidate))
            candidate_predictions.append(bool(np.any(candidate_component)))
            local_sensor_index = int(batch["sensor_index"][index].item())
            sensor_indices.append(local_sensor_index)
            add_pixels(candidate_pixels, candidate, truth, observable)
            sensor_name = SENSOR_NAMES[local_sensor_index]
            add_pixels(
                candidate_pixels_by_sensor[sensor_name], candidate, truth, observable
            )
            if baseline is not None:
                baseline_component = component_mask(baseline)
                baseline_scores.append(connected_scene_score(baseline))
                baseline_predictions.append(bool(np.any(baseline_component)))
                add_pixels(baseline_pixels, baseline, truth, observable)
                add_pixels(
                    baseline_pixels_by_sensor[sensor_name], baseline, truth, observable
                )
    y = np.asarray(labels, dtype=np.uint8)
    group_array = np.asarray(groups)
    sensor_array = np.asarray(sensor_indices, dtype=np.uint8)
    cohort_fingerprint = hashlib.sha256(
        json.dumps(
            list(zip(sample_ids, groups, labels, sensor_indices)),
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    if (
        baseline_reference is not None
        and baseline_reference.get("cohort_fingerprint") != cohort_fingerprint
    ):
        raise ValueError("Cached released baseline covers a different validation cohort")

    def summarize(
        scores: list[float],
        predictions: list[bool],
        pixels: dict[str, float],
        selection: np.ndarray | None = None,
    ) -> dict[str, Any]:
        score_array = np.asarray(scores, dtype=np.float32)
        prediction_array = np.asarray(predictions, dtype=bool)
        local_y = y if selection is None else y[selection]
        local_scores = score_array if selection is None else score_array[selection]
        local_predictions = (
            prediction_array if selection is None else prediction_array[selection]
        )
        result = scene_metrics(local_y, local_predictions, local_scores)
        result["operating_points"] = {
            str(target): choose_threshold_at_fpr(local_y, local_scores, target)
            for target in TARGET_FPRS
        }
        result["pixel_fixed_0_5"] = finish_pixels(pixels)
        return result

    candidate = summarize(candidate_scores, candidate_predictions, candidate_pixels)
    baseline = (
        summarize(baseline_scores, baseline_predictions, baseline_pixels)
        if baseline_reference is None
        else baseline_reference["released_baseline"]
    )
    sensor_strata: dict[str, Any] = {}
    for sensor_index, sensor_name in enumerate(SENSOR_NAMES):
        selection = sensor_array == sensor_index
        local_y = y[selection]
        if local_y.size == 0 or np.unique(local_y).size < 2:
            sensor_strata[sensor_name] = {
                "rows": int(local_y.size),
                "positive": int(np.count_nonzero(local_y == 1)),
                "eligible_for_promotion": False,
                "reason": "A sensor stratum needs both plume and no-plume scenes.",
            }
            continue
        local_candidate = summarize(
            candidate_scores,
            candidate_predictions,
            candidate_pixels_by_sensor[sensor_name],
            selection,
        )
        local_baseline = (
            summarize(
                baseline_scores,
                baseline_predictions,
                baseline_pixels_by_sensor[sensor_name],
                selection,
            )
            if baseline_reference is None
            else baseline_reference["sensor_strata"][sensor_name][
                "released_baseline"
            ]
        )
        sensor_strata[sensor_name] = {
            "rows": int(np.count_nonzero(selection)),
            "positive": int(np.count_nonzero(y[selection] == 1)),
            "eligible_for_promotion": True,
            "candidate": local_candidate,
            "released_baseline": local_baseline,
            "delta": {
                "average_precision": local_candidate["average_precision"]
                - local_baseline["average_precision"],
                "pixel_iou": local_candidate["pixel_fixed_0_5"][
                    "intersection_over_union"
                ]
                - local_baseline["pixel_fixed_0_5"]["intersection_over_union"],
            },
        }
    return {
        "rows": int(y.size),
        "positive": int(np.count_nonzero(y == 1)),
        "negative": int(np.count_nonzero(y == 0)),
        "sites": int(np.unique(group_array).size),
        "cohort_fingerprint": cohort_fingerprint,
        "candidate": candidate,
        "released_baseline": baseline,
        "sensor_strata": sensor_strata,
        "delta": {
            "average_precision": candidate["average_precision"] - baseline["average_precision"],
            "pixel_iou": candidate["pixel_fixed_0_5"]["intersection_over_union"]
            - baseline["pixel_fixed_0_5"]["intersection_over_union"],
            "recall_at_fpr_0_0713": candidate["operating_points"]["0.0713"]["recall"]
            - baseline["operating_points"]["0.0713"]["recall"],
        },
    }


def artifact_payload(
    model: MarsPaperResidualModel,
    *,
    fold: int,
    seed: int,
    epoch: int,
    protocol_hash: str,
    validation: dict[str, Any],
    loss_weights: dict[str, float],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "model": model.artifact_metadata(),
        "fold": fold,
        "seed": seed,
        "epoch": epoch,
        "protocol_sha256": protocol_hash,
        "released_checkpoint_sha256": RELEASED_CHECKPOINT_SHA256,
        "correction_state_dict": model.correction.state_dict(),
        "sensor_log_scale": model.sensor_log_scale.detach().cpu(),
        "sensor_bias": model.sensor_bias.detach().cpu(),
        "validation": validation,
        "loss_weights": loss_weights,
    }


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
    loss_weights: dict[str, float],
) -> tuple[list[dict[str, Any]], int]:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history: list[dict[str, Any]] = []
    best_rank = (-math.inf, -math.inf, -math.inf)
    best_epoch = -1
    stale = 0
    baseline_reference: dict[str, Any] | None = None
    artifact.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, epochs + 1):
        model.train()
        started = time.perf_counter()
        parts: list[dict[str, float]] = []
        for batch in train_loader:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                "cuda", dtype=torch.float16, enabled=device.type == "cuda"
            ):
                output = model(
                    batch["inputs"], batch["observable"], batch["sensor_index"]
                )
                loss, values = successor_loss(output, batch, **loss_weights)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(parameters, 5.0)
            scaler.step(optimizer)
            scaler.update()
            parts.append(values)
        scheduler.step()
        validation = validation_summary(
            model,
            validation_loader,
            device,
            baseline_reference=baseline_reference,
        )
        if baseline_reference is None:
            baseline_reference = validation
        primary_deltas = (
            float(validation["delta"]["average_precision"]),
            float(validation["delta"]["pixel_iou"]),
        )
        rank = (
            min(primary_deltas),
            sum(primary_deltas),
            float(validation["delta"]["recall_at_fpr_0_0713"]),
        )
        record = {
            "epoch": epoch,
            "seconds": round(time.perf_counter() - started, 3),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "loss": {
                key: float(np.mean([part[key] for part in parts])) for key in parts[0]
            },
            "validation": validation,
        }
        history.append(record)
        print(json.dumps({"epoch": epoch, "rank": rank, "seconds": record["seconds"]}))
        if rank > best_rank:
            best_rank = rank
            best_epoch = epoch
            stale = 0
            torch.save(
                artifact_payload(
                    model,
                    fold=fold,
                    seed=seed,
                    epoch=epoch,
                    protocol_hash=protocol_hash,
                    validation=validation,
                    loss_weights=loss_weights,
                ),
                artifact,
            )
        else:
            stale += 1
            if stale >= patience:
                break
    return history, best_epoch


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    best = report["best_validation"]
    candidate = best["candidate"]
    baseline = best["released_baseline"]
    delta = best["delta"]
    lines = [
        f"# MARS paper residual fold {report['experiment']['held_out_fold']} validation",
        "",
        f"- Scope: {'smoke only' if report['experiment']['smoke'] else 'frozen site-held development'}",
        f"- Fit / held-out scenes: {report['cohort']['fit_rows']:,} / {report['cohort']['validation_rows']:,}",
        f"- Best epoch: {report['training']['best_epoch']}",
        "",
        "| Model | AP | Recall at <=7.13% FPR | Pixel IoU at 0.5 |",
        "|---|---:|---:|---:|",
        f"| Released MARS-S2L | {baseline['average_precision']:.4f} | {baseline['operating_points']['0.0713']['recall']:.4f} | {baseline['pixel_fixed_0_5']['intersection_over_union']:.4f} |",
        f"| ERSRR residual | {candidate['average_precision']:.4f} | {candidate['operating_points']['0.0713']['recall']:.4f} | {candidate['pixel_fixed_0_5']['intersection_over_union']:.4f} |",
        "",
        f"Deltas (ERSRR minus released): AP {delta['average_precision']:+.4f}, recall {delta['recall_at_fpr_0_0713']:+.4f}, pixel IoU {delta['pixel_iou']:+.4f}.",
        "",
        report["decision"],
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-dir", default=DEFAULT_OUTPUT.as_posix())
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST.as_posix())
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL.as_posix())
    parser.add_argument(
        "--acquisition-receipt", default=DEFAULT_ACQUISITION_RECEIPT.as_posix()
    )
    parser.add_argument("--released-checkpoint", default=DEFAULT_CHECKPOINT.as_posix())
    parser.add_argument("--artifact", default=DEFAULT_ARTIFACT.as_posix())
    parser.add_argument("--output-json", default=DEFAULT_JSON.as_posix())
    parser.add_argument("--output-markdown", default=DEFAULT_MARKDOWN.as_posix())
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=606)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--samples-per-epoch", type=int, default=32768)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--scene-weight", type=float, default=0.25)
    parser.add_argument("--negative-upward-weight", type=float, default=0.25)
    parser.add_argument("--positive-downward-weight", type=float, default=0.10)
    parser.add_argument("--correction-l2-weight", type=float, default=0.002)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.epochs <= 0 or args.samples_per_epoch <= 0 or args.batch_size <= 0:
        parser.error("epochs, samples-per-epoch, and batch-size must be positive")
    loss_weights = {
        "scene_weight": args.scene_weight,
        "negative_upward_weight": args.negative_upward_weight,
        "positive_downward_weight": args.positive_downward_weight,
        "correction_l2_weight": args.correction_l2_weight,
    }
    if any(not math.isfinite(value) or value < 0 for value in loss_weights.values()):
        parser.error("loss weights must be non-negative")
    root = repo_root()
    seed_everything(args.seed)
    metadata_dir = (root / args.metadata_dir).resolve()
    manifest = (root / args.manifest).resolve()
    protocol_path = (root / args.protocol).resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if sha256(manifest) != protocol["development_manifest_sha256"]:
        raise ValueError("Development manifest differs from the frozen fold protocol")
    if not 0 <= args.fold < int(protocol["n_folds"]):
        parser.error("fold is outside the frozen protocol")
    group_to_fold = {
        str(item["group_id"]): int(item["fold"])
        for item in protocol["assignments"]
    }
    records = list(iter_development_manifest(manifest))
    fit_records = [
        record
        for record in records
        if group_to_fold[str(record["group_id"])] != args.fold
    ]
    validation_records = [
        record
        for record in records
        if group_to_fold[str(record["group_id"])] == args.fold
    ]
    if args.smoke:
        fit_records = available_smoke_subset(metadata_dir, fit_records)
        validation_records = available_smoke_subset(metadata_dir, validation_records)
        args.epochs = min(args.epochs, 1)
        args.samples_per_epoch = min(args.samples_per_epoch, 128)
        args.batch_size = min(args.batch_size, 4)
        args.workers = min(args.workers, 2)
        missing_fit_count = -1
        missing_validation_count = -1
    else:
        verify_acquisition_receipt(
            (root / args.acquisition_receipt).resolve(), sha256(manifest)
        )
        missing_fit_count = 0
        missing_validation_count = 0

    train_dataset = MarsPaperDataset(
        metadata_dir, fit_records, augment=True, seed=args.seed
    )
    validation_dataset = MarsPaperDataset(
        metadata_dir, validation_records, augment=False, seed=args.seed
    )
    sampler = WeightedRandomSampler(
        sampling_weights(fit_records),
        num_samples=args.samples_per_epoch,
        replacement=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": True,
        "persistent_workers": args.workers > 0,
    }
    train_loader = DataLoader(train_dataset, sampler=sampler, **loader_options)
    validation_loader = DataLoader(
        validation_dataset, shuffle=False, **loader_options
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MarsPaperResidualModel().to(device)
    released_checkpoint = (root / args.released_checkpoint).resolve()
    model.load_released_checkpoint(released_checkpoint)
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
        loss_weights=loss_weights,
    )
    if best_epoch < 1:
        raise RuntimeError("Training did not produce a checkpoint")
    best = next(item["validation"] for item in history if item["epoch"] == best_epoch)
    checks = {
        "ap_higher": best["delta"]["average_precision"] > 0,
        "pixel_iou_higher": best["delta"]["pixel_iou"] > 0,
        "recall_at_fpr_0_0713_higher": best["delta"]["recall_at_fpr_0_0713"] > 0,
        "no_material_sensor_regression": all(
            stratum["eligible_for_promotion"]
            and stratum["delta"]["average_precision"] >= -0.01
            and stratum["delta"]["pixel_iou"] >= -0.01
            for stratum in best["sensor_strata"].values()
        ),
    }
    decision = (
        "Advance this architecture from the primary fold to the independent confirmation fold."
        if not args.smoke and all(checks.values())
        else (
            "Smoke execution only; this result cannot promote an architecture."
            if args.smoke
            else "Do not advance: at least one predeclared primary-fold point gate failed."
        )
    )
    report = {
        "schema_version": 1,
        "scope": "site-held architecture development; sealed paper test not loaded",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": {
            "held_out_fold": args.fold,
            "seed": args.seed,
            "smoke": args.smoke,
            "loss_weights": loss_weights,
        },
        "cohort": {
            "fit_rows": len(fit_records),
            "validation_rows": len(validation_records),
            "fit_sites": len({record["group_id"] for record in fit_records}),
            "validation_sites": len(
                {record["group_id"] for record in validation_records}
            ),
            "missing_full_fit_at_start": missing_fit_count,
            "missing_full_validation_at_start": missing_validation_count,
        },
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
        "model": model.artifact_metadata(),
        "artifact": {
            "path": artifact.relative_to(root).as_posix(),
            "bytes": artifact.stat().st_size,
            "sha256": sha256(artifact),
            "tracked": False,
        },
        "provenance": {
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True
            ).strip(),
            "script": "tools/train_mars_paper_residual.py",
            "script_sha256": sha256(Path(__file__)),
            "protocol": protocol_path.relative_to(root).as_posix(),
            "protocol_sha256": sha256(protocol_path),
            "development_manifest_sha256": sha256(manifest),
            "released_checkpoint_sha256": sha256(released_checkpoint),
            "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
            "torch": torch.__version__,
            "numpy": np.__version__,
            "rasterio": rasterio.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
        },
    }
    output_json = (root / args.output_json).resolve()
    output_markdown = (root / args.output_markdown).resolve()
    write_json(output_json, report)
    write_markdown(output_markdown, report)
    print(json.dumps({"ok": True, "best_epoch": best_epoch, "checks": checks, "decision": decision}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
