#!/usr/bin/env python3
"""Cross-fit the NDMI-guided bi-temporal successor on authorized development data."""

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
from torch.utils.data import ConcatDataset, DataLoader, Dataset, WeightedRandomSampler


ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "EarthRemoteSensingRapidResponse"
for path in (MODEL_ROOT, ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from acquire_mars_metadata import sha256  # noqa: E402
from analyze_mars_mask_routing import paired_group_bootstrap as pixel_bootstrap  # noqa: E402
from analyze_mars_mask_thresholds import component_mask_at  # noqa: E402
from mars_ndmi_bitemporal_fusion import (  # noqa: E402
    NdmiBitemporalFusionAdapter,
)
from mars_paper_model import SENSOR_NAMES, released_state  # noqa: E402
from mars_s2l_adapter import label_state  # noqa: E402
from train_mars_dense_prithvi_teacher_pilot import (  # noqa: E402
    fusion_loss,
    pixel_counts,
    pixel_summary,
)
from train_mars_dinov3_methane_fusion_pilot import (  # noqa: E402
    MASK_SCENE_GATE,
    MASK_THRESHOLDS,
    MINIMUM_CONNECTED_PIXELS,
    load_base_score_contract,
    partial_auc_pair_loss,
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


DEFAULT_PROTOCOL = Path("configs/mars_ndmi_bitemporal_fusion_pilot_protocol.json")


def iter_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


class TaggedDataset(Dataset[dict[str, Any]]):
    """Attach a stable source index without changing the underlying sample."""

    def __init__(self, dataset: MarsPaperDataset, source_index: int) -> None:
        self.dataset = dataset
        self.source_index = source_index

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.dataset[index]
        item["source_index"] = torch.tensor(self.source_index, dtype=torch.long)
        return item


def group_balanced_weights(records: list[dict[str, Any]]) -> torch.Tensor:
    groups = [str(row["group_id"]) for row in records]
    counts = Counter(groups)
    weights = torch.tensor([1.0 / counts[group] for group in groups], dtype=torch.double)
    return weights / weights.sum()


def combined_sampling_weights(
    mars_records: list[dict[str, Any]],
    unep_records: list[dict[str, Any]],
    cloudsen_records: list[dict[str, Any]],
    source_mass: dict[str, float],
) -> tuple[torch.Tensor, dict[str, float]]:
    mars, mars_mass = balanced_request_weights(mars_records)
    unep = group_balanced_weights(unep_records)
    cloudsen = group_balanced_weights(cloudsen_records)
    weights = torch.cat(
        (
            mars * float(source_mass["mars"]),
            unep * float(source_mass["unep_positive"]),
            cloudsen * float(source_mass["cloudsen_negative"]),
        )
    )
    weights /= weights.sum()
    mass = {
        "mars": float(weights[: len(mars_records)].sum()),
        "unep_positive": float(
            weights[len(mars_records) : len(mars_records) + len(unep_records)].sum()
        ),
        "cloudsen_negative": float(weights[-len(cloudsen_records) :].sum()),
    }
    for key, value in mars_mass.items():
        mass[f"mars:{key}"] = float(value) * mass["mars"]
    return weights, mass


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


def train_endpoint(
    model: NdmiBitemporalFusionAdapter,
    loader: DataLoader[dict[str, Any]],
    spec: dict[str, Any],
    device: torch.device,
) -> list[dict[str, float]]:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(spec["learning_rate"]),
        weight_decay=float(spec["weight_decay"]),
    )
    epochs = int(spec["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda")
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
                base_loss, parts = fusion_loss(output, batch, spec)
                partial_auc = partial_auc_pair_loss(
                    output["scene_logit"],
                    batch["presence"],
                    negative_fraction=float(spec["partial_auc_negative_fraction"]),
                    margin=float(spec["partial_auc_margin"]),
                )
                loss = base_loss + float(spec["partial_auc_weight"]) * partial_auc
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(parameters, float(spec["gradient_clip"]))
            scaler.step(optimizer)
            scaler.update()
            parts["loss"] = float(loss.detach())
            parts["partial_auc"] = float(partial_auc.detach())
            batches += 1
            for key, value in parts.items():
                sums[key] = sums.get(key, 0.0) + value
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
    model: NdmiBitemporalFusionAdapter,
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
            [base_scores[value] for value in sample_ids], dtype=torch.float32
        )
        batch = move_batch(batch, device)
        local_base = local_base.to(device)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            output = model(batch["inputs"], batch["observable"], batch["sensor_index"])
        baseline_probability = torch.sigmoid(output["baseline_logits"]).float()
        correction = output["correction_logits"].float()
        for strength in strengths:
            scores = model.fuse_scene_score(
                local_base,
                output["scene_logit"],
                batch["sensor_index"],
                strength,
                sentinel_only=True,
            )
            rows["candidate_scores"][str(strength)].extend(
                float(value) for value in scores.cpu().numpy()
            )
        for index in range(baseline_probability.shape[0]):
            sensor = int(batch["sensor_index"][index])
            threshold = MASK_THRESHOLDS[sensor]
            observable = batch["observable"][index, 0].cpu().numpy() > 0.5
            clear = batch["clear"][index, 0].cpu().numpy() > 0.5
            truth = (batch["mask"][index, 0].cpu().numpy() > 0.5) & observable
            base_score = float(local_base[index])
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
                    + float(strength) * correction[index, 0]
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
        rows["sensors"].extend(
            int(value) for value in batch["sensor_index"].cpu().numpy()
        )
        rows["groups"].extend(group_ids)
        rows["sample_ids"].extend(sample_ids)
        rows["folds"].extend([fold] * len(sample_ids))
        rows["base_scores"].extend(float(value) for value in local_base.cpu().numpy())
        if batch_index % 128 == 0:
            print(
                json.dumps(
                    {"progress": "inference_batch", "fold": fold, "batch": batch_index}
                ),
                flush=True,
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
        for key in (
            "labels",
            "sensors",
            "groups",
            "sample_ids",
            "folds",
            "base_scores",
            "base_pixels",
        ):
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
    labels = np.asarray(raw["labels"], dtype=np.uint8)
    sensors = np.asarray(raw["sensors"], dtype=np.uint8)
    groups = np.asarray(raw["groups"])
    folds = np.asarray(raw["folds"], dtype=np.uint8)
    base = np.asarray(raw["base_scores"], dtype=np.float64)
    base_pixels = np.asarray(raw["base_pixels"], dtype=np.int64)
    base_metrics = metric_summary(labels, base, sensors)
    base_pixel_metrics = pixel_summary(base_pixels)
    candidates: list[dict[str, Any]] = []
    for strength in strengths:
        key = str(strength)
        scores = np.asarray(raw["candidate_scores"][key], dtype=np.float64)
        candidate_pixels = np.asarray(raw["candidate_pixels"][key], dtype=np.int64)
        metrics = metric_summary(labels, scores, sensors)
        versus = comparison(metrics, base_metrics)
        ap_interval = ap_group_bootstrap(
            labels,
            base,
            scores,
            groups,
            replicates=int(bootstrap["replicates"]),
            seed=int(bootstrap["seed"]),
        )
        pixel_metrics = pixel_summary(candidate_pixels)
        iou_delta = float(
            pixel_metrics["intersection_over_union"]
            - base_pixel_metrics["intersection_over_union"]
        )
        iou_interval = pixel_bootstrap(
            base_pixels,
            candidate_pixels,
            groups,
            replicates=int(bootstrap["replicates"]),
            seed=int(bootstrap["seed"]) + 1,
            confidence=float(bootstrap["confidence"]),
        )
        by_fold: dict[str, Any] = {}
        for fold in sorted(np.unique(folds)):
            selected = folds == fold
            local_base = metric_summary(labels[selected], base[selected], sensors[selected])
            local_candidate = metric_summary(
                labels[selected], scores[selected], sensors[selected]
            )
            local_base_pixels = pixel_summary(base_pixels[selected])
            local_candidate_pixels = pixel_summary(candidate_pixels[selected])
            by_fold[str(int(fold))] = {
                "rows": int(np.count_nonzero(selected)),
                "scene": comparison(local_candidate, local_base),
                "base_pixel_rule": local_base_pixels,
                "candidate_pixel_rule": local_candidate_pixels,
                "pixel_iou_delta": float(
                    local_candidate_pixels["intersection_over_union"]
                    - local_base_pixels["intersection_over_union"]
                ),
            }
        sensor_delta = versus["delta"]["sensor_average_precision"]
        fold_ap = [row["scene"]["delta"]["average_precision"] for row in by_fold.values()]
        fold_recall = [
            row["scene"]["delta"]["recall_at_fpr_0_0713"] for row in by_fold.values()
        ]
        fold_iou = [row["pixel_iou_delta"] for row in by_fold.values()]
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
            "each_fold_recall": min(fold_recall)
            >= float(gates["each_fold_recall_delta_minimum"]),
            "paired_site_ap": ap_interval["lower"] > 0.0,
            "pixel_iou": iou_delta > 0.0,
            "each_fold_pixel_iou": min(fold_iou) > 0.0,
            "paired_site_pixel_iou": iou_interval["lower"] > 0.0,
        }
        passed = all(checks.values())
        candidates.append(
            {
                "strength": strength,
                "metrics": metrics,
                "versus_current": versus,
                "paired_site_ap_delta": ap_interval,
                "base_pixel_rule": base_pixel_metrics,
                "candidate_pixel_rule": pixel_metrics,
                "pixel_iou_delta": iou_delta,
                "paired_site_pixel_iou_delta": iou_interval,
                "by_fold": by_fold,
                "checks": checks,
                "passed": passed,
                "rank": [
                    int(passed),
                    min(fold_ap),
                    min(fold_recall),
                    min(sensor_delta.values()),
                    ap_interval["lower"],
                    iou_interval["lower"],
                    versus["delta"]["average_precision"],
                    -strength,
                ],
            }
        )
    identity = {
        "rows": len(labels),
        "fold_counts": dict(Counter(map(int, folds))),
        "sample_id_sha256": hashlib.sha256(
            "\n".join(raw["sample_ids"]).encode()
        ).hexdigest(),
        "current_metrics": base_metrics,
        "current_pixel_rule": base_pixel_metrics,
    }
    return candidates, identity


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    selected = report["selected"]
    scene = selected["versus_current"]["delta"]
    ap_ci = selected["paired_site_ap_delta"]
    iou_ci = selected["paired_site_pixel_iou_delta"]
    lines = [
        "# NDMI-guided bi-temporal fusion pilot",
        "",
        f"- Promotion gates pass: **{report['all_promotion_gates_pass']}**",
        f"- Selected residual strength: **{selected['strength']}**",
        f"- AP delta versus current spatial-Prithvi score: **{scene['average_precision']:+.6f}**",
        f"- Matched-FPR recall delta: **{scene['recall_at_fpr_0_0713']:+.6f}**",
        f"- Paired-site AP interval: **[{ap_ci['lower']:+.6f}, {ap_ci['upper']:+.6f}]**",
        f"- Dense-mask IoU delta: **{selected['pixel_iou_delta']:+.6f}**",
        f"- Paired-site IoU interval: **[{iou_ci['lower']:+.6f}, {iou_ci['upper']:+.6f}]**",
        "",
        report["decision"],
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    os.replace(temporary, path)


def verify_protocol(protocol: dict[str, Any], *, smoke: bool) -> dict[str, Path]:
    frozen = str(protocol["status"]).startswith("frozen")
    if not frozen and not smoke:
        raise ValueError("Outcome evaluation requires a frozen protocol")
    if frozen and sha256(Path(__file__).resolve()) != protocol["trainer"]["sha256"]:
        raise ValueError("Frozen NDMI bi-temporal trainer hash mismatch")
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


def smoke_records(records: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    groups: Counter[str] = Counter()
    selected: list[dict[str, Any]] = []
    for row in records:
        group = str(row["group_id"])
        if groups[group] == 0:
            selected.append(row)
            groups[group] += 1
        if len(selected) >= count:
            break
    if len(selected) < count:
        raise ValueError("Not enough source-disjoint smoke groups")
    return selected


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
        str(row["group_id"]): int(row["fold"])
        for row in fold_protocol["assignments"]
    }
    all_records = list(iter_development_manifest(paths["manifest"]))
    selected_folds = set(map(int, protocol["folds"]))
    records = [
        row for row in all_records if group_to_fold[str(row["group_id"])] in selected_folds
    ]
    unep_records = iter_jsonl(paths["unep_auxiliary_manifest"])
    cloudsen_records = iter_jsonl(paths["cloudsen_auxiliary_manifest"])
    if any(label_state(row) != "PLUME" for row in unep_records):
        raise ValueError("UNEP auxiliary cohort must be positive-only")
    if any(label_state(row) != "NO_PLUME" for row in cloudsen_records):
        raise ValueError("CloudSEN12 auxiliary cohort must be negative-only")
    base_scores, score_identity = load_base_score_contract(
        all_records, group_to_fold, paths["score_cache"]
    )
    spec = protocol["training"]
    strengths = [float(value) for value in protocol["search"]["strengths"]]
    batch_size = int(spec["batch_size"])
    workers = int(spec["loader_workers"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("NDMI bi-temporal pilot requires CUDA")
    torch.cuda.reset_peak_memory_stats()

    if args.smoke:
        seed_everything(int(spec["seed"]))
        mars_smoke = smoke_subset(records, 2)
        unep_smoke = smoke_records(unep_records, 4)
        cloudsen_smoke = smoke_records(cloudsen_records, 4)
        datasets = (
            TaggedDataset(
                MarsPaperDataset(paths["metadata_root"], mars_smoke, augment=True, seed=int(spec["seed"])),
                0,
            ),
            TaggedDataset(
                MarsPaperDataset(ROOT, unep_smoke, augment=True, seed=int(spec["seed"]) + 1),
                1,
            ),
            TaggedDataset(
                MarsPaperDataset(ROOT, cloudsen_smoke, augment=True, seed=int(spec["seed"]) + 2),
                2,
            ),
        )
        weights, request_mass = combined_sampling_weights(
            mars_smoke,
            unep_smoke,
            cloudsen_smoke,
            protocol["sampling"]["source_mass"],
        )
        sampler = WeightedRandomSampler(
            weights,
            num_samples=batch_size * 2,
            replacement=True,
            generator=torch.Generator().manual_seed(int(spec["seed"])),
        )
        loader = make_loader(
            ConcatDataset(datasets), batch_size=batch_size, workers=0, sampler=sampler
        )
        model = NdmiBitemporalFusionAdapter().to(device)
        model.load_released_checkpoint(released_state(paths["released_checkpoint"]))
        first = move_batch(next(iter(loader)), device)
        model.eval()
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
            initial = model(first["inputs"], first["observable"], first["sensor_index"])
        identity_max = float(initial["correction_logits"].abs().max())
        if identity_max != 0.0:
            raise ValueError("Zero-initialized dense path is not exact released identity")
        model.train()
        history = train_endpoint(model, loader, {**spec, "epochs": 1}, device)
        finite = all(math.isfinite(value) for value in history[-1].values())
        result = {
            "ok": finite,
            "identity_pixel_max_abs": identity_max,
            "finite_optimization": finite,
            "request_mass": request_mass,
            "trainable_parameters": model.artifact_metadata()["trainable_parameter_count"],
            "peak_cuda_bytes": int(torch.cuda.max_memory_allocated()),
            "device": torch.cuda.get_device_name(device),
            "history": history,
        }
        print(json.dumps(result, sort_keys=True))
        return 0 if finite else 1

    endpoint_results: list[dict[str, Any]] = []
    prediction_parts: list[dict[str, Any]] = []
    endpoint_states: dict[str, Any] = {}
    for held_fold in sorted(selected_folds):
        fit_folds = selected_folds - {held_fold}
        fit_records = [
            row for row in records if group_to_fold[str(row["group_id"])] in fit_folds
        ]
        held_records = [
            row for row in records if group_to_fold[str(row["group_id"])] == held_fold
        ]
        seed = int(spec["seed"]) + held_fold
        seed_everything(seed)
        datasets = (
            TaggedDataset(
                MarsPaperDataset(paths["metadata_root"], fit_records, augment=True, seed=seed),
                0,
            ),
            TaggedDataset(
                MarsPaperDataset(ROOT, unep_records, augment=True, seed=seed + 100),
                1,
            ),
            TaggedDataset(
                MarsPaperDataset(ROOT, cloudsen_records, augment=True, seed=seed + 200),
                2,
            ),
        )
        weights, request_mass = combined_sampling_weights(
            fit_records,
            unep_records,
            cloudsen_records,
            protocol["sampling"]["source_mass"],
        )
        sampler = WeightedRandomSampler(
            weights,
            num_samples=int(spec["samples_per_epoch"]),
            replacement=True,
            generator=torch.Generator().manual_seed(seed),
        )
        train_loader = make_loader(
            ConcatDataset(datasets),
            batch_size=batch_size,
            workers=workers,
            sampler=sampler,
        )
        evaluation_loader = make_loader(
            MarsPaperDataset(paths["metadata_root"], held_records, augment=False, seed=seed),
            batch_size=batch_size,
            workers=workers,
        )
        model = NdmiBitemporalFusionAdapter().to(device)
        model.load_released_checkpoint(released_state(paths["released_checkpoint"]))
        print(
            json.dumps(
                {
                    "progress": "endpoint_start",
                    "held_fold": held_fold,
                    "seed": seed,
                    "fit_rows": len(fit_records),
                    "held_rows": len(held_records),
                    "request_mass": request_mass,
                }
            ),
            flush=True,
        )
        history = train_endpoint(model, train_loader, spec, device)
        predictions = collect_predictions(
            model,
            evaluation_loader,
            base_scores,
            strengths,
            device,
            held_fold,
        )
        endpoint_results.append(
            {
                "held_fold": held_fold,
                "fit_folds": sorted(fit_folds),
                "fit_rows": len(fit_records),
                "held_rows": len(held_records),
                "seed": seed,
                "request_mass": request_mass,
                "history": history,
            }
        )
        prediction_parts.append(predictions)
        endpoint_states[str(held_fold)] = model.trainable_state()
        del model, train_loader, evaluation_loader
        torch.cuda.empty_cache()

    raw = merge_predictions(prediction_parts, strengths)
    candidates, identity = summarize_predictions(
        raw, strengths, protocol["bootstrap"], protocol["gates"]
    )
    selected = max(candidates, key=lambda row: row["rank"])
    passed = bool(selected["passed"])
    artifact = None
    if passed:
        artifact_path = (ROOT / protocol["outputs"]["artifact"]).resolve()
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = artifact_path.with_suffix(artifact_path.suffix + ".tmp")
        torch.save(
            {
                "schema_version": 1,
                "model": NdmiBitemporalFusionAdapter().artifact_metadata(),
                "states_by_held_fold": endpoint_states,
                "selected_strength": selected["strength"],
                "protocol_sha256": sha256(protocol_path),
            },
            temporary,
        )
        os.replace(temporary, artifact_path)
        artifact = {
            "path": protocol["outputs"]["artifact"],
            "bytes": artifact_path.stat().st_size,
            "sha256": sha256(artifact_path),
            "tracked": False,
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
        "score_identity": score_identity,
        "development_identity": identity,
        "external_auxiliary": {
            "unep_rows": len(unep_records),
            "unep_groups": len({str(row["group_id"]) for row in unep_records}),
            "cloudsen_rows": len(cloudsen_records),
            "cloudsen_groups": len({str(row["group_id"]) for row in cloudsen_records}),
        },
        "endpoints": endpoint_results,
        "candidates": candidates,
        "selected": selected,
        "all_promotion_gates_pass": passed,
        "artifact": artifact,
        "peak_cuda_bytes": int(torch.cuda.max_memory_allocated()),
        "device": torch.cuda.get_device_name(device),
        "decision": (
            "Freeze a multi-seed source-disjoint confirmation; external development and official test remain closed."
            if passed
            else "Reject this architecture before external development, fold 2, or official-test scoring."
        ),
    }
    json_path = (ROOT / protocol["outputs"]["json"]).resolve()
    json_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_json = json_path.with_suffix(json_path.suffix + ".tmp")
    temporary_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary_json, json_path)
    write_markdown((ROOT / protocol["outputs"]["markdown"]).resolve(), report)
    print(
        json.dumps(
            {
                "ok": passed,
                "strength": selected["strength"],
                "ap_delta": selected["versus_current"]["delta"]["average_precision"],
                "recall_delta": selected["versus_current"]["delta"][
                    "recall_at_fpr_0_0713"
                ],
                "iou_delta": selected["pixel_iou_delta"],
                "artifact": artifact,
            },
            sort_keys=True,
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
