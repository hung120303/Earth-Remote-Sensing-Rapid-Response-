#!/usr/bin/env python3
"""Cross-fit the product-aware v6 scene branch before dense v6 investment."""

from __future__ import annotations

import argparse
import gc
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
from calibrate_mars_v6_group_risk import calibration_report  # noqa: E402
from evaluate_mars_dofa_gaussian_protected_ensemble import (  # noqa: E402
    gaussian_local_candidate,
)
from mars_v6_product_model import (  # noqa: E402
    ProductHarmonizedMultiCohortV6,
    canonicalize_mars,
    canonicalize_methanes2cm,
)
from train_mars_dinov3_methane_fusion_pilot import partial_auc_pair_loss  # noqa: E402
from train_mars_oof_scene_ensemble_v2 import ap_group_bootstrap  # noqa: E402
from train_mars_paper_residual import MarsPaperDataset, iter_development_manifest  # noqa: E402
from train_mars_scene_ranker import comparison, metric_summary  # noqa: E402
from train_mars_v6_unified import (  # noqa: E402
    build_model,
    iter_jsonl,
    mars_temporal_location,
    methanes2cm_temporal_location,
    verify_protocol,
)
from train_methanes2cm_v5 import PackedMethaneS2CMDataset  # noqa: E402


DEFAULT_PROTOCOL = Path("configs/mars_v6_scene_pilot_protocol.json")


class MarsSceneDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        root: Path,
        rows: list[dict[str, Any]],
        baseline_scores: dict[str, float],
        *,
        protected: bool,
        augment: bool,
        seed: int,
    ) -> None:
        self.rows = rows
        self.dataset = MarsPaperDataset(
            root,
            rows,
            augment=augment,
            seed=seed,
            allow_missing_positive_mask=True,
        )
        self.baseline_scores = baseline_scores
        self.protected = protected

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        item = self.dataset[index]
        temporal, location, available = mars_temporal_location([row], torch.device("cpu"))
        item.update(
            temporal=temporal[0],
            location=location[0],
            reference90_available=available[0],
            baseline_score=torch.tensor(
                self.baseline_scores.get(str(row["sample_id"]), 0.5), dtype=torch.float32
            ),
            protected=torch.tensor(self.protected, dtype=torch.bool),
        )
        return item


class MethaneSceneDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        packed: Path,
        rows: list[dict[str, Any]],
        *,
        augment: bool,
        seed: int,
    ) -> None:
        self.rows = rows
        self.dataset = PackedMethaneS2CMDataset(
            packed, rows, augment=augment, seed=seed
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        item = self.dataset[index]
        temporal, location = methanes2cm_temporal_location([row], torch.device("cpu"))
        item.update(
            temporal=temporal[0],
            location=location[0],
            baseline_score=torch.tensor(0.5, dtype=torch.float32),
            protected=torch.tensor(False, dtype=torch.bool),
            sensor_index=torch.tensor(0, dtype=torch.long),
            reference90_available=torch.tensor(1.0, dtype=torch.float32),
        )
        return item


def group_balanced_weights(
    rows: list[dict[str, Any]],
    *,
    label_key: str,
    hard_negative_scores: dict[str, float] | None = None,
    hard_negative_quantile: float = 0.9,
) -> torch.Tensor:
    labels = [
        int(row[label_key]) if label_key == "label" else int(row[label_key] == "PLUME")
        for row in rows
    ]
    group_counts = Counter((labels[index], str(row["group_id"])) for index, row in enumerate(rows))
    groups_by_label = Counter(label for label, _ in group_counts)
    threshold = math.inf
    if hard_negative_scores:
        negative = [
            float(hard_negative_scores[str(row["sample_id"])])
            for row, value in zip(rows, labels)
            if value == 0 and str(row["sample_id"]) in hard_negative_scores
        ]
        if negative:
            threshold = float(np.quantile(negative, hard_negative_quantile))
    values = []
    for row, value in zip(rows, labels):
        weight = 0.5 / groups_by_label[value] / group_counts[(value, str(row["group_id"]))]
        if (
            value == 0
            and hard_negative_scores
            and float(hard_negative_scores.get(str(row["sample_id"]), -math.inf)) >= threshold
        ):
            weight *= 2.0
        values.append(weight)
    result = torch.tensor(values, dtype=torch.double)
    return result / result.sum()


def group_only_weights(rows: list[dict[str, Any]]) -> torch.Tensor:
    counts = Counter(str(row["group_id"]) for row in rows)
    values = torch.tensor(
        [1.0 / counts[str(row["group_id"])] for row in rows], dtype=torch.double
    )
    return values / values.sum()


def mars_source_weights(
    mars: list[dict[str, Any]],
    unep: list[dict[str, Any]],
    cloudsen: list[dict[str, Any]],
    masses: dict[str, float],
    *,
    baseline_scores: dict[str, float],
    hard_negative: bool,
) -> torch.Tensor:
    mars_weight = group_balanced_weights(
        mars,
        label_key="label_state",
        hard_negative_scores=baseline_scores if hard_negative else None,
    )
    unep_weight = group_only_weights(unep)
    cloudsen_weight = group_only_weights(cloudsen)
    result = torch.cat(
        (
            mars_weight * float(masses["mars"]),
            unep_weight * float(masses["unep_positive"]),
            cloudsen_weight * float(masses["cloudsen_negative"]),
        )
    )
    return result / result.sum()


def move(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def canonical(batch: dict[str, Any], source: str) -> Any:
    if source == "methanes2cm":
        return canonicalize_methanes2cm(batch["inputs"], batch["observable"])
    return canonicalize_mars(
        batch["inputs"],
        batch["observable"],
        batch["sensor_index"],
        reference90_available=batch["reference90_available"],
    )


def focal_loss(logits: torch.Tensor, target: torch.Tensor, gamma: float) -> torch.Tensor:
    rows = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    probability = torch.exp(-rows)
    return (((1.0 - probability) ** gamma) * rows).mean()


def train_batch(
    model: ProductHarmonizedMultiCohortV6,
    batch: dict[str, Any],
    source: str,
    optimizer: torch.optim.Optimizer,
    spec: dict[str, Any],
    device: torch.device,
) -> dict[str, float]:
    batch = move(batch, device)
    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        raw = model(
            canonical(batch, source),
            batch["temporal"],
            batch["location"],
            branch="scene",
        )["scene_logit"]
        protected_score = model.protected_scene_score(
            batch["baseline_score"],
            raw,
            strength=float(spec["training_strength"]),
            protection_gate=float(spec["protection_gate"]),
        )
        effective = torch.where(
            batch["protected"],
            torch.logit(protected_score.clamp(1e-6, 1.0 - 1e-6)),
            raw,
        )
        focal = focal_loss(effective, batch["presence"], float(spec["focal_gamma"]))
        pair = partial_auc_pair_loss(
            effective,
            batch["presence"],
            negative_fraction=float(spec["partial_auc_negative_fraction"]),
            margin=float(spec["partial_auc_margin"]),
        )
        loss = focal + float(spec["partial_auc_weight"]) * pair
    loss.backward()
    parameters = [value for value in model.parameters() if value.requires_grad]
    if not all(
        torch.isfinite(value.grad).all()
        for value in parameters
        if value.grad is not None
    ):
        raise FloatingPointError("V6 scene pilot produced a non-finite gradient")
    norm = torch.nn.utils.clip_grad_norm_(parameters, float(spec["gradient_clip"]))
    optimizer.step()
    return {
        "loss": float(loss.detach()),
        "focal": float(focal.detach()),
        "pair": float(pair.detach()),
        "gradient_norm": float(norm),
    }


def optimizer_for(model: ProductHarmonizedMultiCohortV6, spec: dict[str, Any]) -> torch.optim.Optimizer:
    model.set_trainable_phase("scene")
    lora = []
    head = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (lora if ".a." in name or ".b." in name else head).append(parameter)
    return torch.optim.AdamW(
        [
            {"params": lora, "lr": float(spec["lora_learning_rate"])},
            {"params": head, "lr": float(spec["scene_learning_rate"])},
        ],
        weight_decay=float(spec["weight_decay"]),
    )


def train_endpoint(
    model: ProductHarmonizedMultiCohortV6,
    mars_rows: list[dict[str, Any]],
    unep_rows: list[dict[str, Any]],
    cloudsen_rows: list[dict[str, Any]],
    methane_rows: list[dict[str, Any]],
    baseline_scores: dict[str, float],
    paths: dict[str, Path],
    protocol: dict[str, Any],
    seed: int,
    device: torch.device,
    *,
    smoke: bool,
) -> list[dict[str, float]]:
    spec = dict(protocol["training"])
    architecture = protocol["architecture"]
    spec["protection_gate"] = architecture["protection_gate"]
    optimizer = optimizer_for(model, spec)
    epochs = 1 if smoke else int(spec["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    history = []
    for epoch in range(1, epochs + 1):
        hard = not smoke and epoch > epochs - int(spec["hard_negative_epochs"])
        mars_datasets = (
            MarsSceneDataset(
                paths["mars_root"], mars_rows, baseline_scores, protected=True, augment=True, seed=seed + epoch
            ),
            MarsSceneDataset(
                ROOT, unep_rows, baseline_scores, protected=False, augment=True, seed=seed + 100 + epoch
            ),
            MarsSceneDataset(
                ROOT, cloudsen_rows, baseline_scores, protected=False, augment=True, seed=seed + 200 + epoch
            ),
        )
        mars_weights = mars_source_weights(
            mars_rows,
            unep_rows,
            cloudsen_rows,
            protocol["sampling"]["mars_compatible_source_mass"],
            baseline_scores=baseline_scores,
            hard_negative=hard,
        )
        mars_requests = int(spec["smoke_requests"] if smoke else spec["mars_requests_per_epoch"])
        methane_requests = int(spec["smoke_requests"] if smoke else spec["methanes2cm_requests_per_epoch"])
        workers = 0 if smoke else int(spec["loader_workers"])
        mars_loader = DataLoader(
            ConcatDataset(mars_datasets),
            batch_size=int(spec["mars_batch_size"]),
            sampler=WeightedRandomSampler(
                mars_weights,
                num_samples=mars_requests,
                replacement=True,
                generator=torch.Generator().manual_seed(seed + epoch * 17),
            ),
            num_workers=workers,
            pin_memory=True,
            persistent_workers=workers > 0,
        )
        methane_weights = group_balanced_weights(methane_rows, label_key="label")
        methane_loader = DataLoader(
            MethaneSceneDataset(
                paths["methanes2cm_packed"], methane_rows, augment=True, seed=seed + 300 + epoch
            ),
            batch_size=int(spec["methanes2cm_batch_size"]),
            sampler=WeightedRandomSampler(
                methane_weights,
                num_samples=methane_requests,
                replacement=True,
                generator=torch.Generator().manual_seed(seed + epoch * 19),
            ),
            num_workers=workers,
            pin_memory=True,
            persistent_workers=workers > 0,
        )
        model.train()
        sums = Counter()
        batches = 0
        started = time.perf_counter()
        for source, loader in (("mars", mars_loader), ("methanes2cm", methane_loader)):
            for batch in loader:
                row = train_batch(model, batch, source, optimizer, spec, device)
                batches += 1
                for key, value in row.items():
                    sums[key] += value
        scheduler.step()
        summary = {
            "epoch": epoch,
            "hard_negative": hard,
            "seconds": time.perf_counter() - started,
            **{key: value / batches for key, value in sums.items()},
        }
        history.append(summary)
        print(json.dumps(summary, sort_keys=True), flush=True)
    return history


@torch.no_grad()
def score_endpoint(
    model: ProductHarmonizedMultiCohortV6,
    rows: list[dict[str, Any]],
    baseline_scores: dict[str, float],
    paths: dict[str, Path],
    spec: dict[str, Any],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    dataset = MarsSceneDataset(
        paths["mars_root"], rows, baseline_scores, protected=True, augment=False, seed=0
    )
    loader = DataLoader(
        dataset,
        batch_size=int(spec["evaluation_batch_size"]),
        shuffle=False,
        num_workers=int(spec["loader_workers"]),
        pin_memory=True,
        persistent_workers=int(spec["loader_workers"]) > 0,
    )
    model.eval()
    ids = []
    logits = []
    for index, batch in enumerate(loader, start=1):
        batch = move(batch, device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            raw = model(
                canonical(batch, "mars"),
                batch["temporal"],
                batch["location"],
                branch="scene",
            )["scene_logit"]
        ids.extend(str(value) for value in batch["sample_id"])
        logits.extend(float(value) for value in raw.float().cpu().numpy())
        if index % 64 == 0:
            print(json.dumps({"progress": "score", "batch": index}), flush=True)
    return np.asarray(ids), np.asarray(logits, dtype=np.float64)


def load_champion(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        values = {name: np.asarray(source[name]) for name in source.files}
    if values["sample_ids"].size != 17745 or set(map(int, values["folds"])) != {3, 4}:
        raise ValueError("V6 champion cache row contract differs")
    return values


def candidate_evaluation(
    values: dict[str, np.ndarray], raw: np.ndarray, strength: float, gate: float
) -> dict[str, Any]:
    scores = gaussian_local_candidate(
        values["champion_scores"].astype(float), raw, strength=strength, gate=gate
    )
    candidate = metric_summary(values["labels"], scores, values["sensors"])
    baseline = metric_summary(
        values["labels"], values["champion_scores"], values["sensors"]
    )
    versus = comparison(candidate, baseline)
    per_fold = {}
    for fold in (3, 4):
        local = values["folds"] == fold
        per_fold[str(fold)] = comparison(
            metric_summary(values["labels"][local], scores[local], values["sensors"][local]),
            metric_summary(
                values["labels"][local],
                values["champion_scores"][local],
                values["sensors"][local],
            ),
        )
    return {"strength": strength, "scores": scores, "metrics": candidate, "versus_champion": versus, "per_fold": per_fold}


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    selected = report["selected"]
    delta = selected["versus_champion"]["delta"]
    interval = selected["paired_site_ap_delta"]
    lines = [
        "# ERSRR v6 product-aware scene pilot",
        "",
        f"- Promotion gates pass: **{report['all_promotion_gates_pass']}**",
        f"- Selected residual strength: **{selected['strength']}**",
        f"- AP delta versus Gaussian+DOFA champion: **{delta['average_precision']:+.6f}**",
        f"- Matched-FPR recall delta: **{delta['recall_at_fpr_0_0713']:+.6f}**",
        f"- Paired-site AP interval: **[{interval['lower']:+.6f}, {interval['upper']:+.6f}]**",
        "",
        report["decision"],
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL.as_posix())
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    protocol_path = (ROOT / args.protocol).resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    paths = verify_protocol(protocol_path, protocol, smoke=args.smoke)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("V6 scene pilot requires CUDA")
    fold_payload = json.loads(paths["mars_folds"].read_text(encoding="utf-8"))
    group_to_fold = {str(item["group_id"]): int(item["fold"]) for item in fold_payload["assignments"]}
    all_rows = list(iter_development_manifest(paths["mars_manifest"]))
    rows = [row for row in all_rows if group_to_fold[str(row["group_id"])] in {3, 4}]
    unep = iter_jsonl(paths["unep_auxiliary"])
    cloudsen = iter_jsonl(paths["cloudsen_auxiliary"])
    methane = iter_jsonl(paths["methanes2cm_auxiliary"])
    champion = load_champion(paths["champion_cache"])
    baseline_scores = dict(
        zip(champion["sample_ids"].astype(str), champion["champion_scores"].astype(float))
    )
    if args.smoke:
        mars_smoke = []
        for target in (0, 1):
            mars_smoke.append(next(row for row in rows if int(row["label_state"] == "PLUME") == target))
        methane_smoke = []
        for target in (0, 1):
            methane_smoke.append(next(row for row in methane if int(row["label"]) == target))
        model = build_model(paths, protocol["architecture"], device)
        torch.cuda.reset_peak_memory_stats()
        history = train_endpoint(
            model,
            mars_smoke,
            unep[:2],
            cloudsen[:2],
            methane_smoke,
            baseline_scores,
            paths,
            protocol,
            int(protocol["training"]["seed"]),
            device,
            smoke=True,
        )
        smoke_report = {
            "schema_version": 1,
            "scope": "v6 scene-pilot mixed-source optimization smoke; no held-fold score",
            "history": history,
            "peak_cuda_bytes": int(torch.cuda.max_memory_allocated()),
            "held_outcomes_accessed": False,
        }
        output = (ROOT / protocol["outputs"]["smoke"]).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(smoke_report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(smoke_report, sort_keys=True))
        return 0

    if (ROOT / protocol["outputs"]["json"]).exists():
        raise FileExistsError("Refusing to repeat the v6 scene pilot")
    seed = int(protocol["training"]["seed"])
    raw_by_id: dict[str, float] = {}
    histories = {}
    states = {}
    for held in (3, 4):
        fit = [row for row in rows if group_to_fold[str(row["group_id"])] == 7 - held]
        held_rows = [row for row in rows if group_to_fold[str(row["group_id"])] == held]
        torch.manual_seed(seed + held)
        np.random.seed(seed + held)
        torch.cuda.manual_seed_all(seed + held)
        model = build_model(paths, protocol["architecture"], device)
        histories[str(held)] = train_endpoint(
            model,
            fit,
            unep,
            cloudsen,
            methane,
            baseline_scores,
            paths,
            protocol,
            seed + held,
            device,
            smoke=False,
        )
        ids, logits = score_endpoint(
            model, held_rows, baseline_scores, paths, protocol["training"], device
        )
        raw_by_id.update(zip(ids.astype(str), logits.astype(float)))
        model.set_trainable_phase("scene")
        states[str(held)] = {
            name: value.detach().cpu()
            for name, value in model.named_parameters()
            if value.requires_grad
        }
        del model
        gc.collect()
        torch.cuda.empty_cache()
    try:
        raw = np.asarray([raw_by_id[str(value)] for value in champion["sample_ids"]], dtype=np.float64)
    except KeyError as exc:
        raise ValueError("V6 scene pilot did not score every champion identity") from exc
    candidates = []
    for index, strength in enumerate(map(float, protocol["search"]["strengths"])):
        row = candidate_evaluation(
            champion, raw, strength, float(protocol["architecture"]["protection_gate"])
        )
        interval = ap_group_bootstrap(
            champion["labels"],
            champion["champion_scores"],
            row["scores"],
            champion["groups"],
            replicates=int(protocol["bootstrap"]["replicates"]),
            seed=int(protocol["bootstrap"]["seed"]) + index,
        )
        delta = row["versus_champion"]["delta"]
        fold_ap = [
            row["per_fold"][str(fold)]["delta"]["average_precision"] for fold in (3, 4)
        ]
        sensor_ap = list(delta["sensor_average_precision"].values())
        checks = {
            "minimum_ap_delta": delta["average_precision"] >= float(protocol["gates"]["average_precision_delta_minimum"]),
            "paired_site_lower_positive": interval["lower"] > 0.0,
            "each_fold_ap_positive": min(fold_ap) > 0.0,
            "each_sensor_ap_positive": min(sensor_ap) > 0.0,
            "matched_fpr_recall_no_worse": delta["recall_at_fpr_0_0713"] >= 0.0,
        }
        row["paired_site_ap_delta"] = interval
        row["checks"] = checks
        row["passed"] = all(checks.values())
        row["rank"] = [
            int(row["passed"]),
            min(fold_ap),
            interval["lower"],
            min(sensor_ap),
            delta["average_precision"],
            -strength,
        ]
        del row["scores"]
        candidates.append(row)
    selected = max(candidates, key=lambda value: tuple(value["rank"]))
    for row in candidates:
        del row["rank"]
    passed = bool(selected["passed"])
    report = {
        "schema_version": 1,
        "status": "passed" if passed else "rejected",
        "scope": protocol["scope"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": sha256(protocol_path),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "seed": seed,
        "histories": histories,
        "candidates": candidates,
        "selected": selected,
        "all_promotion_gates_pass": passed,
        "held_external_or_official_outcomes_accessed": False,
        "decision": (
            "Authorize the frozen second-seed replication and dense v6 branch."
            if passed
            else "Reject this v6 scene schedule before dense or external confirmation work."
        ),
    }
    output_json = (ROOT / protocol["outputs"]["json"]).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_markdown((ROOT / protocol["outputs"]["markdown"]).resolve(), report)
    if passed:
        artifact = (ROOT / protocol["outputs"]["artifact"]).resolve()
        artifact.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"states": states, "selected_strength": selected["strength"]}, artifact)
    print(json.dumps({"passed": passed, "selected": selected}, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

