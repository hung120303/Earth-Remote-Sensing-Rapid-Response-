#!/usr/bin/env python3
"""Cross-fit v6 with an unconstrained champion error-correction objective."""

from __future__ import annotations

import argparse
import gc
import json
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
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "EarthRemoteSensingRapidResponse"
for path in (MODEL_ROOT, ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from acquire_mars_metadata import sha256  # noqa: E402
from train_mars_dinov3_methane_fusion_pilot import partial_auc_pair_loss  # noqa: E402
from train_mars_oof_scene_ensemble_v2 import ap_group_bootstrap  # noqa: E402
from train_mars_paper_residual import iter_development_manifest  # noqa: E402
from train_mars_v6_scene_pilot import (  # noqa: E402
    MarsSceneDataset,
    MethaneSceneDataset,
    build_model,
    candidate_evaluation,
    canonical,
    focal_loss,
    iter_jsonl,
    load_champion,
    mars_source_weights,
    move,
    optimizer_for,
    score_endpoint,
    verify_protocol,
)

DEFAULT_PROTOCOL = Path("configs/mars_v6_error_correcting_pilot_protocol.json")


def effective_training_logits(
    baseline_score: torch.Tensor,
    residual_logit: torch.Tensor,
    protected: torch.Tensor,
    *,
    residual_scale: float,
) -> torch.Tensor:
    """Use an unconstrained residual during fitting; protection is inference-only."""
    baseline_logit = torch.logit(baseline_score.clamp(1e-6, 1.0 - 1e-6))
    corrected = baseline_logit + float(residual_scale) * residual_logit
    return torch.where(protected, corrected, residual_logit)


def train_batch(
    model: torch.nn.Module,
    batch: dict[str, Any],
    source: str,
    optimizer: torch.optim.Optimizer,
    spec: dict[str, Any],
    device: torch.device,
) -> dict[str, float]:
    batch = move(batch, device)
    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        residual = model(
            canonical(batch, source),
            batch["temporal"],
            batch["location"],
            branch="scene",
        )["scene_logit"]
        effective = effective_training_logits(
            batch["baseline_score"],
            residual,
            batch["protected"],
            residual_scale=float(spec["training_residual_scale"]),
        )
        focal = focal_loss(effective, batch["presence"], float(spec["focal_gamma"]))
        pair = partial_auc_pair_loss(
            effective,
            batch["presence"],
            negative_fraction=float(spec["partial_auc_negative_fraction"]),
            margin=float(spec["partial_auc_margin"]),
        )
        regularizer = torch.where(batch["protected"], residual.square(), 0.0).mean()
        loss = (
            focal
            + float(spec["partial_auc_weight"]) * pair
            + float(spec["residual_l2_weight"]) * regularizer
        )
    loss.backward()
    parameters = [value for value in model.parameters() if value.requires_grad]
    if not all(
        torch.isfinite(value.grad).all()
        for value in parameters
        if value.grad is not None
    ):
        raise FloatingPointError("V6.1 produced a non-finite gradient")
    norm = torch.nn.utils.clip_grad_norm_(parameters, float(spec["gradient_clip"]))
    optimizer.step()
    return {
        "loss": float(loss.detach()),
        "focal": float(focal.detach()),
        "pair": float(pair.detach()),
        "residual_l2": float(regularizer.detach()),
        "gradient_norm": float(norm),
    }


def train_endpoint(
    model: torch.nn.Module,
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
    optimizer = optimizer_for(model, spec)
    epochs = 1 if smoke else int(spec["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        hard = not smoke and epoch > epochs - int(spec["hard_negative_epochs"])
        mars_datasets = (
            MarsSceneDataset(
                paths["mars_root"], mars_rows, baseline_scores,
                protected=True, augment=True, seed=seed + epoch,
            ),
            MarsSceneDataset(
                ROOT, unep_rows, baseline_scores,
                protected=False, augment=True, seed=seed + 100 + epoch,
            ),
            MarsSceneDataset(
                ROOT, cloudsen_rows, baseline_scores,
                protected=False, augment=True, seed=seed + 200 + epoch,
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
        methane_requests = int(
            spec["smoke_requests"] if smoke else spec["methanes2cm_requests_per_epoch"]
        )
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
        from train_mars_v6_scene_pilot import group_balanced_weights

        methane_loader = DataLoader(
            MethaneSceneDataset(
                paths["methanes2cm_packed"], methane_rows,
                augment=True, seed=seed + 300 + epoch,
            ),
            batch_size=int(spec["methanes2cm_batch_size"]),
            sampler=WeightedRandomSampler(
                group_balanced_weights(methane_rows, label_key="label"),
                num_samples=methane_requests,
                replacement=True,
                generator=torch.Generator().manual_seed(seed + epoch * 19),
            ),
            num_workers=workers,
            pin_memory=True,
            persistent_workers=workers > 0,
        )
        model.train()
        sums: Counter[str] = Counter()
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


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    selected = report["selected"]
    delta = selected["versus_champion"]["delta"]
    interval = selected["paired_site_ap_delta"]
    lines = [
        "# ERSRR v6.1 error-correcting residual pilot",
        "",
        f"- Promotion gates pass: **{report['all_promotion_gates_pass']}**",
        f"- Selected protected residual strength: **{selected['strength']}**",
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
        raise RuntimeError("V6.1 scene pilot requires CUDA")
    fold_payload = json.loads(paths["mars_folds"].read_text(encoding="utf-8"))
    group_to_fold = {
        str(item["group_id"]): int(item["fold"])
        for item in fold_payload["assignments"]
    }
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
        mars_smoke = [
            next(row for row in rows if int(row["label_state"] == "PLUME") == target)
            for target in (0, 1)
        ]
        methane_smoke = [
            next(row for row in methane if int(row["label"]) == target)
            for target in (0, 1)
        ]
        model = build_model(paths, protocol["architecture"], device)
        torch.cuda.reset_peak_memory_stats()
        history = train_endpoint(
            model, mars_smoke, unep[:2], cloudsen[:2], methane_smoke,
            baseline_scores, paths, protocol, int(protocol["training"]["seed"]),
            device, smoke=True,
        )
        smoke_report = {
            "schema_version": 1,
            "scope": "v6.1 unconstrained residual mixed-source optimization smoke",
            "history": history,
            "peak_cuda_bytes": int(torch.cuda.max_memory_allocated()),
            "held_outcomes_accessed": False,
        }
        output = (ROOT / protocol["outputs"]["smoke"]).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(smoke_report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(smoke_report, sort_keys=True))
        return 0

    output_json = (ROOT / protocol["outputs"]["json"]).resolve()
    if output_json.exists():
        raise FileExistsError("Refusing to repeat the v6.1 scene pilot")
    seed = int(protocol["training"]["seed"])
    raw_by_id: dict[str, float] = {}
    histories: dict[str, Any] = {}
    states: dict[str, Any] = {}
    for held in (3, 4):
        fit = [row for row in rows if group_to_fold[str(row["group_id"])] == 7 - held]
        held_rows = [row for row in rows if group_to_fold[str(row["group_id"])] == held]
        torch.manual_seed(seed + held)
        np.random.seed(seed + held)
        torch.cuda.manual_seed_all(seed + held)
        model = build_model(paths, protocol["architecture"], device)
        histories[str(held)] = train_endpoint(
            model, fit, unep, cloudsen, methane, baseline_scores, paths,
            protocol, seed + held, device, smoke=False,
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
    raw = np.asarray(
        [raw_by_id[str(value)] for value in champion["sample_ids"]], dtype=np.float64
    )
    raw_path = (ROOT / protocol["outputs"]["raw_cache"]).resolve()
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(raw_path, sample_ids=champion["sample_ids"], raw_logits=raw)
    candidates = []
    for index, strength in enumerate(map(float, protocol["search"]["strengths"])):
        row = candidate_evaluation(
            champion, raw, strength, float(protocol["architecture"]["protection_gate"])
        )
        interval = ap_group_bootstrap(
            champion["labels"], champion["champion_scores"], row["scores"],
            champion["groups"], replicates=int(protocol["bootstrap"]["replicates"]),
            seed=int(protocol["bootstrap"]["seed"]) + index,
        )
        delta = row["versus_champion"]["delta"]
        fold_ap = [row["per_fold"][str(fold)]["delta"]["average_precision"] for fold in (3, 4)]
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
            int(row["passed"]), min(fold_ap), interval["lower"], min(sensor_ap),
            delta["average_precision"], -strength,
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
        "raw_cache": {
            "path": protocol["outputs"]["raw_cache"],
            "bytes": raw_path.stat().st_size,
            "sha256": sha256(raw_path),
            "tracked": False,
        },
        "candidates": candidates,
        "selected": selected,
        "all_promotion_gates_pass": passed,
        "held_external_or_official_outcomes_accessed": False,
        "decision": (
            "Authorize a second seed and dense-v6 branch."
            if passed
            else "Reject the unconstrained v6.1 residual schedule before external confirmation."
        ),
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_markdown((ROOT / protocol["outputs"]["markdown"]).resolve(), report)
    if passed:
        artifact = (ROOT / protocol["outputs"]["artifact"]).resolve()
        artifact.parent.mkdir(parents=True, exist_ok=True)
        temporary = artifact.with_suffix(artifact.suffix + ".tmp")
        torch.save(
            {"states": states, "selected_strength": selected["strength"]}, temporary
        )
        os.replace(temporary, artifact)
    print(json.dumps({"passed": passed, "selected": selected}, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
