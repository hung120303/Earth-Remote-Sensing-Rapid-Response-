#!/usr/bin/env python3
"""Train and test a conservative instance-signal scene ensemble on fold 2."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "EarthRemoteSensingRapidResponse"
for path in (MODEL_ROOT, ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from acquire_mars_metadata import sha256  # noqa: E402
from analyze_mars_scene_gated_masks import (  # noqa: E402
    count_metrics,
    gate_counts,
    paired_group_bootstrap,
)
from evaluate_released_marss2l import connected_scene_score  # noqa: E402
from mars_instance_guided_teacher import InstanceGuidedTeacherAdapter  # noqa: E402
from mars_paper_model import SENSOR_NAMES, released_state  # noqa: E402
from train_mars_instance_guided_teacher_pilot import (  # noqa: E402
    InstanceTargetDataset,
    train_endpoint,
)
from train_mars_oof_scene_ensemble_v2 import ap_group_bootstrap  # noqa: E402
from train_mars_paper_residual import (  # noqa: E402
    MarsPaperDataset,
    iter_development_manifest,
    move_batch,
    verify_acquisition_receipt,
)
from train_mars_physics_guided_teacher_balanced_pilot import (  # noqa: E402
    balanced_request_weights,
)
from train_mars_scene_ranker import comparison, metric_summary  # noqa: E402
from train_mars_source_aligned_residual import offshore_flags  # noqa: E402


DEFAULT_PROTOCOL = Path("configs/mars_instance_scene_ensemble_pilot_protocol.json")


def sensor_mask_summary(
    baseline: np.ndarray,
    candidate: np.ndarray,
    sensors: np.ndarray,
) -> dict[str, dict[str, float]]:
    values: dict[str, dict[str, float]] = {}
    for sensor_index, name in enumerate(SENSOR_NAMES):
        rows = sensors == sensor_index
        base = count_metrics(baseline[rows])
        local = count_metrics(candidate[rows])
        values[name] = {
            "baseline_iou": float(base["iou"]),
            "candidate_iou": float(local["iou"]),
            "delta": float(local["iou"] - base["iou"]),
        }
    return values


def frozen_mask_evidence(
    pixel_cache: Path,
    labels: np.ndarray,
    sensors: np.ndarray,
    groups: np.ndarray,
    current_scores: np.ndarray,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    with np.load(pixel_cache, allow_pickle=False) as pixel:
        rows = pixel["folds"] == int(protocol["folds"]["held"])
        pixel_labels = pixel["labels"][rows]
        pixel_sensors = pixel["sensors"][rows]
        pixel_groups = pixel["groups"][rows]
        if not np.array_equal(pixel_labels, labels):
            raise ValueError("Frozen pixel-cache labels do not align with held fold")
        if not np.array_equal(pixel_sensors, sensors):
            raise ValueError("Frozen pixel-cache sensors do not align with held fold")
        if not np.array_equal(pixel_groups, groups):
            raise ValueError("Frozen pixel-cache groups do not align with held fold")
        sentinel_index = SENSOR_NAMES.index("Sentinel-2")
        baseline = np.where(
            (sensors == sentinel_index)[:, None],
            pixel["threshold_08"][rows],
            pixel["threshold_07"][rows],
        )
    mask_rule = protocol["mask_rule"]
    candidate = gate_counts(
        baseline,
        current_scores,
        float(mask_rule["scene_probability_cutoff"]),
    )
    base_metrics = count_metrics(baseline)
    candidate_metrics = count_metrics(candidate)
    interval = paired_group_bootstrap(
        baseline,
        candidate,
        groups,
        replicates=int(protocol["bootstrap"]["replicates"]),
        seed=int(protocol["bootstrap"]["mask_seed"]),
        confidence=float(protocol["bootstrap"]["confidence"]),
    )
    sensors_summary = sensor_mask_summary(baseline, candidate, sensors)
    retained_tp = float(candidate_metrics["tp"] / max(int(base_metrics["tp"]), 1))
    passed = bool(
        candidate_metrics["iou"] > base_metrics["iou"]
        and interval["lower"] > 0.0
        and min(value["delta"] for value in sensors_summary.values()) >= 0.0
        and retained_tp >= float(mask_rule["minimum_retained_true_positive_fraction"])
    )
    return {
        "rule": mask_rule,
        "baseline": base_metrics,
        "candidate": candidate_metrics,
        "delta": float(candidate_metrics["iou"] - base_metrics["iou"]),
        "paired_site_bootstrap": interval,
        "sensors": sensors_summary,
        "retained_true_positive_fraction": retained_tp,
        "passed": passed,
    }


@torch.no_grad()
def collect_instance_signals(
    model: InstanceGuidedTeacherAdapter,
    loader: DataLoader[dict[str, Any]],
    device: torch.device,
    correction_strength: float,
) -> dict[str, np.ndarray]:
    model.eval()
    labels: list[int] = []
    sensors: list[int] = []
    groups: list[str] = []
    sample_ids: list[str] = []
    connected: list[float] = []
    scene_head: list[float] = []
    proposal: list[float] = []
    released: list[float] = []
    for batch in loader:
        local_ids = [str(value) for value in batch["sample_id"]]
        local_groups = [str(value) for value in batch["group_id"]]
        batch = move_batch(batch, device)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            output = model(batch["inputs"], batch["observable"], batch["sensor_index"])
            proposal_logit = model.proposal_scene_surrogate(
                output["object_logits"], batch["observable"]
            )
        corrected_probability = torch.sigmoid(
            output["baseline_logits"]
            + float(correction_strength) * output["correction_logits"]
        ).float()
        released_probability = torch.sigmoid(output["baseline_logits"]).float()
        clear = batch["clear"] > 0.5
        corrected_probability = corrected_probability.masked_fill(~clear, 0.0)
        released_probability = released_probability.masked_fill(~clear, 0.0)
        for index in range(corrected_probability.shape[0]):
            connected.append(
                connected_scene_score(corrected_probability[index, 0].cpu().numpy())
            )
            released.append(
                connected_scene_score(released_probability[index, 0].cpu().numpy())
            )
        scene_head.extend(torch.sigmoid(output["scene_logit"]).float().cpu().numpy())
        proposal.extend(torch.sigmoid(proposal_logit).float().cpu().numpy())
        labels.extend(int(value) for value in batch["presence"].cpu().numpy())
        sensors.extend(int(value) for value in batch["sensor_index"].cpu().numpy())
        groups.extend(local_groups)
        sample_ids.extend(local_ids)
    return {
        "labels": np.asarray(labels, dtype=np.uint8),
        "sensors": np.asarray(sensors, dtype=np.uint8),
        "groups": np.asarray(groups),
        "sample_ids": np.asarray(sample_ids),
        "released": np.asarray(released, dtype=np.float64),
        "connected": np.asarray(connected, dtype=np.float64),
        "scene_head": np.asarray(scene_head, dtype=np.float64),
        "proposal": np.asarray(proposal, dtype=np.float64),
    }


def current_fold_scores(
    score_cache: Path,
    signals: dict[str, np.ndarray],
    held_fold: int,
) -> np.ndarray:
    with np.load(score_cache, allow_pickle=False) as scores:
        rows = scores["inner_folds"] == held_fold
        if not np.array_equal(scores["inner_labels"][rows], signals["labels"]):
            raise ValueError("Current-score labels do not align with held fold")
        if not np.array_equal(scores["inner_sensors"][rows], signals["sensors"]):
            raise ValueError("Current-score sensors do not align with held fold")
        if not np.array_equal(scores["inner_groups"][rows], signals["groups"]):
            raise ValueError("Current-score groups do not align with held fold")
        return scores["inner_new"][rows].astype(np.float64)


def evaluate_candidates(
    signals: dict[str, np.ndarray],
    current: np.ndarray,
    protocol: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    labels = signals["labels"]
    sensors = signals["sensors"]
    groups = signals["groups"]
    current_metrics = metric_summary(labels, current, sensors)
    candidates: list[dict[str, Any]] = []
    for signal_name in protocol["search"]["signals"]:
        signal = signals[signal_name]
        for weight in protocol["search"]["weights"]:
            candidate_score = (1.0 - float(weight)) * current + float(weight) * signal
            metrics = metric_summary(labels, candidate_score, sensors)
            versus = comparison(metrics, current_metrics)
            interval = ap_group_bootstrap(
                labels,
                current,
                candidate_score,
                groups,
                replicates=int(protocol["bootstrap"]["replicates"]),
                seed=int(protocol["bootstrap"]["ap_seed"]),
            )
            delta = versus["delta"]
            sensor_delta = delta["sensor_average_precision"]
            passed = bool(
                delta["average_precision"]
                >= float(protocol["gates"]["average_precision_delta_minimum"])
                and delta["recall_at_fpr_0_0713"]
                >= float(protocol["gates"]["matched_fpr_recall_delta_minimum"])
                and min(sensor_delta.values())
                >= float(protocol["gates"]["each_sensor_ap_delta_minimum"])
                and interval["lower"] > 0.0
            )
            candidates.append(
                {
                    "signal": signal_name,
                    "weight": float(weight),
                    "metrics": metrics,
                    "versus_current": versus,
                    "bootstrap": interval,
                    "passed": passed,
                    "rank": [
                        int(passed),
                        interval["lower"],
                        delta["average_precision"],
                        delta["recall_at_fpr_0_0713"],
                        min(sensor_delta.values()),
                        -float(weight),
                    ],
                }
            )
    candidates.sort(key=lambda row: row["rank"], reverse=True)
    identity = {
        "rows": int(len(labels)),
        "sample_id_sha256": hashlib.sha256(
            "\n".join(signals["sample_ids"].tolist()).encode()
        ).hexdigest(),
        "released_metrics": metric_summary(labels, signals["released"], sensors),
        "current_metrics": current_metrics,
        "signal_metrics": {
            name: metric_summary(labels, signals[name], sensors)
            for name in protocol["search"]["signals"]
        },
    }
    return candidates, identity


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL.as_posix())
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    protocol_path = (ROOT / args.protocol).resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if sha256(Path(__file__).resolve()) != protocol["trainer"]["sha256"]:
        raise ValueError("Frozen instance-ensemble trainer hash mismatch")
    for dependency in protocol["code_dependencies"]:
        path = (ROOT / dependency["path"]).resolve()
        if sha256(path) != dependency["sha256"]:
            raise ValueError(f"Frozen dependency mismatch: {dependency['path']}")
    paths: dict[str, Path] = {}
    for name, contract in protocol["inputs"].items():
        path = (ROOT / contract["path"]).resolve()
        if path.is_file() and sha256(path) != contract["sha256"]:
            raise ValueError(f"Frozen input mismatch: {name}")
        paths[name] = path
    if not paths["metadata_root"].is_dir():
        raise ValueError("MARS metadata root is unavailable")
    verify_acquisition_receipt(paths["acquisition_receipt"], sha256(paths["manifest"]))
    fold_protocol = json.loads(paths["fold_protocol"].read_text(encoding="utf-8"))
    group_to_fold = {
        str(row["group_id"]): int(row["fold"]) for row in fold_protocol["assignments"]
    }
    records = list(iter_development_manifest(paths["manifest"]))
    fit_folds = set(map(int, protocol["folds"]["fit"]))
    held_fold = int(protocol["folds"]["held"])
    fit_records = [
        row for row in records if group_to_fold[str(row["group_id"])] in fit_folds
    ]
    held_records = [
        row for row in records if group_to_fold[str(row["group_id"])] == held_fold
    ]
    spec = protocol["training"]
    if args.smoke:
        selected = []
        counts: Counter[tuple[str, str]] = Counter()
        for row in fit_records:
            key = (str(row["label_state"]), str(row["sensor_family"]))
            if counts[key] < 4:
                selected.append(row)
                counts[key] += 1
            if len(counts) == 4 and min(counts.values()) >= 4:
                break
        fit_records = selected
        held_records = selected

    weights, request_mass = balanced_request_weights(fit_records)
    flags = offshore_flags(paths["metadata_csv"], {str(row["sample_id"]) for row in fit_records})
    dataset = InstanceTargetDataset(
        paths["metadata_root"],
        fit_records,
        flags,
        lut_path=paths["lut"],
        augment=True,
        simulation_fraction=float(spec["simulation_fraction"]),
        crop_size=int(spec["crop_size"]),
        seed=int(spec["seed"]),
    )
    samples = 64 if args.smoke else int(spec["samples_per_epoch"])
    workers = 2 if args.smoke else int(spec["loader_workers"])
    batch_size = 4 if args.smoke else int(spec["batch_size"])
    sampler = WeightedRandomSampler(
        weights,
        num_samples=samples,
        replacement=True,
        generator=torch.Generator().manual_seed(int(spec["seed"])),
    )
    options = {
        "batch_size": batch_size,
        "num_workers": workers,
        "pin_memory": True,
        "persistent_workers": workers > 0,
    }
    train_loader = DataLoader(dataset, sampler=sampler, **options)
    evaluation_loader = DataLoader(
        MarsPaperDataset(
            paths["metadata_root"], held_records, augment=False, seed=int(spec["seed"])
        ),
        shuffle=False,
        **options,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Instance-scene ensemble pilot requires CUDA")
    model = InstanceGuidedTeacherAdapter().to(device)
    model.load_released_checkpoint(released_state(paths["released_checkpoint"]))
    with torch.no_grad():
        first = move_batch(next(iter(evaluation_loader)), device)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            initial = model(first["inputs"], first["observable"], first["sensor_index"])
        identity_max = float(initial["correction_logits"].abs().max())
    if identity_max != 0.0:
        raise ValueError(f"Adapter initialization is not exact identity: {identity_max}")

    history = train_endpoint(
        model, train_loader, spec, device, 1 if args.smoke else int(spec["epochs"])
    )
    if args.smoke:
        finite = all(torch.isfinite(value).all() for value in model.trainable_state().values())
        print(
            json.dumps(
                {
                    "ok": finite,
                    "identity_max_abs": identity_max,
                    "request_mass": request_mass,
                    "trainable_parameters": model.trainable_parameter_count(),
                    "history": history,
                }
            )
        )
        return 0 if finite else 1

    signals = collect_instance_signals(
        model,
        evaluation_loader,
        device,
        float(protocol["search"]["correction_strength"]),
    )
    current = current_fold_scores(paths["score_cache"], signals, held_fold)
    candidates, identity = evaluate_candidates(signals, current, protocol)
    mask_evidence = frozen_mask_evidence(
        paths["pixel_cache"],
        signals["labels"],
        signals["sensors"],
        signals["groups"],
        current,
        protocol,
    )
    selected = candidates[0]
    passed = bool(selected["passed"] and mask_evidence["passed"])
    artifact = None
    if passed:
        artifact_path = (ROOT / protocol["outputs"]["artifact"]).resolve()
        if artifact_path.exists():
            raise FileExistsError(f"Refusing to overwrite existing artifact: {artifact_path}")
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "schema_version": 1,
                "adapter_state": model.trainable_state(),
                "selected_signal": selected["signal"],
                "selected_weight": selected["weight"],
                "correction_strength": protocol["search"]["correction_strength"],
                "mask_rule": protocol["mask_rule"],
                "protocol_sha256": sha256(protocol_path),
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
        "scope": "conservative two-output instance-signal scene ensemble pilot",
        "all_promotion_gates_pass": passed,
        "decision": (
            "Authorize preregistered multi-seed full development cross-fit."
            if passed
            else "Reject instance-scene ensemble before external scoring."
        ),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "provenance": {
            "git_commit": commit,
            "protocol_sha256": sha256(protocol_path),
            "trainer_sha256": sha256(Path(__file__).resolve()),
            "device": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
        "fit_folds": sorted(fit_folds),
        "held_fold": held_fold,
        "fit_rows": len(fit_records),
        "held_rows": len(held_records),
        "request_mass": request_mass,
        "trainable_parameters": model.trainable_parameter_count(),
        "identity_max_abs": identity_max,
        "history": history,
        "score_identity": identity,
        "mask_evidence": mask_evidence,
        "candidates": candidates,
        "selected": selected,
        "artifact": artifact,
        "external_inputs_accessed": False,
    }
    output = (ROOT / protocol["outputs"]["json"]).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    delta = selected["versus_current"]["delta"]
    print(
        json.dumps(
            {
                "ok": passed,
                "signal": selected["signal"],
                "weight": selected["weight"],
                "ap_delta": delta["average_precision"],
                "ap_lower": selected["bootstrap"]["lower"],
                "recall_delta": delta["recall_at_fpr_0_0713"],
                "minimum_sensor_ap_delta": min(
                    delta["sensor_average_precision"].values()
                ),
                "mask_iou_delta": mask_evidence["delta"],
                "mask_iou_lower": mask_evidence["paired_site_bootstrap"]["lower"],
            },
            indent=2,
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

