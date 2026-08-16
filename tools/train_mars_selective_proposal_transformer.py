#!/usr/bin/env python3
"""Train the frozen selective proposal-verifier transformer on MARS folds 3/4."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.special import expit
from sklearn.metrics import average_precision_score
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "EarthRemoteSensingRapidResponse"
for path in (ROOT / "tools", MODEL_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from acquire_mars_metadata import sha256  # noqa: E402
from analyze_mars_recall_anchor import align_feature_rows  # noqa: E402
from mars_selective_proposal_transformer import SelectiveProposalTransformer  # noqa: E402
from train_mars_oof_scene_ensemble_v2 import ap_group_bootstrap  # noqa: E402
from train_mars_scene_ranker import comparison, metric_summary  # noqa: E402


DEFAULT_PROTOCOL = Path("configs/mars_selective_proposal_transformer_protocol.json")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def balanced_cell_weights(
    groups: np.ndarray,
    labels: np.ndarray,
    sensors: np.ndarray,
) -> np.ndarray:
    keys = [
        f"{group}|{int(label)}|{int(sensor)}"
        for group, label, sensor in zip(groups, labels, sensors)
    ]
    counts = Counter(keys)
    weights = np.asarray([1.0 / counts[key] for key in keys], dtype=np.float64)
    for label in (0, 1):
        selection = labels == label
        if not np.any(selection):
            raise ValueError("Proposal-verifier fitting population lacks a class")
        weights[selection] *= 0.5 / weights[selection].sum()
    return weights / weights.mean()


def channel_statistics(
    images: np.ndarray,
    indices: np.ndarray,
    batch_size: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    sums = np.zeros(images.shape[1], dtype=np.float64)
    squares = np.zeros(images.shape[1], dtype=np.float64)
    count = 0
    for start in range(0, indices.size, batch_size):
        batch = np.asarray(images[indices[start : start + batch_size]], dtype=np.float32)
        sums += batch.sum(axis=(0, 2, 3), dtype=np.float64)
        squares += np.square(batch, dtype=np.float64).sum(axis=(0, 2, 3))
        count += batch.shape[0] * batch.shape[2] * batch.shape[3]
    mean = sums / count
    variance = np.maximum(squares / count - np.square(mean), 1e-6)
    return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)


def augment(values: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    if torch.rand((), generator=generator).item() < 0.5:
        values = values.flip(-1)
    if torch.rand((), generator=generator).item() < 0.5:
        values = values.flip(-2)
    turns = int(torch.randint(0, 4, (), generator=generator).item())
    return torch.rot90(values, turns, dims=(-2, -1))


def build_model(spec: dict[str, Any]) -> SelectiveProposalTransformer:
    architecture = spec["architecture"]
    return SelectiveProposalTransformer(
        input_channels=9,
        image_size=64,
        patch_size=int(architecture["patch_size"]),
        dimension=int(architecture["dimension"]),
        layers=int(architecture["layers"]),
        heads=int(architecture["heads"]),
        dropout=float(architecture["dropout"]),
    )


def train_endpoint(
    protocol: dict[str, Any],
    images: np.ndarray,
    indices: np.ndarray,
    labels: np.ndarray,
    sensors: np.ndarray,
    groups: np.ndarray,
    *,
    seed: int,
    device: torch.device,
    epochs: int,
    maximum_rows: int | None = None,
) -> dict[str, Any]:
    seed_everything(seed)
    if maximum_rows is not None and indices.size > maximum_rows:
        rng = np.random.default_rng(seed)
        selected = []
        for label in (0, 1):
            local = indices[labels[indices] == label]
            take = min(local.size, maximum_rows // 2)
            selected.extend(rng.choice(local, size=take, replace=False).tolist())
        indices = np.asarray(sorted(selected), dtype=np.int64)
    local_labels = labels[indices]
    local_sensors = sensors[indices]
    local_groups = groups[indices]
    weights = balanced_cell_weights(local_groups, local_labels, local_sensors)
    mean, standard_deviation = channel_statistics(images, indices)
    mean_tensor = torch.from_numpy(mean)[None, :, None, None].to(device)
    std_tensor = torch.from_numpy(standard_deviation)[None, :, None, None].to(device)
    model = build_model(protocol).to(device)
    training = protocol["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    batch_size = int(training["batch_size"])
    history = []
    for epoch in range(1, epochs + 1):
        order = torch.randperm(indices.size, generator=generator).numpy()
        sums = {"loss": 0.0, "bce": 0.0, "pair": 0.0}
        batches = 0
        model.train()
        for start in range(0, indices.size, batch_size):
            rows = order[start : start + batch_size]
            global_rows = indices[rows]
            raw = torch.from_numpy(
                np.asarray(images[global_rows], dtype=np.float32)
            ).to(device)
            raw = augment(raw, generator)
            target = torch.from_numpy(local_labels[rows].astype(np.float32)).to(device)
            sensor = torch.from_numpy(local_sensors[rows].astype(np.int64)).to(device)
            row_weight = torch.from_numpy(weights[rows].astype(np.float32)).to(device)
            normalized = (raw - mean_tensor) / std_tensor
            logits = model(
                normalized,
                sensor,
                raw[:, 1:2].clamp(0.0, 1.0),
                raw[:, 8:9].clamp(0.0, 1.0),
            )
            bce_rows = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
            bce = (bce_rows * row_weight).sum() / row_weight.sum().clamp_min(1e-6)
            positive = logits[target > 0.5]
            negative = logits[target < 0.5]
            pair = (
                F.softplus(0.5 - positive[:, None] + negative[None, :]).mean()
                if positive.numel() and negative.numel()
                else logits.sum() * 0.0
            )
            loss = bce + float(training["pairwise_loss_weight"]) * pair
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(training["gradient_clip"])
            )
            optimizer.step()
            batches += 1
            sums["loss"] += float(loss.detach())
            sums["bce"] += float(bce.detach())
            sums["pair"] += float(pair.detach())
        scheduler.step()
        record = {
            "epoch": epoch,
            **{name: value / batches for name, value in sums.items()},
        }
        history.append(record)
        print(json.dumps({"seed": seed, **record}), flush=True)
    return {
        "state_dict": {
            name: value.detach().cpu() for name, value in model.state_dict().items()
        },
        "channel_mean": mean,
        "channel_standard_deviation": standard_deviation,
        "history": history,
        "training_rows": int(indices.size),
        "training_positives": int(local_labels.sum()),
        "parameter_count": int(sum(value.numel() for value in model.parameters())),
    }


@torch.no_grad()
def predict_endpoint(
    protocol: dict[str, Any],
    endpoint: dict[str, Any],
    images: np.ndarray,
    indices: np.ndarray,
    sensors: np.ndarray,
    *,
    device: torch.device,
    maximum_rows: int | None = None,
) -> np.ndarray:
    if maximum_rows is not None:
        indices = indices[:maximum_rows]
    model = build_model(protocol).to(device)
    model.load_state_dict(endpoint["state_dict"])
    model.eval()
    mean = torch.from_numpy(endpoint["channel_mean"])[None, :, None, None].to(device)
    standard_deviation = torch.from_numpy(endpoint["channel_standard_deviation"])[
        None, :, None, None
    ].to(device)
    parts = []
    batch_size = 256
    for start in range(0, indices.size, batch_size):
        rows = indices[start : start + batch_size]
        raw = torch.from_numpy(np.asarray(images[rows], dtype=np.float32)).to(device)
        normalized = (raw - mean) / standard_deviation
        logits = model(
            normalized,
            torch.from_numpy(sensors[rows].astype(np.int64)).to(device),
            raw[:, 1:2].clamp(0.0, 1.0),
            raw[:, 8:9].clamp(0.0, 1.0),
        )
        parts.append(logits.cpu().numpy())
    return np.concatenate(parts) if parts else np.empty(0, dtype=np.float32)


def candidate_scores(
    champion: np.ndarray,
    released: np.ndarray,
    verifier: np.ndarray,
    weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    route = (released > 0.5) & (champion < 0.25)
    candidate = champion.copy()
    candidate[route] = np.maximum(candidate[route], weight * verifier[route])
    if np.any(candidate + 1e-15 < champion):
        raise AssertionError("Selective verifier suppressed a champion score")
    return candidate, route


def evaluate_candidate(
    labels: np.ndarray,
    sensors: np.ndarray,
    groups: np.ndarray,
    folds: np.ndarray,
    champion: np.ndarray,
    candidate: np.ndarray,
    weight: float,
    route: np.ndarray,
    gates: dict[str, Any],
) -> dict[str, Any]:
    pooled = comparison(
        metric_summary(labels, candidate, sensors),
        metric_summary(labels, champion, sensors),
    )
    per_fold = {}
    for fold in (3, 4):
        selection = folds == fold
        per_fold[str(fold)] = comparison(
            metric_summary(labels[selection], candidate[selection], sensors[selection]),
            metric_summary(labels[selection], champion[selection], sensors[selection]),
        )
    delta = pooled["delta"]
    fold_ap = [value["delta"]["average_precision"] for value in per_fold.values()]
    fold_recall = [
        value["delta"]["recall_at_fpr_0_0713"] for value in per_fold.values()
    ]
    checks = {
        "minimum_pooled_ap_delta": delta["average_precision"]
        >= float(gates["minimum_pooled_ap_delta"]),
        "strictly_positive_pooled_recall": delta["recall_at_fpr_0_0713"] > 0.0,
        "each_fold_ap_nonnegative": min(fold_ap)
        >= float(gates["minimum_each_fold_ap_delta"]),
        "each_fold_recall_nonnegative": min(fold_recall)
        >= float(gates["minimum_each_fold_recall_delta"]),
        "each_sensor_ap_nonnegative": min(delta["sensor_average_precision"].values())
        >= float(gates["minimum_each_sensor_ap_delta"]),
    }
    raised = candidate > champion + 1e-15
    return {
        "rescue_weight": weight,
        "route_rows": int(route.sum()),
        "raised_rows": int(raised.sum()),
        "raised_positives": int(labels[raised].sum()),
        "raised_negatives": int(raised.sum() - labels[raised].sum()),
        "pooled": pooled,
        "per_fold": per_fold,
        "point_checks": checks,
        "all_point_gates_pass": all(checks.values()),
        "rank": [
            int(all(checks.values())),
            min(fold_recall),
            min(fold_ap),
            delta["average_precision"],
            -weight,
        ],
        "groups": int(np.unique(groups).size),
    }


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    selected = report["selected"]
    delta = selected["pooled"]["delta"]
    interval = selected["paired_group_bootstrap_ap_delta"]
    lines = [
        "# Selective proposal-verifier transformer: folds 3/4",
        "",
        f"- Rescue weight: {selected['rescue_weight']:.6f}",
        f"- AP delta: {delta['average_precision']:+.6f}",
        f"- Matched-FPR recall delta: {delta['recall_at_fpr_0_0713']:+.6f}",
        f"- Paired 25 km-group AP interval: [{interval['lower']:+.6f}, {interval['upper']:+.6f}]",
        f"- Raised rows: {selected['raised_rows']} ({selected['raised_positives']} positives, {selected['raised_negatives']} negatives)",
        "",
        "| Fold | AP delta | Recall delta |",
        "|---|---:|---:|",
    ]
    for fold, value in selected["per_fold"].items():
        local = value["delta"]
        lines.append(
            f"| {fold} | {local['average_precision']:+.6f} | "
            f"{local['recall_at_fpr_0_0713']:+.6f} |"
        )
    lines.extend(["", f"Decision: **{report['decision']}**", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL.as_posix())
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    protocol_path = (ROOT / args.protocol).resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    paths = {
        name: (ROOT / value["path"]).resolve()
        for name, value in protocol["inputs"].items()
    }
    for name, value in protocol["inputs"].items():
        if sha256(paths[name]) != value["sha256"]:
            raise ValueError(f"Frozen input hash mismatch: {name}")
    prior = json.loads(paths["deterministic_rescue_result"].read_text(encoding="utf-8"))
    if prior.get("decision") != "reject_deterministic_dual_teacher_rescue":
        raise ValueError("Selective verifier requires the frozen deterministic rejection")

    images = np.load(paths["spatial_images"], mmap_mode="r", allow_pickle=False)
    with np.load(paths["spatial_metadata"], allow_pickle=False) as bundle:
        spatial_ids = bundle["sample_ids"].astype(str)
        spatial_folds = bundle["folds"].astype(np.uint8)
        spatial_labels = bundle["labels"].astype(np.uint8)
        spatial_sensors = bundle["sensors"].astype(np.uint8)
        spatial_groups = bundle["groups"].astype(str)
    with np.load(paths["champion_scores"], allow_pickle=False) as bundle:
        sample_ids = bundle["sample_ids"].astype(str)
        labels = bundle["labels"].astype(np.uint8)
        sensors = bundle["sensors"].astype(np.uint8)
        groups = bundle["groups"].astype(str)
        folds = bundle["folds"].astype(np.uint8)
        champion = bundle["champion_scores"].astype(np.float64)
    spatial_indices = align_feature_rows(sample_ids, spatial_ids, spatial_folds)
    if not (
        np.array_equal(labels, spatial_labels[spatial_indices])
        and np.array_equal(sensors, spatial_sensors[spatial_indices])
        and np.array_equal(groups, spatial_groups[spatial_indices])
    ):
        raise ValueError("Spatial cache metadata does not align with champion rows")
    with np.load(paths["scene_features"], allow_pickle=False) as bundle:
        feature_names = bundle["feature_names"].astype(str)
        feature_indices = align_feature_rows(sample_ids, bundle["sample_ids"], bundle["folds"])
        features = bundle["features"][feature_indices]
    released_column = int(np.flatnonzero(feature_names == "released_connected_score")[0])
    released = features[:, released_column].astype(np.float64)
    eligible = released > 0.5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.smoke:
        held_fold = 3
        fit = (folds == 4) & eligible
        held = (folds == held_fold) & eligible
        fit_indices = spatial_indices[fit]
        held_indices = spatial_indices[held]
        endpoint = train_endpoint(
            protocol,
            images,
            fit_indices,
            spatial_labels,
            spatial_sensors,
            spatial_groups,
            seed=int(protocol["training"]["seeds"][0]),
            device=device,
            epochs=1,
            maximum_rows=128,
        )
        logits = predict_endpoint(
            protocol,
            endpoint,
            images,
            held_indices,
            spatial_sensors,
            device=device,
            maximum_rows=128,
        )
        smoke = {
            "schema_version": 1,
            "status": "passed",
            "device": str(device),
            "training_rows": endpoint["training_rows"],
            "training_positives": endpoint["training_positives"],
            "parameter_count": endpoint["parameter_count"],
            "prediction_rows": int(logits.size),
            "finite_predictions": bool(np.isfinite(logits).all()),
            "protocol_sha256": sha256(protocol_path),
            "script_sha256": sha256(Path(__file__).resolve()),
        }
        if not smoke["finite_predictions"] or not logits.size:
            raise RuntimeError("Selective-proposal smoke failed")
        write_json((ROOT / protocol["execution"]["smoke_output"]).resolve(), smoke)
        print(json.dumps(smoke, sort_keys=True))
        return 0

    smoke_path = (ROOT / protocol["execution"]["smoke_output"]).resolve()
    if not smoke_path.is_file() or json.loads(smoke_path.read_text(encoding="utf-8")).get(
        "status"
    ) != "passed":
        raise ValueError("Passing selective-proposal smoke report is required")

    seeds = [int(value) for value in protocol["training"]["seeds"]]
    verifier_logits = np.full((len(seeds), labels.size), np.nan, dtype=np.float64)
    endpoints = []
    audits = []
    epochs = int(protocol["training"]["epochs"])
    for held_fold in (3, 4):
        training_fold = 4 if held_fold == 3 else 3
        fit = (folds == training_fold) & eligible
        held = (folds == held_fold) & eligible
        fit_indices = spatial_indices[fit]
        held_indices = spatial_indices[held]
        for seed_index, seed in enumerate(seeds):
            endpoint = train_endpoint(
                protocol,
                images,
                fit_indices,
                spatial_labels,
                spatial_sensors,
                spatial_groups,
                seed=seed,
                device=device,
                epochs=epochs,
            )
            local_logits = predict_endpoint(
                protocol,
                endpoint,
                images,
                held_indices,
                spatial_sensors,
                device=device,
            )
            verifier_logits[seed_index, held] = local_logits
            local_labels = labels[held]
            audits.append(
                {
                    "held_fold": held_fold,
                    "training_fold": training_fold,
                    "seed": seed,
                    "training_rows": endpoint["training_rows"],
                    "training_positives": endpoint["training_positives"],
                    "held_rows": int(held.sum()),
                    "held_positives": int(local_labels.sum()),
                    "held_verifier_ap": float(
                        average_precision_score(local_labels, expit(local_logits))
                    ),
                    "final_loss": endpoint["history"][-1]["loss"],
                }
            )
            endpoints.append(
                {
                    "held_fold": held_fold,
                    "training_fold": training_fold,
                    "seed": seed,
                    **endpoint,
                }
            )
    if not np.isfinite(verifier_logits[:, eligible]).all():
        raise RuntimeError("Selective verifier left non-finite eligible predictions")
    verifier = np.zeros(labels.size, dtype=np.float64)
    verifier[eligible] = expit(verifier_logits[:, eligible].mean(axis=0))
    candidates = []
    candidate_vectors = {}
    for weight in protocol["architecture"]["rescue_weights"]:
        candidate, route = candidate_scores(champion, released, verifier, float(weight))
        candidate_vectors[float(weight)] = candidate
        candidates.append(
            evaluate_candidate(
                labels,
                sensors,
                groups,
                folds,
                champion,
                candidate,
                float(weight),
                route,
                protocol["promotion_gates"],
            )
        )
    if len(candidates) != int(protocol["architecture"]["candidate_count"]):
        raise AssertionError("Selective-proposal candidate count changed")
    selected = max(candidates, key=lambda value: tuple(value["rank"]))
    selected_scores = candidate_vectors[selected["rescue_weight"]]
    selected["paired_group_bootstrap_ap_delta"] = ap_group_bootstrap(
        labels,
        champion,
        selected_scores,
        groups,
        replicates=int(protocol["promotion_gates"]["paired_group_bootstrap_replicates"]),
        seed=int(protocol["promotion_gates"]["paired_group_bootstrap_seed"]),
    )
    passed = bool(
        selected["all_point_gates_pass"]
        and selected["paired_group_bootstrap_ap_delta"]["lower"] > 0.0
    )
    selected["all_promotion_gates_pass"] = passed
    state_path = (ROOT / protocol["execution"]["full_state_output"]).resolve()
    score_path = (ROOT / protocol["execution"]["full_score_output"]).resolve()
    artifact = None
    if passed:
        state_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "schema_version": 1,
                "protocol_sha256": sha256(protocol_path),
                "endpoints": endpoints,
                "selected_rescue_weight": selected["rescue_weight"],
            },
            state_path,
        )
        np.savez_compressed(
            score_path,
            sample_ids=sample_ids,
            labels=labels,
            sensors=sensors,
            groups=groups,
            folds=folds,
            champion_scores=champion,
            verifier_scores=verifier,
            candidate_scores=selected_scores,
        )
        artifact = {
            "state_path": state_path.relative_to(ROOT).as_posix(),
            "state_sha256": sha256(state_path),
            "score_path": score_path.relative_to(ROOT).as_posix(),
            "score_sha256": sha256(score_path),
            "tracked": False,
        }
    report = {
        "schema_version": 1,
        "scope": protocol["scope"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "protocol": {
            "path": protocol_path.relative_to(ROOT).as_posix(),
            "sha256": sha256(protocol_path),
        },
        "cohort": {
            "rows": int(labels.size),
            "positives": int(labels.sum()),
            "negatives": int((labels == 0).sum()),
            "groups": int(np.unique(groups).size),
            "released_positive_training_population": int(eligible.sum()),
        },
        "endpoint_audits": audits,
        "candidates": candidates,
        "selected": selected,
        "artifact": artifact,
        "all_promotion_gates_pass": passed,
        "decision": (
            "promote_selective_proposal_transformer_for_separate_posttest_protocol"
            if passed
            else "reject_selective_proposal_transformer"
        ),
        "holdout_boundary": protocol["holdout_boundary"],
    }
    output_json = (ROOT / protocol["outputs"]["json"]).resolve()
    output_markdown = (ROOT / protocol["outputs"]["markdown"]).resolve()
    write_json(output_json, report)
    write_markdown(output_markdown, report)
    print(
        json.dumps(
            {
                "decision": report["decision"],
                "weight": selected["rescue_weight"],
                "ap_delta": selected["pooled"]["delta"]["average_precision"],
                "recall_delta": selected["pooled"]["delta"]["recall_at_fpr_0_0713"],
                "ap_lower": selected["paired_group_bootstrap_ap_delta"]["lower"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
