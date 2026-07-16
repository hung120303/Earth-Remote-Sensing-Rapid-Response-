#!/usr/bin/env python3
"""Train a compact worst-domain and pairwise-AUC Prithvi scene ranker."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "EarthRemoteSensingRapidResponse"
for path in (MODEL_ROOT, ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from acquire_mars_metadata import sha256  # noqa: E402
from train_mars_adaptive_prithvi_probe import load_features  # noqa: E402
from train_mars_crossfold_bagged_scene_head import load_development  # noqa: E402
from train_mars_oof_scene_ensemble_v2 import ap_group_bootstrap  # noqa: E402
from train_mars_scene_ranker import blend_scores, comparison, metric_summary  # noqa: E402
from train_mars_temporal_spatial_ensemble import align_spatial_scores  # noqa: E402


DEFAULT_PROTOCOL = Path("configs/mars_robust_prithvi_ranker_protocol.json")


class RobustRanker(nn.Module):
    def __init__(self, width: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.hidden = nn.Linear(width, hidden)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden, 1)
        self.skip = nn.Linear(width, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        normalized = self.norm(features)
        nonlinear = self.output(self.dropout(F.gelu(self.hidden(normalized))))
        return (nonlinear + self.skip(normalized)).squeeze(1)


def independent_normalize(
    source: np.ndarray, target: np.ndarray, epsilon: float = 1e-4
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    source_mean = source.mean(0)
    source_scale = np.maximum(source.std(0), epsilon)
    target_mean = target.mean(0)
    target_scale = np.maximum(target.std(0), epsilon)
    return (
        ((source - source_mean) / source_scale).astype(np.float32),
        ((target - target_mean) / target_scale).astype(np.float32),
        source_mean.astype(np.float32),
        source_scale.astype(np.float32),
    )


def site_weights(groups: np.ndarray, labels: np.ndarray) -> np.ndarray:
    unique, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    del unique
    weights = 1.0 / counts[inverse].astype(np.float64)
    class_counts = np.bincount(labels.astype(np.int64), minlength=2).astype(np.float64)
    positive_weight = float(np.sqrt(class_counts[0] / max(class_counts[1], 1.0)))
    weights *= np.where(labels == 1, positive_weight, 1.0)
    return (weights / weights.mean()).astype(np.float32)


def domain_ids(folds: np.ndarray, sensors: np.ndarray, labels: np.ndarray) -> np.ndarray:
    triples = np.column_stack((folds, sensors, labels)).astype(np.int64)
    _, inverse = np.unique(triples, axis=0, return_inverse=True)
    return inverse.astype(np.int64)


def fit_ranker(
    source: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    folds: np.ndarray,
    sensors: np.ndarray,
    spec: dict[str, Any],
    seed: int,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], list[dict[str, float]]]:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = RobustRanker(source.shape[1], int(spec["hidden"]), float(spec["dropout"])).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(spec["learning_rate"]),
        weight_decay=float(spec["weight_decay"]),
    )
    x = torch.from_numpy(source).to(device)
    y = torch.from_numpy(labels.astype(np.float32)).to(device)
    weights = torch.from_numpy(site_weights(groups, labels)).to(device)
    domains = torch.from_numpy(domain_ids(folds, sensors, labels)).to(device)
    domain_values = torch.unique(domains).tolist()
    positive_indices = torch.nonzero(y > 0.5, as_tuple=False).flatten()
    negative_indices = torch.nonzero(y < 0.5, as_tuple=False).flatten()
    history: list[dict[str, float]] = []
    generator = torch.Generator(device=device)
    generator.manual_seed(seed + 91)
    for epoch in range(int(spec["epochs"])):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        row_loss = F.binary_cross_entropy_with_logits(logits, y, reduction="none") * weights
        local_losses = torch.stack([row_loss[domains == value].mean() for value in domain_values])
        temperature = float(spec["group_temperature"])
        robust_bce = temperature * torch.logsumexp(local_losses / temperature, dim=0)
        hard_count = min(int(spec["hard_negative_pool"]), int(negative_indices.numel()))
        hard_local = torch.topk(logits[negative_indices].detach(), hard_count).indices
        hard_negative_indices = negative_indices[hard_local]
        pair_count = int(spec["pair_count"])
        positive_draw = torch.randint(
            positive_indices.numel(), (pair_count,), generator=generator, device=device
        )
        negative_draw = torch.randint(
            hard_negative_indices.numel(), (pair_count,), generator=generator, device=device
        )
        positive_logits = logits[positive_indices[positive_draw]]
        negative_logits = logits[hard_negative_indices[negative_draw]]
        pair_loss = F.softplus(
            float(spec["pair_margin"]) - positive_logits + negative_logits
        ).mean()
        loss = robust_bce + float(spec["pair_weight"]) * pair_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(spec["gradient_clip"]))
        optimizer.step()
        history.append({
            "epoch": epoch + 1,
            "loss": float(loss.detach().cpu()),
            "robust_bce": float(robust_bce.detach().cpu()),
            "pair_loss": float(pair_loss.detach().cpu()),
            "worst_domain_bce": float(local_losses.max().detach().cpu()),
        })
    state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    return state, history


@torch.no_grad()
def predict(
    state: dict[str, torch.Tensor], features: np.ndarray, spec: dict[str, Any], device: torch.device
) -> np.ndarray:
    model = RobustRanker(features.shape[1], int(spec["hidden"]), float(spec["dropout"])).to(device)
    model.load_state_dict(state)
    model.eval()
    logits = model(torch.from_numpy(features).to(device)).float().cpu().numpy()
    return (1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))).astype(np.float64)


def subset(values: dict[str, Any], rows: np.ndarray) -> dict[str, Any]:
    return {
        key: value[rows] if isinstance(value, np.ndarray) and value.shape[:1] == rows.shape else value
        for key, value in values.items()
    }


def evaluate(
    values: dict[str, Any], raw: np.ndarray, folds: list[int], blend: float
) -> dict[str, Any]:
    rows = np.isin(values["folds"], folds)
    scores = blend_scores(values["current"][rows], raw[rows], blend)
    labels = values["labels"][rows]
    sensors = values["sensors"][rows]
    current = metric_summary(labels, values["current"][rows], sensors)
    candidate = metric_summary(labels, scores, sensors)
    versus = comparison(candidate, current)
    per_fold: dict[str, Any] = {}
    for fold in folds:
        local = values["folds"] == fold
        local_scores = blend_scores(values["current"][local], raw[local], blend)
        per_fold[str(fold)] = comparison(
            metric_summary(values["labels"][local], local_scores, values["sensors"][local]),
            metric_summary(values["labels"][local], values["current"][local], values["sensors"][local]),
        )
    fold_ap = [value["delta"]["average_precision"] for value in per_fold.values()]
    fold_recall = [value["delta"]["recall_at_fpr_0_0713"] for value in per_fold.values()]
    stable = (
        versus["delta"]["average_precision"] > 0.0
        and versus["delta"]["recall_at_fpr_0_0713"] >= 0.0
        and min(fold_ap) > 0.0
        and min(fold_recall) >= -0.002
        and min(versus["delta"]["sensor_average_precision"].values()) >= -0.0025
    )
    return {
        "blend_weight": blend,
        "metrics": candidate,
        "versus_current": versus,
        "per_fold": per_fold,
        "stable": bool(stable),
        "rank": [
            int(stable),
            min(fold_ap),
            versus["delta"]["average_precision"],
            versus["delta"]["recall_at_fpr_0_0713"],
            -blend,
        ],
    }


def build_features(
    values: dict[str, Any], prithvi_path: Path, spatial_path: Path
) -> tuple[np.ndarray, list[str]]:
    prithvi = load_features(prithvi_path, values, "cls").astype(np.float32)
    spatial = align_spatial_scores(values, spatial_path).astype(np.float32)
    clipped_spatial = np.clip(spatial, 1e-6, 1.0 - 1e-6)
    spatial_logit = np.log(clipped_spatial) - np.log1p(-clipped_spatial)
    features = np.concatenate(
        (prithvi, values["features"].astype(np.float32), spatial_logit[:, None]), axis=1
    )
    names = [
        *[f"prithvi_cls_{index}" for index in range(prithvi.shape[1])],
        *list(map(str, values["augmented_feature_names"])),
        "site_relative_spatial_logit",
    ]
    if features.shape[1] != len(names) or not np.isfinite(features).all():
        raise ValueError("Robust Prithvi feature matrix is invalid")
    return features, names


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    selected = report["selected"]
    confirmation = report["reused_holdout_audit"]
    lines = [
        "# Robust Prithvi scene ranker",
        "",
        f"- Selected pair weight: **{selected['spec']['pair_weight']:.2f}**; current blend: **{selected['blend_weight']:.2f}**.",
        f"- Selection AP delta: **{selected['versus_current']['delta']['average_precision']:+.6f}**.",
        f"- Selection paired-site AP interval: **[{selected['bootstrap']['lower']:+.6f}, {selected['bootstrap']['upper']:+.6f}]**.",
        f"- Reused holdout AP delta: **{confirmation['versus_current']['delta']['average_precision']:+.6f}**.",
        f"- Reused holdout paired-site AP interval: **[{confirmation['bootstrap']['lower']:+.6f}, {confirmation['bootstrap']['upper']:+.6f}]**.",
        "",
        report["decision"],
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL.as_posix())
    args = parser.parse_args()
    protocol_path = (ROOT / args.protocol).resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if sha256(Path(__file__).resolve()) != protocol["trainer"]["sha256"]:
        raise ValueError("Robust Prithvi trainer hash mismatch")
    paths: dict[str, Path] = {}
    for name, contract in protocol["inputs"].items():
        path = (ROOT / contract["path"]).resolve()
        if sha256(path) != contract["sha256"]:
            raise ValueError(f"Frozen robust-ranker input hash mismatch: {name}")
        paths[name] = path
    values = load_development(
        {"inner": paths["inner"], "fold0": paths["fold0"], "fold1": paths["fold1"]},
        paths["scores"],
    )
    features, feature_names = build_features(values, paths["prithvi"], paths["spatial_scores"])
    selection_folds = list(map(int, protocol["folds"]["selection"]))
    audit_folds = list(map(int, protocol["folds"]["reused_holdout_audit"]))
    seeds = list(map(int, protocol["training"]["seeds"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Robust Prithvi ranker requires CUDA")

    candidate_results: list[dict[str, Any]] = []
    raw_by_spec: dict[str, np.ndarray] = {}
    histories: dict[str, Any] = {}
    for spec_index, pair_weight in enumerate(protocol["search"]["pair_weights"]):
        spec = {**protocol["training"]["model"], "pair_weight": float(pair_weight)}
        key = f"pair_weight_{float(pair_weight):.2f}"
        raw = np.full(values["labels"].shape, np.nan, dtype=np.float64)
        histories[key] = {}
        for held_fold in selection_folds:
            fit = np.isin(values["folds"], [fold for fold in selection_folds if fold != held_fold])
            held = values["folds"] == held_fold
            source, target, _, _ = independent_normalize(features[fit], features[held])
            predictions = []
            histories[key][str(held_fold)] = []
            for seed in seeds:
                state, history = fit_ranker(
                    source, values["labels"][fit], values["groups"][fit],
                    values["folds"][fit], values["sensors"][fit], spec,
                    seed + held_fold, device,
                )
                predictions.append(predict(state, target, spec, device))
                histories[key][str(held_fold)].append(history)
            clipped = np.clip(np.stack(predictions), 1e-6, 1.0 - 1e-6)
            mean_logit = np.mean(np.log(clipped) - np.log1p(-clipped), axis=0)
            raw[held] = 1.0 / (1.0 + np.exp(-mean_logit))
            print(json.dumps({
                "candidate": spec_index + 1,
                "pair_weight": pair_weight,
                "completed_selection_fold": held_fold,
            }), flush=True)
        raw_by_spec[key] = raw
        for blend in protocol["search"]["current_blends"]:
            result = evaluate(values, raw, selection_folds, float(blend))
            result.update({"spec_key": key, "spec": spec})
            candidate_results.append(result)

    selected = max(candidate_results, key=lambda value: tuple(value["rank"]))
    selection_raw = raw_by_spec[selected["spec_key"]]
    selection_rows = np.isin(values["folds"], selection_folds)
    selection_scores = blend_scores(
        values["current"][selection_rows], selection_raw[selection_rows], selected["blend_weight"]
    )
    selected["bootstrap"] = ap_group_bootstrap(
        values["labels"][selection_rows], values["current"][selection_rows], selection_scores,
        values["groups"][selection_rows], replicates=int(protocol["bootstrap"]["replicates"]),
        seed=int(protocol["bootstrap"]["selection_seed"]),
    )
    selection_passed = bool(selected["stable"] and selected["bootstrap"]["lower"] > 0.0)

    fixed_spec = selected["spec"]
    fit = np.isin(values["folds"], selection_folds)
    source_mean = features[fit].mean(0)
    source_scale = np.maximum(features[fit].std(0), 1e-4)
    source = ((features[fit] - source_mean) / source_scale).astype(np.float32)
    audit_raw = np.full(values["labels"].shape, np.nan, dtype=np.float64)
    audit_histories: dict[str, Any] = {}
    states = []
    for seed in seeds:
        state, history = fit_ranker(
            source, values["labels"][fit], values["groups"][fit], values["folds"][fit],
            values["sensors"][fit], fixed_spec, seed + 50, device,
        )
        states.append(state)
        audit_histories[str(seed)] = history
    for held_fold in audit_folds:
        held = values["folds"] == held_fold
        target_mean = features[held].mean(0)
        target_scale = np.maximum(features[held].std(0), 1e-4)
        target = ((features[held] - target_mean) / target_scale).astype(np.float32)
        predictions = np.stack([predict(state, target, fixed_spec, device) for state in states])
        clipped = np.clip(predictions, 1e-6, 1.0 - 1e-6)
        mean_logit = np.mean(np.log(clipped) - np.log1p(-clipped), axis=0)
        audit_raw[held] = 1.0 / (1.0 + np.exp(-mean_logit))
    audit = evaluate(values, audit_raw, audit_folds, float(selected["blend_weight"]))
    audit_rows = np.isin(values["folds"], audit_folds)
    audit_scores = blend_scores(
        values["current"][audit_rows], audit_raw[audit_rows], selected["blend_weight"]
    )
    audit["bootstrap"] = ap_group_bootstrap(
        values["labels"][audit_rows], values["current"][audit_rows], audit_scores,
        values["groups"][audit_rows], replicates=int(protocol["bootstrap"]["replicates"]),
        seed=int(protocol["bootstrap"]["audit_seed"]),
    )
    audit_passed = bool(audit["stable"] and audit["bootstrap"]["lower"] > 0.0)
    passed = selection_passed and audit_passed

    artifact_record = None
    score_record = None
    if passed:
        final_source = features.astype(np.float32)
        final_mean = final_source.mean(0)
        final_scale = np.maximum(final_source.std(0), 1e-4)
        final_normalized = ((final_source - final_mean) / final_scale).astype(np.float32)
        final_states = []
        for seed in seeds:
            state, _ = fit_ranker(
                final_normalized, values["labels"], values["groups"], values["folds"],
                values["sensors"], fixed_spec, seed + 100, device,
            )
            final_states.append(state)
        artifact_path = (ROOT / protocol["outputs"]["artifact"]).resolve()
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = artifact_path.with_suffix(artifact_path.suffix + ".tmp")
        torch.save({
            "schema_version": 1,
            "kind": "mars_robust_prithvi_ranker",
            "spec": fixed_spec,
            "seeds": seeds,
            "blend_weight": float(selected["blend_weight"]),
            "feature_names": feature_names,
            "source_mean": final_mean.astype(np.float32),
            "source_scale": final_scale.astype(np.float32),
            "states": final_states,
            "target_normalization": "independent unlabeled target mean and standard deviation",
            "operational_scene_threshold": float(protocol["training"]["operational_scene_threshold"]),
            "protocol_sha256": sha256(protocol_path),
        }, temporary)
        os.replace(temporary, artifact_path)
        artifact_record = {
            "path": protocol["outputs"]["artifact"],
            "bytes": artifact_path.stat().st_size,
            "sha256": sha256(artifact_path),
            "tracked": False,
        }
        score_path = (ROOT / protocol["outputs"]["development_scores"]).resolve()
        score_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_score = score_path.with_suffix(".tmp.npz")
        combined_raw = selection_raw.copy()
        combined_raw[audit_rows] = audit_raw[audit_rows]
        np.savez_compressed(
            temporary_score,
            sample_ids=values["sample_ids"], groups=values["groups"], folds=values["folds"],
            labels=values["labels"], raw_scores=combined_raw,
            candidate_scores=blend_scores(values["current"], combined_raw, selected["blend_weight"]),
        )
        os.replace(temporary_score, score_path)
        score_record = {
            "path": protocol["outputs"]["development_scores"],
            "bytes": score_path.stat().st_size,
            "sha256": sha256(score_path),
            "tracked": False,
        }

    report = {
        "schema_version": 1,
        "scope": "development-only worst-domain pairwise-AUC Prithvi ranker; no fresh or paper inputs",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_count": len(candidate_results),
        "selected": selected,
        "selection_passed": selection_passed,
        "reused_holdout_audit": {
            **audit,
            "independent_confirmation": False,
            "reason": "folds 0/1 have been exposed by predecessor architecture experiments",
        },
        "reused_holdout_audit_passed": audit_passed,
        "all_promotion_gates_pass": passed,
        "artifact": artifact_record,
        "development_score_cache": score_record,
        "training_history": {"selection": histories, "audit": audit_histories},
        "decision": (
            "Freeze robust Prithvi ranker for a fresh safety evaluation."
            if passed else "Reject robust Prithvi ranker before fresh or paper scoring."
        ),
        "provenance": {
            "protocol_sha256": sha256(protocol_path),
            "script_sha256": sha256(Path(__file__).resolve()),
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "device": str(torch.cuda.get_device_name(device)),
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
    }
    output_json = (ROOT / protocol["outputs"]["json"]).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown((ROOT / protocol["outputs"]["markdown"]).resolve(), report)
    print(json.dumps({
        "ok": passed,
        "selected_pair_weight": selected["spec"]["pair_weight"],
        "selected_blend": selected["blend_weight"],
        "selection_ap_delta": selected["versus_current"]["delta"]["average_precision"],
        "selection_ap_lower": selected["bootstrap"]["lower"],
        "audit_ap_delta": audit["versus_current"]["delta"]["average_precision"],
        "audit_ap_lower": audit["bootstrap"]["lower"],
    }, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
