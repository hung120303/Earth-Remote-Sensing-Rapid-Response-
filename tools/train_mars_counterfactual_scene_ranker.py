#!/usr/bin/env python3
"""Cross-fit a counterfactual temporal-physics scene ranker on MARS folds 3/4."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
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
from extract_mars_counterfactual_scene_inputs import CHANNEL_NAMES  # noqa: E402
from extract_mars_scene_features import atomic_savez  # noqa: E402
from train_mars_crossfold_bagged_scene_head import load_development  # noqa: E402
from train_mars_oof_scene_ensemble_v2 import (  # noqa: E402
    ap_group_bootstrap,
    sample_weights,
)
from train_mars_scene_ranker import blend_scores, comparison, metric_summary  # noqa: E402
from train_mars_spatial_prithvi_ensemble import align_prithvi_scores  # noqa: E402
from train_mars_temporal_spatial_ensemble import align_spatial_scores  # noqa: E402
from train_mars_unseen_low_prevalence_router import low_prevalence_mask  # noqa: E402

DEFAULT_PROTOCOL = Path("configs/mars_counterfactual_scene_ranker_protocol.json")


def _groups(channels: int) -> int:
    for groups in (16, 8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class ResidualBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                input_channels,
                output_channels,
                3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(_groups(output_channels), output_channels),
            nn.GELU(),
            nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(output_channels), output_channels),
        )
        self.skip = (
            nn.Identity()
            if input_channels == output_channels and stride == 1
            else nn.Conv2d(input_channels, output_channels, 1, stride=stride, bias=False)
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.block(values) + self.skip(values))


class CounterfactualSceneRanker(nn.Module):
    """Compact spatial encoder over factual and counterfactual evidence maps."""

    def __init__(self, input_channels: int, dropout: float) -> None:
        super().__init__()
        self.input_channels = input_channels
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, 32, 5, stride=2, padding=2, bias=False),
            nn.GroupNorm(8, 32),
            nn.GELU(),
        )
        self.stage1 = ResidualBlock(32, 32)
        self.stage2 = ResidualBlock(32, 64, stride=2)
        self.stage3 = ResidualBlock(64, 128, stride=2)
        self.stage4 = ResidualBlock(128, 192, stride=2)
        self.attention = nn.Conv2d(192, 1, kernel_size=1)
        self.sensor_embedding = nn.Embedding(2, 8)
        self.classifier = nn.Sequential(
            nn.Linear(192 * 4 + 8, 160),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(160, 1),
        )

    def forward(self, values: torch.Tensor, sensors: torch.Tensor) -> torch.Tensor:
        if values.ndim != 4 or values.shape[1] != self.input_channels:
            raise ValueError("Counterfactual tensor differs from the ranker schema")
        features = self.stage1(self.stem(values))
        features = self.stage2(features)
        features = self.stage3(features)
        features = self.stage4(features)
        flat = features.flatten(2)
        attention = torch.softmax(self.attention(features).flatten(1), dim=1)
        attended = (flat * attention[:, None, :]).sum(dim=2)
        average = flat.mean(dim=2)
        maximum = flat.amax(dim=2)
        spread = flat.std(dim=2, unbiased=False)
        sensor = self.sensor_embedding(sensors)
        return self.classifier(
            torch.cat([attended, average, maximum, spread, sensor], dim=1)
        ).squeeze(1)


def feature_indices(feature_set: str) -> tuple[int, ...]:
    if feature_set == "directional_teacher":
        return tuple(range(12)) + (26, 27)
    if feature_set == "counterfactual_physics":
        return tuple(range(len(CHANNEL_NAMES)))
    if feature_set == "physics_without_teacher":
        return tuple(range(8, len(CHANNEL_NAMES)))
    raise ValueError(f"Unknown counterfactual feature set: {feature_set}")


def augment_batch(values: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    if torch.rand((), generator=generator).item() < 0.5:
        values = values.flip(-1)
    if torch.rand((), generator=generator).item() < 0.5:
        values = values.flip(-2)
    turns = int(torch.randint(0, 4, (), generator=generator).item())
    return torch.rot90(values, turns, dims=(-2, -1))


def batch_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    row_weights: torch.Tensor,
    *,
    positive_weight: float,
    pair_weight: float,
) -> torch.Tensor:
    class_weights = torch.where(labels > 0.5, positive_weight, 1.0)
    losses = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    bce = (losses * row_weights * class_weights).sum() / (
        row_weights * class_weights
    ).sum().clamp_min(1e-8)
    positive = logits[labels > 0.5]
    negative = logits[labels <= 0.5]
    if positive.numel() == 0 or negative.numel() == 0:
        return bce
    hard_count = min(max(positive.numel() * 4, 16), negative.numel())
    hard_negative = torch.topk(negative, k=hard_count).values
    pair = F.softplus(hard_negative[None, :] - positive[:, None]).mean()
    return bce + pair_weight * pair


def train_model(
    spec: dict[str, Any],
    images: np.ndarray,
    image_rows: np.ndarray,
    labels: np.ndarray,
    sensors: np.ndarray,
    weights: np.ndarray,
    *,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    channels = feature_indices(str(spec["feature_set"]))
    model = CounterfactualSceneRanker(len(channels), float(spec["dropout"])).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(spec["learning_rate"]),
        weight_decay=float(spec["weight_decay"]),
    )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    positive_weight = min(
        4.0,
        float(np.sqrt(weights[labels == 0].sum() / max(weights[labels == 1].sum(), 1e-8))),
    )
    batch_size = int(spec["batch_size"])
    model.train()
    history: list[float] = []
    for _ in range(int(spec["epochs"])):
        order = torch.randperm(labels.size, generator=generator).numpy()
        epoch_loss = 0.0
        epoch_rows = 0
        for start in range(0, labels.size, batch_size):
            rows = order[start : start + batch_size]
            array = np.asarray(images[image_rows[rows]][:, channels], dtype=np.float32)
            values = augment_batch(torch.from_numpy(array), generator).to(device)
            target = torch.from_numpy(labels[rows].astype(np.float32)).to(device)
            sensor = torch.from_numpy(sensors[rows].astype(np.int64)).to(device)
            row_weight = torch.from_numpy(weights[rows].astype(np.float32)).to(device)
            logits = model(values, sensor)
            loss = batch_loss(
                logits,
                target,
                row_weight,
                positive_weight=positive_weight,
                pair_weight=float(spec["pair_weight"]),
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            epoch_loss += float(loss.detach().cpu()) * rows.size
            epoch_rows += rows.size
        history.append(epoch_loss / epoch_rows)
    return {
        "state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "input_channels": len(channels),
        "channel_indices": channels,
        "dropout": float(spec["dropout"]),
        "positive_weight": positive_weight,
        "history": history,
    }


@torch.inference_mode()
def predict_model(
    fitted: dict[str, Any],
    images: np.ndarray,
    image_rows: np.ndarray,
    sensors: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    model = CounterfactualSceneRanker(
        int(fitted["input_channels"]), float(fitted["dropout"])
    ).to(device)
    model.load_state_dict(fitted["state_dict"])
    model.eval()
    channels = tuple(int(value) for value in fitted["channel_indices"])
    parts: list[np.ndarray] = []
    for start in range(0, image_rows.size, 256):
        rows = slice(start, start + 256)
        array = np.asarray(images[image_rows[rows]][:, channels], dtype=np.float32)
        values = torch.from_numpy(array).to(device)
        sensor = torch.from_numpy(sensors[rows].astype(np.int64)).to(device)
        parts.append(torch.sigmoid(model(values, sensor)).cpu().numpy())
    result = np.concatenate(parts).astype(np.float64)
    if result.shape != (image_rows.size,) or not np.isfinite(result).all():
        raise RuntimeError("Counterfactual model produced invalid scores")
    return result


def current_spatial_prithvi_scores(
    paths: dict[str, Path], expected_weight: float
) -> dict[str, np.ndarray]:
    values = load_development(
        {name: paths[name] for name in ("inner", "fold0", "fold1")},
        paths["score_cache"],
    )
    spatial = align_spatial_scores(values, paths["spatial_scores"])
    prithvi = align_prithvi_scores(values, paths["prithvi_scores"])
    artifact = joblib.load(paths["ensemble_artifact"])
    weight = float(artifact["prithvi_weight"])
    if weight != expected_weight:
        raise ValueError("Frozen spatial-Prithvi weight differs from ranker protocol")
    values["current"] = blend_scores(spatial, prithvi, weight)
    return values


def align_images(
    values: dict[str, np.ndarray], metadata_path: Path
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    with np.load(metadata_path, allow_pickle=False) as metadata:
        sample_ids = metadata["sample_ids"].astype(str)
        local = {
            key: metadata[key].copy()
            for key in ("labels", "sensors", "groups", "folds", "channel_names")
        }
    if not np.array_equal(local["channel_names"].astype(str), np.asarray(CHANNEL_NAMES)):
        raise ValueError("Counterfactual channel schema differs from the ranker")
    lookup = {identifier: index for index, identifier in enumerate(sample_ids)}
    if len(lookup) != sample_ids.size:
        raise ValueError("Counterfactual metadata has duplicate sample IDs")
    selected = np.isin(values["folds"], (3, 4))
    selected_ids = values["sample_ids"][selected].astype(str)
    if set(selected_ids.tolist()) != set(sample_ids.tolist()):
        raise ValueError("Counterfactual cache does not exactly cover development folds 3/4")
    image_rows = np.asarray([lookup[identifier] for identifier in selected_ids], dtype=np.int64)
    for key in ("labels", "sensors", "groups", "folds"):
        if not np.array_equal(np.asarray(values[key])[selected], local[key][image_rows]):
            raise ValueError(f"Counterfactual metadata {key} differs from development")
    subset = {
        key: np.asarray(value)[selected]
        for key, value in values.items()
        if isinstance(value, np.ndarray) and value.shape[:1] == selected.shape
    }
    return image_rows, subset


def evaluate_view(
    values: dict[str, np.ndarray], candidate: np.ndarray, rows: np.ndarray
) -> dict[str, Any]:
    baseline = metric_summary(
        values["labels"][rows], values["current"][rows], values["sensors"][rows]
    )
    metrics = metric_summary(
        values["labels"][rows], candidate[rows], values["sensors"][rows]
    )
    per_fold: dict[str, Any] = {}
    for fold in (3, 4):
        local = rows & (values["folds"] == fold)
        base_local = metric_summary(
            values["labels"][local], values["current"][local], values["sensors"][local]
        )
        candidate_local = metric_summary(
            values["labels"][local], candidate[local], values["sensors"][local]
        )
        per_fold[str(fold)] = comparison(candidate_local, base_local)
    return {
        "rows": int(rows.sum()),
        "positive": int(values["labels"][rows].sum()),
        "sites": len(set(values["groups"][rows].tolist())),
        "baseline": baseline,
        "candidate": metrics,
        "versus_current": comparison(metrics, baseline),
        "per_fold": per_fold,
    }


def gates(
    whole: dict[str, Any], rare: dict[str, Any], protocol_gates: dict[str, Any]
) -> dict[str, bool]:
    whole_delta = whole["versus_current"]["delta"]
    rare_delta = rare["versus_current"]["delta"]
    return {
        "whole_ap_minimum": whole_delta["average_precision"]
        >= float(protocol_gates["whole_ap_delta"]),
        "whole_recall_no_lower": whole_delta["recall_at_fpr_0_0713"] >= 0.0,
        "whole_each_fold_ap_positive": min(
            value["delta"]["average_precision"] for value in whole["per_fold"].values()
        )
        >= 0.0,
        "whole_each_sensor_ap_positive": min(
            whole_delta["sensor_average_precision"].values()
        )
        >= 0.0,
        "whole_ap_interval_positive": whole["bootstrap"]["lower"] > 0.0,
        "rare_ap_minimum": rare_delta["average_precision"]
        >= float(protocol_gates["rare_ap_delta"]),
        "rare_recall_no_lower": rare_delta["recall_at_fpr_0_0713"] >= 0.0,
        "rare_each_fold_ap_positive": min(
            value["delta"]["average_precision"] for value in rare["per_fold"].values()
        )
        >= 0.0,
        "rare_each_sensor_ap_positive": min(
            rare_delta["sensor_average_precision"].values()
        )
        >= 0.0,
        "rare_ap_interval_positive": rare["bootstrap"]["lower"] > 0.0,
    }


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    selected = report.get("selected")
    lines = [
        "# Counterfactual temporal-physics scene ranker",
        "",
        "Cross-fitted selection used only development folds 3 and 4 and compared against the frozen spatial-Prithvi ensemble.",
        "",
    ]
    if selected is None:
        lines.append("No candidate passed every preregistered whole-view and low-prevalence gate.")
    else:
        lines.extend(
            [
                f"- Feature set: `{selected['feature_set']}`",
                f"- Blend weight: {selected['blend_weight']:.3f}",
                f"- Whole AP delta: {selected['whole']['versus_current']['delta']['average_precision']:+.6f}",
                f"- Rare-site AP delta: {selected['rare']['versus_current']['delta']['average_precision']:+.6f}",
                f"- Whole AP interval: [{selected['whole']['bootstrap']['lower']:+.6f}, {selected['whole']['bootstrap']['upper']:+.6f}]",
                f"- Rare-site AP interval: [{selected['rare']['bootstrap']['lower']:+.6f}, {selected['rare']['bootstrap']['upper']:+.6f}]",
            ]
        )
    lines.extend(["", report["decision"]])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL.as_posix())
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    protocol_path = (ROOT / args.protocol).resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if sha256(Path(__file__).resolve()) != protocol["trainer"]["sha256"]:
        raise ValueError("Frozen counterfactual ranker hash mismatch")
    paths: dict[str, Path] = {}
    for name, contract in protocol["inputs"].items():
        path = (ROOT / contract["path"]).resolve()
        if sha256(path) != contract["sha256"]:
            raise ValueError(f"Frozen ranker input hash mismatch: {name}")
        paths[name] = path
    images = np.load(paths["images"], mmap_mode="r", allow_pickle=False)
    if images.shape[1:] != (len(CHANNEL_NAMES), 64, 64) or images.dtype != np.float16:
        raise ValueError("Counterfactual image cache schema mismatch")
    values = current_spatial_prithvi_scores(
        paths, float(protocol["current_comparator"]["prithvi_weight"])
    )
    image_rows, values = align_images(values, paths["metadata"])
    if args.smoke:
        chosen: list[int] = []
        strata: Counter[tuple[int, int]] = Counter()
        for index, key in enumerate(zip(values["labels"], values["sensors"], strict=True)):
            pair = (int(key[0]), int(key[1]))
            if strata[pair] >= 4:
                continue
            chosen.append(index)
            strata[pair] += 1
            if len(strata) == 4 and min(strata.values()) >= 4:
                break
        chosen_rows = np.asarray(chosen, dtype=np.int64)
        spec = dict(protocol["models"][0])
        spec["epochs"] = 1
        weights = np.ones(chosen_rows.size, dtype=np.float64)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        fitted = train_model(
            spec,
            images,
            image_rows[chosen_rows],
            values["labels"][chosen_rows],
            values["sensors"][chosen_rows],
            weights,
            seed=int(protocol["seeds"][0]),
            device=device,
        )
        scores = predict_model(
            fitted,
            images,
            image_rows[chosen_rows],
            values["sensors"][chosen_rows],
            device,
        )
        print(
            json.dumps(
                {
                    "ok": True,
                    "smoke": True,
                    "rows": int(chosen_rows.size),
                    "strata": {f"{key[0]}|{key[1]}": value for key, value in strata.items()},
                    "finite_scores": bool(np.isfinite(scores).all()),
                    "history": fitted["history"],
                },
                indent=2,
            )
        )
        return 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Counterfactual ranker selection requires CUDA")
    torch.set_float32_matmul_precision("high")
    rare_rows = low_prevalence_mask(
        values["labels"],
        values["groups"],
        float(protocol["target_view"]["maximum_site_positive_rate"]),
    )
    all_rows = np.ones(values["labels"].shape, dtype=bool)
    candidates: list[dict[str, Any]] = []
    raw_by_feature: dict[str, np.ndarray] = {}
    fitted_by_feature: dict[str, list[dict[str, Any]]] = {}
    for spec_index, spec in enumerate(protocol["models"]):
        raw = np.empty(values["labels"].shape, dtype=np.float64)
        fitted_by_feature[str(spec["feature_set"])] = []
        for holdout in (3, 4):
            fit_rows = values["folds"] != holdout
            held_rows = values["folds"] == holdout
            weights = sample_weights(
                str(spec["weighting"]),
                values["groups"][fit_rows],
                values["labels"][fit_rows],
                values["sensors"][fit_rows],
            )
            seed_scores: list[np.ndarray] = []
            for seed in protocol["seeds"]:
                fitted = train_model(
                    spec,
                    images,
                    image_rows[fit_rows],
                    values["labels"][fit_rows],
                    values["sensors"][fit_rows],
                    weights,
                    seed=int(seed) + holdout,
                    device=device,
                )
                seed_scores.append(
                    predict_model(
                        fitted,
                        images,
                        image_rows[held_rows],
                        values["sensors"][held_rows],
                        device,
                    )
                )
            stacked = np.stack(seed_scores)
            clipped = np.clip(stacked, 1e-5, 1.0 - 1e-5)
            raw[held_rows] = 1.0 / (
                1.0 + np.exp(-np.mean(np.log(clipped / (1.0 - clipped)), axis=0))
            )
        raw_by_feature[str(spec["feature_set"])] = raw
        for weight in protocol["blend_weights"]:
            scores = blend_scores(values["current"], raw, float(weight))
            whole = evaluate_view(values, scores, all_rows)
            rare = evaluate_view(values, scores, rare_rows)
            whole["bootstrap"] = ap_group_bootstrap(
                values["labels"],
                values["current"],
                scores,
                values["groups"],
                replicates=int(protocol["bootstrap"]["replicates"]),
                seed=int(protocol["bootstrap"]["whole_seed"]) + spec_index * 100 + int(float(weight) * 100),
            )
            rare["bootstrap"] = ap_group_bootstrap(
                values["labels"][rare_rows],
                values["current"][rare_rows],
                scores[rare_rows],
                values["groups"][rare_rows],
                replicates=int(protocol["bootstrap"]["replicates"]),
                seed=int(protocol["bootstrap"]["rare_seed"]) + spec_index * 100 + int(float(weight) * 100),
            )
            checks = gates(whole, rare, protocol["gates"])
            candidate = {
                "feature_set": str(spec["feature_set"]),
                "blend_weight": float(weight),
                "whole": whole,
                "rare": rare,
                "checks": checks,
                "passed": all(checks.values()),
            }
            candidate["rank"] = [
                int(candidate["passed"]),
                rare["bootstrap"]["lower"],
                whole["bootstrap"]["lower"],
                rare["versus_current"]["delta"]["average_precision"],
                whole["versus_current"]["delta"]["average_precision"],
                -float(weight),
            ]
            candidates.append(candidate)
        best = max(
            [value for value in candidates if value["feature_set"] == spec["feature_set"]],
            key=lambda value: tuple(value["rank"]),
        )
        print(
            json.dumps(
                {
                    "model": spec_index + 1,
                    "models": len(protocol["models"]),
                    "feature_set": spec["feature_set"],
                    "best_weight": best["blend_weight"],
                    "whole_ap_delta": best["whole"]["versus_current"]["delta"]["average_precision"],
                    "rare_ap_delta": best["rare"]["versus_current"]["delta"]["average_precision"],
                    "passed": best["passed"],
                }
            ),
            flush=True,
        )
    winner = max(candidates, key=lambda value: tuple(value["rank"]))
    selected = winner if winner["passed"] else None
    artifact_record = None
    if selected is not None:
        spec = next(
            value
            for value in protocol["models"]
            if value["feature_set"] == selected["feature_set"]
        )
        weights = sample_weights(
            str(spec["weighting"]), values["groups"], values["labels"], values["sensors"]
        )
        endpoints = [
            train_model(
                spec,
                images,
                image_rows,
                values["labels"],
                values["sensors"],
                weights,
                seed=int(seed) + 34,
                device=device,
            )
            for seed in protocol["seeds"]
        ]
        artifact_path = (ROOT / protocol["outputs"]["artifact"]).resolve()
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = artifact_path.with_suffix(artifact_path.suffix + ".tmp")
        torch.save(
            {
                "schema_version": 1,
                "kind": "mars_counterfactual_scene_ranker_folds34",
                "fit_folds": [3, 4],
                "forbidden_folds": [0, 1, 2],
                "spec": spec,
                "blend_weight": selected["blend_weight"],
                "channel_names": list(CHANNEL_NAMES),
                "endpoints": endpoints,
                "protocol_sha256": sha256(protocol_path),
            },
            temporary,
        )
        os.replace(temporary, artifact_path)
        artifact_record = {
            "path": protocol["outputs"]["artifact"],
            "bytes": artifact_path.stat().st_size,
            "sha256": sha256(artifact_path),
            "tracked": False,
        }
    raw_output = (ROOT / protocol["outputs"]["crossfit_scores"]).resolve()
    atomic_savez(
        raw_output,
        sample_ids=values["sample_ids"],
        labels=values["labels"],
        sensors=values["sensors"],
        groups=values["groups"],
        folds=values["folds"],
        current_scores=values["current"],
        **{f"raw_{name}": scores for name, scores in raw_by_feature.items()},
        protocol_sha256=np.asarray(sha256(protocol_path)),
    )
    report = {
        "schema_version": 1,
        "scope": "development folds 3/4 honest cross-fit; no folds 0/1/2, exact-paper, or fresh-external access",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "current_comparator": protocol["current_comparator"],
        "candidate_count": len(candidates),
        "candidates": candidates,
        "selected": selected,
        "artifact": artifact_record,
        "all_promotion_gates_pass": selected is not None,
        "decision": (
            "Freeze the counterfactual ranker for separately preregistered fold-2 confirmation."
            if selected is not None
            else "Reject this counterfactual ranker family before any fold-0/1/2 or external extraction."
        ),
        "provenance": {
            "protocol_sha256": sha256(protocol_path),
            "script_sha256": sha256(Path(__file__).resolve()),
            "crossfit_scores_sha256": sha256(raw_output),
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "device": torch.cuda.get_device_name(device),
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
    }
    output_json = (ROOT / protocol["outputs"]["json"]).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    temporary_json = output_json.with_suffix(output_json.suffix + ".tmp")
    temporary_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary_json, output_json)
    write_markdown((ROOT / protocol["outputs"]["markdown"]).resolve(), report)
    print(
        json.dumps(
            {
                "ok": selected is not None,
                "selected": None
                if selected is None
                else {
                    "feature_set": selected["feature_set"],
                    "blend_weight": selected["blend_weight"],
                    "whole_ap_delta": selected["whole"]["versus_current"]["delta"]["average_precision"],
                    "rare_ap_delta": selected["rare"]["versus_current"]["delta"]["average_precision"],
                },
                "artifact": artifact_record,
            },
            indent=2,
        )
    )
    return 0 if selected is not None else 2


if __name__ == "__main__":
    raise SystemExit(main())
