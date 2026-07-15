#!/usr/bin/env python3
"""Train a grouped scene probe over frozen multi-scale MARS encoder features."""

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

from acquire_mars_metadata import repo_root, sha256  # noqa: E402
from train_mars_oof_scene_ensemble_v2 import (  # noqa: E402
    ap_group_bootstrap,
    sample_weights,
)
from train_mars_scene_ranker import comparison, metric_summary  # noqa: E402

DEFAULT_ENCODER_CACHE = Path("outputs/mars_encoder_scene_features_all_folds.npz")
DEFAULT_ENCODER_SHA256 = "397501c6230436cb677047abb8b2895f3bafb9916150087370c9586a47559e1f"
DEFAULT_SCORE_CACHE = Path("outputs/mars_scene_domain_routing_development_scores.npz")
DEFAULT_SCORE_SHA256 = "fd955b78b26a3b2a5165b4abab02180ccf4dad433511bf4da7afbff44275c1c7"
DEFAULT_INNER_CACHE = Path("outputs/mars_scene_features_folds234.npz")
DEFAULT_INNER_SHA256 = "01d8587e283c1179d61a7c789eb514b3f699d3e7a75bf8c50e4baff3f1698b89"
DEFAULT_FOLD0_CACHE = Path("outputs/mars_scene_features_fold0.npz")
DEFAULT_FOLD0_SHA256 = "372e152734db1314417ed385b099af54acd182bf758b1d2eabcedfeb64a709e7"
DEFAULT_FOLD1_CACHE = Path("outputs/mars_scene_features_fold1_crossfit.npz")
DEFAULT_FOLD1_SHA256 = "2b62e03215047d6a49639fdaead7e9d3cf7939b8eda26fb9442210b49c3ba108"
DEFAULT_ARTIFACT = Path("EarthRemoteSensingRapidResponse/artifacts/mars_encoder_scene_probe.pt")
DEFAULT_JSON = Path("reports/experiments/mars_encoder_scene_probe.json")
DEFAULT_MARKDOWN = Path("reports/experiments/MARS_ENCODER_SCENE_PROBE.md")
INNER_FOLDS = (2, 3, 4)
LEVEL5_WIDTH = 1536


class SceneProbe(nn.Module):
    def __init__(self, input_features: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.input_features = input_features
        self.hidden = hidden
        if hidden == 0:
            self.network = nn.Linear(input_features, 1)
        else:
            self.network = nn.Sequential(
                nn.Linear(input_features, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, 1),
            )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 2 or values.shape[1] != self.input_features:
            raise ValueError("Scene probe input shape differs from its frozen schema")
        return self.network(values).squeeze(1)


def candidate_specs() -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for weighting in ("uniform", "group", "site_cell"):
        specs.append(
            {
                "feature_set": "all_encoder_plus_base",
                "hidden": 0,
                "dropout": 0.0,
                "weighting": weighting,
                "epochs": 20,
                "learning_rate": 0.001,
                "weight_decay": 0.001,
            }
        )
        specs.append(
            {
                "feature_set": "level5_plus_base",
                "hidden": 64,
                "dropout": 0.2,
                "weighting": weighting,
                "epochs": 20,
                "learning_rate": 0.001,
                "weight_decay": 0.001,
            }
        )
    specs.append(
        {
            "feature_set": "all_encoder_plus_base",
            "hidden": 64,
            "dropout": 0.2,
            "weighting": "uniform",
            "epochs": 20,
            "learning_rate": 0.001,
            "weight_decay": 0.001,
        }
    )
    return specs


def spec_key(spec: dict[str, Any]) -> str:
    return "_".join(f"{key}-{spec[key]}" for key in sorted(spec))


def select_features(
    encoder: np.ndarray,
    base: np.ndarray,
    encoder_names: np.ndarray,
    base_names: np.ndarray,
    feature_set: str,
) -> tuple[np.ndarray, list[str]]:
    if encoder.shape[0] != base.shape[0]:
        raise ValueError("Encoder and base scene features must align")
    if feature_set == "all_encoder_plus_base":
        selected_encoder = encoder
        selected_names = encoder_names.astype(str).tolist()
    elif feature_set == "level5_plus_base":
        selected_encoder = encoder[:, -LEVEL5_WIDTH:]
        selected_names = encoder_names[-LEVEL5_WIDTH:].astype(str).tolist()
        if not selected_names[0].startswith("level5_"):
            raise ValueError("Level-5 encoder slice differs from the frozen schema")
    else:
        raise ValueError(f"Unknown encoder probe feature set: {feature_set}")
    values = np.concatenate(
        [selected_encoder.astype(np.float32), base.astype(np.float32)], axis=1
    )
    names = [*selected_names, *base_names.astype(str).tolist()]
    return values, names


def fit_scaler(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = values.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = values.std(axis=0, dtype=np.float64).astype(np.float32)
    std = np.maximum(std, 1e-4)
    return mean, std


def transform(values: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return np.clip((values.astype(np.float32) - mean) / std, -12.0, 12.0)


def train_model(
    spec: dict[str, Any],
    features: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    *,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    mean, std = fit_scaler(features)
    values = torch.from_numpy(transform(features, mean, std)).to(device)
    targets = torch.from_numpy(labels.astype(np.float32)).to(device)
    row_weights = torch.from_numpy(weights.astype(np.float32)).to(device)
    model = SceneProbe(features.shape[1], int(spec["hidden"]), float(spec["dropout"])).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(spec["learning_rate"]),
        weight_decay=float(spec["weight_decay"]),
    )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    batch_size = 512
    model.train()
    for _ in range(int(spec["epochs"])):
        order = torch.randperm(values.shape[0], generator=generator)
        for start in range(0, order.numel(), batch_size):
            indices = order[start : start + batch_size].to(device)
            logits = model(values[indices])
            losses = F.binary_cross_entropy_with_logits(
                logits, targets[indices], reduction="none"
            )
            loss = (losses * row_weights[indices]).sum() / row_weights[indices].sum()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
    return {
        "mean": mean,
        "std": std,
        "state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "input_features": features.shape[1],
        "hidden": int(spec["hidden"]),
        "dropout": float(spec["dropout"]),
    }


@torch.no_grad()
def predict_model(
    fitted: dict[str, Any], features: np.ndarray, device: torch.device
) -> np.ndarray:
    model = SceneProbe(
        int(fitted["input_features"]), int(fitted["hidden"]), float(fitted["dropout"])
    ).to(device)
    model.load_state_dict(fitted["state_dict"])
    model.eval()
    values = transform(features, fitted["mean"], fitted["std"])
    parts: list[np.ndarray] = []
    for start in range(0, values.shape[0], 2048):
        batch = torch.from_numpy(values[start : start + 2048]).to(device)
        parts.append(torch.sigmoid(model(batch)).cpu().numpy())
    return np.concatenate(parts).astype(np.float64)


def metric_comparison(
    partition: dict[str, np.ndarray], scores: np.ndarray, reference: str
) -> dict[str, Any]:
    candidate_metrics = metric_summary(partition["labels"], scores, partition["sensors"])
    reference_metrics = metric_summary(
        partition["labels"], partition[reference], partition["sensors"]
    )
    return comparison(candidate_metrics, reference_metrics)


def load_aligned_partitions(
    encoder_path: Path,
    score_path: Path,
    base_paths: dict[str, Path],
) -> tuple[dict[str, dict[str, np.ndarray]], np.ndarray, np.ndarray]:
    with np.load(encoder_path, allow_pickle=False) as cache:
        encoder = cache["features"]
        encoder_ids = cache["sample_ids"].astype(str)
        encoder_names = cache["feature_names"].astype(str)
        encoder_labels = cache["labels"].astype(np.uint8)
        encoder_groups = cache["groups"].astype(str)
        encoder_sensors = cache["sensors"].astype(np.uint8)
    lookup = {sample_id: index for index, sample_id in enumerate(encoder_ids)}
    if len(lookup) != encoder_ids.size:
        raise ValueError("Encoder feature cache contains duplicate sample IDs")
    partitions: dict[str, dict[str, np.ndarray]] = {}
    with np.load(score_path, allow_pickle=False) as scores:
        for name, prefix in (("inner", "inner"), ("fold0", "fold0"), ("fold1", "fold1")):
            with np.load(base_paths[name], allow_pickle=False) as base:
                sample_ids = base["sample_ids"].astype(str)
                indices = np.asarray([lookup[sample_id] for sample_id in sample_ids])
                labels = scores[f"{prefix}_labels"].astype(np.uint8)
                sensors = scores[f"{prefix}_sensors"].astype(np.uint8)
                groups = scores[f"{prefix}_groups"].astype(str)
                if not (
                    np.array_equal(labels, base["labels"].astype(np.uint8))
                    and np.array_equal(sensors, base["sensors"].astype(np.uint8))
                    and np.array_equal(groups, base["groups"].astype(str))
                    and np.array_equal(labels, encoder_labels[indices])
                    and np.array_equal(sensors, encoder_sensors[indices])
                    and np.array_equal(groups, encoder_groups[indices])
                ):
                    raise ValueError(f"{name} feature/score cache alignment failed")
                partitions[name] = {
                    "encoder": encoder[indices],
                    "base": base["features"],
                    "labels": labels,
                    "sensors": sensors,
                    "groups": groups,
                    "primary": scores[f"{prefix}_primary"].astype(np.float64),
                    "new": scores[f"{prefix}_new"].astype(np.float64),
                }
                if name == "inner":
                    partitions[name]["folds"] = base["folds"].astype(np.uint8)
                if "base_names" not in locals():
                    base_names = base["feature_names"].astype(str)
                elif not np.array_equal(base_names, base["feature_names"].astype(str)):
                    raise ValueError("Base feature schemas differ across partitions")
    return partitions, encoder_names, base_names


def crossfit_candidate(
    spec: dict[str, Any],
    partition: dict[str, np.ndarray],
    encoder_names: np.ndarray,
    base_names: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, list[str]]:
    features, names = select_features(
        partition["encoder"],
        partition["base"],
        encoder_names,
        base_names,
        str(spec["feature_set"]),
    )
    scores = np.empty(partition["labels"].shape, dtype=np.float64)
    for holdout in INNER_FOLDS:
        fit_rows = partition["folds"] != holdout
        held_rows = partition["folds"] == holdout
        weights = sample_weights(
            str(spec["weighting"]),
            partition["groups"][fit_rows],
            partition["labels"][fit_rows],
            partition["sensors"][fit_rows],
        )
        fitted = train_model(
            spec,
            features[fit_rows],
            partition["labels"][fit_rows],
            weights,
            seed=20260800 + holdout,
            device=device,
        )
        scores[held_rows] = predict_model(fitted, features[held_rows], device)
    return scores, names


def screen_candidate(
    spec: dict[str, Any], partition: dict[str, np.ndarray], scores: np.ndarray
) -> dict[str, Any]:
    versus_primary = metric_comparison(partition, scores, "primary")
    versus_new = metric_comparison(partition, scores, "new")
    per_fold: dict[str, Any] = {}
    for fold in INNER_FOLDS:
        rows = partition["folds"] == fold
        local = {key: value[rows] for key, value in partition.items() if key not in {"folds", "encoder", "base"}}
        per_fold[str(fold)] = {
            "versus_primary": metric_comparison(local, scores[rows], "primary"),
            "versus_new": metric_comparison(local, scores[rows], "new"),
        }
    primary_fold_ap = [
        value["versus_primary"]["delta"]["average_precision"] for value in per_fold.values()
    ]
    new_fold_ap = [
        value["versus_new"]["delta"]["average_precision"] for value in per_fold.values()
    ]
    stable = (
        versus_primary["delta"]["average_precision"] > 0.0
        and versus_primary["delta"]["recall_at_fpr_0_0713"] > 0.0
        and min(primary_fold_ap) > 0.0
        and min(versus_primary["delta"]["sensor_average_precision"].values()) >= -0.005
        and versus_new["delta"]["average_precision"] > 0.0
        and versus_new["delta"]["recall_at_fpr_0_0713"] >= -0.0025
        and min(new_fold_ap) >= -0.005
    )
    return {
        "spec": spec,
        "spec_key": spec_key(spec),
        "stable": stable,
        "versus_primary": versus_primary,
        "versus_new": versus_new,
        "per_fold": per_fold,
        "rank": [
            int(stable),
            min(new_fold_ap),
            versus_new["delta"]["average_precision"],
            versus_primary["delta"]["average_precision"],
        ],
    }


def confirm_partition(
    partition: dict[str, np.ndarray], scores: np.ndarray, *, seed: int
) -> dict[str, Any]:
    versus_primary = metric_comparison(partition, scores, "primary")
    versus_new = metric_comparison(partition, scores, "new")
    bootstrap = ap_group_bootstrap(
        partition["labels"],
        partition["primary"],
        scores,
        partition["groups"],
        replicates=10000,
        seed=seed,
    )
    checks = {
        "ap_higher_than_primary": versus_primary["delta"]["average_precision"] > 0.0,
        "recall_higher_than_primary": versus_primary["delta"]["recall_at_fpr_0_0713"] > 0.0,
        "ap_ci_lower_positive_vs_primary": bootstrap["lower"] > 0.0,
        "no_sensor_regression_vs_primary": min(
            versus_primary["delta"]["sensor_average_precision"].values()
        )
        >= -0.005,
        "no_material_ap_regression_vs_new": versus_new["delta"]["average_precision"]
        >= -0.005,
    }
    return {
        "rows": int(partition["labels"].size),
        "positive": int(partition["labels"].sum()),
        "sites": len(set(partition["groups"].tolist())),
        "versus_primary": versus_primary,
        "versus_new": versus_new,
        "paired_group_bootstrap_ap_delta_vs_primary": bootstrap,
        "checks": checks,
        "passed": all(checks.values()),
    }


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    selected = report["selected"]
    lines = [
        "# Frozen-encoder MARS scene probe",
        "",
        "Selection used cross-fitted folds 2/3/4; the selected probe was frozen before folds 0/1 were scored.",
        "",
        f"- Selected probe: `{selected['spec_key']}`",
        f"- Inner AP delta vs primary: {selected['versus_primary']['delta']['average_precision']:+.5f}",
        f"- Inner AP delta vs stronger head: {selected['versus_new']['delta']['average_precision']:+.5f}",
        f"- Inner paired AP interval vs primary: [{selected['paired_group_bootstrap_ap_delta_vs_primary']['lower']:+.5f}, {selected['paired_group_bootstrap_ap_delta_vs_primary']['upper']:+.5f}]",
        "",
        "| Partition | AP delta vs primary | Recall delta | AP 95% CI | AP delta vs new | Gates |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for name, value in report["confirmation"].items():
        primary = value["versus_primary"]["delta"]
        ci = value["paired_group_bootstrap_ap_delta_vs_primary"]
        lines.append(
            f"| {name} | {primary['average_precision']:+.5f} | "
            f"{primary['recall_at_fpr_0_0713']:+.5f} | [{ci['lower']:+.5f}, {ci['upper']:+.5f}] | "
            f"{value['versus_new']['delta']['average_precision']:+.5f} | "
            f"{'PASS' if value['passed'] else 'FAIL'} |"
        )
    lines.extend(["", report["decision"]])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder-cache", default=DEFAULT_ENCODER_CACHE.as_posix())
    parser.add_argument("--encoder-sha256", default=DEFAULT_ENCODER_SHA256)
    parser.add_argument("--score-cache", default=DEFAULT_SCORE_CACHE.as_posix())
    parser.add_argument("--score-sha256", default=DEFAULT_SCORE_SHA256)
    parser.add_argument("--inner-cache", default=DEFAULT_INNER_CACHE.as_posix())
    parser.add_argument("--inner-sha256", default=DEFAULT_INNER_SHA256)
    parser.add_argument("--fold0-cache", default=DEFAULT_FOLD0_CACHE.as_posix())
    parser.add_argument("--fold0-sha256", default=DEFAULT_FOLD0_SHA256)
    parser.add_argument("--fold1-cache", default=DEFAULT_FOLD1_CACHE.as_posix())
    parser.add_argument("--fold1-sha256", default=DEFAULT_FOLD1_SHA256)
    parser.add_argument("--artifact", default=DEFAULT_ARTIFACT.as_posix())
    parser.add_argument("--output-json", default=DEFAULT_JSON.as_posix())
    parser.add_argument("--output-markdown", default=DEFAULT_MARKDOWN.as_posix())
    args = parser.parse_args()
    root = repo_root()
    paths = {
        "encoder": (root / args.encoder_cache).resolve(),
        "score": (root / args.score_cache).resolve(),
        "inner": (root / args.inner_cache).resolve(),
        "fold0": (root / args.fold0_cache).resolve(),
        "fold1": (root / args.fold1_cache).resolve(),
    }
    expected = {
        "encoder": args.encoder_sha256,
        "score": args.score_sha256,
        "inner": args.inner_sha256,
        "fold0": args.fold0_sha256,
        "fold1": args.fold1_sha256,
    }
    for name, path in paths.items():
        if sha256(path) != expected[name]:
            raise ValueError(f"Frozen {name} cache hash mismatch")
    partitions, encoder_names, base_names = load_aligned_partitions(
        paths["encoder"],
        paths["score"],
        {name: paths[name] for name in ("inner", "fold0", "fold1")},
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")
    candidates: list[dict[str, Any]] = []
    scores_by_key: dict[str, np.ndarray] = {}
    names_by_key: dict[str, list[str]] = {}
    for index, spec in enumerate(candidate_specs()):
        scores, names = crossfit_candidate(
            spec, partitions["inner"], encoder_names, base_names, device
        )
        candidate = screen_candidate(spec, partitions["inner"], scores)
        candidates.append(candidate)
        scores_by_key[candidate["spec_key"]] = scores
        names_by_key[candidate["spec_key"]] = names
        print(
            json.dumps(
                {
                    "candidate": index + 1,
                    "total": len(candidate_specs()),
                    "spec": candidate["spec_key"],
                    "ap_delta_vs_primary": candidate["versus_primary"]["delta"][
                        "average_precision"
                    ],
                    "ap_delta_vs_new": candidate["versus_new"]["delta"][
                        "average_precision"
                    ],
                    "stable": candidate["stable"],
                }
            ),
            flush=True,
        )
    selected = max(candidates, key=lambda value: tuple(value["rank"]))
    selected_scores = scores_by_key[selected["spec_key"]]
    selected["paired_group_bootstrap_ap_delta_vs_primary"] = ap_group_bootstrap(
        partitions["inner"]["labels"],
        partitions["inner"]["primary"],
        selected_scores,
        partitions["inner"]["groups"],
        replicates=10000,
        seed=20260820,
    )
    selected["paired_group_bootstrap_ap_delta_vs_new"] = ap_group_bootstrap(
        partitions["inner"]["labels"],
        partitions["inner"]["new"],
        selected_scores,
        partitions["inner"]["groups"],
        replicates=10000,
        seed=20260821,
    )
    selected["inner_passed"] = bool(
        selected["stable"]
        and selected["paired_group_bootstrap_ap_delta_vs_primary"]["lower"] > 0.0
        and selected["paired_group_bootstrap_ap_delta_vs_new"]["lower"] > -0.0025
    )

    selected_spec = selected["spec"]
    inner_features, feature_names = select_features(
        partitions["inner"]["encoder"],
        partitions["inner"]["base"],
        encoder_names,
        base_names,
        str(selected_spec["feature_set"]),
    )
    final_weights = sample_weights(
        str(selected_spec["weighting"]),
        partitions["inner"]["groups"],
        partitions["inner"]["labels"],
        partitions["inner"]["sensors"],
    )
    fitted = train_model(
        selected_spec,
        inner_features,
        partitions["inner"]["labels"],
        final_weights,
        seed=20260830,
        device=device,
    )
    confirmation: dict[str, Any] = {}
    thresholds: list[float] = []
    for index, name in enumerate(("fold0", "fold1")):
        partition = partitions[name]
        features, local_names = select_features(
            partition["encoder"],
            partition["base"],
            encoder_names,
            base_names,
            str(selected_spec["feature_set"]),
        )
        if local_names != feature_names:
            raise ValueError("Held encoder probe schema mismatch")
        scores = predict_model(fitted, features, device)
        confirmation[name] = confirm_partition(partition, scores, seed=20260840 + index)
        thresholds.append(
            confirmation[name]["versus_primary"]["metrics"]["operating_point"]["threshold"]
        )

    artifact_path = (root / args.artifact).resolve()
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = artifact_path.with_suffix(artifact_path.suffix + ".tmp")
    payload = {
        "schema_version": 1,
        "kind": "mars_frozen_encoder_scene_probe",
        "spec": selected_spec,
        "feature_names": feature_names,
        "fitted": fitted,
        "operational_scene_threshold": max(thresholds),
        "encoder_cache_sha256": args.encoder_sha256,
        "score_cache_sha256": args.score_sha256,
    }
    torch.save(payload, temporary)
    os.replace(temporary, artifact_path)
    passed = selected["inner_passed"] and all(value["passed"] for value in confirmation.values())
    report = {
        "schema_version": 1,
        "scope": "development only; paper-test labels and imagery are not loaded",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_count": len(candidates),
        "candidates": candidates,
        "selected": selected,
        "confirmation": confirmation,
        "operational_scene_threshold": max(thresholds),
        "all_promotion_gates_pass": passed,
        "decision": (
            "Freeze the encoder scene probe for a transparent post-test paper benchmark."
            if passed
            else "Reject the encoder scene probe before paper-test feature extraction."
        ),
        "literature_rationale": {
            "S2MAE_CVPR_2024": "spatial-spectral representation learning benefits multispectral downstream tasks",
            "Scale_MAE_ICCV_2023": "multi-scale representations improve remote-sensing transfer",
            "AttMetNet_2025": "attention-enhanced scene discrimination targets methane false positives",
        },
        "provenance": {
            **{f"{name}_cache_sha256": expected[name] for name in expected},
            "artifact_sha256": sha256(artifact_path),
            "script_sha256": sha256(Path(__file__).resolve()),
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True
            ).strip(),
            "device": str(torch.cuda.get_device_name(device) if device.type == "cuda" else device),
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
    }
    write_json((root / args.output_json).resolve(), report)
    write_markdown((root / args.output_markdown).resolve(), report)
    print(
        json.dumps(
            {
                "ok": passed,
                "selected": selected["spec_key"],
                "inner_ap_delta_vs_primary": selected["versus_primary"]["delta"][
                    "average_precision"
                ],
                "inner_ap_delta_vs_new": selected["versus_new"]["delta"][
                    "average_precision"
                ],
                "confirmation": {
                    name: {
                        "passed": value["passed"],
                        "ap_delta_vs_primary": value["versus_primary"]["delta"][
                            "average_precision"
                        ],
                        "ap_delta_vs_new": value["versus_new"]["delta"]["average_precision"],
                    }
                    for name, value in confirmation.items()
                },
            },
            indent=2,
        )
    )
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
