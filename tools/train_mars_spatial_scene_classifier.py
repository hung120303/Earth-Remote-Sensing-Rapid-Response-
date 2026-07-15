#!/usr/bin/env python3
"""Train a physics-guided spatial morphology classifier for MARS scenes."""

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
from extract_mars_spatial_scene_inputs import CHANNEL_NAMES  # noqa: E402
from train_mars_oof_scene_ensemble_v2 import ap_group_bootstrap, sample_weights  # noqa: E402
from train_mars_scene_ranker import blend_scores, comparison, metric_summary  # noqa: E402

DEFAULT_IMAGES = Path("outputs/mars_spatial_scene_inputs_all_folds.npy")
DEFAULT_IMAGES_SHA256 = "6530fa2d07d94bd57ba1ac757039dedd18745227e7a476fb0d69f78f996134a5"
DEFAULT_METADATA = Path("outputs/mars_spatial_scene_inputs_all_folds_metadata.npz")
DEFAULT_METADATA_SHA256 = "0160ed93371396a487b819dd170a682f55756d285f0ba94d0f0d093ba51a8d01"
DEFAULT_SCORE_CACHE = Path("outputs/mars_scene_domain_routing_development_scores.npz")
DEFAULT_SCORE_SHA256 = "fd955b78b26a3b2a5165b4abab02180ccf4dad433511bf4da7afbff44275c1c7"
DEFAULT_INNER_CACHE = Path("outputs/mars_scene_features_folds234.npz")
DEFAULT_INNER_SHA256 = "01d8587e283c1179d61a7c789eb514b3f699d3e7a75bf8c50e4baff3f1698b89"
DEFAULT_FOLD0_CACHE = Path("outputs/mars_scene_features_fold0.npz")
DEFAULT_FOLD0_SHA256 = "372e152734db1314417ed385b099af54acd182bf758b1d2eabcedfeb64a709e7"
DEFAULT_FOLD1_CACHE = Path("outputs/mars_scene_features_fold1_crossfit.npz")
DEFAULT_FOLD1_SHA256 = "2b62e03215047d6a49639fdaead7e9d3cf7939b8eda26fb9442210b49c3ba108"
DEFAULT_ARTIFACT = Path("EarthRemoteSensingRapidResponse/artifacts/mars_spatial_scene_classifier.pt")
DEFAULT_JSON = Path("reports/experiments/mars_spatial_scene_classifier.json")
DEFAULT_MARKDOWN = Path("reports/experiments/MARS_SPATIAL_SCENE_CLASSIFIER.md")
INNER_FOLDS = (2, 3, 4)
BLEND_WEIGHTS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.625, 0.75, 1.0)


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


class SpatialSceneClassifier(nn.Module):
    def __init__(self, input_channels: int, dropout: float = 0.2) -> None:
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
            nn.Linear(192 * 3 + 8, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, values: torch.Tensor, sensors: torch.Tensor) -> torch.Tensor:
        if values.ndim != 4 or values.shape[1] != self.input_channels:
            raise ValueError("Spatial scene tensor differs from the classifier schema")
        if sensors.shape != (values.shape[0],):
            raise ValueError("Sensor indices must have shape B")
        features = self.stage1(self.stem(values))
        features = self.stage2(features)
        features = self.stage3(features)
        features = self.stage4(features)
        attention = self.attention(features).flatten(1)
        attention = torch.softmax(attention, dim=1)
        flat = features.flatten(2)
        attended = (flat * attention[:, None, :]).sum(dim=2)
        average = flat.mean(dim=2)
        maximum = flat.amax(dim=2)
        sensor = self.sensor_embedding(sensors)
        return self.classifier(torch.cat([attended, average, maximum, sensor], dim=1)).squeeze(1)


def channel_indices(feature_set: str) -> tuple[int, ...]:
    if feature_set == "physics_spatial":
        return tuple(range(len(CHANNEL_NAMES)))
    if feature_set == "probability_spatial":
        return (0, 1, 7, 8)
    raise ValueError(f"Unknown spatial feature set: {feature_set}")


def candidate_specs() -> list[dict[str, Any]]:
    return [
        {
            "feature_set": feature_set,
            "weighting": weighting,
            "epochs": 8,
            "learning_rate": 0.0003,
            "weight_decay": 0.001,
            "dropout": 0.2,
        }
        for feature_set, weighting in (
            ("physics_spatial", "uniform"),
            ("physics_spatial", "group"),
            ("physics_spatial", "site_cell"),
            ("probability_spatial", "group"),
        )
    ]


def spec_key(spec: dict[str, Any]) -> str:
    return "_".join(f"{key}-{spec[key]}" for key in sorted(spec))


def augment_batch(values: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    if torch.rand((), generator=generator).item() < 0.5:
        values = values.flip(-1)
    if torch.rand((), generator=generator).item() < 0.5:
        values = values.flip(-2)
    rotations = int(torch.randint(0, 4, (), generator=generator).item())
    return torch.rot90(values, rotations, dims=(-2, -1))


def train_model(
    spec: dict[str, Any],
    images: np.ndarray,
    global_indices: np.ndarray,
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
    channels = channel_indices(str(spec["feature_set"]))
    model = SpatialSceneClassifier(len(channels), float(spec["dropout"])).to(device)
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
    batch_size = 128
    model.train()
    for _ in range(int(spec["epochs"])):
        order = torch.randperm(labels.size, generator=generator).numpy()
        for start in range(0, labels.size, batch_size):
            rows = order[start : start + batch_size]
            array = np.asarray(images[global_indices[rows]][:, channels], dtype=np.float32)
            values = augment_batch(torch.from_numpy(array), generator).to(device)
            target = torch.from_numpy(labels[rows].astype(np.float32)).to(device)
            sensor = torch.from_numpy(sensors[rows].astype(np.int64)).to(device)
            row_weight = torch.from_numpy(weights[rows].astype(np.float32)).to(device)
            class_weight = torch.where(target > 0.5, positive_weight, 1.0)
            losses = F.binary_cross_entropy_with_logits(
                model(values, sensor), target, reduction="none"
            )
            loss = (losses * row_weight * class_weight).sum() / (row_weight * class_weight).sum()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
    return {
        "state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "input_channels": len(channels),
        "channel_indices": channels,
        "dropout": float(spec["dropout"]),
        "positive_weight": positive_weight,
    }


@torch.no_grad()
def predict_model(
    fitted: dict[str, Any],
    images: np.ndarray,
    global_indices: np.ndarray,
    sensors: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    model = SpatialSceneClassifier(
        int(fitted["input_channels"]), float(fitted["dropout"])
    ).to(device)
    model.load_state_dict(fitted["state_dict"])
    model.eval()
    channels = tuple(int(value) for value in fitted["channel_indices"])
    parts: list[np.ndarray] = []
    for start in range(0, global_indices.size, 256):
        rows = slice(start, start + 256)
        array = np.asarray(images[global_indices[rows]][:, channels], dtype=np.float32)
        values = torch.from_numpy(array).to(device)
        sensor = torch.from_numpy(sensors[rows].astype(np.int64)).to(device)
        parts.append(torch.sigmoid(model(values, sensor)).cpu().numpy())
    return np.concatenate(parts).astype(np.float64)


def metric_comparison(
    partition: dict[str, np.ndarray], scores: np.ndarray, reference: str
) -> dict[str, Any]:
    candidate = metric_summary(partition["labels"], scores, partition["sensors"])
    baseline = metric_summary(
        partition["labels"], partition[reference], partition["sensors"]
    )
    return comparison(candidate, baseline)


def load_partitions(
    metadata_path: Path,
    score_path: Path,
    base_paths: dict[str, Path],
) -> dict[str, dict[str, np.ndarray]]:
    with np.load(metadata_path, allow_pickle=False) as metadata:
        sample_ids = metadata["sample_ids"].astype(str)
        labels = metadata["labels"].astype(np.uint8)
        sensors = metadata["sensors"].astype(np.uint8)
        groups = metadata["groups"].astype(str)
    lookup = {sample_id: index for index, sample_id in enumerate(sample_ids)}
    if len(lookup) != sample_ids.size:
        raise ValueError("Spatial metadata contains duplicate sample IDs")
    partitions: dict[str, dict[str, np.ndarray]] = {}
    with np.load(score_path, allow_pickle=False) as scores:
        for name in ("inner", "fold0", "fold1"):
            with np.load(base_paths[name], allow_pickle=False) as base:
                local_ids = base["sample_ids"].astype(str)
                indices = np.asarray([lookup[sample_id] for sample_id in local_ids])
                local_labels = scores[f"{name}_labels"].astype(np.uint8)
                local_sensors = scores[f"{name}_sensors"].astype(np.uint8)
                local_groups = scores[f"{name}_groups"].astype(str)
                if not (
                    np.array_equal(local_labels, labels[indices])
                    and np.array_equal(local_sensors, sensors[indices])
                    and np.array_equal(local_groups, groups[indices])
                    and np.array_equal(local_labels, base["labels"].astype(np.uint8))
                ):
                    raise ValueError(f"{name} spatial/score cache alignment failed")
                partitions[name] = {
                    "image_indices": indices,
                    "labels": local_labels,
                    "sensors": local_sensors,
                    "groups": local_groups,
                    "primary": scores[f"{name}_primary"].astype(np.float64),
                    "new": scores[f"{name}_new"].astype(np.float64),
                }
                if name == "inner":
                    partitions[name]["folds"] = base["folds"].astype(np.uint8)
    return partitions


def crossfit_raw_scores(
    spec: dict[str, Any],
    images: np.ndarray,
    partition: dict[str, np.ndarray],
    device: torch.device,
) -> np.ndarray:
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
            images,
            partition["image_indices"][fit_rows],
            partition["labels"][fit_rows],
            partition["sensors"][fit_rows],
            weights,
            seed=20260900 + holdout,
            device=device,
        )
        scores[held_rows] = predict_model(
            fitted,
            images,
            partition["image_indices"][held_rows],
            partition["sensors"][held_rows],
            device,
        )
    return scores


def evaluate_candidate(
    spec: dict[str, Any],
    blend_weight: float,
    partition: dict[str, np.ndarray],
    raw_scores: np.ndarray,
) -> dict[str, Any]:
    scores = blend_scores(partition["new"], raw_scores, blend_weight)
    versus_primary = metric_comparison(partition, scores, "primary")
    versus_new = metric_comparison(partition, scores, "new")
    per_fold: dict[str, Any] = {}
    for fold in INNER_FOLDS:
        rows = partition["folds"] == fold
        local = {
            key: value[rows]
            for key, value in partition.items()
            if key not in {"folds", "image_indices"}
        }
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
        "blend_weight": blend_weight,
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
        "# Physics-guided spatial MARS scene classifier",
        "",
        "Selection used cross-fitted folds 2/3/4; the selected model and blend were frozen before folds 0/1 were scored.",
        "",
        f"- Selected model: `{selected['spec_key']}`",
        f"- Spatial blend weight: {selected['blend_weight']:.3f}",
        f"- Inner AP delta vs primary: {selected['versus_primary']['delta']['average_precision']:+.5f}",
        f"- Inner AP delta vs stronger head: {selected['versus_new']['delta']['average_precision']:+.5f}",
        f"- Inner AP interval vs primary: [{selected['paired_group_bootstrap_ap_delta_vs_primary']['lower']:+.5f}, {selected['paired_group_bootstrap_ap_delta_vs_primary']['upper']:+.5f}]",
        "",
        "| Partition | AP delta vs primary | Recall delta | AP 95% CI | AP delta vs new | Gates |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for name, value in report["confirmation"].items():
        delta = value["versus_primary"]["delta"]
        interval = value["paired_group_bootstrap_ap_delta_vs_primary"]
        lines.append(
            f"| {name} | {delta['average_precision']:+.5f} | "
            f"{delta['recall_at_fpr_0_0713']:+.5f} | "
            f"[{interval['lower']:+.5f}, {interval['upper']:+.5f}] | "
            f"{value['versus_new']['delta']['average_precision']:+.5f} | "
            f"{'PASS' if value['passed'] else 'FAIL'} |"
        )
    lines.extend(["", report["decision"]])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", default=DEFAULT_IMAGES.as_posix())
    parser.add_argument("--images-sha256", default=DEFAULT_IMAGES_SHA256)
    parser.add_argument("--metadata", default=DEFAULT_METADATA.as_posix())
    parser.add_argument("--metadata-sha256", default=DEFAULT_METADATA_SHA256)
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
        "images": (root / args.images).resolve(),
        "metadata": (root / args.metadata).resolve(),
        "score": (root / args.score_cache).resolve(),
        "inner": (root / args.inner_cache).resolve(),
        "fold0": (root / args.fold0_cache).resolve(),
        "fold1": (root / args.fold1_cache).resolve(),
    }
    expected = {
        "images": args.images_sha256,
        "metadata": args.metadata_sha256,
        "score": args.score_sha256,
        "inner": args.inner_sha256,
        "fold0": args.fold0_sha256,
        "fold1": args.fold1_sha256,
    }
    for name, path in paths.items():
        if sha256(path) != expected[name]:
            raise ValueError(f"Frozen {name} cache hash mismatch")
    images = np.load(paths["images"], mmap_mode="r", allow_pickle=False)
    if images.shape[1:] != (len(CHANNEL_NAMES), 64, 64) or images.dtype != np.float16:
        raise ValueError("Spatial image cache schema mismatch")
    partitions = load_partitions(
        paths["metadata"],
        paths["score"],
        {name: paths[name] for name in ("inner", "fold0", "fold1")},
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")
    raw_by_key: dict[str, np.ndarray] = {}
    candidates: list[dict[str, Any]] = []
    specs = candidate_specs()
    for index, spec in enumerate(specs):
        raw = crossfit_raw_scores(spec, images, partitions["inner"], device)
        raw_by_key[spec_key(spec)] = raw
        local = [
            evaluate_candidate(spec, weight, partitions["inner"], raw)
            for weight in BLEND_WEIGHTS
        ]
        candidates.extend(local)
        best = max(local, key=lambda value: tuple(value["rank"]))
        print(
            json.dumps(
                {
                    "candidate_model": index + 1,
                    "total_models": len(specs),
                    "spec": best["spec_key"],
                    "best_blend": best["blend_weight"],
                    "ap_delta_vs_primary": best["versus_primary"]["delta"][
                        "average_precision"
                    ],
                    "ap_delta_vs_new": best["versus_new"]["delta"]["average_precision"],
                    "stable": best["stable"],
                }
            ),
            flush=True,
        )
    selected = max(candidates, key=lambda value: tuple(value["rank"]))
    selected_raw = raw_by_key[selected["spec_key"]]
    selected_scores = blend_scores(
        partitions["inner"]["new"], selected_raw, float(selected["blend_weight"])
    )
    selected["paired_group_bootstrap_ap_delta_vs_primary"] = ap_group_bootstrap(
        partitions["inner"]["labels"],
        partitions["inner"]["primary"],
        selected_scores,
        partitions["inner"]["groups"],
        replicates=10000,
        seed=20260920,
    )
    selected["paired_group_bootstrap_ap_delta_vs_new"] = ap_group_bootstrap(
        partitions["inner"]["labels"],
        partitions["inner"]["new"],
        selected_scores,
        partitions["inner"]["groups"],
        replicates=10000,
        seed=20260921,
    )
    selected["inner_passed"] = bool(
        selected["stable"]
        and selected["paired_group_bootstrap_ap_delta_vs_primary"]["lower"] > 0.0
        and selected["paired_group_bootstrap_ap_delta_vs_new"]["lower"] > -0.0025
    )

    spec = selected["spec"]
    weights = sample_weights(
        str(spec["weighting"]),
        partitions["inner"]["groups"],
        partitions["inner"]["labels"],
        partitions["inner"]["sensors"],
    )
    fitted = train_model(
        spec,
        images,
        partitions["inner"]["image_indices"],
        partitions["inner"]["labels"],
        partitions["inner"]["sensors"],
        weights,
        seed=20260930,
        device=device,
    )
    confirmation: dict[str, Any] = {}
    thresholds: list[float] = []
    for index, name in enumerate(("fold0", "fold1")):
        partition = partitions[name]
        raw = predict_model(
            fitted,
            images,
            partition["image_indices"],
            partition["sensors"],
            device,
        )
        scores = blend_scores(partition["new"], raw, float(selected["blend_weight"]))
        confirmation[name] = confirm_partition(partition, scores, seed=20260940 + index)
        thresholds.append(
            confirmation[name]["versus_primary"]["metrics"]["operating_point"]["threshold"]
        )

    artifact_path = (root / args.artifact).resolve()
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = artifact_path.with_suffix(artifact_path.suffix + ".tmp")
    payload = {
        "schema_version": 1,
        "kind": "mars_physics_guided_spatial_scene_classifier",
        "spec": spec,
        "fitted": fitted,
        "blend_weight": float(selected["blend_weight"]),
        "channel_names": list(CHANNEL_NAMES),
        "operational_scene_threshold": max(thresholds),
        "images_sha256": args.images_sha256,
        "metadata_sha256": args.metadata_sha256,
    }
    torch.save(payload, temporary)
    os.replace(temporary, artifact_path)
    passed = selected["inner_passed"] and all(value["passed"] for value in confirmation.values())
    report = {
        "schema_version": 1,
        "scope": "development only; paper-test labels and imagery are not loaded",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_model_count": len(specs),
        "candidate_blend_count": len(candidates),
        "candidates": candidates,
        "selected": selected,
        "confirmation": confirmation,
        "operational_scene_threshold": max(thresholds),
        "all_promotion_gates_pass": passed,
        "decision": (
            "Freeze the spatial classifier for a transparent post-test paper benchmark."
            if passed
            else "Reject the spatial classifier before paper-test extraction."
        ),
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
                "blend_weight": selected["blend_weight"],
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
