#!/usr/bin/env python3
"""Train a cross-fitted scene/mask head on frozen spatial Prithvi patch differences."""

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
for path in (ROOT / "EarthRemoteSensingRapidResponse", ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from acquire_mars_metadata import repo_root, sha256  # noqa: E402
from extract_mars_prithvi_spatial_features import (  # noqa: E402
    BLOCKS,
    EMBED_DIM,
    GRID_SIZE,
    feature_names,
)
from train_mars_oof_scene_ensemble_v2 import ap_group_bootstrap, sample_weights  # noqa: E402
from train_mars_scene_ranker import blend_scores  # noqa: E402
from train_mars_spatial_scene_classifier import (  # noqa: E402
    DEFAULT_FOLD0_CACHE,
    DEFAULT_FOLD0_SHA256,
    DEFAULT_FOLD1_CACHE,
    DEFAULT_FOLD1_SHA256,
    DEFAULT_IMAGES as DEFAULT_PHYSICS,
    DEFAULT_IMAGES_SHA256 as DEFAULT_PHYSICS_SHA256,
    DEFAULT_INNER_CACHE,
    DEFAULT_INNER_SHA256,
    DEFAULT_METADATA as DEFAULT_PHYSICS_METADATA,
    DEFAULT_METADATA_SHA256 as DEFAULT_PHYSICS_METADATA_SHA256,
    DEFAULT_SCORE_CACHE,
    DEFAULT_SCORE_SHA256,
    load_partitions,
)
from train_mars_target_weighted_scene_head import evaluate_candidate  # noqa: E402

DEFAULT_FEATURES = Path("outputs/mars_prithvi_spatial_features_all_folds.npy")
DEFAULT_FEATURES_SHA256 = "c0bd358da7563bec1e6e3cd706aeec86c6558b5bde6181c66bab9c5b30621925"
DEFAULT_TARGETS = Path("outputs/mars_prithvi_spatial_targets_all_folds.npy")
DEFAULT_TARGETS_SHA256 = "6aca9c26dd72d56d3a8c95b7ccc2ef55331f2c8f376a06e4fc2298099744c3c9"
DEFAULT_METADATA = Path("outputs/mars_prithvi_spatial_features_all_folds_metadata.npz")
DEFAULT_METADATA_SHA256 = "de89016e60ab5b6b71cf3f2c6b5d4b9885a1d7257b0cf34e09158396fc0fe813"
DEFAULT_ARTIFACT = Path(
    "EarthRemoteSensingRapidResponse/artifacts/mars_prithvi_spatial_head.pt"
)
DEFAULT_JSON = Path("reports/experiments/mars_prithvi_spatial_head.json")
DEFAULT_MARKDOWN = Path("reports/experiments/MARS_PRITHVI_SPATIAL_HEAD.md")
FOLDS = (0, 1, 2, 3, 4)
SEEDS = (20261900, 20262000)
BLENDS = (0.05, 0.1, 0.2, 0.3, 0.4)
EPOCHS = 8
PATCH_LOSS_WEIGHT = 0.25


def _groups(channels: int) -> int:
    for groups in (16, 12, 8, 6, 4, 3, 2):
        if channels % groups == 0:
            return groups
    return 1


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(channels), channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(channels), channels),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return F.gelu(values + self.block(values))


class SpatialPrithviHead(nn.Module):
    """Fuse multi-depth patch changes with the existing physics maps."""

    def __init__(self, dropout: float = 0.3) -> None:
        super().__init__()
        token_channels = len(BLOCKS) * EMBED_DIM
        self.token_channels = token_channels
        self.token_projection = nn.Sequential(
            nn.Conv2d(token_channels, 192, 1, groups=len(BLOCKS), bias=False),
            nn.GroupNorm(12, 192),
            nn.GELU(),
        )
        self.physics_projection = nn.Sequential(
            nn.Conv2d(9, 32, 1, bias=False),
            nn.GroupNorm(8, 32),
            nn.GELU(),
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(224, 192, 1, bias=False),
            nn.GroupNorm(12, 192),
            nn.GELU(),
            ResidualBlock(192),
            ResidualBlock(192),
        )
        self.patch_head = nn.Conv2d(192, 1, 1)
        self.sensor_embedding = nn.Embedding(2, 8)
        self.scene_head = nn.Sequential(
            nn.Linear(192 * 3 + 8, 192),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(192, 1),
        )

    def forward(
        self, tokens: torch.Tensor, physics: torch.Tensor, sensors: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        if tokens.shape[1:] != (self.token_channels, GRID_SIZE, GRID_SIZE):
            raise ValueError("Prithvi patch tensor differs from the frozen schema")
        if physics.shape[1:] != (9, GRID_SIZE, GRID_SIZE):
            raise ValueError("Physics patch tensor differs from the frozen schema")
        features = self.fusion(
            torch.cat((self.token_projection(tokens), self.physics_projection(physics)), dim=1)
        )
        patch_logits = self.patch_head(features)
        visible = physics[:, 8:9].clamp(0.0, 1.0)
        flat = features.flatten(2)
        visible_flat = visible.flatten(2)
        count = visible_flat.sum(dim=2).clamp_min(1e-4)
        average = (flat * visible_flat).sum(dim=2) / count
        maximum = flat.masked_fill(visible_flat <= 0.05, -1e4).amax(dim=2)
        attention_logits = patch_logits.flatten(2).masked_fill(visible_flat <= 0.05, -1e4)
        attention = torch.softmax(attention_logits, dim=2)
        attended = (flat * attention).sum(dim=2)
        scene_logits = self.scene_head(
            torch.cat((attended, average, maximum, self.sensor_embedding(sensors)), dim=1)
        ).squeeze(1)
        return {"scene_logits": scene_logits, "patch_logits": patch_logits}


def patch_supervision_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    labels: torch.Tensor,
    pixel_truth_available: torch.Tensor,
) -> torch.Tensor:
    """Per-scene observable patch BCE+Dice; missing positive masks are excluded."""
    plume = targets[:, 0:1].clamp(0.0, 1.0)
    visible = targets[:, 1:2].clamp(0.0, 1.0)
    supervised = (labels < 0.5) | pixel_truth_available.bool()
    valid = (visible > 0.05) & supervised[:, None, None, None]
    positive = (plume > 0.0) & valid
    negative_count = valid.sum().float() - positive.sum().float()
    positive_weight = torch.sqrt(
        negative_count / positive.sum().float().clamp_min(1.0)
    ).clamp(1.0, 20.0)
    bce = F.binary_cross_entropy_with_logits(
        logits,
        plume,
        reduction="none",
        pos_weight=positive_weight,
    )
    valid_float = valid.to(bce.dtype)
    per_scene_bce = (bce * valid_float).flatten(1).sum(dim=1) / valid_float.flatten(1).sum(
        dim=1
    ).clamp_min(1.0)
    probability = torch.sigmoid(logits) * valid_float
    truth = plume * valid_float
    intersection = (probability * truth).flatten(1).sum(dim=1)
    denominator = probability.flatten(1).sum(dim=1) + truth.flatten(1).sum(dim=1)
    dice = 1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)
    return torch.where(supervised, per_scene_bce + dice, torch.zeros_like(dice))


def augment_batch(
    tokens: torch.Tensor,
    physics: torch.Tensor,
    targets: torch.Tensor,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if torch.rand((), generator=generator).item() < 0.5:
        tokens, physics, targets = tokens.flip(-1), physics.flip(-1), targets.flip(-1)
    if torch.rand((), generator=generator).item() < 0.5:
        tokens, physics, targets = tokens.flip(-2), physics.flip(-2), targets.flip(-2)
    turns = int(torch.randint(0, 4, (), generator=generator).item())
    return tuple(torch.rot90(value, turns, dims=(-2, -1)) for value in (tokens, physics, targets))  # type: ignore[return-value]


def combine_partitions(parts: dict[str, dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    order = ("fold0", "fold1", "inner")
    counts = [parts[name]["labels"].size for name in order]
    return {
        "image_indices": np.concatenate([parts[name]["image_indices"] for name in order]),
        "labels": np.concatenate([parts[name]["labels"] for name in order]),
        "sensors": np.concatenate([parts[name]["sensors"] for name in order]),
        "groups": np.concatenate([parts[name]["groups"] for name in order]),
        "primary": np.concatenate([parts[name]["primary"] for name in order]),
        "current": np.concatenate([parts[name]["new"] for name in order]),
        "folds": np.concatenate(
            (
                np.zeros(counts[0], dtype=np.uint8),
                np.ones(counts[1], dtype=np.uint8),
                parts["inner"]["folds"].astype(np.uint8),
            )
        ),
    }


def physics_patches(images: np.ndarray, indices: np.ndarray) -> torch.Tensor:
    values = torch.from_numpy(np.asarray(images[indices], dtype=np.float32))
    return F.adaptive_avg_pool2d(values, GRID_SIZE)


def train_model(
    features: np.ndarray,
    targets: np.ndarray,
    physics: np.ndarray,
    pixel_truth_available: np.ndarray,
    values: dict[str, np.ndarray],
    fit: np.ndarray,
    *,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = SpatialPrithviHead().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    labels = values["labels"][fit]
    sensors = values["sensors"][fit]
    indices = values["image_indices"][fit]
    row_weights = sample_weights(
        "site_cell", values["groups"][fit], labels, sensors
    )
    positive_weight = min(
        4.0,
        float(
            np.sqrt(
                row_weights[labels == 0].sum()
                / max(row_weights[labels == 1].sum(), 1e-8)
            )
        ),
    )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    model.train()
    for epoch in range(EPOCHS):
        order = torch.randperm(labels.size, generator=generator).numpy()
        losses: list[float] = []
        for start in range(0, labels.size, 256):
            rows = order[start : start + 256]
            global_rows = indices[rows]
            token_batch = torch.from_numpy(
                np.asarray(features[global_rows], dtype=np.float32)
            )
            physics_batch = physics_patches(physics, global_rows)
            target_batch = torch.from_numpy(
                np.asarray(targets[global_rows], dtype=np.float32)
            )
            token_batch, physics_batch, target_batch = augment_batch(
                token_batch, physics_batch, target_batch, generator
            )
            token_batch = token_batch.to(device)
            physics_batch = physics_batch.to(device)
            target_batch = target_batch.to(device)
            label_batch = torch.from_numpy(labels[rows].astype(np.float32)).to(device)
            sensor_batch = torch.from_numpy(sensors[rows].astype(np.int64)).to(device)
            available_batch = torch.from_numpy(
                pixel_truth_available[global_rows].astype(np.bool_)
            ).to(device)
            weight_batch = torch.from_numpy(row_weights[rows].astype(np.float32)).to(device)
            class_weight = torch.where(label_batch > 0.5, positive_weight, 1.0)
            output = model(token_batch, physics_batch, sensor_batch)
            scene = F.binary_cross_entropy_with_logits(
                output["scene_logits"], label_batch, reduction="none"
            )
            patch = patch_supervision_loss(
                output["patch_logits"], target_batch, label_batch, available_batch
            )
            combined = scene + PATCH_LOSS_WEIGHT * patch
            weight = weight_batch * class_weight
            loss = (combined * weight).sum() / weight.sum().clamp_min(1e-8)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        print(
            json.dumps(
                {"seed": seed, "epoch": epoch + 1, "mean_loss": float(np.mean(losses))}
            ),
            flush=True,
        )
    return {
        "state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "seed": seed,
        "epochs": EPOCHS,
        "dropout": 0.3,
        "patch_loss_weight": PATCH_LOSS_WEIGHT,
        "positive_weight": positive_weight,
    }


@torch.no_grad()
def predict_model(
    fitted: dict[str, Any],
    features: np.ndarray,
    physics: np.ndarray,
    indices: np.ndarray,
    sensors: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    model = SpatialPrithviHead(float(fitted["dropout"])).to(device)
    model.load_state_dict(fitted["state_dict"])
    model.eval()
    scores: list[np.ndarray] = []
    for start in range(0, indices.size, 512):
        rows = slice(start, start + 512)
        global_rows = indices[rows]
        token_batch = torch.from_numpy(
            np.asarray(features[global_rows], dtype=np.float32)
        ).to(device)
        physics_batch = physics_patches(physics, global_rows).to(device)
        sensor_batch = torch.from_numpy(sensors[rows].astype(np.int64)).to(device)
        output = model(token_batch, physics_batch, sensor_batch)
        scores.append(torch.sigmoid(output["scene_logits"]).cpu().numpy())
    return np.concatenate(scores).astype(np.float64)


def crossfit(
    features: np.ndarray,
    targets: np.ndarray,
    physics: np.ndarray,
    pixel_truth_available: np.ndarray,
    values: dict[str, np.ndarray],
    seed: int,
    device: torch.device,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    scores = np.empty(values["labels"].shape, dtype=np.float64)
    models = []
    for holdout in FOLDS:
        fit = values["folds"] != holdout
        held = ~fit
        fitted = train_model(
            features,
            targets,
            physics,
            pixel_truth_available,
            values,
            fit,
            seed=seed + holdout,
            device=device,
        )
        scores[held] = predict_model(
            fitted,
            features,
            physics,
            values["image_indices"][held],
            values["sensors"][held],
            device,
        )
        models.append({"holdout": holdout, "fitted": fitted})
        print(json.dumps({"seed": seed, "completed_holdout": holdout}), flush=True)
    return scores, models


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    selected = report["selected"]
    delta = selected["versus_current"]["delta"]
    interval = selected["paired_group_bootstrap_ap_delta_vs_current"]
    lines = [
        "# Spatial Prithvi patch head",
        "",
        f"- Blend: {selected['blend_weight']:.2f}",
        f"- AP delta vs current: {delta['average_precision']:+.5f}",
        f"- Recall delta vs current: {delta['recall_at_fpr_0_0713']:+.5f}",
        f"- Paired-site AP interval: [{interval['lower']:+.5f}, {interval['upper']:+.5f}]",
        "",
        report["decision"],
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", default=DEFAULT_FEATURES.as_posix())
    parser.add_argument("--features-sha256", default=DEFAULT_FEATURES_SHA256)
    parser.add_argument("--targets", default=DEFAULT_TARGETS.as_posix())
    parser.add_argument("--targets-sha256", default=DEFAULT_TARGETS_SHA256)
    parser.add_argument("--metadata", default=DEFAULT_METADATA.as_posix())
    parser.add_argument("--metadata-sha256", default=DEFAULT_METADATA_SHA256)
    parser.add_argument("--physics", default=DEFAULT_PHYSICS.as_posix())
    parser.add_argument("--physics-sha256", default=DEFAULT_PHYSICS_SHA256)
    parser.add_argument("--physics-metadata", default=DEFAULT_PHYSICS_METADATA.as_posix())
    parser.add_argument("--physics-metadata-sha256", default=DEFAULT_PHYSICS_METADATA_SHA256)
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
        "features": (root / args.features).resolve(),
        "targets": (root / args.targets).resolve(),
        "metadata": (root / args.metadata).resolve(),
        "physics": (root / args.physics).resolve(),
        "physics_metadata": (root / args.physics_metadata).resolve(),
        "score": (root / args.score_cache).resolve(),
        "inner": (root / args.inner_cache).resolve(),
        "fold0": (root / args.fold0_cache).resolve(),
        "fold1": (root / args.fold1_cache).resolve(),
    }
    expected = {
        "features": args.features_sha256,
        "targets": args.targets_sha256,
        "metadata": args.metadata_sha256,
        "physics": args.physics_sha256,
        "physics_metadata": args.physics_metadata_sha256,
        "score": args.score_sha256,
        "inner": args.inner_sha256,
        "fold0": args.fold0_sha256,
        "fold1": args.fold1_sha256,
    }
    for name, digest in expected.items():
        if sha256(paths[name]) != digest:
            raise ValueError(f"Frozen {name} hash mismatch")
    features = np.load(paths["features"], mmap_mode="r", allow_pickle=False)
    targets = np.load(paths["targets"], mmap_mode="r", allow_pickle=False)
    physics = np.load(paths["physics"], mmap_mode="r", allow_pickle=False)
    with np.load(paths["metadata"], allow_pickle=False) as metadata:
        sample_ids = metadata["sample_ids"].astype(str)
        pixel_truth_available = metadata["pixel_truth_available"].astype(np.bool_)
        if metadata["feature_names"].astype(str).tolist() != feature_names():
            raise ValueError("Prithvi spatial feature schema differs")
    with np.load(paths["physics_metadata"], allow_pickle=False) as physics_metadata:
        if not np.array_equal(sample_ids, physics_metadata["sample_ids"].astype(str)):
            raise ValueError("Prithvi and physics image rows are not aligned")
    if features.shape != (sample_ids.size, len(feature_names()), GRID_SIZE, GRID_SIZE):
        raise ValueError("Prithvi spatial cache geometry differs")
    if targets.shape != (sample_ids.size, 2, GRID_SIZE, GRID_SIZE):
        raise ValueError("Prithvi spatial target geometry differs")
    parts = load_partitions(
        paths["metadata"], paths["score"], {name: paths[name] for name in ("inner", "fold0", "fold1")}
    )
    values = combine_partitions(parts)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_scores = []
    members = []
    for seed in SEEDS:
        raw, models = crossfit(
            features,
            targets,
            physics,
            pixel_truth_available,
            values,
            seed,
            device,
        )
        seed_scores.append(raw)
        members.extend(models)
    ensemble = np.mean(seed_scores, axis=0)
    candidates = []
    for blend in BLENDS:
        candidate = evaluate_candidate(values, ensemble, {"architecture": "prithvi_spatial"}, blend)
        candidate.update({"blend_weight": blend})
        seed_checks = []
        for raw in seed_scores:
            local = evaluate_candidate(values, raw, {"architecture": "prithvi_spatial"}, blend)
            seed_checks.append(
                bool(
                    local["versus_current"]["delta"]["average_precision"] > 0
                    and min(
                        fold["versus_current"]["delta"]["recall_at_fpr_0_0713"]
                        for fold in local["per_fold"].values()
                    )
                    >= 0
                )
            )
        candidate["all_seed_stability_pass"] = all(seed_checks)
        candidate["stable"] = bool(candidate["stable"] and all(seed_checks))
        candidate["rank"][0] = int(candidate["stable"])
        candidates.append(candidate)
    selected = max(candidates, key=lambda value: tuple(value["rank"]))
    scores = blend_scores(values["current"], ensemble, selected["blend_weight"])
    selected["paired_group_bootstrap_ap_delta_vs_primary"] = ap_group_bootstrap(
        values["labels"], values["primary"], scores, values["groups"], replicates=10_000, seed=20262120
    )
    selected["paired_group_bootstrap_ap_delta_vs_current"] = ap_group_bootstrap(
        values["labels"], values["current"], scores, values["groups"], replicates=10_000, seed=20262121
    )
    passed = bool(
        selected["stable"]
        and selected["paired_group_bootstrap_ap_delta_vs_primary"]["lower"] > 0
        and selected["paired_group_bootstrap_ap_delta_vs_current"]["lower"] > 0
    )
    artifact = (root / args.artifact).resolve()
    artifact_hash = None
    if passed:
        artifact.parent.mkdir(parents=True, exist_ok=True)
        temporary = artifact.with_suffix(artifact.suffix + ".tmp")
        torch.save(
            {
                "schema_version": 1,
                "kind": "mars_prithvi_spatial_crossfold_ensemble",
                "blend_weight": selected["blend_weight"],
                "seeds": SEEDS,
                "members": members,
            },
            temporary,
        )
        os.replace(temporary, artifact)
        artifact_hash = sha256(artifact)
    report = {
        "schema_version": 1,
        "scope": "two-seed five-fold spatial Prithvi development experiment; paper cache not loaded",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "architecture": {
            "blocks": list(BLOCKS),
            "grid_size": GRID_SIZE,
            "epochs": EPOCHS,
            "patch_loss_weight": PATCH_LOSS_WEIGHT,
        },
        "seeds": list(SEEDS),
        "blends": list(BLENDS),
        "selected": selected,
        "all_promotion_gates_pass": passed,
        "decision": (
            "Freeze the ten-member spatial Prithvi head for label-free paper scoring."
            if passed
            else "Reject the spatial Prithvi head before paper scoring."
        ),
        "provenance": {
            **{f"{name}_sha256": digest for name, digest in expected.items()},
            "artifact_sha256": artifact_hash,
            "script_sha256": sha256(Path(__file__).resolve()),
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True
            ).strip(),
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
    }
    write_json((root / args.output_json).resolve(), report)
    write_markdown((root / args.output_markdown).resolve(), report)
    print(
        json.dumps(
            {
                "ok": passed,
                "blend": selected["blend_weight"],
                "ap_delta": selected["versus_current"]["delta"]["average_precision"],
                "recall_delta": selected["versus_current"]["delta"]["recall_at_fpr_0_0713"],
                "ap_lower": selected["paired_group_bootstrap_ap_delta_vs_current"]["lower"],
                "artifact_sha256": artifact_hash,
            },
            indent=2,
        )
    )
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
