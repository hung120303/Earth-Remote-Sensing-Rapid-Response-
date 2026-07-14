#!/usr/bin/env python3
"""Extract compact scene-ranking features from a frozen MARS residual model."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from acquire_mars_metadata import DEFAULT_OUTPUT, repo_root, sha256
from evaluate_mars_residual_endpoint_blend import load_residual_model, trust_region_logits
from evaluate_released_marss2l import connected_scene_score
from train_mars_paper_residual import (
    DEFAULT_ACQUISITION_RECEIPT,
    DEFAULT_CHECKPOINT,
    DEFAULT_MANIFEST,
    DEFAULT_PROTOCOL,
    MarsPaperDataset,
    iter_development_manifest,
    move_batch,
    verify_acquisition_receipt,
)

DEFAULT_ARTIFACT = Path(
    "EarthRemoteSensingRapidResponse/artifacts/mars_paper_residual_fold0_seed606.pt"
)
DEFAULT_ARTIFACT_SHA256 = (
    "b94880d858e1e7791591eeb5f7d0da9be84b99a324e980437ebe83cfae6c7d49"
)
DEFAULT_OUTPUT_CACHE = Path("outputs/mars_scene_features_folds234.npz")
TOP_COUNTS = (25, 50, 100, 200, 500, 1000, 2500)
AREA_THRESHOLDS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
REGION_COUNTS = (100, 500)


def tensor_feature_names() -> list[str]:
    names: list[str] = []
    for prefix in ("primary", "released"):
        names.extend(f"{prefix}_top_{count}_mean" for count in TOP_COUNTS)
        names.extend((f"{prefix}_valid_mean", f"{prefix}_valid_std"))
        names.extend(
            f"{prefix}_area_above_{threshold:.1f}" for threshold in AREA_THRESHOLDS
        )
    names.extend(
        (
            "logit_delta_valid_mean",
            "logit_delta_valid_std",
            "logit_delta_valid_min",
            "logit_delta_valid_max",
        )
    )
    for statistic in ("mean", "std"):
        names.extend(f"input_{channel}_{statistic}" for channel in range(16))
    for count in REGION_COUNTS:
        names.extend(f"input_{channel}_top_{count}_mean" for channel in range(16))
    names.extend(("clear_fraction", "observable_fraction"))
    return names


def _masked_mean_std(
    values: torch.Tensor, valid: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    weight = valid.to(values.dtype)
    count = weight.sum(dim=-1).clamp_min(1.0)
    mean = (values * weight).sum(dim=-1) / count
    variance = ((values - mean.unsqueeze(-1)).square() * weight).sum(dim=-1) / count
    return mean, variance.sqrt()


def pooled_scene_features(
    inputs: torch.Tensor,
    primary_logits: torch.Tensor,
    released_logits: torch.Tensor,
    clear: torch.Tensor,
    observable: torch.Tensor,
) -> torch.Tensor:
    """Return deterministic batch features; connected scores are added separately."""
    if inputs.ndim != 4 or inputs.shape[1] != 16:
        raise ValueError("Expected Bx16xHxW inputs")
    valid = (clear > 0.5) & (observable > 0.5)
    valid_flat = valid.flatten(1)
    valid_count = valid_flat.sum(dim=1).clamp_min(1).to(torch.float32)
    probabilities = []
    for logits in (primary_logits, released_logits):
        probabilities.append(
            torch.sigmoid(logits).float().masked_fill(~valid, 0.0).flatten(1)
        )

    parts: list[torch.Tensor] = []
    primary_indices: torch.Tensor | None = None
    maximum_count = min(TOP_COUNTS[-1], probabilities[0].shape[1])
    for probability in probabilities:
        top_values, top_indices = torch.topk(probability, k=maximum_count, dim=1)
        if primary_indices is None:
            primary_indices = top_indices
        cumulative = top_values.cumsum(dim=1)
        for count in TOP_COUNTS:
            local_count = min(count, maximum_count)
            parts.append((cumulative[:, local_count - 1] / local_count).unsqueeze(1))
        mean, std = _masked_mean_std(probability, valid_flat)
        parts.extend((mean.unsqueeze(1), std.unsqueeze(1)))
        parts.extend(
            ((probability > threshold).sum(dim=1).float() / valid_count).unsqueeze(1)
            for threshold in AREA_THRESHOLDS
        )

    delta = (primary_logits.float() - released_logits.float()).flatten(1)
    delta_mean, delta_std = _masked_mean_std(delta, valid_flat)
    delta_min = delta.masked_fill(~valid_flat, torch.inf).amin(dim=1)
    delta_max = delta.masked_fill(~valid_flat, -torch.inf).amax(dim=1)
    delta_min = torch.where(torch.isfinite(delta_min), delta_min, torch.zeros_like(delta_min))
    delta_max = torch.where(torch.isfinite(delta_max), delta_max, torch.zeros_like(delta_max))
    parts.extend(
        value.unsqueeze(1) for value in (delta_mean, delta_std, delta_min, delta_max)
    )

    flat_inputs = inputs.float().flatten(2)
    channel_valid = valid_flat[:, None, :]
    channel_mean, channel_std = _masked_mean_std(flat_inputs, channel_valid)
    parts.extend((channel_mean, channel_std))
    if primary_indices is None:
        raise RuntimeError("Primary top-pixel indices were not produced")
    for count in REGION_COUNTS:
        local_count = min(count, primary_indices.shape[1])
        indices = primary_indices[:, None, :local_count].expand(-1, 16, -1)
        parts.append(torch.gather(flat_inputs, dim=2, index=indices).mean(dim=2))

    total_pixels = float(valid_flat.shape[1])
    parts.extend(
        (
            (clear.flatten(1) > 0.5).sum(dim=1, keepdim=True).float() / total_pixels,
            (observable.flatten(1) > 0.5).sum(dim=1, keepdim=True).float() / total_pixels,
        )
    )
    features = torch.cat(parts, dim=1)
    expected = len(tensor_feature_names())
    if features.shape[1] != expected:
        raise RuntimeError(f"Feature width {features.shape[1]} does not match schema {expected}")
    if not torch.isfinite(features).all():
        raise RuntimeError("Scene feature extraction produced non-finite values")
    return features


def atomic_savez(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


@torch.no_grad()
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-dir", default=DEFAULT_OUTPUT.as_posix())
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST.as_posix())
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL.as_posix())
    parser.add_argument("--acquisition-receipt", default=DEFAULT_ACQUISITION_RECEIPT.as_posix())
    parser.add_argument("--released-checkpoint", default=DEFAULT_CHECKPOINT.as_posix())
    parser.add_argument("--artifact", default=DEFAULT_ARTIFACT.as_posix())
    parser.add_argument("--artifact-sha256", default=DEFAULT_ARTIFACT_SHA256)
    parser.add_argument("--folds", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", default=DEFAULT_OUTPUT_CACHE.as_posix())
    args = parser.parse_args()
    folds = tuple(sorted(set(args.folds)))
    if not folds or any(fold not in range(5) for fold in folds):
        parser.error("folds must be a non-empty subset of 0..4")
    if 1 in folds:
        parser.error("fold 1 is reserved for independent confirmation")
    if args.batch_size <= 0 or args.workers < 0:
        parser.error("batch size must be positive and workers non-negative")

    root = repo_root()
    manifest = (root / args.manifest).resolve()
    protocol_path = (root / args.protocol).resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    manifest_hash = sha256(manifest)
    if manifest_hash != protocol["development_manifest_sha256"]:
        raise ValueError("Development manifest differs from the frozen protocol")
    verify_acquisition_receipt((root / args.acquisition_receipt).resolve(), manifest_hash)
    artifact_path = (root / args.artifact).resolve()
    if sha256(artifact_path) != args.artifact_sha256:
        raise ValueError("Residual artifact hash mismatch")
    artifact = torch.load(artifact_path, map_location="cpu", weights_only=True)
    if int(artifact["fold"]) != 0 or artifact["protocol_sha256"] != sha256(protocol_path):
        raise ValueError("Residual artifact does not cover the frozen fold-0 protocol")

    group_to_fold = {
        str(item["group_id"]): int(item["fold"])
        for item in protocol["assignments"]
    }
    records = [
        record for record in iter_development_manifest(manifest)
        if group_to_fold[str(record["group_id"])] in folds
    ]
    loader = DataLoader(
        MarsPaperDataset((root / args.metadata_dir).resolve(), records, augment=False, seed=0),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_residual_model(
        (root / args.released_checkpoint).resolve(), artifact, device
    )
    rows: list[np.ndarray] = []
    labels: list[int] = []
    sensors: list[int] = []
    sample_ids: list[str] = []
    groups: list[str] = []
    row_folds: list[int] = []
    for batch_index, batch in enumerate(loader, start=1):
        batch = move_batch(batch, device)
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            output = model(batch["inputs"], batch["observable"], batch["sensor_index"])
        primary_logits = trust_region_logits(
            output["baseline_logits"], output["segmentation_logits"], 0.5
        )
        tensor_features = pooled_scene_features(
            batch["inputs"], primary_logits, output["baseline_logits"],
            batch["clear"], batch["observable"],
        ).cpu().numpy()
        primary_probability = torch.sigmoid(primary_logits).float().masked_fill(
            batch["clear"] <= 0.5, 0.0
        ).cpu().numpy()
        released_probability = torch.sigmoid(output["baseline_logits"]).float().masked_fill(
            batch["clear"] <= 0.5, 0.0
        ).cpu().numpy()
        for index in range(primary_probability.shape[0]):
            connected = np.asarray(
                [
                    connected_scene_score(primary_probability[index, 0]),
                    connected_scene_score(released_probability[index, 0]),
                ],
                dtype=np.float32,
            )
            rows.append(np.concatenate((connected, tensor_features[index])).astype(np.float32))
            labels.append(int(batch["presence"][index].item()))
            sensors.append(int(batch["sensor_index"][index].item()))
            sample_id = str(batch["sample_id"][index])
            group = str(batch["group_id"][index])
            sample_ids.append(sample_id)
            groups.append(group)
            row_folds.append(group_to_fold[group])
        if batch_index % 100 == 0:
            print(json.dumps({"batches": batch_index, "rows": len(rows)}), flush=True)

    feature_names = ["primary_connected_score", "released_connected_score", *tensor_feature_names()]
    output_path = (root / args.output).resolve()
    atomic_savez(
        output_path,
        features=np.stack(rows),
        feature_names=np.asarray(feature_names),
        labels=np.asarray(labels, dtype=np.uint8),
        sensors=np.asarray(sensors, dtype=np.uint8),
        sample_ids=np.asarray(sample_ids),
        groups=np.asarray(groups),
        folds=np.asarray(row_folds, dtype=np.uint8),
        artifact_sha256=np.asarray(args.artifact_sha256),
        manifest_sha256=np.asarray(manifest_hash),
        protocol_sha256=np.asarray(sha256(protocol_path)),
    )
    print(json.dumps({
        "ok": True, "rows": len(rows), "features": len(feature_names),
        "folds": list(folds), "output": args.output,
        "sha256": sha256(output_path),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
