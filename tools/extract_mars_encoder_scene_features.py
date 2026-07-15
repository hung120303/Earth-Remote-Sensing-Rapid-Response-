#!/usr/bin/env python3
"""Extract frozen multi-scale MARS U-Net encoder moments for scene probing."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "EarthRemoteSensingRapidResponse"
for path in (MODEL_ROOT, ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from acquire_mars_metadata import DEFAULT_OUTPUT, repo_root, sha256  # noqa: E402
from extract_mars_scene_features import atomic_savez  # noqa: E402
from mars_paper_model import ReleasedMarsUNet, released_state  # noqa: E402
from train_mars_paper_residual import (  # noqa: E402
    DEFAULT_ACQUISITION_RECEIPT,
    DEFAULT_CHECKPOINT,
    DEFAULT_MANIFEST,
    DEFAULT_PROTOCOL,
    MarsPaperDataset,
    iter_development_manifest,
    move_batch,
    verify_acquisition_receipt,
)

DEFAULT_OUTPUT_CACHE = Path("outputs/mars_encoder_scene_features_all_folds.npz")
LEVEL_CHANNELS = (("level3", 256), ("level4", 512), ("level5", 512))
STATISTICS = ("mean", "std", "max")


def encoder_feature_names() -> list[str]:
    return [
        f"{level}_channel_{channel}_{statistic}"
        for level, channels in LEVEL_CHANNELS
        for statistic in STATISTICS
        for channel in range(channels)
    ]


def masked_channel_moments(
    values: torch.Tensor, observable: torch.Tensor
) -> torch.Tensor:
    if values.ndim != 4 or observable.ndim != 4 or observable.shape[1] != 1:
        raise ValueError("Expected BxCxHxW values and Bx1xHxW observable mask")
    mask = F.interpolate(observable.float(), size=values.shape[-2:], mode="nearest") > 0.5
    flat = values.float().flatten(2)
    valid = mask.flatten(2)
    weight = valid.to(flat.dtype)
    count = weight.sum(dim=2).clamp_min(1.0)
    mean = (flat * weight).sum(dim=2) / count
    variance = ((flat - mean[:, :, None]).square() * weight).sum(dim=2) / count
    maximum = flat.masked_fill(~valid, -torch.inf).amax(dim=2)
    maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
    return torch.cat((mean, variance.sqrt(), maximum), dim=1)


def encoder_moments(
    model: ReleasedMarsUNet, inputs: torch.Tensor, observable: torch.Tensor
) -> torch.Tensor:
    level1 = model.inc(inputs)
    level2 = model.down1(level1)
    level3 = model.down2(level2)
    level4 = model.down3(level3)
    level5 = model.down4(level4)
    features = torch.cat(
        [
            masked_channel_moments(level, observable)
            for level in (level3, level4, level5)
        ],
        dim=1,
    )
    if features.shape[1] != len(encoder_feature_names()):
        raise RuntimeError("Encoder moment width differs from the frozen schema")
    if not torch.isfinite(features).all():
        raise RuntimeError("Encoder feature extraction produced non-finite values")
    return features


@torch.no_grad()
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-dir", default=DEFAULT_OUTPUT.as_posix())
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST.as_posix())
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL.as_posix())
    parser.add_argument("--acquisition-receipt", default=DEFAULT_ACQUISITION_RECEIPT.as_posix())
    parser.add_argument("--released-checkpoint", default=DEFAULT_CHECKPOINT.as_posix())
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", default=DEFAULT_OUTPUT_CACHE.as_posix())
    args = parser.parse_args()
    folds = tuple(sorted(set(args.folds)))
    if not folds or any(fold not in range(5) for fold in folds):
        parser.error("folds must be a non-empty subset of 0..4")
    if args.batch_size <= 0 or args.workers < 0:
        parser.error("batch size must be positive and workers non-negative")

    root = repo_root()
    manifest = (root / args.manifest).resolve()
    protocol_path = (root / args.protocol).resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    manifest_hash = sha256(manifest)
    protocol_hash = sha256(protocol_path)
    if manifest_hash != protocol["development_manifest_sha256"]:
        raise ValueError("Development manifest differs from the frozen protocol")
    verify_acquisition_receipt((root / args.acquisition_receipt).resolve(), manifest_hash)
    group_to_fold = {
        str(item["group_id"]): int(item["fold"]) for item in protocol["assignments"]
    }
    records = [
        record
        for record in iter_development_manifest(manifest)
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
    checkpoint = (root / args.released_checkpoint).resolve()
    model = ReleasedMarsUNet().to(device)
    model.load_state_dict(released_state(checkpoint), strict=False)
    model.eval()

    rows: list[np.ndarray] = []
    labels: list[int] = []
    sensors: list[int] = []
    sample_ids: list[str] = []
    groups: list[str] = []
    row_folds: list[int] = []
    for batch_index, batch in enumerate(loader, start=1):
        batch = move_batch(batch, device)
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            features = encoder_moments(model, batch["inputs"], batch["observable"])
        rows.append(features.cpu().numpy().astype(np.float16))
        labels.extend(int(value) for value in batch["presence"].cpu().numpy())
        sensors.extend(int(value) for value in batch["sensor_index"].cpu().numpy())
        for sample_id, group in zip(batch["sample_id"], batch["group_id"]):
            sample_ids.append(str(sample_id))
            groups.append(str(group))
            row_folds.append(group_to_fold[str(group)])
        if batch_index % 100 == 0:
            print(
                json.dumps({"batches": batch_index, "rows": len(labels)}),
                flush=True,
            )

    output_path = (root / args.output).resolve()
    atomic_savez(
        output_path,
        features=np.concatenate(rows),
        feature_names=np.asarray(encoder_feature_names()),
        labels=np.asarray(labels, dtype=np.uint8),
        sensors=np.asarray(sensors, dtype=np.uint8),
        sample_ids=np.asarray(sample_ids),
        groups=np.asarray(groups),
        folds=np.asarray(row_folds, dtype=np.uint8),
        checkpoint_sha256=np.asarray(sha256(checkpoint)),
        manifest_sha256=np.asarray(manifest_hash),
        protocol_sha256=np.asarray(protocol_hash),
    )
    print(
        json.dumps(
            {
                "ok": True,
                "rows": len(labels),
                "features": len(encoder_feature_names()),
                "folds": list(folds),
                "output": args.output,
                "sha256": sha256(output_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
