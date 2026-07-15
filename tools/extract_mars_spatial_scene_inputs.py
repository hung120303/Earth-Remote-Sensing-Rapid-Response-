#!/usr/bin/env python3
"""Extract compact physics-guided spatial inputs for MARS scene classification."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from numpy.lib.format import open_memmap
from torch.nn import functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "EarthRemoteSensingRapidResponse"
for path in (MODEL_ROOT, ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from acquire_mars_metadata import DEFAULT_OUTPUT, repo_root, sha256  # noqa: E402
from extract_mars_scene_features import atomic_savez  # noqa: E402
from mars_paper_model import ReleasedMarsUNet, SENSOR_NAMES, released_state  # noqa: E402
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

DEFAULT_OUTPUT_IMAGES = Path("outputs/mars_spatial_scene_inputs_all_folds.npy")
DEFAULT_OUTPUT_METADATA = Path("outputs/mars_spatial_scene_inputs_all_folds_metadata.npz")
OUTPUT_SIZE = 64
CHANNEL_NAMES = (
    "released_probability_meanpool",
    "released_probability_maxpool",
    "mbmp_centered",
    "target_reference_B11_difference",
    "target_reference_B12_difference",
    "target_reference_B11_normalized_difference",
    "target_reference_B12_normalized_difference",
    "cloud_fraction",
    "observable_fraction",
)


def spatial_scene_channels(
    inputs: torch.Tensor,
    released_logits: torch.Tensor,
    observable: torch.Tensor,
    *,
    output_size: int = OUTPUT_SIZE,
) -> torch.Tensor:
    if inputs.ndim != 4 or inputs.shape[1] != 16:
        raise ValueError("Expected Bx16xHxW MARS inputs")
    if released_logits.shape != inputs[:, :1].shape or observable.shape != inputs[:, :1].shape:
        raise ValueError("Released logits and observable mask must align with MARS inputs")
    if output_size <= 0:
        raise ValueError("output size must be positive")
    probability = torch.sigmoid(released_logits.float())
    b11_difference = inputs[:, 5:6].float() - inputs[:, 11:12].float()
    b12_difference = inputs[:, 6:7].float() - inputs[:, 12:13].float()
    b11_normalized = b11_difference / (
        inputs[:, 5:6].float().abs() + inputs[:, 11:12].float().abs() + 0.02
    )
    b12_normalized = b12_difference / (
        inputs[:, 6:7].float().abs() + inputs[:, 12:13].float().abs() + 0.02
    )
    continuous = torch.cat(
        [
            probability,
            inputs[:, 0:1].float() - 1.0,
            b11_difference,
            b12_difference,
            b11_normalized,
            b12_normalized,
            inputs[:, 15:16].float(),
            observable.float(),
        ],
        dim=1,
    )
    means = F.adaptive_avg_pool2d(continuous, (output_size, output_size))
    probability_max = F.adaptive_max_pool2d(probability, (output_size, output_size))
    values = torch.cat([means[:, :1], probability_max, means[:, 1:]], dim=1)
    if values.shape[1] != len(CHANNEL_NAMES):
        raise RuntimeError("Spatial scene channel width differs from its frozen schema")
    if not torch.isfinite(values).all():
        raise RuntimeError("Spatial scene extraction produced non-finite values")
    return values


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
    parser.add_argument("--output-images", default=DEFAULT_OUTPUT_IMAGES.as_posix())
    parser.add_argument("--output-metadata", default=DEFAULT_OUTPUT_METADATA.as_posix())
    parser.add_argument("--finalize-existing", action="store_true")
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
    output_images = (root / args.output_images).resolve()
    output_metadata = (root / args.output_metadata).resolve()
    expected_shape = (len(records), len(CHANNEL_NAMES), OUTPUT_SIZE, OUTPUT_SIZE)
    if args.finalize_existing:
        existing = np.load(output_images, mmap_mode="r", allow_pickle=False)
        if existing.shape != expected_shape or existing.dtype != np.float16:
            raise ValueError("Existing spatial image cache differs from the frozen schema")
        del existing
        images_hash = sha256(output_images)
        atomic_savez(
            output_metadata,
            channel_names=np.asarray(CHANNEL_NAMES),
            labels=np.asarray(
                [int(record["label_state"] == "PLUME") for record in records], dtype=np.uint8
            ),
            sensors=np.asarray(
                [SENSOR_NAMES.index(str(record["sensor_family"])) for record in records],
                dtype=np.uint8,
            ),
            sample_ids=np.asarray([str(record["sample_id"]) for record in records]),
            groups=np.asarray([str(record["group_id"]) for record in records]),
            folds=np.asarray(
                [group_to_fold[str(record["group_id"])] for record in records], dtype=np.uint8
            ),
            images_sha256=np.asarray(images_hash),
            checkpoint_sha256=np.asarray(sha256((root / args.released_checkpoint).resolve())),
            manifest_sha256=np.asarray(manifest_hash),
            protocol_sha256=np.asarray(protocol_hash),
        )
        print(
            json.dumps(
                {
                    "ok": True,
                    "finalized_existing": True,
                    "shape": list(expected_shape),
                    "images_sha256": images_hash,
                    "metadata_sha256": sha256(output_metadata),
                },
                indent=2,
            )
        )
        return 0
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

    output_images.parent.mkdir(parents=True, exist_ok=True)
    temporary_images = output_images.with_suffix(".tmp.npy")
    images = open_memmap(
        temporary_images,
        mode="w+",
        dtype=np.float16,
        shape=expected_shape,
    )
    labels: list[int] = []
    sensors: list[int] = []
    sample_ids: list[str] = []
    groups: list[str] = []
    row_folds: list[int] = []
    cursor = 0
    for batch_index, batch in enumerate(loader, start=1):
        batch = move_batch(batch, device)
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            released_logits = model(batch["inputs"])
        values = spatial_scene_channels(
            batch["inputs"], released_logits, batch["observable"]
        ).cpu().numpy().astype(np.float16)
        end = cursor + values.shape[0]
        images[cursor:end] = values
        labels.extend(int(value) for value in batch["presence"].cpu().numpy())
        sensors.extend(int(value) for value in batch["sensor_index"].cpu().numpy())
        for sample_id, group in zip(batch["sample_id"], batch["group_id"]):
            sample_ids.append(str(sample_id))
            groups.append(str(group))
            row_folds.append(group_to_fold[str(group)])
        cursor = end
        if batch_index % 100 == 0:
            images.flush()
            print(json.dumps({"batches": batch_index, "rows": cursor}), flush=True)
    if cursor != len(records):
        raise RuntimeError("Spatial scene extraction row count is incomplete")
    images.flush()
    del images
    images_hash = sha256(temporary_images)
    os.replace(temporary_images, output_images)

    atomic_savez(
        output_metadata,
        channel_names=np.asarray(CHANNEL_NAMES),
        labels=np.asarray(labels, dtype=np.uint8),
        sensors=np.asarray(sensors, dtype=np.uint8),
        sample_ids=np.asarray(sample_ids),
        groups=np.asarray(groups),
        folds=np.asarray(row_folds, dtype=np.uint8),
        images_sha256=np.asarray(images_hash),
        checkpoint_sha256=np.asarray(sha256(checkpoint)),
        manifest_sha256=np.asarray(manifest_hash),
        protocol_sha256=np.asarray(protocol_hash),
    )
    print(
        json.dumps(
            {
                "ok": True,
                "rows": cursor,
                "channels": list(CHANNEL_NAMES),
                "shape": [cursor, len(CHANNEL_NAMES), OUTPUT_SIZE, OUTPUT_SIZE],
                "folds": list(folds),
                "output_images": args.output_images,
                "images_sha256": images_hash,
                "output_metadata": args.output_metadata,
                "metadata_sha256": sha256(output_metadata),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
