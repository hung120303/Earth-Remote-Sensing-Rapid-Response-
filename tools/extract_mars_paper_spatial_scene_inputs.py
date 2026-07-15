#!/usr/bin/env python3
"""Extract label-independent spatial inputs for the sealed MARS paper cohort."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.lib.format import open_memmap
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "EarthRemoteSensingRapidResponse"
for path in (MODEL_ROOT, ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from acquire_mars_metadata import DEFAULT_OUTPUT, repo_root, sha256  # noqa: E402
from evaluate_mars_successor_paper_test import (  # noqa: E402
    DEFAULT_MANIFEST,
    DEFAULT_RECEIPT,
    load_sealed_records,
    verify_receipt,
)
from extract_mars_scene_features import atomic_savez  # noqa: E402
from extract_mars_spatial_scene_inputs import (  # noqa: E402
    CHANNEL_NAMES,
    OUTPUT_SIZE,
    spatial_scene_channels,
)
from mars_paper_model import ReleasedMarsUNet, SENSOR_NAMES, released_state  # noqa: E402
from train_mars_paper_residual import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    MarsPaperDataset,
    move_batch,
)

DEFAULT_MANIFEST_SHA256 = "685937eb599ce4612116dc5c16edc74c4c0c6273f2fa864e651cfee73d1f4f6d"
DEFAULT_OUTPUT_IMAGES = Path("outputs/mars_paper_spatial_scene_inputs.npy")
DEFAULT_OUTPUT_METADATA = Path("outputs/mars_paper_spatial_scene_inputs_metadata.npz")


def label_free_metadata(records: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    return {
        "sample_ids": np.asarray([str(record["sample_id"]) for record in records]),
        "groups": np.asarray([str(record["group_id"]) for record in records]),
        "sensors": np.asarray(
            [SENSOR_NAMES.index(str(record["sensor_family"])) for record in records],
            dtype=np.uint8,
        ),
    }


@torch.no_grad()
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-dir", default=DEFAULT_OUTPUT.as_posix())
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST.as_posix())
    parser.add_argument("--manifest-sha256", default=DEFAULT_MANIFEST_SHA256)
    parser.add_argument("--receipt", default=DEFAULT_RECEIPT.as_posix())
    parser.add_argument("--released-checkpoint", default=DEFAULT_CHECKPOINT.as_posix())
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output-images", default=DEFAULT_OUTPUT_IMAGES.as_posix())
    parser.add_argument("--output-metadata", default=DEFAULT_OUTPUT_METADATA.as_posix())
    parser.add_argument("--finalize-existing", action="store_true")
    args = parser.parse_args()
    if args.batch_size <= 0 or args.workers < 0:
        parser.error("batch size must be positive and workers non-negative")

    root = repo_root()
    manifest = (root / args.manifest).resolve()
    records = load_sealed_records(manifest, args.manifest_sha256)
    verify_receipt((root / args.receipt).resolve(), args.manifest_sha256)
    output_images = (root / args.output_images).resolve()
    output_metadata = (root / args.output_metadata).resolve()
    expected_shape = (len(records), len(CHANNEL_NAMES), OUTPUT_SIZE, OUTPUT_SIZE)
    metadata = label_free_metadata(records)
    checkpoint = (root / args.released_checkpoint).resolve()
    if args.finalize_existing:
        existing = np.load(output_images, mmap_mode="r", allow_pickle=False)
        if existing.shape != expected_shape or existing.dtype != np.float16:
            raise ValueError("Existing paper spatial cache differs from the frozen schema")
        del existing
        images_hash = sha256(output_images)
        atomic_savez(
            output_metadata,
            channel_names=np.asarray(CHANNEL_NAMES),
            **metadata,
            images_sha256=np.asarray(images_hash),
            checkpoint_sha256=np.asarray(sha256(checkpoint)),
            sealed_manifest_sha256=np.asarray(args.manifest_sha256),
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
        MarsPaperDataset(
            (root / args.metadata_dir).resolve(),
            records,
            augment=False,
            seed=0,
            allow_missing_positive_mask=True,
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    extracted_ids: list[str] = []
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
        extracted_ids.extend(str(sample_id) for sample_id in batch["sample_id"])
        cursor = end
        if batch_index % 100 == 0:
            images.flush()
            print(json.dumps({"batches": batch_index, "rows": cursor}), flush=True)
    if cursor != len(records) or not np.array_equal(
        np.asarray(extracted_ids), metadata["sample_ids"]
    ):
        raise RuntimeError("Paper spatial extraction row identity is incomplete")
    images.flush()
    del images
    images_hash = sha256(temporary_images)
    os.replace(temporary_images, output_images)
    atomic_savez(
        output_metadata,
        channel_names=np.asarray(CHANNEL_NAMES),
        **metadata,
        images_sha256=np.asarray(images_hash),
        checkpoint_sha256=np.asarray(sha256(checkpoint)),
        sealed_manifest_sha256=np.asarray(args.manifest_sha256),
    )
    print(
        json.dumps(
            {
                "ok": True,
                "label_independent": True,
                "rows": cursor,
                "shape": list(expected_shape),
                "images_sha256": images_hash,
                "metadata_sha256": sha256(output_metadata),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
