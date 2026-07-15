#!/usr/bin/env python3
"""Extract label-independent Prithvi CLS features for the exact MARS paper cohort."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
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
from extract_mars_prithvi_scene_features import (  # noqa: E402
    BLOCKS,
    DEFAULT_FOUNDATION_DIR,
    DEFAULT_FOUNDATION_RECEIPT,
    EMBED_DIM,
    INPUT_SIZE,
    build_input,
    date_coordinate,
    feature_names,
    reference_date_coordinate,
)
from extract_mars_scene_features import atomic_savez  # noqa: E402
from mars_paper_model import SENSOR_NAMES  # noqa: E402
from train_mars_paper_residual import MarsPaperDataset, move_batch  # noqa: E402

DEFAULT_MANIFEST_SHA256 = "685937eb599ce4612116dc5c16edc74c4c0c6273f2fa864e651cfee73d1f4f6d"
DEFAULT_SHARD_COUNT = 5
CLS_WIDTH = len(BLOCKS) * EMBED_DIM


def cls_features(outputs: list[torch.Tensor]) -> torch.Tensor:
    if len(outputs) != 12:
        raise ValueError("Prithvi tiny encoder block count differs from the frozen schema")
    values = torch.cat([outputs[index - 1][:, 0].float() for index in BLOCKS], dim=1)
    if values.shape[1] != CLS_WIDTH or not torch.isfinite(values).all():
        raise RuntimeError("Prithvi CLS feature schema or finiteness failure")
    return values


@torch.no_grad()
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-dir", default=DEFAULT_OUTPUT.as_posix())
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST.as_posix())
    parser.add_argument("--manifest-sha256", default=DEFAULT_MANIFEST_SHA256)
    parser.add_argument("--receipt", default=DEFAULT_RECEIPT.as_posix())
    parser.add_argument("--foundation-dir", default=DEFAULT_FOUNDATION_DIR.as_posix())
    parser.add_argument("--foundation-receipt", default=DEFAULT_FOUNDATION_RECEIPT.as_posix())
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, default=DEFAULT_SHARD_COUNT)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
        parser.error("shard-index must be in [0, shard-count)")
    if args.batch_size <= 0 or args.workers < 0:
        parser.error("batch size must be positive and workers non-negative")
    root = repo_root()
    manifest = (root / args.manifest).resolve()
    records = load_sealed_records(manifest, args.manifest_sha256)
    verify_receipt((root / args.receipt).resolve(), args.manifest_sha256)
    boundaries = np.linspace(0, len(records), args.shard_count + 1, dtype=np.int64)
    start = int(boundaries[args.shard_index])
    end = int(boundaries[args.shard_index + 1])
    local_records = records[start:end]

    foundation_dir = (root / args.foundation_dir).resolve()
    foundation_receipt = (root / args.foundation_receipt).resolve()
    receipt = json.loads(foundation_receipt.read_text(encoding="utf-8"))
    for item in receipt["files"]:
        path = (root / item["path"]).resolve()
        if path.stat().st_size != int(item["bytes"]) or sha256(path) != item["sha256"]:
            raise ValueError(f"Prithvi acquisition identity mismatch: {item['path']}")
    sys.path.insert(0, str(foundation_dir))
    from prithvi_mae import PrithviMAE  # type: ignore  # noqa: E402

    config_path = foundation_dir / "config.json"
    checkpoint = foundation_dir / "Prithvi_EO_V2_tiny_TL.pt"
    config = json.loads(config_path.read_text(encoding="utf-8"))["pretrained_cfg"]
    means = torch.tensor(config["mean"], dtype=torch.float32)[None, :, None, None, None]
    stds = torch.tensor(config["std"], dtype=torch.float32)[None, :, None, None, None]
    model_config = dict(config)
    model_config.update(img_size=INPUT_SIZE, num_frames=2, in_chans=6)
    model = PrithviMAE(**model_config)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state["encoder.pos_embed"] = model.encoder.pos_embed
    state["decoder.decoder_pos_embed"] = model.decoder.decoder_pos_embed
    model.load_state_dict(state, strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    means = means.to(device)
    stds = stds.to(device)

    metadata = {str(record["sample_id"]): record for record in local_records}
    loader = DataLoader(
        MarsPaperDataset(
            (root / args.metadata_dir).resolve(),
            local_records,
            augment=False,
            seed=0,
            allow_missing_positive_mask=True,
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=False,
        persistent_workers=args.workers > 0,
    )
    rows: list[np.ndarray] = []
    sample_ids: list[str] = []
    for batch_index, batch in enumerate(loader, start=1):
        local_ids = [str(value) for value in batch["sample_id"]]
        temporal = torch.tensor(
            [
                [
                    reference_date_coordinate(metadata[sample_id]),
                    date_coordinate(str(metadata[sample_id]["target_datetime"])),
                ]
                for sample_id in local_ids
            ],
            dtype=torch.float32,
            device=device,
        )
        location = torch.tensor(
            [
                [
                    float(metadata[sample_id]["latitude"]),
                    float(metadata[sample_id]["longitude"]),
                ]
                for sample_id in local_ids
            ],
            dtype=torch.float32,
            device=device,
        )
        batch = move_batch(batch, device)
        values = build_input(batch, means, stds)
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            outputs = model.forward_features(values, temporal, location)
        rows.append(cls_features(outputs).cpu().numpy().astype(np.float16))
        sample_ids.extend(local_ids)
        if batch_index % 5 == 0:
            print(json.dumps({"batches": batch_index, "rows": len(sample_ids)}), flush=True)
    expected_ids = [str(record["sample_id"]) for record in local_records]
    if sample_ids != expected_ids:
        raise RuntimeError("Paper Prithvi shard row identity is incomplete")
    output_value = args.output or (
        f"outputs/mars_paper_prithvi_cls_shard{args.shard_index}of{args.shard_count}.npz"
    )
    output = (root / output_value).resolve()
    missing = sum(not str(record["reference_scene_id"]).strip() for record in local_records)
    atomic_savez(
        output,
        features=np.concatenate(rows),
        feature_names=np.asarray(feature_names()[:CLS_WIDTH]),
        sample_ids=np.asarray(sample_ids),
        groups=np.asarray([str(record["group_id"]) for record in local_records]),
        sensors=np.asarray(
            [SENSOR_NAMES.index(str(record["sensor_family"])) for record in local_records],
            dtype=np.uint8,
        ),
        shard_index=np.asarray(args.shard_index),
        shard_count=np.asarray(args.shard_count),
        shard_start=np.asarray(start),
        shard_end=np.asarray(end),
        total_available_rows=np.asarray(len(records)),
        foundation_revision=np.asarray(receipt["source"]["revision"]),
        checkpoint_sha256=np.asarray(sha256(checkpoint)),
        foundation_receipt_sha256=np.asarray(sha256(foundation_receipt)),
        sealed_manifest_sha256=np.asarray(args.manifest_sha256),
        test_acquisition_receipt_sha256=np.asarray(sha256((root / args.receipt).resolve())),
        input_contract=np.asarray(
            "reference then target; 128x128 bilinear; HLS means/std; temporal and location coordinates"
        ),
        nir_transfer_contract=np.asarray(
            "Prithvi HLS Narrow NIR slot receives MARS Sentinel-2 B08 broad NIR or Landsat B05"
        ),
        missing_reference_datetime_rows=np.asarray(missing),
        missing_reference_datetime_policy=np.asarray(
            "use target time coordinate for reference frame when reference_scene_id is blank"
        ),
    )
    print(
        json.dumps(
            {
                "ok": True,
                "label_independent_output": True,
                "shard_index": args.shard_index,
                "shard_count": args.shard_count,
                "rows": len(sample_ids),
                "features": CLS_WIDTH,
                "missing_reference_datetime_rows": missing,
                "output": output_value,
                "sha256": sha256(output),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
