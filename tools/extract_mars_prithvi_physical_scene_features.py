#!/usr/bin/env python3
"""Extract Prithvi scene features with MARS radiometry restored to raw DN units.

The released MARS loader divides raw reflectance digital numbers by 5,000 for
its own U-Net. Prithvi's pinned normalization statistics are expressed in raw
reflectance x 10,000 units. Therefore the physically correct conversion from
the 16-channel MARS tensor back to Prithvi units is multiplication by 5,000,
not 10,000. This versioned extractor leaves all earlier caches untouched.
"""

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
from extract_mars_prithvi_scene_features import (  # noqa: E402
    INPUT_SIZE,
    build_features,
    date_coordinate,
    feature_names,
    reference_date_coordinate,
)
from extract_mars_scene_features import atomic_savez  # noqa: E402
from train_mars_paper_residual import (  # noqa: E402
    DEFAULT_ACQUISITION_RECEIPT,
    DEFAULT_MANIFEST,
    DEFAULT_PROTOCOL,
    MarsPaperDataset,
    iter_development_manifest,
    move_batch,
    verify_acquisition_receipt,
)

DEFAULT_FOUNDATION_DIR = Path(
    "EarthRemoteSensingRapidResponse/artifacts/foundation/prithvi_eo_2_tiny_tl"
)
DEFAULT_FOUNDATION_RECEIPT = Path("reports/acquisition/prithvi_eo_2_tiny_tl.json")
DEFAULT_OUTPUT_CACHE = Path(
    "outputs/mars_prithvi_eo_2_tiny_tl_physical_features_folds34.npz"
)
MARS_LOADER_DIVISOR = 5_000.0
PRITHVI_REFLECTANCE_SCALE = 10_000.0
MARS_TO_PRITHVI_MULTIPLIER = MARS_LOADER_DIVISOR


def build_physical_input(
    batch: dict[str, torch.Tensor], mean: torch.Tensor, std: torch.Tensor
) -> torch.Tensor:
    """Convert the MARS 16-channel contract to Prithvi's physical DN scale."""
    inputs = batch["inputs"]
    if inputs.ndim != 4 or inputs.shape[1] != 16:
        raise ValueError("Expected the frozen 16-channel MARS input")
    if mean.shape != (1, 6, 1, 1, 1) or std.shape != mean.shape:
        raise ValueError("Expected six-channel Prithvi normalization statistics")
    # MARS spectral layout is MBMP, target[6], reference[6], wind[2], cloud.
    # Prithvi expects chronological C,T,H,W: reference followed by target.
    frames = torch.stack([inputs[:, 7:13], inputs[:, 1:7]], dim=2)
    resized = F.interpolate(
        frames.permute(0, 2, 1, 3, 4).flatten(0, 1),
        size=(INPUT_SIZE, INPUT_SIZE),
        mode="bilinear",
        align_corners=False,
    ).reshape(inputs.shape[0], 2, 6, INPUT_SIZE, INPUT_SIZE).permute(0, 2, 1, 3, 4)
    observable = F.interpolate(
        batch["observable"].float(), size=(INPUT_SIZE, INPUT_SIZE), mode="nearest"
    )[:, :, None]
    normalized = (resized * MARS_TO_PRITHVI_MULTIPLIER - mean) / std
    return normalized.masked_fill(observable <= 0.5, 0.0)


@torch.no_grad()
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-dir", default=DEFAULT_OUTPUT.as_posix())
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST.as_posix())
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL.as_posix())
    parser.add_argument(
        "--acquisition-receipt", default=DEFAULT_ACQUISITION_RECEIPT.as_posix()
    )
    parser.add_argument("--foundation-dir", default=DEFAULT_FOUNDATION_DIR.as_posix())
    parser.add_argument(
        "--foundation-receipt", default=DEFAULT_FOUNDATION_RECEIPT.as_posix()
    )
    parser.add_argument("--folds", type=int, nargs="+", default=[3, 4])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--output", default=DEFAULT_OUTPUT_CACHE.as_posix())
    args = parser.parse_args()

    folds = tuple(sorted(set(args.folds)))
    if not folds or any(fold not in range(5) for fold in folds):
        parser.error("folds must be a non-empty subset of 0..4")
    if args.batch_size <= 0 or args.workers < 0:
        parser.error("batch size must be positive and workers non-negative")
    if args.max_rows is not None and args.max_rows <= 0:
        parser.error("max-rows must be positive when provided")

    root = repo_root()
    foundation_dir = (root / args.foundation_dir).resolve()
    foundation_receipt_path = (root / args.foundation_receipt).resolve()
    foundation_receipt = json.loads(
        foundation_receipt_path.read_text(encoding="utf-8")
    )
    for item in foundation_receipt["files"]:
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
    means, stds = means.to(device), stds.to(device)

    manifest = (root / args.manifest).resolve()
    fold_protocol_path = (root / args.protocol).resolve()
    fold_protocol = json.loads(fold_protocol_path.read_text(encoding="utf-8"))
    manifest_hash = sha256(manifest)
    if manifest_hash != fold_protocol["development_manifest_sha256"]:
        raise ValueError("Development manifest differs from the frozen fold protocol")
    verify_acquisition_receipt(
        (root / args.acquisition_receipt).resolve(), manifest_hash
    )
    group_to_fold = {
        str(item["group_id"]): int(item["fold"])
        for item in fold_protocol["assignments"]
    }
    all_records = list(iter_development_manifest(manifest))
    metadata = {str(record["sample_id"]): record for record in all_records}
    records = [
        record
        for record in all_records
        if group_to_fold[str(record["group_id"])] in folds
    ]
    if args.max_rows is not None:
        records = records[: args.max_rows]
    if not records:
        raise ValueError("No development rows selected")

    loader = DataLoader(
        MarsPaperDataset((root / args.metadata_dir).resolve(), records, augment=False, seed=0),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=False,
        persistent_workers=args.workers > 0,
    )
    rows: list[np.ndarray] = []
    labels: list[int] = []
    sensors: list[int] = []
    sample_ids: list[str] = []
    groups: list[str] = []
    row_folds: list[int] = []
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
        values = build_physical_input(batch, means, stds)
        with torch.amp.autocast(
            "cuda", dtype=torch.float16, enabled=device.type == "cuda"
        ):
            outputs = model.forward_features(values, temporal, location)
        rows.append(build_features(outputs).cpu().numpy().astype(np.float16))
        labels.extend(int(value) for value in batch["presence"].cpu().numpy())
        sensors.extend(int(value) for value in batch["sensor_index"].cpu().numpy())
        for sample_id, group in zip(local_ids, batch["group_id"]):
            sample_ids.append(sample_id)
            groups.append(str(group))
            row_folds.append(group_to_fold[str(group)])
        if batch_index % 10 == 0:
            print(json.dumps({"batches": batch_index, "rows": len(labels)}), flush=True)

    output_path = (root / args.output).resolve()
    atomic_savez(
        output_path,
        features=np.concatenate(rows),
        feature_names=np.asarray(feature_names()),
        labels=np.asarray(labels, dtype=np.uint8),
        sensors=np.asarray(sensors, dtype=np.uint8),
        sample_ids=np.asarray(sample_ids),
        groups=np.asarray(groups),
        folds=np.asarray(row_folds, dtype=np.uint8),
        foundation_revision=np.asarray(foundation_receipt["source"]["revision"]),
        checkpoint_sha256=np.asarray(sha256(checkpoint)),
        foundation_receipt_sha256=np.asarray(sha256(foundation_receipt_path)),
        manifest_sha256=np.asarray(manifest_hash),
        fold_protocol_sha256=np.asarray(sha256(fold_protocol_path)),
        mars_loader_divisor=np.asarray(MARS_LOADER_DIVISOR),
        prithvi_reflectance_scale=np.asarray(PRITHVI_REFLECTANCE_SCALE),
        mars_to_prithvi_multiplier=np.asarray(MARS_TO_PRITHVI_MULTIPLIER),
        input_contract=np.asarray(
            "reference then target; MARS normalized values x5000 restore raw reflectance DN; "
            "128x128 bilinear; pinned HLS means/std; temporal and location coordinates"
        ),
        nir_transfer_contract=np.asarray(
            "Prithvi HLS narrow-NIR slot receives Sentinel-2 B08 broad NIR or Landsat B05"
        ),
    )
    print(
        json.dumps(
            {
                "ok": True,
                "rows": len(labels),
                "features": len(feature_names()),
                "folds": list(folds),
                "input_size": INPUT_SIZE,
                "mars_to_prithvi_multiplier": MARS_TO_PRITHVI_MULTIPLIER,
                "output": args.output,
                "sha256": sha256(output_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
