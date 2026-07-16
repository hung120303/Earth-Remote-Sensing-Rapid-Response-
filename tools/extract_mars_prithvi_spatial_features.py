#!/usr/bin/env python3
"""Extract spatial multi-depth Prithvi patch differences for MARS development scenes."""

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
from torch.nn import functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "EarthRemoteSensingRapidResponse"
for path in (MODEL_ROOT, ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from acquire_mars_metadata import DEFAULT_OUTPUT, repo_root, sha256  # noqa: E402
from extract_mars_prithvi_scene_features import (  # noqa: E402
    BLOCKS,
    EMBED_DIM,
    INPUT_SIZE,
    build_input,
    date_coordinate,
    reference_date_coordinate,
)
from extract_mars_scene_features import atomic_savez  # noqa: E402
from mars_paper_model import SENSOR_NAMES  # noqa: E402
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
DEFAULT_OUTPUT_FEATURES = Path("outputs/mars_prithvi_spatial_features_all_folds.npy")
DEFAULT_OUTPUT_TARGETS = Path("outputs/mars_prithvi_spatial_targets_all_folds.npy")
DEFAULT_OUTPUT_METADATA = Path("outputs/mars_prithvi_spatial_features_all_folds_metadata.npz")
PATCH_SIZE = 16
GRID_SIZE = INPUT_SIZE // PATCH_SIZE
TARGET_NAMES = ("plume_fraction", "observable_fraction")


def feature_names() -> list[str]:
    return [
        f"prithvi_block{block}_target_minus_reference_{channel}"
        for block in BLOCKS
        for channel in range(EMBED_DIM)
    ]


def spatial_patch_differences(outputs: list[torch.Tensor]) -> torch.Tensor:
    """Return B,C,H,W signed temporal patch differences from selected depths."""
    if len(outputs) != 12:
        raise ValueError("Prithvi tiny encoder block count differs from the frozen schema")
    maps: list[torch.Tensor] = []
    expected_tokens = 2 * GRID_SIZE * GRID_SIZE
    for block in BLOCKS:
        tokens = outputs[block - 1][:, 1:].float()
        if tokens.shape[1:] != (expected_tokens, EMBED_DIM):
            raise ValueError("Prithvi temporal patch geometry differs from the frozen schema")
        frames = tokens.reshape(
            tokens.shape[0], 2, GRID_SIZE, GRID_SIZE, EMBED_DIM
        )
        maps.append((frames[:, 1] - frames[:, 0]).permute(0, 3, 1, 2))
    values = torch.cat(maps, dim=1)
    if values.shape[1:] != (len(feature_names()), GRID_SIZE, GRID_SIZE):
        raise RuntimeError("Spatial Prithvi feature width differs from its frozen schema")
    if not torch.isfinite(values).all():
        raise RuntimeError("Spatial Prithvi extraction produced non-finite values")
    return values


def pooled_patch_targets(mask: torch.Tensor, observable: torch.Tensor) -> torch.Tensor:
    if mask.ndim != 4 or mask.shape[1] != 1 or observable.shape != mask.shape:
        raise ValueError("Mask and observable tensors must align as Bx1xHxW")
    plume = F.adaptive_avg_pool2d(mask.float() * observable.float(), GRID_SIZE)
    visible = F.adaptive_avg_pool2d(observable.float(), GRID_SIZE)
    values = torch.cat((plume, visible), dim=1)
    if values.shape[1:] != (len(TARGET_NAMES), GRID_SIZE, GRID_SIZE):
        raise RuntimeError("Spatial Prithvi target geometry differs from its frozen schema")
    return values


def write_state(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_foundation(
    root: Path, foundation_dir: Path, receipt_path: Path
) -> tuple[torch.nn.Module, torch.Tensor, torch.Tensor, dict[str, Any]]:
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
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
    return model, means, stds, receipt


@torch.no_grad()
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-dir", default=DEFAULT_OUTPUT.as_posix())
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST.as_posix())
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL.as_posix())
    parser.add_argument("--acquisition-receipt", default=DEFAULT_ACQUISITION_RECEIPT.as_posix())
    parser.add_argument("--foundation-dir", default=DEFAULT_FOUNDATION_DIR.as_posix())
    parser.add_argument("--foundation-receipt", default=DEFAULT_FOUNDATION_RECEIPT.as_posix())
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output-features", default=DEFAULT_OUTPUT_FEATURES.as_posix())
    parser.add_argument("--output-targets", default=DEFAULT_OUTPUT_TARGETS.as_posix())
    parser.add_argument("--output-metadata", default=DEFAULT_OUTPUT_METADATA.as_posix())
    args = parser.parse_args()
    folds = tuple(sorted(set(args.folds)))
    if not folds or any(fold not in range(5) for fold in folds):
        parser.error("folds must be a non-empty subset of 0..4")
    if args.batch_size <= 0 or args.workers < 0:
        parser.error("batch size must be positive and workers non-negative")
    if args.max_rows is not None and args.max_rows <= 0:
        parser.error("max-rows must be positive when provided")

    root = repo_root()
    manifest = (root / args.manifest).resolve()
    protocol_path = (root / args.protocol).resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    manifest_hash = sha256(manifest)
    protocol_hash = sha256(protocol_path)
    if manifest_hash != protocol["development_manifest_sha256"]:
        raise ValueError("Development manifest differs from the frozen protocol")
    acquisition_receipt = (root / args.acquisition_receipt).resolve()
    verify_acquisition_receipt(acquisition_receipt, manifest_hash)
    group_to_fold = {
        str(item["group_id"]): int(item["fold"]) for item in protocol["assignments"]
    }
    all_records = list(iter_development_manifest(manifest))
    record_by_id = {str(record["sample_id"]): record for record in all_records}
    records = [
        record
        for record in all_records
        if group_to_fold[str(record["group_id"])] in folds
    ]
    if args.max_rows is not None:
        records = records[: args.max_rows]
    if not records:
        raise ValueError("No development records selected")

    foundation_dir = (root / args.foundation_dir).resolve()
    foundation_receipt = (root / args.foundation_receipt).resolve()
    model, means, stds, receipt = load_foundation(
        root, foundation_dir, foundation_receipt
    )
    checkpoint = foundation_dir / "Prithvi_EO_V2_tiny_TL.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    means, stds = means.to(device), stds.to(device)

    output_features = (root / args.output_features).resolve()
    output_targets = (root / args.output_targets).resolve()
    output_metadata = (root / args.output_metadata).resolve()
    output_features.parent.mkdir(parents=True, exist_ok=True)
    feature_tmp = output_features.with_suffix(".tmp.npy")
    target_tmp = output_targets.with_suffix(".tmp.npy")
    state_path = output_features.with_suffix(".state.json")
    feature_shape = (len(records), len(feature_names()), GRID_SIZE, GRID_SIZE)
    target_shape = (len(records), len(TARGET_NAMES), GRID_SIZE, GRID_SIZE)
    identity = {
        "schema_version": 1,
        "folds": list(folds),
        "rows": len(records),
        "feature_shape": list(feature_shape),
        "target_shape": list(target_shape),
        "manifest_sha256": manifest_hash,
        "protocol_sha256": protocol_hash,
        "foundation_receipt_sha256": sha256(foundation_receipt),
        "checkpoint_sha256": sha256(checkpoint),
    }
    cursor = 0
    if args.resume:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if {key: state[key] for key in identity} != identity:
            raise ValueError("Resume state differs from the requested extraction identity")
        cursor = int(state["cursor"])
        features = open_memmap(feature_tmp, mode="r+")
        targets = open_memmap(target_tmp, mode="r+")
        if features.shape != feature_shape or targets.shape != target_shape:
            raise ValueError("Resume memmap geometry differs from the frozen schema")
    else:
        for path in (feature_tmp, target_tmp, state_path):
            if path.exists():
                raise FileExistsError(f"Stale extraction state exists: {path}")
        features = open_memmap(feature_tmp, mode="w+", dtype=np.float16, shape=feature_shape)
        targets = open_memmap(target_tmp, mode="w+", dtype=np.float16, shape=target_shape)
        write_state(state_path, {**identity, "cursor": 0})

    loader = DataLoader(
        MarsPaperDataset(
            (root / args.metadata_dir).resolve(), records[cursor:], augment=False, seed=0
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=False,
        persistent_workers=args.workers > 0,
    )
    for batch_index, batch in enumerate(loader, start=1):
        local_ids = [str(value) for value in batch["sample_id"]]
        temporal = torch.tensor(
            [
                [
                    reference_date_coordinate(record_by_id[sample_id]),
                    date_coordinate(str(record_by_id[sample_id]["target_datetime"])),
                ]
                for sample_id in local_ids
            ],
            dtype=torch.float32,
            device=device,
        )
        location = torch.tensor(
            [
                [
                    float(record_by_id[sample_id]["latitude"]),
                    float(record_by_id[sample_id]["longitude"]),
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
        spatial = spatial_patch_differences(outputs).cpu().numpy().astype(np.float16)
        patch_targets = pooled_patch_targets(batch["mask"], batch["observable"])
        patch_targets = patch_targets.cpu().numpy().astype(np.float16)
        end = cursor + spatial.shape[0]
        features[cursor:end] = spatial
        targets[cursor:end] = patch_targets
        cursor = end
        if batch_index % 25 == 0 or cursor == len(records):
            features.flush()
            targets.flush()
            write_state(state_path, {**identity, "cursor": cursor})
            print(json.dumps({"batches": batch_index, "rows": cursor}), flush=True)
    if cursor != len(records):
        raise RuntimeError("Spatial Prithvi extraction row count is incomplete")
    features.flush()
    targets.flush()
    del features, targets
    feature_hash = sha256(feature_tmp)
    target_hash = sha256(target_tmp)
    os.replace(feature_tmp, output_features)
    os.replace(target_tmp, output_targets)
    state_path.unlink()

    atomic_savez(
        output_metadata,
        feature_names=np.asarray(feature_names()),
        target_names=np.asarray(TARGET_NAMES),
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
        pixel_truth_available=np.asarray(
            [bool(record.get("pixel_truth_available", True)) for record in records],
            dtype=np.bool_,
        ),
        features_sha256=np.asarray(feature_hash),
        targets_sha256=np.asarray(target_hash),
        checkpoint_sha256=np.asarray(sha256(checkpoint)),
        foundation_receipt_sha256=np.asarray(sha256(foundation_receipt)),
        foundation_revision=np.asarray(receipt["source"]["revision"]),
        manifest_sha256=np.asarray(manifest_hash),
        protocol_sha256=np.asarray(protocol_hash),
        input_contract=np.asarray(
            "reference then target; 128x128 bilinear; four encoder depths; signed target-reference patch tokens"
        ),
    )
    print(
        json.dumps(
            {
                "ok": True,
                "rows": len(records),
                "shape": list(feature_shape),
                "features_sha256": feature_hash,
                "targets_sha256": target_hash,
                "metadata_sha256": sha256(output_metadata),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
