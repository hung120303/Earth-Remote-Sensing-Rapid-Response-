#!/usr/bin/env python3
"""Extract frozen Prithvi-EO-2.0 temporal features for MARS development scenes."""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
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
from train_mars_paper_residual import (  # noqa: E402
    DEFAULT_ACQUISITION_RECEIPT,
    DEFAULT_MANIFEST,
    DEFAULT_PROTOCOL,
    MarsPaperDataset,
    iter_development_manifest,
    move_batch,
    verify_acquisition_receipt,
)

DEFAULT_FOUNDATION_DIR = Path("EarthRemoteSensingRapidResponse/artifacts/foundation/prithvi_eo_2_tiny_tl")
DEFAULT_FOUNDATION_RECEIPT = Path("reports/acquisition/prithvi_eo_2_tiny_tl.json")
DEFAULT_OUTPUT_CACHE = Path("outputs/mars_prithvi_eo_2_tiny_tl_features_all_folds.npz")
INPUT_SIZE = 128
EMBED_DIM = 192
BLOCKS = (3, 6, 9, 12)
STATISTICS = ("mean", "std", "max")
TEMPORAL_PARTS = ("reference", "target", "target_minus_reference", "absolute_difference")


def feature_names() -> list[str]:
    names = [
        f"prithvi_block{block}_cls_{channel}"
        for block in BLOCKS
        for channel in range(EMBED_DIM)
    ]
    names.extend(
        f"prithvi_{part}_{statistic}_{channel}"
        for part in TEMPORAL_PARTS
        for statistic in STATISTICS
        for channel in range(EMBED_DIM)
    )
    return names


def date_coordinate(value: str) -> tuple[int, int]:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed.year, parsed.timetuple().tm_yday


def scene_date_coordinate(scene_id: str) -> tuple[int, int]:
    match = re.search(r"(?:19|20)\d{6}", scene_id)
    if match is None:
        raise ValueError(f"Cannot parse reference date from scene id: {scene_id}")
    parsed = datetime.strptime(match.group(0), "%Y%m%d")
    return parsed.year, parsed.timetuple().tm_yday


def reference_date_coordinate(record: dict[str, object]) -> tuple[int, int]:
    """Return the reference acquisition coordinate, with a neutral missing-date policy."""
    scene_id = str(record["reference_scene_id"])
    if scene_id.strip():
        return scene_date_coordinate(scene_id)
    # MARS omits the reference product ID for a small minority of otherwise valid
    # pairs, and its GeoTIFF metadata contains no recoverable reference timestamp.
    # Equal frame coordinates express unknown/zero separation without inventing a date.
    return date_coordinate(str(record["target_datetime"]))


def token_statistics(values: torch.Tensor) -> torch.Tensor:
    if values.ndim != 3:
        raise ValueError("Expected BxTokensxChannels")
    return torch.cat(
        [values.mean(dim=1), values.std(dim=1, unbiased=False), values.amax(dim=1)],
        dim=1,
    )


def build_features(outputs: list[torch.Tensor]) -> torch.Tensor:
    if len(outputs) != 12:
        raise ValueError("Prithvi tiny encoder block count differs from the frozen schema")
    cls = [outputs[index - 1][:, 0].float() for index in BLOCKS]
    tokens = outputs[-1][:, 1:].float()
    if tokens.shape[1] % 2 != 0 or tokens.shape[2] != EMBED_DIM:
        raise ValueError("Prithvi temporal token shape differs from the frozen schema")
    frames = tokens.reshape(tokens.shape[0], 2, tokens.shape[1] // 2, EMBED_DIM)
    reference = frames[:, 0]
    target = frames[:, 1]
    difference = target - reference
    values = torch.cat(
        [
            *cls,
            token_statistics(reference),
            token_statistics(target),
            token_statistics(difference),
            token_statistics(difference.abs()),
        ],
        dim=1,
    )
    if values.shape[1] != len(feature_names()) or not torch.isfinite(values).all():
        raise RuntimeError("Prithvi transfer feature schema or finiteness failure")
    return values


def build_input(batch: dict[str, torch.Tensor], mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    inputs = batch["inputs"]
    if inputs.ndim != 4 or inputs.shape[1] != 16:
        raise ValueError("Expected the frozen 16-channel MARS input")
    # Prithvi expects chronological C,T,H,W ordering and reflectance scaled by 10,000.
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
    normalized = (resized * 10_000.0 - mean) / std
    return normalized.masked_fill(observable <= 0.5, 0.0)


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
    # Each MARS sample is comparatively large. Multiprocess prefetching duplicates enough
    # batch state to exhaust the default 16 GiB WSL VM, so the reproducible safe default is
    # deliberately single-process. Callers may opt in after measuring their host.
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
    receipt_path = (root / args.foundation_receipt).resolve()
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    means = means.to(device)
    stds = stds.to(device)

    manifest = (root / args.manifest).resolve()
    protocol_path = (root / args.protocol).resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    manifest_hash = sha256(manifest)
    if manifest_hash != protocol["development_manifest_sha256"]:
        raise ValueError("Development manifest differs from the frozen protocol")
    verify_acquisition_receipt((root / args.acquisition_receipt).resolve(), manifest_hash)
    group_to_fold = {
        str(item["group_id"]): int(item["fold"]) for item in protocol["assignments"]
    }
    all_records = list(iter_development_manifest(manifest))
    metadata = {str(record["sample_id"]): record for record in all_records}
    records = [
        record for record in all_records if group_to_fold[str(record["group_id"])] in folds
    ]
    if args.max_rows is not None:
        records = records[: args.max_rows]
    missing_reference_datetime_rows = sum(
        not str(record["reference_scene_id"]).strip() for record in records
    )
    loader = DataLoader(
        MarsPaperDataset((root / args.metadata_dir).resolve(), records, augment=False, seed=0),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        # WSL's CUDA host allocator retains each large pinned MARS batch and grows
        # shared memory across the shard. Blocking copies keep memory bounded and
        # are effectively free here because GeoTIFF reads dominate throughput.
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
                [float(metadata[sample_id]["latitude"]), float(metadata[sample_id]["longitude"])]
                for sample_id in local_ids
            ],
            dtype=torch.float32,
            device=device,
        )
        batch = move_batch(batch, device)
        values = build_input(batch, means, stds)
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            outputs = model.forward_features(values, temporal, location)
        rows.append(build_features(outputs).cpu().numpy().astype(np.float16))
        labels.extend(int(value) for value in batch["presence"].cpu().numpy())
        sensors.extend(int(value) for value in batch["sensor_index"].cpu().numpy())
        for sample_id, group in zip(local_ids, batch["group_id"]):
            sample_ids.append(sample_id)
            groups.append(str(group))
            row_folds.append(group_to_fold[str(group)])
        if batch_index % 5 == 0:
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
        foundation_revision=np.asarray(receipt["source"]["revision"]),
        checkpoint_sha256=np.asarray(sha256(checkpoint)),
        foundation_receipt_sha256=np.asarray(sha256(receipt_path)),
        manifest_sha256=np.asarray(manifest_hash),
        protocol_sha256=np.asarray(sha256(protocol_path)),
        input_contract=np.asarray(
            "reference then target; 128x128 bilinear; HLS means/std; temporal and location coordinates"
        ),
        nir_transfer_contract=np.asarray(
            "Prithvi HLS Narrow NIR slot receives MARS Sentinel-2 B08 broad NIR or Landsat B05"
        ),
        missing_reference_datetime_rows=np.asarray(missing_reference_datetime_rows),
        missing_reference_datetime_policy=np.asarray(
            "use target time coordinate for reference frame when reference_scene_id is blank"
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
                "missing_reference_datetime_rows": missing_reference_datetime_rows,
                "output": args.output,
                "sha256": sha256(output_path),
                "checkpoint_sha256": sha256(checkpoint),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
