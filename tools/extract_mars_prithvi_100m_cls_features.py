#!/usr/bin/env python3
"""Extract frozen multiscale CLS features from Prithvi-EO-2.0-100M-TL."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
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
from extract_mars_prithvi_scene_features import (  # noqa: E402
    INPUT_SIZE,
    build_input,
    date_coordinate,
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
    "EarthRemoteSensingRapidResponse/artifacts/foundation/prithvi_eo_2_100m_tl"
)
DEFAULT_FOUNDATION_RECEIPT = Path("reports/acquisition/prithvi_eo_2_100m_tl.json")
DEFAULT_OUTPUT_CACHE = Path("outputs/mars_prithvi_eo_2_100m_tl_cls_folds34.npz")
CHECKPOINT_NAME = "Prithvi_EO_V2_100M_TL.pt"
BLOCKS = (3, 6, 9, 12)
EMBED_DIM = 768


def feature_names() -> list[str]:
    return [
        f"prithvi100m_block{block}_cls_{channel}"
        for block in BLOCKS
        for channel in range(EMBED_DIM)
    ]


def build_features(outputs: list[torch.Tensor]) -> torch.Tensor:
    if len(outputs) != 12:
        raise ValueError("Prithvi 100M encoder block count differs")
    values = torch.cat([outputs[index - 1][:, 0].float() for index in BLOCKS], dim=1)
    if values.shape[1] != len(feature_names()) or not torch.isfinite(values).all():
        raise RuntimeError("Prithvi 100M CLS schema or finiteness failure")
    return values


@torch.no_grad()
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-dir", default=DEFAULT_OUTPUT.as_posix())
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST.as_posix())
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL.as_posix())
    parser.add_argument("--acquisition-receipt", default=DEFAULT_ACQUISITION_RECEIPT.as_posix())
    parser.add_argument("--foundation-dir", default=DEFAULT_FOUNDATION_DIR.as_posix())
    parser.add_argument("--foundation-receipt", default=DEFAULT_FOUNDATION_RECEIPT.as_posix())
    parser.add_argument("--folds", type=int, nargs="+", default=[3, 4])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--output", default=DEFAULT_OUTPUT_CACHE.as_posix())
    args = parser.parse_args()
    folds = tuple(sorted(set(args.folds)))
    if not folds or any(fold not in range(5) for fold in folds):
        parser.error("folds must be a non-empty subset of 0..4")
    if args.batch_size <= 0 or args.workers < 0:
        parser.error("batch size must be positive and workers non-negative")

    root = repo_root()
    foundation_dir = (root / args.foundation_dir).resolve()
    receipt_path = (root / args.foundation_receipt).resolve()
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    for item in receipt["files"]:
        path = (root / item["path"]).resolve()
        if path.stat().st_size != int(item["bytes"]) or sha256(path) != item["sha256"]:
            raise ValueError(f"Prithvi 100M acquisition identity mismatch: {item['path']}")
    sys.path.insert(0, str(foundation_dir))
    from prithvi_mae import PrithviMAE  # type: ignore  # noqa: E402

    config = json.loads((foundation_dir / "config.json").read_text(encoding="utf-8"))[
        "pretrained_cfg"
    ]
    means = torch.tensor(config["mean"], dtype=torch.float32)[None, :, None, None, None]
    stds = torch.tensor(config["std"], dtype=torch.float32)[None, :, None, None, None]
    model_config = dict(config)
    model_config.update(img_size=INPUT_SIZE, num_frames=2, in_chans=6)
    model = PrithviMAE(**model_config)
    checkpoint = foundation_dir / CHECKPOINT_NAME
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state["encoder.pos_embed"] = model.encoder.pos_embed
    state["decoder.decoder_pos_embed"] = model.decoder.decoder_pos_embed
    model.load_state_dict(state, strict=True)
    del model.decoder
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Prithvi 100M extraction requires CUDA")
    model = model.to(device).eval()
    means, stds = means.to(device), stds.to(device)

    manifest = (root / args.manifest).resolve()
    protocol_path = (root / args.protocol).resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    manifest_hash = sha256(manifest)
    if manifest_hash != protocol["development_manifest_sha256"]:
        raise ValueError("Development manifest differs from frozen protocol")
    verify_acquisition_receipt((root / args.acquisition_receipt).resolve(), manifest_hash)
    group_to_fold = {
        str(item["group_id"]): int(item["fold"])
        for item in protocol["assignments"]
    }
    all_records = list(iter_development_manifest(manifest))
    metadata = {str(record["sample_id"]): record for record in all_records}
    records = [
        record for record in all_records
        if group_to_fold[str(record["group_id"])] in folds
    ]
    if args.max_rows is not None:
        records = records[: args.max_rows]
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
    missing_reference_datetime_rows = 0
    for batch_index, batch in enumerate(loader, start=1):
        local_ids = [str(value) for value in batch["sample_id"]]
        missing_reference_datetime_rows += sum(
            not str(metadata[sample_id]["reference_scene_id"]).strip()
            for sample_id in local_ids
        )
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
        with torch.amp.autocast("cuda", dtype=torch.float16):
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
        foundation_revision=np.asarray(receipt["source"]["revision"]),
        checkpoint_sha256=np.asarray(sha256(checkpoint)),
        foundation_receipt_sha256=np.asarray(sha256(receipt_path)),
        manifest_sha256=np.asarray(manifest_hash),
        protocol_sha256=np.asarray(sha256(protocol_path)),
        input_contract=np.asarray("MARS reference/target six-band physical reflectance at 128x128"),
        nir_transfer_contract=np.asarray("MARS broad NIR mapped to pretrained HLS narrow-NIR slot"),
        missing_reference_datetime_rows=np.asarray(missing_reference_datetime_rows),
    )
    print(json.dumps({
        "ok": True,
        "rows": len(labels),
        "features": len(feature_names()),
        "folds": list(folds),
        "output": args.output,
        "sha256": sha256(output_path),
        "checkpoint_sha256": sha256(checkpoint),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
