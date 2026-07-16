#!/usr/bin/env python3
"""Extract frozen spatial and Prithvi representations for fresh CloudSEN negatives."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "EarthRemoteSensingRapidResponse"
for path in (MODEL_ROOT, ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from acquire_mars_metadata import sha256  # noqa: E402
from evaluate_mars_residual_endpoint_blend import load_residual_model  # noqa: E402
from extract_cloudsen12_spatial_scene_features import validate_records  # noqa: E402
from extract_mars_prithvi_scene_features import (  # noqa: E402
    INPUT_SIZE,
    build_features,
    build_input,
    date_coordinate,
    reference_date_coordinate,
)
from extract_mars_scene_features import atomic_savez  # noqa: E402
from extract_mars_spatial_scene_inputs import CHANNEL_NAMES, spatial_scene_channels  # noqa: E402
from mars_s2l_adapter import iter_manifest  # noqa: E402
from train_mars_paper_residual import MarsPaperDataset, move_batch  # noqa: E402


DEFAULT_PROTOCOL = Path("configs/cloudsen12_fresh_spatial_prithvi_protocol.json")
CLS_WIDTH = 768


@torch.no_grad()
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL.as_posix())
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()
    if args.batch_size <= 0 or args.workers < 0:
        parser.error("batch size must be positive and workers nonnegative")
    protocol_path = (ROOT / args.protocol).resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if sha256(Path(__file__).resolve()) != protocol["extractor"]["sha256"]:
        raise ValueError("Fresh spatial-Prithvi extractor hash mismatch")
    paths: dict[str, Path] = {}
    for name, contract in protocol["inputs"].items():
        path = (ROOT / contract["path"]).resolve()
        if sha256(path) != contract["sha256"]:
            raise ValueError(f"Fresh representation input hash mismatch: {name}")
        paths[name] = path
    records = list(iter_manifest(paths["manifest"]))
    validate_records(records, "fresh_external_test", int(protocol["expected"]["rows"]))
    metadata = {str(record["sample_id"]): record for record in records}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    residual_artifact = torch.load(paths["residual_artifact"], map_location="cpu", weights_only=True)
    residual_model = load_residual_model(paths["released_checkpoint"], residual_artifact, device).eval()

    foundation_receipt = json.loads(paths["foundation_receipt"].read_text(encoding="utf-8"))
    foundation_dir = paths["foundation_dir_receipt_anchor"].parent
    for item in foundation_receipt["files"]:
        item_path = (ROOT / item["path"]).resolve()
        if item_path.stat().st_size != int(item["bytes"]) or sha256(item_path) != item["sha256"]:
            raise ValueError(f"Prithvi file identity mismatch: {item['path']}")
    sys.path.insert(0, str(foundation_dir))
    from prithvi_mae import PrithviMAE  # type: ignore  # noqa: E402

    config_path = foundation_dir / "config.json"
    foundation_checkpoint = foundation_dir / "Prithvi_EO_V2_tiny_TL.pt"
    config = json.loads(config_path.read_text(encoding="utf-8"))["pretrained_cfg"]
    means = torch.tensor(config["mean"], dtype=torch.float32, device=device)[None, :, None, None, None]
    stds = torch.tensor(config["std"], dtype=torch.float32, device=device)[None, :, None, None, None]
    model_config = dict(config)
    model_config.update(img_size=INPUT_SIZE, num_frames=2, in_chans=6)
    prithvi = PrithviMAE(**model_config)
    state = torch.load(foundation_checkpoint, map_location="cpu", weights_only=True)
    state["encoder.pos_embed"] = prithvi.encoder.pos_embed
    state["decoder.decoder_pos_embed"] = prithvi.decoder.decoder_pos_embed
    prithvi.load_state_dict(state, strict=True)
    prithvi = prithvi.to(device).eval()

    loader = DataLoader(
        MarsPaperDataset(ROOT, records, augment=False, seed=0),
        batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
        pin_memory=False, persistent_workers=args.workers > 0,
    )
    spatial_rows: list[np.ndarray] = []
    prithvi_rows: list[np.ndarray] = []
    sample_ids: list[str] = []
    groups: list[str] = []
    for batch_index, batch in enumerate(loader, start=1):
        local_ids = [str(value) for value in batch["sample_id"]]
        temporal = torch.tensor(
            [[reference_date_coordinate(metadata[sample_id]), date_coordinate(str(metadata[sample_id]["target_datetime"]))] for sample_id in local_ids],
            dtype=torch.float32, device=device,
        )
        location = torch.tensor(
            [[float(metadata[sample_id]["latitude"]), float(metadata[sample_id]["longitude"])] for sample_id in local_ids],
            dtype=torch.float32, device=device,
        )
        batch = move_batch(batch, device)
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            residual_output = residual_model(batch["inputs"], batch["observable"], batch["sensor_index"])
            prithvi_output = prithvi.forward_features(build_input(batch, means, stds), temporal, location)
        spatial_rows.append(
            spatial_scene_channels(batch["inputs"], residual_output["baseline_logits"], batch["observable"])
            .cpu().numpy().astype(np.float16)
        )
        prithvi_rows.append(build_features(prithvi_output)[:, :CLS_WIDTH].cpu().numpy().astype(np.float16))
        sample_ids.extend(local_ids)
        groups.extend(str(value) for value in batch["group_id"])
        if batch_index % 10 == 0:
            print(json.dumps({"batches": batch_index, "rows": len(sample_ids)}), flush=True)
    output_path = (ROOT / protocol["outputs"]["cache"]).resolve()
    atomic_savez(
        output_path,
        spatial_images=np.concatenate(spatial_rows),
        spatial_channel_names=np.asarray(CHANNEL_NAMES),
        prithvi_cls=np.concatenate(prithvi_rows),
        sample_ids=np.asarray(sample_ids), groups=np.asarray(groups),
        labels=np.zeros(len(sample_ids), dtype=np.uint8), sensors=np.zeros(len(sample_ids), dtype=np.uint8),
        manifest_sha256=np.asarray(protocol["inputs"]["manifest"]["sha256"]),
        protocol_sha256=np.asarray(sha256(protocol_path)),
        foundation_checkpoint_sha256=np.asarray(sha256(foundation_checkpoint)),
    )
    report = {
        "schema_version": 1, "scope": "no-fit fresh external-negative spatial and Prithvi extraction",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows": len(sample_ids), "groups": len(set(groups)),
        "spatial_shape": list(np.concatenate(spatial_rows).shape),
        "prithvi_shape": list(np.concatenate(prithvi_rows).shape),
        "output": {"path": protocol["outputs"]["cache"], "bytes": output_path.stat().st_size, "sha256": sha256(output_path), "tracked": False},
        "paper_test_accessed": False,
        "provenance": {"protocol_sha256": sha256(protocol_path), "script_sha256": sha256(Path(__file__).resolve()), "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), "device": str(torch.cuda.get_device_name(device) if device.type == "cuda" else device), "torch": torch.__version__, "numpy": np.__version__},
    }
    report_path = (ROOT / protocol["outputs"]["report"]).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"ok": True, "output": report["output"], "rows": len(sample_ids)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
