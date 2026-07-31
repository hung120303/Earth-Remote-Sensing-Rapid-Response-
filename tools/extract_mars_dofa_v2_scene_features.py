#!/usr/bin/env python3
"""Extract wavelength-conditioned DOFA-v2 temporal scene features for MARS."""

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

from acquire_dofa_v2_base import (  # noqa: E402
    CHECKPOINT_SHA256,
    DEFAULT_CHECKPOINT,
    DEFAULT_RECEIPT as DEFAULT_FOUNDATION_RECEIPT,
    MODEL_REVISION,
)
from acquire_mars_metadata import DEFAULT_OUTPUT, repo_root, sha256  # noqa: E402
from dofa_v2_backbone import vit_base_patch14  # noqa: E402
from extract_mars_scene_features import atomic_savez  # noqa: E402
from mars_paper_model import SENSOR_NAMES  # noqa: E402
from train_mars_paper_residual import (  # noqa: E402
    DEFAULT_ACQUISITION_RECEIPT,
    DEFAULT_MANIFEST,
    DEFAULT_PROTOCOL,
    MarsPaperDataset,
    available_smoke_subset,
    iter_development_manifest,
    move_batch,
    verify_acquisition_receipt,
)

DEFAULT_OUTPUT_CACHE = Path("outputs/mars_dofa_v2_scene_features_folds34.npz")
INPUT_SIZE = 224
EMBED_DIM = 768
BLOCK_INDICES = (4, 6, 10, 11)
MARS_TO_PHYSICAL_REFLECTANCE = 0.5
PHYSICAL_TO_DOFA_BYTE_SCALE = 255.0
MARS_TO_DOFA_MULTIPLIER = MARS_TO_PHYSICAL_REFLECTANCE * PHYSICAL_TO_DOFA_BYTE_SCALE
S2_WAVELENGTHS = (0.490, 0.560, 0.665, 0.842, 1.610, 2.190)
LANDSAT_WAVELENGTHS = (0.482, 0.561, 0.655, 0.865, 1.609, 2.201)
# Official DOFA Sentinel-2 statistics reordered from
# [red, green, blue, red-edge1, red-edge2, red-edge3, NIR, SWIR1, SWIR2]
# to the MARS [blue, green, red, NIR, SWIR1, SWIR2] band contract.
DOFA_MEAN = (126.63977424, 114.81779093, 114.10997390, 101.43563300, 72.32804172, 56.66528851)
DOFA_STD = (67.42465279, 69.96844919, 77.84352553, 60.29744676, 47.88519516, 42.55886798)
SUMMARY_PARTS = (
    "reference_final_mean",
    "target_final_mean",
    "absolute_difference_mean",
    "signed_difference_std",
    "signed_extreme_difference",
)
FEATURE_WIDTH = 14 * EMBED_DIM


def feature_names() -> list[str]:
    names = [
        f"dofa_v2_{part}_{channel}"
        for part in ("reference_final_mean", "target_final_mean")
        for channel in range(EMBED_DIM)
    ]
    for part in SUMMARY_PARTS[2:]:
        names.extend(
            f"dofa_v2_block{block_index}_{part}_{channel}"
            for block_index in BLOCK_INDICES
            for channel in range(EMBED_DIM)
        )
    if len(names) != FEATURE_WIDTH:
        raise RuntimeError("DOFA-v2 feature-name width differs from the frozen schema")
    return names


def sensor_wavelengths(sensor_index: int) -> tuple[float, ...]:
    sensor_name = SENSOR_NAMES[sensor_index]
    if sensor_name == "Sentinel-2":
        return S2_WAVELENGTHS
    if sensor_name == "Landsat":
        return LANDSAT_WAVELENGTHS
    raise ValueError(f"Unsupported MARS sensor: {sensor_name}")


def build_dofa_frames(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """Return normalized chronological B,T,C,224,224 DOFA inputs."""
    inputs = batch["inputs"]
    if inputs.ndim != 4 or inputs.shape[1] != 16:
        raise ValueError("Expected the frozen 16-channel MARS input")
    frames = torch.stack((inputs[:, 7:13], inputs[:, 1:7]), dim=1)
    batch_size = frames.shape[0]
    resized = F.interpolate(
        frames.flatten(0, 1),
        size=(INPUT_SIZE, INPUT_SIZE),
        mode="bilinear",
        align_corners=False,
    ).reshape(batch_size, 2, 6, INPUT_SIZE, INPUT_SIZE)
    mean = torch.tensor(DOFA_MEAN, device=inputs.device, dtype=torch.float32)[
        None, None, :, None, None
    ]
    std = torch.tensor(DOFA_STD, device=inputs.device, dtype=torch.float32)[
        None, None, :, None, None
    ]
    normalized = (resized * MARS_TO_DOFA_MULTIPLIER - mean) / std
    observable = F.interpolate(
        batch["observable"].float(), size=(INPUT_SIZE, INPUT_SIZE), mode="nearest"
    )[:, None]
    return normalized.masked_fill(observable <= 0.5, 0.0)


def signed_extreme(values: torch.Tensor) -> torch.Tensor:
    flat = values.flatten(2)
    indices = flat.abs().argmax(dim=2, keepdim=True)
    return torch.gather(flat, dim=2, index=indices).squeeze(2)


def temporal_scene_features(
    reference_maps: list[torch.Tensor], target_maps: list[torch.Tensor]
) -> torch.Tensor:
    if len(reference_maps) != len(BLOCK_INDICES) or len(target_maps) != len(BLOCK_INDICES):
        raise ValueError("Expected four official DOFA-v2 intermediate maps per frame")
    for reference, target in zip(reference_maps, target_maps):
        if reference.shape != target.shape or reference.shape[1] != EMBED_DIM:
            raise ValueError("DOFA-v2 temporal map geometry differs")
    final_reference = reference_maps[-1].float().mean(dim=(2, 3))
    final_target = target_maps[-1].float().mean(dim=(2, 3))
    differences = [target.float() - reference.float() for reference, target in zip(reference_maps, target_maps)]
    absolute_means = [difference.abs().mean(dim=(2, 3)) for difference in differences]
    difference_stds = [
        difference.flatten(2).std(dim=2, unbiased=False) for difference in differences
    ]
    extremes = [signed_extreme(difference) for difference in differences]
    features = torch.cat(
        (final_reference, final_target, *absolute_means, *difference_stds, *extremes),
        dim=1,
    )
    if features.shape[1] != FEATURE_WIDTH or not torch.isfinite(features).all():
        raise RuntimeError("DOFA-v2 scene feature schema or finiteness failure")
    return features


@torch.no_grad()
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-dir", default=DEFAULT_OUTPUT.as_posix())
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST.as_posix())
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL.as_posix())
    parser.add_argument(
        "--acquisition-receipt", default=DEFAULT_ACQUISITION_RECEIPT.as_posix()
    )
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT.as_posix())
    parser.add_argument(
        "--foundation-receipt", default=DEFAULT_FOUNDATION_RECEIPT.as_posix()
    )
    parser.add_argument("--folds", type=int, nargs="+", default=[3, 4])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--smoke-per-stratum", type=int)
    parser.add_argument("--output", default=DEFAULT_OUTPUT_CACHE.as_posix())
    args = parser.parse_args()
    folds = tuple(sorted(set(args.folds)))
    if not folds or any(fold not in range(5) for fold in folds):
        parser.error("folds must be a non-empty subset of 0..4")
    if args.batch_size <= 0 or args.workers < 0:
        parser.error("batch size must be positive and workers non-negative")
    if args.max_rows is not None and args.max_rows <= 0:
        parser.error("max-rows must be positive when provided")
    if args.smoke_per_stratum is not None and args.smoke_per_stratum <= 0:
        parser.error("smoke-per-stratum must be positive when provided")
    if args.max_rows is not None and args.smoke_per_stratum is not None:
        parser.error("max-rows and smoke-per-stratum are mutually exclusive")

    root = repo_root()
    checkpoint = (root / args.checkpoint).resolve()
    foundation_receipt_path = (root / args.foundation_receipt).resolve()
    foundation_receipt = json.loads(
        foundation_receipt_path.read_text(encoding="utf-8")
    )
    if checkpoint.stat().st_size != 421_811_730 or sha256(checkpoint) != CHECKPOINT_SHA256:
        raise ValueError("DOFA-v2 checkpoint identity mismatch")
    if foundation_receipt["source"]["model_revision"] != MODEL_REVISION:
        raise ValueError("DOFA-v2 foundation receipt revision mismatch")
    if foundation_receipt["files"][0]["sha256"] != CHECKPOINT_SHA256:
        raise ValueError("DOFA-v2 foundation receipt checkpoint mismatch")

    model = vit_base_patch14()
    model.load_state_dict(
        torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

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
    records = [
        record
        for record in all_records
        if group_to_fold[str(record["group_id"])] in folds
    ]
    if args.smoke_per_stratum is not None:
        records = available_smoke_subset(
            (root / args.metadata_dir).resolve(),
            records,
            limit_per_stratum=args.smoke_per_stratum,
        )
    elif args.max_rows is not None:
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
        batch = move_batch(batch, device)
        frames = build_dofa_frames(batch)
        local_features = torch.empty(
            (frames.shape[0], FEATURE_WIDTH), device=device, dtype=torch.float32
        )
        for sensor_index in torch.unique(batch["sensor_index"]).tolist():
            selection = torch.nonzero(
                batch["sensor_index"] == int(sensor_index), as_tuple=False
            ).flatten()
            sensor_frames = frames[selection].flatten(0, 1)
            with torch.amp.autocast(
                "cuda", dtype=torch.float16, enabled=device.type == "cuda"
            ):
                outputs = model.forward_features(
                    sensor_frames, sensor_wavelengths(int(sensor_index))
                )
            reference_maps = [output[0::2] for output in outputs]
            target_maps = [output[1::2] for output in outputs]
            local_features[selection] = temporal_scene_features(
                reference_maps, target_maps
            )
        rows.append(local_features.cpu().numpy().astype(np.float16))
        labels.extend(int(value) for value in batch["presence"].cpu().numpy())
        sensors.extend(int(value) for value in batch["sensor_index"].cpu().numpy())
        for sample_id, group in zip(batch["sample_id"], batch["group_id"]):
            sample_ids.append(str(sample_id))
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
        checkpoint_sha256=np.asarray(CHECKPOINT_SHA256),
        foundation_model_revision=np.asarray(MODEL_REVISION),
        foundation_receipt_sha256=np.asarray(sha256(foundation_receipt_path)),
        manifest_sha256=np.asarray(manifest_hash),
        fold_protocol_sha256=np.asarray(sha256(fold_protocol_path)),
        input_size=np.asarray(INPUT_SIZE),
        mars_to_dofa_multiplier=np.asarray(MARS_TO_DOFA_MULTIPLIER),
        s2_wavelengths=np.asarray(S2_WAVELENGTHS),
        landsat_wavelengths=np.asarray(LANDSAT_WAVELENGTHS),
        dofa_mean=np.asarray(DOFA_MEAN),
        dofa_std=np.asarray(DOFA_STD),
        input_contract=np.asarray(
            "reference and target encoded independently; physical reflectance x255; "
            "official reordered Sentinel-2 statistics; sensor-specific center wavelengths; "
            "224x224 bilinear; invalid pixels zero after normalization"
        ),
    )
    print(
        json.dumps(
            {
                "ok": True,
                "rows": len(labels),
                "features": FEATURE_WIDTH,
                "folds": list(folds),
                "output": args.output,
                "sha256": sha256(output_path),
                "peak_cuda_gib": (
                    torch.cuda.max_memory_allocated() / 2**30
                    if device.type == "cuda"
                    else 0.0
                ),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
