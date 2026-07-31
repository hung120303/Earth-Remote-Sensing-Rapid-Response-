#!/usr/bin/env python3
"""Extract directional counterfactual and frequency-aligned MARS scene inputs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
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

from acquire_mars_metadata import sha256  # noqa: E402
from extract_mars_scene_features import atomic_savez  # noqa: E402
from mars_paper_model import ReleasedMarsUNet, released_state  # noqa: E402
from train_mars_paper_residual import (  # noqa: E402
    MarsPaperDataset,
    iter_development_manifest,
    move_batch,
    verify_acquisition_receipt,
)

DEFAULT_PROTOCOL = Path("configs/mars_counterfactual_scene_inputs_protocol.json")
OUTPUT_SIZE = 64
LOW_FREQUENCY_FRACTION = 0.10
CHANNEL_NAMES = (
    "factual_probability_meanpool",
    "factual_probability_maxpool",
    "swapped_probability_meanpool",
    "swapped_probability_maxpool",
    "target_self_probability_meanpool",
    "target_self_probability_maxpool",
    "reference_self_probability_meanpool",
    "reference_self_probability_maxpool",
    "factual_minus_swapped_probability_meanpool",
    "factual_minus_self_probability_meanpool",
    "factual_mbmp_centered",
    "swapped_mbmp_centered",
    "target_reference_B11_difference",
    "target_reference_B12_difference",
    "target_reference_B11_normalized_difference",
    "target_reference_B12_normalized_difference",
    "B12_minus_B11_difference",
    "B12_minus_B11_normalized_difference",
    "frequency_aligned_target_minus_reference_B11",
    "frequency_aligned_target_minus_reference_B12",
    "target_minus_frequency_aligned_reference_B11",
    "target_minus_frequency_aligned_reference_B12",
    "highpass_target_reference_B11_difference",
    "highpass_target_reference_B12_difference",
    "low_frequency_change_magnitude",
    "frequency_style_correction_magnitude",
    "cloud_fraction",
    "observable_fraction",
)


def release_mbmp_torch(target: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Reproduce the released per-scene MBMP transform on a tensor batch."""
    if target.shape != reference.shape or target.ndim != 4 or target.shape[1] != 6:
        raise ValueError("MBMP requires matching Bx6xHxW target/reference tensors")

    def normalized_ratio(values: torch.Tensor) -> torch.Tensor:
        b11 = values[:, 4]
        b12 = values[:, 5]
        usable = b11 != 0
        ratio = torch.where(usable, b12 / b11.clamp_min(1e-8), torch.ones_like(b11))
        candidates = ratio.masked_fill(~usable, torch.nan).flatten(1)
        median = torch.nanmedian(candidates, dim=1).values
        median = torch.where(
            torch.isfinite(median) & (median.abs() >= 1e-8),
            median,
            torch.ones_like(median),
        )
        ratio = ratio / median[:, None, None]
        return ratio.nan_to_num(1.0, posinf=10.0, neginf=0.0).clamp(0.0, 10.0)

    target_ratio = normalized_ratio(target.float())
    reference_ratio = normalized_ratio(reference.float())
    result = torch.where(
        reference_ratio != 0,
        target_ratio / reference_ratio.clamp_min(1e-8),
        torch.ones_like(target_ratio),
    )
    return result.nan_to_num(1.0, posinf=10.0, neginf=0.0).clamp(0.0, 10.0)[:, None]


def counterfactual_inputs(inputs: torch.Tensor) -> dict[str, torch.Tensor]:
    """Create swapped-date and same-date controls under the 16-channel contract."""
    if inputs.ndim != 4 or inputs.shape[1] != 16:
        raise ValueError("Counterfactual construction requires Bx16xHxW inputs")
    target = inputs[:, 1:7].float()
    reference = inputs[:, 7:13].float()
    context = inputs[:, 13:16].float()
    ones = torch.ones_like(inputs[:, :1], dtype=torch.float32)
    return {
        "swapped": torch.cat(
            [release_mbmp_torch(reference, target), reference, target, context], dim=1
        ),
        "target_self": torch.cat([ones, target, target, context], dim=1),
        "reference_self": torch.cat([ones, reference, reference, context], dim=1),
    }


def frequency_align(
    source: torch.Tensor,
    style: torch.Tensor,
    *,
    low_frequency_fraction: float = LOW_FREQUENCY_FRACTION,
) -> torch.Tensor:
    """Replace a centered low-frequency amplitude box while preserving phase."""
    if source.shape != style.shape or source.ndim != 4:
        raise ValueError("Frequency alignment requires matching BCHW tensors")
    if not 0.0 < low_frequency_fraction < 0.5:
        raise ValueError("Low-frequency fraction must be in (0, 0.5)")
    source_fft = torch.fft.fftshift(torch.fft.fft2(source.float()), dim=(-2, -1))
    style_fft = torch.fft.fftshift(torch.fft.fft2(style.float()), dim=(-2, -1))
    source_amplitude = source_fft.abs()
    source_phase = source_fft / source_amplitude.clamp_min(1e-8)
    amplitude = source_amplitude.clone()
    height, width = source.shape[-2:]
    half_height = max(1, int(round(height * low_frequency_fraction / 2.0)))
    half_width = max(1, int(round(width * low_frequency_fraction / 2.0)))
    center_height, center_width = height // 2, width // 2
    rows = slice(center_height - half_height, center_height + half_height + 1)
    cols = slice(center_width - half_width, center_width + half_width + 1)
    amplitude[..., rows, cols] = style_fft.abs()[..., rows, cols]
    transformed = torch.fft.ifft2(
        torch.fft.ifftshift(source_phase * amplitude, dim=(-2, -1))
    ).real
    return transformed


def _pooled_probability(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    probability = torch.sigmoid(logits.float())
    return (
        F.adaptive_avg_pool2d(probability, (OUTPUT_SIZE, OUTPUT_SIZE)),
        F.adaptive_max_pool2d(probability, (OUTPUT_SIZE, OUTPUT_SIZE)),
    )


def counterfactual_scene_channels(
    inputs: torch.Tensor,
    observable: torch.Tensor,
    logits: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Build the frozen 28-channel label-free scene representation."""
    required = {"factual", "swapped", "target_self", "reference_self"}
    if set(logits) != required:
        raise ValueError("Counterfactual logits differ from the four-view contract")
    if inputs.ndim != 4 or inputs.shape[1] != 16:
        raise ValueError("Scene channels require Bx16xHxW inputs")
    if observable.shape != inputs[:, :1].shape:
        raise ValueError("Observable mask does not align with scene inputs")

    probability = {name: _pooled_probability(value) for name, value in logits.items()}
    target = F.adaptive_avg_pool2d(inputs[:, 1:7].float(), (OUTPUT_SIZE, OUTPUT_SIZE))
    reference = F.adaptive_avg_pool2d(
        inputs[:, 7:13].float(), (OUTPUT_SIZE, OUTPUT_SIZE)
    )
    difference = target - reference
    normalized = difference / (target.abs() + reference.abs() + 0.02)
    swapped_mbmp = F.adaptive_avg_pool2d(
        release_mbmp_torch(inputs[:, 7:13], inputs[:, 1:7]),
        (OUTPUT_SIZE, OUTPUT_SIZE),
    )
    factual_mbmp = F.adaptive_avg_pool2d(
        inputs[:, :1].float(), (OUTPUT_SIZE, OUTPUT_SIZE)
    )

    target_aligned = frequency_align(target, reference)
    reference_aligned = frequency_align(reference, target)
    aligned_target_difference = target_aligned - reference
    aligned_reference_difference = target - reference_aligned
    low_frequency = F.avg_pool2d(difference, kernel_size=9, stride=1, padding=4)
    high_frequency = difference - low_frequency
    low_frequency_magnitude = low_frequency.abs().mean(dim=1, keepdim=True)
    style_correction_magnitude = (target_aligned - target).abs().mean(
        dim=1, keepdim=True
    )

    factual_mean, factual_max = probability["factual"]
    swapped_mean, swapped_max = probability["swapped"]
    target_self_mean, target_self_max = probability["target_self"]
    reference_self_mean, reference_self_max = probability["reference_self"]
    maximum_self = torch.maximum(target_self_mean, reference_self_mean)
    cloud = F.adaptive_avg_pool2d(inputs[:, 15:16].float(), (OUTPUT_SIZE, OUTPUT_SIZE))
    observed = F.adaptive_avg_pool2d(
        observable.float(), (OUTPUT_SIZE, OUTPUT_SIZE)
    )
    values = torch.cat(
        [
            factual_mean,
            factual_max,
            swapped_mean,
            swapped_max,
            target_self_mean,
            target_self_max,
            reference_self_mean,
            reference_self_max,
            factual_mean - swapped_mean,
            factual_mean - maximum_self,
            factual_mbmp - 1.0,
            swapped_mbmp - 1.0,
            difference[:, 4:5],
            difference[:, 5:6],
            normalized[:, 4:5],
            normalized[:, 5:6],
            difference[:, 5:6] - difference[:, 4:5],
            normalized[:, 5:6] - normalized[:, 4:5],
            aligned_target_difference[:, 4:5],
            aligned_target_difference[:, 5:6],
            aligned_reference_difference[:, 4:5],
            aligned_reference_difference[:, 5:6],
            high_frequency[:, 4:5],
            high_frequency[:, 5:6],
            low_frequency_magnitude,
            style_correction_magnitude,
            cloud,
            observed,
        ],
        dim=1,
    )
    if values.shape[1:] != (len(CHANNEL_NAMES), OUTPUT_SIZE, OUTPUT_SIZE):
        raise RuntimeError("Counterfactual scene representation has the wrong schema")
    if not torch.isfinite(values).all():
        raise RuntimeError("Counterfactual scene representation contains non-finite values")
    return values


def _verify_contract(protocol_path: Path, protocol: dict[str, Any]) -> dict[str, Path]:
    if sha256(Path(__file__).resolve()) != protocol["extractor"]["sha256"]:
        raise ValueError("Frozen counterfactual extractor hash mismatch")
    paths: dict[str, Path] = {}
    for name, contract in protocol["inputs"].items():
        path = (ROOT / contract["path"]).resolve()
        if contract["sha256"] != "directory_verified_by_acquisition_receipt":
            if sha256(path) != contract["sha256"]:
                raise ValueError(f"Frozen counterfactual input hash mismatch: {name}")
        paths[name] = path
    verify_acquisition_receipt(paths["acquisition_receipt"], sha256(paths["manifest"]))
    return paths


@torch.inference_mode()
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL.as_posix())
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    protocol_path = (ROOT / args.protocol).resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    paths = _verify_contract(protocol_path, protocol)

    fold_protocol = json.loads(paths["fold_protocol"].read_text(encoding="utf-8"))
    group_to_fold = {
        str(item["group_id"]): int(item["fold"]) for item in fold_protocol["assignments"]
    }
    selected_folds = set(map(int, protocol["folds"]))
    records = [
        record
        for record in iter_development_manifest(paths["manifest"])
        if group_to_fold[str(record["group_id"])] in selected_folds
    ]
    if args.smoke:
        smoke: list[dict[str, Any]] = []
        strata: set[tuple[str, str]] = set()
        for record in records:
            key = (str(record["label_state"]), str(record["sensor_family"]))
            if key in strata:
                continue
            smoke.append(record)
            strata.add(key)
            if len(strata) == 4:
                break
        records = smoke
    if not records:
        raise ValueError("No records selected for counterfactual extraction")

    dataset = MarsPaperDataset(paths["metadata_root"], records, augment=False, seed=0)
    loader = DataLoader(
        dataset,
        batch_size=(len(records) if args.smoke else int(protocol["runtime"]["batch_size"])),
        shuffle=False,
        num_workers=(0 if args.smoke else int(protocol["runtime"]["workers"])),
        pin_memory=True,
        persistent_workers=(not args.smoke and int(protocol["runtime"]["workers"]) > 0),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Counterfactual extraction requires CUDA")
    torch.set_float32_matmul_precision("high")
    model = ReleasedMarsUNet().to(device)
    model.load_state_dict(released_state(paths["released_checkpoint"]), strict=False)
    model.eval()

    if args.smoke:
        batch = move_batch(next(iter(loader)), device)
        variants = counterfactual_inputs(batch["inputs"])
        with torch.amp.autocast("cuda", dtype=torch.float16):
            logits = {"factual": model(batch["inputs"])}
            logits.update({name: model(value) for name, value in variants.items()})
        values = counterfactual_scene_channels(batch["inputs"], batch["observable"], logits)
        print(
            json.dumps(
                {
                    "ok": True,
                    "smoke": True,
                    "rows": int(values.shape[0]),
                    "shape": list(values.shape),
                    "finite": bool(torch.isfinite(values).all()),
                    "channel_names": list(CHANNEL_NAMES),
                    "maximum_absolute_value": float(values.abs().max().item()),
                },
                indent=2,
            )
        )
        return 0

    output_images = (ROOT / protocol["outputs"]["images"]).resolve()
    output_metadata = (ROOT / protocol["outputs"]["metadata"]).resolve()
    output_receipt = (ROOT / protocol["outputs"]["receipt"]).resolve()
    output_images.parent.mkdir(parents=True, exist_ok=True)
    temporary_images = output_images.with_suffix(".tmp.npy")
    images = open_memmap(
        temporary_images,
        mode="w+",
        dtype=np.float16,
        shape=(len(records), len(CHANNEL_NAMES), OUTPUT_SIZE, OUTPUT_SIZE),
    )
    labels: list[int] = []
    sensors: list[int] = []
    sample_ids: list[str] = []
    groups: list[str] = []
    folds: list[int] = []
    cursor = 0
    for batch_index, batch in enumerate(loader, start=1):
        batch = move_batch(batch, device)
        variants = counterfactual_inputs(batch["inputs"])
        with torch.amp.autocast("cuda", dtype=torch.float16):
            logits = {"factual": model(batch["inputs"])}
            logits.update({name: model(value) for name, value in variants.items()})
        values = counterfactual_scene_channels(
            batch["inputs"], batch["observable"], logits
        ).cpu().numpy().astype(np.float16)
        end = cursor + values.shape[0]
        images[cursor:end] = values
        labels.extend(int(value) for value in batch["presence"].cpu().numpy())
        sensors.extend(int(value) for value in batch["sensor_index"].cpu().numpy())
        for sample_id, group_id in zip(batch["sample_id"], batch["group_id"], strict=True):
            sample_ids.append(str(sample_id))
            groups.append(str(group_id))
            folds.append(group_to_fold[str(group_id)])
        cursor = end
        if batch_index % 100 == 0:
            images.flush()
            print(json.dumps({"batches": batch_index, "rows": cursor}), flush=True)
    if cursor != len(records):
        raise RuntimeError("Counterfactual extraction did not write every selected row")
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
        folds=np.asarray(folds, dtype=np.uint8),
        images_sha256=np.asarray(images_hash),
        manifest_sha256=np.asarray(sha256(paths["manifest"])),
        fold_protocol_sha256=np.asarray(sha256(paths["fold_protocol"])),
        released_checkpoint_sha256=np.asarray(sha256(paths["released_checkpoint"])),
        protocol_sha256=np.asarray(sha256(protocol_path)),
    )
    receipt = {
        "schema_version": 1,
        "scope": "development folds 3/4 label-free counterfactual feature extraction",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows": cursor,
        "shape": [cursor, len(CHANNEL_NAMES), OUTPUT_SIZE, OUTPUT_SIZE],
        "folds": sorted(selected_folds),
        "channels": list(CHANNEL_NAMES),
        "outputs": {
            "images": {
                "path": protocol["outputs"]["images"],
                "bytes": output_images.stat().st_size,
                "sha256": images_hash,
                "tracked": False,
            },
            "metadata": {
                "path": protocol["outputs"]["metadata"],
                "bytes": output_metadata.stat().st_size,
                "sha256": sha256(output_metadata),
                "tracked": False,
            },
        },
        "provenance": {
            "protocol_sha256": sha256(protocol_path),
            "script_sha256": sha256(Path(__file__).resolve()),
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "device": torch.cuda.get_device_name(device),
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
        "invariants": [
            "Only development folds 3 and 4 were loaded.",
            "Counterfactual transforms use imagery and the frozen released model without labels.",
            "Bulk arrays remain Git-ignored; only this compact receipt is tracked.",
            "No fold 0/1/2, exact-paper, or fresh-external sample was accessed.",
        ],
    }
    output_receipt.parent.mkdir(parents=True, exist_ok=True)
    temporary_receipt = output_receipt.with_suffix(output_receipt.suffix + ".tmp")
    temporary_receipt.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary_receipt, output_receipt)
    print(json.dumps({"ok": True, **receipt["outputs"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
