#!/usr/bin/env python3
"""Extract fit-fold-only simulated counterfactual scene representations."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.lib.format import open_memmap

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "EarthRemoteSensingRapidResponse"
for path in (MODEL_ROOT, ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from acquire_mars_metadata import sha256  # noqa: E402
from extract_mars_counterfactual_scene_inputs import (  # noqa: E402
    CHANNEL_NAMES,
    counterfactual_inputs,
    counterfactual_scene_channels,
)
from extract_mars_scene_features import atomic_savez  # noqa: E402
from mars_paper_model import ReleasedMarsUNet, SENSOR_NAMES, released_state  # noqa: E402
from mars_s2l_adapter import (  # noqa: E402
    REFLECTANCE_DIVISOR,
    REFLECTANCE_MAX,
    compute_mbmp,
    load_sample,
)
from mars_v4_model import INPUT_CHANNELS  # noqa: E402
from train_mars_paper_residual import (  # noqa: E402
    iter_development_manifest,
    verify_acquisition_receipt,
)
from train_mars_v4 import MarsV4Dataset, metadata_and_plume_library  # noqa: E402

DEFAULT_PROTOCOL = Path("configs/mars_counterfactual_simulation_inputs_protocol.json")


def visible_size(path: Path) -> int:
    """Wait briefly for an atomic DrvFS rename to become visible to WSL."""
    for _ in range(100):
        try:
            return path.stat().st_size
        except FileNotFoundError:
            time.sleep(0.1)
    raise FileNotFoundError(f"Atomic output did not become visible: {path}")


def strict_simulated_positive(
    dataset: MarsV4Dataset,
    rng: np.random.Generator,
) -> tuple[dict[str, Any], Any, np.ndarray, np.ndarray, str, dict[str, Any]] | None:
    """Inject one fit-fold plume with the paper's strict 1.5 m/s wind gate."""
    target_record = dataset.records[int(rng.choice(dataset.eligible_negative_indices))]
    target_sample = load_sample(
        dataset.metadata_dir, target_record, require_enhancement=False
    )
    metadata = dataset.scene_metadata[target_sample.sample_id]
    target_speed = float(np.linalg.norm(metadata["wind"]))
    distance = np.abs(dataset._library_speeds - target_speed)
    candidates = np.flatnonzero(distance <= 1.5)
    if candidates.size == 0:
        return None
    source = dataset.plume_library[int(rng.choice(candidates))]
    ch4, source_mask = dataset.plume_arrays(source)
    result = dataset.simulator().simulate(
        target_sample.raw_pair[:6],
        ch4,
        source_mask,
        source_wind=source["wind"],
        target_wind=metadata["wind"],
        satellite=metadata["satellite"],
        solar_zenith_degrees=metadata["sza"],
        view_zenith_degrees=metadata["vza"],
        rng=rng,
    )
    visible = result.mask & target_sample.observable_mask
    visible_fraction = np.count_nonzero(visible) / max(np.count_nonzero(result.mask), 1)
    if visible_fraction < 0.5:
        return None
    raw_pair = np.concatenate([result.target, target_sample.raw_pair[6:]], axis=0)
    reflectance = np.clip(
        raw_pair.astype(np.float32) / REFLECTANCE_DIVISOR, 0.0, REFLECTANCE_MAX
    )
    identifier = f"sim:{target_sample.sample_id}:{source['sample_id']}"
    diagnostics = {
        "source_sample_id": str(source["sample_id"]),
        "wind_speed_delta": float(abs(float(source["wind_speed"]) - target_speed)),
        "scale": float(result.scale),
        "rotation_degrees": float(result.rotation_degrees),
        "visible_fraction": float(visible_fraction),
        "plume_pixels": int(np.count_nonzero(result.mask)),
    }
    return target_record, target_sample, reflectance, result.mask, identifier, diagnostics


def simulated_input(
    sample: Any, reflectance: np.ndarray, wind: tuple[float, float]
) -> tuple[torch.Tensor, torch.Tensor]:
    cloud = (sample.cloud_classes > 0).astype(np.float32)
    observable = sample.observable_mask.astype(np.float32)
    mbmp = compute_mbmp(reflectance[:6], reflectance[6:])
    height, width = mbmp.shape
    wind_channels = np.broadcast_to(
        np.asarray(wind, dtype=np.float32)[:, None, None] / 8.0,
        (2, height, width),
    ).copy()
    inputs = np.concatenate(
        [mbmp[None], reflectance, wind_channels, cloud[None]], axis=0
    ).astype(np.float32)
    if inputs.shape != (len(INPUT_CHANNELS), height, width):
        raise ValueError("Simulated input violates the released 16-channel contract")
    return torch.from_numpy(inputs), torch.from_numpy(observable[None])


def feature_batch(
    model: ReleasedMarsUNet,
    inputs: list[torch.Tensor],
    observable: list[torch.Tensor],
    device: torch.device,
) -> np.ndarray:
    values = torch.stack(inputs).to(device, non_blocking=True)
    observed = torch.stack(observable).to(device, non_blocking=True)
    variants = counterfactual_inputs(values)
    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.float16):
        logits = {"factual": model(values)}
        logits.update({name: model(item) for name, item in variants.items()})
    return (
        counterfactual_scene_channels(values, observed, logits)
        .cpu()
        .numpy()
        .astype(np.float16)
    )


def _paths(protocol: dict[str, Any]) -> dict[str, Path]:
    if sha256(Path(__file__).resolve()) != protocol["extractor"]["sha256"]:
        raise ValueError("Frozen simulation extractor hash mismatch")
    for dependency in protocol["code_dependencies"]:
        path = (ROOT / dependency["path"]).resolve()
        if sha256(path) != dependency["sha256"]:
            raise ValueError(f"Frozen simulation dependency mismatch: {dependency['path']}")
    paths: dict[str, Path] = {}
    for name, contract in protocol["inputs"].items():
        path = (ROOT / contract["path"]).resolve()
        if contract["sha256"] != "directory_verified_by_acquisition_receipt":
            if sha256(path) != contract["sha256"]:
                raise ValueError(f"Frozen simulation input hash mismatch: {name}")
        paths[name] = path
    verify_acquisition_receipt(paths["acquisition_receipt"], sha256(paths["manifest"]))
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL.as_posix())
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    protocol_path = (ROOT / args.protocol).resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    paths = _paths(protocol)
    fold_protocol = json.loads(paths["fold_protocol"].read_text(encoding="utf-8"))
    group_to_fold = {
        str(item["group_id"]): int(item["fold"]) for item in fold_protocol["assignments"]
    }
    all_records = list(iter_development_manifest(paths["manifest"]))
    records_by_fold = {
        fold: [
            record
            for record in all_records
            if group_to_fold[str(record["group_id"])] == fold
        ]
        for fold in map(int, protocol["fit_folds"])
    }
    count_per_stratum = (
        1 if args.smoke else int(protocol["samples_per_fit_fold_per_sensor"])
    )
    expected_rows = count_per_stratum * len(protocol["fit_folds"]) * len(SENSOR_NAMES)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Counterfactual simulation extraction requires CUDA")
    model = ReleasedMarsUNet().to(device)
    model.load_state_dict(released_state(paths["released_checkpoint"]), strict=False)
    model.eval()

    images: np.ndarray | None = None
    output_images: Path | None = None
    temporary_images: Path | None = None
    if not args.smoke:
        output_images = (ROOT / protocol["outputs"]["images"]).resolve()
        output_images.parent.mkdir(parents=True, exist_ok=True)
        temporary_images = output_images.with_suffix(".tmp.npy")
        images = open_memmap(
            temporary_images,
            mode="w+",
            dtype=np.float16,
            shape=(expected_rows, len(CHANNEL_NAMES), 64, 64),
        )

    simulation_ids: list[str] = []
    background_sample_ids: list[str] = []
    source_sample_ids: list[str] = []
    groups: list[str] = []
    sensors: list[int] = []
    fit_folds: list[int] = []
    wind_deltas: list[float] = []
    scales: list[float] = []
    rotations: list[float] = []
    visible_fractions: list[float] = []
    plume_pixels: list[int] = []
    failure_counts: Counter[str] = Counter()
    cursor = 0
    batch_inputs: list[torch.Tensor] = []
    batch_observable: list[torch.Tensor] = []

    def flush() -> None:
        nonlocal cursor, batch_inputs, batch_observable
        if not batch_inputs:
            return
        values = feature_batch(model, batch_inputs, batch_observable, device)
        if images is not None:
            images[cursor : cursor + values.shape[0]] = values
        cursor += values.shape[0]
        batch_inputs = []
        batch_observable = []

    for fit_fold in map(int, protocol["fit_folds"]):
        fold_records = records_by_fold[fit_fold]
        fit_positive_ids = {
            str(record["sample_id"])
            for record in fold_records
            if record["label_state"] == "PLUME"
        }
        for sensor_index, sensor_name in enumerate(SENSOR_NAMES):
            sensor_records = [
                record
                for record in fold_records
                if record["sensor_family"] == sensor_name
            ]
            required_ids = {str(record["sample_id"]) for record in sensor_records}
            scene_metadata, plume_library = metadata_and_plume_library(
                paths["metadata_root"],
                paths["metadata_csv"],
                required_ids,
                fit_positive_ids,
            )
            dataset = MarsV4Dataset(
                paths["metadata_root"],
                sensor_records,
                scene_metadata,
                lut_path=paths["transmittance_lut"],
                plume_library=plume_library,
                augment=False,
                simulation_fraction=0.0,
                seed=int(protocol["seed"]) + fit_fold * 10 + sensor_index,
            )
            if not dataset.eligible_negative_indices:
                raise ValueError(f"No eligible {sensor_name} negatives in fold {fit_fold}")
            rng = np.random.default_rng(int(protocol["seed"]) + fit_fold * 10 + sensor_index)
            accepted = 0
            attempts = 0
            maximum_attempts = count_per_stratum * int(protocol["maximum_attempt_multiplier"])
            while accepted < count_per_stratum and attempts < maximum_attempts:
                attempts += 1
                try:
                    result = strict_simulated_positive(dataset, rng)
                except (OSError, ValueError, FloatingPointError):
                    failure_counts["exception"] += 1
                    continue
                if result is None:
                    failure_counts["wind_or_visibility"] += 1
                    continue
                record, sample, reflectance, mask, identifier, diagnostics = result
                if not np.any(mask):
                    failure_counts["empty_mask"] += 1
                    continue
                inputs, observable = simulated_input(
                    sample, reflectance, scene_metadata[sample.sample_id]["wind"]
                )
                unique_id = f"{identifier}:fold{fit_fold}:sensor{sensor_index}:rep{accepted}"
                simulation_ids.append(unique_id)
                background_sample_ids.append(sample.sample_id)
                source_sample_ids.append(str(diagnostics["source_sample_id"]))
                groups.append(str(record["group_id"]))
                sensors.append(sensor_index)
                fit_folds.append(fit_fold)
                wind_deltas.append(float(diagnostics["wind_speed_delta"]))
                scales.append(float(diagnostics["scale"]))
                rotations.append(float(diagnostics["rotation_degrees"]))
                visible_fractions.append(float(diagnostics["visible_fraction"]))
                plume_pixels.append(int(diagnostics["plume_pixels"]))
                batch_inputs.append(inputs)
                batch_observable.append(observable)
                accepted += 1
                if len(batch_inputs) >= int(protocol["runtime"]["batch_size"]):
                    flush()
            if accepted != count_per_stratum:
                raise RuntimeError(
                    f"Only accepted {accepted}/{count_per_stratum} simulations for "
                    f"fold {fit_fold} {sensor_name} after {attempts} attempts"
                )
            print(
                json.dumps(
                    {
                        "fit_fold": fit_fold,
                        "sensor": sensor_name,
                        "accepted": accepted,
                        "attempts": attempts,
                        "rows_buffered_or_written": cursor + len(batch_inputs),
                    }
                ),
                flush=True,
            )
    flush()
    if cursor != expected_rows:
        raise RuntimeError("Simulation extraction row count is incomplete")

    if args.smoke:
        print(
            json.dumps(
                {
                    "ok": True,
                    "smoke": True,
                    "rows": cursor,
                    "fit_folds": fit_folds,
                    "sensors": sensors,
                    "maximum_wind_delta": max(wind_deltas),
                    "minimum_visible_fraction": min(visible_fractions),
                    "failure_counts": dict(failure_counts),
                },
                indent=2,
            )
        )
        return 0

    assert images is not None and output_images is not None and temporary_images is not None
    images.flush()
    del images
    images_hash = sha256(temporary_images)
    os.replace(temporary_images, output_images)
    metadata_path = (ROOT / protocol["outputs"]["metadata"]).resolve()
    atomic_savez(
        metadata_path,
        channel_names=np.asarray(CHANNEL_NAMES),
        simulation_ids=np.asarray(simulation_ids),
        background_sample_ids=np.asarray(background_sample_ids),
        source_sample_ids=np.asarray(source_sample_ids),
        groups=np.asarray(groups),
        sensors=np.asarray(sensors, dtype=np.uint8),
        fit_folds=np.asarray(fit_folds, dtype=np.uint8),
        labels=np.ones(expected_rows, dtype=np.uint8),
        wind_speed_deltas=np.asarray(wind_deltas, dtype=np.float32),
        scales=np.asarray(scales, dtype=np.float32),
        rotations=np.asarray(rotations, dtype=np.float32),
        visible_fractions=np.asarray(visible_fractions, dtype=np.float32),
        plume_pixels=np.asarray(plume_pixels, dtype=np.int32),
        images_sha256=np.asarray(images_hash),
        protocol_sha256=np.asarray(sha256(protocol_path)),
        manifest_sha256=np.asarray(sha256(paths["manifest"])),
    )
    receipt = {
        "schema_version": 1,
        "scope": "fit-fold-only wind-matched methane interventions for scene-ranker training",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows": expected_rows,
        "shape": [expected_rows, len(CHANNEL_NAMES), 64, 64],
        "fit_fold_counts": {
            str(fold): int(np.count_nonzero(np.asarray(fit_folds) == fold))
            for fold in sorted(set(fit_folds))
        },
        "sensor_counts": {
            SENSOR_NAMES[index]: int(np.count_nonzero(np.asarray(sensors) == index))
            for index in range(len(SENSOR_NAMES))
        },
        "simulation": {
            "maximum_wind_speed_delta": max(wind_deltas),
            "minimum_visible_fraction": min(visible_fractions),
            "scale_range": [min(scales), max(scales)],
            "failure_counts": dict(failure_counts),
        },
        "outputs": {
            "images": {
                "path": output_images.relative_to(ROOT).as_posix(),
                "bytes": visible_size(output_images),
                "sha256": images_hash,
                "tracked": False,
            },
            "metadata": {
                "path": metadata_path.relative_to(ROOT).as_posix(),
                "bytes": visible_size(metadata_path),
                "sha256": sha256(metadata_path),
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
            "Each branch uses only plume fields and no-plume backgrounds from its fitting fold.",
            "Each sensor contributes exactly half of each fitting-fold cache.",
            "Source and target wind speeds differ by no more than 1.5 m/s.",
            "Target wind exceeds neither 9 m/s nor the clear/onshore gate.",
            "No fold 0/1/2, exact-paper, or fresh-external input was accessed.",
        ],
    }
    receipt_path = (ROOT / protocol["outputs"]["receipt"]).resolve()
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_receipt = receipt_path.with_suffix(receipt_path.suffix + ".tmp")
    temporary_receipt.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary_receipt, receipt_path)
    print(json.dumps({"ok": True, **receipt}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
