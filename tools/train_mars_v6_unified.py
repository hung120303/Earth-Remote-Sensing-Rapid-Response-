#!/usr/bin/env python3
"""Train or smoke-test the product-aware ERSRR v6 unified architecture."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "EarthRemoteSensingRapidResponse"
for path in (MODEL_ROOT, ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from acquire_mars_metadata import sha256  # noqa: E402
from extract_mars_prithvi_scene_features import (  # noqa: E402
    date_coordinate,
    reference_date_coordinate,
)
from mars_v6_product_model import (  # noqa: E402
    PrithviPairEncoder,
    ProductHarmonizedMultiCohortV6,
    canonicalize_mars,
    canonicalize_methanes2cm,
)
from train_mars_paper_residual import (  # noqa: E402
    MarsPaperDataset,
    iter_development_manifest,
)
from train_methanes2cm_v5 import PackedMethaneS2CMDataset  # noqa: E402


DEFAULT_PROTOCOL = Path("configs/mars_v6_unified_protocol.json")


def iter_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def verify_protocol(
    protocol_path: Path, protocol: dict[str, Any], *, smoke: bool
) -> dict[str, Path]:
    status = str(protocol["status"])
    frozen = status.startswith("frozen")
    if not smoke and not frozen:
        raise ValueError("V6 outcome training requires a frozen protocol")
    trainer = (ROOT / protocol["trainer"]["path"]).resolve()
    if frozen and sha256(trainer) != protocol["trainer"]["sha256"]:
        raise ValueError("Frozen v6 trainer hash mismatch")
    if frozen:
        for dependency in protocol["code_dependencies"]:
            path = (ROOT / dependency["path"]).resolve()
            if sha256(path) != dependency["sha256"]:
                raise ValueError(f"Frozen v6 dependency mismatch: {dependency['path']}")
    paths: dict[str, Path] = {}
    for name, contract in protocol["inputs"].items():
        path = (ROOT / contract["path"]).resolve()
        if not path.exists():
            raise FileNotFoundError(f"V6 input is unavailable: {name}={path}")
        if frozen and path.is_file() and sha256(path) != contract["sha256"]:
            raise ValueError(f"Frozen v6 input mismatch: {name}")
        paths[name] = path
    return paths


def build_foundation(
    paths: dict[str, Path], device: torch.device
) -> tuple[torch.nn.Module, dict[str, Any]]:
    receipt = json.loads(paths["foundation_receipt"].read_text(encoding="utf-8"))
    foundation_dir = paths["foundation_config"].parent
    for item in receipt["files"]:
        path = (ROOT / item["path"]).resolve()
        if path.stat().st_size != int(item["bytes"]) or sha256(path) != item["sha256"]:
            raise ValueError(f"Prithvi foundation identity mismatch: {item['path']}")
    if str(foundation_dir) not in sys.path:
        sys.path.insert(0, str(foundation_dir))
    from prithvi_mae import PrithviMAE  # type: ignore  # noqa: E402

    config = json.loads(paths["foundation_config"].read_text(encoding="utf-8"))[
        "pretrained_cfg"
    ]
    model_config = dict(config)
    model_config.update(img_size=128, num_frames=2, in_chans=6)
    model = PrithviMAE(**model_config)
    state = torch.load(paths["foundation_checkpoint"], map_location="cpu", weights_only=True)
    # Positional embeddings are fixed sin/cos buffers. Recreate them for the
    # declared two-frame 128px contract, matching the established repository path.
    state["encoder.pos_embed"] = model.encoder.pos_embed
    state["decoder.decoder_pos_embed"] = model.decoder.decoder_pos_embed
    model.load_state_dict(state, strict=True)
    return model.to(device), config


def build_model(
    paths: dict[str, Path], spec: dict[str, Any], device: torch.device
) -> ProductHarmonizedMultiCohortV6:
    scene_foundation, config = build_foundation(paths, device)
    dense_foundation, second_config = build_foundation(paths, device)
    if second_config != config:
        raise RuntimeError("Prithvi branch configurations differ")
    adapter = spec["adapter"]
    model = ProductHarmonizedMultiCohortV6(
        PrithviPairEncoder(scene_foundation, adapter),
        PrithviPairEncoder(dense_foundation, adapter),
        torch.tensor(config["mean"]),
        torch.tensor(config["std"]),
        product_embedding_dim=int(spec["product_embedding_dim"]),
        sensor_embedding_dim=int(spec["sensor_embedding_dim"]),
        scene_hidden=int(spec["scene_hidden"]),
        scene_topk=int(spec["scene_topk"]),
    )
    return model.to(device)


def stack_items(items: list[dict[str, Any]], keys: tuple[str, ...]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in keys:
        values = [item[key] for item in items]
        result[key] = torch.stack(values) if torch.is_tensor(values[0]) else values
    return result


def balanced_pair(rows: list[dict[str, Any]], *, source: str) -> list[dict[str, Any]]:
    selected = []
    for target in (0, 1):
        for row in rows:
            value = int(row["label"]) if "label" in row else int(row["label_state"] == "PLUME")
            if value == target:
                selected.append(row)
                break
    if len(selected) != 2:
        raise ValueError(f"{source} smoke requires one positive and one negative")
    return selected


def mars_temporal_location(
    rows: list[dict[str, Any]], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    temporal = []
    location = []
    available = []
    for row in rows:
        target = date_coordinate(str(row["target_datetime"]))
        has_reference = bool(str(row.get("reference_scene_id", "")).strip())
        reference = reference_date_coordinate(row) if has_reference else target
        temporal.append((reference, reference, target))
        location.append((float(row["latitude"]), float(row["longitude"])))
        available.append(float(has_reference))
    return (
        torch.tensor(temporal, dtype=torch.float32, device=device),
        torch.tensor(location, dtype=torch.float32, device=device),
        torch.tensor(available, dtype=torch.float32, device=device),
    )


def methanes2cm_temporal_location(
    rows: list[dict[str, Any]], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    # MethaneS2CM publishes relative T/T-90/T-365 frames but omits timestamps.
    # Use fixed calendar-valid proxies; product identity remains explicit.
    temporal = torch.tensor(
        [[[2020.0, 276.0], [2020.0, 1.0], [2021.0, 1.0]]] * len(rows),
        dtype=torch.float32,
        device=device,
    )
    location = torch.tensor(
        [[float(row["latitude"]), float(row["longitude"])] for row in rows],
        dtype=torch.float32,
        device=device,
    )
    return temporal, location


def center_crop(batch: dict[str, Any], pixels: int) -> dict[str, Any]:
    height, width = batch["inputs"].shape[-2:]
    if height < pixels or width < pixels:
        raise ValueError("Dense smoke crop is larger than the source")
    y = (height - pixels) // 2
    x = (width - pixels) // 2
    region = np.s_[..., y : y + pixels, x : x + pixels]
    result = dict(batch)
    for key in ("inputs", "observable", "mask"):
        result[key] = result[key][region]
    return result


def dense_loss(logits: torch.Tensor, mask: torch.Tensor, observable: torch.Tensor) -> torch.Tensor:
    valid = observable > 0.5
    target = mask.float()
    per_pixel = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    weights = 1.0 + 3.0 * target
    bce = (per_pixel[valid] * weights[valid]).mean()
    probability = torch.sigmoid(logits) * observable
    intersection = (probability * target).sum(dim=(1, 2, 3))
    dice = 1.0 - ((2.0 * intersection + 1.0) / (
        probability.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + 1.0
    )).mean()
    return bce + 0.3 * dice


def optimization_step(
    model: ProductHarmonizedMultiCohortV6,
    *,
    branch: str,
    canonical: Any,
    temporal: torch.Tensor,
    location: torch.Tensor,
    target: torch.Tensor,
    observable: torch.Tensor,
    learning_rate: float,
    gradient_clip: float,
) -> dict[str, Any]:
    model.set_trainable_phase(branch)  # type: ignore[arg-type]
    parameters = [value for value in model.parameters() if value.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=0.01)
    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        output = model(canonical, temporal, location, branch=branch)  # type: ignore[arg-type]
        value = output["scene_logit"] if branch == "scene" else output["dense_logits"]
        loss = (
            F.binary_cross_entropy_with_logits(value, target.float())
            if branch == "scene"
            else dense_loss(value, target, observable)
        )
    loss.backward()
    gradients = [value.grad for value in parameters if value.grad is not None]
    if not gradients or not all(torch.isfinite(value).all() for value in gradients):
        raise FloatingPointError(f"Non-finite or missing {branch} gradients")
    norm = torch.nn.utils.clip_grad_norm_(parameters, gradient_clip)
    optimizer.step()
    return {
        "loss": float(loss.detach()),
        "output_shape": list(value.shape),
        "finite": bool(torch.isfinite(value).all()),
        "trainable_parameters": int(sum(item.numel() for item in parameters)),
        "gradient_norm_before_clip": float(norm),
    }


def run_smoke(
    protocol: dict[str, Any], paths: dict[str, Path], output: Path
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("The preregistered v6 smoke requires CUDA")
    device = torch.device("cuda")
    torch.manual_seed(int(protocol["training"]["seeds"][0]))
    torch.cuda.manual_seed_all(int(protocol["training"]["seeds"][0]))
    all_mars = list(iter_development_manifest(paths["mars_manifest"]))
    folds = json.loads(paths["mars_folds"].read_text(encoding="utf-8"))
    group_to_fold = {str(item["group_id"]): int(item["fold"]) for item in folds["assignments"]}
    mars_rows = balanced_pair(
        [row for row in all_mars if group_to_fold[str(row["group_id"])] in {3, 4}],
        source="MARS",
    )
    methane_rows = balanced_pair(iter_jsonl(paths["methanes2cm_auxiliary"]), source="MethaneS2CM")
    mars_items = [
        MarsPaperDataset(paths["mars_root"], [row], augment=False, seed=0)[0]
        for row in mars_rows
    ]
    methane_dataset = PackedMethaneS2CMDataset(
        paths["methanes2cm_packed"], methane_rows, augment=False, seed=0
    )
    methane_items = [methane_dataset[index] for index in range(2)]
    mars_batch = stack_items(
        mars_items, ("inputs", "observable", "mask", "presence", "sensor_index")
    )
    methane_batch = stack_items(
        methane_items, ("inputs", "observable", "mask", "presence")
    )
    mars_batch = {
        name: value.to(device) if torch.is_tensor(value) else value
        for name, value in mars_batch.items()
    }
    methane_batch = {
        name: value.to(device) if torch.is_tensor(value) else value
        for name, value in methane_batch.items()
    }
    mars_temporal, mars_location, mars_available = mars_temporal_location(mars_rows, device)
    methane_temporal, methane_location = methanes2cm_temporal_location(methane_rows, device)
    model = build_model(paths, protocol["architecture"], device)
    total_parameters = sum(value.numel() for value in model.parameters())
    torch.cuda.reset_peak_memory_stats()
    results: dict[str, Any] = {}
    for source, raw, temporal, location in (
        ("mars", mars_batch, mars_temporal, mars_location),
        ("methanes2cm", methane_batch, methane_temporal, methane_location),
    ):
        canonical = (
            canonicalize_mars(
                raw["inputs"],
                raw["observable"],
                raw["sensor_index"],
                reference90_available=mars_available,
            )
            if source == "mars"
            else canonicalize_methanes2cm(raw["inputs"], raw["observable"])
        )
        results[f"{source}_scene"] = optimization_step(
            model,
            branch="scene",
            canonical=canonical,
            temporal=temporal,
            location=location,
            target=raw["presence"],
            observable=raw["observable"],
            learning_rate=float(protocol["training"]["scene_learning_rate"]),
            gradient_clip=float(protocol["training"]["gradient_clip"]),
        )
        dense_raw = center_crop(raw, 64) if source == "mars" else raw
        dense_canonical = (
            canonicalize_mars(
                dense_raw["inputs"],
                dense_raw["observable"],
                dense_raw["sensor_index"],
                reference90_available=mars_available,
            )
            if source == "mars"
            else canonicalize_methanes2cm(dense_raw["inputs"], dense_raw["observable"])
        )
        results[f"{source}_dense"] = optimization_step(
            model,
            branch="dense",
            canonical=dense_canonical,
            temporal=temporal,
            location=location,
            target=dense_raw["mask"],
            observable=dense_raw["observable"],
            learning_rate=float(protocol["training"]["dense_learning_rate"]),
            gradient_clip=float(protocol["training"]["gradient_clip"]),
        )
    baseline = torch.tensor([0.1, 0.3], device=device)
    residual = torch.tensor([10.0, -2.0], device=device)
    protected = model.protected_scene_score(
        baseline,
        residual,
        strength=float(protocol["search"]["strengths"][0]),
        protection_gate=float(protocol["architecture"]["protection_gate"]),
    )
    report = {
        "schema_version": 1,
        "scope": "real mixed-source v6 finite-optimization smoke; no held-fold outcome scoring",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": torch.cuda.get_device_name(device),
        "total_parameters": int(total_parameters),
        "peak_cuda_bytes": int(torch.cuda.max_memory_allocated()),
        "steps": results,
        "protected_identity_below_gate_exact": bool(protected[0].item() == baseline[0].item()),
        "mars_sample_ids": [str(row["sample_id"]) for row in mars_rows],
        "methanes2cm_sample_ids": [str(row["id"]) for row in methane_rows],
        "held_outcomes_accessed": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL.as_posix())
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--smoke-output", default="reports/experiments/mars_v6_unified_smoke.json"
    )
    args = parser.parse_args()
    protocol_path = (ROOT / args.protocol).resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    paths = verify_protocol(protocol_path, protocol, smoke=args.smoke)
    if args.smoke:
        report = run_smoke(protocol, paths, (ROOT / args.smoke_output).resolve())
        print(json.dumps(report, sort_keys=True))
        return 0
    raise NotImplementedError(
        "The current preregistration permits only mixed-source smoke. Full outcome training "
        "will be enabled only after smoke hashes and the endpoint implementation are frozen."
    )


if __name__ == "__main__":
    raise SystemExit(main())
