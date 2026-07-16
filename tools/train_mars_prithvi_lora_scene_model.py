#!/usr/bin/env python3
"""Train a patch-supervised LoRA adaptation of Prithvi on MARS development folds."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "EarthRemoteSensingRapidResponse"
for path in (MODEL_ROOT, ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from acquire_mars_metadata import sha256  # noqa: E402
from extract_mars_prithvi_scene_features import (  # noqa: E402
    INPUT_SIZE,
    build_input,
    date_coordinate,
    reference_date_coordinate,
)
from mars_prithvi_lora_model import (  # noqa: E402
    PrithviLoRASceneModel,
    load_trainable_state,
    trainable_parameter_count,
    trainable_state,
)
from train_mars_crossfold_bagged_scene_head import load_development  # noqa: E402
from train_mars_oof_scene_ensemble_v2 import ap_group_bootstrap  # noqa: E402
from train_mars_paper_residual import (  # noqa: E402
    MarsPaperDataset,
    iter_development_manifest,
    move_batch,
    verify_acquisition_receipt,
)
from train_mars_prithvi_spatial_head import patch_supervision_loss  # noqa: E402
from train_mars_scene_ranker import blend_scores, comparison, metric_summary  # noqa: E402


DEFAULT_PROTOCOL = Path("configs/mars_prithvi_lora_scene_protocol.json")


def foundation_model(paths: dict[str, Path], device: torch.device) -> tuple[nn.Module, torch.Tensor, torch.Tensor]:
    receipt = json.loads(paths["foundation_receipt"].read_text(encoding="utf-8"))
    foundation_dir = paths["foundation_config"].parent
    for item in receipt["files"]:
        path = (ROOT / item["path"]).resolve()
        if path.stat().st_size != int(item["bytes"]) or sha256(path) != item["sha256"]:
            raise ValueError(f"Prithvi foundation identity mismatch: {item['path']}")
    sys.path.insert(0, str(foundation_dir))
    from prithvi_mae import PrithviMAE  # type: ignore  # noqa: E402

    config = json.loads(paths["foundation_config"].read_text(encoding="utf-8"))["pretrained_cfg"]
    means = torch.tensor(config["mean"], dtype=torch.float32, device=device)[None, :, None, None, None]
    stds = torch.tensor(config["std"], dtype=torch.float32, device=device)[None, :, None, None, None]
    model_config = dict(config)
    model_config.update(img_size=INPUT_SIZE, num_frames=2, in_chans=6)
    model = PrithviMAE(**model_config)
    state = torch.load(paths["foundation_checkpoint"], map_location="cpu", weights_only=True)
    state["encoder.pos_embed"] = model.encoder.pos_embed
    state["decoder.decoder_pos_embed"] = model.decoder.decoder_pos_embed
    model.load_state_dict(state, strict=True)
    return model.to(device), means, stds


def coordinates(
    sample_ids: list[str], metadata: dict[str, dict[str, Any]], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    temporal = torch.tensor(
        [[reference_date_coordinate(metadata[value]), date_coordinate(str(metadata[value]["target_datetime"]))] for value in sample_ids],
        dtype=torch.float32,
        device=device,
    )
    location = torch.tensor(
        [[float(metadata[value]["latitude"]), float(metadata[value]["longitude"])] for value in sample_ids],
        dtype=torch.float32,
        device=device,
    )
    return temporal, location


def record_weights(records: list[dict[str, Any]]) -> torch.Tensor:
    groups = Counter(str(record["group_id"]) for record in records)
    classes = Counter(str(record["label_state"]) for record in records)
    sensors = Counter(str(record["sensor_family"]) for record in records)
    values = []
    for record in records:
        weight = 1.0 / groups[str(record["group_id"])]
        weight *= math.sqrt(len(records) / classes[str(record["label_state"])])
        weight *= math.sqrt(len(records) / sensors[str(record["sensor_family"])])
        values.append(weight)
    result = torch.tensor(values, dtype=torch.double)
    return result / result.mean()


def patch_targets(batch: dict[str, Any], grid: int) -> torch.Tensor:
    plume = F.adaptive_max_pool2d(batch["mask"] * batch["observable"], (grid, grid))
    visible = F.adaptive_avg_pool2d(batch["observable"], (grid, grid))
    return torch.cat((plume, visible), dim=1)


def build_model(
    paths: dict[str, Path], spec: dict[str, Any], device: torch.device,
    state: dict[str, torch.Tensor] | None = None,
) -> tuple[PrithviLoRASceneModel, torch.Tensor, torch.Tensor]:
    foundation, means, stds = foundation_model(paths, device)
    model = PrithviLoRASceneModel(foundation, spec).to(device)
    if state is not None:
        load_trainable_state(model, state)
    return model, means, stds


def train_one(
    records: list[dict[str, Any]],
    metadata: dict[str, dict[str, Any]],
    paths: dict[str, Path],
    spec: dict[str, Any],
    seed: int,
    device: torch.device,
    *,
    epochs: int,
) -> tuple[dict[str, torch.Tensor], list[dict[str, float]], int]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    model, means, stds = build_model(paths, spec, device)
    lora_parameters = []
    head_parameters = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if ".a." in name or ".b." in name:
            lora_parameters.append(parameter)
        else:
            head_parameters.append(parameter)
    optimizer = torch.optim.AdamW(
        [
            {"params": lora_parameters, "lr": float(spec["lora_learning_rate"])},
            {"params": head_parameters, "lr": float(spec["head_learning_rate"])},
        ],
        weight_decay=float(spec["weight_decay"]),
    )
    sampler = WeightedRandomSampler(
        record_weights(records), num_samples=len(records), replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )
    loader = DataLoader(
        MarsPaperDataset(paths["metadata_root"], records, augment=True, seed=seed),
        batch_size=int(spec["batch_size"]), sampler=sampler, num_workers=0,
        pin_memory=False,
    )
    history: list[dict[str, float]] = []
    for epoch in range(epochs):
        model.train()
        sums = {"loss": 0.0, "scene": 0.0, "patch": 0.0, "pair": 0.0}
        batches = 0
        for batch in loader:
            local_ids = [str(value) for value in batch["sample_id"]]
            temporal, location = coordinates(local_ids, metadata, device)
            batch = move_batch(batch, device)
            values = build_input(batch, means, stds)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                output = model(values, temporal, location)
                scene = F.binary_cross_entropy_with_logits(output["scene_logit"], batch["presence"])
                target = patch_targets(batch, output["patch_logits"].shape[-1])
                patch_rows = patch_supervision_loss(
                    output["patch_logits"], target, batch["presence"], batch["pixel_truth_available"]
                )
                patch = patch_rows.mean()
                positive = output["scene_logit"][batch["presence"] > 0.5]
                negative = output["scene_logit"][batch["presence"] < 0.5]
                pair = (
                    F.softplus(float(spec["pair_margin"]) - positive[:, None] + negative[None, :]).mean()
                    if positive.numel() and negative.numel()
                    else output["scene_logit"].sum() * 0.0
                )
                loss = scene + float(spec["patch_loss_weight"]) * patch + float(spec["pair_loss_weight"]) * pair
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                float(spec["gradient_clip"]),
            )
            optimizer.step()
            batches += 1
            for name, value in (("loss", loss), ("scene", scene), ("patch", patch), ("pair", pair)):
                sums[name] += float(value.detach())
        row = {"epoch": epoch + 1, **{name: value / max(batches, 1) for name, value in sums.items()}}
        history.append(row)
        print(json.dumps({"seed": seed, "rows": len(records), **row}), flush=True)
    state = trainable_state(model)
    count = trainable_parameter_count(model)
    del model
    torch.cuda.empty_cache()
    return state, history, count


@torch.no_grad()
def score_records(
    records: list[dict[str, Any]],
    metadata: dict[str, dict[str, Any]],
    paths: dict[str, Path],
    spec: dict[str, Any],
    state: dict[str, torch.Tensor],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model, means, stds = build_model(paths, spec, device, state)
    model.eval()
    loader = DataLoader(
        MarsPaperDataset(paths["metadata_root"], records, augment=False, seed=0),
        batch_size=int(spec["evaluation_batch_size"]), shuffle=False, num_workers=0,
        pin_memory=False,
    )
    ids: list[str] = []
    scores: list[np.ndarray] = []
    for batch in loader:
        local_ids = [str(value) for value in batch["sample_id"]]
        temporal, location = coordinates(local_ids, metadata, device)
        batch = move_batch(batch, device)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            output = model(build_input(batch, means, stds), temporal, location)
        ids.extend(local_ids)
        scores.append(torch.sigmoid(output["scene_logit"]).float().cpu().numpy())
    del model
    torch.cuda.empty_cache()
    return np.asarray(ids), np.concatenate(scores).astype(np.float64)


def evaluate(
    values: dict[str, Any], raw: np.ndarray, folds: list[int], blend: float
) -> dict[str, Any]:
    rows = np.isin(values["folds"], folds)
    scores = blend_scores(values["current"][rows], raw[rows], blend)
    candidate = metric_summary(values["labels"][rows], scores, values["sensors"][rows])
    current = metric_summary(values["labels"][rows], values["current"][rows], values["sensors"][rows])
    versus = comparison(candidate, current)
    per_fold = {}
    for fold in folds:
        local = values["folds"] == fold
        local_scores = blend_scores(values["current"][local], raw[local], blend)
        per_fold[str(fold)] = comparison(
            metric_summary(values["labels"][local], local_scores, values["sensors"][local]),
            metric_summary(values["labels"][local], values["current"][local], values["sensors"][local]),
        )
    fold_ap = [item["delta"]["average_precision"] for item in per_fold.values()]
    fold_recall = [item["delta"]["recall_at_fpr_0_0713"] for item in per_fold.values()]
    stable = (
        versus["delta"]["average_precision"] > 0.0
        and versus["delta"]["recall_at_fpr_0_0713"] >= 0.0
        and min(fold_ap) > 0.0
        and min(fold_recall) >= -0.002
        and min(versus["delta"]["sensor_average_precision"].values()) >= -0.0025
    )
    return {
        "blend_weight": blend,
        "metrics": candidate,
        "versus_current": versus,
        "per_fold": per_fold,
        "stable": bool(stable),
        "rank": [int(stable), min(fold_ap), versus["delta"]["average_precision"], versus["delta"]["recall_at_fpr_0_0713"], -blend],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL.as_posix())
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    protocol_path = (ROOT / args.protocol).resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if sha256(Path(__file__).resolve()) != protocol["trainer"]["sha256"]:
        raise ValueError("Prithvi LoRA trainer hash mismatch")
    for dependency in protocol["code_dependencies"]:
        path = (ROOT / dependency["path"]).resolve()
        if sha256(path) != dependency["sha256"]:
            raise ValueError(f"Frozen LoRA code dependency mismatch: {dependency['path']}")
    paths: dict[str, Path] = {}
    for name, contract in protocol["inputs"].items():
        path = (ROOT / contract["path"]).resolve()
        if path.is_file() and sha256(path) != contract["sha256"]:
            raise ValueError(f"Frozen LoRA input hash mismatch: {name}")
        paths[name] = path
    if not paths["metadata_root"].is_dir():
        raise ValueError("MARS development metadata root is unavailable")
    manifest_hash = sha256(paths["manifest"])
    verify_acquisition_receipt(paths["acquisition_receipt"], manifest_hash)
    fold_protocol = json.loads(paths["fold_protocol"].read_text(encoding="utf-8"))
    group_to_fold = {str(item["group_id"]): int(item["fold"]) for item in fold_protocol["assignments"]}
    records = list(iter_development_manifest(paths["manifest"]))
    metadata = {str(record["sample_id"]): record for record in records}
    if len(metadata) != len(records):
        raise ValueError("Development manifest sample IDs are not unique")
    values = load_development(
        {"inner": paths["inner"], "fold0": paths["fold0"], "fold1": paths["fold1"]},
        paths["scores"],
    )
    if set(values["sample_ids"].tolist()) != set(metadata):
        raise ValueError("Development feature and raw manifest identities differ")
    spec = protocol["architecture"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Prithvi LoRA training requires CUDA")
    if args.smoke:
        selected = []
        counts: Counter[tuple[str, str]] = Counter()
        for record in records:
            key = (str(record["label_state"]), str(record["sensor_family"]))
            if counts[key] < 4:
                selected.append(record)
                counts[key] += 1
            if len(counts) == 4 and min(counts.values()) >= 4:
                break
        state, history, count = train_one(selected, metadata, paths, spec, 20261800, device, epochs=1)
        ids, score = score_records(selected, metadata, paths, spec, state, device)
        print(json.dumps({"ok": bool(len(ids) == len(selected) and np.isfinite(score).all()), "rows": len(ids), "trainable_parameters": count, "history": history}))
        return 0

    selection_folds = list(map(int, protocol["folds"]["selection"]))
    audit_folds = list(map(int, protocol["folds"]["reused_holdout_audit"]))
    seed = int(protocol["training"]["pilot_seed"])
    raw = np.full(values["labels"].shape, np.nan, dtype=np.float64)
    lookup = {value: index for index, value in enumerate(values["sample_ids"].tolist())}
    selection_history: dict[str, Any] = {}
    parameter_count = None
    for held_fold in selection_folds:
        fit_records = [record for record in records if group_to_fold[str(record["group_id"])] in selection_folds and group_to_fold[str(record["group_id"])] != held_fold]
        held_records = [record for record in records if group_to_fold[str(record["group_id"])] == held_fold]
        state, history, count = train_one(
            fit_records, metadata, paths, spec, seed + held_fold, device,
            epochs=int(protocol["training"]["selection_epochs"]),
        )
        ids, local_scores = score_records(held_records, metadata, paths, spec, state, device)
        raw[np.asarray([lookup[value] for value in ids])] = local_scores
        selection_history[str(held_fold)] = history
        parameter_count = count
        print(json.dumps({"completed_selection_fold": held_fold, "fit_rows": len(fit_records), "held_rows": len(held_records)}), flush=True)
    candidates = [evaluate(values, raw, selection_folds, float(blend)) for blend in protocol["search"]["current_blends"]]
    selected = max(candidates, key=lambda item: tuple(item["rank"]))
    selection_rows = np.isin(values["folds"], selection_folds)
    selection_scores = blend_scores(values["current"][selection_rows], raw[selection_rows], selected["blend_weight"])
    selected["bootstrap"] = ap_group_bootstrap(
        values["labels"][selection_rows], values["current"][selection_rows], selection_scores,
        values["groups"][selection_rows], replicates=int(protocol["bootstrap"]["replicates"]),
        seed=int(protocol["bootstrap"]["selection_seed"]),
    )
    selection_passed = bool(selected["stable"] and selected["bootstrap"]["lower"] > 0.0)

    fit_records = [record for record in records if group_to_fold[str(record["group_id"])] in selection_folds]
    state, audit_history, _ = train_one(
        fit_records, metadata, paths, spec, seed + 50, device,
        epochs=int(protocol["training"]["audit_epochs"]),
    )
    for held_fold in audit_folds:
        held_records = [record for record in records if group_to_fold[str(record["group_id"])] == held_fold]
        ids, local_scores = score_records(held_records, metadata, paths, spec, state, device)
        raw[np.asarray([lookup[value] for value in ids])] = local_scores
    audit = evaluate(values, raw, audit_folds, float(selected["blend_weight"]))
    audit_rows = np.isin(values["folds"], audit_folds)
    audit_scores = blend_scores(values["current"][audit_rows], raw[audit_rows], selected["blend_weight"])
    audit["bootstrap"] = ap_group_bootstrap(
        values["labels"][audit_rows], values["current"][audit_rows], audit_scores,
        values["groups"][audit_rows], replicates=int(protocol["bootstrap"]["replicates"]),
        seed=int(protocol["bootstrap"]["audit_seed"]),
    )
    audit_passed = bool(audit["stable"] and audit["bootstrap"]["lower"] > 0.0)
    passed = selection_passed and audit_passed
    artifact_record = None
    score_record = None
    if passed:
        final_state, final_history, _ = train_one(
            records, metadata, paths, spec, seed + 100, device,
            epochs=int(protocol["training"]["final_epochs"]),
        )
        artifact_path = (ROOT / protocol["outputs"]["artifact"]).resolve()
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = artifact_path.with_suffix(artifact_path.suffix + ".tmp")
        torch.save({
            "schema_version": 1,
            "kind": "mars_prithvi_lora_scene_model",
            "architecture": spec,
            "seed": seed + 100,
            "blend_weight": float(selected["blend_weight"]),
            "trainable_state": final_state,
            "trainable_parameters": parameter_count,
            "foundation_checkpoint_sha256": protocol["inputs"]["foundation_checkpoint"]["sha256"],
            "operational_scene_threshold": float(protocol["training"]["operational_scene_threshold"]),
            "protocol_sha256": sha256(protocol_path),
            "final_history": final_history,
        }, temporary)
        os.replace(temporary, artifact_path)
        artifact_record = {"path": protocol["outputs"]["artifact"], "bytes": artifact_path.stat().st_size, "sha256": sha256(artifact_path), "tracked": False}
        score_path = (ROOT / protocol["outputs"]["development_scores"]).resolve()
        score_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_score = score_path.with_suffix(".tmp.npz")
        np.savez_compressed(
            temporary_score, sample_ids=values["sample_ids"], groups=values["groups"], folds=values["folds"],
            labels=values["labels"], raw_scores=raw,
            candidate_scores=blend_scores(values["current"], raw, selected["blend_weight"]),
        )
        os.replace(temporary_score, score_path)
        score_record = {"path": protocol["outputs"]["development_scores"], "bytes": score_path.stat().st_size, "sha256": sha256(score_path), "tracked": False}
    report = {
        "schema_version": 1,
        "scope": "development-only patch-supervised Prithvi LoRA pilot; no fresh or paper inputs",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "architecture": spec,
        "trainable_parameters": parameter_count,
        "selection_candidates": candidates,
        "selected": selected,
        "selection_passed": selection_passed,
        "reused_holdout_audit": {**audit, "independent_confirmation": False, "reason": "folds 0/1 were exposed by predecessor experiments"},
        "reused_holdout_audit_passed": audit_passed,
        "all_promotion_gates_pass": passed,
        "artifact": artifact_record,
        "development_score_cache": score_record,
        "training_history": {"selection": selection_history, "audit": audit_history},
        "decision": "Freeze LoRA pilot for a second-seed replication." if passed else "Reject LoRA pilot before external scoring.",
        "provenance": {"protocol_sha256": sha256(protocol_path), "script_sha256": sha256(Path(__file__).resolve()), "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), "device": str(torch.cuda.get_device_name(device)), "torch": torch.__version__, "numpy": np.__version__},
    }
    output = (ROOT / protocol["outputs"]["json"]).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"ok": passed, "blend": selected["blend_weight"], "selection_ap_delta": selected["versus_current"]["delta"]["average_precision"], "selection_ap_lower": selected["bootstrap"]["lower"], "audit_ap_delta": audit["versus_current"]["delta"]["average_precision"], "audit_ap_lower": audit["bootstrap"]["lower"]}, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
