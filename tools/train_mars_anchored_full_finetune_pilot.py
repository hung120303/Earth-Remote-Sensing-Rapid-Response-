#!/usr/bin/env python3
"""Cross-fit weight-anchored full-model MARS-S2L adaptation."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import ConcatDataset, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "EarthRemoteSensingRapidResponse"
for path in (MODEL_ROOT, ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from acquire_mars_metadata import sha256  # noqa: E402
from mars_anchored_full_finetune import AnchoredMarsFullFinetune  # noqa: E402
from mars_paper_model import released_state  # noqa: E402
from mars_s2l_adapter import label_state  # noqa: E402
from train_mars_dense_prithvi_teacher_pilot import fusion_loss  # noqa: E402
from train_mars_dinov3_methane_fusion_pilot import (  # noqa: E402
    load_base_score_contract,
    partial_auc_pair_loss,
)
from train_mars_ndmi_bitemporal_fusion_pilot import (  # noqa: E402
    TaggedDataset,
    collect_predictions,
    combined_sampling_weights,
    iter_jsonl,
    make_loader,
    merge_predictions,
    smoke_records,
    summarize_predictions,
)
from train_mars_paper_residual import (  # noqa: E402
    MarsPaperDataset,
    iter_development_manifest,
    move_batch,
    smoke_subset,
    verify_acquisition_receipt,
)
from train_mars_physics_guided_teacher_pilot import seed_everything  # noqa: E402

DEFAULT_PROTOCOL = Path("configs/mars_anchored_full_finetune_pilot_protocol.json")


def resolve_fold_contract(
    protocol: dict[str, Any],
) -> tuple[set[int], set[int], dict[int, set[int]]]:
    """Return evaluation folds, authorized rows, and per-endpoint fit folds."""
    evaluation = set(map(int, protocol["folds"]))
    if not evaluation:
        raise ValueError("At least one evaluation fold is required")
    configured = protocol.get("fit_folds_by_held")
    if configured is None:
        mapping = {held: evaluation - {held} for held in evaluation}
    else:
        mapping = {
            int(held): set(map(int, fit_folds))
            for held, fit_folds in configured.items()
        }
        if set(mapping) != evaluation:
            raise ValueError("Fit-fold mapping keys must equal evaluation folds")
    for held, fit_folds in mapping.items():
        if not fit_folds or held in fit_folds:
            raise ValueError(f"Invalid fit folds for held fold {held}")
    authorized = evaluation | set().union(*mapping.values())
    return evaluation, authorized, mapping


def write_scene_prediction_cache(
    path: Path,
    raw: dict[str, Any],
    strengths: list[float],
    *,
    protocol_sha256: str,
) -> dict[str, Any]:
    """Write the compact, identity-aligned scene outputs needed by later fusion."""
    rows = len(raw["sample_ids"])
    aligned_keys = ("labels", "sensors", "groups", "sample_ids", "folds", "base_scores")
    if any(len(raw[key]) != rows for key in aligned_keys):
        raise ValueError("Scene prediction cache fields are not row-aligned")
    candidates = {
        f"candidate_{index}": np.asarray(
            raw["candidate_scores"][str(strength)], dtype=np.float64
        )
        for index, strength in enumerate(strengths)
    }
    if any(values.shape != (rows,) for values in candidates.values()):
        raise ValueError("Candidate scene scores are not row-aligned")
    arrays = {
        "schema_version": np.asarray(1, dtype=np.uint8),
        "protocol_sha256": np.asarray(protocol_sha256),
        "strengths": np.asarray(strengths, dtype=np.float64),
        "labels": np.asarray(raw["labels"], dtype=np.uint8),
        "sensors": np.asarray(raw["sensors"], dtype=np.uint8),
        "groups": np.asarray(raw["groups"]),
        "sample_ids": np.asarray(raw["sample_ids"]),
        "folds": np.asarray(raw["folds"], dtype=np.uint8),
        "base_scores": np.asarray(raw["base_scores"], dtype=np.float64),
        **candidates,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)
    try:
        receipt_path = path.relative_to(ROOT).as_posix()
    except ValueError:
        receipt_path = path.as_posix()
    return {
        "path": receipt_path,
        "rows": rows,
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "tracked": False,
        "contains_dense_pixels": False,
    }


def write_endpoint_state_cache(
    path: Path,
    endpoint_states: dict[str, Any],
    *,
    strengths: list[float],
    protocol_sha256: str,
) -> dict[str, Any]:
    """Persist ignored cross-fit endpoint states for a downstream frozen fusion."""
    if not endpoint_states:
        raise ValueError("Endpoint state cache cannot be empty")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "schema_version": 1,
            "model": AnchoredMarsFullFinetune().artifact_metadata(),
            "states_by_held_fold": endpoint_states,
            "strengths": list(strengths),
            "protocol_sha256": protocol_sha256,
            "research_only_until_downstream_gates_pass": True,
        },
        temporary,
    )
    os.replace(temporary, path)
    try:
        receipt_path = path.relative_to(ROOT).as_posix()
    except ValueError:
        receipt_path = path.as_posix()
    return {
        "path": receipt_path,
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "tracked": False,
        "research_only_until_downstream_gates_pass": True,
    }


def train_endpoint(
    model: AnchoredMarsFullFinetune,
    loader: torch.utils.data.DataLoader[dict[str, Any]],
    spec: dict[str, Any],
    device: torch.device,
) -> list[dict[str, float]]:
    groups = model.parameter_groups(
        backbone_learning_rate=float(spec["backbone_learning_rate"]),
        output_learning_rate=float(spec["output_learning_rate"]),
    )
    optimizer = torch.optim.AdamW(groups, weight_decay=0.0)
    epochs = int(spec["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        sums: dict[str, float] = {}
        batches = 0
        started = time.perf_counter()
        for batch_index, batch in enumerate(loader, start=1):
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                output = model(
                    batch["inputs"], batch["observable"], batch["sensor_index"]
                )
                task_loss, parts = fusion_loss(output, batch, spec)
                partial_auc = partial_auc_pair_loss(
                    output["scene_logit"],
                    batch["presence"],
                    negative_fraction=float(spec["partial_auc_negative_fraction"]),
                    margin=float(spec["partial_auc_margin"]),
                )
                anchor = model.anchor_penalty()
                loss = (
                    task_loss
                    + float(spec["partial_auc_weight"]) * partial_auc
                    + float(spec["anchor_weight"]) * anchor
                )
            loss.backward()
            if not all(
                parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
                for parameter in parameters
            ):
                raise FloatingPointError("Anchored full-model gradient is non-finite")
            torch.nn.utils.clip_grad_norm_(parameters, float(spec["gradient_clip"]))
            optimizer.step()
            parts["loss"] = float(loss.detach())
            parts["partial_auc"] = float(partial_auc.detach())
            parts["anchor"] = float(anchor.detach())
            batches += 1
            for key, value in parts.items():
                sums[key] = sums.get(key, 0.0) + value
            if batch_index % int(spec["progress_every_batches"]) == 0:
                print(
                    json.dumps(
                        {
                            "progress": "training_batch",
                            "epoch": epoch,
                            "batch": batch_index,
                            "elapsed_seconds": time.perf_counter() - started,
                        }
                    ),
                    flush=True,
                )
        scheduler.step()
        row = {
            "epoch": epoch,
            "seconds": time.perf_counter() - started,
            **{key: value / batches for key, value in sums.items()},
        }
        history.append(row)
        print(json.dumps(row), flush=True)
    return history


def verify_protocol(protocol: dict[str, Any], *, smoke: bool) -> dict[str, Path]:
    frozen = str(protocol["status"]).startswith("frozen")
    if not frozen and not smoke:
        raise ValueError("Outcome evaluation requires a frozen protocol")
    if frozen and sha256(Path(__file__).resolve()) != protocol["trainer"]["sha256"]:
        raise ValueError("Frozen anchored-finetune trainer hash mismatch")
    if frozen:
        for dependency in protocol["code_dependencies"]:
            path = (ROOT / dependency["path"]).resolve()
            if sha256(path) != dependency["sha256"]:
                raise ValueError(f"Frozen dependency mismatch: {dependency['path']}")
    paths: dict[str, Path] = {}
    for name, contract in protocol["inputs"].items():
        path = (ROOT / contract["path"]).resolve()
        if frozen and path.is_file() and sha256(path) != contract["sha256"]:
            raise ValueError(f"Frozen input mismatch: {name}")
        paths[name] = path
    if not paths["metadata_root"].is_dir():
        raise ValueError("MARS metadata root is unavailable")
    verify_acquisition_receipt(paths["acquisition_receipt"], sha256(paths["manifest"]))
    return paths


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    selected = report["selected"]
    scene = selected["versus_current"]["delta"]
    ap_ci = selected["paired_site_ap_delta"]
    iou_ci = selected["paired_site_pixel_iou_delta"]
    lines = [
        "# Weight-anchored released-U-Net full fine-tune pilot",
        "",
        f"- Promotion gates pass: **{report['all_promotion_gates_pass']}**",
        f"- Selected interpolation strength: **{selected['strength']}**",
        f"- AP delta versus current spatial-Prithvi score: **{scene['average_precision']:+.6f}**",
        f"- Matched-FPR recall delta: **{scene['recall_at_fpr_0_0713']:+.6f}**",
        f"- Paired-site AP interval: **[{ap_ci['lower']:+.6f}, {ap_ci['upper']:+.6f}]**",
        f"- Dense-mask IoU delta: **{selected['pixel_iou_delta']:+.6f}**",
        f"- Paired-site IoU interval: **[{iou_ci['lower']:+.6f}, {iou_ci['upper']:+.6f}]**",
        "",
        report["decision"],
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL.as_posix())
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    protocol_path = (ROOT / args.protocol).resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    paths = verify_protocol(protocol, smoke=args.smoke)
    fold_protocol = json.loads(paths["fold_protocol"].read_text(encoding="utf-8"))
    group_to_fold = {
        str(row["group_id"]): int(row["fold"])
        for row in fold_protocol["assignments"]
    }
    all_records = list(iter_development_manifest(paths["manifest"]))
    selected_folds, authorized_folds, fit_folds_by_held = resolve_fold_contract(protocol)
    records = [
        row
        for row in all_records
        if group_to_fold[str(row["group_id"])] in authorized_folds
    ]
    unep_records = iter_jsonl(paths["unep_auxiliary_manifest"])
    cloudsen_records = iter_jsonl(paths["cloudsen_auxiliary_manifest"])
    if any(label_state(row) != "PLUME" for row in unep_records):
        raise ValueError("UNEP auxiliary cohort must be positive-only")
    if any(label_state(row) != "NO_PLUME" for row in cloudsen_records):
        raise ValueError("CloudSEN12 auxiliary cohort must be negative-only")
    base_scores, score_identity = load_base_score_contract(
        all_records, group_to_fold, paths["score_cache"]
    )
    spec = protocol["training"]
    strengths = [float(value) for value in protocol["search"]["strengths"]]
    batch_size = int(spec["batch_size"])
    workers = int(spec["loader_workers"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Anchored full-model fine-tune requires CUDA")
    torch.cuda.reset_peak_memory_stats()

    if args.smoke:
        seed_everything(int(spec["seed"]))
        smoke_fit_folds = set().union(*fit_folds_by_held.values())
        smoke_pool = [
            row
            for row in records
            if group_to_fold[str(row["group_id"])] in smoke_fit_folds
        ]
        mars_smoke = smoke_subset(smoke_pool, 2)
        unep_smoke = smoke_records(unep_records, 4)
        cloudsen_smoke = smoke_records(cloudsen_records, 4)
        datasets = (
            TaggedDataset(
                MarsPaperDataset(paths["metadata_root"], mars_smoke, augment=True, seed=int(spec["seed"])),
                0,
            ),
            TaggedDataset(
                MarsPaperDataset(ROOT, unep_smoke, augment=True, seed=int(spec["seed"]) + 1),
                1,
            ),
            TaggedDataset(
                MarsPaperDataset(ROOT, cloudsen_smoke, augment=True, seed=int(spec["seed"]) + 2),
                2,
            ),
        )
        weights, request_mass = combined_sampling_weights(
            mars_smoke, unep_smoke, cloudsen_smoke, protocol["sampling"]["source_mass"]
        )
        sampler = WeightedRandomSampler(
            weights,
            num_samples=batch_size * 2,
            replacement=True,
            generator=torch.Generator().manual_seed(int(spec["seed"])),
        )
        loader = make_loader(
            ConcatDataset(datasets), batch_size=batch_size, workers=0, sampler=sampler
        )
        model = AnchoredMarsFullFinetune(
            scene_topk_fraction=float(protocol["architecture"]["scene_topk_fraction"])
        ).to(device)
        model.load_released_checkpoint(released_state(paths["released_checkpoint"]))
        first = move_batch(next(iter(loader)), device)
        model.eval()
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            initial = model(first["inputs"], first["observable"], first["sensor_index"])
        identity_max = float(initial["correction_logits"].abs().max())
        if identity_max != 0.0:
            raise ValueError("Initial student is not exact released identity")
        history = train_endpoint(model, loader, {**spec, "epochs": 1}, device)
        finite = all(math.isfinite(value) for value in history[-1].values())
        result = {
            "ok": finite and model.anchor_penalty().item() > 0.0,
            "identity_pixel_max_abs": identity_max,
            "finite_optimization": finite,
            "final_anchor_penalty": model.anchor_penalty().item(),
            "request_mass": request_mass,
            "trainable_parameters": model.artifact_metadata()["trainable_parameter_count"],
            "peak_cuda_bytes": int(torch.cuda.max_memory_allocated()),
            "device": torch.cuda.get_device_name(device),
            "history": history,
        }
        print(json.dumps(result, sort_keys=True))
        return 0 if result["ok"] else 1

    endpoint_results: list[dict[str, Any]] = []
    prediction_parts: list[dict[str, Any]] = []
    endpoint_states: dict[str, Any] = {}
    for held_fold in sorted(selected_folds):
        fit_folds = fit_folds_by_held[held_fold]
        fit_records = [
            row for row in records if group_to_fold[str(row["group_id"])] in fit_folds
        ]
        held_records = [
            row for row in records if group_to_fold[str(row["group_id"])] == held_fold
        ]
        seed = int(spec["seed"]) + held_fold
        seed_everything(seed)
        datasets = (
            TaggedDataset(MarsPaperDataset(paths["metadata_root"], fit_records, augment=True, seed=seed), 0),
            TaggedDataset(MarsPaperDataset(ROOT, unep_records, augment=True, seed=seed + 100), 1),
            TaggedDataset(MarsPaperDataset(ROOT, cloudsen_records, augment=True, seed=seed + 200), 2),
        )
        weights, request_mass = combined_sampling_weights(
            fit_records, unep_records, cloudsen_records, protocol["sampling"]["source_mass"]
        )
        sampler = WeightedRandomSampler(
            weights,
            num_samples=int(spec["samples_per_epoch"]),
            replacement=True,
            generator=torch.Generator().manual_seed(seed),
        )
        train_loader = make_loader(
            ConcatDataset(datasets), batch_size=batch_size, workers=workers, sampler=sampler
        )
        evaluation_loader = make_loader(
            MarsPaperDataset(paths["metadata_root"], held_records, augment=False, seed=seed),
            batch_size=batch_size,
            workers=workers,
        )
        model = AnchoredMarsFullFinetune(
            scene_topk_fraction=float(protocol["architecture"]["scene_topk_fraction"])
        ).to(device)
        model.load_released_checkpoint(released_state(paths["released_checkpoint"]))
        print(json.dumps({"progress": "endpoint_start", "held_fold": held_fold, "seed": seed, "fit_rows": len(fit_records), "held_rows": len(held_records), "request_mass": request_mass}), flush=True)
        history = train_endpoint(model, train_loader, spec, device)
        predictions = collect_predictions(
            model, evaluation_loader, base_scores, strengths, device, held_fold
        )
        endpoint_results.append({"held_fold": held_fold, "fit_folds": sorted(fit_folds), "fit_rows": len(fit_records), "held_rows": len(held_records), "seed": seed, "request_mass": request_mass, "history": history})
        prediction_parts.append(predictions)
        endpoint_states[str(held_fold)] = model.trainable_state()
        del model, train_loader, evaluation_loader
        torch.cuda.empty_cache()

    raw = merge_predictions(prediction_parts, strengths)
    scene_cache = None
    if protocol["outputs"].get("scene_cache"):
        scene_cache = write_scene_prediction_cache(
            (ROOT / protocol["outputs"]["scene_cache"]).resolve(),
            raw,
            strengths,
            protocol_sha256=sha256(protocol_path),
        )
    endpoint_state_cache = None
    if protocol["outputs"].get("endpoint_state_cache"):
        endpoint_state_cache = write_endpoint_state_cache(
            (ROOT / protocol["outputs"]["endpoint_state_cache"]).resolve(),
            endpoint_states,
            strengths=strengths,
            protocol_sha256=sha256(protocol_path),
        )
    candidates, identity = summarize_predictions(
        raw, strengths, protocol["bootstrap"], protocol["gates"]
    )
    selected = max(candidates, key=lambda row: row["rank"])
    passed = bool(selected["passed"])
    artifact = None
    if passed:
        artifact_path = (ROOT / protocol["outputs"]["artifact"]).resolve()
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = artifact_path.with_suffix(artifact_path.suffix + ".tmp")
        torch.save({"schema_version": 1, "model": AnchoredMarsFullFinetune().artifact_metadata(), "states_by_held_fold": endpoint_states, "selected_strength": selected["strength"], "protocol_sha256": sha256(protocol_path)}, temporary)
        os.replace(temporary, artifact_path)
        artifact = {"path": protocol["outputs"]["artifact"], "bytes": artifact_path.stat().st_size, "sha256": sha256(artifact_path), "tracked": False}
    report = {
        "schema_version": 1,
        "status": "passed" if passed else "rejected",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": protocol["scope"],
        "protocol": protocol_path.relative_to(ROOT).as_posix(),
        "protocol_sha256": sha256(protocol_path),
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip(),
        "score_identity": score_identity,
        "development_identity": identity,
        "external_auxiliary": {"unep_rows": len(unep_records), "unep_groups": len({str(row["group_id"]) for row in unep_records}), "cloudsen_rows": len(cloudsen_records), "cloudsen_groups": len({str(row["group_id"]) for row in cloudsen_records})},
        "endpoints": endpoint_results,
        "candidates": candidates,
        "selected": selected,
        "all_promotion_gates_pass": passed,
        "artifact": artifact,
        "scene_cache": scene_cache,
        "endpoint_state_cache": endpoint_state_cache,
        "peak_cuda_bytes": int(torch.cuda.max_memory_allocated()),
        "device": torch.cuda.get_device_name(device),
        "decision": "Freeze a new-seed source-disjoint confirmation; external development and official test remain closed." if passed else "Reject this architecture before external development, fold 2, or official-test scoring.",
    }
    json_path = (ROOT / protocol["outputs"]["json"]).resolve()
    json_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_json = json_path.with_suffix(json_path.suffix + ".tmp")
    temporary_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary_json, json_path)
    write_markdown((ROOT / protocol["outputs"]["markdown"]).resolve(), report)
    print(json.dumps({"ok": passed, "strength": selected["strength"], "ap_delta": selected["versus_current"]["delta"]["average_precision"], "recall_delta": selected["versus_current"]["delta"]["recall_at_fpr_0_0713"], "iou_delta": selected["pixel_iou_delta"], "artifact": artifact}, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
