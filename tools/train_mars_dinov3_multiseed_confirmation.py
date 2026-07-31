#!/usr/bin/env python3
"""Confirm DINOv3 dense evidence with new seeds and Sentinel-2-only routing."""

from __future__ import annotations

import argparse
import hashlib
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
from torch.utils.data import DataLoader, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "EarthRemoteSensingRapidResponse"
for path in (MODEL_ROOT, ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from acquire_mars_metadata import sha256  # noqa: E402
from analyze_mars_mask_thresholds import component_mask_at  # noqa: E402
from mars_dinov3_methane_fusion import DinoMethaneFusionAdapter  # noqa: E402
from mars_paper_model import SENSOR_NAMES, released_state  # noqa: E402
from train_mars_dinov3_methane_fusion_pilot import (  # noqa: E402
    MASK_SCENE_GATE,
    MASK_THRESHOLDS,
    MINIMUM_CONNECTED_PIXELS,
    DinoMethaneDataset,
    load_base_score_contract,
    load_counterfactual_contract,
    make_loader,
    merge_predictions,
    summarize_predictions,
    train_endpoint,
)
from train_mars_dense_prithvi_teacher_pilot import pixel_counts  # noqa: E402
from train_mars_paper_residual import (  # noqa: E402
    iter_development_manifest,
    move_batch,
    smoke_subset,
    verify_acquisition_receipt,
)
from train_mars_physics_guided_teacher_balanced_pilot import (  # noqa: E402
    balanced_request_weights,
)
from train_mars_physics_guided_teacher_pilot import seed_everything  # noqa: E402


DEFAULT_PROTOCOL = Path("configs/mars_dinov3_multiseed_confirmation_protocol.json")


class ProgressLoader:
    """Expose batch timing without changing the wrapped loader or its order."""

    def __init__(self, loader: DataLoader[dict[str, Any]], label: str, every: int) -> None:
        self.loader = loader
        self.label = label
        self.every = every
        self.epoch = 0

    def __iter__(self):
        self.epoch += 1
        started = time.perf_counter()
        print(
            json.dumps(
                {"progress": "epoch_start", "endpoint": self.label, "epoch": self.epoch}
            ),
            flush=True,
        )
        for batch_index, batch in enumerate(self.loader, start=1):
            if batch_index % self.every == 0:
                print(
                    json.dumps(
                        {
                            "progress": "training_batch",
                            "endpoint": self.label,
                            "epoch": self.epoch,
                            "batch": batch_index,
                            "elapsed_seconds": time.perf_counter() - started,
                        }
                    ),
                    flush=True,
                )
            yield batch


def verify_protocol(
    protocol_path: Path, protocol: dict[str, Any], *, smoke: bool
) -> dict[str, Path]:
    frozen = str(protocol["status"]).startswith("frozen")
    if not frozen and not smoke:
        raise ValueError("Outcome evaluation requires a frozen confirmation protocol")
    if frozen and sha256(Path(__file__).resolve()) != protocol["trainer"]["sha256"]:
        raise ValueError("Frozen multi-seed confirmation trainer hash mismatch")
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


@torch.no_grad()
def collect_seed_evidence(
    model: DinoMethaneFusionAdapter,
    loader: DataLoader[dict[str, Any]],
    strengths: list[float],
    device: torch.device,
    fold: int,
    *,
    collect_masks: bool,
) -> dict[str, Any]:
    """Collect dense evidence; only the first seed carries the fixed mask endpoint."""

    model.eval()
    rows: dict[str, Any] = {
        "labels": [],
        "sensors": [],
        "groups": [],
        "sample_ids": [],
        "folds": [],
        "base_scores": [],
        "base_pixels": [],
        "evidence": {str(value): [] for value in strengths},
        "candidate_pixels": {str(value): [] for value in strengths},
    }
    for batch in loader:
        local_ids = [str(value) for value in batch["sample_id"]]
        local_groups = [str(value) for value in batch["group_id"]]
        batch = move_batch(batch, device)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            output = model(
                batch["inputs"],
                batch["observable"],
                batch["sensor_index"],
                batch["prithvi_tokens"],
                batch["base_scene_score"],
            )
        baseline_logits = output["baseline_logits"].float()
        pixel_delta = output["correction_logits"].float()
        baseline_surrogate = model.scene_surrogate(
            baseline_logits, batch["observable"].float()
        )
        for strength in strengths:
            candidate_logits = baseline_logits + float(strength) * pixel_delta
            candidate_surrogate = model.scene_surrogate(
                candidate_logits, batch["observable"].float()
            )
            evidence = 2.0 * torch.tanh(candidate_surrogate - baseline_surrogate)
            rows["evidence"][str(strength)].extend(
                float(value) for value in evidence.cpu().numpy()
            )

        if collect_masks:
            baseline_probability = torch.sigmoid(baseline_logits)
            for index in range(baseline_probability.shape[0]):
                sensor = int(batch["sensor_index"][index])
                threshold = MASK_THRESHOLDS[sensor]
                observable = batch["observable"][index, 0].cpu().numpy() > 0.5
                clear = batch["clear"][index, 0].cpu().numpy() > 0.5
                truth = (batch["mask"][index, 0].cpu().numpy() > 0.5) & observable
                base_score = float(batch["base_scene_score"][index])
                base_map = baseline_probability[index, 0].cpu().numpy()
                base_map[~clear] = 0.0
                base_prediction = component_mask_at(
                    base_map, threshold, MINIMUM_CONNECTED_PIXELS
                )
                if base_score < MASK_SCENE_GATE:
                    base_prediction[:] = False
                rows["base_pixels"].append(
                    pixel_counts(base_prediction, truth, observable)
                )
                for strength in strengths:
                    probability = torch.sigmoid(
                        baseline_logits[index, 0]
                        + float(strength) * pixel_delta[index, 0]
                    ).cpu().numpy()
                    probability[~clear] = 0.0
                    prediction = component_mask_at(
                        probability, threshold, MINIMUM_CONNECTED_PIXELS
                    )
                    if base_score < MASK_SCENE_GATE:
                        prediction[:] = False
                    rows["candidate_pixels"][str(strength)].append(
                        pixel_counts(prediction, truth, observable)
                    )

        rows["labels"].extend(int(value) for value in batch["presence"].cpu().numpy())
        rows["sensors"].extend(
            int(value) for value in batch["sensor_index"].cpu().numpy()
        )
        rows["groups"].extend(local_groups)
        rows["sample_ids"].extend(local_ids)
        rows["folds"].extend([fold] * len(local_ids))
        rows["base_scores"].extend(
            float(value) for value in batch["base_scene_score"].cpu().numpy()
        )
    return rows


def protected_sensor_score(
    base_scores: np.ndarray,
    evidence: np.ndarray,
    sensors: np.ndarray,
    *,
    route_sensor_index: int,
    evidence_weight: float,
) -> np.ndarray:
    """Apply float32 protected scoring only to the predeclared sensor."""

    base = torch.as_tensor(base_scores, dtype=torch.float32)
    delta = torch.as_tensor(evidence, dtype=torch.float32)
    routed = DinoMethaneFusionAdapter.protected_scene_score(
        base, delta, evidence_weight
    )
    selected = torch.as_tensor(sensors == route_sensor_index)
    result = torch.where(selected, routed, base).numpy().astype(np.float64)
    if not np.array_equal(result[~selected.numpy()], base.numpy()[~selected.numpy()]):
        raise RuntimeError("Non-routed sensor scores changed")
    return result


def aggregate_seed_parts(
    parts: list[dict[str, Any]],
    strengths: list[float],
    *,
    route_sensor_index: int,
    evidence_weight: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not parts:
        raise ValueError("At least one seed prediction is required")
    first = parts[0]
    identity_keys = (
        "labels",
        "sensors",
        "groups",
        "sample_ids",
        "folds",
        "base_scores",
    )
    for index, part in enumerate(parts[1:], start=1):
        for key in identity_keys:
            if part[key] != first[key]:
                raise ValueError(f"Seed {index} prediction order differs for {key}")
    if not first["base_pixels"]:
        raise ValueError("The first seed must supply the mask endpoint")
    sensors = np.asarray(first["sensors"], dtype=np.uint8)
    base = np.asarray(first["base_scores"], dtype=np.float64)
    raw = {
        **{key: list(first[key]) for key in identity_keys},
        "base_pixels": list(first["base_pixels"]),
        "candidate_scores": {},
        "candidate_pixels": {
            str(value): list(first["candidate_pixels"][str(value)])
            for value in strengths
        },
    }
    evidence_identity: dict[str, Any] = {}
    for strength in strengths:
        key = str(strength)
        stacked = np.stack(
            [np.asarray(part["evidence"][key], dtype=np.float64) for part in parts]
        )
        if not np.isfinite(stacked).all():
            raise ValueError("Non-finite dense evidence")
        mean_evidence = stacked.mean(axis=0)
        scores = protected_sensor_score(
            base,
            mean_evidence,
            sensors,
            route_sensor_index=route_sensor_index,
            evidence_weight=evidence_weight,
        )
        raw["candidate_scores"][key] = scores.tolist()
        evidence_identity[key] = {
            "seed_count": int(stacked.shape[0]),
            "mean": float(mean_evidence.mean()),
            "standard_deviation": float(mean_evidence.std()),
            "minimum": float(mean_evidence.min()),
            "maximum": float(mean_evidence.max()),
            "sha256": hashlib.sha256(mean_evidence.tobytes()).hexdigest(),
        }
    routing_identity = {
        "route_sensor": SENSOR_NAMES[route_sensor_index],
        "route_sensor_index": route_sensor_index,
        "routed_rows": int(np.count_nonzero(sensors == route_sensor_index)),
        "unchanged_rows": int(np.count_nonzero(sensors != route_sensor_index)),
        "landsat_scores_exact": bool(
            all(
                np.array_equal(
                    np.asarray(raw["candidate_scores"][str(value)])[sensors != route_sensor_index],
                    base[sensors != route_sensor_index].astype(np.float32).astype(np.float64),
                )
                for value in strengths
            )
        ),
        "evidence": evidence_identity,
    }
    if not routing_identity["landsat_scores_exact"]:
        raise RuntimeError("The non-routed scene scores are not exact")
    return raw, routing_identity


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    selected = report["selected"]
    scene = selected["versus_current"]["delta"]
    ap_ci = selected["paired_site_ap_delta"]
    pixel_ci = selected["paired_site_pixel_iou_delta"]
    lines = [
        "# DINOv3 multi-seed Sentinel-2 confirmation",
        "",
        f"- Promotion gates pass: {report['all_promotion_gates_pass']}",
        f"- Selected residual strength: {selected['strength']}",
        f"- AP delta versus current cross-fitted ranker: {scene['average_precision']:+.6f}",
        f"- Matched-FPR recall delta: {scene['recall_at_fpr_0_0713']:+.6f}",
        f"- Paired-site AP interval: [{ap_ci['lower']:+.6f}, {ap_ci['upper']:+.6f}]",
        f"- Dense-mask IoU delta: {selected['pixel_iou_delta']:+.6f}",
        f"- Paired-site IoU interval: [{pixel_ci['lower']:+.6f}, {pixel_ci['upper']:+.6f}]",
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
    paths = verify_protocol(protocol_path, protocol, smoke=args.smoke)
    fold_protocol = json.loads(paths["fold_protocol"].read_text(encoding="utf-8"))
    group_to_fold = {
        str(row["group_id"]): int(row["fold"])
        for row in fold_protocol["assignments"]
    }
    all_records = list(iter_development_manifest(paths["manifest"]))
    selected_folds = set(map(int, protocol["folds"]))
    records = [
        row
        for row in all_records
        if group_to_fold[str(row["group_id"])] in selected_folds
    ]
    images, row_by_id, counterfactual_identity = load_counterfactual_contract(
        paths["counterfactual_images"],
        paths["counterfactual_metadata"],
        expected_images_sha256=protocol["inputs"]["counterfactual_images"]["sha256"],
    )
    base_scores, score_identity = load_base_score_contract(
        all_records, group_to_fold, paths["score_cache"]
    )
    spec = protocol["training"]
    seed_bases = [int(value) for value in spec["seed_bases"]]
    strengths = [float(value) for value in protocol["search"]["strengths"]]
    route_sensor = str(protocol["search"]["scene_route_sensor"])
    if route_sensor not in SENSOR_NAMES:
        raise ValueError(f"Unknown scene route sensor: {route_sensor}")
    route_sensor_index = SENSOR_NAMES.index(route_sensor)
    evidence_weight = float(protocol["search"]["scene_evidence_weight"])
    batch_size = int(spec["batch_size"])
    workers = int(spec["loader_workers"])
    progress_every = int(protocol["logging"]["progress_every_batches"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("DINOv3 methane fusion requires CUDA")
    torch.cuda.reset_peak_memory_stats()

    if args.smoke:
        smoke_seed = seed_bases[0]
        seed_everything(smoke_seed)
        fit = smoke_subset(
            [row for row in records if group_to_fold[str(row["group_id"])] == 3], 2
        )
        dataset = DinoMethaneDataset(
            paths["metadata_root"],
            fit,
            images=images,
            row_by_id=row_by_id,
            base_scores=base_scores,
            augment=True,
            seed=smoke_seed,
        )
        weights, request_mass = balanced_request_weights(fit)
        sampler = WeightedRandomSampler(
            weights,
            num_samples=max(batch_size * 2, len(fit)),
            replacement=True,
            generator=torch.Generator().manual_seed(smoke_seed),
        )
        loader = make_loader(dataset, batch_size=batch_size, workers=0, sampler=sampler)
        model = DinoMethaneFusionAdapter(paths["dino_checkpoint"]).to(device)
        model.load_released_checkpoint(released_state(paths["released_checkpoint"]))
        with torch.no_grad():
            first = move_batch(next(iter(loader)), device)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                initial = model(
                    first["inputs"],
                    first["observable"],
                    first["sensor_index"],
                    first["prithvi_tokens"],
                    first["base_scene_score"],
                )
            identity_pixel = float(initial["correction_logits"].abs().max())
            identity_scene = float(initial["scene_delta_logit"].abs().max())
            identity_score = float(
                (initial["scene_score"] - first["base_scene_score"]).abs().max()
            )
        synthetic_base = np.asarray([0.49, 0.75, 0.75], dtype=np.float64)
        synthetic_sensor = np.asarray([0, 0, 1], dtype=np.uint8)
        zero_score = protected_sensor_score(
            synthetic_base,
            np.zeros(3),
            synthetic_sensor,
            route_sensor_index=route_sensor_index,
            evidence_weight=evidence_weight,
        )
        routed_score = protected_sensor_score(
            synthetic_base,
            np.ones(3),
            synthetic_sensor,
            route_sensor_index=route_sensor_index,
            evidence_weight=evidence_weight,
        )
        routing_identity = bool(
            np.array_equal(zero_score, synthetic_base.astype(np.float32).astype(np.float64))
            and routed_score[0] == synthetic_base.astype(np.float32)[0]
            and routed_score[2] == synthetic_base.astype(np.float32)[2]
            and routed_score[1] > synthetic_base[1]
        )
        if (
            identity_pixel != 0.0
            or identity_scene != 0.0
            or identity_score != 0.0
            or not routing_identity
        ):
            raise ValueError("Multi-seed confirmation initialization is not exact identity")
        started = time.perf_counter()
        endpoint_spec = dict(spec)
        endpoint_spec["seed"] = smoke_seed
        history = train_endpoint(
            model,
            ProgressLoader(loader, "smoke", progress_every),
            endpoint_spec,
            device,
            1,
        )
        elapsed = time.perf_counter() - started
        finite = all(
            torch.isfinite(value).all() for value in model.trainable_state().values()
        )
        print(
            json.dumps(
                {
                    "ok": bool(finite),
                    "identity_pixel_max_abs": identity_pixel,
                    "identity_scene_max_abs": identity_scene,
                    "identity_score_max_abs": identity_score,
                    "routing_identity": routing_identity,
                    "elapsed_seconds": elapsed,
                    "peak_cuda_bytes": int(torch.cuda.max_memory_allocated()),
                    "trainable_parameters": model.trainable_parameter_count(),
                    "request_mass": request_mass,
                    "history": history,
                },
                indent=2,
            )
        )
        return 0 if finite else 1

    raw_parts: list[dict[str, Any]] = []
    endpoint_states: dict[str, dict[str, torch.Tensor]] = {}
    endpoints: list[dict[str, Any]] = []
    routing_identities: dict[str, Any] = {}
    trainable_parameters: int | None = None
    for held_fold in sorted(selected_folds):
        fit_records = [
            row for row in records if group_to_fold[str(row["group_id"])] != held_fold
        ]
        held_records = [
            row for row in records if group_to_fold[str(row["group_id"])] == held_fold
        ]
        weights, request_mass = balanced_request_weights(fit_records)
        seed_parts: list[dict[str, Any]] = []
        for seed_index, seed_base in enumerate(seed_bases):
            endpoint_seed = seed_base + held_fold
            seed_everything(endpoint_seed)
            endpoint_spec = dict(spec)
            endpoint_spec["seed"] = endpoint_seed
            train_dataset = DinoMethaneDataset(
                paths["metadata_root"],
                fit_records,
                images=images,
                row_by_id=row_by_id,
                base_scores=base_scores,
                augment=True,
                seed=endpoint_seed,
            )
            held_dataset = DinoMethaneDataset(
                paths["metadata_root"],
                held_records,
                images=images,
                row_by_id=row_by_id,
                base_scores=base_scores,
                augment=False,
                seed=endpoint_seed,
            )
            sampler = WeightedRandomSampler(
                weights,
                num_samples=int(spec["samples_per_epoch"]),
                replacement=True,
                generator=torch.Generator().manual_seed(endpoint_seed),
            )
            train_loader = make_loader(
                train_dataset, batch_size=batch_size, workers=workers, sampler=sampler
            )
            held_loader = make_loader(
                held_dataset, batch_size=batch_size, workers=workers
            )
            model = DinoMethaneFusionAdapter(paths["dino_checkpoint"]).to(device)
            model.load_released_checkpoint(released_state(paths["released_checkpoint"]))
            local_parameter_count = model.trainable_parameter_count()
            if trainable_parameters is None:
                trainable_parameters = local_parameter_count
            elif trainable_parameters != local_parameter_count:
                raise RuntimeError("Cross-fit endpoints have different parameter counts")
            with torch.no_grad():
                first = move_batch(next(iter(held_loader)), device)
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    initial = model(
                        first["inputs"],
                        first["observable"],
                        first["sensor_index"],
                        first["prithvi_tokens"],
                        first["base_scene_score"],
                    )
                identity = {
                    "pixel_max_abs": float(initial["correction_logits"].abs().max()),
                    "scene_delta_max_abs": float(initial["scene_delta_logit"].abs().max()),
                    "scene_score_max_abs": float(
                        (initial["scene_score"] - first["base_scene_score"]).abs().max()
                    ),
                }
            if any(value != 0.0 for value in identity.values()):
                raise ValueError(
                    f"Endpoint fold={held_fold} seed={endpoint_seed} is not identity"
                )
            endpoint_label = f"held_fold={held_fold},seed={endpoint_seed}"
            history = train_endpoint(
                model,
                ProgressLoader(train_loader, endpoint_label, progress_every),
                endpoint_spec,
                device,
                int(spec["epochs"]),
            )
            seed_parts.append(
                collect_seed_evidence(
                    model,
                    held_loader,
                    strengths,
                    device,
                    held_fold,
                    collect_masks=seed_index == 0,
                )
            )
            state_key = f"seed_{seed_base}_held_fold_{held_fold}"
            endpoint_states[state_key] = model.trainable_state()
            endpoints.append(
                {
                    "seed_base": seed_base,
                    "endpoint_seed": endpoint_seed,
                    "held_fold": held_fold,
                    "fit_fold": next(iter(selected_folds - {held_fold})),
                    "fit_rows": len(fit_records),
                    "held_rows": len(held_records),
                    "request_mass": request_mass,
                    "identity": identity,
                    "history": history,
                    "supplies_mask_endpoint": seed_index == 0,
                }
            )
            del model, train_loader, held_loader, train_dataset, held_dataset
            torch.cuda.empty_cache()
        aggregated, routing_identity = aggregate_seed_parts(
            seed_parts,
            strengths,
            route_sensor_index=route_sensor_index,
            evidence_weight=evidence_weight,
        )
        raw_parts.append(aggregated)
        routing_identities[str(held_fold)] = routing_identity

    candidates, evaluation_identity = summarize_predictions(
        merge_predictions(raw_parts, strengths),
        strengths,
        protocol["bootstrap"],
        protocol["gates"],
    )
    selected = max(candidates, key=lambda row: tuple(row["rank"]))
    passed = bool(selected["passed"])
    artifact = None
    if passed:
        artifact_path = (ROOT / protocol["outputs"]["artifact"]).resolve()
        if artifact_path.exists():
            raise FileExistsError(f"Refusing to overwrite artifact: {artifact_path}")
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "schema_version": 1,
                "endpoint_states": endpoint_states,
                "selected_strength": selected["strength"],
                "seed_bases": seed_bases,
                "scene_route_sensor": route_sensor,
                "scene_evidence_weight": evidence_weight,
                "protocol_sha256": sha256(protocol_path),
                "counterfactual_identity": counterfactual_identity,
                "score_identity": score_identity,
            },
            artifact_path,
        )
        artifact = {
            "path": artifact_path.relative_to(ROOT).as_posix(),
            "bytes": artifact_path.stat().st_size,
            "sha256": sha256(artifact_path),
        }
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    report = {
        "schema_version": 1,
        "scope": "three-seed DINOv3 dense-evidence confirmation with Sentinel-2-only scene routing",
        "all_promotion_gates_pass": passed,
        "decision": (
            "Authorize a separately frozen source-disjoint confirmation without official-test access."
            if passed
            else "Reject this fixed confirmation without fold-2, external, or official-test scoring."
        ),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "provenance": {
            "git_commit": commit,
            "protocol_sha256": sha256(protocol_path),
            "trainer_sha256": sha256(Path(__file__).resolve()),
            "device": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "peak_cuda_bytes": int(torch.cuda.max_memory_allocated()),
        },
        "folds": sorted(selected_folds),
        "seed_bases": seed_bases,
        "endpoints": endpoints,
        "routing_identity": routing_identities,
        "counterfactual_identity": counterfactual_identity,
        "score_identity": score_identity,
        "trainable_parameters": trainable_parameters,
        "evaluation_identity": evaluation_identity,
        "candidates": candidates,
        "selected": selected,
        "artifact": artifact,
        "fold_2_accessed": False,
        "external_inputs_accessed": False,
        "official_test_inputs_accessed": False,
    }
    output_json = (ROOT / protocol["outputs"]["json"]).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_json.with_suffix(output_json.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, output_json)
    write_markdown((ROOT / protocol["outputs"]["markdown"]).resolve(), report)
    print(
        json.dumps(
            {
                "ok": passed,
                "strength": selected["strength"],
                "ap_delta": selected["versus_current"]["delta"]["average_precision"],
                "ap_lower": selected["paired_site_ap_delta"]["lower"],
                "recall_delta": selected["versus_current"]["delta"][
                    "recall_at_fpr_0_0713"
                ],
                "iou_delta": selected["pixel_iou_delta"],
                "iou_lower": selected["paired_site_pixel_iou_delta"]["lower"],
            },
            indent=2,
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
