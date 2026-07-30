#!/usr/bin/env python3
"""Reproduce and preserve the dense Prithvi mask branch with scene scores fixed."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
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
from analyze_mars_mask_routing import paired_group_bootstrap  # noqa: E402
from analyze_mars_mask_thresholds import component_mask_at  # noqa: E402
from mars_dense_prithvi_teacher import DensePrithviTeacherAdapter  # noqa: E402
from mars_paper_model import SENSOR_NAMES, released_state  # noqa: E402
from train_mars_dense_prithvi_teacher_pilot import (  # noqa: E402
    DensePrithviDataset,
    MASK_THRESHOLDS,
    MINIMUM_CONNECTED_PIXELS,
    SCENE_GATE,
    load_feature_contract,
    pixel_counts,
    pixel_summary,
    train_endpoint,
)
from train_mars_paper_residual import (  # noqa: E402
    iter_development_manifest,
    move_batch,
    verify_acquisition_receipt,
)
from train_mars_physics_guided_teacher_balanced_pilot import (  # noqa: E402
    balanced_request_weights,
)
from train_mars_physics_guided_teacher_pilot import seed_everything  # noqa: E402


DEFAULT_PROTOCOL = Path("configs/mars_dense_prithvi_mask_preservation_protocol.json")


def summarize_domains(
    baseline: np.ndarray,
    candidate: np.ndarray,
    sensors: np.ndarray,
    groups: np.ndarray,
    bootstrap: dict[str, Any],
) -> dict[str, Any]:
    baseline_metrics = pixel_summary(baseline)
    candidate_metrics = pixel_summary(candidate)
    sensor_values: dict[str, Any] = {}
    for index, name in enumerate(SENSOR_NAMES):
        rows = sensors == index
        local_base = pixel_summary(baseline[rows])
        local_candidate = pixel_summary(candidate[rows])
        sensor_values[name] = {
            "baseline": local_base,
            "candidate": local_candidate,
            "iou_delta": (
                local_candidate["intersection_over_union"]
                - local_base["intersection_over_union"]
            ),
        }
    interval = paired_group_bootstrap(
        baseline,
        candidate,
        groups,
        replicates=int(bootstrap["replicates"]),
        seed=int(bootstrap["seed"]),
        confidence=float(bootstrap["confidence"]),
    )
    return {
        "baseline": baseline_metrics,
        "candidate": candidate_metrics,
        "iou_delta": (
            candidate_metrics["intersection_over_union"]
            - baseline_metrics["intersection_over_union"]
        ),
        "retained_true_positive_pixel_fraction": (
            candidate_metrics["tp"] / max(int(baseline_metrics["tp"]), 1)
        ),
        "sensors": sensor_values,
        "paired_site_iou_delta": interval,
    }


@torch.no_grad()
def evaluate_mask_only(
    model: DensePrithviTeacherAdapter,
    loader: DataLoader[dict[str, Any]],
    *,
    strength: float,
    device: torch.device,
    bootstrap: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    model.eval()
    baseline_counts: list[np.ndarray] = []
    candidate_counts: list[np.ndarray] = []
    sensors: list[int] = []
    groups: list[str] = []
    sample_ids: list[str] = []
    scene_identity_max_abs = 0.0
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
        # The mask-only branch is not permitted to alter scene ranking or gating.
        scene_identity_max_abs = max(
            scene_identity_max_abs,
            float(
                (
                    torch.sigmoid(output["base_scene_logit"])
                    - batch["base_scene_score"].float().clamp(1e-6, 1 - 1e-6)
                )
                .abs()
                .max()
            ),
        )
        baseline_probability = torch.sigmoid(output["baseline_logits"]).float()
        candidate_probability = torch.sigmoid(
            output["baseline_logits"].float()
            + float(strength) * output["correction_logits"].float()
        )
        for index in range(baseline_probability.shape[0]):
            sensor = int(batch["sensor_index"][index])
            threshold = MASK_THRESHOLDS[sensor]
            observable = batch["observable"][index, 0].cpu().numpy() > 0.5
            clear = batch["clear"][index, 0].cpu().numpy() > 0.5
            truth = (batch["mask"][index, 0].cpu().numpy() > 0.5) & observable
            base_map = baseline_probability[index, 0].cpu().numpy()
            candidate_map = candidate_probability[index, 0].cpu().numpy()
            base_map[~clear] = 0.0
            candidate_map[~clear] = 0.0
            base_prediction = component_mask_at(
                base_map, threshold, MINIMUM_CONNECTED_PIXELS
            )
            candidate_prediction = component_mask_at(
                candidate_map, threshold, MINIMUM_CONNECTED_PIXELS
            )
            if float(batch["base_scene_score"][index]) < SCENE_GATE:
                base_prediction[:] = False
                candidate_prediction[:] = False
            baseline_counts.append(pixel_counts(base_prediction, truth, observable))
            candidate_counts.append(
                pixel_counts(candidate_prediction, truth, observable)
            )
        sensors.extend(int(value) for value in batch["sensor_index"].cpu().numpy())
        groups.extend(local_groups)
        sample_ids.extend(local_ids)
    baseline = np.asarray(baseline_counts, dtype=np.int64)
    candidate = np.asarray(candidate_counts, dtype=np.int64)
    sensor_array = np.asarray(sensors, dtype=np.uint8)
    group_array = np.asarray(groups)
    summary = summarize_domains(
        baseline, candidate, sensor_array, group_array, bootstrap
    )
    identity = {
        "rows": len(sample_ids),
        "sample_id_sha256": __import__("hashlib")
        .sha256("\n".join(sample_ids).encode())
        .hexdigest(),
        "scene_score_identity_max_abs": scene_identity_max_abs,
    }
    return summary, identity


def histories_match(
    observed: list[dict[str, float]],
    reference: list[dict[str, float]],
    tolerance: float,
) -> tuple[bool, float]:
    if len(observed) != len(reference):
        return False, float("inf")
    maximum = 0.0
    for left, right in zip(observed, reference):
        if set(left) != set(right):
            return False, float("inf")
        for key in left:
            maximum = max(maximum, abs(float(left[key]) - float(right[key])))
    return maximum <= tolerance, maximum


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    result = report["mask_result"]
    interval = result["paired_site_iou_delta"]
    lines = [
        "# Dense Prithvi mask preservation",
        "",
        "Development fold 2 only. The current cross-fitted scene score and 0.75 "
        "scene gate are unchanged.",
        "",
        f"- Mask residual strength: {report['mask_strength']:.2f}",
        f"- IoU delta: {result['iou_delta']:+.6f}",
        f"- Paired-site IoU interval: [{interval['lower']:+.6f}, {interval['upper']:+.6f}]",
        f"- True-positive pixel ratio: {result['retained_true_positive_pixel_fraction']:.6f}",
        f"- Training-history reproduction max delta: {report['history_reproduction']['maximum_absolute_delta']:.3g}",
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
    if sha256(Path(__file__).resolve()) != protocol["trainer"]["sha256"]:
        raise ValueError("Frozen mask-preservation trainer hash mismatch")
    for dependency in protocol["code_dependencies"]:
        path = (ROOT / dependency["path"]).resolve()
        if sha256(path) != dependency["sha256"]:
            raise ValueError(f"Frozen dependency mismatch: {dependency['path']}")
    paths: dict[str, Path] = {}
    for name, contract in protocol["inputs"].items():
        path = (ROOT / contract["path"]).resolve()
        if path.is_file() and sha256(path) != contract["sha256"]:
            raise ValueError(f"Frozen input mismatch: {name}")
        paths[name] = path
    if not paths["metadata_root"].is_dir():
        raise ValueError("MARS metadata root is unavailable")
    verify_acquisition_receipt(paths["acquisition_receipt"], sha256(paths["manifest"]))
    features, row_by_id, base_scores, feature_identity = load_feature_contract(
        paths["features"],
        paths["feature_metadata"],
        paths["score_cache"],
        str(protocol["inputs"]["features"]["sha256"]),
    )
    prior = json.loads(paths["prior_result"].read_text(encoding="utf-8"))
    reference_history = prior["history"]
    fold_protocol = json.loads(paths["fold_protocol"].read_text(encoding="utf-8"))
    group_to_fold = {
        str(row["group_id"]): int(row["fold"]) for row in fold_protocol["assignments"]
    }
    records = list(iter_development_manifest(paths["manifest"]))
    fit_folds = set(map(int, protocol["folds"]["fit"]))
    held_fold = int(protocol["folds"]["held"])
    fit_records = [
        row for row in records if group_to_fold[str(row["group_id"])] in fit_folds
    ]
    held_records = [
        row for row in records if group_to_fold[str(row["group_id"])] == held_fold
    ]
    spec = protocol["training"]
    if args.smoke:
        selected: list[dict[str, Any]] = []
        counts: Counter[tuple[str, str]] = Counter()
        for row in fit_records:
            key = (str(row["label_state"]), str(row["sensor_family"]))
            if counts[key] < 4:
                selected.append(row)
                counts[key] += 1
            if len(counts) == 4 and min(counts.values()) >= 4:
                break
        fit_records = selected
        held_records = selected
    weights, request_mass = balanced_request_weights(fit_records)
    train_dataset = DensePrithviDataset(
        paths["metadata_root"],
        fit_records,
        features=features,
        row_by_id=row_by_id,
        base_scores=base_scores,
        augment=True,
        seed=int(spec["seed"]),
    )
    evaluation_dataset = DensePrithviDataset(
        paths["metadata_root"],
        held_records,
        features=features,
        row_by_id=row_by_id,
        base_scores=base_scores,
        augment=False,
        seed=int(spec["seed"]),
    )
    samples = 32 if args.smoke else int(spec["samples_per_epoch"])
    workers = 0 if args.smoke else int(spec["loader_workers"])
    sampler = WeightedRandomSampler(
        weights,
        num_samples=samples,
        replacement=True,
        generator=torch.Generator().manual_seed(int(spec["seed"])),
    )
    options = {
        "batch_size": int(spec["batch_size"]),
        "num_workers": workers,
        "pin_memory": True,
        "persistent_workers": workers > 0,
    }
    train_loader = DataLoader(train_dataset, sampler=sampler, **options)
    evaluation_loader = DataLoader(
        evaluation_dataset, shuffle=False, **options
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Dense Prithvi mask preservation requires CUDA")
    # The original joint pilot seeded immediately before optimization, after
    # random module construction.  This replication closes that provenance gap
    # by binding initialization as well as sampling/optimization to the seed.
    seed_everything(int(spec["seed"]))
    model = DensePrithviTeacherAdapter().to(device)
    model.load_released_checkpoint(released_state(paths["released_checkpoint"]))
    with torch.no_grad():
        first = move_batch(next(iter(evaluation_loader)), device)
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
    if identity_pixel != 0.0 or identity_scene != 0.0:
        raise ValueError("Mask-preservation initialization is not exact identity")
    history = train_endpoint(
        model,
        train_loader,
        spec,
        device,
        1 if args.smoke else int(spec["epochs"]),
    )
    if args.smoke:
        finite = all(
            torch.isfinite(value).all() for value in model.trainable_state().values()
        )
        print(
            json.dumps(
                {
                    "ok": finite,
                    "identity_pixel_max_abs": identity_pixel,
                    "identity_scene_max_abs": identity_scene,
                    "request_mass": request_mass,
                    "feature_identity": feature_identity,
                    "trainable_parameters": model.trainable_parameter_count(),
                    "history": history,
                }
            )
        )
        return 0 if finite else 1

    history_ok, history_delta = histories_match(
        history,
        reference_history,
        float(protocol["reproduction"]["history_absolute_tolerance"]),
    )
    mask_result, evaluation_identity = evaluate_mask_only(
        model,
        evaluation_loader,
        strength=float(protocol["mask"]["strength"]),
        device=device,
        bootstrap=protocol["bootstrap"],
    )
    sensor_deltas = [
        value["iou_delta"] for value in mask_result["sensors"].values()
    ]
    passed = bool(
        mask_result["iou_delta"] > 0
        and mask_result["paired_site_iou_delta"]["lower"] > 0
        and min(sensor_deltas) >= 0
        and mask_result["retained_true_positive_pixel_fraction"]
        >= float(protocol["gates"]["minimum_true_positive_pixel_ratio"])
        and evaluation_identity["scene_score_identity_max_abs"] <= 1e-6
    )
    artifact = None
    if passed:
        artifact_path = (ROOT / protocol["outputs"]["artifact"]).resolve()
        if artifact_path.exists():
            raise FileExistsError(f"Refusing to overwrite existing artifact: {artifact_path}")
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "schema_version": 1,
                "kind": "mars_dense_prithvi_mask_adapter",
                "adapter_state": model.trainable_state(),
                "mask_strength": float(protocol["mask"]["strength"]),
                "scene_strength": 0.0,
                "protocol_sha256": sha256(protocol_path),
                "feature_identity": feature_identity,
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
        "scope": "development-only reproduction and preservation of the dense Prithvi mask branch",
        "all_preservation_gates_pass": passed,
        "decision": (
            "Preserve the mask adapter for independent full-development cross-fitting."
            if passed
            else "Do not preserve the mask adapter; reproduction or domain gates failed."
        ),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "provenance": {
            "git_commit": commit,
            "protocol_sha256": sha256(protocol_path),
            "trainer_sha256": sha256(Path(__file__).resolve()),
            "device": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
        "fit_folds": sorted(fit_folds),
        "held_fold": held_fold,
        "fit_rows": len(fit_records),
        "held_rows": len(held_records),
        "feature_identity": feature_identity,
        "request_mass": request_mass,
        "identity_pixel_max_abs": identity_pixel,
        "identity_scene_max_abs": identity_scene,
        "mask_strength": float(protocol["mask"]["strength"]),
        "scene_strength": 0.0,
        "history": history,
        "history_reproduction": {
            "passed": history_ok,
            "role": "descriptive only; the prior pilot did not seed random module initialization, while this replication does",
            "maximum_absolute_delta": history_delta,
            "tolerance": float(
                protocol["reproduction"]["history_absolute_tolerance"]
            ),
        },
        "evaluation_identity": evaluation_identity,
        "mask_result": mask_result,
        "artifact": artifact,
        "external_inputs_accessed": False,
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
                "history_delta": history_delta,
                "iou_delta": mask_result["iou_delta"],
                "iou_lower": mask_result["paired_site_iou_delta"]["lower"],
                "sensor_deltas": {
                    name: value["iou_delta"]
                    for name, value in mask_result["sensors"].items()
                },
                "tp_ratio": mask_result["retained_true_positive_pixel_fraction"],
                "artifact": artifact,
            },
            indent=2,
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
