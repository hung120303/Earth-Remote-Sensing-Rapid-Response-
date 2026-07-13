#!/usr/bin/env python3
"""Run the single frozen MethaneS2CM location-test comparison campaign."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import h5py
import numpy as np
import sklearn
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "EarthRemoteSensingRapidResponse"
for directory in (MODEL_ROOT, ROOT / "tools"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from mars_v4_model import MarsV4Model  # noqa: E402
from methanes2cm_adapter import coordinate_group_id  # noqa: E402
from methanes2cm_v5_model import MethaneS2CMV5Model  # noqa: E402

from acquire_mars_metadata import repo_root, sha256  # noqa: E402
from acquire_methanes2cm_v5_test import (  # noqa: E402
    DEFAULT_PACKED_TEST,
    DEFAULT_REPORT as DEFAULT_ACQUISITION,
    EXPECTED_TEST_CSV_SHA256,
)
from aggregate_methanes2cm_v5_1 import (  # noqa: E402
    binary_metrics,
    empirical_percentile,
)
from build_methanes2cm_v5_protocol import location_components, read_rows  # noqa: E402
from evaluate_mars_v4_3_ensemble import calibrated_ensemble  # noqa: E402
from evaluate_released_marss2l import (  # noqa: E402
    MINIMUM_CONNECTED_PIXELS,
    PIXEL_THRESHOLD as RELEASED_PIXEL_THRESHOLD,
    RELEASE_SPECS,
    component_mask,
    connected_scene_score,
    load_released_model,
)
from train_mars_v3 import safe_output, tracked_dirty, write_json  # noqa: E402
from train_methanes2cm_v5 import PackedMethaneS2CMDataset, move_batch  # noqa: E402

DATA_DIR = Path(
    "EarthRemoteSensingRapidResponse/Data Collection/s2_emit_pairs/publication-v1/"
    "external/MethaneS2CM/l2a_location_split_32x32"
)
TEST_CSV = DATA_DIR / "test.csv"
V5_ENSEMBLE_REPORT = Path(
    "reports/experiments/methanes2cm_v5_1_ensemble_validation.json"
)
V4_ENSEMBLE_REPORT = Path("reports/experiments/mars_v4_3_ensemble_validation.json")
RELEASED_MARS_REPORT = Path(
    "reports/experiments/mars_released_model_full_strict_baseline.json"
)
MARS_STRICT_COMPARISON = Path("reports/experiments/mars_v4_3_strict_comparison.json")
DEFAULT_JSON = Path("reports/experiments/methanes2cm_v5_1_location_test.json")
DEFAULT_MARKDOWN = Path("reports/experiments/METHANES2CM_V5_1_LOCATION_TEST.md")
DEFAULT_CACHE = DATA_DIR / "v5_1_location_test_predictions.npz"

EXPECTED_V5_ENSEMBLE_SHA256 = (
    "03691437f3ce2c384aece9f00c7dc4462eebe5e0b580ca340a7db043dc0cdeca"
)
EXPECTED_V4_ENSEMBLE_SHA256 = (
    "2499aff4b949e1499b0378141d4bf12541de3b6814b5b35b79d685308bbf105b"
)
EXPECTED_RELEASED_MARS_REPORT_SHA256 = (
    "433301db9591a10d6f702fb31b9fc34992f2704c1d074a8e897b71fe6c83c653"
)
BOOTSTRAP_SEED = 20_260_713
BOOTSTRAP_REPLICATES = 2_000
MODEL_ORDER = ("ersrr_v5_1", "ersrr_v4_3", "released_mars_s2l")


def load_json_identity(path: Path, expected_sha256: str) -> dict[str, Any]:
    if sha256(path) != expected_sha256:
        raise ValueError(f"Frozen report identity mismatch: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def cache_is_ignored(root: Path, path: Path) -> bool:
    relative = path.resolve().relative_to(root.resolve())
    return (
        subprocess.run(
            ["git", "check-ignore", "--quiet", "--", relative.as_posix()],
            cwd=root,
            check=False,
        ).returncode
        == 0
    )


def build_v4_batch(v5_inputs: torch.Tensor) -> torch.Tensor:
    if v5_inputs.ndim != 4 or v5_inputs.shape[1] != 20:
        raise ValueError("V4 compatibility conversion requires a Bx20xHxW v5 tensor")
    batch, _, height, width = v5_inputs.shape
    wind = torch.full(
        (batch, 2, height, width),
        0.5,
        dtype=v5_inputs.dtype,
        device=v5_inputs.device,
    )
    cloud = torch.zeros(
        (batch, 1, height, width),
        dtype=v5_inputs.dtype,
        device=v5_inputs.device,
    )
    values = torch.cat([v5_inputs[:, 0:1], v5_inputs[:, 2:14], wind, cloud], dim=1)
    if values.shape[1] != 16:
        raise ValueError("Constructed v4 compatibility tensor has the wrong channel count")
    return values


def read_test_records(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = read_rows(path)
    rows.sort(key=lambda row: int(row["id"]))
    by_location, geographic = location_components(rows)
    records: list[dict[str, Any]] = []
    for row in rows:
        value: dict[str, Any] = dict(row)
        value["exact_location_id"] = coordinate_group_id(row)
        value["group_id"] = by_location[value["exact_location_id"]]
        records.append(value)
    return records, geographic


def verify_acquisition(
    root: Path, acquisition_path: Path, packed_path: Path
) -> tuple[dict[str, Any], str]:
    acquisition = json.loads(acquisition_path.read_text(encoding="utf-8"))
    if acquisition["scope"] != "methanes2cm_v5_once_authorized_location_test_acquisition":
        raise ValueError("Unexpected location-test acquisition scope")
    evaluator = acquisition["unlock_authorization"]["evaluator"]
    if evaluator["sha256"] != sha256(Path(__file__).resolve()):
        raise ValueError("Acquisition was not unlocked by this frozen evaluator")
    extraction = acquisition["extraction"]
    if (root / extraction["packed_path"]).resolve() != packed_path:
        raise ValueError("Acquisition and evaluator packed-test paths differ")
    packed_identity = sha256(packed_path)
    if packed_identity != extraction["packed_sha256"]:
        raise ValueError("Packed location-test identity mismatch")
    if not acquisition["seal_transition"]["images_opened_by_this_acquisition"]:
        raise ValueError("Acquisition did not record the authorized seal transition")
    current_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    if acquisition["provenance"]["git_commit"] != current_commit:
        raise ValueError("Acquisition and one-shot evaluation must use the same code commit")
    with h5py.File(packed_path, "r") as source:
        if str(source.attrs.get("source_split")) != (
            "l2a_location_split_32x32/test.csv"
        ):
            raise ValueError("Packed location-test source partition mismatch")
        if source["sample_id"].shape != (20_789,):
            raise ValueError("Packed location-test count mismatch")
    return acquisition, packed_identity


def load_v5_models(
    root: Path,
    ensemble: dict[str, Any],
    device: torch.device,
) -> tuple[list[MethaneS2CMV5Model], dict[str, np.ndarray]]:
    models: list[MethaneS2CMV5Model] = []
    if [int(item["seed"]) for item in ensemble["seeds"]] != [1101, 2202, 3303]:
        raise ValueError("Frozen v5.1 seed ordering mismatch")
    for item in ensemble["seeds"]:
        report_path = root / item["report_path"]
        if sha256(report_path) != item["report_sha256"]:
            raise ValueError("V5.1 seed-report identity mismatch")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        checkpoint = root / item["checkpoint"]["path"]
        if sha256(checkpoint) != item["checkpoint"]["sha256"]:
            raise ValueError("V5.1 checkpoint identity mismatch")
        model = MethaneS2CMV5Model(
            scene_topk_fraction=float(report["model"]["scene_topk_fraction"]),
            scene_max_weight=float(report["model"]["scene_max_weight"]),
            context_scene_weight=float(report["model"]["context_scene_weight"]),
        ).to(device)
        payload = torch.load(checkpoint, map_location=device, weights_only=True)
        if payload["model_metadata"] != report["model"]:
            raise ValueError("V5.1 checkpoint metadata mismatch")
        model.load_state_dict(payload["state_dict"], strict=True)
        model.eval()
        models.append(model)
    cache_path = root / ensemble["calibration_cache"]["path"]
    if sha256(cache_path) != ensemble["calibration_cache"]["sha256"]:
        raise ValueError("V5.1 calibration-cache identity mismatch")
    with np.load(cache_path, allow_pickle=False) as source:
        calibration = {name: source[name].copy() for name in source.files}
    if calibration["seeds"].tolist() != [1101, 2202, 3303]:
        raise ValueError("V5.1 calibration-cache seed ordering mismatch")
    return models, calibration


def load_v4_models(
    root: Path,
    ensemble: dict[str, Any],
    device: torch.device,
) -> tuple[list[MarsV4Model], np.ndarray]:
    models: list[MarsV4Model] = []
    if [int(item["seed"]) for item in ensemble["source_reports"]] != [606, 707, 808]:
        raise ValueError("Frozen v4.3 seed ordering mismatch")
    for item in ensemble["source_reports"]:
        report_path = root / item["path"]
        if sha256(report_path) != item["sha256"]:
            raise ValueError("V4.3 seed-report identity mismatch")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        checkpoint = root / item["checkpoint"]["path"]
        if sha256(checkpoint) != item["checkpoint"]["sha256"]:
            raise ValueError("V4.3 checkpoint identity mismatch")
        model = MarsV4Model(
            scene_topk_fraction=float(report["model"]["scene_topk_fraction"]),
            scene_max_weight=float(report["model"]["scene_max_weight"]),
        ).to(device)
        payload = torch.load(checkpoint, map_location=device, weights_only=True)
        if payload["model_metadata"] != report["model"]:
            raise ValueError("V4.3 checkpoint metadata mismatch")
        model.load_state_dict(payload["state_dict"], strict=True)
        model.eval()
        models.append(model)
    cache_path = root / ensemble["ignored_calibration_cache"]["path"]
    if sha256(cache_path) != ensemble["ignored_calibration_cache"]["sha256"]:
        raise ValueError("V4.3 calibration-cache identity mismatch")
    with np.load(cache_path, allow_pickle=False) as source:
        validation_scores = source["seed_scores"].astype(np.float64)
    if validation_scores.shape[1] != 3:
        raise ValueError("V4.3 calibration scores have the wrong shape")
    return models, validation_scores


@torch.inference_mode()
def collect_v5(
    models: Sequence[MethaneS2CMV5Model],
    loader: DataLoader[dict[str, Any]],
    device: torch.device,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    identities: dict[str, list[np.ndarray] | list[str]] = {
        "sample_id": [],
        "label": [],
        "group_id": [],
        "observable": [],
        "truth": [],
    }
    raw_scores: list[np.ndarray] = []
    probability_maps: list[np.ndarray] = []
    completed = 0
    for batch in loader:
        moved = move_batch(batch, device)
        probability_sum: torch.Tensor | None = None
        local_scores: list[np.ndarray] = []
        for model in models:
            with torch.amp.autocast("cuda", dtype=torch.float16):
                output = model(moved["inputs"], moved["observable"])
            probability = torch.sigmoid(output["segmentation_logits"]).float()
            probability_sum = probability if probability_sum is None else probability_sum + probability
            local_scores.append(torch.sigmoid(output["scene_logit"]).float().cpu().numpy())
        assert probability_sum is not None
        probability_maps.append(
            (probability_sum / len(models))[:, 0].cpu().numpy().astype(np.float32)
        )
        raw_scores.append(np.column_stack(local_scores).astype(np.float32))
        identities["sample_id"].append(batch["sample_id"].numpy().astype(np.int64))
        identities["label"].append(batch["presence"].numpy().astype(np.uint8))
        identities["group_id"].extend(str(value) for value in batch["group_id"])
        identities["observable"].append(
            (batch["observable"][:, 0].numpy() > 0.5).astype(np.uint8)
        )
        identities["truth"].append((batch["mask"][:, 0].numpy() > 0.5).astype(np.uint8))
        completed += len(batch["sample_id"])
        if completed // 2_000 != (completed - len(batch["sample_id"])) // 2_000:
            print(f"ERSRR v5.1: {completed:,}/{len(loader.dataset):,}", flush=True)
    aligned = {
        "sample_id": np.concatenate(identities["sample_id"]),
        "label": np.concatenate(identities["label"]),
        "group_id": np.asarray(identities["group_id"]),
        "observable": np.concatenate(identities["observable"]).astype(bool),
        "truth": np.concatenate(identities["truth"]).astype(bool),
    }
    return aligned, np.concatenate(raw_scores), np.concatenate(probability_maps)


@torch.inference_mode()
def collect_v4(
    models: Sequence[MarsV4Model],
    loader: DataLoader[dict[str, Any]],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    raw_scores: list[np.ndarray] = []
    probability_maps: list[np.ndarray] = []
    completed = 0
    for batch in loader:
        moved = move_batch(batch, device)
        inputs = build_v4_batch(moved["inputs"])
        probability_sum: torch.Tensor | None = None
        local_scores: list[np.ndarray] = []
        for model in models:
            with torch.amp.autocast("cuda", dtype=torch.float16):
                output = model(inputs, moved["observable"])
            probability = torch.sigmoid(output["segmentation_logits"]).float()
            probability_sum = probability if probability_sum is None else probability_sum + probability
            local_scores.append(torch.sigmoid(output["scene_logit"]).float().cpu().numpy())
        assert probability_sum is not None
        probability_maps.append(
            (probability_sum / len(models))[:, 0].cpu().numpy().astype(np.float32)
        )
        raw_scores.append(np.column_stack(local_scores).astype(np.float32))
        completed += len(batch["sample_id"])
        if completed // 2_000 != (completed - len(batch["sample_id"])) // 2_000:
            print(f"ERSRR v4.3: {completed:,}/{len(loader.dataset):,}", flush=True)
    return np.concatenate(raw_scores), np.concatenate(probability_maps)


@torch.inference_mode()
def collect_released_mars(
    model: torch.nn.Module,
    loader: DataLoader[dict[str, Any]],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scene_scores: list[float] = []
    scene_decisions: list[bool] = []
    probability_maps: list[np.ndarray] = []
    completed = 0
    for batch in loader:
        moved = move_batch(batch, device)
        inputs = build_v4_batch(moved["inputs"])
        with torch.amp.autocast("cuda", dtype=torch.float16):
            probability = torch.sigmoid(model(inputs)).float().cpu().numpy()
        observable = batch["observable"][:, 0].numpy() > 0.5
        probability[~observable] = 0.0
        probability_maps.append(probability.astype(np.float32))
        for score in probability:
            decision = component_mask(score)
            scene_decisions.append(bool(np.any(decision)))
            scene_scores.append(connected_scene_score(score))
        completed += len(batch["sample_id"])
        if completed // 2_000 != (completed - len(batch["sample_id"])) // 2_000:
            print(
                f"Released MARS-S2L: {completed:,}/{len(loader.dataset):,}",
                flush=True,
            )
    return (
        np.asarray(scene_scores, dtype=np.float32),
        np.asarray(scene_decisions, dtype=bool),
        np.concatenate(probability_maps),
    )


def model_metrics(
    labels: np.ndarray,
    scene_scores: np.ndarray,
    scene_decisions: np.ndarray,
    pixel_probability: np.ndarray,
    pixel_decision: np.ndarray,
    truth: np.ndarray,
    observable: np.ndarray,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    local_truth = truth & observable
    local_prediction = pixel_decision & observable
    intersection = (local_prediction & local_truth).reshape(len(labels), -1).sum(axis=1)
    predicted = local_prediction.reshape(len(labels), -1).sum(axis=1)
    truth_area = local_truth.reshape(len(labels), -1).sum(axis=1)
    total_intersection = int(intersection.sum())
    total_predicted = int(predicted.sum())
    total_truth = int(truth_area.sum())
    union = total_predicted + total_truth - total_intersection
    scene = binary_metrics(labels, scene_decisions)
    result = {
        "scene": {
            "average_precision": float(average_precision_score(labels, scene_scores)),
            "auroc": float(roc_auc_score(labels, scene_scores)),
            **scene,
        },
        "pixel": {
            "average_precision": float(
                average_precision_score(local_truth[observable], pixel_probability[observable])
            ),
            "dice": 2.0 * total_intersection / max(total_predicted + total_truth, 1),
            "intersection_over_union": total_intersection / max(union, 1),
            "intersection_pixels": total_intersection,
            "predicted_positive_pixels": total_predicted,
            "truth_positive_pixels": total_truth,
            "observable_pixels": int(np.count_nonzero(observable)),
        },
    }
    per_scene = {
        "pixel_intersection": intersection.astype(np.int32),
        "pixel_predicted": predicted.astype(np.int32),
        "pixel_truth": truth_area.astype(np.int32),
    }
    return result, per_scene


def metric_vector(
    labels: np.ndarray,
    scores: np.ndarray,
    decisions: np.ndarray,
    per_scene: dict[str, np.ndarray],
    indices: np.ndarray,
) -> dict[str, float]:
    truth = labels[indices]
    binary = binary_metrics(truth, decisions[indices])
    intersection = int(per_scene["pixel_intersection"][indices].sum())
    predicted = int(per_scene["pixel_predicted"][indices].sum())
    truth_area = int(per_scene["pixel_truth"][indices].sum())
    union = predicted + truth_area - intersection
    return {
        "average_precision": float(average_precision_score(truth, scores[indices])),
        "auroc": float(roc_auc_score(truth, scores[indices])),
        "recall": float(binary["recall"]),
        "false_positive_rate": float(binary["false_positive_rate"]),
        "precision": float(binary["precision"]),
        "pixel_dice": 2.0 * intersection / max(predicted + truth_area, 1),
        "pixel_intersection_over_union": intersection / max(union, 1),
    }


def distribution_summary(values: list[float], point: float) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "point_estimate": float(point),
        "bootstrap_mean": float(np.mean(array)),
        "bootstrap_standard_deviation": float(np.std(array, ddof=1)),
        "95ci": [float(value) for value in np.quantile(array, (0.025, 0.975))],
    }


def paired_group_bootstrap(
    labels: np.ndarray,
    groups: np.ndarray,
    scores: dict[str, np.ndarray],
    decisions: dict[str, np.ndarray],
    per_scene: dict[str, dict[str, np.ndarray]],
    point_metrics: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    unique = np.asarray(sorted(set(groups.astype(str))))
    by_group = {group: np.flatnonzero(groups.astype(str) == group) for group in unique}
    metric_names = (
        "average_precision",
        "auroc",
        "recall",
        "false_positive_rate",
        "precision",
        "pixel_dice",
        "pixel_intersection_over_union",
    )
    samples = {
        model: {metric: [] for metric in metric_names} for model in MODEL_ORDER
    }
    comparisons = {
        "ersrr_v5_1_minus_ersrr_v4_3": ("ersrr_v5_1", "ersrr_v4_3"),
        "ersrr_v5_1_minus_released_mars_s2l": (
            "ersrr_v5_1",
            "released_mars_s2l",
        ),
    }
    deltas = {
        name: {metric: [] for metric in metric_names} for name in comparisons
    }
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    for _ in range(BOOTSTRAP_REPLICATES):
        selected = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([by_group[group] for group in selected])
        local = {
            model: metric_vector(
                labels, scores[model], decisions[model], per_scene[model], indices
            )
            for model in MODEL_ORDER
        }
        for model in MODEL_ORDER:
            for metric in metric_names:
                samples[model][metric].append(local[model][metric])
        for name, (candidate, baseline) in comparisons.items():
            for metric in metric_names:
                deltas[name][metric].append(
                    local[candidate][metric] - local[baseline][metric]
                )

    point = {
        model: {
            "average_precision": point_metrics[model]["scene"]["average_precision"],
            "auroc": point_metrics[model]["scene"]["auroc"],
            "recall": point_metrics[model]["scene"]["recall"],
            "false_positive_rate": point_metrics[model]["scene"][
                "false_positive_rate"
            ],
            "precision": point_metrics[model]["scene"]["precision"],
            "pixel_dice": point_metrics[model]["pixel"]["dice"],
            "pixel_intersection_over_union": point_metrics[model]["pixel"][
                "intersection_over_union"
            ],
        }
        for model in MODEL_ORDER
    }
    result: dict[str, Any] = {
        "method": "paired nonparametric resampling of all 100 frozen 25 km test components",
        "seed": BOOTSTRAP_SEED,
        "replicates": BOOTSTRAP_REPLICATES,
        "models": {},
        "deltas": {},
    }
    for model in MODEL_ORDER:
        result["models"][model] = {
            metric: distribution_summary(samples[model][metric], point[model][metric])
            for metric in metric_names
        }
    for name, (candidate, baseline) in comparisons.items():
        result["deltas"][name] = {
            metric: distribution_summary(
                deltas[name][metric], point[candidate][metric] - point[baseline][metric]
            )
            for metric in metric_names
        }
    return result


def comparison_checks(
    candidate: dict[str, Any], baseline: dict[str, Any], delta_ci: dict[str, Any]
) -> dict[str, Any]:
    point = {
        "scene_average_precision_higher": candidate["scene"]["average_precision"]
        > baseline["scene"]["average_precision"],
        "scene_auroc_higher": candidate["scene"]["auroc"]
        > baseline["scene"]["auroc"],
        "scene_recall_higher": candidate["scene"]["recall"]
        > baseline["scene"]["recall"],
        "scene_false_positive_rate_lower": candidate["scene"]["false_positive_rate"]
        < baseline["scene"]["false_positive_rate"],
        "pixel_average_precision_higher": candidate["pixel"]["average_precision"]
        > baseline["pixel"]["average_precision"],
        "pixel_dice_higher": candidate["pixel"]["dice"] > baseline["pixel"]["dice"],
        "pixel_iou_higher": candidate["pixel"]["intersection_over_union"]
        > baseline["pixel"]["intersection_over_union"],
    }
    supported = {
        "scene_average_precision": delta_ci["average_precision"]["95ci"][0] > 0,
        "scene_auroc": delta_ci["auroc"]["95ci"][0] > 0,
        "scene_recall": delta_ci["recall"]["95ci"][0] > 0,
        "scene_false_positive_rate": delta_ci["false_positive_rate"]["95ci"][1] < 0,
        "pixel_dice": delta_ci["pixel_dice"]["95ci"][0] > 0,
        "pixel_intersection_over_union": delta_ci[
            "pixel_intersection_over_union"
        ]["95ci"][0]
        > 0,
    }
    return {
        "point_checks": point,
        "all_point_checks_pass": all(point.values()),
        "group_bootstrap_supported_checks": supported,
        "all_bootstrap_checks_pass": all(supported.values()),
    }


def write_prediction_cache(
    root: Path,
    path: Path,
    *,
    identity: dict[str, np.ndarray],
    scores: dict[str, np.ndarray],
    decisions: dict[str, np.ndarray],
    probabilities: dict[str, np.ndarray],
    packed_sha256: str,
    acquisition_sha256: str,
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as destination:
        np.savez_compressed(
            destination,
            schema_version=np.asarray([1], dtype=np.int16),
            sample_id=identity["sample_id"],
            label=identity["label"],
            group_id=identity["group_id"].astype(str),
            observable=identity["observable"].astype(np.uint8),
            truth=identity["truth"].astype(np.uint8),
            ersrr_v5_1_scene_score=scores["ersrr_v5_1"].astype(np.float32),
            ersrr_v4_3_scene_score=scores["ersrr_v4_3"].astype(np.float32),
            released_mars_s2l_scene_score=scores["released_mars_s2l"].astype(np.float32),
            ersrr_v5_1_scene_decision=decisions["ersrr_v5_1"].astype(np.uint8),
            ersrr_v4_3_scene_decision=decisions["ersrr_v4_3"].astype(np.uint8),
            released_mars_s2l_scene_decision=decisions[
                "released_mars_s2l"
            ].astype(np.uint8),
            ersrr_v5_1_probability=probabilities["ersrr_v5_1"].astype(np.float16),
            ersrr_v4_3_probability=probabilities["ersrr_v4_3"].astype(np.float16),
            released_mars_s2l_probability=probabilities[
                "released_mars_s2l"
            ].astype(np.float16),
            packed_sha256=np.asarray([packed_sha256]),
            acquisition_sha256=np.asarray([acquisition_sha256]),
        )
    os.replace(temporary, path)
    if not cache_is_ignored(root, path):
        raise ValueError("Location-test prediction cache must be ignored by Git")
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "tracked": False,
    }


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    models = report["models"]
    lines = [
        "# MethaneS2CM v5.1 one-shot location test",
        "",
        "The sealed location test was opened once after architecture, checkpoints, calibration, thresholds, acquisition, and evaluator code were frozen.",
        "",
        "| Frozen model/rule | Scene AP | AUROC | Recall | FPR | Precision | Pixel AP | Dice | IoU |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    labels = {
        "ersrr_v5_1": "ERSRR v5.1 (three-seed)",
        "ersrr_v4_3": "ERSRR v4.3 zero-shot",
        "released_mars_s2l": "Released MARS-S2L zero-shot",
    }
    for name in MODEL_ORDER:
        scene = models[name]["metrics"]["scene"]
        pixel = models[name]["metrics"]["pixel"]
        lines.append(
            f"| {labels[name]} | {scene['average_precision']:.4f} | {scene['auroc']:.4f} | "
            f"{scene['recall']:.4f} | {scene['false_positive_rate']:.4f} | "
            f"{scene['precision']:.4f} | {pixel['average_precision']:.4f} | "
            f"{pixel['dice']:.4f} | {pixel['intersection_over_union']:.4f} |"
        )
    comparison = report["comparison"]["v5_1_vs_released_mars_s2l"]
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            report["decision"],
            "",
            f"- All same-cohort point checks versus released MARS-S2L pass: {str(comparison['all_point_checks_pass']).lower()}",
            f"- All predeclared group-bootstrap checks pass: {str(comparison['all_bootstrap_checks_pass']).lower()}",
            "- Precision is benchmark precision on a roughly balanced crop set, not deployment positive predictive value.",
            "- V5.1 was trained on MethaneS2CM L2A; v4.3 and released MARS-S2L are L1C-trained zero-shot comparators, so this is not an architecture-only causal comparison.",
            "- The test result is frozen evidence. No retuning from it is permitted.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packed-test", default=DEFAULT_PACKED_TEST.as_posix())
    parser.add_argument("--acquisition", default=DEFAULT_ACQUISITION.as_posix())
    parser.add_argument("--output-json", default=DEFAULT_JSON.as_posix())
    parser.add_argument("--output-markdown", default=DEFAULT_MARKDOWN.as_posix())
    parser.add_argument("--prediction-cache", default=DEFAULT_CACHE.as_posix())
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    root = repo_root()
    if tracked_dirty(root):
        raise RuntimeError("Refusing one-shot evaluation from a dirty tracked worktree")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the one-shot comparison")
    device = torch.device("cuda")
    prediction_cache = safe_output(root, args.prediction_cache)
    if not cache_is_ignored(root, prediction_cache):
        raise ValueError("Location-test prediction cache must be ignored by Git")

    v5_ensemble = load_json_identity(
        root / V5_ENSEMBLE_REPORT, EXPECTED_V5_ENSEMBLE_SHA256
    )
    v4_ensemble = load_json_identity(
        root / V4_ENSEMBLE_REPORT, EXPECTED_V4_ENSEMBLE_SHA256
    )
    released_report = load_json_identity(
        root / RELEASED_MARS_REPORT, EXPECTED_RELEASED_MARS_REPORT_SHA256
    )
    packed_path = (root / args.packed_test).resolve()
    acquisition_path = (root / args.acquisition).resolve()
    acquisition, packed_identity = verify_acquisition(
        root, acquisition_path, packed_path
    )
    acquisition_identity = sha256(acquisition_path)
    test_csv = root / TEST_CSV
    if sha256(test_csv) != EXPECTED_TEST_CSV_SHA256:
        raise ValueError("MethaneS2CM test CSV identity mismatch")
    records, geographic = read_test_records(test_csv)
    if len(records) != 20_789 or geographic["connected_groups"] != 100:
        raise ValueError("Frozen test metadata/group count mismatch")
    dataset = PackedMethaneS2CMDataset(
        packed_path, records, augment=False, seed=BOOTSTRAP_SEED
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )

    v5_models, v5_calibration = load_v5_models(root, v5_ensemble, device)
    identity, v5_raw_scores, v5_probability = collect_v5(v5_models, loader, device)
    del v5_models
    torch.cuda.empty_cache()
    v5_scores = np.mean(
        np.stack(
            [
                empirical_percentile(
                    v5_calibration["seed_sorted_development_scores"][index],
                    v5_raw_scores[:, index],
                )
                for index in range(3)
            ]
        ),
        axis=0,
    )
    target_index = int(
        np.flatnonzero(np.isclose(v5_calibration["target_fprs"], 0.05))[0]
    )
    v5_scene_threshold = float(v5_calibration["scene_thresholds"][target_index])
    v5_pixel_threshold = float(v5_calibration["pixel_threshold"][0])

    v4_models, v4_validation_scores = load_v4_models(root, v4_ensemble, device)
    v4_raw_scores, v4_probability = collect_v4(v4_models, loader, device)
    del v4_models
    torch.cuda.empty_cache()
    v4_scores = calibrated_ensemble(v4_validation_scores, v4_raw_scores)
    v4_scene_threshold = float(
        v4_ensemble["final_development_rule"]["operating_points"]["0.05"][
            "threshold"
        ]
    )
    v4_pixel_threshold = float(v4_ensemble["segmentation"]["selected"]["threshold"])

    spec = RELEASE_SPECS["mars-s2l"]
    model_directory = root / Path(spec["directory"])
    released_checkpoint = model_directory / "best_epoch"
    released_config = model_directory / "config_experiment.json"
    if (
        released_checkpoint.stat().st_size != int(spec["checkpoint_bytes"])
        or sha256(released_checkpoint) != spec["checkpoint_sha256"]
        or sha256(released_config) != spec["config_sha256"]
    ):
        raise ValueError("Released MARS-S2L artifact identity mismatch")
    with released_config.open("r", encoding="utf-8") as source:
        config = json.load(source)
    if any(config.get(key) != value for key, value in spec["expected_config"].items()):
        raise ValueError("Released MARS-S2L configuration mismatch")
    released_model = load_released_model(
        released_checkpoint, device, int(spec["input_channels"])
    )
    released_scores, released_decisions, released_probability = collect_released_mars(
        released_model, loader, device
    )
    del released_model
    torch.cuda.empty_cache()

    truth = identity["truth"] & identity["observable"]
    probabilities = {
        "ersrr_v5_1": v5_probability.astype(np.float32),
        "ersrr_v4_3": v4_probability.astype(np.float32),
        "released_mars_s2l": released_probability.astype(np.float32),
    }
    scores = {
        "ersrr_v5_1": v5_scores,
        "ersrr_v4_3": v4_scores,
        "released_mars_s2l": released_scores,
    }
    decisions = {
        "ersrr_v5_1": v5_scores >= v5_scene_threshold,
        "ersrr_v4_3": v4_scores >= v4_scene_threshold,
        "released_mars_s2l": released_decisions,
    }
    pixel_decisions = {
        "ersrr_v5_1": probabilities["ersrr_v5_1"] >= v5_pixel_threshold,
        "ersrr_v4_3": probabilities["ersrr_v4_3"] >= v4_pixel_threshold,
        "released_mars_s2l": np.stack(
            [component_mask(score) for score in probabilities["released_mars_s2l"]]
        ),
    }
    metrics: dict[str, dict[str, Any]] = {}
    per_scene: dict[str, dict[str, np.ndarray]] = {}
    for name in MODEL_ORDER:
        metrics[name], per_scene[name] = model_metrics(
            identity["label"],
            scores[name],
            decisions[name],
            probabilities[name],
            pixel_decisions[name],
            truth,
            identity["observable"],
        )
    bootstrap = paired_group_bootstrap(
        identity["label"],
        identity["group_id"],
        scores,
        decisions,
        per_scene,
        metrics,
    )
    comparison_mars = comparison_checks(
        metrics["ersrr_v5_1"],
        metrics["released_mars_s2l"],
        bootstrap["deltas"]["ersrr_v5_1_minus_released_mars_s2l"],
    )
    comparison_v4 = comparison_checks(
        metrics["ersrr_v5_1"],
        metrics["ersrr_v4_3"],
        bootstrap["deltas"]["ersrr_v5_1_minus_ersrr_v4_3"],
    )

    cache_artifact = write_prediction_cache(
        root,
        prediction_cache,
        identity=identity,
        scores=scores,
        decisions=decisions,
        probabilities=probabilities,
        packed_sha256=packed_identity,
        acquisition_sha256=acquisition_identity,
    )
    strict_context = json.loads(
        (root / MARS_STRICT_COMPARISON).read_text(encoding="utf-8")
    )
    all_supported = comparison_mars["all_bootstrap_checks_pass"]
    decision = (
        "ERSRR v5.1 outperforms released MARS-S2L on every predeclared scene and overlap "
        "criterion on this same MethaneS2CM location test, with all group-bootstrap direction "
        "checks supported. This is strong benchmark evidence, but not an architecture-only "
        "claim because v5.1 is in-domain L2A while MARS-S2L is a zero-shot L1C comparator."
        if all_supported and comparison_mars["all_point_checks_pass"]
        else "ERSRR v5.1 does not establish across-the-board superiority over released MARS-S2L "
        "on the frozen MethaneS2CM location test. Preserve the partial result and do not retune "
        "from this test."
    )
    report = {
        "schema_version": 1,
        "scope": "methanes2cm_v5_1_once_only_location_test_comparison",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "cohort": {
            "scenes": int(len(identity["label"])),
            "positives": int(np.count_nonzero(identity["label"] == 1)),
            "negatives": int(np.count_nonzero(identity["label"] == 0)),
            "exact_locations": int(geographic["exact_locations"]),
            "geographic_25km_groups": int(geographic["connected_groups"]),
            "largest_group_scenes": int(geographic["largest_group_samples"]),
            "product": "Sentinel-2 L2A surface reflectance",
            "balanced_crop_benchmark": True,
        },
        "seal": {
            "opened_once_after_code_and_rules_committed": True,
            "architecture_or_threshold_retuning_permitted": False,
            "acquisition": {
                "path": acquisition_path.relative_to(root).as_posix(),
                "sha256": acquisition_identity,
            },
            "packed_test": {
                "path": packed_path.relative_to(root).as_posix(),
                "bytes": packed_path.stat().st_size,
                "sha256": packed_identity,
                "tracked": False,
            },
        },
        "models": {
            "ersrr_v5_1": {
                "rule": {
                    "scene": "mean of three frozen empirical-CDF percentiles",
                    "scene_threshold": v5_scene_threshold,
                    "scene_target_development_fpr": 0.05,
                    "pixel": "equal mean probability",
                    "pixel_threshold": v5_pixel_threshold,
                },
                "source_report": {
                    "path": V5_ENSEMBLE_REPORT.as_posix(),
                    "sha256": EXPECTED_V5_ENSEMBLE_SHA256,
                },
                "metrics": metrics["ersrr_v5_1"],
            },
            "ersrr_v4_3": {
                "rule": {
                    "scene": "mean of three frozen empirical-CDF percentiles",
                    "scene_threshold": v4_scene_threshold,
                    "scene_target_development_fpr": 0.05,
                    "pixel": "equal mean probability",
                    "pixel_threshold": v4_pixel_threshold,
                    "wind_mps_imputed": [4.0, 4.0],
                    "cloud_mask_unavailable_zero": True,
                },
                "source_report": {
                    "path": V4_ENSEMBLE_REPORT.as_posix(),
                    "sha256": EXPECTED_V4_ENSEMBLE_SHA256,
                },
                "metrics": metrics["ersrr_v4_3"],
            },
            "released_mars_s2l": {
                "rule": {
                    "pixel_threshold": RELEASED_PIXEL_THRESHOLD,
                    "pixel_comparison": ">",
                    "minimum_8_connected_pixels": MINIMUM_CONNECTED_PIXELS,
                    "wind_mps_imputed": [4.0, 4.0],
                    "cloud_mask_unavailable_zero": True,
                    "selected_on": "released author configuration; no MethaneS2CM tuning",
                },
                "source_report": {
                    "path": RELEASED_MARS_REPORT.as_posix(),
                    "sha256": EXPECTED_RELEASED_MARS_REPORT_SHA256,
                },
                "artifact": released_report["artifact"],
                "metrics": metrics["released_mars_s2l"],
            },
        },
        "group_bootstrap": bootstrap,
        "comparison": {
            "v5_1_vs_released_mars_s2l": comparison_mars,
            "v5_1_vs_v4_3": comparison_v4,
        },
        "prior_same_mars_strict_cohort_context": {
            "path": MARS_STRICT_COMPARISON.as_posix(),
            "sha256": sha256(root / MARS_STRICT_COMPARISON),
            "v4_3": strict_context["strict_spatial_test"],
            "released_mars_s2l": strict_context["same_cohort_comparison"][
                "released_mars_s2l"
            ],
            "official_paper_targets_not_same_cohort": strict_context[
                "official_mars_s2l_paper_targets_not_same_cohort"
            ],
        },
        "prediction_cache": cache_artifact,
        "decision": decision,
        "interpretation_limits": [
            "MethaneS2CM is approximately balanced by crop construction; precision is not operational PPV.",
            "V5.1 is trained on MethaneS2CM L2A while v4.3 and MARS-S2L are L1C-trained zero-shot comparators.",
            "MethaneS2CM omits acquisition timestamps, wind, and per-pixel cloud masks.",
            "The largest spatial component contains 5,487 crops; group bootstrap, not crop independence, defines uncertainty.",
            "No model or operating rule may be changed in response to this test outcome.",
        ],
        "provenance": {
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True
            ).strip(),
            "script": Path(__file__).resolve().relative_to(root).as_posix(),
            "script_sha256": sha256(Path(__file__).resolve()),
            "tracked_worktree_dirty_at_start": False,
            "runtime": {
                "torch": torch.__version__,
                "cuda_runtime": torch.version.cuda,
                "numpy": np.__version__,
                "sklearn": sklearn.__version__,
                "device": torch.cuda.get_device_name(0),
            },
        },
    }
    output_json = safe_output(root, args.output_json)
    output_markdown = safe_output(root, args.output_markdown)
    write_json(output_json, report)
    write_markdown(output_markdown, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
