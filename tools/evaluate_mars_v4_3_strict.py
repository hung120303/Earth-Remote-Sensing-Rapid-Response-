#!/usr/bin/env python3
"""Run the one authorized frozen v4.3 ensemble comparison on strict MARS data."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import rasterio
import sklearn
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "EarthRemoteSensingRapidResponse"
for directory in (MODEL_ROOT, ROOT / "tools"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from mars_s2l_adapter import iter_manifest  # noqa: E402
from mars_v4_model import MarsV4Model  # noqa: E402

from acquire_mars_metadata import DEFAULT_OUTPUT, checked_output_dir, repo_root, sha256  # noqa: E402
from aggregate_mars_v3_strict import PAPER_TARGETS, summary  # noqa: E402
from build_mars_v3_strict_cohort import (  # noqa: E402
    DEFAULT_JSON as STRICT_COHORT_JSON,
    V3_STRICT_SAMPLES,
)
from build_mars_v3_training_cohort import V3_SAMPLES  # noqa: E402
from evaluate_mars_v4_3_ensemble import (  # noqa: E402
    FIXED_SEEDS,
    binary_metrics,
    calibrated_ensemble,
)
from run_mars_dev_scene_baselines import bootstrap_ci, metrics  # noqa: E402
from train_mars_v3 import DEFAULT_METADATA_CSV, safe_output, tracked_dirty, write_json  # noqa: E402
from train_mars_v4 import (  # noqa: E402
    DEFAULT_LUT,
    MarsV4Dataset,
    metadata_and_plume_library,
    move_batch,
)

DEFAULT_EXPERIMENT = Path("reports/experiments/mars_v4_3_ensemble_validation.json")
DEFAULT_BASELINE = Path("reports/experiments/mars_released_model_full_strict_baseline.json")
DEFAULT_JSON = Path("reports/experiments/mars_v4_3_strict_comparison.json")
DEFAULT_MARKDOWN = Path("reports/experiments/MARS_V4_3_STRICT_COMPARISON.md")
DEFAULT_CACHE = DEFAULT_OUTPUT / "publication_v4_3_strict_scene_predictions.npz"
BOOTSTRAP_REPLICATES = 2_000
BOOTSTRAP_SEED = 20_260_713
PIXELS_PER_SCENE = 40_000
IMAGE_SIZE = 200


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


def load_validation_cache(
    root: Path, experiment: dict[str, Any]
) -> tuple[Path, dict[str, np.ndarray]]:
    artifact = experiment["ignored_calibration_cache"]
    path = (root / artifact["path"]).resolve()
    if path.stat().st_size != int(artifact["bytes"]) or sha256(path) != artifact["sha256"]:
        raise ValueError("V4.3 validation calibration cache identity mismatch")
    with np.load(path, allow_pickle=False) as source:
        cache = {name: source[name].copy() for name in source.files}
    if int(cache["schema_version"][0]) != 1:
        raise ValueError("Unsupported v4.3 validation cache schema")
    if tuple(int(value) for value in cache["seeds"]) != FIXED_SEEDS:
        raise ValueError("V4.3 validation cache seed ordering mismatch")
    expected_hashes = [item["sha256"] for item in experiment["source_reports"]]
    if list(cache["source_report_sha256"].astype(str)) != expected_hashes:
        raise ValueError("V4.3 validation cache source-report mismatch")
    scores = np.asarray(cache["seed_scores"], dtype=np.float64)
    if scores.shape != (int(experiment["cohort"]["scenes"]), len(FIXED_SEEDS)):
        raise ValueError("V4.3 validation cache score shape mismatch")
    return path, cache


def load_released_cache(
    root: Path, baseline: dict[str, Any]
) -> tuple[Path, dict[str, np.ndarray]]:
    artifact = baseline["scene_prediction_cache"]
    path = (root / artifact["path"]).resolve()
    if path.stat().st_size != int(artifact["bytes"]) or sha256(path) != artifact["sha256"]:
        raise ValueError("Released MARS scene cache identity mismatch")
    with np.load(path, allow_pickle=False) as source:
        cache = {name: source[name].copy() for name in source.files}
    required = ("sample_ids", "groups", "labels", "scores", "predictions")
    arrays = [np.asarray(cache[name]) for name in required]
    if any(array.ndim != 1 for array in arrays) or len({array.shape for array in arrays}) != 1:
        raise ValueError("Released MARS scene cache arrays are not aligned")
    return path, cache


def align_released_cache(
    cache: dict[str, np.ndarray], sample_ids: Sequence[str]
) -> dict[str, np.ndarray]:
    source_ids = cache["sample_ids"].astype(str)
    destination = np.asarray(sample_ids).astype(str)
    if len(set(source_ids)) != source_ids.size or set(source_ids) != set(destination):
        raise ValueError("Released MARS and v4.3 strict sample IDs differ")
    lookup = {value: index for index, value in enumerate(source_ids)}
    order = np.asarray([lookup[value] for value in destination], dtype=np.int64)
    return {name: np.asarray(values)[order] for name, values in cache.items() if np.asarray(values).ndim == 1 and np.asarray(values).shape == source_ids.shape}


def unpack(packed: np.ndarray) -> np.ndarray:
    return np.unpackbits(packed, count=PIXELS_PER_SCENE).reshape(IMAGE_SIZE, IMAGE_SIZE).astype(bool)


def strict_pixel_metrics(
    probabilities: np.ndarray,
    observable_packed: np.ndarray,
    truth_packed: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    if probabilities.shape[1:] != (IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError("Unexpected strict segmentation raster shape")
    pixel_truth: list[np.ndarray] = []
    pixel_scores: list[np.ndarray] = []
    intersection = predicted_area = truth_area = 0
    for score, obs_bits, truth_bits in zip(
        probabilities, observable_packed, truth_packed
    ):
        observable = unpack(obs_bits)
        truth = unpack(truth_bits) & observable
        prediction = (score.astype(np.float32) >= threshold) & observable
        intersection += int(np.count_nonzero(prediction & truth))
        predicted_area += int(np.count_nonzero(prediction))
        truth_area += int(np.count_nonzero(truth))
        pixel_truth.append(truth[observable].astype(np.uint8))
        pixel_scores.append(score.ravel()[observable.ravel()])
    truth_values = np.concatenate(pixel_truth)
    score_values = np.concatenate(pixel_scores).astype(np.float32)
    union = predicted_area + truth_area - intersection
    return {
        "average_precision": float(average_precision_score(truth_values, score_values)),
        "intersection_over_union": 0.0 if union == 0 else intersection / union,
        "dice": 0.0
        if predicted_area + truth_area == 0
        else 2.0 * intersection / (predicted_area + truth_area),
        "intersection_pixels": intersection,
        "truth_positive_pixels": truth_area,
        "predicted_positive_pixels": predicted_area,
        "observable_pixels": int(truth_values.size),
        "pixel_threshold": threshold,
        "minimum_connected_pixels": 1,
    }


@torch.no_grad()
def collect_predictions(
    models: Sequence[MarsV4Model],
    loader: DataLoader[dict[str, Any]],
    device: torch.device,
) -> dict[str, Any]:
    for model in models:
        model.eval()
    sample_ids: list[str] = []
    groups: list[str] = []
    labels: list[np.ndarray] = []
    scene_scores: list[np.ndarray] = []
    segmentation: list[np.ndarray] = []
    observable: list[np.ndarray] = []
    truth: list[np.ndarray] = []
    completed = 0
    for batch in loader:
        moved = move_batch(batch, device)
        probability_sum: torch.Tensor | None = None
        local_scene_scores = []
        for model in models:
            with torch.amp.autocast("cuda", dtype=torch.float16):
                output = model(moved["inputs"], moved["observable"])
            probability = torch.sigmoid(output["segmentation_logits"]).float()
            probability_sum = probability if probability_sum is None else probability_sum + probability
            local_scene_scores.append(torch.sigmoid(output["scene_logit"]).float().cpu().numpy())
        assert probability_sum is not None
        ensemble_probability = (
            probability_sum / len(models) * moved["observable"]
        ).cpu().numpy()[:, 0]
        segmentation.append(ensemble_probability.astype(np.float16))
        scene_scores.append(np.column_stack(local_scene_scores))
        labels.append(batch["presence"].numpy().astype(np.uint8))
        sample_ids.extend(str(value) for value in batch["sample_id"])
        groups.extend(str(value) for value in batch["group_id"])
        observable.extend(
            np.packbits(item[0].numpy().astype(bool).ravel()) for item in batch["observable"]
        )
        truth.extend(
            np.packbits(item[0].numpy().astype(bool).ravel()) for item in batch["mask"]
        )
        completed += len(batch["sample_id"])
        if completed // 500 != (completed - len(batch["sample_id"])) // 500:
            print(f"Evaluated v4.3 strict ensemble for {completed} scenes", flush=True)
    return {
        "sample_ids": np.asarray(sample_ids),
        "groups": np.asarray(groups),
        "labels": np.concatenate(labels),
        "seed_scores": np.concatenate(scene_scores, axis=0),
        "segmentation": np.concatenate(segmentation, axis=0),
        "observable": np.stack(observable),
        "truth": np.stack(truth),
    }


def paired_group_bootstrap(
    labels: np.ndarray,
    groups: np.ndarray,
    candidate_scores: np.ndarray,
    candidate_predictions: np.ndarray,
    baseline_scores: np.ndarray,
    baseline_predictions: np.ndarray,
) -> dict[str, Any]:
    truth = np.asarray(labels, dtype=np.uint8)
    group_values = np.asarray(groups).astype(str)
    unique = np.unique(group_values)
    by_group = {group: np.flatnonzero(group_values == group) for group in unique}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    values: dict[str, list[float]] = {
        "recall_delta": [],
        "false_positive_rate_delta": [],
        "average_precision_delta": [],
        "auroc_delta": [],
    }
    for _ in range(BOOTSTRAP_REPLICATES):
        sampled = rng.choice(unique, size=unique.size, replace=True)
        indices = np.concatenate([by_group[group] for group in sampled])
        local_truth = truth[indices]
        if np.unique(local_truth).size != 2:
            continue
        candidate_binary = binary_metrics(local_truth, candidate_predictions[indices])
        baseline_binary = binary_metrics(local_truth, baseline_predictions[indices])
        values["recall_delta"].append(
            float(candidate_binary["recall"] - baseline_binary["recall"])
        )
        values["false_positive_rate_delta"].append(
            float(
                candidate_binary["false_positive_rate"]
                - baseline_binary["false_positive_rate"]
            )
        )
        values["average_precision_delta"].append(
            float(
                average_precision_score(local_truth, candidate_scores[indices])
                - average_precision_score(local_truth, baseline_scores[indices])
            )
        )
        values["auroc_delta"].append(
            float(
                roc_auc_score(local_truth, candidate_scores[indices])
                - roc_auc_score(local_truth, baseline_scores[indices])
            )
        )
    if any(len(rows) != BOOTSTRAP_REPLICATES for rows in values.values()):
        raise ValueError("A paired strict bootstrap replicate lacked both scene classes")
    return {
        "method": "paired nonparametric bootstrap of the 150 frozen 25 km groups",
        "replicates": BOOTSTRAP_REPLICATES,
        "random_seed": BOOTSTRAP_SEED,
        **{name: summary(rows) for name, rows in values.items()},
    }


def write_prediction_cache(
    root: Path,
    path: Path,
    *,
    predictions: dict[str, Any],
    ensemble_scores: np.ndarray,
    threshold: float,
    strict_manifest_sha256: str,
    experiment_sha256: str,
    validation_cache_sha256: str,
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fixed = (ensemble_scores >= threshold).astype(np.uint8)
    with temporary.open("wb") as destination:
        np.savez_compressed(
            destination,
            schema_version=np.asarray([1], dtype=np.int16),
            sample_ids=predictions["sample_ids"].astype(str),
            groups=predictions["groups"].astype(str),
            labels=predictions["labels"].astype(np.uint8),
            seeds=np.asarray(FIXED_SEEDS, dtype=np.int32),
            seed_scores=predictions["seed_scores"].astype(np.float32),
            ensemble_scores=np.asarray(ensemble_scores, dtype=np.float32),
            predictions=fixed,
            threshold=np.asarray([threshold], dtype=np.float64),
            strict_manifest_sha256=np.asarray([strict_manifest_sha256]),
            experiment_sha256=np.asarray([experiment_sha256]),
            validation_cache_sha256=np.asarray([validation_cache_sha256]),
        )
    os.replace(temporary, path)
    if not cache_is_ignored(root, path):
        raise ValueError("V4.3 strict scene cache is not ignored by Git")
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "tracked": False,
        "contents": "compact strict scene labels, groups, three seed scores, calibrated ensemble scores, and frozen predictions; no raster pixels",
    }


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    candidate = report["strict_spatial_test"]["primary_scene"]
    pixel = report["strict_spatial_test"]["segmentation"]
    baseline = report["same_cohort_comparison"]["released_mars_s2l"]
    delta = report["same_cohort_comparison"]["delta"]
    bootstrap = report["paired_group_bootstrap"]
    lines = [
        "# ERSRR v4.3 frozen strict MARS comparison",
        "",
        "Development benchmark on the already-opened ERSRR strict cohort. All v4.3 calibrators and thresholds were frozen on internal validation.",
        "",
        "| Model | Recall | FPR | Precision | AP | AUROC | Pixel IoU | Pixel Dice |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| ERSRR v4.3 ensemble | {candidate['recall']:.4f} | "
            f"{candidate['false_positive_rate']:.4f} | {candidate['precision']:.4f} | "
            f"{candidate['average_precision']:.4f} | {candidate['auroc']:.4f} | "
            f"{pixel['intersection_over_union']:.4f} | {pixel['dice']:.4f} |"
        ),
        (
            f"| Released MARS-S2L | {baseline['recall']:.4f} | "
            f"{baseline['false_positive_rate']:.4f} | {baseline['precision']:.4f} | "
            f"{baseline['average_precision']:.4f} | {baseline['auroc']:.4f} | "
            f"{baseline['pixel_intersection_over_union']:.4f} | {baseline['pixel_dice']:.4f} |"
        ),
        "",
        "## Same-cohort deltas (ERSRR - released MARS-S2L)",
        "",
        f"- Recall: {delta['recall']:+.4f} (paired group-bootstrap 95% CI {bootstrap['recall_delta']['95ci'][0]:+.4f} to {bootstrap['recall_delta']['95ci'][1]:+.4f})",
        f"- FPR: {delta['false_positive_rate']:+.4f} (95% CI {bootstrap['false_positive_rate_delta']['95ci'][0]:+.4f} to {bootstrap['false_positive_rate_delta']['95ci'][1]:+.4f})",
        f"- AP: {delta['average_precision']:+.4f} (95% CI {bootstrap['average_precision_delta']['95ci'][0]:+.4f} to {bootstrap['average_precision_delta']['95ci'][1]:+.4f})",
        f"- AUROC: {delta['auroc']:+.4f} (95% CI {bootstrap['auroc_delta']['95ci'][0]:+.4f} to {bootstrap['auroc_delta']['95ci'][1]:+.4f})",
        f"- Pixel IoU / Dice: {delta['pixel_intersection_over_union']:+.4f} / {delta['pixel_dice']:+.4f}",
        "",
        "## Interpretation",
        "",
        report["decision"],
        "",
        "The official MARS-S2L paper benchmarks use different, much larger test cohorts and are contextual only; they are not substituted for this paired comparison.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-dir", default=DEFAULT_OUTPUT.as_posix())
    parser.add_argument("--metadata-csv", default=DEFAULT_METADATA_CSV.as_posix())
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT.as_posix())
    parser.add_argument("--baseline", default=DEFAULT_BASELINE.as_posix())
    parser.add_argument("--output-json", default=DEFAULT_JSON.as_posix())
    parser.add_argument("--output-markdown", default=DEFAULT_MARKDOWN.as_posix())
    parser.add_argument("--prediction-cache", default=DEFAULT_CACHE.as_posix())
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    root = repo_root()
    start_dirty = tracked_dirty(root)
    if start_dirty:
        raise RuntimeError("Refusing strict evaluation from a dirty tracked worktree")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for v4.3 strict evaluation")
    metadata_dir = checked_output_dir(root, args.metadata_dir)
    metadata_csv = (root / args.metadata_csv).resolve()
    experiment_path = (root / args.experiment).resolve()
    baseline_path = (root / args.baseline).resolve()
    experiment = json.loads(experiment_path.read_text(encoding="utf-8"))
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    if experiment.get("scope") != "v4_3_predeclared_internal_validation_ensemble":
        raise ValueError("Expected the frozen v4.3 ensemble validation report")
    if experiment.get("strict_evaluation_authorized") is not True:
        raise ValueError("V4.3 internal validation did not authorize strict evaluation")
    if experiment["cohort"].get("strict_spatial_test_loaded") is not False:
        raise ValueError("V4.3 selection report is not strict-isolated")
    if baseline.get("scope") != "released_mars-s2l_on_frozen_full_strict_spatial_cohort":
        raise ValueError("Unexpected released MARS baseline scope")

    validation_cache_path, validation_cache = load_validation_cache(root, experiment)
    released_cache_path, released_cache = load_released_cache(root, baseline)
    source_reports = []
    models = []
    device = torch.device("cuda")
    for expected_seed, source in zip(FIXED_SEEDS, experiment["source_reports"]):
        if int(source["seed"]) != expected_seed:
            raise ValueError("V4.3 source seed ordering mismatch")
        report_path = (root / source["path"]).resolve()
        if sha256(report_path) != source["sha256"]:
            raise ValueError("V4.3 source report identity mismatch")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        checkpoint = (root / source["checkpoint"]["path"]).resolve()
        if sha256(checkpoint) != source["checkpoint"]["sha256"]:
            raise ValueError("V4.3 checkpoint identity mismatch")
        model = MarsV4Model(
            scene_topk_fraction=float(report["model"]["scene_topk_fraction"]),
            scene_max_weight=float(report["model"]["scene_max_weight"]),
        ).to(device)
        payload = torch.load(checkpoint, map_location=device, weights_only=True)
        if payload["model_metadata"] != report["model"]:
            raise ValueError("V4.3 checkpoint metadata mismatch")
        model.load_state_dict(payload["state_dict"], strict=True)
        models.append(model)
        source_reports.append(report)

    manifest = metadata_dir / V3_STRICT_SAMPLES
    strict_cohort_path = root / STRICT_COHORT_JSON
    strict_cohort = json.loads(strict_cohort_path.read_text(encoding="utf-8"))
    strict_manifest_identity = sha256(manifest)
    if strict_manifest_identity != strict_cohort["identities"]["sample_manifest_sha256"]:
        raise ValueError("Strict MARS manifest identity mismatch")
    if strict_manifest_identity != baseline["source"]["evaluation_manifest_sha256"]:
        raise ValueError("Candidate and released MARS strict manifests differ")
    strict_records = [
        record
        for record in iter_manifest(manifest)
        if record["research_role"] == "strict_spatial_test"
    ]
    if len(strict_records) != int(strict_cohort["samples"]["total"]):
        raise ValueError("Strict MARS row count mismatch")
    training_records = list(iter_manifest(metadata_dir / V3_SAMPLES))
    fit_positive_ids = {
        str(record["sample_id"])
        for record in training_records
        if record["research_role"] == "internal_training" and record["label_state"] == "PLUME"
    }
    required_ids = {str(record["sample_id"]) for record in strict_records}
    scene_metadata, _ = metadata_and_plume_library(
        metadata_dir, metadata_csv, required_ids, fit_positive_ids
    )
    dataset = MarsV4Dataset(
        metadata_dir,
        strict_records,
        scene_metadata,
        lut_path=(root / source_reports[0]["simulation"].get("lut_path", DEFAULT_LUT)).resolve(),
        plume_library=[],
        augment=False,
        simulation_fraction=0.0,
        seed=0,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    predictions = collect_predictions(models, loader, device)
    expected_ids = [str(record["sample_id"]) for record in strict_records]
    if list(predictions["sample_ids"].astype(str)) != expected_ids:
        raise ValueError("V4.3 strict inference order differs from the frozen manifest")
    labels = predictions["labels"]
    groups = predictions["groups"].astype(str)
    validation_scores = np.asarray(validation_cache["seed_scores"], dtype=np.float64)
    ensemble_scores = calibrated_ensemble(validation_scores, predictions["seed_scores"])
    frozen_points = experiment["final_development_rule"]["operating_points"]
    operating_points = {
        target: metrics(labels, ensemble_scores, float(rule["threshold"]))
        for target, rule in frozen_points.items()
    }
    primary_threshold = float(frozen_points["0.05"]["threshold"])
    primary_scene = operating_points["0.05"]
    interval = bootstrap_ci(labels, ensemble_scores, groups, primary_threshold, BOOTSTRAP_SEED)
    pixel_threshold = float(experiment["segmentation"]["selected"]["threshold"])
    segmentation = strict_pixel_metrics(
        predictions["segmentation"],
        predictions["observable"],
        predictions["truth"],
        pixel_threshold,
    )

    released = baseline["strict_spatial_test"]["scene_unweighted"]
    released_pixel = baseline["strict_spatial_test"]["pixel_validity_aware"]
    aligned = align_released_cache(released_cache, predictions["sample_ids"])
    if not np.array_equal(aligned["labels"].astype(np.uint8), labels):
        raise ValueError("Released MARS cache labels differ from the v4.3 strict labels")
    candidate_fixed = ensemble_scores >= primary_threshold
    baseline_fixed = aligned["predictions"].astype(bool)
    paired = paired_group_bootstrap(
        labels,
        groups,
        ensemble_scores,
        candidate_fixed,
        aligned["scores"].astype(np.float64),
        baseline_fixed,
    )
    comparison_baseline = {
        **released,
        "pixel_average_precision": released_pixel["average_precision"],
        "pixel_intersection_over_union": released_pixel["intersection_over_union"],
        "pixel_dice": released_pixel["dice"],
    }
    delta = {
        "recall": float(primary_scene["recall"] - released["recall"]),
        "false_positive_rate": float(
            primary_scene["false_positive_rate"] - released["false_positive_rate"]
        ),
        "precision": float(primary_scene["precision"] - released["precision"]),
        "average_precision": float(
            primary_scene["average_precision"] - released["average_precision"]
        ),
        "auroc": float(primary_scene["auroc"] - released["auroc"]),
        "pixel_average_precision": float(
            segmentation["average_precision"] - released_pixel["average_precision"]
        ),
        "pixel_intersection_over_union": float(
            segmentation["intersection_over_union"]
            - released_pixel["intersection_over_union"]
        ),
        "pixel_dice": float(segmentation["dice"] - released_pixel["dice"]),
    }
    point_checks = {
        "recall_higher": delta["recall"] > 0,
        "false_positive_rate_lower": delta["false_positive_rate"] < 0,
        "average_precision_higher": delta["average_precision"] > 0,
        "auroc_higher": delta["auroc"] > 0,
        "pixel_iou_higher": delta["pixel_intersection_over_union"] > 0,
        "pixel_dice_higher": delta["pixel_dice"] > 0,
    }
    uncertainty_checks = {
        "recall_delta_lower_95ci_above_zero": paired["recall_delta"]["95ci"][0] > 0,
        "fpr_delta_upper_95ci_below_zero": paired["false_positive_rate_delta"]["95ci"][1] < 0,
        "ap_delta_lower_95ci_above_zero": paired["average_precision_delta"]["95ci"][0] > 0,
        "auroc_delta_lower_95ci_above_zero": paired["auroc_delta"]["95ci"][0] > 0,
    }
    point_outperformance = all(point_checks.values())
    uncertainty_supported = all(uncertainty_checks.values())
    if point_outperformance and uncertainty_supported:
        decision = (
            "ERSRR v4.3 outperforms the released MARS-S2L checkpoint on this same frozen strict "
            "cohort at the predeclared operating rule, with paired group-bootstrap support for all "
            "four scene-level deltas. This is strong development evidence, not a publishable "
            "superiority claim, because the cohort was opened during earlier v3 research."
        )
    elif point_outperformance:
        decision = (
            "ERSRR v4.3 has better same-cohort point estimates than the released MARS-S2L checkpoint "
            "on every declared scene and segmentation metric, but at least one paired group-bootstrap "
            "interval crosses zero. Treat this as promising development evidence only."
        )
    else:
        failed = [name for name, passed in point_checks.items() if not passed]
        decision = (
            "ERSRR v4.3 does not outperform the released MARS-S2L checkpoint on every same-cohort "
            f"point metric; failed comparisons: {', '.join(failed)}. Preserve this result and do not "
            "retune from strict behavior."
        )

    experiment_identity = sha256(experiment_path)
    prediction_cache_path = safe_output(root, args.prediction_cache)
    prediction_cache = write_prediction_cache(
        root,
        prediction_cache_path,
        predictions=predictions,
        ensemble_scores=ensemble_scores,
        threshold=primary_threshold,
        strict_manifest_sha256=strict_manifest_identity,
        experiment_sha256=experiment_identity,
        validation_cache_sha256=sha256(validation_cache_path),
    )
    report = {
        "schema_version": 1,
        "scope": "frozen_v4_3_ensemble_on_opened_full_strict_spatial_cohort",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "interpretation": (
            "development comparison on the already-opened ERSRR strict cohort; all v4.3 rules were "
            "frozen on internal validation and no strict threshold selection was performed"
        ),
        "source": {
            "dataset": "UNEP-IMEO/MARS-S2L",
            "strict_manifest_sha256": strict_manifest_identity,
            "strict_cohort_report": {
                "path": strict_cohort_path.relative_to(root).as_posix(),
                "sha256": sha256(strict_cohort_path),
            },
            "validation_experiment": {
                "path": experiment_path.relative_to(root).as_posix(),
                "sha256": experiment_identity,
            },
            "validation_calibration_cache": {
                "path": validation_cache_path.relative_to(root).as_posix(),
                "sha256": sha256(validation_cache_path),
            },
        },
        "cohort": {
            "samples": int(labels.size),
            "positives": int(np.count_nonzero(labels == 1)),
            "negatives": int(np.count_nonzero(labels == 0)),
            "groups": int(np.unique(groups).size),
            "previously_opened_by_v3_research": True,
        },
        "model": {
            "name": "ERSRR v4.3 three-checkpoint percentile ensemble",
            "seeds": list(FIXED_SEEDS),
            "parameter_count_per_checkpoint": source_reports[0]["model"]["parameter_count"],
            "scene_rule": experiment["ensemble_rule"]["scene"],
            "segmentation_rule": experiment["ensemble_rule"]["segmentation"],
        },
        "operating_rule": {
            "selected_on": "internal validation only",
            "scene_thresholds": frozen_points,
            "pixel_threshold": pixel_threshold,
            "minimum_connected_pixels": 1,
        },
        "strict_spatial_test": {
            "primary_scene": primary_scene,
            "all_frozen_operating_points": operating_points,
            "group_bootstrap": interval,
            "segmentation": segmentation,
        },
        "scene_prediction_cache": prediction_cache,
        "same_cohort_comparison": {
            "released_mars_report": {
                "path": baseline_path.relative_to(root).as_posix(),
                "sha256": sha256(baseline_path),
                "scene_cache_path": released_cache_path.relative_to(root).as_posix(),
                "scene_cache_sha256": sha256(released_cache_path),
            },
            "released_mars_s2l": comparison_baseline,
            "delta": delta,
            "point_outperformance_checks": point_checks,
            "all_point_metrics_better": point_outperformance,
            "uncertainty_checks": uncertainty_checks,
            "all_scene_deltas_bootstrap_supported": uncertainty_supported,
        },
        "paired_group_bootstrap": paired,
        "official_mars_s2l_paper_targets_not_same_cohort": PAPER_TARGETS,
        "decision": decision,
        "provenance": {
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True
            ).strip(),
            "git_tracked_worktree_dirty_at_start": start_dirty,
            "script": Path(__file__).resolve().relative_to(root).as_posix(),
            "script_sha256": sha256(Path(__file__).resolve()),
            "runtime": {
                "torch": torch.__version__,
                "cuda_runtime": torch.version.cuda,
                "numpy": np.__version__,
                "rasterio": rasterio.__version__,
                "sklearn": sklearn.__version__,
                "device": torch.cuda.get_device_name(0),
            },
        },
    }
    output_json = safe_output(root, args.output_json)
    output_markdown = safe_output(root, args.output_markdown)
    write_json(output_json, report)
    write_markdown(output_markdown, report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
