#!/usr/bin/env python3
"""Evaluate the predeclared v4.3 three-checkpoint ensemble on internal validation."""

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
from aggregate_mars_v4_2_validation import (  # noqa: E402
    DEFAULT_REPORTS,
    FIXED_SEEDS,
    frozen_contract,
    verify_seed_report,
)
from analyze_mars_v4_scoring import empirical_percentile  # noqa: E402
from build_mars_v3_training_cohort import V3_SAMPLES  # noqa: E402
from train_mars_v3 import DEFAULT_METADATA_CSV, safe_output, tracked_dirty, write_json  # noqa: E402
from train_mars_v4 import (  # noqa: E402
    DEFAULT_LUT,
    MarsV4Dataset,
    metadata_and_plume_library,
    move_batch,
)
from train_mars_v4_cascade import (  # noqa: E402
    balanced_group_splits,
    choose_threshold_at_fpr,
)

DEFAULT_CACHE = DEFAULT_OUTPUT / "publication_v4_3_internal_validation_ensemble.npz"
DEFAULT_JSON = Path("reports/experiments/mars_v4_3_ensemble_validation.json")
DEFAULT_MARKDOWN = Path("reports/experiments/MARS_V4_3_ENSEMBLE_VALIDATION.md")
PROTOCOL = Path("reports/experiments/MARS_V4_3_ENSEMBLE_PROTOCOL.md")
TARGET_FPRS = (0.05, 0.08, 0.095)
PIXEL_THRESHOLDS = tuple(float(value) for value in np.linspace(0.1, 0.9, 9))
OUTER_FOLDS = 5
FOLD_SEED = 20_260_713


def binary_metrics(labels: np.ndarray, decisions: np.ndarray) -> dict[str, float | int]:
    truth = np.asarray(labels, dtype=np.uint8)
    predicted = np.asarray(decisions, dtype=bool)
    if truth.shape != predicted.shape or truth.ndim != 1:
        raise ValueError("Binary metrics require aligned one-dimensional arrays")
    positive = truth == 1
    negative = ~positive
    tp = int(np.count_nonzero(predicted & positive))
    fp = int(np.count_nonzero(predicted & negative))
    fn = int(np.count_nonzero(~predicted & positive))
    tn = int(np.count_nonzero(~predicted & negative))
    return {
        "recall": tp / max(tp + fn, 1),
        "false_positive_rate": fp / max(fp + tn, 1),
        "precision": tp / max(tp + fp, 1),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def calibrated_ensemble(training_scores: np.ndarray, evaluation_scores: np.ndarray) -> np.ndarray:
    training = np.asarray(training_scores, dtype=np.float64)
    evaluation = np.asarray(evaluation_scores, dtype=np.float64)
    if (
        training.ndim != 2
        or evaluation.ndim != 2
        or training.shape[1] != len(FIXED_SEEDS)
        or evaluation.shape[1] != len(FIXED_SEEDS)
        or not training.shape[0]
    ):
        raise ValueError("Ensemble calibration requires aligned three-seed score matrices")
    calibrated = np.column_stack(
        [
            empirical_percentile(training[:, column], evaluation[:, column])
            for column in range(training.shape[1])
        ]
    )
    return np.mean(calibrated, axis=1)


def group_held_audit(
    labels: np.ndarray, groups: np.ndarray, seed_scores: np.ndarray
) -> dict[str, Any]:
    truth = np.asarray(labels, dtype=np.uint8)
    group_values = np.asarray(groups).astype(str)
    scores = np.asarray(seed_scores, dtype=np.float64)
    if truth.shape != group_values.shape or scores.shape != (truth.size, len(FIXED_SEEDS)):
        raise ValueError("Group-held ensemble inputs are misaligned")
    splits = balanced_group_splits(
        truth, group_values, folds=OUTER_FOLDS, seed=FOLD_SEED
    )
    held_scores = np.full(truth.shape, np.nan, dtype=np.float64)
    held_decisions = {
        target: np.zeros(truth.shape, dtype=bool) for target in TARGET_FPRS
    }
    folds = []
    for fold, (training, held_out) in enumerate(splits, start=1):
        training_ensemble = calibrated_ensemble(scores[training], scores[training])
        local_held_scores = calibrated_ensemble(scores[training], scores[held_out])
        held_scores[held_out] = local_held_scores
        thresholds = {}
        operating = {}
        for target in TARGET_FPRS:
            selection = choose_threshold_at_fpr(truth[training], training_ensemble, target)
            decisions = local_held_scores >= float(selection["threshold"])
            held_decisions[target][held_out] = decisions
            thresholds[str(target)] = selection
            operating[str(target)] = binary_metrics(truth[held_out], decisions)
        folds.append(
            {
                "fold": fold,
                "training_groups": int(np.unique(group_values[training]).size),
                "held_out_groups": int(np.unique(group_values[held_out]).size),
                "training_scenes": int(training.size),
                "held_out_scenes": int(held_out.size),
                "held_out_positives": int(np.count_nonzero(truth[held_out] == 1)),
                "thresholds_selected_on_training_groups": thresholds,
                "held_out_operating_points": operating,
                "held_out_ranking": {
                    "average_precision": float(
                        average_precision_score(truth[held_out], local_held_scores)
                    ),
                    "auroc": float(roc_auc_score(truth[held_out], local_held_scores)),
                },
            }
        )
    if not np.all(np.isfinite(held_scores)):
        raise ValueError("Group-held audit did not score every scene")
    return {
        "method": (
            "five fixed 25 km group folds; fit each seed empirical CDF and every operating "
            "threshold on four folds, then apply the mean calibrated percentile to the held fold"
        ),
        "ranking": {
            "average_precision": float(average_precision_score(truth, held_scores)),
            "auroc": float(roc_auc_score(truth, held_scores)),
        },
        "operating_points": {
            str(target): binary_metrics(truth, decisions)
            for target, decisions in held_decisions.items()
        },
        "folds": folds,
    }


def final_rule(labels: np.ndarray, seed_scores: np.ndarray) -> dict[str, Any]:
    truth = np.asarray(labels, dtype=np.uint8)
    scores = np.asarray(seed_scores, dtype=np.float64)
    ensemble = calibrated_ensemble(scores, scores)
    return {
        "calibration": (
            "per-seed empirical CDF fitted on all internal-validation scores; arithmetic mean "
            "of the three percentiles"
        ),
        "ranking": {
            "average_precision": float(average_precision_score(truth, ensemble)),
            "auroc": float(roc_auc_score(truth, ensemble)),
        },
        "operating_points": {
            str(target): choose_threshold_at_fpr(truth, ensemble, target)
            for target in TARGET_FPRS
        },
    }


def select_pixel_rule(
    intersections: np.ndarray, predicted: np.ndarray, truth_pixels: float
) -> dict[str, Any]:
    intersection = np.asarray(intersections, dtype=np.float64)
    prediction = np.asarray(predicted, dtype=np.float64)
    if intersection.shape != prediction.shape or intersection.size != len(PIXEL_THRESHOLDS):
        raise ValueError("Pixel statistics do not match the frozen threshold grid")
    rows = []
    for threshold, overlap, predicted_pixels in zip(
        PIXEL_THRESHOLDS, intersection, prediction
    ):
        denominator = predicted_pixels + truth_pixels
        dice = 0.0 if denominator == 0 else float(2.0 * overlap / denominator)
        union = predicted_pixels + truth_pixels - overlap
        rows.append(
            {
                "threshold": threshold,
                "positive_pixel_dice": dice,
                "positive_pixel_iou": 0.0 if union == 0 else float(overlap / union),
                "intersection_pixels": int(overlap),
                "predicted_pixels": int(predicted_pixels),
                "truth_pixels": int(truth_pixels),
            }
        )
    selected = max(rows, key=lambda row: (row["positive_pixel_dice"], -row["threshold"]))
    return {
        "selection": "maximum positive-pixel Dice; lower threshold breaks an exact tie",
        "threshold_grid": rows,
        "selected": selected,
    }


def cache_is_ignored(root: Path, path: Path) -> bool:
    relative = path.resolve().relative_to(root.resolve())
    result = subprocess.run(
        ["git", "check-ignore", "--quiet", "--", relative.as_posix()],
        cwd=root,
        check=False,
    )
    return result.returncode == 0


def write_cache(
    path: Path,
    *,
    sample_ids: Sequence[str],
    groups: Sequence[str],
    labels: np.ndarray,
    scores: np.ndarray,
    report_hashes: Sequence[str],
    manifest_sha256: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as destination:
        np.savez_compressed(
            destination,
            schema_version=np.asarray([1], dtype=np.int16),
            sample_ids=np.asarray(sample_ids),
            groups=np.asarray(groups),
            labels=np.asarray(labels, dtype=np.uint8),
            seeds=np.asarray(FIXED_SEEDS, dtype=np.int32),
            seed_scores=np.asarray(scores, dtype=np.float32),
            source_report_sha256=np.asarray(report_hashes),
            manifest_sha256=np.asarray([manifest_sha256]),
        )
    os.replace(temporary, path)


@torch.no_grad()
def collect_predictions(
    models: Sequence[MarsV4Model],
    loader: DataLoader[dict[str, Any]],
    device: torch.device,
) -> tuple[list[str], list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    for model in models:
        model.eval()
    sample_ids: list[str] = []
    groups: list[str] = []
    labels: list[np.ndarray] = []
    scene_rows: list[np.ndarray] = []
    intersections = np.zeros(len(PIXEL_THRESHOLDS), dtype=np.float64)
    predicted_pixels = np.zeros(len(PIXEL_THRESHOLDS), dtype=np.float64)
    truth_pixels = 0.0
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
        ensemble_probability = probability_sum / len(models)
        truth = (moved["mask"] * moved["observable"]) > 0.5
        positive = moved["presence"] > 0.5
        if torch.any(positive):
            local_probability = (
                ensemble_probability[positive] * moved["observable"][positive]
            )
            local_truth = truth[positive]
            truth_pixels += float(local_truth.sum())
            for index, threshold in enumerate(PIXEL_THRESHOLDS):
                prediction = local_probability >= threshold
                intersections[index] += float((prediction & local_truth).sum())
                predicted_pixels[index] += float(prediction.sum())
        sample_ids.extend(str(value) for value in batch["sample_id"])
        groups.extend(str(value) for value in batch["group_id"])
        labels.append(batch["presence"].numpy().astype(np.uint8))
        scene_rows.append(np.column_stack(local_scene_scores))
        completed += len(batch["sample_id"])
        if completed // 500 != (completed - len(batch["sample_id"])) // 500:
            print(f"Evaluated v4.3 ensemble for {completed} scenes", flush=True)
    return (
        sample_ids,
        groups,
        np.concatenate(labels),
        np.concatenate(scene_rows, axis=0),
        intersections,
        predicted_pixels,
        truth_pixels,
    )


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    group = report["group_held_audit"]
    final = report["final_development_rule"]
    pixel = report["segmentation"]["selected"]
    reference = report["v3_internal_reference"]["mean"]
    op5 = group["operating_points"]["0.05"]
    final5 = final["operating_points"]["0.05"]
    lines = [
        "# ERSRR v4.3 frozen ensemble validation",
        "",
        "Predeclared internal-development evaluation; the strict cohort was not loaded.",
        "",
        "| Estimate | AP | AUROC | Recall @ <=5% FPR | FPR | Pixel Dice |",
        "|---|---:|---:|---:|---:|---:|",
        (
            f"| Five-fold 25 km group-held | {group['ranking']['average_precision']:.4f} | "
            f"{group['ranking']['auroc']:.4f} | {op5['recall']:.4f} | "
            f"{op5['false_positive_rate']:.4f} | {pixel['positive_pixel_dice']:.4f} |"
        ),
        (
            f"| Final all-validation rule | {final['ranking']['average_precision']:.4f} | "
            f"{final['ranking']['auroc']:.4f} | {final5['training_recall']:.4f} | "
            f"{final5['training_fpr']:.4f} | {pixel['positive_pixel_dice']:.4f} |"
        ),
        (
            f"| v3 five-seed mean | {reference['average_precision']:.4f} | "
            f"{reference['auroc']:.4f} | {reference['recall_at_fpr5']:.4f} | <=0.0500 | "
            f"{reference['positive_pixel_dice']:.4f} |"
        ),
        "",
        f"- Selected ensemble pixel threshold: {pixel['threshold']:.1f}",
        f"- Strict evaluation authorized: {str(report['strict_evaluation_authorized']).lower()}",
        "",
        "## Decision",
        "",
        report["decision"],
        "",
        "These internal values are not directly comparable to the MARS-S2L strict or paper benchmarks.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-dir", default=DEFAULT_OUTPUT.as_posix())
    parser.add_argument("--metadata-csv", default=DEFAULT_METADATA_CSV.as_posix())
    parser.add_argument("--cache", default=DEFAULT_CACHE.as_posix())
    parser.add_argument("--output-json", default=DEFAULT_JSON.as_posix())
    parser.add_argument("--output-markdown", default=DEFAULT_MARKDOWN.as_posix())
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    root = repo_root()
    start_dirty = tracked_dirty(root)
    if start_dirty:
        raise RuntimeError("Refusing to evaluate the frozen ensemble from a dirty tracked worktree")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for v4.3 ensemble evaluation")
    protocol = root / PROTOCOL
    reports = []
    report_paths = []
    for seed, relative in zip(FIXED_SEEDS, DEFAULT_REPORTS):
        path = root / relative
        report = json.loads(path.read_text(encoding="utf-8"))
        verify_seed_report(report, seed)
        reports.append(report)
        report_paths.append(path)
    contract = frozen_contract(reports[0])
    if any(frozen_contract(report) != contract for report in reports[1:]):
        raise ValueError("V4.2 seed reports do not share a frozen architecture contract")
    if any(report["cohort"]["strict_spatial_test_loaded"] for report in reports):
        raise ValueError("A source report loaded the strict cohort")

    metadata_dir = checked_output_dir(root, args.metadata_dir)
    metadata_csv = (root / args.metadata_csv).resolve()
    manifest = metadata_dir / V3_SAMPLES
    manifest_identity = sha256(manifest)
    if manifest_identity != reports[0]["source"]["manifest_sha256"]:
        raise ValueError("Validation manifest differs from the frozen reports")
    all_records = list(iter_manifest(manifest))
    records = [
        record for record in all_records if record["research_role"] == "internal_validation"
    ]
    fit_positive_ids = {
        str(record["sample_id"])
        for record in all_records
        if record["research_role"] == "internal_training" and record["label_state"] == "PLUME"
    }
    required_ids = {str(record["sample_id"]) for record in records}
    scene_metadata, _ = metadata_and_plume_library(
        metadata_dir, metadata_csv, required_ids, fit_positive_ids
    )
    dataset = MarsV4Dataset(
        metadata_dir,
        records,
        scene_metadata,
        lut_path=(root / reports[0]["simulation"].get("lut_path", DEFAULT_LUT)).resolve(),
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
    device = torch.device("cuda")
    models = []
    for report in reports:
        checkpoint = (root / report["artifact"]["path"]).resolve()
        if sha256(checkpoint) != report["artifact"]["sha256"]:
            raise ValueError("V4.2 checkpoint identity mismatch")
        model = MarsV4Model(
            scene_topk_fraction=float(report["model"]["scene_topk_fraction"]),
            scene_max_weight=float(report["model"]["scene_max_weight"]),
        ).to(device)
        payload = torch.load(checkpoint, map_location=device, weights_only=True)
        if payload["model_metadata"] != report["model"]:
            raise ValueError("V4.2 checkpoint metadata differs from its frozen report")
        model.load_state_dict(payload["state_dict"], strict=True)
        models.append(model)

    sample_ids, groups, labels, seed_scores, intersections, predictions, truth_pixels = (
        collect_predictions(models, loader, device)
    )
    expected_ids = [str(record["sample_id"]) for record in records]
    if sample_ids != expected_ids:
        raise ValueError("Ensemble inference order differs from the frozen manifest")
    group_audit = group_held_audit(labels, np.asarray(groups), seed_scores)
    final = final_rule(labels, seed_scores)
    pixel = select_pixel_rule(intersections, predictions, truth_pixels)
    reference = reports[0]["v3_internal_reference"]
    baseline = reference["mean"]
    group5 = group_audit["operating_points"]["0.05"]
    final5 = final["operating_points"]["0.05"]
    checks = {
        "group_held_ap_not_below_v3_mean": (
            group_audit["ranking"]["average_precision"] >= baseline["average_precision"]
        ),
        "group_held_auroc_not_below_v3_mean": (
            group_audit["ranking"]["auroc"] >= baseline["auroc"]
        ),
        "group_held_recall_at_fpr5_not_below_v3_mean": (
            group5["recall"] >= baseline["recall_at_fpr5"]
        ),
        "group_held_fpr_at_most_0_05": group5["false_positive_rate"] <= 0.05,
        "final_ap_not_below_v3_mean": (
            final["ranking"]["average_precision"] >= baseline["average_precision"]
        ),
        "final_auroc_not_below_v3_mean": (
            final["ranking"]["auroc"] >= baseline["auroc"]
        ),
        "final_recall_at_fpr5_not_below_v3_mean": (
            final5["training_recall"] >= baseline["recall_at_fpr5"]
        ),
        "final_fpr_at_most_0_05": final5["training_fpr"] <= 0.05,
        "pixel_dice_not_below_v3_mean": (
            pixel["selected"]["positive_pixel_dice"] >= baseline["positive_pixel_dice"]
        ),
    }
    promoted = all(checks.values())
    failed = [name for name, passed in checks.items() if not passed]
    decision = (
        "Promote the frozen v4.3 ensemble to one evaluation on the already-opened strict MARS "
        "cohort using only these validation calibrators, scene thresholds, and pixel threshold. "
        "Treat the comparison as development evidence, not a new untouched paper test."
        if promoted
        else "Do not load the strict MARS cohort for v4.3. The predeclared ensemble failed: "
        + ", ".join(failed)
        + ". Preserve the result and revise the architecture hypothesis."
    )
    cache = safe_output(root, args.cache)
    report_hashes = [sha256(path) for path in report_paths]
    write_cache(
        cache,
        sample_ids=sample_ids,
        groups=groups,
        labels=labels,
        scores=seed_scores,
        report_hashes=report_hashes,
        manifest_sha256=manifest_identity,
    )
    if not cache_is_ignored(root, cache):
        raise ValueError("Compact ensemble cache is not ignored by Git")
    output = {
        "schema_version": 1,
        "scope": "v4_3_predeclared_internal_validation_ensemble",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "cohort": {
            "scenes": int(labels.size),
            "positives": int(np.count_nonzero(labels == 1)),
            "negatives": int(np.count_nonzero(labels == 0)),
            "groups": int(np.unique(groups).size),
            "strict_spatial_test_loaded": False,
        },
        "protocol": {"path": PROTOCOL.as_posix(), "sha256": sha256(protocol)},
        "source_reports": [
            {
                "seed": seed,
                "path": path.relative_to(root).as_posix(),
                "sha256": identity,
                "checkpoint": report["artifact"],
            }
            for seed, path, identity, report in zip(
                FIXED_SEEDS, report_paths, report_hashes, reports
            )
        ],
        "ensemble_rule": {
            "scene": (
                "mean of three per-seed empirical percentiles; each seed score is sigmoid(mean "
                "of top 2% observable segmentation logits)"
            ),
            "segmentation": "arithmetic mean of three per-pixel plume probabilities",
            "learned_combiner": False,
        },
        "group_held_audit": group_audit,
        "final_development_rule": final,
        "segmentation": pixel,
        "v3_internal_reference": reference,
        "promotion_checks": checks,
        "strict_evaluation_authorized": promoted,
        "decision": decision,
        "ignored_calibration_cache": {
            "path": cache.relative_to(root).as_posix(),
            "bytes": cache.stat().st_size,
            "sha256": sha256(cache),
            "tracked": False,
        },
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
    write_json(output_json, output)
    write_markdown(output_markdown, output)
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
