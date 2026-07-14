#!/usr/bin/env python3
"""One-shot exact-paper evaluation of the frozen ERSRR MARS successor."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import rasterio
import scipy
import sklearn
import torch
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "EarthRemoteSensingRapidResponse"
for path in (MODEL_ROOT, ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from acquire_mars_metadata import DEFAULT_OUTPUT, repo_root, sha256  # noqa: E402
from analyze_mars_mask_thresholds import component_mask_at  # noqa: E402
from audit_mars_paper_benchmark import (  # noqa: E402
    DEFAULT_ASSET_METADATA,
    DEFAULT_CONFIG as PAPER_CONFIG,
    DEFAULT_METADATA as PAPER_METADATA,
    DEFAULT_OFFSHORE,
    DEFAULT_ONSHORE,
    PUBLISHED,
    reconstruct,
)
from evaluate_mars_residual_endpoint_blend import (  # noqa: E402
    load_residual_model,
    trust_region_logits,
)
from evaluate_released_marss2l import connected_scene_score  # noqa: E402
from extract_mars_scene_features import pooled_scene_features, tensor_feature_names  # noqa: E402
from mars_paper_model import SENSOR_NAMES  # noqa: E402
from train_mars_context_scene_ranker import augment_site_context  # noqa: E402
from train_mars_paper_residual import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    MarsPaperDataset,
    move_batch,
)
from train_mars_scene_ranker import blend_scores, predict_model  # noqa: E402
from train_mars_v4 import choose_threshold_at_fpr  # noqa: E402

DEFAULT_SPEC = Path("configs/mars_successor_paper_test_v1.json")
DEFAULT_MANIFEST = DEFAULT_OUTPUT / "paper_v3_sealed_test_samples.jsonl"
DEFAULT_RECEIPT = Path("reports/acquisition/mars_s2l_paper_v3_test_download.json")
DEFAULT_RESIDUAL = Path("EarthRemoteSensingRapidResponse/artifacts/mars_paper_residual_fold0_seed606.pt")
DEFAULT_HEAD = Path("EarthRemoteSensingRapidResponse/artifacts/mars_oof_context_ranker_folds234.joblib")
DEFAULT_SELECTION = Path("reports/experiments/mars_oof_context_minimum_blend.json")
DEFAULT_SCENE_FOLD0 = Path("reports/experiments/mars_oof_context_minimum_blend_fold0.json")
DEFAULT_MASK_SELECTION = Path("reports/experiments/mars_mask_threshold_folds234.json")
DEFAULT_MASK_CONFIRMATION = Path("reports/experiments/mars_mask_threshold_folds01_confirmation.json")
DEFAULT_PAPER_BENCHMARK = Path("reports/acquisition/mars_s2l_paper_v3_benchmark.json")
DEFAULT_MIXED_COHORT = Path("reports/acquisition/mars_s2l_paper_v3_mixed_cohort.json")
DEFAULT_JSON = Path("reports/experiments/mars_successor_paper_test.json")
DEFAULT_MARKDOWN = Path("reports/experiments/MARS_SUCCESSOR_PAPER_TEST.md")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_sealed_records(path: Path, expected_sha256: str) -> list[dict[str, Any]]:
    if sha256(path) != expected_sha256:
        raise ValueError("Sealed paper-test manifest hash mismatch")
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            record = json.loads(line)
            if record.get("research_role") != "sealed_paper_test":
                raise ValueError(f"Non-test role in sealed manifest line {line_number}")
            records.append(record)
    if len(records) != 43_524 or len({row["sample_id"] for row in records}) != len(records):
        raise ValueError("Sealed manifest does not contain 43,524 unique available scenes")
    if {(int(row["width"]), int(row["height"])) for row in records} != {(200, 200)}:
        raise ValueError("Paper-test grid differs from the frozen 200x200 missing-scene policy")
    return records


def verify_receipt(path: Path, manifest_sha256: str) -> None:
    receipt = load_json(path)
    result = receipt.get("result", {})
    if (
        not result.get("ok")
        or result.get("manifest_filter", {}).get("sha256") != manifest_sha256
        or int(result.get("remaining_bytes", -1)) != 0
        or int(result.get("partial_count", -1)) != 0
        or int(result.get("selected_asset_count", 0)) <= 0
    ):
        raise ValueError("Paper-test acquisition receipt is incomplete or covers another manifest")


def binary_counts(labels: np.ndarray, predictions: np.ndarray) -> dict[str, int | float]:
    tp = int(np.count_nonzero((labels == 1) & predictions))
    fp = int(np.count_nonzero((labels == 0) & predictions))
    tn = int(np.count_nonzero((labels == 0) & ~predictions))
    fn = int(np.count_nonzero((labels == 1) & ~predictions))
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "recall": tp / max(tp + fn, 1),
        "false_positive_rate": fp / max(fp + tn, 1),
        "precision": tp / max(tp + fp, 1),
    }


def pixel_summary(tp: np.ndarray, fp: np.ndarray, fn: np.ndarray) -> dict[str, int | float]:
    values = {"tp": int(tp.sum()), "fp": int(fp.sum()), "fn": int(fn.sum())}
    values["intersection_over_union"] = values["tp"] / max(
        values["tp"] + values["fp"] + values["fn"], 1
    )
    return values


def candidate_pixel_counts(
    prediction: np.ndarray,
    truth: np.ndarray,
    observable: np.ndarray,
    *,
    truth_available: bool,
) -> dict[str, int | bool]:
    """Return mutually exclusive pixel confusion counts.

    When truth is unavailable, every predicted observable pixel remains an
    adversarial false positive; the caller supplies archived truth pixels as
    false negatives. With truth available, an intersection pixel is a true
    positive and must never also be counted as a false positive.
    """
    if truth_available:
        return {
            "truth_available": True,
            "tp": int(np.count_nonzero(prediction & truth)),
            "fp": int(np.count_nonzero(prediction & observable & ~truth)),
            "fn": int(np.count_nonzero(truth & ~prediction)),
        }
    return {
        "truth_available": False,
        "tp": 0,
        "fp": int(np.count_nonzero(prediction & observable)),
        "fn": 0,
    }


def view_metrics(
    labels: np.ndarray,
    baseline_scores: np.ndarray,
    candidate_scores: np.ndarray,
    baseline_pixels: tuple[np.ndarray, np.ndarray, np.ndarray],
    candidate_pixels: tuple[np.ndarray, np.ndarray, np.ndarray],
    operational_threshold: float,
) -> dict[str, Any]:
    baseline_fixed = binary_counts(labels, baseline_scores > 0.5)
    candidate_fixed = binary_counts(labels, candidate_scores > operational_threshold)
    matched = choose_threshold_at_fpr(
        labels,
        candidate_scores,
        float(baseline_fixed["false_positive_rate"]),
    )
    baseline_pixel = pixel_summary(*baseline_pixels)
    candidate_pixel = pixel_summary(*candidate_pixels)
    baseline_ap = float(average_precision_score(labels, baseline_scores))
    candidate_ap = float(average_precision_score(labels, candidate_scores))
    return {
        "rows": int(labels.size),
        "positive": int(labels.sum()),
        "baseline": {
            "average_precision": baseline_ap,
            "fixed_operating_point": baseline_fixed,
            "pixels": baseline_pixel,
        },
        "candidate": {
            "average_precision": candidate_ap,
            "fixed_operating_point": {"threshold": operational_threshold, **candidate_fixed},
            "matched_fpr_operating_point": matched,
            "pixels": candidate_pixel,
        },
        "delta": {
            "average_precision": candidate_ap - baseline_ap,
            "fixed_recall": float(candidate_fixed["recall"] - baseline_fixed["recall"]),
            "fixed_false_positive_rate": float(
                candidate_fixed["false_positive_rate"] - baseline_fixed["false_positive_rate"]
            ),
            "matched_fpr_recall": float(matched["recall"] - baseline_fixed["recall"]),
            "matched_false_positive_rate": float(
                matched["false_positive_rate"] - baseline_fixed["false_positive_rate"]
            ),
            "pixel_iou": float(
                candidate_pixel["intersection_over_union"]
                - baseline_pixel["intersection_over_union"]
            ),
        },
    }


def score_plan(labels: np.ndarray, scores: np.ndarray, site_index: np.ndarray) -> dict[str, np.ndarray]:
    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    ends = np.flatnonzero(np.r_[sorted_scores[1:] != sorted_scores[:-1], True])
    return {
        "labels": labels[order].astype(np.int8),
        "sites": site_index[order],
        "ends": ends,
    }


def plan_cumulative(
    site_draws: np.ndarray, plan: dict[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    weights = site_draws[:, plan["sites"]]
    positives = weights * plan["labels"][None, :]
    tp = np.cumsum(positives, axis=1)[:, plan["ends"]]
    fp = np.cumsum(weights - positives, axis=1)[:, plan["ends"]]
    total_positive = tp[:, -1]
    return tp, fp, total_positive


def average_precision_from_cumulative(
    tp: np.ndarray, fp: np.ndarray, total_positive: np.ndarray
) -> np.ndarray:
    increments = np.diff(tp, axis=1, prepend=np.zeros((tp.shape[0], 1), dtype=tp.dtype))
    precision = tp / np.maximum(tp + fp, 1)
    return np.sum(increments * precision, axis=1) / np.maximum(total_positive, 1)


def interval(values: np.ndarray, confidence: float) -> dict[str, float]:
    finite = values[np.isfinite(values)]
    alpha = (1.0 - confidence) / 2.0
    return {
        "mean": float(np.mean(finite)),
        "lower": float(np.quantile(finite, alpha)),
        "upper": float(np.quantile(finite, 1.0 - alpha)),
    }


def site_sum(values: np.ndarray, site_index: np.ndarray, sites: int) -> np.ndarray:
    return np.bincount(site_index, weights=values, minlength=sites)


def bootstrap_view(
    *,
    labels: np.ndarray,
    sites: np.ndarray,
    baseline_scores: np.ndarray,
    candidate_scores: np.ndarray,
    baseline_predictions: np.ndarray,
    candidate_predictions: np.ndarray,
    baseline_pixels: tuple[np.ndarray, np.ndarray, np.ndarray],
    candidate_pixels: tuple[np.ndarray, np.ndarray, np.ndarray],
    replicates: int,
    seed: int,
    confidence: float,
    batch_size: int = 64,
) -> dict[str, Any]:
    _, site_index = np.unique(sites, return_inverse=True)
    n_sites = int(site_index.max()) + 1
    baseline_plan = score_plan(labels, baseline_scores, site_index)
    candidate_plan = score_plan(labels, candidate_scores, site_index)
    positive_site = site_sum((labels == 1).astype(float), site_index, n_sites)
    negative_site = site_sum((labels == 0).astype(float), site_index, n_sites)
    baseline_tp_site = site_sum(((labels == 1) & baseline_predictions).astype(float), site_index, n_sites)
    baseline_fp_site = site_sum(((labels == 0) & baseline_predictions).astype(float), site_index, n_sites)
    candidate_tp_site = site_sum(((labels == 1) & candidate_predictions).astype(float), site_index, n_sites)
    candidate_fp_site = site_sum(((labels == 0) & candidate_predictions).astype(float), site_index, n_sites)
    baseline_pixel_sites = [site_sum(value, site_index, n_sites) for value in baseline_pixels]
    candidate_pixel_sites = [site_sum(value, site_index, n_sites) for value in candidate_pixels]
    rng = np.random.default_rng(seed)
    deltas = {name: [] for name in (
        "average_precision", "pixel_iou", "matched_fpr_recall",
        "fixed_recall", "fixed_false_positive_rate",
    )}
    probabilities = np.full(n_sites, 1.0 / n_sites)
    for start in range(0, replicates, batch_size):
        size = min(batch_size, replicates - start)
        draws = rng.multinomial(n_sites, probabilities, size=size).astype(np.int32)
        base_tp_cum, base_fp_cum, base_pos_total = plan_cumulative(draws, baseline_plan)
        cand_tp_cum, cand_fp_cum, cand_pos_total = plan_cumulative(draws, candidate_plan)
        base_ap = average_precision_from_cumulative(base_tp_cum, base_fp_cum, base_pos_total)
        cand_ap = average_precision_from_cumulative(cand_tp_cum, cand_fp_cum, cand_pos_total)
        deltas["average_precision"].append(cand_ap - base_ap)

        positive_total = draws @ positive_site
        negative_total = draws @ negative_site
        base_tp = draws @ baseline_tp_site
        base_fp = draws @ baseline_fp_site
        cand_tp = draws @ candidate_tp_site
        cand_fp = draws @ candidate_fp_site
        base_recall = base_tp / np.maximum(positive_total, 1)
        base_fpr = base_fp / np.maximum(negative_total, 1)
        deltas["fixed_recall"].append(cand_tp / np.maximum(positive_total, 1) - base_recall)
        deltas["fixed_false_positive_rate"].append(
            cand_fp / np.maximum(negative_total, 1) - base_fpr
        )
        allowed = cand_fp_cum <= base_fpr[:, None] * negative_total[:, None] + 1e-12
        matched_tp = np.max(np.where(allowed, cand_tp_cum, -1), axis=1)
        matched_tp = np.maximum(matched_tp, 0)
        deltas["matched_fpr_recall"].append(
            matched_tp / np.maximum(positive_total, 1) - base_recall
        )

        base_pixel_values = [draws @ value for value in baseline_pixel_sites]
        cand_pixel_values = [draws @ value for value in candidate_pixel_sites]
        base_iou = base_pixel_values[0] / np.maximum(sum(base_pixel_values), 1)
        cand_iou = cand_pixel_values[0] / np.maximum(sum(cand_pixel_values), 1)
        deltas["pixel_iou"].append(cand_iou - base_iou)
    arrays = {name: np.concatenate(parts) for name, parts in deltas.items()}
    return {
        "replicates": replicates,
        "sites": n_sites,
        "confidence": confidence,
        "delta_intervals": {
            name: interval(values, confidence) for name, values in arrays.items()
        },
    }


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# ERSRR successor exact MARS-S2L paper benchmark",
        "",
        "| View | Model | AP | Recall (matched FPR) | FPR | Pixel IoU |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for name, view in report["views"].items():
        baseline = view["metrics"]["baseline"]
        candidate = view["metrics"]["candidate"]
        lines.extend(
            [
                f"| {name} | Reconstructed paper model | {baseline['average_precision']:.5f} | {baseline['fixed_operating_point']['recall']:.5f} | {baseline['fixed_operating_point']['false_positive_rate']:.5f} | {baseline['pixels']['intersection_over_union']:.5f} |",
                f"| {name} | Frozen ERSRR successor | {candidate['average_precision']:.5f} | {candidate['matched_fpr_operating_point']['recall']:.5f} | {candidate['matched_fpr_operating_point']['false_positive_rate']:.5f} | {candidate['pixels']['intersection_over_union']:.5f} |",
            ]
        )
    lines.extend(["", report["decision"]])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@torch.no_grad()
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", default=DEFAULT_SPEC.as_posix())
    parser.add_argument("--metadata-dir", default=DEFAULT_OUTPUT.as_posix())
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST.as_posix())
    parser.add_argument("--receipt", default=DEFAULT_RECEIPT.as_posix())
    parser.add_argument("--released-checkpoint", default=DEFAULT_CHECKPOINT.as_posix())
    parser.add_argument("--residual", default=DEFAULT_RESIDUAL.as_posix())
    parser.add_argument("--head", default=DEFAULT_HEAD.as_posix())
    parser.add_argument("--selection", default=DEFAULT_SELECTION.as_posix())
    parser.add_argument("--scene-fold0", default=DEFAULT_SCENE_FOLD0.as_posix())
    parser.add_argument("--mask-selection", default=DEFAULT_MASK_SELECTION.as_posix())
    parser.add_argument("--mask-confirmation", default=DEFAULT_MASK_CONFIRMATION.as_posix())
    parser.add_argument("--paper-benchmark", default=DEFAULT_PAPER_BENCHMARK.as_posix())
    parser.add_argument("--mixed-cohort", default=DEFAULT_MIXED_COHORT.as_posix())
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output-json", default=DEFAULT_JSON.as_posix())
    parser.add_argument("--output-markdown", default=DEFAULT_MARKDOWN.as_posix())
    args = parser.parse_args()
    root = repo_root()
    spec_path = (root / args.spec).resolve()
    spec = load_json(spec_path)
    expected = spec["expected"]
    paths = {
        "residual": (root / args.residual).resolve(),
        "scene_head": (root / args.head).resolve(),
        "scene_selection": (root / args.selection).resolve(),
        "scene_fold0_result": (root / args.scene_fold0).resolve(),
        "mask_selection": (root / args.mask_selection).resolve(),
        "mask_confirmation": (root / args.mask_confirmation).resolve(),
        "paper_benchmark": (root / args.paper_benchmark).resolve(),
        "mixed_cohort": (root / args.mixed_cohort).resolve(),
    }
    for name, path in paths.items():
        if sha256(path) != expected[f"{name}_sha256"]:
            raise ValueError(f"Frozen {name} hash mismatch")
    manifest_path = (root / args.manifest).resolve()
    records = load_sealed_records(manifest_path, expected["sealed_manifest_sha256"])
    verify_receipt((root / args.receipt).resolve(), expected["sealed_manifest_sha256"])

    comparator_rows, comparator_audit = reconstruct(
        (root / PAPER_METADATA).resolve(),
        (root / DEFAULT_ONSHORE).resolve(),
        (root / DEFAULT_OFFSHORE).resolve(),
        (root / PAPER_CONFIG).resolve(),
        (root / DEFAULT_ASSET_METADATA).resolve(),
    )
    record_by_id = {str(record["sample_id"]): record for record in records}
    comparator_ids = {str(row["id_loc_image"]) for row in comparator_rows}
    missing_ids = comparator_ids - set(record_by_id)
    if len(missing_ids) != 5 or set(record_by_id) - comparator_ids:
        raise ValueError("Available/missing paper-test scene identity mismatch")

    residual_artifact = torch.load(paths["residual"], map_location="cpu", weights_only=True)
    if int(residual_artifact["fold"]) != 0 or int(residual_artifact["epoch"]) != 7:
        raise ValueError("Frozen scene residual is not fold 0 epoch 7")
    head_payload = joblib.load(paths["scene_head"])
    selection = load_json(paths["scene_selection"])
    architecture = spec["architecture"]
    if float(selection["selected"]["blend_lambda"]) != architecture["scene_head_blend"]:
        raise ValueError("Scene blend differs from frozen selection")

    dataset = MarsPaperDataset(
        (root / args.metadata_dir).resolve(),
        records,
        augment=False,
        seed=0,
        allow_missing_positive_mask=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_residual_model(
        (root / args.released_checkpoint).resolve(), residual_artifact, device
    )
    feature_rows: list[np.ndarray] = []
    available_ids: list[str] = []
    available_groups: list[str] = []
    pixel_outputs: dict[str, dict[str, int | bool]] = {}
    for batch in loader:
        batch = move_batch(batch, device)
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            output = model(batch["inputs"], batch["observable"], batch["sensor_index"])
        primary_logits = trust_region_logits(
            output["baseline_logits"],
            output["segmentation_logits"],
            architecture["scene_residual_alpha"],
        )
        pooled = pooled_scene_features(
            batch["inputs"], primary_logits, output["baseline_logits"], batch["clear"], batch["observable"]
        ).cpu().numpy()
        primary_probability = torch.sigmoid(primary_logits).float().masked_fill(
            batch["clear"] <= 0.5, 0.0
        ).cpu().numpy()
        released_probability = torch.sigmoid(output["baseline_logits"]).float().masked_fill(
            batch["clear"] <= 0.5, 0.0
        ).cpu().numpy()
        for index in range(primary_probability.shape[0]):
            sample_id = str(batch["sample_id"][index])
            primary_score = float(connected_scene_score(primary_probability[index, 0]))
            released_score = float(connected_scene_score(released_probability[index, 0]))
            feature_rows.append(
                np.concatenate(
                    (np.asarray([primary_score, released_score], dtype=np.float32), pooled[index])
                ).astype(np.float32)
            )
            available_ids.append(sample_id)
            available_groups.append(str(batch["group_id"][index]))
            observable = batch["observable"][index, 0].cpu().numpy() > 0.5
            truth = (batch["mask"][index, 0].cpu().numpy() > 0.5) & observable
            prediction = component_mask_at(
                released_probability[index, 0],
                architecture["mask_probability_threshold"],
                architecture["mask_minimum_connected_pixels"],
            )
            truth_available = bool(batch["pixel_truth_available"][index].item())
            pixel_outputs[sample_id] = candidate_pixel_counts(
                prediction,
                truth,
                observable,
                truth_available=truth_available,
            )

    base_names = np.asarray(["primary_connected_score", "released_connected_score", *tensor_feature_names()])
    base_features = np.stack(feature_rows).astype(np.float64)
    context_features, augmented_names = augment_site_context(
        base_features, base_names, np.asarray(available_groups)
    )
    if base_names.tolist() != head_payload["feature_names"] or augmented_names != head_payload["augmented_feature_names"]:
        raise ValueError("Paper-test scene feature schema differs from the frozen head")
    head_probability = predict_model(head_payload["fitted"], context_features)
    available_scores = blend_scores(
        base_features[:, 0], head_probability, architecture["scene_head_blend"]
    )
    score_by_id = dict(zip(available_ids, available_scores, strict=True))

    labels: list[int] = []
    sites: list[str] = []
    test_only: list[bool] = []
    baseline_scores: list[float] = []
    candidate_scores: list[float] = []
    baseline_tp: list[int] = []
    baseline_fp: list[int] = []
    baseline_fn: list[int] = []
    candidate_tp: list[int] = []
    candidate_fp: list[int] = []
    candidate_fn: list[int] = []
    for row in comparator_rows:
        sample_id = str(row["id_loc_image"])
        label = int(row["target"])
        truth_pixels = int(float(row["TP"]) + float(row["FN"]))
        labels.append(label)
        sites.append(str(row["location_name"]).strip())
        test_only.append(bool(row["test_only_site"]))
        baseline_scores.append(float(row["scene_pred"]))
        baseline_tp.append(int(float(row["TP"])))
        baseline_fp.append(int(float(row["FP"])))
        baseline_fn.append(int(float(row["FN"])))
        if sample_id in score_by_id:
            record = record_by_id[sample_id]
            if int(record["label_state"] == "PLUME") != label:
                raise ValueError(f"Paper label mismatch for {sample_id}")
            candidate_scores.append(float(score_by_id[sample_id]))
            pixels = pixel_outputs[sample_id]
            if pixels["truth_available"]:
                candidate_tp.append(int(pixels["tp"]))
                candidate_fp.append(int(pixels["fp"]))
                candidate_fn.append(int(pixels["fn"]))
            else:
                candidate_tp.append(0)
                candidate_fp.append(int(pixels["fp"]))
                candidate_fn.append(truth_pixels)
        else:
            candidate_scores.append(0.0 if label else 1.0)
            candidate_tp.append(0)
            candidate_fp.append(int(spec["missing_data_policy"]["missing_raster_grid_pixels"]))
            candidate_fn.append(truth_pixels)

    arrays = {
        "labels": np.asarray(labels, dtype=np.uint8),
        "sites": np.asarray(sites),
        "test_only": np.asarray(test_only, dtype=bool),
        "baseline_scores": np.asarray(baseline_scores, dtype=np.float64),
        "candidate_scores": np.asarray(candidate_scores, dtype=np.float64),
        "baseline_tp": np.asarray(baseline_tp, dtype=np.int64),
        "baseline_fp": np.asarray(baseline_fp, dtype=np.int64),
        "baseline_fn": np.asarray(baseline_fn, dtype=np.int64),
        "candidate_tp": np.asarray(candidate_tp, dtype=np.int64),
        "candidate_fp": np.asarray(candidate_fp, dtype=np.int64),
        "candidate_fn": np.asarray(candidate_fn, dtype=np.int64),
    }
    views: dict[str, Any] = {}
    bootstrap_config = spec["bootstrap"]
    selections = {"full": np.ones(arrays["labels"].size, dtype=bool), "test_only_sites": arrays["test_only"]}
    for view_index, (name, selected) in enumerate(selections.items()):
        labels_view = arrays["labels"][selected]
        baseline_scores_view = arrays["baseline_scores"][selected]
        candidate_scores_view = arrays["candidate_scores"][selected]
        baseline_pixels_view = tuple(arrays[key][selected] for key in ("baseline_tp", "baseline_fp", "baseline_fn"))
        candidate_pixels_view = tuple(arrays[key][selected] for key in ("candidate_tp", "candidate_fp", "candidate_fn"))
        metrics = view_metrics(
            labels_view,
            baseline_scores_view,
            candidate_scores_view,
            baseline_pixels_view,
            candidate_pixels_view,
            architecture["operational_scene_threshold"],
        )
        bootstrap = bootstrap_view(
            labels=labels_view,
            sites=arrays["sites"][selected],
            baseline_scores=baseline_scores_view,
            candidate_scores=candidate_scores_view,
            baseline_predictions=baseline_scores_view > 0.5,
            candidate_predictions=candidate_scores_view > architecture["operational_scene_threshold"],
            baseline_pixels=baseline_pixels_view,
            candidate_pixels=candidate_pixels_view,
            replicates=int(bootstrap_config["replicates"]),
            seed=int(bootstrap_config["seed"]) + view_index,
            confidence=float(bootstrap_config["confidence"]),
        )
        intervals = bootstrap["delta_intervals"]
        published = PUBLISHED["full" if name == "full" else "test_only_sites"]
        checks = {
            "ap_point_beats_reconstructed": metrics["delta"]["average_precision"] > 0,
            "ap_point_beats_published": metrics["candidate"]["average_precision"] > published["average_precision"],
            "ap_lower_95_above_zero": intervals["average_precision"]["lower"] > 0,
            "iou_point_beats_reconstructed": metrics["delta"]["pixel_iou"] > 0,
            "iou_lower_95_above_zero": intervals["pixel_iou"]["lower"] > 0,
            "matched_recall_point_higher": metrics["delta"]["matched_fpr_recall"] > 0,
            "matched_recall_lower_95_above_zero": intervals["matched_fpr_recall"]["lower"] > 0,
            "matched_fpr_no_worse": metrics["delta"]["matched_false_positive_rate"] <= 0,
            "operational_fpr_upper_95_no_worse": intervals["fixed_false_positive_rate"]["upper"] <= 0,
        }
        if "pixel_iou" in published:
            checks["iou_point_beats_published"] = metrics["candidate"]["pixels"]["intersection_over_union"] > published["pixel_iou"]
        views[name] = {"metrics": metrics, "bootstrap": bootstrap, "published": published, "checks": checks}

    passed = all(all(view["checks"].values()) for view in views.values())
    report = {
        "schema_version": 1,
        "scope": "deterministic metric correction of the frozen official MARS-S2L paper-v3 test evaluation",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "architecture": architecture,
        "views": views,
        "comparator_identity": comparator_audit,
        "missing_scene_ids": sorted(missing_ids),
        "decision": (
            "Frozen ERSRR successor passes every preregistered paper-superiority gate."
            if passed
            else "Frozen ERSRR successor does not pass every preregistered paper-superiority gate."
        ),
        "passed": passed,
        "metric_correction": {
            "status": "post_result_evaluator_bug_correction",
            "superseded_result_sha256": "589210e313fd1c6e93daf83e22db2582223ad065162e98cb93be5627f3934119",
            "architecture_changed": False,
            "predictions_changed": False,
            "reason": "The frozen evaluator counted prediction & observable as FP even when those pixels were already TP. Correct FP is prediction & observable & ~truth when pixel truth is available.",
        },
        "provenance": {
            "spec_sha256": sha256(spec_path),
            "manifest_sha256": sha256(manifest_path),
            "receipt_sha256": sha256((root / args.receipt).resolve()),
            **{f"{name}_sha256": sha256(path) for name, path in paths.items()},
            "script_sha256": sha256(Path(__file__).resolve()),
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
            "device": str(torch.cuda.get_device_name(device) if device.type == "cuda" else device),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
            "rasterio": rasterio.__version__,
            "joblib": joblib.__version__,
        },
    }
    write_json((root / args.output_json).resolve(), report)
    write_markdown((root / args.output_markdown).resolve(), report)
    print(json.dumps({"ok": passed, "decision": report["decision"], "checks": {name: value["checks"] for name, value in views.items()}}, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
