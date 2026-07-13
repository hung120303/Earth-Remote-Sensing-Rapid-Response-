#!/usr/bin/env python3
"""Diagnose the frozen EMIT confirmation without tuning a model or decision rule."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.stats import ks_2samp, rankdata, spearmanr

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "EarthRemoteSensingRapidResponse"
if str(MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_ROOT))

from external_emit_adapter import load_external_scene  # noqa: E402
from mars_s2l_adapter import MARS_BANDS, iter_manifest, load_sample  # noqa: E402

from acquire_mars_metadata import DEFAULT_OUTPUT, repo_root, sha256  # noqa: E402
from build_mars_v3_strict_cohort import V3_STRICT_SAMPLES  # noqa: E402
from evaluate_emit_v002_external import (  # noqa: E402
    DEFAULT_RAW_ROOT,
    DEFAULT_SEAL,
    DEFAULT_WIND,
    FIXED_SEEDS,
)

DEFAULT_CONFIRMATION = Path("reports/experiments/emit_v002_external_confirmation.json")
DEFAULT_CAMPAIGN = Path("reports/experiments/mars_v3_strict_campaign.json")
DEFAULT_JSON = Path("reports/experiments/emit_v002_external_posthoc_diagnostic.json")
DEFAULT_MARKDOWN = Path("reports/experiments/EMIT_V002_EXTERNAL_POSTHOC_DIAGNOSTIC.md")
REFERENCE_TIME = re.compile(r"_(\d{8}T\d{6})_")
FROZEN_CONFIRMATION_SCOPE = "once_only_emit_v002_external_positive_confirmation"


def safe_path(root: Path, value: str | Path) -> Path:
    result = (root / value).resolve()
    if result != root and root not in result.parents:
        raise ValueError("Path must resolve beneath the repository root")
    return result


def tracked_dirty(root: Path) -> bool:
    status = subprocess.check_output(
        [
            "git",
            "-c",
            "core.autocrlf=true",
            "status",
            "--porcelain",
            "--untracked-files=no",
        ],
        cwd=root,
        text=True,
    )
    return bool(status.strip())


def finite_summary(values: Sequence[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        raise ValueError("Cannot summarize an empty finite array")
    return {
        "count": int(array.size),
        "minimum": float(np.min(array)),
        "q1": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "q3": float(np.quantile(array, 0.75)),
        "maximum": float(np.max(array)),
        "mean": float(np.mean(array)),
    }


def binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    truth = np.asarray(labels, dtype=bool).ravel()
    values = np.asarray(scores, dtype=np.float64).ravel()
    if truth.shape != values.shape or not np.all(np.isfinite(values)):
        raise ValueError("AUC labels and scores must be aligned and finite")
    positives = int(np.count_nonzero(truth))
    negatives = int(truth.size - positives)
    if positives == 0 or negatives == 0:
        raise ValueError("AUC requires both classes")
    ranks = rankdata(values, method="average")
    rank_sum = float(np.sum(ranks[truth]))
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def mask_signal_features(
    mbmp: np.ndarray, observable: np.ndarray, plume_mask: np.ndarray
) -> dict[str, float | int]:
    values = np.asarray(mbmp, dtype=np.float64)
    valid = np.asarray(observable, dtype=bool)
    plume = np.asarray(plume_mask, dtype=bool) & valid
    background = (~np.asarray(plume_mask, dtype=bool)) & valid
    if values.shape != valid.shape or values.shape != plume.shape:
        raise ValueError("MBMP, observable, and plume masks must be aligned")
    if not np.any(plume) or not np.any(background):
        raise ValueError("Signal diagnostics require observable plume and background pixels")
    inside = values[plume]
    outside = values[background]
    signed = float(np.median(inside) - np.median(outside))
    background_mad = float(np.median(np.abs(outside - np.median(outside))))
    robust_scale = 1.4826 * background_mad
    signed_robust = 0.0 if robust_scale == 0.0 else signed / robust_scale
    labels = np.concatenate(
        [np.ones(inside.size, dtype=np.uint8), np.zeros(outside.size, dtype=np.uint8)]
    )
    auc = binary_auc(labels, np.concatenate([inside, outside]))
    observable_values = values[valid]
    return {
        "observable_pixels": int(np.count_nonzero(valid)),
        "observable_plume_pixels": int(inside.size),
        "observable_background_pixels": int(outside.size),
        "mbmp_scene_p01": float(np.quantile(observable_values, 0.01)),
        "mbmp_scene_p05": float(np.quantile(observable_values, 0.05)),
        "mbmp_scene_median": float(np.median(observable_values)),
        "mbmp_scene_p95": float(np.quantile(observable_values, 0.95)),
        "mbmp_scene_p99": float(np.quantile(observable_values, 0.99)),
        "mbmp_plume_median": float(np.median(inside)),
        "mbmp_background_median": float(np.median(outside)),
        "mbmp_mask_signed_median_contrast": signed,
        "mbmp_mask_signed_robust_contrast": float(signed_robust),
        "mbmp_mask_absolute_robust_contrast": float(abs(signed_robust)),
        "mbmp_mask_auc": float(auc),
        "mbmp_mask_direction_free_auc": float(max(auc, 1.0 - auc)),
    }


def reflectance_features(
    target: np.ndarray, reference: np.ndarray, observable: np.ndarray
) -> dict[str, float]:
    valid = np.asarray(observable, dtype=bool)
    if target.shape != reference.shape or target.shape[0] != len(MARS_BANDS):
        raise ValueError("Expected matching six-band target/reference reflectance")
    if target.shape[1:] != valid.shape or not np.any(valid):
        raise ValueError("Reflectance and observable mask must be aligned and nonempty")
    result: dict[str, float] = {}
    for prefix, image in (("target", target), ("reference", reference)):
        for index, band in enumerate(MARS_BANDS):
            values = np.asarray(image[index][valid], dtype=np.float64)
            result[f"{prefix}_{band}_median"] = float(np.median(values))
            result[f"{prefix}_{band}_p95"] = float(np.quantile(values, 0.95))
    return result


def reference_interval_days(record: dict[str, Any]) -> float:
    target = datetime.fromisoformat(str(record["target_datetime"]))
    match = REFERENCE_TIME.search(str(record["reference_scene_id"]))
    if match is None:
        raise ValueError(f"Cannot parse reference time for {record['sample_id']}")
    reference = datetime.strptime(match.group(1), "%Y%m%dT%H%M%S").replace(
        tzinfo=target.tzinfo
    )
    return abs((target - reference).total_seconds()) / 86_400.0


def rate_summary(rows: list[dict[str, Any]], indices: np.ndarray) -> dict[str, Any]:
    if indices.size == 0:
        raise ValueError("Rate summary requires at least one row")
    released = np.asarray([rows[index]["released_prediction"] for index in indices])
    candidate = np.asarray([rows[index]["seed_predictions"] for index in indices]).T
    per_seed = np.mean(candidate, axis=1)
    return {
        "samples": int(indices.size),
        "released_mars_s2l_recall": float(np.mean(released)),
        "ersrr_seed_mean_recall": float(np.mean(per_seed)),
        "ersrr_seed_standard_deviation": float(np.std(per_seed)),
        "ersrr_per_seed_recall": [float(value) for value in per_seed],
    }


def rank_strata_indices(values: Sequence[float], names: Sequence[str]) -> list[np.ndarray]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size < len(names) or not np.all(np.isfinite(array)):
        raise ValueError("Rank strata require enough finite observations")
    order = np.lexsort((np.arange(array.size), array))
    return [np.asarray(part, dtype=np.int64) for part in np.array_split(order, len(names))]


def ranked_strata(
    rows: list[dict[str, Any]], field: str, names: Sequence[str]
) -> list[dict[str, Any]]:
    values = [float(row[field]) for row in rows]
    result: list[dict[str, Any]] = []
    for name, indices in zip(names, rank_strata_indices(values, names)):
        selected = np.asarray([values[index] for index in indices])
        result.append(
            {
                "name": name,
                "field": field,
                "minimum": float(np.min(selected)),
                "maximum": float(np.max(selected)),
                **rate_summary(rows, indices),
            }
        )
    return result


def time_offset_strata(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    bins = (
        ("at_most_1_hour", lambda value: value <= 1.0),
        ("over_1_to_2_hours", lambda value: 1.0 < value <= 2.0),
        ("over_2_hours", lambda value: value > 2.0),
    )
    result = []
    for name, predicate in bins:
        indices = np.asarray(
            [index for index, row in enumerate(rows) if predicate(row["absolute_time_offset_hours"])],
            dtype=np.int64,
        )
        if indices.size:
            values = np.asarray([rows[index]["absolute_time_offset_hours"] for index in indices])
            result.append(
                {
                    "name": name,
                    "field": "absolute_time_offset_hours",
                    "minimum": float(np.min(values)),
                    "maximum": float(np.max(values)),
                    **rate_summary(rows, indices),
                }
            )
    return result


def distribution_comparison(
    external_rows: list[dict[str, Any]], strict_rows: list[dict[str, Any]], field: str
) -> dict[str, Any]:
    left = np.asarray([row[field] for row in external_rows], dtype=np.float64)
    right = np.asarray([row[field] for row in strict_rows], dtype=np.float64)
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        raise ValueError(f"Non-finite distribution feature: {field}")
    greater = int(np.count_nonzero(left[:, None] > right[None, :]))
    less = int(np.count_nonzero(left[:, None] < right[None, :]))
    ks = ks_2samp(left, right, alternative="two-sided", method="auto")
    return {
        "feature": field,
        "external_emit": finite_summary(left),
        "mars_strict_positive": finite_summary(right),
        "external_minus_mars_median": float(np.median(left) - np.median(right)),
        "cliffs_delta_external_vs_mars": float((greater - less) / (left.size * right.size)),
        "ks_statistic": float(ks.statistic),
        "ks_pvalue_exploratory_unadjusted": float(ks.pvalue),
    }


def score_correlations(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    predictors = (
        "absolute_time_offset_hours",
        "reference_interval_days",
        "wind_speed_m_s",
        "plume_pixels",
        "mbmp_mask_direction_free_auc",
        "mbmp_mask_absolute_robust_contrast",
    )
    outcomes = ("released_score", "ersrr_mean_score")
    result = []
    for outcome in outcomes:
        y = np.asarray([row[outcome] for row in rows], dtype=np.float64)
        for predictor in predictors:
            x = np.asarray([row[predictor] for row in rows], dtype=np.float64)
            statistic = spearmanr(x, y)
            result.append(
                {
                    "outcome": outcome,
                    "predictor": predictor,
                    "spearman_rho": float(statistic.statistic),
                    "pvalue_exploratory_unadjusted": float(statistic.pvalue),
                    "samples": int(x.size),
                }
            )
    return result


def external_rows(
    root: Path,
    raw_root: Path,
    seal: dict[str, Any],
    wind: dict[str, Any],
    confirmation: dict[str, Any],
) -> list[dict[str, Any]]:
    wind_by_group = {item["group_id"]: item for item in wind["records"]}
    released = {item["group_id"]: item for item in confirmation["released_mars_s2l"]["records"]}
    seeds = [int(item["seed"]) for item in confirmation["seed_results"]]
    if tuple(seeds) != FIXED_SEEDS:
        raise ValueError("External confirmation does not contain the fixed seeds in order")
    per_seed = [
        {record["group_id"]: record for record in item["records"]}
        for item in confirmation["seed_results"]
    ]
    rows = []
    for seal_record in seal["records"]:
        if not seal_record["final_gate_pass"]:
            continue
        group = str(seal_record["group_id"])
        scene_dir = raw_root / seal_record["granule_id"]
        manifest_path = scene_dir / "manifest.json"
        if sha256(manifest_path) != seal_record["crop_manifest_sha256"]:
            raise ValueError(f"Crop manifest identity mismatch for {group}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        scene = load_external_scene(
            root,
            manifest_path,
            scene_dir / "cloudsen12.manifest.json",
            wind_by_group[group],
        )
        if scene.group_id != group:
            raise ValueError(f"Loaded scene identity mismatch for {group}")
        signal = mask_signal_features(scene.inputs[0], scene.observable, scene.plume_mask)
        reflectance = reflectance_features(scene.inputs[1:7], scene.inputs[7:13], scene.observable)
        seed_records = [item[group] for item in per_seed]
        predictions = [int(bool(item["prediction"])) for item in seed_records]
        scores = [float(item["score"]) for item in seed_records]
        rows.append(
            {
                "group_id": group,
                "granule_id": scene.granule_id,
                "target_scene_id": scene.target_scene_id,
                "reference_scene_id": manifest["reference_scene_id"],
                "signed_time_offset_hours": float(manifest["emit_to_target_offset_hours"]),
                "absolute_time_offset_hours": abs(float(manifest["emit_to_target_offset_hours"])),
                "reference_interval_days": float(manifest["reference_to_target_gap_hours"]) / 24.0,
                "plume_pixels": int(np.count_nonzero(scene.plume_mask)),
                "observable_fraction": float(np.mean(scene.observable)),
                "observable_fraction_on_plume": float(np.mean(scene.observable[scene.plume_mask])),
                "wind_speed_m_s": float(np.hypot(scene.wind_u_m_s, scene.wind_v_m_s)),
                "released_prediction": int(bool(released[group]["prediction"])),
                "released_score": float(released[group]["score"]),
                "seed_predictions": predictions,
                "seed_scores": scores,
                "seed_hits": int(sum(predictions)),
                "ersrr_mean_score": float(np.mean(scores)),
                "ersrr_mean_neural_score": float(
                    np.mean([float(item["neural_score"]) for item in seed_records])
                ),
                "ersrr_mean_proposal_score": float(
                    np.mean([float(item["proposal_score"]) for item in seed_records])
                ),
                **signal,
                **reflectance,
            }
        )
    rows.sort(key=lambda item: item["group_id"])
    return rows


def strict_positive_rows(metadata_dir: Path, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        if str(record["label_state"]).upper() != "PLUME":
            continue
        sample = load_sample(metadata_dir, record, require_enhancement=False)
        signal = mask_signal_features(
            sample.mbmp_release_compatible, sample.observable_mask, sample.plume_mask
        )
        reflectance = reflectance_features(sample.target, sample.reference, sample.observable_mask)
        rows.append(
            {
                "sample_id": sample.sample_id,
                "group_id": str(record["group_id"]),
                "reference_interval_days": reference_interval_days(record),
                "plume_pixels": int(np.count_nonzero(sample.plume_mask)),
                "observable_fraction": float(np.mean(sample.observable_mask)),
                "observable_fraction_on_plume": float(
                    np.mean(sample.observable_mask[sample.plume_mask])
                ),
                **signal,
                **reflectance,
            }
        )
    rows.sort(key=lambda item: item["sample_id"])
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def markdown_strata(lines: list[str], title: str, items: list[dict[str, Any]]) -> None:
    lines.extend(
        [
            f"## {title}",
            "",
            "| Stratum | Range | n | Released MARS recall | ERSRR seed-mean recall | Seed SD |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for item in items:
        lines.append(
            f"| {item['name']} | {item['minimum']:.3f}-{item['maximum']:.3f} | "
            f"{item['samples']} | {item['released_mars_s2l_recall']:.3f} | "
            f"{item['ersrr_seed_mean_recall']:.3f} | "
            f"{item['ersrr_seed_standard_deviation']:.3f} |"
        )
    lines.append("")


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    external_signal = report["distribution_shift"]["headline"]["mbmp_mask_direction_free_auc"]
    external_contrast = report["distribution_shift"]["headline"][
        "mbmp_mask_absolute_robust_contrast"
    ]
    near = report["external_strata"]["time_offset"][0]
    agreement = report["agreement"]
    lines = [
        "# EMIT V002 external post-hoc diagnostic",
        "",
        "Frozen exploratory diagnosis only. This analysis does not alter the cohort, model, threshold, or once-only primary result, and it must not be used to retune v3.",
        "",
        f"- External cohort: {report['cohorts']['external_emit']['samples']} positive scenes / independent groups",
        f"- MARS comparator: {report['cohorts']['mars_strict_positive']['samples']} frozen strict positive scenes",
        f"- EMIT/Sentinel-2 absolute offset: median {report['external_summaries']['absolute_time_offset_hours']['median']:.3f} h; {near['samples']} scenes at most one hour",
        f"- At most one hour: released MARS recall {near['released_mars_s2l_recall']:.3f}; ERSRR seed-mean recall {near['ersrr_seed_mean_recall']:.3f}",
        f"- Direction-free MBMP mask AUC median: external {external_signal['external_emit']['median']:.3f} vs strict MARS positives {external_signal['mars_strict_positive']['median']:.3f}",
        f"- Absolute robust MBMP mask contrast median: external {external_contrast['external_emit']['median']:.3f} vs strict MARS positives {external_contrast['mars_strict_positive']['median']:.3f}",
        f"- External scenes missed by both released MARS and every ERSRR seed: {agreement['neither_released_nor_any_ersrr']}",
        "",
    ]
    markdown_strata(lines, "Recall by fixed EMIT/Sentinel-2 offset", report["external_strata"]["time_offset"])
    markdown_strata(lines, "Recall by EMIT footprint area", report["external_strata"]["plume_area"])
    markdown_strata(lines, "Recall by mask-aligned MBMP separability", report["external_strata"]["mbmp_separability"])
    lines.extend(
        [
            "## Interpretation boundary",
            "",
            "The EMIT mask is a cross-sensor, time-offset footprint, not simultaneous Sentinel-2 methane truth. Mask-aligned MBMP statistics test whether a colocated Sentinel-2 spectral signal is present; they do not prove methane causality. The MARS strict cohort remains the primary same-distribution plume/no-plume benchmark.",
            "",
            "The JSON report contains all 55 external rows, all 67 strict-positive diagnostic rows, reflectance distribution comparisons, rank strata, score correlations, source hashes, and the exact frozen hit identities.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirmation", default=DEFAULT_CONFIRMATION.as_posix())
    parser.add_argument("--seal", default=DEFAULT_SEAL.as_posix())
    parser.add_argument("--wind", default=DEFAULT_WIND.as_posix())
    parser.add_argument("--raw-root", default=DEFAULT_RAW_ROOT.as_posix())
    parser.add_argument("--metadata-dir", default=DEFAULT_OUTPUT.as_posix())
    parser.add_argument("--campaign", default=DEFAULT_CAMPAIGN.as_posix())
    parser.add_argument("--output-json", default=DEFAULT_JSON.as_posix())
    parser.add_argument("--output-markdown", default=DEFAULT_MARKDOWN.as_posix())
    args = parser.parse_args()

    root = repo_root()
    confirmation_path = safe_path(root, args.confirmation)
    seal_path = safe_path(root, args.seal)
    wind_path = safe_path(root, args.wind)
    raw_root = safe_path(root, args.raw_root)
    metadata_dir = safe_path(root, args.metadata_dir)
    campaign_path = safe_path(root, args.campaign)
    output_json = safe_path(root, args.output_json)
    output_markdown = safe_path(root, args.output_markdown)
    confirmation = json.loads(confirmation_path.read_text(encoding="utf-8"))
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    wind = json.loads(wind_path.read_text(encoding="utf-8"))
    campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
    if confirmation.get("scope") != FROZEN_CONFIRMATION_SCOPE:
        raise ValueError("Expected the frozen once-only external confirmation")
    if campaign.get("scope") != "frozen_v3_five_seed_full_strict_campaign":
        raise ValueError("Expected the frozen strict MARS campaign")
    if not wind["summary"]["complete"]:
        raise ValueError("Expected complete frozen ERA5-Land acquisition")
    if sha256(seal_path) != confirmation["cohort"]["seal_sha256"]:
        raise ValueError("External seal identity differs from the confirmation")
    if sha256(wind_path) != confirmation["cohort"]["wind_acquisition_sha256"]:
        raise ValueError("External wind identity differs from the confirmation")

    ext_rows = external_rows(root, raw_root, seal, wind, confirmation)
    manifest_path = metadata_dir / V3_STRICT_SAMPLES
    if sha256(manifest_path) != campaign["cohort"]["strict_manifest_sha256"]:
        raise ValueError("Strict MARS manifest differs from the frozen campaign")
    mars_rows = strict_positive_rows(metadata_dir, list(iter_manifest(manifest_path)))
    if len(ext_rows) != int(confirmation["cohort"]["scenes"]):
        raise ValueError("External diagnostic row count differs from the confirmation")
    if len(mars_rows) != int(campaign["cohort"]["positives"]):
        raise ValueError("Strict-positive diagnostic row count differs from the campaign")

    headline_features = (
        "mbmp_mask_direction_free_auc",
        "mbmp_mask_absolute_robust_contrast",
        "mbmp_mask_signed_median_contrast",
        "mbmp_scene_p05",
        "mbmp_scene_median",
        "mbmp_scene_p95",
        "plume_pixels",
        "observable_fraction_on_plume",
        "reference_interval_days",
    )
    reflectance_feature_names = tuple(
        f"{prefix}_{band}_{statistic}"
        for prefix in ("target", "reference")
        for band in MARS_BANDS
        for statistic in ("median", "p95")
    )
    headline = {
        field: distribution_comparison(ext_rows, mars_rows, field)
        for field in headline_features
    }
    reflectance_shift = [
        distribution_comparison(ext_rows, mars_rows, field)
        for field in reflectance_feature_names
    ]

    hit_counts = Counter(int(row["seed_hits"]) for row in ext_rows)
    report: dict[str, Any] = {
        "schema_version": 1,
        "scope": "frozen_emit_v002_external_posthoc_diagnostic_no_tuning",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract": {
            "primary_result_changed": False,
            "cohort_changed": False,
            "model_or_threshold_selected": False,
            "v3_retuning_permitted": False,
            "purpose": "separate detector/domain failure from cross-sensor timing and Sentinel-2 signal-support limitations",
            "time_offset_bins_hours": ["<=1", "(1,2]", ">2"],
            "distribution_tests": "descriptive effect sizes; KS p-values are exploratory, unadjusted, and not confirmatory",
        },
        "cohorts": {
            "external_emit": {"samples": len(ext_rows), "groups": len(ext_rows), "label_scope": "positive-only cross-sensor confirmation"},
            "mars_strict_positive": {"samples": len(mars_rows), "source": "67 positives from the frozen strict-spatial MARS-S2L cohort"},
        },
        "external_summaries": {
            field: finite_summary([row[field] for row in ext_rows])
            for field in (
                "absolute_time_offset_hours",
                "reference_interval_days",
                "wind_speed_m_s",
                "plume_pixels",
                "observable_fraction",
                "observable_fraction_on_plume",
                "mbmp_mask_direction_free_auc",
                "mbmp_mask_absolute_robust_contrast",
            )
        },
        "external_strata": {
            "time_offset": time_offset_strata(ext_rows),
            "plume_area": ranked_strata(ext_rows, "plume_pixels", ("small", "medium", "large")),
            "wind_speed": ranked_strata(ext_rows, "wind_speed_m_s", ("low", "middle", "high")),
            "reference_interval": ranked_strata(ext_rows, "reference_interval_days", ("short", "middle", "long")),
            "mbmp_separability": ranked_strata(ext_rows, "mbmp_mask_direction_free_auc", ("low", "middle", "high")),
        },
        "agreement": {
            "seed_hit_counts": {str(value): int(hit_counts[value]) for value in range(6)},
            "any_ersrr_hit_group_ids": [row["group_id"] for row in ext_rows if row["seed_hits"] > 0],
            "released_hit_group_ids": [row["group_id"] for row in ext_rows if row["released_prediction"]],
            "both_released_and_any_ersrr": int(sum(row["released_prediction"] and row["seed_hits"] > 0 for row in ext_rows)),
            "neither_released_nor_any_ersrr": int(sum(not row["released_prediction"] and row["seed_hits"] == 0 for row in ext_rows)),
        },
        "score_correlations": score_correlations(ext_rows),
        "distribution_shift": {"headline": headline, "reflectance": reflectance_shift},
        "external_rows": ext_rows,
        "mars_strict_positive_rows": mars_rows,
        "limitations": [
            "The cohort has no negative scenes, so it cannot estimate false-positive rate, specificity, precision, average precision, or AUROC.",
            "EMIT detections and Sentinel-2 acquisitions differ by up to six hours; plume existence, position, and strength can change within that interval.",
            "An EMIT footprint is not simultaneous Sentinel-2 methane truth; mask-aligned MBMP statistics are descriptive signal-support proxies.",
            "The external and MARS plume masks have different producer semantics and footprint geometry, so pixel metrics and plume area are not directly interchangeable.",
            "All p-values are exploratory and unadjusted; no hypothesis or architecture is selected from significance thresholds.",
        ],
        "provenance": {
            "script": "tools/analyze_emit_v002_external_posthoc.py",
            "script_sha256": sha256(Path(__file__).resolve()),
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
            "git_tracked_worktree_dirty_at_start": tracked_dirty(root),
            "inputs": {
                "external_confirmation": {"path": confirmation_path.relative_to(root).as_posix(), "sha256": sha256(confirmation_path)},
                "external_seal": {"path": seal_path.relative_to(root).as_posix(), "sha256": sha256(seal_path)},
                "wind_acquisition": {"path": wind_path.relative_to(root).as_posix(), "sha256": sha256(wind_path)},
                "strict_campaign": {"path": campaign_path.relative_to(root).as_posix(), "sha256": sha256(campaign_path)},
                "strict_manifest": {"path": manifest_path.relative_to(root).as_posix(), "sha256": sha256(manifest_path)},
            },
        },
    }
    write_json(output_json, report)
    write_markdown(output_markdown, report)
    print(f"Wrote {output_json.relative_to(root)}")
    print(f"Wrote {output_markdown.relative_to(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
