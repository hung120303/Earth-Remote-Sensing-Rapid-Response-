#!/usr/bin/env python3
"""Select and confirm scene-gated dense masks on MARS development folds only."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "EarthRemoteSensingRapidResponse"
for path in (MODEL_ROOT, ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from acquire_mars_metadata import repo_root, sha256  # noqa: E402
SENSOR_NAMES = ("Sentinel-2", "Landsat")

DEFAULT_SCORE_CACHE = Path("outputs/mars_scene_domain_routing_development_scores.npz")
DEFAULT_PIXEL_CACHE = Path("outputs/mars_offshore_mask_threshold_development_counts.npz")
DEFAULT_METADATA_CACHE = Path("outputs/mars_spatial_scene_inputs_all_folds_metadata.npz")
DEFAULT_FEATURE_CACHES = (
    Path("outputs/mars_scene_features_folds234.npz"),
    Path("outputs/mars_scene_features_fold0.npz"),
    Path("outputs/mars_scene_features_fold1_crossfit.npz"),
)
DEFAULT_DOMAIN_REPORT = Path("reports/experiments/mars_scene_domain_routing_confirmation.json")
DEFAULT_MASK_REPORT = Path("reports/experiments/mars_offshore_mask_threshold_confirmation.json")
DEFAULT_SPATIAL_REPORT = Path("reports/experiments/mars_spatial_scene_classifier.json")
DEFAULT_JSON = Path("reports/experiments/mars_scene_gated_masks.json")
DEFAULT_MARKDOWN = Path("reports/experiments/MARS_SCENE_GATED_MASKS.md")
SELECTION_FOLDS = (2, 3, 4)
CONFIRMATION_FOLDS = (0, 1)
DEFAULT_CUTOFFS = tuple(float(value) for value in np.arange(0.3, 0.901, 0.025))


def iou(counts: np.ndarray) -> float:
    total = counts.sum(axis=0, dtype=np.int64)
    return float(total[0] / max(int(total.sum()), 1))


def paired_group_bootstrap(
    baseline: np.ndarray,
    candidate: np.ndarray,
    groups: np.ndarray,
    *,
    replicates: int,
    seed: int,
    confidence: float,
    batch_size: int = 256,
) -> dict[str, float | int]:
    """Paired nonparametric group bootstrap for aggregate IoU delta."""
    if baseline.shape != candidate.shape or baseline.ndim != 2 or baseline.shape[1] != 3:
        raise ValueError("baseline and candidate must be matching Nx3 count arrays")
    if groups.shape != (baseline.shape[0],):
        raise ValueError("groups must align one-to-one with count rows")
    _, group_index = np.unique(groups.astype(str), return_inverse=True)
    group_count = int(group_index.max()) + 1
    base_group = np.stack(
        [np.bincount(group_index, weights=baseline[:, index], minlength=group_count) for index in range(3)],
        axis=1,
    )
    candidate_group = np.stack(
        [np.bincount(group_index, weights=candidate[:, index], minlength=group_count) for index in range(3)],
        axis=1,
    )
    rng = np.random.default_rng(seed)
    probabilities = np.full(group_count, 1.0 / group_count)
    parts = []
    for start in range(0, replicates, batch_size):
        size = min(batch_size, replicates - start)
        draws = rng.multinomial(group_count, probabilities, size=size)
        base_total = draws @ base_group
        candidate_total = draws @ candidate_group
        base_iou = base_total[:, 0] / np.maximum(base_total.sum(axis=1), 1)
        candidate_iou = candidate_total[:, 0] / np.maximum(candidate_total.sum(axis=1), 1)
        parts.append(candidate_iou - base_iou)
    values = np.concatenate(parts)
    alpha = (1.0 - confidence) / 2.0
    return {
        "replicates": replicates,
        "groups": group_count,
        "confidence": confidence,
        "mean": float(values.mean()),
        "lower": float(np.quantile(values, alpha)),
        "upper": float(np.quantile(values, 1.0 - alpha)),
    }


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as values:
        return {name: values[name] for name in values.files}


def gate_counts(counts: np.ndarray, scores: np.ndarray, cutoff: float) -> np.ndarray:
    """Zero masks below the scene cutoff while preserving exact truth counts."""
    if counts.ndim != 2 or counts.shape[1] != 3 or scores.shape != (counts.shape[0],):
        raise ValueError("Pixel counts and scene scores do not align")
    gated = counts.copy()
    rejected = scores < cutoff
    gated[rejected, 2] += gated[rejected, 0]
    gated[rejected, 0] = 0
    gated[rejected, 1] = 0
    return gated


def count_metrics(counts: np.ndarray) -> dict[str, float | int]:
    total = counts.sum(axis=0, dtype=np.int64)
    tp, fp, fn = (int(value) for value in total)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "iou": tp / max(tp + fp + fn, 1),
        "recall": tp / max(tp + fn, 1),
        "precision": tp / max(tp + fp, 1),
    }


def summarize(
    rows: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    folds: np.ndarray,
    sensors: np.ndarray,
    groups: np.ndarray,
    *,
    cutoff: float,
    replicates: int,
    seed: int,
    confidence: float,
) -> dict[str, Any]:
    base = baseline[rows]
    cand = candidate[rows]
    base_metrics = count_metrics(base)
    candidate_metrics = count_metrics(cand)
    fold_values = {}
    for fold in np.unique(folds[rows]):
        local = rows & (folds == fold)
        fold_values[str(int(fold))] = {
            "baseline_iou": iou(baseline[local]),
            "candidate_iou": iou(candidate[local]),
            "delta": iou(candidate[local]) - iou(baseline[local]),
        }
    sensor_values = {}
    for index, name in enumerate(SENSOR_NAMES):
        local = rows & (sensors == index)
        sensor_values[name] = {
            "baseline_iou": iou(baseline[local]),
            "candidate_iou": iou(candidate[local]),
            "delta": iou(candidate[local]) - iou(baseline[local]),
        }
    bootstrap = paired_group_bootstrap(
        base,
        cand,
        groups[rows],
        replicates=replicates,
        seed=seed,
        confidence=confidence,
    )
    positive = rows & (labels == 1)
    return {
        "rows": int(rows.sum()),
        "positive_scenes": int(positive.sum()),
        "cutoff": cutoff,
        "retained_scene_fraction": float(np.mean(scores[rows] >= cutoff)),
        "retained_positive_scene_fraction": float(np.mean(scores[positive] >= cutoff)),
        "retained_true_positive_pixel_fraction": (
            candidate_metrics["tp"] / max(int(base_metrics["tp"]), 1)
        ),
        "removed_false_positive_pixel_fraction": (
            1.0 - candidate_metrics["fp"] / max(int(base_metrics["fp"]), 1)
        ),
        "baseline": base_metrics,
        "candidate": candidate_metrics,
        "delta": candidate_metrics["iou"] - base_metrics["iou"],
        "folds": fold_values,
        "sensors": sensor_values,
        "paired_group_bootstrap_delta": bootstrap,
        "checks": {
            "iou_higher": candidate_metrics["iou"] > base_metrics["iou"],
            "bootstrap_lower_positive": bootstrap["lower"] > 0.0,
            "all_fold_deltas_positive": all(value["delta"] > 0.0 for value in fold_values.values()),
            "no_sensor_regression": all(value["delta"] >= 0.0 for value in sensor_values.values()),
            "retain_at_least_95pct_true_positive_pixels": (
                candidate_metrics["tp"] >= 0.95 * base_metrics["tp"]
            ),
        },
    }


def write_json(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    selected = report["selection"]["selected"]
    confirmation = report["confirmation"]
    all_folds = report["all_folds"]
    lines = [
        "# Development-only scene-gated MARS masks",
        "",
        "The paper test was not loaded. Released-model masks use the confirmed sensor thresholds "
        "(Sentinel-2 0.80; Landsat 0.70), then are suppressed when the cross-fitted v3 scene "
        "probability is below a development-selected cutoff.",
        "",
        "| Partition | Cutoff | Baseline IoU | Gated IoU | Delta | 95% CI | TP pixels retained | FP pixels removed | Gates |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for name, value in (
        ("Selection folds 2/3/4", selected),
        ("Confirmation folds 0/1", confirmation),
        ("All five folds", all_folds),
    ):
        interval = value["paired_group_bootstrap_delta"]
        lines.append(
            f"| {name} | {value['cutoff']:.3f} | {value['baseline']['iou']:.5f} | "
            f"{value['candidate']['iou']:.5f} | {value['delta']:+.5f} | "
            f"[{interval['lower']:+.5f}, {interval['upper']:+.5f}] | "
            f"{value['retained_true_positive_pixel_fraction']:.2%} | "
            f"{value['removed_false_positive_pixel_fraction']:.2%} | "
            f"{'PASS' if all(value['checks'].values()) else 'FAIL'} |"
        )
    lines.extend(["", report["decision"]])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-cache", default=DEFAULT_SCORE_CACHE.as_posix())
    parser.add_argument("--pixel-cache", default=DEFAULT_PIXEL_CACHE.as_posix())
    parser.add_argument("--metadata-cache", default=DEFAULT_METADATA_CACHE.as_posix())
    parser.add_argument("--feature-caches", nargs=3, default=[path.as_posix() for path in DEFAULT_FEATURE_CACHES])
    parser.add_argument("--domain-report", default=DEFAULT_DOMAIN_REPORT.as_posix())
    parser.add_argument("--mask-report", default=DEFAULT_MASK_REPORT.as_posix())
    parser.add_argument("--spatial-report", default=DEFAULT_SPATIAL_REPORT.as_posix())
    parser.add_argument("--cutoffs", type=float, nargs="+", default=list(DEFAULT_CUTOFFS))
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260760)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--output-json", default=DEFAULT_JSON.as_posix())
    parser.add_argument("--output-markdown", default=DEFAULT_MARKDOWN.as_posix())
    args = parser.parse_args()
    if len(set(args.cutoffs)) != len(args.cutoffs) or any(not 0.0 < value < 1.0 for value in args.cutoffs):
        parser.error("cutoffs must be unique values in (0,1)")

    root = repo_root()
    paths = {
        "score": (root / args.score_cache).resolve(),
        "pixel": (root / args.pixel_cache).resolve(),
        "metadata": (root / args.metadata_cache).resolve(),
    }
    domain_report_path = (root / args.domain_report).resolve()
    mask_report_path = (root / args.mask_report).resolve()
    spatial_report_path = (root / args.spatial_report).resolve()
    domain_report = json.loads(domain_report_path.read_text(encoding="utf-8"))
    mask_report = json.loads(mask_report_path.read_text(encoding="utf-8"))
    spatial_report = json.loads(spatial_report_path.read_text(encoding="utf-8"))
    expected_hashes = {
        "score": domain_report["provenance"]["development_score_cache_sha256"],
        "pixel": mask_report["provenance"]["development_count_cache_sha256"],
        "metadata": spatial_report["provenance"]["metadata_cache_sha256"],
    }
    for name, path in paths.items():
        if sha256(path) != expected_hashes[name]:
            raise ValueError(f"Frozen {name} cache hash mismatch")

    feature_paths = [(root / value).resolve() for value in args.feature_caches]
    feature_expected = (
        domain_report["provenance"]["inner_cache_sha256"],
        domain_report["provenance"]["fold0_cache_sha256"],
        domain_report["provenance"]["fold1_cache_sha256"],
    )
    for path, expected in zip(feature_paths, feature_expected, strict=True):
        if sha256(path) != expected:
            raise ValueError("Frozen feature cache hash mismatch")

    score = load_npz(paths["score"])
    features = [load_npz(path) for path in feature_paths]
    sample_ids = np.concatenate([value["sample_ids"] for value in features]).astype(str)
    labels = np.concatenate([score["inner_labels"], score["fold0_labels"], score["fold1_labels"]])
    sensors = np.concatenate([score["inner_sensors"], score["fold0_sensors"], score["fold1_sensors"]])
    groups = np.concatenate([score["inner_groups"], score["fold0_groups"], score["fold1_groups"]]).astype(str)
    folds = np.concatenate(
        [
            score["inner_folds"],
            np.zeros(score["fold0_labels"].shape, dtype=np.uint8),
            np.ones(score["fold1_labels"].shape, dtype=np.uint8),
        ]
    )
    scene_scores = np.concatenate([score["inner_new"], score["fold0_new"], score["fold1_new"]])
    for values, start, stop in (
        (features[0], 0, len(features[0]["labels"])),
        (features[1], len(features[0]["labels"]), len(features[0]["labels"]) + len(features[1]["labels"])),
        (features[2], len(features[0]["labels"]) + len(features[1]["labels"]), len(labels)),
    ):
        if not np.array_equal(values["labels"], labels[start:stop]):
            raise ValueError("Score and feature cache labels differ")

    metadata = load_npz(paths["metadata"])
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("Development feature caches contain duplicate sample IDs")
    lookup = {sample_id: index for index, sample_id in enumerate(sample_ids)}
    try:
        order = np.asarray([lookup[value] for value in metadata["sample_ids"].astype(str)])
    except KeyError as error:
        raise ValueError("Metadata sample ID is missing from scene feature caches") from error
    labels = labels[order].astype(np.uint8)
    sensors = sensors[order].astype(np.uint8)
    groups = groups[order].astype(str)
    folds = folds[order].astype(np.uint8)
    scene_scores = scene_scores[order].astype(np.float64)

    pixel = load_npz(paths["pixel"])
    for name, expected in (
        ("labels", labels),
        ("sensors", sensors),
        ("groups", groups),
        ("folds", folds),
    ):
        actual = pixel[name].astype(expected.dtype)
        if not np.array_equal(actual, expected):
            raise ValueError(f"Pixel and scene cache {name} differ")
    sentinel_index = SENSOR_NAMES.index("Sentinel-2")
    baseline = np.where(
        (sensors == sentinel_index)[:, None], pixel["threshold_08"], pixel["threshold_07"]
    ).astype(np.int64)
    selection_rows = np.isin(folds, SELECTION_FOLDS)
    confirmation_rows = np.isin(folds, CONFIRMATION_FOLDS)
    candidates = {}
    for index, cutoff in enumerate(args.cutoffs):
        candidate = gate_counts(baseline, scene_scores, cutoff)
        candidates[format(cutoff, ".8g")] = summarize(
            selection_rows,
            baseline,
            candidate,
            scene_scores,
            labels,
            folds,
            sensors,
            groups,
            cutoff=cutoff,
            replicates=args.bootstrap_replicates,
            seed=args.bootstrap_seed + index,
            confidence=args.confidence,
        )
    eligible = {
        key: value for key, value in candidates.items() if all(value["checks"].values())
    }
    selected_key, selected = max(
        (eligible or candidates).items(), key=lambda item: item[1]["candidate"]["iou"]
    )
    selected_cutoff = float(selected_key)
    selected_counts = gate_counts(baseline, scene_scores, selected_cutoff)
    confirmation = summarize(
        confirmation_rows,
        baseline,
        selected_counts,
        scene_scores,
        labels,
        folds,
        sensors,
        groups,
        cutoff=selected_cutoff,
        replicates=args.bootstrap_replicates,
        seed=args.bootstrap_seed + 100,
        confidence=args.confidence,
    )
    all_folds = summarize(
        np.ones(len(labels), dtype=bool),
        baseline,
        selected_counts,
        scene_scores,
        labels,
        folds,
        sensors,
        groups,
        cutoff=selected_cutoff,
        replicates=args.bootstrap_replicates,
        seed=args.bootstrap_seed + 101,
        confidence=args.confidence,
    )
    passed = bool(eligible) and all(selected["checks"].values()) and all(confirmation["checks"].values())
    report = {
        "schema_version": 1,
        "scope": "development-only cross-fitted scene-gated masks; paper test not loaded",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "architecture": {
            "scene_score": "frozen v3 stronger head, cross-fitted within folds 2/3/4 and fixed on held folds 0/1",
            "dense_mask": {"Sentinel-2": 0.8, "Landsat": 0.7},
            "minimum_connected_pixels": 100,
            "gate": "set the dense prediction to empty when scene probability is below cutoff",
        },
        "selection": {
            "folds": list(SELECTION_FOLDS),
            "candidate_count": len(candidates),
            "eligible_candidate_count": len(eligible),
            "selected_cutoff": selected_cutoff,
            "selected": selected,
            "candidates": candidates,
        },
        "confirmation": confirmation,
        "all_folds": all_folds,
        "all_selection_and_confirmation_gates_pass": passed,
        "decision": (
            "Freeze the scene-gated mask rule for one transparent post-test paper-cache replay."
            if passed
            else "Reject scene-gated masks because a development selection or held-fold gate failed."
        ),
        "provenance": {
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
            "script_sha256": sha256(Path(__file__).resolve()),
            "score_cache_sha256": sha256(paths["score"]),
            "pixel_cache_sha256": sha256(paths["pixel"]),
            "metadata_cache_sha256": sha256(paths["metadata"]),
            "domain_report_sha256": sha256(domain_report_path),
            "mask_report_sha256": sha256(mask_report_path),
            "spatial_report_sha256": sha256(spatial_report_path),
            "feature_cache_sha256": [sha256(path) for path in feature_paths],
            "numpy": np.__version__,
        },
    }
    write_json((root / args.output_json).resolve(), report)
    write_markdown((root / args.output_markdown).resolve(), report)
    print(
        json.dumps(
            {
                "ok": True,
                "selected_cutoff": selected_cutoff,
                "selection_delta": selected["delta"],
                "selection_lower": selected["paired_group_bootstrap_delta"]["lower"],
                "confirmation_delta": confirmation["delta"],
                "confirmation_lower": confirmation["paired_group_bootstrap_delta"]["lower"],
                "all_folds_delta": all_folds["delta"],
                "passed": passed,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
