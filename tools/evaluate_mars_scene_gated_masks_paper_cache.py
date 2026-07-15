#!/usr/bin/env python3
"""Replay a frozen scene-gated mask rule on the exact MARS paper cache."""

from __future__ import annotations

import argparse
import copy
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

DEFAULT_CACHE = Path("outputs/mars_paper_test_v3_diagnostic_cache.npz")
DEFAULT_GATE_REPORT = Path("reports/experiments/mars_scene_gated_masks.json")
DEFAULT_SCENE_REPORT = Path("reports/experiments/mars_spatial_successor_paper_posttest.json")
DEFAULT_JSON = Path("reports/experiments/mars_scene_gated_spatial_paper_posttest.json")
DEFAULT_MARKDOWN = Path("reports/experiments/MARS_SCENE_GATED_SPATIAL_PAPER_POSTTEST.md")


def pixel_summary(counts: np.ndarray) -> dict[str, int | float]:
    total = counts.sum(axis=0, dtype=np.int64)
    tp, fp, fn = (int(value) for value in total)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "intersection_over_union": tp / max(tp + fp + fn, 1),
    }


def gate_counts(counts: np.ndarray, scores: np.ndarray, cutoff: float) -> np.ndarray:
    gated = counts.copy()
    rejected = scores < cutoff
    gated[rejected, 2] += gated[rejected, 0]
    gated[rejected, 0] = 0
    gated[rejected, 1] = 0
    return gated


def paired_site_bootstrap_pixel_delta(
    baseline: np.ndarray,
    candidate: np.ndarray,
    sites: np.ndarray,
    *,
    replicates: int,
    seed: int,
    confidence: float,
    batch_size: int = 256,
) -> dict[str, float | int]:
    _, site_index = np.unique(sites.astype(str), return_inverse=True)
    site_count = int(site_index.max()) + 1
    baseline_site = np.stack(
        [np.bincount(site_index, weights=baseline[:, index], minlength=site_count) for index in range(3)],
        axis=1,
    )
    candidate_site = np.stack(
        [np.bincount(site_index, weights=candidate[:, index], minlength=site_count) for index in range(3)],
        axis=1,
    )
    rng = np.random.default_rng(seed)
    probabilities = np.full(site_count, 1.0 / site_count)
    parts = []
    for start in range(0, replicates, batch_size):
        size = min(batch_size, replicates - start)
        draws = rng.multinomial(site_count, probabilities, size=size)
        baseline_total = draws @ baseline_site
        candidate_total = draws @ candidate_site
        baseline_iou = baseline_total[:, 0] / np.maximum(baseline_total.sum(axis=1), 1)
        candidate_iou = candidate_total[:, 0] / np.maximum(candidate_total.sum(axis=1), 1)
        parts.append(candidate_iou - baseline_iou)
    values = np.concatenate(parts)
    alpha = (1.0 - confidence) / 2.0
    return {
        "replicates": replicates,
        "sites": site_count,
        "confidence": confidence,
        "mean": float(values.mean()),
        "lower": float(np.quantile(values, alpha)),
        "upper": float(np.quantile(values, 1.0 - alpha)),
    }


def write_json(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Scene-gated spatial successor: exact MARS-S2L paper benchmark",
        "",
        "Transparent post-test cache replay; it is not an untouched confirmation cohort. "
        "Scene metrics come unchanged from the frozen ordinary-BCE spatial classifier; "
        "only the independently development-confirmed dense-mask gate changes.",
        "",
        "| View | AP delta (95% CI) | Recall delta (95% CI) | IoU | IoU delta (95% CI) | Gates |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for name, value in report["views"].items():
        intervals = value["bootstrap"]["delta_intervals"]
        delta = value["metrics"]["delta"]
        candidate = value["metrics"]["candidate"]
        lines.append(
            f"| {name} | {delta['average_precision']:+.5f} "
            f"([{intervals['average_precision']['lower']:+.5f}, {intervals['average_precision']['upper']:+.5f}]) | "
            f"{delta['matched_fpr_recall']:+.5f} "
            f"([{intervals['matched_fpr_recall']['lower']:+.5f}, {intervals['matched_fpr_recall']['upper']:+.5f}]) | "
            f"{candidate['pixels']['intersection_over_union']:.5f} | {delta['pixel_iou']:+.5f} "
            f"([{intervals['pixel_iou']['lower']:+.5f}, {intervals['pixel_iou']['upper']:+.5f}]) | "
            f"{'PASS' if value['passed'] else 'FAIL'} |"
        )
    lines.extend(["", report["decision"]])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", default=DEFAULT_CACHE.as_posix())
    parser.add_argument("--gate-report", default=DEFAULT_GATE_REPORT.as_posix())
    parser.add_argument("--scene-report", default=DEFAULT_SCENE_REPORT.as_posix())
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260761)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--output-json", default=DEFAULT_JSON.as_posix())
    parser.add_argument("--output-markdown", default=DEFAULT_MARKDOWN.as_posix())
    args = parser.parse_args()

    root = repo_root()
    cache_path = (root / args.cache).resolve()
    gate_path = (root / args.gate_report).resolve()
    scene_path = (root / args.scene_report).resolve()
    gate_report = json.loads(gate_path.read_text(encoding="utf-8"))
    scene_report = json.loads(scene_path.read_text(encoding="utf-8"))
    if gate_report.get("all_selection_and_confirmation_gates_pass") is not True:
        raise ValueError("Development scene-gated mask rule was not promoted")
    if gate_report.get("scope") != "development-only cross-fitted scene-gated masks; paper test not loaded":
        raise ValueError("Unexpected gate report scope")
    if scene_report.get("scope") != "transparent post-test development evaluation on exact paper rows and comparator":
        raise ValueError("Unexpected spatial scene report scope")
    if sha256(cache_path) != scene_report["provenance"]["diagnostic_sha256"]:
        raise ValueError("Exact paper diagnostic cache hash mismatch")
    cutoff = float(gate_report["selection"]["selected_cutoff"])

    with np.load(cache_path, allow_pickle=False) as source:
        values = {name: source[name] for name in source.files}
    baseline = values["baseline_pixels"].astype(np.int64)
    ungated = values["candidate_pixels"].astype(np.int64)
    scores = values["candidate_scores"].astype(np.float64)
    sites = values["sites"].astype(str)
    test_only = values["test_only"].astype(bool)
    gated = gate_counts(ungated, scores, cutoff)
    selections = {
        "full": np.ones(len(scores), dtype=bool),
        "test_only_sites": test_only,
    }
    views = copy.deepcopy(scene_report["views"])
    for index, (name, selected) in enumerate(selections.items()):
        source_candidate = views[name]["metrics"]["candidate"]["pixels"]
        if source_candidate != pixel_summary(ungated[selected]):
            raise ValueError(f"Ungated pixel counts differ from frozen {name} scene report")
        baseline_pixels = pixel_summary(baseline[selected])
        if baseline_pixels != views[name]["metrics"]["baseline"]["pixels"]:
            raise ValueError(f"Baseline pixel counts differ from frozen {name} scene report")
        candidate_pixels = pixel_summary(gated[selected])
        interval = paired_site_bootstrap_pixel_delta(
            baseline[selected],
            gated[selected],
            sites[selected],
            replicates=args.bootstrap_replicates,
            seed=args.bootstrap_seed + index,
            confidence=args.confidence,
        )
        delta = candidate_pixels["intersection_over_union"] - baseline_pixels["intersection_over_union"]
        views[name]["metrics"]["candidate"]["pixels"] = candidate_pixels
        views[name]["metrics"]["delta"]["pixel_iou"] = delta
        views[name]["bootstrap"]["delta_intervals"]["pixel_iou"] = {
            "lower": interval["lower"],
            "mean": interval["mean"],
            "upper": interval["upper"],
        }
        views[name]["bootstrap"]["sites"] = interval["sites"]
        views[name]["bootstrap"]["replicates"] = interval["replicates"]
        views[name]["checks"]["pixel_iou_point_higher"] = delta > 0.0
        views[name]["checks"]["pixel_iou_lower_positive"] = interval["lower"] > 0.0
        views[name]["passed"] = all(views[name]["checks"].values())
        views[name]["mask_gate_audit"] = {
            "cutoff": cutoff,
            "retained_scene_fraction": float(np.mean(scores[selected] >= cutoff)),
            "retained_true_positive_pixel_fraction": (
                candidate_pixels["tp"] / max(int(source_candidate["tp"]), 1)
            ),
            "removed_false_positive_pixel_fraction": (
                1.0 - candidate_pixels["fp"] / max(int(source_candidate["fp"]), 1)
            ),
        }

    passed = all(value["passed"] for value in views.values())
    report = {
        "schema_version": 1,
        "scope": "transparent post-test scene-gated spatial successor on exact paper cache",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "architecture": {
            "scene_ranking": "unchanged frozen ordinary-BCE spatial classifier",
            "mask_probability": "released MARS-S2L dense probability",
            "mask_thresholds": {"Sentinel-2": 0.8, "Landsat": 0.7},
            "mask_gate_score": "frozen v3 stronger scene head",
            "mask_gate_cutoff": cutoff,
            "mask_gate_rule": "empty dense mask below cutoff",
        },
        "views": views,
        "all_exact_paper_gates_pass": passed,
        "decision": (
            "All exact paper gates pass on both views; require independent external confirmation before a publication claim."
            if passed
            else "Dense segmentation now passes both exact paper views, but at least one scene-level superiority gate remains unresolved."
        ),
        "provenance": {
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
            "script_sha256": sha256(Path(__file__).resolve()),
            "diagnostic_cache_sha256": sha256(cache_path),
            "gate_report_sha256": sha256(gate_path),
            "scene_report_sha256": sha256(scene_path),
            "numpy": np.__version__,
        },
    }
    write_json((root / args.output_json).resolve(), report)
    write_markdown((root / args.output_markdown).resolve(), report)
    print(
        json.dumps(
            {
                "ok": True,
                "passed": passed,
                "views": {
                    name: {
                        "pixel_iou": value["metrics"]["candidate"]["pixels"]["intersection_over_union"],
                        "pixel_delta": value["metrics"]["delta"]["pixel_iou"],
                        "pixel_lower": value["bootstrap"]["delta_intervals"]["pixel_iou"]["lower"],
                        "view_passed": value["passed"],
                    }
                    for name, value in views.items()
                },
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
