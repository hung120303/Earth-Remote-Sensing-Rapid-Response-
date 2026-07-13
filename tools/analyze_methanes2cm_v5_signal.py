#!/usr/bin/env python3
"""Audit frozen MethaneS2CM masks and reference-only MBMP baselines."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "EarthRemoteSensingRapidResponse"
if str(MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_ROOT))

from acquire_mars_metadata import repo_root, sha256  # noqa: E402
from train_mars_v3 import safe_output, tracked_dirty, write_json  # noqa: E402
from train_methanes2cm_v5 import (  # noqa: E402
    DEFAULT_MANIFEST,
    DEFAULT_PACKED,
    PackedMethaneS2CMDataset,
    choose_threshold_at_fpr,
    read_manifest,
)

DEFAULT_V5_REPORT = Path("reports/experiments/methanes2cm_v5_seed1101_validation.json")
DEFAULT_REPORT = Path("reports/experiments/methanes2cm_v5_signal_audit.json")
DEFAULT_MARKDOWN = Path("reports/experiments/METHANES2CM_V5_SIGNAL_AUDIT.md")


def robust_scene_score(
    evidence: np.ndarray,
    observable: np.ndarray,
    *,
    topk_fraction: float = 0.01,
    max_weight: float = 0.15,
) -> np.ndarray:
    values = np.asarray(evidence, dtype=np.float32)
    valid = np.asarray(observable, dtype=bool)
    if values.shape != valid.shape or values.ndim != 3:
        raise ValueError("Evidence and observability must be matching BxHxW arrays")
    flattened = values.reshape(values.shape[0], -1)
    flattened_valid = valid.reshape(valid.shape[0], -1)
    masked = np.where(flattened_valid, flattened, -np.inf)
    count = max(1, int(masked.shape[1] * topk_fraction))
    top = np.partition(masked, -count, axis=1)[:, -count:]
    result = (1.0 - max_weight) * np.mean(top, axis=1) + max_weight * np.max(
        top, axis=1
    )
    result[~np.isfinite(result)] = -1e4
    return result


def quantiles(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "minimum": float(np.min(array)),
        "p10": float(np.quantile(array, 0.10)),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.90)),
        "maximum": float(np.max(array)),
        "mean": float(np.mean(array)),
    }


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# MethaneS2CM v5 signal and mask audit",
        "",
        "Exploratory analysis on the frozen internal-development groups only; location-test imagery remains sealed.",
        "",
        "| Evidence | Scene AP | AUROC | Recall at <=5% FPR | Pixel AP |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, result in sorted(
        report["physics_baselines"].items(),
        key=lambda item: item[1]["scene_average_precision"],
        reverse=True,
    ):
        lines.append(
            f"| `{name}` | {result['scene_average_precision']:.4f} | "
            f"{result['scene_auroc']:.4f} | {result['operating_point_0.05']['recall']:.4f} | "
            f"{result['pixel_average_precision']:.4f} |"
        )
    mask = report["mask_audit"]
    lines.extend(
        [
            "",
            f"- Positive-mask median area: {mask['positive_mask_pixels']['median']:.0f} / 1024 pixels",
            f"- Positive crops with >=50% mask coverage: {mask['positive_crops_at_least_half_masked']:,}",
            f"- Positive crops with 100% mask coverage: {mask['positive_crops_fully_masked']:,}",
            f"- Observable-pixel median: {mask['observable_pixels']['median']:.0f} / 1024",
            "- Baselines are diagnostic reference-only ratios, not fitted models or test results.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packed", default=DEFAULT_PACKED.as_posix())
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST.as_posix())
    parser.add_argument("--v5-report", default=DEFAULT_V5_REPORT.as_posix())
    parser.add_argument("--report", default=DEFAULT_REPORT.as_posix())
    parser.add_argument("--markdown", default=DEFAULT_MARKDOWN.as_posix())
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    root = repo_root()
    if tracked_dirty(root):
        raise RuntimeError("Refusing MethaneS2CM signal audit from a dirty tracked worktree")
    packed_path = (root / args.packed).resolve()
    manifest_path = (root / args.manifest).resolve()
    v5_report_path = (root / args.v5_report).resolve()
    report_path = safe_output(root, args.report)
    markdown_path = safe_output(root, args.markdown)
    rows = [
        row
        for row in read_manifest(manifest_path)
        if row["research_role"] == "internal_development"
    ]
    dataset = PackedMethaneS2CMDataset(
        packed_path, rows, augment=False, seed=20260713
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
        prefetch_factor=2 if args.workers > 0 else None,
    )
    labels: list[np.ndarray] = []
    groups: list[str] = []
    pixel_truth: list[np.ndarray] = []
    mask_pixels: list[np.ndarray] = []
    observable_pixels: list[np.ndarray] = []
    scene_scores: dict[str, list[np.ndarray]] = {}
    pixel_scores: dict[str, list[np.ndarray]] = {}
    for batch in loader:
        inputs = batch["inputs"].numpy()
        observable = batch["observable"][:, 0].numpy() > 0.5
        truth = batch["mask"][:, 0].numpy() > 0.5
        truth &= observable
        mbmp90 = np.clip(inputs[:, 0], 0.1, 10.0)
        mbmp365 = np.clip(inputs[:, 1], 0.1, 10.0)
        log90 = np.log(mbmp90)
        log365 = np.log(mbmp365)
        candidates = {
            "T_over_Tminus90_high": log90,
            "T_over_Tminus90_low": -log90,
            "T_over_Tminus365_high": log365,
            "T_over_Tminus365_low": -log365,
            "reference_max_low": np.maximum(-log90, -log365),
            "reference_mean_low": (-log90 - log365) / 2.0,
            "reference_consensus_low": np.minimum(-log90, -log365),
            "reference_max_absolute_change": np.maximum(np.abs(log90), np.abs(log365)),
        }
        labels.append(batch["presence"].numpy().astype(np.uint8))
        groups.extend(str(value) for value in batch["group_id"])
        pixel_truth.append(truth[observable].astype(np.uint8))
        mask_pixels.append(truth.reshape(truth.shape[0], -1).sum(axis=1))
        observable_pixels.append(observable.reshape(observable.shape[0], -1).sum(axis=1))
        for name, evidence in candidates.items():
            scene_scores.setdefault(name, []).append(
                robust_scene_score(evidence, observable)
            )
            # Float16 is sufficient for a diagnostic rank baseline and keeps
            # eight 15.8M-pixel candidates below 300 MB total resident memory.
            pixel_scores.setdefault(name, []).append(
                evidence[observable].astype(np.float16)
            )

    y = np.concatenate(labels)
    truth_values = np.concatenate(pixel_truth)
    areas = np.concatenate(mask_pixels)
    visible = np.concatenate(observable_pixels)
    physics: dict[str, Any] = {}
    for name in scene_scores:
        scene = np.concatenate(scene_scores[name]).astype(np.float32)
        pixels = np.concatenate(pixel_scores[name]).astype(np.float32)
        physics[name] = {
            "scene_average_precision": float(average_precision_score(y, scene)),
            "scene_auroc": float(roc_auc_score(y, scene)),
            "operating_point_0.05": choose_threshold_at_fpr(y, scene, 0.05),
            "pixel_average_precision": float(
                average_precision_score(truth_values, pixels)
            ),
            "scene_score": "0.85 * mean(top 1% evidence pixels) + 0.15 * maximum evidence pixel",
        }
    v5 = json.loads(v5_report_path.read_text(encoding="utf-8"))
    positive_areas = areas[y == 1]
    report = {
        "schema_version": 1,
        "scope": "methanes2cm_v5_internal_development_signal_audit",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "cohort": {
            "samples": int(y.size),
            "positives": int(np.count_nonzero(y == 1)),
            "negatives": int(np.count_nonzero(y == 0)),
            "geographic_groups": len(set(groups)),
            "location_test_images_opened": False,
        },
        "mask_audit": {
            "positive_mask_pixels": quantiles(positive_areas),
            "observable_pixels": quantiles(visible),
            "positive_crops_at_least_half_masked": int(
                np.count_nonzero(positive_areas >= 512)
            ),
            "positive_crops_at_least_90pct_masked": int(
                np.count_nonzero(positive_areas >= 922)
            ),
            "positive_crops_fully_masked": int(
                np.count_nonzero(positive_areas == 1024)
            ),
            "negative_crops_with_positive_mask_pixels": int(
                np.count_nonzero(areas[y == 0] > 0)
            ),
        },
        "physics_baselines": physics,
        "learned_v5_seed1101_reference": {
            "scene_average_precision": v5["validation"]["average_precision"],
            "scene_auroc": v5["validation"]["auroc"],
            "operating_point_0.05": v5["validation"]["operating_points"]["0.05"],
            "pixel_average_precision": v5["validation"]["segmentation"][
                "average_precision_all_observable_pixels"
            ],
            "pixel_dice": v5["validation"]["segmentation"]["dice"],
        },
        "interpretation": (
            "Exploratory internal-development diagnostics. Candidate evidence definitions and "
            "their ranking may inform later fitting experiments, so none are test claims."
        ),
        "provenance": {
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True
            ).strip(),
            "script": Path(__file__).resolve().relative_to(root).as_posix(),
            "script_sha256": sha256(Path(__file__).resolve()),
            "manifest_sha256": sha256(manifest_path),
            "packed_sha256": v5["seal"]["packed_sha256"],
            "v5_report_sha256": sha256(v5_report_path),
            "tracked_worktree_dirty_at_start": False,
        },
    }
    write_json(report_path, report)
    write_markdown(markdown_path, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
