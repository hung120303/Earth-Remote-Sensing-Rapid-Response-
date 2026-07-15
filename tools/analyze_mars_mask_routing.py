#!/usr/bin/env python3
"""Select a scene-conditioned dense-mask rule on development folds only."""

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
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "EarthRemoteSensingRapidResponse"
for path in (MODEL_ROOT, ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from acquire_mars_metadata import DEFAULT_OUTPUT, repo_root, sha256  # noqa: E402
from analyze_mars_mask_thresholds import component_mask_at  # noqa: E402
from evaluate_released_marss2l import connected_scene_score  # noqa: E402
from mars_paper_model import ReleasedMarsUNet, SENSOR_NAMES, released_state  # noqa: E402
from train_mars_paper_residual import (  # noqa: E402
    DEFAULT_ACQUISITION_RECEIPT,
    DEFAULT_CHECKPOINT,
    DEFAULT_MANIFEST,
    DEFAULT_PROTOCOL,
    MarsPaperDataset,
    iter_development_manifest,
    move_batch,
    verify_acquisition_receipt,
)

DEFAULT_CUTOFFS = tuple(float(value) for value in np.arange(0.5, 0.976, 0.025))
DEFAULT_JSON = Path("reports/experiments/mars_mask_routing_folds234.json")
DEFAULT_MARKDOWN = Path("reports/experiments/MARS_MASK_ROUTING_FOLDS234.md")


def pixel_counts(
    prediction: np.ndarray, truth: np.ndarray, observable: np.ndarray
) -> np.ndarray:
    """Return mutually exclusive TP, FP, and FN counts for one scene."""
    return np.asarray(
        [
            np.count_nonzero(prediction & truth),
            np.count_nonzero(prediction & observable & ~truth),
            np.count_nonzero(truth & ~prediction),
        ],
        dtype=np.int64,
    )


def iou(counts: np.ndarray) -> float:
    total = counts.sum(axis=0)
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
    """Paired nonparametric group bootstrap for an aggregate IoU delta."""
    if baseline.shape != candidate.shape or baseline.ndim != 2 or baseline.shape[1] != 3:
        raise ValueError("baseline and candidate must be matching Nx3 count arrays")
    if groups.shape != (baseline.shape[0],):
        raise ValueError("groups must align one-to-one with count rows")
    if replicates < 100 or not 0.0 < confidence < 1.0:
        raise ValueError("replicates must be >=100 and confidence must be in (0,1)")
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
    parts: list[np.ndarray] = []
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


def summarize(
    baseline: np.ndarray,
    candidate: np.ndarray,
    folds: np.ndarray,
    sensors: np.ndarray,
    groups: np.ndarray,
    *,
    replicates: int,
    seed: int,
    confidence: float,
) -> dict[str, Any]:
    baseline_iou = iou(baseline)
    candidate_iou = iou(candidate)
    fold_values = {}
    for fold in np.unique(folds):
        rows = folds == fold
        fold_values[str(int(fold))] = {
            "baseline_iou": iou(baseline[rows]),
            "candidate_iou": iou(candidate[rows]),
            "delta": iou(candidate[rows]) - iou(baseline[rows]),
        }
    sensor_values = {}
    for sensor_index, name in enumerate(SENSOR_NAMES):
        rows = sensors == sensor_index
        sensor_values[name] = {
            "baseline_iou": iou(baseline[rows]),
            "candidate_iou": iou(candidate[rows]),
            "delta": iou(candidate[rows]) - iou(baseline[rows]),
        }
    bootstrap = paired_group_bootstrap(
        baseline,
        candidate,
        groups,
        replicates=replicates,
        seed=seed,
        confidence=confidence,
    )
    return {
        "baseline_iou": baseline_iou,
        "candidate_iou": candidate_iou,
        "delta": candidate_iou - baseline_iou,
        "folds": fold_values,
        "sensors": sensor_values,
        "paired_group_bootstrap_delta": bootstrap,
    }


def select_candidate(candidates: dict[str, dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    """Prefer a positive CI, then worst-domain and pooled improvements."""
    def rank(item: tuple[str, dict[str, Any]]) -> tuple[bool, float, float, float, float]:
        value = item[1]
        fold_floor = min(row["delta"] for row in value["folds"].values())
        sensor_floor = min(row["delta"] for row in value["sensors"].values())
        lower = value["paired_group_bootstrap_delta"]["lower"]
        all_domains_positive = fold_floor > 0.0 and sensor_floor > 0.0
        return lower > 0.0 and all_domains_positive, lower, fold_floor, sensor_floor, value["delta"]

    return max(candidates.items(), key=rank)


def write_json(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Development-only scene-conditioned mask routing",
        "",
        "The paper test was not loaded. A released-model connected score routes strong scenes to a 0.5 mask and all other scenes to the frozen 0.7 mask.",
        "",
        "| Route cutoff | IoU | Delta vs 0.7 | 95% lower | Worst fold | Worst sensor |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for cutoff, value in report["candidates"].items():
        lines.append(
            f"| {float(cutoff):.3f} | {value['candidate_iou']:.5f} | {value['delta']:+.5f} | "
            f"{value['paired_group_bootstrap_delta']['lower']:+.5f} | "
            f"{min(row['delta'] for row in value['folds'].values()):+.5f} | "
            f"{min(row['delta'] for row in value['sensors'].values()):+.5f} |"
        )
    selected = report["selected"]
    lines.extend(
        [
            "",
            f"Selected released-scene cutoff: **{report['selected_cutoff']:.3f}**.",
            "",
            f"Development IoU: {selected['candidate_iou']:.5f} ({selected['delta']:+.5f} versus global 0.7).",
            "",
            report["decision"],
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@torch.no_grad()
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-dir", default=DEFAULT_OUTPUT.as_posix())
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST.as_posix())
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL.as_posix())
    parser.add_argument("--acquisition-receipt", default=DEFAULT_ACQUISITION_RECEIPT.as_posix())
    parser.add_argument("--released-checkpoint", default=DEFAULT_CHECKPOINT.as_posix())
    parser.add_argument("--folds", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--routing-cutoffs", type=float, nargs="+", default=list(DEFAULT_CUTOFFS))
    parser.add_argument("--wide-threshold", type=float, default=0.5)
    parser.add_argument("--conservative-threshold", type=float, default=0.7)
    parser.add_argument("--minimum-connected-pixels", type=int, default=100)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260714)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output-json", default=DEFAULT_JSON.as_posix())
    parser.add_argument("--output-markdown", default=DEFAULT_MARKDOWN.as_posix())
    args = parser.parse_args()
    folds_requested = tuple(args.folds)
    cutoffs = tuple(args.routing_cutoffs)
    if len(set(folds_requested)) != len(folds_requested) or any(not 0 <= value < 5 for value in folds_requested):
        parser.error("folds must be unique values in [0,4]")
    if len(set(cutoffs)) != len(cutoffs) or any(not 0.0 < value < 1.0 for value in cutoffs):
        parser.error("routing cutoffs must be unique values in (0,1)")
    if not 0.0 < args.wide_threshold < args.conservative_threshold < 1.0:
        parser.error("mask thresholds must satisfy 0 < wide < conservative < 1")
    if args.minimum_connected_pixels != 100:
        parser.error("released connected_scene_score is frozen to 100 connected pixels")
    if args.batch_size <= 0 or args.workers < 0:
        parser.error("area/batch must be positive and workers non-negative")

    root = repo_root()
    manifest = (root / args.manifest).resolve()
    protocol_path = (root / args.protocol).resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    manifest_hash = sha256(manifest)
    if manifest_hash != protocol["development_manifest_sha256"]:
        raise ValueError("Development manifest differs from the frozen fold protocol")
    verify_acquisition_receipt((root / args.acquisition_receipt).resolve(), manifest_hash)
    group_to_fold = {str(row["group_id"]): int(row["fold"]) for row in protocol["assignments"]}
    records = [
        row for row in iter_development_manifest(manifest)
        if group_to_fold[str(row["group_id"])] in folds_requested
    ]
    loader = DataLoader(
        MarsPaperDataset((root / args.metadata_dir).resolve(), records, augment=False, seed=0),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = (root / args.released_checkpoint).resolve()
    model = ReleasedMarsUNet().to(device)
    model.load_state_dict(released_state(checkpoint), strict=False)
    model.eval()

    scores: list[float] = []
    wide_counts: list[np.ndarray] = []
    conservative_counts: list[np.ndarray] = []
    row_folds: list[int] = []
    row_sensors: list[int] = []
    row_groups: list[str] = []
    for batch_index, batch in enumerate(loader, start=1):
        batch = move_batch(batch, device)
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(batch["inputs"])
        probabilities = torch.sigmoid(logits).float().masked_fill(batch["clear"] <= 0.5, 0.0).cpu().numpy()
        for index in range(probabilities.shape[0]):
            score = probabilities[index, 0]
            observable = batch["observable"][index, 0].cpu().numpy() > 0.5
            truth = (batch["mask"][index, 0].cpu().numpy() > 0.5) & observable
            scores.append(float(connected_scene_score(score)))
            wide_counts.append(
                pixel_counts(component_mask_at(score, args.wide_threshold, args.minimum_connected_pixels), truth, observable)
            )
            conservative_counts.append(
                pixel_counts(component_mask_at(score, args.conservative_threshold, args.minimum_connected_pixels), truth, observable)
            )
            group = str(batch["group_id"][index])
            row_groups.append(group)
            row_folds.append(group_to_fold[group])
            row_sensors.append(int(batch["sensor_index"][index].item()))
        if batch_index % 100 == 0:
            print(json.dumps({"batches": batch_index, "rows": len(scores)}), flush=True)

    scene_scores = np.asarray(scores)
    wide = np.stack(wide_counts)
    conservative = np.stack(conservative_counts)
    fold_array = np.asarray(row_folds, dtype=np.uint8)
    sensor_array = np.asarray(row_sensors, dtype=np.uint8)
    group_array = np.asarray(row_groups)
    candidates: dict[str, dict[str, Any]] = {}
    for cutoff_index, cutoff in enumerate(cutoffs):
        use_wide = scene_scores > cutoff
        candidate = np.where(use_wide[:, None], wide, conservative)
        candidates[format(cutoff, ".8g")] = {
            "wide_scene_rows": int(use_wide.sum()),
            **summarize(
                conservative,
                candidate,
                fold_array,
                sensor_array,
                group_array,
                replicates=args.bootstrap_replicates,
                seed=args.bootstrap_seed + cutoff_index,
                confidence=args.confidence,
            ),
        }
    selected_key, selected = select_candidate(candidates)
    fold_floor = min(row["delta"] for row in selected["folds"].values())
    sensor_floor = min(row["delta"] for row in selected["sensors"].values())
    passed = (
        selected["paired_group_bootstrap_delta"]["lower"] > 0.0
        and fold_floor > 0.0
        and sensor_floor > 0.0
    )
    report = {
        "schema_version": 1,
        "scope": "development folds only; paper test not loaded",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "folds": list(folds_requested),
        "rows": len(records),
        "routing_feature": "released MARS-S2L connected-component scene score",
        "rule": "use wide mask when released_scene_score > cutoff; otherwise conservative mask",
        "wide_threshold": args.wide_threshold,
        "conservative_threshold": args.conservative_threshold,
        "minimum_connected_pixels": args.minimum_connected_pixels,
        "selected_cutoff": float(selected_key),
        "selected": selected,
        "selected_passes_fold_sensor_and_bootstrap_gates": passed,
        "decision": (
            "Advance this routed mask rule to independent folds 0/1 confirmation."
            if passed
            else "Reject routed masking; it lacks a positive paired group-bootstrap lower bound across development domains."
        ),
        "candidates": candidates,
        "provenance": {
            "manifest_sha256": manifest_hash,
            "protocol_sha256": sha256(protocol_path),
            "checkpoint_sha256": sha256(checkpoint),
            "script_sha256": sha256(Path(__file__).resolve()),
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
            "device": str(torch.cuda.get_device_name(device) if device.type == "cuda" else device),
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
    }
    write_json((root / args.output_json).resolve(), report)
    write_markdown((root / args.output_markdown).resolve(), report)
    print(json.dumps({
        "ok": True,
        "selected_cutoff": float(selected_key),
        "candidate_iou": selected["candidate_iou"],
        "delta": selected["delta"],
        "bootstrap_lower": selected["paired_group_bootstrap_delta"]["lower"],
        "passed": passed,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
