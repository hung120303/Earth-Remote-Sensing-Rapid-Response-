#!/usr/bin/env python3
"""Confirm frozen sensor-specific MARS mask thresholds on all development folds."""

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
from analyze_mars_mask_routing import (  # noqa: E402
    iou,
    paired_group_bootstrap,
    pixel_counts,
)
from analyze_mars_mask_thresholds import component_mask_at  # noqa: E402
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

DEFAULT_JSON = Path("reports/experiments/mars_sensor_mask_thresholds_confirmation.json")
DEFAULT_MARKDOWN = Path("reports/experiments/MARS_SENSOR_MASK_THRESHOLDS_CONFIRMATION.md")
SELECTION_FOLDS = (2, 3, 4)
CONFIRMATION_FOLDS = (0, 1)


def route_sensor_counts(
    baseline: np.ndarray, sentinel: np.ndarray, sensors: np.ndarray
) -> np.ndarray:
    """Use the Sentinel-specific counts for S2 and baseline counts for Landsat."""
    if baseline.shape != sentinel.shape or baseline.ndim != 2 or baseline.shape[1] != 3:
        raise ValueError("baseline and sentinel must be matching Nx3 arrays")
    if sensors.shape != (baseline.shape[0],):
        raise ValueError("sensors must align one-to-one with count rows")
    sentinel_index = SENSOR_NAMES.index("Sentinel-2")
    return np.where((sensors == sentinel_index)[:, None], sentinel, baseline)


def summarize_partition(
    rows: np.ndarray,
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
    base = baseline[rows]
    cand = candidate[rows]
    base_iou = iou(base)
    candidate_iou = iou(cand)
    by_fold = {}
    for fold in np.unique(folds[rows]):
        local = rows & (folds == fold)
        by_fold[str(int(fold))] = {
            "baseline_iou": iou(baseline[local]),
            "candidate_iou": iou(candidate[local]),
            "delta": iou(candidate[local]) - iou(baseline[local]),
        }
    by_sensor = {}
    for index, name in enumerate(SENSOR_NAMES):
        local = rows & (sensors == index)
        by_sensor[name] = {
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
    return {
        "rows": int(rows.sum()),
        "baseline_iou": base_iou,
        "candidate_iou": candidate_iou,
        "delta": candidate_iou - base_iou,
        "folds": by_fold,
        "sensors": by_sensor,
        "paired_group_bootstrap_delta": bootstrap,
        "checks": {
            "point_delta_positive": candidate_iou > base_iou,
            "all_fold_deltas_positive": all(value["delta"] > 0.0 for value in by_fold.values()),
            "no_sensor_regression": all(value["delta"] >= 0.0 for value in by_sensor.values()),
            "sentinel_delta_positive": by_sensor["Sentinel-2"]["delta"] > 0.0,
            "bootstrap_lower_positive": bootstrap["lower"] > 0.0,
        },
    }


def write_json(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Sensor-specific MARS mask-threshold confirmation",
        "",
        "Frozen rule: Sentinel-2 uses 0.80; Landsat retains 0.70. The paper test was not loaded.",
        "",
        "| Partition | Rows | Baseline IoU | Routed IoU | Delta | 95% CI | Gates |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for name, value in report["partitions"].items():
        interval = value["paired_group_bootstrap_delta"]
        passed = all(value["checks"].values())
        lines.append(
            f"| {name} | {value['rows']:,} | {value['baseline_iou']:.5f} | "
            f"{value['candidate_iou']:.5f} | {value['delta']:+.5f} | "
            f"[{interval['lower']:+.5f}, {interval['upper']:+.5f}] | {'PASS' if passed else 'FAIL'} |"
        )
    lines.extend(["", report["decision"]])
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
    parser.add_argument("--baseline-threshold", type=float, default=0.7)
    parser.add_argument("--sentinel-threshold", type=float, default=0.8)
    parser.add_argument("--minimum-connected-pixels", type=int, default=100)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260715)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output-json", default=DEFAULT_JSON.as_posix())
    parser.add_argument("--output-markdown", default=DEFAULT_MARKDOWN.as_posix())
    args = parser.parse_args()
    if args.minimum_connected_pixels != 100:
        parser.error("the frozen paper contract requires 100 connected pixels")
    if not 0.0 < args.baseline_threshold < args.sentinel_threshold < 1.0:
        parser.error("thresholds must satisfy 0 < baseline < sentinel < 1")
    if args.batch_size <= 0 or args.workers < 0:
        parser.error("batch size must be positive and workers non-negative")

    root = repo_root()
    manifest = (root / args.manifest).resolve()
    protocol_path = (root / args.protocol).resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    manifest_hash = sha256(manifest)
    if manifest_hash != protocol["development_manifest_sha256"]:
        raise ValueError("Development manifest differs from the frozen fold protocol")
    verify_acquisition_receipt((root / args.acquisition_receipt).resolve(), manifest_hash)
    group_to_fold = {str(row["group_id"]): int(row["fold"]) for row in protocol["assignments"]}
    records = list(iter_development_manifest(manifest))
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

    baseline_counts: list[np.ndarray] = []
    sentinel_counts: list[np.ndarray] = []
    folds: list[int] = []
    sensors: list[int] = []
    groups: list[str] = []
    for batch_index, batch in enumerate(loader, start=1):
        batch = move_batch(batch, device)
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(batch["inputs"])
        probabilities = torch.sigmoid(logits).float().masked_fill(batch["clear"] <= 0.5, 0.0).cpu().numpy()
        for index in range(probabilities.shape[0]):
            score = probabilities[index, 0]
            observable = batch["observable"][index, 0].cpu().numpy() > 0.5
            truth = (batch["mask"][index, 0].cpu().numpy() > 0.5) & observable
            baseline_counts.append(
                pixel_counts(component_mask_at(score, args.baseline_threshold, 100), truth, observable)
            )
            sentinel_counts.append(
                pixel_counts(component_mask_at(score, args.sentinel_threshold, 100), truth, observable)
            )
            group = str(batch["group_id"][index])
            groups.append(group)
            folds.append(group_to_fold[group])
            sensors.append(int(batch["sensor_index"][index].item()))
        if batch_index % 100 == 0:
            print(json.dumps({"batches": batch_index, "rows": len(folds)}), flush=True)

    baseline = np.stack(baseline_counts)
    sentinel = np.stack(sentinel_counts)
    fold_array = np.asarray(folds, dtype=np.uint8)
    sensor_array = np.asarray(sensors, dtype=np.uint8)
    group_array = np.asarray(groups)
    candidate = route_sensor_counts(baseline, sentinel, sensor_array)
    partition_folds = {
        "selection_folds_2_3_4": SELECTION_FOLDS,
        "confirmation_folds_0_1": CONFIRMATION_FOLDS,
        "all_five_folds": tuple(range(5)),
    }
    partitions = {}
    for index, (name, selected_folds) in enumerate(partition_folds.items()):
        rows = np.isin(fold_array, selected_folds)
        partitions[name] = summarize_partition(
            rows,
            baseline,
            candidate,
            fold_array,
            sensor_array,
            group_array,
            replicates=args.bootstrap_replicates,
            seed=args.bootstrap_seed + index,
            confidence=args.confidence,
        )
    passed = all(all(value["checks"].values()) for value in partitions.values())
    report = {
        "schema_version": 1,
        "scope": "all development folds; paper test not loaded",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "rule": {"Sentinel-2": args.sentinel_threshold, "Landsat": args.baseline_threshold},
        "baseline_threshold": args.baseline_threshold,
        "minimum_connected_pixels": args.minimum_connected_pixels,
        "partitions": partitions,
        "all_selection_confirmation_and_domain_gates_pass": passed,
        "decision": (
            "Advance the frozen sensor-specific mask rule to a transparent post-test benchmark evaluation."
            if passed
            else "Reject the sensor-specific mask rule because at least one development confirmation gate failed."
        ),
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
        "passed": passed,
        "partitions": {
            name: {
                "delta": value["delta"],
                "lower": value["paired_group_bootstrap_delta"]["lower"],
            }
            for name, value in partitions.items()
        },
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
