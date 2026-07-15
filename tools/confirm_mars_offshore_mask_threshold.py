#!/usr/bin/env python3
"""Reverse-validate an offshore 0.90 mask threshold over the v2 sensor rule."""

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
from analyze_mars_mask_routing import iou, paired_group_bootstrap, pixel_counts  # noqa: E402
from analyze_mars_mask_thresholds import component_mask_at  # noqa: E402
from confirm_mars_sensor_mask_thresholds import route_sensor_counts  # noqa: E402
from mars_paper_model import ReleasedMarsUNet  # noqa: E402
from mars_paper_model import released_state  # noqa: E402
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

DEFAULT_JSON = Path("reports/experiments/mars_offshore_mask_threshold_confirmation.json")
DEFAULT_MARKDOWN = Path("reports/experiments/MARS_OFFSHORE_MASK_THRESHOLD_CONFIRMATION.md")
DEFAULT_CACHE = Path("outputs/mars_offshore_mask_threshold_development_counts.npz")
SELECTION_FOLDS = (2, 3, 4)
CONFIRMATION_FOLDS = (0, 1)


def route_offshore_counts(
    baseline: np.ndarray, offshore_counts: np.ndarray, offshore: np.ndarray
) -> np.ndarray:
    if baseline.shape != offshore_counts.shape or baseline.ndim != 2 or baseline.shape[1] != 3:
        raise ValueError("baseline and offshore counts must be matching Nx3 arrays")
    if offshore.shape != (baseline.shape[0],):
        raise ValueError("offshore flags must align one-to-one with count rows")
    return np.where(offshore[:, None], offshore_counts, baseline)


def domain_summary(
    rows: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    offshore: np.ndarray,
) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for name, domain_rows in (
        ("onshore", rows & ~offshore),
        ("offshore", rows & offshore),
    ):
        if not np.any(domain_rows):
            values[name] = {"rows": 0, "baseline_iou": None, "candidate_iou": None, "delta": 0.0}
            continue
        base_iou = iou(baseline[domain_rows])
        candidate_iou = iou(candidate[domain_rows])
        values[name] = {
            "rows": int(domain_rows.sum()),
            "baseline_iou": base_iou,
            "candidate_iou": candidate_iou,
            "delta": candidate_iou - base_iou,
        }
    return values


def summarize_partition(
    rows: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    folds: np.ndarray,
    groups: np.ndarray,
    offshore: np.ndarray,
    labels: np.ndarray,
    *,
    replicates: int,
    seed: int,
    confidence: float,
    require_positive_interval: bool,
) -> dict[str, Any]:
    base_iou = iou(baseline[rows])
    candidate_iou = iou(candidate[rows])
    by_fold: dict[str, Any] = {}
    for fold in np.unique(folds[rows]):
        local = rows & (folds == fold)
        local_base = iou(baseline[local])
        local_candidate = iou(candidate[local])
        by_fold[str(int(fold))] = {
            "rows": int(local.sum()),
            "baseline_iou": local_base,
            "candidate_iou": local_candidate,
            "delta": local_candidate - local_base,
        }
    bootstrap = paired_group_bootstrap(
        baseline[rows],
        candidate[rows],
        groups[rows],
        replicates=replicates,
        seed=seed,
        confidence=confidence,
    )
    domains = domain_summary(rows, baseline, candidate, offshore)
    offshore_positive = int(labels[rows & offshore].sum())
    checks = {
        "point_nonnegative": candidate_iou >= base_iou,
        "all_fold_deltas_nonnegative": all(value["delta"] >= 0.0 for value in by_fold.values()),
        "onshore_unchanged": abs(domains["onshore"]["delta"]) < 1e-15,
        "offshore_nonnegative": domains["offshore"]["delta"] >= 0.0,
        "offshore_positive_when_truth_present": (
            offshore_positive == 0 or domains["offshore"]["delta"] > 0.0
        ),
        "bootstrap_interval_acceptable": (
            bootstrap["lower"] > 0.0 if require_positive_interval else bootstrap["lower"] >= 0.0
        ),
    }
    return {
        "rows": int(rows.sum()),
        "positive": int(labels[rows].sum()),
        "offshore_rows": int((rows & offshore).sum()),
        "offshore_positive": offshore_positive,
        "baseline_iou": base_iou,
        "candidate_iou": candidate_iou,
        "delta": candidate_iou - base_iou,
        "folds": by_fold,
        "domains": domains,
        "paired_group_bootstrap_delta": bootstrap,
        "checks": checks,
        "passed": all(checks.values()),
    }


def write_json(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Offshore MARS mask-threshold reverse validation",
        "",
        "Baseline: Sentinel-2 0.80 / Landsat 0.70. Candidate: offshore scenes use 0.90. The paper test was not loaded.",
        "",
        "| Partition | Rows | Offshore positive | Baseline IoU | Candidate IoU | Delta | 95% CI | Gates |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for name, value in report["partitions"].items():
        interval = value["paired_group_bootstrap_delta"]
        lines.append(
            f"| {name} | {value['rows']:,} | {value['offshore_positive']} | "
            f"{value['baseline_iou']:.5f} | {value['candidate_iou']:.5f} | "
            f"{value['delta']:+.5f} | [{interval['lower']:+.5f}, {interval['upper']:+.5f}] | "
            f"{'PASS' if value['passed'] else 'FAIL'} |"
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
    parser.add_argument("--landsat-threshold", type=float, default=0.7)
    parser.add_argument("--sentinel-threshold", type=float, default=0.8)
    parser.add_argument("--offshore-threshold", type=float, default=0.9)
    parser.add_argument("--minimum-connected-pixels", type=int, default=100)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260750)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output-json", default=DEFAULT_JSON.as_posix())
    parser.add_argument("--output-markdown", default=DEFAULT_MARKDOWN.as_posix())
    parser.add_argument("--output-cache", default=DEFAULT_CACHE.as_posix())
    args = parser.parse_args()
    if args.minimum_connected_pixels != 100:
        parser.error("the frozen paper contract requires 100 connected pixels")
    if not 0.0 < args.landsat_threshold < args.sentinel_threshold < args.offshore_threshold < 1.0:
        parser.error("thresholds must satisfy 0 < Landsat < Sentinel-2 < offshore < 1")
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
    group_offshore: dict[str, bool] = {}
    for record in records:
        group = str(record["group_id"])
        value = str(record.get("country", "")) == "Offshore"
        if group in group_offshore and group_offshore[group] != value:
            raise ValueError("Development group spans onshore and offshore records")
        group_offshore[group] = value
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

    landsat_counts: list[np.ndarray] = []
    sentinel_counts: list[np.ndarray] = []
    offshore_counts: list[np.ndarray] = []
    folds: list[int] = []
    sensors: list[int] = []
    groups: list[str] = []
    offshore_flags: list[bool] = []
    labels: list[int] = []
    thresholds = (args.landsat_threshold, args.sentinel_threshold, args.offshore_threshold)
    count_lists = (landsat_counts, sentinel_counts, offshore_counts)
    for batch_index, batch in enumerate(loader, start=1):
        batch = move_batch(batch, device)
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(batch["inputs"])
        probabilities = (
            torch.sigmoid(logits).float().masked_fill(batch["clear"] <= 0.5, 0.0).cpu().numpy()
        )
        for index in range(probabilities.shape[0]):
            score = probabilities[index, 0]
            observable = batch["observable"][index, 0].cpu().numpy() > 0.5
            truth = (batch["mask"][index, 0].cpu().numpy() > 0.5) & observable
            for threshold, destination in zip(thresholds, count_lists):
                destination.append(
                    pixel_counts(component_mask_at(score, threshold, 100), truth, observable)
                )
            group = str(batch["group_id"][index])
            groups.append(group)
            folds.append(group_to_fold[group])
            sensors.append(int(batch["sensor_index"][index].item()))
            offshore_flags.append(group_offshore[group])
            labels.append(int(batch["presence"][index].item()))
        if batch_index % 100 == 0:
            print(json.dumps({"batches": batch_index, "rows": len(folds)}), flush=True)

    landsat = np.stack(landsat_counts)
    sentinel = np.stack(sentinel_counts)
    threshold_09 = np.stack(offshore_counts)
    fold_array = np.asarray(folds, dtype=np.uint8)
    sensor_array = np.asarray(sensors, dtype=np.uint8)
    group_array = np.asarray(groups)
    offshore_array = np.asarray(offshore_flags, dtype=bool)
    label_array = np.asarray(labels, dtype=np.uint8)
    baseline = route_sensor_counts(landsat, sentinel, sensor_array)
    candidate = route_offshore_counts(baseline, threshold_09, offshore_array)
    cache_path = (root / args.output_cache).resolve()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_cache = cache_path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary_cache,
        threshold_07=landsat,
        threshold_08=sentinel,
        threshold_09=threshold_09,
        folds=fold_array,
        sensors=sensor_array,
        groups=group_array,
        offshore=offshore_array,
        labels=label_array,
    )
    os.replace(temporary_cache, cache_path)

    partition_specs = {
        "selection_folds_2_3_4": (SELECTION_FOLDS, True),
        "confirmation_folds_0_1": (CONFIRMATION_FOLDS, False),
        "all_five_folds": (tuple(range(5)), True),
    }
    partitions: dict[str, Any] = {}
    for index, (name, (selected_folds, require_positive)) in enumerate(partition_specs.items()):
        rows = np.isin(fold_array, selected_folds)
        partitions[name] = summarize_partition(
            rows,
            baseline,
            candidate,
            fold_array,
            group_array,
            offshore_array,
            label_array,
            replicates=args.bootstrap_replicates,
            seed=args.bootstrap_seed + index,
            confidence=args.confidence,
            require_positive_interval=require_positive,
        )
    passed = all(value["passed"] for value in partitions.values())
    report = {
        "schema_version": 1,
        "scope": "post-test-diagnosed mask rule reverse-validated on all development folds",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "baseline_rule": {"Sentinel-2": args.sentinel_threshold, "Landsat": args.landsat_threshold},
        "candidate_rule": {
            "Sentinel-2": args.sentinel_threshold,
            "Landsat": args.landsat_threshold,
            "offshore_all_sensors": args.offshore_threshold,
        },
        "minimum_connected_pixels": args.minimum_connected_pixels,
        "partitions": partitions,
        "all_development_reverse_validation_gates_pass": passed,
        "decision": (
            "Development evidence supports the offshore 0.90 mask threshold for transparent post-test evaluation."
            if passed
            else "Reject the offshore 0.90 mask threshold because development reverse validation failed."
        ),
        "provenance": {
            "manifest_sha256": manifest_hash,
            "protocol_sha256": sha256(protocol_path),
            "checkpoint_sha256": sha256(checkpoint),
            "development_count_cache_sha256": sha256(cache_path),
            "script_sha256": sha256(Path(__file__).resolve()),
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True
            ).strip(),
            "device": str(torch.cuda.get_device_name(device) if device.type == "cuda" else device),
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
    }
    write_json((root / args.output_json).resolve(), report)
    write_markdown((root / args.output_markdown).resolve(), report)
    print(
        json.dumps(
            {
                "ok": passed,
                "partitions": {
                    name: {
                        "delta": value["delta"],
                        "lower": value["paired_group_bootstrap_delta"]["lower"],
                        "offshore_positive": value["offshore_positive"],
                        "passed": value["passed"],
                    }
                    for name, value in partitions.items()
                },
            },
            indent=2,
        )
    )
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
