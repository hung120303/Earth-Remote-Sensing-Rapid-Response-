#!/usr/bin/env python3
"""Select a robust released-logit mask threshold on development folds only."""

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
import rasterio
import scipy
import torch
from scipy import ndimage
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "EarthRemoteSensingRapidResponse"
if str(MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_ROOT))

from acquire_mars_metadata import DEFAULT_OUTPUT, repo_root, sha256
from mars_paper_model import ReleasedMarsUNet, SENSOR_NAMES, released_state
from mars_s2l_adapter import iter_development_manifest
from train_mars_paper_residual import (
    DEFAULT_ACQUISITION_RECEIPT,
    DEFAULT_CHECKPOINT,
    DEFAULT_MANIFEST,
    DEFAULT_PROTOCOL,
    MarsPaperDataset,
    move_batch,
    verify_acquisition_receipt,
)

DEFAULT_THRESHOLDS = (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8)
DEFAULT_JSON = Path("reports/experiments/mars_mask_threshold_development.json")
DEFAULT_MARKDOWN = Path("reports/experiments/MARS_MASK_THRESHOLD_DEVELOPMENT.md")


def key(value: float) -> str:
    return format(value, ".8g")


def component_mask_at(
    score: np.ndarray, threshold: float, minimum_connected_pixels: int
) -> np.ndarray:
    labels, count = ndimage.label(
        score > threshold, structure=np.ones((3, 3), dtype=np.uint8)
    )
    if count == 0:
        return np.zeros(score.shape, dtype=bool)
    sizes = np.bincount(labels.ravel())
    keep = sizes >= minimum_connected_pixels
    keep[0] = False
    return keep[labels]


def accumulator() -> dict[str, int]:
    return {"intersection": 0, "predicted": 0, "truth": 0}


def add(
    target: dict[str, int], prediction: np.ndarray, truth: np.ndarray, observable: np.ndarray
) -> None:
    target["intersection"] += int(np.count_nonzero(prediction & truth))
    target["predicted"] += int(np.count_nonzero(prediction & observable))
    target["truth"] += int(np.count_nonzero(truth))


def finish(target: dict[str, int]) -> dict[str, int | float]:
    union = target["predicted"] + target["truth"] - target["intersection"]
    return {
        **target,
        "union": union,
        "intersection_over_union": 0.0 if union == 0 else target["intersection"] / union,
    }


def choose_threshold(
    summaries: dict[str, dict[str, Any]], baseline_key: str
) -> tuple[str, dict[str, Any]]:
    candidates = [
        (threshold, summary)
        for threshold, summary in summaries.items()
        if threshold != baseline_key
    ]
    if not candidates:
        raise ValueError("At least one non-baseline threshold is required")

    def rank(item: tuple[str, dict[str, Any]]) -> tuple[float, float, float]:
        summary = item[1]
        fold_deltas = [float(value["delta"]) for value in summary["folds"].values()]
        sensor_deltas = [
            float(value["delta"]) for value in summary["sensors"].values()
        ]
        return min(fold_deltas), min(sensor_deltas), float(summary["delta"])

    return max(candidates, key=rank)


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Development-only mask-threshold analysis",
        "",
        "The paper test was not loaded. Scene ranking is unchanged; this analysis only calibrates the dense-mask decision rule.",
        "",
        "| Threshold | Pooled IoU | Delta | Worst fold delta | Worst sensor delta |",
        "|---:|---:|---:|---:|---:|",
    ]
    for threshold, summary in report["thresholds"].items():
        lines.append(
            f"| {float(threshold):.4f} | {summary['pooled']['intersection_over_union']:.5f} | "
            f"{summary['delta']:+.5f} | "
            f"{min(value['delta'] for value in summary['folds'].values()):+.5f} | "
            f"{min(value['delta'] for value in summary['sensors'].values()):+.5f} |"
        )
    lines.extend(
        [
            "",
            f"Selected threshold: **{report['selected_threshold']:.4f}**.",
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
    parser.add_argument("--thresholds", type=float, nargs="+", default=list(DEFAULT_THRESHOLDS))
    parser.add_argument("--minimum-connected-pixels", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output-json", default=DEFAULT_JSON.as_posix())
    parser.add_argument("--output-markdown", default=DEFAULT_MARKDOWN.as_posix())
    args = parser.parse_args()
    folds = tuple(args.folds)
    thresholds = tuple(args.thresholds)
    if len(set(folds)) != len(folds) or any(not 0 <= value < 5 for value in folds):
        parser.error("folds must be unique values in [0,4]")
    if len(set(thresholds)) != len(thresholds) or 0.5 not in thresholds or any(
        not 0.0 < value < 1.0 for value in thresholds
    ):
        parser.error("thresholds must be unique values in (0,1) and include 0.5")
    if args.minimum_connected_pixels <= 0 or args.batch_size <= 0 or args.workers < 0:
        parser.error("area/batch must be positive and workers non-negative")

    root = repo_root()
    metadata_dir = (root / args.metadata_dir).resolve()
    manifest = (root / args.manifest).resolve()
    protocol_path = (root / args.protocol).resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    manifest_hash = sha256(manifest)
    if manifest_hash != protocol["development_manifest_sha256"]:
        raise ValueError("Development manifest differs from the frozen fold protocol")
    verify_acquisition_receipt((root / args.acquisition_receipt).resolve(), manifest_hash)
    group_to_fold = {
        str(item["group_id"]): int(item["fold"]) for item in protocol["assignments"]
    }
    records = list(iter_development_manifest(manifest))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ReleasedMarsUNet().to(device)
    model.load_state_dict(
        released_state((root / args.released_checkpoint).resolve()), strict=False
    )
    model.eval()
    states = {
        key(threshold): {
            "pooled": accumulator(),
            "folds": {str(fold): accumulator() for fold in folds},
            "sensors": {name: accumulator() for name in SENSOR_NAMES},
        }
        for threshold in thresholds
    }
    fold_rows: dict[str, int] = {}
    for fold in folds:
        held_out = [
            record for record in records if group_to_fold[str(record["group_id"])] == fold
        ]
        fold_rows[str(fold)] = len(held_out)
        loader = DataLoader(
            MarsPaperDataset(metadata_dir, held_out, augment=False, seed=0),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
            persistent_workers=args.workers > 0,
        )
        for batch in loader:
            batch = move_batch(batch, device)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(batch["inputs"])
            probabilities = torch.sigmoid(logits).float().masked_fill(
                batch["clear"] <= 0.5, 0.0
            )
            for index in range(probabilities.shape[0]):
                score = probabilities[index, 0].cpu().numpy()
                observable = batch["observable"][index, 0].cpu().numpy() > 0.5
                truth = (batch["mask"][index, 0].cpu().numpy() > 0.5) & observable
                sensor = SENSOR_NAMES[int(batch["sensor_index"][index].item())]
                for threshold in thresholds:
                    prediction = component_mask_at(
                        score, threshold, args.minimum_connected_pixels
                    )
                    state = states[key(threshold)]
                    add(state["pooled"], prediction, truth, observable)
                    add(state["folds"][str(fold)], prediction, truth, observable)
                    add(state["sensors"][sensor], prediction, truth, observable)

    baseline_key = key(0.5)
    baseline = states[baseline_key]
    summaries: dict[str, dict[str, Any]] = {}
    for threshold in thresholds:
        threshold_key = key(threshold)
        state = states[threshold_key]
        pooled = finish(state["pooled"])
        folds_summary = {
            fold: {
                **finish(value),
                "delta": finish(value)["intersection_over_union"]
                - finish(baseline["folds"][fold])["intersection_over_union"],
            }
            for fold, value in state["folds"].items()
        }
        sensors_summary = {
            sensor: {
                **finish(value),
                "delta": finish(value)["intersection_over_union"]
                - finish(baseline["sensors"][sensor])["intersection_over_union"],
            }
            for sensor, value in state["sensors"].items()
        }
        summaries[threshold_key] = {
            "pooled": pooled,
            "delta": pooled["intersection_over_union"]
            - finish(baseline["pooled"])["intersection_over_union"],
            "folds": folds_summary,
            "sensors": sensors_summary,
        }
    selected_key, selected = choose_threshold(summaries, baseline_key)
    passed = (
        selected["delta"] > 0
        and all(value["delta"] > 0 for value in selected["folds"].values())
        and all(value["delta"] > 0 for value in selected["sensors"].values())
    )
    report = {
        "schema_version": 1,
        "scope": "development folds only; paper test not loaded",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "folds": list(folds),
        "fold_rows": fold_rows,
        "minimum_connected_pixels": args.minimum_connected_pixels,
        "thresholds": summaries,
        "selected_threshold": float(selected_key),
        "selected_passes_all_fold_and_sensor_gates": passed,
        "decision": (
            "Advance this mask threshold to a separate development-fold confirmation."
            if passed
            else "Do not advance this mask rule; at least one fold or sensor IoU gate failed."
        ),
        "provenance": {
            "manifest_sha256": manifest_hash,
            "protocol_sha256": sha256(protocol_path),
            "checkpoint_sha256": sha256((root / args.released_checkpoint).resolve()),
            "script_sha256": sha256(Path(__file__).resolve()),
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
            "device": str(torch.cuda.get_device_name(device) if device.type == "cuda" else device),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "rasterio": rasterio.__version__,
        },
    }
    write_json((root / args.output_json).resolve(), report)
    write_markdown((root / args.output_markdown).resolve(), report)
    print(json.dumps({"ok": True, "selected_threshold": float(selected_key), "passed": passed, "decision": report["decision"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
