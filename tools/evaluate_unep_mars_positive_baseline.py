#!/usr/bin/env python3
"""Score frozen MARS-S2L mask endpoints on positive-only UNEP exact crops."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import scipy
import torch
from scipy import ndimage
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "EarthRemoteSensingRapidResponse"
for path in (MODEL_ROOT, ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from acquire_mars_metadata import sha256  # noqa: E402
from evaluate_released_marss2l import connected_scene_score  # noqa: E402
from mars_paper_model import MarsPaperResidualModel  # noqa: E402
from mars_s2l_adapter import iter_manifest  # noqa: E402
from train_mars_paper_residual import MarsPaperDataset, move_batch  # noqa: E402


DEFAULT_PROTOCOL = Path("configs/unep_mars_post2024_baseline_protocol.json")
DEFAULT_JSON = Path("reports/experiments/unep_mars_post2024_positive_baseline.json")
DEFAULT_MARKDOWN = Path("reports/experiments/UNEP_MARS_POST2024_POSITIVE_BASELINE.md")


def retained_mask(probability: np.ndarray, threshold: float, minimum: int) -> np.ndarray:
    labels, count = ndimage.label(
        probability > threshold, structure=np.ones((3, 3), dtype=np.uint8)
    )
    if count == 0:
        return np.zeros(probability.shape, dtype=bool)
    sizes = np.bincount(labels.ravel())
    keep = sizes >= minimum
    keep[0] = False
    return keep[labels]


def aggregate(rows: list[dict[str, Any]], endpoint: str) -> dict[str, float | int]:
    detected = sum(bool(row[endpoint]["detected"]) for row in rows)
    intersection = sum(int(row[endpoint]["intersection"]) for row in rows)
    predicted = sum(int(row[endpoint]["predicted"]) for row in rows)
    truth = sum(int(row[endpoint]["truth"]) for row in rows)
    union = predicted + truth - intersection
    total = predicted + truth
    return {
        "rows": len(rows),
        "detected": detected,
        "positive_recall": detected / len(rows),
        "intersection_pixels": intersection,
        "predicted_positive_pixels": predicted,
        "truth_positive_pixels": truth,
        "pixel_iou": 0.0 if union == 0 else intersection / union,
        "pixel_dice": 0.0 if total == 0 else 2.0 * intersection / total,
    }


def group_bootstrap(
    rows: list[dict[str, Any]],
    endpoint: str,
    *,
    replicates: int,
    seed: int,
    confidence: float,
) -> dict[str, Any]:
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_group[str(row["group_id"])].append(row)
    groups = sorted(by_group)
    rng = np.random.default_rng(seed)
    recall = np.empty(replicates, dtype=np.float64)
    iou = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        selected = [row for group in sampled for row in by_group[str(group)]]
        result = aggregate(selected, endpoint)
        recall[index] = float(result["positive_recall"])
        iou[index] = float(result["pixel_iou"])
    alpha = (1.0 - confidence) / 2.0

    def interval(values: np.ndarray) -> dict[str, float]:
        return {
            "lower": float(np.quantile(values, alpha)),
            "median": float(np.median(values)),
            "upper": float(np.quantile(values, 1.0 - alpha)),
        }

    return {
        "groups": len(groups),
        "replicates": replicates,
        "confidence": confidence,
        "positive_recall": interval(recall),
        "pixel_iou": interval(iou),
    }


def summarize(
    rows: list[dict[str, Any]], protocol: dict[str, Any], role_index: int
) -> dict[str, Any]:
    endpoints: dict[str, Any] = {}
    uncertainty = protocol["metrics"]["uncertainty"]
    for endpoint in protocol["endpoints"]:
        endpoints[endpoint] = {
            "metrics": aggregate(rows, endpoint),
            "bootstrap": group_bootstrap(
                rows,
                endpoint,
                replicates=int(uncertainty["replicates"]),
                seed=int(uncertainty["seed"]) + role_index,
                confidence=float(uncertainty["confidence"]),
            ),
        }
    scores = np.asarray([row["connected_scene_score"] for row in rows])
    return {
        "rows": len(rows),
        "groups": len({row["group_id"] for row in rows}),
        "connected_scene_score": {
            "minimum": float(np.min(scores)),
            "q05": float(np.quantile(scores, 0.05)),
            "median": float(np.median(scores)),
            "q95": float(np.quantile(scores, 0.95)),
            "maximum": float(np.max(scores)),
        },
        "endpoints": endpoints,
    }


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# UNEP MARS post-2024 positive baseline",
        "",
        "This positive-only result supports recall and mask-overlap conclusions only. It cannot estimate AP, false-positive rate, precision, specificity, or AUROC.",
        "",
        "| Role | Endpoint | Rows | Groups | Recall | 95% group CI | Pixel IoU | 95% group CI |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for role, result in report["roles"].items():
        for endpoint, endpoint_result in result["endpoints"].items():
            metrics = endpoint_result["metrics"]
            bootstrap = endpoint_result["bootstrap"]
            recall = bootstrap["positive_recall"]
            iou = bootstrap["pixel_iou"]
            lines.append(
                f"| {role} | {endpoint} | {result['rows']} | {result['groups']} | "
                f"{metrics['positive_recall']:.4f} | [{recall['lower']:.4f}, {recall['upper']:.4f}] | "
                f"{metrics['pixel_iou']:.4f} | [{iou['lower']:.4f}, {iou['upper']:.4f}] |"
            )
    lines.extend(
        [
            "",
            "The auxiliary rows are a training-domain diagnostic. The four development groups are isolated confirmation and did not select either endpoint.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


@torch.no_grad()
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL.as_posix())
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output-json", default=DEFAULT_JSON.as_posix())
    parser.add_argument("--output-markdown", default=DEFAULT_MARKDOWN.as_posix())
    args = parser.parse_args()
    if args.batch_size <= 0 or args.workers < 0:
        parser.error("batch size must be positive and workers non-negative")

    root = ROOT
    protocol_path = (root / args.protocol).resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    checkpoint = (root / protocol["model"]["checkpoint"]).resolve()
    if sha256(checkpoint) != protocol["model"]["checkpoint_sha256"]:
        raise ValueError("Released checkpoint differs from frozen protocol")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MarsPaperResidualModel().to(device)
    model.load_released_checkpoint(checkpoint)
    model.eval()
    role_rows: dict[str, list[dict[str, Any]]] = {}
    for role_index, (role, manifest_record) in enumerate(protocol["manifests"].items()):
        manifest = (root / manifest_record["path"]).resolve()
        if sha256(manifest) != manifest_record["sha256"]:
            raise ValueError(f"{role} manifest differs from frozen protocol")
        records = list(iter_manifest(manifest))
        if len(records) != int(manifest_record["rows"]):
            raise ValueError(f"{role} row count differs from frozen protocol")
        dataset = MarsPaperDataset(root, records, augment=False, seed=0)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
            persistent_workers=args.workers > 0,
        )
        observations: list[dict[str, Any]] = []
        for batch in loader:
            batch = move_batch(batch, device)
            with torch.amp.autocast(
                "cuda", dtype=torch.float16, enabled=device.type == "cuda"
            ):
                output = model(
                    batch["inputs"], batch["observable"], batch["sensor_index"]
                )
            probability = torch.sigmoid(output["baseline_logits"]).float()
            probability = probability.masked_fill(~(batch["clear"] > 0.5), 0.0)
            for index in range(probability.shape[0]):
                local = probability[index, 0].cpu().numpy()
                observable = batch["observable"][index, 0].cpu().numpy() > 0.5
                truth = (batch["mask"][index, 0].cpu().numpy() > 0.5) & observable
                row: dict[str, Any] = {
                    "sample_id": str(batch["sample_id"][index]),
                    "group_id": str(batch["group_id"][index]),
                    "connected_scene_score": connected_scene_score(local),
                }
                for endpoint, endpoint_protocol in protocol["endpoints"].items():
                    prediction = retained_mask(
                        local,
                        float(endpoint_protocol["pixel_probability_threshold"]),
                        int(endpoint_protocol["minimum_connected_pixels"]),
                    )
                    row[endpoint] = {
                        "detected": bool(np.any(prediction)),
                        "intersection": int(np.count_nonzero(prediction & truth)),
                        "predicted": int(np.count_nonzero(prediction & observable)),
                        "truth": int(np.count_nonzero(truth)),
                    }
                observations.append(row)
        role_rows[role] = observations
        print(json.dumps({"role": role, "rows": len(observations)}), flush=True)

    report = {
        "schema_version": 1,
        "scope": protocol["scope"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "limitations": protocol["metrics"]["prohibited"],
        "roles": {
            role: summarize(rows, protocol, index)
            for index, (role, rows) in enumerate(role_rows.items())
        },
        "provenance": {
            "protocol_sha256": sha256(protocol_path),
            "checkpoint_sha256": sha256(checkpoint),
            "script_sha256": sha256(Path(__file__).resolve()),
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True
            ).strip(),
            "device": str(
                torch.cuda.get_device_name(device) if device.type == "cuda" else device
            ),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
    }
    output_json = (root / args.output_json).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_json.with_suffix(output_json.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output_json)
    write_markdown((root / args.output_markdown).resolve(), report)
    print(json.dumps({"ok": True, "roles": list(report["roles"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
