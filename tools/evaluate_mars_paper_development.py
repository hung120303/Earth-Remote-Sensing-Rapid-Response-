#!/usr/bin/env python3
"""Evaluate the released MARS-S2L baseline on frozen development site folds."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import scipy
import sklearn
import torch
from torch.utils.data import DataLoader

from acquire_mars_metadata import DEFAULT_OUTPUT, repo_root, sha256
from train_mars_paper_residual import (
    DEFAULT_ACQUISITION_RECEIPT,
    DEFAULT_CHECKPOINT,
    DEFAULT_MANIFEST,
    DEFAULT_PROTOCOL,
    MarsPaperDataset,
    MarsPaperResidualModel,
    iter_development_manifest,
    validation_summary,
    verify_acquisition_receipt,
)

DEFAULT_JSON = Path("reports/experiments/mars_paper_released_development.json")
DEFAULT_MARKDOWN = Path("reports/experiments/MARS_PAPER_RELEASED_DEVELOPMENT.md")


def assert_zero_residual_equivalence(summary: dict[str, Any]) -> None:
    deltas = [float(value) for value in summary["delta"].values()]
    for stratum in summary["sensor_strata"].values():
        if not stratum.get("eligible_for_promotion"):
            raise ValueError("A baseline sensor stratum is not evaluable")
        deltas.extend(float(value) for value in stratum["delta"].values())
    if any(value != 0.0 for value in deltas):
        raise RuntimeError(
            "Zero-initialized successor differs from the released baseline on a full fold"
        )


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Released MARS-S2L on frozen development folds",
        "",
        "This is development-only evidence. The sealed paper test was not loaded.",
        "",
        "| Fold | Scenes | Plume | Sites | AP | Recall at <=7.13% FPR | Pixel IoU |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for fold, result in report["folds"].items():
        baseline = result["released_baseline"]
        lines.append(
            f"| {fold} | {result['rows']:,} | {result['positive']:,} | "
            f"{result['sites']:,} | {baseline['average_precision']:.4f} | "
            f"{baseline['operating_points']['0.0713']['recall']:.4f} | "
            f"{baseline['pixel_fixed_0_5']['intersection_over_union']:.4f} |"
        )
    lines.extend(
        [
            "",
            "The zero-initialized successor matched the released logits and every reported metric exactly on each fold.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-dir", default=DEFAULT_OUTPUT.as_posix())
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST.as_posix())
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL.as_posix())
    parser.add_argument(
        "--acquisition-receipt", default=DEFAULT_ACQUISITION_RECEIPT.as_posix()
    )
    parser.add_argument("--released-checkpoint", default=DEFAULT_CHECKPOINT.as_posix())
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output-json", default=DEFAULT_JSON.as_posix())
    parser.add_argument("--output-markdown", default=DEFAULT_MARKDOWN.as_posix())
    args = parser.parse_args()
    if args.batch_size <= 0 or args.workers < 0:
        parser.error("batch size must be positive and workers non-negative")

    root = repo_root()
    metadata_dir = (root / args.metadata_dir).resolve()
    manifest = (root / args.manifest).resolve()
    protocol_path = (root / args.protocol).resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    manifest_hash = sha256(manifest)
    if manifest_hash != protocol["development_manifest_sha256"]:
        raise ValueError("Development manifest differs from the frozen protocol")
    verify_acquisition_receipt(
        (root / args.acquisition_receipt).resolve(), manifest_hash
    )
    n_folds = int(protocol["n_folds"])
    if len(set(args.folds)) != len(args.folds) or any(
        fold < 0 or fold >= n_folds for fold in args.folds
    ):
        parser.error("folds must be unique members of the frozen protocol")

    group_to_fold = {
        str(item["group_id"]): int(item["fold"])
        for item in protocol["assignments"]
    }
    records = list(iter_development_manifest(manifest))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MarsPaperResidualModel().to(device)
    model.load_released_checkpoint((root / args.released_checkpoint).resolve())
    fold_results: dict[str, Any] = {}
    for fold in args.folds:
        held_out = [
            record
            for record in records
            if group_to_fold[str(record["group_id"])] == fold
        ]
        dataset = MarsPaperDataset(
            metadata_dir, held_out, augment=False, seed=0
        )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
            persistent_workers=args.workers > 0,
        )
        summary = validation_summary(model, loader, device)
        assert_zero_residual_equivalence(summary)
        fold_results[str(fold)] = summary
        print(
            json.dumps(
                {
                    "fold": fold,
                    "rows": summary["rows"],
                    "average_precision": summary["released_baseline"][
                        "average_precision"
                    ],
                    "pixel_iou": summary["released_baseline"]["pixel_fixed_0_5"][
                        "intersection_over_union"
                    ],
                }
            ),
            flush=True,
        )

    report = {
        "schema_version": 1,
        "scope": "frozen site-held development; sealed paper test not loaded",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "folds": fold_results,
        "provenance": {
            "development_manifest_sha256": manifest_hash,
            "protocol_sha256": sha256(protocol_path),
            "released_checkpoint_sha256": model.artifact_metadata()[
                "released_checkpoint_sha256"
            ],
            "script_sha256": sha256(Path(__file__).resolve()),
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True
            ).strip(),
            "device": str(torch.cuda.get_device_name(device) if device.type == "cuda" else device),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
            "rasterio": rasterio.__version__,
        },
    }
    output_json = (root / args.output_json).resolve()
    output_markdown = (root / args.output_markdown).resolve()
    write_json(output_json, report)
    write_markdown(output_markdown, report)
    print(json.dumps({"ok": True, "folds": args.folds}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
