#!/usr/bin/env python3
"""Evaluate the frozen spatial successor on the exact cached MARS paper contract."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "EarthRemoteSensingRapidResponse"
for path in (MODEL_ROOT, ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from acquire_mars_metadata import repo_root, sha256  # noqa: E402
from diagnose_mars_scene_stacker_paper_cache import aligned_indices, triplet  # noqa: E402
from evaluate_mars_successor_paper_test import bootstrap_view, view_metrics  # noqa: E402
from train_mars_scene_ranker import blend_scores  # noqa: E402
from train_mars_spatial_scene_classifier import predict_model  # noqa: E402

DEFAULT_DIAGNOSTIC = Path("outputs/mars_paper_test_v3_diagnostic_cache.npz")
DEFAULT_DIAGNOSTIC_SHA256 = "1624fddc0222f8ffc5137f557c7fc3e465d53b335c82cc8014711baa35bb94a1"
DEFAULT_IMAGES = Path("outputs/mars_paper_spatial_scene_inputs.npy")
DEFAULT_IMAGES_SHA256 = "7cce444a552c6c873c05ae3a972f62a8081fde89d31ac63d94303dbfac3b1b94"
DEFAULT_METADATA = Path("outputs/mars_paper_spatial_scene_inputs_metadata.npz")
DEFAULT_METADATA_SHA256 = "b2b58eabf4478b46912d3c3437c1ee0b039841ee283bd7fe0164775b9d448022"
DEFAULT_ARTIFACT = Path("EarthRemoteSensingRapidResponse/artifacts/mars_spatial_scene_classifier.pt")
DEFAULT_ARTIFACT_SHA256 = "36135b7c8f9538f3ce7b896df0c2b767ee85b81d57e2de3eede2cf33384730c3"
DEFAULT_SELECTION = Path("reports/experiments/mars_spatial_scene_classifier.json")
DEFAULT_SELECTION_SHA256 = "dfeba0d4e8dde28ae880077c0db2010ccda5c57f58d293f4d3f834b541605c22"
DEFAULT_JSON = Path("reports/experiments/mars_spatial_successor_paper_posttest.json")
DEFAULT_MARKDOWN = Path("reports/experiments/MARS_SPATIAL_SUCCESSOR_PAPER_POSTTEST.md")


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Spatial successor: exact MARS-S2L paper benchmark",
        "",
        "Transparent post-test development evaluation; it is not an untouched confirmation cohort.",
        "",
        "| View | Candidate AP | AP delta | AP 95% CI | Recall delta | Recall 95% CI | IoU delta | IoU 95% CI | Gates |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for name, value in report["views"].items():
        metrics = value["metrics"]
        intervals = value["bootstrap"]["delta_intervals"]
        lines.append(
            f"| {name} | {metrics['candidate']['average_precision']:.5f} | "
            f"{metrics['delta']['average_precision']:+.5f} | "
            f"[{intervals['average_precision']['lower']:+.5f}, {intervals['average_precision']['upper']:+.5f}] | "
            f"{metrics['delta']['matched_fpr_recall']:+.5f} | "
            f"[{intervals['matched_fpr_recall']['lower']:+.5f}, {intervals['matched_fpr_recall']['upper']:+.5f}] | "
            f"{metrics['delta']['pixel_iou']:+.5f} | "
            f"[{intervals['pixel_iou']['lower']:+.5f}, {intervals['pixel_iou']['upper']:+.5f}] | "
            f"{'PASS' if value['passed'] else 'FAIL'} |"
        )
    lines.extend(["", report["decision"]])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic", default=DEFAULT_DIAGNOSTIC.as_posix())
    parser.add_argument("--diagnostic-sha256", default=DEFAULT_DIAGNOSTIC_SHA256)
    parser.add_argument("--images", default=DEFAULT_IMAGES.as_posix())
    parser.add_argument("--images-sha256", default=DEFAULT_IMAGES_SHA256)
    parser.add_argument("--metadata", default=DEFAULT_METADATA.as_posix())
    parser.add_argument("--metadata-sha256", default=DEFAULT_METADATA_SHA256)
    parser.add_argument("--artifact", default=DEFAULT_ARTIFACT.as_posix())
    parser.add_argument("--artifact-sha256", default=DEFAULT_ARTIFACT_SHA256)
    parser.add_argument("--selection", default=DEFAULT_SELECTION.as_posix())
    parser.add_argument("--selection-sha256", default=DEFAULT_SELECTION_SHA256)
    parser.add_argument("--replicates", type=int, default=10000)
    parser.add_argument("--output-json", default=DEFAULT_JSON.as_posix())
    parser.add_argument("--output-markdown", default=DEFAULT_MARKDOWN.as_posix())
    args = parser.parse_args()
    root = repo_root()
    paths = {
        "diagnostic": (root / args.diagnostic).resolve(),
        "images": (root / args.images).resolve(),
        "metadata": (root / args.metadata).resolve(),
        "artifact": (root / args.artifact).resolve(),
        "selection": (root / args.selection).resolve(),
    }
    expected = {
        "diagnostic": args.diagnostic_sha256,
        "images": args.images_sha256,
        "metadata": args.metadata_sha256,
        "artifact": args.artifact_sha256,
        "selection": args.selection_sha256,
    }
    for name, path in paths.items():
        if sha256(path) != expected[name]:
            raise ValueError(f"Frozen {name} hash mismatch")

    artifact = torch.load(paths["artifact"], map_location="cpu", weights_only=True)
    selection = json.loads(paths["selection"].read_text(encoding="utf-8"))
    if (
        float(artifact["blend_weight"]) != float(selection["selected"]["blend_weight"])
        or float(artifact["operational_scene_threshold"])
        != float(selection["operational_scene_threshold"])
    ):
        raise ValueError("Spatial artifact differs from its frozen development selection")
    images = np.load(paths["images"], mmap_mode="r", allow_pickle=False)
    with np.load(paths["metadata"], allow_pickle=False) as metadata:
        sample_ids = metadata["sample_ids"].astype(str)
        sensors = metadata["sensors"].astype(np.uint8)
        if str(metadata["images_sha256"].item()) != args.images_sha256:
            raise ValueError("Paper spatial metadata points to a different image cache")
        if "labels" in metadata.files:
            raise ValueError("Paper spatial metadata must remain label-independent")
    with np.load(paths["diagnostic"], allow_pickle=False) as diagnostic:
        values = {name: diagnostic[name] for name in diagnostic.files}
    indices = aligned_indices(values["aligned_sample_ids"], sample_ids)
    if not np.array_equal(values["sensors"][indices].astype(np.uint8), sensors):
        raise ValueError("Paper spatial sensor alignment failed")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw_scores = predict_model(
        artifact["fitted"], images, np.arange(images.shape[0]), sensors, device
    )
    candidate_scores = values["candidate_scores"].astype(np.float64).copy()
    candidate_scores[indices] = blend_scores(
        candidate_scores[indices], raw_scores, float(artifact["blend_weight"])
    )
    missing = np.ones(candidate_scores.shape, dtype=bool)
    missing[indices] = False

    labels = values["labels"].astype(np.uint8)
    sites = values["sites"].astype(str)
    baseline_scores = values["baseline_scores"].astype(np.float64)
    baseline_pixels = values["baseline_pixels"].astype(np.int64)
    candidate_pixels = values["candidate_pixels"].astype(np.int64)
    threshold = float(artifact["operational_scene_threshold"])
    selections = {
        "full": np.ones(labels.shape, dtype=bool),
        "test_only_sites": values["test_only"].astype(bool),
    }
    views: dict[str, Any] = {}
    for index, (name, rows) in enumerate(selections.items()):
        metrics = view_metrics(
            labels[rows],
            baseline_scores[rows],
            candidate_scores[rows],
            triplet(baseline_pixels[rows]),
            triplet(candidate_pixels[rows]),
            threshold,
        )
        bootstrap = bootstrap_view(
            labels=labels[rows],
            sites=sites[rows],
            baseline_scores=baseline_scores[rows],
            candidate_scores=candidate_scores[rows],
            baseline_predictions=baseline_scores[rows] > 0.5,
            candidate_predictions=candidate_scores[rows] > threshold,
            baseline_pixels=triplet(baseline_pixels[rows]),
            candidate_pixels=triplet(candidate_pixels[rows]),
            replicates=args.replicates,
            seed=20260960 + index,
            confidence=0.95,
        )
        intervals = bootstrap["delta_intervals"]
        checks = {
            "ap_point_higher": metrics["delta"]["average_precision"] > 0.0,
            "ap_lower_positive": intervals["average_precision"]["lower"] > 0.0,
            "matched_recall_point_higher": metrics["delta"]["matched_fpr_recall"] > 0.0,
            "matched_recall_lower_positive": intervals["matched_fpr_recall"]["lower"] > 0.0,
            "fixed_fpr_upper_nonpositive": intervals["fixed_false_positive_rate"]["upper"]
            <= 0.0,
            "pixel_iou_point_higher": metrics["delta"]["pixel_iou"] > 0.0,
            "pixel_iou_lower_positive": intervals["pixel_iou"]["lower"] > 0.0,
        }
        views[name] = {
            "metrics": metrics,
            "bootstrap": bootstrap,
            "checks": checks,
            "passed": all(checks.values()),
        }
    passed = all(value["passed"] for value in views.values())
    report = {
        "schema_version": 1,
        "scope": "transparent post-test development evaluation on exact paper rows and comparator",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "architecture": {
            "scene": "v3 stronger ExtraTrees head plus 0.10 physics-guided spatial morphology CNN",
            "segmentation": "v2 sensor-specific released-model masks",
            "operational_scene_threshold": threshold,
        },
        "available_spatial_rows": int(indices.size),
        "missing_rows_fallback_to_v3": int(missing.sum()),
        "views": views,
        "all_exact_paper_gates_pass": passed,
        "decision": (
            "All exact paper gates pass; independent future confirmation is still required."
            if passed
            else "At least one exact paper superiority gate remains unresolved."
        ),
        "provenance": {f"{name}_sha256": expected[name] for name in expected},
    }
    write_json((root / args.output_json).resolve(), report)
    write_markdown((root / args.output_markdown).resolve(), report)
    print(
        json.dumps(
            {
                "ok": passed,
                "views": {
                    name: {
                        "candidate_ap": value["metrics"]["candidate"]["average_precision"],
                        "ap_lower": value["bootstrap"]["delta_intervals"]["average_precision"][
                            "lower"
                        ],
                        "recall_lower": value["bootstrap"]["delta_intervals"][
                            "matched_fpr_recall"
                        ]["lower"],
                        "iou_lower": value["bootstrap"]["delta_intervals"]["pixel_iou"][
                            "lower"
                        ],
                        "passed": value["passed"],
                    }
                    for name, value in views.items()
                },
            },
            indent=2,
        )
    )
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
