#!/usr/bin/env python3
"""Evaluate the frozen site-relative spatial model on the exact MARS paper cache."""

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
from evaluate_mars_scene_gated_masks_paper_cache import gate_counts  # noqa: E402
from evaluate_mars_successor_paper_test import bootstrap_view, view_metrics  # noqa: E402
from train_mars_scene_ranker import blend_scores  # noqa: E402
from train_mars_site_relative_spatial_classifier import (  # noqa: E402
    build_site_templates,
    predict_model,
)

DEFAULT_DIAGNOSTIC = Path("outputs/mars_paper_test_v3_diagnostic_cache.npz")
DEFAULT_DIAGNOSTIC_SHA256 = "1624fddc0222f8ffc5137f557c7fc3e465d53b335c82cc8014711baa35bb94a1"
DEFAULT_IMAGES = Path("outputs/mars_paper_spatial_scene_inputs.npy")
DEFAULT_IMAGES_SHA256 = "7cce444a552c6c873c05ae3a972f62a8081fde89d31ac63d94303dbfac3b1b94"
DEFAULT_METADATA = Path("outputs/mars_paper_spatial_scene_inputs_metadata.npz")
DEFAULT_METADATA_SHA256 = "b2b58eabf4478b46912d3c3437c1ee0b039841ee283bd7fe0164775b9d448022"
DEFAULT_ARTIFACT = Path(
    "EarthRemoteSensingRapidResponse/artifacts/mars_site_relative_spatial_classifier.pt"
)
DEFAULT_ARTIFACT_SHA256 = "9401678aa4f38fb3b54a914318dc5c2a553b39fa8b309d1a127b9d21abbfc496"
DEFAULT_DEVELOPMENT_REPORT = Path(
    "reports/experiments/mars_site_relative_spatial_classifier.json"
)
DEFAULT_DEVELOPMENT_REPORT_SHA256 = (
    "c2b6695dbe24f0b5bc097d89c6fa6332c2af93441079cd55bec99e77aeeb820f"
)
DEFAULT_GATE_REPORT = Path("reports/experiments/mars_scene_gated_masks.json")
DEFAULT_GATE_REPORT_SHA256 = "c1e5a1497abebba80d42898a8165b30fd255ff252478a0ee1fd90fd32456a51c"
DEFAULT_JSON = Path(
    "reports/experiments/mars_site_relative_spatial_paper_posttest.json"
)
DEFAULT_MARKDOWN = Path(
    "reports/experiments/MARS_SITE_RELATIVE_SPATIAL_PAPER_POSTTEST.md"
)


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Site-relative spatial successor: exact MARS-S2L paper benchmark",
        "",
        "Transparent post-test replay; this is not an untouched confirmation cohort. Site templates use no labels, and the dense-mask gate remains driven by the unchanged v3 scene score.",
        "",
        "| View | AP | AP delta (95% CI) | Recall delta (95% CI) | FPR delta | IoU delta (95% CI) | Gates |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for name, value in report["views"].items():
        metrics = value["metrics"]
        intervals = value["bootstrap"]["delta_intervals"]
        lines.append(
            f"| {name} | {metrics['candidate']['average_precision']:.5f} | "
            f"{metrics['delta']['average_precision']:+.5f} "
            f"([{intervals['average_precision']['lower']:+.5f}, {intervals['average_precision']['upper']:+.5f}]) | "
            f"{metrics['delta']['matched_fpr_recall']:+.5f} "
            f"([{intervals['matched_fpr_recall']['lower']:+.5f}, {intervals['matched_fpr_recall']['upper']:+.5f}]) | "
            f"{metrics['delta']['fixed_false_positive_rate']:+.5f} | "
            f"{metrics['delta']['pixel_iou']:+.5f} "
            f"([{intervals['pixel_iou']['lower']:+.5f}, {intervals['pixel_iou']['upper']:+.5f}]) | "
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
    parser.add_argument(
        "--development-report", default=DEFAULT_DEVELOPMENT_REPORT.as_posix()
    )
    parser.add_argument(
        "--development-report-sha256", default=DEFAULT_DEVELOPMENT_REPORT_SHA256
    )
    parser.add_argument("--gate-report", default=DEFAULT_GATE_REPORT.as_posix())
    parser.add_argument("--gate-report-sha256", default=DEFAULT_GATE_REPORT_SHA256)
    parser.add_argument("--replicates", type=int, default=10_000)
    parser.add_argument("--output-json", default=DEFAULT_JSON.as_posix())
    parser.add_argument("--output-markdown", default=DEFAULT_MARKDOWN.as_posix())
    args = parser.parse_args()
    root = repo_root()
    paths = {
        "diagnostic": (root / args.diagnostic).resolve(),
        "images": (root / args.images).resolve(),
        "metadata": (root / args.metadata).resolve(),
        "artifact": (root / args.artifact).resolve(),
        "development_report": (root / args.development_report).resolve(),
        "gate_report": (root / args.gate_report).resolve(),
    }
    expected = {
        "diagnostic": args.diagnostic_sha256,
        "images": args.images_sha256,
        "metadata": args.metadata_sha256,
        "artifact": args.artifact_sha256,
        "development_report": args.development_report_sha256,
        "gate_report": args.gate_report_sha256,
    }
    for name, digest in expected.items():
        if sha256(paths[name]) != digest:
            raise ValueError(f"Frozen {name} hash mismatch")
    development = json.loads(paths["development_report"].read_text(encoding="utf-8"))
    if development.get("all_promotion_gates_pass") is not True:
        raise ValueError("Development site-relative model was not promoted")
    gate_report = json.loads(paths["gate_report"].read_text(encoding="utf-8"))
    if gate_report.get("all_selection_and_confirmation_gates_pass") is not True:
        raise ValueError("Development dense-mask gate was not promoted")
    cutoff = float(gate_report["selection"]["selected_cutoff"])
    artifact = torch.load(paths["artifact"], map_location="cpu", weights_only=True)
    if (
        float(artifact["blend_weight"])
        != float(development["selected"]["blend_weight"])
        or artifact["spec"] != development["selected"]["spec"]
        or float(artifact["operational_scene_threshold"])
        != float(development["operational_scene_threshold"])
    ):
        raise ValueError("Site-relative artifact differs from its development report")

    images = np.load(paths["images"], mmap_mode="r", allow_pickle=False)
    if images.shape != (43_524, 9, 64, 64) or images.dtype != np.float16:
        raise ValueError("Paper spatial image cache schema differs")
    with np.load(paths["metadata"], allow_pickle=False) as metadata:
        if "labels" in metadata.files:
            raise ValueError("Paper spatial metadata must remain label-independent")
        sample_ids = metadata["sample_ids"].astype(str)
        groups = metadata["groups"].astype(str)
        sensors = metadata["sensors"].astype(np.uint8)
        if str(metadata["images_sha256"].item()) != args.images_sha256:
            raise ValueError("Paper spatial metadata points to a different image cache")
    # Compute all site-relative scores before opening the diagnostic cache,
    # whose arrays include the paper labels and comparator scores.
    means, counts, group_indices = build_site_templates(images, groups)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw = predict_model(
        artifact["fitted"],
        images,
        np.arange(images.shape[0]),
        sensors,
        means,
        counts,
        group_indices,
        device,
    )
    with np.load(paths["diagnostic"], allow_pickle=False) as diagnostic:
        values = {name: diagnostic[name] for name in diagnostic.files}
    if not (
        np.array_equal(sample_ids, values["available_ids"].astype(str))
        and np.array_equal(groups, values["available_groups"].astype(str))
    ):
        raise ValueError("Paper spatial rows differ from the exact available cohort")
    indices = aligned_indices(values["aligned_sample_ids"], sample_ids)
    if not np.array_equal(sensors, values["sensors"][indices].astype(np.uint8)):
        raise ValueError("Paper spatial sensor alignment failed")
    candidate_scores = values["candidate_scores"].astype(np.float64).copy()
    candidate_scores[indices] = blend_scores(
        candidate_scores[indices], raw, float(artifact["blend_weight"])
    )
    baseline_scores = values["baseline_scores"].astype(np.float64)
    labels = values["labels"].astype(np.uint8)
    sites = values["sites"].astype(str)
    baseline_pixels = values["baseline_pixels"].astype(np.int64)
    gated_pixels = gate_counts(
        values["candidate_pixels"].astype(np.int64),
        values["candidate_scores"].astype(np.float64),
        cutoff,
    )
    threshold = float(artifact["operational_scene_threshold"])
    selections = {
        "full": np.ones(labels.shape, dtype=bool),
        "test_only_sites": values["test_only"].astype(bool),
    }
    views: dict[str, Any] = {}
    for index, (name, selected) in enumerate(selections.items()):
        metrics = view_metrics(
            labels[selected],
            baseline_scores[selected],
            candidate_scores[selected],
            triplet(baseline_pixels[selected]),
            triplet(gated_pixels[selected]),
            threshold,
        )
        bootstrap = bootstrap_view(
            labels=labels[selected],
            sites=sites[selected],
            baseline_scores=baseline_scores[selected],
            candidate_scores=candidate_scores[selected],
            baseline_predictions=baseline_scores[selected] > 0.5,
            candidate_predictions=candidate_scores[selected] > threshold,
            baseline_pixels=triplet(baseline_pixels[selected]),
            candidate_pixels=triplet(gated_pixels[selected]),
            replicates=args.replicates,
            seed=20261160 + index,
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
        "scope": "transparent post-test site-relative spatial replay on exact paper comparator",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "architecture": {
            "scene_ranking": "current v3 stronger head plus frozen 0.10 site-relative spatial CNN",
            "site_context": "label-free leave-one-out pixelwise mean over same-site observations",
            "singleton_policy": "template equals scene and residual is zero",
            "scene_threshold": threshold,
            "mask_probability": "released MARS-S2L probability with sensor thresholds",
            "mask_gate_score": "unchanged frozen v3 stronger scene score",
            "mask_gate_cutoff": cutoff,
        },
        "available_spatial_rows": int(sample_ids.size),
        "missing_rows_fallback_to_v3": int(labels.size - sample_ids.size),
        "site_template_count": int(means.shape[0]),
        "views": views,
        "all_exact_paper_gates_pass": passed,
        "decision": (
            "All exact paper gates pass on both views; independent external confirmation remains required."
            if passed
            else "Reject the site-relative spatial model as the final successor; at least one exact paper gate fails."
        ),
        "provenance": {
            **{f"{name}_sha256": digest for name, digest in expected.items()},
            "script_sha256": sha256(Path(__file__).resolve()),
        },
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
                        "ap_lower": value["bootstrap"]["delta_intervals"][
                            "average_precision"
                        ]["lower"],
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
