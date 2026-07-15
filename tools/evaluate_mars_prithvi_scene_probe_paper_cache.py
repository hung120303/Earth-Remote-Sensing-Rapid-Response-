#!/usr/bin/env python3
"""Evaluate the frozen Prithvi scene probe on the exact cached MARS paper contract."""

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
from train_mars_encoder_scene_probe import predict_model  # noqa: E402
from train_mars_scene_ranker import blend_scores  # noqa: E402

DEFAULT_DIAGNOSTIC = Path("outputs/mars_paper_test_v3_diagnostic_cache.npz")
DEFAULT_DIAGNOSTIC_SHA256 = "1624fddc0222f8ffc5137f557c7fc3e465d53b335c82cc8014711baa35bb94a1"
DEFAULT_FEATURES = Path("outputs/mars_paper_prithvi_cls_features.npz")
DEFAULT_ARTIFACT = Path("EarthRemoteSensingRapidResponse/artifacts/mars_prithvi_scene_probe.pt")
DEFAULT_DEVELOPMENT_REPORT = Path("reports/experiments/mars_prithvi_scene_probe.json")
DEFAULT_GATE_REPORT = Path("reports/experiments/mars_scene_gated_masks.json")
DEFAULT_JSON = Path("reports/experiments/mars_prithvi_scene_probe_paper_posttest.json")
DEFAULT_MARKDOWN = Path("reports/experiments/MARS_PRITHVI_SCENE_PROBE_PAPER_POSTTEST.md")


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Prithvi scene probe: exact MARS-S2L paper benchmark",
        "",
        "Transparent post-test replay; this is not an untouched confirmation cohort. The dense-mask gate remains driven by the separately frozen v3 scene score.",
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
    parser.add_argument("--features", default=DEFAULT_FEATURES.as_posix())
    parser.add_argument("--features-sha256", required=True)
    parser.add_argument("--artifact", default=DEFAULT_ARTIFACT.as_posix())
    parser.add_argument("--artifact-sha256", required=True)
    parser.add_argument("--development-report", default=DEFAULT_DEVELOPMENT_REPORT.as_posix())
    parser.add_argument("--development-report-sha256", required=True)
    parser.add_argument("--gate-report", default=DEFAULT_GATE_REPORT.as_posix())
    parser.add_argument("--replicates", type=int, default=10_000)
    parser.add_argument("--output-json", default=DEFAULT_JSON.as_posix())
    parser.add_argument("--output-markdown", default=DEFAULT_MARKDOWN.as_posix())
    args = parser.parse_args()
    root = repo_root()
    paths = {
        "diagnostic": (root / args.diagnostic).resolve(),
        "features": (root / args.features).resolve(),
        "artifact": (root / args.artifact).resolve(),
        "development_report": (root / args.development_report).resolve(),
        "gate_report": (root / args.gate_report).resolve(),
    }
    expected = {
        "diagnostic": args.diagnostic_sha256,
        "features": args.features_sha256,
        "artifact": args.artifact_sha256,
        "development_report": args.development_report_sha256,
    }
    for name, digest in expected.items():
        if sha256(paths[name]) != digest:
            raise ValueError(f"Frozen {name} hash mismatch")
    development = json.loads(paths["development_report"].read_text(encoding="utf-8"))
    if development.get("all_promotion_gates_pass") is not True:
        raise ValueError("Development Prithvi probe was not promoted")
    gate_report = json.loads(paths["gate_report"].read_text(encoding="utf-8"))
    if gate_report.get("all_selection_and_confirmation_gates_pass") is not True:
        raise ValueError("Development dense-mask gate was not promoted")
    cutoff = float(gate_report["selection"]["selected_cutoff"])
    artifact = torch.load(paths["artifact"], map_location="cpu", weights_only=True)
    if (
        float(artifact["blend_weight"]) != float(development["selected"]["blend_weight"])
        or artifact["spec"] != development["selected"]["spec"]
    ):
        raise ValueError("Prithvi artifact differs from its development report")
    with np.load(paths["features"], allow_pickle=False) as cache:
        if "labels" in cache.files:
            raise ValueError("Paper Prithvi cache must remain label-independent")
        prithvi = cache["features"].astype(np.float32)
        prithvi_names = cache["feature_names"].astype(str)
        sample_ids = cache["sample_ids"].astype(str)
        groups = cache["groups"].astype(str)
        sensors = cache["sensors"].astype(np.uint8)
    with np.load(paths["diagnostic"], allow_pickle=False) as diagnostic:
        values = {name: diagnostic[name] for name in diagnostic.files}
    if not (
        np.array_equal(sample_ids, values["available_ids"].astype(str))
        and np.array_equal(groups, values["available_groups"].astype(str))
    ):
        raise ValueError("Paper Prithvi rows differ from the exact available cohort")
    indices = aligned_indices(values["aligned_sample_ids"], sample_ids)
    if not np.array_equal(sensors, values["sensors"][indices].astype(np.uint8)):
        raise ValueError("Paper Prithvi sensor alignment failed")
    base = values["available_base_features"].astype(np.float32)
    base_names = values["base_feature_names"].astype(str)
    names = [*prithvi_names.tolist(), *base_names.tolist()]
    if names != artifact["feature_names"]:
        raise ValueError("Paper Prithvi probe feature schema differs from the artifact")
    features = np.concatenate([prithvi, base], axis=1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw = predict_model(artifact["fitted"], features, device)
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
            seed=20261080 + index,
            confidence=0.95,
        )
        intervals = bootstrap["delta_intervals"]
        checks = {
            "ap_point_higher": metrics["delta"]["average_precision"] > 0.0,
            "ap_lower_positive": intervals["average_precision"]["lower"] > 0.0,
            "matched_recall_point_higher": metrics["delta"]["matched_fpr_recall"] > 0.0,
            "matched_recall_lower_positive": intervals["matched_fpr_recall"]["lower"] > 0.0,
            "fixed_fpr_upper_nonpositive": intervals["fixed_false_positive_rate"]["upper"] <= 0.0,
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
        "scope": "transparent post-test Prithvi scene-probe replay on exact paper comparator",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "architecture": {
            "scene_ranking": "current v3 stronger head plus frozen 0.05 Prithvi CLS linear probe",
            "scene_threshold": threshold,
            "mask_probability": "released MARS-S2L probability with sensor thresholds",
            "mask_gate_score": "unchanged frozen v3 stronger scene score",
            "mask_gate_cutoff": cutoff,
        },
        "available_prithvi_rows": int(indices.size),
        "missing_rows_fallback_to_v3": int(labels.size - indices.size),
        "views": views,
        "all_exact_paper_gates_pass": passed,
        "decision": (
            "All exact paper gates pass; independent external confirmation is still required."
            if passed
            else "Reject the Prithvi complement as the final successor; at least one exact paper gate fails."
        ),
        "provenance": {
            **{f"{name}_sha256": digest for name, digest in expected.items()},
            "gate_report_sha256": sha256(paths["gate_report"]),
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
                        "ap_lower": value["bootstrap"]["delta_intervals"]["average_precision"]["lower"],
                        "recall_lower": value["bootstrap"]["delta_intervals"]["matched_fpr_recall"]["lower"],
                        "iou_lower": value["bootstrap"]["delta_intervals"]["pixel_iou"]["lower"],
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
