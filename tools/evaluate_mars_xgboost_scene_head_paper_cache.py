#!/usr/bin/env python3
"""Replay the frozen XGBoost successor on the exact MARS-S2L paper benchmark."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import xgboost

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "EarthRemoteSensingRapidResponse"
for path in (MODEL_ROOT, ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from acquire_mars_metadata import repo_root, sha256  # noqa: E402
from diagnose_mars_scene_stacker_paper_cache import aligned_indices, triplet  # noqa: E402
from evaluate_mars_scene_gated_masks_paper_cache import gate_counts  # noqa: E402
from evaluate_mars_successor_paper_test import bootstrap_view, view_metrics  # noqa: E402
from train_mars_context_scene_ranker import augment_site_context  # noqa: E402
from train_mars_scene_ranker import blend_scores  # noqa: E402

DEFAULT_FEATURES = Path("outputs/mars_paper_scene_features_label_free.npz")
DEFAULT_FEATURES_SHA256 = "8a35e60e7c396e58639f940239020adb36def885124841e0b20901e10db52f33"
DEFAULT_FEATURE_RECEIPT = Path(
    "reports/acquisition/mars_paper_scene_features_label_free.json"
)
DEFAULT_FEATURE_RECEIPT_SHA256 = (
    "8f82262d60e9a47a40f2c7ded63042544c87bdfe00c74ed9c34cd5db462b168f"
)
DEFAULT_DIAGNOSTIC = Path("outputs/mars_paper_test_v3_diagnostic_cache.npz")
DEFAULT_DIAGNOSTIC_SHA256 = "1624fddc0222f8ffc5137f557c7fc3e465d53b335c82cc8014711baa35bb94a1"
DEFAULT_ARTIFACT = Path(
    "EarthRemoteSensingRapidResponse/artifacts/mars_xgboost_scene_head.joblib"
)
DEFAULT_ARTIFACT_SHA256 = "e383b9e4e0c3879aa1db4b33d12a823a396a66c8c7abd86ad6813bb44c56fb4b"
DEFAULT_DEVELOPMENT_REPORT = Path("reports/experiments/mars_xgboost_scene_head.json")
DEFAULT_DEVELOPMENT_REPORT_SHA256 = (
    "f79fb3c8b0a0ff2832d6d7c9d1ae945b6bf6fc6d795e661e49bcaba0910da1db"
)
DEFAULT_GATE_REPORT = Path("reports/experiments/mars_scene_gated_masks.json")
DEFAULT_GATE_REPORT_SHA256 = "c1e5a1497abebba80d42898a8165b30fd255ff252478a0ee1fd90fd32456a51c"
DEFAULT_JSON = Path("reports/experiments/mars_xgboost_scene_head_paper_posttest.json")
DEFAULT_MARKDOWN = Path("reports/experiments/MARS_XGBOOST_SCENE_HEAD_PAPER_POSTTEST.md")


def operational_threshold(development: dict[str, Any]) -> float:
    """Use the conservative maximum candidate threshold across all OOF folds."""
    folds = development["selected"]["per_fold"]
    thresholds = [
        float(value["versus_current"]["metrics"]["operating_point"]["threshold"])
        for value in folds.values()
    ]
    if len(thresholds) != 5 or not np.isfinite(thresholds).all():
        raise ValueError("Development report does not contain five valid OOF thresholds")
    return max(thresholds)


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# XGBoost successor: exact MARS-S2L paper benchmark",
        "",
        "Transparent post-test replay; this is not an untouched confirmation cohort. XGBoost scores were computed from a separate label-free cache, and the dense-mask gate remains driven by the unchanged v3 scene score.",
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
    parser.add_argument("--features", default=DEFAULT_FEATURES.as_posix())
    parser.add_argument("--features-sha256", default=DEFAULT_FEATURES_SHA256)
    parser.add_argument("--feature-receipt", default=DEFAULT_FEATURE_RECEIPT.as_posix())
    parser.add_argument(
        "--feature-receipt-sha256", default=DEFAULT_FEATURE_RECEIPT_SHA256
    )
    parser.add_argument("--diagnostic", default=DEFAULT_DIAGNOSTIC.as_posix())
    parser.add_argument("--diagnostic-sha256", default=DEFAULT_DIAGNOSTIC_SHA256)
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
        "features": (root / args.features).resolve(),
        "feature_receipt": (root / args.feature_receipt).resolve(),
        "diagnostic": (root / args.diagnostic).resolve(),
        "artifact": (root / args.artifact).resolve(),
        "development_report": (root / args.development_report).resolve(),
        "gate_report": (root / args.gate_report).resolve(),
    }
    expected = {
        "features": args.features_sha256,
        "feature_receipt": args.feature_receipt_sha256,
        "diagnostic": args.diagnostic_sha256,
        "artifact": args.artifact_sha256,
        "development_report": args.development_report_sha256,
        "gate_report": args.gate_report_sha256,
    }
    for name, digest in expected.items():
        if sha256(paths[name]) != digest:
            raise ValueError(f"Frozen {name} hash mismatch")

    development = json.loads(paths["development_report"].read_text(encoding="utf-8"))
    if development.get("all_promotion_gates_pass") is not True:
        raise ValueError("Development XGBoost model was not promoted")
    gate_report = json.loads(paths["gate_report"].read_text(encoding="utf-8"))
    if gate_report.get("all_selection_and_confirmation_gates_pass") is not True:
        raise ValueError("Development dense-mask gate was not promoted")
    receipt = json.loads(paths["feature_receipt"].read_text(encoding="utf-8"))
    if receipt.get("output_sha256") != args.features_sha256:
        raise ValueError("Label-free feature receipt points to a different cache")
    artifact = joblib.load(paths["artifact"])
    selected = development["selected"]
    if (
        artifact.get("model_spec") != selected["model_spec"]
        or float(artifact.get("xgboost_blend")) != float(selected["xgboost_blend"])
    ):
        raise ValueError("XGBoost artifact differs from its development report")

    # Score the model from the label-free cache before opening the separate
    # diagnostic archive containing comparator outcomes and pixel truth.
    with np.load(paths["features"], allow_pickle=False) as cache:
        if tuple(cache.files) != tuple(receipt["output_fields"]):
            raise ValueError("Label-free feature cache schema differs from its receipt")
        sample_ids = cache["sample_ids"].astype(str)
        groups = cache["groups"].astype(str)
        base = cache["base_features"].astype(np.float32)
        base_names = cache["base_feature_names"].astype(str)
        current_available = cache["current_v3_scores"].astype(np.float64)
    features, augmented_names = augment_site_context(base, base_names, groups)
    if augmented_names != artifact["feature_names"]:
        raise ValueError("Paper augmented-feature schema differs from XGBoost artifact")
    raw = artifact["model"].predict_proba(features)[:, 1]
    available_scores = blend_scores(
        current_available, raw, float(artifact["xgboost_blend"])
    )

    with np.load(paths["diagnostic"], allow_pickle=False) as diagnostic:
        values = {name: diagnostic[name] for name in diagnostic.files}
    indices = aligned_indices(values["aligned_sample_ids"], sample_ids)
    if not (
        np.array_equal(groups, values["available_groups"].astype(str))
        and np.allclose(
            current_available,
            values["candidate_scores"].astype(np.float64)[indices],
            rtol=0.0,
            atol=0.0,
        )
    ):
        raise ValueError("Label-free paper rows differ from the exact diagnostic cohort")
    candidate_scores = values["candidate_scores"].astype(np.float64).copy()
    candidate_scores[indices] = available_scores
    baseline_scores = values["baseline_scores"].astype(np.float64)
    labels = values["labels"].astype(np.uint8)
    sites = values["sites"].astype(str)
    baseline_pixels = values["baseline_pixels"].astype(np.int64)
    cutoff = float(gate_report["selection"]["selected_cutoff"])
    gated_pixels = gate_counts(
        values["candidate_pixels"].astype(np.int64),
        values["candidate_scores"].astype(np.float64),
        cutoff,
    )
    threshold = operational_threshold(development)
    selections = {
        "full": np.ones(labels.shape, dtype=bool),
        "test_only_sites": values["test_only"].astype(bool),
    }
    views: dict[str, Any] = {}
    for index, (name, selected_rows) in enumerate(selections.items()):
        metrics = view_metrics(
            labels[selected_rows],
            baseline_scores[selected_rows],
            candidate_scores[selected_rows],
            triplet(baseline_pixels[selected_rows]),
            triplet(gated_pixels[selected_rows]),
            threshold,
        )
        bootstrap = bootstrap_view(
            labels=labels[selected_rows],
            sites=sites[selected_rows],
            baseline_scores=baseline_scores[selected_rows],
            candidate_scores=candidate_scores[selected_rows],
            baseline_predictions=baseline_scores[selected_rows] > 0.5,
            candidate_predictions=candidate_scores[selected_rows] > threshold,
            baseline_pixels=triplet(baseline_pixels[selected_rows]),
            candidate_pixels=triplet(gated_pixels[selected_rows]),
            replicates=args.replicates,
            seed=20261460 + index,
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
        "scope": "transparent post-test XGBoost replay on exact MARS-S2L paper comparator",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "architecture": {
            "scene_ranking": "current v3 stronger head plus frozen 0.10 regularized XGBoost head",
            "scene_threshold": threshold,
            "scene_threshold_rule": "maximum candidate threshold across all five development OOF folds",
            "mask_probability": "released MARS-S2L probability with frozen sensor thresholds",
            "mask_gate_score": "unchanged frozen v3 stronger scene score",
            "mask_gate_cutoff": cutoff,
        },
        "available_xgboost_rows": int(sample_ids.size),
        "missing_rows_fallback_to_v3": int(labels.size - sample_ids.size),
        "views": views,
        "all_exact_paper_gates_pass": passed,
        "decision": (
            "All exact paper gates pass on both views; independent external confirmation remains required."
            if passed
            else "Reject the XGBoost complement as the final successor; at least one exact paper gate fails."
        ),
        "provenance": {
            **{f"{name}_sha256": digest for name, digest in expected.items()},
            "script_sha256": sha256(Path(__file__).resolve()),
            "xgboost": xgboost.__version__,
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
