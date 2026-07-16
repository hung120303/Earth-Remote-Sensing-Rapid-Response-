#!/usr/bin/env python3
"""Replay the UNEP-augmented scene head on the exact MARS-S2L paper cache."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
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

from acquire_mars_metadata import sha256  # noqa: E402
from diagnose_mars_scene_stacker_paper_cache import aligned_indices, triplet  # noqa: E402
from evaluate_mars_scene_gated_masks_paper_cache import gate_counts  # noqa: E402
from evaluate_mars_successor_paper_test import bootstrap_view, view_metrics  # noqa: E402
from train_mars_context_scene_ranker import augment_site_context  # noqa: E402
from train_mars_scene_ranker import blend_scores  # noqa: E402


DEFAULT_PROTOCOL = Path("configs/mars_unep_positive_augmented_paper_replay.json")
DEFAULT_JSON = Path(
    "reports/experiments/mars_unep_positive_augmented_paper_posttest.json"
)
DEFAULT_MARKDOWN = Path(
    "reports/experiments/MARS_UNEP_POSITIVE_AUGMENTED_PAPER_POSTTEST.md"
)


def validate_candidate(
    artifact: dict[str, Any], development: dict[str, Any], protocol: dict[str, Any]
) -> None:
    candidate = protocol["candidate"]
    if development.get("all_promotion_gates_pass") is not True:
        raise ValueError("Development model was not promoted")
    if development.get("artifact", {}).get("sha256") != candidate["artifact_sha256"]:
        raise ValueError("Development report names a different artifact")
    if artifact.get("kind") != "mars_unep_positive_augmented_xgboost":
        raise ValueError("Unexpected candidate artifact kind")
    if float(artifact.get("candidate_blend")) != float(candidate["candidate_logit_blend"]):
        raise ValueError("Candidate blend differs from frozen replay")
    if float(artifact.get("operational_scene_threshold")) != float(
        candidate["operational_scene_threshold"]
    ):
        raise ValueError("Candidate threshold differs from frozen replay")


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# UNEP-augmented successor: exact MARS-S2L paper benchmark",
        "",
        "Transparent post-test replay, not an untouched confirmation. Candidate scores were computed from the label-free cache before comparator outcomes were opened; dense masks remain the unchanged promoted v3 branch.",
        "",
        "| View | AP | AP delta (95% CI) | Matched-FPR recall delta (95% CI) | FPR delta | IoU delta (95% CI) | Gates |",
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
    lines.extend(["", report["decision"], ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL.as_posix())
    parser.add_argument("--output-json", default=DEFAULT_JSON.as_posix())
    parser.add_argument("--output-markdown", default=DEFAULT_MARKDOWN.as_posix())
    args = parser.parse_args()
    protocol_path = (ROOT / args.protocol).resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    candidate_contract = protocol["candidate"]
    feature_contract = protocol["paper_caches"]["label_free_features"]
    diagnostic_contract = protocol["paper_caches"]["diagnostic_outcomes"]
    dense_contract = protocol["dense_mask"]
    paths = {
        "artifact": (ROOT / candidate_contract["artifact"]).resolve(),
        "development_report": (ROOT / candidate_contract["development_report"]).resolve(),
        "features": (ROOT / feature_contract["path"]).resolve(),
        "feature_receipt": (ROOT / feature_contract["receipt"]).resolve(),
        "diagnostic": (ROOT / diagnostic_contract["path"]).resolve(),
        "gate_report": (ROOT / dense_contract["gate_report"]).resolve(),
    }
    expected = {
        "artifact": candidate_contract["artifact_sha256"],
        "development_report": candidate_contract["development_report_sha256"],
        "features": feature_contract["sha256"],
        "feature_receipt": feature_contract["receipt_sha256"],
        "diagnostic": diagnostic_contract["sha256"],
        "gate_report": dense_contract["gate_report_sha256"],
    }
    for name, digest in expected.items():
        if sha256(paths[name]) != digest:
            raise ValueError(f"Frozen {name} hash mismatch")

    development = json.loads(paths["development_report"].read_text(encoding="utf-8"))
    artifact = joblib.load(paths["artifact"])
    validate_candidate(artifact, development, protocol)
    gate_report = json.loads(paths["gate_report"].read_text(encoding="utf-8"))
    if gate_report.get("all_selection_and_confirmation_gates_pass") is not True:
        raise ValueError("Dense-mask gate was not promoted")
    if float(gate_report["selection"]["selected_cutoff"]) != float(
        dense_contract["gate_cutoff"]
    ):
        raise ValueError("Dense-mask gate cutoff differs from frozen replay")
    receipt = json.loads(paths["feature_receipt"].read_text(encoding="utf-8"))
    if receipt.get("output_sha256") != expected["features"]:
        raise ValueError("Label-free feature receipt names a different cache")

    # Candidate inference occurs before loading the outcome-bearing diagnostic
    # cache. This preserves an auditable ordering even though the paper test is
    # transparently post-test at the project level.
    with np.load(paths["features"], allow_pickle=False) as cache:
        if tuple(cache.files) != tuple(receipt["output_fields"]):
            raise ValueError("Label-free paper feature schema differs from receipt")
        sample_ids = cache["sample_ids"].astype(str)
        groups = cache["groups"].astype(str)
        base = cache["base_features"].astype(np.float32)
        base_names = cache["base_feature_names"].astype(str)
        current_available = cache["current_v3_scores"].astype(np.float64)
    features, augmented_names = augment_site_context(base, base_names, groups)
    if augmented_names != artifact["feature_names"]:
        raise ValueError("Paper feature schema differs from candidate artifact")
    raw = artifact["model"].predict_proba(features)[:, 1]
    available_scores = blend_scores(
        current_available, raw, float(artifact["candidate_blend"])
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
        raise ValueError("Label-free rows differ from exact diagnostic cohort")
    candidate_scores = values["candidate_scores"].astype(np.float64).copy()
    candidate_scores[indices] = available_scores
    baseline_scores = values["baseline_scores"].astype(np.float64)
    labels = values["labels"].astype(np.uint8)
    sites = values["sites"].astype(str)
    baseline_pixels = values["baseline_pixels"].astype(np.int64)
    cutoff = float(dense_contract["gate_cutoff"])
    gated_pixels = gate_counts(
        values["candidate_pixels"].astype(np.int64),
        values["candidate_scores"].astype(np.float64),
        cutoff,
    )
    threshold = float(candidate_contract["operational_scene_threshold"])
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
            replicates=int(protocol["bootstrap"]["replicates"]),
            seed=int(protocol["bootstrap"]["seed"]) + index,
            confidence=float(protocol["bootstrap"]["confidence"]),
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
        "scope": protocol["scope"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "architecture": {
            "scene_ranking": "frozen v3 stronger head plus 0.20 UNEP-positive augmented XGBoost",
            "scene_threshold": threshold,
            "mask_probability": dense_contract["probability"],
            "mask_gate_score": dense_contract["gate_score"],
            "mask_gate_cutoff": cutoff,
            "dense_mask_changed": False,
        },
        "available_augmented_rows": int(sample_ids.size),
        "missing_rows_fallback_to_v3": int(labels.size - sample_ids.size),
        "views": views,
        "all_exact_paper_gates_pass": passed,
        "decision": (
            "All exact MARS-S2L v3 paper gates pass on both views."
            if passed
            else "At least one exact MARS-S2L v3 paper gate remains unresolved."
        ),
        "provenance": {
            **{f"{name}_sha256": digest for name, digest in expected.items()},
            "protocol_sha256": sha256(protocol_path),
            "script_sha256": sha256(Path(__file__).resolve()),
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "xgboost": xgboost.__version__,
            "numpy": np.__version__,
        },
    }
    write_json((ROOT / args.output_json).resolve(), report)
    write_markdown((ROOT / args.output_markdown).resolve(), report)
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
