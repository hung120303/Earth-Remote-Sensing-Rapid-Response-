#!/usr/bin/env python3
"""Evaluate the frozen calibrated spatial-Prithvi ensemble on exact MARS-S2L v3."""

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
import torch

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "EarthRemoteSensingRapidResponse"
for path in (MODEL_ROOT, ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from acquire_mars_metadata import sha256  # noqa: E402
from diagnose_mars_scene_stacker_paper_cache import aligned_indices, triplet  # noqa: E402
from evaluate_mars_scene_gated_masks_paper_cache import gate_counts  # noqa: E402
from evaluate_mars_successor_paper_test import bootstrap_view, view_metrics  # noqa: E402
from train_mars_scene_ranker import blend_scores  # noqa: E402
from train_mars_site_relative_spatial_classifier import (  # noqa: E402
    build_site_templates,
    predict_model,
)


DEFAULT_PROTOCOL = Path("configs/mars_spatial_prithvi_ensemble_paper_protocol.json")


def apply_offset(values: np.ndarray, offset: float) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=np.float64), 1e-8, 1.0 - 1e-8)
    logits = np.log(clipped) - np.log1p(-clipped) + float(offset)
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Calibrated spatial-Prithvi ensemble: exact MARS-S2L v3 benchmark",
        "",
        "Transparent post-test architecture evaluation against the exact reconstructed v3 paper comparator.",
        "",
        "| View | Exact v3 AP | Candidate AP | AP delta (95% CI) | Matched-FPR recall delta (95% CI) | Exact v3 IoU | Candidate IoU | IoU delta (95% CI) | Result |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for name, value in report["views"].items():
        metrics = value["metrics"]
        intervals = value["bootstrap"]["delta_intervals"]
        lines.append(
            f"| {name} | {metrics['baseline']['average_precision']:.6f} | "
            f"{metrics['candidate']['average_precision']:.6f} | "
            f"{metrics['delta']['average_precision']:+.6f} "
            f"([{intervals['average_precision']['lower']:+.6f}, {intervals['average_precision']['upper']:+.6f}]) | "
            f"{metrics['delta']['matched_fpr_recall']:+.6f} "
            f"([{intervals['matched_fpr_recall']['lower']:+.6f}, {intervals['matched_fpr_recall']['upper']:+.6f}]) | "
            f"{metrics['baseline']['pixels']['intersection_over_union']:.6f} | "
            f"{metrics['candidate']['pixels']['intersection_over_union']:.6f} | "
            f"{metrics['delta']['pixel_iou']:+.6f} "
            f"([{intervals['pixel_iou']['lower']:+.6f}, {intervals['pixel_iou']['upper']:+.6f}]) | "
            f"{'PASS' if value['passed'] else 'FAIL'} |"
        )
    lines.extend([
        "",
        f"Official paper revision: [{report['paper']['revision']}]({report['paper']['url']}).",
        "",
        report["decision"],
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def exact_comparator_check(
    name: str,
    metrics: dict[str, Any],
    receipt: dict[str, Any],
    selected: np.ndarray,
    sites: np.ndarray,
) -> dict[str, bool]:
    key = "full" if name == "full" else "test_only_sites"
    expected = receipt["reconstruction"][key]
    baseline = metrics["baseline"]
    fixed = baseline["fixed_operating_point"]
    pixels = baseline["pixels"]
    tolerance = 1e-12
    return {
        "rows_exact": int(selected.sum()) == int(expected["rows"]),
        "sites_exact": len(set(sites[selected].tolist())) == int(expected["sites"]),
        "positives_exact": int(metrics["positive"]) == int(expected["positive"]),
        "average_precision_exact": abs(float(baseline["average_precision"]) - float(expected["average_precision"])) <= tolerance,
        "recall_exact": abs(float(fixed["recall"]) - float(expected["recall"])) <= tolerance,
        "false_positive_rate_exact": abs(float(fixed["false_positive_rate"]) - float(expected["false_positive_rate"])) <= tolerance,
        "pixel_iou_exact": abs(float(pixels["intersection_over_union"]) - float(expected["pixel_iou"])) <= tolerance,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL.as_posix())
    args = parser.parse_args()
    protocol_path = (ROOT / args.protocol).resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if sha256(Path(__file__).resolve()) != protocol["evaluator"]["sha256"]:
        raise ValueError("Exact ensemble paper evaluator hash mismatch")
    paths: dict[str, Path] = {}
    for name, contract in protocol["inputs"].items():
        path = (ROOT / contract["path"]).resolve()
        if sha256(path) != contract["sha256"]:
            raise ValueError(f"Frozen exact-paper input hash mismatch: {name}")
        paths[name] = path

    reports = {
        name: json.loads(paths[name].read_text(encoding="utf-8"))
        for name in (
            "paper_benchmark", "label_free_receipt", "prithvi_receipt",
            "spatial_development", "prithvi_development", "ensemble_development",
            "calibration_development", "fresh_safety", "mask_gate_development",
        )
    }
    paper = reports["paper_benchmark"]
    paper_contract = protocol["paper_contract"]
    receipt_exact = paper["reconstruction"]
    protocol_exact = paper_contract["exact_reconstructed_comparator"]
    exact_contract_matches = all(
        (
            int(receipt_exact[view][metric]) == int(expected)
            if metric in {"rows", "sites", "positive"}
            else abs(float(receipt_exact[view][metric]) - float(expected)) <= 1e-15
        )
        for view, metrics in protocol_exact.items()
        for metric, expected in metrics.items()
    )
    if (
        paper["paper"]["revision"] != paper_contract["revision"]
        or paper["paper"]["url"] != paper_contract["url"]
        or not exact_contract_matches
        or int(paper["artifacts"]["assignment_rows"]) != 43_529
        or reports["label_free_receipt"]["output_sha256"] != protocol["inputs"]["label_free_scores"]["sha256"]
        or reports["prithvi_receipt"]["output_sha256"] != protocol["inputs"]["prithvi_scores"]["sha256"]
        or reports["prithvi_receipt"]["labels_accessed"] is not False
        or reports["spatial_development"].get("all_promotion_gates_pass") is not True
        or reports["prithvi_development"].get("all_promotion_gates_pass") is not True
        or reports["ensemble_development"].get("all_promotion_gates_pass") is not True
        or reports["calibration_development"].get("all_calibration_gates_pass") is not True
        or reports["fresh_safety"].get("all_safety_gates_pass") is not True
        or reports["mask_gate_development"].get("all_selection_and_confirmation_gates_pass") is not True
    ):
        raise ValueError("Exact-paper provenance or promotion chain failed")

    with np.load(paths["label_free_scores"], allow_pickle=False) as cache:
        sample_ids = cache["sample_ids"].astype(str)
        groups = cache["groups"].astype(str)
        current_available = cache["current_v3_scores"].astype(np.float64)
        if any(token in " ".join(cache.files).lower() for token in ("label", "truth", "test_only")):
            raise ValueError("Label-free score cache contains a forbidden outcome field")
    with np.load(paths["spatial_metadata"], allow_pickle=False) as cache:
        spatial_ids = cache["sample_ids"].astype(str)
        spatial_groups = cache["groups"].astype(str)
        sensors = cache["sensors"].astype(np.uint8)
        if str(cache["images_sha256"].item()) != protocol["inputs"]["spatial_images"]["sha256"]:
            raise ValueError("Spatial metadata points to a different image cache")
    with np.load(paths["prithvi_scores"], allow_pickle=False) as cache:
        prithvi_ids = cache["sample_ids"].astype(str)
        prithvi_available = cache["scores"].astype(np.float64)
    if not (
        sample_ids.shape == (43_524,)
        and np.array_equal(sample_ids, spatial_ids)
        and np.array_equal(sample_ids, prithvi_ids)
        and np.array_equal(groups, spatial_groups)
        and len(set(sample_ids.tolist())) == sample_ids.size
        and len(set(groups.tolist())) == 1_289
    ):
        raise ValueError("Label-free component score identities differ")

    images = np.load(paths["spatial_images"], mmap_mode="r", allow_pickle=False)
    if images.shape != (43_524, 9, 64, 64) or images.dtype != np.float16:
        raise ValueError("Paper spatial image cache schema differs")
    spatial_control = torch.load(paths["spatial_artifact"], map_location="cpu", weights_only=True)
    means, counts, group_indices = build_site_templates(images, groups)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spatial_raw = predict_model(
        spatial_control["fitted"], images, np.arange(images.shape[0]), sensors,
        means, counts, group_indices, device,
    )
    spatial_available = blend_scores(
        current_available, spatial_raw, float(spatial_control["blend_weight"])
    )
    ensemble_control = joblib.load(paths["ensemble_artifact"])
    calibration_control = joblib.load(paths["calibration_artifact"])
    if calibration_control["base_ensemble_sha256"] != protocol["inputs"]["ensemble_artifact"]["sha256"]:
        raise ValueError("Calibration artifact does not bind the exact ensemble")
    raw_available = blend_scores(
        spatial_available, prithvi_available, float(ensemble_control["prithvi_weight"])
    )
    offset = float(calibration_control["logit_offset"])
    calibrated_available = apply_offset(raw_available, offset)
    if (
        offset >= 0.0
        or np.any(calibrated_available >= raw_available)
        or not all(np.isfinite(value).all() for value in (spatial_raw, spatial_available, prithvi_available, raw_available, calibrated_available))
    ):
        raise ValueError("Exact paper ensemble scores violate calibration contract")

    score_cache = (ROOT / protocol["outputs"]["label_free_scores"]).resolve()
    score_cache.parent.mkdir(parents=True, exist_ok=True)
    temporary_cache = score_cache.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary_cache,
        sample_ids=sample_ids,
        groups=groups,
        raw_scores=raw_available,
        calibrated_scores=calibrated_available,
        logit_offset=np.asarray(offset),
        protocol_sha256=np.asarray(sha256(protocol_path)),
    )
    os.replace(temporary_cache, score_cache)
    score_cache_record = {
        "path": protocol["outputs"]["label_free_scores"],
        "bytes": score_cache.stat().st_size,
        "sha256": sha256(score_cache),
        "labels_accessed_during_score_construction": False,
        "tracked": False,
    }

    # Outcome arrays are opened only after the complete available score vector exists.
    with np.load(paths["diagnostic"], allow_pickle=False) as cache:
        values = {name: cache[name] for name in cache.files}
    indices = aligned_indices(values["aligned_sample_ids"], sample_ids)
    if not (
        np.array_equal(values["available_ids"].astype(str), sample_ids)
        and np.array_equal(values["available_groups"].astype(str), groups)
        and np.array_equal(values["sensors"][indices].astype(np.uint8), sensors)
    ):
        raise ValueError("Exact diagnostic alignment differs from label-free components")
    candidate = values["candidate_scores"].astype(np.float64).copy()
    candidate[indices] = calibrated_available
    labels = values["labels"].astype(np.uint8)
    baseline = values["baseline_scores"].astype(np.float64)
    sites = values["sites"].astype(str)
    baseline_pixels = values["baseline_pixels"].astype(np.int64)
    cutoff = float(reports["mask_gate_development"]["selection"]["selected_cutoff"])
    gated_pixels = gate_counts(
        values["candidate_pixels"].astype(np.int64),
        values["candidate_scores"].astype(np.float64),
        cutoff,
    )
    threshold = float(calibration_control["operational_scene_threshold"])
    selections = {
        "full": np.ones(labels.shape, dtype=bool),
        "test_only_sites": values["test_only"].astype(bool),
    }
    views: dict[str, Any] = {}
    comparator_identity: dict[str, Any] = {}
    for index, (name, selected) in enumerate(selections.items()):
        metrics = view_metrics(
            labels[selected], baseline[selected], candidate[selected],
            triplet(baseline_pixels[selected]), triplet(gated_pixels[selected]), threshold,
        )
        identity = exact_comparator_check(name, metrics, paper, selected, sites)
        if not all(identity.values()):
            raise ValueError(f"Exact v3 comparator identity failed for {name}: {identity}")
        comparator_identity[name] = identity
        bootstrap = bootstrap_view(
            labels=labels[selected],
            sites=sites[selected],
            baseline_scores=baseline[selected],
            candidate_scores=candidate[selected],
            baseline_predictions=baseline[selected] > 0.5,
            candidate_predictions=candidate[selected] > threshold,
            baseline_pixels=triplet(baseline_pixels[selected]),
            candidate_pixels=triplet(gated_pixels[selected]),
            replicates=int(protocol["bootstrap"]["replicates"]),
            seed=int(protocol["bootstrap"]["seed"]) + index,
            confidence=float(protocol["bootstrap"]["confidence"]),
        )
        intervals = bootstrap["delta_intervals"]
        published_key = "full" if name == "full" else "test_only_sites"
        published = paper["reconstruction"]["published"][published_key]
        checks = {
            "ap_point_beats_exact_v3": metrics["delta"]["average_precision"] > 0.0,
            "ap_point_beats_published_table": metrics["candidate"]["average_precision"] > float(published["average_precision"]),
            "ap_site_bootstrap_lower_positive": intervals["average_precision"]["lower"] > 0.0,
            "matched_fpr_recall_point_higher": metrics["delta"]["matched_fpr_recall"] > 0.0,
            "matched_fpr_recall_site_bootstrap_lower_positive": intervals["matched_fpr_recall"]["lower"] > 0.0,
            "matched_false_positive_rate_no_worse": metrics["delta"]["matched_false_positive_rate"] <= 0.0,
            "fixed_fpr_site_bootstrap_upper_nonpositive": intervals["fixed_false_positive_rate"]["upper"] <= 0.0,
            "pixel_iou_point_beats_exact_v3": metrics["delta"]["pixel_iou"] > 0.0,
            "pixel_iou_site_bootstrap_lower_positive": intervals["pixel_iou"]["lower"] > 0.0,
        }
        if "pixel_iou" in published:
            checks["pixel_iou_point_beats_published_table"] = (
                metrics["candidate"]["pixels"]["intersection_over_union"]
                > float(published["pixel_iou"])
            )
        views[name] = {
            "metrics": metrics,
            "bootstrap": bootstrap,
            "published_table": published,
            "checks": checks,
            "passed": all(checks.values()),
        }

    passed = all(value["passed"] for value in views.values())
    report = {
        "schema_version": 1,
        "scope": "transparent post-test calibrated spatial-Prithvi ensemble replay on exact MARS-S2L v3 comparator",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "paper": paper["paper"],
        "architecture": {
            "scene_ranking": "0.75 adaptive Prithvi plus 0.25 site-relative spatial logit ensemble",
            "calibration": "development-negative empirical-CDF dominance via constant logit offset",
            "logit_offset": offset,
            "operational_scene_threshold": threshold,
            "mask_probability": "released MARS-S2L probability with frozen sensor thresholds",
            "mask_gate_score": "unchanged frozen v3 stronger scene score",
            "mask_gate_cutoff": cutoff,
        },
        "available_rows": int(sample_ids.size),
        "missing_rows_adversarial_policy": int(labels.size - sample_ids.size),
        "site_template_count": int(means.shape[0]),
        "label_free_score_cache": score_cache_record,
        "comparator_identity": comparator_identity,
        "views": views,
        "all_exact_paper_gates_pass": passed,
        "decision": (
            "The frozen ensemble unequivocally beats the exact MARS-S2L v3 comparator on every predeclared gate in both views."
            if passed else "The frozen ensemble does not beat the exact MARS-S2L v3 comparator on every required gate."
        ),
        "audit_status": {
            "phase": "transparent post-test architecture evaluation",
            "independent_external_confirmation": False,
            "paper_labels_used_for_architecture_or_calibration": False,
            "exact_comparator": True,
        },
        "provenance": {
            "protocol_sha256": sha256(protocol_path),
            "script_sha256": sha256(Path(__file__).resolve()),
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "device": str(torch.cuda.get_device_name(device) if device.type == "cuda" else device),
            **{f"{name}_sha256": contract["sha256"] for name, contract in protocol["inputs"].items()},
        },
    }
    write_json((ROOT / protocol["outputs"]["json"]).resolve(), report)
    write_markdown((ROOT / protocol["outputs"]["markdown"]).resolve(), report)
    print(json.dumps({
        "ok": passed,
        "views": {
            name: {
                "candidate_ap": value["metrics"]["candidate"]["average_precision"],
                "ap_delta": value["metrics"]["delta"]["average_precision"],
                "ap_lower": value["bootstrap"]["delta_intervals"]["average_precision"]["lower"],
                "matched_recall_delta": value["metrics"]["delta"]["matched_fpr_recall"],
                "matched_recall_lower": value["bootstrap"]["delta_intervals"]["matched_fpr_recall"]["lower"],
                "iou_delta": value["metrics"]["delta"]["pixel_iou"],
                "iou_lower": value["bootstrap"]["delta_intervals"]["pixel_iou"]["lower"],
                "passed": value["passed"],
            }
            for name, value in views.items()
        },
    }, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
