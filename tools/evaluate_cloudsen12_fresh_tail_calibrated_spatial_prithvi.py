#!/usr/bin/env python3
"""Post-test fresh-negative safety replay for the tail-calibrated ensemble."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "EarthRemoteSensingRapidResponse"
for path in (MODEL_ROOT, ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from acquire_mars_metadata import sha256  # noqa: E402
from evaluate_cloudsen12_fresh_test_scene_heads import compare_stratum  # noqa: E402
from train_mars_adaptive_prithvi_probe import domain_normalize, load_features  # noqa: E402
from train_mars_context_scene_ranker import augment_site_context  # noqa: E402
from train_mars_crossfold_bagged_scene_head import load_development  # noqa: E402
from train_mars_scene_ranker import blend_scores  # noqa: E402
from train_mars_site_relative_spatial_classifier import (  # noqa: E402
    build_site_templates,
    predict_model,
)
from train_mars_unep_positive_augmented_xgboost import current_scores  # noqa: E402


DEFAULT_PROTOCOL = Path("configs/cloudsen12_fresh_tail_calibrated_spatial_prithvi_protocol.json")


def apply_offset(values: np.ndarray, offset: float) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=np.float64), 1e-8, 1.0 - 1e-8)
    logits = np.log(clipped) - np.log1p(-clipped) + float(offset)
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    available = report["strata"]["available"]
    lines = [
        "# Post-test fresh CloudSEN tail-calibrated ensemble replay",
        "",
        "> This reuses the previously inspected fresh cohort and is not an untouched test.",
        "",
        f"- Available rows: **{available['current']['rows']}**; unavailable: **{report['full_cohort_bounds']['unavailable_rows']}**.",
        f"- Current false positives: **{available['current']['false_positives']}**; candidate: **{available['candidate']['false_positives']}**.",
        f"- Current/candidate p95: **{available['current']['raw_score']['p95']:.6f} / {available['candidate']['raw_score']['p95']:.6f}**.",
        f"- Frozen logit offset: **{report['calibration']['logit_offset']:.6f}**.",
        f"- All safety gates pass: **{str(report['all_safety_gates_pass']).lower()}**.",
        "",
        "The exact MARS paper cache remained unopened by this evaluator.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL.as_posix())
    args = parser.parse_args()
    protocol_path = (ROOT / args.protocol).resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if sha256(Path(__file__).resolve()) != protocol["evaluator"]["sha256"]:
        raise ValueError("Calibrated fresh evaluator hash mismatch")
    paths: dict[str, Path] = {}
    for name, contract in protocol["inputs"].items():
        path = (ROOT / contract["path"]).resolve()
        if sha256(path) != contract["sha256"]:
            raise ValueError(f"Frozen calibrated fresh input hash mismatch: {name}")
        paths[name] = path

    with np.load(paths["fresh_features"], allow_pickle=False) as cache:
        raw_features = cache["features"].astype(np.float64)
        base_names = cache["feature_names"].astype(str)
        sample_ids = cache["sample_ids"].astype(str)
        groups = cache["groups"].astype(str)
        labels = cache["labels"].astype(np.uint8)
        all_clear = cache["published_all_clear"].astype(bool)
        nonclear_pixels = cache["published_nonclear_pixels"].astype(np.int64)
    with np.load(paths["fresh_representations"], allow_pickle=False) as cache:
        representation_ids = cache["sample_ids"].astype(str)
        representation_groups = cache["groups"].astype(str)
        spatial_images = cache["spatial_images"].astype(np.float32)
        prithvi_cls = cache["prithvi_cls"].astype(np.float32)
    expected = protocol["expected"]
    if (
        not np.array_equal(sample_ids, representation_ids)
        or not np.array_equal(groups, representation_groups)
        or raw_features.shape != (int(expected["available_rows"]), int(expected["features"]))
        or spatial_images.shape != tuple(expected["spatial_shape"])
        or prithvi_cls.shape != tuple(expected["prithvi_shape"])
        or np.any(labels)
        or len(set(groups.tolist())) != int(expected["groups"])
        or int(np.count_nonzero(all_clear)) != int(expected["all_clear_rows"])
        or int(nonclear_pixels.sum()) != int(expected["nonclear_pixels"])
    ):
        raise ValueError("Fresh calibrated row or stratum contract changed")

    augmented, augmented_names = augment_site_context(raw_features, base_names, groups)
    current_payload = joblib.load(paths["current_artifact"])
    if augmented_names != current_payload["augmented_feature_names"]:
        raise ValueError("Current scene-head feature schema changed")
    current_values = {
        "features": augmented,
        "augmented_names": augmented_names,
        "primary": raw_features[:, int(np.flatnonzero(base_names == "primary_connected_score")[0])],
    }
    current = current_scores(current_values, current_payload)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spatial_artifact = torch.load(paths["spatial_artifact"], map_location="cpu", weights_only=False)
    means, counts, inverse = build_site_templates(spatial_images, groups)
    spatial_raw = predict_model(
        spatial_artifact["fitted"], spatial_images, np.arange(labels.size),
        np.zeros(labels.size, dtype=np.uint8), means, counts, inverse, device,
    )
    spatial = blend_scores(current, spatial_raw, float(spatial_artifact["blend_weight"]))

    development = load_development(
        {"inner": paths["inner"], "fold0": paths["fold0"], "fold1": paths["fold1"]},
        paths["development_scores"],
    )
    source = load_features(paths["development_prithvi"], development, "cls_plus_base").astype(np.float64)
    target = np.concatenate((prithvi_cls, augmented.astype(np.float32)), axis=1).astype(np.float64)
    source_norm, target_norm = domain_normalize(source, target)
    positive_weight = float(np.sqrt((development["labels"] == 0).sum() / (development["labels"] == 1).sum()))
    weights = np.where(development["labels"] == 1, positive_weight, 1.0)
    adaptive_control = joblib.load(paths["prithvi_artifact"])
    prithvi_model = LogisticRegression(
        C=float(adaptive_control["C"]), max_iter=500, solver="lbfgs", random_state=20261550
    ).fit(source_norm, development["labels"], sample_weight=weights)
    prithvi_raw = prithvi_model.predict_proba(target_norm)[:, 1]
    prithvi = blend_scores(current, prithvi_raw, float(adaptive_control["blend_weight"]))

    ensemble_control = joblib.load(paths["ensemble_artifact"])
    calibration_control = joblib.load(paths["calibration_artifact"])
    if calibration_control["base_ensemble_sha256"] != protocol["inputs"]["ensemble_artifact"]["sha256"]:
        raise ValueError("Calibration artifact does not bind the frozen ensemble")
    candidate_raw = blend_scores(spatial, prithvi, float(ensemble_control["prithvi_weight"]))
    offset = float(calibration_control["logit_offset"])
    candidate = apply_offset(candidate_raw, offset)
    if not all(np.isfinite(values).all() for values in (current, spatial, prithvi, candidate_raw, candidate)):
        raise ValueError("Calibrated fresh scores contain non-finite values")
    if offset >= 0.0 or np.any(candidate >= candidate_raw):
        raise ValueError("Frozen calibration is not strictly score-lowering")

    current_threshold = float(protocol["thresholds"]["current"])
    candidate_threshold = float(protocol["thresholds"]["candidate"])
    masks = {
        "available": np.ones(labels.size, dtype=bool),
        "published_all_clear": all_clear,
        "published_nonclear": ~all_clear,
    }
    strata = {
        name: compare_stratum(current, candidate, mask, current_threshold, candidate_threshold)
        for name, mask in masks.items()
    }
    checks: dict[str, bool] = {}
    gate_spec = protocol["safety_gates"]
    for stratum in gate_spec["false_positive_count_strata"]:
        checks[f"{stratum}_false_positive_count_no_higher"] = strata[stratum]["checks"]["false_positive_count_no_higher"]
    for stratum in gate_spec["raw_score_p95_strata"]:
        checks[f"{stratum}_raw_score_p95_no_higher"] = strata[stratum]["checks"]["raw_score_p95_no_higher"]
    for stratum in gate_spec["logit_margin_p95_strata"]:
        checks[f"{stratum}_logit_margin_p95_no_higher"] = strata[stratum]["checks"]["logit_margin_p95_no_higher"]
    current_pred = current >= current_threshold
    candidate_pred = candidate >= candidate_threshold
    unavailable = int(expected["full_rows"]) - int(expected["available_rows"])
    full_bounds = {
        "full_rows": int(expected["full_rows"]),
        "available_rows": int(expected["available_rows"]),
        "unavailable_rows": unavailable,
        "current_worst_case_fpr": float((current_pred.sum() + unavailable) / int(expected["full_rows"])),
        "candidate_worst_case_fpr": float((candidate_pred.sum() + unavailable) / int(expected["full_rows"])),
        "symmetric_worst_case_candidate_no_worse": bool(candidate_pred.sum() <= current_pred.sum()),
    }
    checks["symmetric_adversarial_full_bound_no_worse"] = full_bounds["symmetric_worst_case_candidate_no_worse"]
    passed = all(checks.values())
    report = {
        "schema_version": 1,
        "scope": "post-test fresh external-negative safety replay for tail-calibrated spatial-Prithvi ensemble",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "post_test_reuse": True,
        "independent_external_confirmation": False,
        "reuse_reason": "the identical fresh cohort scores were inspected during the uncalibrated safety evaluation",
        "strata": strata,
        "full_cohort_bounds": full_bounds,
        "paired_transitions": {
            "both_false_positive": int(np.count_nonzero(current_pred & candidate_pred)),
            "current_only_false_positive": int(np.count_nonzero(current_pred & ~candidate_pred)),
            "candidate_only_false_positive": int(np.count_nonzero(~current_pred & candidate_pred)),
            "both_true_negative": int(np.count_nonzero(~current_pred & ~candidate_pred)),
        },
        "component_score_summary": {
            "spatial_p95": float(np.quantile(spatial, 0.95)),
            "prithvi_p95": float(np.quantile(prithvi, 0.95)),
            "raw_ensemble_p95": float(np.quantile(candidate_raw, 0.95)),
            "calibrated_ensemble_p95": float(np.quantile(candidate, 0.95)),
        },
        "calibration": {
            "logit_offset": offset,
            "artifact_sha256": protocol["inputs"]["calibration_artifact"]["sha256"],
            "selected_without_fresh_inputs": True,
            "ranking_invariant": True,
        },
        "safety_checks": checks,
        "all_safety_gates_pass": passed,
        "decision": "Authorize frozen exact MARS paper evaluation." if passed else "Reject calibrated ensemble before exact MARS paper evaluation.",
        "paper_test_accessed": False,
        "provenance": {
            "protocol_sha256": sha256(protocol_path),
            "script_sha256": sha256(Path(__file__).resolve()),
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "device": str(torch.cuda.get_device_name(device) if device.type == "cuda" else device),
        },
    }
    output_json = (ROOT / protocol["outputs"]["json"]).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown((ROOT / protocol["outputs"]["markdown"]).resolve(), report)
    print(json.dumps({
        "ok": passed,
        "checks": checks,
        "transitions": report["paired_transitions"],
        "offset": offset,
    }, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
