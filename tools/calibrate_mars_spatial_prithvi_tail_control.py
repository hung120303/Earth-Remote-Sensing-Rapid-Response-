#!/usr/bin/env python3
"""Fit a monotone development-tail offset for the spatial-Prithvi ensemble."""

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
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "EarthRemoteSensingRapidResponse"
for path in (MODEL_ROOT, ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from acquire_mars_metadata import sha256  # noqa: E402
from train_mars_crossfold_bagged_scene_head import load_development  # noqa: E402
from train_mars_scene_ranker import blend_scores  # noqa: E402
from train_mars_spatial_prithvi_ensemble import align_prithvi_scores  # noqa: E402
from train_mars_temporal_spatial_ensemble import align_spatial_scores  # noqa: E402


DEFAULT_PROTOCOL = Path("configs/mars_spatial_prithvi_tail_calibration_protocol.json")


def logit(values: np.ndarray | float) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=np.float64), 1e-8, 1.0 - 1e-8)
    return np.log(clipped) - np.log1p(-clipped)


def apply_offset(values: np.ndarray, offset: float) -> np.ndarray:
    shifted = logit(values) + float(offset)
    return 1.0 / (1.0 + np.exp(-np.clip(shifted, -40.0, 40.0)))


def selection_constraints(
    labels: np.ndarray,
    current: np.ndarray,
    raw: np.ndarray,
    folds: np.ndarray,
    selected_folds: list[int],
    threshold: float,
) -> dict[str, dict[str, float | int]]:
    """Return per-fold offsets that satisfy p95 and exact current FP budgets."""
    threshold_logit = float(logit(threshold))
    results: dict[str, dict[str, float | int]] = {}
    for fold in selected_folds:
        negatives = (folds == fold) & (labels == 0)
        current_negative = current[negatives]
        raw_negative = raw[negatives]
        current_fp = int(np.count_nonzero(current_negative >= threshold))
        descending = np.sort(raw_negative)[::-1]
        if current_fp >= descending.size:
            tail_offset = float("inf")
            first_disallowed_score = None
        else:
            first_disallowed_score = float(descending[current_fp])
            tail_offset = threshold_logit - float(logit(first_disallowed_score))
        p95_offset = (
            float(np.quantile(logit(current_negative), 0.95))
            - float(np.quantile(logit(raw_negative), 0.95))
        )
        results[str(fold)] = {
            "negative_count": int(raw_negative.size),
            "current_false_positive_budget": current_fp,
            "first_disallowed_raw_score": first_disallowed_score,
            "p95_offset_limit": p95_offset,
            "false_positive_tail_offset_limit": tail_offset,
            "binding_offset_limit": min(p95_offset, tail_offset),
        }
    return results


def fold_summary(
    labels: np.ndarray,
    current: np.ndarray,
    raw: np.ndarray,
    calibrated: np.ndarray,
    folds: np.ndarray,
    selected_folds: list[int],
    threshold: float,
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for fold in selected_folds:
        rows = folds == fold
        negatives = rows & (labels == 0)
        raw_ap = float(average_precision_score(labels[rows], raw[rows]))
        calibrated_ap = float(average_precision_score(labels[rows], calibrated[rows]))
        results[str(fold)] = {
            "current_negative_logit_p95": float(np.quantile(logit(current[negatives]), 0.95)),
            "raw_negative_logit_p95": float(np.quantile(logit(raw[negatives]), 0.95)),
            "calibrated_negative_logit_p95": float(np.quantile(logit(calibrated[negatives]), 0.95)),
            "raw_ap": raw_ap,
            "calibrated_ap": calibrated_ap,
            "ap_absolute_difference": abs(raw_ap - calibrated_ap),
            "current_false_positives": int(np.count_nonzero(current[negatives] >= threshold)),
            "raw_false_positives": int(np.count_nonzero(raw[negatives] >= threshold)),
            "calibrated_false_positives": int(np.count_nonzero(calibrated[negatives] >= threshold)),
        }
    return results


def checks(results: dict[str, Any], tolerance: float) -> dict[str, bool]:
    return {
        "every_fold_negative_p95_no_higher": all(
            value["calibrated_negative_logit_p95"] <= value["current_negative_logit_p95"] + 1e-12
            for value in results.values()
        ),
        "every_fold_false_positives_no_higher": all(
            value["calibrated_false_positives"] <= value["current_false_positives"]
            for value in results.values()
        ),
        "every_fold_ap_exact": all(
            value["ap_absolute_difference"] <= tolerance for value in results.values()
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL.as_posix())
    args = parser.parse_args()
    protocol_path = (ROOT / args.protocol).resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if sha256(Path(__file__).resolve()) != protocol["calibrator"]["sha256"]:
        raise ValueError("Spatial-Prithvi tail calibrator hash mismatch")
    paths: dict[str, Path] = {}
    for name, contract in protocol["inputs"].items():
        path = (ROOT / contract["path"]).resolve()
        if sha256(path) != contract["sha256"]:
            raise ValueError(f"Frozen tail-calibration input hash mismatch: {name}")
        paths[name] = path

    values = load_development(
        {"inner": paths["inner"], "fold0": paths["fold0"], "fold1": paths["fold1"]},
        paths["scores"],
    )
    spatial = align_spatial_scores(values, paths["spatial_scores"])
    prithvi = align_prithvi_scores(values, paths["prithvi_scores"])
    ensemble_control = joblib.load(paths["ensemble_artifact"])
    raw = blend_scores(spatial, prithvi, float(ensemble_control["prithvi_weight"]))
    selection_folds = list(map(int, protocol["folds"]["selection"]))
    audit_folds = list(map(int, protocol["folds"]["reused_holdout_audit"]))
    threshold = float(protocol["calibration"]["operational_threshold"])
    constraints = selection_constraints(
        values["labels"], values["current"], raw, values["folds"], selection_folds, threshold
    )
    binding_limit = min(float(value["binding_offset_limit"]) for value in constraints.values())
    offset = binding_limit - float(protocol["calibration"]["conservative_logit_margin"])
    if not np.isfinite(offset) or offset >= 0.0:
        raise RuntimeError("Frozen tail calibration did not produce a finite negative offset")
    calibrated = apply_offset(raw, offset)
    if np.any(calibrated >= raw):
        raise RuntimeError("Negative logit calibration did not strictly lower every score")

    tolerance = float(protocol["gates"]["ap_absolute_tolerance"])
    selection = fold_summary(
        values["labels"], values["current"], raw, calibrated,
        values["folds"], selection_folds, threshold,
    )
    audit = fold_summary(
        values["labels"], values["current"], raw, calibrated,
        values["folds"], audit_folds, threshold,
    )
    selection_checks = checks(selection, tolerance)
    audit_checks = checks(audit, tolerance)
    passed = all(selection_checks.values()) and all(audit_checks.values())
    artifact_record = None
    if passed:
        artifact_path = (ROOT / protocol["outputs"]["artifact"]).resolve()
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = artifact_path.with_suffix(artifact_path.suffix + ".tmp")
        joblib.dump(
            {
                "schema_version": 1,
                "kind": "mars_tail_calibrated_spatial_prithvi_ensemble",
                "base_ensemble_path": protocol["inputs"]["ensemble_artifact"]["path"],
                "base_ensemble_sha256": protocol["inputs"]["ensemble_artifact"]["sha256"],
                "logit_offset": offset,
                "binding_selection_limit": binding_limit,
                "conservative_logit_margin": protocol["calibration"]["conservative_logit_margin"],
                "operational_scene_threshold": threshold,
                "ranking_contract": "strictly monotone; AP, AUROC, and matched-FPR recall invariant",
                "protocol_sha256": sha256(protocol_path),
            },
            temporary,
            compress=3,
        )
        os.replace(temporary, artifact_path)
        artifact_record = {
            "path": protocol["outputs"]["artifact"],
            "bytes": artifact_path.stat().st_size,
            "sha256": sha256(artifact_path),
            "tracked": False,
        }

    report = {
        "schema_version": 1,
        "scope": "development-only monotone p95 and exact-tail calibration",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_constraints": constraints,
        "binding_selection_limit": binding_limit,
        "logit_offset": offset,
        "selection": {"folds": selection, "checks": selection_checks},
        "reused_holdout_audit": {
            "independent_confirmation": False,
            "reason": "folds 0/1 were exposed by the preceding p95-only calibration",
            "folds": audit,
            "checks": audit_checks,
        },
        "all_calibration_gates_pass": passed,
        "artifact": artifact_record,
        "decision": (
            "Freeze tail-calibrated ensemble for transparent fresh safety replay."
            if passed else "Reject tail calibration before fresh safety replay."
        ),
        "provenance": {
            "protocol_sha256": sha256(protocol_path),
            "script_sha256": sha256(Path(__file__).resolve()),
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "numpy": np.__version__,
            "joblib": joblib.__version__,
        },
    }
    output_json = (ROOT / protocol["outputs"]["json"]).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "ok": passed,
        "offset": offset,
        "selection": selection_checks,
        "reused_holdout_audit": audit_checks,
        "artifact": artifact_record,
    }, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
