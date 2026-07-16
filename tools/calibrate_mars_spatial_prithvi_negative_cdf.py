#!/usr/bin/env python3
"""Fit a monotone offset enforcing development-negative empirical-CDF dominance."""

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

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "EarthRemoteSensingRapidResponse"
for path in (MODEL_ROOT, ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from acquire_mars_metadata import sha256  # noqa: E402
from calibrate_mars_spatial_prithvi_tail_control import (  # noqa: E402
    apply_offset,
    checks,
    fold_summary,
    logit,
)
from train_mars_crossfold_bagged_scene_head import load_development  # noqa: E402
from train_mars_scene_ranker import blend_scores  # noqa: E402
from train_mars_spatial_prithvi_ensemble import align_prithvi_scores  # noqa: E402
from train_mars_temporal_spatial_ensemble import align_spatial_scores  # noqa: E402


DEFAULT_PROTOCOL = Path("configs/mars_spatial_prithvi_negative_cdf_calibration_protocol.json")


def cdf_constraints(
    labels: np.ndarray,
    current: np.ndarray,
    raw: np.ndarray,
    folds: np.ndarray,
    selected_folds: list[int],
) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for fold in selected_folds:
        negatives = (folds == fold) & (labels == 0)
        current_sorted = np.sort(current[negatives])
        raw_sorted = np.sort(raw[negatives])
        if current_sorted.shape != raw_sorted.shape or current_sorted.size == 0:
            raise ValueError(f"Invalid negative-score arrays for fold {fold}")
        limits = logit(current_sorted) - logit(raw_sorted)
        binding_rank = int(np.argmin(limits))
        results[str(fold)] = {
            "negative_count": int(current_sorted.size),
            "offset_limit": float(limits[binding_rank]),
            "binding_ascending_rank_zero_based": binding_rank,
            "binding_empirical_quantile": float(binding_rank / max(current_sorted.size - 1, 1)),
            "binding_current_score": float(current_sorted[binding_rank]),
            "binding_raw_score": float(raw_sorted[binding_rank]),
            "minimum_raw_minus_current_score": float(np.min(raw_sorted - current_sorted)),
            "maximum_raw_minus_current_score": float(np.max(raw_sorted - current_sorted)),
        }
    return results


def dominance_checks(
    labels: np.ndarray,
    current: np.ndarray,
    calibrated: np.ndarray,
    folds: np.ndarray,
    selected_folds: list[int],
) -> dict[str, bool]:
    return {
        str(fold): bool(np.all(
            np.sort(calibrated[(folds == fold) & (labels == 0)])
            <= np.sort(current[(folds == fold) & (labels == 0)]) + 1e-12
        ))
        for fold in selected_folds
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL.as_posix())
    args = parser.parse_args()
    protocol_path = (ROOT / args.protocol).resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if sha256(Path(__file__).resolve()) != protocol["calibrator"]["sha256"]:
        raise ValueError("Negative-CDF calibrator hash mismatch")
    for dependency in protocol["code_dependencies"]:
        path = (ROOT / dependency["path"]).resolve()
        if sha256(path) != dependency["sha256"]:
            raise ValueError(f"Frozen code dependency mismatch: {dependency['path']}")
    paths: dict[str, Path] = {}
    for name, contract in protocol["inputs"].items():
        path = (ROOT / contract["path"]).resolve()
        if sha256(path) != contract["sha256"]:
            raise ValueError(f"Frozen negative-CDF input hash mismatch: {name}")
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
    constraints = cdf_constraints(
        values["labels"], values["current"], raw, values["folds"], selection_folds
    )
    binding_limit = min(float(value["offset_limit"]) for value in constraints.values())
    offset = binding_limit - float(protocol["calibration"]["conservative_logit_margin"])
    if not np.isfinite(offset) or offset >= 0.0:
        raise RuntimeError("Frozen negative-CDF calibration did not produce a finite negative offset")
    calibrated = apply_offset(raw, offset)
    if np.any(calibrated >= raw):
        raise RuntimeError("Negative-CDF calibration did not strictly lower every score")

    threshold = float(protocol["calibration"]["operational_threshold"])
    tolerance = float(protocol["gates"]["ap_absolute_tolerance"])
    selection = fold_summary(
        values["labels"], values["current"], raw, calibrated,
        values["folds"], selection_folds, threshold,
    )
    audit = fold_summary(
        values["labels"], values["current"], raw, calibrated,
        values["folds"], audit_folds, threshold,
    )
    selection_standard = checks(selection, tolerance)
    audit_standard = checks(audit, tolerance)
    selection_dominance = dominance_checks(
        values["labels"], values["current"], calibrated, values["folds"], selection_folds
    )
    audit_dominance = dominance_checks(
        values["labels"], values["current"], calibrated, values["folds"], audit_folds
    )
    passed = (
        all(selection_standard.values())
        and all(audit_standard.values())
        and all(selection_dominance.values())
        and all(audit_dominance.values())
    )

    artifact_record = None
    if passed:
        artifact_path = (ROOT / protocol["outputs"]["artifact"]).resolve()
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = artifact_path.with_suffix(artifact_path.suffix + ".tmp")
        joblib.dump(
            {
                "schema_version": 1,
                "kind": "mars_negative_cdf_calibrated_spatial_prithvi_ensemble",
                "base_ensemble_path": protocol["inputs"]["ensemble_artifact"]["path"],
                "base_ensemble_sha256": protocol["inputs"]["ensemble_artifact"]["sha256"],
                "logit_offset": offset,
                "binding_selection_limit": binding_limit,
                "conservative_logit_margin": protocol["calibration"]["conservative_logit_margin"],
                "operational_scene_threshold": threshold,
                "ranking_contract": "strictly monotone; AP, AUROC, and matched-FPR recall invariant",
                "negative_calibration_contract": "selection-fold empirical CDF first-order dominance",
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
        "scope": "development-only negative empirical-CDF dominance calibration",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_constraints": constraints,
        "binding_selection_limit": binding_limit,
        "logit_offset": offset,
        "selection": {
            "folds": selection,
            "standard_checks": selection_standard,
            "negative_cdf_dominance_by_fold": selection_dominance,
        },
        "reused_holdout_audit": {
            "independent_confirmation": False,
            "reason": "folds 0/1 were exposed by preceding calibration-family experiments",
            "folds": audit,
            "standard_checks": audit_standard,
            "negative_cdf_dominance_by_fold": audit_dominance,
        },
        "all_calibration_gates_pass": passed,
        "artifact": artifact_record,
        "decision": (
            "Freeze negative-CDF calibrated ensemble for transparent fresh replay."
            if passed else "Reject negative-CDF calibration before fresh replay."
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
        "selection_cdf": selection_dominance,
        "reused_holdout_cdf": audit_dominance,
        "artifact": artifact_record,
    }, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
