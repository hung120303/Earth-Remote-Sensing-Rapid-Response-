#!/usr/bin/env python3
"""Select a non-temporal ensemble of spatial residual and adaptive Prithvi scores."""

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
from train_mars_crossfold_bagged_scene_head import load_development  # noqa: E402
from train_mars_oof_scene_ensemble_v2 import ap_group_bootstrap  # noqa: E402
from train_mars_scene_ranker import blend_scores, comparison, metric_summary  # noqa: E402
from train_mars_temporal_spatial_ensemble import align_spatial_scores  # noqa: E402


DEFAULT_PROTOCOL = Path("configs/mars_spatial_prithvi_ensemble_protocol.json")


def align_prithvi_scores(values: dict[str, Any], path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as cache:
        identifiers = cache["sample_ids"].astype(str)
        scores = cache["scores"].astype(np.float64)
    if identifiers.size != scores.size or len(set(identifiers.tolist())) != identifiers.size:
        raise ValueError("Adaptive Prithvi score identities are invalid")
    lookup = dict(zip(identifiers.tolist(), scores.tolist(), strict=True))
    if len(lookup) != values["sample_ids"].size:
        raise ValueError("Adaptive Prithvi score count differs from development")
    aligned = np.asarray([lookup[identifier] for identifier in values["sample_ids"]], dtype=np.float64)
    if not np.isfinite(aligned).all():
        raise ValueError("Adaptive Prithvi scores are non-finite")
    return aligned


def evaluate(
    values: dict[str, Any], scores: np.ndarray, folds: list[int]
) -> dict[str, Any]:
    rows = np.isin(values["folds"], folds)
    old = metric_summary(values["labels"][rows], values["current"][rows], values["sensors"][rows])
    new = metric_summary(values["labels"][rows], scores[rows], values["sensors"][rows])
    combined = {"current": old, "candidate": new, "versus_current": comparison(new, old)}
    per_fold = {}
    for fold in folds:
        local = values["folds"] == fold
        old_local = metric_summary(values["labels"][local], values["current"][local], values["sensors"][local])
        new_local = metric_summary(values["labels"][local], scores[local], values["sensors"][local])
        per_fold[str(fold)] = {"current": old_local, "candidate": new_local, "versus_current": comparison(new_local, old_local)}
    fold_ap = [value["versus_current"]["delta"]["average_precision"] for value in per_fold.values()]
    fold_recall = [value["versus_current"]["delta"]["recall_at_fpr_0_0713"] for value in per_fold.values()]
    delta = combined["versus_current"]["delta"]
    return {
        "combined": combined, "per_fold": per_fold,
        "rank": [
            min(fold_ap), delta["average_precision"], delta["recall_at_fpr_0_0713"],
            min(delta["sensor_average_precision"].values()),
        ],
    }


def checks(
    result: dict[str, Any], bootstrap: dict[str, Any], gates: dict[str, Any]
) -> dict[str, bool]:
    delta = result["combined"]["versus_current"]["delta"]
    fold_ap = [value["versus_current"]["delta"]["average_precision"] for value in result["per_fold"].values()]
    fold_recall = [value["versus_current"]["delta"]["recall_at_fpr_0_0713"] for value in result["per_fold"].values()]
    return {
        "ap_point_higher": delta["average_precision"] > 0.0,
        "ap_bootstrap_lower_positive": bootstrap["lower"] > 0.0,
        "recall_point_no_lower": delta["recall_at_fpr_0_0713"] >= 0.0,
        "each_fold_ap_within_tolerance": min(fold_ap) >= -float(gates["per_fold_ap_tolerance"]),
        "each_fold_recall_within_tolerance": min(fold_recall) >= -float(gates["per_fold_recall_tolerance"]),
        "each_sensor_ap_within_tolerance": min(delta["sensor_average_precision"].values()) >= -float(gates["sensor_ap_tolerance"]),
    }


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    selected = report["selection"]["selected"]
    confirmation = report["confirmation"]["metrics"]["combined"]["versus_current"]["delta"]
    interval = report["confirmation"]["bootstrap"]
    lines = [
        "# Spatial-Prithvi representation ensemble",
        "",
        f"- Selected Prithvi weight: **{selected['prithvi_weight']:.2f}**.",
        f"- Confirmation AP delta: **{confirmation['average_precision']:+.5f}**, interval **[{interval['lower']:+.5f}, {interval['upper']:+.5f}]**.",
        f"- Confirmation recall delta: **{confirmation['recall_at_fpr_0_0713']:+.5f}**.",
        f"- All promotion gates pass: **{str(report['all_promotion_gates_pass']).lower()}**.",
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
    if sha256(Path(__file__).resolve()) != protocol["trainer"]["sha256"]:
        raise ValueError("Spatial-Prithvi trainer hash mismatch")
    paths: dict[str, Path] = {}
    for name, contract in protocol["inputs"].items():
        path = (ROOT / contract["path"]).resolve()
        if sha256(path) != contract["sha256"]:
            raise ValueError(f"Frozen spatial-Prithvi input hash mismatch: {name}")
        paths[name] = path
    values = load_development(
        {"inner": paths["inner"], "fold0": paths["fold0"], "fold1": paths["fold1"]}, paths["scores"]
    )
    spatial = align_spatial_scores(values, paths["spatial_scores"])
    prithvi = align_prithvi_scores(values, paths["prithvi_scores"])
    selection_folds = list(map(int, protocol["folds"]["selection"]))
    confirmation_folds = list(map(int, protocol["folds"]["confirmation"]))
    candidates: list[dict[str, Any]] = []
    scores_by_weight: dict[float, np.ndarray] = {}
    for weight in protocol["search"]["prithvi_weights"]:
        scores = blend_scores(spatial, prithvi, float(weight))
        result = evaluate(values, scores, selection_folds)
        result["prithvi_weight"] = float(weight)
        result["rank"].append(-abs(float(weight) - 0.5))
        candidates.append(result)
        scores_by_weight[float(weight)] = scores
    selected = max(candidates, key=lambda value: tuple(value["rank"]))
    scores = scores_by_weight[selected["prithvi_weight"]]
    selection_rows = np.isin(values["folds"], selection_folds)
    selection_bootstrap = ap_group_bootstrap(
        values["labels"][selection_rows], values["current"][selection_rows], scores[selection_rows],
        values["groups"][selection_rows], replicates=int(protocol["bootstrap"]["replicates"]), seed=int(protocol["bootstrap"]["selection_seed"]),
    )
    selection_checks = checks(selected, selection_bootstrap, protocol["gates"])
    confirmation = evaluate(values, scores, confirmation_folds)
    confirmation_rows = np.isin(values["folds"], confirmation_folds)
    confirmation_bootstrap = ap_group_bootstrap(
        values["labels"][confirmation_rows], values["current"][confirmation_rows], scores[confirmation_rows],
        values["groups"][confirmation_rows], replicates=int(protocol["bootstrap"]["replicates"]), seed=int(protocol["bootstrap"]["confirmation_seed"]),
    )
    confirmation_checks = checks(confirmation, confirmation_bootstrap, protocol["gates"])
    passed = all(selection_checks.values()) and all(confirmation_checks.values())
    artifact_record = None
    if passed:
        artifact_path = (ROOT / protocol["outputs"]["artifact"]).resolve()
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = artifact_path.with_suffix(artifact_path.suffix + ".tmp")
        joblib.dump({
            "schema_version": 1, "kind": "mars_spatial_adaptive_prithvi_ensemble",
            "prithvi_weight": selected["prithvi_weight"],
            "spatial_artifact_path": protocol["inputs"]["spatial_artifact"]["path"],
            "spatial_artifact_sha256": protocol["inputs"]["spatial_artifact"]["sha256"],
            "adaptive_prithvi_artifact_path": protocol["inputs"]["prithvi_artifact"]["path"],
            "adaptive_prithvi_artifact_sha256": protocol["inputs"]["prithvi_artifact"]["sha256"],
            "operational_scene_threshold": float(protocol["base_architecture"]["operational_scene_threshold"]),
            "protocol_sha256": sha256(protocol_path),
        }, temporary, compress=3)
        os.replace(temporary, artifact_path)
        artifact_record = {"path": protocol["outputs"]["artifact"], "bytes": artifact_path.stat().st_size, "sha256": sha256(artifact_path), "tracked": False}
    report = {
        "schema_version": 1, "scope": "development-only non-temporal representation ensemble; paper cache not loaded",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection": {"candidates": candidates, "selected": selected, "bootstrap": selection_bootstrap, "checks": selection_checks},
        "confirmation": {"metrics": confirmation, "bootstrap": confirmation_bootstrap, "checks": confirmation_checks},
        "all_promotion_gates_pass": passed, "artifact": artifact_record,
        "decision": "Freeze the spatial-Prithvi ensemble for fresh safety evaluation." if passed else "Reject the spatial-Prithvi ensemble before fresh or paper evaluation.",
        "provenance": {"protocol_sha256": sha256(protocol_path), "script_sha256": sha256(Path(__file__).resolve()), "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), "numpy": np.__version__, "joblib": joblib.__version__},
    }
    output_json = (ROOT / protocol["outputs"]["json"]).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown((ROOT / protocol["outputs"]["markdown"]).resolve(), report)
    print(json.dumps({"ok": passed, "weight": selected["prithvi_weight"], "selection": selection_checks, "confirmation": confirmation_checks, "artifact": artifact_record}, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
