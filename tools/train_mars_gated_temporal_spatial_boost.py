#!/usr/bin/env python3
"""Select a one-sided high-confidence temporal boost above the spatial scene head."""

from __future__ import annotations

import argparse
import itertools
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
from train_mars_temporal_spatial_ensemble import (  # noqa: E402
    align_spatial_scores,
    bootstrap_view,
    evaluate_candidate,
    promotion_checks,
)
from train_mars_unseen_low_prevalence_router import low_prevalence_mask  # noqa: E402


DEFAULT_PROTOCOL = Path("configs/mars_gated_temporal_spatial_boost_protocol.json")


def logit(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=np.float64), 1e-7, 1.0 - 1e-7)
    return np.log(clipped) - np.log1p(-clipped)


def one_sided_site_boost(
    spatial: np.ndarray,
    groups: np.ndarray,
    *,
    min_site_size: int,
    top_k: int,
    confidence_cutoff: float,
    weight: float,
) -> np.ndarray:
    spatial = np.asarray(spatial, dtype=np.float64)
    groups = np.asarray(groups).astype(str)
    source_logits = logit(spatial)
    output_logits = source_logits.copy()
    changed = np.zeros(spatial.size, dtype=bool)
    cutoff_logit = float(logit(np.asarray([confidence_cutoff]))[0])
    for group in np.unique(groups):
        rows = np.flatnonzero(groups == group)
        if rows.size < min_site_size:
            continue
        local = source_logits[rows]
        evidence = float(np.mean(np.sort(local)[-min(top_k, rows.size) :]))
        excess = max(0.0, evidence - cutoff_logit)
        output_logits[rows] += float(weight) * excess
        if excess > 0.0 and weight > 0.0:
            changed[rows] = True
    output = 1.0 / (1.0 + np.exp(-np.clip(output_logits, -40.0, 40.0)))
    output[~changed] = spatial[~changed]
    if np.any(output + 1e-14 < spatial):
        raise RuntimeError("One-sided temporal boost lowered a spatial score")
    return output


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    selected = report["selection"]["selected"]
    low = report["confirmation"]["metrics"]["low_prevalence_combined"]["versus_current"]["delta"]
    whole = report["confirmation"]["metrics"]["whole_combined"]["versus_current"]["delta"]
    lines = [
        "# Gated temporal-spatial boost",
        "",
        f"- Selected minimum history: **{selected['min_site_size']}**; top-k: **{selected['top_k']}**; cutoff: **{selected['confidence_cutoff']:.2f}**; weight: **{selected['weight']:.2f}**.",
        f"- Confirmation target AP/recall deltas: **{low['average_precision']:+.5f} / {low['recall_at_fpr_0_0713']:+.5f}**.",
        f"- Confirmation whole AP/recall deltas: **{whole['average_precision']:+.5f} / {whole['recall_at_fpr_0_0713']:+.5f}**.",
        f"- All promotion gates pass: **{str(report['all_promotion_gates_pass']).lower()}**.",
        "",
        "The temporal rule is mathematically one-sided: candidate scores never fall below the validated spatial score.",
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
        raise ValueError("Gated temporal-spatial trainer hash mismatch")
    paths: dict[str, Path] = {}
    for name, contract in protocol["inputs"].items():
        path = (ROOT / contract["path"]).resolve()
        if sha256(path) != contract["sha256"]:
            raise ValueError(f"Frozen gated-boost input hash mismatch: {name}")
        paths[name] = path
    values = load_development(
        {"inner": paths["inner"], "fold0": paths["fold0"], "fold1": paths["fold1"]}, paths["scores"]
    )
    spatial = align_spatial_scores(values, paths["spatial_scores"])
    low_rows = low_prevalence_mask(
        values["labels"], values["groups"], float(protocol["target_domain"]["maximum_site_positive_rate"])
    )
    selection_folds = list(map(int, protocol["folds"]["selection"]))
    confirmation_folds = list(map(int, protocol["folds"]["confirmation"]))
    candidates: list[dict[str, Any]] = []
    scores_by_key: dict[str, np.ndarray] = {}
    for min_size, top_k, cutoff, weight in itertools.product(
        protocol["search"]["min_site_size"], protocol["search"]["top_k"],
        protocol["search"]["confidence_cutoffs"], protocol["search"]["weights"],
    ):
        scores = one_sided_site_boost(
            spatial, values["groups"], min_site_size=int(min_size), top_k=int(top_k),
            confidence_cutoff=float(cutoff), weight=float(weight),
        )
        result = evaluate_candidate(values, scores, low_rows, selection_folds)
        result.update({
            "min_site_size": int(min_size), "top_k": int(top_k),
            "confidence_cutoff": float(cutoff), "weight": float(weight),
        })
        result["key"] = f"min{min_size}_top{top_k}_cutoff{cutoff}_weight{weight}"
        candidates.append(result)
        scores_by_key[result["key"]] = scores
    selected = max(candidates, key=lambda value: tuple(value["rank"]))
    scores = scores_by_key[selected["key"]]
    replicates = int(protocol["bootstrap"]["replicates"])
    selection_rows = np.isin(values["folds"], selection_folds)
    selection_low = selection_rows & low_rows
    selection_low_bootstrap = bootstrap_view(values, scores, selection_low, replicates, int(protocol["bootstrap"]["selection_low_seed"]))
    selection_whole_bootstrap = bootstrap_view(values, scores, selection_rows, replicates, int(protocol["bootstrap"]["selection_whole_seed"]))
    selection_checks = promotion_checks(selected, selection_low_bootstrap, selection_whole_bootstrap, protocol["gates"], selection_folds)
    confirmation_metrics = evaluate_candidate(values, scores, low_rows, confirmation_folds)
    confirmation_rows = np.isin(values["folds"], confirmation_folds)
    confirmation_low = confirmation_rows & low_rows
    confirmation_low_bootstrap = bootstrap_view(values, scores, confirmation_low, replicates, int(protocol["bootstrap"]["confirmation_low_seed"]))
    confirmation_whole_bootstrap = bootstrap_view(values, scores, confirmation_rows, replicates, int(protocol["bootstrap"]["confirmation_whole_seed"]))
    confirmation_checks = promotion_checks(confirmation_metrics, confirmation_low_bootstrap, confirmation_whole_bootstrap, protocol["gates"], confirmation_folds)
    passed = all(selection_checks.values()) and all(confirmation_checks.values())
    artifact_record = None
    if passed:
        artifact_path = (ROOT / protocol["outputs"]["artifact"]).resolve()
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = artifact_path.with_suffix(artifact_path.suffix + ".tmp")
        joblib.dump({
            "schema_version": 1, "kind": "mars_gated_temporal_spatial_boost",
            "spatial_artifact_path": protocol["inputs"]["spatial_artifact"]["path"],
            "spatial_artifact_sha256": protocol["inputs"]["spatial_artifact"]["sha256"],
            "known_training_sites": sorted(set(values["groups"].tolist())),
            "min_site_size": selected["min_site_size"], "top_k": selected["top_k"],
            "confidence_cutoff": selected["confidence_cutoff"], "weight": selected["weight"],
            "operational_scene_threshold": float(protocol["base_architecture"]["operational_scene_threshold"]),
            "protocol_sha256": sha256(protocol_path),
        }, temporary, compress=3)
        os.replace(temporary, artifact_path)
        artifact_record = {"path": protocol["outputs"]["artifact"], "bytes": artifact_path.stat().st_size, "sha256": sha256(artifact_path), "tracked": False}
    report = {
        "schema_version": 1, "scope": "development-only gated temporal-spatial boost; exact paper cache not loaded",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection": {"candidates": candidates, "selected": selected, "low_bootstrap": selection_low_bootstrap, "whole_bootstrap": selection_whole_bootstrap, "checks": selection_checks},
        "confirmation": {"metrics": confirmation_metrics, "low_bootstrap": confirmation_low_bootstrap, "whole_bootstrap": confirmation_whole_bootstrap, "checks": confirmation_checks},
        "all_promotion_gates_pass": passed, "artifact": artifact_record,
        "decision": "Freeze gated temporal-spatial boost for fresh safety evaluation." if passed else "Reject gated temporal-spatial boost before fresh or paper evaluation.",
        "provenance": {"protocol_sha256": sha256(protocol_path), "script_sha256": sha256(Path(__file__).resolve()), "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), "numpy": np.__version__, "joblib": joblib.__version__},
    }
    output_json = (ROOT / protocol["outputs"]["json"]).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown((ROOT / protocol["outputs"]["markdown"]).resolve(), report)
    print(json.dumps({"ok": passed, "selected": selected["key"], "selection": selection_checks, "confirmation": confirmation_checks, "artifact": artifact_record}, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
