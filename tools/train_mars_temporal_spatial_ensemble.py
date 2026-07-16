#!/usr/bin/env python3
"""Select a site-relative spatial head with temporal corroboration for unseen sites."""

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
from train_mars_oof_scene_ensemble_v2 import ap_group_bootstrap  # noqa: E402
from train_mars_scene_ranker import comparison, metric_summary  # noqa: E402
from train_mars_temporal_site_prior import temporal_site_prior  # noqa: E402
from train_mars_unseen_low_prevalence_router import low_prevalence_mask  # noqa: E402


DEFAULT_PROTOCOL = Path("configs/mars_temporal_spatial_ensemble_protocol.json")


def align_spatial_scores(values: dict[str, Any], path: Path) -> np.ndarray:
    lookup: dict[str, float] = {}
    with np.load(path, allow_pickle=False) as cache:
        for partition in ("inner", "fold0", "fold1"):
            identifiers = cache[f"{partition}_sample_ids"].astype(str)
            scores = cache[f"{partition}_scores"].astype(np.float64)
            if identifiers.size != scores.size:
                raise ValueError(f"{partition} spatial score identity mismatch")
            for identifier, score in zip(identifiers, scores, strict=True):
                if identifier in lookup:
                    raise ValueError(f"Duplicate spatial score identity: {identifier}")
                lookup[identifier] = float(score)
    if len(lookup) != values["sample_ids"].size:
        raise ValueError("Spatial score cache row count differs from development")
    aligned = np.asarray([lookup[identifier] for identifier in values["sample_ids"]], dtype=np.float64)
    if not np.isfinite(aligned).all() or np.any((aligned < 0.0) | (aligned > 1.0)):
        raise ValueError("Aligned spatial scores are invalid")
    return aligned


def evaluate_rows(
    values: dict[str, Any], scores: np.ndarray, rows: np.ndarray
) -> dict[str, Any]:
    current = metric_summary(values["labels"][rows], values["current"][rows], values["sensors"][rows])
    candidate = metric_summary(values["labels"][rows], scores[rows], values["sensors"][rows])
    return {"current": current, "candidate": candidate, "versus_current": comparison(candidate, current)}


def score_candidate(
    spatial: np.ndarray, groups: np.ndarray, min_size: int, top_k: int, weight: float
) -> np.ndarray:
    if weight == 0.0:
        return spatial.copy()
    return temporal_site_prior(spatial, groups, top_k, weight, min_size)


def evaluate_candidate(
    values: dict[str, Any], scores: np.ndarray, low_rows: np.ndarray, folds: list[int]
) -> dict[str, Any]:
    selected = np.isin(values["folds"], folds)
    low_selected = selected & low_rows
    low_combined = evaluate_rows(values, scores, low_selected)
    whole_combined = evaluate_rows(values, scores, selected)
    low_per_fold = {
        str(fold): evaluate_rows(values, scores, (values["folds"] == fold) & low_rows)
        for fold in folds
    }
    whole_per_fold = {
        str(fold): evaluate_rows(values, scores, values["folds"] == fold)
        for fold in folds
    }
    low_ap = [value["versus_current"]["delta"]["average_precision"] for value in low_per_fold.values()]
    low_recall = [value["versus_current"]["delta"]["recall_at_fpr_0_0713"] for value in low_per_fold.values()]
    whole_ap = [value["versus_current"]["delta"]["average_precision"] for value in whole_per_fold.values()]
    return {
        "low_prevalence_combined": low_combined,
        "whole_combined": whole_combined,
        "low_prevalence_per_fold": low_per_fold,
        "whole_per_fold": whole_per_fold,
        "rank": [
            min(low_ap),
            low_combined["versus_current"]["delta"]["average_precision"],
            min(whole_ap),
            whole_combined["versus_current"]["delta"]["average_precision"],
            min(low_recall),
        ],
    }


def bootstrap_view(
    values: dict[str, Any], scores: np.ndarray, rows: np.ndarray, replicates: int, seed: int
) -> dict[str, Any]:
    return ap_group_bootstrap(
        values["labels"][rows], values["current"][rows], scores[rows],
        values["groups"][rows], replicates=replicates, seed=seed,
    )


def promotion_checks(
    result: dict[str, Any], low_bootstrap: dict[str, Any], whole_bootstrap: dict[str, Any],
    gates: dict[str, Any], folds: list[int],
) -> dict[str, bool]:
    low_delta = result["low_prevalence_combined"]["versus_current"]["delta"]
    whole_delta = result["whole_combined"]["versus_current"]["delta"]
    low_ap = [result["low_prevalence_per_fold"][str(f)]["versus_current"]["delta"]["average_precision"] for f in folds]
    low_recall = [result["low_prevalence_per_fold"][str(f)]["versus_current"]["delta"]["recall_at_fpr_0_0713"] for f in folds]
    whole_ap = [result["whole_per_fold"][str(f)]["versus_current"]["delta"]["average_precision"] for f in folds]
    whole_recall = [result["whole_per_fold"][str(f)]["versus_current"]["delta"]["recall_at_fpr_0_0713"] for f in folds]
    return {
        "low_ap_point_higher": low_delta["average_precision"] > 0.0,
        "low_recall_point_no_lower": low_delta["recall_at_fpr_0_0713"] >= 0.0,
        "low_ap_bootstrap_lower_positive": low_bootstrap["lower"] > 0.0,
        "whole_ap_point_higher": whole_delta["average_precision"] > 0.0,
        "whole_recall_point_no_lower": whole_delta["recall_at_fpr_0_0713"] >= 0.0,
        "whole_ap_bootstrap_lower_positive": whole_bootstrap["lower"] > 0.0,
        "each_low_fold_ap_within_tolerance": min(low_ap) >= -float(gates["per_fold_low_ap_tolerance"]),
        "each_low_fold_recall_within_tolerance": min(low_recall) >= -float(gates["per_fold_low_recall_tolerance"]),
        "each_whole_fold_ap_within_tolerance": min(whole_ap) >= -float(gates["per_fold_whole_ap_tolerance"]),
        "each_whole_fold_recall_within_tolerance": min(whole_recall) >= -float(gates["per_fold_whole_recall_tolerance"]),
    }


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    selected = report["selection"]["selected"]
    confirmation = report["confirmation"]
    low = confirmation["metrics"]["low_prevalence_combined"]["versus_current"]["delta"]
    whole = confirmation["metrics"]["whole_combined"]["versus_current"]["delta"]
    lines = [
        "# Temporal-spatial unseen-site ensemble",
        "",
        f"- Selected minimum history: **{selected['min_site_size']}**; top-k: **{selected['top_k']}**; temporal weight: **{selected['weight']:.2f}**.",
        f"- Confirmation low-prevalence AP/recall deltas: **{low['average_precision']:+.5f} / {low['recall_at_fpr_0_0713']:+.5f}**.",
        f"- Confirmation whole-view AP/recall deltas: **{whole['average_precision']:+.5f} / {whole['recall_at_fpr_0_0713']:+.5f}**.",
        f"- All promotion gates pass: **{str(report['all_promotion_gates_pass']).lower()}**.",
        "",
        "The spatial component scores all scenes; temporal corroboration is applied only to unseen sites at deployment.",
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
        raise ValueError("Temporal-spatial trainer hash mismatch")
    paths: dict[str, Path] = {}
    for name, contract in protocol["inputs"].items():
        path = (ROOT / contract["path"]).resolve()
        if sha256(path) != contract["sha256"]:
            raise ValueError(f"Frozen temporal-spatial input hash mismatch: {name}")
        paths[name] = path
    values = load_development(
        {"inner": paths["inner"], "fold0": paths["fold0"], "fold1": paths["fold1"]},
        paths["scores"],
    )
    spatial = align_spatial_scores(values, paths["spatial_scores"])
    maximum_rate = float(protocol["target_domain"]["maximum_site_positive_rate"])
    low_rows = low_prevalence_mask(values["labels"], values["groups"], maximum_rate)
    selection_folds = list(map(int, protocol["folds"]["selection"]))
    confirmation_folds = list(map(int, protocol["folds"]["confirmation"]))
    candidates: list[dict[str, Any]] = []
    scores_by_key: dict[str, np.ndarray] = {}
    for min_size, top_k, weight in itertools.product(
        protocol["search"]["min_site_size"], protocol["search"]["top_k"], protocol["search"]["weights"]
    ):
        scores = score_candidate(spatial, values["groups"], int(min_size), int(top_k), float(weight))
        result = evaluate_candidate(values, scores, low_rows, selection_folds)
        result.update({"min_site_size": int(min_size), "top_k": int(top_k), "weight": float(weight)})
        result["key"] = f"min{min_size}_top{top_k}_weight{weight}"
        candidates.append(result)
        scores_by_key[result["key"]] = scores
    selected = max(candidates, key=lambda value: tuple(value["rank"]))
    selected_scores = scores_by_key[selected["key"]]
    replicates = int(protocol["bootstrap"]["replicates"])
    selection_rows = np.isin(values["folds"], selection_folds)
    selection_low_rows = selection_rows & low_rows
    selection_low_bootstrap = bootstrap_view(values, selected_scores, selection_low_rows, replicates, int(protocol["bootstrap"]["selection_low_seed"]))
    selection_whole_bootstrap = bootstrap_view(values, selected_scores, selection_rows, replicates, int(protocol["bootstrap"]["selection_whole_seed"]))
    selection_checks = promotion_checks(selected, selection_low_bootstrap, selection_whole_bootstrap, protocol["gates"], selection_folds)

    confirmation_metrics = evaluate_candidate(values, selected_scores, low_rows, confirmation_folds)
    confirmation_rows = np.isin(values["folds"], confirmation_folds)
    confirmation_low_rows = confirmation_rows & low_rows
    confirmation_low_bootstrap = bootstrap_view(values, selected_scores, confirmation_low_rows, replicates, int(protocol["bootstrap"]["confirmation_low_seed"]))
    confirmation_whole_bootstrap = bootstrap_view(values, selected_scores, confirmation_rows, replicates, int(protocol["bootstrap"]["confirmation_whole_seed"]))
    confirmation_checks = promotion_checks(confirmation_metrics, confirmation_low_bootstrap, confirmation_whole_bootstrap, protocol["gates"], confirmation_folds)
    passed = all(selection_checks.values()) and all(confirmation_checks.values())
    artifact_record = None
    if passed:
        artifact_path = (ROOT / protocol["outputs"]["artifact"]).resolve()
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = artifact_path.with_suffix(artifact_path.suffix + ".tmp")
        joblib.dump({
            "schema_version": 1,
            "kind": "mars_temporal_spatial_unseen_site_ensemble",
            "spatial_artifact_path": protocol["inputs"]["spatial_artifact"]["path"],
            "spatial_artifact_sha256": protocol["inputs"]["spatial_artifact"]["sha256"],
            "known_training_sites": sorted(set(values["groups"].tolist())),
            "min_site_size": selected["min_site_size"], "top_k": selected["top_k"], "weight": selected["weight"],
            "operational_scene_threshold": float(protocol["base_architecture"]["operational_scene_threshold"]),
            "protocol_sha256": sha256(protocol_path),
        }, temporary, compress=3)
        os.replace(temporary, artifact_path)
        artifact_record = {
            "path": protocol["outputs"]["artifact"], "bytes": artifact_path.stat().st_size,
            "sha256": sha256(artifact_path), "tracked": False,
            "known_training_sites": len(set(values["groups"].tolist())),
        }
    report = {
        "schema_version": 1,
        "scope": "development-only temporal-spatial ensemble; exact paper cache not loaded",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "target_domain": {"maximum_site_positive_rate": maximum_rate},
        "selection": {
            "candidates": candidates, "selected": selected,
            "low_prevalence_bootstrap": selection_low_bootstrap,
            "whole_view_bootstrap": selection_whole_bootstrap, "checks": selection_checks,
        },
        "confirmation": {
            "metrics": confirmation_metrics,
            "low_prevalence_bootstrap": confirmation_low_bootstrap,
            "whole_view_bootstrap": confirmation_whole_bootstrap, "checks": confirmation_checks,
        },
        "all_promotion_gates_pass": passed, "artifact": artifact_record,
        "decision": "Freeze the temporal-spatial ensemble for fresh safety evaluation." if passed else "Reject the temporal-spatial ensemble before fresh or paper evaluation.",
        "provenance": {
            "protocol_sha256": sha256(protocol_path), "script_sha256": sha256(Path(__file__).resolve()),
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "numpy": np.__version__, "joblib": joblib.__version__,
        },
    }
    output_json = (ROOT / protocol["outputs"]["json"]).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown((ROOT / protocol["outputs"]["markdown"]).resolve(), report)
    print(json.dumps({"ok": passed, "selected": selected["key"], "selection": selection_checks, "confirmation": confirmation_checks, "artifact": artifact_record}, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
