#!/usr/bin/env python3
"""Select temporal-spatial routing under a preregistered rare-site target mixture."""

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
from sklearn.metrics import average_precision_score, roc_curve

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "EarthRemoteSensingRapidResponse"
for path in (MODEL_ROOT, ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from acquire_mars_metadata import sha256  # noqa: E402
from train_mars_crossfold_bagged_scene_head import load_development  # noqa: E402
from train_mars_scene_ranker import comparison, metric_summary  # noqa: E402
from train_mars_temporal_spatial_ensemble import align_spatial_scores, score_candidate  # noqa: E402


DEFAULT_PROTOCOL = Path("configs/mars_target_mixture_temporal_spatial_protocol.json")
TARGET_FPR = 0.0713


def build_cohort_plans(
    labels: np.ndarray,
    groups: np.ndarray,
    folds: np.ndarray,
    selected_folds: list[int],
    *,
    replicates: int,
    sites_per_fold: int,
    positive_sites_per_fold: int,
    maximum_scenes_per_site: int,
    seed: int,
) -> list[dict[int, np.ndarray]]:
    rng = np.random.default_rng(seed)
    site_tables: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for fold in selected_folds:
        fold_sites = np.unique(groups[folds == fold])
        positive = np.asarray(
            [site for site in fold_sites if np.any(labels[(folds == fold) & (groups == site)] == 1)]
        )
        negative = np.asarray(
            [site for site in fold_sites if not np.any(labels[(folds == fold) & (groups == site)] == 1)]
        )
        required_negative = sites_per_fold - positive_sites_per_fold
        if positive.size < positive_sites_per_fold or negative.size < required_negative:
            raise ValueError(f"Fold {fold} cannot support the frozen target mixture")
        site_tables[fold] = (positive, negative)
    plans: list[dict[int, np.ndarray]] = []
    for _ in range(replicates):
        replicate: dict[int, np.ndarray] = {}
        for fold in selected_folds:
            positive, negative = site_tables[fold]
            chosen = np.concatenate(
                [
                    rng.choice(positive, size=positive_sites_per_fold, replace=False),
                    rng.choice(
                        negative,
                        size=sites_per_fold - positive_sites_per_fold,
                        replace=False,
                    ),
                ]
            )
            rows: list[np.ndarray] = []
            for site in chosen:
                available = np.flatnonzero((folds == fold) & (groups == site))
                if available.size > maximum_scenes_per_site:
                    available = rng.choice(
                        available, size=maximum_scenes_per_site, replace=False
                    )
                rows.append(np.sort(available))
            replicate[fold] = np.concatenate(rows)
        plans.append(replicate)
    return plans


def ap_recall(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    ap = float(average_precision_score(labels, scores))
    fpr, tpr, _ = roc_curve(labels, scores, drop_intermediate=False)
    valid = fpr <= TARGET_FPR + 1e-12
    recall = float(np.max(tpr[valid]))
    return ap, recall


def distribution_summary(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "lower": float(np.quantile(values, 0.025)),
        "upper": float(np.quantile(values, 0.975)),
        "fraction_positive": float(np.mean(values > 0.0)),
    }


def simulate(
    values: dict[str, Any],
    spatial: np.ndarray,
    plans: list[dict[int, np.ndarray]],
    folds: list[int],
    min_site_size: int,
    top_k: int,
    weight: float,
) -> dict[str, Any]:
    combined_ap: list[float] = []
    combined_recall: list[float] = []
    per_fold_ap: dict[int, list[float]] = {fold: [] for fold in folds}
    per_fold_recall: dict[int, list[float]] = {fold: [] for fold in folds}
    row_counts: list[int] = []
    for plan in plans:
        combined_rows = np.concatenate([plan[fold] for fold in folds])
        row_counts.append(int(combined_rows.size))
        candidate = score_candidate(
            spatial[combined_rows], values["groups"][combined_rows],
            min_site_size, top_k, weight,
        )
        current_ap, current_recall = ap_recall(
            values["labels"][combined_rows], values["current"][combined_rows]
        )
        candidate_ap, candidate_recall = ap_recall(
            values["labels"][combined_rows], candidate
        )
        combined_ap.append(candidate_ap - current_ap)
        combined_recall.append(candidate_recall - current_recall)
        offset = 0
        for fold in folds:
            rows = plan[fold]
            local_candidate = candidate[offset : offset + rows.size]
            offset += rows.size
            old_ap, old_recall = ap_recall(values["labels"][rows], values["current"][rows])
            new_ap, new_recall = ap_recall(values["labels"][rows], local_candidate)
            per_fold_ap[fold].append(new_ap - old_ap)
            per_fold_recall[fold].append(new_recall - old_recall)
    return {
        "combined_ap_delta": distribution_summary(np.asarray(combined_ap)),
        "combined_recall_delta": distribution_summary(np.asarray(combined_recall)),
        "per_fold": {
            str(fold): {
                "ap_delta": distribution_summary(np.asarray(per_fold_ap[fold])),
                "recall_delta": distribution_summary(np.asarray(per_fold_recall[fold])),
            }
            for fold in folds
        },
        "replicates": len(plans),
        "rows_per_replicate": {
            "minimum": min(row_counts), "median": float(np.median(row_counts)), "maximum": max(row_counts)
        },
    }


def natural_metrics(
    values: dict[str, Any], spatial: np.ndarray, folds: list[int],
    min_site_size: int, top_k: int, weight: float,
) -> dict[str, Any]:
    rows = np.isin(values["folds"], folds)
    candidate = score_candidate(
        spatial[rows], values["groups"][rows], min_site_size, top_k, weight
    )
    current = metric_summary(values["labels"][rows], values["current"][rows], values["sensors"][rows])
    new = metric_summary(values["labels"][rows], candidate, values["sensors"][rows])
    return {"current": current, "candidate": new, "versus_current": comparison(new, current)}


def checks(
    simulation: dict[str, Any], natural: dict[str, Any], gates: dict[str, Any]
) -> dict[str, bool]:
    delta = natural["versus_current"]["delta"]
    return {
        "simulated_ap_lower_positive": simulation["combined_ap_delta"]["lower"] > 0.0,
        "simulated_ap_median_positive": simulation["combined_ap_delta"]["median"] > 0.0,
        "simulated_recall_median_no_lower": simulation["combined_recall_delta"]["median"] >= 0.0,
        "each_fold_simulated_ap_median_positive": min(
            value["ap_delta"]["median"] for value in simulation["per_fold"].values()
        ) > 0.0,
        "each_fold_simulated_recall_lower_within_tolerance": min(
            value["recall_delta"]["lower"] for value in simulation["per_fold"].values()
        ) >= -float(gates["simulated_fold_recall_lower_tolerance"]),
        "natural_ap_within_tolerance": delta["average_precision"] >= -float(gates["natural_ap_noninferiority"]),
        "natural_recall_within_tolerance": delta["recall_at_fpr_0_0713"] >= -float(gates["natural_recall_noninferiority"]),
    }


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    selected = report["selection"]["selected"]
    confirmation = report["confirmation"]["simulation"]
    lines = [
        "# Target-mixture temporal-spatial transport experiment",
        "",
        f"- Selected minimum history: **{selected['min_site_size']}**; top-k: **{selected['top_k']}**; weight: **{selected['weight']:.2f}**.",
        f"- Confirmation simulated AP delta: median **{confirmation['combined_ap_delta']['median']:+.5f}**, 95% interval **[{confirmation['combined_ap_delta']['lower']:+.5f}, {confirmation['combined_ap_delta']['upper']:+.5f}]**.",
        f"- Confirmation simulated recall delta median: **{confirmation['combined_recall_delta']['median']:+.5f}**.",
        f"- All promotion gates pass: **{str(report['all_promotion_gates_pass']).lower()}**.",
        "",
        "Each simulated fold samples 4 positive and 46 negative sites and uniformly caps site histories at 23 scenes.",
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
        raise ValueError("Target-mixture trainer hash mismatch")
    paths: dict[str, Path] = {}
    for name, contract in protocol["inputs"].items():
        path = (ROOT / contract["path"]).resolve()
        if sha256(path) != contract["sha256"]:
            raise ValueError(f"Frozen target-mixture input hash mismatch: {name}")
        paths[name] = path
    values = load_development(
        {"inner": paths["inner"], "fold0": paths["fold0"], "fold1": paths["fold1"]}, paths["scores"]
    )
    spatial = align_spatial_scores(values, paths["spatial_scores"])
    selection_folds = list(map(int, protocol["folds"]["selection"]))
    confirmation_folds = list(map(int, protocol["folds"]["confirmation"]))
    mixture = protocol["target_mixture"]
    selection_plans = build_cohort_plans(
        values["labels"], values["groups"], values["folds"], selection_folds,
        replicates=int(protocol["simulation"]["selection_replicates"]),
        sites_per_fold=int(mixture["sites_per_fold"]), positive_sites_per_fold=int(mixture["positive_sites_per_fold"]),
        maximum_scenes_per_site=int(mixture["maximum_scenes_per_site"]), seed=int(protocol["simulation"]["selection_seed"]),
    )
    candidates: list[dict[str, Any]] = []
    for min_size, top_k, weight in itertools.product(
        protocol["search"]["min_site_size"], protocol["search"]["top_k"], protocol["search"]["weights"]
    ):
        simulation = simulate(values, spatial, selection_plans, selection_folds, int(min_size), int(top_k), float(weight))
        natural = natural_metrics(values, spatial, selection_folds, int(min_size), int(top_k), float(weight))
        candidate = {
            "key": f"min{min_size}_top{top_k}_weight{weight}", "min_site_size": int(min_size),
            "top_k": int(top_k), "weight": float(weight), "simulation": simulation, "natural": natural,
            "rank": [
                simulation["combined_ap_delta"]["lower"], simulation["combined_ap_delta"]["median"],
                min(v["ap_delta"]["median"] for v in simulation["per_fold"].values()),
                simulation["combined_recall_delta"]["median"], -float(weight),
            ],
        }
        candidates.append(candidate)
        print(json.dumps({"completed_candidate": candidate["key"]}), flush=True)
    selected = max(candidates, key=lambda value: tuple(value["rank"]))
    selection_checks = checks(selected["simulation"], selected["natural"], protocol["gates"])
    confirmation_plans = build_cohort_plans(
        values["labels"], values["groups"], values["folds"], confirmation_folds,
        replicates=int(protocol["simulation"]["confirmation_replicates"]),
        sites_per_fold=int(mixture["sites_per_fold"]), positive_sites_per_fold=int(mixture["positive_sites_per_fold"]),
        maximum_scenes_per_site=int(mixture["maximum_scenes_per_site"]), seed=int(protocol["simulation"]["confirmation_seed"]),
    )
    confirmation_simulation = simulate(
        values, spatial, confirmation_plans, confirmation_folds,
        selected["min_site_size"], selected["top_k"], selected["weight"],
    )
    confirmation_natural = natural_metrics(
        values, spatial, confirmation_folds,
        selected["min_site_size"], selected["top_k"], selected["weight"],
    )
    confirmation_checks = checks(confirmation_simulation, confirmation_natural, protocol["gates"])
    passed = all(selection_checks.values()) and all(confirmation_checks.values())
    artifact_record = None
    if passed:
        artifact_path = (ROOT / protocol["outputs"]["artifact"]).resolve()
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = artifact_path.with_suffix(artifact_path.suffix + ".tmp")
        joblib.dump({
            "schema_version": 1, "kind": "mars_target_mixture_temporal_spatial_ensemble",
            "spatial_artifact_path": protocol["inputs"]["spatial_artifact"]["path"],
            "spatial_artifact_sha256": protocol["inputs"]["spatial_artifact"]["sha256"],
            "known_training_sites": sorted(set(values["groups"].tolist())),
            "min_site_size": selected["min_site_size"], "top_k": selected["top_k"], "weight": selected["weight"],
            "operational_scene_threshold": float(protocol["base_architecture"]["operational_scene_threshold"]),
            "protocol_sha256": sha256(protocol_path),
        }, temporary, compress=3)
        os.replace(temporary, artifact_path)
        artifact_record = {"path": protocol["outputs"]["artifact"], "bytes": artifact_path.stat().st_size, "sha256": sha256(artifact_path), "tracked": False}
    report = {
        "schema_version": 1,
        "scope": "development-only target-mixture transport simulation; exact paper cache not loaded",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "target_mixture": mixture,
        "selection": {"candidates": candidates, "selected": selected, "checks": selection_checks},
        "confirmation": {"simulation": confirmation_simulation, "natural": confirmation_natural, "checks": confirmation_checks},
        "all_promotion_gates_pass": passed, "artifact": artifact_record,
        "decision": "Freeze target-mixture ensemble for fresh safety evaluation." if passed else "Reject target-mixture ensemble before fresh or paper evaluation.",
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
