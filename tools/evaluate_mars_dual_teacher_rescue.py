#!/usr/bin/env python3
"""Evaluate a bounded released/primary dense-evidence rescue of the MARS champion."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "tools", ROOT / "EarthRemoteSensingRapidResponse"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from acquire_mars_metadata import sha256  # noqa: E402
from analyze_mars_recall_anchor import align_feature_rows  # noqa: E402
from train_mars_oof_scene_ensemble_v2 import ap_group_bootstrap  # noqa: E402
from train_mars_scene_ranker import comparison, metric_summary  # noqa: E402


DEFAULT_PROTOCOL = Path("configs/mars_dual_teacher_rescue_protocol.json")


def anchor_score(
    name: str,
    primary: np.ndarray,
    released: np.ndarray,
    primary_top_200: np.ndarray,
) -> np.ndarray:
    if name == "geometric_dense":
        return np.sqrt(np.clip(primary * released, 0.0, 1.0))
    if name == "conservative_dense":
        return np.minimum(primary, released)
    if name == "topology_consensus":
        return np.cbrt(np.clip(primary * released * primary_top_200, 0.0, 1.0))
    raise ValueError(f"Unknown rescue anchor: {name}")


def rescue_scores(
    champion: np.ndarray,
    released: np.ndarray,
    anchor: np.ndarray,
    *,
    weight: float,
    released_gate: float = 0.5,
    champion_gate: float = 0.25,
) -> tuple[np.ndarray, np.ndarray]:
    if not 0.0 < weight <= 1.0:
        raise ValueError("Rescue weight must be in (0, 1]")
    if not (champion.shape == released.shape == anchor.shape):
        raise ValueError("Rescue score arrays must align")
    route = (released > released_gate) & (champion < champion_gate)
    candidate = np.asarray(champion, dtype=np.float64).copy()
    candidate[route] = np.maximum(candidate[route], weight * anchor[route])
    if np.any(candidate + 1e-15 < champion):
        raise AssertionError("Recall rescue suppressed a champion score")
    return candidate, route


def summarize_candidate(
    labels: np.ndarray,
    sensors: np.ndarray,
    groups: np.ndarray,
    folds: np.ndarray,
    champion: np.ndarray,
    candidate: np.ndarray,
    *,
    anchor: str,
    weight: float,
    route: np.ndarray,
    gates: dict[str, Any],
) -> dict[str, Any]:
    baseline = metric_summary(labels, champion, sensors)
    candidate_metrics = metric_summary(labels, candidate, sensors)
    pooled = comparison(candidate_metrics, baseline)
    per_fold: dict[str, Any] = {}
    for fold in (3, 4):
        selection = folds == fold
        per_fold[str(fold)] = comparison(
            metric_summary(labels[selection], candidate[selection], sensors[selection]),
            metric_summary(labels[selection], champion[selection], sensors[selection]),
        )
    fold_ap = [value["delta"]["average_precision"] for value in per_fold.values()]
    fold_recall = [
        value["delta"]["recall_at_fpr_0_0713"] for value in per_fold.values()
    ]
    delta = pooled["delta"]
    checks = {
        "minimum_pooled_ap_delta": delta["average_precision"]
        >= float(gates["minimum_pooled_ap_delta"]),
        "strictly_positive_pooled_recall": delta["recall_at_fpr_0_0713"]
        > float(gates["minimum_pooled_matched_fpr_recall_delta"]),
        "each_fold_ap_nonnegative": min(fold_ap)
        >= float(gates["minimum_each_fold_ap_delta"]),
        "each_fold_recall_nonnegative": min(fold_recall)
        >= float(gates["minimum_each_fold_recall_delta"]),
        "each_sensor_ap_nonnegative": min(delta["sensor_average_precision"].values())
        >= float(gates["minimum_each_sensor_ap_delta"]),
    }
    changed = candidate > champion + 1e-15
    return {
        "anchor": anchor,
        "rescue_weight": weight,
        "route_rows": int(route.sum()),
        "raised_rows": int(changed.sum()),
        "raised_positives": int(labels[changed].sum()),
        "raised_negatives": int(changed.sum() - labels[changed].sum()),
        "pooled": pooled,
        "per_fold": per_fold,
        "point_checks": checks,
        "all_point_gates_pass": all(checks.values()),
        "rank": [
            int(all(checks.values())),
            min(fold_recall),
            min(fold_ap),
            delta["average_precision"],
            -weight,
        ],
        "groups": int(np.unique(groups).size),
    }


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    selected = report["selected"]
    delta = selected["pooled"]["delta"]
    interval = selected["paired_group_bootstrap_ap_delta"]
    lines = [
        "# Recall-anchored dual-teacher rescue: folds 3/4",
        "",
        "The candidate can only raise the frozen Gaussian+DOFA score inside the released-detector rescue route.",
        "",
        f"- Anchor: `{selected['anchor']}`",
        f"- Rescue weight: {selected['rescue_weight']:.6f}",
        f"- Raised rows: {selected['raised_rows']} ({selected['raised_positives']} positives, {selected['raised_negatives']} negatives)",
        f"- AP delta: {delta['average_precision']:+.6f}",
        f"- Matched-FPR recall delta: {delta['recall_at_fpr_0_0713']:+.6f}",
        f"- Paired 25 km-group AP interval: [{interval['lower']:+.6f}, {interval['upper']:+.6f}]",
        "",
        "| Fold | AP delta | Recall delta |",
        "|---|---:|---:|",
    ]
    for fold, value in selected["per_fold"].items():
        local = value["delta"]
        lines.append(
            f"| {fold} | {local['average_precision']:+.6f} | "
            f"{local['recall_at_fpr_0_0713']:+.6f} |"
        )
    lines.extend(["", f"Decision: **{report['decision']}**", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL.as_posix())
    args = parser.parse_args()
    protocol_path = (ROOT / args.protocol).resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    inputs = {
        name: (ROOT / value["path"]).resolve()
        for name, value in protocol["inputs"].items()
    }
    for name, value in protocol["inputs"].items():
        if sha256(inputs[name]) != value["sha256"]:
            raise ValueError(f"Frozen input hash mismatch: {name}")
    feasibility = json.loads(inputs["feasibility_report"].read_text(encoding="utf-8"))
    if feasibility.get("decision") != "continue_to_recall_anchored_architecture":
        raise ValueError("Recall-anchor feasibility gate did not authorize this pilot")

    with np.load(inputs["champion_scores"], allow_pickle=False) as bundle:
        sample_ids = bundle["sample_ids"].astype(str)
        labels = bundle["labels"].astype(np.uint8)
        sensors = bundle["sensors"].astype(np.uint8)
        groups = bundle["groups"].astype(str)
        folds = bundle["folds"].astype(np.uint8)
        champion = bundle["champion_scores"].astype(np.float64)
    with np.load(inputs["scene_features"], allow_pickle=False) as bundle:
        feature_names = bundle["feature_names"].astype(str)
        indices = align_feature_rows(sample_ids, bundle["sample_ids"], bundle["folds"])
        features = bundle["features"][indices].astype(np.float64)
    feature_lookup = {name: index for index, name in enumerate(feature_names)}
    required = (
        "primary_connected_score",
        "released_connected_score",
        "primary_top_200_mean",
    )
    if any(name not in feature_lookup for name in required):
        raise ValueError("Scene features lack a required dual-teacher signal")
    primary = features[:, feature_lookup["primary_connected_score"]]
    released = features[:, feature_lookup["released_connected_score"]]
    primary_top_200 = features[:, feature_lookup["primary_top_200_mean"]]

    candidates = []
    candidate_scores: dict[tuple[str, float], np.ndarray] = {}
    for anchor_name in protocol["architecture"]["anchors"]:
        anchor = anchor_score(anchor_name, primary, released, primary_top_200)
        for weight in protocol["architecture"]["rescue_weights"]:
            candidate, route = rescue_scores(
                champion,
                released,
                anchor,
                weight=float(weight),
            )
            candidate_scores[(anchor_name, float(weight))] = candidate
            candidates.append(
                summarize_candidate(
                    labels,
                    sensors,
                    groups,
                    folds,
                    champion,
                    candidate,
                    anchor=anchor_name,
                    weight=float(weight),
                    route=route,
                    gates=protocol["promotion_gates"],
                )
            )
    if len(candidates) != int(protocol["architecture"]["candidate_count"]):
        raise AssertionError("Candidate count differs from the frozen protocol")
    selected = max(candidates, key=lambda value: tuple(value["rank"]))
    selected_scores = candidate_scores[(selected["anchor"], selected["rescue_weight"])]
    selected["paired_group_bootstrap_ap_delta"] = ap_group_bootstrap(
        labels,
        champion,
        selected_scores,
        groups,
        replicates=int(protocol["promotion_gates"]["paired_group_bootstrap_replicates"]),
        seed=int(protocol["promotion_gates"]["paired_group_bootstrap_seed"]),
    )
    passed = bool(
        selected["all_point_gates_pass"]
        and selected["paired_group_bootstrap_ap_delta"]["lower"] > 0.0
    )
    selected["all_promotion_gates_pass"] = passed
    report = {
        "schema_version": 1,
        "scope": protocol["scope"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "path": protocol_path.relative_to(ROOT).as_posix(),
            "sha256": sha256(protocol_path),
        },
        "cohort": {
            "rows": int(labels.size),
            "positives": int(labels.sum()),
            "negatives": int((labels == 0).sum()),
            "groups": int(np.unique(groups).size),
            "folds": sorted(np.unique(folds).astype(int).tolist()),
        },
        "baseline": metric_summary(labels, champion, sensors),
        "candidates": candidates,
        "selected": selected,
        "all_promotion_gates_pass": passed,
        "decision": (
            "promote_dual_teacher_rescue_for_separate_posttest_protocol"
            if passed
            else "reject_deterministic_dual_teacher_rescue"
        ),
        "holdout_boundary": protocol["holdout_boundary"],
    }
    output_json = (ROOT / protocol["outputs"]["json"]).resolve()
    output_markdown = (ROOT / protocol["outputs"]["markdown"]).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_json.with_suffix(".tmp.json")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output_json)
    write_markdown(output_markdown, report)
    print(
        json.dumps(
            {
                "decision": report["decision"],
                "anchor": selected["anchor"],
                "weight": selected["rescue_weight"],
                "ap_delta": selected["pooled"]["delta"]["average_precision"],
                "recall_delta": selected["pooled"]["delta"]["recall_at_fpr_0_0713"],
                "ap_lower": selected["paired_group_bootstrap_ap_delta"]["lower"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
