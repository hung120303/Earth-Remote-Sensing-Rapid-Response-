#!/usr/bin/env python3
"""Audit invariant univariate signals in honest dense-Prithvi features."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import ndtri
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "EarthRemoteSensingRapidResponse"
for path in (MODEL_ROOT, ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from acquire_mars_metadata import sha256  # noqa: E402
from train_mars_dense_ap_residual_ranker import (  # noqa: E402
    load_cache,
    logit,
    sigmoid,
)
from train_mars_oof_scene_ensemble_v2 import ap_group_bootstrap  # noqa: E402
from train_mars_scene_ranker import comparison, metric_summary  # noqa: E402


DEFAULT_PROTOCOL = Path(
    "configs/mars_dense_invariant_univariate_exploration_protocol.json"
)


def rank_normalize_by_domain(
    values: np.ndarray,
    folds: np.ndarray,
    sensors: np.ndarray,
) -> np.ndarray:
    output = np.empty(values.size, dtype=np.float64)
    for fold in np.unique(folds):
        for sensor in np.unique(sensors):
            rows = np.flatnonzero((folds == fold) & (sensors == sensor))
            local = np.asarray(values[rows], dtype=np.float64).copy()
            missing = ~np.isfinite(local) | (np.abs(local) >= 1000)
            valid = local[~missing]
            if not valid.size:
                raise ValueError(
                    f"Feature has no finite values for fold={fold}, sensor={sensor}"
                )
            local[missing] = float(np.median(valid))
            probability = rankdata(local, method="average") / (local.size + 1.0)
            output[rows] = ndtri(np.clip(probability, 1e-4, 1 - 1e-4))
    if not np.isfinite(output).all():
        raise ValueError("Domain-rank normalization produced non-finite values")
    return output


def direction_effects(
    values: np.ndarray,
    labels: np.ndarray,
    folds: np.ndarray,
) -> dict[int, float]:
    return {
        int(fold): float(
            values[(folds == fold) & (labels == 1)].mean()
            - values[(folds == fold) & (labels == 0)].mean()
        )
        for fold in np.unique(folds)
    }


def crossfit_residual(
    values: np.ndarray,
    effects: dict[int, float],
    folds: np.ndarray,
) -> np.ndarray:
    output = np.empty(values.size, dtype=np.float64)
    unique_folds = sorted(effects)
    if len(unique_folds) != 2:
        raise ValueError("Univariate exploration requires exactly two fit folds")
    for held in unique_folds:
        fit = unique_folds[1] if held == unique_folds[0] else unique_folds[0]
        direction = float(np.sign(effects[fit]))
        if direction == 0.0:
            raise ValueError("Univariate fit-fold direction is exactly zero")
        output[folds == held] = direction * values[folds == held]
    return output


def ap_views(
    labels: np.ndarray,
    scores: np.ndarray,
    folds: np.ndarray,
    sensors: np.ndarray,
) -> dict[str, Any]:
    return {
        "pooled": float(average_precision_score(labels, scores)),
        "folds": {
            str(int(fold)): float(
                average_precision_score(labels[folds == fold], scores[folds == fold])
            )
            for fold in np.unique(folds)
        },
        "sensors": {
            str(int(sensor)): float(
                average_precision_score(
                    labels[sensors == sensor], scores[sensors == sensor]
                )
            )
            for sensor in np.unique(sensors)
        },
    }


def deltas(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    return {
        "pooled": candidate["pooled"] - baseline["pooled"],
        "folds": {
            key: candidate["folds"][key] - baseline["folds"][key]
            for key in baseline["folds"]
        },
        "sensors": {
            key: candidate["sensors"][key] - baseline["sensors"][key]
            for key in baseline["sensors"]
        },
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL.as_posix())
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    protocol_path = (ROOT / args.protocol).resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if sha256(Path(__file__).resolve()) != protocol["explorer"]["sha256"]:
        raise ValueError("Frozen invariant-univariate explorer hash mismatch")
    for dependency in protocol["code_dependencies"]:
        path = (ROOT / dependency["path"]).resolve()
        if sha256(path) != dependency["sha256"]:
            raise ValueError(f"Frozen dependency mismatch: {dependency['path']}")
    paths: dict[str, Path] = {}
    for name, contract in protocol["inputs"].items():
        path = (ROOT / contract["path"]).resolve()
        if sha256(path) != contract["sha256"]:
            raise ValueError(f"Frozen input mismatch: {name}")
        paths[name] = path
    matrix, metadata, _columns = load_cache(paths["features"], paths["metadata"])
    labels = metadata["labels"].astype(np.uint8)
    sensors = metadata["sensors"].astype(np.uint8)
    groups = metadata["groups"].astype(str)
    folds = metadata["folds"].astype(np.uint8)
    base_scores = metadata["exact_base_scores"].astype(np.float64)
    expected_folds = sorted(map(int, protocol["folds"]))
    if sorted(map(int, np.unique(folds))) != expected_folds:
        raise ValueError("Invariant-univariate cache contains unexpected folds")
    if not np.isfinite(base_scores).all():
        raise ValueError("Exact current-score floor is non-finite")
    names = metadata["feature_names"].astype(str)
    feature_indices = list(
        range(
            int(protocol["feature_range"]["start_inclusive"]),
            int(protocol["feature_range"]["end_exclusive"]),
        )
    )
    if args.smoke:
        feature_indices = feature_indices[:8]
    strengths = [float(value) for value in protocol["strengths"]]
    if args.smoke:
        strengths = strengths[:2]
    baseline_ap = ap_views(labels, base_scores, folds, sensors)
    baseline_metrics = metric_summary(labels, base_scores, sensors)
    base_logit = logit(base_scores)
    minimum_ap = float(protocol["gates"]["minimum_pooled_ap_delta"])
    candidates = []
    signs_agree = 0
    for feature_index in feature_indices:
        normalized = rank_normalize_by_domain(
            matrix[:, feature_index],
            folds,
            sensors,
        )
        effects = direction_effects(normalized, labels, folds)
        if np.prod(list(effects.values())) <= 0.0:
            continue
        signs_agree += 1
        residual = crossfit_residual(normalized, effects, folds)
        for strength in strengths:
            scores = sigmoid(base_logit + strength * residual)
            local_ap = ap_views(labels, scores, folds, sensors)
            local_delta = deltas(local_ap, baseline_ap)
            if (
                local_delta["pooled"] < minimum_ap
                or min(local_delta["folds"].values()) < 0.0
                or min(local_delta["sensors"].values()) < 0.0
            ):
                continue
            metrics = metric_summary(labels, scores, sensors)
            versus = comparison(metrics, baseline_metrics)
            recall_delta = float(
                versus["delta"]["recall_at_fpr_0_0713"]
            )
            if recall_delta < 0.0:
                continue
            candidates.append(
                {
                    "feature_index": feature_index,
                    "feature_name": str(names[feature_index]),
                    "effects": {str(key): value for key, value in effects.items()},
                    "strength": strength,
                    "average_precision": local_ap,
                    "average_precision_delta": local_delta,
                    "matched_fpr_recall_delta": recall_delta,
                    "rank": [
                        min(local_delta["folds"].values()),
                        min(local_delta["sensors"].values()),
                        local_delta["pooled"],
                        recall_delta,
                        -strength,
                    ],
                }
            )
    candidates.sort(key=lambda row: tuple(row["rank"]), reverse=True)
    bootstrap_top = min(
        len(candidates),
        2 if args.smoke else int(protocol["bootstrap"]["top_k"]),
    )
    evaluated = []
    for rank_index, candidate in enumerate(candidates[:bootstrap_top]):
        normalized = rank_normalize_by_domain(
            matrix[:, int(candidate["feature_index"])],
            folds,
            sensors,
        )
        effects = {
            int(key): float(value)
            for key, value in candidate["effects"].items()
        }
        residual = crossfit_residual(normalized, effects, folds)
        scores = sigmoid(
            base_logit + float(candidate["strength"]) * residual
        )
        interval = ap_group_bootstrap(
            labels,
            base_scores,
            scores,
            groups,
            replicates=(
                100 if args.smoke else int(protocol["bootstrap"]["replicates"])
            ),
            seed=int(protocol["bootstrap"]["seed"]) + rank_index,
        )
        passed = bool(interval["lower"] > 0.0)
        evaluated.append(
            {
                **candidate,
                "paired_site_ap_delta": interval,
                "passed": passed,
            }
        )
    passed = [row for row in evaluated if row["passed"]]
    selected = (
        max(passed, key=lambda row: tuple(row["rank"])) if passed else None
    )
    summary = {
        "features_considered": len(feature_indices),
        "direction_agreement_features": signs_agree,
        "point_gate_candidates": len(candidates),
        "bootstrapped_candidates": len(evaluated),
        "passing_candidates": len(passed),
        "selected": (
            None
            if selected is None
            else {
                key: value
                for key, value in selected.items()
                if key != "rank"
            }
        ),
    }
    if args.smoke:
        print(json.dumps({"ok": True, **summary}, indent=2))
        return 0
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    report = {
        "schema_version": 1,
        "scope": "exploratory invariant-univariate audit on honest dense-Prithvi development features",
        "status": (
            "candidate_found_for_preregistered_fold2_test"
            if selected is not None
            else "rejected_before_fold2"
        ),
        "decision": (
            "Freeze the selected sparse signal in a separate protocol before fold 2."
            if selected is not None
            else "Retire invariant univariate dense signals before fold 2 or external access."
        ),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "provenance": {
            "git_commit": commit,
            "protocol_sha256": sha256(protocol_path),
            "explorer_sha256": sha256(Path(__file__).resolve()),
            "features_sha256": sha256(paths["features"]),
            "metadata_sha256": sha256(paths["metadata"]),
            "numpy": np.__version__,
        },
        "normalization": protocol["normalization"],
        "baseline": {
            "average_precision": baseline_ap,
            "metrics": baseline_metrics,
        },
        "summary": summary,
        "evaluated_candidates": [
            {
                key: value
                for key, value in row.items()
                if key != "rank"
            }
            for row in evaluated
        ],
        "fold2_accessed": False,
        "fresh_inputs_accessed": False,
        "exact_paper_inputs_accessed": False,
    }
    write_json((ROOT / protocol["output"]).resolve(), report)
    print(json.dumps({"ok": selected is not None, **summary}, indent=2))
    return 0 if selected is not None else 2


if __name__ == "__main__":
    raise SystemExit(main())
