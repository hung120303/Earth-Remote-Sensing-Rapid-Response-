#!/usr/bin/env python3
"""Test sparse invariant ensembles discovered on honest dense-Prithvi features."""

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

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "EarthRemoteSensingRapidResponse"
for path in (MODEL_ROOT, ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from acquire_mars_metadata import sha256  # noqa: E402
from explore_mars_dense_invariant_univariate import (  # noqa: E402
    ap_views,
    crossfit_residual,
    deltas,
    direction_effects,
    rank_normalize_by_domain,
)
from train_mars_dense_ap_residual_ranker import (  # noqa: E402
    load_cache,
    logit,
    sigmoid,
)
from train_mars_oof_scene_ensemble_v2 import ap_group_bootstrap  # noqa: E402
from train_mars_scene_ranker import comparison, metric_summary  # noqa: E402
from train_mars_unseen_low_prevalence_router import (  # noqa: E402
    low_prevalence_mask,
)


DEFAULT_PROTOCOL = Path(
    "configs/mars_dense_sparse_invariant_ensemble_protocol.json"
)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def evaluate_views(
    labels: np.ndarray,
    baseline_scores: np.ndarray,
    candidate_scores: np.ndarray,
    folds: np.ndarray,
    sensors: np.ndarray,
    rows: np.ndarray,
) -> dict[str, Any]:
    local_labels = labels[rows]
    local_baseline = baseline_scores[rows]
    local_candidate = candidate_scores[rows]
    local_folds = folds[rows]
    local_sensors = sensors[rows]
    baseline_ap = ap_views(
        local_labels,
        local_baseline,
        local_folds,
        local_sensors,
    )
    candidate_ap = ap_views(
        local_labels,
        local_candidate,
        local_folds,
        local_sensors,
    )
    baseline_metrics = metric_summary(
        local_labels,
        local_baseline,
        local_sensors,
    )
    candidate_metrics = metric_summary(
        local_labels,
        local_candidate,
        local_sensors,
    )
    versus = comparison(candidate_metrics, baseline_metrics)
    return {
        "rows": int(np.count_nonzero(rows)),
        "positive": int(local_labels.sum()),
        "average_precision": candidate_ap,
        "average_precision_delta": deltas(candidate_ap, baseline_ap),
        "matched_fpr_metrics": candidate_metrics,
        "matched_fpr_recall_delta": float(
            versus["delta"]["recall_at_fpr_0_0713"]
        ),
    }


def point_gates(
    whole: dict[str, Any],
    low: dict[str, Any],
    protocol: dict[str, Any],
) -> dict[str, bool]:
    gates = protocol["gates"]
    whole_delta = whole["average_precision_delta"]
    low_delta = low["average_precision_delta"]
    return {
        "whole_pooled_ap_delta": bool(
            whole_delta["pooled"]
            >= float(gates["minimum_whole_pooled_ap_delta"])
        ),
        "whole_each_fold_ap_nondecreasing": bool(
            min(whole_delta["folds"].values())
            >= float(gates["minimum_each_whole_fold_ap_delta"])
        ),
        "whole_each_sensor_ap_nondecreasing": bool(
            min(whole_delta["sensors"].values())
            >= float(gates["minimum_each_whole_sensor_ap_delta"])
        ),
        "whole_matched_fpr_recall_nondecreasing": bool(
            whole["matched_fpr_recall_delta"]
            >= float(gates["minimum_whole_matched_fpr_recall_delta"])
        ),
        "low_pooled_ap_delta": bool(
            low_delta["pooled"]
            >= float(gates["minimum_low_prevalence_pooled_ap_delta"])
        ),
        "low_each_fold_ap_nondecreasing": bool(
            min(low_delta["folds"].values())
            >= float(gates["minimum_each_low_prevalence_fold_ap_delta"])
        ),
        "low_each_sensor_ap_nondecreasing": bool(
            min(low_delta["sensors"].values())
            >= float(gates["minimum_each_low_prevalence_sensor_ap_delta"])
        ),
        "low_matched_fpr_recall_nondecreasing": bool(
            low["matched_fpr_recall_delta"]
            >= float(gates["minimum_low_prevalence_matched_fpr_recall_delta"])
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL.as_posix())
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    protocol_path = (ROOT / args.protocol).resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if sha256(Path(__file__).resolve()) != protocol["explorer"]["sha256"]:
        raise ValueError("Frozen sparse-invariant ensemble explorer hash mismatch")
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

    matrix, metadata, columns = load_cache(paths["features"], paths["metadata"])
    labels = metadata["labels"].astype(np.uint8)
    sensors = metadata["sensors"].astype(np.uint8)
    groups = metadata["groups"].astype(str)
    folds = metadata["folds"].astype(np.uint8)
    base_scores = metadata["exact_base_scores"].astype(np.float64)
    names = metadata["feature_names"].astype(str)
    expected_folds = sorted(map(int, protocol["folds"]))
    if sorted(map(int, np.unique(folds))) != expected_folds:
        raise ValueError("Sparse-invariant cache contains unexpected folds")
    if len(names) != matrix.shape[1]:
        raise ValueError("Sparse-invariant cache column metadata mismatch")
    if 1 in set(map(int, columns)) or 0 not in set(map(int, columns)):
        raise ValueError("Sparse-invariant cache did not exclude only the rejected residual")
    if not np.isfinite(base_scores).all():
        raise ValueError("Exact current-score floor is non-finite")

    feature_order = [int(value) for value in protocol["feature_order"]]
    if len(feature_order) != len(set(feature_order)):
        raise ValueError("Sparse-invariant feature order contains duplicates")
    expected_names = protocol["feature_names"]
    for index in feature_order:
        if str(names[index]) != str(expected_names[str(index)]):
            raise ValueError(f"Frozen feature name mismatch for column {index}")

    effects_by_feature: dict[int, dict[int, float]] = {}
    residual_by_feature: dict[int, np.ndarray] = {}
    for feature_index in feature_order:
        if feature_index not in set(map(int, columns)):
            raise ValueError(f"Frozen feature {feature_index} is not authorized")
        normalized = rank_normalize_by_domain(
            matrix[:, feature_index],
            folds,
            sensors,
        )
        effects = direction_effects(normalized, labels, folds)
        if np.prod(list(effects.values())) <= 0.0:
            raise ValueError(
                f"Frozen feature {feature_index} lost cross-fold direction agreement"
            )
        effects_by_feature[feature_index] = effects
        residual_by_feature[feature_index] = crossfit_residual(
            normalized,
            effects,
            folds,
        )

    maximum_rate = float(
        protocol["target_domain"]["maximum_site_positive_rate"]
    )
    low_rows = low_prevalence_mask(labels, groups, maximum_rate)
    if not np.any(low_rows) or len(np.unique(labels[low_rows])) != 2:
        raise ValueError("Low-prevalence target view is empty or single-class")
    for fold in expected_folds:
        local = low_rows & (folds == fold)
        if len(np.unique(labels[local])) != 2:
            raise ValueError(f"Low-prevalence fold {fold} is single-class")
    for sensor in np.unique(sensors):
        local = low_rows & (sensors == sensor)
        if len(np.unique(labels[local])) != 2:
            raise ValueError(f"Low-prevalence sensor {sensor} is single-class")

    top_ks = [int(value) for value in protocol["top_k"]]
    strengths = [float(value) for value in protocol["strengths"]]
    if args.smoke:
        top_ks = top_ks[:2]
        strengths = strengths[:2]
    all_rows = np.ones(labels.size, dtype=bool)
    base_logit = logit(base_scores)
    candidates: list[dict[str, Any]] = []
    scores_by_key: dict[str, np.ndarray] = {}
    for top_k in top_ks:
        selected_features = feature_order[:top_k]
        ensemble_residual = np.mean(
            np.column_stack(
                [residual_by_feature[index] for index in selected_features]
            ),
            axis=1,
        )
        for strength in strengths:
            scores = sigmoid(base_logit + strength * ensemble_residual)
            key = f"top{top_k}_strength{strength:g}"
            scores_by_key[key] = scores
            whole = evaluate_views(
                labels,
                base_scores,
                scores,
                folds,
                sensors,
                all_rows,
            )
            low = evaluate_views(
                labels,
                base_scores,
                scores,
                folds,
                sensors,
                low_rows,
            )
            checks = point_gates(whole, low, protocol)
            candidates.append(
                {
                    "key": key,
                    "top_k": top_k,
                    "feature_indices": selected_features,
                    "feature_names": [
                        str(names[index]) for index in selected_features
                    ],
                    "strength": strength,
                    "whole": whole,
                    "low_prevalence": low,
                    "point_checks": checks,
                    "passed_point_gates": all(checks.values()),
                    "rank": [
                        int(all(checks.values())),
                        min(
                            low["average_precision_delta"]["folds"].values()
                        ),
                        min(
                            low["average_precision_delta"]["sensors"].values()
                        ),
                        low["average_precision_delta"]["pooled"],
                        min(
                            whole["average_precision_delta"]["folds"].values()
                        ),
                        whole["average_precision_delta"]["pooled"],
                        -top_k,
                        -strength,
                    ],
                }
            )

    point_survivors = [
        row for row in candidates if row["passed_point_gates"]
    ]
    point_survivors.sort(
        key=lambda row: tuple(row["rank"]),
        reverse=True,
    )
    bootstrap_pool = point_survivors
    if args.smoke and not bootstrap_pool:
        bootstrap_pool = sorted(
            candidates,
            key=lambda row: tuple(row["rank"]),
            reverse=True,
        )
    bootstrap_top = min(
        len(bootstrap_pool),
        2 if args.smoke else int(protocol["bootstrap"]["top_k"]),
    )
    evaluated: list[dict[str, Any]] = []
    for rank_index, candidate in enumerate(bootstrap_pool[:bootstrap_top]):
        scores = scores_by_key[str(candidate["key"])]
        replicates = (
            100
            if args.smoke
            else int(protocol["bootstrap"]["replicates"])
        )
        whole_interval = ap_group_bootstrap(
            labels,
            base_scores,
            scores,
            groups,
            replicates=replicates,
            seed=int(protocol["bootstrap"]["whole_seed"]) + rank_index,
        )
        low_interval = ap_group_bootstrap(
            labels[low_rows],
            base_scores[low_rows],
            scores[low_rows],
            groups[low_rows],
            replicates=replicates,
            seed=int(protocol["bootstrap"]["low_prevalence_seed"])
            + rank_index,
        )
        bootstrap_checks = {
            "whole_paired_site_ap_lower_strictly_positive": bool(
                whole_interval["lower"] > 0.0
            ),
            "low_prevalence_paired_site_ap_lower_strictly_positive": bool(
                low_interval["lower"] > 0.0
            ),
        }
        evaluated.append(
            {
                **candidate,
                "paired_site_ap_delta": {
                    "whole": whole_interval,
                    "low_prevalence": low_interval,
                },
                "bootstrap_checks": bootstrap_checks,
                "passed": bool(
                    candidate["passed_point_gates"]
                    and all(bootstrap_checks.values())
                ),
            }
        )

    passed = [row for row in evaluated if row["passed"]]
    selected = (
        max(passed, key=lambda row: tuple(row["rank"])) if passed else None
    )
    summary = {
        "candidate_count": len(candidates),
        "point_gate_candidates": len(point_survivors),
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
        "scope": (
            "exploratory sparse invariant ensemble on honest dense-Prithvi "
            "development features"
        ),
        "status": (
            "candidate_found_for_preregistered_fold2_test"
            if selected is not None
            else "rejected_before_fold2"
        ),
        "decision": (
            "Freeze the selected sparse ensemble in a separate fold-2 protocol."
            if selected is not None
            else (
                "Retire this sparse invariant ensemble family before fold 2 "
                "or external access."
            )
        ),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "provenance": {
            "git_commit": commit,
            "protocol_sha256": sha256(protocol_path),
            "explorer_sha256": sha256(Path(__file__).resolve()),
            "features_sha256": sha256(paths["features"]),
            "metadata_sha256": sha256(paths["metadata"]),
            "univariate_report_sha256": sha256(
                paths["univariate_report"]
            ),
            "numpy": np.__version__,
        },
        "target_domain": {
            "definition": protocol["target_domain"],
            "rows": int(np.count_nonzero(low_rows)),
            "positive": int(labels[low_rows].sum()),
            "sites": int(len(np.unique(groups[low_rows]))),
            "folds": {
                str(fold): {
                    "rows": int(np.count_nonzero(low_rows & (folds == fold))),
                    "positive": int(labels[low_rows & (folds == fold)].sum()),
                    "sites": int(
                        len(np.unique(groups[low_rows & (folds == fold)]))
                    ),
                }
                for fold in expected_folds
            },
        },
        "feature_order": feature_order,
        "feature_names": {
            str(index): str(names[index]) for index in feature_order
        },
        "direction_effects": {
            str(index): {
                str(fold): value
                for fold, value in effects_by_feature[index].items()
            }
            for index in feature_order
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
