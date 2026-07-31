#!/usr/bin/env python3
"""Explore label-free nonlinear gates over direction-stable dense residuals."""

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
    crossfit_residual,
    direction_effects,
    rank_normalize_by_domain,
)
from explore_mars_dense_sparse_invariant_ensemble import (  # noqa: E402
    evaluate_views,
    point_gates,
)
from train_mars_dense_ap_residual_ranker import (  # noqa: E402
    load_cache,
    logit,
    sigmoid,
)
from train_mars_oof_scene_ensemble_v2 import ap_group_bootstrap  # noqa: E402
from train_mars_unseen_low_prevalence_router import (  # noqa: E402
    low_prevalence_mask,
)


DEFAULT_PROTOCOL = Path(
    "configs/mars_dense_invariant_nonlinear_residual_protocol.json"
)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def standardize_by_domain(
    values: np.ndarray,
    folds: np.ndarray,
    sensors: np.ndarray,
) -> np.ndarray:
    output = np.empty(values.size, dtype=np.float64)
    for fold in np.unique(folds):
        for sensor in np.unique(sensors):
            rows = (folds == fold) & (sensors == sensor)
            local = np.asarray(values[rows], dtype=np.float64)
            center = float(np.mean(local))
            scale = float(np.std(local))
            if not np.isfinite(center) or not np.isfinite(scale) or scale <= 0.0:
                raise ValueError(
                    f"Nonlinear residual is degenerate for fold={fold}, sensor={sensor}"
                )
            output[rows] = (local - center) / scale
    if not np.isfinite(output).all():
        raise ValueError("Nonlinear domain standardization produced non-finite values")
    return output


def transform_residual(
    name: str,
    residual: np.ndarray,
    base_scores: np.ndarray,
    folds: np.ndarray,
    sensors: np.ndarray,
    winsor_limit: float,
) -> np.ndarray:
    if name == "symmetric":
        return np.asarray(residual, dtype=np.float64)
    if name == "positive_only":
        values = np.maximum(residual, 0.0)
    elif name == "negative_only":
        values = np.minimum(residual, 0.0)
    elif name == "uncertainty_weighted":
        values = residual * (4.0 * base_scores * (1.0 - base_scores))
    elif name == "low_score_weighted":
        values = residual * (1.0 - base_scores)
    elif name == "high_score_weighted":
        values = residual * base_scores
    elif name == "tanh":
        values = np.tanh(residual)
    elif name == "winsorized":
        values = np.clip(residual, -winsor_limit, winsor_limit)
    else:
        raise ValueError(f"Unknown nonlinear residual transform: {name}")
    return standardize_by_domain(values, folds, sensors)


def compact_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "key": candidate["key"],
        "feature_index": candidate["feature_index"],
        "feature_name": candidate["feature_name"],
        "transform": candidate["transform"],
        "strength": candidate["strength"],
        "whole_average_precision_delta": candidate["whole"][
            "average_precision_delta"
        ],
        "whole_matched_fpr_recall_delta": candidate["whole"][
            "matched_fpr_recall_delta"
        ],
        "low_prevalence_average_precision_delta": candidate[
            "low_prevalence"
        ]["average_precision_delta"],
        "low_prevalence_matched_fpr_recall_delta": candidate[
            "low_prevalence"
        ]["matched_fpr_recall_delta"],
        "point_checks": candidate["point_checks"],
        "passed_point_gates": candidate["passed_point_gates"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL.as_posix())
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    protocol_path = (ROOT / args.protocol).resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if sha256(Path(__file__).resolve()) != protocol["explorer"]["sha256"]:
        raise ValueError("Frozen invariant-nonlinear explorer hash mismatch")
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
        raise ValueError("Invariant-nonlinear cache contains unexpected folds")
    authorized = set(map(int, columns))
    feature_indices = [int(value) for value in protocol["feature_indices"]]
    if args.smoke:
        feature_indices = feature_indices[:1]
    for feature_index in feature_indices:
        if feature_index not in authorized:
            raise ValueError(f"Frozen feature {feature_index} is not authorized")
        expected = str(protocol["feature_names"][str(feature_index)])
        if str(names[feature_index]) != expected:
            raise ValueError(f"Frozen feature name mismatch for {feature_index}")

    transforms = [str(value) for value in protocol["transforms"]]
    strengths = [float(value) for value in protocol["strengths"]]
    if args.smoke:
        transforms = transforms[:2]
        strengths = strengths[:2]
    maximum_rate = float(
        protocol["target_domain"]["maximum_site_positive_rate"]
    )
    low_rows = low_prevalence_mask(labels, groups, maximum_rate)
    all_rows = np.ones(labels.size, dtype=bool)
    base_logit = logit(base_scores)
    winsor_limit = float(protocol["transform_parameters"]["winsor_limit"])
    effects_by_feature: dict[int, dict[int, float]] = {}
    candidates: list[dict[str, Any]] = []
    scores_by_key: dict[str, np.ndarray] = {}

    for feature_index in feature_indices:
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
        residual = crossfit_residual(normalized, effects, folds)
        for transform in transforms:
            transformed = transform_residual(
                transform,
                residual,
                base_scores,
                folds,
                sensors,
                winsor_limit,
            )
            for strength in strengths:
                scores = sigmoid(base_logit + strength * transformed)
                key = (
                    f"feature{feature_index}_{transform}_"
                    f"strength{strength:g}"
                )
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
                        "feature_index": feature_index,
                        "feature_name": str(names[feature_index]),
                        "transform": transform,
                        "strength": strength,
                        "whole": whole,
                        "low_prevalence": low,
                        "point_checks": checks,
                        "passed_point_gates": all(checks.values()),
                        "rank": [
                            int(all(checks.values())),
                            min(
                                low["average_precision_delta"][
                                    "folds"
                                ].values()
                            ),
                            min(
                                low["average_precision_delta"][
                                    "sensors"
                                ].values()
                            ),
                            low["average_precision_delta"]["pooled"],
                            min(
                                whole["average_precision_delta"][
                                    "folds"
                                ].values()
                            ),
                            min(
                                whole["average_precision_delta"][
                                    "sensors"
                                ].values()
                            ),
                            whole["average_precision_delta"]["pooled"],
                            -strength,
                        ],
                    }
                )

    candidates.sort(key=lambda row: tuple(row["rank"]), reverse=True)
    point_survivors = [
        row for row in candidates if row["passed_point_gates"]
    ]
    bootstrap_pool = point_survivors
    if args.smoke and not bootstrap_pool:
        bootstrap_pool = candidates
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
        "features": len(feature_indices),
        "transforms": len(transforms),
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
            "exploratory label-free nonlinear gates over direction-stable "
            "honest dense-Prithvi development residuals"
        ),
        "status": (
            "candidate_found_for_preregistered_fold2_test"
            if selected is not None
            else "rejected_before_fold2"
        ),
        "decision": (
            "Freeze the selected nonlinear invariant residual for fold 2."
            if selected is not None
            else (
                "Retire invariant nonlinear residual gates before fold 2 "
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
            "linear_ranker_report_sha256": sha256(
                paths["linear_ranker_report"]
            ),
            "numpy": np.__version__,
        },
        "target_domain": {
            "definition": protocol["target_domain"],
            "rows": int(np.count_nonzero(low_rows)),
            "positive": int(labels[low_rows].sum()),
            "sites": int(len(np.unique(groups[low_rows]))),
        },
        "feature_direction_effects": {
            str(index): {
                str(fold): value
                for fold, value in effects_by_feature[index].items()
            }
            for index in feature_indices
        },
        "screening_candidates": [
            compact_candidate(row)
            for row in candidates[
                : int(protocol["reporting"]["top_screening_candidates"])
            ]
        ],
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
