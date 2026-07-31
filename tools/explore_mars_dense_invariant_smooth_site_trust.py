#!/usr/bin/env python3
"""Apply smooth label-free site trust to the fixed invariant residual pair."""

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
from evaluate_mars_dense_invariant_pair_fold2 import fixed_scores  # noqa: E402
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
    "configs/mars_dense_invariant_smooth_site_trust_protocol.json"
)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def trust_value(family: str, rate: float, scale: float) -> float:
    if scale <= 0.0:
        raise ValueError("Smooth site-trust scale must be positive")
    if family == "linear":
        return float(np.clip(1.0 - rate / scale, 0.0, 1.0))
    if family == "exponential":
        return float(np.exp(-rate / scale))
    if family == "inverse":
        return float(1.0 / (1.0 + rate / scale))
    raise ValueError(f"Unknown smooth site-trust family: {family}")


def site_trust(
    base_scores: np.ndarray,
    groups: np.ndarray,
    *,
    scene_threshold: float,
    family: str,
    scale: float,
) -> tuple[np.ndarray, dict[str, dict[str, float | int]]]:
    groups = groups.astype(str)
    weights = np.empty(base_scores.size, dtype=np.float64)
    statistics: dict[str, dict[str, float | int]] = {}
    for group in np.unique(groups):
        rows = groups == group
        rate = float(np.mean(base_scores[rows] >= scene_threshold))
        weight = trust_value(family, rate, scale)
        weights[rows] = weight
        statistics[str(group)] = {
            "size": int(np.count_nonzero(rows)),
            "predicted_positive_rate": rate,
            "trust_weight": weight,
        }
    if not np.isfinite(weights).all() or np.any((weights < 0) | (weights > 1)):
        raise ValueError("Smooth site trust produced an invalid weight")
    return weights, statistics


def trust_summary(
    weights: np.ndarray,
    low_rows: np.ndarray,
    groups: np.ndarray,
) -> dict[str, Any]:
    groups = groups.astype(str)
    site_weights = np.asarray(
        [float(weights[groups == group][0]) for group in np.unique(groups)]
    )
    low_groups = np.unique(groups[low_rows])
    low_site_weights = np.asarray(
        [float(weights[groups == group][0]) for group in low_groups]
    )
    return {
        "whole_row_mean": float(np.mean(weights)),
        "whole_row_median": float(np.median(weights)),
        "whole_rows_at_least_half": float(np.mean(weights >= 0.5)),
        "whole_site_mean": float(np.mean(site_weights)),
        "low_prevalence_row_mean": float(np.mean(weights[low_rows])),
        "low_prevalence_row_median": float(np.median(weights[low_rows])),
        "low_prevalence_rows_at_least_half": float(
            np.mean(weights[low_rows] >= 0.5)
        ),
        "low_prevalence_site_mean": float(np.mean(low_site_weights)),
    }


def compact_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "key": candidate["key"],
        "family": candidate["family"],
        "scale": candidate["scale"],
        "trust": candidate["trust"],
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
        raise ValueError("Frozen smooth-site-trust explorer hash mismatch")
    for dependency in protocol["code_dependencies"]:
        path = (ROOT / dependency["path"]).resolve()
        if sha256(path) != dependency["sha256"]:
            raise ValueError(f"Frozen dependency mismatch: {dependency['path']}")
    paths: dict[str, Path] = {}
    for name, contract in protocol["inputs"].items():
        path = (ROOT / contract["path"]).resolve()
        if sha256(path) != contract["sha256"]:
            raise ValueError(f"Frozen smooth-trust input mismatch: {name}")
        paths[name] = path
    adjudication = json.loads(
        paths["hard_router_adjudication"].read_text(encoding="utf-8")
    )
    if bool(adjudication["promotion_authorized"]):
        raise ValueError("Hard-router adjudication unexpectedly authorized promotion")

    matrix, metadata, _columns = load_cache(
        paths["features"],
        paths["metadata"],
    )
    labels = metadata["labels"].astype(np.uint8)
    sensors = metadata["sensors"].astype(np.uint8)
    groups = metadata["groups"].astype(str)
    folds = metadata["folds"].astype(np.uint8)
    base_scores = metadata["exact_base_scores"].astype(np.float64)
    if sorted(map(int, np.unique(folds))) != list(map(int, protocol["folds"])):
        raise ValueError("Smooth-site-trust cache contains unexpected folds")
    pair_scores = fixed_scores(matrix, metadata, protocol)
    pair_residual = logit(pair_scores) - logit(base_scores)
    if not np.isfinite(pair_residual).all():
        raise ValueError("Fixed pair residual is non-finite")
    maximum_rate = float(
        protocol["target_domain"]["maximum_site_positive_rate"]
    )
    low_rows = low_prevalence_mask(labels, groups, maximum_rate)
    all_rows = np.ones(labels.size, dtype=bool)
    specs = [
        {"family": str(row["family"]), "scale": float(scale)}
        for row in protocol["families"]
        for scale in row["scales"]
    ]
    if args.smoke:
        specs = specs[:2]
    candidates: list[dict[str, Any]] = []
    scores_by_key: dict[str, np.ndarray] = {}
    scene_threshold = float(protocol["routing"]["scene_threshold"])
    trust_gates = protocol["trust_gates"]
    for spec in specs:
        weights, _statistics = site_trust(
            base_scores,
            groups,
            scene_threshold=scene_threshold,
            family=str(spec["family"]),
            scale=float(spec["scale"]),
        )
        scores = sigmoid(logit(base_scores) + weights * pair_residual)
        key = f"{spec['family']}_scale{float(spec['scale']):g}"
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
        local_trust = trust_summary(weights, low_rows, groups)
        checks = point_gates(whole, low, protocol)
        checks.update(
            {
                "minimum_whole_mean_trust": bool(
                    local_trust["whole_row_mean"]
                    >= float(trust_gates["minimum_whole_row_mean"])
                ),
                "minimum_low_prevalence_mean_trust": bool(
                    local_trust["low_prevalence_row_mean"]
                    >= float(
                        trust_gates["minimum_low_prevalence_row_mean"]
                    )
                ),
                "minimum_low_prevalence_half_trust_coverage": bool(
                    local_trust["low_prevalence_rows_at_least_half"]
                    >= float(
                        trust_gates[
                            "minimum_low_prevalence_rows_at_least_half"
                        ]
                    )
                ),
            }
        )
        candidates.append(
            {
                "key": key,
                "family": str(spec["family"]),
                "scale": float(spec["scale"]),
                "trust": local_trust,
                "whole": whole,
                "low_prevalence": low,
                "point_checks": checks,
                "passed_point_gates": all(checks.values()),
                "rank": [
                    int(all(checks.values())),
                    min(low["average_precision_delta"]["folds"].values()),
                    min(low["average_precision_delta"]["sensors"].values()),
                    low["average_precision_delta"]["pooled"],
                    min(whole["average_precision_delta"]["folds"].values()),
                    min(whole["average_precision_delta"]["sensors"].values()),
                    whole["average_precision_delta"]["pooled"],
                    local_trust["low_prevalence_row_mean"],
                    -local_trust["whole_row_mean"],
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
        max(
            passed,
            key=lambda row: (
                row["paired_site_ap_delta"]["low_prevalence"]["lower"],
                row["paired_site_ap_delta"]["whole"]["lower"],
                *row["rank"][1:],
            ),
        )
        if passed
        else None
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
            "post-fold2-failure smooth label-free site trust on the fixed "
            "invariant pair using folds 3+4 only"
        ),
        "status": (
            "candidate_found_for_independent_folds01_confirmation"
            if selected is not None
            else "rejected_before_folds01"
        ),
        "decision": (
            "Freeze the selected smooth-trust rule for folds-0/1 confirmation."
            if selected is not None
            else (
                "Retire smooth site-trust routing before folds 0/1 or "
                "external access."
            )
        ),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "provenance": {
            "git_commit": commit,
            "protocol_sha256": sha256(protocol_path),
            "explorer_sha256": sha256(Path(__file__).resolve()),
            "features_sha256": sha256(paths["features"]),
            "metadata_sha256": sha256(paths["metadata"]),
            "hard_router_adjudication_sha256": sha256(
                paths["hard_router_adjudication"]
            ),
            "fold2_report_sha256": sha256(paths["fold2_report"]),
            "numpy": np.__version__,
        },
        "routing": protocol["routing"],
        "target_domain": {
            "definition": protocol["target_domain"],
            "rows": int(np.count_nonzero(low_rows)),
            "positive": int(labels[low_rows].sum()),
            "sites": int(len(np.unique(groups[low_rows]))),
        },
        "screening_candidates": [
            compact_candidate(row) for row in candidates
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
        "fold2_post_failure_diagnostic_referenced": True,
        "fold2_cache_or_labels_reloaded": False,
        "folds01_accessed": False,
        "fresh_inputs_accessed": False,
        "exact_paper_inputs_accessed": False,
    }
    write_json((ROOT / protocol["output"]).resolve(), report)
    print(json.dumps({"ok": selected is not None, **summary}, indent=2))
    return 0 if selected is not None else 2


if __name__ == "__main__":
    raise SystemExit(main())
