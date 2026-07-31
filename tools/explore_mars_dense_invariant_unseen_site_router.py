#!/usr/bin/env python3
"""Route the fixed invariant pair using label-free unseen-site history statistics."""

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
from train_mars_dense_ap_residual_ranker import load_cache  # noqa: E402
from train_mars_oof_scene_ensemble_v2 import ap_group_bootstrap  # noqa: E402
from train_mars_unseen_low_prevalence_router import (  # noqa: E402
    low_prevalence_mask,
)


DEFAULT_PROTOCOL = Path(
    "configs/mars_dense_invariant_unseen_site_router_protocol.json"
)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def route_rows(
    base_scores: np.ndarray,
    groups: np.ndarray,
    *,
    scene_threshold: float,
    maximum_site_size: int,
    maximum_predicted_positive_rate: float,
) -> tuple[np.ndarray, dict[str, dict[str, float | int | bool]]]:
    selected = np.zeros(base_scores.size, dtype=bool)
    statistics: dict[str, dict[str, float | int | bool]] = {}
    for group in np.unique(groups.astype(str)):
        rows = groups.astype(str) == group
        size = int(np.count_nonzero(rows))
        predicted_positive_rate = float(
            np.mean(base_scores[rows] >= scene_threshold)
        )
        route = bool(
            size <= maximum_site_size
            and predicted_positive_rate <= maximum_predicted_positive_rate
        )
        selected[rows] = route
        statistics[str(group)] = {
            "size": size,
            "predicted_positive_rate": predicted_positive_rate,
            "routed": route,
        }
    return selected, statistics


def coverage(
    routed: np.ndarray,
    low_rows: np.ndarray,
    groups: np.ndarray,
) -> dict[str, Any]:
    all_sites = np.unique(groups.astype(str))
    low_sites = np.unique(groups[low_rows].astype(str))
    routed_sites = {
        str(group)
        for group in all_sites
        if np.any(routed[groups.astype(str) == str(group)])
    }
    routed_low_sites = routed_sites & set(map(str, low_sites))
    return {
        "whole_rows": int(np.count_nonzero(routed)),
        "whole_row_fraction": float(np.mean(routed)),
        "whole_sites": len(routed_sites),
        "whole_site_fraction": float(len(routed_sites) / len(all_sites)),
        "low_prevalence_rows": int(np.count_nonzero(routed & low_rows)),
        "low_prevalence_row_fraction": float(np.mean(routed[low_rows])),
        "low_prevalence_sites": len(routed_low_sites),
        "low_prevalence_site_fraction": float(
            len(routed_low_sites) / len(low_sites)
        ),
    }


def compact_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "key": candidate["key"],
        "maximum_site_size": candidate["maximum_site_size"],
        "maximum_predicted_positive_rate": candidate[
            "maximum_predicted_positive_rate"
        ],
        "coverage": candidate["coverage"],
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
        raise ValueError("Frozen unseen-site invariant router explorer hash mismatch")
    for dependency in protocol["code_dependencies"]:
        path = (ROOT / dependency["path"]).resolve()
        if sha256(path) != dependency["sha256"]:
            raise ValueError(f"Frozen dependency mismatch: {dependency['path']}")
    paths: dict[str, Path] = {}
    for name, contract in protocol["inputs"].items():
        path = (ROOT / contract["path"]).resolve()
        if sha256(path) != contract["sha256"]:
            raise ValueError(f"Frozen unseen-router input mismatch: {name}")
        paths[name] = path
    fold2_report = json.loads(
        paths["fold2_report"].read_text(encoding="utf-8")
    )
    if str(fold2_report["status"]) != "rejected_on_fold2_confirmation":
        raise ValueError("Unseen-router motivation report has unexpected status")

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
        raise ValueError("Unseen-router cache contains unexpected folds")
    pair_scores = fixed_scores(matrix, metadata, protocol)
    maximum_rate = float(
        protocol["target_domain"]["maximum_site_positive_rate"]
    )
    low_rows = low_prevalence_mask(labels, groups, maximum_rate)
    all_rows = np.ones(labels.size, dtype=bool)
    sizes = [int(value) for value in protocol["search"]["maximum_site_size"]]
    rates = [
        float(value)
        for value in protocol["search"]["maximum_predicted_positive_rate"]
    ]
    if args.smoke:
        sizes = sizes[:2]
        rates = rates[:2]
    candidates: list[dict[str, Any]] = []
    scores_by_key: dict[str, np.ndarray] = {}
    scene_threshold = float(protocol["routing"]["scene_threshold"])
    coverage_gates = protocol["coverage_gates"]
    for maximum_site_size in sizes:
        for maximum_predicted_positive_rate in rates:
            routed, _statistics = route_rows(
                base_scores,
                groups,
                scene_threshold=scene_threshold,
                maximum_site_size=maximum_site_size,
                maximum_predicted_positive_rate=maximum_predicted_positive_rate,
            )
            scores = np.where(routed, pair_scores, base_scores)
            key = (
                f"maxsize{maximum_site_size}_"
                f"maxrate{maximum_predicted_positive_rate:g}"
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
            local_coverage = coverage(routed, low_rows, groups)
            checks = point_gates(whole, low, protocol)
            checks.update(
                {
                    "minimum_whole_row_coverage": bool(
                        local_coverage["whole_row_fraction"]
                        >= float(coverage_gates["minimum_whole_row_fraction"])
                    ),
                    "minimum_low_prevalence_row_coverage": bool(
                        local_coverage["low_prevalence_row_fraction"]
                        >= float(
                            coverage_gates[
                                "minimum_low_prevalence_row_fraction"
                            ]
                        )
                    ),
                    "minimum_low_prevalence_site_coverage": bool(
                        local_coverage["low_prevalence_site_fraction"]
                        >= float(
                            coverage_gates[
                                "minimum_low_prevalence_site_fraction"
                            ]
                        )
                    ),
                }
            )
            candidates.append(
                {
                    "key": key,
                    "maximum_site_size": maximum_site_size,
                    "maximum_predicted_positive_rate": (
                        maximum_predicted_positive_rate
                    ),
                    "coverage": local_coverage,
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
                        min(
                            whole["average_precision_delta"]["sensors"].values()
                        ),
                        whole["average_precision_delta"]["pooled"],
                        local_coverage["low_prevalence_row_fraction"],
                        -local_coverage["whole_row_fraction"],
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
            "post-fold2-failure development-only label-free unseen-site "
            "routing on folds 3+4"
        ),
        "status": (
            "candidate_found_for_independent_folds01_confirmation"
            if selected is not None
            else "rejected_before_folds01"
        ),
        "decision": (
            "Freeze the selected router for independent folds-0/1 confirmation."
            if selected is not None
            else (
                "Retire this label-free unseen-site router before folds 0/1 "
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
            "development_pair_report_sha256": sha256(
                paths["development_pair_report"]
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
