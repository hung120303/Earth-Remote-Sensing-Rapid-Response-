#!/usr/bin/env python3
"""Test uniform global shrinkage of the fixed invariant residual pair."""

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
    "configs/mars_dense_invariant_global_shrinkage_protocol.json"
)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def compact_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "key": candidate["key"],
        "shrinkage": candidate["shrinkage"],
        "effective_anchor_strength": candidate["effective_anchor_strength"],
        "effective_booster_strength": candidate["effective_booster_strength"],
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
        raise ValueError("Frozen global-shrinkage explorer hash mismatch")
    for dependency in protocol["code_dependencies"]:
        path = (ROOT / dependency["path"]).resolve()
        if sha256(path) != dependency["sha256"]:
            raise ValueError(f"Frozen dependency mismatch: {dependency['path']}")
    paths: dict[str, Path] = {}
    for name, contract in protocol["inputs"].items():
        path = (ROOT / contract["path"]).resolve()
        if sha256(path) != contract["sha256"]:
            raise ValueError(f"Frozen global-shrinkage input mismatch: {name}")
        paths[name] = path
    fold2_report = json.loads(
        paths["fold2_report"].read_text(encoding="utf-8")
    )
    if str(fold2_report["status"]) != "rejected_on_fold2_confirmation":
        raise ValueError("Global shrinkage lacks the expected fold-2 motivation")
    smooth_report = json.loads(
        paths["smooth_trust_report"].read_text(encoding="utf-8")
    )
    if str(smooth_report["status"]) != "rejected_before_folds01":
        raise ValueError(
            "Global shrinkage lacks the expected smooth-router rejection"
        )
    output_path = (ROOT / protocol["output"]).resolve()
    if not args.smoke and output_path.exists():
        raise FileExistsError(
            "Refusing to repeat or overwrite the global-shrinkage report"
        )

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
        raise ValueError("Global-shrinkage cache contains unexpected folds")
    pair_scores = fixed_scores(matrix, metadata, protocol)
    pair_residual = logit(pair_scores) - logit(base_scores)
    if not np.isfinite(pair_residual).all():
        raise ValueError("Fixed pair residual is non-finite")
    low_rows = low_prevalence_mask(
        labels,
        groups,
        float(protocol["target_domain"]["maximum_site_positive_rate"]),
    )
    all_rows = np.ones(labels.size, dtype=bool)
    shrinkages = [float(value) for value in protocol["shrinkages"]]
    if args.smoke:
        shrinkages = shrinkages[:2]
    candidates: list[dict[str, Any]] = []
    scores_by_key: dict[str, np.ndarray] = {}
    for shrinkage in shrinkages:
        if not 0.0 < shrinkage < 1.0:
            raise ValueError("Global shrinkage must be strictly between zero and one")
        scores = sigmoid(logit(base_scores) + shrinkage * pair_residual)
        key = f"shrinkage{shrinkage:g}"
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
                "shrinkage": shrinkage,
                "effective_anchor_strength": float(
                    protocol["candidate"]["anchor_strength"]
                )
                * shrinkage,
                "effective_booster_strength": float(
                    protocol["candidate"]["booster_strength"]
                )
                * shrinkage,
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
                    -shrinkage,
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
                min(
                    row["paired_site_ap_delta"]["whole"]["lower"],
                    row["paired_site_ap_delta"]["low_prevalence"]["lower"],
                ),
                row["paired_site_ap_delta"]["whole"]["lower"],
                row["paired_site_ap_delta"]["low_prevalence"]["lower"],
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
            "post-fold2-failure uniform shrinkage of the fixed invariant pair "
            "using folds 3+4 only"
        ),
        "status": (
            "candidate_found_for_independent_folds01_confirmation"
            if selected is not None
            else "rejected_before_folds01"
        ),
        "decision": (
            "Freeze the selected global shrinkage for folds-0/1 confirmation."
            if selected is not None
            else (
                "Retire globally shrunken invariant pairs before folds 0/1 "
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
            "smooth_trust_report_sha256": sha256(
                paths["smooth_trust_report"]
            ),
            "fold2_report_sha256": sha256(paths["fold2_report"]),
            "numpy": np.__version__,
        },
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
    write_json(output_path, report)
    print(json.dumps({"ok": selected is not None, **summary}, indent=2))
    return 0 if selected is not None else 2


if __name__ == "__main__":
    raise SystemExit(main())
