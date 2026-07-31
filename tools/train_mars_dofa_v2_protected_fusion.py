#!/usr/bin/env python3
"""Select a recall-protected DOFA-v2 fusion on MARS folds 3 and 4."""

from __future__ import annotations

import argparse
import gc
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import sklearn

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "EarthRemoteSensingRapidResponse", ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from acquire_mars_metadata import repo_root, sha256  # noqa: E402
from confirm_mars_dofa_v2_projection_ensemble import (  # noqa: E402
    mean_logit_probabilities,
)
from train_mars_crossfold_bagged_scene_head import (  # noqa: E402
    DEFAULT_FOLD0_CACHE,
    DEFAULT_FOLD0_SHA256,
    DEFAULT_FOLD1_CACHE,
    DEFAULT_FOLD1_SHA256,
    DEFAULT_INNER_CACHE,
    DEFAULT_INNER_SHA256,
    DEFAULT_SCORE_CACHE,
    DEFAULT_SCORE_SHA256,
    load_development,
)
from train_mars_dofa_v2_scene_probe import (  # noqa: E402
    DEFAULT_DOFA,
    PROJECTION_DIM,
    SELECTION_FOLDS,
    align_features,
    build_projected_views,
    candidate_summary,
    crossfit_scores,
    evaluate_candidate,
    select_features,
    write_json,
)
from train_mars_oof_scene_ensemble_v2 import ap_group_bootstrap  # noqa: E402
from train_mars_scene_ranker import blend_scores, metric_summary  # noqa: E402

DEFAULT_PROTOCOL = Path("configs/mars_dofa_v2_protected_fusion_protocol.json")
DEFAULT_ENSEMBLE_REPORT = Path(
    "reports/experiments/mars_dofa_v2_projection_ensemble_folds34.json"
)
DEFAULT_JSON = Path("reports/experiments/mars_dofa_v2_protected_fusion_folds34.json")
DEFAULT_MARKDOWN = Path(
    "reports/experiments/MARS_DOFA_V2_PROTECTED_FUSION_FOLDS34.md"
)


def protected_logit_blend(
    current: np.ndarray, dofa: np.ndarray, *, gate: float, weight: float
) -> np.ndarray:
    """Blend only above ``gate`` while preserving separation from lower scores."""
    current_values = np.asarray(current, dtype=np.float64)
    dofa_values = np.asarray(dofa, dtype=np.float64)
    if current_values.shape != dofa_values.shape:
        raise ValueError("Current and DOFA probabilities must have identical shapes")
    if not 0.0 < gate < 1.0:
        raise ValueError("Gate must be strictly inside (0,1)")
    if not 0.0 <= weight <= 1.0:
        raise ValueError("Blend weight must be in [0,1]")
    if (
        not np.isfinite(current_values).all()
        or not np.isfinite(dofa_values).all()
        or (current_values < 0.0).any()
        or (current_values > 1.0).any()
        or (dofa_values < 0.0).any()
        or (dofa_values > 1.0).any()
    ):
        raise ValueError("Inputs must be finite probabilities")
    high = current_values >= gate
    result = current_values.copy()
    local_current = (current_values[high] - gate) / (1.0 - gate)
    local_blend = blend_scores(local_current, dofa_values[high], weight)
    result[high] = gate + (1.0 - gate) * local_blend
    if (result[~high] != current_values[~high]).any():
        raise RuntimeError("Protected fusion altered a below-gate score")
    if high.any() and float(result[high].min()) < gate:
        raise RuntimeError("Protected fusion crossed its confidence gate")
    return result


def operating_counts_preserved(
    candidate: dict[str, Any], current_metrics: dict[str, Any]
) -> bool:
    metrics = candidate["versus_current"]["metrics"]
    return all(metrics[key] == current_metrics[key] for key in ("tp", "fp", "tn", "fn"))


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    selected = report["selected"]
    evaluation = selected["evaluation"]
    delta = evaluation["versus_current"]["delta"]
    interval = selected["paired_group_bootstrap_ap_delta_vs_current"]
    lines = [
        "# Recall-protected DOFA-v2 fusion - folds 3/4",
        "",
        "The fusion leaves every current-model score below a fixed confidence gate "
        "unchanged and maps every affected score back above that gate. DOFA-v2 can "
        "therefore rerank likely-plume scenes without altering the no-plume operating "
        "region.",
        "",
        f"- Selected gate / DOFA weight: {selected['gate']:.2f} / "
        f"{selected['weight']:.2f}",
        f"- AP delta vs current: {delta['average_precision']:+.6f}",
        f"- Recall delta at FPR 0.0713: {delta['recall_at_fpr_0_0713']:+.6f}",
        f"- Paired-site AP 95% interval: [{interval['lower']:+.6f}, "
        f"{interval['upper']:+.6f}]",
        f"- Operating confusion counts preserved: "
        f"{'yes' if selected['operating_counts_preserved'] else 'no'}",
        "",
        "| Gate | Weight | AP delta | Recall delta | AP CI lower | Promoted |",
        "|---:|---:|---:|---:|---:|:---:|",
    ]
    for candidate in report["candidate_summaries"]:
        delta = candidate["evaluation"]["delta_vs_current"]
        lower = candidate["paired_group_bootstrap_ap_delta_vs_current"]["lower"]
        lines.append(
            f"| {candidate['gate']:.2f} | {candidate['weight']:.2f} | "
            f"{delta['average_precision']:+.6f} | "
            f"{delta['recall_at_fpr_0_0713']:+.6f} | {lower:+.6f} | "
            f"{'yes' if candidate['promotion_gates_pass'] else 'no'} |"
        )
    lines.extend(["", report["decision"]])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_protocol(path: Path) -> dict[str, Any]:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol["fixed_representation"] != {
        "feature_set": "change_extreme",
        "C": 0.01,
        "projection_dimension": 2048,
        "projection_seeds": [20260780, 20260781, 20260782, 20260783, 20260784],
        "aggregation": "mean_logit",
    }:
        raise ValueError("Protected-fusion representation differs from confirmation")
    if int(protocol["fixed_representation"]["projection_dimension"]) != PROJECTION_DIM:
        raise ValueError("Projection dimension differs")
    expected_count = len(protocol["candidate_grid"]["gates"]) * len(
        protocol["candidate_grid"]["weights"]
    )
    if expected_count != int(protocol["eligible_candidates"]):
        raise ValueError("Candidate grid count differs")
    return protocol


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL.as_posix())
    parser.add_argument("--dofa", default=DEFAULT_DOFA.as_posix())
    parser.add_argument("--dofa-sha256", required=True)
    parser.add_argument("--ensemble-report", default=DEFAULT_ENSEMBLE_REPORT.as_posix())
    parser.add_argument("--inner-cache", default=DEFAULT_INNER_CACHE.as_posix())
    parser.add_argument("--inner-sha256", default=DEFAULT_INNER_SHA256)
    parser.add_argument("--fold0-cache", default=DEFAULT_FOLD0_CACHE.as_posix())
    parser.add_argument("--fold0-sha256", default=DEFAULT_FOLD0_SHA256)
    parser.add_argument("--fold1-cache", default=DEFAULT_FOLD1_CACHE.as_posix())
    parser.add_argument("--fold1-sha256", default=DEFAULT_FOLD1_SHA256)
    parser.add_argument("--score-cache", default=DEFAULT_SCORE_CACHE.as_posix())
    parser.add_argument("--score-sha256", default=DEFAULT_SCORE_SHA256)
    parser.add_argument("--output-json", default=DEFAULT_JSON.as_posix())
    parser.add_argument("--output-markdown", default=DEFAULT_MARKDOWN.as_posix())
    args = parser.parse_args()

    root = repo_root()
    protocol_path = (root / args.protocol).resolve()
    ensemble_report_path = (root / args.ensemble_report).resolve()
    protocol = load_protocol(protocol_path)
    dependencies = protocol["dependencies"]
    if sha256(ensemble_report_path) != dependencies["ensemble_report_sha256"]:
        raise ValueError("Projection-ensemble report hash mismatch")
    previous_report = json.loads(ensemble_report_path.read_text(encoding="utf-8"))

    paths = {
        "dofa": (root / args.dofa).resolve(),
        "inner": (root / args.inner_cache).resolve(),
        "fold0": (root / args.fold0_cache).resolve(),
        "fold1": (root / args.fold1_cache).resolve(),
        "score": (root / args.score_cache).resolve(),
    }
    expected = {
        "dofa": args.dofa_sha256,
        "inner": args.inner_sha256,
        "fold0": args.fold0_sha256,
        "fold1": args.fold1_sha256,
        "score": args.score_sha256,
    }
    for name, digest in expected.items():
        if sha256(paths[name]) != digest:
            raise ValueError(f"Frozen {name} hash mismatch")
    if expected["dofa"] != dependencies["dofa_cache_sha256"]:
        raise ValueError("DOFA cache differs from the frozen protocol")

    all_values = load_development(
        {name: paths[name] for name in ("inner", "fold0", "fold1")}, paths["score"]
    )
    encoded, names = align_features(paths["dofa"], all_values)
    selection = np.isin(all_values["folds"], SELECTION_FOLDS)
    values = {
        key: np.asarray(all_values[key])[selection]
        for key in (
            "labels",
            "sensors",
            "sample_ids",
            "groups",
            "folds",
            "primary",
            "current",
        )
    }
    representation = protocol["fixed_representation"]
    features, _ = select_features(encoded, names, representation["feature_set"])
    del encoded
    gc.collect()
    raw_scores: list[np.ndarray] = []
    for seed in map(int, representation["projection_seeds"]):
        views = build_projected_views(features, values["folds"], seed=seed)
        raw_scores.append(
            crossfit_scores(views, values["labels"], float(representation["C"]))
        )
        del views
        gc.collect()
    aggregate_raw = mean_logit_probabilities(raw_scores)

    reproduced = evaluate_candidate(
        values,
        aggregate_raw,
        {"role": "confirmation_reproduction"},
        float(protocol["reproduction_blend"]),
    )
    expected_delta = previous_report["aggregate"]["evaluation"]["versus_current"][
        "delta"
    ]
    observed_delta = reproduced["versus_current"]["delta"]
    for key in ("average_precision", "recall_at_fpr_0_0713"):
        if not np.isclose(observed_delta[key], expected_delta[key], rtol=0.0, atol=1e-14):
            raise RuntimeError("Fixed five-seed representation did not reproduce")

    current_metrics = metric_summary(
        values["labels"], values["current"], values["sensors"]
    )
    current_threshold = float(current_metrics["operating_point"]["threshold"])
    margin = float(protocol["minimum_gate_margin_above_current_operating_threshold"])
    candidates: list[dict[str, Any]] = []
    index = 0
    for gate in map(float, protocol["candidate_grid"]["gates"]):
        if gate < current_threshold + margin:
            raise ValueError("A frozen protection gate is too near the operating point")
        for weight in map(float, protocol["candidate_grid"]["weights"]):
            scores = protected_logit_blend(
                values["current"], aggregate_raw, gate=gate, weight=weight
            )
            evaluation = evaluate_candidate(
                values,
                scores,
                {
                    "family": "protected_logit_blend",
                    "gate": gate,
                    "weight": weight,
                },
                1.0,
            )
            preserved = operating_counts_preserved(evaluation, current_metrics)
            bootstrap = ap_group_bootstrap(
                values["labels"],
                values["current"],
                scores,
                values["groups"],
                replicates=int(protocol["bootstrap"]["replicates"]),
                seed=int(protocol["bootstrap"]["seed"]),
            )
            promoted = bool(
                evaluation["stable"] and preserved and bootstrap["lower"] > 0.0
            )
            fold_ap = [
                evaluation["per_fold"][str(fold)]["versus_current"]["delta"][
                    "average_precision"
                ]
                for fold in SELECTION_FOLDS
            ]
            sensor_ap = evaluation["versus_current"]["delta"][
                "sensor_average_precision"
            ]
            candidates.append(
                {
                    "gate": gate,
                    "weight": weight,
                    "evaluation": evaluation,
                    "operating_counts_preserved": preserved,
                    "paired_group_bootstrap_ap_delta_vs_current": bootstrap,
                    "promotion_gates_pass": promoted,
                    "rank": [
                        int(promoted),
                        int(evaluation["stable"]),
                        bootstrap["lower"],
                        min(fold_ap),
                        min(sensor_ap.values()),
                        evaluation["versus_current"]["delta"]["average_precision"],
                        -gate,
                        -weight,
                    ],
                    "candidate_index": index,
                }
            )
            index += 1

    selected = max(candidates, key=lambda value: tuple(value["rank"]))
    passed = bool(selected["promotion_gates_pass"])
    summaries = [
        {
            "gate": value["gate"],
            "weight": value["weight"],
            "evaluation": candidate_summary(value["evaluation"]),
            "operating_counts_preserved": value["operating_counts_preserved"],
            "paired_group_bootstrap_ap_delta_vs_current": value[
                "paired_group_bootstrap_ap_delta_vs_current"
            ],
            "promotion_gates_pass": value["promotion_gates_pass"],
            "rank": value["rank"],
        }
        for value in candidates
    ]
    selected_report = {
        key: value
        for key, value in selected.items()
        if key not in ("rank", "candidate_index")
    }
    report = {
        "schema_version": 1,
        "scope": "development-only recall-protected DOFA-v2 fusion on folds 3/4",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_folds": list(SELECTION_FOLDS),
        "fixed_representation": representation,
        "reproduction": {
            "previous_aggregate_delta_vs_current": expected_delta,
            "observed_aggregate_delta_vs_current": observed_delta,
            "exact_within_1e_14": True,
        },
        "current_operating_threshold": current_threshold,
        "candidate_contract": {
            **protocol["candidate_grid"],
            "candidate_count": len(candidates),
            "minimum_gate_margin_above_current_operating_threshold": margin,
        },
        "candidate_summaries": summaries,
        "selected": selected_report,
        "all_promotion_gates_pass": passed,
        "decision": (
            "Freeze the selected recall-protected DOFA-v2 fusion for one-shot fold-2 extraction."
            if passed
            else "Reject recall-protected DOFA-v2 fusion before fold-2 extraction."
        ),
        "provenance": {
            **{f"{name}_sha256": digest for name, digest in expected.items()},
            "protocol_sha256": sha256(protocol_path),
            "ensemble_report_sha256": sha256(ensemble_report_path),
            "script_sha256": sha256(Path(__file__).resolve()),
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True
            ).strip(),
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
        },
    }
    write_json((root / args.output_json).resolve(), report)
    write_markdown((root / args.output_markdown).resolve(), report)
    delta = selected["evaluation"]["versus_current"]["delta"]
    print(
        json.dumps(
            {
                "ok": passed,
                "gate": selected["gate"],
                "weight": selected["weight"],
                "ap_delta_vs_current": delta["average_precision"],
                "recall_delta_vs_current": delta["recall_at_fpr_0_0713"],
                "ap_lower_vs_current": selected[
                    "paired_group_bootstrap_ap_delta_vs_current"
                ]["lower"],
            },
            indent=2,
        )
    )
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
