#!/usr/bin/env python3
"""Select a recall-protecting trust region around the current MARS scene head."""

from __future__ import annotations

import argparse
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

from acquire_mars_metadata import repo_root, sha256  # noqa: E402
from train_mars_crossfold_bagged_scene_head import (  # noqa: E402
    AGGREGATIONS,
    FOLDS,
    DEFAULT_FOLD0_CACHE,
    DEFAULT_FOLD0_SHA256,
    DEFAULT_FOLD1_CACHE,
    DEFAULT_FOLD1_SHA256,
    DEFAULT_INNER_CACHE,
    DEFAULT_INNER_SHA256,
    DEFAULT_SCORE_CACHE,
    DEFAULT_SCORE_SHA256,
    aggregate_predictions,
    load_development,
    oof_member_predictions,
)
from train_mars_oof_scene_ensemble_v2 import ap_group_bootstrap  # noqa: E402
from train_mars_scene_ranker import blend_scores, comparison, metric_summary  # noqa: E402

DEFAULT_SOURCE_ARTIFACT = Path(
    "EarthRemoteSensingRapidResponse/artifacts/mars_crossfold_bagged_scene_head.joblib"
)
DEFAULT_SOURCE_ARTIFACT_SHA256 = (
    "c9efecc2315305bb306b6deb84037f60096bdfae94164e3e9e646a2612fdcafb"
)
DEFAULT_ARTIFACT = Path(
    "EarthRemoteSensingRapidResponse/artifacts/mars_crossfold_bagged_trust_region.joblib"
)
DEFAULT_JSON = Path("reports/experiments/mars_crossfold_bagged_trust_region.json")
DEFAULT_MARKDOWN = Path(
    "reports/experiments/MARS_CROSSFOLD_BAGGED_TRUST_REGION.md"
)
WEIGHTS = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.625, 0.75, 0.875, 1.0)
MINIMUM_POOLED_AP_GAIN = 0.002


def evaluate_candidate(
    values: dict[str, Any], raw: np.ndarray, aggregation: str, weight: float
) -> dict[str, Any]:
    scores = blend_scores(values["current"], raw, weight)
    candidate = metric_summary(values["labels"], scores, values["sensors"])
    primary = metric_summary(values["labels"], values["primary"], values["sensors"])
    current = metric_summary(values["labels"], values["current"], values["sensors"])
    versus_primary = comparison(candidate, primary)
    versus_current = comparison(candidate, current)
    per_fold = {}
    for fold in FOLDS:
        rows = values["folds"] == fold
        local = metric_summary(values["labels"][rows], scores[rows], values["sensors"][rows])
        per_fold[str(fold)] = {
            "versus_primary": comparison(
                local,
                metric_summary(
                    values["labels"][rows], values["primary"][rows], values["sensors"][rows]
                ),
            ),
            "versus_current": comparison(
                local,
                metric_summary(
                    values["labels"][rows], values["current"][rows], values["sensors"][rows]
                ),
            ),
        }
    fold_ap = [
        value["versus_current"]["delta"]["average_precision"]
        for value in per_fold.values()
    ]
    fold_recall = [
        value["versus_current"]["delta"]["recall_at_fpr_0_0713"]
        for value in per_fold.values()
    ]
    stable = (
        versus_current["delta"]["average_precision"] >= MINIMUM_POOLED_AP_GAIN
        and versus_current["delta"]["recall_at_fpr_0_0713"] > 0.0
        and min(fold_ap) > 0.0
        and min(fold_recall) >= 0.0
        and min(versus_current["delta"]["sensor_average_precision"].values()) >= 0.0
        and versus_primary["delta"]["average_precision"] > 0.0
        and versus_primary["delta"]["recall_at_fpr_0_0713"] > 0.0
    )
    return {
        "aggregation": aggregation,
        "bagged_weight": weight,
        "stable": bool(stable),
        "versus_primary": versus_primary,
        "versus_current": versus_current,
        "per_fold": per_fold,
        "rank": [
            int(stable),
            min(fold_recall),
            min(fold_ap),
            versus_current["delta"]["average_precision"],
            -weight,
        ],
    }


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    selected = report["selected"]
    delta = selected["versus_current"]["delta"]
    interval = selected["paired_group_bootstrap_ap_delta_vs_current"]
    lines = [
        "# Crossfold-bagged scene-head trust region",
        "",
        "The frozen bagged ensemble is blended around the current v3 head to protect held-fold recall.",
        "",
        f"- Aggregation: `{selected['aggregation']}`",
        f"- Bagged weight: {selected['bagged_weight']:.3f}",
        f"- AP delta vs current: {delta['average_precision']:+.5f}",
        f"- Recall delta vs current: {delta['recall_at_fpr_0_0713']:+.5f}",
        f"- Paired-site AP interval vs current: [{interval['lower']:+.5f}, {interval['upper']:+.5f}]",
        "",
        "| Fold | AP delta vs current | Recall delta vs current |",
        "|---|---:|---:|",
    ]
    for fold, value in selected["per_fold"].items():
        local = value["versus_current"]["delta"]
        lines.append(
            f"| {fold} | {local['average_precision']:+.5f} | "
            f"{local['recall_at_fpr_0_0713']:+.5f} |"
        )
    lines.extend(["", report["decision"]])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inner-cache", default=DEFAULT_INNER_CACHE.as_posix())
    parser.add_argument("--inner-sha256", default=DEFAULT_INNER_SHA256)
    parser.add_argument("--fold0-cache", default=DEFAULT_FOLD0_CACHE.as_posix())
    parser.add_argument("--fold0-sha256", default=DEFAULT_FOLD0_SHA256)
    parser.add_argument("--fold1-cache", default=DEFAULT_FOLD1_CACHE.as_posix())
    parser.add_argument("--fold1-sha256", default=DEFAULT_FOLD1_SHA256)
    parser.add_argument("--score-cache", default=DEFAULT_SCORE_CACHE.as_posix())
    parser.add_argument("--score-sha256", default=DEFAULT_SCORE_SHA256)
    parser.add_argument("--source-artifact", default=DEFAULT_SOURCE_ARTIFACT.as_posix())
    parser.add_argument(
        "--source-artifact-sha256", default=DEFAULT_SOURCE_ARTIFACT_SHA256
    )
    parser.add_argument("--artifact", default=DEFAULT_ARTIFACT.as_posix())
    parser.add_argument("--output-json", default=DEFAULT_JSON.as_posix())
    parser.add_argument("--output-markdown", default=DEFAULT_MARKDOWN.as_posix())
    args = parser.parse_args()
    root = repo_root()
    paths = {
        "inner": (root / args.inner_cache).resolve(),
        "fold0": (root / args.fold0_cache).resolve(),
        "fold1": (root / args.fold1_cache).resolve(),
        "score": (root / args.score_cache).resolve(),
        "source_artifact": (root / args.source_artifact).resolve(),
    }
    expected = {
        "inner": args.inner_sha256,
        "fold0": args.fold0_sha256,
        "fold1": args.fold1_sha256,
        "score": args.score_sha256,
        "source_artifact": args.source_artifact_sha256,
    }
    for name, digest in expected.items():
        if sha256(paths[name]) != digest:
            raise ValueError(f"Frozen {name} hash mismatch")
    source = joblib.load(paths["source_artifact"])
    if source.get("kind") != "mars_crossfold_bagged_extra_trees_scene_head" or len(
        source.get("models", [])
    ) != 5:
        raise ValueError("Unexpected crossfold source artifact")
    values = load_development(
        {name: paths[name] for name in ("inner", "fold0", "fold1")}, paths["score"]
    )
    members = oof_member_predictions(values)
    raw_by_aggregation = {
        aggregation: aggregate_predictions(members, aggregation)
        for aggregation in AGGREGATIONS
    }
    candidates = [
        evaluate_candidate(values, raw_by_aggregation[aggregation], aggregation, weight)
        for aggregation in AGGREGATIONS
        for weight in WEIGHTS
    ]
    selected = max(candidates, key=lambda value: tuple(value["rank"]))
    selected_scores = blend_scores(
        values["current"],
        raw_by_aggregation[selected["aggregation"]],
        float(selected["bagged_weight"]),
    )
    selected["paired_group_bootstrap_ap_delta_vs_primary"] = ap_group_bootstrap(
        values["labels"],
        values["primary"],
        selected_scores,
        values["groups"],
        replicates=10_000,
        seed=20261240,
    )
    selected["paired_group_bootstrap_ap_delta_vs_current"] = ap_group_bootstrap(
        values["labels"],
        values["current"],
        selected_scores,
        values["groups"],
        replicates=10_000,
        seed=20261241,
    )
    passed = bool(
        selected["stable"]
        and selected["paired_group_bootstrap_ap_delta_vs_primary"]["lower"] > 0.0
        and selected["paired_group_bootstrap_ap_delta_vs_current"]["lower"] > 0.0
    )
    thresholds = [
        value["versus_primary"]["metrics"]["operating_point"]["threshold"]
        for value in selected["per_fold"].values()
    ]
    artifact_path = (root / args.artifact).resolve()
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = artifact_path.with_suffix(artifact_path.suffix + ".tmp")
    joblib.dump(
        {
            "schema_version": 1,
            "kind": "mars_crossfold_bagged_scene_trust_region",
            "source_artifact_sha256": args.source_artifact_sha256,
            "aggregation": selected["aggregation"],
            "bagged_weight": float(selected["bagged_weight"]),
            "base_score": "frozen v3 stronger scene head",
            "operational_scene_threshold": max(thresholds),
        },
        temporary,
        compress=3,
    )
    os.replace(temporary, artifact_path)
    report = {
        "schema_version": 1,
        "scope": "five-fold nested OOF trust-region selection; paper cache not loaded",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "minimum_pooled_ap_gain": MINIMUM_POOLED_AP_GAIN,
        "aggregations": list(AGGREGATIONS),
        "bagged_weights": list(WEIGHTS),
        "candidate_summaries": [
            {
                "aggregation": value["aggregation"],
                "bagged_weight": value["bagged_weight"],
                "stable": value["stable"],
                "ap_delta_vs_current": value["versus_current"]["delta"][
                    "average_precision"
                ],
                "recall_delta_vs_current": value["versus_current"]["delta"][
                    "recall_at_fpr_0_0713"
                ],
                "worst_fold_ap_delta_vs_current": min(
                    fold["versus_current"]["delta"]["average_precision"]
                    for fold in value["per_fold"].values()
                ),
                "worst_fold_recall_delta_vs_current": min(
                    fold["versus_current"]["delta"]["recall_at_fpr_0_0713"]
                    for fold in value["per_fold"].values()
                ),
            }
            for value in candidates
        ],
        "selected": selected,
        "operational_scene_threshold": max(thresholds),
        "all_promotion_gates_pass": passed,
        "decision": (
            "Freeze the bagged trust-region scene head for one transparent paper replay."
            if passed
            else "Reject the bagged trust region before paper-cache scoring."
        ),
        "provenance": {
            **{f"{name}_sha256": digest for name, digest in expected.items()},
            "artifact_sha256": sha256(artifact_path),
            "script_sha256": sha256(Path(__file__).resolve()),
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True
            ).strip(),
            "numpy": np.__version__,
            "joblib": joblib.__version__,
        },
    }
    write_json((root / args.output_json).resolve(), report)
    write_markdown((root / args.output_markdown).resolve(), report)
    print(
        json.dumps(
            {
                "ok": passed,
                "aggregation": selected["aggregation"],
                "bagged_weight": selected["bagged_weight"],
                "ap_delta_vs_current": selected["versus_current"]["delta"][
                    "average_precision"
                ],
                "recall_delta_vs_current": selected["versus_current"]["delta"][
                    "recall_at_fpr_0_0713"
                ],
                "ap_lower_vs_current": selected[
                    "paired_group_bootstrap_ap_delta_vs_current"
                ]["lower"],
                "artifact_sha256": report["provenance"]["artifact_sha256"],
            },
            indent=2,
        )
    )
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
