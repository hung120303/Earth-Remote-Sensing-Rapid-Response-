#!/usr/bin/env python3
"""Confirm the frozen DOFA-v2 scene probe across new projection seeds."""

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
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "EarthRemoteSensingRapidResponse", ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from acquire_mars_metadata import repo_root, sha256  # noqa: E402
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
from train_mars_scene_ranker import blend_scores, safe_logit  # noqa: E402

DEFAULT_PROTOCOL = Path("configs/mars_dofa_v2_projection_ensemble_protocol.json")
DEFAULT_SELECTION_REPORT = Path(
    "reports/experiments/mars_dofa_v2_scene_probe_folds34.json"
)
DEFAULT_JSON = Path(
    "reports/experiments/mars_dofa_v2_projection_ensemble_folds34.json"
)
DEFAULT_MARKDOWN = Path(
    "reports/experiments/MARS_DOFA_V2_PROJECTION_ENSEMBLE_FOLDS34.md"
)


def mean_logit_probabilities(probabilities: list[np.ndarray]) -> np.ndarray:
    """Average independent predictions in log-odds space."""
    if not probabilities:
        raise ValueError("At least one probability vector is required")
    shape = np.asarray(probabilities[0]).shape
    if any(np.asarray(value).shape != shape for value in probabilities):
        raise ValueError("Probability vectors must have identical shapes")
    logits = np.stack([safe_logit(value) for value in probabilities], axis=0)
    result = expit(logits.mean(axis=0))
    if not np.isfinite(result).all():
        raise RuntimeError("Mean-logit probabilities are not finite")
    return result.astype(np.float64)


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    aggregate = report["aggregate"]
    delta = aggregate["evaluation"]["versus_current"]["delta"]
    interval = aggregate["paired_group_bootstrap_ap_delta_vs_current"]
    lines = [
        "# DOFA-v2 projection-seed confirmation — folds 3/4",
        "",
        "This preregistered confirmation holds the feature family, logistic "
        "regularization, blend, and folds fixed. It averages five newly seeded "
        "sparse-projection probes in log-odds space; the seed used during initial "
        "selection is excluded.",
        "",
        f"- Fixed feature set / C / blend: `{report['fixed_candidate']['feature_set']}` / "
        f"{report['fixed_candidate']['C']} / {report['fixed_candidate']['blend']:.3f}",
        f"- New projection seeds: `{report['projection_seeds']}`",
        f"- Aggregate AP delta vs current: {delta['average_precision']:+.6f}",
        f"- Aggregate recall delta at FPR 0.0713: "
        f"{delta['recall_at_fpr_0_0713']:+.6f}",
        f"- Paired-site AP 95% interval: [{interval['lower']:+.6f}, "
        f"{interval['upper']:+.6f}]",
        "",
        "| Projection seed | AP delta | Recall delta | Stable point gates |",
        "|---:|---:|---:|:---:|",
    ]
    for item in report["individual_seed_diagnostics"]:
        local = item["evaluation"]["versus_current"]["delta"]
        lines.append(
            f"| {item['seed']} | {local['average_precision']:+.6f} | "
            f"{local['recall_at_fpr_0_0713']:+.6f} | "
            f"{'yes' if item['evaluation']['stable'] else 'no'} |"
        )
    lines.extend(["", report["decision"]])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_protocol(path: Path) -> dict[str, Any]:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    candidate = protocol["fixed_candidate"]
    if candidate != {
        "feature_set": "change_extreme",
        "C": 0.01,
        "blend": 0.05,
    }:
        raise ValueError("Confirmation candidate differs from frozen selection")
    seeds = [int(value) for value in protocol["projection_seeds"]]
    if len(seeds) != 5 or len(set(seeds)) != 5 or 20260760 in seeds:
        raise ValueError("Expected five unique new projection seeds")
    if int(protocol["projection_dimension"]) != PROJECTION_DIM:
        raise ValueError("Projection dimension differs from frozen probe")
    return protocol


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL.as_posix())
    parser.add_argument("--dofa", default=DEFAULT_DOFA.as_posix())
    parser.add_argument("--dofa-sha256", required=True)
    parser.add_argument(
        "--selection-report", default=DEFAULT_SELECTION_REPORT.as_posix()
    )
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
    selection_report_path = (root / args.selection_report).resolve()
    protocol = load_protocol(protocol_path)
    dependencies = protocol["dependencies"]
    if sha256(selection_report_path) != dependencies["selection_report_sha256"]:
        raise ValueError("Initial DOFA-v2 selection report hash mismatch")
    if sha256((root / dependencies["selection_script"]).resolve()) != dependencies[
        "selection_script_sha256"
    ]:
        raise ValueError("Initial DOFA-v2 selection script hash mismatch")

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
    fixed = protocol["fixed_candidate"]
    features, _ = select_features(encoded, names, str(fixed["feature_set"]))
    del encoded
    gc.collect()

    raw_scores: list[np.ndarray] = []
    diagnostics: list[dict[str, Any]] = []
    for seed in map(int, protocol["projection_seeds"]):
        views = build_projected_views(features, values["folds"], seed=seed)
        raw = crossfit_scores(views, values["labels"], float(fixed["C"]))
        raw_scores.append(raw)
        evaluation = evaluate_candidate(
            values,
            raw,
            {
                "feature_set": fixed["feature_set"],
                "C": fixed["C"],
                "projection_seed": seed,
                "role": "diagnostic_only",
            },
            float(fixed["blend"]),
        )
        diagnostics.append(
            {"seed": seed, "evaluation": candidate_summary(evaluation)}
        )
        del views
        gc.collect()

    aggregate_raw = mean_logit_probabilities(raw_scores)
    aggregate_evaluation = evaluate_candidate(
        values,
        aggregate_raw,
        {
            "feature_set": fixed["feature_set"],
            "C": fixed["C"],
            "projection_seeds": protocol["projection_seeds"],
            "aggregation": "mean_logit",
        },
        float(fixed["blend"]),
    )
    aggregate_scores = blend_scores(
        values["current"], aggregate_raw, float(fixed["blend"])
    )
    bootstrap = ap_group_bootstrap(
        values["labels"],
        values["current"],
        aggregate_scores,
        values["groups"],
        replicates=int(protocol["bootstrap"]["replicates"]),
        seed=int(protocol["bootstrap"]["seed"]),
    )
    passed = bool(aggregate_evaluation["stable"] and bootstrap["lower"] > 0.0)
    report = {
        "schema_version": 1,
        "scope": "development-only DOFA-v2 projection-seed confirmation on folds 3/4",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_folds": list(SELECTION_FOLDS),
        "fixed_candidate": fixed,
        "projection_seeds": protocol["projection_seeds"],
        "aggregation": "mean_logit",
        "individual_seed_diagnostics": diagnostics,
        "aggregate": {
            "evaluation": aggregate_evaluation,
            "paired_group_bootstrap_ap_delta_vs_current": bootstrap,
        },
        "all_promotion_gates_pass": passed,
        "decision": (
            "Promote the fixed five-projection DOFA-v2 ensemble for one-shot fold-2 extraction."
            if passed
            else "Reject the DOFA-v2 projection ensemble before fold-2 extraction."
        ),
        "provenance": {
            **{f"{name}_sha256": digest for name, digest in expected.items()},
            "protocol_sha256": sha256(protocol_path),
            "selection_report_sha256": sha256(selection_report_path),
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
    delta = aggregate_evaluation["versus_current"]["delta"]
    print(
        json.dumps(
            {
                "ok": passed,
                "ap_delta_vs_current": delta["average_precision"],
                "recall_delta_vs_current": delta["recall_at_fpr_0_0713"],
                "ap_lower_vs_current": bootstrap["lower"],
            },
            indent=2,
        )
    )
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
