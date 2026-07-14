#!/usr/bin/env python3
"""Select the minimum OOF-stable blend for the frozen context head family."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import scipy
import sklearn

from acquire_mars_metadata import repo_root, sha256
from train_mars_context_scene_ranker import augment_site_context
from train_mars_oof_context_ranker import INNER_FOLDS, stability_checks
from train_mars_scene_ranker import (
    blend_scores,
    comparison,
    fit_model,
    metric_summary,
    predict_model,
    site_cell_weights,
)

DEFAULT_CACHE = Path("outputs/mars_scene_features_folds234.npz")
DEFAULT_CACHE_SHA256 = "01d8587e283c1179d61a7c789eb514b3f699d3e7a75bf8c50e4baff3f1698b89"
DEFAULT_OOF_REPORT = Path("reports/experiments/mars_oof_context_ranker_folds234.json")
DEFAULT_OOF_REPORT_SHA256 = "a125830c41d1d592a7d3a52ee2343ad5faa883061869e4638113c9df09f421e0"
DEFAULT_JSON = Path("reports/experiments/mars_oof_context_minimum_blend.json")
DEFAULT_MARKDOWN = Path("reports/experiments/MARS_OOF_CONTEXT_MINIMUM_BLEND.md")
BLEND_GRID = (0.015625, 0.03125, 0.046875, 0.0625, 0.09375, 0.125, 0.15625, 0.1875, 0.21875, 0.25)
FROZEN_SPEC = {
    "family": "hist_gradient_boosting",
    "max_leaf_nodes": 31,
    "min_samples_leaf": 20,
    "l2_regularization": 10.0,
}


def select_minimum_stable(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    passing = [candidate for candidate in candidates if all(candidate["stability_checks"].values())]
    if not passing:
        return max(candidates, key=lambda candidate: tuple(candidate["rank"]))
    return min(passing, key=lambda candidate: float(candidate["blend_lambda"]))


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    selected = report["selected"]
    lines = [
        "# Minimum OOF-stable context-head blend",
        "",
        "Fixed HGB model; complete OOF predictions on folds 2/3/4. Folds 0/1 and the paper test were not loaded.",
        "",
        f"- Selected blend: {selected['blend_lambda']}",
        f"- Pooled AP delta: {selected['delta']['average_precision']:+.5f}",
        f"- Pooled recall delta: {selected['delta']['recall_at_fpr_0_0713']:+.5f}",
        f"- Extra true positives: {selected['extra_true_positives']}",
        f"- Stability pass: {'yes' if all(selected['stability_checks'].values()) else 'no'}",
        "",
        report["decision"],
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", default=DEFAULT_CACHE.as_posix())
    parser.add_argument("--cache-sha256", default=DEFAULT_CACHE_SHA256)
    parser.add_argument("--oof-report", default=DEFAULT_OOF_REPORT.as_posix())
    parser.add_argument("--oof-report-sha256", default=DEFAULT_OOF_REPORT_SHA256)
    parser.add_argument("--output-json", default=DEFAULT_JSON.as_posix())
    parser.add_argument("--output-markdown", default=DEFAULT_MARKDOWN.as_posix())
    args = parser.parse_args()
    root = repo_root()
    cache_path = (root / args.cache).resolve()
    report_path = (root / args.oof_report).resolve()
    if sha256(cache_path) != args.cache_sha256 or sha256(report_path) != args.oof_report_sha256:
        raise ValueError("Frozen cache or OOF report hash mismatch")
    previous = json.loads(report_path.read_text(encoding="utf-8"))
    if previous["selected"]["spec"] != FROZEN_SPEC:
        raise ValueError("Frozen HGB spec differs from the OOF-selected family")
    with np.load(cache_path, allow_pickle=False) as cache:
        base_features = cache["features"].astype(np.float64)
        base_names = cache["feature_names"].astype(str)
        labels = cache["labels"].astype(np.uint8)
        sensors = cache["sensors"].astype(np.uint8)
        groups = cache["groups"].astype(str)
        folds = cache["folds"].astype(np.uint8)
    if set(np.unique(folds).tolist()) != set(INNER_FOLDS):
        raise ValueError("Minimum-blend cache must contain folds 2/3/4 only")
    features, _ = augment_site_context(base_features, base_names, groups)
    primary_index = int(np.flatnonzero(base_names == "primary_connected_score")[0])
    oof_head = np.empty(labels.shape, dtype=np.float64)
    for holdout in INNER_FOLDS:
        fit_rows = folds != holdout
        held_rows = folds == holdout
        fitted = fit_model(
            FROZEN_SPEC, features[fit_rows], labels[fit_rows],
            site_cell_weights(groups[fit_rows], labels[fit_rows], sensors[fit_rows]),
        )
        oof_head[held_rows] = predict_model(fitted, features[held_rows])
        print(json.dumps({"completed_holdout": holdout}), flush=True)
    baseline = metric_summary(labels, base_features[:, primary_index], sensors)
    baseline_by_fold = {
        str(fold): metric_summary(
            labels[folds == fold], base_features[folds == fold, primary_index], sensors[folds == fold]
        )
        for fold in INNER_FOLDS
    }
    positive_count = int(np.count_nonzero(labels == 1))
    candidates: list[dict[str, Any]] = []
    for blend_lambda in BLEND_GRID:
        scores = blend_scores(base_features[:, primary_index], oof_head, blend_lambda)
        pooled = comparison(metric_summary(labels, scores, sensors), baseline)
        per_fold = {
            str(fold): comparison(
                metric_summary(labels[folds == fold], scores[folds == fold], sensors[folds == fold]),
                baseline_by_fold[str(fold)],
            )
            for fold in INNER_FOLDS
        }
        candidate = {**pooled, "blend_lambda": blend_lambda, "per_fold": per_fold}
        candidate["stability_checks"] = stability_checks(candidate, per_fold, positive_count)
        candidate["extra_true_positives"] = round(
            candidate["delta"]["recall_at_fpr_0_0713"] * positive_count
        )
        candidates.append(candidate)
    selected = select_minimum_stable(candidates)
    passed = all(selected["stability_checks"].values())
    decision = (
        "Freeze the minimum stable blend for one fold-0 evaluation."
        if passed else "Reject the fine minimum-intervention trust region."
    )
    report = {
        "schema_version": 1,
        "scope": "minimum-intervention OOF trust region on folds 2/3/4",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "frozen_spec": FROZEN_SPEC, "blend_grid": list(BLEND_GRID),
        "selection_rule": "smallest blend passing every prior OOF stability gate",
        "candidates": candidates, "selected": selected, "decision": decision,
        "provenance": {
            "cache_sha256": args.cache_sha256,
            "oof_report_sha256": args.oof_report_sha256,
            "script_sha256": sha256(Path(__file__).resolve()),
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
            "numpy": np.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__,
        },
    }
    output_json = (root / args.output_json).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown((root / args.output_markdown).resolve(), report)
    print(json.dumps({
        "ok": passed, "selected_blend": selected["blend_lambda"],
        "deltas": selected["delta"], "extra_true_positives": selected["extra_true_positives"],
        "stability_checks": selected["stability_checks"], "decision": decision,
    }, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
