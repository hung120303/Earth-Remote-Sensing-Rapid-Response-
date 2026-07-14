#!/usr/bin/env python3
"""Select a site-context scene head using three-fold out-of-fold stability."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import scipy
import sklearn

from acquire_mars_metadata import repo_root, sha256
from train_mars_context_scene_ranker import (
    CONTEXT_BASE_FEATURES,
    CONTEXT_BLEND_LAMBDAS,
    CONTEXT_STATISTICS,
    augment_site_context,
    context_specs,
)
from train_mars_scene_ranker import (
    blend_scores,
    comparison,
    fit_model,
    metric_summary,
    predict_model,
    site_cell_weights,
    spec_key,
)

DEFAULT_CACHE = Path("outputs/mars_scene_features_folds234.npz")
DEFAULT_CACHE_SHA256 = "01d8587e283c1179d61a7c789eb514b3f699d3e7a75bf8c50e4baff3f1698b89"
DEFAULT_ARTIFACT = Path("EarthRemoteSensingRapidResponse/artifacts/mars_oof_context_ranker_folds234.joblib")
DEFAULT_JSON = Path("reports/experiments/mars_oof_context_ranker_folds234.json")
DEFAULT_MARKDOWN = Path("reports/experiments/MARS_OOF_CONTEXT_RANKER_FOLDS234.md")
INNER_FOLDS = (2, 3, 4)
MINIMUM_EXTRA_TRUE_POSITIVES = 6
MINIMUM_POSITIVE_FOLDS = 2
MAXIMUM_FOLD_AP_REGRESSION = 0.005


def stability_checks(
    pooled: dict[str, Any], per_fold: dict[str, dict[str, Any]], positive_count: int
) -> dict[str, bool]:
    checks = dict(pooled["checks"])
    extra_true_positives = round(
        pooled["delta"]["recall_at_fpr_0_0713"] * positive_count
    )
    recall_deltas = [value["delta"]["recall_at_fpr_0_0713"] for value in per_fold.values()]
    ap_deltas = [value["delta"]["average_precision"] for value in per_fold.values()]
    checks.update(
        {
            "minimum_six_pooled_extra_true_positives": extra_true_positives
            >= MINIMUM_EXTRA_TRUE_POSITIVES,
            "no_inner_fold_recall_regression": all(value >= 0 for value in recall_deltas),
            "at_least_two_inner_folds_raise_recall": sum(value > 0 for value in recall_deltas)
            >= MINIMUM_POSITIVE_FOLDS,
            "no_material_inner_fold_ap_regression": all(
                value >= -MAXIMUM_FOLD_AP_REGRESSION for value in ap_deltas
            ),
        }
    )
    return checks


def select_stable_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    passing = [candidate for candidate in candidates if all(candidate["stability_checks"].values())]
    def rank(candidate: dict[str, Any]) -> tuple[float, ...]:
        fold_recall = [
            value["delta"]["recall_at_fpr_0_0713"]
            for value in candidate["per_fold"].values()
        ]
        return (*candidate["rank"], min(fold_recall))
    return max(passing or candidates, key=rank)


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    selected = report["selected"]
    lines = [
        "# Three-fold OOF site-context MARS ranker",
        "",
        "Every fold-2/3/4 prediction comes from a model trained on the other two folds. Folds 0/1 and the paper test were not loaded.",
        "",
        f"- Selected model: `{spec_key(selected['spec'])}`",
        f"- Head blend weight: {selected['blend_lambda']}",
        f"- Pooled AP delta: {selected['delta']['average_precision']:+.5f}",
        f"- Pooled recall delta: {selected['delta']['recall_at_fpr_0_0713']:+.5f}",
        f"- Stable authorization gate: {'pass' if all(selected['stability_checks'].values()) else 'fail'}",
        "",
        report["decision"],
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", default=DEFAULT_CACHE.as_posix())
    parser.add_argument("--cache-sha256", default=DEFAULT_CACHE_SHA256)
    parser.add_argument("--artifact", default=DEFAULT_ARTIFACT.as_posix())
    parser.add_argument("--output-json", default=DEFAULT_JSON.as_posix())
    parser.add_argument("--output-markdown", default=DEFAULT_MARKDOWN.as_posix())
    args = parser.parse_args()
    root = repo_root()
    cache_path = (root / args.cache).resolve()
    if sha256(cache_path) != args.cache_sha256:
        raise ValueError("Scene feature cache hash mismatch")
    with np.load(cache_path, allow_pickle=False) as cache:
        base_features = cache["features"].astype(np.float64)
        base_names = cache["feature_names"].astype(str)
        labels = cache["labels"].astype(np.uint8)
        sensors = cache["sensors"].astype(np.uint8)
        groups = cache["groups"].astype(str)
        folds = cache["folds"].astype(np.uint8)
        source_provenance = {
            key: str(cache[key].item())
            for key in ("artifact_sha256", "manifest_sha256", "protocol_sha256")
        }
    if set(np.unique(folds).tolist()) != set(INNER_FOLDS):
        raise ValueError("OOF cache must contain folds 2, 3, and 4 only")
    features, augmented_names = augment_site_context(base_features, base_names, groups)
    primary_index = int(np.flatnonzero(base_names == "primary_connected_score")[0])
    baseline = metric_summary(labels, base_features[:, primary_index], sensors)
    baseline_by_fold = {
        str(fold): metric_summary(
            labels[folds == fold], base_features[folds == fold, primary_index], sensors[folds == fold]
        )
        for fold in INNER_FOLDS
    }
    positive_count = int(np.count_nonzero(labels == 1))
    candidates: list[dict[str, Any]] = []
    for spec in context_specs():
        oof_head = np.empty(labels.shape, dtype=np.float64)
        for holdout in INNER_FOLDS:
            fit_rows = folds != holdout
            held_rows = folds == holdout
            fitted = fit_model(
                spec,
                features[fit_rows],
                labels[fit_rows],
                site_cell_weights(groups[fit_rows], labels[fit_rows], sensors[fit_rows]),
            )
            oof_head[held_rows] = predict_model(fitted, features[held_rows])
        for blend_lambda in CONTEXT_BLEND_LAMBDAS:
            scores = blend_scores(base_features[:, primary_index], oof_head, blend_lambda)
            pooled = comparison(metric_summary(labels, scores, sensors), baseline)
            per_fold = {
                str(fold): comparison(
                    metric_summary(labels[folds == fold], scores[folds == fold], sensors[folds == fold]),
                    baseline_by_fold[str(fold)],
                )
                for fold in INNER_FOLDS
            }
            candidate = {**pooled, "spec": spec, "blend_lambda": blend_lambda, "per_fold": per_fold}
            candidate["stability_checks"] = stability_checks(candidate, per_fold, positive_count)
            candidate["extra_true_positives"] = round(
                candidate["delta"]["recall_at_fpr_0_0713"] * positive_count
            )
            candidates.append(candidate)
        print(json.dumps({"completed": spec_key(spec)}), flush=True)
    selected = select_stable_candidate(candidates)
    passed = all(selected["stability_checks"].values())
    decision = (
        "Refit the OOF-stable context head on folds 2-4 and freeze fold-0 evaluation."
        if passed else "Reject context rankers lacking three-fold OOF recall stability."
    )
    artifact_path = (root / args.artifact).resolve()
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_hash: str | None = None
    if passed:
        fitted = fit_model(selected["spec"], features, labels, site_cell_weights(groups, labels, sensors))
        payload = {
            "schema_version": 1,
            "architecture": "oof_stable_context_scene_ranker_v1",
            "spec": selected["spec"],
            "blend_lambda": selected["blend_lambda"],
            "feature_names": base_names.tolist(),
            "augmented_feature_names": augmented_names,
            "primary_feature": "primary_connected_score",
            "context_base_features": list(CONTEXT_BASE_FEATURES),
            "context_statistics": list(CONTEXT_STATISTICS),
            "fitted": fitted,
            "training_folds": list(INNER_FOLDS),
            "cache_sha256": args.cache_sha256,
            "source_provenance": source_provenance,
            "oof_stability_checks": selected["stability_checks"],
        }
        temporary = artifact_path.with_suffix(artifact_path.suffix + ".tmp")
        joblib.dump(payload, temporary, compress=3)
        os.replace(temporary, artifact_path)
        artifact_hash = sha256(artifact_path)
    report = {
        "schema_version": 1,
        "scope": "three-fold OOF selection on folds 2/3/4; folds 0/1 and paper test not loaded",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows": int(labels.size), "positive": positive_count,
        "folds": list(INNER_FOLDS),
        "model_specs": context_specs(), "blend_lambdas": list(CONTEXT_BLEND_LAMBDAS),
        "stability_contract": {
            "minimum_extra_true_positives": MINIMUM_EXTRA_TRUE_POSITIVES,
            "minimum_positive_recall_folds": MINIMUM_POSITIVE_FOLDS,
            "maximum_fold_ap_regression": MAXIMUM_FOLD_AP_REGRESSION,
        },
        "baseline": baseline, "baseline_by_fold": baseline_by_fold,
        "candidates": candidates, "selected": selected, "decision": decision,
        "artifact": None if artifact_hash is None else {
            "path": args.artifact, "sha256": artifact_hash,
            "bytes": artifact_path.stat().st_size, "tracked": False,
        },
        "provenance": {
            "cache_path": args.cache, "cache_sha256": args.cache_sha256,
            **source_provenance,
            "script_sha256": sha256(Path(__file__).resolve()),
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
            "numpy": np.__version__, "scipy": scipy.__version__,
            "sklearn": sklearn.__version__, "joblib": joblib.__version__,
        },
    }
    output_json = (root / args.output_json).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown((root / args.output_markdown).resolve(), report)
    print(json.dumps({
        "ok": passed, "selected": {"spec": selected["spec"], "blend_lambda": selected["blend_lambda"]},
        "deltas": selected["delta"], "extra_true_positives": selected["extra_true_positives"],
        "stability_checks": selected["stability_checks"],
        "artifact_sha256": artifact_hash, "decision": decision,
    }, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
