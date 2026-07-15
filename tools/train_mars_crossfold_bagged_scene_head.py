#!/usr/bin/env python3
"""Train a crossfold-bagged ExtraTrees scene head on all MARS development folds."""

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
import sklearn

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "EarthRemoteSensingRapidResponse"
for path in (MODEL_ROOT, ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from acquire_mars_metadata import repo_root, sha256  # noqa: E402
from train_mars_context_scene_ranker import augment_site_context  # noqa: E402
from train_mars_oof_scene_ensemble_v2 import (  # noqa: E402
    ap_group_bootstrap,
    fit_model,
)
from train_mars_scene_ranker import (  # noqa: E402
    blend_scores,
    comparison,
    metric_summary,
)
from train_mars_spatial_scene_classifier import (  # noqa: E402
    DEFAULT_FOLD0_CACHE,
    DEFAULT_FOLD0_SHA256,
    DEFAULT_FOLD1_CACHE,
    DEFAULT_FOLD1_SHA256,
    DEFAULT_INNER_CACHE,
    DEFAULT_INNER_SHA256,
    DEFAULT_SCORE_CACHE,
    DEFAULT_SCORE_SHA256,
)

DEFAULT_ARTIFACT = Path(
    "EarthRemoteSensingRapidResponse/artifacts/mars_crossfold_bagged_scene_head.joblib"
)
DEFAULT_JSON = Path("reports/experiments/mars_crossfold_bagged_scene_head.json")
DEFAULT_MARKDOWN = Path("reports/experiments/MARS_CROSSFOLD_BAGGED_SCENE_HEAD.md")
FOLDS = (0, 1, 2, 3, 4)
AGGREGATIONS = ("mean_probability", "mean_logit", "median_probability")
BLENDS = (0.25, 0.5, 0.625, 0.75, 0.875, 1.0)
SPEC: dict[str, Any] = {
    "family": "extra_trees",
    "weighting": "uniform",
    "n_estimators": 400,
    "min_samples_leaf": 5,
    "max_features": 0.5,
}


def nested_training_folds(holdout: int) -> list[tuple[int, ...]]:
    available = [fold for fold in FOLDS if fold != holdout]
    return [tuple(fold for fold in available if fold != omitted) for omitted in available]


def aggregate_predictions(predictions: np.ndarray, mode: str) -> np.ndarray:
    if predictions.ndim != 2 or predictions.shape[0] < 2:
        raise ValueError("Bagged predictions must have shape members by rows")
    if mode == "mean_probability":
        values = predictions.mean(axis=0)
    elif mode == "mean_logit":
        clipped = np.clip(predictions, 1e-6, 1.0 - 1e-6)
        logits = np.log(clipped / (1.0 - clipped))
        values = 1.0 / (1.0 + np.exp(-logits.mean(axis=0)))
    elif mode == "median_probability":
        values = np.median(predictions, axis=0)
    else:
        raise ValueError(f"Unknown bagging aggregation: {mode}")
    if values.ndim != 1 or not np.isfinite(values).all():
        raise RuntimeError("Bagged scene scores are invalid")
    return values.astype(np.float64)


def load_development(
    cache_paths: dict[str, Path], score_path: Path
) -> dict[str, np.ndarray | list[str]]:
    parts = []
    names: np.ndarray | None = None
    for key in ("fold0", "fold1", "inner"):
        with np.load(cache_paths[key], allow_pickle=False) as cache:
            local_names = cache["feature_names"].astype(str)
            if names is None:
                names = local_names
            elif not np.array_equal(names, local_names):
                raise ValueError("Development scene feature schemas differ")
            parts.append(
                {
                    "features": cache["features"].astype(np.float64),
                    "labels": cache["labels"].astype(np.uint8),
                    "sensors": cache["sensors"].astype(np.uint8),
                    "sample_ids": cache["sample_ids"].astype(str),
                    "groups": cache["groups"].astype(str),
                    "folds": cache["folds"].astype(np.uint8),
                }
            )
    assert names is not None
    features = np.concatenate([part["features"] for part in parts])
    labels = np.concatenate([part["labels"] for part in parts])
    sensors = np.concatenate([part["sensors"] for part in parts])
    sample_ids = np.concatenate([part["sample_ids"] for part in parts])
    groups = np.concatenate([part["groups"] for part in parts])
    folds = np.concatenate([part["folds"] for part in parts])
    if set(np.unique(folds).tolist()) != set(FOLDS) or len(set(sample_ids.tolist())) != len(
        sample_ids
    ):
        raise ValueError("Development rows do not cover five unique folds")
    with np.load(score_path, allow_pickle=False) as scores:
        score_parts = []
        for key in ("fold0", "fold1", "inner"):
            local = parts[("fold0", "fold1", "inner").index(key)]
            for field in ("labels", "sensors", "groups"):
                if not np.array_equal(
                    scores[f"{key}_{field}"], local[field]
                ):
                    raise ValueError(f"{key} score cache {field} alignment failed")
            score_parts.append(
                {
                    "primary": scores[f"{key}_primary"].astype(np.float64),
                    "current": scores[f"{key}_new"].astype(np.float64),
                }
            )
    primary = np.concatenate([part["primary"] for part in score_parts])
    current = np.concatenate([part["current"] for part in score_parts])
    primary_index = int(np.flatnonzero(names == "primary_connected_score")[0])
    if not np.allclose(features[:, primary_index], primary, rtol=0.0, atol=1e-7):
        raise ValueError("Primary score differs between feature and score caches")
    augmented, augmented_names = augment_site_context(features, names, groups)
    return {
        "features": augmented,
        "feature_names": names.tolist(),
        "augmented_feature_names": augmented_names,
        "labels": labels,
        "sensors": sensors,
        "sample_ids": sample_ids,
        "groups": groups,
        "folds": folds,
        "primary": primary,
        "current": current,
    }


def oof_member_predictions(values: dict[str, Any]) -> np.ndarray:
    labels = values["labels"]
    folds = values["folds"]
    features = values["features"]
    predictions = np.empty((4, labels.size), dtype=np.float64)
    for holdout in FOLDS:
        held = folds == holdout
        plans = nested_training_folds(holdout)
        for member, training in enumerate(plans):
            fit = np.isin(folds, training)
            model = fit_model(SPEC, features[fit], labels[fit], np.ones(int(fit.sum())))
            predictions[member, held] = model.predict_proba(features[held])[:, 1]
        print(
            json.dumps(
                {
                    "completed_holdout": holdout,
                    "held_rows": int(held.sum()),
                    "member_training_folds": [list(plan) for plan in plans],
                }
            ),
            flush=True,
        )
    if not np.isfinite(predictions).all():
        raise RuntimeError("Nested OOF bagging produced non-finite scores")
    return predictions


def evaluate_candidate(
    values: dict[str, Any], raw: np.ndarray, aggregation: str, blend: float
) -> dict[str, Any]:
    candidate_scores = blend_scores(values["primary"], raw, blend)
    primary = metric_summary(values["labels"], values["primary"], values["sensors"])
    current = metric_summary(values["labels"], values["current"], values["sensors"])
    candidate = metric_summary(values["labels"], candidate_scores, values["sensors"])
    per_fold = {}
    for fold in FOLDS:
        rows = values["folds"] == fold
        per_fold[str(fold)] = {
            "versus_primary": comparison(
                metric_summary(values["labels"][rows], candidate_scores[rows], values["sensors"][rows]),
                metric_summary(values["labels"][rows], values["primary"][rows], values["sensors"][rows]),
            ),
            "versus_current": comparison(
                metric_summary(values["labels"][rows], candidate_scores[rows], values["sensors"][rows]),
                metric_summary(values["labels"][rows], values["current"][rows], values["sensors"][rows]),
            ),
        }
    versus_primary = comparison(candidate, primary)
    versus_current = comparison(candidate, current)
    fold_ap = [
        value["versus_current"]["delta"]["average_precision"]
        for value in per_fold.values()
    ]
    fold_recall = [
        value["versus_current"]["delta"]["recall_at_fpr_0_0713"]
        for value in per_fold.values()
    ]
    stable = (
        versus_primary["delta"]["average_precision"] > 0.0
        and versus_primary["delta"]["recall_at_fpr_0_0713"] > 0.0
        and versus_current["delta"]["average_precision"] > 0.0
        and versus_current["delta"]["recall_at_fpr_0_0713"] >= 0.0
        and min(fold_ap) >= 0.0
        and min(fold_recall) >= -0.002
        and min(versus_current["delta"]["sensor_average_precision"].values()) >= -0.0025
    )
    return {
        "aggregation": aggregation,
        "blend_weight": blend,
        "stable": bool(stable),
        "versus_primary": versus_primary,
        "versus_current": versus_current,
        "per_fold": per_fold,
        "rank": [
            int(stable),
            min(fold_ap),
            versus_current["delta"]["average_precision"],
            versus_current["delta"]["recall_at_fpr_0_0713"],
        ],
    }


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    selected = report["selected"]
    current = selected["versus_current"]["delta"]
    interval = selected["paired_group_bootstrap_ap_delta_vs_current"]
    lines = [
        "# Crossfold-bagged MARS scene head",
        "",
        "Every OOF score averages four ExtraTrees members trained on distinct subsets that exclude the held fold.",
        "",
        f"- Aggregation: `{selected['aggregation']}`",
        f"- Primary/head blend: {selected['blend_weight']:.3f}",
        f"- AP delta vs current head: {current['average_precision']:+.5f}",
        f"- Recall delta vs current head: {current['recall_at_fpr_0_0713']:+.5f}",
        f"- Paired-site AP interval vs current: [{interval['lower']:+.5f}, {interval['upper']:+.5f}]",
        "",
        "| Fold | AP delta vs current | Recall delta vs current |",
        "|---|---:|---:|",
    ]
    for fold, value in selected["per_fold"].items():
        delta = value["versus_current"]["delta"]
        lines.append(
            f"| {fold} | {delta['average_precision']:+.5f} | "
            f"{delta['recall_at_fpr_0_0713']:+.5f} |"
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
    }
    expected = {
        "inner": args.inner_sha256,
        "fold0": args.fold0_sha256,
        "fold1": args.fold1_sha256,
        "score": args.score_sha256,
    }
    for name, digest in expected.items():
        if sha256(paths[name]) != digest:
            raise ValueError(f"Frozen {name} cache hash mismatch")
    values = load_development(
        {name: paths[name] for name in ("inner", "fold0", "fold1")},
        paths["score"],
    )
    members = oof_member_predictions(values)
    candidates = []
    raw_by_aggregation = {}
    for aggregation in AGGREGATIONS:
        raw = aggregate_predictions(members, aggregation)
        raw_by_aggregation[aggregation] = raw
        candidates.extend(
            evaluate_candidate(values, raw, aggregation, blend) for blend in BLENDS
        )
    selected = max(candidates, key=lambda value: tuple(value["rank"]))
    selected_scores = blend_scores(
        values["primary"],
        raw_by_aggregation[selected["aggregation"]],
        float(selected["blend_weight"]),
    )
    selected["paired_group_bootstrap_ap_delta_vs_primary"] = ap_group_bootstrap(
        values["labels"],
        values["primary"],
        selected_scores,
        values["groups"],
        replicates=10_000,
        seed=20261220,
    )
    selected["paired_group_bootstrap_ap_delta_vs_current"] = ap_group_bootstrap(
        values["labels"],
        values["current"],
        selected_scores,
        values["groups"],
        replicates=10_000,
        seed=20261221,
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

    final_models = []
    for omitted in FOLDS:
        fit = values["folds"] != omitted
        final_models.append(
            fit_model(
                SPEC,
                values["features"][fit],
                values["labels"][fit],
                np.ones(int(fit.sum())),
            )
        )
    artifact_path = (root / args.artifact).resolve()
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = artifact_path.with_suffix(artifact_path.suffix + ".tmp")
    joblib.dump(
        {
            "schema_version": 1,
            "kind": "mars_crossfold_bagged_extra_trees_scene_head",
            "spec": SPEC,
            "training": "five members; each omits one complete physical-site fold",
            "aggregation": selected["aggregation"],
            "blend_weight": float(selected["blend_weight"]),
            "operational_scene_threshold": max(thresholds),
            "feature_names": values["feature_names"],
            "augmented_feature_names": values["augmented_feature_names"],
            "primary_feature": "primary_connected_score",
            "models": final_models,
            "cache_sha256": expected,
        },
        temporary,
        compress=3,
    )
    os.replace(temporary, artifact_path)
    report = {
        "schema_version": 1,
        "scope": "five-fold nested OOF development selection; paper cache not loaded",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows": int(values["labels"].size),
        "positive": int(values["labels"].sum()),
        "groups": len(set(values["groups"].tolist())),
        "spec": SPEC,
        "aggregations": list(AGGREGATIONS),
        "blends": list(BLENDS),
        "candidate_summaries": [
            {
                "aggregation": value["aggregation"],
                "blend_weight": value["blend_weight"],
                "stable": value["stable"],
                "ap_delta_vs_primary": value["versus_primary"]["delta"][
                    "average_precision"
                ],
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
            }
            for value in candidates
        ],
        "selected": selected,
        "operational_scene_threshold": max(thresholds),
        "all_promotion_gates_pass": passed,
        "decision": (
            "Freeze the five-member crossfold bagged head for one transparent paper replay."
            if passed
            else "Reject crossfold bagging before paper-cache scoring."
        ),
        "provenance": {
            **{f"{name}_cache_sha256": digest for name, digest in expected.items()},
            "artifact_sha256": sha256(artifact_path),
            "script_sha256": sha256(Path(__file__).resolve()),
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True
            ).strip(),
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
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
                "blend_weight": selected["blend_weight"],
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
