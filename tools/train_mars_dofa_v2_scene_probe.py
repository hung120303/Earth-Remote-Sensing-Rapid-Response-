#!/usr/bin/env python3
"""Cross-fit sensor-aware frozen DOFA-v2 scene probes on MARS folds 3 and 4."""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.random_projection import SparseRandomProjection

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "EarthRemoteSensingRapidResponse", ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from acquire_dofa_v2_base import CHECKPOINT_SHA256  # noqa: E402
from acquire_mars_metadata import repo_root, sha256  # noqa: E402
from extract_mars_dofa_v2_scene_features import (  # noqa: E402
    EMBED_DIM,
    FEATURE_WIDTH,
    MARS_TO_DOFA_MULTIPLIER,
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
from train_mars_oof_scene_ensemble_v2 import ap_group_bootstrap  # noqa: E402
from train_mars_scene_ranker import blend_scores, comparison, metric_summary  # noqa: E402

DEFAULT_DOFA = Path("outputs/mars_dofa_v2_scene_features_folds34.npz")
DEFAULT_JSON = Path("reports/experiments/mars_dofa_v2_scene_probe_folds34.json")
DEFAULT_MARKDOWN = Path("reports/experiments/MARS_DOFA_V2_SCENE_PROBE_FOLDS34.md")
SELECTION_FOLDS = (3, 4)
FEATURE_SETS = ("change_extreme", "change_full", "context_change")
REGULARIZATION = (0.001, 0.01, 0.1)
BLENDS = (0.01, 0.025, 0.05, 0.1, 0.2)
PROJECTION_DIM = 2048
MINIMUM_POOLED_AP_GAIN = 0.001
CONTEXT_END = 2 * EMBED_DIM
ABSOLUTE_END = CONTEXT_END + 4 * EMBED_DIM
STD_END = ABSOLUTE_END + 4 * EMBED_DIM
EXTREME_END = STD_END + 4 * EMBED_DIM


def domain_normalize(
    source: np.ndarray, target: np.ndarray, epsilon: float = 1e-4
) -> tuple[np.ndarray, np.ndarray]:
    source_mean, target_mean = source.mean(0), target.mean(0)
    source_scale = np.maximum(source.std(0), epsilon)
    target_scale = np.maximum(target.std(0), epsilon)
    return (
        ((source - source_mean) / source_scale).astype(np.float32),
        ((target - target_mean) / target_scale).astype(np.float32),
    )


def select_features(
    encoded: np.ndarray, names: np.ndarray, feature_set: str
) -> tuple[np.ndarray, np.ndarray]:
    if encoded.ndim != 2 or encoded.shape[1] != FEATURE_WIDTH:
        raise ValueError("Expected the frozen 10,752-feature DOFA-v2 schema")
    if names.shape != (FEATURE_WIDTH,) or EXTREME_END != FEATURE_WIDTH:
        raise ValueError("DOFA-v2 feature names or offsets differ")
    absolute = encoded[:, CONTEXT_END:ABSOLUTE_END]
    extreme = encoded[:, STD_END:EXTREME_END]
    if feature_set == "change_extreme":
        values = np.concatenate((absolute, extreme), axis=1)
        selected_names = np.concatenate(
            (names[CONTEXT_END:ABSOLUTE_END], names[STD_END:EXTREME_END])
        )
    elif feature_set == "change_full":
        values = encoded[:, CONTEXT_END:]
        selected_names = names[CONTEXT_END:]
    elif feature_set == "context_change":
        values, selected_names = encoded, names
    else:
        raise ValueError(f"Unknown feature set: {feature_set}")
    return values.astype(np.float32), selected_names.astype(str)


def align_features(path: Path, values: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as cache:
        cache_folds = cache["folds"].astype(np.uint8)
        if set(np.unique(cache_folds).tolist()) != set(SELECTION_FOLDS):
            raise ValueError("DOFA-v2 cache must contain only folds 3 and 4")
        if float(cache["mars_to_dofa_multiplier"]) != MARS_TO_DOFA_MULTIPLIER:
            raise ValueError("DOFA-v2 cache radiometric multiplier differs")
        if str(cache["checkpoint_sha256"]) != CHECKPOINT_SHA256:
            raise ValueError("DOFA-v2 cache checkpoint differs")
        ids = cache["sample_ids"].astype(str)
        encoded = cache["features"]
        names = cache["feature_names"].astype(str)
        cached_labels = cache["labels"].astype(np.uint8)
        cached_groups = cache["groups"].astype(str)
        cached_sensors = cache["sensors"].astype(np.uint8)
    lookup = {sample_id: index for index, sample_id in enumerate(ids)}
    if len(lookup) != ids.size:
        raise ValueError("DOFA-v2 cache sample IDs are not unique")
    selected = np.isin(values["folds"], SELECTION_FOLDS)
    target_ids = values["sample_ids"][selected].astype(str)
    if set(target_ids.tolist()) != set(ids.tolist()):
        raise ValueError("DOFA-v2 cache differs from folds 3/4 development rows")
    order = np.asarray([lookup[sample_id] for sample_id in target_ids])
    for expected, observed, label in (
        (values["labels"][selected], cached_labels[order], "labels"),
        (values["groups"][selected].astype(str), cached_groups[order], "groups"),
        (values["sensors"][selected], cached_sensors[order], "sensors"),
    ):
        if not np.array_equal(expected, observed):
            raise ValueError(f"DOFA-v2 cache {label} alignment failed")
    return encoded[order], names


def build_projected_views(
    features: np.ndarray, folds: np.ndarray, seed: int
) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    views = {}
    for holdout in SELECTION_FOLDS:
        fit, held = folds != holdout, folds == holdout
        source, target = domain_normalize(features[fit], features[held])
        projection = SparseRandomProjection(
            n_components=PROJECTION_DIM,
            density="auto",
            dense_output=True,
            random_state=seed,
        )
        source_projected = projection.fit_transform(source).astype(np.float32)
        target_projected = projection.transform(target).astype(np.float32)
        source_projected, target_projected = domain_normalize(
            source_projected, target_projected
        )
        views[holdout] = (fit, held, source_projected, target_projected)
    return views


def crossfit_scores(
    views: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    labels: np.ndarray,
    c_value: float,
) -> np.ndarray:
    scores = np.empty(labels.shape, dtype=np.float64)
    for holdout in SELECTION_FOLDS:
        fit, held, source, target = views[holdout]
        positives = int((labels[fit] == 1).sum())
        negatives = int((labels[fit] == 0).sum())
        positive_weight = float(np.sqrt(negatives / positives))
        weights = np.where(labels[fit] == 1, positive_weight, 1.0)
        model = LogisticRegression(
            C=c_value,
            max_iter=500,
            solver="lbfgs",
            random_state=20260750 + holdout,
        ).fit(source, labels[fit], sample_weight=weights)
        scores[held] = model.predict_proba(target)[:, 1]
        print(
            json.dumps(
                {
                    "C": c_value,
                    "completed_holdout": holdout,
                    "fit_rows": int(fit.sum()),
                    "held_rows": int(held.sum()),
                    "projected_features": source.shape[1],
                }
            ),
            flush=True,
        )
    if not np.isfinite(scores).all():
        raise RuntimeError("Cross-fitted DOFA-v2 scores are not finite")
    return scores


def evaluate_candidate(
    values: dict[str, np.ndarray], raw: np.ndarray, spec: dict[str, Any], blend: float
) -> dict[str, Any]:
    scores = blend_scores(values["current"], raw, blend)
    candidate = metric_summary(values["labels"], scores, values["sensors"])
    current = metric_summary(values["labels"], values["current"], values["sensors"])
    primary = metric_summary(values["labels"], values["primary"], values["sensors"])
    versus_current = comparison(candidate, current)
    versus_primary = comparison(candidate, primary)
    per_fold: dict[str, Any] = {}
    for fold in SELECTION_FOLDS:
        rows = values["folds"] == fold
        local = metric_summary(values["labels"][rows], scores[rows], values["sensors"][rows])
        per_fold[str(fold)] = {
            "versus_current": comparison(
                local,
                metric_summary(
                    values["labels"][rows],
                    values["current"][rows],
                    values["sensors"][rows],
                ),
            ),
            "versus_primary": comparison(
                local,
                metric_summary(
                    values["labels"][rows],
                    values["primary"][rows],
                    values["sensors"][rows],
                ),
            ),
        }
    fold_ap = [
        per_fold[str(fold)]["versus_current"]["delta"]["average_precision"]
        for fold in SELECTION_FOLDS
    ]
    fold_recall = [
        per_fold[str(fold)]["versus_current"]["delta"]["recall_at_fpr_0_0713"]
        for fold in SELECTION_FOLDS
    ]
    sensor_ap = versus_current["delta"]["sensor_average_precision"]
    stable = bool(
        versus_current["delta"]["average_precision"] >= MINIMUM_POOLED_AP_GAIN
        and versus_current["delta"]["recall_at_fpr_0_0713"] >= 0.0
        and min(fold_ap) > 0.0
        and min(fold_recall) >= -0.002
        and min(sensor_ap.values()) >= 0.0
        and versus_primary["delta"]["average_precision"] > 0.0
        and versus_primary["delta"]["recall_at_fpr_0_0713"] > 0.0
    )
    return {
        "spec": spec,
        "blend_weight": blend,
        "stable": stable,
        "versus_current": versus_current,
        "versus_primary": versus_primary,
        "per_fold": per_fold,
        "rank": [
            int(stable),
            min(fold_ap),
            min(sensor_ap.values()),
            versus_current["delta"]["average_precision"],
            versus_current["delta"]["recall_at_fpr_0_0713"],
            -blend,
        ],
    }


def candidate_summary(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "spec": candidate["spec"],
        "blend_weight": candidate["blend_weight"],
        "stable": candidate["stable"],
        "rank": candidate["rank"],
        "delta_vs_current": candidate["versus_current"]["delta"],
        "fold_ap_delta_vs_current": {
            fold: candidate["per_fold"][fold]["versus_current"]["delta"][
                "average_precision"
            ]
            for fold in map(str, SELECTION_FOLDS)
        },
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
        "# DOFA-v2 sensor-aware scene probe — folds 3/4",
        "",
        "Frozen wavelength-conditioned target/reference features were projected with a "
        "fixed sparse random map and scored by a regularized linear probe. Selection is "
        "a two-way physical-site cross-fit on folds 3 and 4 only.",
        "",
        f"- Feature set / C / blend: `{selected['spec']['feature_set']}` / "
        f"{selected['spec']['C']} / {selected['blend_weight']:.3f}",
        f"- AP delta vs current: {delta['average_precision']:+.6f}",
        f"- Recall delta at FPR 0.0713: {delta['recall_at_fpr_0_0713']:+.6f}",
        f"- Paired-site AP 95% interval: [{interval['lower']:+.6f}, {interval['upper']:+.6f}]",
        "",
        "| Fold | AP delta vs current | Recall delta vs current |",
        "|---:|---:|---:|",
    ]
    for fold in SELECTION_FOLDS:
        local = selected["per_fold"][str(fold)]["versus_current"]["delta"]
        lines.append(
            f"| {fold} | {local['average_precision']:+.6f} | "
            f"{local['recall_at_fpr_0_0713']:+.6f} |"
        )
    lines.extend(["", report["decision"]])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dofa", default=DEFAULT_DOFA.as_posix())
    parser.add_argument("--dofa-sha256", required=True)
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
    all_values = load_development(
        {name: paths[name] for name in ("inner", "fold0", "fold1")}, paths["score"]
    )
    encoded, names = align_features(paths["dofa"], all_values)
    selection = np.isin(all_values["folds"], SELECTION_FOLDS)
    values = {
        key: np.asarray(all_values[key])[selection]
        for key in ("labels", "sensors", "sample_ids", "groups", "folds", "primary", "current")
    }

    candidates: list[dict[str, Any]] = []
    raw_by_spec: dict[tuple[str, float], np.ndarray] = {}
    for feature_index, feature_set in enumerate(FEATURE_SETS):
        features, _ = select_features(encoded, names, feature_set)
        views = build_projected_views(
            features, values["folds"], seed=20260760 + feature_index
        )
        del features
        gc.collect()
        for c_value in REGULARIZATION:
            raw = crossfit_scores(views, values["labels"], c_value)
            raw_by_spec[(feature_set, c_value)] = raw
            spec = {"feature_set": feature_set, "C": c_value}
            candidates.extend(
                evaluate_candidate(values, raw, spec, blend) for blend in BLENDS
            )
        del views
        gc.collect()

    selected = max(candidates, key=lambda value: tuple(value["rank"]))
    selected_raw = raw_by_spec[
        (str(selected["spec"]["feature_set"]), float(selected["spec"]["C"]))
    ]
    selected_scores = blend_scores(values["current"], selected_raw, selected["blend_weight"])
    selected["paired_group_bootstrap_ap_delta_vs_current"] = ap_group_bootstrap(
        values["labels"],
        values["current"],
        selected_scores,
        values["groups"],
        replicates=10_000,
        seed=2026073170,
    )
    selected["paired_group_bootstrap_ap_delta_vs_primary"] = ap_group_bootstrap(
        values["labels"],
        values["primary"],
        selected_scores,
        values["groups"],
        replicates=10_000,
        seed=2026073171,
    )
    passed = bool(
        selected["stable"]
        and selected["paired_group_bootstrap_ap_delta_vs_current"]["lower"] > 0.0
    )
    report = {
        "schema_version": 1,
        "scope": "development-only DOFA-v2 sensor-aware scene selection on folds 3/4",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_folds": list(SELECTION_FOLDS),
        "candidate_contract": {
            "feature_sets": list(FEATURE_SETS),
            "projection_dimension": PROJECTION_DIM,
            "regularization": list(REGULARIZATION),
            "blends": list(BLENDS),
            "minimum_pooled_ap_gain": MINIMUM_POOLED_AP_GAIN,
            "candidate_count": len(candidates),
        },
        "candidate_summaries": [candidate_summary(value) for value in candidates],
        "selected": selected,
        "all_promotion_gates_pass": passed,
        "decision": (
            "Freeze the selected DOFA-v2 candidate before one-shot fold-2 extraction."
            if passed
            else "Reject the DOFA-v2 scene candidate before fold-2 extraction."
        ),
        "provenance": {
            **{f"{name}_sha256": digest for name, digest in expected.items()},
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
    print(
        json.dumps(
            {
                "ok": passed,
                "selected": selected["spec"],
                "blend": selected["blend_weight"],
                "ap_delta_vs_current": selected["versus_current"]["delta"]["average_precision"],
                "recall_delta_vs_current": selected["versus_current"]["delta"]["recall_at_fpr_0_0713"],
                "ap_lower_vs_current": selected["paired_group_bootstrap_ap_delta_vs_current"]["lower"],
            },
            indent=2,
        )
    )
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
