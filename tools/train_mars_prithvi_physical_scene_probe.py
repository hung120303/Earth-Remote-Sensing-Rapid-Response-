#!/usr/bin/env python3
"""Evaluate physically scaled Prithvi features on development folds 3 and 4."""

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
from sklearn.linear_model import LogisticRegression

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
from train_mars_oof_scene_ensemble_v2 import ap_group_bootstrap  # noqa: E402
from train_mars_scene_ranker import blend_scores, comparison, metric_summary  # noqa: E402

DEFAULT_PRITHVI = Path(
    "outputs/mars_prithvi_eo_2_tiny_tl_physical_features_folds34.npz"
)
DEFAULT_ARTIFACT = Path(
    "EarthRemoteSensingRapidResponse/artifacts/mars_prithvi_physical_scene_probe.joblib"
)
DEFAULT_JSON = Path("reports/experiments/mars_prithvi_physical_scene_probe_folds34.json")
DEFAULT_MARKDOWN = Path(
    "reports/experiments/MARS_PRITHVI_PHYSICAL_SCENE_PROBE_FOLDS34.md"
)
SELECTION_FOLDS = (3, 4)
FEATURE_SETS = ("cls", "temporal_change", "all")
REGULARIZATION = (0.001, 0.01, 0.1)
BLENDS = (0.025, 0.05, 0.1, 0.2, 0.3)
CLS_WIDTH = 4 * 192
TEMPORAL_CHANGE_OFFSET = CLS_WIDTH + 2 * 3 * 192
MINIMUM_POOLED_AP_GAIN = 0.001


def domain_normalize(
    source: np.ndarray, target: np.ndarray, epsilon: float = 1e-4
) -> tuple[np.ndarray, np.ndarray]:
    """Use each domain's unlabeled moments without leaking held labels."""
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_scale = np.maximum(source.std(axis=0), epsilon)
    target_scale = np.maximum(target.std(axis=0), epsilon)
    return (
        ((source - source_mean) / source_scale).astype(np.float32),
        ((target - target_mean) / target_scale).astype(np.float32),
    )


def select_features(
    encoded: np.ndarray, names: np.ndarray, feature_set: str
) -> tuple[np.ndarray, np.ndarray]:
    if encoded.ndim != 2 or encoded.shape[1] != 3072 or names.shape != (3072,):
        raise ValueError("Expected the frozen 3,072-feature Prithvi schema")
    if feature_set == "cls":
        values, selected_names = encoded[:, :CLS_WIDTH], names[:CLS_WIDTH]
    elif feature_set == "temporal_change":
        values = encoded[:, TEMPORAL_CHANGE_OFFSET:]
        selected_names = names[TEMPORAL_CHANGE_OFFSET:]
    elif feature_set == "all":
        values, selected_names = encoded, names
    else:
        raise ValueError(f"Unknown feature set: {feature_set}")
    return values.astype(np.float32), selected_names.astype(str)


def align_features(path: Path, values: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as cache:
        cache_folds = cache["folds"].astype(np.uint8)
        if set(np.unique(cache_folds).tolist()) != set(SELECTION_FOLDS):
            raise ValueError("Physical Prithvi cache must contain only folds 3 and 4")
        if float(cache["mars_to_prithvi_multiplier"]) != 5_000.0:
            raise ValueError("Physical Prithvi cache has the wrong radiometric multiplier")
        ids = cache["sample_ids"].astype(str)
        encoded = cache["features"].astype(np.float32)
        names = cache["feature_names"].astype(str)
        cached_labels = cache["labels"].astype(np.uint8)
        cached_groups = cache["groups"].astype(str)
        cached_sensors = cache["sensors"].astype(np.uint8)
    lookup = {sample_id: index for index, sample_id in enumerate(ids)}
    if len(lookup) != ids.size:
        raise ValueError("Physical Prithvi cache sample IDs are not unique")
    selected = np.isin(values["folds"], SELECTION_FOLDS)
    target_ids = values["sample_ids"][selected].astype(str)
    if set(target_ids.tolist()) != set(ids.tolist()):
        raise ValueError("Physical Prithvi cache differs from folds 3/4 development rows")
    order = np.asarray([lookup[sample_id] for sample_id in target_ids])
    for expected, observed, label in (
        (values["labels"][selected], cached_labels[order], "labels"),
        (values["groups"][selected].astype(str), cached_groups[order], "groups"),
        (values["sensors"][selected], cached_sensors[order], "sensors"),
    ):
        if not np.array_equal(expected, observed):
            raise ValueError(f"Physical Prithvi cache {label} alignment failed")
    return encoded[order], names


def crossfit_scores(
    features: np.ndarray,
    labels: np.ndarray,
    folds: np.ndarray,
    c_value: float,
) -> np.ndarray:
    scores = np.empty(labels.shape, dtype=np.float64)
    for holdout in SELECTION_FOLDS:
        fit, held = folds != holdout, folds == holdout
        source, target = domain_normalize(features[fit], features[held])
        positives = int((labels[fit] == 1).sum())
        negatives = int((labels[fit] == 0).sum())
        positive_weight = float(np.sqrt(negatives / positives))
        weights = np.where(labels[fit] == 1, positive_weight, 1.0)
        model = LogisticRegression(
            C=c_value,
            max_iter=500,
            solver="lbfgs",
            random_state=20260731 + holdout,
        ).fit(source, labels[fit], sample_weight=weights)
        scores[held] = model.predict_proba(target)[:, 1]
        print(
            json.dumps(
                {
                    "C": c_value,
                    "completed_holdout": holdout,
                    "fit_rows": int(fit.sum()),
                    "held_rows": int(held.sum()),
                    "features": features.shape[1],
                }
            ),
            flush=True,
        )
    if not np.isfinite(scores).all():
        raise RuntimeError("Cross-fitted physical Prithvi scores are not finite")
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
        "# Physically scaled Prithvi scene probe — folds 3/4",
        "",
        "MARS raw reflectance DN were restored with the correct x5,000 conversion before "
        "the pinned Prithvi normalization. Selection is a two-way physical-site cross-fit "
        "on development folds 3 and 4; no fold-2, folds-0/1, or paper-test outcome was used.",
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
    parser.add_argument("--prithvi", default=DEFAULT_PRITHVI.as_posix())
    parser.add_argument("--prithvi-sha256", required=True)
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
        "prithvi": (root / args.prithvi).resolve(),
        "inner": (root / args.inner_cache).resolve(),
        "fold0": (root / args.fold0_cache).resolve(),
        "fold1": (root / args.fold1_cache).resolve(),
        "score": (root / args.score_cache).resolve(),
    }
    expected = {
        "prithvi": args.prithvi_sha256,
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
    encoded, names = align_features(paths["prithvi"], all_values)
    selection = np.isin(all_values["folds"], SELECTION_FOLDS)
    values = {
        key: np.asarray(all_values[key])[selection]
        for key in ("labels", "sensors", "sample_ids", "groups", "folds", "primary", "current")
    }

    candidates: list[dict[str, Any]] = []
    raw_by_spec: dict[tuple[str, float], np.ndarray] = {}
    feature_names_by_set: dict[str, np.ndarray] = {}
    for feature_set in FEATURE_SETS:
        features, selected_names = select_features(encoded, names, feature_set)
        feature_names_by_set[feature_set] = selected_names
        for c_value in REGULARIZATION:
            raw = crossfit_scores(features, values["labels"], values["folds"], c_value)
            raw_by_spec[(feature_set, c_value)] = raw
            spec = {"feature_set": feature_set, "C": c_value}
            candidates.extend(
                evaluate_candidate(values, raw, spec, blend) for blend in BLENDS
            )

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
        seed=2026073134,
    )
    selected["paired_group_bootstrap_ap_delta_vs_primary"] = ap_group_bootstrap(
        values["labels"],
        values["primary"],
        selected_scores,
        values["groups"],
        replicates=10_000,
        seed=2026073135,
    )
    passed = bool(
        selected["stable"]
        and selected["paired_group_bootstrap_ap_delta_vs_current"]["lower"] > 0.0
    )

    artifact_path = (root / args.artifact).resolve()
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = artifact_path.with_suffix(artifact_path.suffix + ".tmp")
    joblib.dump(
        {
            "schema_version": 1,
            "kind": "mars_prithvi_physical_scene_probe_selection",
            "selection_folds": list(SELECTION_FOLDS),
            "spec": selected["spec"],
            "blend_weight": selected["blend_weight"],
            "feature_names": feature_names_by_set[str(selected["spec"]["feature_set"])],
            "radiometric_contract": "MARS normalized reflectance x5000 restores raw DN",
            "prithvi_cache_sha256": args.prithvi_sha256,
        },
        temporary,
        compress=3,
    )
    os.replace(temporary, artifact_path)
    report = {
        "schema_version": 1,
        "scope": "development-only physical-radiometry Prithvi selection on folds 3/4",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_folds": list(SELECTION_FOLDS),
        "candidate_contract": {
            "feature_sets": list(FEATURE_SETS),
            "regularization": list(REGULARIZATION),
            "blends": list(BLENDS),
            "minimum_pooled_ap_gain": MINIMUM_POOLED_AP_GAIN,
            "candidate_count": len(candidates),
        },
        "selected": selected,
        "all_promotion_gates_pass": passed,
        "decision": (
            "Promote the fixed physical-radiometry Prithvi candidate to one-shot fold-2 confirmation."
            if passed
            else "Reject the physical-radiometry Prithvi candidate before fold-2 extraction."
        ),
        "provenance": {
            **{f"{name}_sha256": digest for name, digest in expected.items()},
            "artifact_sha256": sha256(artifact_path),
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
