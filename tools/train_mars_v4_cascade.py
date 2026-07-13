#!/usr/bin/env python3
"""Cross-fit an ERSRR v4 physics verifier over frozen MARS/v3 scene scores."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import scipy
import sklearn
from scipy import ndimage
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "EarthRemoteSensingRapidResponse"
if str(MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_ROOT))

from mars_s2l_adapter import MARS_BANDS, iter_manifest, load_sample  # noqa: E402

from acquire_mars_metadata import DEFAULT_OUTPUT, repo_root, sha256  # noqa: E402
from analyze_mars_v3_strict_posthoc import (  # noqa: E402
    aligned,
    load_cache,
    metadata_winds,
    reference_interval_days,
)
from build_mars_v3_strict_cohort import V3_STRICT_SAMPLES  # noqa: E402
from evaluate_released_marss2l import DEFAULT_METADATA_CSV  # noqa: E402

DEFAULT_CAMPAIGN = Path("reports/experiments/mars_v3_strict_campaign.json")
DEFAULT_FEATURE_CACHE = DEFAULT_OUTPUT / "publication_v4_strict_development_features.npz"
DEFAULT_ARTIFACT = Path("EarthRemoteSensingRapidResponse/artifacts/mars_v4_cascade.joblib")
DEFAULT_JSON = Path("reports/experiments/mars_v4_nested_development.json")
DEFAULT_MARKDOWN = Path("reports/experiments/MARS_V4_NESTED_DEVELOPMENT.md")
OUTER_FOLDS = 5
INNER_FOLDS = 4
RANDOM_SEED = 20_260_714
BOOTSTRAP_REPLICATES = 2_000
TARGET_FPRS = (0.05, 0.095)


def safe_path(root: Path, value: str | Path) -> Path:
    result = (root / value).resolve()
    if result != root and root not in result.parents:
        raise ValueError("Path must resolve beneath the repository root")
    return result


def tracked_dirty(root: Path) -> bool:
    status = subprocess.check_output(
        [
            "git",
            "-c",
            "core.autocrlf=true",
            "status",
            "--porcelain",
            "--untracked-files=no",
        ],
        cwd=root,
        text=True,
    )
    return bool(status.strip())


def largest_component_pixels(mask: np.ndarray) -> int:
    labels, count = ndimage.label(
        np.asarray(mask, dtype=bool), structure=np.ones((3, 3), dtype=np.uint8)
    )
    if count == 0:
        return 0
    return int(np.max(np.bincount(labels.ravel())[1:]))


def physics_features(
    mbmp: np.ndarray,
    target: np.ndarray,
    reference: np.ndarray,
    observable: np.ndarray,
    *,
    cloud_fraction: float,
    wind_speed_m_s: float,
    reference_interval_days_value: float,
) -> tuple[list[str], np.ndarray]:
    values = np.asarray(mbmp, dtype=np.float64)
    valid = np.asarray(observable, dtype=bool)
    target_values = np.asarray(target, dtype=np.float64)
    reference_values = np.asarray(reference, dtype=np.float64)
    if values.shape != valid.shape or target_values.shape != reference_values.shape:
        raise ValueError("Physics feature arrays must be aligned")
    if target_values.shape != (len(MARS_BANDS), *values.shape) or not np.any(valid):
        raise ValueError("Physics features require six-band observable imagery")
    observed = values[valid]
    names = [
        "cloud_fraction",
        "observable_fraction",
        "wind_speed_m_s",
        "reference_interval_days",
        "mbmp_p01",
        "mbmp_p05",
        "mbmp_median",
        "mbmp_p95",
        "mbmp_p99",
        "mbmp_mad",
    ]
    features = [
        float(cloud_fraction),
        float(np.mean(valid)),
        float(wind_speed_m_s),
        float(reference_interval_days_value),
        *[float(np.quantile(observed, value)) for value in (0.01, 0.05, 0.5, 0.95, 0.99)],
        float(np.median(np.abs(observed - np.median(observed)))),
    ]
    for threshold in (0.99, 0.98, 0.97, 0.95):
        candidate = (values <= threshold) & valid
        suffix = str(threshold).replace("0.", "")
        names.extend(
            [f"mbmp_fraction_le_{suffix}", f"mbmp_largest_component_le_{suffix}"]
        )
        features.extend(
            [float(np.mean(candidate[valid])), float(largest_component_pixels(candidate))]
        )
    for index, band in enumerate(MARS_BANDS):
        current = target_values[index][valid]
        background = reference_values[index][valid]
        names.extend(
            [
                f"target_{band}_median",
                f"reference_{band}_median",
                f"absolute_{band}_median_change",
                f"absolute_{band}_difference_median",
            ]
        )
        features.extend(
            [
                float(np.median(current)),
                float(np.median(background)),
                float(abs(np.median(current) - np.median(background))),
                float(np.median(np.abs(current - background))),
            ]
        )
    result = np.asarray(features, dtype=np.float64)
    if result.shape != (len(names),) or not np.all(np.isfinite(result)):
        raise ValueError("Physics feature vector is invalid")
    return names, result


def write_feature_cache(
    path: Path,
    *,
    sample_ids: np.ndarray,
    feature_names: Sequence[str],
    features: np.ndarray,
    manifest_sha256: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as destination:
        np.savez_compressed(
            destination,
            sample_ids=np.asarray(sample_ids),
            feature_names=np.asarray(feature_names),
            features=np.asarray(features, dtype=np.float64),
            manifest_sha256=np.asarray([manifest_sha256]),
        )
    os.replace(temporary, path)


def build_or_load_physics_cache(
    metadata_dir: Path,
    manifest_path: Path,
    records_by_id: dict[str, dict[str, Any]],
    sample_ids: np.ndarray,
    winds: dict[str, tuple[float, float]],
    cache_path: Path,
    *,
    overwrite: bool,
) -> tuple[list[str], np.ndarray]:
    identity = sha256(manifest_path)
    if cache_path.is_file() and not overwrite:
        with np.load(cache_path, allow_pickle=False) as source:
            if str(source["manifest_sha256"][0]) != identity:
                raise ValueError("V4 physics cache manifest identity mismatch")
            if not np.array_equal(source["sample_ids"], sample_ids):
                raise ValueError("V4 physics cache sample order mismatch")
            return [str(value) for value in source["feature_names"]], source["features"].copy()

    feature_names: list[str] | None = None
    rows = []
    for index, identifier_value in enumerate(sample_ids, start=1):
        identifier = str(identifier_value)
        record = records_by_id[identifier]
        sample = load_sample(metadata_dir, record, require_enhancement=False)
        wind = winds[identifier]
        names, values = physics_features(
            sample.mbmp_release_compatible,
            sample.target,
            sample.reference,
            sample.observable_mask,
            cloud_fraction=float(record["cloud_fraction"]),
            wind_speed_m_s=float(np.hypot(*wind)),
            reference_interval_days_value=reference_interval_days(record),
        )
        if feature_names is None:
            feature_names = names
        elif feature_names != names:
            raise ValueError("Physics feature schema changed between scenes")
        rows.append(values)
        if index % 500 == 0:
            print(f"Extracted label-blind physics features for {index}/{len(sample_ids)} scenes")
    assert feature_names is not None
    matrix = np.stack(rows)
    write_feature_cache(
        cache_path,
        sample_ids=sample_ids,
        feature_names=feature_names,
        features=matrix,
        manifest_sha256=identity,
    )
    return feature_names, matrix


def candidate_specs(feature_sets: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    values = []
    for name in ("released_only", "released_plus_v3", "released_plus_physics", "all"):
        values.append({"name": f"{name}:logistic", "features": name, "model": "logistic"})
    for name in ("released_plus_physics", "all"):
        values.append({"name": f"{name}:hist_gradient_boosting", "features": name, "model": "hist_gradient_boosting"})
    for item in values:
        if item["features"] not in feature_sets:
            raise ValueError(f"Unknown feature set: {item['features']}")
    return values


def estimator(kind: str, seed: int) -> Any:
    if kind == "logistic":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(
                C=1.0,
                class_weight="balanced",
                max_iter=2_000,
                random_state=seed,
            ),
        )
    if kind == "hist_gradient_boosting":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_iter=120,
                max_leaf_nodes=7,
                min_samples_leaf=30,
                l2_regularization=2.0,
                class_weight="balanced",
                random_state=seed,
            ),
        )
    raise ValueError(f"Unknown estimator kind: {kind}")


def balanced_group_splits(
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    folds: int,
    seed: int,
    trials: int = 500,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Balance rows and both labels while keeping every 25 km group intact."""
    truth = np.asarray(labels, dtype=np.uint8)
    group_values = np.asarray(groups)
    unique, inverse = np.unique(group_values, return_inverse=True)
    if truth.shape != group_values.shape or unique.size < folds:
        raise ValueError("Balanced group splitting requires aligned labels and enough groups")
    stats = np.zeros((unique.size, 4), dtype=np.float64)
    for index in range(unique.size):
        local = truth[inverse == index]
        stats[index] = [local.size, np.count_nonzero(local == 1), np.count_nonzero(local == 0), 1]
    targets = np.sum(stats, axis=0) / folds
    if targets[1] < 1.0 or targets[2] < 1.0:
        raise ValueError("Every fold requires expected positive and negative support")
    weights = np.asarray([1.0, 5.0, 1.0, 0.25])

    def objective(counts: np.ndarray) -> float:
        relative = (counts - targets[None, :]) / np.maximum(targets[None, :], 1.0)
        value = float(np.sum(weights[None, :] * np.square(relative)))
        value += 100.0 * float(np.count_nonzero(counts[:, 1] == 0))
        value += 100.0 * float(np.count_nonzero(counts[:, 2] == 0))
        return value

    rng = np.random.default_rng(seed)
    best: tuple[float, tuple[int, ...], np.ndarray] | None = None
    difficulty = np.max(stats[:, :3] / np.maximum(targets[None, :3], 1.0), axis=1)
    for _ in range(trials):
        order = np.argsort(-(difficulty + 0.05 * rng.random(unique.size)), kind="stable")
        counts = np.zeros((folds, 4), dtype=np.float64)
        assignment = np.full(unique.size, -1, dtype=np.int64)
        tie_order = rng.permutation(folds)
        for group_index in order:
            choices = []
            for fold in tie_order:
                candidate = counts.copy()
                candidate[fold] += stats[group_index]
                # With fixed assigned totals, minimizing squared load balances the folds.
                load = candidate / np.maximum(targets[None, :], 1.0)
                choices.append((float(np.sum(weights[None, :] * np.square(load))), int(fold)))
            _, selected = min(choices)
            assignment[group_index] = selected
            counts[selected] += stats[group_index]
        score = objective(counts)
        candidate_key = (score, tuple(int(value) for value in assignment), counts.copy())
        if best is None or candidate_key[:2] < best[:2]:
            best = candidate_key
    assert best is not None
    assignment = np.asarray(best[1], dtype=np.int64)
    result = []
    for fold in range(folds):
        validation = np.flatnonzero(assignment[inverse] == fold)
        training = np.flatnonzero(assignment[inverse] != fold)
        if not validation.size or np.unique(truth[validation]).size != 2:
            raise ValueError("Balanced group splitter produced an invalid fold")
        result.append((training, validation))
    return result


def spec_crossfit(
    spec: dict[str, Any],
    feature_sets: dict[str, np.ndarray],
    labels: np.ndarray,
    groups: np.ndarray,
    splits: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    seed: int,
) -> np.ndarray:
    values = feature_sets[spec["features"]]
    predictions = np.full(labels.shape, np.nan, dtype=np.float64)
    for fold, (train, validation) in enumerate(splits, start=1):
        model = estimator(spec["model"], seed + fold)
        model.fit(values[train], labels[train])
        predictions[validation] = model.predict_proba(values[validation])[:, 1]
    if not np.all(np.isfinite(predictions)):
        raise ValueError("Candidate cross-fitting did not predict every row")
    return predictions


def choose_threshold_at_fpr(
    labels: np.ndarray, scores: np.ndarray, target_fpr: float
) -> dict[str, float | int]:
    truth = np.asarray(labels, dtype=np.uint8)
    values = np.asarray(scores, dtype=np.float64)
    if truth.shape != values.shape or not 0.0 < target_fpr < 1.0:
        raise ValueError("Threshold selection inputs are invalid")
    candidates = np.concatenate(([np.inf], np.unique(values)[::-1]))
    best: tuple[float, float, float, float] | None = None
    result: dict[str, float | int] | None = None
    positives = truth == 1
    negatives = ~positives
    for threshold in candidates:
        decision = values >= threshold
        recall = float(np.mean(decision[positives]))
        fpr = float(np.mean(decision[negatives]))
        if fpr > target_fpr + 1e-12:
            continue
        precision = float(np.mean(truth[decision])) if np.any(decision) else 1.0
        ordering = (recall, precision, -fpr, float(threshold))
        if best is None or ordering > best:
            best = ordering
            result = {
                "threshold": float(threshold),
                "training_recall": recall,
                "training_fpr": fpr,
                "training_precision": precision,
                "target_fpr": float(target_fpr),
            }
    if result is None:
        raise AssertionError("At least the no-positive threshold must be feasible")
    return result


def metrics(
    labels: np.ndarray, scores: np.ndarray, decisions: np.ndarray
) -> dict[str, float | int]:
    truth = np.asarray(labels, dtype=np.uint8)
    values = np.asarray(scores, dtype=np.float64)
    predicted = np.asarray(decisions, dtype=bool)
    positive = truth == 1
    negative = ~positive
    tp = int(np.count_nonzero(predicted & positive))
    fp = int(np.count_nonzero(predicted & negative))
    fn = int(np.count_nonzero(~predicted & positive))
    tn = int(np.count_nonzero(~predicted & negative))
    return {
        "scenes": int(truth.size),
        "positives": int(np.count_nonzero(positive)),
        "negatives": int(np.count_nonzero(negative)),
        "average_precision": float(average_precision_score(truth, values)),
        "auroc": float(roc_auc_score(truth, values)),
        "recall": tp / max(tp + fn, 1),
        "false_positive_rate": fp / max(fp + tn, 1),
        "precision": tp / max(tp + fp, 1),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def nested_crossfit(
    specs: list[dict[str, Any]],
    feature_sets: dict[str, np.ndarray],
    labels: np.ndarray,
    groups: np.ndarray,
) -> dict[str, Any]:
    oof_scores = np.full(labels.shape, np.nan, dtype=np.float64)
    oof_decisions = {
        target: np.zeros(labels.shape, dtype=bool) for target in TARGET_FPRS
    }
    folds = []
    outer_splits = balanced_group_splits(
        labels, groups, folds=OUTER_FOLDS, seed=RANDOM_SEED
    )
    for outer_fold, (train, test) in enumerate(outer_splits, start=1):
        inner_results = []
        inner_predictions: dict[str, np.ndarray] = {}
        inner_splits = balanced_group_splits(
            labels[train],
            groups[train],
            folds=INNER_FOLDS,
            seed=RANDOM_SEED + outer_fold * 100,
        )
        for spec_index, spec in enumerate(specs):
            predictions = spec_crossfit(
                spec,
                {name: values[train] for name, values in feature_sets.items()},
                labels[train],
                groups[train],
                inner_splits,
                seed=RANDOM_SEED + outer_fold * 100 + spec_index,
            )
            inner_predictions[spec["name"]] = predictions
            inner_results.append(
                {
                    **spec,
                    "average_precision": float(average_precision_score(labels[train], predictions)),
                    "auroc": float(roc_auc_score(labels[train], predictions)),
                    "feature_count": int(feature_sets[spec["features"]].shape[1]),
                }
            )
        selected = max(
            inner_results,
            key=lambda item: (
                item["average_precision"],
                item["auroc"],
                -item["feature_count"],
                item["name"],
            ),
        )
        spec = next(item for item in specs if item["name"] == selected["name"])
        model = estimator(spec["model"], RANDOM_SEED + outer_fold)
        model.fit(feature_sets[spec["features"]][train], labels[train])
        fold_scores = model.predict_proba(feature_sets[spec["features"]][test])[:, 1]
        oof_scores[test] = fold_scores
        thresholds = {}
        for target in TARGET_FPRS:
            selection = choose_threshold_at_fpr(
                labels[train], inner_predictions[spec["name"]], target
            )
            oof_decisions[target][test] = fold_scores >= float(selection["threshold"])
            thresholds[str(target)] = selection
        folds.append(
            {
                "outer_fold": outer_fold,
                "training_scenes": int(train.size),
                "held_out_scenes": int(test.size),
                "training_groups": int(np.unique(groups[train]).size),
                "held_out_groups": int(np.unique(groups[test]).size),
                "held_out_positives": int(np.count_nonzero(labels[test] == 1)),
                "selected_candidate": selected,
                "thresholds_selected_from_inner_oof": thresholds,
                "held_out_ranking": {
                    "average_precision": float(average_precision_score(labels[test], fold_scores)),
                    "auroc": float(roc_auc_score(labels[test], fold_scores)),
                },
            }
        )
    if not np.all(np.isfinite(oof_scores)):
        raise ValueError("Nested cross-fitting did not predict every scene")
    return {"scores": oof_scores, "decisions": oof_decisions, "folds": folds}


def percentile_interval(values: Sequence[float]) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    return [float(np.quantile(array, 0.025)), float(np.quantile(array, 0.975))]


def group_bootstrap(
    labels: np.ndarray,
    groups: np.ndarray,
    baseline_scores: np.ndarray,
    baseline_decisions: np.ndarray,
    candidate_scores: np.ndarray,
    candidate_decisions: np.ndarray,
) -> dict[str, Any]:
    unique = np.unique(groups)
    indices = {group: np.flatnonzero(groups == group) for group in unique}
    rng = np.random.default_rng(RANDOM_SEED)
    deltas: dict[str, list[float]] = {
        "average_precision": [],
        "auroc": [],
        "recall": [],
        "false_positive_rate": [],
    }
    for _ in range(BOOTSTRAP_REPLICATES):
        sampled = rng.choice(unique, size=unique.size, replace=True)
        selected = np.concatenate([indices[group] for group in sampled])
        truth = labels[selected]
        if np.unique(truth).size < 2:
            continue
        base = metrics(truth, baseline_scores[selected], baseline_decisions[selected])
        candidate = metrics(truth, candidate_scores[selected], candidate_decisions[selected])
        for name in deltas:
            deltas[name].append(float(candidate[name]) - float(base[name]))
    return {
        "replicates_requested": BOOTSTRAP_REPLICATES,
        "replicates_valid": len(deltas["average_precision"]),
        "random_seed": RANDOM_SEED,
        "candidate_minus_released_mars_s2l": {
            name: {
                "mean": float(np.mean(values)),
                "95ci": percentile_interval(values),
            }
            for name, values in deltas.items()
        },
    }


def artifact_record(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "tracked": False,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    baseline = report["released_mars_s2l"]
    ranking = report["nested_crossfit"]["ranking"]
    op5 = report["nested_crossfit"]["operating_points"]["0.05"]
    op95 = report["nested_crossfit"]["operating_points"]["0.095"]
    selection = report["final_development_fit"]["selected_candidate"]
    lines = [
        "# ERSRR v4 nested development experiment",
        "",
        "Development result only. The former v3 strict cohort was already opened for v3 evaluation and is now used with group-nested cross-fitting; it is not an untouched v4 test set.",
        "",
        f"- Scenes/groups: {report['cohort']['scenes']:,} / {report['cohort']['groups']:,}",
        f"- Plume/no-plume: {report['cohort']['positives']:,} / {report['cohort']['negatives']:,}",
        f"- Released MARS-S2L: AP {baseline['average_precision']:.3f}, AUROC {baseline['auroc']:.3f}, recall {baseline['recall']:.3f}, FPR {baseline['false_positive_rate']:.3f}",
        f"- Nested v4 ranking: AP {ranking['average_precision']:.3f}, AUROC {ranking['auroc']:.3f}",
        f"- Nested v4 at 5% FPR target: recall {op5['recall']:.3f}, observed FPR {op5['false_positive_rate']:.3f}, precision {op5['precision']:.3f}",
        f"- Nested v4 at 9.5% FPR target: recall {op95['recall']:.3f}, observed FPR {op95['false_positive_rate']:.3f}, precision {op95['precision']:.3f}",
        f"- Final development fit: `{selection['name']}` with {selection['feature_count']} features",
        "",
        "## Outer-fold architecture selections",
        "",
        "| Fold | Candidate selected on inner OOF | Inner AP | Held-out AP | Held-out AUROC | Held-out positives |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for fold in report["nested_crossfit"]["folds"]:
        selected = fold["selected_candidate"]
        held = fold["held_out_ranking"]
        lines.append(
            f"| {fold['outer_fold']} | `{selected['name']}` | {selected['average_precision']:.3f} | "
            f"{held['average_precision']:.3f} | {held['auroc']:.3f} | {fold['held_out_positives']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            report["decision"],
            "",
            "The final serialized verifier and feature cache remain ignored because they are derived artifacts. A new prediction-blind, same-sensor, spatially and temporally novel plume/no-plume cohort is required for any v4 paper claim.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-dir", default=DEFAULT_OUTPUT.as_posix())
    parser.add_argument("--metadata-csv", default=DEFAULT_METADATA_CSV.as_posix())
    parser.add_argument("--campaign", default=DEFAULT_CAMPAIGN.as_posix())
    parser.add_argument("--feature-cache", default=DEFAULT_FEATURE_CACHE.as_posix())
    parser.add_argument("--artifact", default=DEFAULT_ARTIFACT.as_posix())
    parser.add_argument("--output-json", default=DEFAULT_JSON.as_posix())
    parser.add_argument("--output-markdown", default=DEFAULT_MARKDOWN.as_posix())
    parser.add_argument("--overwrite-features", action="store_true")
    args = parser.parse_args()

    root = repo_root()
    metadata_dir = safe_path(root, args.metadata_dir)
    metadata_csv = safe_path(root, args.metadata_csv)
    campaign_path = safe_path(root, args.campaign)
    feature_cache_path = safe_path(root, args.feature_cache)
    artifact_path = safe_path(root, args.artifact)
    output_json = safe_path(root, args.output_json)
    output_markdown = safe_path(root, args.output_markdown)
    campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
    if campaign.get("scope") != "frozen_v3_five_seed_full_strict_campaign":
        raise ValueError("Expected the frozen v3 strict campaign")

    manifest_path = metadata_dir / V3_STRICT_SAMPLES
    if sha256(manifest_path) != campaign["cohort"]["strict_manifest_sha256"]:
        raise ValueError("Strict manifest identity differs from the frozen campaign")
    records = list(iter_manifest(manifest_path))
    by_id = {str(record["sample_id"]): record for record in records}

    baseline_input = campaign["inputs"]["released_baseline_report"]["scene_cache"]
    baseline = load_cache(
        safe_path(root, baseline_input["path"]), baseline_input["sha256"]
    )
    sample_ids = baseline["sample_ids"]
    labels = baseline["labels"].astype(np.uint8)
    groups = baseline["groups"]
    baseline_scores = baseline["scores"].astype(np.float64)
    baseline_decisions = baseline["predictions"].astype(bool)
    if set(str(value) for value in sample_ids) != set(by_id):
        raise ValueError("Baseline cache and strict manifest sample ids differ")

    seed_scores = []
    seed_predictions = []
    seeds = []
    for item in campaign["inputs"]["v3_reports"]:
        cache_input = item["scene_cache"]
        cache = aligned(
            load_cache(safe_path(root, cache_input["path"]), cache_input["sha256"]),
            sample_ids,
        )
        if not np.array_equal(cache["labels"], labels) or not np.array_equal(cache["groups"], groups):
            raise ValueError("Candidate and baseline cache targets differ")
        seed_scores.append(cache["primary_scores"].astype(np.float64))
        seed_predictions.append(cache["primary_predictions"].astype(np.float64))
        seeds.append(int(cache["seed"][0]))
    if tuple(seeds) != (101, 202, 303, 404, 505):
        raise ValueError("Expected the five frozen ERSRR seeds")
    score_matrix = np.stack(seed_scores)
    decision_matrix = np.stack(seed_predictions)

    winds = metadata_winds(metadata_csv)
    required_winds = {str(value): winds[str(value)] for value in sample_ids}
    physics_names, physics = build_or_load_physics_cache(
        metadata_dir,
        manifest_path,
        by_id,
        sample_ids,
        required_winds,
        feature_cache_path,
        overwrite=args.overwrite_features,
    )
    v3 = np.column_stack(
        [
            np.mean(score_matrix, axis=0),
            np.std(score_matrix, axis=0),
            np.min(score_matrix, axis=0),
            np.max(score_matrix, axis=0),
            np.mean(decision_matrix, axis=0),
        ]
    )
    v3_names = [
        "v3_score_mean",
        "v3_score_standard_deviation",
        "v3_score_minimum",
        "v3_score_maximum",
        "v3_seed_hit_fraction",
    ]
    released = baseline_scores[:, None]
    feature_sets = {
        "released_only": released,
        "released_plus_v3": np.column_stack([released, v3]),
        "released_plus_physics": np.column_stack([released, physics]),
        "all": np.column_stack([released, v3, physics]),
    }
    feature_names = {
        "released_only": ["released_mars_s2l_score"],
        "released_plus_v3": ["released_mars_s2l_score", *v3_names],
        "released_plus_physics": ["released_mars_s2l_score", *physics_names],
        "all": ["released_mars_s2l_score", *v3_names, *physics_names],
    }
    specs = candidate_specs(feature_sets)
    nested = nested_crossfit(specs, feature_sets, labels, groups)
    baseline_metrics = metrics(labels, baseline_scores, baseline_decisions)
    nested_ranking = {
        "average_precision": float(average_precision_score(labels, nested["scores"])),
        "auroc": float(roc_auc_score(labels, nested["scores"])),
    }
    operating = {
        str(target): metrics(labels, nested["scores"], nested["decisions"][target])
        for target in TARGET_FPRS
    }

    full_cv = []
    full_predictions: dict[str, np.ndarray] = {}
    full_splits = balanced_group_splits(
        labels, groups, folds=OUTER_FOLDS, seed=RANDOM_SEED + 10_000
    )
    for index, spec in enumerate(specs):
        predictions = spec_crossfit(
            spec,
            feature_sets,
            labels,
            groups,
            full_splits,
            seed=RANDOM_SEED + 10_000 + index,
        )
        full_predictions[spec["name"]] = predictions
        full_cv.append(
            {
                **spec,
                "feature_count": int(feature_sets[spec["features"]].shape[1]),
                "average_precision": float(average_precision_score(labels, predictions)),
                "auroc": float(roc_auc_score(labels, predictions)),
            }
        )
    final_selection = max(
        full_cv,
        key=lambda item: (
            item["average_precision"],
            item["auroc"],
            -item["feature_count"],
            item["name"],
        ),
    )
    final_spec = next(item for item in specs if item["name"] == final_selection["name"])
    final_thresholds = {
        str(target): choose_threshold_at_fpr(
            labels, full_predictions[final_spec["name"]], target
        )
        for target in TARGET_FPRS
    }
    final_model = estimator(final_spec["model"], RANDOM_SEED)
    final_model.fit(feature_sets[final_spec["features"]], labels)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_artifact = artifact_path.with_suffix(artifact_path.suffix + ".tmp")
    joblib.dump(
        {
            "schema_version": 1,
            "model": final_model,
            "candidate": final_spec,
            "feature_names": feature_names[final_spec["features"]],
            "thresholds": final_thresholds,
            "development_manifest_sha256": sha256(manifest_path),
            "warning": "development fit; requires untouched external same-sensor validation",
        },
        temporary_artifact,
    )
    os.replace(temporary_artifact, artifact_path)

    bootstrap = {
        str(target): group_bootstrap(
            labels,
            groups,
            baseline_scores,
            baseline_decisions,
            nested["scores"],
            nested["decisions"][target],
        )
        for target in TARGET_FPRS
    }
    promotion = {
        "development_only": True,
        "ap_exceeds_released_mars": nested_ranking["average_precision"] > baseline_metrics["average_precision"],
        "auroc_exceeds_released_mars": nested_ranking["auroc"] > baseline_metrics["auroc"],
        "recall_at_0_095_exceeds_released_mars": operating["0.095"]["recall"] > baseline_metrics["recall"],
        "fpr_at_0_095_not_above_released_mars": operating["0.095"]["false_positive_rate"] <= baseline_metrics["false_positive_rate"],
        "confirmatory_promotion_permitted": False,
    }
    decision = (
        "Use the winning cross-fitted candidate as the v4 development architecture only. "
        "Do not claim improvement or tune it on the former strict cohort again. Freeze its "
        "feature contract and evaluate once on a newly acquired, prediction-blind, same-sensor "
        "plume/no-plume cohort with spatial and temporal isolation from every fit location."
    )
    report = {
        "schema_version": 1,
        "scope": "ersrr_v4_group_nested_cascade_development_not_final_test",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "cohort": {
            "scenes": int(labels.size),
            "groups": int(np.unique(groups).size),
            "positives": int(np.count_nonzero(labels == 1)),
            "negatives": int(np.count_nonzero(labels == 0)),
            "status": "former frozen v3 strict test; opened once for v3 and now irreversibly development-only for v4",
        },
        "method": {
            "outer_folds": OUTER_FOLDS,
            "inner_folds": INNER_FOLDS,
            "group_unit": "frozen 25 km connected component",
            "fold_assignment": "500-trial deterministic label-count/row-count/group-count balancing; independent of model scores and features",
            "candidate_selection": "maximum inner out-of-fold average precision; AUROC, fewer features, then name break ties",
            "operating_thresholds": "selected separately inside each outer-training split from inner OOF predictions",
            "target_false_positive_rates": list(TARGET_FPRS),
            "raw_feature_label_blinding": "physics features use imagery, cloud, wind, and reference interval; never plume masks or labels",
            "candidate_specs": specs,
            "feature_schemas": feature_names,
        },
        "released_mars_s2l": baseline_metrics,
        "nested_crossfit": {
            "ranking": nested_ranking,
            "operating_points": operating,
            "folds": nested["folds"],
            "selection_counts": dict(
                sorted(Counter(fold["selected_candidate"]["name"] for fold in nested["folds"]).items())
            ),
        },
        "paired_group_bootstrap": bootstrap,
        "final_development_fit": {
            "all_candidate_crossfit_results": full_cv,
            "selected_candidate": final_selection,
            "thresholds_from_full_group_oof": final_thresholds,
            "artifact": artifact_record(artifact_path, root),
            "feature_cache": artifact_record(feature_cache_path, root),
        },
        "development_promotion_checks": promotion,
        "decision": decision,
        "limitations": [
            "The cohort was already used for the v3 primary result, so v4 results are development estimates even though every v4 prediction is group-cross-fitted.",
            "Only 67 positive scenes are available; outer-fold estimates and architecture selection remain high variance.",
            "The cascade depends on the non-commercial ShareAlike MARS-S2L release and inherits its deployment/license constraints.",
            "Cross-fitted probabilities come from fold-specific models; ranking metrics estimate the procedure, not one already-frozen final artifact.",
            "No EMIT cross-sensor label is used for training or final selection.",
        ],
        "runtime": {
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
        },
        "provenance": {
            "script": "tools/train_mars_v4_cascade.py",
            "script_sha256": sha256(Path(__file__).resolve()),
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
            "git_tracked_worktree_dirty_at_start": tracked_dirty(root),
            "inputs": {
                "campaign": {"path": campaign_path.relative_to(root).as_posix(), "sha256": sha256(campaign_path)},
                "strict_manifest": {"path": manifest_path.relative_to(root).as_posix(), "sha256": sha256(manifest_path)},
                "released_scene_cache": baseline_input,
                "v3_scene_caches": [item["scene_cache"] for item in campaign["inputs"]["v3_reports"]],
            },
        },
    }
    write_json(output_json, report)
    write_markdown(output_markdown, report)
    print(f"Wrote {output_json.relative_to(root)}")
    print(f"Wrote {output_markdown.relative_to(root)}")
    print(f"Selected development candidate: {final_selection['name']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
