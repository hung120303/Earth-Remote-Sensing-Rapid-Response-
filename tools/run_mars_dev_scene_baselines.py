#!/usr/bin/env python3
"""Run group-disjoint scene-presence baselines on the MARS development tranche."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import sklearn
from scipy import ndimage
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "EarthRemoteSensingRapidResponse"
if str(MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_ROOT))

from mars_s2l_adapter import MARS_BANDS, iter_manifest, load_sample  # noqa: E402

from acquire_mars_metadata import DEFAULT_OUTPUT, REVISION, checked_output_dir, repo_root, sha256  # noqa: E402
from build_mars_dev_cohort import DEV_SAMPLES, DEFAULT_JSON as DEV_REPORT_JSON  # noqa: E402

FEATURE_CACHE = "publication_dev_scene_features.npz"
DEFAULT_JSON = Path("reports/experiments/mars_dev_scene_baselines.json")
DEFAULT_MARKDOWN = Path("reports/experiments/MARS_DEV_SCENE_BASELINES.md")
BOOTSTRAP_REPLICATES = 2_000
MODEL_SEED = 101
EPSILON = 1e-4
COMPONENT_SCORE_THRESHOLDS = (0.02, 0.05, 0.08, 0.10)
FULL_ROLE_COUNTS = {
    "internal_training": {"positive": 2007, "negative": 21756},
    "internal_validation": {"positive": 505, "negative": 5440},
    "strict_spatial_test": {"positive": 67, "negative": 4334},
}


def git_commit(root: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


def tracked_dirty(root: Path) -> bool:
    output = subprocess.check_output(
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
    return bool(output.strip())


def safe_output(root: Path, value: str) -> Path:
    result = (root / value).resolve()
    if root not in result.parents:
        raise ValueError("Experiment output must resolve beneath the repository root")
    return result


def quantile_features(values: np.ndarray, prefix: str) -> tuple[list[str], list[float]]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError(f"No finite observable values for {prefix}")
    names = [
        f"{prefix}_mean",
        f"{prefix}_std",
        f"{prefix}_p10",
        f"{prefix}_p50",
        f"{prefix}_p90",
    ]
    q10, q50, q90 = np.quantile(finite, (0.10, 0.50, 0.90))
    values_out = [
        float(np.mean(finite)),
        float(np.std(finite)),
        float(q10),
        float(q50),
        float(q90),
    ]
    return names, values_out


def largest_component(mask: np.ndarray) -> int:
    labels, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    if count == 0:
        return 0
    return int(np.max(np.bincount(labels.ravel())[1:]))


def scene_features(sample: Any) -> tuple[list[str], np.ndarray, int]:
    observable = sample.observable_mask
    names: list[str] = []
    values: list[float] = []
    for date_name, image in (("target", sample.target), ("reference", sample.reference)):
        for band_index, band in enumerate(MARS_BANDS):
            local_names, local_values = quantile_features(
                image[band_index][observable], f"raw_{date_name}_{band}"
            )
            names.extend(local_names)
            values.extend(local_values)
    normalized_change = (sample.target - sample.reference) / (
        sample.target + sample.reference + EPSILON
    )
    for band_index, band in enumerate(MARS_BANDS):
        local = normalized_change[band_index][observable]
        local_names, local_values = quantile_features(local, f"raw_change_{band}")
        names.extend(local_names)
        values.extend(local_values)
        names.append(f"raw_change_{band}_fraction_abs_gt_0p1")
        values.append(float(np.mean(np.abs(local) > 0.1)))
    names.extend(["raw_observable_fraction", "raw_clear_fraction"])
    values.extend(
        [float(np.mean(observable)), float(np.mean(sample.clear_mask))]
    )
    raw_count = len(names)

    for method, mbmp in (
        ("release", sample.mbmp_release_compatible),
        ("valid", sample.mbmp_valid_aware),
    ):
        score = (1.0 - mbmp)[observable]
        prefix = f"physics_mbmp_{method}_score"
        q = np.quantile(score, (0.50, 0.90, 0.95, 0.99, 0.999))
        names.extend(
            [
                f"{prefix}_mean",
                f"{prefix}_std",
                f"{prefix}_p50",
                f"{prefix}_p90",
                f"{prefix}_p95",
                f"{prefix}_p99",
                f"{prefix}_p999",
            ]
        )
        values.extend([float(np.mean(score)), float(np.std(score)), *map(float, q)])
        full_score = np.zeros(observable.shape, dtype=np.float32)
        full_score[observable] = score
        for threshold in COMPONENT_SCORE_THRESHOLDS:
            encoded = str(threshold).replace(".", "p")
            mask = (full_score >= threshold) & observable
            names.extend(
                [
                    f"{prefix}_fraction_gt_{encoded}",
                    f"{prefix}_largest_component_gt_{encoded}",
                ]
            )
            values.extend(
                [float(np.mean(mask[observable])), float(largest_component(mask))]
            )
    return names, np.asarray(values, dtype=np.float32), raw_count


def cache_identity(manifest_path: Path) -> dict[str, str]:
    return {
        "manifest_sha256": sha256(manifest_path),
        "adapter_sha256": sha256(MODEL_ROOT / "mars_s2l_adapter.py"),
        "feature_schema": "mars_scene_features_v1",
    }


def extract_features(
    metadata_dir: Path, manifest_path: Path, cache_path: Path
) -> dict[str, np.ndarray]:
    records = list(iter_manifest(manifest_path))
    matrix: list[np.ndarray] = []
    labels: list[int] = []
    sample_ids: list[str] = []
    roles: list[str] = []
    groups: list[str] = []
    feature_names: list[str] | None = None
    raw_count: int | None = None
    for index, record in enumerate(records, start=1):
        sample = load_sample(metadata_dir, record)
        current_names, current_values, current_raw_count = scene_features(sample)
        if feature_names is None:
            feature_names = current_names
            raw_count = current_raw_count
        elif feature_names != current_names or raw_count != current_raw_count:
            raise ValueError("Scene feature schema changed within the development cohort")
        matrix.append(current_values)
        labels.append(sample.presence)
        sample_ids.append(sample.sample_id)
        roles.append(record["research_role"])
        groups.append(record["group_id"])
        if index % 250 == 0 or index == len(records):
            print(f"Extracted features: {index:,}/{len(records):,}", file=sys.stderr, flush=True)
    identity = cache_identity(manifest_path)
    temporary = cache_path.with_suffix(cache_path.suffix + ".tmp.npz")
    np.savez_compressed(
        temporary,
        x=np.stack(matrix),
        y=np.asarray(labels, dtype=np.uint8),
        sample_ids=np.asarray(sample_ids),
        roles=np.asarray(roles),
        groups=np.asarray(groups),
        feature_names=np.asarray(feature_names),
        raw_feature_count=np.asarray([raw_count], dtype=np.int64),
        identity_json=np.asarray([json.dumps(identity, sort_keys=True)]),
    )
    os.replace(temporary, cache_path)
    return load_feature_cache(cache_path, identity)


def load_feature_cache(path: Path, expected_identity: dict[str, str]) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        observed_identity = json.loads(str(payload["identity_json"][0]))
        if observed_identity != expected_identity:
            raise ValueError("Feature cache identity does not match the current manifest/adapter")
        return {key: payload[key].copy() for key in payload.files}


def role_weights(y: np.ndarray, role: str) -> np.ndarray:
    full = FULL_ROLE_COUNTS[role]
    observed_positive = int(np.sum(y == 1))
    observed_negative = int(np.sum(y == 0))
    return np.where(
        y == 1,
        full["positive"] / observed_positive,
        full["negative"] / observed_negative,
    ).astype(np.float64)


def confusion(y: np.ndarray, predicted: np.ndarray, weights: np.ndarray | None = None) -> dict[str, float]:
    w = np.ones(y.shape, dtype=np.float64) if weights is None else weights
    tp = float(np.sum(w[(y == 1) & (predicted == 1)]))
    tn = float(np.sum(w[(y == 0) & (predicted == 0)]))
    fp = float(np.sum(w[(y == 0) & (predicted == 1)]))
    fn = float(np.sum(w[(y == 1) & (predicted == 0)]))
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def ratio(numerator: float, denominator: float) -> float | None:
    return None if denominator == 0 else numerator / denominator


def metrics(
    y: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    *,
    weights: np.ndarray | None = None,
) -> dict[str, Any]:
    predicted = (scores >= threshold).astype(np.uint8)
    c = confusion(y, predicted, weights)
    result = {
        **c,
        "precision": ratio(c["tp"], c["tp"] + c["fp"]),
        "recall": ratio(c["tp"], c["tp"] + c["fn"]),
        "specificity": ratio(c["tn"], c["tn"] + c["fp"]),
        "false_positive_rate": ratio(c["fp"], c["fp"] + c["tn"]),
        "negative_predictive_value": ratio(c["tn"], c["tn"] + c["fn"]),
        "accuracy": ratio(c["tp"] + c["tn"], sum(c.values())),
    }
    result["auroc"] = float(roc_auc_score(y, scores, sample_weight=weights))
    result["average_precision"] = float(
        average_precision_score(y, scores, sample_weight=weights)
    )
    if np.all((scores >= 0.0) & (scores <= 1.0)):
        result["brier"] = float(brier_score_loss(y, scores, sample_weight=weights))
    else:
        result["brier"] = None
    return result


def threshold_candidates(scores: np.ndarray) -> np.ndarray:
    unique = np.unique(scores)
    if unique.size == 1:
        return unique
    midpoints = (unique[:-1] + unique[1:]) / 2.0
    return np.concatenate(
        [[np.nextafter(unique[0], -np.inf)], midpoints, [np.nextafter(unique[-1], np.inf)]]
    )


def choose_upper_threshold(y: np.ndarray, scores: np.ndarray) -> tuple[float, dict[str, Any]]:
    candidates: list[tuple[tuple[float, ...], float, dict[str, Any]]] = []
    for threshold in threshold_candidates(scores):
        result = metrics(y, scores, float(threshold))
        fpr = float(result["false_positive_rate"] or 0.0)
        recall = float(result["recall"] or 0.0)
        feasible = 1.0 if fpr <= 0.05 else 0.0
        rank = (feasible, recall if feasible else -fpr, -fpr, -float(threshold))
        candidates.append((rank, float(threshold), result))
    _, threshold, result = max(candidates, key=lambda item: item[0])
    return threshold, result


def choose_lower_threshold(
    y: np.ndarray, scores: np.ndarray, upper: float, weights: np.ndarray
) -> tuple[float | None, dict[str, Any]]:
    best: tuple[tuple[float, ...], float, dict[str, Any]] | None = None
    for threshold in threshold_candidates(scores):
        if threshold >= upper:
            continue
        accept_no = scores <= threshold
        accepted = int(np.sum(accept_no))
        if accepted == 0:
            continue
        c = confusion(y[accept_no], np.zeros(accepted, dtype=np.uint8), weights[accept_no])
        npv = ratio(c["tn"], c["tn"] + c["fn"])
        if npv is None or npv < 0.95:
            continue
        coverage = float(np.sum(weights[accept_no]) / np.sum(weights))
        candidate = ((coverage, npv, threshold), float(threshold), {"npv": npv, "weighted_coverage": coverage, "accepted_samples": accepted})
        if best is None or candidate[0] > best[0]:
            best = candidate
    if best is None:
        return None, {"npv": None, "weighted_coverage": 0.0, "accepted_samples": 0}
    return best[1], best[2]


def selective_metrics(
    y: np.ndarray,
    scores: np.ndarray,
    lower: float | None,
    upper: float,
    weights: np.ndarray,
) -> dict[str, Any]:
    plume = scores >= upper
    no_plume = np.zeros(scores.shape, dtype=bool) if lower is None else scores <= lower
    accepted = plume | no_plume
    predicted = plume.astype(np.uint8)
    c = confusion(y[accepted], predicted[accepted], weights[accepted]) if np.any(accepted) else {"tp": 0.0, "tn": 0.0, "fp": 0.0, "fn": 0.0}
    return {
        "weighted_coverage": float(np.sum(weights[accepted]) / np.sum(weights)),
        "sample_coverage": float(np.mean(accepted)),
        "abstention_rate": float(1.0 - np.mean(accepted)),
        "accepted_samples": int(np.sum(accepted)),
        "accepted_plume": int(np.sum(plume)),
        "accepted_no_plume": int(np.sum(no_plume)),
        "accepted_precision": ratio(c["tp"], c["tp"] + c["fp"]),
        "accepted_recall": ratio(c["tp"], c["tp"] + c["fn"]),
        "accepted_specificity": ratio(c["tn"], c["tn"] + c["fp"]),
        "accepted_no_plume_npv": ratio(c["tn"], c["tn"] + c["fn"]),
        "accepted_error_rate": ratio(c["fp"] + c["fn"], sum(c.values())),
        "weighted_confusion": c,
    }


def bootstrap_ci(
    y: np.ndarray,
    scores: np.ndarray,
    groups: np.ndarray,
    threshold: float,
    seed: int,
) -> dict[str, list[float]]:
    rng = np.random.default_rng(seed)
    unique_groups = np.unique(groups)
    indices_by_group = {group: np.flatnonzero(groups == group) for group in unique_groups}
    recalls: list[float] = []
    specificities: list[float] = []
    for _ in range(BOOTSTRAP_REPLICATES):
        sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        indices = np.concatenate([indices_by_group[group] for group in sampled_groups])
        result = metrics(y[indices], scores[indices], threshold)
        if result["recall"] is not None:
            recalls.append(float(result["recall"]))
        if result["specificity"] is not None:
            specificities.append(float(result["specificity"]))
    return {
        "recall_95ci": [float(value) for value in np.quantile(recalls, (0.025, 0.975))],
        "specificity_95ci": [
            float(value) for value in np.quantile(specificities, (0.025, 0.975))
        ],
        "replicates": BOOTSTRAP_REPLICATES,
        "unit": "frozen 25 km group",
    }


def run_model(
    name: str,
    train_y: np.ndarray,
    val_y: np.ndarray,
    test_y: np.ndarray,
    val_scores: np.ndarray,
    test_scores: np.ndarray,
    val_weights: np.ndarray,
    test_weights: np.ndarray,
    test_groups: np.ndarray,
) -> dict[str, Any]:
    upper, validation = choose_upper_threshold(val_y, val_scores)
    lower, lower_selection = choose_lower_threshold(val_y, val_scores, upper, val_weights)
    test_unweighted = metrics(test_y, test_scores, upper)
    test_weighted = metrics(test_y, test_scores, upper, weights=test_weights)
    return {
        "name": name,
        "train_positive": int(np.sum(train_y)),
        "operating_rule": {
            "upper_plume_threshold": upper,
            "lower_no_plume_threshold": lower,
            "selected_on": "internal_validation",
            "upper_objective": "maximum recall at observed FPR <= 0.05",
            "lower_objective": "maximum representative-weighted coverage at NPV >= 0.95",
            "lower_selection": lower_selection,
        },
        "validation": validation,
        "test_unweighted": test_unweighted,
        "test_representative_weighted": test_weighted,
        "test_selective": selective_metrics(test_y, test_scores, lower, upper, test_weights),
        "test_group_bootstrap": bootstrap_ci(
            test_y, test_scores, test_groups, upper, seed=MODEL_SEED + len(name)
        ),
    }


def fit_models(data: dict[str, np.ndarray]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    x = data["x"].astype(np.float64)
    y = data["y"].astype(np.uint8)
    roles = data["roles"].astype(str)
    groups = data["groups"].astype(str)
    raw_count = int(data["raw_feature_count"][0])
    feature_names = data["feature_names"].astype(str).tolist()
    train = roles == "internal_training"
    val = roles == "internal_validation"
    test = roles == "strict_spatial_test"
    train_weights = role_weights(y[train], "internal_training")
    val_weights = role_weights(y[val], "internal_validation")
    test_weights = role_weights(y[test], "strict_spatial_test")

    specifications: list[tuple[str, Any, np.ndarray]] = [
        (
            "raw_scene_logistic",
            make_pipeline(
                StandardScaler(),
                LogisticRegression(C=0.5, max_iter=2_000, random_state=MODEL_SEED),
            ),
            np.arange(raw_count),
        ),
        (
            "physics_scene_logistic",
            make_pipeline(
                StandardScaler(),
                LogisticRegression(C=0.5, max_iter=2_000, random_state=MODEL_SEED),
            ),
            np.arange(x.shape[1]),
        ),
        (
            "physics_hist_gradient_boosting",
            HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_iter=200,
                max_leaf_nodes=15,
                l2_regularization=1.0,
                random_state=MODEL_SEED,
            ),
            np.arange(x.shape[1]),
        ),
    ]
    outputs: list[dict[str, Any]] = []

    mbmp_index = feature_names.index("physics_mbmp_valid_score_p99")
    outputs.append(
        run_model(
            "valid_aware_mbmp_p99",
            y[train],
            y[val],
            y[test],
            x[val, mbmp_index],
            x[test, mbmp_index],
            val_weights,
            test_weights,
            groups[test],
        )
    )
    for name, model, indices in specifications:
        if hasattr(model, "named_steps"):
            final_step = next(reversed(model.named_steps))
            model.fit(
                x[train][:, indices],
                y[train],
                **{f"{final_step}__sample_weight": train_weights},
            )
        else:
            model.fit(x[train][:, indices], y[train], sample_weight=train_weights)
        val_scores = model.predict_proba(x[val][:, indices])[:, 1]
        test_scores = model.predict_proba(x[test][:, indices])[:, 1]
        outputs.append(
            run_model(
                name,
                y[train],
                y[val],
                y[test],
                val_scores,
                test_scores,
                val_weights,
                test_weights,
                groups[test],
            )
        )
    context = {
        "feature_count": int(x.shape[1]),
        "raw_feature_count": raw_count,
        "physics_feature_count": int(x.shape[1] - raw_count),
        "role_counts": {
            role: {
                "rows": int(np.sum(roles == role)),
                "positive": int(np.sum(y[roles == role])),
                "negative": int(np.sum(roles == role) - np.sum(y[roles == role])),
                "groups": int(len(np.unique(groups[roles == role]))),
            }
            for role in ("internal_training", "internal_validation", "strict_spatial_test")
        },
        "representative_weight_targets": FULL_ROLE_COUNTS,
    }
    return outputs, context


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# MARS-S2L group-disjoint scene baselines",
        "",
        "Development-tranche result; architecture-screening evidence, not the final paper estimate.",
        "",
        f"- Training: {report['context']['role_counts']['internal_training']['rows']:,} scenes",
        f"- Validation: {report['context']['role_counts']['internal_validation']['rows']:,} scenes; thresholds selected here only",
        f"- Strict spatial test: {report['context']['role_counts']['strict_spatial_test']['rows']:,} scenes / {report['context']['role_counts']['strict_spatial_test']['groups']:,} groups",
        f"- Features: {report['context']['raw_feature_count']} raw/change + {report['context']['physics_feature_count']} MBMP/physics",
        "- Confidence intervals: 2,000 bootstrap resamples of strict 25 km groups",
        "",
        "| Model | Val recall | Val FPR | Test recall | Test specificity | Recall 95% CI | Specificity 95% CI | Selective coverage |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model in report["models"]:
        validation = model["validation"]
        test = model["test_unweighted"]
        ci = model["test_group_bootstrap"]
        selective = model["test_selective"]
        lines.append(
            f"| {model['name']} | {fmt(validation['recall'])} | {fmt(validation['false_positive_rate'])} | "
            f"{fmt(test['recall'])} | {fmt(test['specificity'])} | "
            f"{ci['recall_95ci'][0]:.3f}-{ci['recall_95ci'][1]:.3f} | "
            f"{ci['specificity_95ci'][0]:.3f}-{ci['specificity_95ci'][1]:.3f} | "
            f"{selective['weighted_coverage']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            report["decision"],
            "",
            "Specificity and FPR are estimable from 512 group-diverse strict-test negatives, while recall uses all 67 strict-test positives. The tranche is class-enriched, so representative-weighted calibration metrics are reported separately and final claims still require the full frozen cohort and five learned-model seeds.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-dir", default=DEFAULT_OUTPUT.as_posix())
    parser.add_argument("--output-json", default=DEFAULT_JSON.as_posix())
    parser.add_argument("--output-markdown", default=DEFAULT_MARKDOWN.as_posix())
    parser.add_argument("--refresh-features", action="store_true")
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args()
    root = repo_root()
    try:
        metadata_dir = checked_output_dir(root, args.metadata_dir)
        manifest_path = metadata_dir / DEV_SAMPLES
        dev_report = json.loads((root / DEV_REPORT_JSON).read_text(encoding="utf-8"))
        if sha256(manifest_path) != dev_report["identities"]["sample_manifest_sha256"]:
            raise ValueError("Development manifest identity does not match the tracked report")
        cache_path = metadata_dir / FEATURE_CACHE
        identity = cache_identity(manifest_path)
        if cache_path.is_file() and not args.refresh_features:
            data = load_feature_cache(cache_path, identity)
        else:
            data = extract_features(metadata_dir, manifest_path, cache_path)
        models, context = fit_models(data)
        def validation_rank(model: dict[str, Any]) -> tuple[float, ...]:
            validation = model["validation"]
            fpr = float(validation["false_positive_rate"] or 0.0)
            feasible = 1.0 if fpr <= 0.05 else 0.0
            return (
                feasible,
                float(validation["recall"] or 0.0) if feasible else -fpr,
                float(validation["auroc"]),
                -fpr,
            )

        strongest = max(models, key=validation_rank)
        gate_pass = (
            float(strongest["test_group_bootstrap"]["recall_95ci"][0]) >= 0.75
            and float(strongest["test_unweighted"]["false_positive_rate"] or 1.0) <= 0.05
            and float(strongest["test_unweighted"]["specificity"] or 0.0) >= 0.95
        )
        decision = (
            f"Validation-selected development baseline: `{strongest['name']}`. "
            + (
                "It clears the provisional recall/FPR/specificity gate on this tranche, but full-cohort five-seed confirmation is still required."
                if gate_pass
                else "It does not clear the research promotion gate; proceed to a joint presence/segmentation model with hard-negative mining, not a larger backbone alone."
            )
        )
        output_json = safe_output(root, args.output_json)
        output_markdown = safe_output(root, args.output_markdown)
        report = {
            "schema_version": 1,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "scope": "group_disjoint_development_tranche_not_final_paper_claim",
            "source": {
                "repository": "UNEP-IMEO/MARS-S2L",
                "revision": REVISION,
                "development_manifest_sha256": identity["manifest_sha256"],
                "feature_cache_identity": identity,
            },
            "context": context,
            "models": models,
            "decision": decision,
            "promotion_gate_passed_on_development_tranche": gate_pass,
            "limitations": [
                "Class-enriched deterministic development subset, not deployment prevalence.",
                "Single fit seed for learned classical models; the final protocol requires five seeds.",
                "Scene-presence screening only; pixel segmentation is evaluated in the next experiment.",
                "Full released MARS-S2L and CH4Net baselines still require their runtime/checkpoints.",
                "The strict spatial development subset is used only for this predeclared baseline ladder; candidate architecture selection remains validation-only.",
            ],
            "provenance": {
                "git_commit": git_commit(root),
                "git_tracked_worktree_dirty_at_start": tracked_dirty(root),
                "script": "tools/run_mars_dev_scene_baselines.py",
                "script_sha256": sha256(Path(__file__)),
                "adapter_sha256": identity["adapter_sha256"],
                "python": sys.version.split()[0],
                "numpy": np.__version__,
                "rasterio": rasterio.__version__,
                "sklearn": sklearn.__version__,
            },
        }
        write_json(output_json, report)
        write_markdown(output_markdown, report)
    except (FileNotFoundError, OSError, RuntimeError, ValueError, rasterio.errors.RasterioError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, indent=None if args.compact else 2))
        return 2
    payload = {
        "ok": True,
        "scope": report["scope"],
        "strongest": strongest["name"],
        "gate_passed": gate_pass,
        "models": [
            {
                "name": model["name"],
                "test_recall": model["test_unweighted"]["recall"],
                "test_specificity": model["test_unweighted"]["specificity"],
            }
            for model in models
        ],
        "output_json": output_json.relative_to(root).as_posix(),
        "output_markdown": output_markdown.relative_to(root).as_posix(),
    }
    print(json.dumps(payload, indent=None if args.compact else 2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
