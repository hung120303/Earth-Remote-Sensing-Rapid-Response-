#!/usr/bin/env python3
"""Train the frozen CloudSEN12-negative scene complement on development data."""

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
import xgboost
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "EarthRemoteSensingRapidResponse"
for path in (MODEL_ROOT, ROOT / "tools"):
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


DEFAULT_PROTOCOL = Path("configs/mars_cloudsen12_negative_augmented_xgboost_protocol.json")
DEFAULT_RECEIPT = Path("reports/acquisition/mars_cloudsen12_common_stats.json")
DEFAULT_MARS_STATS = Path("outputs/mars_cloudsen12_common_stats_development.npz")
DEFAULT_CLOUD_STATS = Path("outputs/cloudsen12_common_stats_nonsealed.npz")
DEFAULT_ARTIFACT = Path(
    "EarthRemoteSensingRapidResponse/artifacts/mars_cloudsen12_negative_augmented_xgboost.joblib"
)
DEFAULT_JSON = Path("reports/experiments/mars_cloudsen12_negative_augmented_xgboost.json")
DEFAULT_MARKDOWN = Path("reports/experiments/MARS_CLOUDSEN12_NEGATIVE_AUGMENTED_XGBOOST.md")

INNER_FOLDS = (2, 3, 4)
CONFIRMATION_FOLDS = (0, 1)
SEED = 20260716
TARGET_FPR = 0.0713


def build_model(spec: dict[str, Any], fixed: dict[str, Any], *, seed: int) -> XGBClassifier:
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method=str(fixed["tree_method"]),
        device="cpu",
        n_estimators=int(spec["n_estimators"]),
        max_depth=int(spec["max_depth"]),
        learning_rate=float(spec["learning_rate"]),
        min_child_weight=float(spec["min_child_weight"]),
        subsample=float(fixed["subsample"]),
        colsample_bytree=float(fixed["colsample_bytree"]),
        reg_alpha=float(fixed["reg_alpha"]),
        reg_lambda=float(fixed["reg_lambda"]),
        gamma=0.0,
        max_bin=256,
        random_state=seed,
        n_jobs=int(fixed["n_jobs"]),
        verbosity=0,
    )


def augmented_fit_arrays(
    mars_features: np.ndarray,
    mars_labels: np.ndarray,
    cloud_features: np.ndarray,
    cloud_multiplier: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if cloud_multiplier <= 0.0:
        raise ValueError("CloudSEN12 multiplier must be positive")
    features = np.concatenate([mars_features, cloud_features], axis=0)
    labels = np.concatenate(
        [mars_labels.astype(np.uint8), np.zeros(cloud_features.shape[0], dtype=np.uint8)]
    )
    weights = np.concatenate(
        [
            np.ones(mars_features.shape[0], dtype=np.float64),
            np.full(cloud_features.shape[0], cloud_multiplier, dtype=np.float64),
        ]
    )
    return features, labels, weights


def align_common_features(
    values: dict[str, Any], path: Path, expected_hash: str
) -> tuple[np.ndarray, list[str]]:
    if sha256(path) != expected_hash:
        raise ValueError("Frozen MARS common-statistic cache hash mismatch")
    with np.load(path, allow_pickle=False) as cache:
        ids = cache["sample_ids"].astype(str)
        features = cache["features"].astype(np.float64)
        names = cache["feature_names"].astype(str).tolist()
    if np.unique(ids).size != ids.size or features.shape != (ids.size, len(names)):
        raise ValueError("MARS common-statistic cache is malformed")
    index = {identifier: row for row, identifier in enumerate(ids.tolist())}
    missing = [identifier for identifier in values["sample_ids"] if identifier not in index]
    if missing:
        raise ValueError(f"MARS common statistics miss {len(missing)} development rows")
    order = np.asarray([index[identifier] for identifier in values["sample_ids"]])
    aligned = features[order]
    if not np.isfinite(aligned).all():
        raise ValueError("Aligned MARS common statistics are non-finite")
    return aligned, names


def load_cloud_features(
    path: Path, expected_hash: str, expected_names: list[str]
) -> dict[str, np.ndarray]:
    if sha256(path) != expected_hash:
        raise ValueError("Frozen CloudSEN12 common-statistic cache hash mismatch")
    with np.load(path, allow_pickle=False) as cache:
        values = {name: cache[name].copy() for name in cache.files}
    names = values["feature_names"].astype(str).tolist()
    features = values["features"].astype(np.float64)
    splits = values["splits"].astype(str)
    labels = values["labels"].astype(np.uint8)
    if names != expected_names or features.shape != (splits.size, len(names)):
        raise ValueError("CloudSEN12 common-statistic schema mismatch")
    if set(np.unique(splits).tolist()) != {"train", "validation"}:
        raise ValueError("CloudSEN12 nonsealed cache has unexpected partitions")
    if labels.any() or not np.isfinite(features).all():
        raise ValueError("CloudSEN12 nonsealed cache violates the negative contract")
    return {"features": features, "splits": splits, "labels": labels}


def fit_and_predict(
    spec: dict[str, Any],
    fixed: dict[str, Any],
    mars_features: np.ndarray,
    mars_labels: np.ndarray,
    cloud_train: np.ndarray,
    cloud_multiplier: float,
    held_features: np.ndarray,
    *,
    seed: int,
) -> tuple[np.ndarray, XGBClassifier]:
    features, labels, weights = augmented_fit_arrays(
        mars_features, mars_labels, cloud_train, cloud_multiplier
    )
    model = build_model(spec, fixed, seed=seed)
    model.fit(features, labels, sample_weight=weights, verbose=False)
    scores = model.predict_proba(held_features)[:, 1].astype(np.float64)
    if not np.isfinite(scores).all():
        raise RuntimeError("CloudSEN12-augmented head emitted non-finite scores")
    return scores, model


def compare_subset(
    values: dict[str, Any], rows: np.ndarray, scores: np.ndarray
) -> dict[str, Any]:
    candidate = metric_summary(values["labels"][rows], scores, values["sensors"][rows])
    current = metric_summary(
        values["labels"][rows], values["current"][rows], values["sensors"][rows]
    )
    return comparison(candidate, current)


def candidate_summary(
    values: dict[str, Any], raw: np.ndarray, blend: float, rows: np.ndarray
) -> dict[str, Any]:
    scores = blend_scores(values["current"][rows], raw[rows], blend)
    pooled = compare_subset(values, rows, scores)
    per_fold = {}
    for fold in INNER_FOLDS:
        local_global = np.flatnonzero(rows)[values["folds"][rows] == fold]
        local_scores = blend_scores(
            values["current"][local_global], raw[local_global], blend
        )
        per_fold[str(fold)] = compare_subset(values, local_global, local_scores)
    fold_ap = [item["delta"]["average_precision"] for item in per_fold.values()]
    fold_recall = [item["delta"]["recall_at_fpr_0_0713"] for item in per_fold.values()]
    sensor_delta = pooled["delta"]["sensor_average_precision"]
    stable = (
        pooled["delta"]["average_precision"] >= 0.002
        and pooled["delta"]["recall_at_fpr_0_0713"] > 0.0
        and min(fold_ap) > 0.0
        and min(fold_recall) >= 0.0
        and min(sensor_delta.values()) >= -0.002
    )
    return {
        "blend": blend,
        "stable": bool(stable),
        "versus_current": pooled,
        "per_fold": per_fold,
        "rank": [
            int(stable),
            min(fold_recall),
            min(fold_ap),
            pooled["delta"]["average_precision"],
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
    lines = [
        "# CloudSEN12-negative augmented MARS scene head",
        "",
        "The candidate was selected on cross-fitted MARS folds 2/3/4 with only CloudSEN12 train negatives. Folds 0/1 and CloudSEN12 validation were evaluated only after selection.",
        "",
        f"- Selected model: `{selected['model_spec_name']}`",
        f"- CloudSEN12 negative weight: {selected['cloud_multiplier']:.2f}",
        f"- Current/candidate logit blend: {selected['blend']:.3f}",
        f"- Inner AP delta: {selected['inner']['versus_current']['delta']['average_precision']:+.5f}",
        f"- Inner recall delta: {selected['inner']['versus_current']['delta']['recall_at_fpr_0_0713']:+.5f}",
        "",
        "| Confirmation | AP delta | Recall delta | AP lower 95% CI |",
        "|---|---:|---:|---:|",
    ]
    for fold, item in selected["confirmation"].items():
        delta = item["versus_current"]["delta"]
        lines.append(
            f"| fold {fold} | {delta['average_precision']:+.5f} | "
            f"{delta['recall_at_fpr_0_0713']:+.5f} | {item['ap_bootstrap']['lower']:+.5f} |"
        )
    lines.extend(
        [
            "",
            f"CloudSEN12 validation raw-head p95/p99: {selected['cloudsen12_validation']['score_p95']:.5f} / {selected['cloudsen12_validation']['score_p99']:.5f}.",
            "",
            report["decision"],
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL.as_posix())
    parser.add_argument("--receipt", default=DEFAULT_RECEIPT.as_posix())
    parser.add_argument("--mars-stats", default=DEFAULT_MARS_STATS.as_posix())
    parser.add_argument("--cloud-stats", default=DEFAULT_CLOUD_STATS.as_posix())
    parser.add_argument("--inner-cache", default=DEFAULT_INNER_CACHE.as_posix())
    parser.add_argument("--fold0-cache", default=DEFAULT_FOLD0_CACHE.as_posix())
    parser.add_argument("--fold1-cache", default=DEFAULT_FOLD1_CACHE.as_posix())
    parser.add_argument("--score-cache", default=DEFAULT_SCORE_CACHE.as_posix())
    parser.add_argument("--artifact", default=DEFAULT_ARTIFACT.as_posix())
    parser.add_argument("--output-json", default=DEFAULT_JSON.as_posix())
    parser.add_argument("--output-markdown", default=DEFAULT_MARKDOWN.as_posix())
    args = parser.parse_args()

    root = repo_root()
    protocol_path = (root / args.protocol).resolve()
    receipt_path = (root / args.receipt).resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt["protocol_sha256"] != sha256(protocol_path):
        raise ValueError("Feature receipt does not match the frozen protocol")
    paths = {
        "inner": (root / args.inner_cache).resolve(),
        "fold0": (root / args.fold0_cache).resolve(),
        "fold1": (root / args.fold1_cache).resolve(),
        "score": (root / args.score_cache).resolve(),
    }
    expected_base = {
        "inner": DEFAULT_INNER_SHA256,
        "fold0": DEFAULT_FOLD0_SHA256,
        "fold1": DEFAULT_FOLD1_SHA256,
        "score": DEFAULT_SCORE_SHA256,
    }
    for name, digest in expected_base.items():
        if sha256(paths[name]) != digest:
            raise ValueError(f"Frozen {name} development cache hash mismatch")
    values = load_development(
        {name: paths[name] for name in ("inner", "fold0", "fold1")}, paths["score"]
    )
    mars_path = (root / args.mars_stats).resolve()
    cloud_path = (root / args.cloud_stats).resolve()
    mars_features, feature_names = align_common_features(
        values, mars_path, receipt["outputs"]["mars_development"]["sha256"]
    )
    cloud = load_cloud_features(
        cloud_path,
        receipt["outputs"]["cloudsen12_nonsealed"]["sha256"],
        feature_names,
    )
    cloud_train = cloud["features"][cloud["splits"] == "train"]
    cloud_validation = cloud["features"][cloud["splits"] == "validation"]

    family = protocol["candidate_family"]
    fixed = family["fixed_parameters"]
    inner_rows = np.isin(values["folds"], INNER_FOLDS)
    raw_store: dict[tuple[str, float], np.ndarray] = {}
    validation_store: dict[tuple[str, float], np.ndarray] = {}
    candidates = []
    for spec in family["specifications"]:
        for multiplier in family["cloudsen12_negative_weight_multipliers"]:
            key = (spec["name"], float(multiplier))
            raw = np.full(values["labels"].shape, np.nan, dtype=np.float64)
            validation_parts = []
            for holdout in INNER_FOLDS:
                fit = inner_rows & (values["folds"] != holdout)
                held = values["folds"] == holdout
                raw[held], model = fit_and_predict(
                    spec,
                    fixed,
                    mars_features[fit],
                    values["labels"][fit],
                    cloud_train,
                    float(multiplier),
                    mars_features[held],
                    seed=SEED + holdout,
                )
                validation_parts.append(model.predict_proba(cloud_validation)[:, 1])
            validation_store[key] = np.mean(np.stack(validation_parts), axis=0)
            raw_store[key] = raw
            for blend in family["candidate_logit_blends"]:
                item = candidate_summary(values, raw, float(blend), inner_rows)
                item.update(
                    {
                        "model_spec": spec,
                        "model_spec_name": spec["name"],
                        "cloud_multiplier": float(multiplier),
                    }
                )
                candidates.append(item)
            print(json.dumps({"completed": spec["name"], "cloud_multiplier": multiplier}), flush=True)

    selected = max(candidates, key=lambda item: tuple(item["rank"]))
    selected_key = (selected["model_spec_name"], selected["cloud_multiplier"])
    selected_raw = raw_store[selected_key]
    selected_inner_scores = blend_scores(
        values["current"][inner_rows], selected_raw[inner_rows], selected["blend"]
    )
    selected["inner"] = {
        "versus_current": selected.pop("versus_current"),
        "per_fold": selected.pop("per_fold"),
        "ap_bootstrap": ap_group_bootstrap(
            values["labels"][inner_rows],
            values["current"][inner_rows],
            selected_inner_scores,
            values["groups"][inner_rows],
            replicates=10_000,
            seed=SEED + 100,
        ),
    }
    inner_pass = bool(
        selected["stable"] and selected["inner"]["ap_bootstrap"]["lower"] > 0.0
    )

    confirmation: dict[str, Any] = {}
    confirmation_model = None
    confirmation_raw = None
    if inner_pass:
        fit = inner_rows
        confirmation_raw, confirmation_model = fit_and_predict(
            selected["model_spec"],
            fixed,
            mars_features[fit],
            values["labels"][fit],
            cloud_train,
            selected["cloud_multiplier"],
            mars_features[np.isin(values["folds"], CONFIRMATION_FOLDS)],
            seed=SEED + 200,
        )
        confirmation_rows = np.flatnonzero(np.isin(values["folds"], CONFIRMATION_FOLDS))
        for fold in CONFIRMATION_FOLDS:
            local_positions = np.flatnonzero(values["folds"][confirmation_rows] == fold)
            global_rows = confirmation_rows[local_positions]
            local_raw = confirmation_raw[local_positions]
            scores = blend_scores(values["current"][global_rows], local_raw, selected["blend"])
            versus = compare_subset(values, global_rows, scores)
            bootstrap = ap_group_bootstrap(
                values["labels"][global_rows],
                values["current"][global_rows],
                scores,
                values["groups"][global_rows],
                replicates=10_000,
                seed=SEED + 300 + fold,
            )
            delta = versus["delta"]
            passed = bool(
                delta["average_precision"] > 0.0
                and delta["recall_at_fpr_0_0713"] >= 0.0
                and min(delta["sensor_average_precision"].values()) >= -0.002
                and bootstrap["lower"] > 0.0
            )
            confirmation[str(fold)] = {
                "versus_current": versus,
                "ap_bootstrap": bootstrap,
                "passed": passed,
            }

    cloud_scores = validation_store[selected_key]
    mars_inner_negative = selected_raw[inner_rows & (values["labels"] == 0)]
    cloud_gate = {
        "rows": int(cloud_scores.size),
        "score_p95": float(np.quantile(cloud_scores, 0.95)),
        "score_p99": float(np.quantile(cloud_scores, 0.99)),
        "mars_inner_negative_score_p95": float(np.quantile(mars_inner_negative, 0.95)),
    }
    cloud_gate["passed"] = bool(
        cloud_gate["score_p95"] <= cloud_gate["mars_inner_negative_score_p95"]
        and cloud_gate["score_p99"] < 0.5
    )
    selected["confirmation"] = confirmation
    selected["cloudsen12_validation"] = cloud_gate
    passed = bool(
        inner_pass
        and len(confirmation) == len(CONFIRMATION_FOLDS)
        and all(item["passed"] for item in confirmation.values())
        and cloud_gate["passed"]
    )

    artifact_path = (root / args.artifact).resolve()
    artifact_digest = None
    if passed:
        all_features, all_labels, all_weights = augmented_fit_arrays(
            mars_features,
            values["labels"],
            cloud_train,
            selected["cloud_multiplier"],
        )
        final_model = build_model(selected["model_spec"], fixed, seed=SEED + 400)
        final_model.fit(all_features, all_labels, sample_weight=all_weights, verbose=False)
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = artifact_path.with_suffix(artifact_path.suffix + ".tmp")
        joblib.dump(
            {
                "schema_version": 1,
                "kind": "mars_cloudsen12_negative_augmented_xgboost",
                "model": final_model,
                "model_spec": selected["model_spec"],
                "cloud_multiplier": selected["cloud_multiplier"],
                "blend": selected["blend"],
                "feature_names": feature_names,
                "base_score": "frozen v3 stronger scene head",
                "protocol_sha256": sha256(protocol_path),
            },
            temporary,
            compress=3,
        )
        os.replace(temporary, artifact_path)
        artifact_digest = sha256(artifact_path)

    report = {
        "schema_version": 1,
        "scope": "development-only CloudSEN12-negative scene complement; no paper or CloudSEN12 test outcomes loaded",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_summaries": [
            {
                "model_spec_name": item["model_spec_name"],
                "cloud_multiplier": item["cloud_multiplier"],
                "blend": item["blend"],
                "stable": item["stable"],
                "ap_delta": item["inner"]["versus_current"]["delta"]["average_precision"]
                if item is selected
                else item["versus_current"]["delta"]["average_precision"],
                "recall_delta": item["inner"]["versus_current"]["delta"]["recall_at_fpr_0_0713"]
                if item is selected
                else item["versus_current"]["delta"]["recall_at_fpr_0_0713"],
            }
            for item in candidates
        ],
        "selected": selected,
        "all_promotion_gates_pass": passed,
        "decision": (
            "Freeze the CloudSEN12-negative artifact for sealed-negative and exact-paper replay."
            if passed
            else "Reject the CloudSEN12-negative branch before any sealed-negative or paper replay."
        ),
        "provenance": {
            "protocol_sha256": sha256(protocol_path),
            "receipt_sha256": sha256(receipt_path),
            "mars_stats_sha256": sha256(mars_path),
            "cloud_stats_sha256": sha256(cloud_path),
            **{f"{name}_cache_sha256": digest for name, digest in expected_base.items()},
            "artifact_sha256": artifact_digest,
            "script_sha256": sha256(Path(__file__).resolve()),
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True
            ).strip(),
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "joblib": joblib.__version__,
        },
    }
    write_json((root / args.output_json).resolve(), report)
    write_markdown((root / args.output_markdown).resolve(), report)
    print(
        json.dumps(
            {
                "ok": passed,
                "model": selected["model_spec_name"],
                "cloud_multiplier": selected["cloud_multiplier"],
                "blend": selected["blend"],
                "inner_ap_delta": selected["inner"]["versus_current"]["delta"]["average_precision"],
                "inner_ap_lower": selected["inner"]["ap_bootstrap"]["lower"],
                "confirmation": {fold: item["passed"] for fold, item in confirmation.items()},
                "cloud_validation_passed": cloud_gate["passed"],
                "artifact_sha256": artifact_digest,
            },
            indent=2,
        )
    )
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
