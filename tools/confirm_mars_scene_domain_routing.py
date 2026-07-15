#!/usr/bin/env python3
"""Reverse-validate a post-test sensor/offshore scene-head routing rule."""

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
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "EarthRemoteSensingRapidResponse"
for path in (MODEL_ROOT, ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from acquire_mars_metadata import repo_root, sha256  # noqa: E402
from train_mars_context_scene_ranker import augment_site_context  # noqa: E402
from train_mars_oof_scene_ensemble_v2 import (  # noqa: E402
    ap_group_bootstrap,
    fit_model as fit_v2,
    sample_weights,
)
from train_mars_paper_residual import (  # noqa: E402
    DEFAULT_MANIFEST,
    DEFAULT_PROTOCOL,
    SENSOR_NAMES,
    iter_development_manifest,
)
from train_mars_scene_ranker import (  # noqa: E402
    blend_scores,
    comparison,
    fit_model as fit_legacy,
    metric_summary,
    predict_model,
    safe_logit,
    site_cell_weights,
)

DEFAULT_INNER_CACHE = Path("outputs/mars_scene_features_folds234.npz")
DEFAULT_INNER_SHA256 = "01d8587e283c1179d61a7c789eb514b3f699d3e7a75bf8c50e4baff3f1698b89"
DEFAULT_FOLD0_CACHE = Path("outputs/mars_scene_features_fold0.npz")
DEFAULT_FOLD0_SHA256 = "372e152734db1314417ed385b099af54acd182bf758b1d2eabcedfeb64a709e7"
DEFAULT_FOLD1_CACHE = Path("outputs/mars_scene_features_fold1_crossfit.npz")
DEFAULT_FOLD1_SHA256 = "2b62e03215047d6a49639fdaead7e9d3cf7939b8eda26fb9442210b49c3ba108"
DEFAULT_LEGACY_HEAD = Path("EarthRemoteSensingRapidResponse/artifacts/mars_oof_context_ranker_folds234.joblib")
DEFAULT_LEGACY_SHA256 = "2d014f54918f68726d2ca4da19f35a1f29cb1b622fe7c32b56afc554ec27c370"
DEFAULT_NEW_HEAD = Path("EarthRemoteSensingRapidResponse/artifacts/mars_oof_scene_ensemble_v2.joblib")
DEFAULT_NEW_SHA256 = "9e6fa18b83ef065ac24c94a06a510057a0c382cecf1efa3b54e818566a45c9ac"
DEFAULT_JSON = Path("reports/experiments/mars_scene_domain_routing_confirmation.json")
DEFAULT_MARKDOWN = Path("reports/experiments/MARS_SCENE_DOMAIN_ROUTING_CONFIRMATION.md")
DEFAULT_SCORE_CACHE = Path("outputs/mars_scene_domain_routing_development_scores.npz")
INNER_FOLDS = (2, 3, 4)


def route_scores(
    legacy: np.ndarray,
    new: np.ndarray,
    sensors: np.ndarray,
    offshore: np.ndarray,
    *,
    sentinel_new_weight: float,
    landsat_new_weight: float,
    offshore_logit_shift: float,
) -> np.ndarray:
    if not (legacy.shape == new.shape == sensors.shape == offshore.shape):
        raise ValueError("scene routing arrays must align")
    routed = np.empty(legacy.shape, dtype=np.float64)
    weights = (sentinel_new_weight, landsat_new_weight)
    for index, weight in enumerate(weights):
        rows = sensors == index
        routed[rows] = blend_scores(legacy[rows], new[rows], weight)
    if np.any((sensors < 0) | (sensors >= len(SENSOR_NAMES))):
        raise ValueError("unknown sensor in scene routing")
    routed[offshore] = expit(safe_logit(routed[offshore]) + offshore_logit_shift)
    return routed


def load_cache(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as cache:
        return {
            name: cache[name]
            for name in ("features", "feature_names", "labels", "sensors", "groups", "folds")
        }


def summarize(
    labels: np.ndarray,
    primary: np.ndarray,
    new: np.ndarray,
    routed: np.ndarray,
    sensors: np.ndarray,
    groups: np.ndarray,
    offshore: np.ndarray,
    *,
    seed: int,
) -> dict[str, Any]:
    baseline_metrics = metric_summary(labels, primary, sensors)
    new_metrics = metric_summary(labels, new, sensors)
    routed_metrics = metric_summary(labels, routed, sensors)
    versus_primary = comparison(routed_metrics, baseline_metrics)
    versus_new = comparison(routed_metrics, new_metrics)
    bootstrap = ap_group_bootstrap(
        labels,
        primary,
        routed,
        groups,
        replicates=10000,
        seed=seed,
    )
    checks = {
        "ap_higher_than_primary": versus_primary["delta"]["average_precision"] > 0.0,
        "recall_higher_than_primary": versus_primary["delta"]["recall_at_fpr_0_0713"] > 0.0,
        "paired_ap_lower_positive": bootstrap["lower"] > 0.0,
        "no_material_sensor_ap_regression": min(
            versus_primary["delta"]["sensor_average_precision"].values()
        ) >= -0.005,
        "no_material_ap_regression_vs_new": versus_new["delta"]["average_precision"] >= -0.005,
    }
    return {
        "rows": int(labels.size),
        "positive": int(labels.sum()),
        "sites": len(set(groups.tolist())),
        "offshore_rows": int(offshore.sum()),
        "offshore_positive": int(labels[offshore].sum()),
        "primary": baseline_metrics,
        "new_head": new_metrics,
        "routed": routed_metrics,
        "versus_primary": versus_primary,
        "versus_new_head": versus_new,
        "paired_group_bootstrap_ap_delta_vs_primary": bootstrap,
        "checks": checks,
        "passed": all(checks.values()),
    }


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Scene domain-routing reverse validation",
        "",
        "The rule was diagnosed after opening the paper test, then evaluated unchanged on development folds.",
        "",
        "| Partition | AP delta vs primary | Recall delta | AP 95% CI | Offshore positive | Gates |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for name, value in report["partitions"].items():
        delta = value["versus_primary"]["delta"]
        ci = value["paired_group_bootstrap_ap_delta_vs_primary"]
        lines.append(
            f"| {name} | {delta['average_precision']:+.5f} | "
            f"{delta['recall_at_fpr_0_0713']:+.5f} | [{ci['lower']:+.5f}, {ci['upper']:+.5f}] | "
            f"{value['offshore_positive']} | {'PASS' if value['passed'] else 'FAIL'} |"
        )
    lines.extend(["", report["decision"]])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_score_cache(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inner-cache", default=DEFAULT_INNER_CACHE.as_posix())
    parser.add_argument("--inner-sha256", default=DEFAULT_INNER_SHA256)
    parser.add_argument("--fold0-cache", default=DEFAULT_FOLD0_CACHE.as_posix())
    parser.add_argument("--fold0-sha256", default=DEFAULT_FOLD0_SHA256)
    parser.add_argument("--fold1-cache", default=DEFAULT_FOLD1_CACHE.as_posix())
    parser.add_argument("--fold1-sha256", default=DEFAULT_FOLD1_SHA256)
    parser.add_argument("--legacy-head", default=DEFAULT_LEGACY_HEAD.as_posix())
    parser.add_argument("--legacy-sha256", default=DEFAULT_LEGACY_SHA256)
    parser.add_argument("--new-head", default=DEFAULT_NEW_HEAD.as_posix())
    parser.add_argument("--new-sha256", default=DEFAULT_NEW_SHA256)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST.as_posix())
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL.as_posix())
    parser.add_argument("--sentinel-new-weight", type=float, default=0.7)
    parser.add_argument("--landsat-new-weight", type=float, default=1.0)
    parser.add_argument("--offshore-logit-shift", type=float, default=-4.0)
    parser.add_argument("--output-json", default=DEFAULT_JSON.as_posix())
    parser.add_argument("--output-markdown", default=DEFAULT_MARKDOWN.as_posix())
    parser.add_argument("--score-cache", default=DEFAULT_SCORE_CACHE.as_posix())
    args = parser.parse_args()
    root = repo_root()
    cache_paths = {
        "inner": (root / args.inner_cache).resolve(),
        "fold0": (root / args.fold0_cache).resolve(),
        "fold1": (root / args.fold1_cache).resolve(),
    }
    for name, expected in (
        ("inner", args.inner_sha256),
        ("fold0", args.fold0_sha256),
        ("fold1", args.fold1_sha256),
    ):
        if sha256(cache_paths[name]) != expected:
            raise ValueError(f"Frozen {name} cache hash mismatch")
    legacy_path = (root / args.legacy_head).resolve()
    new_path = (root / args.new_head).resolve()
    if sha256(legacy_path) != args.legacy_sha256 or sha256(new_path) != args.new_sha256:
        raise ValueError("Frozen scene-head hash mismatch")
    legacy_payload = joblib.load(legacy_path)
    new_payload = joblib.load(new_path)
    manifest = (root / args.manifest).resolve()
    protocol_path = (root / args.protocol).resolve()
    group_offshore: dict[str, bool] = {}
    for record in iter_development_manifest(manifest):
        group = str(record["group_id"])
        value = str(record.get("country", "")) == "Offshore"
        if group in group_offshore and group_offshore[group] != value:
            raise ValueError("Development group spans onshore and offshore records")
        group_offshore[group] = value

    inner = load_cache(cache_paths["inner"])
    base = inner["features"].astype(np.float64)
    names = inner["feature_names"].astype(str)
    labels = inner["labels"].astype(np.uint8)
    sensors = inner["sensors"].astype(np.uint8)
    groups = inner["groups"].astype(str)
    folds = inner["folds"].astype(np.uint8)
    features, augmented_names = augment_site_context(base, names, groups)
    if augmented_names != new_payload["augmented_feature_names"] or names.tolist() != new_payload["feature_names"]:
        raise ValueError("Inner cache schema differs from the frozen heads")
    primary = base[:, int(np.flatnonzero(names == "primary_connected_score")[0])]
    legacy_oof = np.empty(labels.shape, dtype=np.float64)
    new_oof = np.empty(labels.shape, dtype=np.float64)
    for holdout in INNER_FOLDS:
        fit_rows = folds != holdout
        held_rows = folds == holdout
        legacy_fit = fit_legacy(
            legacy_payload["spec"],
            features[fit_rows],
            labels[fit_rows],
            site_cell_weights(groups[fit_rows], labels[fit_rows], sensors[fit_rows]),
        )
        legacy_head = predict_model(legacy_fit, features[held_rows])
        legacy_oof[held_rows] = blend_scores(primary[held_rows], legacy_head, 0.25)
        new_fit = fit_v2(
            new_payload["spec"],
            features[fit_rows],
            labels[fit_rows],
            sample_weights("uniform", groups[fit_rows], labels[fit_rows], sensors[fit_rows]),
        )
        new_head = new_fit.predict_proba(features[held_rows])[:, 1]
        new_oof[held_rows] = blend_scores(primary[held_rows], new_head, 0.625)
    offshore = np.asarray([group_offshore[group] for group in groups], dtype=bool)
    routed = route_scores(
        legacy_oof,
        new_oof,
        sensors,
        offshore,
        sentinel_new_weight=args.sentinel_new_weight,
        landsat_new_weight=args.landsat_new_weight,
        offshore_logit_shift=args.offshore_logit_shift,
    )
    partitions = {
        "cross_fitted_folds_2_3_4": summarize(
            labels, primary, new_oof, routed, sensors, groups, offshore, seed=20260715
        )
    }
    score_arrays = {
        "inner_labels": labels,
        "inner_sensors": sensors,
        "inner_groups": groups,
        "inner_folds": folds,
        "inner_offshore": offshore,
        "inner_primary": primary,
        "inner_legacy": legacy_oof,
        "inner_new": new_oof,
    }

    held_thresholds = []
    for fold_name, fold_number in (("held_fold_0", 0), ("held_fold_1", 1)):
        cache = load_cache(cache_paths[f"fold{fold_number}"])
        local_base = cache["features"].astype(np.float64)
        local_names = cache["feature_names"].astype(str)
        local_labels = cache["labels"].astype(np.uint8)
        local_sensors = cache["sensors"].astype(np.uint8)
        local_groups = cache["groups"].astype(str)
        local_features, local_augmented = augment_site_context(local_base, local_names, local_groups)
        if local_names.tolist() != names.tolist() or local_augmented != augmented_names:
            raise ValueError("Held-fold feature schema mismatch")
        local_primary = local_base[:, int(np.flatnonzero(local_names == "primary_connected_score")[0])]
        local_legacy = blend_scores(
            local_primary, predict_model(legacy_payload["fitted"], local_features), 0.25
        )
        local_new = blend_scores(
            local_primary, new_payload["fitted"].predict_proba(local_features)[:, 1], 0.625
        )
        local_offshore = np.asarray([group_offshore[group] for group in local_groups], dtype=bool)
        local_routed = route_scores(
            local_legacy,
            local_new,
            local_sensors,
            local_offshore,
            sentinel_new_weight=args.sentinel_new_weight,
            landsat_new_weight=args.landsat_new_weight,
            offshore_logit_shift=args.offshore_logit_shift,
        )
        partitions[fold_name] = summarize(
            local_labels,
            local_primary,
            local_new,
            local_routed,
            local_sensors,
            local_groups,
            local_offshore,
            seed=20260716 + fold_number,
        )
        prefix = f"fold{fold_number}"
        score_arrays.update(
            {
                f"{prefix}_labels": local_labels,
                f"{prefix}_sensors": local_sensors,
                f"{prefix}_groups": local_groups,
                f"{prefix}_offshore": local_offshore,
                f"{prefix}_primary": local_primary,
                f"{prefix}_legacy": local_legacy,
                f"{prefix}_new": local_new,
            }
        )
        held_thresholds.append(partitions[fold_name]["routed"]["operating_point"]["threshold"])

    passed = all(value["passed"] for value in partitions.values())
    score_cache_path = (root / args.score_cache).resolve()
    write_score_cache(score_cache_path, score_arrays)
    report = {
        "schema_version": 1,
        "scope": "post-test diagnosed rule reverse-validated on development folds; paper labels not loaded by this tool",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "rule": {
            "Sentinel-2_new_head_weight": args.sentinel_new_weight,
            "Landsat_new_head_weight": args.landsat_new_weight,
            "offshore_logit_shift": args.offshore_logit_shift,
        },
        "partitions": partitions,
        "operational_scene_threshold": max(held_thresholds),
        "all_development_reverse_validation_gates_pass": passed,
        "decision": (
            "Development evidence supports a transparent post-test domain-routed scene ensemble."
            if passed
            else "Reject the post-test scene routing rule because development reverse validation failed."
        ),
        "provenance": {
            "inner_cache_sha256": args.inner_sha256,
            "fold0_cache_sha256": args.fold0_sha256,
            "fold1_cache_sha256": args.fold1_sha256,
            "legacy_head_sha256": args.legacy_sha256,
            "new_head_sha256": args.new_sha256,
            "development_score_cache_sha256": sha256(score_cache_path),
            "manifest_sha256": sha256(manifest),
            "protocol_sha256": sha256(protocol_path),
            "script_sha256": sha256(Path(__file__).resolve()),
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        },
    }
    write_json((root / args.output_json).resolve(), report)
    write_markdown((root / args.output_markdown).resolve(), report)
    print(json.dumps({
        "ok": passed,
        "operational_scene_threshold": report["operational_scene_threshold"],
        "partitions": {
            name: {
                "ap_delta": value["versus_primary"]["delta"]["average_precision"],
                "recall_delta": value["versus_primary"]["delta"]["recall_at_fpr_0_0713"],
                "bootstrap_lower": value["paired_group_bootstrap_ap_delta_vs_primary"]["lower"],
                "passed": value["passed"],
            }
            for name, value in partitions.items()
        },
    }, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
