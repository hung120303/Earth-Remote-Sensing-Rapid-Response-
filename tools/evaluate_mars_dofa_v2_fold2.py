#!/usr/bin/env python3
"""Run the one-shot fold-2 confirmation of the fixed protected DOFA-v2 fusion."""

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
from acquire_mars_metadata import sha256  # noqa: E402
from confirm_mars_dofa_v2_projection_ensemble import (  # noqa: E402
    mean_logit_probabilities,
)
from confirm_mars_dofa_v2_train_fitted_normalization import (  # noqa: E402
    build_source_fitted_views,
    source_fitted_normalize,
)
from extract_mars_dofa_v2_scene_features import (  # noqa: E402
    FEATURE_WIDTH,
    MARS_TO_DOFA_MULTIPLIER,
)
from train_mars_crossfold_bagged_scene_head import load_development  # noqa: E402
from train_mars_dofa_v2_protected_fusion import protected_logit_blend  # noqa: E402
from train_mars_dofa_v2_scene_probe import (  # noqa: E402
    PROJECTION_DIM,
    SELECTION_FOLDS,
    align_features,
    crossfit_scores,
    evaluate_candidate,
    select_features,
)
from train_mars_oof_scene_ensemble_v2 import ap_group_bootstrap  # noqa: E402
from train_mars_scene_ranker import comparison, metric_summary  # noqa: E402

DEFAULT_PROTOCOL = Path("configs/mars_dofa_v2_fold2_protocol.json")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    delta = report["versus_current"]["delta"]
    interval = report["paired_group_bootstrap_ap_delta_vs_current"]
    lines = [
        "# One-shot DOFA-v2 fold-2 confirmation",
        "",
        f"- AP delta vs current: {delta['average_precision']:+.6f}",
        f"- Recall delta at FPR 0.0713: {delta['recall_at_fpr_0_0713']:+.6f}",
        f"- Landsat AP delta: {delta['sensor_average_precision']['Landsat']:+.6f}",
        f"- Sentinel-2 AP delta: "
        f"{delta['sensor_average_precision']['Sentinel-2']:+.6f}",
        f"- Paired-site AP 95% interval: [{interval['lower']:+.6f}, "
        f"{interval['upper']:+.6f}]",
        f"- Operating confusion counts preserved: "
        f"{'yes' if report['operating_counts_preserved'] else 'no'}",
        "",
        report["decision"],
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def verify_static_contract(protocol: dict[str, Any]) -> dict[str, Path]:
    if sha256(Path(__file__).resolve()) != protocol["evaluator"]["sha256"]:
        raise ValueError("Frozen fold-2 evaluator hash mismatch")
    for dependency in protocol["code_dependencies"]:
        path = (ROOT / dependency["path"]).resolve()
        if sha256(path) != dependency["sha256"]:
            raise ValueError(f"Frozen dependency mismatch: {dependency['path']}")
    paths = {}
    for name, contract in protocol["inputs"].items():
        path = (ROOT / contract["path"]).resolve()
        if sha256(path) != contract["sha256"]:
            raise ValueError(f"Frozen fold-2 input mismatch: {name}")
        paths[name] = path
    return paths


def align_fold_cache(
    path: Path, values: dict[str, Any], fold: int
) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as cache:
        features = cache["features"]
        names = cache["feature_names"].astype(str)
        ids = cache["sample_ids"].astype(str)
        labels = cache["labels"].astype(np.uint8)
        groups = cache["groups"].astype(str)
        sensors = cache["sensors"].astype(np.uint8)
        folds = cache["folds"].astype(np.uint8)
        if features.ndim != 2 or features.shape[1] != FEATURE_WIDTH:
            raise ValueError("Fold cache DOFA feature width differs")
        if np.unique(folds).tolist() != [fold]:
            raise ValueError("Fold cache contains unexpected folds")
        if str(cache["checkpoint_sha256"]) != CHECKPOINT_SHA256:
            raise ValueError("Fold cache checkpoint differs")
        if float(cache["mars_to_dofa_multiplier"]) != MARS_TO_DOFA_MULTIPLIER:
            raise ValueError("Fold cache radiometric contract differs")
    selected = np.asarray(values["folds"]) == fold
    target_ids = np.asarray(values["sample_ids"])[selected].astype(str)
    lookup = {sample_id: index for index, sample_id in enumerate(ids)}
    if len(lookup) != ids.size or set(target_ids.tolist()) != set(ids.tolist()):
        raise ValueError("Fold cache sample identities differ")
    order = np.asarray([lookup[sample_id] for sample_id in target_ids])
    for expected, observed, label in (
        (np.asarray(values["labels"])[selected], labels[order], "labels"),
        (np.asarray(values["groups"])[selected].astype(str), groups[order], "groups"),
        (np.asarray(values["sensors"])[selected], sensors[order], "sensors"),
    ):
        if not np.array_equal(expected, observed):
            raise ValueError(f"Fold cache {label} alignment failed")
    return features[order], names


def fit_predict_fold2(
    source: np.ndarray,
    target: np.ndarray,
    source_labels: np.ndarray,
    source_sensors: np.ndarray,
    target_sensors: np.ndarray,
    *,
    seed: int,
    c_value: float,
) -> np.ndarray:
    normalized_source, normalized_target = source_fitted_normalize(
        source,
        target,
        source_sensors,
        target_sensors,
        mode="global_train_fitted",
    )
    projection = SparseRandomProjection(
        n_components=PROJECTION_DIM,
        density="auto",
        dense_output=True,
        random_state=seed,
    )
    source_projected = projection.fit_transform(normalized_source).astype(np.float32)
    target_projected = projection.transform(normalized_target).astype(np.float32)
    source_projected, target_projected = source_fitted_normalize(
        source_projected,
        target_projected,
        source_sensors,
        target_sensors,
        mode="global_train_fitted",
    )
    positives = int((source_labels == 1).sum())
    negatives = int((source_labels == 0).sum())
    weights = np.where(source_labels == 1, np.sqrt(negatives / positives), 1.0)
    model = LogisticRegression(
        C=c_value,
        max_iter=500,
        solver="lbfgs",
        random_state=20260752,
    ).fit(source_projected, source_labels, sample_weight=weights)
    scores = model.predict_proba(target_projected)[:, 1]
    if not np.isfinite(scores).all():
        raise RuntimeError("Fold-2 DOFA probabilities are not finite")
    return scores.astype(np.float64)


def load_all_values(paths: dict[str, Path]) -> dict[str, Any]:
    return load_development(
        {name: paths[name] for name in ("inner", "fold0", "fold1")}, paths["score"]
    )


def reproduce_development(
    protocol: dict[str, Any], paths: dict[str, Path]
) -> dict[str, Any]:
    all_values = load_all_values(paths)
    encoded, names = align_features(paths["dofa_development"], all_values)
    selection = np.isin(all_values["folds"], SELECTION_FOLDS)
    values = {
        key: np.asarray(all_values[key])[selection]
        for key in ("labels", "sensors", "sample_ids", "groups", "folds", "primary", "current")
    }
    fixed = protocol["candidate"]
    features, _ = select_features(encoded, names, fixed["feature_set"])
    raw_scores = []
    for seed in map(int, fixed["projection_seeds"]):
        views = build_source_fitted_views(
            features,
            values["folds"],
            values["sensors"],
            seed=seed,
            mode="global_train_fitted",
        )
        raw_scores.append(crossfit_scores(views, values["labels"], float(fixed["C"])))
        del views
        gc.collect()
    aggregate = mean_logit_probabilities(raw_scores)
    scores = protected_logit_blend(
        values["current"], aggregate, gate=float(fixed["gate"]), weight=float(fixed["weight"])
    )
    observed = evaluate_candidate(values, scores, {"role": "fold2_smoke"}, 1.0)
    reference = json.loads(paths["development_report"].read_text(encoding="utf-8"))[
        "selected"
    ]["evaluation"]
    checks: dict[str, bool] = {}
    tolerance = float(protocol["smoke"]["absolute_tolerance"])
    for key in ("average_precision", "recall_at_fpr_0_0713"):
        checks[f"pooled_{key}"] = bool(
            np.isclose(
                observed["versus_current"]["delta"][key],
                reference["versus_current"]["delta"][key],
                rtol=0.0,
                atol=tolerance,
            )
        )
    for sensor in ("Landsat", "Sentinel-2"):
        checks[f"sensor_{sensor}_ap"] = bool(
            np.isclose(
                observed["versus_current"]["delta"]["sensor_average_precision"][sensor],
                reference["versus_current"]["delta"]["sensor_average_precision"][sensor],
                rtol=0.0,
                atol=tolerance,
            )
        )
    for fold in map(str, SELECTION_FOLDS):
        checks[f"fold_{fold}_ap"] = bool(
            np.isclose(
                observed["per_fold"][fold]["versus_current"]["delta"]["average_precision"],
                reference["per_fold"][fold]["versus_current"]["delta"]["average_precision"],
                rtol=0.0,
                atol=tolerance,
            )
        )
    if not all(checks.values()):
        raise ValueError(f"Fold-2 evaluator failed development reproduction: {checks}")
    return {"ok": True, "checks": checks}


def verify_receipt(
    protocol: dict[str, Any], protocol_path: Path
) -> tuple[Path, dict[str, Any]]:
    receipt_path = (ROOT / protocol["fold2_cache"]["receipt"]).resolve()
    if not receipt_path.is_file():
        raise FileNotFoundError("Frozen fold-2 extraction receipt is unavailable")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt["protocol_sha256"] != sha256(protocol_path):
        raise ValueError("Fold-2 receipt protocol identity mismatch")
    if receipt["extractor_sha256"] != protocol["extractor"]["sha256"]:
        raise ValueError("Fold-2 receipt extractor identity mismatch")
    features = (ROOT / protocol["fold2_cache"]["features"]).resolve()
    if receipt["features"]["path"] != protocol["fold2_cache"]["features"]:
        raise ValueError("Fold-2 receipt feature path mismatch")
    if sha256(features) != receipt["features"]["sha256"]:
        raise ValueError("Fold-2 feature hash mismatch")
    return features, receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL.as_posix())
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    protocol_path = (ROOT / args.protocol).resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    paths = verify_static_contract(protocol)
    if args.smoke:
        print(json.dumps(reproduce_development(protocol, paths), indent=2))
        return 0

    output = (ROOT / protocol["output"]["json"]).resolve()
    markdown = (ROOT / protocol["output"]["markdown"]).resolve()
    if output.exists() or markdown.exists():
        raise FileExistsError("Refusing to repeat the one-shot fold-2 evaluation")
    fold2_cache, receipt = verify_receipt(protocol, protocol_path)
    all_values = load_all_values(paths)
    source_encoded, source_names = align_features(paths["dofa_development"], all_values)
    target_encoded, target_names = align_fold_cache(fold2_cache, all_values, 2)
    if not np.array_equal(source_names, target_names):
        raise ValueError("Development and fold-2 DOFA schemas differ")
    source_rows = np.isin(all_values["folds"], SELECTION_FOLDS)
    target_rows = np.asarray(all_values["folds"]) == 2
    source_features, _ = select_features(
        source_encoded, source_names, protocol["candidate"]["feature_set"]
    )
    target_features, _ = select_features(
        target_encoded, target_names, protocol["candidate"]["feature_set"]
    )
    raw_scores = []
    for seed in map(int, protocol["candidate"]["projection_seeds"]):
        raw_scores.append(
            fit_predict_fold2(
                source_features,
                target_features,
                np.asarray(all_values["labels"])[source_rows],
                np.asarray(all_values["sensors"])[source_rows],
                np.asarray(all_values["sensors"])[target_rows],
                seed=seed,
                c_value=float(protocol["candidate"]["C"]),
            )
        )
        gc.collect()
    aggregate = mean_logit_probabilities(raw_scores)
    current = np.asarray(all_values["current"])[target_rows]
    primary = np.asarray(all_values["primary"])[target_rows]
    labels = np.asarray(all_values["labels"])[target_rows]
    sensors = np.asarray(all_values["sensors"])[target_rows]
    groups = np.asarray(all_values["groups"])[target_rows].astype(str)
    scores = protected_logit_blend(
        current,
        aggregate,
        gate=float(protocol["candidate"]["gate"]),
        weight=float(protocol["candidate"]["weight"]),
    )
    candidate_metrics = metric_summary(labels, scores, sensors)
    current_metrics = metric_summary(labels, current, sensors)
    primary_metrics = metric_summary(labels, primary, sensors)
    versus_current = comparison(candidate_metrics, current_metrics)
    versus_primary = comparison(candidate_metrics, primary_metrics)
    preserved = all(
        candidate_metrics[key] == current_metrics[key] for key in ("tp", "fp", "tn", "fn")
    )
    interval = ap_group_bootstrap(
        labels,
        current,
        scores,
        groups,
        replicates=int(protocol["bootstrap"]["replicates"]),
        seed=int(protocol["bootstrap"]["seed"]),
    )
    gates = protocol["gates"]
    checks = {
        "minimum_ap_delta": versus_current["delta"]["average_precision"]
        >= float(gates["minimum_ap_delta_vs_current"]),
        "recall_no_worse": versus_current["delta"]["recall_at_fpr_0_0713"] >= 0.0,
        "fpr_no_worse": candidate_metrics["false_positive_rate"]
        <= current_metrics["false_positive_rate"],
        "operating_counts_preserved": preserved,
        "each_sensor_ap_no_worse": min(
            versus_current["delta"]["sensor_average_precision"].values()
        )
        >= 0.0,
        "paired_site_ap_lower_positive": interval["lower"] > 0.0,
        "ap_vs_released_primary_positive": versus_primary["delta"]["average_precision"] > 0.0,
        "recall_vs_released_primary_positive": versus_primary["delta"][
            "recall_at_fpr_0_0713"
        ]
        > 0.0,
    }
    passed = all(checks.values())
    report = {
        "schema_version": 1,
        "scope": "one-shot fixed protected DOFA-v2 fold-2 confirmation",
        "status": "passed_fold2_confirmation" if passed else "rejected_on_fold2_confirmation",
        "decision": (
            "Authorize the fixed train-fitted DOFA-v2 fusion for source-disjoint confirmation."
            if passed
            else "Retire the fixed DOFA-v2 fusion before external or official-test evaluation."
        ),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate": protocol["candidate"],
        "rows": {
            "total": int(labels.size),
            "positive": int(labels.sum()),
            "negative": int(labels.size - labels.sum()),
            "physical_groups": int(len(np.unique(groups))),
        },
        "metrics": candidate_metrics,
        "versus_current": versus_current,
        "versus_released_primary": versus_primary,
        "operating_counts_preserved": preserved,
        "paired_group_bootstrap_ap_delta_vs_current": interval,
        "promotion_checks": checks,
        "all_promotion_gates_pass": passed,
        "provenance": {
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "protocol_sha256": sha256(protocol_path),
            "evaluator_sha256": sha256(Path(__file__).resolve()),
            "fold2_receipt_sha256": sha256(
                (ROOT / protocol["fold2_cache"]["receipt"]).resolve()
            ),
            "fold2_features_sha256": sha256(fold2_cache),
            "development_report_sha256": sha256(paths["development_report"]),
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
        },
        "extraction_receipt": receipt,
        "fold2_accessed": True,
        "folds_0_1_accessed_for_outcomes": False,
        "external_outcomes_accessed": False,
        "paper_test_outcomes_accessed": False,
    }
    write_json(output, report)
    write_markdown(markdown, report)
    print(
        json.dumps(
            {
                "ok": passed,
                "ap_delta_vs_current": versus_current["delta"]["average_precision"],
                "recall_delta_vs_current": versus_current["delta"][
                    "recall_at_fpr_0_0713"
                ],
                "ap_lower_vs_current": interval["lower"],
                "checks": checks,
            },
            indent=2,
        )
    )
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
