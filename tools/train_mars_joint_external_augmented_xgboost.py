#!/usr/bin/env python3
"""Fit a leakage-controlled MARS scene head with joint external positives/negatives."""

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

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "EarthRemoteSensingRapidResponse"
for path in (MODEL_ROOT, ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from acquire_mars_metadata import sha256  # noqa: E402
from train_mars_cloudsen12_spatial_augmented_xgboost import (  # noqa: E402
    compact_result,
    confirmation_checks,
    evaluate,
    load_external_negative,
    negative_confirmation,
    selection_checks,
)
from train_mars_crossfold_bagged_scene_head import load_development  # noqa: E402
from train_mars_oof_scene_ensemble_v2 import ap_group_bootstrap  # noqa: E402
from train_mars_scene_ranker import blend_scores  # noqa: E402
from train_mars_unep_positive_augmented_xgboost import (  # noqa: E402
    current_scores,
    load_external as load_external_positive,
)
from train_mars_xgboost_scene_head import build_model  # noqa: E402


DEFAULT_PROTOCOL = Path("configs/mars_joint_external_augmented_xgboost_protocol.json")
DEFAULT_ARTIFACT = Path(
    "EarthRemoteSensingRapidResponse/artifacts/"
    "mars_joint_external_augmented_xgboost.joblib"
)
DEFAULT_JSON = Path("reports/experiments/mars_joint_external_augmented_xgboost.json")
DEFAULT_MARKDOWN = Path("reports/experiments/MARS_JOINT_EXTERNAL_AUGMENTED_XGBOOST.md")
INNER_FOLDS = (2, 3, 4)
CONFIRMATION_FOLDS = (0, 1)


def joint_training_arrays(
    original: dict[str, Any],
    positive: dict[str, Any],
    negative: dict[str, Any],
    fit: np.ndarray,
    positive_multiplier: float,
    negative_multiplier: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = int(np.count_nonzero(fit))
    features = np.concatenate(
        [original["features"][fit], positive["features"], negative["features"]]
    )
    labels = np.concatenate(
        [original["labels"][fit], positive["labels"], negative["labels"]]
    )
    weights = np.concatenate(
        [
            np.ones(count, dtype=np.float64),
            np.full(positive["labels"].size, positive_multiplier, dtype=np.float64),
            np.full(negative["labels"].size, negative_multiplier, dtype=np.float64),
        ]
    )
    return features, labels, weights


def fit_joint(
    original: dict[str, Any],
    positive: dict[str, Any],
    negative: dict[str, Any],
    fit: np.ndarray,
    positive_multiplier: float,
    negative_multiplier: float,
    seed: int,
) -> Any:
    features, labels, weights = joint_training_arrays(
        original,
        positive,
        negative,
        fit,
        positive_multiplier,
        negative_multiplier,
    )
    model = build_model(
        {
            "name": "depth3_lr004",
            "n_estimators": 600,
            "max_depth": 3,
            "learning_rate": 0.04,
            "min_child_weight": 10.0,
        },
        seed=seed,
    )
    model.fit(features, labels, sample_weight=weights, verbose=False)
    return model


def positive_confirmation(
    current: np.ndarray,
    candidate: np.ndarray,
    current_threshold: float,
    candidate_threshold: float,
) -> dict[str, Any]:
    current_recall = float(np.mean(current >= current_threshold))
    candidate_recall = float(np.mean(candidate >= candidate_threshold))
    return {
        "rows": int(current.size),
        "current_threshold": current_threshold,
        "candidate_threshold": candidate_threshold,
        "current_positive_recall": current_recall,
        "candidate_positive_recall": candidate_recall,
        "candidate_no_worse": candidate_recall >= current_recall,
    }


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    selected = report["selection"]["selected"]
    lines = [
        "# Joint external positive/negative augmented MARS scene head",
        "",
        f"- Positive multiplier: **{selected['positive_multiplier']:.4f}**.",
        f"- Negative multiplier: **{selected['negative_multiplier']:.4f}**.",
        f"- Complement blend: **{selected['candidate_blend']:.3f}**.",
        "",
        "| Partition | AP delta | Recall delta at 7.13% FPR | AP 95% CI | Gates |",
        "|---|---:|---:|---:|---|",
    ]
    inner_delta = selected["versus_current"]["delta"]
    inner_interval = selected["paired_group_bootstrap_ap_delta_vs_current"]
    lines.append(
        f"| folds 2/3/4 selection | {inner_delta['average_precision']:+.5f} | "
        f"{inner_delta['recall_at_fpr_0_0713']:+.5f} | "
        f"[{inner_interval['lower']:+.5f}, {inner_interval['upper']:+.5f}] | "
        f"{'PASS' if selected['passed'] else 'FAIL'} |"
    )
    for fold, result in report.get("confirmation", {}).items():
        delta = result["versus_current"]["delta"]
        interval = result["paired_group_bootstrap_ap_delta_vs_current"]
        lines.append(
            f"| fold {fold} confirmation | {delta['average_precision']:+.5f} | "
            f"{delta['recall_at_fpr_0_0713']:+.5f} | "
            f"[{interval['lower']:+.5f}, {interval['upper']:+.5f}] | "
            f"{'PASS' if result['passed'] else 'FAIL'} |"
        )
    lines.extend(["", report["decision"], ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL.as_posix())
    parser.add_argument("--artifact", default=DEFAULT_ARTIFACT.as_posix())
    parser.add_argument("--output-json", default=DEFAULT_JSON.as_posix())
    parser.add_argument("--output-markdown", default=DEFAULT_MARKDOWN.as_posix())
    args = parser.parse_args()
    protocol_path = (ROOT / args.protocol).resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    caches = protocol["feature_caches"]
    paths = {
        key: (ROOT / caches[key]["path"]).resolve()
        for key in ("original_inner", "original_fold0", "original_fold1")
    }
    for key, path in paths.items():
        if sha256(path) != caches[key]["sha256"]:
            raise ValueError(f"{key} cache hash mismatch")
    score_path = (ROOT / caches["original_scores"]["path"]).resolve()
    if sha256(score_path) != caches["original_scores"]["sha256"]:
        raise ValueError("Original score cache hash mismatch")
    original = load_development(
        {"inner": paths["original_inner"], "fold0": paths["original_fold0"], "fold1": paths["original_fold1"]},
        score_path,
    )
    current_path = (ROOT / protocol["base_architecture"]["artifact"]).resolve()
    if sha256(current_path) != protocol["base_architecture"]["artifact_sha256"]:
        raise ValueError("Current scene-head artifact hash mismatch")
    current_payload = joblib.load(current_path)
    if float(current_payload["blend_lambda"]) != float(
        protocol["base_architecture"]["blend_lambda"]
    ):
        raise ValueError("Current scene-head blend differs from protocol")

    positive_contract = caches["unep_positive_auxiliary"]
    positive = load_external_positive(
        (ROOT / positive_contract["path"]).resolve(),
        positive_contract["sha256"],
        int(positive_contract["rows"]),
        "auxiliary_training",
        original["feature_names"],
    )
    negative_contract = caches["cloudsen12_negative_auxiliary"]
    negative = load_external_negative(
        (ROOT / negative_contract["path"]).resolve(),
        negative_contract["sha256"],
        int(negative_contract["rows"]),
        "auxiliary_training",
        original["feature_names"],
    )
    if (
        positive["augmented_names"] != original["augmented_feature_names"]
        or negative["augmented_names"] != original["augmented_feature_names"]
    ):
        raise ValueError("External feature schema differs from original development")

    inner_rows = np.isin(original["folds"], INNER_FOLDS)
    family = protocol["candidate_family"]
    candidates: list[dict[str, Any]] = []
    for positive_value in family["positive_weight_multipliers"]:
        for negative_value in family["negative_weight_multipliers"]:
            positive_multiplier = float(positive_value)
            negative_multiplier = float(negative_value)
            raw = np.full(original["labels"].shape, np.nan, dtype=np.float64)
            for holdout in INNER_FOLDS:
                fit = inner_rows & (original["folds"] != holdout)
                held = original["folds"] == holdout
                seed = (
                    20260717
                    + holdout
                    + int(round(positive_multiplier * 100))
                    + int(round(negative_multiplier * 1000))
                )
                model = fit_joint(
                    original,
                    positive,
                    negative,
                    fit,
                    positive_multiplier,
                    negative_multiplier,
                    seed,
                )
                raw[held] = model.predict_proba(original["features"][held])[:, 1]
            for blend_value in family["candidate_logit_blends"]:
                blend = float(blend_value)
                result = evaluate(original, inner_rows, raw[inner_rows], blend)
                result["checks"] = selection_checks(result)
                per_fold = {}
                per_fold_ap = []
                for fold in INNER_FOLDS:
                    rows = original["folds"] == fold
                    local = evaluate(original, rows, raw[rows], blend)
                    delta = local["versus_current"]["delta"]
                    local["checks"] = {
                        "ap_strictly_positive": delta["average_precision"] > 0.0,
                        "recall_nonnegative": delta["recall_at_fpr_0_0713"] >= 0.0,
                    }
                    local["passed"] = all(local["checks"].values())
                    per_fold[str(fold)] = compact_result(local)
                    per_fold_ap.append(float(delta["average_precision"]))
                result["per_fold"] = per_fold
                result["passed"] = all(result["checks"].values()) and all(
                    value["passed"] for value in per_fold.values()
                )
                delta = result["versus_current"]["delta"]
                result.update(
                    {
                        "positive_multiplier": positive_multiplier,
                        "negative_multiplier": negative_multiplier,
                        "candidate_blend": blend,
                        "rank": [
                            int(result["passed"]),
                            min(per_fold_ap),
                            delta["recall_at_fpr_0_0713"],
                            delta["average_precision"],
                            -blend,
                            -positive_multiplier,
                        ],
                    }
                )
                candidates.append(result)
            print(
                json.dumps(
                    {
                        "joint_weight_complete": [
                            positive_multiplier,
                            negative_multiplier,
                        ]
                    }
                ),
                flush=True,
            )

    selected = max(candidates, key=lambda value: tuple(value["rank"]))
    selected["paired_group_bootstrap_ap_delta_vs_current"] = ap_group_bootstrap(
        original["labels"][inner_rows],
        original["current"][inner_rows],
        selected["candidate_scores"],
        original["groups"][inner_rows],
        replicates=int(protocol["bootstrap"]["replicates"]),
        seed=int(protocol["bootstrap"]["seed"]),
    )
    selected["checks"]["paired_ap_lower_positive"] = (
        selected["paired_group_bootstrap_ap_delta_vs_current"]["lower"] > 0.0
    )
    selected["passed"] = selected["passed"] and selected["checks"][
        "paired_ap_lower_positive"
    ]
    selection_report = {
        "candidate_summaries": [
            {
                "positive_multiplier": value["positive_multiplier"],
                "negative_multiplier": value["negative_multiplier"],
                "candidate_blend": value["candidate_blend"],
                "ap_delta": value["versus_current"]["delta"]["average_precision"],
                "recall_delta": value["versus_current"]["delta"]["recall_at_fpr_0_0713"],
                "worst_fold_ap_delta": min(
                    fold["versus_current"]["delta"]["average_precision"]
                    for fold in value["per_fold"].values()
                ),
                "passed_before_bootstrap": bool(value["passed"]),
            }
            for value in candidates
        ],
        "selected": compact_result(selected)
        | {
            "positive_multiplier": selected["positive_multiplier"],
            "negative_multiplier": selected["negative_multiplier"],
            "candidate_blend": selected["candidate_blend"],
            "per_fold": selected["per_fold"],
        },
    }

    confirmation: dict[str, Any] = {}
    candidate_thresholds: list[float] = []
    current_thresholds: list[float] = []
    passed = bool(selected["passed"])
    if passed:
        model = fit_joint(
            original,
            positive,
            negative,
            inner_rows,
            float(selected["positive_multiplier"]),
            float(selected["negative_multiplier"]),
            20260817,
        )
        for fold in CONFIRMATION_FOLDS:
            rows = original["folds"] == fold
            raw = model.predict_proba(original["features"][rows])[:, 1]
            result = evaluate(original, rows, raw, float(selected["candidate_blend"]))
            result["paired_group_bootstrap_ap_delta_vs_current"] = ap_group_bootstrap(
                original["labels"][rows],
                original["current"][rows],
                result["candidate_scores"],
                original["groups"][rows],
                replicates=int(protocol["bootstrap"]["replicates"]),
                seed=int(protocol["bootstrap"]["seed"]) + fold,
            )
            result["checks"] = confirmation_checks(result)
            result["passed"] = all(result["checks"].values())
            confirmation[str(fold)] = compact_result(result)
            candidate_thresholds.append(float(result["candidate"]["operating_point"]["threshold"]))
            current_thresholds.append(float(result["current"]["operating_point"]["threshold"]))
        passed = all(value["passed"] for value in confirmation.values())

    positive_development = None
    negative_development = None
    final_model = None
    if passed:
        final_model = fit_joint(
            original,
            positive,
            negative,
            np.ones(original["labels"].shape, dtype=bool),
            float(selected["positive_multiplier"]),
            float(selected["negative_multiplier"]),
            20260917,
        )
        positive_dev_contract = caches["unep_positive_development"]
        positive_dev = load_external_positive(
            (ROOT / positive_dev_contract["path"]).resolve(),
            positive_dev_contract["sha256"],
            int(positive_dev_contract["rows"]),
            "development",
            original["feature_names"],
        )
        negative_dev_contract = caches["cloudsen12_negative_development"]
        negative_dev = load_external_negative(
            (ROOT / negative_dev_contract["path"]).resolve(),
            negative_dev_contract["sha256"],
            int(negative_dev_contract["rows"]),
            "development",
            original["feature_names"],
        )
        positive_dev["current"] = current_scores(positive_dev, current_payload)
        negative_dev["current"] = current_scores(negative_dev, current_payload)
        blend = float(selected["candidate_blend"])
        positive_candidate = blend_scores(
            positive_dev["current"],
            final_model.predict_proba(positive_dev["features"])[:, 1],
            blend,
        )
        negative_candidate = blend_scores(
            negative_dev["current"],
            final_model.predict_proba(negative_dev["features"])[:, 1],
            blend,
        )
        positive_development = positive_confirmation(
            positive_dev["current"],
            positive_candidate,
            max(current_thresholds),
            max(candidate_thresholds),
        )
        positive_development["groups"] = len(set(positive_dev["groups"].tolist()))
        negative_development = negative_confirmation(
            negative_dev["current"],
            negative_candidate,
            max(current_thresholds),
            max(candidate_thresholds),
        )
        negative_development["groups"] = len(set(negative_dev["groups"].tolist()))
        passed = bool(
            positive_development["candidate_no_worse"]
            and negative_development["passed"]
        )

    artifact_path = (ROOT / args.artifact).resolve()
    artifact_record = None
    if passed and final_model is not None:
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = artifact_path.with_suffix(artifact_path.suffix + ".tmp")
        joblib.dump(
            {
                "schema_version": 1,
                "kind": "mars_joint_external_augmented_xgboost",
                "model": final_model,
                "positive_multiplier": selected["positive_multiplier"],
                "negative_multiplier": selected["negative_multiplier"],
                "candidate_blend": selected["candidate_blend"],
                "base_score": "frozen v3 stronger OOF ExtraTrees scene head",
                "feature_names": original["augmented_feature_names"],
                "operational_scene_threshold": max(candidate_thresholds),
                "protocol_sha256": sha256(protocol_path),
            },
            temporary,
            compress=3,
        )
        os.replace(temporary, artifact_path)
        artifact_record = {
            "path": artifact_path.relative_to(ROOT).as_posix(),
            "bytes": artifact_path.stat().st_size,
            "sha256": sha256(artifact_path),
            "tracked": False,
            "operational_scene_threshold": max(candidate_thresholds),
        }

    decision = (
        "Freeze the joint external-data scene head for exact paper-cache replay."
        if passed
        else "Reject the joint external-data scene head before paper-cache replay."
    )
    report = {
        "schema_version": 1,
        "scope": "development-only joint external augmentation; paper test not loaded",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection": selection_report,
        "confirmation": confirmation,
        "unep_positive_development_confirmation": positive_development,
        "cloudsen12_negative_development_confirmation": negative_development,
        "all_promotion_gates_pass": passed,
        "decision": decision,
        "artifact": artifact_record,
        "provenance": {
            "protocol_sha256": sha256(protocol_path),
            "script_sha256": sha256(Path(__file__).resolve()),
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "joblib": joblib.__version__,
        },
    }
    output_json = (ROOT / args.output_json).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_json.with_suffix(output_json.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output_json)
    write_markdown((ROOT / args.output_markdown).resolve(), report)
    print(json.dumps({"ok": passed, "decision": decision, "artifact": artifact_record}))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
