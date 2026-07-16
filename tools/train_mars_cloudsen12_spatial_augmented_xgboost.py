#!/usr/bin/env python3
"""Train a leakage-controlled MARS scene complement with spatial clear negatives."""

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
from train_mars_context_scene_ranker import augment_site_context  # noqa: E402
from train_mars_crossfold_bagged_scene_head import load_development  # noqa: E402
from train_mars_oof_scene_ensemble_v2 import ap_group_bootstrap  # noqa: E402
from train_mars_scene_ranker import blend_scores, comparison, metric_summary  # noqa: E402
from train_mars_unep_positive_augmented_xgboost import (  # noqa: E402
    current_scores,
    fit_augmented,
)


DEFAULT_PROTOCOL = Path(
    "configs/mars_cloudsen12_spatial_augmented_xgboost_protocol.json"
)
DEFAULT_ARTIFACT = Path(
    "EarthRemoteSensingRapidResponse/artifacts/"
    "mars_cloudsen12_spatial_augmented_xgboost.joblib"
)
DEFAULT_JSON = Path(
    "reports/experiments/mars_cloudsen12_spatial_augmented_xgboost.json"
)
DEFAULT_MARKDOWN = Path(
    "reports/experiments/MARS_CLOUDSEN12_SPATIAL_AUGMENTED_XGBOOST.md"
)
INNER_FOLDS = (2, 3, 4)
CONFIRMATION_FOLDS = (0, 1)


def load_external_negative(
    path: Path,
    expected_hash: str,
    expected_rows: int,
    expected_role: str,
    original_names: list[str],
) -> dict[str, Any]:
    if sha256(path) != expected_hash:
        raise ValueError(f"{expected_role} feature cache hash mismatch")
    with np.load(path, allow_pickle=False) as cache:
        names = cache["feature_names"].astype(str)
        values = {
            "features": cache["features"].astype(np.float64),
            "labels": cache["labels"].astype(np.uint8),
            "sensors": cache["sensors"].astype(np.uint8),
            "sample_ids": cache["sample_ids"].astype(str),
            "groups": cache["groups"].astype(str),
            "role": str(cache["research_role"].item()),
        }
    if names.tolist() != original_names or values["role"] != expected_role:
        raise ValueError(f"{expected_role} feature schema or role mismatch")
    if values["labels"].size != expected_rows or values["labels"].any():
        raise ValueError(f"{expected_role} must contain frozen negative-only rows")
    if len(set(values["sample_ids"].tolist())) != expected_rows:
        raise ValueError(f"{expected_role} contains duplicate samples")
    augmented, augmented_names = augment_site_context(
        values["features"], names, values["groups"]
    )
    values["features"] = augmented
    values["augmented_names"] = augmented_names
    values["primary"] = values["features"][:, original_names.index("primary_connected_score")]
    return values


def evaluate(
    original: dict[str, Any], rows: np.ndarray, raw: np.ndarray, blend: float
) -> dict[str, Any]:
    scores = blend_scores(original["current"][rows], raw, blend)
    candidate = metric_summary(
        original["labels"][rows], scores, original["sensors"][rows]
    )
    current = metric_summary(
        original["labels"][rows],
        original["current"][rows],
        original["sensors"][rows],
    )
    return {
        "candidate_scores": scores,
        "candidate": candidate,
        "current": current,
        "versus_current": comparison(candidate, current),
    }


def selection_checks(result: dict[str, Any]) -> dict[str, bool]:
    delta = result["versus_current"]["delta"]
    return {
        "ap_delta_at_least_0_002": delta["average_precision"] >= 0.002,
        "recall_nonnegative": delta["recall_at_fpr_0_0713"] >= 0.0,
        "sensor_noninferiority": min(delta["sensor_average_precision"].values()) >= -0.002,
    }


def confirmation_checks(result: dict[str, Any]) -> dict[str, bool]:
    delta = result["versus_current"]["delta"]
    return {
        "ap_delta_at_least_0_002": delta["average_precision"] >= 0.002,
        "recall_strictly_positive": delta["recall_at_fpr_0_0713"] > 0.0,
        "sensor_noninferiority": min(delta["sensor_average_precision"].values()) >= -0.002,
        "paired_ap_lower_positive": (
            result["paired_group_bootstrap_ap_delta_vs_current"]["lower"] > 0.0
        ),
    }


def compact_result(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate": result["candidate"],
        "current": result["current"],
        "versus_current": result["versus_current"],
        **(
            {
                "paired_group_bootstrap_ap_delta_vs_current": result[
                    "paired_group_bootstrap_ap_delta_vs_current"
                ]
            }
            if "paired_group_bootstrap_ap_delta_vs_current" in result
            else {}
        ),
        "checks": result["checks"],
        "passed": result["passed"],
    }


def negative_confirmation(
    current: np.ndarray,
    candidate: np.ndarray,
    current_threshold: float,
    candidate_threshold: float,
) -> dict[str, Any]:
    current_margin = current - current_threshold
    candidate_margin = candidate - candidate_threshold
    current_false_positives = int(np.count_nonzero(current >= current_threshold))
    candidate_false_positives = int(np.count_nonzero(candidate >= candidate_threshold))
    current_p95 = float(np.quantile(current_margin, 0.95))
    candidate_p95 = float(np.quantile(candidate_margin, 0.95))
    checks = {
        "false_positive_count_no_higher": candidate_false_positives <= current_false_positives,
        "threshold_margin_p95_no_higher": candidate_p95 <= current_p95,
    }
    return {
        "rows": int(current.size),
        "current_threshold": current_threshold,
        "candidate_threshold": candidate_threshold,
        "current_false_positives": current_false_positives,
        "candidate_false_positives": candidate_false_positives,
        "current_false_positive_rate": current_false_positives / current.size,
        "candidate_false_positive_rate": candidate_false_positives / candidate.size,
        "current_threshold_margin_p95": current_p95,
        "candidate_threshold_margin_p95": candidate_p95,
        "checks": checks,
        "passed": all(checks.values()),
    }


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    selected = report["selection"]["selected"]
    lines = [
        "# CloudSEN12+ spatial-negative augmented MARS scene head",
        "",
        f"- Selected negative multiplier: **{selected['auxiliary_multiplier']:.2f}**.",
        f"- Selected complement blend: **{selected['candidate_blend']:.3f}**.",
        f"- Cross-fitted folds 2/3/4 gates: **{'PASS' if selected['passed'] else 'FAIL'}**.",
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
    external = report.get("cloudsen12_development_confirmation")
    if external is not None:
        lines.extend(
            [
                "",
                f"CloudSEN12+ development false positives: current {external['current_false_positives']}/{external['rows']}, candidate {external['candidate_false_positives']}/{external['rows']}; gate **{'PASS' if external['passed'] else 'FAIL'}**.",
            ]
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
        {
            "inner": paths["original_inner"],
            "fold0": paths["original_fold0"],
            "fold1": paths["original_fold1"],
        },
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

    auxiliary_contract = caches["cloudsen12_auxiliary"]
    auxiliary = load_external_negative(
        (ROOT / auxiliary_contract["path"]).resolve(),
        auxiliary_contract["sha256"],
        int(auxiliary_contract["rows"]),
        "auxiliary_training",
        original["feature_names"],
    )
    if auxiliary["augmented_names"] != original["augmented_feature_names"]:
        raise ValueError("Auxiliary feature schema differs from original development")

    inner_rows = np.isin(original["folds"], INNER_FOLDS)
    candidates = []
    raw_store: dict[float, np.ndarray] = {}
    family = protocol["candidate_family"]
    for multiplier_value in family["auxiliary_negative_weight_multipliers"]:
        multiplier = float(multiplier_value)
        raw = np.full(original["labels"].shape, np.nan, dtype=np.float64)
        for holdout in INNER_FOLDS:
            fit = inner_rows & (original["folds"] != holdout)
            held = original["folds"] == holdout
            model = fit_augmented(
                original, auxiliary, fit, multiplier, 20260716 + holdout + int(multiplier * 10)
            )
            raw[held] = model.predict_proba(original["features"][held])[:, 1]
        raw_store[multiplier] = raw
        for blend_value in family["candidate_logit_blends"]:
            blend = float(blend_value)
            result = evaluate(original, inner_rows, raw[inner_rows], blend)
            result["checks"] = selection_checks(result)
            per_fold = {}
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
            result["per_fold"] = per_fold
            result["passed"] = all(result["checks"].values()) and all(
                value["passed"] for value in per_fold.values()
            )
            delta = result["versus_current"]["delta"]
            result.update(
                {
                    "auxiliary_multiplier": multiplier,
                    "candidate_blend": blend,
                    "rank": [
                        int(result["passed"]),
                        delta["recall_at_fpr_0_0713"],
                        delta["average_precision"],
                        -blend,
                        -multiplier,
                    ],
                }
            )
            candidates.append(result)
        print(json.dumps({"inner_multiplier_complete": multiplier}), flush=True)

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
                "auxiliary_multiplier": value["auxiliary_multiplier"],
                "candidate_blend": value["candidate_blend"],
                "ap_delta": value["versus_current"]["delta"]["average_precision"],
                "recall_delta": value["versus_current"]["delta"][
                    "recall_at_fpr_0_0713"
                ],
                "passed_before_bootstrap": all(value["checks"].values()),
            }
            for value in candidates
        ],
        "selected": compact_result(selected)
        | {
            "auxiliary_multiplier": selected["auxiliary_multiplier"],
            "candidate_blend": selected["candidate_blend"],
            "per_fold": selected["per_fold"],
        },
    }

    confirmation: dict[str, Any] = {}
    candidate_thresholds: list[float] = []
    current_thresholds: list[float] = []
    passed = bool(selected["passed"])
    if passed:
        fit = inner_rows
        model = fit_augmented(
            original,
            auxiliary,
            fit,
            float(selected["auxiliary_multiplier"]),
            20260800,
        )
        for fold in CONFIRMATION_FOLDS:
            rows = original["folds"] == fold
            raw = model.predict_proba(original["features"][rows])[:, 1]
            result = evaluate(
                original, rows, raw, float(selected["candidate_blend"])
            )
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
            candidate_thresholds.append(
                float(result["candidate"]["operating_point"]["threshold"])
            )
            current_thresholds.append(
                float(result["current"]["operating_point"]["threshold"])
            )
        passed = all(value["passed"] for value in confirmation.values())

    external_confirmation = None
    final_model = None
    if passed:
        final_model = fit_augmented(
            original,
            auxiliary,
            np.ones(original["labels"].shape, dtype=bool),
            float(selected["auxiliary_multiplier"]),
            20260900,
        )
        development_contract = caches["cloudsen12_development"]
        development = load_external_negative(
            (ROOT / development_contract["path"]).resolve(),
            development_contract["sha256"],
            int(development_contract["rows"]),
            "development",
            original["feature_names"],
        )
        development["current"] = current_scores(development, current_payload)
        raw = final_model.predict_proba(development["features"])[:, 1]
        candidate = blend_scores(
            development["current"], raw, float(selected["candidate_blend"])
        )
        external_confirmation = negative_confirmation(
            development["current"],
            candidate,
            max(current_thresholds),
            max(candidate_thresholds),
        )
        external_confirmation["groups"] = len(set(development["groups"].tolist()))
        external_confirmation["accessed_after_original_gates"] = True
        passed = bool(external_confirmation["passed"])

    artifact_path = (ROOT / args.artifact).resolve()
    artifact_record = None
    if passed and final_model is not None:
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = artifact_path.with_suffix(artifact_path.suffix + ".tmp")
        joblib.dump(
            {
                "schema_version": 1,
                "kind": "mars_cloudsen12_spatial_augmented_xgboost",
                "model": final_model,
                "auxiliary_multiplier": selected["auxiliary_multiplier"],
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
        "Freeze the spatial-negative scene complement for exact paper-cache replay."
        if passed
        else "Reject the spatial-negative scene complement before paper-cache replay."
    )
    report = {
        "schema_version": 1,
        "scope": "development-only CloudSEN12+ spatial-negative augmentation; paper test not loaded",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection": selection_report,
        "confirmation": confirmation,
        "cloudsen12_development_confirmation": external_confirmation,
        "all_promotion_gates_pass": passed,
        "decision": decision,
        "artifact": artifact_record,
        "provenance": {
            "protocol_sha256": sha256(protocol_path),
            "script_sha256": sha256(Path(__file__).resolve()),
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
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
