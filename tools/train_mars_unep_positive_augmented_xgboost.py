#!/usr/bin/env python3
"""Train a leakage-controlled MARS scene head with new UNEP positives."""

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
from train_mars_xgboost_scene_head import build_model  # noqa: E402


DEFAULT_PROTOCOL = Path("configs/mars_unep_positive_augmented_xgboost_protocol.json")
DEFAULT_ARTIFACT = Path(
    "EarthRemoteSensingRapidResponse/artifacts/mars_unep_positive_augmented_xgboost.joblib"
)
DEFAULT_JSON = Path(
    "reports/experiments/mars_unep_positive_augmented_xgboost.json"
)
DEFAULT_MARKDOWN = Path(
    "reports/experiments/MARS_UNEP_POSITIVE_AUGMENTED_XGBOOST.md"
)
SELECTION_FOLD = 2
CONFIRMATION_FOLDS = (0, 1)
FIT_FOLDS = (2, 3, 4)


def load_external(
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
    if values["labels"].size != expected_rows or not np.all(values["labels"] == 1):
        raise ValueError(f"{expected_role} must contain the frozen positive-only rows")
    augmented, augmented_names = augment_site_context(
        values["features"], names, values["groups"]
    )
    values["features"] = augmented
    values["augmented_names"] = augmented_names
    values["primary"] = values["features"][:, original_names.index("primary_connected_score")]
    return values


def current_scores(
    values: dict[str, Any], current_payload: dict[str, Any]
) -> np.ndarray:
    if values["augmented_names"] != current_payload["augmented_feature_names"]:
        raise ValueError("External augmented feature schema differs from current head")
    raw = current_payload["fitted"].predict_proba(values["features"])[:, 1]
    return blend_scores(values["primary"], raw, float(current_payload["blend_lambda"]))


def fit_augmented(
    original: dict[str, Any],
    auxiliary: dict[str, Any],
    fit: np.ndarray,
    multiplier: float,
    seed: int,
) -> Any:
    features = np.concatenate([original["features"][fit], auxiliary["features"]])
    labels = np.concatenate([original["labels"][fit], auxiliary["labels"]])
    weights = np.concatenate(
        [
            np.ones(int(np.count_nonzero(fit)), dtype=np.float64),
            np.full(auxiliary["labels"].size, multiplier, dtype=np.float64),
        ]
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


def original_checks(result: dict[str, Any], *, confirmation: bool) -> dict[str, bool]:
    delta = result["versus_current"]["delta"]
    checks = {
        "ap_delta_at_least_0_002": delta["average_precision"] >= 0.002,
        "sensor_noninferiority": min(delta["sensor_average_precision"].values()) >= -0.002,
        "recall_gate": (
            delta["recall_at_fpr_0_0713"] > 0.0
            if confirmation
            else delta["recall_at_fpr_0_0713"] >= 0.0
        ),
    }
    if confirmation:
        checks["paired_ap_lower_positive"] = (
            result["paired_group_bootstrap_ap_delta_vs_current"]["lower"] > 0.0
        )
    return checks


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


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    selected = report["selection"]["selected"]
    lines = [
        "# UNEP-positive augmented MARS XGBoost scene head",
        "",
        f"- Selected auxiliary multiplier: **{selected['auxiliary_multiplier']:.1f}**.",
        f"- Selected candidate blend: **{selected['candidate_blend']:.3f}**.",
        f"- Selection gates: **{'PASS' if selected['passed'] else 'FAIL'}**.",
        "",
        "| Partition | AP delta vs current | Recall delta at 7.13% FPR | AP 95% CI | Gates |",
        "|---|---:|---:|---:|---|",
    ]
    selection_delta = selected["versus_current"]["delta"]
    lines.append(
        f"| fold 2 selection | {selection_delta['average_precision']:+.5f} | "
        f"{selection_delta['recall_at_fpr_0_0713']:+.5f} | not used | "
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
    cache_contract = protocol["feature_caches"]
    cache_paths = {
        "inner": (ROOT / cache_contract["original_inner"]["path"]).resolve(),
        "fold0": (ROOT / cache_contract["original_fold0"]["path"]).resolve(),
        "fold1": (ROOT / cache_contract["original_fold1"]["path"]).resolve(),
    }
    for name, key in (("inner", "original_inner"), ("fold0", "original_fold0"), ("fold1", "original_fold1")):
        if sha256(cache_paths[name]) != cache_contract[key]["sha256"]:
            raise ValueError(f"Original {name} cache hash mismatch")
    score_path = (ROOT / cache_contract["original_scores"]["path"]).resolve()
    if sha256(score_path) != cache_contract["original_scores"]["sha256"]:
        raise ValueError("Original score cache hash mismatch")
    original = load_development(cache_paths, score_path)
    current_path = (ROOT / protocol["base_architecture"]["artifact"]).resolve()
    if sha256(current_path) != protocol["base_architecture"]["artifact_sha256"]:
        raise ValueError("Current scene-head artifact hash mismatch")
    current_payload = joblib.load(current_path)
    if float(current_payload["blend_lambda"]) != float(protocol["base_architecture"]["blend_lambda"]):
        raise ValueError("Current scene-head blend differs from protocol")
    if original["augmented_feature_names"] != current_payload["augmented_feature_names"]:
        raise ValueError("Original feature schema differs from current head")

    aux_contract = cache_contract["unep_auxiliary"]
    auxiliary = load_external(
        (ROOT / aux_contract["path"]).resolve(),
        aux_contract["sha256"],
        int(aux_contract["rows"]),
        "auxiliary_training",
        original["feature_names"],
    )
    if auxiliary["augmented_names"] != original["augmented_feature_names"]:
        raise ValueError("Auxiliary feature schema differs from original development")
    auxiliary["current"] = current_scores(auxiliary, current_payload)

    selection_fit = np.isin(original["folds"], (3, 4))
    selection_rows = original["folds"] == SELECTION_FOLD
    multipliers = protocol["candidate_family"]["auxiliary_positive_weight_multipliers"]
    blends = protocol["candidate_family"]["candidate_logit_blends"]
    candidates = []
    for multiplier in multipliers:
        model = fit_augmented(
            original, auxiliary, selection_fit, float(multiplier), 20260716 + int(multiplier)
        )
        raw = model.predict_proba(original["features"][selection_rows])[:, 1]
        for blend in blends:
            result = evaluate(original, selection_rows, raw, float(blend))
            result["auxiliary_multiplier"] = float(multiplier)
            result["candidate_blend"] = float(blend)
            result["checks"] = original_checks(result, confirmation=False)
            result["passed"] = all(result["checks"].values())
            delta = result["versus_current"]["delta"]
            result["rank"] = [
                int(result["passed"]),
                delta["recall_at_fpr_0_0713"],
                delta["average_precision"],
                -float(blend),
                -float(multiplier),
            ]
            candidates.append(result)
        print(json.dumps({"selection_multiplier_complete": multiplier}), flush=True)
    selected = max(candidates, key=lambda value: tuple(value["rank"]))
    selection_report = {
        "training_rows": int(selection_fit.sum()),
        "validation_rows": int(selection_rows.sum()),
        "candidate_summaries": [
            {
                "auxiliary_multiplier": value["auxiliary_multiplier"],
                "candidate_blend": value["candidate_blend"],
                "ap_delta": value["versus_current"]["delta"]["average_precision"],
                "recall_delta": value["versus_current"]["delta"]["recall_at_fpr_0_0713"],
                "passed": value["passed"],
            }
            for value in candidates
        ],
        "selected": compact_result(selected)
        | {
            "auxiliary_multiplier": selected["auxiliary_multiplier"],
            "candidate_blend": selected["candidate_blend"],
        },
    }

    confirmation: dict[str, Any] = {}
    passed = bool(selected["passed"])
    thresholds: list[float] = []
    if passed:
        confirmation_fit = np.isin(original["folds"], FIT_FOLDS)
        for fold in CONFIRMATION_FOLDS:
            rows = original["folds"] == fold
            model = fit_augmented(
                original,
                auxiliary,
                confirmation_fit,
                float(selected["auxiliary_multiplier"]),
                20260800 + fold,
            )
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
            result["checks"] = original_checks(result, confirmation=True)
            result["passed"] = all(result["checks"].values())
            confirmation[str(fold)] = compact_result(result)
            thresholds.append(float(result["candidate"]["operating_point"]["threshold"]))
            print(json.dumps({"confirmation_fold": fold, "passed": result["passed"]}), flush=True)
        passed = all(value["passed"] for value in confirmation.values())

    artifact_path = (ROOT / args.artifact).resolve()
    artifact_record = None
    external_confirmation = None
    if passed:
        final_model = fit_augmented(
            original,
            auxiliary,
            np.ones(original["labels"].shape, dtype=bool),
            float(selected["auxiliary_multiplier"]),
            20260900,
        )
        operational_threshold = max(thresholds)
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = artifact_path.with_suffix(artifact_path.suffix + ".tmp")
        joblib.dump(
            {
                "schema_version": 1,
                "kind": "mars_unep_positive_augmented_xgboost",
                "model": final_model,
                "auxiliary_multiplier": selected["auxiliary_multiplier"],
                "candidate_blend": selected["candidate_blend"],
                "base_score": "frozen v3 stronger OOF ExtraTrees scene head",
                "feature_names": original["augmented_feature_names"],
                "operational_scene_threshold": operational_threshold,
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
            "operational_scene_threshold": operational_threshold,
        }
        dev_contract = cache_contract["unep_development"]
        development = load_external(
            (ROOT / dev_contract["path"]).resolve(),
            dev_contract["sha256"],
            int(dev_contract["rows"]),
            "development",
            original["feature_names"],
        )
        development["current"] = current_scores(development, current_payload)
        raw = final_model.predict_proba(development["features"])[:, 1]
        scores = blend_scores(
            development["current"], raw, float(selected["candidate_blend"])
        )
        current_recall = float(np.mean(development["current"] >= operational_threshold))
        candidate_recall = float(np.mean(scores >= operational_threshold))
        external_confirmation = {
            "rows": int(scores.size),
            "groups": len(set(development["groups"].tolist())),
            "current_positive_recall": current_recall,
            "candidate_positive_recall": candidate_recall,
            "candidate_no_worse": candidate_recall >= current_recall,
            "development_accessed_after_original_gates": True,
        }
        passed = passed and external_confirmation["candidate_no_worse"]

    decision = (
        "Freeze the augmented scene head for exact paper-cache evaluation; dense masks remain unchanged."
        if passed
        else "Reject the augmented scene head before paper-cache evaluation."
    )
    report = {
        "schema_version": 1,
        "scope": "development-only UNEP positive augmentation; paper test not loaded",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection": selection_report,
        "confirmation": confirmation,
        "unep_development_confirmation": external_confirmation,
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
    print(json.dumps({"ok": passed, "decision": decision, "artifact": artifact_record}, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
