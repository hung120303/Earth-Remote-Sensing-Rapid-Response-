#!/usr/bin/env python3
"""Finalize a fixed joint MARS head under calibration-aware negative safety gates."""

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

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "EarthRemoteSensingRapidResponse"
for path in (MODEL_ROOT, ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from acquire_mars_metadata import sha256  # noqa: E402
from train_mars_cloudsen12_spatial_augmented_xgboost import load_external_negative  # noqa: E402
from train_mars_crossfold_bagged_scene_head import load_development  # noqa: E402
from train_mars_joint_external_augmented_xgboost import (  # noqa: E402
    fit_joint,
    positive_confirmation,
)
from train_mars_scene_ranker import blend_scores  # noqa: E402
from train_mars_unep_positive_augmented_xgboost import (  # noqa: E402
    current_scores,
    load_external as load_external_positive,
)


DEFAULT_PROTOCOL = Path("configs/mars_joint_calibration_aware_finalization_protocol.json")
DEFAULT_ARTIFACT = Path(
    "EarthRemoteSensingRapidResponse/artifacts/"
    "mars_joint_calibration_aware_xgboost.joblib"
)
DEFAULT_JSON = Path("reports/experiments/mars_joint_calibration_aware_finalization.json")
DEFAULT_MARKDOWN = Path("reports/experiments/MARS_JOINT_CALIBRATION_AWARE_FINALIZATION.md")


def logit(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=np.float64), 1e-8, 1.0 - 1e-8)
    return np.log(clipped) - np.log1p(-clipped)


def calibration_aware_negative_safety(
    current: np.ndarray,
    candidate: np.ndarray,
    current_threshold: float,
    candidate_threshold: float,
) -> dict[str, Any]:
    current = np.asarray(current, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    current_fp = int(np.count_nonzero(current >= current_threshold))
    candidate_fp = int(np.count_nonzero(candidate >= candidate_threshold))
    current_p95 = float(np.quantile(current, 0.95))
    candidate_p95 = float(np.quantile(candidate, 0.95))
    current_logit_margin = float(
        np.quantile(logit(current) - logit(np.asarray(current_threshold)), 0.95)
    )
    candidate_logit_margin = float(
        np.quantile(logit(candidate) - logit(np.asarray(candidate_threshold)), 0.95)
    )
    checks = {
        "false_positive_count_no_higher": candidate_fp <= current_fp,
        "raw_score_p95_no_higher": candidate_p95 <= current_p95,
        "logit_threshold_margin_p95_no_higher": (
            candidate_logit_margin <= current_logit_margin
        ),
    }
    return {
        "rows": int(current.size),
        "current_threshold": current_threshold,
        "candidate_threshold": candidate_threshold,
        "current_false_positives": current_fp,
        "candidate_false_positives": candidate_fp,
        "current_false_positive_rate": current_fp / current.size,
        "candidate_false_positive_rate": candidate_fp / candidate.size,
        "current_raw_score_p95": current_p95,
        "candidate_raw_score_p95": candidate_p95,
        "current_logit_threshold_margin_p95": current_logit_margin,
        "candidate_logit_threshold_margin_p95": candidate_logit_margin,
        "checks": checks,
        "passed": all(checks.values()),
    }


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    positive = report["unep_positive_development"]
    negative = report["cloudsen12_negative_development"]
    lines = [
        "# Calibration-aware finalization of the fixed joint MARS scene head",
        "",
        f"- Fixed candidate: positive weight **{report['fixed_candidate']['positive_multiplier']:.4f}**, negative weight **{report['fixed_candidate']['negative_multiplier']:.1f}**, blend **{report['fixed_candidate']['candidate_blend']:.2f}**.",
        f"- UNEP development recall: current **{positive['current_positive_recall']:.3f}**, candidate **{positive['candidate_positive_recall']:.3f}**.",
        f"- CloudSEN development false positives: current **{negative['current_false_positives']}/{negative['rows']}**, candidate **{negative['candidate_false_positives']}/{negative['rows']}**.",
        f"- CloudSEN raw-score p95: current **{negative['current_raw_score_p95']:.5f}**, candidate **{negative['candidate_raw_score_p95']:.5f}**.",
        f"- CloudSEN logit-margin p95: current **{negative['current_logit_threshold_margin_p95']:.5f}**, candidate **{negative['candidate_logit_threshold_margin_p95']:.5f}**.",
        "",
        report["decision"],
        "",
    ]
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
    source = protocol["source_experiment"]
    source_path = (ROOT / source["path"]).resolve()
    if sha256(source_path) != source["sha256"]:
        raise ValueError("Source experiment hash mismatch")
    source_report = json.loads(source_path.read_text(encoding="utf-8"))
    fixed = protocol["fixed_candidate"]
    selected = source_report["selection"]["selected"]
    for key in ("positive_multiplier", "negative_multiplier", "candidate_blend"):
        if float(selected[key]) != float(fixed[key]):
            raise ValueError(f"Fixed candidate {key} differs from source experiment")
    if not selected["passed"] or not all(
        result["passed"] for result in source_report["confirmation"].values()
    ):
        raise ValueError("Source candidate did not pass every original-development gate")
    if source_report["artifact"] is not None:
        raise ValueError("Source experiment unexpectedly wrote an artifact")

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
        raise ValueError("Original scores hash mismatch")
    original = load_development(
        {"inner": paths["original_inner"], "fold0": paths["original_fold0"], "fold1": paths["original_fold1"]},
        score_path,
    )
    current_path = (ROOT / protocol["base_architecture"]["artifact"]).resolve()
    if sha256(current_path) != protocol["base_architecture"]["artifact_sha256"]:
        raise ValueError("Current architecture hash mismatch")
    current_payload = joblib.load(current_path)

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
    final_model = fit_joint(
        original,
        positive,
        negative,
        np.ones(original["labels"].shape, dtype=bool),
        float(fixed["positive_multiplier"]),
        float(fixed["negative_multiplier"]),
        int(fixed["final_fit_seed"]),
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
    blend = float(fixed["candidate_blend"])
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
    current_threshold = max(
        float(value["current"]["operating_point"]["threshold"])
        for value in source_report["confirmation"].values()
    )
    candidate_threshold = max(
        float(value["candidate"]["operating_point"]["threshold"])
        for value in source_report["confirmation"].values()
    )
    positive_result = positive_confirmation(
        positive_dev["current"], positive_candidate, current_threshold, candidate_threshold
    )
    positive_result["groups"] = len(set(positive_dev["groups"].tolist()))
    negative_result = calibration_aware_negative_safety(
        negative_dev["current"], negative_candidate, current_threshold, candidate_threshold
    )
    negative_result["groups"] = len(set(negative_dev["groups"].tolist()))
    passed = bool(positive_result["candidate_no_worse"] and negative_result["passed"])

    artifact_path = (ROOT / args.artifact).resolve()
    artifact_record = None
    if passed:
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = artifact_path.with_suffix(artifact_path.suffix + ".tmp")
        joblib.dump(
            {
                "schema_version": 1,
                "kind": "mars_joint_calibration_aware_xgboost",
                "model": final_model,
                **fixed,
                "base_score": "frozen v3 stronger OOF ExtraTrees scene head",
                "feature_names": original["augmented_feature_names"],
                "operational_scene_threshold": candidate_threshold,
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
            "operational_scene_threshold": candidate_threshold,
        }
    decision = (
        "Freeze the fixed joint head for fresh CloudSEN-test and exact-paper evaluation."
        if passed
        else "Reject the fixed joint head before fresh external or paper evaluation."
    )
    report = {
        "schema_version": 1,
        "scope": "calibration-aware finalization; no fresh external or paper test loaded",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "fixed_candidate": fixed,
        "source_experiment": source,
        "unep_positive_development": positive_result,
        "cloudsen12_negative_development": negative_result,
        "all_finalization_gates_pass": passed,
        "decision": decision,
        "artifact": artifact_record,
        "provenance": {
            "protocol_sha256": sha256(protocol_path),
            "script_sha256": sha256(Path(__file__).resolve()),
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
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
