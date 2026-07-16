#!/usr/bin/env python3
"""One-shot safety comparison of frozen scene heads on fresh CloudSEN12+ negatives."""

from __future__ import annotations

import argparse
import json
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
from train_mars_context_scene_ranker import augment_site_context  # noqa: E402
from train_mars_scene_ranker import blend_scores  # noqa: E402
from train_mars_unep_positive_augmented_xgboost import current_scores  # noqa: E402


DEFAULT_PROTOCOL = Path("configs/cloudsen12_fresh_test_scene_head_evaluation_protocol.json")


def logit(values: np.ndarray | float) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=np.float64), 1e-8, 1.0 - 1e-8)
    return np.log(clipped) - np.log1p(-clipped)


def score_summary(scores: np.ndarray, threshold: float) -> dict[str, Any]:
    values = np.asarray(scores, dtype=np.float64)
    margins = logit(values) - logit(threshold)
    return {
        "rows": int(values.size),
        "threshold": float(threshold),
        "false_positives": int(np.count_nonzero(values >= threshold)),
        "false_positive_rate": float(np.mean(values >= threshold)),
        "raw_score": {
            name: float(np.quantile(values, quantile))
            for name, quantile in (("p50", 0.50), ("p90", 0.90), ("p95", 0.95), ("p99", 0.99))
        }
        | {"max": float(np.max(values))},
        "logit_threshold_margin": {
            name: float(np.quantile(margins, quantile))
            for name, quantile in (("p50", 0.50), ("p90", 0.90), ("p95", 0.95), ("p99", 0.99))
        }
        | {"max": float(np.max(margins))},
    }


def compare_stratum(
    current: np.ndarray,
    candidate: np.ndarray,
    mask: np.ndarray,
    current_threshold: float,
    candidate_threshold: float,
) -> dict[str, Any]:
    current_result = score_summary(current[mask], current_threshold)
    candidate_result = score_summary(candidate[mask], candidate_threshold)
    checks = {
        "false_positive_count_no_higher": (
            candidate_result["false_positives"] <= current_result["false_positives"]
        ),
        "raw_score_p95_no_higher": (
            candidate_result["raw_score"]["p95"] <= current_result["raw_score"]["p95"]
        ),
        "logit_margin_p95_no_higher": (
            candidate_result["logit_threshold_margin"]["p95"]
            <= current_result["logit_threshold_margin"]["p95"]
        ),
    }
    return {"current": current_result, "candidate": candidate_result, "checks": checks}


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    overall = report["strata"]["available"]
    lines = [
        "# Fresh CloudSEN12+ scene-head safety test",
        "",
        f"- Available exact-product rows: **{overall['current']['rows']:,}**; unavailable rows: **{report['full_cohort_bounds']['unavailable_rows']:,}**.",
        f"- Current false positives: **{overall['current']['false_positives']}**; candidate: **{overall['candidate']['false_positives']}**.",
        f"- Current FPR: **{overall['current']['false_positive_rate']:.4%}**; candidate: **{overall['candidate']['false_positive_rate']:.4%}**.",
        f"- Symmetric adversarial full-cohort FPR bound: current **{report['full_cohort_bounds']['current_worst_case_fpr']:.4%}**, candidate **{report['full_cohort_bounds']['candidate_worst_case_fpr']:.4%}**.",
        f"- All predeclared fresh-safety gates pass: **{str(report['all_safety_gates_pass']).lower()}**.",
        "",
        "The exact MARS paper cache remained unopened. The controlled zero cloud proxy was identical for both heads and is not claimed as producer spatial cloud truth.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL.as_posix())
    args = parser.parse_args()
    protocol_path = (ROOT / args.protocol).resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if sha256(Path(__file__).resolve()) != protocol["evaluator"]["sha256"]:
        raise ValueError("Fresh-test evaluator hash mismatch")
    paths: dict[str, Path] = {}
    for name, source in protocol["inputs"].items():
        path = (ROOT / source["path"]).resolve()
        if sha256(path) != source["sha256"]:
            raise ValueError(f"Frozen evaluation input hash mismatch: {name}")
        paths[name] = path
    finalization = json.loads(paths["finalization_report"].read_text(encoding="utf-8"))
    frozen_negative = finalization["cloudsen12_negative_development"]
    if (
        not finalization["all_finalization_gates_pass"]
        or finalization["artifact"]["sha256"] != protocol["inputs"]["candidate_artifact"]["sha256"]
        or float(frozen_negative["current_threshold"]) != float(protocol["thresholds"]["current"])
        or float(frozen_negative["candidate_threshold"]) != float(protocol["thresholds"]["candidate"])
    ):
        raise ValueError("Finalization authorization or frozen thresholds changed")
    with np.load(paths["features"], allow_pickle=False) as cache:
        raw_features = cache["features"].astype(np.float64)
        feature_names = cache["feature_names"].astype(str).tolist()
        labels = cache["labels"].astype(np.uint8)
        sample_ids = cache["sample_ids"].astype(str)
        groups = cache["groups"].astype(str)
        role = str(cache["research_role"].item())
        all_clear = cache["published_all_clear"].astype(bool)
        nonclear_pixels = cache["published_nonclear_pixels"].astype(np.int64)
    expected = protocol["expected"]
    if (
        raw_features.shape != (expected["available_rows"], expected["features"])
        or not np.isfinite(raw_features).all()
        or np.any(labels)
        or role != "fresh_external_test"
        or len(set(sample_ids.tolist())) != sample_ids.size
        or len(set(groups.tolist())) != expected["groups"]
        or int(np.count_nonzero(all_clear)) != expected["all_clear_rows"]
        or int(np.count_nonzero(~all_clear)) != expected["nonclear_rows"]
        or int(nonclear_pixels.sum()) != expected["nonclear_pixels"]
    ):
        raise ValueError("Fresh-test feature cache contract changed")
    augmented, augmented_names = augment_site_context(raw_features, np.asarray(feature_names), groups)
    current_payload = joblib.load(paths["current_artifact"])
    candidate_payload = joblib.load(paths["candidate_artifact"])
    if augmented_names != current_payload["augmented_feature_names"]:
        raise ValueError("Current-head augmented feature schema changed")
    if augmented_names != candidate_payload["feature_names"]:
        raise ValueError("Candidate augmented feature schema changed")
    current_threshold = float(protocol["thresholds"]["current"])
    candidate_threshold = float(protocol["thresholds"]["candidate"])
    if candidate_threshold != float(candidate_payload["operational_scene_threshold"]):
        raise ValueError("Candidate threshold differs from frozen artifact")
    values = {
        "features": augmented,
        "augmented_names": augmented_names,
        "primary": raw_features[:, feature_names.index("primary_connected_score")],
    }
    current = current_scores(values, current_payload)
    candidate_raw = candidate_payload["model"].predict_proba(augmented)[:, 1]
    candidate = blend_scores(current, candidate_raw, float(candidate_payload["candidate_blend"]))
    if not np.isfinite(current).all() or not np.isfinite(candidate).all():
        raise ValueError("Nonfinite fresh-test scene scores")
    masks = {
        "available": np.ones(labels.size, dtype=bool),
        "published_all_clear": all_clear,
        "published_nonclear": ~all_clear,
    }
    strata = {
        name: compare_stratum(
            current, candidate, mask, current_threshold, candidate_threshold
        )
        for name, mask in masks.items()
    }
    gate_spec = protocol["safety_gates"]
    checks: dict[str, bool] = {}
    for stratum in gate_spec["false_positive_count_strata"]:
        checks[f"{stratum}_false_positive_count_no_higher"] = strata[stratum]["checks"]["false_positive_count_no_higher"]
    for stratum in gate_spec["raw_score_p95_strata"]:
        checks[f"{stratum}_raw_score_p95_no_higher"] = strata[stratum]["checks"]["raw_score_p95_no_higher"]
    for stratum in gate_spec["logit_margin_p95_strata"]:
        checks[f"{stratum}_logit_margin_p95_no_higher"] = strata[stratum]["checks"]["logit_margin_p95_no_higher"]
    current_pred = current >= current_threshold
    candidate_pred = candidate >= candidate_threshold
    unavailable_rows = expected["full_rows"] - expected["available_rows"]
    full_bounds = {
        "full_rows": expected["full_rows"],
        "available_rows": expected["available_rows"],
        "unavailable_rows": unavailable_rows,
        "current_best_case_fpr": float(np.count_nonzero(current_pred) / expected["full_rows"]),
        "candidate_best_case_fpr": float(np.count_nonzero(candidate_pred) / expected["full_rows"]),
        "current_worst_case_fpr": float((np.count_nonzero(current_pred) + unavailable_rows) / expected["full_rows"]),
        "candidate_worst_case_fpr": float((np.count_nonzero(candidate_pred) + unavailable_rows) / expected["full_rows"]),
        "symmetric_worst_case_candidate_no_worse": bool(np.count_nonzero(candidate_pred) <= np.count_nonzero(current_pred)),
        "candidate_only_adversarial_stress_fpr": float((np.count_nonzero(candidate_pred) + unavailable_rows) / expected["full_rows"]),
    }
    checks["symmetric_adversarial_full_bound_no_worse"] = full_bounds["symmetric_worst_case_candidate_no_worse"]
    passed = all(checks.values())
    transitions = {
        "both_false_positive": int(np.count_nonzero(current_pred & candidate_pred)),
        "current_only_false_positive": int(np.count_nonzero(current_pred & ~candidate_pred)),
        "candidate_only_false_positive": int(np.count_nonzero(~current_pred & candidate_pred)),
        "both_true_negative": int(np.count_nonzero(~current_pred & ~candidate_pred)),
        "candidate_false_positive_sample_ids": sorted(sample_ids[candidate_pred].tolist()),
    }
    report = {
        "schema_version": 1,
        "scope": "one-shot fresh external-negative safety evaluation; exact MARS paper cache unopened",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "strata": strata,
        "full_cohort_bounds": full_bounds,
        "paired_transitions": transitions,
        "safety_checks": checks,
        "all_safety_gates_pass": passed,
        "decision": (
            "Authorize the fixed candidate for exact MARS paper evaluation."
            if passed
            else "Reject the fixed candidate before exact MARS paper evaluation."
        ),
        "cloud_input_limit": protocol["cloud_input_limit"],
        "paper_test_accessed": False,
        "provenance": {
            "protocol": args.protocol,
            "protocol_sha256": sha256(protocol_path),
            "script_sha256": sha256(Path(__file__).resolve()),
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        },
    }
    output_json = (ROOT / protocol["outputs"]["json"]).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown((ROOT / protocol["outputs"]["markdown"]).resolve(), report)
    print(json.dumps({"ok": passed, "decision": report["decision"], "checks": checks}, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
