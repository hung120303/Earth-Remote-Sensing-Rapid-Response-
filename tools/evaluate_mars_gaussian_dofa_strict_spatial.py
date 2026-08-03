#!/usr/bin/env python3
"""One-shot scene evaluation of frozen Gaussian+DOFA on MARS strict spatial rows."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = Path("configs/mars_gaussian_dofa_strict_spatial_evaluation_protocol.json")
EXPECTED_SCORE_KEYS = {
    "schema_version",
    "sample_ids",
    "groups",
    "released_mars_v3_scores",
    "current_v3_scores",
    "gaussian_raw_logits",
    "dofa_raw_scores",
    "gaussian_dofa_scores",
    "released_mars_v3_decisions",
    "gaussian_dofa_decisions",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bindings(value: Any, prefix: str = "") -> list[tuple[str, dict[str, str]]]:
    found: list[tuple[str, dict[str, str]]] = []
    if isinstance(value, dict):
        if isinstance(value.get("path"), str) and isinstance(value.get("sha256"), str):
            found.append((prefix.rstrip("."), value))
        else:
            for key, child in value.items():
                found.extend(_bindings(child, f"{prefix}{key}."))
    return found


def validate_bindings(dependencies: dict[str, Any]) -> list[dict[str, str]]:
    verified: list[dict[str, str]] = []
    for name, binding in _bindings(dependencies):
        path = (ROOT / binding["path"]).resolve()
        try:
            relative = path.relative_to(ROOT)
        except ValueError as exc:
            raise ValueError(f"Dependency escapes repository root: {name}") from exc
        if not path.is_file() or sha256(path) != binding["sha256"]:
            raise ValueError(f"Frozen evaluation dependency differs: {name}")
        verified.append({"name": name, "path": relative.as_posix(), "sha256": binding["sha256"]})
    return verified


def load_score_bundle(path: Path, expected_rows: int) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        if set(source.files) != EXPECTED_SCORE_KEYS:
            raise ValueError(f"Strict score-bundle schema differs: {source.files}")
        if int(np.asarray(source["schema_version"]).item()) != 1:
            raise ValueError("Unsupported strict score-bundle schema")
        result = {key: source[key].copy() for key in source.files if key != "schema_version"}
    for name, values in result.items():
        if values.shape != (expected_rows,):
            raise ValueError(f"Strict score array does not have {expected_rows} rows: {name}")
        if name not in {"sample_ids", "groups"} and not np.isfinite(values).all():
            raise ValueError(f"Strict score array contains non-finite values: {name}")
    ids = result["sample_ids"].astype(str)
    if len(set(ids.tolist())) != expected_rows:
        raise ValueError("Strict score identities are not unique")
    if not np.isin(result["released_mars_v3_decisions"], (0, 1)).all():
        raise ValueError("Released decisions are not binary")
    if not np.isin(result["gaussian_dofa_decisions"], (0, 1)).all():
        raise ValueError("Gaussian+DOFA decisions are not binary")
    result["sample_ids"] = ids
    result["groups"] = result["groups"].astype(str)
    return result


def validate_score_side(protocol: dict[str, Any]) -> tuple[dict[str, np.ndarray], list[dict[str, str]]]:
    dependencies = protocol["score_dependencies"]
    verified = validate_bindings(dependencies)
    score_path = ROOT / dependencies["score_bundle"]["path"]
    scores = load_score_bundle(score_path, int(protocol["cohort"]["rows"]))
    released_expected = scores["released_mars_v3_scores"] > float(protocol["models"]["released_mars_v3"]["threshold"])
    candidate_expected = scores["gaussian_dofa_scores"] >= float(protocol["models"]["gaussian_dofa"]["threshold"])
    np.testing.assert_array_equal(scores["released_mars_v3_decisions"].astype(bool), released_expected)
    np.testing.assert_array_equal(scores["gaussian_dofa_decisions"].astype(bool), candidate_expected)
    return scores, verified


def load_outcomes(
    protocol: dict[str, Any], scores: dict[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    validate_bindings(protocol["outcome_dependencies"])
    manifest_path = ROOT / protocol["outcome_dependencies"]["strict_labeled_manifest"]["path"]
    records: dict[str, tuple[int, str]] = {}
    with manifest_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            record = json.loads(line)
            sample_id = str(record["sample_id"])
            state = str(record["label_state"])
            if state not in {"PLUME", "NO_PLUME"}:
                raise ValueError(f"Unexpected outcome state on line {line_number}: {state}")
            if sample_id in records:
                raise ValueError(f"Duplicate strict outcome identity: {sample_id}")
            records[sample_id] = (int(state == "PLUME"), str(record["group_id"]))
    ids = scores["sample_ids"].astype(str)
    if set(records) != set(ids.tolist()):
        raise ValueError("Strict outcome manifest and score bundle identities differ")
    labels = np.asarray([records[sample_id][0] for sample_id in ids], dtype=np.int8)
    components = np.asarray([records[sample_id][1] for sample_id in ids])
    expected_components = int(protocol["cohort"]["strict_spatial_components"])
    if np.unique(components).size != expected_components:
        raise ValueError("Strict spatial-component count differs from frozen protocol")

    diagnostic_path = ROOT / protocol["outcome_dependencies"]["exact_paper_diagnostic_cache"]["path"]
    with np.load(diagnostic_path, allow_pickle=False) as diagnostic:
        required = {"aligned_sample_ids", "labels", "baseline_scores", "candidate_scores"}
        if not required.issubset(diagnostic.files):
            raise ValueError("Exact-paper diagnostic cache lacks required verification arrays")
        diagnostic_ids = diagnostic["aligned_sample_ids"].astype(str)
        lookup = {sample_id: index for index, sample_id in enumerate(diagnostic_ids)}
        if set(ids.tolist()) - set(lookup):
            raise ValueError("Strict score identities are absent from exact-paper cache")
        indices = np.asarray([lookup[sample_id] for sample_id in ids], dtype=np.int64)
        diagnostic_labels = np.asarray(diagnostic["labels"][indices], dtype=np.int8)
        diagnostic_released = np.asarray(diagnostic["baseline_scores"][indices], dtype=np.float64)
        diagnostic_current = np.asarray(diagnostic["candidate_scores"][indices], dtype=np.float64)
    np.testing.assert_array_equal(labels, diagnostic_labels)
    np.testing.assert_allclose(scores["current_v3_scores"], diagnostic_current, rtol=0.0, atol=1e-12)
    provenance = {
        "rows": int(labels.size),
        "positives": int(labels.sum()),
        "negatives": int(np.count_nonzero(labels == 0)),
        "strict_spatial_components": int(np.unique(components).size),
        "manifest_and_diagnostic_labels_exact": True,
        "released_comparator": "input-only reconstruction of the released checkpoint connected-component score; the published CSV scene_pred is retained only as a provenance cross-check",
        "released_vs_published_scene_pred_max_abs_difference": float(
            np.max(np.abs(scores["released_mars_v3_scores"] - diagnostic_released))
        ),
        "current_scores_max_abs_error": float(np.max(np.abs(scores["current_v3_scores"] - diagnostic_current))),
    }
    return labels, components, provenance


def binary_counts(labels: np.ndarray, predictions: np.ndarray) -> dict[str, int | float]:
    predictions = np.asarray(predictions, dtype=bool)
    tp = int(np.count_nonzero((labels == 1) & predictions))
    fp = int(np.count_nonzero((labels == 0) & predictions))
    tn = int(np.count_nonzero((labels == 0) & ~predictions))
    fn = int(np.count_nonzero((labels == 1) & ~predictions))
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "recall": tp / max(tp + fn, 1),
        "false_positive_rate": fp / max(fp + tn, 1),
        "precision": tp / max(tp + fp, 1),
    }


def model_metrics(labels: np.ndarray, scores: np.ndarray, predictions: np.ndarray, threshold: float, comparator: str) -> dict[str, Any]:
    return {
        "average_precision": float(average_precision_score(labels, scores)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "fixed_operating_point": {
            "threshold": threshold,
            "comparator": comparator,
            **binary_counts(labels, predictions),
        },
    }


def score_plan(labels: np.ndarray, scores: np.ndarray, component_index: np.ndarray) -> dict[str, np.ndarray]:
    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    ends = np.flatnonzero(np.r_[sorted_scores[1:] != sorted_scores[:-1], True])
    return {
        "labels": labels[order].astype(np.int8),
        "components": component_index[order],
        "ends": ends,
        "thresholds": sorted_scores[ends],
    }


def plan_cumulative(draws: np.ndarray, plan: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    weights = draws[:, plan["components"]]
    positives = weights * plan["labels"][None, :]
    tp = np.cumsum(positives, axis=1)[:, plan["ends"]]
    fp = np.cumsum(weights - positives, axis=1)[:, plan["ends"]]
    return tp, fp, tp[:, -1]


def average_precision_from_cumulative(tp: np.ndarray, fp: np.ndarray, positives: np.ndarray) -> np.ndarray:
    increments = np.diff(tp, axis=1, prepend=np.zeros((tp.shape[0], 1), dtype=tp.dtype))
    precision = tp / np.maximum(tp + fp, 1)
    return np.sum(increments * precision, axis=1) / np.maximum(positives, 1)


def matched_fpr_point(labels: np.ndarray, candidate_scores: np.ndarray, target_fpr: float) -> dict[str, Any]:
    plan = score_plan(labels, candidate_scores, np.zeros(labels.size, dtype=np.int64))
    cumulative_tp = np.cumsum(plan["labels"])[plan["ends"]]
    cumulative_fp = np.cumsum(1 - plan["labels"])[plan["ends"]]
    positives = max(int(labels.sum()), 1)
    negatives = max(int(np.count_nonzero(labels == 0)), 1)
    allowed = cumulative_fp / negatives <= target_fpr + 1e-12
    if not np.any(allowed):
        return {"threshold": math.inf, "recall": 0.0, "false_positive_rate": 0.0, "tp": 0, "fp": 0}
    allowed_indices = np.flatnonzero(allowed)
    recalls = cumulative_tp[allowed_indices] / positives
    index = int(allowed_indices[np.argmax(recalls)])
    return {
        "threshold": float(plan["thresholds"][index]),
        "comparator": ">=",
        "recall": float(cumulative_tp[index] / positives),
        "false_positive_rate": float(cumulative_fp[index] / negatives),
        "tp": int(cumulative_tp[index]),
        "fp": int(cumulative_fp[index]),
    }


def interval(values: np.ndarray, confidence: float) -> dict[str, float]:
    alpha = (1.0 - confidence) / 2.0
    return {
        "mean": float(np.mean(values)),
        "lower": float(np.quantile(values, alpha)),
        "upper": float(np.quantile(values, 1.0 - alpha)),
    }


def _component_sum(values: np.ndarray, component_index: np.ndarray, count: int) -> np.ndarray:
    return np.bincount(component_index, weights=values, minlength=count)


def paired_component_bootstrap(
    *,
    labels: np.ndarray,
    components: np.ndarray,
    baseline_scores: np.ndarray,
    candidate_scores: np.ndarray,
    baseline_predictions: np.ndarray,
    candidate_predictions: np.ndarray,
    replicates: int,
    seed: int,
    confidence: float,
    batch_size: int = 64,
) -> dict[str, Any]:
    component_names, component_index = np.unique(components, return_inverse=True)
    count = component_names.size
    baseline_plan = score_plan(labels, baseline_scores, component_index)
    candidate_plan = score_plan(labels, candidate_scores, component_index)
    positive_component = _component_sum((labels == 1).astype(float), component_index, count)
    negative_component = _component_sum((labels == 0).astype(float), component_index, count)
    baseline_tp_component = _component_sum(((labels == 1) & baseline_predictions).astype(float), component_index, count)
    baseline_fp_component = _component_sum(((labels == 0) & baseline_predictions).astype(float), component_index, count)
    candidate_tp_component = _component_sum(((labels == 1) & candidate_predictions).astype(float), component_index, count)
    candidate_fp_component = _component_sum(((labels == 0) & candidate_predictions).astype(float), component_index, count)
    rng = np.random.default_rng(seed)
    chunks = {name: [] for name in ("average_precision", "fixed_recall", "fixed_false_positive_rate", "matched_fpr_recall")}
    probabilities = np.full(count, 1.0 / count)
    for start in range(0, replicates, batch_size):
        size = min(batch_size, replicates - start)
        draws = rng.multinomial(count, probabilities, size=size).astype(np.int32)
        base_tp_cumulative, base_fp_cumulative, base_positive_total = plan_cumulative(draws, baseline_plan)
        cand_tp_cumulative, cand_fp_cumulative, cand_positive_total = plan_cumulative(draws, candidate_plan)
        base_ap = average_precision_from_cumulative(base_tp_cumulative, base_fp_cumulative, base_positive_total)
        cand_ap = average_precision_from_cumulative(cand_tp_cumulative, cand_fp_cumulative, cand_positive_total)
        chunks["average_precision"].append(cand_ap - base_ap)
        positives = draws @ positive_component
        negatives = draws @ negative_component
        base_tp = draws @ baseline_tp_component
        base_fp = draws @ baseline_fp_component
        cand_tp = draws @ candidate_tp_component
        cand_fp = draws @ candidate_fp_component
        base_recall = base_tp / np.maximum(positives, 1)
        base_fpr = base_fp / np.maximum(negatives, 1)
        chunks["fixed_recall"].append(cand_tp / np.maximum(positives, 1) - base_recall)
        chunks["fixed_false_positive_rate"].append(cand_fp / np.maximum(negatives, 1) - base_fpr)
        allowed = cand_fp_cumulative <= base_fpr[:, None] * negatives[:, None] + 1e-12
        matched_tp = np.max(np.where(allowed, cand_tp_cumulative, -1), axis=1)
        matched_tp = np.maximum(matched_tp, 0)
        chunks["matched_fpr_recall"].append(matched_tp / np.maximum(positives, 1) - base_recall)
    arrays = {name: np.concatenate(parts) for name, parts in chunks.items()}
    return {
        "replicates": replicates,
        "seed": seed,
        "confidence": confidence,
        "unit": "strict 25 km connected spatial component",
        "components": int(count),
        "delta_intervals": {name: interval(values, confidence) for name, values in arrays.items()},
    }


def exact_mcnemar(labels: np.ndarray, baseline: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    baseline_correct = baseline.astype(bool) == labels.astype(bool)
    candidate_correct = candidate.astype(bool) == labels.astype(bool)
    baseline_only_wrong = int(np.count_nonzero(~baseline_correct & candidate_correct))
    candidate_only_wrong = int(np.count_nonzero(baseline_correct & ~candidate_correct))
    discordant = baseline_only_wrong + candidate_only_wrong
    pvalue = 1.0 if discordant == 0 else float(binomtest(baseline_only_wrong, discordant, 0.5).pvalue)
    return {
        "baseline_wrong_candidate_correct": baseline_only_wrong,
        "baseline_correct_candidate_wrong": candidate_only_wrong,
        "discordant": discordant,
        "two_sided_exact_pvalue": pvalue,
    }


def comparison(
    *,
    labels: np.ndarray,
    components: np.ndarray,
    baseline_name: str,
    baseline_scores: np.ndarray,
    baseline_predictions: np.ndarray,
    baseline_threshold: float,
    baseline_comparator: str,
    candidate_name: str,
    candidate_scores: np.ndarray,
    candidate_predictions: np.ndarray,
    candidate_threshold: float,
    candidate_comparator: str,
    bootstrap: dict[str, Any],
    superiority_rules: dict[str, Any],
) -> dict[str, Any]:
    base_metrics = model_metrics(labels, baseline_scores, baseline_predictions, baseline_threshold, baseline_comparator)
    cand_metrics = model_metrics(labels, candidate_scores, candidate_predictions, candidate_threshold, candidate_comparator)
    matched = matched_fpr_point(labels, candidate_scores, float(base_metrics["fixed_operating_point"]["false_positive_rate"]))
    uncertainty = paired_component_bootstrap(
        labels=labels,
        components=components,
        baseline_scores=baseline_scores,
        candidate_scores=candidate_scores,
        baseline_predictions=baseline_predictions,
        candidate_predictions=candidate_predictions,
        replicates=int(bootstrap["replicates"]),
        seed=int(bootstrap["seed"]),
        confidence=float(bootstrap["confidence"]),
    )
    intervals = uncertainty["delta_intervals"]
    checks = {
        "average_precision_delta_lower_gt_zero": intervals["average_precision"]["lower"] > 0.0,
        "matched_fpr_recall_delta_lower_gt_zero": intervals["matched_fpr_recall"]["lower"] > 0.0,
        "fixed_recall_delta_lower_gte_zero": intervals["fixed_recall"]["lower"] >= 0.0,
        "fixed_fpr_delta_upper_lte_zero": intervals["fixed_false_positive_rate"]["upper"] <= 0.0,
    }
    required = {name: bool(superiority_rules[name]) for name in checks}
    passed = all((not required[name]) or checks[name] for name in checks)
    return {
        "baseline_name": baseline_name,
        "candidate_name": candidate_name,
        "baseline": base_metrics,
        "candidate": {**cand_metrics, "matched_fpr_operating_point": matched},
        "point_deltas": {
            "average_precision": cand_metrics["average_precision"] - base_metrics["average_precision"],
            "roc_auc": cand_metrics["roc_auc"] - base_metrics["roc_auc"],
            "fixed_recall": cand_metrics["fixed_operating_point"]["recall"] - base_metrics["fixed_operating_point"]["recall"],
            "fixed_false_positive_rate": cand_metrics["fixed_operating_point"]["false_positive_rate"] - base_metrics["fixed_operating_point"]["false_positive_rate"],
            "matched_fpr_recall": matched["recall"] - base_metrics["fixed_operating_point"]["recall"],
        },
        "paired_component_bootstrap": uncertainty,
        "fixed_decision_mcnemar": exact_mcnemar(labels, baseline_predictions, candidate_predictions),
        "superiority_gate": {"required": required, "checks": checks, "passed": passed},
    }


def write_json(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite one-shot evaluation output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite one-shot evaluation output: {path}")
    primary = report["comparisons"]["gaussian_dofa_vs_released_mars_v3"]
    lines = [
        "# Gaussian+DOFA strict-spatial candidate-specific replay",
        "",
        "> This is a candidate-specific post-test diagnostic, not a fresh project-level holdout. Earlier ERSRR candidates had already opened this MARS exact-paper outcome view.",
        "",
        f"Rows: {report['cohort']['rows']}; positives: {report['cohort']['positives']}; negatives: {report['cohort']['negatives']}; strict 25 km components: {report['cohort']['strict_spatial_components']}.",
        "",
        "| Model | AP | AUROC | Recall | FPR | Precision |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for role in ("baseline", "candidate"):
        model = primary[role]
        fixed = model["fixed_operating_point"]
        name = primary[f"{role}_name"]
        lines.append(f"| {name} | {model['average_precision']:.6f} | {model['roc_auc']:.6f} | {fixed['recall']:.6f} | {fixed['false_positive_rate']:.6f} | {fixed['precision']:.6f} |")
    lines.extend([
        "",
        f"Primary superiority gate: **{'PASS' if primary['superiority_gate']['passed'] else 'FAIL'}**.",
        "",
        "Gaussian+DOFA changes only the frozen scene score. No pixel-IoU or localization-improvement claim is made.",
        "",
        "Dataset provenance: UNEP-IMEO/MARS-S2L revision `c26b1d7e31a0c5241fa37c9140802622c215eb32`, CC-BY-NC-SA-4.0.",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    protocol_path = ROOT / DEFAULT_PROTOCOL
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    scores, score_bindings = validate_score_side(protocol)
    outputs = {name: ROOT / path for name, path in protocol["outputs"].items()}
    if any(path.exists() for path in outputs.values()):
        raise FileExistsError("One-shot strict-spatial evaluation output already exists")
    if args.dry_run:
        print(json.dumps({"status": "pre_outcome_dry_run_passed", "rows": int(scores["sample_ids"].size), "verified_score_bindings": len(score_bindings), "outcomes_opened": False}, sort_keys=True))
        return 0
    if not protocol["evaluation_gate"]["one_shot_outcome_access_authorized"]:
        raise RuntimeError("One-shot strict-spatial outcome access is not authorized")
    labels, components, provenance = load_outcomes(protocol, scores)
    released_prediction = scores["released_mars_v3_decisions"].astype(bool)
    candidate_prediction = scores["gaussian_dofa_decisions"].astype(bool)
    candidate_threshold = float(protocol["models"]["gaussian_dofa"]["threshold"])
    current_prediction = scores["current_v3_scores"] >= candidate_threshold
    common = {
        "labels": labels,
        "components": components,
        "candidate_name": "Gaussian+DOFA",
        "candidate_scores": scores["gaussian_dofa_scores"],
        "candidate_predictions": candidate_prediction,
        "candidate_threshold": candidate_threshold,
        "candidate_comparator": ">=",
        "bootstrap": protocol["bootstrap"],
        "superiority_rules": protocol["superiority_gate"],
    }
    comparisons = {
        "gaussian_dofa_vs_released_mars_v3": comparison(
            **common,
            baseline_name="Released MARS-S2L v3",
            baseline_scores=scores["released_mars_v3_scores"],
            baseline_predictions=released_prediction,
            baseline_threshold=float(protocol["models"]["released_mars_v3"]["threshold"]),
            baseline_comparator=">",
        ),
        "gaussian_dofa_vs_current_v3": comparison(
            **common,
            baseline_name="Current-v3 ExtraTrees base at candidate threshold",
            baseline_scores=scores["current_v3_scores"],
            baseline_predictions=current_prediction,
            baseline_threshold=candidate_threshold,
            baseline_comparator=">=",
        ),
    }
    primary_passed = comparisons["gaussian_dofa_vs_released_mars_v3"]["superiority_gate"]["passed"]
    report = {
        "schema_version": 1,
        "status": "complete_candidate_specific_post_test_replay",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_boundary": protocol["scientific_boundary"],
        "provenance": provenance,
        "cohort": {
            "rows": int(labels.size),
            "positives": int(labels.sum()),
            "negatives": int(np.count_nonzero(labels == 0)),
            "strict_spatial_components": int(np.unique(components).size),
        },
        "comparisons": comparisons,
        "decision": "primary_superiority_gate_passed" if primary_passed else "primary_superiority_gate_failed",
        "pixel_claim": "not evaluated; Gaussian+DOFA changes only the frozen scene score",
        "protocol": {"path": DEFAULT_PROTOCOL.as_posix(), "sha256": sha256(protocol_path)},
        "score_bundle": protocol["score_dependencies"]["score_bundle"],
    }
    write_json(outputs["json"], report)
    write_markdown(outputs["markdown"], report)
    print(json.dumps({"status": report["status"], "decision": report["decision"], "rows": report["cohort"]["rows"], "json_sha256": sha256(outputs["json"]), "markdown_sha256": sha256(outputs["markdown"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
