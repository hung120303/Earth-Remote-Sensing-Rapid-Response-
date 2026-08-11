#!/usr/bin/env python3
"""Fit and evaluate frozen group-level CRC transport for MethaneS2CM v5.1.

This is a development-only candidate-specific diagnostic. The fixed 24-group
calibration partition fits thresholds; the disjoint 24-group partition is joined
to outcomes only in the separately authorized evaluation mode.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from calibrate_mars_v6_group_risk import crc_threshold


PROTOCOL = Path("configs/methanes2cm_v5_1_group_crc_transport_protocol.json")
FIT_OUTPUT = Path("outputs/methanes2cm_v5_1_group_crc_transport_fit.json")
RESULT_JSON = Path("reports/experiments/methanes2cm_v5_1_group_crc_transport.json")
RESULT_MD = Path("reports/experiments/METHANES2CM_V5_1_GROUP_CRC_TRANSPORT.md")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def verify_bound_file(root: Path, binding: dict[str, Any], name: str) -> Path:
    path = resolve(root, binding["path"])
    if not path.is_file():
        raise FileNotFoundError(f"Missing bound {name}: {path}")
    actual = sha256(path)
    if actual != binding["sha256"]:
        raise ValueError(f"{name} hash mismatch: expected {binding['sha256']}, got {actual}")
    return path


def verify_protocol(root: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    protocol_path = root / PROTOCOL
    protocol = load_json(protocol_path)
    script_binding = protocol["implementation"]
    script_path = Path(__file__).resolve()
    if script_path != resolve(root, script_binding["path"]).resolve():
        raise ValueError("Running script path does not match the bound implementation path")
    if sha256(script_path) != script_binding["sha256"]:
        raise ValueError("Running implementation hash does not match the protocol")
    paths = {
        name: verify_bound_file(root, binding, name)
        for name, binding in protocol["inputs"].items()
    }
    return protocol, paths


def validate_partition(
    rows: list[dict[str, Any]], receipt: dict[str, Any]
) -> tuple[set[str], set[str]]:
    partition = receipt["methanes2cm_confirmation_partition"]
    calibration = set(partition["risk_calibration_groups"])
    confirmation = set(partition["confirmation_groups"])
    manifest_groups = {str(row["group_id"]) for row in rows}
    if calibration & confirmation:
        raise ValueError("Calibration and confirmation groups overlap")
    if manifest_groups != calibration | confirmation:
        raise ValueError("Manifest groups do not equal the frozen 24/24 partition")
    if len(calibration) != 24 or len(confirmation) != 24:
        raise ValueError("Expected exactly 24 calibration and 24 confirmation groups")
    if len(rows) != receipt["held_confirmation_rows"]:
        raise ValueError("Manifest row count does not match cohort receipt")
    return calibration, confirmation


def load_score_cache(path: Path, score_field: str) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        required = {"sample_id", score_field}
        if not required.issubset(source.files):
            raise ValueError(f"Score cache missing {sorted(required - set(source.files))}")
        sample_ids = source["sample_id"].astype(np.int64)
        scores = source[score_field].astype(np.float64)
    if sample_ids.size != scores.size or len(np.unique(sample_ids)) != sample_ids.size:
        raise ValueError("Score cache IDs/scores are not one-to-one")
    if not np.isfinite(scores).all():
        raise ValueError("Score cache contains non-finite values")
    return {"sample_id": sample_ids, "scores": scores}


def align_partition(
    rows: list[dict[str, Any]],
    selected_groups: set[str],
    cache: dict[str, np.ndarray],
    *,
    include_labels: bool,
) -> dict[str, np.ndarray]:
    selected = [row for row in rows if str(row["group_id"]) in selected_groups]
    index = {int(value): i for i, value in enumerate(cache["sample_id"])}
    identifiers = np.asarray([int(row["id"]) for row in selected], dtype=np.int64)
    missing = [int(value) for value in identifiers if int(value) not in index]
    if missing:
        raise ValueError(f"Manifest IDs missing from score cache: {missing[:5]}")
    order = np.asarray([index[int(value)] for value in identifiers], dtype=np.int64)
    result = {
        "sample_id": identifiers,
        "groups": np.asarray([str(row["group_id"]) for row in selected]),
        "scores": cache["scores"][order],
    }
    if include_labels:
        labels = np.asarray([int(row["label"]) for row in selected], dtype=np.uint8)
        if not set(np.unique(labels)).issubset({0, 1}):
            raise ValueError("Labels must be binary")
        result["labels"] = labels
    return result


def pooled_fpr_threshold(scores: np.ndarray, labels: np.ndarray, target: float) -> float:
    if not 0.0 < target < 1.0:
        raise ValueError("target must lie in (0, 1)")
    negative = np.asarray(scores, dtype=np.float64)[np.asarray(labels).astype(int) == 0]
    if not negative.size:
        raise ValueError("Pooled calibration requires negatives")
    candidates = np.concatenate(
        [np.nextafter(np.unique(negative), math.inf), np.asarray([math.inf])]
    )
    for threshold in candidates:
        if float(np.mean(negative >= threshold)) <= target:
            return float(threshold)
    return math.inf


def operating_metrics(
    scores: np.ndarray, labels: np.ndarray, groups: np.ndarray, threshold: float
) -> dict[str, Any]:
    score = np.asarray(scores, dtype=np.float64)
    label = np.asarray(labels).astype(int)
    group = np.asarray(groups).astype(str)
    decision = score >= threshold
    positive = label == 1
    negative = ~positive
    tp = int(np.count_nonzero(decision & positive))
    fp = int(np.count_nonzero(decision & negative))
    tn = int(np.count_nonzero(~decision & negative))
    fn = int(np.count_nonzero(~decision & positive))
    fpr_by_group = []
    recall_by_group = []
    for value in np.unique(group):
        local = group == value
        if np.any(local & negative):
            fpr_by_group.append(float(np.mean(decision[local & negative])))
        if np.any(local & positive):
            recall_by_group.append(float(np.mean(decision[local & positive])))
    return {
        "threshold": float(threshold),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "recall": float(tp / (tp + fn)) if tp + fn else math.nan,
        "false_positive_rate": float(fp / (fp + tn)) if fp + tn else math.nan,
        "precision": float(tp / (tp + fp)) if tp + fp else math.nan,
        "group_balanced_recall": float(np.mean(recall_by_group)),
        "group_balanced_false_positive_rate": float(np.mean(fpr_by_group)),
        "negative_groups": len(fpr_by_group),
        "positive_groups": len(recall_by_group),
    }


def _per_group_rates(
    scores: np.ndarray, labels: np.ndarray, groups: np.ndarray, threshold: float
) -> dict[str, tuple[float, float]]:
    values: dict[str, tuple[float, float]] = {}
    for group in np.unique(groups.astype(str)):
        local = groups.astype(str) == group
        negative = local & (labels.astype(int) == 0)
        positive = local & (labels.astype(int) == 1)
        fpr = float(np.mean(scores[negative] >= threshold)) if np.any(negative) else math.nan
        recall = float(np.mean(scores[positive] >= threshold)) if np.any(positive) else math.nan
        values[str(group)] = (fpr, recall)
    return values


def paired_group_bootstrap(
    scores: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    candidate_threshold: float,
    comparator_threshold: float,
    replicates: int,
    seed: int,
    confidence: float,
) -> dict[str, Any]:
    candidate = _per_group_rates(scores, labels, groups, candidate_threshold)
    comparator = _per_group_rates(scores, labels, groups, comparator_threshold)
    names = np.asarray(sorted(candidate))
    if set(candidate) != set(comparator):
        raise ValueError("Paired bootstrap group identities differ")
    fpr_delta = np.asarray([candidate[g][0] - comparator[g][0] for g in names])
    recall_delta = np.asarray([candidate[g][1] - comparator[g][1] for g in names])
    rng = np.random.default_rng(seed)
    fpr_draws = np.empty(replicates, dtype=np.float64)
    recall_draws = np.empty(replicates, dtype=np.float64)
    for i in range(replicates):
        sampled = rng.integers(0, names.size, size=names.size)
        fpr_draws[i] = float(np.nanmean(fpr_delta[sampled]))
        recall_draws[i] = float(np.nanmean(recall_delta[sampled]))
    tail = (1.0 - confidence) / 2.0
    return {
        "unit": "25 km group",
        "groups": int(names.size),
        "replicates": int(replicates),
        "seed": int(seed),
        "confidence": float(confidence),
        "group_balanced_fpr_delta": {
            "point": float(np.nanmean(fpr_delta)),
            "lower": float(np.quantile(fpr_draws, tail)),
            "upper": float(np.quantile(fpr_draws, 1.0 - tail)),
        },
        "group_balanced_recall_delta": {
            "point": float(np.nanmean(recall_delta)),
            "lower": float(np.quantile(recall_draws, tail)),
            "upper": float(np.quantile(recall_draws, 1.0 - tail)),
        },
    }


def dry_run(root: Path, protocol: dict[str, Any], paths: dict[str, Path]) -> dict[str, Any]:
    rows = load_jsonl(paths["development_manifest"])
    receipt = load_json(paths["cohort_receipt"])
    calibration, confirmation = validate_partition(rows, receipt)
    cache = load_score_cache(paths["score_cache"], protocol["score_field"])
    calibration_view = align_partition(rows, calibration, cache, include_labels=False)
    confirmation_view = align_partition(rows, confirmation, cache, include_labels=False)
    return {
        "status": "pre_outcome_dry_run_passed",
        "calibration_groups": len(np.unique(calibration_view["groups"])),
        "calibration_rows": int(calibration_view["scores"].size),
        "confirmation_groups": len(np.unique(confirmation_view["groups"])),
        "confirmation_rows": int(confirmation_view["scores"].size),
        "outcome_arrays_opened": False,
        "verified_bindings": len(paths) + 1,
    }


def fit(root: Path, protocol: dict[str, Any], paths: dict[str, Path]) -> dict[str, Any]:
    if protocol["status"] != "fit_authorized_before_confirmation_score_label_join":
        raise ValueError("Protocol does not authorize calibration fitting")
    output = root / FIT_OUTPUT
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    rows = load_jsonl(paths["development_manifest"])
    receipt = load_json(paths["cohort_receipt"])
    calibration, _ = validate_partition(rows, receipt)
    cache = load_score_cache(paths["score_cache"], protocol["score_field"])
    view = align_partition(rows, calibration, cache, include_labels=True)
    curve = [
        crc_threshold(view["scores"], view["labels"], view["groups"], alpha)
        for alpha in protocol["calibration"]["crc_alphas"]
    ]
    primary = next(
        value
        for value in curve
        if value["alpha"] == protocol["calibration"]["primary_alpha"]
    )
    if not primary["feasible"]:
        raise ValueError("Primary CRC alpha is infeasible on calibration groups")
    comparator_threshold = pooled_fpr_threshold(
        view["scores"], view["labels"], protocol["calibration"]["pooled_target_fpr"]
    )
    report = {
        "schema_version": 1,
        "status": "calibration_fit_frozen_before_confirmation_score_label_join",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": sha256(root / PROTOCOL),
        "score_field": protocol["score_field"],
        "cohort": {
            "groups": len(np.unique(view["groups"])),
            "rows": int(view["scores"].size),
            "positives": int(np.count_nonzero(view["labels"] == 1)),
            "negatives": int(np.count_nonzero(view["labels"] == 0)),
        },
        "crc_curve": curve,
        "primary_crc": primary,
        "primary_calibration_metrics": operating_metrics(
            view["scores"], view["labels"], view["groups"], primary["threshold"]
        ),
        "pooled_comparator": {
            "target_fpr": protocol["calibration"]["pooled_target_fpr"],
            "threshold": comparator_threshold,
            "calibration_metrics": operating_metrics(
                view["scores"], view["labels"], view["groups"], comparator_threshold
            ),
        },
        "confirmation_outcomes_accessed": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def markdown(report: dict[str, Any]) -> str:
    c = report["confirmation"]["candidate"]
    b = report["confirmation"]["pooled_comparator"]
    boot = report["paired_group_bootstrap"]
    decision = "PROMOTE" if report["all_promotion_gates_pass"] else "REJECT"
    return "\n".join(
        [
            "# MethaneS2CM v5.1 group-CRC calibration transport",
            "",
            "Development-only candidate-specific evidence. Confirmation label marginals were exposed during the prerequisite alignment audit; this is not an untouched external confirmation.",
            "",
            f"Decision: **{decision}**",
            "",
            "| Rule | Threshold | Recall | Crop FPR | Group-balanced recall | Group-balanced FPR |",
            "|---|---:|---:|---:|---:|---:|",
            f"| Group CRC | {c['threshold']:.6f} | {c['recall']:.4f} | {c['false_positive_rate']:.4f} | {c['group_balanced_recall']:.4f} | {c['group_balanced_false_positive_rate']:.4f} |",
            f"| Pooled 5% calibration | {b['threshold']:.6f} | {b['recall']:.4f} | {b['false_positive_rate']:.4f} | {b['group_balanced_recall']:.4f} | {b['group_balanced_false_positive_rate']:.4f} |",
            "",
            f"Paired group FPR delta: {boot['group_balanced_fpr_delta']['point']:+.4f} "
            f"[{boot['group_balanced_fpr_delta']['lower']:+.4f}, {boot['group_balanced_fpr_delta']['upper']:+.4f}]",
            f"Paired group recall delta: {boot['group_balanced_recall_delta']['point']:+.4f} "
            f"[{boot['group_balanced_recall_delta']['lower']:+.4f}, {boot['group_balanced_recall_delta']['upper']:+.4f}]",
            "",
            "No model weights, ranking, dense masks, or opened location-test outcomes were changed or used.",
            "",
        ]
    )


def evaluate(root: Path, protocol: dict[str, Any], paths: dict[str, Path]) -> dict[str, Any]:
    if protocol["status"] != "evaluation_authorized_after_calibrator_freeze":
        raise ValueError("Protocol does not authorize confirmation evaluation")
    result_path = root / RESULT_JSON
    markdown_path = root / RESULT_MD
    if result_path.exists() or markdown_path.exists():
        raise FileExistsError("Refusing to overwrite confirmation result")
    fit_path = verify_bound_file(root, protocol["fitted_calibrator"], "fitted_calibrator")
    fitted = load_json(fit_path)
    if fitted["confirmation_outcomes_accessed"] is not False:
        raise ValueError("Fitted calibrator does not attest outcome blindness")
    rows = load_jsonl(paths["development_manifest"])
    receipt = load_json(paths["cohort_receipt"])
    _, confirmation = validate_partition(rows, receipt)
    cache = load_score_cache(paths["score_cache"], protocol["score_field"])
    view = align_partition(rows, confirmation, cache, include_labels=True)
    candidate_threshold = float(fitted["primary_crc"]["threshold"])
    comparator_threshold = float(fitted["pooled_comparator"]["threshold"])
    candidate = operating_metrics(
        view["scores"], view["labels"], view["groups"], candidate_threshold
    )
    comparator = operating_metrics(
        view["scores"], view["labels"], view["groups"], comparator_threshold
    )
    bootstrap = paired_group_bootstrap(
        view["scores"],
        view["labels"],
        view["groups"],
        candidate_threshold,
        comparator_threshold,
        protocol["bootstrap"]["replicates"],
        protocol["bootstrap"]["seed"],
        protocol["bootstrap"]["confidence"],
    )
    gates = protocol["promotion_gates"]
    checks = {
        "calibration_crc_bound_at_most_primary_alpha": float(
            fitted["primary_crc"]["crc_expected_risk_bound"]
        )
        <= protocol["calibration"]["primary_alpha"],
        "confirmation_group_fpr_at_most": candidate[
            "group_balanced_false_positive_rate"
        ]
        <= gates["confirmation_group_fpr_maximum"],
        "confirmation_crop_fpr_at_most": candidate["false_positive_rate"]
        <= gates["confirmation_crop_fpr_maximum"],
        "confirmation_recall_at_least": candidate["recall"]
        >= gates["confirmation_recall_minimum"],
        "paired_group_fpr_delta_upper_at_most": bootstrap[
            "group_balanced_fpr_delta"
        ]["upper"]
        <= gates["paired_group_fpr_delta_upper_maximum"],
        "paired_group_recall_delta_lower_at_least": bootstrap[
            "group_balanced_recall_delta"
        ]["lower"]
        >= gates["paired_group_recall_delta_lower_minimum"],
    }
    label = view["labels"].astype(int)
    report = {
        "schema_version": 1,
        "status": "completed",
        "scope": "development-only fixed-24-group confirmation of v5.1 group-CRC threshold transport",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": sha256(root / PROTOCOL),
        "fitted_calibrator_sha256": sha256(fit_path),
        "confirmation": {
            "groups": len(np.unique(view["groups"])),
            "rows": int(view["scores"].size),
            "positives": int(np.count_nonzero(label == 1)),
            "negatives": int(np.count_nonzero(label == 0)),
            "ranking": {
                "average_precision": float(average_precision_score(label, view["scores"])),
                "auroc": float(roc_auc_score(label, view["scores"])),
                "invariant_between_threshold_rules": True,
            },
            "candidate": candidate,
            "pooled_comparator": comparator,
        },
        "paired_group_bootstrap": bootstrap,
        "promotion_checks": checks,
        "all_promotion_gates_pass": all(checks.values()),
        "decision": gates["pass_action"] if all(checks.values()) else gates["failure_action"],
        "limitations": [
            "Confirmation label marginals were exposed during prerequisite alignment audit before protocol freeze.",
            "The underlying development labels and v5.1 score family were opened in earlier work.",
            "This is candidate-specific development evidence, not a fresh external confirmation.",
            "The guarantee applies only under exchangeability; product/geographic shift remains empirical.",
        ],
        "prohibited_outcomes_accessed": False,
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(markdown(report), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--fit", action="store_true")
    mode.add_argument("--evaluate", action="store_true")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    protocol, paths = verify_protocol(root)
    if args.dry_run:
        result = dry_run(root, protocol, paths)
    elif args.fit:
        result = fit(root, protocol, paths)
    else:
        result = evaluate(root, protocol, paths)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
