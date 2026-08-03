#!/usr/bin/env python3
"""Evaluate a hash-bound Stanford label-free score bundle exactly once.

The script validates the complete score bundle before opening the frozen clean
event table. It never selects or adjusts a threshold and refuses to overwrite an
existing one-shot report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

DEFAULT_PROTOCOL = Path("configs/stanford_large_controlled_release_scoring_protocol.json")
DEFAULT_SCORES = Path(
    ".research/stanford_controlled_release_2024_2025/l1c_stress/scores/label_free_scores.npz"
)
DEFAULT_SCORE_RECEIPT = Path(
    ".research/stanford_controlled_release_2024_2025/l1c_stress/scores/label_free_score_receipt.json"
)
DEFAULT_CROP_MANIFEST = Path(
    ".research/stanford_controlled_release_2024_2025/l1c_stress/crop_manifest.json"
)
DEFAULT_COHORT_RECEIPT = Path(
    "reports/acquisition/stanford_large_controlled_release_cohort.json"
)
DEFAULT_EVENTS = Path(
    ".research/stanford_controlled_release_2024_2025/clean_events.jsonl"
)
DEFAULT_JOINED = Path(
    ".research/stanford_controlled_release_2024_2025/l1c_stress/scores/one_shot_joined.jsonl"
)
DEFAULT_JSON = Path(
    "reports/experiments/stanford_large_controlled_release_one_shot.json"
)
DEFAULT_MARKDOWN = Path(
    "reports/experiments/STANFORD_LARGE_CONTROLLED_RELEASE_ONE_SHOT.md"
)
REQUIRED_SCORE_FIELDS = (
    "event_ids",
    "released_mars_v3_scores",
    "gaussian_dofa_scores",
    "calibrated_spatial_prithvi_scores",
)
MODEL_CONTRACTS = {
    "released_mars_v3": {
        "field": "released_mars_v3_scores",
        "threshold": 0.5,
        "comparator": ">",
    },
    "gaussian_dofa": {
        "field": "gaussian_dofa_scores",
        "threshold": 0.16728139929966007,
        "comparator": ">=",
    },
    "spatial_prithvi_posttest": {
        "field": "calibrated_spatial_prithvi_scores",
        "threshold": 0.28187603894788654,
        "comparator": ">=",
    },
}


def repo_root() -> Path:
    return Path(
        subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip()
    ).resolve()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def wilson_interval(
    successes: int, total: int, *, confidence: float = 0.95
) -> tuple[float | None, float | None]:
    if total == 0:
        return None, None
    if not 0 <= successes <= total:
        raise ValueError("Wilson successes must be within [0, total]")
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    margin = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def decisions(scores: np.ndarray, threshold: float, comparator: str) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Scores must be a finite one-dimensional vector")
    if comparator == ">":
        return values > threshold
    if comparator == ">=":
        return values >= threshold
    raise ValueError(f"Unsupported threshold comparator: {comparator}")


def binary_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    threshold: float,
    comparator: str,
) -> dict[str, Any]:
    truth = np.asarray(labels, dtype=np.int8)
    predicted = decisions(scores, threshold, comparator)
    if truth.shape != predicted.shape or truth.ndim != 1:
        raise ValueError("Labels and scores must be aligned vectors")
    if not np.isin(truth, [0, 1]).all():
        raise ValueError("Primary labels must be binary")
    tp = int(np.sum((truth == 1) & predicted))
    tn = int(np.sum((truth == 0) & ~predicted))
    fp = int(np.sum((truth == 0) & predicted))
    fn = int(np.sum((truth == 1) & ~predicted))
    positive = tp + fn
    negative = tn + fp
    recall = None if positive == 0 else tp / positive
    fpr = None if negative == 0 else fp / negative
    precision = None if tp + fp == 0 else tp / (tp + fp)
    recall_interval = wilson_interval(tp, positive)
    fpr_interval = wilson_interval(fp, negative)
    return {
        "threshold": threshold,
        "comparator": comparator,
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "recall": recall,
        "recall_interval_95": list(recall_interval),
        "false_positive_rate": fpr,
        "false_positive_rate_interval_95": list(fpr_interval),
        "precision": precision,
    }


def score_model(
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    threshold: float,
    comparator: str,
) -> dict[str, Any]:
    truth = np.asarray(labels, dtype=np.int8)
    values = np.asarray(scores, dtype=np.float64)
    if truth.shape != values.shape or truth.ndim != 1:
        raise ValueError("Labels and scores must be aligned vectors")
    classes = np.unique(truth)
    if not np.array_equal(classes, np.asarray([0, 1], dtype=np.int8)):
        raise ValueError("Primary view must contain both classes")
    return {
        "rows": int(len(truth)),
        "positive": int(np.sum(truth == 1)),
        "negative": int(np.sum(truth == 0)),
        "average_precision": float(average_precision_score(truth, values)),
        "auroc_descriptive": float(roc_auc_score(truth, values)),
        "fixed_threshold": binary_metrics(
            truth, values, threshold=threshold, comparator=comparator
        ),
    }


def _metric_delta(
    labels: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    *,
    baseline_threshold: float,
    baseline_comparator: str,
    candidate_threshold: float,
    candidate_comparator: str,
) -> dict[str, float | None]:
    if len(np.unique(labels)) < 2:
        ap_delta = None
    else:
        ap_delta = float(
            average_precision_score(labels, candidate)
            - average_precision_score(labels, baseline)
        )
    baseline_fixed = binary_metrics(
        labels,
        baseline,
        threshold=baseline_threshold,
        comparator=baseline_comparator,
    )
    candidate_fixed = binary_metrics(
        labels,
        candidate,
        threshold=candidate_threshold,
        comparator=candidate_comparator,
    )
    recall_delta = (
        None
        if baseline_fixed["recall"] is None or candidate_fixed["recall"] is None
        else float(candidate_fixed["recall"] - baseline_fixed["recall"])
    )
    fpr_delta = (
        None
        if baseline_fixed["false_positive_rate"] is None
        or candidate_fixed["false_positive_rate"] is None
        else float(
            candidate_fixed["false_positive_rate"]
            - baseline_fixed["false_positive_rate"]
        )
    )
    return {
        "average_precision_delta": ap_delta,
        "recall_delta": recall_delta,
        "false_positive_rate_delta": fpr_delta,
    }


def paired_date_bootstrap(
    labels: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    dates: np.ndarray,
    *,
    baseline_threshold: float,
    baseline_comparator: str,
    candidate_threshold: float,
    candidate_comparator: str,
    replicates: int,
    seed: int,
    confidence: float = 0.95,
) -> dict[str, Any]:
    truth = np.asarray(labels, dtype=np.int8)
    primary = np.asarray(baseline, dtype=np.float64)
    alternative = np.asarray(candidate, dtype=np.float64)
    blocks = np.asarray(dates).astype(str)
    if not (truth.shape == primary.shape == alternative.shape == blocks.shape):
        raise ValueError("Bootstrap inputs must be aligned vectors")
    unique_blocks = np.unique(blocks)
    if len(unique_blocks) == 0 or replicates <= 0:
        raise ValueError("Bootstrap requires blocks and positive replicate count")
    row_by_block = {block: np.flatnonzero(blocks == block) for block in unique_blocks}
    rng = np.random.default_rng(seed)
    values: dict[str, list[float]] = {
        "average_precision_delta": [],
        "recall_delta": [],
        "false_positive_rate_delta": [],
    }
    for _ in range(replicates):
        sampled = rng.choice(unique_blocks, size=len(unique_blocks), replace=True)
        indices = np.concatenate([row_by_block[block] for block in sampled])
        local = _metric_delta(
            truth[indices],
            primary[indices],
            alternative[indices],
            baseline_threshold=baseline_threshold,
            baseline_comparator=baseline_comparator,
            candidate_threshold=candidate_threshold,
            candidate_comparator=candidate_comparator,
        )
        for name, value in local.items():
            if value is not None and np.isfinite(value):
                values[name].append(float(value))
    alpha = (1.0 - confidence) / 2.0
    result: dict[str, Any] = {
        "confidence": confidence,
        "unit": "UTC acquisition date",
        "blocks": int(len(unique_blocks)),
        "requested_replicates": int(replicates),
        "seed": int(seed),
    }
    point = _metric_delta(
        truth,
        primary,
        alternative,
        baseline_threshold=baseline_threshold,
        baseline_comparator=baseline_comparator,
        candidate_threshold=candidate_threshold,
        candidate_comparator=candidate_comparator,
    )
    for name, local_values in values.items():
        array = np.asarray(local_values, dtype=np.float64)
        result[name] = {
            "point": point[name],
            "lower": None if len(array) == 0 else float(np.quantile(array, alpha)),
            "upper": None if len(array) == 0 else float(np.quantile(array, 1.0 - alpha)),
            "valid_replicates": int(len(array)),
        }
    return result


def superiority_gate(bootstrap: dict[str, Any]) -> dict[str, Any]:
    ap = bootstrap["average_precision_delta"]
    recall = bootstrap["recall_delta"]
    fpr = bootstrap["false_positive_rate_delta"]
    checks = {
        "ap_delta_lower_positive": ap["lower"] is not None and ap["lower"] > 0.0,
        "recall_delta_lower_positive": recall["lower"] is not None
        and recall["lower"] > 0.0,
        "fpr_delta_upper_nonpositive": fpr["upper"] is not None and fpr["upper"] <= 0.0,
    }
    return {"checks": checks, "passed": bool(all(checks.values()))}


def load_crop_event_ids(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    samples = payload.get("samples")
    if not isinstance(samples, list):
        raise ValueError("Crop manifest lacks samples")
    return [str(row["event_id"]) for row in samples]


def validate_score_bundle(
    arrays: dict[str, np.ndarray], crop_manifest_path: Path, *, expected_rows: int = 169
) -> list[str]:
    missing = sorted(set(REQUIRED_SCORE_FIELDS) - set(arrays))
    if missing:
        raise ValueError(f"Score bundle lacks required fields: {missing}")
    event_ids = np.asarray(arrays["event_ids"]).astype(str)
    if event_ids.ndim != 1 or len(event_ids) != expected_rows:
        raise ValueError(f"Expected {expected_rows} score event IDs")
    if len(set(event_ids.tolist())) != len(event_ids):
        raise ValueError("Score bundle contains duplicate event IDs")
    expected_ids = load_crop_event_ids(crop_manifest_path)
    if len(expected_ids) != expected_rows or len(set(expected_ids)) != len(expected_ids):
        raise ValueError("Crop manifest event IDs are incomplete or duplicated")
    if set(event_ids.tolist()) != set(expected_ids):
        raise ValueError("Score bundle event IDs do not exactly match the crop manifest event IDs")
    for field in REQUIRED_SCORE_FIELDS[1:]:
        values = np.asarray(arrays[field], dtype=np.float64)
        if values.shape != event_ids.shape or not np.isfinite(values).all():
            raise ValueError(f"Invalid score vector: {field}")
        if (values < 0.0).any() or (values > 1.0).any():
            raise ValueError(f"Score probabilities outside [0,1]: {field}")
    return event_ids.tolist()


def validate_score_receipt(
    receipt: dict[str, Any], scores_path: Path, protocol_path: Path, expected_rows: int
) -> None:
    bundle = receipt.get("score_bundle") or receipt.get("scores")
    if not isinstance(bundle, dict) or bundle.get("sha256") != sha256(scores_path):
        raise ValueError("Label-free score bundle hash is not bound by its receipt")
    protocol_binding = receipt.get("protocol") or receipt.get("bindings", {}).get("protocol")
    if not isinstance(protocol_binding, dict) or protocol_binding.get("sha256") != sha256(protocol_path):
        raise ValueError("Score receipt does not bind the current scoring protocol")
    outcome = receipt.get("outcome_blindness", {})
    blind_attestation = outcome.get(
        "labels_or_rates_accessed", outcome.get("labels_or_outcomes_accessed")
    )
    if blind_attestation is not False:
        raise ValueError("Score receipt does not affirm label/rate blindness")
    if outcome.get("detector_outcomes_accessed") not in {False, None}:
        raise ValueError("Score receipt indicates detector outcome access")
    summary = receipt.get("summary", {})
    complete = summary.get("complete")
    rows = summary.get("rows", summary.get("complete_rows"))
    if complete is not True or int(rows or -1) != expected_rows:
        raise ValueError("Score receipt is not a complete frozen-cohort receipt")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            rows.append(value)
    return rows


def score_distribution(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "minimum": float(np.min(array)),
        "p25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "p75": float(np.quantile(array, 0.75)),
        "maximum": float(np.max(array)),
    }


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Stanford controlled-release one-shot stress test",
        "",
        f"Generated: {report['generated_at_utc']}",
        "",
        "This is one-site temporal operating-point evidence, not geographic generalization. Stanford has no pixel plume masks, so IoU is not estimable.",
        "",
        "| Model | AP | Recall | FPR | Superiority vs released MARS |",
        "|---|---:|---:|---:|---|",
    ]
    for name, result in report["primary_view"]["models"].items():
        fixed = result["fixed_threshold"]
        gate = report["comparisons"].get(name, {}).get("superiority_gate")
        claim = "baseline" if gate is None else ("passed" if gate["passed"] else "not passed")
        lines.append(
            f"| {name} | {result['average_precision']:.6f} | "
            f"{fixed['recall']:.6f} | {fixed['false_positive_rate']:.6f} | {claim} |"
        )
    lines.extend(
        [
            "",
            "No threshold was selected or modified on this cohort. Spatial-Prithvi is a separately labeled post-test candidate. Official benchmark conclusions remain post-test and unchanged.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL.as_posix())
    parser.add_argument("--scores", default=DEFAULT_SCORES.as_posix())
    parser.add_argument("--score-receipt", default=DEFAULT_SCORE_RECEIPT.as_posix())
    parser.add_argument("--crop-manifest", default=DEFAULT_CROP_MANIFEST.as_posix())
    parser.add_argument("--cohort-receipt", default=DEFAULT_COHORT_RECEIPT.as_posix())
    parser.add_argument("--events", default=DEFAULT_EVENTS.as_posix())
    parser.add_argument("--joined", default=DEFAULT_JOINED.as_posix())
    parser.add_argument("--output-json", default=DEFAULT_JSON.as_posix())
    parser.add_argument("--output-markdown", default=DEFAULT_MARKDOWN.as_posix())
    parser.add_argument("--dry-run", action="store_true", help="Validate scores without opening outcomes")
    args = parser.parse_args()

    root = repo_root()
    protocol_path = (root / args.protocol).resolve()
    scores_path = (root / args.scores).resolve()
    score_receipt_path = (root / args.score_receipt).resolve()
    crop_manifest_path = (root / args.crop_manifest).resolve()
    cohort_receipt_path = (root / args.cohort_receipt).resolve()
    events_path = (root / args.events).resolve()
    joined_path = (root / args.joined).resolve()
    output_json = (root / args.output_json).resolve()
    output_markdown = (root / args.output_markdown).resolve()

    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    expected_rows = int(protocol["acquisition"]["pairs"])
    receipt = json.loads(score_receipt_path.read_text(encoding="utf-8"))
    validate_score_receipt(receipt, scores_path, protocol_path, expected_rows)
    with np.load(scores_path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    event_ids = validate_score_bundle(arrays, crop_manifest_path, expected_rows=expected_rows)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "label-free score bundle valid; outcomes not opened",
                    "rows": expected_rows,
                    "scores_sha256": sha256(scores_path),
                },
                sort_keys=True,
            )
        )
        return 0

    if output_json.exists() or output_markdown.exists() or joined_path.exists():
        raise FileExistsError("One-shot outcome artifact already exists; overwrite is forbidden")
    cohort_receipt = json.loads(cohort_receipt_path.read_text(encoding="utf-8"))
    source_binding = cohort_receipt["source"]
    if source_binding["clean_event_rows_sha256"] != sha256(events_path):
        raise ValueError("Frozen clean-event outcome table hash changed")
    events = load_jsonl(events_path)
    event_by_id = {str(row["release_id"]): row for row in events}
    if len(event_by_id) != len(events):
        raise ValueError("Duplicate release IDs in frozen clean-event table")
    if any(event_id not in event_by_id for event_id in event_ids):
        raise ValueError("A score event is absent from the frozen outcome table")
    score_index = {event_id: index for index, event_id in enumerate(event_ids)}
    joined: list[dict[str, Any]] = []
    for event_id in event_ids:
        event = event_by_id[event_id]
        index = score_index[event_id]
        joined.append(
            {
                "event_id": event_id,
                "observed_at_utc": event["observed_at_utc"],
                "truth_stratum": event["truth_stratum"],
                "metered_ch4_kgh": float(event["metered_ch4_kgh"]),
                **{
                    field: float(np.asarray(arrays[field])[index])
                    for field in REQUIRED_SCORE_FIELDS[1:]
                },
            }
        )
    joined_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_joined = joined_path.with_suffix(joined_path.suffix + ".tmp")
    temporary_joined.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in joined),
        encoding="utf-8",
    )
    os.replace(temporary_joined, joined_path)

    strata = Counter(row["truth_stratum"] for row in joined)
    primary_rows = [
        row
        for row in joined
        if row["truth_stratum"] in {"primary_negative", "primary_positive"}
    ]
    labels = np.asarray(
        [int(row["truth_stratum"] == "primary_positive") for row in primary_rows],
        dtype=np.int8,
    )
    dates = np.asarray([row["observed_at_utc"][:10] for row in primary_rows])
    model_results: dict[str, Any] = {}
    primary_scores: dict[str, np.ndarray] = {}
    for model, contract in MODEL_CONTRACTS.items():
        values = np.asarray([row[contract["field"]] for row in primary_rows])
        primary_scores[model] = values
        model_results[model] = score_model(
            labels,
            values,
            threshold=float(contract["threshold"]),
            comparator=str(contract["comparator"]),
        )
    comparisons: dict[str, Any] = {}
    baseline_contract = MODEL_CONTRACTS["released_mars_v3"]
    for model in ("gaussian_dofa", "spatial_prithvi_posttest"):
        candidate_contract = MODEL_CONTRACTS[model]
        bootstrap = paired_date_bootstrap(
            labels,
            primary_scores["released_mars_v3"],
            primary_scores[model],
            dates,
            baseline_threshold=float(baseline_contract["threshold"]),
            baseline_comparator=str(baseline_contract["comparator"]),
            candidate_threshold=float(candidate_contract["threshold"]),
            candidate_comparator=str(candidate_contract["comparator"]),
            replicates=int(protocol["uncertainty"]["paired_bootstrap_replicates"]),
            seed=int(protocol["uncertainty"]["paired_bootstrap_seed"]),
            confidence=float(protocol["uncertainty"]["confidence"]),
        )
        comparisons[model] = {
            "paired_date_bootstrap": bootstrap,
            "superiority_gate": superiority_gate(bootstrap),
        }
    challenge_rows = [
        row for row in joined if row["truth_stratum"] == "subthreshold_challenge"
    ]
    challenge: dict[str, Any] = {"rows": len(challenge_rows), "models": {}}
    for model, contract in MODEL_CONTRACTS.items():
        values = np.asarray([row[contract["field"]] for row in challenge_rows])
        predicted = decisions(values, float(contract["threshold"]), str(contract["comparator"]))
        challenge["models"][model] = {
            "detected": int(np.sum(predicted)),
            "detection_fraction": None if len(values) == 0 else float(np.mean(predicted)),
            "score_distribution": None if len(values) == 0 else score_distribution(values),
        }
    report = {
        "schema_version": 1,
        "status": "completed one-shot; thresholds unchanged",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "one-site temporal controlled-release operating-point stress test",
        "bindings": {
            "protocol": {"path": args.protocol, "sha256": sha256(protocol_path)},
            "score_bundle": {"path": args.scores, "sha256": sha256(scores_path)},
            "score_receipt": {"path": args.score_receipt, "sha256": sha256(score_receipt_path)},
            "outcomes": {"path": args.events, "sha256": sha256(events_path)},
            "joined": {"path": args.joined, "sha256": sha256(joined_path)},
            "script": {
                "path": Path(__file__).resolve().relative_to(root).as_posix(),
                "sha256": sha256(Path(__file__).resolve()),
            },
        },
        "access_sequence": {
            "score_bundle_validated_before_outcomes": True,
            "threshold_selection_on_stanford": False,
            "joined_outcomes_once": True,
        },
        "cohort": {"rows": len(joined), "truth_strata": dict(sorted(strata.items()))},
        "primary_view": {"rows": len(primary_rows), "models": model_results},
        "comparisons": comparisons,
        "subthreshold_challenge": challenge,
        "claim_boundary": (
            "Stanford is one-site temporal evidence only. IoU is unavailable. Spatial-Prithvi "
            "is post-test. Superiority requires every preregistered paired interval gate."
        ),
    }
    write_json(output_json, report)
    write_markdown(output_markdown, report)
    print(json.dumps({"status": report["status"], "report": args.output_json}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
