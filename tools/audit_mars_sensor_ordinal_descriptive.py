#!/usr/bin/env python3
"""Independently verify and describe the frozen MARS ordinal folds-3/4 result."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from sklearn.metrics import average_precision_score

from train_mars_sensor_ordinal import (
    aggregate_iou,
    align_comparator,
    fold_lookup,
    matched_fpr_recall,
    metric_gates,
    reconstruct_dense_comparator,
    records_for_folds,
    verify_protocol,
    verify_runtime_environment,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORTING_PROTOCOL = Path("configs/mars_sensor_ordinal_reporting_protocol.json")
SENSOR_NAMES = {0: "Sentinel-2", 1: "Landsat"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(value: str | Path) -> Path:
    path = (ROOT / value).resolve()
    if ROOT != path and ROOT not in path.parents:
        raise ValueError(f"Path escapes repository root: {value}")
    return path


def _verify_hash(binding: Mapping[str, Any]) -> Path:
    path = _resolve(str(binding["path"]))
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256(path)
    if actual != str(binding["sha256"]):
        raise RuntimeError(f"Frozen hash mismatch for {path}: {actual}")
    return path


def _require_immutable(path: Path) -> None:
    if not path.is_file() or path.stat().st_mode & 0o222:
        raise RuntimeError(f"Required one-shot artifact is missing or mutable: {path}")


def assert_nested_close(
    expected: Any,
    actual: Any,
    *,
    tolerance: float,
    location: str = "root",
) -> None:
    """Require exact structure and tolerance-bounded numeric equality."""
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping) or set(expected) != set(actual):
            raise AssertionError(f"Mapping schema mismatch at {location}")
        for key in expected:
            assert_nested_close(
                expected[key], actual[key], tolerance=tolerance, location=f"{location}.{key}"
            )
        return
    if isinstance(expected, Sequence) and not isinstance(expected, (str, bytes, bytearray)):
        if not isinstance(actual, Sequence) or isinstance(actual, (str, bytes, bytearray)):
            raise AssertionError(f"Sequence type mismatch at {location}")
        if len(expected) != len(actual):
            raise AssertionError(f"Sequence length mismatch at {location}")
        for index, (left, right) in enumerate(zip(expected, actual)):
            assert_nested_close(
                left, right, tolerance=tolerance, location=f"{location}[{index}]"
            )
        return
    if isinstance(expected, bool) or isinstance(actual, bool):
        if expected is not actual:
            raise AssertionError(f"Boolean mismatch at {location}: {expected!r} != {actual!r}")
        return
    if isinstance(expected, (int, float, np.integer, np.floating)) and isinstance(
        actual, (int, float, np.integer, np.floating)
    ):
        left, right = float(expected), float(actual)
        if not np.isfinite(left) or not np.isfinite(right) or abs(left - right) > tolerance:
            raise AssertionError(f"Numeric mismatch at {location}: {left!r} != {right!r}")
        return
    if expected != actual:
        raise AssertionError(f"Value mismatch at {location}: {expected!r} != {actual!r}")


def absolute_scene_metrics(
    labels: np.ndarray,
    candidate: np.ndarray,
    comparator: np.ndarray,
    folds: np.ndarray,
    sensors: np.ndarray,
    fprs: Sequence[float],
) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.uint8)
    candidate = np.asarray(candidate, dtype=np.float64)
    comparator = np.asarray(comparator, dtype=np.float64)
    folds = np.asarray(folds, dtype=np.uint8)
    sensors = np.asarray(sensors, dtype=np.uint8)
    if any(len(values) != len(labels) for values in (candidate, comparator, folds, sensors)):
        raise ValueError("Scene metric vectors are not aligned")

    def ap(rows: np.ndarray, values: np.ndarray) -> float:
        if set(np.unique(labels[rows]).tolist()) != {0, 1}:
            raise ValueError("Every reported AP stratum must contain both classes")
        return float(average_precision_score(labels[rows], values[rows]))

    def ap_pair(rows: np.ndarray) -> dict[str, float]:
        candidate_ap = ap(rows, candidate)
        comparator_ap = ap(rows, comparator)
        return {
            "candidate": candidate_ap,
            "comparator": comparator_ap,
            "delta": candidate_ap - comparator_ap,
        }

    pooled_rows = np.ones(len(labels), dtype=bool)
    by_fold = {str(int(value)): ap_pair(folds == value) for value in np.unique(folds)}
    by_sensor = {
        SENSOR_NAMES.get(int(value), f"sensor-{int(value)}"): ap_pair(sensors == value)
        for value in np.unique(sensors)
    }
    recall_curve: list[dict[str, float]] = []
    for fpr in map(float, fprs):
        candidate_recall = matched_fpr_recall(labels, candidate, fpr)
        comparator_recall = matched_fpr_recall(labels, comparator, fpr)
        recall_curve.append(
            {
                "fpr": fpr,
                "candidate_recall": candidate_recall,
                "comparator_recall": comparator_recall,
                "delta": candidate_recall - comparator_recall,
            }
        )
    return {
        "rows": int(len(labels)),
        "positive": int(np.count_nonzero(labels)),
        "negative": int(np.count_nonzero(labels == 0)),
        "pooled_average_precision": ap_pair(pooled_rows),
        "fold_average_precision": by_fold,
        "sensor_average_precision": by_sensor,
        "matched_fpr_recall_curve": recall_curve,
        "matched_fpr_mean_delta": float(np.mean([row["delta"] for row in recall_curve])),
    }


def absolute_dense_metrics(
    candidate_counts: np.ndarray,
    comparator_counts: np.ndarray,
) -> dict[str, Any]:
    candidate_counts = np.asarray(candidate_counts, dtype=np.int64)
    comparator_counts = np.asarray(comparator_counts, dtype=np.int64)
    if candidate_counts.shape != comparator_counts.shape or candidate_counts.ndim != 2:
        raise ValueError("Dense count arrays must be matching Nx3 arrays")

    def summarize(values: np.ndarray) -> dict[str, Any]:
        total = values.sum(axis=0, dtype=np.int64)
        return {
            "true_positive": int(total[0]),
            "false_positive": int(total[1]),
            "false_negative": int(total[2]),
            "intersection_over_union": aggregate_iou(values),
        }

    candidate = summarize(candidate_counts)
    comparator = summarize(comparator_counts)
    return {
        "candidate": candidate,
        "comparator": comparator,
        "iou_delta": candidate["intersection_over_union"]
        - comparator["intersection_over_union"],
    }


def validate_access_ledger(ledger: Mapping[str, Any]) -> None:
    required = {
        "comparator_integrity_bytes_hashed": True,
        "comparator_values_decoded": True,
        "held_folds_opened": [3, 4],
        "folds_0_1_2_opened": False,
        "external_or_official_evidence_opened": False,
    }
    for key, expected in required.items():
        if ledger.get(key) != expected:
            raise RuntimeError(f"Result access-ledger mismatch for {key}: {ledger.get(key)!r}")


def build_descriptive_report(
    reporting_protocol: Mapping[str, Any],
    compact_result: Mapping[str, Any],
    candidate: Mapping[str, np.ndarray],
    scene_comparator: Mapping[str, np.ndarray],
    comparator_dense: np.ndarray,
    recomputed_metrics: Mapping[str, Any],
    *,
    compact_result_sha256: str,
    reporting_protocol_sha256: str,
    runtime: Mapping[str, Any],
) -> dict[str, Any]:
    tolerance = float(reporting_protocol["independent_consistency_checks"]["numeric_tolerance"])
    assert_nested_close(
        compact_result["metrics"], recomputed_metrics, tolerance=tolerance, location="metrics"
    )
    checks = compact_result["metrics"]["checks"]
    if bool(compact_result["metrics"]["passed"]) != all(bool(value) for value in checks.values()):
        raise AssertionError("Compact result pass flag does not equal all frozen checks")
    validate_access_ledger(compact_result["access_ledger"])

    execution_protocol = json.loads(
        _resolve(reporting_protocol["frozen_execution"]["protocol"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    scene = absolute_scene_metrics(
        candidate["labels"],
        candidate["scores"],
        scene_comparator["champion_scores"],
        candidate["folds"],
        candidate["sensors"],
        execution_protocol["evaluation"]["matched_fpr_grid"],
    )
    dense = absolute_dense_metrics(candidate["dense_counts"], comparator_dense)

    if abs(scene["pooled_average_precision"]["delta"] - compact_result["metrics"]["pooled_ap_delta"]) > tolerance:
        raise AssertionError("Descriptive pooled AP delta does not match compact result")
    if abs(scene["matched_fpr_mean_delta"] - compact_result["metrics"]["matched_fpr_recall_delta"]) > tolerance:
        raise AssertionError("Descriptive recall delta does not match compact result")
    if abs(dense["iou_delta"] - compact_result["metrics"]["dense_iou_delta"]) > tolerance:
        raise AssertionError("Descriptive dense IoU delta does not match compact result")

    passed = bool(compact_result["metrics"]["passed"])
    language_key = "claim_language_if_development_passes" if passed else "claim_language_if_development_fails"
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": reporting_protocol["scope"],
        "decision": "PASS_DEVELOPMENT" if passed else "REJECT_DEVELOPMENT",
        "claim_language": reporting_protocol["publication_report"][language_key],
        "forbidden_claim": reporting_protocol["publication_report"][
            "forbidden_claim_until_exact_official_gate_passes"
        ],
        "scene": scene,
        "dense": dense,
        "frozen_metrics": compact_result["metrics"],
        "paper_context": reporting_protocol["paper_context"],
        "access_ledger": compact_result["access_ledger"],
        "provenance": {
            "reporting_protocol": reporting_protocol["frozen_execution"],
            "reporting_protocol_sha256": reporting_protocol_sha256,
            "compact_result_path": reporting_protocol["future_inputs"]["compact_result"],
            "compact_result_sha256": compact_result_sha256,
            "candidate_predictions": compact_result["candidate_predictions"],
            "endpoint_states": compact_result["endpoint_states"],
            "consistency_tolerance": tolerance,
            "runtime": dict(runtime),
        },
    }


def markdown_report(report: Mapping[str, Any]) -> str:
    scene = report["scene"]
    dense = report["dense"]
    lines = [
        "# MARS sensor-aware ordinal folds 3/4 — descriptive verification",
        "",
        f"- Decision: **{report['decision']}**",
        f"- Candidate AP: **{scene['pooled_average_precision']['candidate']:.6f}**",
        f"- Comparator AP: **{scene['pooled_average_precision']['comparator']:.6f}**",
        f"- AP delta: **{scene['pooled_average_precision']['delta']:+.6f}**",
        f"- Candidate IoU: **{dense['candidate']['intersection_over_union']:.6f}**",
        f"- Comparator IoU: **{dense['comparator']['intersection_over_union']:.6f}**",
        f"- IoU delta: **{dense['iou_delta']:+.6f}**",
        "",
        report["claim_language"],
        "",
        "## Matched-FPR recall",
        "",
        "| FPR | Candidate | Comparator | Delta |",
        "|---:|---:|---:|---:|",
    ]
    lines.extend(
        f"| {row['fpr']:.3f} | {row['candidate_recall']:.6f} | {row['comparator_recall']:.6f} | {row['delta']:+.6f} |"
        for row in scene["matched_fpr_recall_curve"]
    )
    lines.extend(["", "## Frozen gate checks", ""])
    lines.extend(
        f"- `{name}`: **{bool(value)}**"
        for name, value in report["frozen_metrics"]["checks"].items()
    )
    lines.extend(["", f"Forbidden until the exact official gate passes: {report['forbidden_claim']}", ""])
    return "\n".join(lines)


def write_new(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite one-shot descriptive output: {path}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.rename(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=DEFAULT_REPORTING_PROTOCOL.as_posix())
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    reporting_path = _resolve(args.protocol)
    reporting_protocol = json.loads(reporting_path.read_text(encoding="utf-8"))
    if reporting_protocol.get("status") != "frozen_before_held_outcomes":
        raise RuntimeError("Reporting protocol is not frozen")
    execution_protocol_path = _verify_hash(reporting_protocol["frozen_execution"]["protocol"])
    _verify_hash(reporting_protocol["frozen_execution"]["trainer"])
    _verify_hash(reporting_protocol["frozen_execution"]["model"])
    _verify_hash(reporting_protocol["future_inputs"]["scene_comparator"])
    _verify_hash(reporting_protocol["future_inputs"]["dense_comparator_state"])
    _verify_hash(reporting_protocol["future_inputs"]["paper_v3_benchmark_receipt"])

    execution_protocol = json.loads(execution_protocol_path.read_text(encoding="utf-8"))
    runtime = verify_runtime_environment(require_native_windows=True)
    paths = verify_protocol(execution_protocol, execution_protocol_path, smoke=False)
    compact_path = _resolve(reporting_protocol["future_inputs"]["compact_result"])
    candidate_path = _resolve(reporting_protocol["future_inputs"]["candidate_predictions"])
    endpoint_path = _resolve(reporting_protocol["future_inputs"]["endpoint_states"])
    for path in (compact_path, candidate_path, endpoint_path):
        _require_immutable(path)
    compact_result = json.loads(compact_path.read_text(encoding="utf-8"))
    if compact_result.get("protocol_sha256") != reporting_protocol["frozen_execution"]["protocol"]["sha256"]:
        raise RuntimeError("Compact result is not bound to the frozen execution protocol")
    if compact_result.get("scientific_digest") != reporting_protocol["frozen_execution"]["science_digest"]:
        raise RuntimeError("Compact result science digest mismatch")
    if compact_result["candidate_predictions"].get("path") != reporting_protocol["future_inputs"]["candidate_predictions"]:
        raise RuntimeError("Compact result candidate path mismatch")
    if compact_result["candidate_predictions"].get("sha256") != sha256(candidate_path):
        raise RuntimeError("Compact result candidate hash mismatch")
    if compact_result["endpoint_states"].get("path") != reporting_protocol["future_inputs"]["endpoint_states"]:
        raise RuntimeError("Compact result endpoint-state path mismatch")
    if compact_result["endpoint_states"].get("sha256") != sha256(endpoint_path):
        raise RuntimeError("Compact result endpoint-state hash mismatch")

    with np.load(candidate_path, allow_pickle=False) as source:
        candidate = {name: source[name].copy() for name in source.files}
    scene_comparator = align_comparator(candidate, paths["champion_scene_cache"])
    groups = fold_lookup(paths["fold_protocol"])
    all_records = records_for_folds(paths["manifest"], groups, [3, 4])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    comparator_dense = reconstruct_dense_comparator(
        candidate, all_records, paths, device, scene_comparator
    )
    bootstrap = execution_protocol["bootstrap"]
    recomputed = metric_gates(
        candidate["labels"],
        candidate["scores"],
        scene_comparator["champion_scores"],
        candidate["folds"],
        candidate["sensors"],
        candidate["groups"],
        candidate["dense_counts"],
        comparator_dense,
        replicates=int(bootstrap["replicates"]),
        ap_seed=int(bootstrap["ap_seed"]),
        dense_seed=int(bootstrap["dense_seed"]),
    )
    report = build_descriptive_report(
        reporting_protocol,
        compact_result,
        candidate,
        scene_comparator,
        comparator_dense,
        recomputed,
        compact_result_sha256=sha256(compact_path),
        reporting_protocol_sha256=sha256(reporting_path),
        runtime=runtime,
    )
    outputs = reporting_protocol["outputs"]
    write_new(_resolve(outputs["descriptive_json"]), json.dumps(report, indent=2, sort_keys=True) + "\n")
    write_new(_resolve(outputs["descriptive_markdown"]), markdown_report(report))
    print(json.dumps({"decision": report["decision"], "json": outputs["descriptive_json"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
