#!/usr/bin/env python3
"""Audit released-detector sensitivity that the frozen MARS champion suppresses."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "tools", ROOT / "EarthRemoteSensingRapidResponse"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from acquire_mars_metadata import sha256  # noqa: E402
from train_mars_scene_ranker import metric_summary  # noqa: E402


DEFAULT_PROTOCOL = Path("configs/mars_recall_anchor_diagnostic_protocol.json")


def scalar(value: np.generic | float | int) -> float | int:
    return value.item() if isinstance(value, np.generic) else value


def align_feature_rows(
    champion_ids: np.ndarray,
    feature_ids: np.ndarray,
    feature_folds: np.ndarray,
) -> np.ndarray:
    if len(set(feature_ids.astype(str).tolist())) != feature_ids.size:
        raise ValueError("Scene-feature sample identities are not unique")
    lookup = {sample_id: index for index, sample_id in enumerate(feature_ids.astype(str))}
    try:
        indices = np.asarray([lookup[value] for value in champion_ids.astype(str)], dtype=np.int64)
    except KeyError as error:
        raise ValueError(f"Champion identity missing from scene features: {error}") from error
    if not np.isin(feature_folds[indices], (3, 4)).all():
        raise ValueError("Aligned scene features escaped authorized folds 3/4")
    return indices


def decision_cell(selection: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    rows = int(selection.sum())
    positives = int(labels[selection].sum())
    negatives = rows - positives
    return {
        "rows": rows,
        "positives": positives,
        "negatives": negatives,
        "precision": float(positives / rows) if rows else None,
    }


def decision_table(
    first: np.ndarray,
    second: np.ndarray,
    labels: np.ndarray,
) -> dict[str, Any]:
    return {
        "both": decision_cell(first & second, labels),
        "first_only": decision_cell(first & ~second, labels),
        "second_only": decision_cell(~first & second, labels),
        "neither": decision_cell(~first & ~second, labels),
    }


def standardized_feature_contrasts(
    features: np.ndarray,
    names: np.ndarray,
    positive_selection: np.ndarray,
    negative_selection: np.ndarray,
    maximum: int,
) -> list[dict[str, Any]]:
    if int(positive_selection.sum()) < 2 or int(negative_selection.sum()) < 2:
        return []
    positive = features[positive_selection].astype(np.float64)
    negative = features[negative_selection].astype(np.float64)
    mean_positive = positive.mean(axis=0)
    mean_negative = negative.mean(axis=0)
    variance = (
        positive.var(axis=0, ddof=1) * (positive.shape[0] - 1)
        + negative.var(axis=0, ddof=1) * (negative.shape[0] - 1)
    ) / max(positive.shape[0] + negative.shape[0] - 2, 1)
    denominator = np.sqrt(np.maximum(variance, 1e-12))
    effects = (mean_positive - mean_negative) / denominator
    order = np.argsort(-np.abs(effects), kind="stable")[:maximum]
    return [
        {
            "feature": str(names[index]),
            "standardized_mean_difference": float(effects[index]),
            "positive_mean": float(mean_positive[index]),
            "negative_mean": float(mean_negative[index]),
        }
        for index in order
        if np.isfinite(effects[index])
    ]


def summarize_score(
    labels: np.ndarray,
    scores: np.ndarray,
    sensors: np.ndarray,
) -> dict[str, Any]:
    result = metric_summary(labels, scores, sensors)
    operating = result["operating_point"]
    return {
        "average_precision": float(result["average_precision"]),
        "recall": float(operating["recall"]),
        "false_positive_rate": float(operating["false_positive_rate"]),
        "threshold": float(operating["threshold"]),
        "sensor_average_precision": {
            name: float(value) for name, value in result["sensor_average_precision"].items()
        },
    }


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    metrics = report["score_metrics"]
    table = report["matched_fpr_decision_complementarity"]
    lines = [
        "# MARS released-detector recall-anchor diagnostic",
        "",
        "Development-only folds 3/4 diagnostic; no candidate was fitted or promoted.",
        "",
        "| Score | AP | Recall at <=7.13% FPR | Realized FPR |",
        "|---|---:|---:|---:|",
    ]
    for name, value in metrics.items():
        lines.append(
            f"| {name} | {value['average_precision']:.6f} | "
            f"{value['recall']:.6f} | {value['false_positive_rate']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Matched-FPR decision complementarity",
            "",
            "`first` is the Gaussian+DOFA champion; `second` is the released detector.",
            "",
            "| Cell | Rows | Positives | Negatives | Precision |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name, value in table.items():
        precision = "n/a" if value["precision"] is None else f"{value['precision']:.6f}"
        lines.append(
            f"| {name} | {value['rows']} | {value['positives']} | "
            f"{value['negatives']} | {precision} |"
        )
    lines.extend(["", f"Decision: **{report['decision']}**", ""])
    if report["top_released_rescue_feature_contrasts"]:
        lines.extend(
            [
                "## Strongest released-rescue TP/FP feature contrasts",
                "",
                "| Feature | Standardized mean difference |",
                "|---|---:|",
            ]
        )
        for value in report["top_released_rescue_feature_contrasts"][:10]:
            lines.append(
                f"| {value['feature']} | {value['standardized_mean_difference']:+.4f} |"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL.as_posix())
    args = parser.parse_args()
    protocol_path = (ROOT / args.protocol).resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    inputs = {
        name: (ROOT / contract["path"]).resolve()
        for name, contract in protocol["inputs"].items()
    }
    for name, contract in protocol["inputs"].items():
        if sha256(inputs[name]) != contract["sha256"]:
            raise ValueError(f"Frozen input hash mismatch: {name}")

    with np.load(inputs["champion_scores"], allow_pickle=False) as bundle:
        champion_ids = bundle["sample_ids"].astype(str)
        labels = bundle["labels"].astype(np.uint8)
        sensors = bundle["sensors"].astype(np.uint8)
        groups = bundle["groups"].astype(str)
        folds = bundle["folds"].astype(np.uint8)
        scores = {
            "current_v3": bundle["released_primary_scores"].astype(np.float64),
            "spatial_prithvi": bundle["spatial_prithvi_scores"].astype(np.float64),
            "gaussian_dofa_champion": bundle["champion_scores"].astype(np.float64),
        }
    if set(np.unique(folds).tolist()) != {3, 4}:
        raise ValueError("Champion cache must contain only folds 3/4")
    if champion_ids.size != int(protocol["cohort"]["rows"]):
        raise ValueError("Champion row count differs from protocol")

    with np.load(inputs["scene_features"], allow_pickle=False) as bundle:
        feature_ids = bundle["sample_ids"].astype(str)
        feature_folds = bundle["folds"].astype(np.uint8)
        feature_names = bundle["feature_names"].astype(str)
        feature_values = bundle["features"].astype(np.float64)
    indices = align_feature_rows(champion_ids, feature_ids, feature_folds)
    feature_values = feature_values[indices]
    released_index = int(np.flatnonzero(feature_names == "released_connected_score")[0])
    scores["released_detector"] = feature_values[:, released_index]
    if not np.isfinite(feature_values).all() or not all(
        np.isfinite(value).all() for value in scores.values()
    ):
        raise ValueError("Recall-anchor inputs contain non-finite values")

    score_metrics = {
        name: summarize_score(labels, value, sensors) for name, value in scores.items()
    }
    champion_threshold = score_metrics["gaussian_dofa_champion"]["threshold"]
    released_threshold = score_metrics["released_detector"]["threshold"]
    champion_decision = scores["gaussian_dofa_champion"] >= champion_threshold
    released_matched_decision = scores["released_detector"] >= released_threshold
    matched_table = decision_table(champion_decision, released_matched_decision, labels)

    released_paper_decision = scores["released_detector"] > 0.5
    paper_rescue = released_paper_decision & ~champion_decision
    rescue_strata = []
    for threshold in protocol["fixed_analysis"]["released_rescue_thresholds"]:
        selection = (scores["released_detector"] >= float(threshold)) & ~champion_decision
        cell = decision_cell(selection, labels)
        cell.update(
            {
                "released_threshold": float(threshold),
                "positive_sample_ids": champion_ids[selection & (labels == 1)].tolist(),
                "fold_counts": {
                    str(fold): int(np.sum(selection & (folds == fold))) for fold in (3, 4)
                },
                "sensor_counts": {
                    str(sensor): int(np.sum(selection & (sensors == sensor)))
                    for sensor in sorted(np.unique(sensors).tolist())
                },
            }
        )
        rescue_strata.append(cell)

    contrasts = standardized_feature_contrasts(
        feature_values,
        feature_names,
        paper_rescue & (labels == 1),
        paper_rescue & (labels == 0),
        int(protocol["fixed_analysis"]["maximum_reported_features"]),
    )
    decision_checks = {
        "released_recovers_champion_missed_positive": matched_table["second_only"][
            "positives"
        ]
        > 0,
        "released_only_region_is_nontrivial": matched_table["second_only"]["positives"]
        > 0
        and matched_table["second_only"]["negatives"] > 0,
        "model_aligned_feature_contrast_exists": bool(contrasts)
        and abs(contrasts[0]["standardized_mean_difference"]) >= 0.25,
    }
    passed = all(decision_checks.values())
    report = {
        "schema_version": 1,
        "scope": protocol["scope"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "path": protocol_path.relative_to(ROOT).as_posix(),
            "sha256": sha256(protocol_path),
        },
        "cohort": {
            "rows": int(labels.size),
            "positives": int(labels.sum()),
            "negatives": int((labels == 0).sum()),
            "groups": int(np.unique(groups).size),
            "fold_counts": {
                str(fold): int(np.sum(folds == fold)) for fold in sorted(np.unique(folds))
            },
        },
        "score_metrics": score_metrics,
        "matched_fpr_decision_complementarity": matched_table,
        "released_paper_rule_rescue": decision_cell(paper_rescue, labels),
        "released_rescue_strata": rescue_strata,
        "top_released_rescue_feature_contrasts": contrasts,
        "decision_checks": decision_checks,
        "decision": (
            "continue_to_recall_anchored_architecture"
            if passed
            else "retire_released_detector_rescue_before_fitting"
        ),
        "scientific_boundary": protocol["scientific_boundary"],
    }
    output_json = (ROOT / protocol["outputs"]["json"]).resolve()
    output_markdown = (ROOT / protocol["outputs"]["markdown"]).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_json.with_suffix(".tmp.json")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output_json)
    write_markdown(output_markdown, report)
    print(json.dumps({"decision": report["decision"], "checks": decision_checks}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
