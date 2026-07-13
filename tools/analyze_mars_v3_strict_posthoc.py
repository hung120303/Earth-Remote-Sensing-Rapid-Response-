#!/usr/bin/env python3
"""Describe frozen MARS v3 strict errors without selecting a new model or rule."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "EarthRemoteSensingRapidResponse"
if str(MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_ROOT))

from mars_s2l_adapter import iter_manifest, load_sample  # noqa: E402

from acquire_mars_metadata import (  # noqa: E402
    DEFAULT_OUTPUT,
    repo_root,
    sha256,
)
from build_mars_v3_strict_cohort import V3_STRICT_SAMPLES  # noqa: E402
from evaluate_released_marss2l import DEFAULT_METADATA_CSV  # noqa: E402

DEFAULT_CAMPAIGN = Path("reports/experiments/mars_v3_strict_campaign.json")
DEFAULT_JSON = Path("reports/experiments/mars_v3_strict_posthoc_diagnostic.json")
DEFAULT_MARKDOWN = Path("reports/experiments/MARS_V3_STRICT_POSTHOC_DIAGNOSTIC.md")
REFERENCE_TIME = re.compile(r"_(\d{8}T\d{6})_")


def safe_path(root: Path, value: str | Path) -> Path:
    result = (root / value).resolve()
    if result != root and root not in result.parents:
        raise ValueError("Path must resolve beneath the repository root")
    return result


def tracked_dirty(root: Path) -> bool:
    status = subprocess.check_output(
        [
            "git",
            "-c",
            "core.autocrlf=true",
            "status",
            "--porcelain",
            "--untracked-files=no",
        ],
        cwd=root,
        text=True,
    )
    return bool(status.strip())


def rank_strata_indices(values: Sequence[float], names: Sequence[str]) -> list[np.ndarray]:
    """Return deterministic, near-equal rank strata without unstable quantile edges."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size < len(names):
        raise ValueError("Rank strata require at least one observation per stratum")
    if not np.all(np.isfinite(array)):
        raise ValueError("Rank-stratification values must be finite")
    order = np.lexsort((np.arange(array.size), array))
    return [np.asarray(part, dtype=np.int64) for part in np.array_split(order, len(names))]


def load_cache(path: Path, expected_sha256: str) -> dict[str, np.ndarray]:
    if sha256(path) != expected_sha256:
        raise ValueError(f"Prediction-cache identity mismatch: {path}")
    with np.load(path, allow_pickle=False) as source:
        return {name: source[name].copy() for name in source.files}


def aligned(values: dict[str, np.ndarray], sample_ids: np.ndarray) -> dict[str, np.ndarray]:
    positions = {str(value): index for index, value in enumerate(values["sample_ids"])}
    if len(positions) != len(values["sample_ids"]):
        raise ValueError("Prediction cache contains duplicate sample ids")
    try:
        rows = np.asarray([positions[str(value)] for value in sample_ids], dtype=np.int64)
    except KeyError as error:
        raise ValueError(f"Prediction cache is missing sample {error.args[0]}") from error
    return {
        name: value[rows] if value.shape[:1] == values["sample_ids"].shape else value
        for name, value in values.items()
    }


def reference_interval_days(record: dict[str, Any]) -> float:
    target = datetime.fromisoformat(str(record["target_datetime"]))
    match = REFERENCE_TIME.search(str(record["reference_scene_id"]))
    if match is None:
        raise ValueError(f"Cannot parse reference time for {record['sample_id']}")
    reference = datetime.strptime(match.group(1), "%Y%m%dT%H%M%S").replace(
        tzinfo=target.tzinfo
    )
    return abs((target - reference).total_seconds()) / 86_400.0


def metadata_winds(path: Path) -> dict[str, tuple[float, float]]:
    result: dict[str, tuple[float, float]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        for row in csv.DictReader(source):
            identifier = str(row.get("id_loc_image", ""))
            if identifier:
                result[identifier] = (float(row["wind_u"]), float(row["wind_v"]))
    return result


def rate_summary(
    indices: np.ndarray,
    baseline_predictions: np.ndarray,
    candidate_predictions: np.ndarray,
) -> dict[str, Any]:
    per_seed = np.mean(candidate_predictions[:, indices], axis=1)
    return {
        "samples": int(indices.size),
        "released_mars_s2l_rate": float(np.mean(baseline_predictions[indices])),
        "ersrr_seed_mean_rate": float(np.mean(per_seed)),
        "ersrr_seed_standard_deviation": float(np.std(per_seed)),
        "ersrr_per_seed_rate": [float(value) for value in per_seed],
    }


def ranked_strata(
    rows: list[dict[str, Any]],
    indices: np.ndarray,
    field: str,
    names: Sequence[str],
    baseline_predictions: np.ndarray,
    candidate_predictions: np.ndarray,
) -> list[dict[str, Any]]:
    local_values = [float(rows[index][field]) for index in indices]
    result: list[dict[str, Any]] = []
    for name, local in zip(names, rank_strata_indices(local_values, names)):
        selected = indices[local]
        values = np.asarray([rows[index][field] for index in selected], dtype=np.float64)
        result.append(
            {
                "name": name,
                "field": field,
                "minimum": float(np.min(values)),
                "maximum": float(np.max(values)),
                **rate_summary(
                    selected, baseline_predictions, candidate_predictions
                ),
            }
        )
    return result


def country_strata(
    rows: list[dict[str, Any]],
    indices: np.ndarray,
    minimum_samples: int,
    baseline_predictions: np.ndarray,
    candidate_predictions: np.ndarray,
) -> list[dict[str, Any]]:
    countries = Counter(str(rows[index]["country"]) for index in indices)
    result: list[dict[str, Any]] = []
    for country, count in sorted(countries.items(), key=lambda item: (-item[1], item[0])):
        if count < minimum_samples:
            continue
        selected = np.asarray(
            [index for index in indices if str(rows[index]["country"]) == country],
            dtype=np.int64,
        )
        result.append(
            {
                "country": country,
                **rate_summary(selected, baseline_predictions, candidate_predictions),
            }
        )
    return result


def zero_nonzero_strata(
    rows: list[dict[str, Any]],
    indices: np.ndarray,
    field: str,
    baseline_predictions: np.ndarray,
    candidate_predictions: np.ndarray,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for name, predicate in (
        ("zero", lambda value: value == 0.0),
        ("nonzero", lambda value: value > 0.0),
    ):
        selected = np.asarray(
            [index for index in indices if predicate(float(rows[index][field]))],
            dtype=np.int64,
        )
        if selected.size == 0:
            continue
        values = np.asarray([rows[index][field] for index in selected], dtype=np.float64)
        result.append(
            {
                "name": name,
                "field": field,
                "minimum": float(np.min(values)),
                "maximum": float(np.max(values)),
                **rate_summary(selected, baseline_predictions, candidate_predictions),
            }
        )
    return result


def atlas_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        name: row[name]
        for name in (
            "sample_id",
            "group_id",
            "country",
            "latitude",
            "longitude",
            "label",
            "truth_plume_pixels",
            "wind_speed_mps",
            "reference_interval_days",
            "cloud_fraction",
            "released_prediction",
            "released_score",
            "ersrr_seed_hits",
            "ersrr_mean_score",
        )
    }


def error_atlas(rows: list[dict[str, Any]], limit: int = 12) -> dict[str, Any]:
    positives = [row for row in rows if row["label"] == 1]
    negatives = [row for row in rows if row["label"] == 0]
    consensus_false_negatives = sorted(
        [row for row in positives if row["ersrr_seed_hits"] == 0],
        key=lambda row: (-row["truth_plume_pixels"], row["sample_id"]),
    )[:limit]
    consensus_true_positives = sorted(
        [row for row in positives if row["ersrr_seed_hits"] == 5],
        key=lambda row: (-row["truth_plume_pixels"], row["sample_id"]),
    )[:limit]
    persistent_false_positives = sorted(
        [row for row in negatives if row["ersrr_seed_hits"] > 0],
        key=lambda row: (
            -row["ersrr_seed_hits"],
            -row["ersrr_mean_score"],
            row["sample_id"],
        ),
    )[:limit]
    baseline_only_true_positives = sorted(
        [
            row
            for row in positives
            if row["released_prediction"] == 1 and row["ersrr_seed_hits"] == 0
        ],
        key=lambda row: (-row["truth_plume_pixels"], row["sample_id"]),
    )[:limit]
    return {
        "selection_rule": (
            "Deterministic post-hoc examples: largest plume area for consensus FN/TP and "
            "baseline-only TP; then most seed hits and highest mean frozen score for FP; "
            "sample id breaks ties. No visual selection."
        ),
        "consensus_false_negatives": [atlas_row(row) for row in consensus_false_negatives],
        "consensus_true_positives": [atlas_row(row) for row in consensus_true_positives],
        "persistent_false_positives": [atlas_row(row) for row in persistent_false_positives],
        "baseline_only_true_positives": [
            atlas_row(row) for row in baseline_only_true_positives
        ],
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def markdown_strata(lines: list[str], title: str, items: list[dict[str, Any]]) -> None:
    lines.extend(
        [
            f"## {title}",
            "",
            "| Stratum | Range | n | MARS-S2L rate | ERSRR seed-mean rate | Seed SD |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for item in items:
        lines.append(
            f"| {item['name']} | {item['minimum']:.3f}-{item['maximum']:.3f} | "
            f"{item['samples']} | {item['released_mars_s2l_rate']:.3f} | "
            f"{item['ersrr_seed_mean_rate']:.3f} | "
            f"{item['ersrr_seed_standard_deviation']:.3f} |"
        )
    lines.append("")


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    agreement = report["agreement"]
    lines = [
        "# ERSRR v3 strict post-hoc diagnostic",
        "",
        "Exploratory analysis of frozen predictions. It must not be used to retune v3; any resulting v4 hypothesis requires a new untouched test cohort.",
        "",
        f"- Cohort: {report['cohort']['samples']} scenes / {report['cohort']['groups']} frozen 25 km groups",
        f"- Positives / negatives: {report['cohort']['positives']} / {report['cohort']['negatives']}",
        f"- Consensus ERSRR plume misses (0/5 seeds): {agreement['positive_seed_hit_counts']['0']}",
        f"- Plumes detected by all ERSRR seeds: {agreement['positive_seed_hit_counts']['5']}",
        f"- No-plume scenes falsely flagged by at least one ERSRR seed: {agreement['negative_any_seed_false_positive']}",
        "",
    ]
    markdown_strata(lines, "Positive recall by plume area", report["positive_strata"]["plume_area"])
    markdown_strata(lines, "Positive recall by wind speed", report["positive_strata"]["wind_speed"])
    markdown_strata(lines, "Positive recall by target/reference interval", report["positive_strata"]["reference_interval"])
    markdown_strata(lines, "Negative FPR by wind speed", report["negative_strata"]["wind_speed"])
    markdown_strata(lines, "Negative FPR by target/reference interval", report["negative_strata"]["reference_interval"])
    lines.extend(
        [
            "## Interpretation boundary",
            "",
            report["decision"],
            "",
            "The JSON report contains country and cloud strata plus deterministic error-atlas sample ids.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-dir", default=DEFAULT_OUTPUT.as_posix())
    parser.add_argument("--metadata-csv", default=DEFAULT_METADATA_CSV.as_posix())
    parser.add_argument("--campaign", default=DEFAULT_CAMPAIGN.as_posix())
    parser.add_argument("--output-json", default=DEFAULT_JSON.as_posix())
    parser.add_argument("--output-markdown", default=DEFAULT_MARKDOWN.as_posix())
    args = parser.parse_args()

    root = repo_root()
    metadata_dir = safe_path(root, args.metadata_dir)
    metadata_csv = safe_path(root, args.metadata_csv)
    campaign_path = safe_path(root, args.campaign)
    output_json = safe_path(root, args.output_json)
    output_markdown = safe_path(root, args.output_markdown)
    campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
    if campaign.get("scope") != "frozen_v3_five_seed_full_strict_campaign":
        raise ValueError("Expected the frozen five-seed strict campaign")

    manifest = metadata_dir / V3_STRICT_SAMPLES
    manifest_sha = sha256(manifest)
    if manifest_sha != campaign["cohort"]["strict_manifest_sha256"]:
        raise ValueError("Strict manifest differs from the frozen campaign")
    records = list(iter_manifest(manifest))
    if len(records) != int(campaign["cohort"]["samples"]):
        raise ValueError("Strict manifest row count differs from the campaign")
    by_id = {str(record["sample_id"]): record for record in records}
    if len(by_id) != len(records):
        raise ValueError("Strict manifest contains duplicate sample ids")

    baseline_input = campaign["inputs"]["released_baseline_report"]["scene_cache"]
    baseline = load_cache(
        safe_path(root, baseline_input["path"]), baseline_input["sha256"]
    )
    sample_ids = baseline["sample_ids"]
    labels = baseline["labels"].astype(np.uint8)
    groups = baseline["groups"]
    baseline_predictions = baseline["predictions"].astype(np.uint8)
    baseline_scores = baseline["scores"].astype(np.float64)
    if set(str(value) for value in sample_ids) != set(by_id):
        raise ValueError("Baseline cache and strict manifest sample ids differ")

    candidate_predictions: list[np.ndarray] = []
    candidate_scores: list[np.ndarray] = []
    seeds: list[int] = []
    for item in campaign["inputs"]["v3_reports"]:
        cache_input = item["scene_cache"]
        cache = aligned(
            load_cache(safe_path(root, cache_input["path"]), cache_input["sha256"]),
            sample_ids,
        )
        if not np.array_equal(cache["labels"], labels):
            raise ValueError("Candidate and baseline labels differ")
        if not np.array_equal(cache["groups"], groups):
            raise ValueError("Candidate and baseline groups differ")
        candidate_predictions.append(cache["primary_predictions"].astype(np.uint8))
        candidate_scores.append(cache["primary_scores"].astype(np.float64))
        seeds.append(int(cache["seed"][0]))
    if seeds != [101, 202, 303, 404, 505]:
        raise ValueError("Candidate caches do not contain the five fixed seeds in order")
    prediction_matrix = np.stack(candidate_predictions)
    score_matrix = np.stack(candidate_scores)

    winds = metadata_winds(metadata_csv)
    rows: list[dict[str, Any]] = []
    for index, identifier_value in enumerate(sample_ids):
        identifier = str(identifier_value)
        record = by_id[identifier]
        wind_u, wind_v = winds[identifier]
        truth_pixels = 0
        if int(labels[index]) == 1:
            sample = load_sample(metadata_dir, record, require_enhancement=False)
            truth_pixels = int(
                np.count_nonzero(sample.plume_mask & sample.observable_mask)
            )
            if truth_pixels <= 0:
                raise ValueError(f"Positive scene has no observable truth pixels: {identifier}")
        rows.append(
            {
                "sample_id": identifier,
                "group_id": str(groups[index]),
                "country": str(record.get("country") or "Unknown"),
                "latitude": float(record["latitude"]),
                "longitude": float(record["longitude"]),
                "label": int(labels[index]),
                "truth_plume_pixels": truth_pixels,
                "wind_speed_mps": float(np.hypot(wind_u, wind_v)),
                "reference_interval_days": reference_interval_days(record),
                "cloud_fraction": float(record["cloud_fraction"]),
                "released_prediction": int(baseline_predictions[index]),
                "released_score": float(baseline_scores[index]),
                "ersrr_seed_hits": int(np.sum(prediction_matrix[:, index])),
                "ersrr_mean_score": float(np.mean(score_matrix[:, index])),
            }
        )

    positive = np.flatnonzero(labels == 1)
    negative = np.flatnonzero(labels == 0)
    tertiles = ("low", "middle", "high")
    positive_strata = {
        "plume_area": ranked_strata(
            rows,
            positive,
            "truth_plume_pixels",
            ("small", "medium", "large"),
            baseline_predictions,
            prediction_matrix,
        ),
        "wind_speed": ranked_strata(
            rows, positive, "wind_speed_mps", tertiles, baseline_predictions, prediction_matrix
        ),
        "reference_interval": ranked_strata(
            rows,
            positive,
            "reference_interval_days",
            tertiles,
            baseline_predictions,
            prediction_matrix,
        ),
        "cloud_fraction": zero_nonzero_strata(
            rows, positive, "cloud_fraction", baseline_predictions, prediction_matrix
        ),
        "country_minimum_5_positives": country_strata(
            rows, positive, 5, baseline_predictions, prediction_matrix
        ),
    }
    negative_strata = {
        "wind_speed": ranked_strata(
            rows, negative, "wind_speed_mps", tertiles, baseline_predictions, prediction_matrix
        ),
        "reference_interval": ranked_strata(
            rows,
            negative,
            "reference_interval_days",
            tertiles,
            baseline_predictions,
            prediction_matrix,
        ),
        "cloud_fraction": zero_nonzero_strata(
            rows, negative, "cloud_fraction", baseline_predictions, prediction_matrix
        ),
        "country_minimum_100_negatives": country_strata(
            rows, negative, 100, baseline_predictions, prediction_matrix
        ),
    }

    positive_hits = Counter(int(np.sum(prediction_matrix[:, index])) for index in positive)
    negative_hits = Counter(int(np.sum(prediction_matrix[:, index])) for index in negative)
    agreement = {
        "positive_seed_hit_counts": {str(value): int(positive_hits[value]) for value in range(6)},
        "negative_seed_hit_counts": {str(value): int(negative_hits[value]) for value in range(6)},
        "positive_baseline_only_vs_ersrr_zero": int(
            np.count_nonzero((baseline_predictions[positive] == 1) & (np.sum(prediction_matrix[:, positive], axis=0) == 0))
        ),
        "positive_both_baseline_and_any_ersrr": int(
            np.count_nonzero((baseline_predictions[positive] == 1) & (np.sum(prediction_matrix[:, positive], axis=0) > 0))
        ),
        "positive_neither_baseline_nor_ersrr": int(
            np.count_nonzero((baseline_predictions[positive] == 0) & (np.sum(prediction_matrix[:, positive], axis=0) == 0))
        ),
        "negative_any_seed_false_positive": int(
            np.count_nonzero(np.sum(prediction_matrix[:, negative], axis=0) > 0)
        ),
        "negative_all_seed_false_positive": int(
            np.count_nonzero(np.sum(prediction_matrix[:, negative], axis=0) == 5)
        ),
    }

    report = {
        "schema_version": 1,
        "scope": "posthoc_v3_strict_diagnostic_no_tuning",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "cohort": {
            "samples": len(rows),
            "groups": len(set(str(value) for value in groups)),
            "positives": int(positive.size),
            "negatives": int(negative.size),
            "strict_manifest_sha256": manifest_sha,
        },
        "fixed_seeds": seeds,
        "positive_strata": positive_strata,
        "negative_strata": negative_strata,
        "agreement": agreement,
        "error_atlas": error_atlas(rows),
        "decision": (
            "Exploratory description after the once-only strict campaign. Do not retune v3 "
            "from these strata or examples. Use them only to preregister v4 hypotheses and "
            "acquisition strata, then evaluate v4 on a newly untouched cohort."
        ),
        "inputs": {
            "campaign": {
                "path": campaign_path.relative_to(root).as_posix(),
                "sha256": sha256(campaign_path),
            },
            "metadata_csv": {
                "path": metadata_csv.relative_to(root).as_posix(),
                "sha256": sha256(metadata_csv),
            },
        },
        "provenance": {
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True
            ).strip(),
            "git_tracked_worktree_dirty_at_start": tracked_dirty(root),
            "script": "tools/analyze_mars_v3_strict_posthoc.py",
            "script_sha256": sha256(Path(__file__).resolve()),
        },
    }
    write_json(output_json, report)
    write_markdown(output_markdown, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
