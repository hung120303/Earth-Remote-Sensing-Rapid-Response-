#!/usr/bin/env python3
"""Freeze the leakage-resistant MARS-S2L publication evaluation protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from acquire_mars_metadata import DEFAULT_OUTPUT, REPO_ID, REVISION, checked_output_dir, repo_root, sha256
from build_mars_cohort import COHORT_MANIFEST

COHORT_REPORT = Path("reports/acquisition/mars_s2l_publication_cohort.json")
DEFAULT_PROTOCOL = Path("configs/mars_publication_protocol.json")
DEFAULT_MARKDOWN = Path("reports/acquisition/MARS_S2L_EVALUATION_PROTOCOL.md")
ASSIGNMENTS_NAME = "publication_protocol_assignments.jsonl"
VALIDATION_FRACTION = 0.20
SPLIT_SEED = 20260710
SEARCH_TRIALS = 50_000
FIXED_TRAINING_SEEDS = (101, 202, 303, 404, 505)


def git_commit(root: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


def tracked_dirty(root: Path) -> bool:
    output = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=root, text=True
    )
    return bool(output.strip())


def safe_output(root: Path, value: str) -> Path:
    result = (root / value).resolve()
    if root not in result.parents:
        raise ValueError("Protocol output must resolve beneath the repository root")
    return result


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid cohort JSONL at line {line_number}") from exc
    if len({row["sample_id"] for row in rows}) != len(rows):
        raise ValueError("Cohort manifest contains duplicate sample IDs")
    return rows


def group_statistics(rows: list[dict[str, Any]], split: str) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = defaultdict(lambda: {"rows": 0, "positive": 0, "negative": 0})
    for row in rows:
        if row["split"] != split:
            continue
        item = result[row["group_id"]]
        item["rows"] += 1
        if row["label_state"] == "PLUME":
            item["positive"] += 1
        elif row["label_state"] == "NO_PLUME":
            item["negative"] += 1
        else:
            raise ValueError(f"Unexpected label state: {row['label_state']}")
    return dict(result)


def assignment_score(
    selected: set[str], groups: dict[str, dict[str, int]], totals: dict[str, int]
) -> tuple[float, dict[str, int]] | None:
    counts = {
        key: sum(groups[group][key] for group in selected)
        for key in ("rows", "positive", "negative")
    }
    if len(selected) < 10 or counts["positive"] < 100 or counts["negative"] < 1_000:
        return None
    score = sum(
        abs(counts[key] / totals[key] - VALIDATION_FRACTION) / VALIDATION_FRACTION
        for key in ("rows", "positive", "negative")
    )
    score += 0.10 * abs(len(selected) / len(groups) - VALIDATION_FRACTION) / VALIDATION_FRACTION
    return score, counts


def select_validation_groups(groups: dict[str, dict[str, int]]) -> tuple[set[str], dict[str, Any]]:
    identifiers = sorted(groups)
    totals = {
        key: sum(item[key] for item in groups.values())
        for key in ("rows", "positive", "negative")
    }
    rng = random.Random(SPLIT_SEED)
    best: tuple[float, tuple[str, ...], dict[str, int]] | None = None
    for _ in range(SEARCH_TRIALS):
        selected = {
            group for group in identifiers if rng.random() < VALIDATION_FRACTION
        }
        evaluated = assignment_score(selected, groups, totals)
        if evaluated is None:
            continue
        score, counts = evaluated
        candidate = (score, tuple(sorted(selected)), counts)
        if best is None or candidate[:2] < best[:2]:
            best = candidate
    if best is None:
        raise RuntimeError("Could not construct a valid group-disjoint validation assignment")
    score, ordered, counts = best
    selected = set(ordered)
    training_counts = {
        key: totals[key] - counts[key] for key in ("rows", "positive", "negative")
    }
    return selected, {
        "method": "seeded 50,000-candidate group Bernoulli search minimizing row/positive/negative fraction error",
        "seed": SPLIT_SEED,
        "trials": SEARCH_TRIALS,
        "target_fraction": VALIDATION_FRACTION,
        "objective_score": round(score, 9),
        "source_group_count": len(groups),
        "validation_group_count": len(selected),
        "training_group_count": len(groups) - len(selected),
        "training_counts": training_counts,
        "validation_counts": counts,
        "validation_fractions": {
            key: round(counts[key] / totals[key], 8)
            for key in ("rows", "positive", "negative")
        },
        "validation_group_ids": sorted(selected),
    }


def counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "positive": sum(row["label_state"] == "PLUME" for row in rows),
        "negative": sum(row["label_state"] == "NO_PLUME" for row in rows),
        "groups": len({row["group_id"] for row in rows}),
        "locations": len({row["physical_location_id"] for row in rows}),
    }


def make_assignments(
    rows: list[dict[str, Any]], validation_groups: set[str]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    training_groups = {row["group_id"] for row in rows if row["split"] == "train"}
    strict_test_groups = {
        row["group_id"]
        for row in rows
        if row["split"] == "test" and row["group_id"] not in training_groups
    }
    assignments: list[dict[str, Any]] = []
    for row in rows:
        official = row["split"]
        if official == "train":
            role = "internal_validation" if row["group_id"] in validation_groups else "internal_training"
        elif official == "val":
            role = "official_validation_comparability_only"
        elif row["group_id"] in strict_test_groups:
            role = "strict_spatial_test"
        else:
            role = "official_test_overlap_comparability_only"
        assignments.append(
            {
                "sample_id": row["sample_id"],
                "group_id": row["group_id"],
                "physical_location_id": row["physical_location_id"],
                "official_split": official,
                "research_role": role,
                "label_state": row["label_state"],
                "test_only_location": bool(row["test_only_location"]),
                "official_validation_comparability": official == "val",
                "official_test_comparability": official == "test",
                "strict_25km_test": official == "test" and row["group_id"] in strict_test_groups,
            }
        )
    assignments.sort(key=lambda item: item["sample_id"])
    role_by_id = {item["sample_id"]: item["research_role"] for item in assignments}
    by_role = {
        role: counts([row for row in rows if role_by_id[row["sample_id"]] == role])
        for role in sorted(set(role_by_id.values()))
    }
    internal_train_groups = {
        item["group_id"] for item in assignments if item["research_role"] == "internal_training"
    }
    internal_val_groups = {
        item["group_id"] for item in assignments if item["research_role"] == "internal_validation"
    }
    strict_groups_observed = {
        item["group_id"] for item in assignments if item["research_role"] == "strict_spatial_test"
    }
    invariants = {
        "internal_train_validation_group_overlap": len(internal_train_groups & internal_val_groups),
        "official_train_strict_test_group_overlap": len(training_groups & strict_groups_observed),
        "strict_test_groups_expected": len(strict_test_groups),
        "strict_test_groups_observed": len(strict_groups_observed),
        "all_assignments_unique": len({item["sample_id"] for item in assignments}) == len(assignments),
        "all_rows_assigned": len(assignments) == len(rows),
    }
    if (
        invariants["internal_train_validation_group_overlap"] != 0
        or invariants["official_train_strict_test_group_overlap"] != 0
        or not invariants["all_assignments_unique"]
        or not invariants["all_rows_assigned"]
    ):
        raise ValueError(f"Protocol leakage invariant failed: {invariants}")
    return assignments, {"counts_by_role": by_role, "invariants": invariants}


def serialized_assignment(row: dict[str, Any]) -> str:
    return json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"


def assignment_identity(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(serialized_assignment(row).encode("utf-8"))
    return digest.hexdigest()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as target:
        for row in rows:
            target.write(serialized_assignment(row))
    os.replace(temporary, path)


def build_protocol(
    root: Path,
    cohort_report: dict[str, Any],
    cohort_path: Path,
    assignment_path: Path,
    assignment_summary: dict[str, Any],
    split_search: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "frozen_before_full_cohort_training",
        "research_question": "Can a dual-temporal physics-guided selective detector reduce false alarms at matched plume recall on geographically isolated Sentinel-2 sites?",
        "data": {
            "repository": REPO_ID,
            "revision": REVISION,
            "cohort_manifest": cohort_path.relative_to(root).as_posix(),
            "cohort_manifest_sha256": sha256(cohort_path),
            "sample_count": cohort_report["cohort"]["sample_count"],
            "sensor": "Sentinel-2 MSI",
            "product_level": "L1C",
            "label_states": ["PLUME", "NO_PLUME", "UNCERTAIN", "UNOBSERVABLE"],
            "enhancement_units": "unresolved; raw values excluded from unit-bearing primary claims",
        },
        "assignments": {
            "path": assignment_path.relative_to(root).as_posix(),
            "sha256": sha256(assignment_path),
            "bytes": assignment_path.stat().st_size,
            **assignment_summary,
            "internal_split_search": split_search,
        },
        "evaluation_views": {
            "model_selection": "internal_validation only; group-disjoint from internal_training",
            "primary_test": "strict_spatial_test: official test rows whose 25 km connected component contains no official train row",
            "official_comparability": "report untouched released validation and test splits, explicitly labeled as location-overlapping",
            "site_novelty_secondary": "official test rows at physical locations absent from train/validation; weaker than 25 km isolation",
            "external": "locked EMIT V002 cohort after architecture and thresholds are frozen",
        },
        "outputs": {
            "states": ["PLUME", "NO_PLUME", "UNOBSERVABLE_OR_UNCERTAIN"],
            "plume": "scene probability plus binary mask",
            "no_plume": "allowed only for observable accepted scenes below the lower calibrated threshold",
            "abstain": "required for cloud, invalid coverage, low sensitivity, missing reference, disagreement, or intermediate confidence",
        },
        "operating_rule": {
            "calibration_data": "internal_validation only",
            "primary_endpoint": "scene recall at false-positive rate <= 0.05 on accepted observable scenes",
            "thresholds": "fit lower no-plume and upper plume thresholds; intermediate probabilities abstain",
            "minimum_component_area": "selected on internal_validation only",
            "test_tuning": "prohibited",
            "promotion_gate": {
                "scene_recall_lower_95ci_min": 0.75,
                "scene_false_positive_rate_max": 0.05,
                "no_plume_specificity_min": 0.95,
                "relative_fpr_reduction_vs_strongest_baseline_min": 0.25,
                "recall_noninferiority_required": True,
            },
        },
        "models": {
            "baselines": [
                "all-no-plume and all-plume priors",
                "MBMP released operating rule",
                "raw and physics logistic models",
                "legacy compact ERSRR ResUNet",
                "released CH4Net",
                "released MARS-S2L",
            ],
            "candidate": "shared six-band target/reference encoders with change fusion, MBMP channel, scene-presence head, segmentation decoder, and observability/abstention head",
            "training_seeds": list(FIXED_TRAINING_SEEDS),
        },
        "metrics": {
            "scene": [
                "precision",
                "recall",
                "specificity",
                "negative_predictive_value",
                "AUPRC",
                "AUROC",
                "Brier",
                "expected_calibration_error",
                "false_positives_per_100_observable_scenes",
            ],
            "segmentation": ["IoU", "Dice", "pixel_average_precision", "false_positive_area", "boundary_error"],
            "object": ["precision", "recall", "detection_by_plume_area", "source_localization_error"],
            "selective": ["coverage_risk", "abstention_rate", "accepted_no_plume_error_rate"],
            "uncertainty": "95% block bootstrap over physical/group units with 2,000 replicates; paired scene comparisons",
        },
        "frozen_invariants": [
            "No group can occur in both internal training and internal validation.",
            "No strict spatial test group can occur in official training.",
            "All thresholds and normalization statistics are fit without any test view.",
            "All five seeds and failures are reported; no best-seed selection.",
            "Official overlapping views are labeled comparability-only.",
            "Raw enhancement values cannot support unit-bearing regression until the producer resolves the unit conflict.",
        ],
        "provenance": {
            "git_commit": git_commit(root),
            "git_tracked_worktree_dirty_at_start": tracked_dirty(root),
            "script": "tools/build_mars_protocol.py",
            "script_sha256": sha256(Path(__file__)),
            "python": sys.version.split()[0],
        },
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_markdown(path: Path, protocol: dict[str, Any]) -> None:
    role_counts = protocol["assignments"]["counts_by_role"]
    search = protocol["assignments"]["internal_split_search"]
    invariants = protocol["assignments"]["invariants"]
    lines = [
        "# MARS-S2L frozen evaluation protocol",
        "",
        f"- Cohort: {protocol['data']['sample_count']:,} pinned Sentinel-2 L1C samples",
        f"- Assignment identity: `{protocol['assignments']['sha256']}`",
        f"- Internal split seed: {search['seed']}; validation groups: {search['validation_group_count']} / {search['source_group_count']}",
        "- Status: frozen before full-cohort training",
        "",
        "| Research role | Rows | Plume | No plume | Groups | Locations |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for role, item in role_counts.items():
        lines.append(
            f"| {role} | {item['rows']:,} | {item['positive']:,} | {item['negative']:,} | {item['groups']:,} | {item['locations']:,} |"
        )
    lines.extend(
        [
            "",
            "## Leakage result",
            "",
            f"- Internal train/validation 25 km group overlap: {invariants['internal_train_validation_group_overlap']}.",
            f"- Official-train/strict-test 25 km group overlap: {invariants['official_train_strict_test_group_overlap']}.",
            "- Released validation and full test remain comparability-only because their physical/proximity groups overlap training.",
            "- The strict spatial test is the primary test. The test-only-location view remains a weaker secondary analysis.",
            "",
            "## Frozen operating contract",
            "",
            "Thresholds, normalization, component area, calibration, and abstention rules are selected on the internal group-disjoint validation set only. The primary endpoint is scene recall at FPR <= 0.05 among observable accepted scenes. `NO_PLUME` requires observability and probability below the lower threshold; intermediate or invalid scenes abstain.",
            "",
            "Promotion requires lower 95% recall CI >= 0.75, FPR <= 0.05, specificity >= 0.95, and at least 25% relative FPR reduction versus the strongest reproduced baseline without inferior recall. Five fixed seeds and 2,000 group-bootstrap replicates are mandatory.",
            "",
            "This protocol defines evaluation; it does not claim that any current model passes the gate.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-dir", default=DEFAULT_OUTPUT.as_posix())
    parser.add_argument("--output-protocol", default=DEFAULT_PROTOCOL.as_posix())
    parser.add_argument("--output-markdown", default=DEFAULT_MARKDOWN.as_posix())
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args()
    if args.dry_run and args.verify_only:
        parser.error("--dry-run and --verify-only are mutually exclusive")
    root = repo_root()
    try:
        metadata_dir = checked_output_dir(root, args.metadata_dir)
        cohort_path = metadata_dir / COHORT_MANIFEST
        assignment_path = metadata_dir / ASSIGNMENTS_NAME
        if not cohort_path.is_file():
            raise FileNotFoundError(f"Missing frozen cohort manifest: {cohort_path}")
        cohort_report_path = root / COHORT_REPORT
        cohort_report = json.loads(cohort_report_path.read_text(encoding="utf-8"))
        expected_hash = cohort_report["local_ignored_artifacts"]["cohort_manifest_sha256"]
        observed_hash = sha256(cohort_path)
        if observed_hash != expected_hash:
            raise ValueError(
                f"Cohort manifest hash mismatch: expected {expected_hash}, got {observed_hash}"
            )
        rows = load_jsonl(cohort_path)
        if len(rows) != int(cohort_report["cohort"]["sample_count"]):
            raise ValueError("Cohort row count does not match the tracked cohort report")
        train_groups = group_statistics(rows, "train")
        validation_groups, split_search = select_validation_groups(train_groups)
        assignments, assignment_summary = make_assignments(rows, validation_groups)
        output_protocol = safe_output(root, args.output_protocol)
        output_markdown = safe_output(root, args.output_markdown)
        expected_assignment_identity = assignment_identity(assignments)
        if args.verify_only:
            if not assignment_path.is_file() or not output_protocol.is_file() or not output_markdown.is_file():
                raise FileNotFoundError("Protocol verification requires the assignments, JSON protocol, and Markdown report")
            observed_assignment_identity = sha256(assignment_path)
            existing_protocol = json.loads(output_protocol.read_text(encoding="utf-8"))
            if observed_assignment_identity != expected_assignment_identity:
                raise ValueError(
                    f"Assignment identity mismatch: expected {expected_assignment_identity}, got {observed_assignment_identity}"
                )
            if existing_protocol["assignments"]["sha256"] != expected_assignment_identity:
                raise ValueError("Tracked protocol references a different assignment identity")
            if existing_protocol["data"]["cohort_manifest_sha256"] != observed_hash:
                raise ValueError("Tracked protocol references a different cohort identity")
            if existing_protocol["assignments"]["invariants"] != assignment_summary["invariants"]:
                raise ValueError("Tracked protocol leakage invariants differ from recomputed invariants")
            payload = {
                "ok": True,
                "verify_only": True,
                "sample_count": len(rows),
                "assignment_sha256": observed_assignment_identity,
                "invariants": assignment_summary["invariants"],
            }
            print(json.dumps(payload, indent=None if args.compact else 2, sort_keys=True))
            return 0
        if args.dry_run:
            payload = {
                "ok": True,
                "dry_run": True,
                "sample_count": len(rows),
                "assignment_sha256": expected_assignment_identity,
                "counts_by_role": assignment_summary["counts_by_role"],
                "invariants": assignment_summary["invariants"],
                "internal_split_search": split_search,
            }
            print(json.dumps(payload, indent=None if args.compact else 2, sort_keys=True))
            return 0
        write_jsonl(assignment_path, assignments)
        protocol = build_protocol(
            root,
            cohort_report,
            cohort_path,
            assignment_path,
            assignment_summary,
            split_search,
        )
        write_json(output_protocol, protocol)
        write_markdown(output_markdown, protocol)
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, indent=None if args.compact else 2))
        return 2
    strict = protocol["assignments"]["counts_by_role"]["strict_spatial_test"]
    payload = {
        "ok": True,
        "dry_run": False,
        "assignment_sha256": protocol["assignments"]["sha256"],
        "strict_test_rows": strict["rows"],
        "strict_test_positive": strict["positive"],
        "strict_test_negative": strict["negative"],
        "output_protocol": output_protocol.relative_to(root).as_posix(),
        "output_markdown": output_markdown.relative_to(root).as_posix(),
    }
    print(json.dumps(payload, indent=None if args.compact else 2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
