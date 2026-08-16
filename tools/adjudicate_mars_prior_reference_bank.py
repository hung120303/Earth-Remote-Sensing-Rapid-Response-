"""Adjudicate the frozen MARS prior-reference bank without opening outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_PROTOCOL = Path("configs/mars_prior_reference_bank_adjudication_protocol.json")

ALLOWED_MANIFEST_FIELDS = {
    "exact_grid_candidates",
    "fallback_to_original_only",
    "fold",
    "grid_key",
    "original_reference_distance",
    "physical_location_id",
    "recent_pool_candidates",
    "sample_id",
    "selected_distances",
    "selected_sample_ids",
    "selected_target_scene_ids",
    "sensor_family",
    "strictly_prior_clear_candidates",
    "target_datetime",
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def repo_path(root: Path, value: str) -> Path:
    path = (root / value).resolve()
    path.relative_to(root)
    return path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def validate_hash(path: Path, expected: str) -> None:
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"SHA-256 mismatch for {path}: {actual} != {expected}")


def audit_selection_manifest(
    path: Path,
    *,
    allowed_folds: set[int],
    maximum_selected_references: int,
    maximum_recent_pool: int,
) -> dict[str, int]:
    counts = {
        "rows": 0,
        "sentinel_rows": 0,
        "sentinel_rows_with_selected_reference": 0,
        "sentinel_rows_with_five_selected_references": 0,
        "landsat_rows": 0,
        "fallback_rows": 0,
    }
    sample_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            row = json.loads(line)
            if set(row) != ALLOWED_MANIFEST_FIELDS:
                extra = sorted(set(row) - ALLOWED_MANIFEST_FIELDS)
                missing = sorted(ALLOWED_MANIFEST_FIELDS - set(row))
                raise ValueError(
                    f"Unsafe manifest schema at line {line_number}; "
                    f"extra={extra}, missing={missing}"
                )
            sample_id = str(row["sample_id"])
            if sample_id in sample_ids:
                raise ValueError(
                    f"Duplicate sample_id at line {line_number}: {sample_id}"
                )
            sample_ids.add(sample_id)
            if int(row["fold"]) not in allowed_folds:
                raise ValueError(f"Unauthorized fold at line {line_number}")

            selected_ids = list(row["selected_sample_ids"])
            selected_scenes = list(row["selected_target_scene_ids"])
            selected_distances = list(row["selected_distances"])
            if not (
                len(selected_ids) == len(selected_scenes) == len(selected_distances)
            ):
                raise ValueError(
                    f"Misaligned selected-reference fields at line {line_number}"
                )
            if len(selected_ids) > maximum_selected_references:
                raise ValueError(f"Too many selected references at line {line_number}")
            if len(set(selected_ids)) != len(selected_ids):
                raise ValueError(f"Duplicate selected reference at line {line_number}")
            if int(row["recent_pool_candidates"]) > maximum_recent_pool:
                raise ValueError(f"Recent-pool overflow at line {line_number}")
            if int(row["exact_grid_candidates"]) < len(selected_ids):
                raise ValueError(
                    f"Selected non-exact-grid reference at line {line_number}"
                )
            if bool(row["fallback_to_original_only"]) != (len(selected_ids) == 0):
                raise ValueError(
                    f"Fallback/reference inconsistency at line {line_number}"
                )

            counts["rows"] += 1
            if row["fallback_to_original_only"]:
                counts["fallback_rows"] += 1
            if row["sensor_family"] == "Sentinel-2":
                counts["sentinel_rows"] += 1
                if selected_ids:
                    counts["sentinel_rows_with_selected_reference"] += 1
                if len(selected_ids) == maximum_selected_references:
                    counts["sentinel_rows_with_five_selected_references"] += 1
            elif row["sensor_family"] == "Landsat":
                counts["landsat_rows"] += 1
                if selected_ids:
                    raise ValueError(
                        f"Landsat alternate reference is prohibited at line {line_number}"
                    )
            else:
                raise ValueError(f"Unexpected sensor at line {line_number}")
    return counts


def write_markdown(path: Path, result: dict[str, Any]) -> None:
    metrics = result["metrics"]
    gates = result["gates"]
    lines = [
        "# MARS prior-reference-bank alignment adjudication",
        "",
        f"Generated: {result['generated_at_utc']}.",
        "",
        "This post-feasibility engineering adjudication preserves the original FAIL. "
        "It asks whether exact-grid filtering plus exact original-pair fallback makes "
        "a separate label-free inference diagnostic safe and sufficiently representative.",
        "",
        "| Check | Observed | Required | Pass |",
        "|---|---:|---:|:---:|",
        f"| Sentinel-2 reference coverage | {metrics['reference_coverage']:.4%} | "
        f">= {gates['reference_coverage']['minimum']:.2%} | "
        f"{'yes' if gates['reference_coverage']['pass'] else 'no'} |",
        f"| Five-reference coverage | {metrics['five_reference_coverage']:.4%} | "
        f">= {gates['five_reference_coverage']['minimum']:.2%} | "
        f"{'yes' if gates['five_reference_coverage']['pass'] else 'no'} |",
        f"| Grid-excluded among prior-candidate rows | "
        f"{metrics['grid_excluded_candidate_fraction']:.4%} | "
        f"<= {gates['grid_excluded_candidate_fraction']['maximum']:.2%} | "
        f"{'yes' if gates['grid_excluded_candidate_fraction']['pass'] else 'no'} |",
        f"| Label/model access absent | n/a | required | "
        f"{'yes' if gates['outcome_blind']['pass'] else 'no'} |",
        "",
        f"**Decision:** {result['decision']}",
        "",
        "Passing authorizes only the separately preregistered alternate-reference score "
        "extraction on folds 3/4. It does not authorize training, threshold selection, "
        "external replay, official-test access, or a performance claim.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run(protocol_path: Path) -> dict[str, Any]:
    root = repo_root()
    protocol = load_json(protocol_path)
    for implementation in protocol["implementation"].values():
        validate_hash(repo_path(root, implementation["path"]), implementation["sha256"])
    source = protocol["inputs"]["initial_audit"]
    report_path = repo_path(root, source["path"])
    validate_hash(report_path, source["sha256"])
    report = load_json(report_path)

    selection = report["artifacts"]["selection_manifest"]
    if selection["path"] != protocol["inputs"]["selection_manifest_from_initial_audit"]:
        raise ValueError("Initial audit points to an unexpected selection manifest")
    selection_path = repo_path(root, selection["path"])
    validate_hash(selection_path, selection["sha256"])
    counts = audit_selection_manifest(
        selection_path,
        allowed_folds=set(protocol["scope"]["folds"]),
        maximum_selected_references=int(
            protocol["integrity"]["maximum_selected_references"]
        ),
        maximum_recent_pool=int(protocol["integrity"]["maximum_recent_pool"]),
    )

    summary = report["summary"]
    expected_counts = {
        key: int(summary[key])
        for key in (
            "sentinel_rows",
            "sentinel_rows_with_selected_reference",
            "sentinel_rows_with_five_selected_references",
        )
    }
    for key, expected in expected_counts.items():
        if counts[key] != expected:
            raise ValueError(f"Manifest/report count mismatch for {key}")

    sentinel_rows = int(summary["sentinel_rows"])
    raw_candidate_rows = int(
        summary["sentinel_rows_with_any_strict_prior_clear_candidate"]
    )
    exact_candidate_rows = int(summary["sentinel_rows_with_any_exact_grid_candidate"])
    metrics = {
        "reference_coverage": counts["sentinel_rows_with_selected_reference"]
        / sentinel_rows,
        "five_reference_coverage": counts["sentinel_rows_with_five_selected_references"]
        / sentinel_rows,
        "grid_excluded_candidate_rows": raw_candidate_rows - exact_candidate_rows,
        "grid_excluded_candidate_fraction": (
            (raw_candidate_rows - exact_candidate_rows) / raw_candidate_rows
        ),
        "sentinel_fallback_fraction": (
            sentinel_rows - counts["sentinel_rows_with_selected_reference"]
        )
        / sentinel_rows,
    }
    thresholds = protocol["gates"]
    outcome_blind = (
        report["outcome_access"]["labels_accessed"] is False
        and report["outcome_access"]["plume_masks_accessed"] is False
        and report["outcome_access"]["model_scores_accessed"] is False
        and report["outcome_access"]["predictions_computed"] is False
    )
    gate_results = {
        "reference_coverage": {
            "observed": metrics["reference_coverage"],
            "minimum": float(thresholds["minimum_sentinel_reference_coverage"]),
            "pass": metrics["reference_coverage"]
            >= float(thresholds["minimum_sentinel_reference_coverage"]),
        },
        "five_reference_coverage": {
            "observed": metrics["five_reference_coverage"],
            "minimum": float(thresholds["minimum_five_reference_coverage"]),
            "pass": metrics["five_reference_coverage"]
            >= float(thresholds["minimum_five_reference_coverage"]),
        },
        "grid_excluded_candidate_fraction": {
            "observed": metrics["grid_excluded_candidate_fraction"],
            "maximum": float(thresholds["maximum_grid_excluded_candidate_fraction"]),
            "pass": metrics["grid_excluded_candidate_fraction"]
            <= float(thresholds["maximum_grid_excluded_candidate_fraction"]),
        },
        "outcome_blind": {"pass": outcome_blind},
    }
    passed = all(item["pass"] for item in gate_results.values())
    decision = (
        "PASS: authorize a separately frozen, label-free alternate-reference score "
        "extraction on folds 3/4 with original-pair fallback."
        if passed
        else "FAIL: do not run alternate-reference model inference."
    )
    output_json = repo_path(root, protocol["outputs"]["json"])
    output_markdown = repo_path(root, protocol["outputs"]["markdown"])
    result = {
        "schema_version": 1,
        "status": "complete_post_feasibility_alignment_adjudication",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "original_audit_decision_preserved": report["decision"],
        "manifest_counts": counts,
        "metrics": metrics,
        "gates": gate_results,
        "pass": passed,
        "decision": decision,
        "claim_boundary": protocol["claim_boundary"],
        "outcome_access": report["outcome_access"],
        "provenance": {
            "protocol": {
                "path": protocol_path.relative_to(root).as_posix(),
                "sha256": sha256(protocol_path),
            },
            "initial_audit": {"path": source["path"], "sha256": sha256(report_path)},
            "selection_manifest": {
                "path": selection["path"],
                "sha256": sha256(selection_path),
                "tracked": False,
            },
            "script": {
                "path": Path(__file__).resolve().relative_to(root).as_posix(),
                "sha256": sha256(Path(__file__).resolve()),
            },
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True
            ).strip(),
        },
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_markdown(output_markdown, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL.as_posix())
    args = parser.parse_args()
    root = repo_root()
    result = run(repo_path(root, args.protocol))
    print(json.dumps({"ok": True, "pass": result["pass"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
