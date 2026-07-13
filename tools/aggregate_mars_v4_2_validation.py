#!/usr/bin/env python3
"""Aggregate the frozen ERSRR v4.2 internal-validation seed campaign.

This script consumes only compact validation reports. It verifies the frozen
architecture, schedule, data identity, provenance hashes, seed set, and
checkpoint-selection rule before applying the predeclared three-seed mean gate.
It never opens the strict spatial cohort or any imagery.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from acquire_mars_metadata import repo_root, sha256

FIXED_SEEDS = (606, 707, 808)
DEFAULT_REPORTS = tuple(
    Path(f"reports/experiments/mars_v4_2_seed{seed}_validation.json")
    for seed in FIXED_SEEDS
)
DEFAULT_JSON = Path("reports/experiments/mars_v4_2_validation_campaign.json")
DEFAULT_MARKDOWN = Path("reports/experiments/MARS_V4_2_VALIDATION_CAMPAIGN.md")
METRIC_FIELDS = {
    "average_precision": ("validation", "average_precision"),
    "auroc": ("validation", "auroc"),
    "recall_at_fpr5": ("validation", "operating_points", "0.05", "recall"),
    "false_positive_rate_at_fpr5": (
        "validation",
        "operating_points",
        "0.05",
        "false_positive_rate",
    ),
    "positive_pixel_dice": ("validation", "positive_pixel_dice"),
}


def nested(document: dict[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = document
    for key in path:
        value = value[key]
    return value


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


def git_commit(root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()


def frozen_contract(report: dict[str, Any]) -> dict[str, Any]:
    """Return fields that must be identical across all three seed reports."""
    training = report["training"]
    return {
        "architecture_revision": report["architecture_revision"],
        "cohort": report["cohort"],
        "model": report["model"],
        "source": report["source"],
        "simulation": report["simulation"],
        "runtime": report["runtime"],
        "v3_internal_reference": report["v3_internal_reference"],
        "training": {
            key: training[key]
            for key in (
                "batch_size",
                "epochs_requested",
                "learning_rate",
                "loss",
                "objective",
                "samples_per_epoch",
                "scene_max_weight",
                "scene_topk_fraction",
                "validation_every",
            )
        },
        "provenance_source_hashes": {
            key: report["provenance"][key]
            for key in (
                "model_source",
                "model_source_sha256",
                "script",
                "script_sha256",
                "simulation_source",
                "simulation_source_sha256",
            )
        },
    }


def validation_rank(validation: dict[str, Any]) -> tuple[float, float, float]:
    return (
        float(validation["average_precision"]),
        float(validation["operating_points"]["0.08"]["recall"]),
        float(validation["positive_pixel_dice"]),
    )


def verify_selected_checkpoint(report: dict[str, Any]) -> None:
    validations = [
        item for item in report["training"]["history"] if item["validation"] is not None
    ]
    if not validations:
        raise ValueError("Seed report contains no validation checkpoints")
    selected = max(validations, key=lambda item: validation_rank(item["validation"]))
    if int(selected["epoch"]) != int(report["training"]["best_epoch"]):
        raise ValueError("Best epoch does not follow the frozen lexicographic rule")
    selected_rank = validation_rank(selected["validation"])
    reported_rank = validation_rank(report["validation"])
    if any(not math.isclose(a, b, rel_tol=0.0, abs_tol=1e-12) for a, b in zip(selected_rank, reported_rank)):
        raise ValueError("Final validation metrics do not match the selected checkpoint")


def verify_seed_report(report: dict[str, Any], expected_seed: int) -> None:
    if report.get("scope") != "v4_internal_validation_selection":
        raise ValueError(f"Seed {expected_seed} has an unexpected report scope")
    if int(report["training"]["seed"]) != expected_seed:
        raise ValueError(f"Expected seed {expected_seed}, found {report['training']['seed']}")
    if report.get("architecture_revision") != "v4.2":
        raise ValueError(f"Seed {expected_seed} is not architecture v4.2")
    if report.get("smoke_test"):
        raise ValueError(f"Seed {expected_seed} is a smoke report")
    if report["cohort"].get("strict_spatial_test_loaded") is not False:
        raise ValueError(f"Seed {expected_seed} loaded the strict spatial cohort")
    if int(report["cohort"].get("group_overlap", -1)) != 0:
        raise ValueError(f"Seed {expected_seed} has fit/validation group overlap")
    if report["provenance"].get("git_tracked_worktree_dirty_at_start") is not False:
        raise ValueError(f"Seed {expected_seed} did not start from a clean tracked worktree")
    if report["artifact"].get("tracked") is not False:
        raise ValueError(f"Seed {expected_seed} checkpoint is reported as tracked")
    if float(report["model"]["scene_topk_fraction"]) != 0.02:
        raise ValueError(f"Seed {expected_seed} does not use the top-2% scene score")
    if float(report["model"]["scene_max_weight"]) != 0.0:
        raise ValueError(f"Seed {expected_seed} still includes a max-logit scene term")
    fpr5 = float(report["validation"]["operating_points"]["0.05"]["false_positive_rate"])
    if fpr5 > 0.05 + 1e-12:
        raise ValueError(f"Seed {expected_seed} exceeds the 5% validation FPR constraint")
    verify_selected_checkpoint(report)


def metric_summary(values: list[float]) -> dict[str, float]:
    if len(values) != len(FIXED_SEEDS) or not all(math.isfinite(value) for value in values):
        raise ValueError("Campaign metrics require exactly three finite seed values")
    return {
        "mean": float(statistics.fmean(values)),
        "standard_deviation": float(statistics.stdev(values)),
        "minimum": float(min(values)),
        "maximum": float(max(values)),
    }


def build_campaign(
    root: Path,
    report_paths: tuple[Path, ...],
    *,
    capture_provenance: bool = True,
) -> dict[str, Any]:
    if len(report_paths) != len(FIXED_SEEDS):
        raise ValueError("The v4.2 campaign requires exactly three reports")
    documents: list[dict[str, Any]] = []
    inputs: list[dict[str, Any]] = []
    for seed, relative in zip(FIXED_SEEDS, report_paths):
        path = relative if relative.is_absolute() else root / relative
        report = json.loads(path.read_text(encoding="utf-8"))
        verify_seed_report(report, seed)
        documents.append(report)
        try:
            display_path = path.resolve().relative_to(root.resolve()).as_posix()
        except ValueError:
            display_path = path.resolve().as_posix()
        inputs.append({"seed": seed, "path": display_path, "sha256": sha256(path)})

    reference_contract = frozen_contract(documents[0])
    for report in documents[1:]:
        if frozen_contract(report) != reference_contract:
            raise ValueError("Seed reports do not share the same frozen campaign contract")

    per_seed: list[dict[str, Any]] = []
    metric_values = {name: [] for name in METRIC_FIELDS}
    for report in documents:
        seed_metrics = {
            name: float(nested(report, path)) for name, path in METRIC_FIELDS.items()
        }
        for name, value in seed_metrics.items():
            metric_values[name].append(value)
        per_seed.append(
            {
                "seed": int(report["training"]["seed"]),
                "best_epoch": int(report["training"]["best_epoch"]),
                **seed_metrics,
                "precision_at_fpr5": float(
                    report["validation"]["operating_points"]["0.05"]["precision"]
                ),
                "threshold_at_fpr5": float(
                    report["validation"]["operating_points"]["0.05"]["threshold"]
                ),
                "recall_at_fpr8": float(
                    report["validation"]["operating_points"]["0.08"]["recall"]
                ),
                "checkpoint_sha256": report["artifact"]["sha256"],
                "training_git_commit": report["provenance"]["git_commit"],
            }
        )

    aggregate = {name: metric_summary(values) for name, values in metric_values.items()}
    v3_reference = documents[0]["v3_internal_reference"]
    v3_mean = v3_reference["mean"]
    checks = {
        "ap_not_below_v3_mean": (
            aggregate["average_precision"]["mean"] >= float(v3_mean["average_precision"])
        ),
        "auroc_not_below_v3_mean": (
            aggregate["auroc"]["mean"] >= float(v3_mean["auroc"])
        ),
        "recall_at_fpr5_not_below_v3_mean": (
            aggregate["recall_at_fpr5"]["mean"] >= float(v3_mean["recall_at_fpr5"])
        ),
        "pixel_dice_not_below_v3_mean": (
            aggregate["positive_pixel_dice"]["mean"]
            >= float(v3_mean["positive_pixel_dice"])
        ),
        "every_seed_fpr_at_most_0_05": all(
            value <= 0.05 + 1e-12 for value in metric_values["false_positive_rate_at_fpr5"]
        ),
    }
    promotion = all(checks.values())
    failed = [name for name, passed in checks.items() if not passed]
    if promotion:
        decision = (
            "Promote the three frozen v4.2 checkpoints to one comparison on the already-opened "
            "strict MARS cohort. Preserve all seed-specific validation thresholds. This comparison "
            "is a development benchmark, not a new untouched paper test."
        )
    else:
        decision = (
            "Do not load the strict MARS cohort for v4.2. The frozen three-seed campaign failed "
            f"the predeclared internal promotion rule on: {', '.join(failed)}. Preserve this result "
            "and revise the research hypothesis before another strict comparison."
        )

    script = Path(__file__).resolve()
    provenance = (
        {
            "git_commit": git_commit(root),
            "git_tracked_worktree_dirty_at_start": tracked_dirty(root),
            "script": script.relative_to(root).as_posix(),
            "script_sha256": sha256(script),
        }
        if capture_provenance
        else {
            "git_commit": None,
            "git_tracked_worktree_dirty_at_start": None,
            "script": script.as_posix(),
            "script_sha256": sha256(script),
        }
    )
    return {
        "schema_version": 1,
        "scope": "v4_2_three_seed_internal_validation_campaign",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "strict_spatial_test_loaded": False,
        "strict_evaluation_authorized": promotion,
        "comparison_status": (
            "internal-development only; not directly comparable to MARS-S2L strict or paper metrics"
        ),
        "inputs": inputs,
        "frozen_contract": reference_contract,
        "per_seed": per_seed,
        "aggregate": aggregate,
        "v3_internal_reference": v3_reference,
        "delta_from_v3_mean": {
            "average_precision": aggregate["average_precision"]["mean"]
            - float(v3_mean["average_precision"]),
            "auroc": aggregate["auroc"]["mean"] - float(v3_mean["auroc"]),
            "recall_at_fpr5": aggregate["recall_at_fpr5"]["mean"]
            - float(v3_mean["recall_at_fpr5"]),
            "positive_pixel_dice": aggregate["positive_pixel_dice"]["mean"]
            - float(v3_mean["positive_pixel_dice"]),
        },
        "promotion_checks": checks,
        "decision": decision,
        "provenance": provenance,
    }


def render_markdown(campaign: dict[str, Any]) -> str:
    aggregate = campaign["aggregate"]
    reference = campaign["v3_internal_reference"]["mean"]
    delta = campaign["delta_from_v3_mean"]
    checks = campaign["promotion_checks"]
    lines = [
        "# ERSRR v4.2 three-seed validation campaign",
        "",
        "Frozen internal-development result. No strict-cohort imagery or labels were loaded.",
        "",
        "| Seed | Best epoch | AP | AUROC | Recall @ <=5% FPR | FPR | Pixel Dice |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in campaign["per_seed"]:
        lines.append(
            f"| {row['seed']} | {row['best_epoch']} | {row['average_precision']:.4f} | "
            f"{row['auroc']:.4f} | {row['recall_at_fpr5']:.4f} | "
            f"{row['false_positive_rate_at_fpr5']:.4f} | {row['positive_pixel_dice']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Mean-gate result",
            "",
            "| Metric | v4.2 mean +/- sample SD | v3 five-seed mean | Delta | Gate |",
            "|---|---:|---:|---:|:---:|",
            (
                f"| AP | {aggregate['average_precision']['mean']:.4f} +/- "
                f"{aggregate['average_precision']['standard_deviation']:.4f} | "
                f"{reference['average_precision']:.4f} | {delta['average_precision']:+.4f} | "
                f"{'pass' if checks['ap_not_below_v3_mean'] else 'fail'} |"
            ),
            (
                f"| AUROC | {aggregate['auroc']['mean']:.4f} +/- "
                f"{aggregate['auroc']['standard_deviation']:.4f} | "
                f"{reference['auroc']:.4f} | {delta['auroc']:+.4f} | "
                f"{'pass' if checks['auroc_not_below_v3_mean'] else 'fail'} |"
            ),
            (
                f"| Recall @ <=5% FPR | {aggregate['recall_at_fpr5']['mean']:.4f} +/- "
                f"{aggregate['recall_at_fpr5']['standard_deviation']:.4f} | "
                f"{reference['recall_at_fpr5']:.4f} | {delta['recall_at_fpr5']:+.4f} | "
                f"{'pass' if checks['recall_at_fpr5_not_below_v3_mean'] else 'fail'} |"
            ),
            (
                f"| Positive-pixel Dice | {aggregate['positive_pixel_dice']['mean']:.4f} +/- "
                f"{aggregate['positive_pixel_dice']['standard_deviation']:.4f} | "
                f"{reference['positive_pixel_dice']:.4f} | {delta['positive_pixel_dice']:+.4f} | "
                f"{'pass' if checks['pixel_dice_not_below_v3_mean'] else 'fail'} |"
            ),
            "",
            "## Decision",
            "",
            campaign["decision"],
            "",
            "Internal validation and the MARS strict/paper benchmarks are different cohorts. These values must not be used to claim MARS-S2L superiority.",
            "",
        ]
    )
    return "\n".join(lines)


def safe_output(root: Path, relative: Path) -> Path:
    path = (root / relative).resolve()
    if root.resolve() not in path.parents:
        raise ValueError("Output path must stay beneath the repository root")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output-markdown", type=Path, default=DEFAULT_MARKDOWN)
    args = parser.parse_args()
    root = repo_root()
    campaign = build_campaign(root, DEFAULT_REPORTS)
    output_json = safe_output(root, args.output_json)
    output_markdown = safe_output(root, args.output_markdown)
    output_json.write_text(json.dumps(campaign, indent=2) + "\n", encoding="utf-8")
    output_markdown.write_text(render_markdown(campaign), encoding="utf-8")
    print(json.dumps(campaign, indent=2))


if __name__ == "__main__":
    main()
