#!/usr/bin/env python3
"""Measure the exact contribution of the paper's offshore fine-tuned model.

This is a post-test diagnostic. It compares the authors' archived general and
offshore prediction files on the same paper-v3 rows; it does not evaluate an
ERSRR candidate or alter the frozen one-shot result.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from audit_mars_paper_benchmark import (
    DEFAULT_ASSIGNMENTS,
    DEFAULT_OFFSHORE,
    DEFAULT_ONSHORE,
    EXPECTED_HASHES,
    index_unique,
    read_csv,
    scene_metrics,
)
from acquire_mars_metadata import repo_root, sha256

DEFAULT_JSON = Path("reports/experiments/mars_offshore_finetune_diagnostic.json")
DEFAULT_MARKDOWN = Path("reports/experiments/MARS_OFFSHORE_FINETUNE_DIAGNOSTIC.md")


def read_assignments(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as source:
        for line in source:
            if line.strip():
                rows.append(json.loads(line))
    if len(rows) != 43_529:
        raise ValueError(f"Expected 43,529 frozen assignments, found {len(rows)}")
    return rows


def sensor_family(tile: str) -> str:
    return "Landsat" if tile.startswith(("LC", "LE", "LT")) else "Sentinel-2"


def prediction_row(
    assignment: dict[str, Any], prediction: dict[str, str]
) -> dict[str, Any]:
    return {
        "id_loc_image": assignment["sample_id"],
        "location_name": assignment["location_name"],
        "tile": assignment["tile"],
        "target": int(assignment["target"]),
        "scene_pred": float(prediction["scene_pred"]),
        "TP": float(prediction["TP"]),
        "FP": float(prediction["FP"]),
        "TN": float(prediction["TN"]),
        "FN": float(prediction["FN"]),
        "test_only_site": bool(assignment["test_only_site"]),
        "is_offshore": bool(assignment["is_offshore"]),
        "sensor_family": sensor_family(str(assignment["tile"])),
    }


def numeric_delta(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "average_precision",
        "precision",
        "recall",
        "false_positive_rate",
        "pixel_iou",
        "tp",
        "fp",
        "tn",
        "fn",
        "pixel_tp",
        "pixel_fp",
        "pixel_fn",
    )
    return {key: candidate[key] - baseline[key] for key in keys}


def comparison(
    general: list[dict[str, Any]], finetuned: list[dict[str, Any]]
) -> dict[str, Any]:
    if [row["id_loc_image"] for row in general] != [
        row["id_loc_image"] for row in finetuned
    ]:
        raise ValueError("General and fine-tuned rows are not aligned")
    base = scene_metrics(general)
    candidate = scene_metrics(finetuned)
    return {
        "general_model": base,
        "offshore_finetune": candidate,
        "delta": numeric_delta(candidate, base),
    }


def select(
    rows: Iterable[dict[str, Any]],
    *,
    offshore: bool | None = None,
    test_only: bool | None = None,
    sensor: str | None = None,
) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if (offshore is None or bool(row["is_offshore"]) == offshore)
        and (test_only is None or bool(row["test_only_site"]) == test_only)
        and (sensor is None or row["sensor_family"] == sensor)
    ]


def safe_iou(rows: Iterable[dict[str, Any]]) -> float:
    rows = list(rows)
    tp = sum(float(row["TP"]) for row in rows)
    fp = sum(float(row["FP"]) for row in rows)
    fn = sum(float(row["FN"]) for row in rows)
    denominator = tp + fp + fn
    return tp / denominator if denominator else 1.0


def site_effects(
    general: list[dict[str, Any]], finetuned: list[dict[str, Any]]
) -> dict[str, Any]:
    general_by_site: dict[str, list[dict[str, Any]]] = defaultdict(list)
    fine_by_site: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for base, fine in zip(general, finetuned, strict=True):
        site = str(base["location_name"])
        general_by_site[site].append(base)
        fine_by_site[site].append(fine)
    effects = []
    for site in sorted(general_by_site):
        base_rows = general_by_site[site]
        fine_rows = fine_by_site[site]
        base_iou = safe_iou(base_rows)
        fine_iou = safe_iou(fine_rows)
        effects.append(
            {
                "location_name": site,
                "rows": len(base_rows),
                "positive": sum(int(row["target"]) for row in base_rows),
                "general_pixel_iou": base_iou,
                "finetuned_pixel_iou": fine_iou,
                "pixel_iou_delta": fine_iou - base_iou,
            }
        )
    tolerance = 1e-12
    return {
        "sites": len(effects),
        "improved": sum(item["pixel_iou_delta"] > tolerance for item in effects),
        "regressed": sum(item["pixel_iou_delta"] < -tolerance for item in effects),
        "tied": sum(abs(item["pixel_iou_delta"]) <= tolerance for item in effects),
        "largest_improvements": sorted(
            effects, key=lambda item: item["pixel_iou_delta"], reverse=True
        )[:10],
        "largest_regressions": sorted(
            effects, key=lambda item: item["pixel_iou_delta"]
        )[:10],
    }


def analyze(
    assignments: list[dict[str, Any]],
    general_predictions: list[dict[str, str]],
    offshore_predictions: list[dict[str, str]],
) -> dict[str, Any]:
    general_index = index_unique(general_predictions, "id_loc_image")
    offshore_index = index_unique(offshore_predictions, "id_loc_image")
    ids = [str(row["sample_id"]) for row in assignments]
    if set(ids) != set(general_index) or set(ids) != set(offshore_index):
        raise ValueError("Frozen assignments and prediction archives have different IDs")

    general = [prediction_row(row, general_index[str(row["sample_id"])]) for row in assignments]
    fine = [prediction_row(row, offshore_index[str(row["sample_id"])]) for row in assignments]

    hybrid = [
        fine_row if bool(base_row["is_offshore"]) else base_row
        for base_row, fine_row in zip(general, fine, strict=True)
    ]
    global_views: dict[str, Any] = {}
    for name, test_only in (("full", None), ("test_only_sites", True)):
        general_view = select(general, test_only=test_only)
        hybrid_view = select(hybrid, test_only=test_only)
        global_views[name] = comparison(general_view, hybrid_view)
        global_views[name]["offshore_rows"] = len(
            select(general_view, offshore=True)
        )

    strata: dict[str, Any] = {}
    specifications = {
        "all_offshore": (None, None),
        "seen_site_offshore": (False, None),
        "test_only_site_offshore": (True, None),
        "sentinel2_offshore": (None, "Sentinel-2"),
        "landsat_offshore": (None, "Landsat"),
        "seen_site_sentinel2_offshore": (False, "Sentinel-2"),
        "seen_site_landsat_offshore": (False, "Landsat"),
        "test_only_site_sentinel2_offshore": (True, "Sentinel-2"),
        "test_only_site_landsat_offshore": (True, "Landsat"),
    }
    for name, (test_only, sensor) in specifications.items():
        base_rows = select(
            general, offshore=True, test_only=test_only, sensor=sensor
        )
        fine_rows = select(fine, offshore=True, test_only=test_only, sensor=sensor)
        if base_rows and any(row["target"] for row in base_rows) and any(
            not row["target"] for row in base_rows
        ):
            strata[name] = comparison(base_rows, fine_rows)

    offshore_general = select(general, offshore=True)
    offshore_fine = select(fine, offshore=True)
    return {
        "global_hybrid_effect": global_views,
        "offshore_strata": strata,
        "offshore_site_pixel_effects": site_effects(offshore_general, offshore_fine),
    }


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# MARS-S2L offshore fine-tune diagnostic",
        "",
        "This is a post-test diagnostic of the authors' two archived prediction files. It does not change or rerun the frozen ERSRR one-shot result.",
        "",
        "## Effect of substituting the fine-tuned model only offshore",
        "",
        "| View | Offshore rows | General AP | Hybrid AP | ΔAP | General IoU | Hybrid IoU | ΔIoU |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, values in report["analysis"]["global_hybrid_effect"].items():
        base = values["general_model"]
        fine = values["offshore_finetune"]
        delta = values["delta"]
        lines.append(
            f"| {name} | {values['offshore_rows']:,} | {base['average_precision']:.5f} | {fine['average_precision']:.5f} | {delta['average_precision']:+.5f} | {base['pixel_iou']:.5f} | {fine['pixel_iou']:.5f} | {delta['pixel_iou']:+.5f} |"
        )
    lines += [
        "",
        "## Offshore-only strata",
        "",
        "| Stratum | Rows | Plume | General AP | Fine-tuned AP | ΔAP | General IoU | Fine-tuned IoU | ΔIoU |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, values in report["analysis"]["offshore_strata"].items():
        base = values["general_model"]
        fine = values["offshore_finetune"]
        delta = values["delta"]
        lines.append(
            f"| {name} | {base['rows']:,} | {base['positive']:,} | {base['average_precision']:.5f} | {fine['average_precision']:.5f} | {delta['average_precision']:+.5f} | {base['pixel_iou']:.5f} | {fine['pixel_iou']:.5f} | {delta['pixel_iou']:+.5f} |"
        )
    sites = report["analysis"]["offshore_site_pixel_effects"]
    lines += [
        "",
        f"Across {sites['sites']} offshore sites, fine-tuning improved site-level pixel IoU at {sites['improved']}, regressed at {sites['regressed']}, and tied at {sites['tied']}.",
        "",
        "The paper specifies one extra full-model epoch on offshore real data. The public release includes the general checkpoint and both prediction archives, but not the offshore checkpoint, so exact weight reproduction requires retraining.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assignments", default=DEFAULT_ASSIGNMENTS.as_posix())
    parser.add_argument("--general-predictions", default=DEFAULT_ONSHORE.as_posix())
    parser.add_argument("--offshore-predictions", default=DEFAULT_OFFSHORE.as_posix())
    parser.add_argument("--output-json", default=DEFAULT_JSON.as_posix())
    parser.add_argument("--output-markdown", default=DEFAULT_MARKDOWN.as_posix())
    args = parser.parse_args()
    root = repo_root()
    paths = {
        name: (root / value).resolve()
        for name, value in {
            "assignments": args.assignments,
            "general": args.general_predictions,
            "offshore": args.offshore_predictions,
            "json": args.output_json,
            "markdown": args.output_markdown,
        }.items()
    }
    if sha256(paths["general"]) != EXPECTED_HASHES["onshore_predictions"]:
        raise ValueError("General prediction archive hash mismatch")
    if sha256(paths["offshore"]) != EXPECTED_HASHES["offshore_predictions"]:
        raise ValueError("Offshore prediction archive hash mismatch")
    report = {
        "schema_version": 1,
        "status": "post_test_diagnostic",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "provenance": {
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True
            ).strip(),
            "script": Path(__file__).relative_to(root).as_posix(),
            "script_sha256": sha256(Path(__file__)),
            "assignment_sha256": sha256(paths["assignments"]),
            "general_predictions_sha256": sha256(paths["general"]),
            "offshore_predictions_sha256": sha256(paths["offshore"]),
        },
        "analysis": analyze(
            read_assignments(paths["assignments"]),
            read_csv(paths["general"]),
            read_csv(paths["offshore"]),
        ),
    }
    write_json(paths["json"], report)
    write_markdown(paths["markdown"], report)
    print(json.dumps(report["analysis"]["global_hybrid_effect"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
