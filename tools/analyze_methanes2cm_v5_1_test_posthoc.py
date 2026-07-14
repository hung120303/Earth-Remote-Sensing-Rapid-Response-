#!/usr/bin/env python3
"""Describe frozen v5.1 calibration transfer without selecting a test rule."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "tools", ROOT / "EarthRemoteSensingRapidResponse"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from acquire_mars_metadata import repo_root, sha256  # noqa: E402
from aggregate_methanes2cm_v5_1 import binary_metrics  # noqa: E402
from train_mars_v3 import safe_output, tracked_dirty, write_json  # noqa: E402

PRIMARY = Path("reports/experiments/methanes2cm_v5_1_location_test.json")
ENSEMBLE = Path("reports/experiments/methanes2cm_v5_1_ensemble_validation.json")
DEFAULT_JSON = Path(
    "reports/experiments/methanes2cm_v5_1_location_test_posthoc.json"
)
DEFAULT_MARKDOWN = Path(
    "reports/experiments/METHANES2CM_V5_1_LOCATION_TEST_POSTHOC.md"
)
EXPECTED_PRIMARY_SHA256 = (
    "415b71ec53353d791381ed8d9f4e60f80f3120771d721d071cea1b347abcd543"
)
EXPECTED_ENSEMBLE_SHA256 = (
    "03691437f3ce2c384aece9f00c7dc4462eebe5e0b580ca340a7db043dc0cdeca"
)


def frozen_operating_points(
    labels: np.ndarray,
    scores: np.ndarray,
    targets: np.ndarray,
    thresholds: np.ndarray,
    development: dict[str, Any],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for target, threshold in zip(targets, thresholds):
        key = str(float(target))
        test = binary_metrics(labels, scores >= threshold)
        source = development[key]
        result[key] = {
            "threshold_frozen_on_development": float(threshold),
            "development": {
                "recall": float(source["recall"]),
                "false_positive_rate": float(source["false_positive_rate"]),
                "precision": float(source["precision"]),
            },
            "location_test": test,
            "transfer_delta_test_minus_development": {
                "recall": float(test["recall"] - source["recall"]),
                "false_positive_rate": float(
                    test["false_positive_rate"] - source["false_positive_rate"]
                ),
                "precision": float(test["precision"] - source["precision"]),
            },
        }
    return result


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# MethaneS2CM v5.1 frozen-test post-hoc diagnostic",
        "",
        "Exploratory calibration-transfer audit only. It does not select a test threshold, change a model, or authorize retuning.",
        "",
        "| Development FPR target | Frozen threshold | Dev recall | Test recall | Dev FPR | Test FPR |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for target, value in report["frozen_operating_points"].items():
        dev = value["development"]
        test = value["location_test"]
        lines.append(
            f"| {float(target):.1%} | {value['threshold_frozen_on_development']:.6f} | "
            f"{dev['recall']:.4f} | {test['recall']:.4f} | "
            f"{dev['false_positive_rate']:.4f} | {test['false_positive_rate']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            report["interpretation"],
            "",
            "The next study must fit any calibration or risk-control layer on new development/calibration groups and use a newly untouched confirmation cohort. These test labels may not choose that layer.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", default=DEFAULT_JSON.as_posix())
    parser.add_argument("--output-markdown", default=DEFAULT_MARKDOWN.as_posix())
    args = parser.parse_args()

    root = repo_root()
    if tracked_dirty(root):
        raise RuntimeError("Refusing post-hoc analysis from a dirty tracked worktree")
    if sha256(root / PRIMARY) != EXPECTED_PRIMARY_SHA256:
        raise ValueError("Primary one-shot result identity mismatch")
    if sha256(root / ENSEMBLE) != EXPECTED_ENSEMBLE_SHA256:
        raise ValueError("Frozen ensemble result identity mismatch")
    primary = json.loads((root / PRIMARY).read_text(encoding="utf-8"))
    ensemble = json.loads((root / ENSEMBLE).read_text(encoding="utf-8"))

    prediction_path = root / primary["prediction_cache"]["path"]
    calibration_path = root / ensemble["calibration_cache"]["path"]
    if sha256(prediction_path) != primary["prediction_cache"]["sha256"]:
        raise ValueError("Primary prediction-cache identity mismatch")
    if sha256(calibration_path) != ensemble["calibration_cache"]["sha256"]:
        raise ValueError("Frozen calibration-cache identity mismatch")
    with np.load(prediction_path, allow_pickle=False) as source:
        labels = source["label"].astype(np.uint8)
        scores = source["ersrr_v5_1_scene_score"].astype(np.float64)
    with np.load(calibration_path, allow_pickle=False) as source:
        targets = source["target_fprs"].astype(np.float64)
        thresholds = source["scene_thresholds"].astype(np.float64)
    operating = frozen_operating_points(
        labels,
        scores,
        targets,
        thresholds,
        ensemble["final_all_development_rule"]["operating_points"],
    )
    quantiles = (0.5, 0.9, 0.95, 0.975, 0.99)
    report = {
        "schema_version": 1,
        "scope": "methanes2cm_v5_1_frozen_location_test_posthoc_calibration_transfer",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": {
            "posthoc": True,
            "model_changed": False,
            "threshold_selected_from_test": False,
            "deployment_rule_changed": False,
            "retuning_authorized": False,
        },
        "sources": {
            "primary": {"path": PRIMARY.as_posix(), "sha256": EXPECTED_PRIMARY_SHA256},
            "ensemble": {
                "path": ENSEMBLE.as_posix(),
                "sha256": EXPECTED_ENSEMBLE_SHA256,
            },
            "prediction_cache_sha256": primary["prediction_cache"]["sha256"],
            "calibration_cache_sha256": ensemble["calibration_cache"]["sha256"],
        },
        "frozen_operating_points": operating,
        "score_quantiles": {
            "probabilities": list(quantiles),
            "plume": [
                float(value) for value in np.quantile(scores[labels == 1], quantiles)
            ],
            "no_plume": [
                float(value) for value in np.quantile(scores[labels == 0], quantiles)
            ],
        },
        "interpretation": (
            "Every predeclared development threshold produced higher FPR on the location test, "
            "while recall also fell relative to development. The primary 5% target moved from "
            "4.99% development FPR / 46.71% recall to 6.07% test FPR / 37.78% recall. This "
            "supports calibration/domain-transfer risk as a future hypothesis, not a test-driven "
            "threshold adjustment."
        ),
        "provenance": {
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True
            ).strip(),
            "script": Path(__file__).resolve().relative_to(root).as_posix(),
            "script_sha256": sha256(Path(__file__).resolve()),
            "tracked_worktree_dirty_at_start": False,
        },
    }
    output_json = safe_output(root, args.output_json)
    output_markdown = safe_output(root, args.output_markdown)
    write_json(output_json, report)
    write_markdown(output_markdown, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
