#!/usr/bin/env python3
"""Replay the frozen all-development scene refit on the exact MARS paper cache."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "EarthRemoteSensingRapidResponse"
for path in (MODEL_ROOT, ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from acquire_mars_metadata import repo_root, sha256  # noqa: E402
from diagnose_mars_scene_stacker_paper_cache import triplet  # noqa: E402
from evaluate_mars_successor_paper_test import bootstrap_view, view_metrics  # noqa: E402
from train_mars_context_scene_ranker import augment_site_context  # noqa: E402
from train_mars_scene_ranker import blend_scores  # noqa: E402

DEFAULT_DIAGNOSTIC = Path("outputs/mars_paper_test_v3_diagnostic_cache.npz")
DEFAULT_DIAGNOSTIC_SHA256 = "1624fddc0222f8ffc5137f557c7fc3e465d53b335c82cc8014711baa35bb94a1"
DEFAULT_ARTIFACT = Path(
    "EarthRemoteSensingRapidResponse/artifacts/mars_oof_scene_ensemble_v2_all_development.joblib"
)
DEFAULT_ARTIFACT_SHA256 = "7dd81a2f1d9b30b88500eeceb086664c4a3fb1cad21810a10783b2ce72c4ab1a"
DEFAULT_JSON = Path("reports/experiments/mars_all_development_scene_refit_paper_posttest.json")
DEFAULT_MARKDOWN = Path("reports/experiments/MARS_ALL_DEVELOPMENT_SCENE_REFIT_PAPER_POSTTEST.md")


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# All-development scene refit: exact MARS-S2L paper replay",
        "",
        "Transparent post-test diagnostic; this is not an untouched confirmation cohort.",
        "",
        "| View | Candidate AP | AP delta | AP 95% CI | Recall delta | Recall 95% CI | FPR delta | Gates |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for name, value in report["views"].items():
        metrics = value["metrics"]
        intervals = value["bootstrap"]["delta_intervals"]
        lines.append(
            f"| {name} | {metrics['candidate']['average_precision']:.5f} | "
            f"{metrics['delta']['average_precision']:+.5f} | "
            f"[{intervals['average_precision']['lower']:+.5f}, {intervals['average_precision']['upper']:+.5f}] | "
            f"{metrics['delta']['matched_fpr_recall']:+.5f} | "
            f"[{intervals['matched_fpr_recall']['lower']:+.5f}, {intervals['matched_fpr_recall']['upper']:+.5f}] | "
            f"{metrics['delta']['fixed_false_positive_rate']:+.5f} | "
            f"{'PASS' if value['scene_gates_pass'] else 'FAIL'} |"
        )
    lines.extend(["", report["decision"]])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic", default=DEFAULT_DIAGNOSTIC.as_posix())
    parser.add_argument("--diagnostic-sha256", default=DEFAULT_DIAGNOSTIC_SHA256)
    parser.add_argument("--artifact", default=DEFAULT_ARTIFACT.as_posix())
    parser.add_argument("--artifact-sha256", default=DEFAULT_ARTIFACT_SHA256)
    parser.add_argument("--replicates", type=int, default=10_000)
    parser.add_argument("--output-json", default=DEFAULT_JSON.as_posix())
    parser.add_argument("--output-markdown", default=DEFAULT_MARKDOWN.as_posix())
    args = parser.parse_args()
    root = repo_root()
    diagnostic_path = (root / args.diagnostic).resolve()
    artifact_path = (root / args.artifact).resolve()
    if sha256(diagnostic_path) != args.diagnostic_sha256:
        raise ValueError("Frozen exact-paper diagnostic cache hash mismatch")
    if sha256(artifact_path) != args.artifact_sha256:
        raise ValueError("Frozen all-development scene artifact hash mismatch")
    artifact = joblib.load(artifact_path)
    with np.load(diagnostic_path, allow_pickle=False) as cache:
        values = {name: cache[name] for name in cache.files}
    names = values["base_feature_names"].astype(str)
    if names.tolist() != artifact["feature_names"]:
        raise ValueError("Paper base-feature schema differs from the refit artifact")
    features, augmented_names = augment_site_context(
        values["available_base_features"].astype(np.float32),
        names,
        values["available_groups"].astype(str),
    )
    if augmented_names != artifact["augmented_feature_names"]:
        raise ValueError("Paper augmented-feature schema differs from the refit artifact")
    primary_index = int(np.flatnonzero(names == artifact["primary_feature"])[0])
    head = artifact["fitted"].predict_proba(features)[:, 1]
    available_scores = blend_scores(
        values["available_base_features"][:, primary_index].astype(np.float64),
        head,
        float(artifact["blend_lambda"]),
    )
    candidate_scores = values["candidate_scores"].astype(np.float64).copy()
    available_lookup = {
        sample_id: index
        for index, sample_id in enumerate(values["available_ids"].astype(str))
    }
    paper_indices = np.asarray(
        [available_lookup.get(sample_id, -1) for sample_id in values["aligned_sample_ids"].astype(str)]
    )
    available = paper_indices >= 0
    candidate_scores[available] = available_scores[paper_indices[available]]
    labels = values["labels"].astype(np.uint8)
    sites = values["sites"].astype(str)
    baseline_scores = values["baseline_scores"].astype(np.float64)
    threshold = float(artifact["operational_scene_threshold"])
    selections = {
        "full": np.ones(labels.shape, dtype=bool),
        "test_only_sites": values["test_only"].astype(bool),
    }
    views: dict[str, Any] = {}
    for index, (name, rows) in enumerate(selections.items()):
        metrics = view_metrics(
            labels[rows],
            baseline_scores[rows],
            candidate_scores[rows],
            triplet(values["baseline_pixels"][rows]),
            triplet(values["candidate_pixels"][rows]),
            threshold,
        )
        bootstrap = bootstrap_view(
            labels=labels[rows],
            sites=sites[rows],
            baseline_scores=baseline_scores[rows],
            candidate_scores=candidate_scores[rows],
            baseline_predictions=baseline_scores[rows] > 0.5,
            candidate_predictions=candidate_scores[rows] > threshold,
            baseline_pixels=triplet(values["baseline_pixels"][rows]),
            candidate_pixels=triplet(values["candidate_pixels"][rows]),
            replicates=args.replicates,
            seed=20261070 + index,
            confidence=0.95,
        )
        intervals = bootstrap["delta_intervals"]
        checks = {
            "ap_point_higher": metrics["delta"]["average_precision"] > 0.0,
            "ap_lower_positive": intervals["average_precision"]["lower"] > 0.0,
            "matched_recall_point_higher": metrics["delta"]["matched_fpr_recall"] > 0.0,
            "matched_recall_lower_positive": intervals["matched_fpr_recall"]["lower"] > 0.0,
            "fixed_fpr_upper_nonpositive": intervals["fixed_false_positive_rate"]["upper"] <= 0.0,
        }
        views[name] = {
            "metrics": metrics,
            "bootstrap": bootstrap,
            "scene_checks": checks,
            "scene_gates_pass": all(checks.values()),
        }
    passed = all(value["scene_gates_pass"] for value in views.values())
    report = {
        "schema_version": 1,
        "scope": "transparent post-test replay of fixed all-development refit on exact paper rows",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "available_refit_rows": int(available.sum()),
        "missing_rows_fallback_to_crossfit_v3": int((~available).sum()),
        "operational_scene_threshold": threshold,
        "views": views,
        "all_exact_paper_scene_gates_pass": passed,
        "decision": (
            "Scene gates pass, but independent external confirmation remains required."
            if passed
            else "Reject the all-development refit; at least one exact paper scene gate fails."
        ),
        "provenance": {
            "diagnostic_sha256": args.diagnostic_sha256,
            "artifact_sha256": args.artifact_sha256,
            "script_sha256": sha256(Path(__file__).resolve()),
        },
    }
    write_json((root / args.output_json).resolve(), report)
    write_markdown((root / args.output_markdown).resolve(), report)
    print(
        json.dumps(
            {
                "ok": passed,
                "views": {
                    name: {
                        "candidate_ap": value["metrics"]["candidate"]["average_precision"],
                        "ap_lower": value["bootstrap"]["delta_intervals"]["average_precision"]["lower"],
                        "recall_lower": value["bootstrap"]["delta_intervals"]["matched_fpr_recall"]["lower"],
                        "scene_gates_pass": value["scene_gates_pass"],
                    }
                    for name, value in views.items()
                },
            },
            indent=2,
        )
    )
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
