#!/usr/bin/env python3
"""Post-test diagnosis of a development-trained scene stacker on the exact paper cache."""

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
from evaluate_mars_successor_paper_test import bootstrap_view, view_metrics  # noqa: E402
from train_mars_context_scene_ranker import augment_site_context  # noqa: E402
from train_mars_scene_ranker import blend_scores, predict_model  # noqa: E402
from train_mars_scene_stacker_v3 import (  # noqa: E402
    predict_model as predict_stacker,
    stack_features,
)

DEFAULT_CACHE = Path("outputs/mars_paper_test_v3_diagnostic_cache.npz")
DEFAULT_CACHE_SHA256 = "1624fddc0222f8ffc5137f557c7fc3e465d53b335c82cc8014711baa35bb94a1"
DEFAULT_LEGACY = Path("EarthRemoteSensingRapidResponse/artifacts/mars_oof_context_ranker_folds234.joblib")
DEFAULT_LEGACY_SHA256 = "2d014f54918f68726d2ca4da19f35a1f29cb1b622fe7c32b56afc554ec27c370"
DEFAULT_STACKER = Path("EarthRemoteSensingRapidResponse/artifacts/mars_scene_stacker_v3.joblib")
DEFAULT_STACKER_SHA256 = "d2e2da7566ad4080110b1ba5bbd0dce8dfdd9dd41ccaf9bbaf263d17d75ab3be"
DEFAULT_JSON = Path("reports/experiments/mars_scene_stacker_v3_paper_posthoc.json")
DEFAULT_MARKDOWN = Path("reports/experiments/MARS_SCENE_STACKER_V3_PAPER_POSTHOC.md")


def aligned_indices(aligned_ids: np.ndarray, available_ids: np.ndarray) -> np.ndarray:
    lookup = {sample_id: index for index, sample_id in enumerate(aligned_ids.astype(str))}
    if len(lookup) != aligned_ids.size:
        raise ValueError("Aligned paper IDs are not unique")
    try:
        indices = np.asarray([lookup[sample_id] for sample_id in available_ids.astype(str)])
    except KeyError as error:
        raise ValueError("Available paper ID is absent from aligned rows") from error
    if len(set(indices.tolist())) != indices.size:
        raise ValueError("Available paper IDs are not unique")
    return indices


def triplet(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("Pixel cache must contain intersection/FP/FN triplets")
    return values[:, 0], values[:, 1], values[:, 2]


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Scene stacker v3: post-test paper-cache diagnosis",
        "",
        "Exploratory only. The stacker failed its inner non-regression gate before this paper cache was loaded.",
        "",
        "| View | AP delta | AP 95% CI | Recall delta at matched FPR | Recall 95% CI |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, value in report["views"].items():
        delta = value["metrics"]["delta"]
        intervals = value["bootstrap"]["delta_intervals"]
        lines.append(
            f"| {name} | {delta['average_precision']:+.5f} | "
            f"[{intervals['average_precision']['lower']:+.5f}, {intervals['average_precision']['upper']:+.5f}] | "
            f"{delta['matched_fpr_recall']:+.5f} | "
            f"[{intervals['matched_fpr_recall']['lower']:+.5f}, {intervals['matched_fpr_recall']['upper']:+.5f}] |"
        )
    lines.extend(["", report["decision"]])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", default=DEFAULT_CACHE.as_posix())
    parser.add_argument("--cache-sha256", default=DEFAULT_CACHE_SHA256)
    parser.add_argument("--legacy", default=DEFAULT_LEGACY.as_posix())
    parser.add_argument("--legacy-sha256", default=DEFAULT_LEGACY_SHA256)
    parser.add_argument("--stacker", default=DEFAULT_STACKER.as_posix())
    parser.add_argument("--stacker-sha256", default=DEFAULT_STACKER_SHA256)
    parser.add_argument("--replicates", type=int, default=10000)
    parser.add_argument("--output-json", default=DEFAULT_JSON.as_posix())
    parser.add_argument("--output-markdown", default=DEFAULT_MARKDOWN.as_posix())
    args = parser.parse_args()
    root = repo_root()
    paths = {
        "cache": (root / args.cache).resolve(),
        "legacy": (root / args.legacy).resolve(),
        "stacker": (root / args.stacker).resolve(),
    }
    expected = {
        "cache": args.cache_sha256,
        "legacy": args.legacy_sha256,
        "stacker": args.stacker_sha256,
    }
    for name, path in paths.items():
        if sha256(path) != expected[name]:
            raise ValueError(f"Frozen {name} hash mismatch")
    legacy_payload = joblib.load(paths["legacy"])
    stacker_payload = joblib.load(paths["stacker"])
    with np.load(paths["cache"], allow_pickle=False) as cache:
        values = {name: cache[name] for name in cache.files}

    indices = aligned_indices(values["aligned_sample_ids"], values["available_ids"])
    base = values["available_base_features"].astype(np.float64)
    names = values["base_feature_names"].astype(str)
    groups = values["available_groups"].astype(str)
    augmented, augmented_names = augment_site_context(base, names, groups)
    if names.tolist() != legacy_payload["feature_names"]:
        raise ValueError("Paper base-feature schema differs from legacy head")
    if augmented_names != legacy_payload["augmented_feature_names"]:
        raise ValueError("Paper augmented-feature schema differs from legacy head")
    primary_column = int(np.flatnonzero(names == "primary_connected_score")[0])
    available_primary = base[:, primary_column]
    legacy_available = blend_scores(
        available_primary,
        predict_model(legacy_payload["fitted"], augmented),
        0.25,
    )
    primary = values["baseline_scores"].astype(np.float64).copy()
    primary[indices] = available_primary
    legacy = values["candidate_scores"].astype(np.float64).copy()
    legacy[indices] = legacy_available
    new = values["candidate_scores"].astype(np.float64)
    sensors = values["sensors"].astype(np.uint8)
    offshore = values["offshore"].astype(bool)
    features, feature_names = stack_features(
        primary,
        legacy,
        new,
        sensors,
        offshore,
        str(stacker_payload["spec"]["feature_set"]),
    )
    if feature_names != stacker_payload["feature_names"]:
        raise ValueError("Paper stacker feature schema mismatch")
    candidate = predict_stacker(stacker_payload["fitted"], features)
    missing = np.ones(primary.shape, dtype=bool)
    missing[indices] = False
    candidate[missing] = new[missing]

    labels = values["labels"].astype(np.uint8)
    sites = values["sites"].astype(str)
    baseline_scores = values["baseline_scores"].astype(np.float64)
    baseline_pixels = values["baseline_pixels"].astype(np.int64)
    candidate_pixels = values["candidate_pixels"].astype(np.int64)
    operational_threshold = float(stacker_payload["operational_scene_threshold"])
    selections = {
        "full": np.ones(labels.shape, dtype=bool),
        "test_only_sites": values["test_only"].astype(bool),
    }
    views: dict[str, Any] = {}
    for index, (name, rows) in enumerate(selections.items()):
        metrics = view_metrics(
            labels[rows],
            baseline_scores[rows],
            candidate[rows],
            triplet(baseline_pixels[rows]),
            triplet(candidate_pixels[rows]),
            operational_threshold,
        )
        baseline_predictions = baseline_scores[rows] > 0.5
        candidate_predictions = candidate[rows] > operational_threshold
        bootstrap = bootstrap_view(
            labels=labels[rows],
            sites=sites[rows],
            baseline_scores=baseline_scores[rows],
            candidate_scores=candidate[rows],
            baseline_predictions=baseline_predictions,
            candidate_predictions=candidate_predictions,
            baseline_pixels=triplet(baseline_pixels[rows]),
            candidate_pixels=triplet(candidate_pixels[rows]),
            replicates=args.replicates,
            seed=20260740 + index,
            confidence=0.95,
        )
        intervals = bootstrap["delta_intervals"]
        scene_passed = (
            intervals["average_precision"]["lower"] > 0.0
            and intervals["matched_fpr_recall"]["lower"] > 0.0
            and intervals["fixed_false_positive_rate"]["upper"] <= 0.0
        )
        views[name] = {"metrics": metrics, "bootstrap": bootstrap, "scene_gates_pass": scene_passed}
    passed = all(value["scene_gates_pass"] for value in views.values())
    report = {
        "schema_version": 1,
        "scope": "post-test diagnosis of a development-rejected stacker; not confirmation evidence",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "missing_rows_fallback_to_v3": int(missing.sum()),
        "operational_scene_threshold": operational_threshold,
        "views": views,
        "all_exact_paper_scene_gates_pass": passed,
        "decision": (
            "Diagnostic scene gates pass, but the stacker remains rejected by development selection."
            if passed
            else "Reject the stacker; it does not solve the exact paper scene gates."
        ),
        "provenance": {f"{name}_sha256": expected[name] for name in expected},
    }
    write_json((root / args.output_json).resolve(), report)
    write_markdown((root / args.output_markdown).resolve(), report)
    print(
        json.dumps(
            {
                "ok": passed,
                "views": {
                    name: {
                        "ap_delta": value["metrics"]["delta"]["average_precision"],
                        "ap_lower": value["bootstrap"]["delta_intervals"]["average_precision"][
                            "lower"
                        ],
                        "recall_lower": value["bootstrap"]["delta_intervals"][
                            "matched_fpr_recall"
                        ]["lower"],
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
