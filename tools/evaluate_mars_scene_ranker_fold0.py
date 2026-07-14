#!/usr/bin/env python3
"""One-shot fold-0 evaluation of the frozen MARS scene-ranking head."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import scipy
import sklearn

from acquire_mars_metadata import repo_root, sha256
from train_mars_paper_residual import SENSOR_NAMES
from train_mars_scene_ranker import blend_scores, metric_summary, predict_model

DEFAULT_CACHE = Path("outputs/mars_scene_features_fold0.npz")
DEFAULT_ARTIFACT = Path(
    "EarthRemoteSensingRapidResponse/artifacts/mars_scene_ranker_folds234.joblib"
)
DEFAULT_ARTIFACT_SHA256 = (
    "98ce79c62b6af0c97155acdf4255ee4a721f5ef3d5412203bf6a04d2512a336c"
)
DEFAULT_INNER_REPORT = Path("reports/experiments/mars_scene_ranker_inner_fold2.json")
DEFAULT_INNER_REPORT_SHA256 = (
    "20c82ea36024dbaa5f1fd56673a139a63945b07c1ade4a706acbd3cda2f5fcec"
)
DEFAULT_PRIMARY_REPORT = Path(
    "reports/experiments/mars_paper_residual_fold0_trust_region.json"
)
DEFAULT_PRIMARY_REPORT_SHA256 = (
    "bb9763a9bdbf0ddd14c3e1c718af9bceff47e0a1c6f04a4749833968368b79b5"
)
DEFAULT_JSON = Path("reports/experiments/mars_scene_ranker_fold0.json")
DEFAULT_MARKDOWN = Path("reports/experiments/MARS_SCENE_RANKER_FOLD0.md")


def primary_identity_values(metrics: dict[str, Any]) -> list[float]:
    values = [
        metrics["average_precision"],
        metrics["operating_point"]["recall"],
        metrics["operating_point"]["false_positive_rate"],
    ]
    values.extend(metrics["sensor_average_precision"][name] for name in SENSOR_NAMES)
    return [float(value) for value in values]


def primary_report_values(summary: dict[str, Any]) -> list[float]:
    values = [
        summary["candidate"]["average_precision"],
        summary["candidate"]["operating_points"]["0.0713"]["recall"],
        summary["candidate"]["operating_points"]["0.0713"]["false_positive_rate"],
    ]
    values.extend(
        summary["sensor_strata"][name]["candidate"]["average_precision"]
        for name in SENSOR_NAMES
    )
    return [float(value) for value in values]


def assert_primary_identity(metrics: dict[str, Any], summary: dict[str, Any]) -> None:
    if primary_identity_values(metrics) != primary_report_values(summary):
        raise RuntimeError("Fold-0 feature cache does not reproduce the frozen primary endpoint")


def evaluate_candidate(
    candidate: dict[str, Any], primary_summary: dict[str, Any]
) -> dict[str, Any]:
    released = primary_summary["released_baseline"]
    ap_delta = float(candidate["average_precision"] - released["average_precision"])
    recall_delta = float(
        candidate["operating_point"]["recall"]
        - released["operating_points"]["0.0713"]["recall"]
    )
    fpr_delta = float(
        candidate["operating_point"]["false_positive_rate"]
        - released["operating_points"]["0.0713"]["false_positive_rate"]
    )
    iou = float(primary_summary["candidate"]["pixel_fixed_0_5"]["intersection_over_union"])
    released_iou = float(released["pixel_fixed_0_5"]["intersection_over_union"])
    sensor_strata: dict[str, Any] = {}
    for name in SENSOR_NAMES:
        primary_sensor = primary_summary["sensor_strata"][name]
        released_sensor = primary_sensor["released_baseline"]
        candidate_ap = float(candidate["sensor_average_precision"][name])
        candidate_iou = float(
            primary_sensor["candidate"]["pixel_fixed_0_5"]["intersection_over_union"]
        )
        sensor_strata[name] = {
            "average_precision": candidate_ap,
            "pixel_iou": candidate_iou,
            "delta": {
                "average_precision": candidate_ap - float(released_sensor["average_precision"]),
                "pixel_iou": candidate_iou
                - float(released_sensor["pixel_fixed_0_5"]["intersection_over_union"]),
            },
        }
    deltas = {
        "average_precision": ap_delta,
        "recall_at_fpr_0_0713": recall_delta,
        "false_positive_rate_at_target": fpr_delta,
        "pixel_iou": iou - released_iou,
    }
    checks = {
        "ap_higher": deltas["average_precision"] > 0,
        "recall_at_fpr_0_0713_higher": deltas["recall_at_fpr_0_0713"] > 0,
        "fpr_no_worse": deltas["false_positive_rate_at_target"] <= 0,
        "pixel_iou_higher": deltas["pixel_iou"] > 0,
        "no_material_sensor_regression": all(
            value["delta"]["average_precision"] >= -0.01
            and value["delta"]["pixel_iou"] >= -0.01
            for value in sensor_strata.values()
        ),
    }
    return {
        "scene": candidate,
        "pixel_iou": iou,
        "released_baseline": {
            "average_precision": released["average_precision"],
            "recall_at_fpr_0_0713": released["operating_points"]["0.0713"]["recall"],
            "false_positive_rate_at_target": released["operating_points"]["0.0713"]["false_positive_rate"],
            "pixel_iou": released_iou,
        },
        "delta": deltas,
        "sensor_strata": sensor_strata,
        "checks": checks,
    }


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    result = report["result"]
    candidate = result["scene"]
    baseline = result["released_baseline"]
    lines = [
        "# Frozen MARS scene ranker on fold 0",
        "",
        "The scene head was frozen from folds 2-4. Fold 1 and the paper test were not loaded.",
        "",
        "| Model | AP | Recall at <=7.13% FPR | FPR | Pixel IoU |",
        "|---|---:|---:|---:|---:|",
        f"| Released MARS-S2L | {baseline['average_precision']:.5f} | {baseline['recall_at_fpr_0_0713']:.5f} | {baseline['false_positive_rate_at_target']:.5f} | {baseline['pixel_iou']:.5f} |",
        f"| Frozen segmentation + scene head | {candidate['average_precision']:.5f} | {candidate['operating_point']['recall']:.5f} | {candidate['operating_point']['false_positive_rate']:.5f} | {result['pixel_iou']:.5f} |",
        "",
        "All gates pass." if all(result["checks"].values()) else "Fold-0 gate failed.",
        "",
        report["decision"],
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", default=DEFAULT_CACHE.as_posix())
    parser.add_argument("--artifact", default=DEFAULT_ARTIFACT.as_posix())
    parser.add_argument("--artifact-sha256", default=DEFAULT_ARTIFACT_SHA256)
    parser.add_argument("--inner-report", default=DEFAULT_INNER_REPORT.as_posix())
    parser.add_argument("--inner-report-sha256", default=DEFAULT_INNER_REPORT_SHA256)
    parser.add_argument("--primary-report", default=DEFAULT_PRIMARY_REPORT.as_posix())
    parser.add_argument("--primary-report-sha256", default=DEFAULT_PRIMARY_REPORT_SHA256)
    parser.add_argument("--output-json", default=DEFAULT_JSON.as_posix())
    parser.add_argument("--output-markdown", default=DEFAULT_MARKDOWN.as_posix())
    args = parser.parse_args()
    root = repo_root()
    artifact_path = (root / args.artifact).resolve()
    inner_report_path = (root / args.inner_report).resolve()
    primary_report_path = (root / args.primary_report).resolve()
    for path, expected, label in (
        (artifact_path, args.artifact_sha256, "Scene-ranker artifact"),
        (inner_report_path, args.inner_report_sha256, "Inner report"),
        (primary_report_path, args.primary_report_sha256, "Primary report"),
    ):
        if sha256(path) != expected:
            raise ValueError(f"{label} hash mismatch")
    payload = joblib.load(artifact_path)
    inner_report = json.loads(inner_report_path.read_text(encoding="utf-8"))
    primary_report = json.loads(primary_report_path.read_text(encoding="utf-8"))
    primary_summary = primary_report["alphas"]["0.5"]
    if payload["spec"] != inner_report["selected"]["spec"]:
        raise ValueError("Ranker artifact model differs from the frozen inner selection")
    if float(payload["blend_lambda"]) != float(inner_report["selected"]["blend_lambda"]):
        raise ValueError("Ranker artifact blend differs from the frozen inner selection")

    cache_path = (root / args.cache).resolve()
    cache_hash = sha256(cache_path)
    with np.load(cache_path, allow_pickle=False) as cache:
        features = cache["features"].astype(np.float64)
        feature_names = cache["feature_names"].astype(str)
        labels = cache["labels"].astype(np.uint8)
        sensors = cache["sensors"].astype(np.uint8)
        groups = cache["groups"].astype(str)
        folds = cache["folds"].astype(np.uint8)
        cache_provenance = {
            key: str(cache[key].item())
            for key in ("artifact_sha256", "manifest_sha256", "protocol_sha256")
        }
    if set(np.unique(folds).tolist()) != {0}:
        raise ValueError("Fold-0 evaluation cache must contain only fold 0")
    if feature_names.tolist() != payload["feature_names"]:
        raise ValueError("Fold-0 feature schema differs from the fitted ranker")
    if cache_provenance != payload["source_provenance"]:
        raise ValueError("Fold-0 cache provenance differs from ranker fitting data")
    primary_index = int(np.flatnonzero(feature_names == payload["primary_feature"])[0])
    primary_metrics = metric_summary(labels, features[:, primary_index], sensors)
    assert_primary_identity(primary_metrics, primary_summary)
    head = predict_model(payload["fitted"], features)
    scores = blend_scores(features[:, primary_index], head, float(payload["blend_lambda"]))
    candidate = metric_summary(labels, scores, sensors)
    result = evaluate_candidate(candidate, primary_summary)
    passed = all(result["checks"].values())
    decision = (
        "Advance the frozen scene-head architecture to independent fold-1 confirmation."
        if passed else "Reject the frozen scene-head architecture on fold 0."
    )
    report = {
        "schema_version": 1,
        "scope": "one-shot fold-0 architecture selection; fold 1 and paper test not loaded",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows": int(labels.size), "positive": int(np.count_nonzero(labels == 1)),
        "sites": len(set(groups.tolist())),
        "architecture": {
            "segmentation": "frozen primary residual at exact alpha 0.5",
            "scene_head_spec": payload["spec"],
            "scene_head_blend_lambda": payload["blend_lambda"],
        },
        "primary_endpoint_identity": primary_metrics,
        "result": result,
        "decision": decision,
        "provenance": {
            "cache_path": args.cache, "cache_sha256": cache_hash,
            **cache_provenance,
            "artifact_path": args.artifact, "artifact_sha256": args.artifact_sha256,
            "inner_report_sha256": args.inner_report_sha256,
            "primary_report_sha256": args.primary_report_sha256,
            "script_sha256": sha256(Path(__file__).resolve()),
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True
            ).strip(),
            "numpy": np.__version__, "scipy": scipy.__version__,
            "sklearn": sklearn.__version__, "joblib": joblib.__version__,
        },
    }
    output_json = (root / args.output_json).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown((root / args.output_markdown).resolve(), report)
    print(json.dumps({"ok": passed, "checks": result["checks"], "decision": decision}, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
