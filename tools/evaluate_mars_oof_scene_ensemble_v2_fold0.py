#!/usr/bin/env python3
"""One-shot fold-0 evaluation of the stronger OOF MARS scene ensemble."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import sklearn

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "EarthRemoteSensingRapidResponse"
for path in (MODEL_ROOT, ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from acquire_mars_metadata import repo_root, sha256  # noqa: E402
from evaluate_mars_scene_ranker_fold0 import assert_primary_identity, evaluate_candidate  # noqa: E402
from train_mars_context_scene_ranker import augment_site_context  # noqa: E402
from train_mars_oof_scene_ensemble_v2 import ap_group_bootstrap  # noqa: E402
from train_mars_scene_ranker import blend_scores, metric_summary  # noqa: E402

DEFAULT_CACHE = Path("outputs/mars_scene_features_fold0.npz")
DEFAULT_CACHE_SHA256 = "372e152734db1314417ed385b099af54acd182bf758b1d2eabcedfeb64a709e7"
DEFAULT_ARTIFACT = Path("EarthRemoteSensingRapidResponse/artifacts/mars_oof_scene_ensemble_v2.joblib")
DEFAULT_ARTIFACT_SHA256 = "9e6fa18b83ef065ac24c94a06a510057a0c382cecf1efa3b54e818566a45c9ac"
DEFAULT_OOF_REPORT = Path("reports/experiments/mars_oof_scene_ensemble_v2_folds234.json")
DEFAULT_OOF_REPORT_SHA256 = "b12f62e05d864739f8a278b414a415febaba0185c649b5dea901eaf4135ef157"
DEFAULT_PRIMARY_REPORT = Path("reports/experiments/mars_paper_residual_fold0_trust_region.json")
DEFAULT_PRIMARY_REPORT_SHA256 = "bb9763a9bdbf0ddd14c3e1c718af9bceff47e0a1c6f04a4749833968368b79b5"
DEFAULT_JSON = Path("reports/experiments/mars_oof_scene_ensemble_v2_fold0.json")
DEFAULT_MARKDOWN = Path("reports/experiments/MARS_OOF_SCENE_ENSEMBLE_V2_FOLD0.md")


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_markdown(path: Path, report: dict) -> None:
    baseline = report["result"]["released_baseline"]
    candidate = report["result"]["scene"]
    delta = report["result"]["delta"]
    bootstrap = report["paired_group_bootstrap_ap_delta"]
    lines = [
        "# Stronger OOF MARS scene ensemble: one-shot fold 0",
        "",
        f"The head was frozen on folds 2/3/4. This evaluates fold {report['fold']}; the paper test was not loaded.",
        "",
        "| Model | AP | Recall at <=7.13% FPR | FPR | Pixel IoU |",
        "|---|---:|---:|---:|---:|",
        f"| Released MARS-S2L | {baseline['average_precision']:.5f} | {baseline['recall_at_fpr_0_0713']:.5f} | {baseline['false_positive_rate_at_target']:.5f} | {baseline['pixel_iou']:.5f} |",
        f"| OOF scene ensemble v2 | {candidate['average_precision']:.5f} | {candidate['operating_point']['recall']:.5f} | {candidate['operating_point']['false_positive_rate']:.5f} | {report['result']['pixel_iou']:.5f} |",
        "",
        f"AP delta: {delta['average_precision']:+.5f}; paired site-bootstrap 95% CI [{bootstrap['lower']:+.5f}, {bootstrap['upper']:+.5f}].",
        "",
        report["decision"],
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", default=DEFAULT_CACHE.as_posix())
    parser.add_argument("--cache-sha256", default=DEFAULT_CACHE_SHA256)
    parser.add_argument("--artifact", default=DEFAULT_ARTIFACT.as_posix())
    parser.add_argument("--artifact-sha256", default=DEFAULT_ARTIFACT_SHA256)
    parser.add_argument("--oof-report", default=DEFAULT_OOF_REPORT.as_posix())
    parser.add_argument("--oof-report-sha256", default=DEFAULT_OOF_REPORT_SHA256)
    parser.add_argument("--primary-report", default=DEFAULT_PRIMARY_REPORT.as_posix())
    parser.add_argument("--primary-report-sha256", default=DEFAULT_PRIMARY_REPORT_SHA256)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--output-json", default=DEFAULT_JSON.as_posix())
    parser.add_argument("--output-markdown", default=DEFAULT_MARKDOWN.as_posix())
    args = parser.parse_args()
    if args.fold not in range(5):
        parser.error("fold must be in [0,4]")
    root = repo_root()
    paths = {
        "cache": (root / args.cache).resolve(),
        "artifact": (root / args.artifact).resolve(),
        "oof": (root / args.oof_report).resolve(),
        "primary": (root / args.primary_report).resolve(),
    }
    for name, expected in (
        ("cache", args.cache_sha256),
        ("artifact", args.artifact_sha256),
        ("oof", args.oof_report_sha256),
        ("primary", args.primary_report_sha256),
    ):
        if sha256(paths[name]) != expected:
            raise ValueError(f"Frozen {name} hash mismatch")
    payload = joblib.load(paths["artifact"])
    oof_report = json.loads(paths["oof"].read_text(encoding="utf-8"))
    primary_report = json.loads(paths["primary"].read_text(encoding="utf-8"))
    if payload["architecture"] != "mars_oof_scene_ensemble_v2":
        raise ValueError("Unexpected scene-head architecture")
    selected = oof_report["selected"]
    if payload["spec"] != selected["spec"] or float(payload["blend_lambda"]) != float(selected["blend_lambda"]):
        raise ValueError("Artifact differs from the frozen OOF selection")
    if not oof_report["passed"] or selected["paired_group_bootstrap_ap_delta"]["lower"] <= 0.0:
        raise ValueError("OOF report does not authorize fold-0 evaluation")
    with np.load(paths["cache"], allow_pickle=False) as cache:
        base_features = cache["features"].astype(np.float64)
        feature_names = cache["feature_names"].astype(str)
        labels = cache["labels"].astype(np.uint8)
        sensors = cache["sensors"].astype(np.uint8)
        groups = cache["groups"].astype(str)
        folds = cache["folds"].astype(np.uint8)
        cache_provenance = {
            name: str(cache[name].item())
            for name in ("artifact_sha256", "manifest_sha256", "protocol_sha256")
        }
    if set(np.unique(folds).tolist()) != {args.fold}:
        raise ValueError("Evaluation cache contains another fold")
    if feature_names.tolist() != payload["feature_names"]:
        raise ValueError("Evaluation feature schema mismatch")
    for name in ("manifest_sha256", "protocol_sha256"):
        if cache_provenance[name] != payload["source_provenance"][name]:
            raise ValueError("Evaluation cache data provenance mismatch")
    if args.fold == 0 and cache_provenance["artifact_sha256"] != payload["source_provenance"]["artifact_sha256"]:
        raise ValueError("Fold-0 cache residual provenance mismatch")
    features, augmented_names = augment_site_context(base_features, feature_names, groups)
    if augmented_names != payload["augmented_feature_names"]:
        raise ValueError("Augmented context schema mismatch")
    primary_index = int(np.flatnonzero(feature_names == payload["primary_feature"])[0])
    primary_scores = base_features[:, primary_index]
    primary_metrics = metric_summary(labels, primary_scores, sensors)
    primary_summary = primary_report["alphas"]["0.5"]
    assert_primary_identity(primary_metrics, primary_summary)
    head = payload["fitted"].predict_proba(features)[:, 1]
    scores = blend_scores(primary_scores, head, float(payload["blend_lambda"]))
    result = evaluate_candidate(metric_summary(labels, scores, sensors), primary_summary)
    bootstrap = ap_group_bootstrap(
        labels,
        primary_scores,
        scores,
        groups,
        replicates=args.bootstrap_replicates,
        seed=20262715,
    )
    checks = {**result["checks"], "paired_group_bootstrap_ap_lower_positive": bootstrap["lower"] > 0.0}
    passed = all(checks.values())
    report = {
        "schema_version": 1,
        "scope": f"frozen stronger OOF scene-head fold-{args.fold} evaluation; paper test not loaded",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "fold": args.fold,
        "rows": int(labels.size),
        "positive": int(labels.sum()),
        "sites": len(set(groups.tolist())),
        "architecture": {
            "segmentation": "frozen primary residual at exact alpha 0.5",
            "scene_head": payload["architecture"],
            "spec": payload["spec"],
            "blend_lambda": payload["blend_lambda"],
        },
        "primary_endpoint_identity": primary_metrics,
        "result": result,
        "paired_group_bootstrap_ap_delta": bootstrap,
        "checks": checks,
        "passed": passed,
        "decision": (
            "Advance the stronger OOF scene ensemble to the next confirmation stage."
            if passed
            else "Reject the stronger OOF scene ensemble on fold 0."
        ),
        "provenance": {
            "cache_sha256": args.cache_sha256,
            **cache_provenance,
            "artifact_sha256": args.artifact_sha256,
            "oof_report_sha256": args.oof_report_sha256,
            "primary_report_sha256": args.primary_report_sha256,
            "script_sha256": sha256(Path(__file__).resolve()),
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
    }
    write_json((root / args.output_json).resolve(), report)
    write_markdown((root / args.output_markdown).resolve(), report)
    print(json.dumps({
        "ok": passed,
        "ap": result["scene"]["average_precision"],
        "ap_delta": result["delta"]["average_precision"],
        "recall_delta": result["delta"]["recall_at_fpr_0_0713"],
        "bootstrap": bootstrap,
        "checks": checks,
    }, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
