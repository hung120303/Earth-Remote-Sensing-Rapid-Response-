#!/usr/bin/env python3
"""One-shot fold-0 evaluation at the minimum OOF-stable context blend."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import scipy
import sklearn

from acquire_mars_metadata import repo_root, sha256
from evaluate_mars_context_scene_ranker_fold0 import write_markdown
from evaluate_mars_scene_ranker_fold0 import assert_primary_identity, evaluate_candidate
from train_mars_context_scene_ranker import augment_site_context
from train_mars_scene_ranker import blend_scores, metric_summary, predict_model

DEFAULT_CACHE = Path("outputs/mars_scene_features_fold0.npz")
DEFAULT_CACHE_SHA256 = "372e152734db1314417ed385b099af54acd182bf758b1d2eabcedfeb64a709e7"
DEFAULT_ARTIFACT = Path("EarthRemoteSensingRapidResponse/artifacts/mars_oof_context_ranker_folds234.joblib")
DEFAULT_ARTIFACT_SHA256 = "2d014f54918f68726d2ca4da19f35a1f29cb1b622fe7c32b56afc554ec27c370"
DEFAULT_SELECTION = Path("reports/experiments/mars_oof_context_minimum_blend.json")
DEFAULT_SELECTION_SHA256 = "8fc190bf4cac9d3abb24979cd20678930f09143302a69ddbfb944a7959951b0b"
DEFAULT_PRIMARY_REPORT = Path("reports/experiments/mars_paper_residual_fold0_trust_region.json")
DEFAULT_PRIMARY_REPORT_SHA256 = "bb9763a9bdbf0ddd14c3e1c718af9bceff47e0a1c6f04a4749833968368b79b5"
DEFAULT_JSON = Path("reports/experiments/mars_oof_context_minimum_blend_fold0.json")
DEFAULT_MARKDOWN = Path("reports/experiments/MARS_OOF_CONTEXT_MINIMUM_BLEND_FOLD0.md")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", default=DEFAULT_CACHE.as_posix())
    parser.add_argument("--cache-sha256", default=DEFAULT_CACHE_SHA256)
    parser.add_argument("--artifact", default=DEFAULT_ARTIFACT.as_posix())
    parser.add_argument("--artifact-sha256", default=DEFAULT_ARTIFACT_SHA256)
    parser.add_argument("--selection", default=DEFAULT_SELECTION.as_posix())
    parser.add_argument("--selection-sha256", default=DEFAULT_SELECTION_SHA256)
    parser.add_argument("--primary-report", default=DEFAULT_PRIMARY_REPORT.as_posix())
    parser.add_argument("--primary-report-sha256", default=DEFAULT_PRIMARY_REPORT_SHA256)
    parser.add_argument("--output-json", default=DEFAULT_JSON.as_posix())
    parser.add_argument("--output-markdown", default=DEFAULT_MARKDOWN.as_posix())
    args = parser.parse_args()
    root = repo_root()
    paths = {
        "cache": (root / args.cache).resolve(),
        "artifact": (root / args.artifact).resolve(),
        "selection": (root / args.selection).resolve(),
        "primary": (root / args.primary_report).resolve(),
    }
    for key, expected in (
        ("cache", args.cache_sha256),
        ("artifact", args.artifact_sha256),
        ("selection", args.selection_sha256),
        ("primary", args.primary_report_sha256),
    ):
        if sha256(paths[key]) != expected:
            raise ValueError(f"Frozen {key} hash mismatch")
    payload = joblib.load(paths["artifact"])
    selection = json.loads(paths["selection"].read_text(encoding="utf-8"))
    primary_report = json.loads(paths["primary"].read_text(encoding="utf-8"))
    if payload["architecture"] != "oof_stable_context_scene_ranker_v1":
        raise ValueError("Unexpected ranker architecture")
    if payload["spec"] != selection["frozen_spec"]:
        raise ValueError("Artifact model differs from minimum-blend selection")
    if not all(selection["selected"]["stability_checks"].values()):
        raise ValueError("Minimum blend did not pass all OOF stability checks")
    blend_lambda = float(selection["selected"]["blend_lambda"])
    if blend_lambda != 0.25:
        raise ValueError("Frozen minimum blend is not the preregistered 0.25")
    with np.load(paths["cache"], allow_pickle=False) as cache:
        base_features = cache["features"].astype(np.float64)
        base_names = cache["feature_names"].astype(str)
        labels = cache["labels"].astype(np.uint8)
        sensors = cache["sensors"].astype(np.uint8)
        groups = cache["groups"].astype(str)
        folds = cache["folds"].astype(np.uint8)
        cache_provenance = {
            key: str(cache[key].item())
            for key in ("artifact_sha256", "manifest_sha256", "protocol_sha256")
        }
    if set(np.unique(folds).tolist()) != {0}:
        raise ValueError("Evaluation cache must contain fold 0 only")
    if base_names.tolist() != payload["feature_names"] or cache_provenance != payload["source_provenance"]:
        raise ValueError("Fold-0 cache schema or provenance mismatch")
    features, augmented_names = augment_site_context(base_features, base_names, groups)
    if augmented_names != payload["augmented_feature_names"]:
        raise ValueError("Augmented context schema mismatch")
    primary_index = int(np.flatnonzero(base_names == payload["primary_feature"])[0])
    primary_metrics = metric_summary(labels, base_features[:, primary_index], sensors)
    primary_summary = primary_report["alphas"]["0.5"]
    assert_primary_identity(primary_metrics, primary_summary)
    head = predict_model(payload["fitted"], features)
    scores = blend_scores(base_features[:, primary_index], head, blend_lambda)
    result = evaluate_candidate(metric_summary(labels, scores, sensors), primary_summary)
    passed = all(result["checks"].values())
    decision = (
        "Advance the minimum-intervention OOF context architecture to fold-1 confirmation."
        if passed else "Reject the minimum-intervention OOF context architecture on fold 0."
    )
    report = {
        "schema_version": 1,
        "scope": "one-shot minimum-intervention OOF-context fold-0 evaluation",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows": int(labels.size), "positive": int(np.count_nonzero(labels == 1)),
        "sites": len(set(groups.tolist())),
        "architecture": {
            "segmentation": "frozen primary residual at exact alpha 0.5",
            "scene_head": payload["architecture"], "spec": payload["spec"],
            "blend_lambda": blend_lambda,
            "selection_rule": selection["selection_rule"],
        },
        "primary_endpoint_identity": primary_metrics,
        "result": result, "decision": decision,
        "provenance": {
            "cache_sha256": args.cache_sha256, **cache_provenance,
            "artifact_sha256": args.artifact_sha256,
            "selection_sha256": args.selection_sha256,
            "primary_report_sha256": args.primary_report_sha256,
            "script_sha256": sha256(Path(__file__).resolve()),
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
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
