#!/usr/bin/env python3
"""Refit the frozen v2 scene-head specification on all authorized development folds."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
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
from train_mars_context_scene_ranker import augment_site_context  # noqa: E402
from train_mars_oof_scene_ensemble_v2 import fit_model  # noqa: E402
from train_mars_scene_ranker import blend_scores, metric_summary  # noqa: E402

DEFAULT_INNER_CACHE = Path("outputs/mars_scene_features_folds234.npz")
DEFAULT_FOLD0_CACHE = Path("outputs/mars_scene_features_fold0.npz")
DEFAULT_FOLD1_CACHE = Path("outputs/mars_scene_features_fold1_crossfit.npz")
DEFAULT_SELECTION = Path("reports/experiments/mars_oof_scene_ensemble_v2_folds234.json")
DEFAULT_FOLD0 = Path("reports/experiments/mars_oof_scene_ensemble_v2_fold0.json")
DEFAULT_FOLD1 = Path("reports/experiments/mars_oof_scene_ensemble_v2_fold1.json")
DEFAULT_ARTIFACT = Path("EarthRemoteSensingRapidResponse/artifacts/mars_oof_scene_ensemble_v2_all_development.joblib")
DEFAULT_JSON = Path("reports/experiments/mars_oof_scene_ensemble_v2_all_development_refit.json")
DEFAULT_MARKDOWN = Path("reports/experiments/MARS_OOF_SCENE_ENSEMBLE_V2_ALL_DEVELOPMENT_REFIT.md")
EXPECTED_CACHE_HASHES = {
    "inner": "01d8587e283c1179d61a7c789eb514b3f699d3e7a75bf8c50e4baff3f1698b89",
    "fold0": "372e152734db1314417ed385b099af54acd182bf758b1d2eabcedfeb64a709e7",
    "fold1": "2b62e03215047d6a49639fdaead7e9d3cf7939b8eda26fb9442210b49c3ba108",
}
FROZEN_SPEC = {
    "family": "extra_trees",
    "weighting": "uniform",
    "n_estimators": 400,
    "min_samples_leaf": 5,
    "max_features": 0.5,
}
FROZEN_BLEND = 0.625


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    metrics = report["training_fit_audit"]
    lines = [
        "# Frozen MARS scene head: all-development refit",
        "",
        "The ExtraTrees specification, feature schema, and 0.625 logit blend were selected and "
        "confirmed before this refit. All five authorized development folds are now used for the "
        "final fit; paper imagery and labels were not loaded.",
        "",
        f"- Rows / positive scenes / sites: {report['cohort']['rows']:,} / "
        f"{report['cohort']['positive']:,} / {report['cohort']['sites']:,}",
        f"- Frozen specification: `{report['spec']}`",
        f"- Development-calibrated threshold at <=7.13% FPR: "
        f"{report['operational_scene_threshold']:.9f}",
        f"- In-sample fit AP (audit only): {metrics['average_precision']:.5f}",
        f"- Artifact SHA-256: `{report['provenance']['artifact_sha256']}`",
        "",
        report["decision"],
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inner-cache", default=DEFAULT_INNER_CACHE.as_posix())
    parser.add_argument("--fold0-cache", default=DEFAULT_FOLD0_CACHE.as_posix())
    parser.add_argument("--fold1-cache", default=DEFAULT_FOLD1_CACHE.as_posix())
    parser.add_argument("--selection-report", default=DEFAULT_SELECTION.as_posix())
    parser.add_argument("--fold0-report", default=DEFAULT_FOLD0.as_posix())
    parser.add_argument("--fold1-report", default=DEFAULT_FOLD1.as_posix())
    parser.add_argument("--artifact", default=DEFAULT_ARTIFACT.as_posix())
    parser.add_argument("--output-json", default=DEFAULT_JSON.as_posix())
    parser.add_argument("--output-markdown", default=DEFAULT_MARKDOWN.as_posix())
    args = parser.parse_args()
    root = repo_root()
    cache_paths = {
        "inner": (root / args.inner_cache).resolve(),
        "fold0": (root / args.fold0_cache).resolve(),
        "fold1": (root / args.fold1_cache).resolve(),
    }
    for name, path in cache_paths.items():
        if sha256(path) != EXPECTED_CACHE_HASHES[name]:
            raise ValueError(f"Frozen {name} feature cache hash mismatch")
    report_paths = {
        "selection": (root / args.selection_report).resolve(),
        "fold0": (root / args.fold0_report).resolve(),
        "fold1": (root / args.fold1_report).resolve(),
    }
    source_reports = {
        name: json.loads(path.read_text(encoding="utf-8"))
        for name, path in report_paths.items()
    }
    if source_reports["selection"].get("passed") is not True:
        raise ValueError("Folds 2/3/4 did not authorize the selected scene head")
    if source_reports["selection"]["selected"]["spec"] != FROZEN_SPEC:
        raise ValueError("Selected scene-head specification differs from the frozen refit")
    if float(source_reports["selection"]["selected"]["blend_lambda"]) != FROZEN_BLEND:
        raise ValueError("Selected scene-head blend differs from the frozen refit")
    if source_reports["fold0"].get("passed") is not True or source_reports["fold1"].get("passed") is not True:
        raise ValueError("Both held development folds must pass before all-development refit")

    parts = []
    feature_names: np.ndarray | None = None
    for name in ("inner", "fold0", "fold1"):
        with np.load(cache_paths[name], allow_pickle=False) as cache:
            local_names = cache["feature_names"].astype(str)
            if feature_names is None:
                feature_names = local_names
            elif not np.array_equal(feature_names, local_names):
                raise ValueError("Development feature schemas differ")
            parts.append(
                {
                    "features": cache["features"].astype(np.float32),
                    "labels": cache["labels"].astype(np.uint8),
                    "sensors": cache["sensors"].astype(np.uint8),
                    "sample_ids": cache["sample_ids"].astype(str),
                    "groups": cache["groups"].astype(str),
                    "folds": cache["folds"].astype(np.uint8),
                }
            )
    assert feature_names is not None
    base = np.concatenate([part["features"] for part in parts])
    labels = np.concatenate([part["labels"] for part in parts])
    sensors = np.concatenate([part["sensors"] for part in parts])
    sample_ids = np.concatenate([part["sample_ids"] for part in parts])
    groups = np.concatenate([part["groups"] for part in parts])
    folds = np.concatenate([part["folds"] for part in parts])
    if len(set(sample_ids.tolist())) != len(sample_ids) or set(folds.tolist()) != set(range(5)):
        raise ValueError("All-development refit rows are duplicated or do not cover five folds")
    features, augmented_names = augment_site_context(base, feature_names, groups)
    primary_index = int(np.flatnonzero(feature_names == "primary_connected_score")[0])
    primary = base[:, primary_index].astype(np.float64)
    fitted = fit_model(FROZEN_SPEC, features, labels, np.ones(labels.shape, dtype=np.float64))
    head_probability = fitted.predict_proba(features)[:, 1]
    scores = blend_scores(primary, head_probability, FROZEN_BLEND)
    metrics = metric_summary(labels, scores, sensors)
    threshold = float(metrics["operating_point"]["threshold"])

    artifact_path = (root / args.artifact).resolve()
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = artifact_path.with_suffix(artifact_path.suffix + ".tmp")
    payload = {
        "schema_version": 1,
        "architecture": "mars_oof_scene_ensemble_v2_all_development_refit",
        "spec": FROZEN_SPEC,
        "blend_lambda": FROZEN_BLEND,
        "feature_names": feature_names.tolist(),
        "augmented_feature_names": augmented_names,
        "primary_feature": "primary_connected_score",
        "training_folds": [0, 1, 2, 3, 4],
        "training_rows": int(labels.size),
        "operational_scene_threshold": threshold,
        "cache_sha256": EXPECTED_CACHE_HASHES,
        "source_report_sha256": {name: sha256(path) for name, path in report_paths.items()},
        "fitted": fitted,
    }
    joblib.dump(payload, temporary, compress=3)
    os.replace(temporary, artifact_path)
    report = {
        "schema_version": 1,
        "scope": "fixed-spec all-development refit; paper test not loaded",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "authorization": {
            "selection_passed": True,
            "fold0_passed": True,
            "fold1_passed": True,
            "architecture_or_hyperparameter_search": False,
        },
        "cohort": {
            "rows": int(labels.size),
            "positive": int(labels.sum()),
            "negative": int(labels.size - labels.sum()),
            "sites": len(set(groups.tolist())),
            "folds": sorted(int(value) for value in set(folds.tolist())),
        },
        "spec": FROZEN_SPEC,
        "blend_lambda": FROZEN_BLEND,
        "operational_scene_threshold": threshold,
        "threshold_rule": "highest development score threshold with false-positive rate <= 0.0713",
        "training_fit_audit": metrics,
        "decision": "Freeze this all-development refit for one transparent exact-paper cache evaluation.",
        "provenance": {
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
            "script_sha256": sha256(Path(__file__).resolve()),
            "artifact_sha256": sha256(artifact_path),
            "cache_sha256": EXPECTED_CACHE_HASHES,
            "source_report_sha256": {name: sha256(path) for name, path in report_paths.items()},
            "numpy": np.__version__,
            "sklearn": __import__("sklearn").__version__,
            "joblib": joblib.__version__,
        },
    }
    write_json((root / args.output_json).resolve(), report)
    write_markdown((root / args.output_markdown).resolve(), report)
    print(
        json.dumps(
            {
                "ok": True,
                "rows": int(labels.size),
                "positive": int(labels.sum()),
                "sites": len(set(groups.tolist())),
                "operational_scene_threshold": threshold,
                "training_fit_ap_audit_only": metrics["average_precision"],
                "artifact_sha256": sha256(artifact_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
