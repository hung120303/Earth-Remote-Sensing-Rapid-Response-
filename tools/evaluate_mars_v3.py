#!/usr/bin/env python3
"""Evaluate one frozen MARS v3 checkpoint without threshold or architecture tuning."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import sklearn
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "EarthRemoteSensingRapidResponse"
if str(MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_ROOT))

from mars_s2l_adapter import iter_manifest  # noqa: E402
from mars_v3_model import MarsV3Model  # noqa: E402

from acquire_mars_metadata import DEFAULT_OUTPUT, REVISION, checked_output_dir, repo_root, sha256  # noqa: E402
from build_mars_dev_cohort import DEV_SAMPLES, DEFAULT_JSON as DEV_REPORT_JSON  # noqa: E402
from run_mars_dev_pixel_baselines import evaluate_rule  # noqa: E402
from run_mars_dev_scene_baselines import bootstrap_ci, metrics, role_weights  # noqa: E402
from train_mars_joint_model import selective_quality_metrics  # noqa: E402
from train_mars_v3 import (  # noqa: E402
    DEFAULT_METADATA_CSV,
    MarsV3Dataset,
    collect_predictions,
    wind_lookup,
)

DEFAULT_EXPERIMENT = Path("reports/experiments/mars_v3_validation.json")
DEFAULT_JSON = Path("reports/experiments/mars_v3_strict_evaluation.json")
DEFAULT_MARKDOWN = Path("reports/experiments/MARS_V3_STRICT_EVALUATION.md")


def safe_output(root: Path, value: str) -> Path:
    result = (root / value).resolve()
    if root not in result.parents:
        raise ValueError("Output must resolve beneath the repository root")
    return result


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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    scene = report["strict_spatial_test"]["scene_unweighted"]
    pixel = report["strict_spatial_test"]["segmentation"]["pixel"]
    interval = report["strict_spatial_test"]["group_bootstrap"]
    lines = [
        "# Frozen ERSRR MARS v3 strict-spatial evaluation",
        "",
        "Checkpoint and all operating rules were selected on internal validation before this run.",
        "",
        f"- Cohort: {report['cohort']['samples']} scenes / {report['cohort']['groups']} frozen 25 km groups",
        f"- Scene recall / specificity / FPR: {fmt(scene['recall'])} / {fmt(scene['specificity'])} / {fmt(scene['false_positive_rate'])}",
        f"- Scene AUROC / AP: {scene['auroc']:.3f} / {scene['average_precision']:.3f}",
        f"- Recall 95% CI: {interval['recall_95ci'][0]:.3f}-{interval['recall_95ci'][1]:.3f}",
        f"- Specificity 95% CI: {interval['specificity_95ci'][0]:.3f}-{interval['specificity_95ci'][1]:.3f}",
        f"- Pixel AP / IoU / Dice: {pixel['average_precision']:.4f} / {pixel['intersection_over_union']:.4f} / {pixel['dice']:.4f}",
        f"- Promotion gate: {'PASS' if report['promotion_gate_passed'] else 'FAIL'}",
        "",
        "## Decision",
        "",
        report["decision"],
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-dir", default=DEFAULT_OUTPUT.as_posix())
    parser.add_argument("--metadata-csv", default=DEFAULT_METADATA_CSV.as_posix())
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT.as_posix())
    parser.add_argument("--output-json", default=DEFAULT_JSON.as_posix())
    parser.add_argument("--output-markdown", default=DEFAULT_MARKDOWN.as_posix())
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    root = repo_root()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for v3 evaluation")
    metadata_dir = checked_output_dir(root, args.metadata_dir)
    metadata_csv = (root / args.metadata_csv).resolve()
    experiment_path = (root / args.experiment).resolve()
    output_json = safe_output(root, args.output_json)
    output_markdown = safe_output(root, args.output_markdown)
    experiment = json.loads(experiment_path.read_text(encoding="utf-8"))
    if experiment.get("smoke_test"):
        raise ValueError("A pipeline smoke checkpoint cannot consume the strict benchmark")
    if experiment["scope"] != "v3_internal_validation_selection":
        raise ValueError("Experiment is not a frozen v3 validation-selection artifact")
    checkpoint = (root / experiment["artifact"]["path"]).resolve()
    if sha256(checkpoint) != experiment["artifact"]["sha256"]:
        raise ValueError("V3 checkpoint identity does not match the validation report")
    if sha256(MODEL_ROOT / "mars_v3_model.py") != experiment["provenance"]["model_source_sha256"]:
        raise ValueError("V3 model source changed after checkpoint selection")

    manifest = metadata_dir / DEV_SAMPLES
    development = json.loads((root / DEV_REPORT_JSON).read_text(encoding="utf-8"))
    if sha256(manifest) != development["identities"]["sample_manifest_sha256"]:
        raise ValueError("Strict development manifest identity mismatch")
    all_records = list(iter_manifest(manifest))
    records = [
        record for record in all_records if record["research_role"] == "strict_spatial_test"
    ]
    required_ids = {str(record["sample_id"]) for record in records}
    winds = wind_lookup(metadata_csv, required_ids)
    dataset = MarsV3Dataset(metadata_dir, records, winds, augment=False, seed=0)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    device = torch.device("cuda")
    model = MarsV3Model().to(device)
    payload = torch.load(checkpoint, map_location=device, weights_only=True)
    if payload["model_metadata"] != experiment["model"]:
        raise ValueError("Checkpoint model metadata differs from the validation report")
    model.load_state_dict(payload["state_dict"], strict=True)
    predictions = collect_predictions(model, loader, device)

    rule = experiment["operating_rule"]
    upper = float(rule["upper_plume_threshold"])
    lower = rule["lower_no_plume_threshold"]
    lower = None if lower is None else float(lower)
    quality_threshold = float(rule["quality_threshold"])
    segmentation_rule = dict(rule["segmentation"])
    labels = predictions["labels"]
    scores = predictions["presence"]
    groups = predictions["groups"].astype(str)
    weights = role_weights(labels, "strict_spatial_test")
    scene_unweighted = metrics(labels, scores, upper)
    scene_weighted = metrics(labels, scores, upper, weights=weights)
    segmentation = evaluate_rule(
        predictions["segmentation"],
        predictions["observable"],
        predictions["truth"],
        labels,
        groups,
        segmentation_rule,
    )
    interval = bootstrap_ci(labels, scores, groups, upper, int(experiment["training"]["seed"]))
    selective = selective_quality_metrics(
        labels,
        scores,
        predictions["quality"],
        lower,
        upper,
        quality_threshold,
        weights,
    )
    gate = (
        float(interval["recall_95ci"][0]) >= 0.75
        and float(scene_unweighted["false_positive_rate"] or 1.0) <= 0.05
        and float(scene_unweighted["specificity"] or 0.0) >= 0.95
    )
    report = {
        "schema_version": 1,
        "scope": "frozen_v3_strict_spatial_development_evaluation",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "dataset": "UNEP-IMEO/MARS-S2L",
            "revision": REVISION,
            "strict_manifest_sha256": sha256(manifest),
            "validation_experiment_sha256": sha256(experiment_path),
        },
        "cohort": {
            "samples": len(records),
            "positives": int(np.sum(labels)),
            "negatives": int(np.sum(labels == 0)),
            "groups": int(np.unique(groups).size),
        },
        "model": experiment["model"],
        "artifact": experiment["artifact"],
        "operating_rule": rule,
        "strict_spatial_test": {
            "scene_unweighted": scene_unweighted,
            "scene_representative_weighted": scene_weighted,
            "segmentation": segmentation,
            "group_bootstrap": interval,
            "selective_with_quality": selective,
            "quality_auroc": float(
                roc_auc_score(predictions["quality_labels"], predictions["quality"])
            ),
        },
        "promotion_gate_passed": gate,
        "decision": (
            "V3 clears the provisional development gate. Run the remaining fixed seeds and untouched EMIT confirmation before promotion."
            if gate
            else "V3 does not clear the frozen development gate. Preserve this result; do not retune from strict-test behavior."
        ),
        "runtime": {
            "device": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
            "rasterio": rasterio.__version__,
        },
        "provenance": {
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True
            ).strip(),
            "git_tracked_worktree_dirty_at_start": tracked_dirty(root),
            "script": "tools/evaluate_mars_v3.py",
            "script_sha256": sha256(Path(__file__)),
        },
    }
    write_json(output_json, report)
    write_markdown(output_markdown, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
