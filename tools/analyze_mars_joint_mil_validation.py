#!/usr/bin/env python3
"""Audit MIL-v2 errors using the frozen internal validation cohort only."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import scipy
import torch
from scipy import ndimage
from scipy.stats import spearmanr
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "EarthRemoteSensingRapidResponse"
if str(MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_ROOT))

from mars_joint_mil_model import MarsJointMILModel  # noqa: E402
from mars_s2l_adapter import iter_manifest  # noqa: E402

from acquire_mars_metadata import DEFAULT_OUTPUT, REVISION, checked_output_dir, repo_root, sha256  # noqa: E402
from build_mars_dev_cohort import DEV_SAMPLES, DEFAULT_JSON as DEV_REPORT_JSON  # noqa: E402
from train_mars_joint_model import MarsJointDataset, move_batch  # noqa: E402

DEFAULT_CHECKPOINT = Path(
    "EarthRemoteSensingRapidResponse/artifacts/mars_joint_mil_v2_seed101.pt"
)
DEFAULT_EXPERIMENT = Path("reports/experiments/mars_joint_mil_development.json")
DEFAULT_JSON = Path("reports/experiments/mars_joint_mil_validation_audit.json")
DEFAULT_MARKDOWN = Path("reports/experiments/MARS_JOINT_MIL_VALIDATION_AUDIT.md")
DEFAULT_PREDICTIONS = DEFAULT_OUTPUT / "publication_mil_v2_validation_predictions.jsonl"


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


def safe_output(root: Path, value: str) -> Path:
    result = (root / value).resolve()
    if root not in result.parents:
        raise ValueError("Output must resolve beneath the repository root")
    return result


def summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "median": None, "q1": None, "q3": None, "mean": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "median": float(np.median(array)),
        "q1": float(np.quantile(array, 0.25)),
        "q3": float(np.quantile(array, 0.75)),
        "mean": float(np.mean(array)),
    }


def top_fraction_mean(values: np.ndarray, fraction: float = 0.01) -> float:
    flat = values.ravel()
    count = max(1, int(flat.size * fraction))
    selected = np.partition(flat, flat.size - count)[-count:]
    return float(np.mean(selected))


def connected_mask(
    probabilities: np.ndarray, observable: np.ndarray, threshold: float, minimum: int
) -> np.ndarray:
    labels, count = ndimage.label(
        (probabilities >= threshold) & observable,
        structure=np.ones((3, 3), dtype=np.uint8),
    )
    if count == 0:
        return np.zeros(probabilities.shape, dtype=bool)
    sizes = np.bincount(labels.ravel())
    keep = sizes >= minimum
    keep[0] = False
    return keep[labels]


def classify(label: int, score: float, lower: float, upper: float) -> str:
    predicted = score >= upper
    if label == 1:
        return "true_positive" if predicted else "false_negative"
    return "false_positive" if predicted else "true_negative"


def decision_state(score: float, lower: float, upper: float) -> str:
    if score >= upper:
        return "plume"
    if score <= lower:
        return "no_plume"
    return "abstain"


def grouped_summaries(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for outcome in ("true_positive", "false_negative", "false_positive", "true_negative"):
        selected = [row for row in rows if row["outcome"] == outcome]
        result[outcome] = {
            "count": len(selected),
            "presence_score": summary([row["presence_score"] for row in selected]),
            "quality_score": summary([row["quality_score"] for row in selected]),
            "observable_fraction": summary([row["observable_fraction"] for row in selected]),
            "truth_plume_pixels": summary(
                [row["truth_plume_pixels"] for row in selected if row["label"] == 1]
            ),
            "segmentation_top1pct_mean": summary(
                [row["segmentation_top1pct_mean"] for row in selected]
            ),
            "mbmp_top1pct_mean": summary([row["mbmp_top1pct_mean"] for row in selected]),
            "mask_detected_rate": None
            if not selected
            else float(np.mean([row["segmentation_scene_detected"] for row in selected])),
            "top_countries": Counter(row["country"] for row in selected).most_common(10),
            "unique_groups": len({row["group_id"] for row in selected}),
        }
    return result


def plume_size_strata(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    positives = [row for row in rows if row["label"] == 1]
    areas = np.asarray([row["truth_plume_pixels"] for row in positives], dtype=np.float64)
    edges = np.quantile(areas, (0.0, 0.25, 0.5, 0.75, 1.0))
    strata: list[dict[str, Any]] = []
    for index, name in enumerate(("smallest", "small", "large", "largest")):
        low = edges[index]
        high = edges[index + 1]
        selected = [
            row
            for row in positives
            if row["truth_plume_pixels"] >= low
            and (
                row["truth_plume_pixels"] <= high
                if index == 3
                else row["truth_plume_pixels"] < high
            )
        ]
        strata.append(
            {
                "name": name,
                "minimum_truth_pixels": int(low),
                "maximum_truth_pixels": int(high),
                "count": len(selected),
                "presence_recall": float(
                    np.mean([row["outcome"] == "true_positive" for row in selected])
                ),
                "segmentation_scene_recall": float(
                    np.mean([row["segmentation_scene_detected"] for row in selected])
                ),
                "median_presence_score": float(
                    np.median([row["presence_score"] for row in selected])
                ),
            }
        )
    return strata


def correlation(x: list[float], y: list[float]) -> dict[str, float | None]:
    statistic = spearmanr(x, y)
    return {
        "spearman_r": None if not np.isfinite(statistic.statistic) else float(statistic.statistic),
        "p_value": None if not np.isfinite(statistic.pvalue) else float(statistic.pvalue),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_predictions(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as destination:
        for row in rows:
            destination.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    os.replace(temporary, path)


def fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    outcomes = report["outcome_summaries"]
    lines = [
        "# MIL-v2 internal-validation error audit",
        "",
        "Validation-only audit; the strict-spatial benchmark was not loaded or scored.",
        "",
        f"- Cohort: {report['cohort']['samples']} scenes / {report['cohort']['groups']} frozen groups",
        f"- Presence outcomes: {outcomes['true_positive']['count']} TP / {outcomes['false_negative']['count']} FN / {outcomes['false_positive']['count']} FP / {outcomes['true_negative']['count']} TN",
        f"- Selective decisions: {report['decision_counts']['plume']} plume / {report['decision_counts']['no_plume']} no-plume / {report['decision_counts']['abstain']} abstain",
        "",
        "## Plume-size sensitivity",
        "",
        "| Stratum | Pixel area | n | Presence recall | Mask scene recall | Median score |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for item in report["plume_size_strata"]:
        lines.append(
            f"| {item['name']} | {item['minimum_truth_pixels']}-{item['maximum_truth_pixels']} | {item['count']} | {item['presence_recall']:.3f} | {item['segmentation_scene_recall']:.3f} | {item['median_presence_score']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Error signatures",
            "",
            f"- TP median plume area: {fmt(outcomes['true_positive']['truth_plume_pixels']['median'])} pixels; FN median: {fmt(outcomes['false_negative']['truth_plume_pixels']['median'])}.",
            f"- FP median segmentation top-1% probability: {fmt(outcomes['false_positive']['segmentation_top1pct_mean']['median'])}; TN median: {fmt(outcomes['true_negative']['segmentation_top1pct_mean']['median'])}.",
            f"- FP median MBMP top-1% signal: {fmt(outcomes['false_positive']['mbmp_top1pct_mean']['median'])}; TN median: {fmt(outcomes['true_negative']['mbmp_top1pct_mean']['median'])}.",
            "",
            "## Decision",
            "",
            report["decision"],
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-dir", default=DEFAULT_OUTPUT.as_posix())
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT.as_posix())
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT.as_posix())
    parser.add_argument("--output-json", default=DEFAULT_JSON.as_posix())
    parser.add_argument("--output-markdown", default=DEFAULT_MARKDOWN.as_posix())
    parser.add_argument("--predictions", default=DEFAULT_PREDICTIONS.as_posix())
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    root = repo_root()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the MIL-v2 validation audit")
    metadata_dir = checked_output_dir(root, args.metadata_dir)
    manifest = metadata_dir / DEV_SAMPLES
    checkpoint = (root / args.checkpoint).resolve()
    experiment_path = (root / args.experiment).resolve()
    output_json = safe_output(root, args.output_json)
    output_markdown = safe_output(root, args.output_markdown)
    predictions_path = safe_output(root, args.predictions)
    development = json.loads((root / DEV_REPORT_JSON).read_text(encoding="utf-8"))
    experiment = json.loads(experiment_path.read_text(encoding="utf-8"))
    if sha256(manifest) != development["identities"]["sample_manifest_sha256"]:
        raise ValueError("Development manifest identity mismatch")
    if sha256(checkpoint) != experiment["artifact"]["sha256"]:
        raise ValueError("MIL-v2 checkpoint identity mismatch")

    rule = experiment["operating_rule"]
    lower = float(rule["lower_no_plume_threshold"])
    upper = float(rule["upper_plume_threshold"])
    pixel_threshold = float(rule["segmentation"]["pixel_threshold"])
    minimum_pixels = int(rule["segmentation"]["minimum_connected_pixels"])
    all_records = list(iter_manifest(manifest))
    records = [
        record for record in all_records if record["research_role"] == "internal_validation"
    ]
    records_by_id = {str(record["sample_id"]): record for record in records}
    dataset = MarsJointDataset(metadata_dir, records, augment=False, seed=101)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    device = torch.device("cuda")
    model = MarsJointMILModel().to(device)
    payload = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()

    rows: list[dict[str, Any]] = []
    completed = 0
    with torch.inference_mode():
        for batch in loader:
            batch = move_batch(batch, device)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                outputs = model(
                    batch["target"], batch["reference"], batch["mbmp"], batch["observable"]
                )
            presence = torch.sigmoid(outputs["presence_logit"]).float().cpu().numpy()
            quality = torch.sigmoid(outputs["quality_logit"]).float().cpu().numpy()
            segmentation = (
                torch.sigmoid(outputs["segmentation_logits"]).float().cpu().numpy()[:, 0]
            )
            labels = batch["presence"].cpu().numpy().astype(np.uint8)
            masks = batch["mask"].cpu().numpy()[:, 0].astype(bool)
            observable = batch["observable"].cpu().numpy()[:, 0].astype(bool)
            mbmp = batch["mbmp"].cpu().numpy()[:, 0]
            for index, sample_id in enumerate(batch["sample_id"]):
                record = records_by_id[str(sample_id)]
                truth = masks[index] & observable[index]
                predicted_mask = connected_mask(
                    segmentation[index], observable[index], pixel_threshold, minimum_pixels
                )
                intersection = int(np.count_nonzero(predicted_mask & truth))
                union = int(np.count_nonzero(predicted_mask | truth))
                label = int(labels[index])
                score = float(presence[index])
                row = {
                    "sample_id": str(sample_id),
                    "group_id": str(record["group_id"]),
                    "country": str(record.get("country") or "unknown"),
                    "label": label,
                    "presence_score": score,
                    "quality_score": float(quality[index]),
                    "decision": decision_state(score, lower, upper),
                    "outcome": classify(label, score, lower, upper),
                    "observable_fraction": float(np.mean(observable[index])),
                    "cloud_fraction_metadata": float(record.get("cloud_fraction") or 0.0),
                    "truth_plume_pixels": int(np.count_nonzero(truth)),
                    "segmentation_top1pct_mean": top_fraction_mean(segmentation[index]),
                    "segmentation_max": float(np.max(segmentation[index])),
                    "segmentation_scene_detected": bool(np.any(predicted_mask)),
                    "segmentation_iou": 0.0 if union == 0 else intersection / union,
                    "mbmp_top1pct_mean": top_fraction_mean(mbmp[index]),
                    "mbmp_max": float(np.max(mbmp[index])),
                }
                rows.append(row)
            completed += len(batch["sample_id"])
            print(f"MIL-v2 validation audit: {completed}/{len(records)}", flush=True)

    write_predictions(predictions_path, rows)
    positives = [row for row in rows if row["label"] == 1]
    outcome_summaries = grouped_summaries(rows)
    report = {
        "schema_version": 1,
        "scope": "mil_v2_internal_validation_error_audit_only",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "dataset": "UNEP-IMEO/MARS-S2L",
            "dataset_revision": REVISION,
            "development_manifest_sha256": sha256(manifest),
            "checkpoint_sha256": sha256(checkpoint),
        },
        "cohort": {
            "role": "internal_validation",
            "samples": len(rows),
            "positives": sum(row["label"] for row in rows),
            "negatives": sum(row["label"] == 0 for row in rows),
            "groups": len({row["group_id"] for row in rows}),
            "strict_spatial_test_loaded": False,
        },
        "operating_rule": {
            "lower_no_plume_threshold": lower,
            "upper_plume_threshold": upper,
            "pixel_threshold": pixel_threshold,
            "minimum_connected_pixels": minimum_pixels,
        },
        "decision_counts": dict(Counter(row["decision"] for row in rows)),
        "outcome_summaries": outcome_summaries,
        "plume_size_strata": plume_size_strata(rows),
        "correlations_on_positives": {
            "presence_vs_truth_area": correlation(
                [row["presence_score"] for row in positives],
                [row["truth_plume_pixels"] for row in positives],
            ),
            "presence_vs_segmentation_top1pct": correlation(
                [row["presence_score"] for row in positives],
                [row["segmentation_top1pct_mean"] for row in positives],
            ),
        },
        "prediction_receipt": {
            "path": predictions_path.relative_to(root).as_posix(),
            "rows": len(rows),
            "sha256": sha256(predictions_path),
            "tracked": False,
        },
        "runtime": {
            "device": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "provenance": {
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True
            ).strip(),
            "git_tracked_worktree_dirty_at_start": tracked_dirty(root),
            "script": "tools/analyze_mars_joint_mil_validation.py",
            "script_sha256": sha256(Path(__file__)),
            "model_source_sha256": sha256(MODEL_ROOT / "mars_joint_mil_model.py"),
        },
        "decision": (
            "Use these validation-only error strata to define the expanded hard-negative/positive "
            "sampling plan. Do not modify a threshold or architecture from strict-test behavior."
        ),
    }
    write_json(output_json, report)
    write_markdown(output_markdown, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
