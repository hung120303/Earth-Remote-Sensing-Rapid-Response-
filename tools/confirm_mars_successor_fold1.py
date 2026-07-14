#!/usr/bin/env python3
"""Run the single sealed fold-1 confirmation for the frozen MARS successor."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import rasterio
import scipy
import sklearn
import torch
from torch.utils.data import DataLoader

from acquire_mars_metadata import DEFAULT_OUTPUT, repo_root, sha256
from evaluate_mars_residual_endpoint_blend import load_residual_model, trust_region_logits
from evaluate_released_marss2l import connected_scene_score
from extract_mars_scene_features import pooled_scene_features, tensor_feature_names
from train_mars_context_scene_ranker import augment_site_context
from train_mars_paper_residual import (
    DEFAULT_ACQUISITION_RECEIPT,
    DEFAULT_CHECKPOINT,
    DEFAULT_MANIFEST,
    DEFAULT_PROTOCOL,
    SENSOR_NAMES,
    MarsPaperDataset,
    add_pixels,
    finish_pixels,
    iter_development_manifest,
    move_batch,
    pixel_accumulator,
    verify_acquisition_receipt,
)
from train_mars_scene_ranker import blend_scores, metric_summary, predict_model

DEFAULT_RESIDUAL = Path("EarthRemoteSensingRapidResponse/artifacts/mars_paper_residual_fold1_seed606_epoch7.pt")
DEFAULT_RESIDUAL_SHA256 = "f6054d0fc8f17d661bce2a17b3947de0e6e566976730aa88f5bc1b6bed347e12"
DEFAULT_HEAD = Path("EarthRemoteSensingRapidResponse/artifacts/mars_oof_context_ranker_folds234.joblib")
DEFAULT_HEAD_SHA256 = "2d014f54918f68726d2ca4da19f35a1f29cb1b622fe7c32b56afc554ec27c370"
DEFAULT_SELECTION = Path("reports/experiments/mars_oof_context_minimum_blend.json")
DEFAULT_SELECTION_SHA256 = "8fc190bf4cac9d3abb24979cd20678930f09143302a69ddbfb944a7959951b0b"
DEFAULT_BASELINE = Path("reports/experiments/mars_paper_released_development.json")
DEFAULT_BASELINE_SHA256 = "4085d6e1e3683dfe4f73d25fd1bc0906a756b7433cc850ed64392db08f1f7935"
DEFAULT_JSON = Path("reports/experiments/mars_successor_fold1_confirmation.json")
DEFAULT_MARKDOWN = Path("reports/experiments/MARS_SUCCESSOR_FOLD1_CONFIRMATION.md")


def baseline_identity_values(metrics: dict[str, Any], pixels: dict[str, Any], pixels_by_sensor: dict[str, dict[str, Any]]) -> list[float]:
    values = [
        metrics["average_precision"],
        metrics["operating_point"]["recall"],
        metrics["operating_point"]["false_positive_rate"],
        pixels["intersection_over_union"],
    ]
    for name in SENSOR_NAMES:
        values.extend(
            [metrics["sensor_average_precision"][name], pixels_by_sensor[name]["intersection_over_union"]]
        )
    return [float(value) for value in values]


def report_identity_values(report: dict[str, Any]) -> list[float]:
    baseline = report["candidate"]
    values = [
        baseline["average_precision"],
        baseline["operating_points"]["0.0713"]["recall"],
        baseline["operating_points"]["0.0713"]["false_positive_rate"],
        baseline["pixel_fixed_0_5"]["intersection_over_union"],
    ]
    for name in SENSOR_NAMES:
        sensor = report["sensor_strata"][name]["candidate"]
        values.extend([sensor["average_precision"], sensor["pixel_fixed_0_5"]["intersection_over_union"]])
    return [float(value) for value in values]


def assert_released_identity(
    metrics: dict[str, Any],
    pixels: dict[str, Any],
    pixels_by_sensor: dict[str, dict[str, Any]],
    report: dict[str, Any],
) -> None:
    actual = baseline_identity_values(metrics, pixels, pixels_by_sensor)
    expected = report_identity_values(report)
    if actual != expected:
        labels = [
            "overall_ap",
            "overall_recall",
            "overall_fpr",
            "overall_iou",
            "sentinel2_ap",
            "sentinel2_iou",
            "landsat_ap",
            "landsat_iou",
        ]
        differences = {
            label: {"actual": observed, "expected": reference}
            for label, observed, reference in zip(labels, actual, expected)
            if observed != reference
        }
        raise RuntimeError(
            "Fold-1 released outputs do not reproduce the frozen baseline: "
            + json.dumps(differences, sort_keys=True)
        )


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    result = report["result"]
    baseline = result["released_baseline"]
    candidate = result["candidate"]
    lines = [
        "# Independent fold-1 confirmation",
        "",
        "The residual was trained without fold-1 labels; this report is the single sealed confirmation read.",
        "",
        "| Model | AP | Recall at <=7.13% FPR | FPR | Pixel IoU |",
        "|---|---:|---:|---:|---:|",
        f"| Released MARS-S2L | {baseline['average_precision']:.5f} | {baseline['recall']:.5f} | {baseline['false_positive_rate']:.5f} | {baseline['pixel_iou']:.5f} |",
        f"| Frozen successor | {candidate['average_precision']:.5f} | {candidate['recall']:.5f} | {candidate['false_positive_rate']:.5f} | {candidate['pixel_iou']:.5f} |",
        "",
        "All confirmation gates pass." if all(result["checks"].values()) else "Confirmation failed.",
        "",
        report["decision"],
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@torch.no_grad()
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-dir", default=DEFAULT_OUTPUT.as_posix())
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST.as_posix())
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL.as_posix())
    parser.add_argument("--acquisition-receipt", default=DEFAULT_ACQUISITION_RECEIPT.as_posix())
    parser.add_argument("--released-checkpoint", default=DEFAULT_CHECKPOINT.as_posix())
    parser.add_argument("--residual", default=DEFAULT_RESIDUAL.as_posix())
    parser.add_argument("--residual-sha256", default=DEFAULT_RESIDUAL_SHA256)
    parser.add_argument("--head", default=DEFAULT_HEAD.as_posix())
    parser.add_argument("--head-sha256", default=DEFAULT_HEAD_SHA256)
    parser.add_argument("--selection", default=DEFAULT_SELECTION.as_posix())
    parser.add_argument("--selection-sha256", default=DEFAULT_SELECTION_SHA256)
    parser.add_argument("--baseline", default=DEFAULT_BASELINE.as_posix())
    parser.add_argument("--baseline-sha256", default=DEFAULT_BASELINE_SHA256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output-json", default=DEFAULT_JSON.as_posix())
    parser.add_argument("--output-markdown", default=DEFAULT_MARKDOWN.as_posix())
    args = parser.parse_args()
    root = repo_root()
    paths = {
        "residual": (root / args.residual).resolve(),
        "head": (root / args.head).resolve(),
        "selection": (root / args.selection).resolve(),
        "baseline": (root / args.baseline).resolve(),
    }
    for key, expected in (
        ("residual", args.residual_sha256),
        ("head", args.head_sha256),
        ("selection", args.selection_sha256),
        ("baseline", args.baseline_sha256),
    ):
        if sha256(paths[key]) != expected:
            raise ValueError(f"Frozen {key} hash mismatch")
    manifest = (root / args.manifest).resolve()
    protocol_path = (root / args.protocol).resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    manifest_hash = sha256(manifest)
    if manifest_hash != protocol["development_manifest_sha256"]:
        raise ValueError("Development manifest differs from frozen protocol")
    verify_acquisition_receipt((root / args.acquisition_receipt).resolve(), manifest_hash)
    group_to_fold = {str(item["group_id"]): int(item["fold"]) for item in protocol["assignments"]}
    records = [
        record for record in iter_development_manifest(manifest)
        if group_to_fold[str(record["group_id"])] == 1
    ]
    residual_artifact = torch.load(paths["residual"], map_location="cpu", weights_only=True)
    if (
        int(residual_artifact["fold"]) != 1
        or int(residual_artifact["epoch"]) != 7
        or residual_artifact["protocol_sha256"] != sha256(protocol_path)
        or residual_artifact["validation"] != {"deferred_confirmation": True, "validation_reads_during_training": 0}
    ):
        raise ValueError("Residual artifact violates the sealed fold-1 contract")
    head_payload = joblib.load(paths["head"])
    selection = json.loads(paths["selection"].read_text(encoding="utf-8"))
    baseline_report = json.loads(paths["baseline"].read_text(encoding="utf-8"))["folds"]["1"]
    if head_payload["spec"] != selection["frozen_spec"] or float(selection["selected"]["blend_lambda"]) != 0.25:
        raise ValueError("Frozen scene-head selection mismatch")

    loader = DataLoader(
        MarsPaperDataset((root / args.metadata_dir).resolve(), records, augment=False, seed=0),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_residual_model((root / args.released_checkpoint).resolve(), residual_artifact, device)
    feature_rows: list[np.ndarray] = []
    labels: list[int] = []
    sensors: list[int] = []
    groups: list[str] = []
    released_scores: list[float] = []
    primary_scores: list[float] = []
    released_pixels = pixel_accumulator()
    primary_pixels = pixel_accumulator()
    released_pixels_by_sensor = {name: pixel_accumulator() for name in SENSOR_NAMES}
    primary_pixels_by_sensor = {name: pixel_accumulator() for name in SENSOR_NAMES}
    for batch in loader:
        batch = move_batch(batch, device)
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            output = model(batch["inputs"], batch["observable"], batch["sensor_index"])
        primary_logits = trust_region_logits(output["baseline_logits"], output["segmentation_logits"], 0.5)
        pooled = pooled_scene_features(
            batch["inputs"], primary_logits, output["baseline_logits"], batch["clear"], batch["observable"]
        ).cpu().numpy()
        primary_probability = torch.sigmoid(primary_logits).float().masked_fill(batch["clear"] <= 0.5, 0.0).cpu().numpy()
        released_probability = torch.sigmoid(output["baseline_logits"]).float().masked_fill(batch["clear"] <= 0.5, 0.0).cpu().numpy()
        for index in range(primary_probability.shape[0]):
            sensor_index = int(batch["sensor_index"][index].item())
            sensor_name = SENSOR_NAMES[sensor_index]
            observable = batch["observable"][index, 0].cpu().numpy() > 0.5
            truth = (batch["mask"][index, 0].cpu().numpy() > 0.5) & observable
            primary_score = connected_scene_score(primary_probability[index, 0])
            released_score = connected_scene_score(released_probability[index, 0])
            feature_rows.append(
                np.concatenate((np.asarray([primary_score, released_score], dtype=np.float32), pooled[index])).astype(np.float32)
            )
            labels.append(int(batch["presence"][index].item()))
            sensors.append(sensor_index)
            groups.append(str(batch["group_id"][index]))
            primary_scores.append(primary_score)
            released_scores.append(released_score)
            add_pixels(primary_pixels, primary_probability[index, 0], truth, observable)
            add_pixels(released_pixels, released_probability[index, 0], truth, observable)
            add_pixels(primary_pixels_by_sensor[sensor_name], primary_probability[index, 0], truth, observable)
            add_pixels(released_pixels_by_sensor[sensor_name], released_probability[index, 0], truth, observable)

    y = np.asarray(labels, dtype=np.uint8)
    sensor_array = np.asarray(sensors, dtype=np.uint8)
    base_names = np.asarray(["primary_connected_score", "released_connected_score", *tensor_feature_names()])
    base_features = np.stack(feature_rows).astype(np.float64)
    released_metrics = metric_summary(y, np.asarray(released_scores), sensor_array)
    released_pixel_summary = finish_pixels(released_pixels)
    released_pixel_sensor = {name: finish_pixels(value) for name, value in released_pixels_by_sensor.items()}
    assert_released_identity(released_metrics, released_pixel_summary, released_pixel_sensor, baseline_report)
    context_features, augmented_names = augment_site_context(base_features, base_names, np.asarray(groups))
    if base_names.tolist() != head_payload["feature_names"] or augmented_names != head_payload["augmented_feature_names"]:
        raise ValueError("Fold-1 feature schema differs from the frozen scene head")
    head_probability = predict_model(head_payload["fitted"], context_features)
    final_scores = blend_scores(np.asarray(primary_scores), head_probability, 0.25)
    candidate_metrics = metric_summary(y, final_scores, sensor_array)
    candidate_pixels = finish_pixels(primary_pixels)
    candidate_pixel_sensor = {name: finish_pixels(value) for name, value in primary_pixels_by_sensor.items()}

    baseline = baseline_report["candidate"]
    deltas = {
        "average_precision": float(candidate_metrics["average_precision"] - baseline["average_precision"]),
        "recall_at_fpr_0_0713": float(candidate_metrics["operating_point"]["recall"] - baseline["operating_points"]["0.0713"]["recall"]),
        "false_positive_rate_at_target": float(candidate_metrics["operating_point"]["false_positive_rate"] - baseline["operating_points"]["0.0713"]["false_positive_rate"]),
        "pixel_iou": float(candidate_pixels["intersection_over_union"] - baseline["pixel_fixed_0_5"]["intersection_over_union"]),
    }
    sensor_strata: dict[str, Any] = {}
    for name in SENSOR_NAMES:
        released_sensor = baseline_report["sensor_strata"][name]["candidate"]
        sensor_strata[name] = {
            "average_precision": candidate_metrics["sensor_average_precision"][name],
            "pixel_iou": candidate_pixel_sensor[name]["intersection_over_union"],
            "delta": {
                "average_precision": float(candidate_metrics["sensor_average_precision"][name] - released_sensor["average_precision"]),
                "pixel_iou": float(candidate_pixel_sensor[name]["intersection_over_union"] - released_sensor["pixel_fixed_0_5"]["intersection_over_union"]),
            },
        }
    checks = {
        "ap_higher": deltas["average_precision"] > 0,
        "recall_at_fpr_0_0713_higher": deltas["recall_at_fpr_0_0713"] > 0,
        "fpr_no_worse": deltas["false_positive_rate_at_target"] <= 0,
        "pixel_iou_higher": deltas["pixel_iou"] > 0,
        "no_material_sensor_regression": all(
            value["delta"]["average_precision"] >= -0.01 and value["delta"]["pixel_iou"] >= -0.01
            for value in sensor_strata.values()
        ),
    }
    result = {
        "released_baseline": {
            "average_precision": baseline["average_precision"],
            "recall": baseline["operating_points"]["0.0713"]["recall"],
            "false_positive_rate": baseline["operating_points"]["0.0713"]["false_positive_rate"],
            "pixel_iou": baseline["pixel_fixed_0_5"]["intersection_over_union"],
        },
        "candidate": {
            "average_precision": candidate_metrics["average_precision"],
            "recall": candidate_metrics["operating_point"]["recall"],
            "false_positive_rate": candidate_metrics["operating_point"]["false_positive_rate"],
            "threshold": candidate_metrics["operating_point"]["threshold"],
            "pixel_iou": candidate_pixels["intersection_over_union"],
        },
        "delta": deltas,
        "sensor_strata": sensor_strata,
        "checks": checks,
    }
    passed = all(checks.values())
    decision = (
        "Independent fold-1 confirmation passed; authorize final ensemble/protocol freeze before the paper test."
        if passed else "Independent fold-1 confirmation failed; do not open the paper test."
    )
    report = {
        "schema_version": 1,
        "scope": "single sealed independent fold-1 confirmation; paper test not loaded",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows": int(y.size), "positive": int(np.count_nonzero(y == 1)),
        "sites": len(set(groups)),
        "architecture": {
            "residual_fold": 1, "residual_epoch": 7, "residual_alpha": 0.5,
            "scene_head": head_payload["architecture"], "spec": head_payload["spec"],
            "scene_blend": 0.25,
        },
        "released_identity": {
            "metrics": released_metrics,
            "pixels": released_pixel_summary,
            "pixels_by_sensor": released_pixel_sensor,
        },
        "result": result, "decision": decision,
        "provenance": {
            "residual_sha256": args.residual_sha256,
            "head_sha256": args.head_sha256,
            "selection_sha256": args.selection_sha256,
            "baseline_sha256": args.baseline_sha256,
            "manifest_sha256": manifest_hash,
            "protocol_sha256": sha256(protocol_path),
            "script_sha256": sha256(Path(__file__).resolve()),
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
            "device": str(torch.cuda.get_device_name(device) if device.type == "cuda" else device),
            "torch": torch.__version__, "numpy": np.__version__, "scipy": scipy.__version__,
            "sklearn": sklearn.__version__, "rasterio": rasterio.__version__, "joblib": joblib.__version__,
        },
    }
    output_json = (root / args.output_json).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown((root / args.output_markdown).resolve(), report)
    print(json.dumps({"ok": passed, "checks": checks, "decision": decision}, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
