#!/usr/bin/env python3
"""Evaluate a frozen blend between two retained MARS residual endpoints."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import scipy
import sklearn
import torch
from torch.utils.data import DataLoader

from acquire_mars_metadata import DEFAULT_OUTPUT, repo_root, sha256
from evaluate_mars_residual_trust_region import (
    alpha_key,
    balanced_rank,
    promotion_checks,
    summarize_candidate,
    write_json,
)
from evaluate_released_marss2l import component_mask, connected_scene_score
from train_mars_paper_residual import (
    DEFAULT_ACQUISITION_RECEIPT,
    DEFAULT_CHECKPOINT,
    DEFAULT_MANIFEST,
    DEFAULT_PROTOCOL,
    SENSOR_NAMES,
    MarsPaperDataset,
    MarsPaperResidualModel,
    add_pixels,
    finish_pixels,
    iter_development_manifest,
    move_batch,
    pixel_accumulator,
    verify_acquisition_receipt,
)
from train_mars_source_aligned_residual import contract_residual_strength

DEFAULT_PRIMARY_ARTIFACT = Path(
    "EarthRemoteSensingRapidResponse/artifacts/mars_paper_residual_fold0_seed606.pt"
)
DEFAULT_PRIMARY_SHA256 = (
    "b94880d858e1e7791591eeb5f7d0da9be84b99a324e980437ebe83cfae6c7d49"
)
DEFAULT_SOURCE_ARTIFACT = Path(
    "EarthRemoteSensingRapidResponse/artifacts/mars_source_aligned_fold0_seed707.pt"
)
DEFAULT_SOURCE_SHA256 = (
    "8da4abe2bdbbbe3f3b8ca9ab189c59c701f57a8186c4d89bdf2337c11e551629"
)
DEFAULT_PRIMARY_REPORT = Path(
    "reports/experiments/mars_paper_residual_fold0_trust_region.json"
)
DEFAULT_PRIMARY_REPORT_SHA256 = (
    "bb9763a9bdbf0ddd14c3e1c718af9bceff47e0a1c6f04a4749833968368b79b5"
)
DEFAULT_BETAS = (
    0.0,
    0.015625,
    0.03125,
    0.046875,
    0.0625,
    0.09375,
    0.125,
    0.1875,
    0.25,
    0.375,
    0.5,
    0.625,
    0.75,
    0.875,
    1.0,
)
DEFAULT_JSON = Path("reports/experiments/mars_residual_endpoint_blend_fold0.json")
DEFAULT_MARKDOWN = Path("reports/experiments/MARS_RESIDUAL_ENDPOINT_BLEND_FOLD0.md")


def select_beta(summaries: dict[str, dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    candidates = [
        (key, value)
        for key, value in summaries.items()
        if 0.0 < float(key) < 1.0
    ]
    if not candidates:
        raise ValueError("Endpoint blend requires at least one interior beta")
    passing = [
        item for item in candidates if all(item[1]["promotion_checks"].values())
    ]
    return max(passing or candidates, key=lambda item: balanced_rank(item[1]))


def endpoint_values(summary: dict[str, Any]) -> list[float]:
    values = [
        summary["candidate"]["average_precision"],
        summary["candidate"]["operating_points"]["0.0713"]["recall"],
        summary["candidate"]["pixel_fixed_0_5"]["intersection_over_union"],
    ]
    for sensor_name in SENSOR_NAMES:
        sensor = summary["sensor_strata"][sensor_name]["candidate"]
        values.extend(
            [
                sensor["average_precision"],
                sensor["pixel_fixed_0_5"]["intersection_over_union"],
            ]
        )
    return [float(value) for value in values]


def assert_endpoint_identity(
    actual: dict[str, Any], expected: dict[str, Any], name: str
) -> None:
    if endpoint_values(actual) != endpoint_values(expected):
        raise RuntimeError(f"{name} endpoint does not reproduce its frozen metrics")


def load_residual_model(
    checkpoint: Path,
    artifact: dict[str, Any],
    device: torch.device,
) -> MarsPaperResidualModel:
    model = MarsPaperResidualModel().to(device)
    model.load_released_checkpoint(checkpoint)
    model.correction.load_state_dict(artifact["correction_state_dict"])
    model.sensor_log_scale.copy_(artifact["sensor_log_scale"].to(device))
    model.sensor_bias.copy_(artifact["sensor_bias"].to(device))
    return model.eval()


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Retained MARS residual endpoint blend on fold 0",
        "",
        "Development-only architecture selection; fold 1 and the paper test were not loaded.",
        "",
        "| Beta | AP delta | Recall delta at <=7.13% FPR | IoU delta | Passes all gates |",
        "|---:|---:|---:|---:|:---:|",
    ]
    for key, summary in report["betas"].items():
        lines.append(
            f"| {float(key):.6f} | {summary['delta']['average_precision']:+.5f} | "
            f"{summary['delta']['recall_at_fpr_0_0713']:+.5f} | "
            f"{summary['delta']['pixel_iou']:+.5f} | "
            f"{'yes' if all(summary['promotion_checks'].values()) else 'no'} |"
        )
    lines.extend(
        [
            "",
            f"Selected beta: **{float(report['selected_beta']):.6f}**.",
            "",
            report["decision"],
        ]
    )
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
    parser.add_argument("--primary-artifact", default=DEFAULT_PRIMARY_ARTIFACT.as_posix())
    parser.add_argument("--primary-sha256", default=DEFAULT_PRIMARY_SHA256)
    parser.add_argument("--source-artifact", default=DEFAULT_SOURCE_ARTIFACT.as_posix())
    parser.add_argument("--source-sha256", default=DEFAULT_SOURCE_SHA256)
    parser.add_argument("--primary-report", default=DEFAULT_PRIMARY_REPORT.as_posix())
    parser.add_argument("--primary-report-sha256", default=DEFAULT_PRIMARY_REPORT_SHA256)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--betas", type=float, nargs="+", default=list(DEFAULT_BETAS))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output-json", default=DEFAULT_JSON.as_posix())
    parser.add_argument("--output-markdown", default=DEFAULT_MARKDOWN.as_posix())
    args = parser.parse_args()
    betas = tuple(float(value) for value in args.betas)
    if (
        len(set(betas)) != len(betas)
        or 0.0 not in betas
        or 1.0 not in betas
        or any(not 0.0 <= value <= 1.0 for value in betas)
    ):
        parser.error("betas must be unique in [0,1] and include both endpoints")

    root = repo_root()
    metadata_dir = (root / args.metadata_dir).resolve()
    manifest = (root / args.manifest).resolve()
    protocol_path = (root / args.protocol).resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    manifest_hash = sha256(manifest)
    if manifest_hash != protocol["development_manifest_sha256"]:
        raise ValueError("Development manifest differs from the frozen protocol")
    verify_acquisition_receipt((root / args.acquisition_receipt).resolve(), manifest_hash)

    primary_path = (root / args.primary_artifact).resolve()
    source_path = (root / args.source_artifact).resolve()
    primary_report_path = (root / args.primary_report).resolve()
    for path, expected, label in (
        (primary_path, args.primary_sha256, "Primary artifact"),
        (source_path, args.source_sha256, "Source artifact"),
        (primary_report_path, args.primary_report_sha256, "Primary report"),
    ):
        if sha256(path) != expected:
            raise ValueError(f"{label} hash mismatch")
    primary_artifact = torch.load(primary_path, map_location="cpu", weights_only=True)
    source_artifact = torch.load(source_path, map_location="cpu", weights_only=True)
    protocol_hash = sha256(protocol_path)
    for artifact in (primary_artifact, source_artifact):
        if int(artifact["fold"]) != args.fold or artifact["protocol_sha256"] != protocol_hash:
            raise ValueError("Artifact covers a different fold or protocol")

    group_to_fold = {
        str(item["group_id"]): int(item["fold"])
        for item in protocol["assignments"]
    }
    records = list(iter_development_manifest(manifest))
    held_out = [
        record for record in records
        if group_to_fold[str(record["group_id"])] == args.fold
    ]
    loader = DataLoader(
        MarsPaperDataset(metadata_dir, held_out, augment=False, seed=0),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = (root / args.released_checkpoint).resolve()
    primary_model = load_residual_model(checkpoint, primary_artifact, device)
    contract_residual_strength(primary_model, 0.5)
    source_model = load_residual_model(checkpoint, source_artifact, device)

    labels: list[int] = []
    groups: list[str] = []
    sensor_indices: list[int] = []
    baseline_state = {
        "scores": [], "predictions": [], "pixels": pixel_accumulator(),
        "pixels_by_sensor": {name: pixel_accumulator() for name in SENSOR_NAMES},
    }
    states = {
        alpha_key(beta): {
            "scores": [], "predictions": [], "pixels": pixel_accumulator(),
            "pixels_by_sensor": {name: pixel_accumulator() for name in SENSOR_NAMES},
        }
        for beta in betas
    }

    for batch in loader:
        batch = move_batch(batch, device)
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            primary_output = primary_model(
                batch["inputs"], batch["observable"], batch["sensor_index"]
            )
            source_output = source_model(
                batch["inputs"], batch["observable"], batch["sensor_index"]
            )
        baseline_logits = primary_output["baseline_logits"]
        primary_logits = primary_output["segmentation_logits"]
        source_logits = source_output["segmentation_logits"]
        primary_float = primary_logits.float()
        endpoint_delta = source_logits.float() - primary_float
        clear = batch["clear"] > 0.5
        for index in range(baseline_logits.shape[0]):
            label = int(batch["presence"][index].item())
            sensor_index = int(batch["sensor_index"][index].item())
            observable = batch["observable"][index, 0].cpu().numpy() > 0.5
            truth = (batch["mask"][index, 0].cpu().numpy() > 0.5) & observable
            labels.append(label)
            groups.append(str(batch["group_id"][index]))
            sensor_indices.append(sensor_index)
            candidates: list[tuple[dict[str, Any], torch.Tensor]] = [
                (baseline_state, baseline_logits[index, 0])
            ]
            for beta in betas:
                if beta == 0.0:
                    logits = primary_logits[index, 0]
                elif beta == 1.0:
                    logits = source_logits[index, 0]
                else:
                    logits = (
                        primary_float[index, 0] + beta * endpoint_delta[index, 0]
                    ).to(primary_logits.dtype)
                candidates.append((states[alpha_key(beta)], logits))
            for state, logits in candidates:
                score = torch.sigmoid(logits).float().masked_fill(
                    ~clear[index, 0], 0.0
                ).cpu().numpy()
                state["scores"].append(connected_scene_score(score))
                state["predictions"].append(bool(np.any(component_mask(score))))
                add_pixels(state["pixels"], score, truth, observable)
                add_pixels(
                    state["pixels_by_sensor"][SENSOR_NAMES[sensor_index]],
                    score, truth, observable,
                )

    y = np.asarray(labels, dtype=np.uint8)
    sensors = np.asarray(sensor_indices, dtype=np.uint8)
    baseline_metrics = summarize_candidate(
        labels=y, sensor_indices=sensors, scores=baseline_state["scores"],
        predictions=baseline_state["predictions"], pixels=baseline_state["pixels"],
        pixels_by_sensor=baseline_state["pixels_by_sensor"], baseline=None,
    )
    baseline_reference = {
        "candidate": baseline_metrics["candidate"],
        "sensor_strata": {
            name: {"candidate": value["candidate"]}
            for name, value in baseline_metrics["sensor_strata"].items()
        },
    }
    summaries: dict[str, dict[str, Any]] = {}
    for beta in betas:
        key = alpha_key(beta)
        state = states[key]
        summary = summarize_candidate(
            labels=y, sensor_indices=sensors, scores=state["scores"],
            predictions=state["predictions"], pixels=state["pixels"],
            pixels_by_sensor=state["pixels_by_sensor"], baseline=baseline_reference,
        )
        summary["balanced_rank"] = list(balanced_rank(summary))
        summary["promotion_checks"] = promotion_checks(summary)
        summaries[key] = summary

    primary_report = json.loads(primary_report_path.read_text(encoding="utf-8"))
    assert_endpoint_identity(summaries[alpha_key(0.0)], primary_report["alphas"]["0.5"], "Primary")
    assert_endpoint_identity(summaries[alpha_key(1.0)], source_artifact["validation"], "Source")
    selected_key, selected = select_beta(summaries)
    passed = all(selected["promotion_checks"].values())
    decision = (
        "Advance the frozen endpoint blend to independent fold-1 confirmation."
        if passed else "Reject the retained endpoint blend on fold 0."
    )
    report = {
        "schema_version": 1,
        "scope": "fold-0 development architecture selection; fold 1 and paper test not loaded",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows": int(y.size), "positive": int(np.count_nonzero(y == 1)),
        "sites": len(set(groups)), "betas": summaries,
        "selected_beta": float(selected_key),
        "selected_checks": selected["promotion_checks"], "decision": decision,
        "provenance": {
            "primary_artifact_sha256": args.primary_sha256,
            "source_artifact_sha256": args.source_sha256,
            "primary_report_sha256": args.primary_report_sha256,
            "development_manifest_sha256": manifest_hash,
            "protocol_sha256": protocol_hash,
            "script_sha256": sha256(Path(__file__).resolve()),
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True
            ).strip(),
            "device": str(torch.cuda.get_device_name(device) if device.type == "cuda" else device),
            "torch": torch.__version__, "numpy": np.__version__,
            "scipy": scipy.__version__, "sklearn": sklearn.__version__,
            "rasterio": rasterio.__version__,
        },
    }
    write_json((root / args.output_json).resolve(), report)
    write_markdown((root / args.output_markdown).resolve(), report)
    print(json.dumps({
        "ok": True, "selected_beta": float(selected_key),
        "checks": selected["promotion_checks"], "decision": decision,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
