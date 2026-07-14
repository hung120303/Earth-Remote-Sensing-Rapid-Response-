#!/usr/bin/env python3
"""Tune a frozen MARS residual's correction strength on one development fold."""

from __future__ import annotations

import argparse
import json
import os
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
from evaluate_released_marss2l import component_mask, connected_scene_score, scene_metrics
from train_mars_paper_residual import (
    DEFAULT_ACQUISITION_RECEIPT,
    DEFAULT_CHECKPOINT,
    DEFAULT_MANIFEST,
    DEFAULT_PROTOCOL,
    SENSOR_NAMES,
    TARGET_FPRS,
    MarsPaperDataset,
    MarsPaperResidualModel,
    add_pixels,
    choose_threshold_at_fpr,
    finish_pixels,
    iter_development_manifest,
    move_batch,
    pixel_accumulator,
    verify_acquisition_receipt,
)

DEFAULT_ARTIFACT = Path(
    "EarthRemoteSensingRapidResponse/artifacts/mars_paper_residual_fold0_seed606.pt"
)
DEFAULT_ARTIFACT_SHA256 = (
    "b94880d858e1e7791591eeb5f7d0da9be84b99a324e980437ebe83cfae6c7d49"
)
DEFAULT_ALPHAS = (0.0, 0.03125, 0.0625, 0.125, 0.25, 0.375, 0.5, 0.75, 1.0)
DEFAULT_JSON = Path("reports/experiments/mars_paper_residual_fold0_trust_region.json")
DEFAULT_MARKDOWN = Path(
    "reports/experiments/MARS_PAPER_RESIDUAL_FOLD0_TRUST_REGION.md"
)


def alpha_key(alpha: float) -> str:
    return format(alpha, ".8g")


def balanced_rank(summary: dict[str, Any]) -> tuple[float, float, float]:
    ap_delta = float(summary["delta"]["average_precision"])
    iou_delta = float(summary["delta"]["pixel_iou"])
    return (
        min(ap_delta, iou_delta),
        ap_delta + iou_delta,
        float(summary["delta"]["recall_at_fpr_0_0713"]),
    )


def promotion_checks(summary: dict[str, Any]) -> dict[str, bool]:
    return {
        "ap_higher": summary["delta"]["average_precision"] > 0,
        "pixel_iou_higher": summary["delta"]["pixel_iou"] > 0,
        "recall_at_fpr_0_0713_higher": (
            summary["delta"]["recall_at_fpr_0_0713"] > 0
        ),
        "no_material_sensor_regression": all(
            stratum["eligible_for_promotion"]
            and stratum["delta"]["average_precision"] >= -0.01
            and stratum["delta"]["pixel_iou"] >= -0.01
            for stratum in summary["sensor_strata"].values()
        ),
    }


def select_alpha(summaries: dict[str, dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    candidates = [(key, value) for key, value in summaries.items() if float(key) > 0]
    if not candidates:
        raise ValueError("Trust-region selection requires at least one positive alpha")
    return max(candidates, key=lambda item: balanced_rank(item[1]))


def assert_matches_artifact_baseline(
    summary: dict[str, Any], artifact_validation: dict[str, Any]
) -> None:
    expected = artifact_validation["released_baseline"]
    pairs = [
        (summary["candidate"]["average_precision"], expected["average_precision"]),
        (
            summary["candidate"]["pixel_fixed_0_5"]["intersection_over_union"],
            expected["pixel_fixed_0_5"]["intersection_over_union"],
        ),
        (
            summary["candidate"]["operating_points"]["0.0713"]["recall"],
            expected["operating_points"]["0.0713"]["recall"],
        ),
    ]
    for sensor_name in SENSOR_NAMES:
        actual_sensor = summary["sensor_strata"][sensor_name]["candidate"]
        expected_sensor = artifact_validation["sensor_strata"][sensor_name][
            "released_baseline"
        ]
        pairs.extend(
            [
                (actual_sensor["average_precision"], expected_sensor["average_precision"]),
                (
                    actual_sensor["pixel_fixed_0_5"]["intersection_over_union"],
                    expected_sensor["pixel_fixed_0_5"]["intersection_over_union"],
                ),
            ]
        )
    if any(float(actual) != float(expected_value) for actual, expected_value in pairs):
        raise RuntimeError("Alpha zero does not match the artifact's released baseline")


def summarize_candidate(
    *,
    labels: np.ndarray,
    sensor_indices: np.ndarray,
    scores: list[float],
    predictions: list[bool],
    pixels: dict[str, float],
    pixels_by_sensor: dict[str, dict[str, float]],
    baseline: dict[str, Any] | None,
) -> dict[str, Any]:
    def summarize(
        local_scores: list[float],
        local_predictions: list[bool],
        local_pixels: dict[str, float],
        selection: np.ndarray | None = None,
    ) -> dict[str, Any]:
        score_array = np.asarray(local_scores, dtype=np.float32)
        prediction_array = np.asarray(local_predictions, dtype=bool)
        local_y = labels if selection is None else labels[selection]
        selected_scores = score_array if selection is None else score_array[selection]
        selected_predictions = (
            prediction_array if selection is None else prediction_array[selection]
        )
        result = scene_metrics(local_y, selected_predictions, selected_scores)
        result["operating_points"] = {
            str(target): choose_threshold_at_fpr(local_y, selected_scores, target)
            for target in TARGET_FPRS
        }
        result["pixel_fixed_0_5"] = finish_pixels(local_pixels)
        return result

    candidate = summarize(scores, predictions, pixels)
    sensor_strata: dict[str, Any] = {}
    for sensor_index, sensor_name in enumerate(SENSOR_NAMES):
        selection = sensor_indices == sensor_index
        local_y = labels[selection]
        if local_y.size == 0 or np.unique(local_y).size < 2:
            sensor_strata[sensor_name] = {
                "rows": int(local_y.size),
                "positive": int(np.count_nonzero(local_y == 1)),
                "eligible_for_promotion": False,
                "reason": "A sensor stratum needs both plume and no-plume scenes.",
            }
            continue
        local_candidate = summarize(
            scores, predictions, pixels_by_sensor[sensor_name], selection
        )
        if baseline is None:
            sensor_strata[sensor_name] = {
                "rows": int(local_y.size),
                "positive": int(np.count_nonzero(local_y == 1)),
                "eligible_for_promotion": True,
                "candidate": local_candidate,
            }
            continue
        local_baseline = baseline["sensor_strata"][sensor_name]["candidate"]
        sensor_strata[sensor_name] = {
            "rows": int(local_y.size),
            "positive": int(np.count_nonzero(local_y == 1)),
            "eligible_for_promotion": True,
            "candidate": local_candidate,
            "released_baseline": local_baseline,
            "delta": {
                "average_precision": (
                    local_candidate["average_precision"]
                    - local_baseline["average_precision"]
                ),
                "pixel_iou": (
                    local_candidate["pixel_fixed_0_5"]["intersection_over_union"]
                    - local_baseline["pixel_fixed_0_5"]["intersection_over_union"]
                ),
            },
        }
    if baseline is None:
        return {"candidate": candidate, "sensor_strata": sensor_strata}
    released = baseline["candidate"]
    return {
        "candidate": candidate,
        "released_baseline": released,
        "sensor_strata": sensor_strata,
        "delta": {
            "average_precision": candidate["average_precision"]
            - released["average_precision"],
            "pixel_iou": candidate["pixel_fixed_0_5"]["intersection_over_union"]
            - released["pixel_fixed_0_5"]["intersection_over_union"],
            "recall_at_fpr_0_0713": (
                candidate["operating_points"]["0.0713"]["recall"]
                - released["operating_points"]["0.0713"]["recall"]
            ),
        },
    }


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# MARS residual fold-0 trust region",
        "",
        "Development-only architecture selection; the paper test and fold 1 were not loaded.",
        "",
        "| Alpha | AP delta | Recall delta at <=7.13% FPR | IoU delta | Worst primary delta |",
        "|---:|---:|---:|---:|---:|",
    ]
    for key, summary in report["alphas"].items():
        rank = summary["balanced_rank"]
        lines.append(
            f"| {float(key):.5f} | {summary['delta']['average_precision']:+.5f} | "
            f"{summary['delta']['recall_at_fpr_0_0713']:+.5f} | "
            f"{summary['delta']['pixel_iou']:+.5f} | {rank[0]:+.5f} |"
        )
    lines.extend(
        [
            "",
            f"Selected alpha: **{float(report['selected_alpha']):.5f}**.",
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
    parser.add_argument("--artifact", default=DEFAULT_ARTIFACT.as_posix())
    parser.add_argument("--artifact-sha256", default=DEFAULT_ARTIFACT_SHA256)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--alphas", type=float, nargs="+", default=list(DEFAULT_ALPHAS))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output-json", default=DEFAULT_JSON.as_posix())
    parser.add_argument("--output-markdown", default=DEFAULT_MARKDOWN.as_posix())
    args = parser.parse_args()
    if args.batch_size <= 0 or args.workers < 0:
        parser.error("batch size must be positive and workers non-negative")
    alphas = tuple(float(value) for value in args.alphas)
    if len(set(alphas)) != len(alphas) or 0.0 not in alphas or any(
        not 0.0 <= value <= 1.0 for value in alphas
    ):
        parser.error("alphas must be unique in [0,1] and include zero")

    root = repo_root()
    metadata_dir = (root / args.metadata_dir).resolve()
    manifest = (root / args.manifest).resolve()
    protocol_path = (root / args.protocol).resolve()
    artifact_path = (root / args.artifact).resolve()
    if sha256(artifact_path) != args.artifact_sha256:
        raise ValueError("Residual artifact hash mismatch")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    manifest_hash = sha256(manifest)
    if manifest_hash != protocol["development_manifest_sha256"]:
        raise ValueError("Development manifest differs from the frozen protocol")
    verify_acquisition_receipt((root / args.acquisition_receipt).resolve(), manifest_hash)
    group_to_fold = {
        str(item["group_id"]): int(item["fold"])
        for item in protocol["assignments"]
    }
    records = list(iter_development_manifest(manifest))
    held_out = [
        record
        for record in records
        if group_to_fold[str(record["group_id"])] == args.fold
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MarsPaperResidualModel().to(device)
    model.load_released_checkpoint((root / args.released_checkpoint).resolve())
    artifact = torch.load(artifact_path, map_location="cpu", weights_only=True)
    if int(artifact["fold"]) != args.fold:
        raise ValueError("Residual artifact covers a different held-out fold")
    if artifact["protocol_sha256"] != sha256(protocol_path):
        raise ValueError("Residual artifact covers a different fold protocol")
    model.correction.load_state_dict(artifact["correction_state_dict"])
    model.sensor_log_scale.copy_(artifact["sensor_log_scale"].to(device))
    model.sensor_bias.copy_(artifact["sensor_bias"].to(device))
    model.eval()

    loader = DataLoader(
        MarsPaperDataset(metadata_dir, held_out, augment=False, seed=0),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    labels: list[int] = []
    groups: list[str] = []
    sensor_indices: list[int] = []
    states = {
        alpha_key(alpha): {
            "scores": [],
            "predictions": [],
            "pixels": pixel_accumulator(),
            "pixels_by_sensor": {name: pixel_accumulator() for name in SENSOR_NAMES},
        }
        for alpha in alphas
    }
    for batch in loader:
        batch = move_batch(batch, device)
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            output = model(batch["inputs"], batch["observable"], batch["sensor_index"])
        baseline_logits = output["baseline_logits"]
        trained_logits = output["segmentation_logits"]
        baseline_float = baseline_logits.float()
        trained_float = trained_logits.float()
        delta_logits = trained_float - baseline_float
        clear = batch["clear"] > 0.5
        for index in range(baseline_logits.shape[0]):
            label = int(batch["presence"][index].item())
            sensor_index = int(batch["sensor_index"][index].item())
            observable = batch["observable"][index, 0].cpu().numpy() > 0.5
            truth = (batch["mask"][index, 0].cpu().numpy() > 0.5) & observable
            labels.append(label)
            groups.append(str(batch["group_id"][index]))
            sensor_indices.append(sensor_index)
            for alpha in alphas:
                key = alpha_key(alpha)
                if alpha == 0.0:
                    logits = baseline_logits[index, 0]
                elif alpha == 1.0:
                    logits = trained_logits[index, 0]
                else:
                    logits = (
                        baseline_float[index, 0] + alpha * delta_logits[index, 0]
                    ).to(baseline_logits.dtype)
                probability = torch.sigmoid(logits).float().masked_fill(
                    ~clear[index, 0], 0.0
                )
                score = probability.cpu().numpy()
                state = states[key]
                state["scores"].append(connected_scene_score(score))
                state["predictions"].append(bool(np.any(component_mask(score))))
                add_pixels(state["pixels"], score, truth, observable)
                add_pixels(
                    state["pixels_by_sensor"][SENSOR_NAMES[sensor_index]],
                    score,
                    truth,
                    observable,
                )

    y = np.asarray(labels, dtype=np.uint8)
    sensors = np.asarray(sensor_indices, dtype=np.uint8)
    baseline_key = alpha_key(0.0)
    # Build alpha zero without subtraction, then use it as the immutable baseline.
    zero_state = states[baseline_key]
    zero_metrics = summarize_candidate(
        labels=y,
        sensor_indices=sensors,
        scores=zero_state["scores"],
        predictions=zero_state["predictions"],
        pixels=zero_state["pixels"],
        pixels_by_sensor=zero_state["pixels_by_sensor"],
        baseline=None,
    )
    baseline_reference = {
        "candidate": zero_metrics["candidate"],
        "sensor_strata": {
            name: {"candidate": value["candidate"]}
            for name, value in zero_metrics["sensor_strata"].items()
        },
    }
    summaries: dict[str, dict[str, Any]] = {}
    for alpha in alphas:
        key = alpha_key(alpha)
        state = states[key]
        summary = summarize_candidate(
            labels=y,
            sensor_indices=sensors,
            scores=state["scores"],
            predictions=state["predictions"],
            pixels=state["pixels"],
            pixels_by_sensor=state["pixels_by_sensor"],
            baseline=baseline_reference,
        )
        summary["balanced_rank"] = list(balanced_rank(summary))
        summary["promotion_checks"] = promotion_checks(summary)
        summaries[key] = summary
    if any(float(value) != 0.0 for value in summaries[baseline_key]["delta"].values()):
        raise RuntimeError("Alpha zero does not reproduce the released baseline exactly")
    for value in summaries[baseline_key]["sensor_strata"].values():
        if any(float(delta) != 0.0 for delta in value["delta"].values()):
            raise RuntimeError("Alpha zero sensor stratum is not exactly released baseline")
    assert_matches_artifact_baseline(
        summaries[baseline_key], artifact["validation"]
    )

    selected_key, selected = select_alpha(summaries)
    checks = selected["promotion_checks"]
    passed = all(checks.values())
    decision = (
        "Advance the frozen trust-region alpha to the independent fold-1 confirmation."
        if passed
        else "Reject the trust-region blend on fold 0; proceed to source-aligned fitting."
    )
    report = {
        "schema_version": 1,
        "scope": "fold-0 development architecture selection; fold 1 and paper test not loaded",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows": int(y.size),
        "positive": int(np.count_nonzero(y == 1)),
        "sites": len(set(groups)),
        "alphas": summaries,
        "selected_alpha": float(selected_key),
        "selected_checks": checks,
        "decision": decision,
        "provenance": {
            "artifact_path": args.artifact,
            "artifact_sha256": args.artifact_sha256,
            "artifact_epoch": int(artifact["epoch"]),
            "artifact_seed": int(artifact["seed"]),
            "development_manifest_sha256": manifest_hash,
            "protocol_sha256": sha256(protocol_path),
            "script_sha256": sha256(Path(__file__).resolve()),
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
            "device": str(torch.cuda.get_device_name(device) if device.type == "cuda" else device),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
            "rasterio": rasterio.__version__,
        },
    }
    write_json((root / args.output_json).resolve(), report)
    write_markdown((root / args.output_markdown).resolve(), report)
    print(json.dumps({"ok": True, "selected_alpha": float(selected_key), "checks": checks, "decision": decision}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
