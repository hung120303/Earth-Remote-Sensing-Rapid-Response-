#!/usr/bin/env python3
"""Evaluate the pinned released MARS-S2L checkpoint on the frozen strict cohort.

The input construction, U-Net topology, 0.5 pixel threshold, and 100-pixel
8-connected scene rule reproduce UNEP-IMEO-MARS/marss2l at the pinned source
revision. The implementation is intentionally inference-only and loads the
checkpoint with ``weights_only=True``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import scipy
import sklearn
import torch
from scipy import ndimage
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "EarthRemoteSensingRapidResponse"
if str(MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_ROOT))

from mars_s2l_adapter import compute_mbmp, iter_manifest, load_sample  # noqa: E402

from acquire_mars_metadata import DEFAULT_OUTPUT, REVISION, repo_root, sha256  # noqa: E402
from build_mars_dev_cohort import DEV_SAMPLES  # noqa: E402
from run_mars_dev_pixel_baselines import bootstrap_scene  # noqa: E402
from run_mars_dev_scene_baselines import role_weights  # noqa: E402

RELEASE_SOURCE_REVISION = "f7d264c2c845dfba1cb27f76ef6026275f8d8758"
DEFAULT_MODEL_DIR = Path(
    "EarthRemoteSensingRapidResponse/Data Collection/s2_emit_pairs/publication-v1/"
    "external/MARS-S2L/trained_models/MARSS2L_20250326"
)
DEFAULT_CHECKPOINT = DEFAULT_MODEL_DIR / "best_epoch"
DEFAULT_CONFIG = DEFAULT_MODEL_DIR / "config_experiment.json"
DEFAULT_METADATA_CSV = DEFAULT_OUTPUT / "validated_images_all.csv"
DEFAULT_JSON = Path("reports/experiments/mars_released_model_baseline.json")
DEFAULT_MARKDOWN = Path("reports/experiments/MARS_RELEASED_MODEL_BASELINE.md")
EXPECTED_CHECKPOINT_SHA256 = (
    "be634fb9e24dc4877f44c1ff9f69972e6f0453e30d70c0dc03677876340ef246"
)
EXPECTED_CHECKPOINT_BYTES = 163_291_870
EXPECTED_CONFIG_SHA256 = "abeb92d01313fbb2939e6c5fc1c6281846b8102ea5edd7081668fe0db05bf79f"
PIXEL_THRESHOLD = 0.5
MINIMUM_CONNECTED_PIXELS = 100
BOOTSTRAP_SEED = 202


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


def double_conv(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.GELU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.GELU(),
    )


def down(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(nn.MaxPool2d(2), double_conv(in_channels, out_channels))


class Up(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = double_conv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        difference_y = x2.size(2) - x1.size(2)
        difference_x = x2.size(3) - x1.size(3)
        x1 = F.pad(
            x1,
            [
                difference_x // 2,
                difference_x - difference_x // 2,
                difference_y // 2,
                difference_y - difference_y // 2,
            ],
        )
        return self.conv(torch.cat([x2, x1], dim=1))


class ReleasedMarsS2LUNet(nn.Module):
    """Inference-equivalent form of upstream ``UnetOriginal``."""

    def __init__(self) -> None:
        super().__init__()
        self.inc = double_conv(16, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = Up(1024, 256)
        self.up2 = Up(512, 128)
        self.up3 = Up(256, 64)
        self.up4 = Up(128, 128)
        self.out = nn.Conv2d(128, 1, kernel_size=1, stride=1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(values)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        output = self.up1(x5, x4)
        output = self.up2(output, x3)
        output = self.up3(output, x2)
        output = self.up4(output, x1)
        return self.out(output)[:, 0]


def load_released_model(path: Path, device: torch.device) -> ReleasedMarsS2LUNet:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    original = payload["model_state_dict"]
    state = {
        key.removeprefix("_orig_mod.module.").removeprefix("module."): value
        for key, value in original.items()
    }
    model = ReleasedMarsS2LUNet()
    incompatible = model.load_state_dict(state, strict=False)
    # The pinned checkpoint contains an unused legacy ``out_mlp`` branch. The
    # released loader also uses strict=False, while current UnetOriginal.forward
    # uses only ``out``. Permit exactly that documented surplus and no missing
    # parameters so architecture drift still fails loudly.
    if incompatible.missing_keys or not incompatible.unexpected_keys:
        raise ValueError(f"Unexpected checkpoint compatibility result: {incompatible}")
    if any(not key.startswith("out_mlp.") for key in incompatible.unexpected_keys):
        raise ValueError(f"Unknown surplus checkpoint parameters: {incompatible.unexpected_keys}")
    return model.to(device).eval()


def wind_lookup(path: Path, required: set[str]) -> dict[str, tuple[float, float]]:
    result: dict[str, tuple[float, float]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        for row in csv.DictReader(source):
            sample_id = row["id_loc_image"]
            if sample_id not in required:
                continue
            values: list[float] = []
            for key in ("wind_u", "wind_v"):
                try:
                    value = float(row[key])
                except (TypeError, ValueError):
                    value = math.nan
                values.append(4.0 if not math.isfinite(value) else float(np.clip(value, -20, 20)))
            result[sample_id] = (values[0], values[1])
    missing = required - set(result)
    if missing:
        raise ValueError(f"Metadata CSV is missing wind for {len(missing)} strict samples")
    return result


def released_input(sample: Any, wind: tuple[float, float]) -> np.ndarray:
    spectral = np.clip(sample.raw_pair.astype(np.float32) / 5000.0, 0.0, 2.0)
    spectral[~np.isfinite(spectral)] = 0.0
    release_mbmp = compute_mbmp(spectral[:6], spectral[6:])
    height, width = release_mbmp.shape
    wind_channels = np.broadcast_to(
        np.asarray(wind, dtype=np.float32)[:, None, None] / 8.0,
        (2, height, width),
    ).copy()
    cloud = (sample.cloud_classes > 0).astype(np.float32)[None, ...]
    return np.concatenate([release_mbmp[None, ...], spectral, wind_channels, cloud]).astype(
        np.float32
    )


def component_mask(score: np.ndarray) -> np.ndarray:
    labels, count = ndimage.label(
        score > PIXEL_THRESHOLD, structure=np.ones((3, 3), dtype=np.uint8)
    )
    if count == 0:
        return np.zeros(score.shape, dtype=bool)
    sizes = np.bincount(labels.ravel())
    keep = sizes >= MINIMUM_CONNECTED_PIXELS
    keep[0] = False
    return keep[labels]


def connected_scene_score(score: np.ndarray) -> float:
    """Reproduce upstream's connected-component cutoff to 1e-3 tolerance."""
    low = float(np.min(score))
    high = float(np.max(score))
    threshold = (low + high) / 2.0
    while high - low > 1e-3:
        labels, count = ndimage.label(
            score > threshold, structure=np.ones((3, 3), dtype=np.uint8)
        )
        largest = 0 if count == 0 else int(np.bincount(labels.ravel())[1:].max())
        if largest >= MINIMUM_CONNECTED_PIXELS:
            low = threshold
        else:
            high = threshold
        threshold = (low + high) / 2.0
    return threshold


def confusion(
    truth: np.ndarray, predicted: np.ndarray, weights: np.ndarray | None = None
) -> dict[str, float | None]:
    weight = np.ones(truth.shape, dtype=np.float64) if weights is None else weights
    tp = float(np.sum(weight[(truth == 1) & predicted]))
    tn = float(np.sum(weight[(truth == 0) & ~predicted]))
    fp = float(np.sum(weight[(truth == 0) & predicted]))
    fn = float(np.sum(weight[(truth == 1) & ~predicted]))

    def ratio(numerator: float, denominator: float) -> float | None:
        return None if denominator == 0 else numerator / denominator

    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "recall": ratio(tp, tp + fn),
        "specificity": ratio(tn, tn + fp),
        "false_positive_rate": ratio(fp, fp + tn),
        "precision": ratio(tp, tp + fp),
        "negative_predictive_value": ratio(tn, tn + fn),
        "accuracy": ratio(tp + tn, tp + tn + fp + fn),
    }


def scene_metrics(
    truth: np.ndarray,
    predicted: np.ndarray,
    scores: np.ndarray,
    weights: np.ndarray | None = None,
) -> dict[str, Any]:
    result = confusion(truth, predicted, weights)
    result["auroc"] = float(roc_auc_score(truth, scores, sample_weight=weights))
    result["average_precision"] = float(
        average_precision_score(truth, scores, sample_weight=weights)
    )
    return result


def evaluate(
    root: Path,
    metadata_dir: Path,
    records: list[dict[str, Any]],
    winds: dict[str, tuple[float, float]],
    model: ReleasedMarsS2LUNet,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    scene_truth: list[int] = []
    scene_prediction: list[bool] = []
    scene_scores: list[float] = []
    groups: list[str] = []
    pixel_truth: list[np.ndarray] = []
    pixel_scores: list[np.ndarray] = []
    intersection = predicted_area = truth_area = 0

    for start in range(0, len(records), batch_size):
        batch_records = records[start : start + batch_size]
        samples = [load_sample(metadata_dir, record) for record in batch_records]
        batch = np.stack(
            [released_input(sample, winds[sample.sample_id]) for sample in samples]
        )
        with torch.inference_mode(), torch.amp.autocast(
            "cuda", dtype=torch.float16, enabled=device.type == "cuda"
        ):
            logits = model(torch.from_numpy(batch).to(device, non_blocking=True))
            probabilities = torch.sigmoid(logits).float().cpu().numpy()

        for record, sample, score in zip(batch_records, samples, probabilities):
            score = score.astype(np.float32)
            score[~sample.clear_mask] = 0.0
            prediction = component_mask(score)
            truth = sample.plume_mask & sample.observable_mask
            observable = sample.observable_mask
            scene_truth.append(sample.presence)
            scene_prediction.append(bool(np.any(prediction)))
            scene_scores.append(connected_scene_score(score))
            groups.append(str(record["group_id"]))
            intersection += int(np.count_nonzero(prediction & truth))
            predicted_area += int(np.count_nonzero(prediction & observable))
            truth_area += int(np.count_nonzero(truth))
            pixel_truth.append(truth[observable].astype(np.uint8))
            pixel_scores.append(score[observable])

        completed = min(start + batch_size, len(records))
        print(f"Released MARS-S2L: {completed:,}/{len(records):,}", flush=True)

    y = np.asarray(scene_truth, dtype=np.uint8)
    predicted = np.asarray(scene_prediction, dtype=bool)
    scores = np.asarray(scene_scores, dtype=np.float32)
    group_array = np.asarray(groups)
    weights = role_weights(y, "strict_spatial_test")
    union = predicted_area + truth_area - intersection
    return {
        "sample_count": int(y.size),
        "positive_count": int(np.sum(y)),
        "negative_count": int(np.sum(y == 0)),
        "group_count": int(np.unique(group_array).size),
        "author_fixed_rule": {
            "pixel_threshold": PIXEL_THRESHOLD,
            "comparison": ">",
            "minimum_connected_pixels": MINIMUM_CONNECTED_PIXELS,
            "connectivity": 8,
            "selected_on": "released_author_configuration; no ERSRR tuning",
        },
        "scene_unweighted": scene_metrics(y, predicted, scores),
        "scene_representative_weighted": scene_metrics(y, predicted, scores, weights),
        "group_bootstrap": bootstrap_scene(y, predicted, group_array),
        "pixel_validity_aware": {
            "average_precision": float(
                average_precision_score(np.concatenate(pixel_truth), np.concatenate(pixel_scores))
            ),
            "intersection_over_union": 0.0 if union == 0 else intersection / union,
            "dice": 0.0
            if predicted_area + truth_area == 0
            else 2.0 * intersection / (predicted_area + truth_area),
            "truth_positive_pixels": truth_area,
            "predicted_positive_pixels": predicted_area,
        },
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    result = report["strict_spatial_test"]
    scene = result["scene_unweighted"]
    pixel = result["pixel_validity_aware"]
    ci = result["group_bootstrap"]
    lines = [
        "# Released MARS-S2L checkpoint on the frozen ERSRR strict cohort",
        "",
        "Inference-only reproduction using the authors' fixed 0.5 / 100-pixel rule; no ERSRR threshold tuning.",
        "",
        f"- Cohort: {result['sample_count']} scenes / {result['group_count']} frozen 25 km groups; {result['positive_count']} plume / {result['negative_count']} no plume",
        f"- Scene recall / specificity / FPR: {scene['recall']:.3f} / {scene['specificity']:.3f} / {scene['false_positive_rate']:.3f}",
        f"- Scene AUROC / AP: {scene['auroc']:.3f} / {scene['average_precision']:.3f}",
        f"- Recall 95% CI: {ci['recall_95ci'][0]:.3f}-{ci['recall_95ci'][1]:.3f}",
        f"- Specificity 95% CI: {ci['specificity_95ci'][0]:.3f}-{ci['specificity_95ci'][1]:.3f}",
        f"- Validity-aware pixel AP / IoU / Dice: {pixel['average_precision']:.4f} / {pixel['intersection_over_union']:.4f} / {pixel['dice']:.4f}",
        f"- Checkpoint SHA-256: `{report['artifact']['checkpoint_sha256']}`",
        "",
        "## Interpretation",
        "",
        "This is a checkpoint baseline on ERSRR's stricter spatially disjoint cohort, not a reproduction of the paper's official aggregate test metric. The released checkpoint trained on the official training split, so ERSRR does not use its internal official-train validation subset to recalibrate this model.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-dir", default=DEFAULT_OUTPUT.as_posix())
    parser.add_argument("--manifest", default=DEV_SAMPLES.as_posix())
    parser.add_argument("--metadata-csv", default=DEFAULT_METADATA_CSV.as_posix())
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT.as_posix())
    parser.add_argument("--config", default=DEFAULT_CONFIG.as_posix())
    parser.add_argument("--output-json", default=DEFAULT_JSON.as_posix())
    parser.add_argument("--output-markdown", default=DEFAULT_MARKDOWN.as_posix())
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    root = repo_root(ROOT)
    metadata_dir = (root / args.metadata_dir).resolve()
    manifest = (root / args.manifest).resolve()
    metadata_csv = (root / args.metadata_csv).resolve()
    checkpoint = (root / args.checkpoint).resolve()
    config_path = (root / args.config).resolve()
    output_json = safe_output(root, args.output_json)
    output_markdown = safe_output(root, args.output_markdown)

    if checkpoint.stat().st_size != EXPECTED_CHECKPOINT_BYTES:
        raise ValueError("Released checkpoint size does not match the pinned catalog")
    if sha256(checkpoint) != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("Released checkpoint SHA-256 does not match the pinned LFS OID")
    if sha256(config_path) != EXPECTED_CONFIG_SHA256:
        raise ValueError("Released config SHA-256 does not match the acquisition receipt")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    expected_config = {
        "model": "UnetOriginal",
        "multipass": True,
        "wind": True,
        "cloud_mask": True,
        "cat_mbmp": True,
        "norm_wind": True,
    }
    if any(config.get(key) != value for key, value in expected_config.items()):
        raise ValueError("Released model config does not match the implemented input contract")

    all_records = list(iter_manifest(manifest))
    records = [
        record for record in all_records if record["research_role"] == "strict_spatial_test"
    ]
    required_ids = {str(record["sample_id"]) for record in records}
    winds = wind_lookup(metadata_csv, required_ids)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    model = load_released_model(checkpoint, device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    result = evaluate(root, metadata_dir, records, winds, model, device, args.batch_size)

    report = {
        "schema_version": 1,
        "scope": "released_marss2l_on_frozen_strict_spatial_development_cohort",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "dataset": "UNEP-IMEO/MARS-S2L",
            "dataset_revision": REVISION,
            "upstream_repository": "UNEP-IMEO-MARS/marss2l",
            "upstream_revision": RELEASE_SOURCE_REVISION,
            "development_manifest_sha256": sha256(manifest),
            "metadata_csv_sha256": sha256(metadata_csv),
        },
        "artifact": {
            "checkpoint_bytes": checkpoint.stat().st_size,
            "checkpoint_sha256": sha256(checkpoint),
            "config_sha256": sha256(config_path),
            "checkpoint_tracked": False,
        },
        "model": {
            "name": "MARS-S2L released MARSS2L_20250326",
            "architecture": "UnetOriginal",
            "input_channels": 16,
            "parameter_count": parameter_count,
            "checkpoint_loaded_with_weights_only": True,
        },
        "runtime": {
            "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
            "torch": torch.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
        },
        "provenance": {
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True
            ).strip(),
            "git_tracked_worktree_dirty_at_start": tracked_dirty(root),
            "script": str(Path(__file__).resolve().relative_to(root)).replace("\\", "/"),
            "script_sha256": sha256(Path(__file__).resolve()),
        },
        "strict_spatial_test": result,
        "interpretation": (
            "Released checkpoint baseline on the ERSRR strict spatial cohort; not an official-split "
            "paper metric and not eligible for validation recalibration because the checkpoint saw "
            "the official training subset used by ERSRR internal validation."
        ),
    }
    write_json(output_json, report)
    write_markdown(output_markdown, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
