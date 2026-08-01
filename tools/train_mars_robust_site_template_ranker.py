#!/usr/bin/env python3
"""Cross-fit a robust, label-free site-history anomaly ranker on MARS folds 3/4."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "EarthRemoteSensingRapidResponse"
for path in (MODEL_ROOT, ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from acquire_mars_metadata import repo_root, sha256  # noqa: E402
from train_mars_oof_scene_ensemble_v2 import ap_group_bootstrap, sample_weights  # noqa: E402
from train_mars_scene_ranker import blend_scores, comparison, metric_summary  # noqa: E402
from train_mars_spatial_scene_classifier import (  # noqa: E402
    CHANNEL_NAMES,
    SpatialSceneClassifier,
    augment_batch,
)

DEFAULT_IMAGES = Path("outputs/mars_spatial_scene_inputs_all_folds.npy")
DEFAULT_IMAGES_SHA256 = "6530fa2d07d94bd57ba1ac757039dedd18745227e7a476fb0d69f78f996134a5"
DEFAULT_SPATIAL_METADATA = Path("outputs/mars_spatial_scene_inputs_all_folds_metadata.npz")
DEFAULT_SPATIAL_METADATA_SHA256 = "0160ed93371396a487b819dd170a682f55756d285f0ba94d0f0d093ba51a8d01"
DEFAULT_FOLDS34 = Path("outputs/mars_dense_prithvi_crossfit_scene_features_folds34_metadata.npz")
DEFAULT_PROTOCOL = Path("configs/mars_robust_site_template_ranker_protocol.json")
DEFAULT_ARTIFACT = Path(
    "EarthRemoteSensingRapidResponse/artifacts/mars_robust_site_template_ranker.pt"
)
DEFAULT_JSON = Path("reports/experiments/mars_robust_site_template_ranker.json")
DEFAULT_MARKDOWN = Path("reports/experiments/MARS_ROBUST_SITE_TEMPLATE_RANKER.md")

FOLDS = (3, 4)
CHANNELS = len(CHANNEL_NAMES)
BLENDS = (0.05, 0.10, 0.20, 0.30, 0.40)
IQR_FLOORS = np.asarray(
    (0.02, 0.02, 0.02, 0.01, 0.01, 0.02, 0.02, 0.05, 0.05),
    dtype=np.float32,
)[:, None, None]


def candidate_specs() -> list[dict[str, Any]]:
    return [
        {
            "feature_set": feature_set,
            "weighting": weighting,
            "epochs": 8,
            "learning_rate": 3e-4,
            "weight_decay": 1e-3,
            "dropout": 0.2,
        }
        for feature_set in ("original_robust_residual", "original_median_residual")
        for weighting in ("group", "site_cell")
    ]


def spec_key(spec: dict[str, Any]) -> str:
    return "_".join(f"{key}-{spec[key]}" for key in sorted(spec))


def input_channels(feature_set: str) -> int:
    if feature_set == "original_robust_residual":
        return CHANNELS * 2
    if feature_set == "original_median_residual":
        return CHANNELS * 3
    raise ValueError(f"Unknown robust site feature set: {feature_set}")


def build_robust_site_templates(
    images: np.ndarray,
    image_indices: np.ndarray,
    groups: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build label-free per-site q25/median/q75 templates from eligible rows only."""
    rows = np.asarray(image_indices, dtype=np.int64)
    sites = np.asarray(groups).astype(str)
    if rows.shape != sites.shape or images.shape[1:] != (CHANNELS, 64, 64):
        raise ValueError("Robust site-template inputs do not align")
    unique, inverse, counts = np.unique(sites, return_inverse=True, return_counts=True)
    templates = np.empty((unique.size, 3, CHANNELS, 64, 64), dtype=np.float32)
    for site_index in range(unique.size):
        local = np.flatnonzero(inverse == site_index)
        values = np.asarray(images[rows[local]], dtype=np.float32)
        templates[site_index] = np.quantile(
            values, (0.25, 0.50, 0.75), axis=0, method="linear"
        ).astype(np.float32)
    if not np.isfinite(templates).all() or np.any(counts <= 0):
        raise RuntimeError("Robust site templates are non-finite or empty")
    return templates, counts.astype(np.int64), inverse.astype(np.int64), unique


def compose_robust_site_batch(
    images: np.ndarray,
    image_indices: np.ndarray,
    local_rows: np.ndarray,
    templates: np.ndarray,
    group_indices: np.ndarray,
    feature_set: str,
) -> np.ndarray:
    """Compose raw maps with a robust label-free same-site temporal anomaly."""
    selected = np.asarray(local_rows, dtype=np.int64)
    original = np.asarray(images[np.asarray(image_indices)[selected]], dtype=np.float32)
    local_templates = templates[group_indices[selected]]
    q25 = local_templates[:, 0]
    median = local_templates[:, 1]
    q75 = local_templates[:, 2]
    residual = original - median
    if feature_set == "original_robust_residual":
        scale = np.maximum(q75 - q25, IQR_FLOORS[None])
        robust = np.clip(residual / scale, -8.0, 8.0) / 8.0
        values = np.concatenate((original, robust), axis=1)
    elif feature_set == "original_median_residual":
        values = np.concatenate((original, median, residual), axis=1)
    else:
        raise ValueError(f"Unknown robust site feature set: {feature_set}")
    if values.shape[1] != input_channels(feature_set) or not np.isfinite(values).all():
        raise RuntimeError("Robust site batch schema or finiteness failure")
    return values.astype(np.float32, copy=False)


def configure_determinism(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def train_model(
    spec: dict[str, Any],
    images: np.ndarray,
    image_indices: np.ndarray,
    selected_rows: np.ndarray,
    labels: np.ndarray,
    sensors: np.ndarray,
    weights: np.ndarray,
    templates: np.ndarray,
    group_indices: np.ndarray,
    *,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    configure_determinism(seed)
    feature_set = str(spec["feature_set"])
    model = SpatialSceneClassifier(
        input_channels(feature_set), float(spec["dropout"])
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(spec["learning_rate"]),
        weight_decay=float(spec["weight_decay"]),
    )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    positive_weight = min(
        4.0,
        float(np.sqrt(weights[labels == 0].sum() / max(weights[labels == 1].sum(), 1e-8))),
    )
    batch_size = 128
    model.train()
    for _ in range(int(spec["epochs"])):
        order = torch.randperm(labels.size, generator=generator).numpy()
        for start in range(0, labels.size, batch_size):
            batch = order[start : start + batch_size]
            local = selected_rows[batch]
            array = compose_robust_site_batch(
                images,
                image_indices,
                local,
                templates,
                group_indices,
                feature_set,
            )
            values = augment_batch(torch.from_numpy(array), generator).to(device)
            target = torch.from_numpy(labels[batch].astype(np.float32)).to(device)
            sensor = torch.from_numpy(sensors[batch].astype(np.int64)).to(device)
            row_weight = torch.from_numpy(weights[batch].astype(np.float32)).to(device)
            class_weight = torch.where(target > 0.5, positive_weight, 1.0)
            losses = F.binary_cross_entropy_with_logits(
                model(values, sensor), target, reduction="none"
            )
            loss = (losses * row_weight * class_weight).sum() / (
                row_weight * class_weight
            ).sum()
            if not torch.isfinite(loss):
                raise RuntimeError("Robust site training produced a non-finite loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
    return {
        "state_dict": {
            name: value.detach().cpu() for name, value in model.state_dict().items()
        },
        "input_channels": input_channels(feature_set),
        "feature_set": feature_set,
        "dropout": float(spec["dropout"]),
        "positive_weight": positive_weight,
    }


@torch.no_grad()
def predict_model(
    fitted: dict[str, Any],
    images: np.ndarray,
    image_indices: np.ndarray,
    selected_rows: np.ndarray,
    sensors: np.ndarray,
    templates: np.ndarray,
    group_indices: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    model = SpatialSceneClassifier(
        int(fitted["input_channels"]), float(fitted["dropout"])
    ).to(device)
    model.load_state_dict(fitted["state_dict"])
    model.eval()
    parts: list[np.ndarray] = []
    for start in range(0, selected_rows.size, 256):
        local = selected_rows[start : start + 256]
        array = compose_robust_site_batch(
            images,
            image_indices,
            local,
            templates,
            group_indices,
            str(fitted["feature_set"]),
        )
        values = torch.from_numpy(array).to(device)
        sensor = torch.from_numpy(sensors[start : start + 256].astype(np.int64)).to(device)
        parts.append(torch.sigmoid(model(values, sensor)).cpu().numpy())
    result = np.concatenate(parts).astype(np.float64)
    if result.shape != (selected_rows.size,) or not np.isfinite(result).all():
        raise RuntimeError("Robust site model produced invalid scores")
    return result


def evaluate(
    labels: np.ndarray,
    sensors: np.ndarray,
    groups: np.ndarray,
    folds: np.ndarray,
    current: np.ndarray,
    candidate: np.ndarray,
    *,
    bootstrap_seed: int,
) -> dict[str, Any]:
    baseline = metric_summary(labels, current, sensors)
    metrics = metric_summary(labels, candidate, sensors)
    result = comparison(metrics, baseline)
    by_fold: dict[str, Any] = {}
    for fold in FOLDS:
        rows = folds == fold
        by_fold[str(fold)] = comparison(
            metric_summary(labels[rows], candidate[rows], sensors[rows]),
            metric_summary(labels[rows], current[rows], sensors[rows]),
        )
    bootstrap = ap_group_bootstrap(
        labels,
        current,
        candidate,
        groups,
        replicates=10_000,
        seed=bootstrap_seed,
    )
    fold_ap = [value["delta"]["average_precision"] for value in by_fold.values()]
    fold_recall = [
        value["delta"]["recall_at_fpr_0_0713"] for value in by_fold.values()
    ]
    sensor_ap = result["delta"]["sensor_average_precision"]
    checks = {
        "average_precision_delta_at_least_0_002": result["delta"]["average_precision"] >= 0.002,
        "matched_fpr_recall_no_lower": result["delta"]["recall_at_fpr_0_0713"] >= 0.0,
        "each_fold_ap_positive": min(fold_ap) > 0.0,
        "each_fold_recall_at_least_minus_0_002": min(fold_recall) >= -0.002,
        "each_sensor_ap_nonnegative": min(sensor_ap.values()) >= 0.0,
        "paired_site_ap_lower_positive": float(bootstrap["lower"]) > 0.0,
        "no_worse_fpr": metrics["false_positive_rate"] <= baseline["false_positive_rate"] + 1e-12,
    }
    return {
        "metrics": metrics,
        "baseline": baseline,
        "delta": result["delta"],
        "by_fold": by_fold,
        "paired_site_ap_delta": bootstrap,
        "checks": checks,
        "passed": all(checks.values()),
    }


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    selected = report["selected"]
    result = selected["result"]
    lines = [
        "# Robust site-template MARS ranker",
        "",
        "The candidate uses label-free same-site pixelwise medians and interquartile ranges; only folds 3/4 were opened.",
        "",
        f"- Selected feature set: `{selected['spec']['feature_set']}`",
        f"- Selected weighting: `{selected['spec']['weighting']}`",
        f"- Blend: {selected['blend']:.2f}",
        f"- AP delta: {result['delta']['average_precision']:+.6f}",
        f"- Recall delta at matched FPR: {result['delta']['recall_at_fpr_0_0713']:+.6f}",
        f"- Paired-site AP 95% CI: [{result['paired_site_ap_delta']['lower']:+.6f}, {result['paired_site_ap_delta']['upper']:+.6f}]",
        f"- Promotion gates: {'PASS' if result['passed'] else 'FAIL'}",
        "",
        report["decision"],
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", default=DEFAULT_IMAGES.as_posix())
    parser.add_argument("--spatial-metadata", default=DEFAULT_SPATIAL_METADATA.as_posix())
    parser.add_argument("--folds34", default=DEFAULT_FOLDS34.as_posix())
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL.as_posix())
    parser.add_argument("--artifact", default=DEFAULT_ARTIFACT.as_posix())
    parser.add_argument("--output-json", default=DEFAULT_JSON.as_posix())
    parser.add_argument("--output-markdown", default=DEFAULT_MARKDOWN.as_posix())
    args = parser.parse_args()
    root = repo_root()
    protocol_path = (root / args.protocol).resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    paths = {
        "images": (root / args.images).resolve(),
        "spatial_metadata": (root / args.spatial_metadata).resolve(),
        "folds34": (root / args.folds34).resolve(),
    }
    for name, path in paths.items():
        expected = str(protocol["inputs"][name]["sha256"])
        if sha256(path) != expected:
            raise ValueError(f"Frozen {name} hash mismatch")
    for dependency in protocol["code_dependencies"]:
        dependency_path = (root / str(dependency["path"])).resolve()
        if sha256(dependency_path) != str(dependency["sha256"]):
            raise ValueError(f"Frozen dependency hash mismatch: {dependency['path']}")
    current_result = protocol["inputs"]["current_result"]
    current_result_path = (root / str(current_result["path"])).resolve()
    if sha256(current_result_path) != str(current_result["sha256"]):
        raise ValueError("Frozen current-result hash mismatch")
    if sha256(Path(__file__).resolve()) != str(protocol["trainer"]["sha256"]):
        raise ValueError("Frozen robust site trainer hash mismatch")

    images = np.load(paths["images"], mmap_mode="r", allow_pickle=False)
    with np.load(paths["spatial_metadata"], allow_pickle=False) as metadata:
        spatial_ids = metadata["sample_ids"].astype(str)
        spatial_groups = metadata["groups"].astype(str)
        channel_names = metadata["channel_names"].astype(str)
    with np.load(paths["folds34"], allow_pickle=False) as cache:
        sample_ids = cache["sample_ids"].astype(str)
        labels = cache["labels"].astype(np.uint8)
        sensors = cache["sensors"].astype(np.uint8)
        groups = cache["groups"].astype(str)
        folds = cache["folds"].astype(np.uint8)
        current = cache["exact_base_scores"].astype(np.float64)
    if set(np.unique(folds).tolist()) != set(FOLDS):
        raise ValueError("Robust site experiment must contain only folds 3 and 4")
    if not np.array_equal(channel_names, np.asarray(CHANNEL_NAMES)):
        raise ValueError("Spatial channel contract differs")
    lookup = {identifier: index for index, identifier in enumerate(spatial_ids)}
    if len(lookup) != spatial_ids.size:
        raise ValueError("Spatial metadata sample ids are not unique")
    image_indices = np.asarray([lookup[identifier] for identifier in sample_ids], dtype=np.int64)
    if not np.array_equal(spatial_groups[image_indices], groups):
        raise ValueError("Folds34 and spatial map groups do not align")
    templates, counts, group_indices, unique_groups = build_robust_site_templates(
        images, image_indices, groups
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")
    candidates: list[dict[str, Any]] = []
    raw_by_key: dict[str, np.ndarray] = {}
    for spec_index, spec in enumerate(candidate_specs()):
        raw = np.empty(labels.shape, dtype=np.float64)
        for held_fold in FOLDS:
            fit = folds != held_fold
            held = folds == held_fold
            fit_rows = np.flatnonzero(fit)
            held_rows = np.flatnonzero(held)
            weights = sample_weights(
                str(spec["weighting"]), groups[fit], labels[fit], sensors[fit]
            )
            fitted = train_model(
                spec,
                images,
                image_indices,
                fit_rows,
                labels[fit],
                sensors[fit],
                weights,
                templates,
                group_indices,
                seed=20268400 + held_fold,
                device=device,
            )
            raw[held] = predict_model(
                fitted,
                images,
                image_indices,
                held_rows,
                sensors[held],
                templates,
                group_indices,
                device,
            )
        key = spec_key(spec)
        raw_by_key[key] = raw
        for blend_index, blend in enumerate(BLENDS):
            scores = blend_scores(current, raw, blend)
            result = evaluate(
                labels,
                sensors,
                groups,
                folds,
                current,
                scores,
                bootstrap_seed=20268450 + spec_index * len(BLENDS) + blend_index,
            )
            fold_ap = [
                value["delta"]["average_precision"] for value in result["by_fold"].values()
            ]
            fold_recall = [
                value["delta"]["recall_at_fpr_0_0713"]
                for value in result["by_fold"].values()
            ]
            sensor_ap = result["delta"]["sensor_average_precision"].values()
            candidates.append(
                {
                    "spec": spec,
                    "spec_key": key,
                    "blend": blend,
                    "result": result,
                    "rank": [
                        int(result["passed"]),
                        min(fold_ap),
                        min(fold_recall),
                        min(sensor_ap),
                        float(result["paired_site_ap_delta"]["lower"]),
                        result["delta"]["average_precision"],
                        -blend,
                    ],
                }
            )
        print(json.dumps({"completed": key, "models": spec_index + 1}), flush=True)

    selected = max(candidates, key=lambda value: tuple(value["rank"]))
    passed = bool(selected["result"]["passed"])
    artifact_path = (root / args.artifact).resolve()
    artifact_hash: str | None = None
    if passed:
        weights = sample_weights(
            str(selected["spec"]["weighting"]), groups, labels, sensors
        )
        fitted = train_model(
            selected["spec"],
            images,
            image_indices,
            np.arange(labels.size),
            labels,
            sensors,
            weights,
            templates,
            group_indices,
            seed=20268480,
            device=device,
        )
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = artifact_path.with_suffix(artifact_path.suffix + ".tmp")
        torch.save(
            {
                "schema_version": 1,
                "kind": "mars_robust_site_template_ranker",
                "spec": selected["spec"],
                "fitted": fitted,
                "blend": selected["blend"],
                "template_contract": "same-site q25/median/q75 from label-free folds34 maps",
                "fit_folds": list(FOLDS),
                "folds34_sha256": protocol["inputs"]["folds34"]["sha256"],
            },
            temporary,
        )
        os.replace(temporary, artifact_path)
        artifact_hash = sha256(artifact_path)

    report = {
        "schema_version": 1,
        "scope": "fold-3/fold-4 robust site-history crossfit; no other outcome partition loaded",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "identity": {
            "rows": int(labels.size),
            "groups": int(unique_groups.size),
            "fold_counts": {str(fold): int(np.sum(folds == fold)) for fold in FOLDS},
            "site_observation_count": {
                "minimum": int(counts.min()),
                "median": float(np.median(counts)),
                "maximum": int(counts.max()),
            },
        },
        "architecture": {
            "templates": "label-free per-site q25/median/q75 maps",
            "iqr_floors": IQR_FLOORS[:, 0, 0].tolist(),
            "robust_clip": [-8.0, 8.0],
            "model": "existing compact residual spatial CNN with sensor embedding",
        },
        "candidates": candidates,
        "selected": selected,
        "all_promotion_gates_pass": passed,
        "artifact_sha256": artifact_hash,
        "decision": (
            "Freeze a new-seed fold-2 confirmation protocol; folds 0/1 and official outcomes remain closed."
            if passed
            else "Reject robust site-template ranking before fold-2 access."
        ),
        "provenance": {
            "protocol": protocol_path.relative_to(root).as_posix(),
            "protocol_sha256": sha256(protocol_path),
            "trainer_sha256": sha256(Path(__file__).resolve()),
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True
            ).strip(),
            "device": str(torch.cuda.get_device_name(device) if device.type == "cuda" else device),
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
    }
    write_json((root / args.output_json).resolve(), report)
    write_markdown((root / args.output_markdown).resolve(), report)
    print(
        json.dumps(
            {
                "ok": passed,
                "selected": selected["spec_key"],
                "blend": selected["blend"],
                "ap_delta": selected["result"]["delta"]["average_precision"],
                "recall_delta": selected["result"]["delta"]["recall_at_fpr_0_0713"],
                "paired_ap_lower": selected["result"]["paired_site_ap_delta"]["lower"],
            },
            sort_keys=True,
        )
    )
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
