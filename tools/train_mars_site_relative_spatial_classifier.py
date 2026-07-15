#!/usr/bin/env python3
"""Train a label-free site-relative spatial classifier for MARS scenes."""

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
from train_mars_oof_scene_ensemble_v2 import (  # noqa: E402
    ap_group_bootstrap,
    sample_weights,
)
from train_mars_scene_ranker import blend_scores  # noqa: E402
from train_mars_spatial_scene_classifier import (  # noqa: E402
    DEFAULT_FOLD0_CACHE,
    DEFAULT_FOLD0_SHA256,
    DEFAULT_FOLD1_CACHE,
    DEFAULT_FOLD1_SHA256,
    DEFAULT_IMAGES,
    DEFAULT_IMAGES_SHA256,
    DEFAULT_INNER_CACHE,
    DEFAULT_INNER_SHA256,
    DEFAULT_METADATA,
    DEFAULT_METADATA_SHA256,
    DEFAULT_SCORE_CACHE,
    DEFAULT_SCORE_SHA256,
    INNER_FOLDS,
    SpatialSceneClassifier,
    augment_batch,
    confirm_partition,
    evaluate_candidate,
    load_partitions,
)

DEFAULT_ARTIFACT = Path(
    "EarthRemoteSensingRapidResponse/artifacts/mars_site_relative_spatial_classifier.pt"
)
DEFAULT_JSON = Path("reports/experiments/mars_site_relative_spatial_classifier.json")
DEFAULT_MARKDOWN = Path(
    "reports/experiments/MARS_SITE_RELATIVE_SPATIAL_CLASSIFIER.md"
)
CHANNELS = 9
BLENDS = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.625, 0.75, 1.0)


def candidate_specs() -> list[dict[str, Any]]:
    return [
        {
            "feature_set": feature_set,
            "weighting": weighting,
            "epochs": 8,
            "learning_rate": 0.0003,
            "weight_decay": 0.001,
            "dropout": 0.2,
        }
        for feature_set in ("original_residual", "original_template_residual")
        for weighting in ("group", "site_cell")
    ]


def spec_key(spec: dict[str, Any]) -> str:
    return "_".join(f"{key}-{spec[key]}" for key in sorted(spec))


def input_channels(feature_set: str) -> int:
    if feature_set == "original_residual":
        return CHANNELS * 2
    if feature_set == "original_template_residual":
        return CHANNELS * 3
    raise ValueError(f"Unknown site-relative feature set: {feature_set}")


def build_site_templates(
    images: np.ndarray, groups: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return per-site means, site sizes, and a group index for every row."""
    if images.shape[0] != groups.size or images.shape[1:] != (CHANNELS, 64, 64):
        raise ValueError("Site-template inputs do not align")
    unique, inverse, counts = np.unique(groups.astype(str), return_inverse=True, return_counts=True)
    means = np.empty((unique.size, CHANNELS, 64, 64), dtype=np.float32)
    for index in range(unique.size):
        rows = np.flatnonzero(inverse == index)
        means[index] = np.asarray(images[rows], dtype=np.float32).mean(axis=0)
    if not np.isfinite(means).all() or np.any(counts <= 0):
        raise RuntimeError("Site templates are non-finite or empty")
    return means, counts.astype(np.int64), inverse.astype(np.int64)


def compose_site_relative_batch(
    images: np.ndarray,
    global_indices: np.ndarray,
    means: np.ndarray,
    counts: np.ndarray,
    group_indices: np.ndarray,
    feature_set: str,
) -> np.ndarray:
    """Compose original maps and label-free leave-one-out site context."""
    rows = np.asarray(global_indices, dtype=np.int64)
    original = np.asarray(images[rows], dtype=np.float32)
    local_groups = group_indices[rows]
    sizes = counts[local_groups].astype(np.float32)[:, None, None, None]
    templates = means[local_groups]
    leave_one_out = np.where(
        sizes > 1.0,
        (templates * sizes - original) / np.maximum(sizes - 1.0, 1.0),
        original,
    )
    residual = original - leave_one_out
    if feature_set == "original_residual":
        values = np.concatenate((original, residual), axis=1)
    elif feature_set == "original_template_residual":
        values = np.concatenate((original, leave_one_out, residual), axis=1)
    else:
        raise ValueError(f"Unknown site-relative feature set: {feature_set}")
    if values.shape[1] != input_channels(feature_set) or not np.isfinite(values).all():
        raise RuntimeError("Site-relative batch schema or finiteness failure")
    return values


def train_model(
    spec: dict[str, Any],
    images: np.ndarray,
    global_indices: np.ndarray,
    labels: np.ndarray,
    sensors: np.ndarray,
    weights: np.ndarray,
    means: np.ndarray,
    counts: np.ndarray,
    group_indices: np.ndarray,
    *,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
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
            local_rows = order[start : start + batch_size]
            rows = global_indices[local_rows]
            array = compose_site_relative_batch(
                images, rows, means, counts, group_indices, feature_set
            )
            values = augment_batch(torch.from_numpy(array), generator).to(device)
            target = torch.from_numpy(labels[local_rows].astype(np.float32)).to(device)
            sensor = torch.from_numpy(sensors[local_rows].astype(np.int64)).to(device)
            row_weight = torch.from_numpy(weights[local_rows].astype(np.float32)).to(device)
            class_weight = torch.where(target > 0.5, positive_weight, 1.0)
            losses = F.binary_cross_entropy_with_logits(
                model(values, sensor), target, reduction="none"
            )
            loss = (losses * row_weight * class_weight).sum() / (
                row_weight * class_weight
            ).sum()
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
        "template_contract": "label-free leave-one-out pixelwise site mean",
    }


@torch.no_grad()
def predict_model(
    fitted: dict[str, Any],
    images: np.ndarray,
    global_indices: np.ndarray,
    sensors: np.ndarray,
    means: np.ndarray,
    counts: np.ndarray,
    group_indices: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    model = SpatialSceneClassifier(
        int(fitted["input_channels"]), float(fitted["dropout"])
    ).to(device)
    model.load_state_dict(fitted["state_dict"])
    model.eval()
    parts: list[np.ndarray] = []
    for start in range(0, global_indices.size, 256):
        rows = global_indices[start : start + 256]
        array = compose_site_relative_batch(
            images,
            rows,
            means,
            counts,
            group_indices,
            str(fitted["feature_set"]),
        )
        values = torch.from_numpy(array).to(device)
        sensor = torch.from_numpy(
            sensors[start : start + 256].astype(np.int64)
        ).to(device)
        parts.append(torch.sigmoid(model(values, sensor)).cpu().numpy())
    return np.concatenate(parts).astype(np.float64)


def strict_candidate_rank(candidate: dict[str, Any]) -> tuple[float, ...]:
    versus_primary = candidate["versus_primary"]
    versus_current = candidate["versus_new"]
    fold_current_ap = [
        value["versus_new"]["delta"]["average_precision"]
        for value in candidate["per_fold"].values()
    ]
    current_sensor = versus_current["delta"]["sensor_average_precision"]
    strict = (
        versus_primary["delta"]["average_precision"] > 0.0
        and versus_primary["delta"]["recall_at_fpr_0_0713"] > 0.0
        and versus_current["delta"]["average_precision"] > 0.0
        and versus_current["delta"]["recall_at_fpr_0_0713"] >= 0.0
        and min(fold_current_ap) >= 0.0
        and min(current_sensor.values()) >= -0.0025
    )
    candidate["strict_stable"] = bool(strict)
    return (
        float(strict),
        min(fold_current_ap),
        versus_current["delta"]["average_precision"],
        versus_current["delta"]["recall_at_fpr_0_0713"],
        versus_primary["delta"]["average_precision"],
    )


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    selected = report["selected"]
    lines = [
        "# Site-relative spatial MARS scene classifier",
        "",
        "Every spatial template is label-free and computed only from other observations of the same physical site.",
        "",
        f"- Selected model: `{selected['spec_key']}`",
        f"- Blend weight: {selected['blend_weight']:.3f}",
        f"- Inner AP delta vs current head: {selected['versus_new']['delta']['average_precision']:+.5f}",
        f"- Inner AP interval vs current head: [{selected['paired_group_bootstrap_ap_delta_vs_new']['lower']:+.5f}, {selected['paired_group_bootstrap_ap_delta_vs_new']['upper']:+.5f}]",
        "",
        "| Partition | AP delta vs primary | AP delta vs current | Recall delta vs current | Gates |",
        "|---|---:|---:|---:|---|",
    ]
    for name, value in report["confirmation"].items():
        lines.append(
            f"| {name} | {value['versus_primary']['delta']['average_precision']:+.5f} | "
            f"{value['versus_new']['delta']['average_precision']:+.5f} | "
            f"{value['versus_new']['delta']['recall_at_fpr_0_0713']:+.5f} | "
            f"{'PASS' if value['passed'] else 'FAIL'} |"
        )
    pooled = report["held_pooled_bootstrap_ap_delta_vs_new"]
    lines.extend(
        [
            "",
            f"Pooled folds-0/1 paired-site AP interval versus current head: [{pooled['lower']:+.5f}, {pooled['upper']:+.5f}].",
            "",
            report["decision"],
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", default=DEFAULT_IMAGES.as_posix())
    parser.add_argument("--images-sha256", default=DEFAULT_IMAGES_SHA256)
    parser.add_argument("--metadata", default=DEFAULT_METADATA.as_posix())
    parser.add_argument("--metadata-sha256", default=DEFAULT_METADATA_SHA256)
    parser.add_argument("--score-cache", default=DEFAULT_SCORE_CACHE.as_posix())
    parser.add_argument("--score-sha256", default=DEFAULT_SCORE_SHA256)
    parser.add_argument("--inner-cache", default=DEFAULT_INNER_CACHE.as_posix())
    parser.add_argument("--inner-sha256", default=DEFAULT_INNER_SHA256)
    parser.add_argument("--fold0-cache", default=DEFAULT_FOLD0_CACHE.as_posix())
    parser.add_argument("--fold0-sha256", default=DEFAULT_FOLD0_SHA256)
    parser.add_argument("--fold1-cache", default=DEFAULT_FOLD1_CACHE.as_posix())
    parser.add_argument("--fold1-sha256", default=DEFAULT_FOLD1_SHA256)
    parser.add_argument("--artifact", default=DEFAULT_ARTIFACT.as_posix())
    parser.add_argument("--output-json", default=DEFAULT_JSON.as_posix())
    parser.add_argument("--output-markdown", default=DEFAULT_MARKDOWN.as_posix())
    args = parser.parse_args()
    root = repo_root()
    paths = {
        "images": (root / args.images).resolve(),
        "metadata": (root / args.metadata).resolve(),
        "score": (root / args.score_cache).resolve(),
        "inner": (root / args.inner_cache).resolve(),
        "fold0": (root / args.fold0_cache).resolve(),
        "fold1": (root / args.fold1_cache).resolve(),
    }
    expected = {
        "images": args.images_sha256,
        "metadata": args.metadata_sha256,
        "score": args.score_sha256,
        "inner": args.inner_sha256,
        "fold0": args.fold0_sha256,
        "fold1": args.fold1_sha256,
    }
    for name, path in paths.items():
        if sha256(path) != expected[name]:
            raise ValueError(f"Frozen {name} cache hash mismatch")
    images = np.load(paths["images"], mmap_mode="r", allow_pickle=False)
    with np.load(paths["metadata"], allow_pickle=False) as metadata:
        all_groups = metadata["groups"].astype(str)
    means, counts, group_indices = build_site_templates(images, all_groups)
    partitions = load_partitions(
        paths["metadata"],
        paths["score"],
        {name: paths[name] for name in ("inner", "fold0", "fold1")},
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")
    candidates: list[dict[str, Any]] = []
    raw_by_key: dict[str, np.ndarray] = {}
    specs = candidate_specs()
    for spec_index, spec in enumerate(specs):
        raw = np.empty(partitions["inner"]["labels"].shape, dtype=np.float64)
        for holdout in INNER_FOLDS:
            fit_rows = partitions["inner"]["folds"] != holdout
            held_rows = partitions["inner"]["folds"] == holdout
            weights = sample_weights(
                str(spec["weighting"]),
                partitions["inner"]["groups"][fit_rows],
                partitions["inner"]["labels"][fit_rows],
                partitions["inner"]["sensors"][fit_rows],
            )
            fitted = train_model(
                spec,
                images,
                partitions["inner"]["image_indices"][fit_rows],
                partitions["inner"]["labels"][fit_rows],
                partitions["inner"]["sensors"][fit_rows],
                weights,
                means,
                counts,
                group_indices,
                seed=20261100 + holdout,
                device=device,
            )
            raw[held_rows] = predict_model(
                fitted,
                images,
                partitions["inner"]["image_indices"][held_rows],
                partitions["inner"]["sensors"][held_rows],
                means,
                counts,
                group_indices,
                device,
            )
        key = spec_key(spec)
        raw_by_key[key] = raw
        local = [
            evaluate_candidate(spec, blend, partitions["inner"], raw)
            for blend in BLENDS
        ]
        for value in local:
            value["spec_key"] = key
            value["strict_rank"] = list(strict_candidate_rank(value))
        candidates.extend(local)
        best = max(local, key=lambda value: tuple(value["strict_rank"]))
        print(
            json.dumps(
                {
                    "candidate_model": spec_index + 1,
                    "total_models": len(specs),
                    "spec": key,
                    "best_blend": best["blend_weight"],
                    "ap_delta_vs_current": best["versus_new"]["delta"][
                        "average_precision"
                    ],
                    "recall_delta_vs_current": best["versus_new"]["delta"][
                        "recall_at_fpr_0_0713"
                    ],
                    "strict_stable": best["strict_stable"],
                }
            ),
            flush=True,
        )
    selected = max(candidates, key=lambda value: tuple(value["strict_rank"]))
    selected_scores = blend_scores(
        partitions["inner"]["new"],
        raw_by_key[selected["spec_key"]],
        float(selected["blend_weight"]),
    )
    selected["paired_group_bootstrap_ap_delta_vs_primary"] = ap_group_bootstrap(
        partitions["inner"]["labels"],
        partitions["inner"]["primary"],
        selected_scores,
        partitions["inner"]["groups"],
        replicates=10_000,
        seed=20261120,
    )
    selected["paired_group_bootstrap_ap_delta_vs_new"] = ap_group_bootstrap(
        partitions["inner"]["labels"],
        partitions["inner"]["new"],
        selected_scores,
        partitions["inner"]["groups"],
        replicates=10_000,
        seed=20261121,
    )
    selected["inner_passed"] = bool(
        selected["strict_stable"]
        and selected["paired_group_bootstrap_ap_delta_vs_primary"]["lower"] > 0.0
        and selected["paired_group_bootstrap_ap_delta_vs_new"]["lower"] > 0.0
    )

    spec = selected["spec"]
    weights = sample_weights(
        str(spec["weighting"]),
        partitions["inner"]["groups"],
        partitions["inner"]["labels"],
        partitions["inner"]["sensors"],
    )
    fitted = train_model(
        spec,
        images,
        partitions["inner"]["image_indices"],
        partitions["inner"]["labels"],
        partitions["inner"]["sensors"],
        weights,
        means,
        counts,
        group_indices,
        seed=20261130,
        device=device,
    )
    confirmation: dict[str, Any] = {}
    held_scores: list[np.ndarray] = []
    thresholds: list[float] = []
    for index, name in enumerate(("fold0", "fold1")):
        partition = partitions[name]
        raw = predict_model(
            fitted,
            images,
            partition["image_indices"],
            partition["sensors"],
            means,
            counts,
            group_indices,
            device,
        )
        scores = blend_scores(partition["new"], raw, float(selected["blend_weight"]))
        held_scores.append(scores)
        result = confirm_partition(partition, scores, seed=20261140 + index)
        current = result["versus_new"]["delta"]
        result["checks"]["ap_higher_than_current"] = current["average_precision"] > 0.0
        result["checks"]["recall_no_worse_than_current"] = (
            current["recall_at_fpr_0_0713"] >= 0.0
        )
        result["passed"] = all(result["checks"].values())
        confirmation[name] = result
        thresholds.append(
            result["versus_primary"]["metrics"]["operating_point"]["threshold"]
        )
    held_labels = np.concatenate(
        [partitions[name]["labels"] for name in ("fold0", "fold1")]
    )
    held_current = np.concatenate(
        [partitions[name]["new"] for name in ("fold0", "fold1")]
    )
    held_groups = np.concatenate(
        [partitions[name]["groups"] for name in ("fold0", "fold1")]
    )
    held_pooled = ap_group_bootstrap(
        held_labels,
        held_current,
        np.concatenate(held_scores),
        held_groups,
        replicates=10_000,
        seed=20261150,
    )
    passed = bool(
        selected["inner_passed"]
        and all(value["passed"] for value in confirmation.values())
        and held_pooled["lower"] > 0.0
    )

    artifact_path = (root / args.artifact).resolve()
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = artifact_path.with_suffix(artifact_path.suffix + ".tmp")
    torch.save(
        {
            "schema_version": 1,
            "kind": "mars_site_relative_spatial_scene_classifier",
            "spec": spec,
            "fitted": fitted,
            "blend_weight": float(selected["blend_weight"]),
            "operational_scene_threshold": max(thresholds),
            "template_contract": "label-free leave-one-out pixelwise site mean",
            "images_sha256": args.images_sha256,
            "metadata_sha256": args.metadata_sha256,
        },
        temporary,
    )
    os.replace(temporary, artifact_path)
    report = {
        "schema_version": 1,
        "scope": "development only; paper-test labels, scores, and imagery are not loaded",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "architecture": {
            "site_template": "label-free leave-one-out pixelwise mean over same-site observations",
            "singleton_policy": "template equals scene and residual is zero",
            "candidate_feature_sets": [
                "original_residual",
                "original_template_residual",
            ],
        },
        "candidate_model_count": len(specs),
        "candidate_blend_count": len(candidates),
        "candidate_summaries": [
            {
                "spec_key": value["spec_key"],
                "blend_weight": value["blend_weight"],
                "strict_stable": value["strict_stable"],
                "ap_delta_vs_primary": value["versus_primary"]["delta"][
                    "average_precision"
                ],
                "ap_delta_vs_current": value["versus_new"]["delta"][
                    "average_precision"
                ],
                "recall_delta_vs_current": value["versus_new"]["delta"][
                    "recall_at_fpr_0_0713"
                ],
                "worst_fold_ap_delta_vs_current": min(
                    fold["versus_new"]["delta"]["average_precision"]
                    for fold in value["per_fold"].values()
                ),
            }
            for value in candidates
        ],
        "selected": selected,
        "confirmation": confirmation,
        "held_pooled_bootstrap_ap_delta_vs_new": held_pooled,
        "operational_scene_threshold": max(thresholds),
        "all_promotion_gates_pass": passed,
        "decision": (
            "Freeze the site-relative spatial model for one transparent exact-paper replay."
            if passed
            else "Reject the site-relative spatial model before paper-cache scoring."
        ),
        "provenance": {
            **{f"{name}_cache_sha256": value for name, value in expected.items()},
            "artifact_sha256": sha256(artifact_path),
            "script_sha256": sha256(Path(__file__).resolve()),
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True
            ).strip(),
            "device": str(
                torch.cuda.get_device_name(device) if device.type == "cuda" else device
            ),
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
                "blend_weight": selected["blend_weight"],
                "inner_ap_delta_vs_current": selected["versus_new"]["delta"][
                    "average_precision"
                ],
                "inner_ap_lower_vs_current": selected[
                    "paired_group_bootstrap_ap_delta_vs_new"
                ]["lower"],
                "held_pooled_ap_lower_vs_current": held_pooled["lower"],
                "confirmation": {
                    name: {
                        "passed": value["passed"],
                        "ap_delta_vs_current": value["versus_new"]["delta"][
                            "average_precision"
                        ],
                        "recall_delta_vs_current": value["versus_new"]["delta"][
                            "recall_at_fpr_0_0713"
                        ],
                    }
                    for name, value in confirmation.items()
                },
            },
            indent=2,
        )
    )
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
