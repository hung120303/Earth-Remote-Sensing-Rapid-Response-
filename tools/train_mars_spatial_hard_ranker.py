#!/usr/bin/env python3
"""Train a hard-negative pairwise spatial ranker for MARS scene retrieval."""

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
from train_mars_scene_ranker import blend_scores  # noqa: E402
from train_mars_spatial_scene_classifier import (  # noqa: E402
    BLEND_WEIGHTS,
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
    channel_indices,
    confirm_partition,
    evaluate_candidate,
    load_partitions,
    predict_model,
)

DEFAULT_ARTIFACT = Path("EarthRemoteSensingRapidResponse/artifacts/mars_spatial_hard_ranker.pt")
DEFAULT_JSON = Path("reports/experiments/mars_spatial_hard_ranker.json")
DEFAULT_MARKDOWN = Path("reports/experiments/MARS_SPATIAL_HARD_RANKER.md")


def candidate_specs() -> list[dict[str, Any]]:
    return [
        {
            "feature_set": "physics_spatial",
            "weighting": weighting,
            "epochs": 10,
            "learning_rate": 0.0003,
            "weight_decay": 0.001,
            "dropout": 0.2,
            "hard_negative_fraction": hard_fraction,
            "pairwise_weight": pairwise_weight,
        }
        for weighting, hard_fraction, pairwise_weight in (
            ("site_cell", 0.75, 0.25),
            ("site_cell", 0.75, 0.5),
            ("site_cell", 0.75, 1.0),
            ("site_cell", 1.0, 0.5),
            ("group", 0.75, 0.5),
        )
    ]


def spec_key(spec: dict[str, Any]) -> str:
    return "_".join(f"{key}-{spec[key]}" for key in sorted(spec))


def hard_negative_probabilities(scores: np.ndarray, fraction: float) -> np.ndarray:
    if scores.ndim != 1 or scores.size == 0:
        raise ValueError("Hard-negative scores must be a non-empty vector")
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("Hard-negative fraction must be in [0,1]")
    order = np.argsort(np.argsort(scores, kind="stable"), kind="stable")
    percentile = (order.astype(np.float64) + 1.0) / scores.size
    hard = percentile**4
    hard /= hard.sum()
    uniform = np.full(scores.size, 1.0 / scores.size)
    probabilities = (1.0 - fraction) * uniform + fraction * hard
    return probabilities / probabilities.sum()


def train_hard_ranker(
    spec: dict[str, Any],
    images: np.ndarray,
    global_indices: np.ndarray,
    labels: np.ndarray,
    sensors: np.ndarray,
    new_scores: np.ndarray,
    weights: np.ndarray,
    *,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    channels = channel_indices(str(spec["feature_set"]))
    model = SpatialSceneClassifier(len(channels), float(spec["dropout"])).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(spec["learning_rate"]),
        weight_decay=float(spec["weight_decay"]),
    )
    positive_rows = np.flatnonzero(labels == 1)
    negative_rows = np.flatnonzero(labels == 0)
    negative_probabilities = hard_negative_probabilities(
        new_scores[negative_rows], float(spec["hard_negative_fraction"])
    )
    half_batch = 64
    steps = int(np.ceil(labels.size / (2 * half_batch)))
    model.train()
    for _ in range(int(spec["epochs"])):
        for _ in range(steps):
            positive = rng.choice(positive_rows, size=half_batch, replace=True)
            negative = rng.choice(
                negative_rows,
                size=half_batch,
                replace=True,
                p=negative_probabilities,
            )
            rows = np.concatenate([positive, negative])
            array = np.asarray(images[global_indices[rows]][:, channels], dtype=np.float32)
            values = augment_batch(torch.from_numpy(array), generator).to(device)
            sensor = torch.from_numpy(sensors[rows].astype(np.int64)).to(device)
            target = torch.cat(
                [torch.ones(half_batch), torch.zeros(half_batch)]
            ).to(device)
            row_weight = torch.from_numpy(weights[rows].astype(np.float32)).to(device)
            logits = model(values, sensor)
            bce_parts = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
            bce = (bce_parts * row_weight).sum() / row_weight.sum()
            pairwise = F.softplus(-(logits[:half_batch] - logits[half_batch:])).mean()
            loss = bce + float(spec["pairwise_weight"]) * pairwise
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
    return {
        "state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "input_channels": len(channels),
        "channel_indices": channels,
        "dropout": float(spec["dropout"]),
        "training": "balanced positive/hard-negative BCE plus paired softplus ranking",
    }


def crossfit_raw_scores(
    spec: dict[str, Any],
    images: np.ndarray,
    partition: dict[str, np.ndarray],
    device: torch.device,
) -> np.ndarray:
    scores = np.empty(partition["labels"].shape, dtype=np.float64)
    for holdout in INNER_FOLDS:
        fit_rows = partition["folds"] != holdout
        held_rows = partition["folds"] == holdout
        weights = sample_weights(
            str(spec["weighting"]),
            partition["groups"][fit_rows],
            partition["labels"][fit_rows],
            partition["sensors"][fit_rows],
        )
        fitted = train_hard_ranker(
            spec,
            images,
            partition["image_indices"][fit_rows],
            partition["labels"][fit_rows],
            partition["sensors"][fit_rows],
            partition["new"][fit_rows],
            weights,
            seed=20261000 + holdout,
            device=device,
        )
        scores[held_rows] = predict_model(
            fitted,
            images,
            partition["image_indices"][held_rows],
            partition["sensors"][held_rows],
            device,
        )
    return scores


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    selected = report["selected"]
    lines = [
        "# Hard-negative pairwise MARS spatial ranker",
        "",
        "Every selected score was cross-fitted on folds 2/3/4, then frozen before folds 0/1 confirmation.",
        "",
        f"- Selected model: `{selected['spec_key']}`",
        f"- Spatial blend weight: {selected['blend_weight']:.3f}",
        f"- Inner AP delta vs primary: {selected['versus_primary']['delta']['average_precision']:+.5f}",
        f"- Inner AP delta vs stronger head: {selected['versus_new']['delta']['average_precision']:+.5f}",
        f"- Inner AP interval vs stronger head: [{selected['paired_group_bootstrap_ap_delta_vs_new']['lower']:+.5f}, {selected['paired_group_bootstrap_ap_delta_vs_new']['upper']:+.5f}]",
        "",
        "| Partition | AP delta vs primary | Recall delta | AP 95% CI | AP delta vs new | Gates |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for name, value in report["confirmation"].items():
        delta = value["versus_primary"]["delta"]
        interval = value["paired_group_bootstrap_ap_delta_vs_primary"]
        lines.append(
            f"| {name} | {delta['average_precision']:+.5f} | "
            f"{delta['recall_at_fpr_0_0713']:+.5f} | "
            f"[{interval['lower']:+.5f}, {interval['upper']:+.5f}] | "
            f"{value['versus_new']['delta']['average_precision']:+.5f} | "
            f"{'PASS' if value['passed'] else 'FAIL'} |"
        )
    lines.extend(["", report["decision"]])
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
    partitions = load_partitions(
        paths["metadata"],
        paths["score"],
        {name: paths[name] for name in ("inner", "fold0", "fold1")},
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")
    specs = candidate_specs()
    raw_by_key: dict[str, np.ndarray] = {}
    candidates: list[dict[str, Any]] = []
    for index, spec in enumerate(specs):
        raw = crossfit_raw_scores(spec, images, partitions["inner"], device)
        key = spec_key(spec)
        raw_by_key[key] = raw
        local = [
            evaluate_candidate(spec, weight, partitions["inner"], raw)
            for weight in BLEND_WEIGHTS
        ]
        for value in local:
            value["spec_key"] = key
        candidates.extend(local)
        best = max(local, key=lambda value: tuple(value["rank"]))
        print(
            json.dumps(
                {
                    "candidate_model": index + 1,
                    "total_models": len(specs),
                    "spec": key,
                    "best_blend": best["blend_weight"],
                    "ap_delta_vs_primary": best["versus_primary"]["delta"][
                        "average_precision"
                    ],
                    "ap_delta_vs_new": best["versus_new"]["delta"]["average_precision"],
                    "stable": best["stable"],
                }
            ),
            flush=True,
        )
    selected = max(candidates, key=lambda value: tuple(value["rank"]))
    selected_raw = raw_by_key[selected["spec_key"]]
    selected_scores = blend_scores(
        partitions["inner"]["new"], selected_raw, float(selected["blend_weight"])
    )
    selected["paired_group_bootstrap_ap_delta_vs_primary"] = ap_group_bootstrap(
        partitions["inner"]["labels"],
        partitions["inner"]["primary"],
        selected_scores,
        partitions["inner"]["groups"],
        replicates=10000,
        seed=20261020,
    )
    selected["paired_group_bootstrap_ap_delta_vs_new"] = ap_group_bootstrap(
        partitions["inner"]["labels"],
        partitions["inner"]["new"],
        selected_scores,
        partitions["inner"]["groups"],
        replicates=10000,
        seed=20261021,
    )
    selected["inner_passed"] = bool(
        selected["stable"]
        and selected["paired_group_bootstrap_ap_delta_vs_primary"]["lower"] > 0.0
        and selected["paired_group_bootstrap_ap_delta_vs_new"]["lower"] > -0.0025
    )

    spec = selected["spec"]
    weights = sample_weights(
        str(spec["weighting"]),
        partitions["inner"]["groups"],
        partitions["inner"]["labels"],
        partitions["inner"]["sensors"],
    )
    fitted = train_hard_ranker(
        spec,
        images,
        partitions["inner"]["image_indices"],
        partitions["inner"]["labels"],
        partitions["inner"]["sensors"],
        partitions["inner"]["new"],
        weights,
        seed=20261030,
        device=device,
    )
    confirmation: dict[str, Any] = {}
    thresholds: list[float] = []
    for index, name in enumerate(("fold0", "fold1")):
        partition = partitions[name]
        raw = predict_model(
            fitted,
            images,
            partition["image_indices"],
            partition["sensors"],
            device,
        )
        scores = blend_scores(partition["new"], raw, float(selected["blend_weight"]))
        confirmation[name] = confirm_partition(partition, scores, seed=20261040 + index)
        thresholds.append(
            confirmation[name]["versus_primary"]["metrics"]["operating_point"]["threshold"]
        )

    artifact_path = (root / args.artifact).resolve()
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = artifact_path.with_suffix(artifact_path.suffix + ".tmp")
    payload = {
        "schema_version": 1,
        "kind": "mars_spatial_hard_negative_pairwise_ranker",
        "spec": spec,
        "fitted": fitted,
        "blend_weight": float(selected["blend_weight"]),
        "operational_scene_threshold": max(thresholds),
        "images_sha256": args.images_sha256,
        "metadata_sha256": args.metadata_sha256,
    }
    torch.save(payload, temporary)
    os.replace(temporary, artifact_path)
    passed = selected["inner_passed"] and all(value["passed"] for value in confirmation.values())
    report = {
        "schema_version": 1,
        "scope": "development only; paper-test labels and imagery are not loaded",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_model_count": len(specs),
        "candidate_blend_count": len(candidates),
        "candidates": candidates,
        "selected": selected,
        "confirmation": confirmation,
        "operational_scene_threshold": max(thresholds),
        "all_promotion_gates_pass": passed,
        "decision": (
            "Freeze the hard ranker for a transparent post-test paper benchmark."
            if passed
            else "Reject the hard ranker before paper-test scoring."
        ),
        "provenance": {
            **{f"{name}_cache_sha256": expected[name] for name in expected},
            "artifact_sha256": sha256(artifact_path),
            "script_sha256": sha256(Path(__file__).resolve()),
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
                "blend_weight": selected["blend_weight"],
                "inner_ap_delta_vs_primary": selected["versus_primary"]["delta"][
                    "average_precision"
                ],
                "inner_ap_delta_vs_new": selected["versus_new"]["delta"][
                    "average_precision"
                ],
                "confirmation": {
                    name: {
                        "passed": value["passed"],
                        "ap_delta_vs_primary": value["versus_primary"]["delta"][
                            "average_precision"
                        ],
                        "ap_delta_vs_new": value["versus_new"]["delta"]["average_precision"],
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
