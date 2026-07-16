#!/usr/bin/env python3
"""Cross-fit a leakage-safe geographic neighboring-site prior for MARS scenes."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "EarthRemoteSensingRapidResponse", ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from acquire_mars_metadata import repo_root, sha256  # noqa: E402
from train_mars_oof_scene_ensemble_v2 import ap_group_bootstrap  # noqa: E402
from train_mars_scene_ranker import blend_scores  # noqa: E402
from train_mars_spatial_scene_classifier import (  # noqa: E402
    DEFAULT_FOLD0_CACHE,
    DEFAULT_FOLD0_SHA256,
    DEFAULT_FOLD1_CACHE,
    DEFAULT_FOLD1_SHA256,
    DEFAULT_INNER_CACHE,
    DEFAULT_INNER_SHA256,
    DEFAULT_METADATA,
    DEFAULT_METADATA_SHA256,
    DEFAULT_SCORE_CACHE,
    DEFAULT_SCORE_SHA256,
    load_partitions,
)
from train_mars_target_weighted_scene_head import evaluate_candidate  # noqa: E402

DEFAULT_MANIFEST = Path(
    "EarthRemoteSensingRapidResponse/Data Collection/s2_emit_pairs/"
    "publication-v1/external/MARS-S2L/paper_v3_development_samples.jsonl"
)
DEFAULT_MANIFEST_SHA256 = "31ba92e791ba07be781dd700ff1e720b8cd686357b9bec38ebfe41bbaa207e8e"
DEFAULT_ARTIFACT = Path(
    "EarthRemoteSensingRapidResponse/artifacts/mars_geospatial_site_prior.json"
)
DEFAULT_JSON = Path("reports/experiments/mars_geospatial_site_prior.json")
DEFAULT_MARKDOWN = Path("reports/experiments/MARS_GEOSPATIAL_SITE_PRIOR.md")
FOLDS = (0, 1, 2, 3, 4)
NEIGHBORS = (5, 20, 50)
SCALES_KM = (100.0, 500.0, 2000.0)
BLENDS = (0.05, 0.1, 0.2, 0.3, 0.4)


def haversine_km(target: np.ndarray, source: np.ndarray) -> np.ndarray:
    """Pairwise great-circle distance for latitude/longitude degree arrays."""
    if target.ndim != 2 or source.ndim != 2 or target.shape[1] != 2 or source.shape[1] != 2:
        raise ValueError("Coordinates must have shape Nx2 and Mx2")
    target_radians = np.radians(target.astype(np.float64))
    source_radians = np.radians(source.astype(np.float64))
    latitude_delta = target_radians[:, None, 0] - source_radians[None, :, 0]
    longitude_delta = target_radians[:, None, 1] - source_radians[None, :, 1]
    value = (
        np.sin(latitude_delta / 2.0) ** 2
        + np.cos(target_radians[:, None, 0])
        * np.cos(source_radians[None, :, 0])
        * np.sin(longitude_delta / 2.0) ** 2
    )
    return 6371.0088 * 2.0 * np.arcsin(np.sqrt(np.clip(value, 0.0, 1.0)))


def site_table(
    groups: np.ndarray, labels: np.ndarray, coordinates: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    names = np.unique(groups.astype(str))
    locations = np.empty((names.size, 2), dtype=np.float64)
    positives = np.empty(names.size, dtype=np.float64)
    counts = np.empty(names.size, dtype=np.float64)
    for index, name in enumerate(names):
        rows = groups == name
        local = coordinates[rows]
        locations[index] = np.median(local, axis=0)
        if np.max(np.abs(local - locations[index])) > 0.05:
            raise ValueError(f"Group {name} spans inconsistent coordinates")
        positives[index] = float(labels[rows].sum())
        counts[index] = float(rows.sum())
    return names, locations, positives, counts


def spatial_prior(
    source_groups: np.ndarray,
    source_labels: np.ndarray,
    source_coordinates: np.ndarray,
    target_groups: np.ndarray,
    target_coordinates: np.ndarray,
    *,
    neighbors: int,
    scale_km: float,
) -> tuple[np.ndarray, dict[str, float]]:
    if neighbors <= 0 or scale_km <= 0:
        raise ValueError("Neighbor count and distance scale must be positive")
    names, locations, positives, counts = site_table(
        source_groups, source_labels, source_coordinates
    )
    target_names, target_locations, _, _ = site_table(
        target_groups, np.zeros(target_groups.shape, dtype=np.uint8), target_coordinates
    )
    distances = haversine_km(target_locations, locations)
    width = min(neighbors, names.size)
    nearest = np.argpartition(distances, width - 1, axis=1)[:, :width]
    nearest_distance = np.take_along_axis(distances, nearest, axis=1)
    # Jeffreys smoothing controls sites with few historical observations.
    rates = (positives + 0.5) / (counts + 1.0)
    global_rate = float((positives.sum() + 0.5) / (counts.sum() + 1.0))
    local_rates = rates[nearest]
    weights = np.exp(-nearest_distance / scale_km) * np.sqrt(counts[nearest])
    prior = (
        (weights * local_rates).sum(axis=1) + global_rate
    ) / (weights.sum(axis=1) + 1.0)
    lookup = {name: float(value) for name, value in zip(target_names, prior)}
    result = np.asarray([lookup[str(group)] for group in target_groups], dtype=np.float64)
    audit = {
        "source_sites": int(names.size),
        "target_sites": int(target_names.size),
        "global_scene_prevalence": global_rate,
        "nearest_distance_km_median": float(np.median(nearest_distance[:, 0])),
        "nearest_distance_km_p90": float(np.quantile(nearest_distance[:, 0], 0.9)),
        "prior_min": float(prior.min()),
        "prior_median": float(np.median(prior)),
        "prior_max": float(prior.max()),
    }
    return result, audit


def crossfit_prior(
    values: dict[str, np.ndarray], coordinates: np.ndarray, *, neighbors: int, scale_km: float
) -> tuple[np.ndarray, list[dict[str, float]]]:
    scores = np.empty(values["labels"].shape, dtype=np.float64)
    audits = []
    for holdout in FOLDS:
        fit = values["folds"] != holdout
        held = ~fit
        scores[held], audit = spatial_prior(
            values["groups"][fit],
            values["labels"][fit],
            coordinates[fit],
            values["groups"][held],
            coordinates[held],
            neighbors=neighbors,
            scale_km=scale_km,
        )
        audits.append({"holdout": holdout, **audit})
    return scores, audits


def combine_partitions(parts: dict[str, dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    order = ("fold0", "fold1", "inner")
    counts = [parts[name]["labels"].size for name in order]
    return {
        "image_indices": np.concatenate([parts[name]["image_indices"] for name in order]),
        "labels": np.concatenate([parts[name]["labels"] for name in order]),
        "sensors": np.concatenate([parts[name]["sensors"] for name in order]),
        "groups": np.concatenate([parts[name]["groups"] for name in order]),
        "primary": np.concatenate([parts[name]["primary"] for name in order]),
        "current": np.concatenate([parts[name]["new"] for name in order]),
        "folds": np.concatenate(
            (
                np.zeros(counts[0], dtype=np.uint8),
                np.ones(counts[1], dtype=np.uint8),
                parts["inner"]["folds"].astype(np.uint8),
            )
        ),
    }


def read_manifest_coordinates(path: Path) -> dict[str, tuple[float, float]]:
    values: dict[str, tuple[float, float]] = {}
    with path.open("r", encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            values[str(row["sample_id"])] = (
                float(row["latitude"]),
                float(row["longitude"]),
            )
    return values


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST.as_posix())
    parser.add_argument("--manifest-sha256", default=DEFAULT_MANIFEST_SHA256)
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
        "manifest": (root / args.manifest).resolve(),
        "metadata": (root / args.metadata).resolve(),
        "score": (root / args.score_cache).resolve(),
        "inner": (root / args.inner_cache).resolve(),
        "fold0": (root / args.fold0_cache).resolve(),
        "fold1": (root / args.fold1_cache).resolve(),
    }
    expected = {
        "manifest": args.manifest_sha256,
        "metadata": args.metadata_sha256,
        "score": args.score_sha256,
        "inner": args.inner_sha256,
        "fold0": args.fold0_sha256,
        "fold1": args.fold1_sha256,
    }
    for name, digest in expected.items():
        if sha256(paths[name]) != digest:
            raise ValueError(f"Frozen {name} hash mismatch")
    parts = load_partitions(
        paths["metadata"], paths["score"], {name: paths[name] for name in ("inner", "fold0", "fold1")}
    )
    values = combine_partitions(parts)
    coordinate_lookup = read_manifest_coordinates(paths["manifest"])
    with np.load(paths["metadata"], allow_pickle=False) as metadata:
        sample_ids = metadata["sample_ids"].astype(str)
    global_coordinates = np.asarray([coordinate_lookup[sample_id] for sample_id in sample_ids])
    coordinates = global_coordinates[values["image_indices"]]
    candidates = []
    raw_store = {}
    audit_store = {}
    for neighbors in NEIGHBORS:
        for scale_km in SCALES_KM:
            raw, audits = crossfit_prior(
                values, coordinates, neighbors=neighbors, scale_km=scale_km
            )
            key = (neighbors, scale_km)
            raw_store[key] = raw
            audit_store[key] = audits
            for blend in BLENDS:
                candidate = evaluate_candidate(
                    values,
                    raw,
                    {"neighbors": neighbors, "scale_km": scale_km},
                    blend,
                )
                candidate.update(
                    {"neighbors": neighbors, "scale_km": scale_km, "blend_weight": blend}
                )
                candidates.append(candidate)
    selected = max(candidates, key=lambda value: tuple(value["rank"]))
    key = (selected["neighbors"], selected["scale_km"])
    raw = raw_store[key]
    scores = blend_scores(values["current"], raw, selected["blend_weight"])
    selected["fold_audits"] = audit_store[key]
    selected["paired_group_bootstrap_ap_delta_vs_primary"] = ap_group_bootstrap(
        values["labels"], values["primary"], scores, values["groups"], replicates=10_000, seed=20262380
    )
    selected["paired_group_bootstrap_ap_delta_vs_current"] = ap_group_bootstrap(
        values["labels"], values["current"], scores, values["groups"], replicates=10_000, seed=20262381
    )
    passed = bool(
        selected["stable"]
        and selected["paired_group_bootstrap_ap_delta_vs_primary"]["lower"] > 0
        and selected["paired_group_bootstrap_ap_delta_vs_current"]["lower"] > 0
    )
    artifact = (root / args.artifact).resolve()
    artifact_hash = None
    if passed:
        payload = {
            "schema_version": 1,
            "kind": "mars_geospatial_site_prior",
            "neighbors": selected["neighbors"],
            "scale_km": selected["scale_km"],
            "blend_weight": selected["blend_weight"],
            "manifest_sha256": expected["manifest"],
        }
        write_json(artifact, payload)
        artifact_hash = sha256(artifact)
    report = {
        "schema_version": 1,
        "scope": "five-fold geographic neighboring-site prior; held-site labels excluded; paper cache not loaded",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "neighbors": list(NEIGHBORS),
        "scales_km": list(SCALES_KM),
        "blends": list(BLENDS),
        "selected": selected,
        "all_promotion_gates_pass": passed,
        "decision": (
            "Freeze the geospatial site prior for label-free paper scoring."
            if passed
            else "Reject the geospatial site prior before paper scoring."
        ),
        "provenance": {
            **{f"{name}_sha256": digest for name, digest in expected.items()},
            "artifact_sha256": artifact_hash,
            "script_sha256": sha256(Path(__file__).resolve()),
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
            "numpy": np.__version__,
        },
    }
    write_json((root / args.output_json).resolve(), report)
    delta = selected["versus_current"]["delta"]
    interval = selected["paired_group_bootstrap_ap_delta_vs_current"]
    markdown = (root / args.output_markdown).resolve()
    markdown.parent.mkdir(parents=True, exist_ok=True)
    markdown.write_text(
        "\n".join(
            [
                "# Geographic neighboring-site MARS prior",
                "",
                f"- Neighbors / scale / blend: {selected['neighbors']} / {selected['scale_km']:.0f} km / {selected['blend_weight']:.2f}",
                f"- AP delta vs current: {delta['average_precision']:+.5f}",
                f"- Recall delta vs current: {delta['recall_at_fpr_0_0713']:+.5f}",
                f"- Paired-site AP interval: [{interval['lower']:+.5f}, {interval['upper']:+.5f}]",
                "",
                report["decision"],
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "ok": passed,
                "neighbors": selected["neighbors"],
                "scale_km": selected["scale_km"],
                "blend": selected["blend_weight"],
                "ap_delta": delta["average_precision"],
                "recall_delta": delta["recall_at_fpr_0_0713"],
                "ap_lower": interval["lower"],
                "artifact_sha256": artifact_hash,
            },
            indent=2,
        )
    )
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
