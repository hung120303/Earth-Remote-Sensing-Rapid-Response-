#!/usr/bin/env python3
"""Freeze the MethaneS2CM v5 development split and sealed-test protocol."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.neighbors import BallTree

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "EarthRemoteSensingRapidResponse"
for directory in (MODEL_ROOT, ROOT / "tools"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from acquire_mars_metadata import repo_root, sha256  # noqa: E402
from methanes2cm_adapter import (  # noqa: E402
    MODEL_BANDS,
    S2_L2A_BAND_ORDER,
    coordinate_group_id,
)
from train_mars_v3 import tracked_dirty, write_json  # noqa: E402

REVISION = "ee9a96d4994ca6bc45725c1e92d7a06258131eaf"
DATA_ROOT = Path(
    "EarthRemoteSensingRapidResponse/Data Collection/s2_emit_pairs/publication-v1/"
    "external/MethaneS2CM"
)
SPLIT = "l2a_location_split_32x32"
DEFAULT_MANIFEST = DATA_ROOT / SPLIT / "v5_train_development_manifest.jsonl"
DEFAULT_JSON = Path("reports/experiments/methanes2cm_v5_protocol.json")
DEFAULT_MARKDOWN = Path("reports/experiments/METHANES2CM_V5_PROTOCOL.md")
GROUP_RADIUS_KM = 25.0
DEVELOPMENT_FRACTION = 0.20
DEVELOPMENT_GROUP_FRACTION = 0.25
SPLIT_SEED = 20_260_713
EARTH_RADIUS_KM = 6371.0088

EXPECTED_IDENTITIES = {
    "train.csv": {
        "bytes": 9_693_624,
        "sha256": "3b14c6b7d34f469f1cf6fe4fceeac403ba80aae5b047098c83b36ea0482cf698",
    },
    "test.csv": {
        "bytes": 2_479_516,
        "sha256": "117a6de1bb8e0d0cb4a746afc2e5f1727e49870a71b9167e578472e89d41d2c4",
    },
}


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = np.arange(size, dtype=np.int32)
        self.weight = np.ones(size, dtype=np.int32)

    def find(self, value: int) -> int:
        while int(self.parent[value]) != value:
            self.parent[value] = self.parent[int(self.parent[value])]
            value = int(self.parent[value])
        return value

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.weight[left_root] < self.weight[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        self.weight[left_root] += self.weight[right_root]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    required = {
        "id",
        "s2_path",
        "s2_pre_path",
        "s2_pre_pre_path",
        "plume_mask_path",
        "label",
        "latitude",
        "longitude",
    }
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Unexpected MethaneS2CM CSV contract: {path}")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError(f"Duplicate MethaneS2CM ids in {path}")
    return rows


def location_components(rows: list[dict[str, str]]) -> tuple[dict[str, str], dict[str, Any]]:
    coordinate_by_location = {
        coordinate_group_id(row): (float(row["latitude"]), float(row["longitude"]))
        for row in rows
    }
    identifiers = sorted(coordinate_by_location)
    coordinates = np.asarray([coordinate_by_location[item] for item in identifiers])
    tree = BallTree(np.radians(coordinates), metric="haversine")
    neighbors = tree.query_radius(
        np.radians(coordinates), r=GROUP_RADIUS_KM / EARTH_RADIUS_KM
    )
    union = UnionFind(len(identifiers))
    for left, values in enumerate(neighbors):
        for right in values:
            if int(right) > left:
                union.union(left, int(right))
    components: dict[int, list[str]] = defaultdict(list)
    for index, identifier in enumerate(identifiers):
        components[union.find(index)].append(identifier)
    by_location: dict[str, str] = {}
    for members in components.values():
        ordered = sorted(members)
        identity = hashlib.sha256("\0".join(ordered).encode("utf-8")).hexdigest()[:16]
        group_id = f"geo25_{identity}"
        for member in ordered:
            by_location[member] = group_id
    counts = Counter(by_location[coordinate_group_id(row)] for row in rows)
    return by_location, {
        "radius_km": GROUP_RADIUS_KM,
        "exact_locations": len(identifiers),
        "connected_groups": len(components),
        "largest_group_locations": max(len(value) for value in components.values()),
        "largest_group_samples": max(counts.values()),
    }


def choose_development_groups(
    rows: list[dict[str, str]], by_location: dict[str, str]
) -> tuple[set[str], dict[str, Any]]:
    statistics: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        group = by_location[coordinate_group_id(row)]
        statistics[group]["samples"] += 1
        statistics[group]["positives"] += int(row["label"])
    total_samples = len(rows)
    total_positives = sum(int(row["label"]) for row in rows)
    targets = np.asarray(
        [total_samples * DEVELOPMENT_FRACTION, total_positives * DEVELOPMENT_FRACTION],
        dtype=np.float64,
    )
    selected: set[str] = set()
    current = np.zeros(2, dtype=np.float64)
    target_group_count = max(1, round(len(statistics) * DEVELOPMENT_GROUP_FRACTION))
    maximum_development_component = total_samples * 0.05

    def identity(group: str) -> str:
        return hashlib.sha256(f"{SPLIT_SEED}:{group}".encode()).hexdigest()

    # Select exactly 20% of the geographic components.  At each stage, choose
    # the component that most closely tracks the proportional sample/positive
    # target for that stage.  This avoids both a two-basin development set and
    # a split that withholds nearly every small geographic component.
    for stage in range(1, target_group_count + 1):
        stage_target = targets * (stage / target_group_count)
        best: tuple[float, str, str, np.ndarray] | None = None
        for group in statistics:
            if group in selected:
                continue
            counts = statistics[group]
            if counts["samples"] > maximum_development_component:
                continue
            proposal = current + np.asarray(
                [counts["samples"], counts["positives"]], dtype=np.float64
            )
            score = float(
                np.sum(
                    np.abs(proposal - stage_target)
                    / np.maximum(stage_target, 1.0)
                )
            )
            candidate = (score, identity(group), group, proposal)
            if best is None or candidate[:2] < best[:2]:
                best = candidate
        if best is None:
            raise ValueError("Insufficient geographic components for development selection")
        selected.add(best[2])
        current = best[3]
    # Deterministic one-for-one swaps refine the final 20% sample/positive
    # target while preserving the frozen geographic-group count.
    def final_objective(values: np.ndarray) -> float:
        return float(
            np.sum(np.abs(values - targets) / np.maximum(targets, 1.0))
        )

    while True:
        best_swap: tuple[float, str, str, np.ndarray] | None = None
        for outgoing in selected:
            outgoing_counts = statistics[outgoing]
            removed = current - np.asarray(
                [outgoing_counts["samples"], outgoing_counts["positives"]],
                dtype=np.float64,
            )
            for incoming, incoming_counts in statistics.items():
                if incoming in selected or incoming_counts["samples"] > maximum_development_component:
                    continue
                proposal = removed + np.asarray(
                    [incoming_counts["samples"], incoming_counts["positives"]],
                    dtype=np.float64,
                )
                candidate = (
                    final_objective(proposal),
                    identity(outgoing),
                    identity(incoming),
                    proposal,
                )
                if best_swap is None or candidate[:3] < best_swap[:3]:
                    best_swap = candidate
                    best_outgoing = outgoing
                    best_incoming = incoming
        if best_swap is None or best_swap[0] >= final_objective(current) - 1e-12:
            break
        selected.remove(best_outgoing)
        selected.add(best_incoming)
        current = best_swap[3]
    if not selected:
        raise ValueError("Development-group selection produced an empty split")
    development = [
        row
        for row in rows
        if by_location[coordinate_group_id(row)] in selected
    ]
    fitting = [
        row
        for row in rows
        if by_location[coordinate_group_id(row)] not in selected
    ]
    return selected, {
        "method": (
            "deterministic staged greedy selection of 25% of frozen 25 km components followed "
            "by objective-improving one-for-one swaps; sample and positive targets are jointly "
            "optimized with stable-hash tie-breaking; any one component exceeding 5% of all "
            "crops remains in fitting"
        ),
        "seed": SPLIT_SEED,
        "target_fraction": DEVELOPMENT_FRACTION,
        "fitting_samples": len(fitting),
        "development_samples": len(development),
        "fitting_positives": sum(int(row["label"]) for row in fitting),
        "development_positives": sum(int(row["label"]) for row in development),
        "achieved_sample_fraction": len(development) / len(rows),
        "achieved_positive_fraction": (
            sum(int(row["label"]) for row in development)
            / max(sum(int(row["label"]) for row in rows), 1)
        ),
        "fitting_groups": len(set(statistics) - selected),
        "development_groups": len(selected),
        "group_overlap": 0,
    }


def write_manifest(
    path: Path,
    rows: list[dict[str, str]],
    by_location: dict[str, str],
    development_groups: set[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as destination:
        for row in sorted(rows, key=lambda value: int(value["id"])):
            group = by_location[coordinate_group_id(row)]
            payload = {
                **row,
                "source_dataset": "H1deaki/MethaneS2CM",
                "source_revision": REVISION,
                "source_split": f"{SPLIT}/train.csv",
                "exact_location_id": coordinate_group_id(row),
                "group_id": group,
                "research_role": (
                    "internal_development" if group in development_groups else "internal_fitting"
                ),
            }
            destination.write(json.dumps(payload, sort_keys=True) + "\n")
    os.replace(temporary, path)


def ignored(root: Path, path: Path) -> bool:
    relative = path.resolve().relative_to(root.resolve())
    return (
        subprocess.run(
            ["git", "check-ignore", "--quiet", "--", relative.as_posix()],
            cwd=root,
            check=False,
        ).returncode
        == 0
    )


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    split = report["development_protocol"]
    lines = [
        "# MethaneS2CM v5 development and sealed-test protocol",
        "",
        "Status: frozen before any L2A location-test image was extracted or opened.",
        "",
        "## Cohort",
        "",
        f"- Source revision: `{report['source']['revision']}`",
        f"- Train metadata: {report['cohort']['train']['samples']:,} crops / {report['cohort']['train']['exact_locations']:,} exact locations",
        f"- Sealed test metadata: {report['cohort']['test']['samples']:,} crops / {report['cohort']['test']['exact_locations']:,} exact locations",
        f"- Exact train/test coordinate overlap: {report['cohort']['coordinate_overlap']}",
        f"- Product: {report['source']['product_level']} at {report['source']['pixel_size_m']} m; balanced crop benchmark, not operational prevalence",
        "",
        "## Internal development boundary",
        "",
        f"- Fitting: {split['fitting_samples']:,} crops / {split['fitting_groups']:,} frozen 25 km groups",
        f"- Development: {split['development_samples']:,} crops / {split['development_groups']:,} frozen 25 km groups",
        f"- Fitting/development 25 km group overlap: {split['group_overlap']}",
        "- Architecture selection, augmentation, epochs, thresholds, and seed selection may use only this internal development partition.",
        "",
        "## Predeclared v5 direction",
        "",
        "- Shared-weight tri-temporal encoder for T, T-90, and T-365 using B02/B03/B04/B08/B11/B12.",
        "- Two scale-invariant MBMP channels (T vs T-90 and T vs T-365).",
        "- Segmentation-first output; scene presence must be derived from dense plume evidence rather than a free classifier.",
        "- Primary selection: scene AP, AUROC, recall at no more than 5% FPR, and positive-pixel Dice, all on frozen internal development groups.",
        "- Final evidence: a fixed multi-seed ensemble evaluated once on the sealed location test together with zero-shot v4.3 and released MARS-S2L.",
        "",
        "## Test seal",
        "",
        report["sealed_test"]["rule"],
        "",
        "The test is balanced by crop construction and lacks acquisition time, wind, and per-pixel cloud masks. Precision is therefore not an operational positive predictive value, and the comparison is a cross-product (L2A versus MARS L1C) robustness benchmark rather than a substitute for a prevalence-representative deployment trial.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=DATA_ROOT.as_posix())
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST.as_posix())
    parser.add_argument("--output-json", default=DEFAULT_JSON.as_posix())
    parser.add_argument("--output-markdown", default=DEFAULT_MARKDOWN.as_posix())
    args = parser.parse_args()

    root = repo_root()
    if tracked_dirty(root):
        raise RuntimeError("Refusing to freeze the protocol from a dirty tracked worktree")
    data_root = (root / args.data_root).resolve()
    split_dir = data_root / SPLIT
    manifest = (root / args.manifest).resolve()
    output_json = (root / args.output_json).resolve()
    output_markdown = (root / args.output_markdown).resolve()
    for filename, expected in EXPECTED_IDENTITIES.items():
        path = split_dir / filename
        if path.stat().st_size != expected["bytes"] or sha256(path) != expected["sha256"]:
            raise ValueError(f"Pinned MethaneS2CM metadata identity mismatch: {filename}")

    train_rows = read_rows(split_dir / "train.csv")
    test_rows = read_rows(split_dir / "test.csv")
    train_locations = {
        (float(row["latitude"]), float(row["longitude"])) for row in train_rows
    }
    test_locations = {
        (float(row["latitude"]), float(row["longitude"])) for row in test_rows
    }
    existing_test_directories = sum(
        (split_dir / str(row["id"])).is_dir() for row in test_rows
    )
    if existing_test_directories:
        raise RuntimeError(
            f"Test seal invalid: {existing_test_directories} location-test sample directories exist"
        )

    by_location, geographic = location_components(train_rows)
    development_groups, development = choose_development_groups(train_rows, by_location)
    write_manifest(manifest, train_rows, by_location, development_groups)
    if not ignored(root, manifest):
        raise ValueError("Bulk train/development manifest is not ignored by Git")

    report = {
        "schema_version": 1,
        "scope": "methanes2cm_v5_development_and_sealed_location_test_protocol",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "dataset": "H1deaki/MethaneS2CM",
            "revision": REVISION,
            "dataset_url": "https://huggingface.co/datasets/H1deaki/MethaneS2CM",
            "paper_doi": "10.1145/3711896.3737415",
            "license": "CC-BY-NC-4.0",
            "sensor": "Sentinel-2 MSI",
            "product_level": "L2A surface reflectance",
            "pixel_size_m": 20,
            "patch_shape": [32, 32],
            "tiff_page_band_order": list(S2_L2A_BAND_ORDER),
            "model_bands": list(MODEL_BANDS),
        },
        "metadata_identities": EXPECTED_IDENTITIES,
        "cohort": {
            "train": {
                "samples": len(train_rows),
                "positives": sum(int(row["label"]) for row in train_rows),
                "negatives": sum(int(row["label"]) == 0 for row in train_rows),
                "exact_locations": len(train_locations),
            },
            "test": {
                "samples": len(test_rows),
                "positives": sum(int(row["label"]) for row in test_rows),
                "negatives": sum(int(row["label"]) == 0 for row in test_rows),
                "exact_locations": len(test_locations),
            },
            "coordinate_overlap": len(train_locations & test_locations),
            "geographic_training_groups": geographic,
        },
        "development_protocol": {
            **development,
            "development_group_ids": sorted(development_groups),
            "manifest": {
                "path": manifest.relative_to(root).as_posix(),
                "bytes": manifest.stat().st_size,
                "sha256": sha256(manifest),
                "tracked": False,
            },
        },
        "input_contract": {
            "frames": ["T", "T-90", "T-365"],
            "reflectance_divisor": 10_000,
            "observability": "all six selected bands in all three frames are nonzero",
            "cloud": "per-pixel cloud mask unavailable; source paper filtered scenes above 20% cloud before cropping",
            "wind": "not present in the processed dataset and not used by v5",
            "test_comparator_missing_wind": [4.0, 4.0],
        },
        "sealed_test": {
            "image_directories_present_at_seal": existing_test_directories,
            "metadata_used_for_counts_and_group_audit": True,
            "images_opened": False,
            "rule": (
                "Do not extract or open any l2a_location_split_32x32 test imagery until the v5 "
                "architecture, selected checkpoints, ensemble, pixel threshold, scene calibration, "
                "and comparison code are checksum-bound in a clean tracked commit. Then run one "
                "test campaign; never retune from its outcomes."
            ),
        },
        "selection_metrics": {
            "scene": ["average_precision", "auroc", "recall_at_fpr_le_0.05"],
            "pixel": ["average_precision", "dice", "intersection_over_union"],
            "uncertainty": "2,000 nonparametric bootstrap resamples of frozen 25 km groups",
        },
        "interpretation_limits": [
            "balanced positive/negative crops do not represent deployment prevalence",
            "L2A-to-L1C product shift limits direct attribution in the MARS-S2L comparison",
            "processed metadata omits acquisition date, wind, and per-pixel cloud mask",
            "multiple crops per location require group-aware splits and confidence intervals",
        ],
        "provenance": {
            "sealed_from_git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True
            ).strip(),
            "script": Path(__file__).resolve().relative_to(root).as_posix(),
            "script_sha256": sha256(Path(__file__).resolve()),
            "tracked_worktree_dirty_at_start": False,
        },
    }
    write_json(output_json, report)
    write_markdown(output_markdown, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
