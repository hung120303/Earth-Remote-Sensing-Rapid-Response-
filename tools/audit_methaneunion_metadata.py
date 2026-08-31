#!/usr/bin/env python3
"""Audit MethaneUnion metadata and freeze a novel Sentinel-2 candidate inventory.

This tool reads manifests only. It never opens or downloads raster imagery. Novel
training rows must be farther than the requested boundary from every pinned
MethaneS2CM and MARS training/strict coordinate.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RELEASE_ROOT = Path(
    ".research/external/MethaneUnion-release/datasets/geo_split/480m_GSD"
)
DEFAULT_SOURCE = Path(
    ".research/external/MethaneUnion/preprocess_dataset_s2/"
    "CM_S2_L2A_gee90360_std512.csv"
)
DEFAULT_METHANES2CM_ROOT = Path(
    "EarthRemoteSensingRapidResponse/Data Collection/s2_emit_pairs/publication-v1/"
    "external/MethaneS2CM/l2a_location_split_32x32"
)
DEFAULT_MARS_ROOT = Path(
    "EarthRemoteSensingRapidResponse/Data Collection/s2_emit_pairs/publication-v1/"
    "external/MARS-S2L"
)
DEFAULT_JSON = Path("reports/acquisition/methaneunion_metadata_audit.json")
DEFAULT_CANDIDATES = Path("reports/acquisition/methaneunion_novel_s2_candidates.json")
MARS_MANIFESTS = (
    "publication_v3_training_samples.jsonl",
    "publication_v3_strict_samples.jsonl",
)
RELEASE_SPLITS = ("train", "test")
REQUIRED_S2_PATHS = (
    "S2_t0_path",
    "S2_pre_path",
    "S2_pre_pre_path",
    "S2_plume_label_path",
)


def resolve_input_path(value: str | Path) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    path = Path(os.path.abspath(candidate))
    if path != ROOT and ROOT not in path.parents:
        raise ValueError(f"Path escapes repository root: {value}")
    return path


def resolve_output_path(value: str | Path) -> Path:
    path = resolve_input_path(value)
    resolved = path.resolve()
    if resolved != ROOT and ROOT not in resolved.parents:
        raise ValueError(f"Output path resolves outside repository root: {value}")
    return path


def repo_relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.replace(path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def coordinate(row: Mapping[str, str], latitude: str, longitude: str) -> tuple[float, float]:
    point = (float(row[latitude]), float(row[longitude]))
    if not (-90 <= point[0] <= 90 and -180 <= point[1] <= 180):
        raise ValueError(f"Invalid coordinate: {point}")
    return point


def haversine_km(left: tuple[float, float], right: tuple[float, float]) -> float:
    latitude_left, latitude_right = map(math.radians, (left[0], right[0]))
    latitude_delta = math.radians(right[0] - left[0])
    longitude_delta = math.radians(right[1] - left[1])
    haversine = (
        math.sin(latitude_delta / 2) ** 2
        + math.cos(latitude_left)
        * math.cos(latitude_right)
        * math.sin(longitude_delta / 2) ** 2
    )
    return 12742.0 * math.asin(min(1.0, math.sqrt(haversine)))


def nearest_distance_km(
    point: tuple[float, float], references: Sequence[tuple[float, float]]
) -> float:
    if not references:
        raise ValueError("Nearest-distance reference set is empty")
    return min(haversine_km(point, reference) for reference in references)


def distance_summary(
    points: Iterable[tuple[float, float]],
    references: Sequence[tuple[float, float]],
) -> dict[str, float | int]:
    distances = sorted(nearest_distance_km(point, references) for point in set(points))
    if not distances:
        raise ValueError("Distance summary point set is empty")
    return {
        "coordinates": len(distances),
        "within_1km": sum(value <= 1.0 for value in distances),
        "within_25km": sum(value <= 25.0 for value in distances),
        "beyond_25km": sum(value > 25.0 for value in distances),
        "minimum_km": distances[0],
        "median_km": distances[len(distances) // 2],
        "maximum_km": distances[-1],
    }


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"CSV is empty: {path}")
    return rows


def load_methanes2cm_coordinates(root: Path) -> list[tuple[float, float]]:
    points: set[tuple[float, float]] = set()
    for split in RELEASE_SPLITS:
        for row in read_csv(root / f"{split}.csv"):
            points.add(coordinate(row, "latitude", "longitude"))
    return sorted(points)


def load_mars_coordinates(root: Path) -> list[tuple[float, float]]:
    points: set[tuple[float, float]] = set()
    for name in MARS_MANIFESTS:
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
                points.add((float(row["latitude"]), float(row["longitude"])))
    return sorted(points)


def has_sensor(row: Mapping[str, str], sensor: str) -> bool:
    return sensor in {value.strip() for value in row["available_sensor"].split(",")}


def select_novel_s2_rows(
    rows: Sequence[Mapping[str, str]],
    known: Sequence[tuple[float, float]],
    minimum_distance_km: float,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    observed_ids: set[int] = set()
    for row in rows:
        if not has_sensor(row, "S2"):
            continue
        row_id = int(row["id"])
        if row_id in observed_ids:
            raise ValueError(f"Duplicate released training row id: {row_id}")
        observed_ids.add(row_id)
        point = coordinate(row, "latitude", "longitude")
        nearest = nearest_distance_km(point, known)
        if nearest <= minimum_distance_km:
            continue
        missing = [name for name in REQUIRED_S2_PATHS if not row.get(name, "").strip()]
        if missing:
            raise ValueError(f"Novel S2 row {row_id} is missing paths: {missing}")
        selected.append(
            {
                "id": row_id,
                "label": int(row["label"]),
                "latitude": point[0],
                "longitude": point[1],
                "nearest_known_km": nearest,
                "available_sensor": row["available_sensor"],
                **{name: row[name] for name in REQUIRED_S2_PATHS},
            }
        )
    return sorted(selected, key=lambda row: row["id"])


def build_audit(
    release: Mapping[str, Sequence[Mapping[str, str]]],
    source_rows: Sequence[Mapping[str, str]],
    methanes2cm: Sequence[tuple[float, float]],
    mars: Sequence[tuple[float, float]],
    minimum_distance_km: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    known = sorted(set(methanes2cm) | set(mars))
    source_points = sorted(
        {
            coordinate(row, "plume_latitude", "plume_longitude")
            for row in source_rows
            if row.get("s2_0_std_512", "").strip()
        }
    )
    source_distance = distance_summary(source_points, methanes2cm)
    source_distance["exact_coordinate_overlap"] = len(set(source_points) & set(methanes2cm))
    source_distance["source_rows"] = len(source_rows)

    split_summary: dict[str, Any] = {}
    split_coordinates: dict[str, set[tuple[float, float]]] = {}
    for split, rows in release.items():
        split_coordinates[split] = {
            coordinate(row, "latitude", "longitude") for row in rows
        }
        split_summary[split] = {
            "rows": len(rows),
            "labels": {
                str(key): value
                for key, value in sorted(Counter(int(row["label"]) for row in rows).items())
            },
            "available_sensor": dict(
                sorted(Counter(row["available_sensor"] for row in rows).items())
            ),
            "unique_coordinates": len(split_coordinates[split]),
        }

    train_test_distance = distance_summary(
        split_coordinates["test"], sorted(split_coordinates["train"])
    )
    train_test_distance["exact_coordinate_overlap"] = len(
        split_coordinates["train"] & split_coordinates["test"]
    )

    label_novelty: dict[str, Any] = {}
    for label in (0, 1):
        points = {
            coordinate(row, "latitude", "longitude")
            for rows in release.values()
            for row in rows
            if has_sensor(row, "S2") and int(row["label"]) == label
        }
        methanes2cm_summary = distance_summary(points, methanes2cm)
        known_summary = distance_summary(points, known)
        label_novelty[str(label)] = {
            "unique_query_coordinates": len(points),
            "beyond_25km_methanes2cm": methanes2cm_summary["beyond_25km"],
            "beyond_25km_methanes2cm_and_mars": known_summary["beyond_25km"],
            "median_to_methanes2cm_km": methanes2cm_summary["median_km"],
            "median_to_known_km": known_summary["median_km"],
        }

    candidates = select_novel_s2_rows(release["train"], known, minimum_distance_km)
    candidate_coordinates: dict[str, set[tuple[float, float]]] = {"0": set(), "1": set()}
    for row in candidates:
        candidate_coordinates[str(row["label"])].add((row["latitude"], row["longitude"]))

    audit = {
        "schema_version": "ersrr.methaneunion.metadata-audit.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": "metadata only; no raster or model outcome access",
        "distance_contract_km": minimum_distance_km,
        "known_cohorts": {
            "methanes2cm_unique_coordinates": len(set(methanes2cm)),
            "mars_unique_coordinates": len(set(mars)),
            "combined_unique_coordinates": len(known),
        },
        "source_overlap_with_methanes2cm": source_distance,
        "released_geo_480m": {
            "splits": split_summary,
            "train_test_distance": train_test_distance,
            "s2_label_novelty": label_novelty,
        },
        "novel_training_candidates": {
            "rows": len(candidates),
            "label_rows": {
                str(key): value
                for key, value in sorted(Counter(row["label"] for row in candidates).items())
            },
            "label_unique_coordinates": {
                label: len(points) for label, points in candidate_coordinates.items()
            },
            "minimum_nearest_known_km": min(row["nearest_known_km"] for row in candidates),
        },
        "access_ledger": {
            "source_metadata_opened": True,
            "release_manifests_opened": ["geo_split/480m_GSD/train.csv", "geo_split/480m_GSD/test.csv"],
            "release_archives_opened": False,
            "raster_members_opened": False,
            "released_test_rows_used_for_candidate_selection": False,
            "model_outcomes_opened": False,
        },
    }
    return audit, candidates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-root", type=Path, default=DEFAULT_RELEASE_ROOT)
    parser.add_argument("--source-csv", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--methanes2cm-root", type=Path, default=DEFAULT_METHANES2CM_ROOT)
    parser.add_argument("--mars-root", type=Path, default=DEFAULT_MARS_ROOT)
    parser.add_argument("--minimum-distance-km", type=float, default=25.0)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output-candidates", type=Path, default=DEFAULT_CANDIDATES)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.minimum_distance_km <= 0:
        raise ValueError("minimum-distance-km must be positive")
    release_root = resolve_input_path(args.release_root)
    source_path = resolve_input_path(args.source_csv)
    methanes2cm_root = resolve_input_path(args.methanes2cm_root)
    mars_root = resolve_input_path(args.mars_root)
    output_json = resolve_output_path(args.output_json)
    output_candidates = resolve_output_path(args.output_candidates)

    release_paths = {
        split: release_root / f"{split}.csv" for split in RELEASE_SPLITS
    }
    release = {split: read_csv(path) for split, path in release_paths.items()}
    source_rows = read_csv(source_path)
    methanes2cm = load_methanes2cm_coordinates(methanes2cm_root)
    mars = load_mars_coordinates(mars_root)
    audit, candidates = build_audit(
        release, source_rows, methanes2cm, mars, args.minimum_distance_km
    )
    audit["inputs"] = {
        "source": {
            "path": repo_relative(source_path),
            "bytes": source_path.stat().st_size,
            "sha256": sha256(source_path),
        },
        "release": {
            split: {
                "path": repo_relative(path),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for split, path in release_paths.items()
        },
        "methanes2cm": {
            split: {
                "path": repo_relative(methanes2cm_root / f"{split}.csv"),
                "bytes": (methanes2cm_root / f"{split}.csv").stat().st_size,
                "sha256": sha256(methanes2cm_root / f"{split}.csv"),
            }
            for split in RELEASE_SPLITS
        },
        "mars": {
            name: {
                "path": repo_relative(mars_root / name),
                "bytes": (mars_root / name).stat().st_size,
                "sha256": sha256(mars_root / name),
            }
            for name in MARS_MANIFESTS
        },
    }

    grouped_candidates: dict[tuple[int, float, float], list[int]] = {}
    for row in candidates:
        key = (row["label"], row["latitude"], row["longitude"])
        grouped_candidates.setdefault(key, []).append(row["id"])
    candidate_inventory = {
        "schema_version": "ersrr.methaneunion.novel-s2-candidates.v1",
        "selection": {
            "split": "geo_split/480m_GSD/train.csv",
            "sensor_required": "S2",
            "minimum_distance_km_exclusive": args.minimum_distance_km,
            "known_coordinate_sets": [
                "MethaneS2CM train+test",
                "MARS-S2L publication v3 training+strict",
            ],
            "raster_accessed": False,
        },
        "source_manifest_sha256": audit["inputs"]["release"]["train"]["sha256"],
        "candidate_rows": len(candidates),
        "candidate_ids": [row["id"] for row in candidates],
        "coordinate_groups": [
            {
                "label": key[0],
                "latitude": key[1],
                "longitude": key[2],
                "row_ids": sorted(row_ids),
            }
            for key, row_ids in sorted(grouped_candidates.items())
        ],
    }
    atomic_write(output_json, json.dumps(audit, indent=2, sort_keys=True) + "\n")
    atomic_write(
        output_candidates, json.dumps(candidate_inventory, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {
                "output_json": repo_relative(output_json),
                "output_candidates": repo_relative(output_candidates),
                "candidate_rows": len(candidates),
                "candidate_unique_coordinates": audit["novel_training_candidates"][
                    "label_unique_coordinates"
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
