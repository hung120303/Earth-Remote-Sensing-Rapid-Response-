#!/usr/bin/env python3
"""Build a MethaneS2CM cohort spatially disjoint from the MARS paper test."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.neighbors import BallTree


DEFAULT_MARS_METADATA = Path(
    "EarthRemoteSensingRapidResponse/Data Collection/s2_emit_pairs/publication-v1/"
    "external/MARS-S2L-paper-source/validated_images_all_20251129.csv"
)
DEFAULT_SOURCE_MANIFEST = Path(
    "EarthRemoteSensingRapidResponse/Data Collection/s2_emit_pairs/publication-v1/"
    "external/MethaneS2CM/l2a_location_split_32x32/"
    "v5_train_development_manifest.jsonl"
)
DEFAULT_PACKED_DATA = Path(
    "EarthRemoteSensingRapidResponse/Data Collection/s2_emit_pairs/publication-v1/"
    "external/MethaneS2CM/l2a_location_split_32x32/v5_train_packed.h5"
)
DEFAULT_AUXILIARY = Path(
    ".research/methanes2cm_mars_disjoint/model_auxiliary_training.jsonl"
)
DEFAULT_DEVELOPMENT = Path(
    ".research/methanes2cm_mars_disjoint/model_development.jsonl"
)
DEFAULT_JSON = Path(
    "reports/acquisition/methanes2cm_mars_disjoint_model_manifest.json"
)
DEFAULT_MARKDOWN = Path(
    "reports/acquisition/METHANES2CM_MARS_DISJOINT_MODEL_MANIFEST.md"
)
EARTH_RADIUS_KM = 6371.0088
SOURCE_REVISION = "ee9a96d4994ca6bc45725c1e92d7a06258131eaf"
ROLE_MAP = {
    "internal_fitting": "auxiliary_training",
    "internal_development": "development",
}


def repo_root() -> Path:
    value = subprocess.check_output(
        ["git", "rev-parse", "--show-toplevel"], text=True
    ).strip()
    if not value:
        raise RuntimeError("Could not resolve repository root")
    return Path(value).resolve()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc


def read_mars_test_sites(path: Path) -> list[dict[str, Any]]:
    """Read only non-label identity/coordinate columns from the paper metadata."""
    coordinates: dict[str, tuple[float, float]] = {}
    with path.open("r", encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            if row["split_name"] != "test_2023":
                continue
            identifier = str(row["id_location"]).strip()
            point = (float(row["lat"]), float(row["lon"]))
            if not identifier or not np.all(np.isfinite(point)):
                raise ValueError("Invalid MARS paper-test site metadata")
            previous = coordinates.setdefault(identifier, point)
            if not np.allclose(previous, point, rtol=0.0, atol=1e-10):
                raise ValueError(f"MARS site coordinate changed within {identifier}")
    if len(coordinates) != 1289:
        raise ValueError(f"Expected 1,289 MARS paper-test sites, got {len(coordinates):,}")
    return [
        {"id_location": identifier, "latitude": point[0], "longitude": point[1]}
        for identifier, point in sorted(coordinates.items())
    ]


def nearest_site_distances_km(
    source_points: np.ndarray, target_points: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    source = np.asarray(source_points, dtype=np.float64)
    target = np.asarray(target_points, dtype=np.float64)
    if source.ndim != 2 or target.ndim != 2 or source.shape[1:] != (2,) or target.shape[1:] != (2,):
        raise ValueError("Coordinate arrays must have shape Nx2")
    if not len(source) or not len(target) or not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError("Coordinate arrays must be non-empty and finite")
    radians = np.deg2rad(target)
    angular_distance, indices = BallTree(radians, metric="haversine").query(
        np.deg2rad(source), k=1
    )
    return angular_distance[:, 0] * EARTH_RADIUS_KM, indices[:, 0]


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as target:
        for record in records:
            target.write(
                json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False)
                + "\n"
            )
    os.replace(temporary, path)


def safe_output(root: Path, value: str) -> Path:
    path = (root / value).resolve()
    if os.path.commonpath([str(root), str(path)]) != str(root):
        raise ValueError(f"Output escapes repository root: {value}")
    return path


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    summary = report["summary"]
    threshold = report["selection"]["minimum_distance_from_mars_test_km"]
    lines = [
        "# MethaneS2CM / MARS spatial-disjoint auxiliary cohort",
        "",
        f"Generated: {report['generated_at_utc']}.",
        "",
        f"- MARS v3 paper-test physical sites (coordinates only): **{summary['mars_test_sites']:,}**.",
        f"- MethaneS2CM source locations: **{summary['source_exact_locations']:,}** / **{summary['source_rows']:,}** crops.",
        f"- Locations farther than {threshold:g} km from every MARS paper-test site: **{summary['selected_exact_locations']:,}**.",
        f"- Auxiliary-training crops: **{summary['by_role']['auxiliary_training']:,}** across **{summary['groups_by_role']['auxiliary_training']:,}** frozen 25 km groups.",
        f"- Held source-development crops: **{summary['by_role']['development']:,}** across **{summary['groups_by_role']['development']:,}** frozen 25 km groups.",
        f"- Selected positive / no-plume crops: **{summary['by_label']['1']:,} / {summary['by_label']['0']:,}**.",
        "",
        "The exclusion reads no MARS model targets or pixel labels. It uses only the published test split name, physical-location identifier, latitude, and longitude. Roles preserve MethaneS2CM's pre-existing 25 km group-held fitting/development boundary. The large loader manifests and packed HDF5 stay ignored; this tracked receipt binds them by SHA-256.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mars-metadata", default=DEFAULT_MARS_METADATA.as_posix())
    parser.add_argument("--source-manifest", default=DEFAULT_SOURCE_MANIFEST.as_posix())
    parser.add_argument("--packed-data", default=DEFAULT_PACKED_DATA.as_posix())
    parser.add_argument("--minimum-distance-km", type=float, default=25.0)
    parser.add_argument("--auxiliary", default=DEFAULT_AUXILIARY.as_posix())
    parser.add_argument("--development", default=DEFAULT_DEVELOPMENT.as_posix())
    parser.add_argument("--output-json", default=DEFAULT_JSON.as_posix())
    parser.add_argument("--output-markdown", default=DEFAULT_MARKDOWN.as_posix())
    args = parser.parse_args()
    if args.minimum_distance_km <= 0:
        parser.error("minimum distance must be positive")

    root = repo_root()
    mars_path = (root / args.mars_metadata).resolve()
    source_path = (root / args.source_manifest).resolve()
    packed_path = (root / args.packed_data).resolve()
    mars_sites = read_mars_test_sites(mars_path)
    records = list(iter_jsonl(source_path))
    if len(records) != 80217:
        raise ValueError(f"Expected 80,217 MethaneS2CM train crops, got {len(records):,}")

    rows_by_location: dict[str, list[dict[str, Any]]] = defaultdict(list)
    coordinates: dict[str, tuple[float, float]] = {}
    for row in records:
        if str(row.get("source_revision")) != SOURCE_REVISION:
            raise ValueError("MethaneS2CM source revision changed")
        if str(row.get("research_role")) not in ROLE_MAP:
            raise ValueError("Unexpected MethaneS2CM research role")
        exact = str(row["exact_location_id"])
        point = (float(row["latitude"]), float(row["longitude"]))
        previous = coordinates.setdefault(exact, point)
        if not np.allclose(previous, point, rtol=0.0, atol=1e-10):
            raise ValueError(f"MethaneS2CM exact-location coordinate changed: {exact}")
        rows_by_location[exact].append(row)

    exact_locations = sorted(rows_by_location)
    source_points = np.asarray([coordinates[key] for key in exact_locations])
    mars_points = np.asarray(
        [[site["latitude"], site["longitude"]] for site in mars_sites]
    )
    distances, nearest_indices = nearest_site_distances_km(source_points, mars_points)
    selected_locations = {
        key
        for key, distance in zip(exact_locations, distances)
        if float(distance) > args.minimum_distance_km
    }
    distance_by_location = dict(zip(exact_locations, distances.astype(float)))
    nearest_by_location = {
        key: mars_sites[int(index)]["id_location"]
        for key, index in zip(exact_locations, nearest_indices)
    }

    outputs: dict[str, list[dict[str, Any]]] = {role: [] for role in ROLE_MAP.values()}
    for exact in sorted(selected_locations):
        for source_row in rows_by_location[exact]:
            original_role = str(source_row["research_role"])
            role = ROLE_MAP[original_role]
            output = dict(source_row)
            output.update(
                {
                    "research_role": role,
                    "source_research_role": original_role,
                    "minimum_mars_test_distance_km": distance_by_location[exact],
                    "nearest_mars_test_location_id": nearest_by_location[exact],
                    "spatial_exclusion_contract": (
                        f"strictly farther than {args.minimum_distance_km:g} km from every "
                        "MARS-S2L v3 paper-test physical-location centroid"
                    ),
                }
            )
            outputs[role].append(output)
    for values in outputs.values():
        values.sort(key=lambda row: int(row["id"]))

    auxiliary_path = safe_output(root, args.auxiliary)
    development_path = safe_output(root, args.development)
    write_jsonl(auxiliary_path, outputs["auxiliary_training"])
    write_jsonl(development_path, outputs["development"])
    selected_rows = outputs["auxiliary_training"] + outputs["development"]
    labels = Counter(str(row["label"]) for row in selected_rows)
    report = {
        "schema_version": 1,
        "status": "complete; imagery already local; loader manifests remain ignored",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "provenance": {
            "script": Path(__file__).resolve().relative_to(root).as_posix(),
            "script_sha256": sha256(Path(__file__).resolve()),
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True
            ).strip(),
        },
        "inputs": {
            "mars_metadata": {
                "path": args.mars_metadata,
                "sha256": sha256(mars_path),
                "columns_accessed": ["split_name", "id_location", "lat", "lon"],
                "labels_accessed": False,
            },
            "source_manifest": {
                "path": args.source_manifest,
                "sha256": sha256(source_path),
            },
            "packed_data": {
                "path": args.packed_data,
                "bytes": packed_path.stat().st_size,
                "sha256": sha256(packed_path),
                "tracked": False,
            },
            "source_dataset": "H1deaki/MethaneS2CM L2A location split train",
            "source_revision": SOURCE_REVISION,
            "license": "CC-BY-NC-4.0",
        },
        "selection": {
            "minimum_distance_from_mars_test_km": args.minimum_distance_km,
            "distance": "great-circle haversine distance between physical-location centroids",
            "strict_comparison": ">",
            "mars_scope": "all 1,289 physical locations in the exact MARS-S2L v3 43,529-row paper test",
            "source_role_policy": ROLE_MAP,
        },
        "artifacts": {
            "auxiliary_training": {
                "path": args.auxiliary,
                "bytes": auxiliary_path.stat().st_size,
                "sha256": sha256(auxiliary_path),
                "tracked": False,
            },
            "development": {
                "path": args.development,
                "bytes": development_path.stat().st_size,
                "sha256": sha256(development_path),
                "tracked": False,
            },
        },
        "summary": {
            "mars_test_sites": len(mars_sites),
            "source_rows": len(records),
            "source_exact_locations": len(exact_locations),
            "selected_rows": len(selected_rows),
            "selected_exact_locations": len(selected_locations),
            "excluded_rows": len(records) - len(selected_rows),
            "excluded_exact_locations": len(exact_locations) - len(selected_locations),
            "by_role": {role: len(values) for role, values in sorted(outputs.items())},
            "groups_by_role": {
                role: len({str(row["group_id"]) for row in values})
                for role, values in sorted(outputs.items())
            },
            "exact_locations_by_role": {
                role: len({str(row["exact_location_id"]) for row in values})
                for role, values in sorted(outputs.items())
            },
            "by_label": dict(sorted(labels.items())),
            "minimum_selected_distance_km": float(
                min(distance_by_location[key] for key in selected_locations)
            ),
            "maximum_excluded_distance_km": float(
                max(
                    distance_by_location[key]
                    for key in exact_locations
                    if key not in selected_locations
                )
            ),
        },
        "invariants": [
            "No MARS paper-test model target, image, mask, prediction, or label is read.",
            (
                "Every selected MethaneS2CM exact location is strictly farther than "
                f"{args.minimum_distance_km:g} km from every MARS paper-test physical location."
            ),
            "MethaneS2CM's existing 25 km group-held source-development boundary is preserved.",
            "The 4.25 GB packed HDF5 and generated row manifests remain ignored and are not committed.",
        ],
    }
    output_json = safe_output(root, args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown = safe_output(root, args.output_markdown)
    write_markdown(markdown, report)
    print(json.dumps({"ok": True, **report["summary"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
