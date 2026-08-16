#!/usr/bin/env python3
"""Build an outcome-blind prior-reference bank for MARS folds 3/4."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

import numpy as np
import rasterio
from rasterio.enums import Resampling


ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "EarthRemoteSensingRapidResponse"
if str(MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_ROOT))

from mars_s2l_adapter import MARS_IMAGE_BANDS, validate_image_band_order  # noqa: E402


DEFAULT_PROTOCOL = Path("configs/mars_prior_reference_bank_protocol.json")
DEFAULT_CACHE = Path(".research/mars_prior_reference_bank/folds34_descriptors.npz")
DEFAULT_SELECTION = Path(".research/mars_prior_reference_bank/folds34_selection.jsonl")
DEFAULT_JSON = Path("reports/acquisition/mars_prior_reference_bank_folds34.json")
DEFAULT_MARKDOWN = Path("reports/acquisition/MARS_PRIOR_REFERENCE_BANK_FOLDS34.md")
FROZEN_STATUS = "frozen_before_real_descriptor_extraction_or_outcome_access"
SENTINEL_PATTERN = re.compile(r"_T(?P<tile>[0-9]{2}[A-Z]{3})_")
LANDSAT_PATTERN = re.compile(
    r"^(?:LC|LO|LE|LT)[0-9]{2}_[A-Z0-9]+_(?P<pathrow>[0-9]{6})_"
)
VISIBLE_NIR_BANDS = ("B02", "B03", "B04", "B08")
VISIBLE_NIR_INDEXES = (1, 2, 3, 4)
ORIGINAL_REFERENCE_INDEXES = (7, 8, 9, 10)
LABEL_FIELDS = frozenset(
    {
        "label",
        "label_state",
        "isplume",
        "plume",
        "presence",
        "plume_mask",
        "ch4_fluxrate",
        "ch4_fluxrate_std",
    }
)
ALLOWED_RECORD_FIELDS = frozenset(
    {
        "band_order",
        "crs",
        "group_id",
        "height",
        "percentage_clear",
        "physical_location_id",
        "sample_id",
        "satellite",
        "sensor_family",
        "target_datetime",
        "target_scene_id",
        "width",
    }
)


def repo_root() -> Path:
    return Path(
        subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], text=True
        ).strip()
    ).resolve()


def repo_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_protocol(path: Path) -> dict[str, Any]:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("status") != FROZEN_STATUS:
        raise ValueError("Prior-reference-bank protocol is not frozen")
    if protocol.get("outcome_access", {}).get("labels_permitted") is not False:
        raise ValueError("Protocol does not prohibit outcome access")
    selection = protocol.get("selection")
    if not isinstance(selection, dict):
        raise ValueError("Protocol lacks a frozen selection contract")
    if selection.get("descriptor_bands") != list(VISIBLE_NIR_BANDS):
        raise ValueError("Frozen descriptor bands differ from the implementation")
    if selection.get("candidate_sensor") != "Sentinel-2":
        raise ValueError("Frozen pilot must route only Sentinel-2 references")
    if int(selection["recent_pool_size"]) < int(selection["selected_references"]):
        raise ValueError("Recent pool is smaller than the selected reference set")
    if float(selection["minimum_percentage_clear"]) < 0.0:
        raise ValueError("Minimum clear percentage is invalid")
    return protocol


def safe_asset_path(base_dir: Path, value: str) -> Path:
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe asset path: {value}")
    base = base_dir.resolve()
    result = (base / Path(*relative.parts)).resolve()
    if base != result and base not in result.parents:
        raise ValueError(f"Asset escapes the MARS root: {value}")
    return result


def image_asset(record: dict[str, Any]) -> str:
    values = [str(item["path"]) for item in record["assets"] if item["role"] == "image"]
    if len(values) != 1:
        raise ValueError(f"Expected one image asset for {record['sample_id']}")
    return values[0]


def scene_grid_key(scene_id: str) -> str:
    sentinel = SENTINEL_PATTERN.search(scene_id)
    if sentinel is not None:
        return f"S2:{sentinel.group('tile')}"
    landsat = LANDSAT_PATTERN.search(scene_id)
    if landsat is not None:
        return f"Landsat:{landsat.group('pathrow')}"
    raise ValueError(f"Cannot infer spatial tile/path-row from scene ID: {scene_id}")


def select_record_fields(record: dict[str, Any], fold: int) -> dict[str, Any]:
    if LABEL_FIELDS.intersection(ALLOWED_RECORD_FIELDS):
        raise RuntimeError("Label fields entered the allowed reference-bank schema")
    selected = {key: record[key] for key in ALLOWED_RECORD_FIELDS}
    selected["fold"] = int(fold)
    selected["grid_key"] = scene_grid_key(str(selected["target_scene_id"]))
    selected["image_asset"] = image_asset(record)
    return selected


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {line_number}") from exc


def descriptor(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 3 or array.shape[0] != len(VISIBLE_NIR_BANDS):
        raise ValueError("Descriptor input must be four bands by height by width")
    medians = np.empty(array.shape[0], dtype=np.float32)
    normalized = np.zeros_like(array, dtype=np.float32)
    for index, band in enumerate(array):
        valid = np.isfinite(band) & (band > 0.0)
        median = float(np.median(band[valid])) if np.any(valid) else 1.0
        if not np.isfinite(median) or median <= 1e-8:
            median = 1.0
        medians[index] = median
        normalized[index, valid] = np.clip(band[valid] / median, 0.0, 4.0)
    return normalized.reshape(-1), medians


def grid_signature(source: rasterio.io.DatasetReader) -> tuple[str, tuple[float, ...]]:
    crs = "" if source.crs is None else source.crs.to_string()
    transform = tuple(round(float(value), 9) for value in tuple(source.transform)[:6])
    return crs, transform


def read_scene_descriptors(
    path: Path,
    record: dict[str, Any],
    *,
    descriptor_size: int,
) -> dict[str, Any]:
    with rasterio.open(path) as source:
        if source.count != len(MARS_IMAGE_BANDS):
            raise ValueError(f"Expected 12 image bands in {path}")
        validate_image_band_order(record, source.descriptions)
        indexes = VISIBLE_NIR_INDEXES + ORIGINAL_REFERENCE_INDEXES
        values = source.read(
            indexes=indexes,
            out_shape=(len(indexes), descriptor_size, descriptor_size),
            resampling=Resampling.average,
        ).astype(np.float32)
        crs, transform = grid_signature(source)
    target_descriptor, target_medians = descriptor(values[:4])
    reference_descriptor, reference_medians = descriptor(values[4:])
    if not all(
        np.isfinite(item).all()
        for item in (
            target_descriptor,
            target_medians,
            reference_descriptor,
            reference_medians,
        )
    ):
        raise ValueError(f"Non-finite descriptor produced for {record['sample_id']}")
    return {
        "target_descriptor": target_descriptor,
        "target_medians": target_medians,
        "reference_descriptor": reference_descriptor,
        "reference_medians": reference_medians,
        "crs": crs,
        "transform": transform,
    }


def reference_distance(
    target_descriptor: np.ndarray,
    target_medians: np.ndarray,
    candidate_descriptor: np.ndarray,
    candidate_medians: np.ndarray,
    *,
    radiometric_weight: float,
) -> np.ndarray:
    target = np.asarray(target_descriptor, dtype=np.float32)
    candidates = np.asarray(candidate_descriptor, dtype=np.float32)
    if candidates.ndim == target.ndim:
        candidates = candidates[None]
    candidate_scale = np.asarray(candidate_medians, dtype=np.float32)
    if candidate_scale.ndim == 1:
        candidate_scale = candidate_scale[None]
    texture = np.mean(np.abs(candidates - target[None]), axis=1)
    safe_target = np.clip(np.asarray(target_medians, dtype=np.float32), 1e-6, None)
    safe_candidate = np.clip(candidate_scale, 1e-6, None)
    radiometric = np.mean(np.abs(np.log(safe_candidate / safe_target[None])), axis=1)
    return texture + float(radiometric_weight) * radiometric


def parse_timestamp(value: str) -> float:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError(f"Target datetime is not timezone-aware: {value}")
    return parsed.timestamp()


def select_prior_references(
    metadata: list[dict[str, Any]],
    target_descriptors: np.ndarray,
    target_medians: np.ndarray,
    reference_descriptors: np.ndarray,
    reference_medians: np.ndarray,
    *,
    minimum_percentage_clear: float,
    recent_pool_size: int,
    selected_references: int,
    radiometric_weight: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if len(metadata) != len(target_descriptors):
        raise ValueError("Metadata and descriptor rows are not aligned")
    groups: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(metadata):
        key = (
            str(row["physical_location_id"]),
            str(row["sensor_family"]),
            str(row["grid_key"]),
        )
        groups[key].append(index)
    for indices in groups.values():
        indices.sort(
            key=lambda index: (
                metadata[index]["timestamp"],
                metadata[index]["sample_id"],
            )
        )

    selections: list[dict[str, Any]] = [dict() for _ in metadata]
    raw_prior_counts = [0] * len(metadata)
    exact_grid_counts = [0] * len(metadata)
    selected_counts = [0] * len(metadata)
    best_distances: list[float] = []
    original_distances = [float("nan")] * len(metadata)
    closer_than_original: list[bool] = []
    sentinel_rows = 0

    for indices in groups.values():
        for position, row_index in enumerate(indices):
            row = metadata[row_index]
            is_sentinel = row["sensor_family"] == "Sentinel-2"
            if is_sentinel:
                sentinel_rows += 1
            earlier = [
                candidate_index
                for candidate_index in indices[:position]
                if metadata[candidate_index]["timestamp"] < row["timestamp"]
                and metadata[candidate_index]["target_scene_id"]
                != row["target_scene_id"]
                and float(metadata[candidate_index]["percentage_clear"])
                >= minimum_percentage_clear
            ]
            exact = [
                candidate_index
                for candidate_index in earlier
                if metadata[candidate_index]["crs"] == row["crs"]
                and metadata[candidate_index]["transform"] == row["transform"]
            ]
            raw_prior_counts[row_index] = len(earlier) if is_sentinel else 0
            exact_grid_counts[row_index] = len(exact) if is_sentinel else 0
            candidates = exact[-recent_pool_size:] if is_sentinel else []
            original = float(
                reference_distance(
                    target_descriptors[row_index],
                    target_medians[row_index],
                    reference_descriptors[row_index],
                    reference_medians[row_index],
                    radiometric_weight=radiometric_weight,
                )[0]
            )
            original_distances[row_index] = original
            selected_indices: list[int] = []
            selected_distances: list[float] = []
            if candidates:
                distances = reference_distance(
                    target_descriptors[row_index],
                    target_medians[row_index],
                    target_descriptors[candidates],
                    target_medians[candidates],
                    radiometric_weight=radiometric_weight,
                )
                order = np.argsort(distances, kind="stable")[:selected_references]
                selected_indices = [candidates[int(index)] for index in order]
                selected_distances = [float(distances[int(index)]) for index in order]
                best_distances.append(selected_distances[0])
                closer_than_original.append(selected_distances[0] < original)
            selected_counts[row_index] = len(selected_indices)
            selections[row_index] = {
                "sample_id": row["sample_id"],
                "fold": int(row["fold"]),
                "sensor_family": row["sensor_family"],
                "physical_location_id": row["physical_location_id"],
                "grid_key": row["grid_key"],
                "target_datetime": row["target_datetime"],
                "original_reference_distance": original,
                "strictly_prior_clear_candidates": len(earlier),
                "exact_grid_candidates": len(exact),
                "recent_pool_candidates": len(candidates),
                "selected_sample_ids": [
                    metadata[index]["sample_id"] for index in selected_indices
                ],
                "selected_target_scene_ids": [
                    metadata[index]["target_scene_id"] for index in selected_indices
                ],
                "selected_distances": selected_distances,
                "fallback_to_original_only": not selected_indices,
            }
    sentinel_selected = [
        selected_counts[index]
        for index, row in enumerate(metadata)
        if row["sensor_family"] == "Sentinel-2"
    ]
    sentinel_raw = [
        raw_prior_counts[index]
        for index, row in enumerate(metadata)
        if row["sensor_family"] == "Sentinel-2"
    ]
    sentinel_exact = [
        exact_grid_counts[index]
        for index, row in enumerate(metadata)
        if row["sensor_family"] == "Sentinel-2"
    ]

    def percentiles(values: list[int | float]) -> dict[str, float]:
        if not values:
            return {}
        array = np.asarray(values, dtype=np.float64)
        return {
            str(percentile): float(np.percentile(array, percentile))
            for percentile in (0, 25, 50, 75, 90, 95, 99, 100)
        }

    summary = {
        "sentinel_rows": sentinel_rows,
        "sentinel_rows_with_any_strict_prior_clear_candidate": sum(
            count > 0 for count in sentinel_raw
        ),
        "sentinel_rows_with_any_exact_grid_candidate": sum(
            count > 0 for count in sentinel_exact
        ),
        "sentinel_rows_with_selected_reference": sum(
            count > 0 for count in sentinel_selected
        ),
        "sentinel_rows_with_five_selected_references": sum(
            count >= selected_references for count in sentinel_selected
        ),
        "raw_prior_candidate_percentiles": percentiles(sentinel_raw),
        "exact_grid_candidate_percentiles": percentiles(sentinel_exact),
        "selected_reference_count_percentiles": percentiles(sentinel_selected),
        "best_selected_distance_percentiles": percentiles(best_distances),
        "original_reference_distance_percentiles_all_rows": percentiles(
            original_distances
        ),
        "best_selected_closer_than_original_rows": sum(closer_than_original),
        "best_selected_compared_rows": len(closer_than_original),
        "best_selected_closer_than_original_fraction": (
            float(np.mean(closer_than_original)) if closer_than_original else None
        ),
    }
    return selections, summary


def atomic_savez(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as target:
        np.savez_compressed(target, **arrays)
    os.replace(temporary, path)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as target:
        for row in rows:
            target.write(json.dumps(row, sort_keys=True) + "\n")
    os.replace(temporary, path)


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    summary = report["summary"]
    selected_fraction = (
        summary["sentinel_rows_with_selected_reference"] / summary["sentinel_rows"]
        if summary["sentinel_rows"]
        else 0.0
    )
    lines = [
        "# MARS prior-reference-bank audit: folds 3/4",
        "",
        f"Generated: {report['generated_at_utc']}.",
        "",
        "## Outcome-blind feasibility",
        "",
        (
            f"The frozen selector found at least one strictly prior, >=95% clear, exact-grid "
            f"Sentinel-2 reference for {summary['sentinel_rows_with_selected_reference']:,}/"
            f"{summary['sentinel_rows']:,} rows ({selected_fraction:.2%})."
        ),
        "",
        (
            f"A full five-reference set is available for "
            f"{summary['sentinel_rows_with_five_selected_references']:,} Sentinel-2 rows. "
            "Landsat remains an exact champion identity in the proposed pilot."
        ),
        "",
        "| Fold / sensor | Rows | Any selected reference | Five references |",
        "|---|---:|---:|---:|",
    ]
    for key, values in report["by_fold_sensor"].items():
        lines.append(
            f"| {key} | {values['rows']:,} | {values['with_selected_reference']:,} | "
            f"{values['with_five_references']:,} |"
        )
    lines.extend(
        [
            "",
            "No label, plume mask, flux, model score, prediction, or held outcome was read. "
            "The bank remains an ignored data artifact and is not yet authorized for model selection.",
            "",
            "## Research basis",
            "",
            (
                "Project Eucalyptus identifies poor or methane-contaminated reference scenes as a "
                "false-negative mechanism and explicitly recommends averages over the last 5/10 "
                "overpasses, similarity-based selection, or learned attention. This audit tests only "
                "whether the first two ingredients are locally feasible; it makes no performance claim."
            ),
            "",
            "## Decision",
            "",
            report["decision"],
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run(
    protocol_path: Path,
    cache_path: Path,
    selection_path: Path,
    json_path: Path,
    markdown_path: Path,
) -> dict[str, Any]:
    root = repo_root()
    protocol = load_protocol(protocol_path)
    inputs = protocol["inputs"]
    manifest_path = repo_path(root, inputs["development_manifest"]["path"])
    folds_path = repo_path(root, inputs["fold_assignments"]["path"])
    mars_root = repo_path(root, inputs["mars_root"])
    if sha256(manifest_path) != inputs["development_manifest"]["sha256"]:
        raise ValueError("Development manifest differs from the frozen protocol")
    if sha256(folds_path) != inputs["fold_assignments"]["sha256"]:
        raise ValueError("Fold assignment differs from the frozen protocol")
    assignments = {
        str(item["group_id"]): int(item["fold"])
        for item in json.loads(folds_path.read_text(encoding="utf-8"))["assignments"]
    }
    requested_folds = set(int(value) for value in protocol["folds"])
    records: list[dict[str, Any]] = []
    for raw in iter_jsonl(manifest_path):
        fold = assignments[str(raw["group_id"])]
        if fold in requested_folds:
            records.append(select_record_fields(raw, fold))
    if not records:
        raise ValueError("No records match the frozen folds")
    if len({row["sample_id"] for row in records}) != len(records):
        raise ValueError("Reference-bank sample IDs are not unique")

    selection = protocol["selection"]
    descriptor_size = int(selection["descriptor_size"])
    descriptor_width = len(VISIBLE_NIR_BANDS) * descriptor_size * descriptor_size
    target_descriptors = np.empty((len(records), descriptor_width), dtype=np.float32)
    reference_descriptors = np.empty_like(target_descriptors)
    target_medians = np.empty((len(records), len(VISIBLE_NIR_BANDS)), dtype=np.float32)
    reference_medians = np.empty_like(target_medians)
    metadata: list[dict[str, Any]] = []
    descriptor_workers = int(protocol["execution"]["descriptor_workers"])
    if descriptor_workers <= 0:
        raise ValueError("Descriptor workers must be positive")

    def extract(record: dict[str, Any]) -> dict[str, Any]:
        path = safe_asset_path(mars_root, record["image_asset"])
        return read_scene_descriptors(path, record, descriptor_size=descriptor_size)

    # Each worker opens its own immutable raster. Executor.map preserves record
    # order, so concurrency cannot change cache alignment or selection ties.
    with ThreadPoolExecutor(max_workers=descriptor_workers) as executor:
        extracted_rows = executor.map(extract, records)
        for index, (record, extracted) in enumerate(zip(records, extracted_rows)):
            target_descriptors[index] = extracted["target_descriptor"]
            reference_descriptors[index] = extracted["reference_descriptor"]
            target_medians[index] = extracted["target_medians"]
            reference_medians[index] = extracted["reference_medians"]
            metadata.append(
                {
                    **record,
                    "timestamp": parse_timestamp(str(record["target_datetime"])),
                    "crs": extracted["crs"],
                    "transform": list(extracted["transform"]),
                }
            )
            if (index + 1) % 1000 == 0:
                print(json.dumps({"descriptor_rows": index + 1}), flush=True)

    selections, summary = select_prior_references(
        metadata,
        target_descriptors,
        target_medians,
        reference_descriptors,
        reference_medians,
        minimum_percentage_clear=float(selection["minimum_percentage_clear"]),
        recent_pool_size=int(selection["recent_pool_size"]),
        selected_references=int(selection["selected_references"]),
        radiometric_weight=float(selection["radiometric_weight"]),
    )
    by_fold_sensor: dict[str, dict[str, int]] = {}
    for fold in sorted(requested_folds):
        for sensor in ("Sentinel-2", "Landsat"):
            indices = [
                index
                for index, row in enumerate(metadata)
                if int(row["fold"]) == fold and row["sensor_family"] == sensor
            ]
            selected_counts = [
                len(selections[index]["selected_sample_ids"]) for index in indices
            ]
            by_fold_sensor[f"fold {fold} / {sensor}"] = {
                "rows": len(indices),
                "with_selected_reference": sum(count > 0 for count in selected_counts),
                "with_five_references": sum(
                    count >= int(selection["selected_references"])
                    for count in selected_counts
                ),
            }

    atomic_savez(
        cache_path,
        sample_ids=np.asarray([row["sample_id"] for row in metadata]),
        target_descriptors=target_descriptors.astype(np.float16),
        reference_descriptors=reference_descriptors.astype(np.float16),
        target_medians=target_medians,
        reference_medians=reference_medians,
    )
    write_jsonl(selection_path, selections)
    coverage = (
        summary["sentinel_rows_with_selected_reference"] / summary["sentinel_rows"]
    )
    gates = protocol["feasibility_gates"]
    passed = (
        coverage >= float(gates["minimum_sentinel_reference_coverage"])
        and summary["sentinel_rows_with_any_exact_grid_candidate"]
        == summary["sentinel_rows_with_any_strict_prior_clear_candidate"]
    )
    decision = (
        "PASS: authorize a separate, preregistered alternate-reference score extraction and "
        "complementarity diagnostic; do not train or select a model from this audit."
        if passed
        else "FAIL: do not run alternate-reference model inference."
    )
    report = {
        "schema_version": 1,
        "status": "complete_outcome_blind_prior_reference_feasibility_audit",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": protocol["scope"],
        "summary": summary,
        "by_fold_sensor": by_fold_sensor,
        "feasibility_gates_pass": passed,
        "decision": decision,
        "outcome_access": {
            "record_fields_materialized": sorted(ALLOWED_RECORD_FIELDS),
            "label_fields_accessed": [],
            "labels_accessed": False,
            "plume_masks_accessed": False,
            "model_scores_accessed": False,
            "predictions_computed": False,
        },
        "artifacts": {
            "descriptor_cache": {
                "path": cache_path.relative_to(root).as_posix(),
                "bytes": cache_path.stat().st_size,
                "sha256": sha256(cache_path),
                "tracked": False,
            },
            "selection_manifest": {
                "path": selection_path.relative_to(root).as_posix(),
                "bytes": selection_path.stat().st_size,
                "sha256": sha256(selection_path),
                "tracked": False,
            },
        },
        "provenance": {
            "protocol": {
                "path": protocol_path.relative_to(root).as_posix(),
                "sha256": sha256(protocol_path),
            },
            "script": {
                "path": Path(__file__).resolve().relative_to(root).as_posix(),
                "sha256": sha256(Path(__file__).resolve()),
            },
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True
            ).strip(),
        },
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_markdown(markdown_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL.as_posix())
    parser.add_argument("--cache", default=DEFAULT_CACHE.as_posix())
    parser.add_argument("--selection", default=DEFAULT_SELECTION.as_posix())
    parser.add_argument("--output-json", default=DEFAULT_JSON.as_posix())
    parser.add_argument("--output-markdown", default=DEFAULT_MARKDOWN.as_posix())
    args = parser.parse_args()
    root = repo_root()
    report = run(
        repo_path(root, args.protocol),
        repo_path(root, args.cache),
        repo_path(root, args.selection),
        repo_path(root, args.output_json),
        repo_path(root, args.output_markdown),
    )
    print(
        json.dumps(
            {
                "ok": True,
                "pass": report["feasibility_gates_pass"],
                "sentinel_rows": report["summary"]["sentinel_rows"],
                "with_reference": report["summary"][
                    "sentinel_rows_with_selected_reference"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
