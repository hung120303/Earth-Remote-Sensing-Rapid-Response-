#!/usr/bin/env python3
"""Audit the public Zhao et al. DSAN archive before any model use."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import struct
import subprocess
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable


DEFAULT_PROTOCOL = Path("configs/zhao_dsan_2025_acquisition_audit_protocol.json")
DEFAULT_JSON = Path("reports/acquisition/zhao_dsan_2025_dataset_audit.json")
DEFAULT_MARKDOWN = Path("reports/acquisition/ZHAO_DSAN_2025_DATASET_AUDIT.md")
FROZEN_STATUS = (
    "frozen_after_public_archive_download_before_extraction_image_decoding_or_model_use"
)
PNG_PATTERN = re.compile(
    r"Dataset#(?P<dataset>[1-9][0-9]*)/"
    r"(?P<label>plume-containing|plume-free)/"
    r"D(?P<filename_dataset>[1-9][0-9]*)_(?P<date>[0-9]{8})[.]png"
)
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
EARTH_RADIUS_KM = 6371.0088


def repo_root() -> Path:
    return Path(
        subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip()
    ).resolve()


def repo_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def file_digest(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256(path: Path) -> str:
    return file_digest(path, "sha256")


def haversine_km(
    latitude_a: float,
    longitude_a: float,
    latitude_b: float,
    longitude_b: float,
) -> float:
    lat_a = math.radians(latitude_a)
    lat_b = math.radians(latitude_b)
    delta_lat = lat_b - lat_a
    delta_lon = math.radians(longitude_b - longitude_a)
    value = (
        math.sin(delta_lat / 2.0) ** 2
        + math.cos(lat_a) * math.cos(lat_b) * math.sin(delta_lon / 2.0) ** 2
    )
    return 2.0 * EARTH_RADIUS_KM * math.asin(min(1.0, math.sqrt(value)))


def load_protocol(path: Path) -> dict[str, Any]:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("status") != FROZEN_STATUS:
        raise ValueError("Zhao acquisition-audit protocol is not frozen")
    files = protocol.get("files")
    table = protocol.get("official_table_contract")
    overlap = protocol.get("mars_overlap_contract")
    if not isinstance(files, dict) or not isinstance(table, dict):
        raise ValueError("Protocol lacks frozen file or table contracts")
    if not isinstance(overlap, dict) or overlap.get("labels_permitted") is not False:
        raise ValueError("Protocol does not enforce a label-free MARS overlap audit")
    sites = table.get("sites")
    if not isinstance(sites, list) or not sites:
        raise ValueError("Protocol lacks frozen Zhao sites")
    indices = [int(site["dataset_index"]) for site in sites]
    if len(indices) != len(set(indices)):
        raise ValueError("Frozen Zhao dataset indices are not unique")
    positives = sum(int(site["plume_containing"]) for site in sites)
    negatives = sum(int(site["plume_free"]) for site in sites)
    if positives != int(table["expected_plume_containing"]):
        raise ValueError("Frozen positive counts do not sum to the table total")
    if negatives != int(table["expected_plume_free"]):
        raise ValueError("Frozen negative counts do not sum to the table total")
    if positives + negatives != int(table["expected_total_rows"]):
        raise ValueError("Frozen class counts do not sum to the table total")
    for site in sites:
        validate_coordinate(float(site["latitude"]), float(site["longitude"]))
        for alternate in site.get("alternate_coordinates", []):
            validate_coordinate(
                float(alternate["latitude"]), float(alternate["longitude"])
            )
    threshold = float(overlap["distance_threshold_km"])
    if threshold <= 0.0:
        raise ValueError("Frozen distance threshold must be positive")
    return protocol


def validate_coordinate(latitude: float, longitude: float) -> None:
    if not (-90.0 <= latitude <= 90.0 and -180.0 <= longitude <= 180.0):
        raise ValueError(f"Coordinate is out of range: {(latitude, longitude)}")


def verify_frozen_file(root: Path, contract: dict[str, Any]) -> dict[str, Any]:
    path = repo_path(root, str(contract["path"]))
    if not path.is_file():
        raise FileNotFoundError(path)
    actual_bytes = path.stat().st_size
    actual_sha256 = sha256(path)
    if actual_bytes != int(contract["expected_bytes"]):
        raise ValueError(f"Unexpected byte size for {path}")
    if actual_sha256.lower() != str(contract["expected_sha256"]).lower():
        raise ValueError(f"Unexpected SHA-256 for {path}")
    result = {
        "path": path.relative_to(root).as_posix(),
        "bytes": actual_bytes,
        "sha256": actual_sha256,
    }
    expected_md5 = contract.get("expected_md5")
    if expected_md5 is not None:
        actual_md5 = file_digest(path, "md5")
        if actual_md5.lower() != str(expected_md5).lower():
            raise ValueError(f"Unexpected MD5 for {path}")
        result["md5"] = actual_md5
    return result


def safe_zip_member(name: str) -> PurePosixPath:
    normalized = name.replace("\\", "/")
    path = PurePosixPath(normalized)
    if not normalized or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe ZIP member path: {name!r}")
    if path.parts and ":" in path.parts[0]:
        raise ValueError(f"Unsafe drive-qualified ZIP member path: {name!r}")
    return path


def parse_png_header(data: bytes, member: str) -> dict[str, int]:
    if len(data) < 33 or data[:8] != PNG_SIGNATURE:
        raise ValueError(f"Invalid PNG signature/header: {member}")
    chunk_length = struct.unpack(">I", data[8:12])[0]
    if chunk_length != 13 or data[12:16] != b"IHDR":
        raise ValueError(f"PNG does not start with a valid IHDR chunk: {member}")
    width, height, depth, color, compression, filtering, interlace = struct.unpack(
        ">IIBBBBB", data[16:29]
    )
    if not width or not height:
        raise ValueError(f"PNG has an empty raster: {member}")
    return {
        "width": width,
        "height": height,
        "bit_depth": depth,
        "color_type": color,
        "compression": compression,
        "filter": filtering,
        "interlace": interlace,
    }


def audit_archive(
    archive_path: Path,
    table_contract: dict[str, Any],
) -> dict[str, Any]:
    expected_by_site = {
        int(site["dataset_index"]): {
            "plume-containing": int(site["plume_containing"]),
            "plume-free": int(site["plume_free"]),
        }
        for site in table_contract["sites"]
    }
    counts: Counter[tuple[int, str]] = Counter()
    dimensions: Counter[str] = Counter()
    png_formats: Counter[str] = Counter()
    date_ranges: dict[tuple[int, str], list[str]] = defaultdict(list)
    labels_by_site_date: dict[tuple[int, str], set[str]] = defaultdict(set)
    unexpected_files: list[str] = []
    image_members = 0
    directory_members = 0
    uncompressed_bytes = 0

    with zipfile.ZipFile(archive_path) as archive:
        infos = archive.infolist()
        names = [info.filename.replace("\\", "/") for info in infos]
        if len(names) != len(set(names)):
            raise ValueError("ZIP contains duplicate member names")
        for info in infos:
            member = safe_zip_member(info.filename).as_posix()
            if info.is_dir():
                directory_members += 1
                continue
            match = PNG_PATTERN.fullmatch(member)
            if match is None:
                unexpected_files.append(member)
                continue
            dataset_index = int(match.group("dataset"))
            filename_index = int(match.group("filename_dataset"))
            if dataset_index != filename_index or dataset_index not in expected_by_site:
                raise ValueError(f"Dataset identity mismatch in {member}")
            label = match.group("label")
            date_value = match.group("date")
            datetime.strptime(date_value, "%Y%m%d")
            with archive.open(info) as source:
                header = parse_png_header(source.read(33), member)
            counts[(dataset_index, label)] += 1
            dimensions[f"{header['width']}x{header['height']}"] += 1
            png_formats[
                (
                    f"bit_depth={header['bit_depth']};color_type={header['color_type']};"
                    f"compression={header['compression']};filter={header['filter']};"
                    f"interlace={header['interlace']}"
                )
            ] += 1
            date_ranges[(dataset_index, label)].append(date_value)
            labels_by_site_date[(dataset_index, date_value)].add(label)
            image_members += 1
            uncompressed_bytes += info.file_size
        bad_member = archive.testzip()
        if bad_member is not None:
            raise ValueError(f"ZIP CRC failed for {bad_member}")

    if unexpected_files:
        raise ValueError(f"ZIP contains unexpected files: {unexpected_files[:5]}")
    observed_by_site: dict[str, dict[str, Any]] = {}
    for site in table_contract["sites"]:
        dataset_index = int(site["dataset_index"])
        observed = {
            label: counts[(dataset_index, label)]
            for label in ("plume-containing", "plume-free")
        }
        if observed != expected_by_site[dataset_index]:
            raise ValueError(
                f"Archive counts disagree with official Table 3 for Dataset#{dataset_index}: "
                f"{observed} != {expected_by_site[dataset_index]}"
            )
        observed_by_site[str(dataset_index)] = {
            "plume_containing": observed["plume-containing"],
            "plume_free": observed["plume-free"],
            "rows": sum(observed.values()),
            "date_ranges": {
                label.replace("-", "_"): {
                    "minimum": min(date_ranges[(dataset_index, label)]),
                    "maximum": max(date_ranges[(dataset_index, label)]),
                }
                for label in ("plume-containing", "plume-free")
            },
        }
    label_conflicts = [
        {"dataset_index": key[0], "date": key[1], "labels": sorted(labels)}
        for key, labels in labels_by_site_date.items()
        if len(labels) > 1
    ]
    expected_total = int(table_contract["expected_total_rows"])
    if image_members != expected_total:
        raise ValueError(f"Expected {expected_total} PNGs, observed {image_members}")
    return {
        "zip_members": image_members + directory_members,
        "directory_members": directory_members,
        "image_members": image_members,
        "uncompressed_image_bytes": uncompressed_bytes,
        "site_counts": observed_by_site,
        "dimensions": dict(sorted(dimensions.items())),
        "png_formats": dict(sorted(png_formats.items())),
        "same_site_same_date_label_conflicts": label_conflicts,
        "same_site_same_date_label_conflict_count": len(label_conflicts),
        "crc_verified_for_all_members": True,
        "pixel_arrays_decoded": False,
        "archive_extracted": False,
        "dense_masks_present": False,
    }


def load_mars_sites(
    path: Path,
    permitted_columns: Iterable[str],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    columns = set(permitted_columns)
    required = {"id_location", "split_name", "lat", "lon"}
    if columns != required:
        raise ValueError("Frozen MARS permitted columns do not match the audit implementation")
    unique_rows: set[tuple[str, str, float, float]] = set()
    unique_locations_by_split: dict[str, set[str]] = defaultdict(set)
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("MARS metadata lacks required identity/geography columns")
        for row in reader:
            split = str(row["split_name"])
            location_id = str(row["id_location"])
            latitude = float(row["lat"])
            longitude = float(row["lon"])
            validate_coordinate(latitude, longitude)
            unique_rows.add((location_id, split, latitude, longitude))
            unique_locations_by_split[split].add(location_id)
    sites = [
        {
            "id_location": location_id,
            "split_name": split,
            "latitude": latitude,
            "longitude": longitude,
        }
        for location_id, split, latitude, longitude in sorted(unique_rows)
    ]
    counts = {
        split: len(location_ids)
        for split, location_ids in sorted(unique_locations_by_split.items())
    }
    return sites, counts


def nearest_site(
    source_coordinate: dict[str, float],
    candidates: Iterable[dict[str, Any]],
) -> dict[str, Any] | None:
    best: tuple[float, str, dict[str, Any]] | None = None
    for candidate in candidates:
        distance = haversine_km(
            source_coordinate["latitude"],
            source_coordinate["longitude"],
            float(candidate["latitude"]),
            float(candidate["longitude"]),
        )
        tie_break = f"{candidate['split_name']}:{candidate['id_location']}"
        if best is None or (distance, tie_break) < (best[0], best[1]):
            best = (distance, tie_break, candidate)
    if best is None:
        return None
    candidate = best[2]
    return {
        "distance_km": best[0],
        "id_location": candidate["id_location"],
        "split_name": candidate["split_name"],
        "latitude": candidate["latitude"],
        "longitude": candidate["longitude"],
    }


def source_coordinates(site: dict[str, Any]) -> list[dict[str, float | str]]:
    coordinates: list[dict[str, float | str]] = [
        {
            "coordinate_role": "primary",
            "latitude": float(site["latitude"]),
            "longitude": float(site["longitude"]),
        }
    ]
    for index, alternate in enumerate(site.get("alternate_coordinates", []), start=1):
        coordinates.append(
            {
                "coordinate_role": f"alternate_{index}",
                "latitude": float(alternate["latitude"]),
                "longitude": float(alternate["longitude"]),
            }
        )
    return coordinates


def audit_mars_overlap(
    sites: list[dict[str, Any]],
    mars_sites: list[dict[str, Any]],
    threshold_km: float,
) -> dict[str, Any]:
    development = [
        site
        for site in mars_sites
        if site["split_name"] in {"train_2023", "val_2023"}
    ]
    official_test = [site for site in mars_sites if site["split_name"] == "test_2023"]
    results: list[dict[str, Any]] = []
    eligible_indices: list[int] = []
    for site in sites:
        coordinate_results: list[dict[str, Any]] = []
        for coordinate in source_coordinates(site):
            numeric_coordinate = {
                "latitude": float(coordinate["latitude"]),
                "longitude": float(coordinate["longitude"]),
            }
            coordinate_results.append(
                {
                    **coordinate,
                    "nearest_development": nearest_site(numeric_coordinate, development),
                    "nearest_official_test": nearest_site(numeric_coordinate, official_test),
                    "nearest_any_mars": nearest_site(numeric_coordinate, mars_sites),
                }
            )
        minimum_development = min(
            float(row["nearest_development"]["distance_km"])
            for row in coordinate_results
            if row["nearest_development"] is not None
        )
        minimum_official_test = min(
            float(row["nearest_official_test"]["distance_km"])
            for row in coordinate_results
            if row["nearest_official_test"] is not None
        )
        minimum_any = min(
            float(row["nearest_any_mars"]["distance_km"])
            for row in coordinate_results
            if row["nearest_any_mars"] is not None
        )
        overlaps_development = minimum_development <= threshold_km
        overlaps_official_test = minimum_official_test <= threshold_km
        eligible = not overlaps_development and not overlaps_official_test
        dataset_index = int(site["dataset_index"])
        if eligible:
            eligible_indices.append(dataset_index)
        results.append(
            {
                "dataset_index": dataset_index,
                "field": site["field"],
                "country": site["country"],
                "rows": int(site["plume_containing"]) + int(site["plume_free"]),
                "coordinates": coordinate_results,
                "minimum_development_distance_km": minimum_development,
                "minimum_official_test_distance_km": minimum_official_test,
                "minimum_any_mars_distance_km": minimum_any,
                "overlaps_development_within_25_km": overlaps_development,
                "overlaps_official_test_within_25_km": overlaps_official_test,
                "eligible_for_current_auxiliary_training": eligible,
            }
        )
    eligible_rows = sum(row["rows"] for row in results if row["eligible_for_current_auxiliary_training"])
    return {
        "threshold_km": threshold_km,
        "comparison": "strictly greater than threshold required for eligibility",
        "sites": results,
        "eligible_dataset_indices": eligible_indices,
        "eligible_sites": len(eligible_indices),
        "eligible_rows": eligible_rows,
        "development_overlap_sites": sum(
            bool(row["overlaps_development_within_25_km"]) for row in results
        ),
        "official_test_overlap_sites": sum(
            bool(row["overlaps_official_test_within_25_km"]) for row in results
        ),
    }


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    archive = report["archive_audit"]
    overlap = report["mars_overlap_audit"]
    lines = [
        "# Zhao et al. (2025) DSAN dataset audit",
        "",
        f"Generated: {report['generated_at_utc']}.",
        "",
        "## Outcome",
        "",
        report["decision"],
        "",
        "## Archive integrity and composition",
        "",
        (
            f"The official Science Data Bank archive passed its frozen byte-size, MD5, "
            f"SHA-256, path-safety, and ZIP CRC checks. It contains "
            f"{archive['image_members']:,} class-organized retrieval-map PNGs and no dense "
            "mask files. No archive extraction or pixel-array decoding was performed."
        ),
        "",
        f"Observed raster dimensions: `{json.dumps(archive['dimensions'], sort_keys=True)}`.",
        "",
        "| Dataset | Field | Country | Rows | Min dev distance (km) | Min official-test distance (km) | Eligible |",
        "|---:|---|---|---:|---:|---:|:---:|",
    ]
    for site in overlap["sites"]:
        lines.append(
            "| {dataset_index} | {field} | {country} | {rows:,} | {dev:.3f} | "
            "{test:.3f} | {eligible} |".format(
                dataset_index=site["dataset_index"],
                field=site["field"],
                country=site["country"],
                rows=site["rows"],
                dev=site["minimum_development_distance_km"],
                test=site["minimum_official_test_distance_km"],
                eligible="yes" if site["eligible_for_current_auxiliary_training"] else "no",
            )
        )
    lines.extend(
        [
            "",
            "## Claim boundary",
            "",
            report["claim_boundary"],
            "",
            "Primary sources: [ACP paper](https://acp.copernicus.org/articles/25/4035/2025/) and [Science Data Bank record](https://doi.org/10.57760/sciencedb.15792).",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run_audit(
    root: Path,
    protocol_path: Path,
    output_json: Path,
    output_markdown: Path,
) -> dict[str, Any]:
    protocol = load_protocol(protocol_path)
    verified_files = {
        name: verify_frozen_file(root, contract)
        for name, contract in protocol["files"].items()
    }
    archive_path = repo_path(root, protocol["files"]["archive"]["path"])
    archive_audit = audit_archive(archive_path, protocol["official_table_contract"])
    mars_contract = protocol["mars_overlap_contract"]
    mars_path = repo_path(root, mars_contract["mars_metadata"])
    mars_sites, mars_counts = load_mars_sites(
        mars_path, mars_contract["columns_permitted"]
    )
    overlap_audit = audit_mars_overlap(
        protocol["official_table_contract"]["sites"],
        mars_sites,
        float(mars_contract["distance_threshold_km"]),
    )
    if overlap_audit["eligible_sites"]:
        decision = (
            f"Only Zhao Dataset#{', Dataset#'.join(map(str, overlap_audit['eligible_dataset_indices']))} "
            f"passes the frozen 25 km all-MARS boundary ({overlap_audit['eligible_rows']:,} rows). "
            "It remains unused until a separate architecture/training protocol is committed."
        )
        role = "eligible_subset_unassigned_pending_separate_training_protocol"
    else:
        decision = (
            "All six Zhao sites overlap MARS development or official-test geography within "
            "the frozen 25 km boundary. The archive is research evidence only and contributes "
            "zero training, calibration, selection, or independent-evaluation rows."
        )
        role = "research_evidence_only_zero_model_rows"
    report = {
        "schema_version": 1,
        "status": "complete_integrity_composition_and_label_free_geographic_overlap_audit",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": protocol["sources"],
        "verified_files": verified_files,
        "archive_audit": archive_audit,
        "mars_overlap_audit": {
            **overlap_audit,
            "mars_unique_locations_by_split": mars_counts,
            "mars_metadata": {
                "path": mars_path.relative_to(root).as_posix(),
                "bytes": mars_path.stat().st_size,
                "sha256": sha256(mars_path),
                "columns_accessed": sorted(mars_contract["columns_permitted"]),
                "labels_accessed": False,
            },
        },
        "model_data_role": role,
        "decision": decision,
        "claim_boundary": protocol["claim_boundary"],
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
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_markdown(output_markdown, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL.as_posix())
    parser.add_argument("--output-json", default=DEFAULT_JSON.as_posix())
    parser.add_argument("--output-markdown", default=DEFAULT_MARKDOWN.as_posix())
    args = parser.parse_args()
    root = repo_root()
    report = run_audit(
        root,
        repo_path(root, args.protocol),
        repo_path(root, args.output_json),
        repo_path(root, args.output_markdown),
    )
    print(
        json.dumps(
            {
                "ok": True,
                "archive_rows": report["archive_audit"]["image_members"],
                "eligible_sites": report["mars_overlap_audit"]["eligible_sites"],
                "eligible_rows": report["mars_overlap_audit"]["eligible_rows"],
                "model_data_role": report["model_data_role"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
