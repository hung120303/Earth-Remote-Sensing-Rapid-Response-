"""Apply the frozen spatial eligibility filter to resolved CACH4 negatives.

This stage reads only the ignored, train-only CACH4 negative catalog. It uses
safe-column MARS location metadata and hash-bound prior negative-pair metadata;
it does not import, instantiate, or query a target Sentinel/Landsat catalog.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.audit_mars_hyperspectral_train_masks import geographic_group_ids  # noqa: E402
from tools.audit_mars_hyperspectral_transfer import (  # noqa: E402
    FORBIDDEN_MARS_COLUMNS,
    SAFE_MARS_COLUMNS,
    MarsObservation,
    haversine_km,
    read_mars_observations,
    representative_locations,
)


EXPECTED_IGNORED_ROOT = Path(".research/jpl_operational_ghg_supplement")
EXPECTED_FILTER_PROTOCOL = Path(
    "configs/mars_cross_modal_negative_supplement_filter_protocol.json"
)
RESOLVED_NAME = "cach4_train_negative_resolved_rows.jsonl"
FILTERED_NAME = "cach4_train_negative_eligible_rows.jsonl"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_jsonl_sha256(path: Path) -> str:
    payload = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(payload).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    partial = path.with_name(path.name + ".part")
    partial.write_text(payload, encoding="utf-8")
    partial.replace(path)


def write_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    count = 0
    with partial.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            count += 1
    partial.replace(path)
    return count


def validate_resolved_rows(rows: list[dict[str, object]]) -> None:
    identifiers: set[str] = set()
    for row in rows:
        sample_id = str(row.get("sample_id", ""))
        if not sample_id or sample_id in identifiers:
            raise ValueError(f"Missing or duplicate CACH4 sample_id: {sample_id}")
        identifiers.add(sample_id)
        if row.get("label_state") != "NO_PLUME":
            raise ValueError(f"Non-negative row in CACH4 resolved input: {sample_id}")
        if row.get("published_split") != "train":
            raise ValueError(f"Non-train row in CACH4 resolved input: {sample_id}")
        if row.get("sensor") != "AVIRIS-NG":
            raise ValueError(f"Unexpected source sensor for {sample_id}")
        if row.get("coordinate_resolved") is not True:
            raise ValueError(f"Unresolved coordinate for {sample_id}")
        latitude = float(row["latitude"])
        longitude = float(row["longitude"])
        if not (
            math.isfinite(latitude)
            and math.isfinite(longitude)
            and -90.0 <= latitude <= 90.0
            and -180.0 <= longitude <= 180.0
        ):
            raise ValueError(f"Invalid CACH4 WGS84 coordinate for {sample_id}")


def _require_recorded_file(
    *,
    path: Path,
    recorded: dict[str, object],
    role: str,
    normalized_jsonl: bool = False,
) -> None:
    recorded_path = str(recorded.get("path", "")).replace("\\", "/")
    actual_path = path.as_posix()
    if recorded_path != actual_path:
        raise ValueError(
            f"{role} path differs from frozen Stage B report: "
            f"{actual_path} != {recorded_path}"
        )
    observed_hash = (
        normalized_jsonl_sha256(path) if normalized_jsonl else sha256_file(path)
    )
    if observed_hash != recorded.get("sha256"):
        raise ValueError(f"{role} SHA-256 differs from frozen Stage B report")


def load_prior_negative_coordinates(
    *,
    stage_b_report_path: Path,
    pair_catalog_path: Path,
    mask_catalog_path: Path,
) -> tuple[dict[str, tuple[float, float]], dict[str, int]]:
    report = json.loads(stage_b_report_path.read_text(encoding="utf-8"))
    _require_recorded_file(
        path=pair_catalog_path,
        recorded=report["ignored_pair_catalog"],
        role="Prior pair catalog",
        normalized_jsonl=True,
    )
    _require_recorded_file(
        path=mask_catalog_path,
        recorded=report["inputs"]["mask_catalog"],
        role="Prior mask catalog",
    )

    pairs = read_jsonl(pair_catalog_path)
    negative_pair_rows = [row for row in pairs if row.get("label_state") == "NO_PLUME"]
    negative_ids = {str(row["sample_id"]) for row in negative_pair_rows}
    if not negative_ids:
        raise ValueError("Frozen Stage B catalog contains no counted negative sample IDs")

    coordinates: dict[str, tuple[float, float]] = {}
    seen_mask_ids: set[str] = set()
    for row in read_jsonl(mask_catalog_path):
        sample_id = str(row.get("sample_id", ""))
        if sample_id in seen_mask_ids:
            raise ValueError(f"Duplicate sample ID in prior mask catalog: {sample_id}")
        seen_mask_ids.add(sample_id)
        if sample_id not in negative_ids:
            continue
        if row.get("label_state") != "NO_PLUME":
            raise ValueError(f"Prior pair/mask label disagreement for {sample_id}")
        latitude = float(row["latitude"])
        longitude = float(row["longitude"])
        if not (
            math.isfinite(latitude)
            and math.isfinite(longitude)
            and -90.0 <= latitude <= 90.0
            and -180.0 <= longitude <= 180.0
        ):
            raise ValueError(f"Invalid prior negative coordinate for {sample_id}")
        coordinates[sample_id] = (latitude, longitude)
    missing = sorted(negative_ids - set(coordinates))
    if missing:
        raise ValueError(
            f"Counted prior negative IDs absent from mask catalog: {len(missing)}"
        )
    return coordinates, {
        "negative_pair_rows": len(negative_pair_rows),
        "unique_negative_source_samples": len(negative_ids),
    }


def validate_frozen_input(
    *,
    path: Path,
    specification: dict[str, object],
    role: str,
    hash_field: str = "sha256",
) -> None:
    if path.as_posix() != str(specification.get("path", "")).replace("\\", "/"):
        raise ValueError(f"{role} path differs from frozen filter protocol")
    if path.stat().st_size != int(specification.get("bytes", -1)):
        raise ValueError(f"{role} byte count differs from frozen filter protocol")
    if sha256_file(path) != specification.get(hash_field):
        raise ValueError(f"{role} SHA-256 differs from frozen filter protocol")


def validate_filter_protocol_path(path: Path) -> None:
    if path.resolve() != EXPECTED_FILTER_PROTOCOL.resolve():
        raise ValueError("Only the committed frozen CACH4 filter protocol is permitted")


def official_test_locations(
    observations: Iterable[MarsObservation],
) -> tuple[dict[str, tuple[float, float]], dict[str, tuple[float, float]]]:
    materialized = list(observations)
    all_locations = representative_locations(materialized)
    protected_names = {
        row.location_name
        for row in materialized
        if row.split_name.lower().startswith("test")
    }
    protected = {
        name: all_locations[name]
        for name in sorted(protected_names)
    }
    return all_locations, protected


def nearest_named_distance(
    latitude: float,
    longitude: float,
    candidates: dict[str, tuple[float, float]],
) -> tuple[float, str | None]:
    if not candidates:
        return math.inf, None
    distance, name = min(
        (
            haversine_km(latitude, longitude, point[0], point[1]),
            name,
        )
        for name, point in candidates.items()
    )
    return distance, name


def within_exclusion_radius(distance_km: float, radius_km: float) -> bool:
    """The frozen boundary is inclusive: exactly 25.0 km is excluded."""
    return distance_km <= radius_km


def conservative_component_gate(
    *, component_count: int, required_components: int
) -> bool:
    return component_count >= required_components


def filter_rows(
    *,
    rows: list[dict[str, object]],
    all_mars_locations: dict[str, tuple[float, float]],
    protected_mars_locations: dict[str, tuple[float, float]],
    prior_negative_coordinates: dict[str, tuple[float, float]],
    radius_km: float,
) -> list[dict[str, object]]:
    validate_resolved_rows(rows)
    filtered: list[dict[str, object]] = []
    eligible_coordinates: dict[str, tuple[float, float]] = {}
    for source in rows:
        row = dict(source)
        sample_id = str(row["sample_id"])
        latitude = float(row["latitude"])
        longitude = float(row["longitude"])
        nearest_test, nearest_test_name = nearest_named_distance(
            latitude, longitude, protected_mars_locations
        )
        nearest_all, nearest_all_name = nearest_named_distance(
            latitude, longitude, all_mars_locations
        )
        nearest_prior, nearest_prior_id = nearest_named_distance(
            latitude, longitude, prior_negative_coordinates
        )
        protected = within_exclusion_radius(nearest_test, radius_km)
        prior_duplicate = within_exclusion_radius(nearest_prior, radius_km)
        eligible = not protected and not prior_duplicate
        novel = nearest_all > radius_km
        if eligible:
            eligible_coordinates[sample_id] = (latitude, longitude)
        reasons: list[str] = []
        if protected:
            reasons.append("within_25km_of_official_mars_test_location")
        if prior_duplicate:
            reasons.append("within_25km_of_counted_prior_negative_source_crop")
        row.update(
            {
                "nearest_mars_test_km": nearest_test,
                "nearest_mars_test_location": nearest_test_name,
                "mars_test_protected": protected,
                "nearest_any_mars_km": nearest_all,
                "nearest_any_mars_location": nearest_all_name,
                "novel_beyond_all_mars_25km": novel,
                "nearest_prior_negative_pair_km": nearest_prior,
                "nearest_prior_negative_sample_id": nearest_prior_id,
                "prior_pair_duplicate_25km": prior_duplicate,
                "eligible_for_target_catalog": eligible,
                "eligibility_status": (
                    "eligible_after_frozen_metadata_filter"
                    if eligible
                    else ";".join(reasons)
                ),
                "group_id": None,
                "component_novel_beyond_all_mars_25km": None,
            }
        )
        filtered.append(row)

    group_by_sample = geographic_group_ids(eligible_coordinates, radius_km)
    component_members: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in filtered:
        sample_id = str(row["sample_id"])
        group_id = group_by_sample.get(sample_id)
        row["group_id"] = group_id
        if group_id is not None:
            component_members[group_id].append(row)
    component_novel = {
        group_id: all(
            bool(member["novel_beyond_all_mars_25km"])
            for member in members
        )
        for group_id, members in component_members.items()
    }
    for row in filtered:
        group_id = row["group_id"]
        if group_id is not None:
            row["component_novel_beyond_all_mars_25km"] = component_novel[
                str(group_id)
            ]
    return filtered


def numeric_summary(values: Iterable[float]) -> dict[str, float | int | None]:
    finite = sorted(value for value in values if math.isfinite(value))
    if not finite:
        return {
            "count": 0,
            "minimum": None,
            "p05": None,
            "median": None,
            "p95": None,
            "maximum": None,
        }

    def quantile(fraction: float) -> float:
        position = (len(finite) - 1) * fraction
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return finite[lower]
        weight = position - lower
        return finite[lower] * (1.0 - weight) + finite[upper] * weight

    return {
        "count": len(finite),
        "minimum": finite[0],
        "p05": quantile(0.05),
        "median": quantile(0.5),
        "p95": quantile(0.95),
        "maximum": finite[-1],
    }


def summarize_filter(
    *,
    filtered: list[dict[str, object]],
    raw_report: dict[str, object],
    required_components: int,
    prior_counts: dict[str, int],
    all_mars_location_count: int,
    protected_mars_location_count: int,
) -> dict[str, object]:
    eligible = [row for row in filtered if row["eligible_for_target_catalog"]]
    groups = {str(row["group_id"]) for row in eligible}
    novel_groups = {
        str(row["group_id"])
        for row in eligible
        if row["component_novel_beyond_all_mars_25km"]
    }
    protected_count = sum(bool(row["mars_test_protected"]) for row in filtered)
    duplicate_count = sum(bool(row["prior_pair_duplicate_25km"]) for row in filtered)
    overlap_count = sum(
        bool(row["mars_test_protected"]) and bool(row["prior_pair_duplicate_25km"])
        for row in filtered
    )
    component_gate = conservative_component_gate(
        component_count=len(groups), required_components=required_components
    )
    return {
        "counts": {
            "raw_cach4_train_rows": raw_report["counts"]["cach4_train_rows"],
            "raw_cach4_train_negative_rows": raw_report["counts"][
                "cach4_train_negative_rows"
            ],
            "resolved_negative_rows": len(filtered),
            "mars_test_protected_rows": protected_count,
            "prior_pair_duplicate_rows": duplicate_count,
            "protected_and_prior_duplicate_rows": overlap_count,
            "excluded_rows_any_reason": len(filtered) - len(eligible),
            "eligible_rows": len(eligible),
            "eligible_flightlines": len({str(row["tile"]) for row in eligible}),
            "eligible_25km_connected_components": len(groups),
            "components_novel_beyond_all_mars_25km": len(novel_groups),
            "all_mars_representative_locations": all_mars_location_count,
            "official_test_representative_locations": protected_mars_location_count,
            **prior_counts,
        },
        "distance_summaries_km": {
            "nearest_official_mars_test": numeric_summary(
                float(row["nearest_mars_test_km"]) for row in filtered
            ),
            "nearest_counted_prior_negative_source_crop": numeric_summary(
                float(row["nearest_prior_negative_pair_km"]) for row in filtered
            ),
            "nearest_any_mars_representative_location": numeric_summary(
                float(row["nearest_any_mars_km"]) for row in filtered
            ),
            "eligible_nearest_any_mars_representative_location": numeric_summary(
                float(row["nearest_any_mars_km"]) for row in eligible
            ),
        },
        "conservative_location_gate": {
            "unit": "eligible_25km_connected_components",
            "required": required_components,
            "observed": len(groups),
            "pass": component_gate,
            "interpretation": (
                "Rows and flightlines are not counted as independent candidate "
                "locations; each transitive 25km connected component counts once."
            ),
        },
    }


def run_filter(
    *,
    resolved_path: Path,
    initial_report_path: Path,
    filter_protocol_path: Path,
    mars_manifest_path: Path,
    stage_b_report_path: Path,
    prior_pairs_path: Path,
    prior_mask_catalog_path: Path,
    output_filtered_path: Path,
    output_markdown_path: Path,
) -> dict[str, object]:
    expected_root = EXPECTED_IGNORED_ROOT.resolve()
    if resolved_path.resolve() != (expected_root / RESOLVED_NAME):
        raise ValueError("Resolved CACH4 input must be the ignored train-negative catalog")
    if output_filtered_path.resolve() != (expected_root / FILTERED_NAME):
        raise ValueError("Detailed filtered rows must remain in the ignored CACH4 root")
    if SAFE_MARS_COLUMNS & FORBIDDEN_MARS_COLUMNS:
        raise AssertionError("Safe and protected MARS field declarations overlap")
    validate_filter_protocol_path(filter_protocol_path)

    filter_protocol = json.loads(filter_protocol_path.read_text(encoding="utf-8"))
    parent_spec = filter_protocol["parent_protocol"]
    parent_protocol_path = Path(str(parent_spec["path"]))
    if sha256_file(parent_protocol_path) != parent_spec["sha256"]:
        raise ValueError("Parent supplement protocol differs from frozen filter protocol")
    radius_km = float(filter_protocol["distance_contract"]["radius_km"])
    if not math.isfinite(radius_km) or radius_km <= 0:
        raise ValueError("Frozen filter protocol has an invalid numeric radius")
    outputs = filter_protocol["outputs"]
    if initial_report_path.as_posix() != outputs["compact_json"]:
        raise ValueError("Compact JPL report path differs from frozen protocol")
    if output_markdown_path.as_posix() != outputs["compact_markdown"]:
        raise ValueError("Compact JPL Markdown path differs from frozen protocol")
    if output_filtered_path.as_posix() != outputs["ignored_eligible_rows"]:
        raise ValueError("Filtered JPL row path differs from frozen protocol")

    frozen_inputs = filter_protocol["inputs"]
    validate_frozen_input(
        path=resolved_path,
        specification=frozen_inputs["coordinate_resolved_jpl_backgrounds"],
        role="Resolved CACH4 negatives",
    )
    validate_frozen_input(
        path=mars_manifest_path,
        specification=frozen_inputs["safe_mars_manifest"],
        role="Safe-column MARS manifest",
    )
    validate_frozen_input(
        path=stage_b_report_path,
        specification=frozen_inputs["prior_stage_b_report"],
        role="Prior Stage B report",
    )
    validate_frozen_input(
        path=prior_pairs_path,
        specification=frozen_inputs["prior_stage_b_pairs"],
        role="Prior Stage B pair catalog",
        hash_field="byte_sha256",
    )
    if normalized_jsonl_sha256(prior_pairs_path) != frozen_inputs[
        "prior_stage_b_pairs"
    ]["normalized_lf_sha256"]:
        raise ValueError("Normalized prior pair identity differs from frozen protocol")
    validate_frozen_input(
        path=prior_mask_catalog_path,
        specification=frozen_inputs["prior_mask_catalog"],
        role="Prior mask catalog",
    )
    if set(frozen_inputs["safe_mars_manifest"]["permitted_columns"]) != SAFE_MARS_COLUMNS:
        raise ValueError("Frozen safe MARS column contract differs from implementation")
    if set(frozen_inputs["safe_mars_manifest"]["forbidden_columns"]) != FORBIDDEN_MARS_COLUMNS:
        raise ValueError("Frozen forbidden MARS column contract differs from implementation")

    stage_b_receipt = json.loads(stage_b_report_path.read_text(encoding="utf-8"))
    declared_normalized = frozen_inputs["prior_stage_b_report"][
        "declared_normalized_pair_catalog_sha256"
    ]
    if stage_b_receipt["ignored_pair_catalog"]["sha256"] != declared_normalized:
        raise ValueError("Stage B receipt no longer declares the frozen normalized pair hash")

    report = json.loads(initial_report_path.read_text(encoding="utf-8"))
    observed_resolved_hash = sha256_file(resolved_path)
    expected_resolved_hash = report["ignored_artifacts"]["resolved_rows_sha256"]
    if observed_resolved_hash != expected_resolved_hash:
        raise ValueError("Resolved CACH4 row SHA-256 differs from header-stage report")
    rows = read_jsonl(resolved_path)
    validate_resolved_rows(rows)

    prior_coordinates, prior_counts = load_prior_negative_coordinates(
        stage_b_report_path=stage_b_report_path,
        pair_catalog_path=prior_pairs_path,
        mask_catalog_path=prior_mask_catalog_path,
    )
    mars = read_mars_observations(mars_manifest_path)
    all_mars, protected_mars = official_test_locations(mars)
    filtered = filter_rows(
        rows=rows,
        all_mars_locations=all_mars,
        protected_mars_locations=protected_mars,
        prior_negative_coordinates=prior_coordinates,
        radius_km=radius_km,
    )
    written = write_jsonl(output_filtered_path, filtered)
    if written != len(rows):
        raise AssertionError("Filtered CACH4 row count changed during serialization")

    required_components = int(
        filter_protocol["gates"]["minimum_nonprotected_candidate_locations"][
            "required_components"
        ]
    )
    eligibility = summarize_filter(
        filtered=filtered,
        raw_report=report,
        required_components=required_components,
        prior_counts=prior_counts,
        all_mars_location_count=len(all_mars),
        protected_mars_location_count=len(protected_mars),
    )
    eligibility["radius_km"] = radius_km
    eligibility["component_novelty_contract"] = (
        "A component is novel beyond all MARS only when every eligible member "
        "is strictly farther than 25km from every MARS representative location."
    )
    eligibility["inputs"] = {
        "filter_protocol": {
            "path": filter_protocol_path.as_posix(),
            "sha256": sha256_file(filter_protocol_path),
        },
        "parent_protocol": {
            "path": parent_protocol_path.as_posix(),
            "sha256": sha256_file(parent_protocol_path),
        },
        "resolved_cach4_negatives": {
            "path": resolved_path.as_posix(),
            "sha256": observed_resolved_hash,
        },
        "mars_safe_column_manifest": {
            "path": mars_manifest_path.as_posix(),
            "sha256": sha256_file(mars_manifest_path),
            "columns_accessed": sorted(SAFE_MARS_COLUMNS),
            "protected_outcome_columns_accessed": [],
        },
        "prior_stage_b_report": {
            "path": stage_b_report_path.as_posix(),
            "sha256": sha256_file(stage_b_report_path),
        },
        "prior_pair_catalog": {
            "path": prior_pairs_path.as_posix(),
            "byte_sha256": sha256_file(prior_pairs_path),
            "normalized_lf_sha256": normalized_jsonl_sha256(prior_pairs_path),
        },
        "prior_mask_catalog": {
            "path": prior_mask_catalog_path.as_posix(),
            "sha256": sha256_file(prior_mask_catalog_path),
        },
    }
    eligibility["output"] = {
        "path": output_filtered_path.as_posix(),
        "sha256": sha256_file(output_filtered_path),
        "rows": written,
    }
    eligibility["security_boundary"] = {
        "jpl_test_definition_or_content_accessed": False,
        "mars_protected_outcome_fields_accessed": [],
        "target_catalog_queried": False,
        "target_assets_downloaded": False,
    }

    old_gates = report["gates"]
    gates = {
        "explicit_dataset_research_use_license_or_written_permission": bool(
            old_gates[
                "explicit_dataset_research_use_license_or_written_permission"
            ]
        ),
        "minimum_metadata_resolved_train_background_tiles": old_gates[
            "minimum_metadata_resolved_train_background_tiles"
        ],
        "exact_utc_and_crop_center_available_without_bulk_download": bool(
            old_gates["exact_utc_and_crop_center_available_without_bulk_download"]
        ),
        "minimum_nonprotected_candidate_locations": eligibility[
            "conservative_location_gate"
        ],
    }
    pass_flags = [
        bool(gates["explicit_dataset_research_use_license_or_written_permission"]),
        bool(gates["minimum_metadata_resolved_train_background_tiles"]["pass"]),
        bool(gates["exact_utc_and_crop_center_available_without_bulk_download"]),
        bool(gates["minimum_nonprotected_candidate_locations"]["pass"]),
    ]
    decision = "PASS" if all(pass_flags) else "FAIL"
    gates["metadata_stage_decision"] = decision
    report.update(
        {
            "schema_version": 2,
            "scope": (
                "final_CACH4_train_metadata_eligibility_no_target_catalog_query"
            ),
            "metadata_eligibility": eligibility,
            "gates": gates,
            "eligibility_boundary": {
                "coordinate_resolved": True,
                "eligible_for_target_catalog_rows": eligibility["counts"][
                    "eligible_rows"
                ],
                "target_catalog_stage_authorized": decision == "PASS",
                "target_catalog_queried": False,
            },
        }
    )
    report["ignored_artifacts"]["filtered_rows_jsonl"] = output_filtered_path.as_posix()
    report["ignored_artifacts"]["filtered_rows_sha256"] = sha256_file(
        output_filtered_path
    )
    write_json(initial_report_path, report)
    write_markdown(report, output_markdown_path)
    return report


def write_markdown(report: dict[str, object], path: Path) -> None:
    eligibility = report["metadata_eligibility"]
    counts = eligibility["counts"]
    gate = eligibility["conservative_location_gate"]
    summaries = eligibility["distance_summaries_km"]
    decision = report["gates"]["metadata_stage_decision"]
    target_status = (
        "authorized but not queried"
        if decision == "PASS"
        else "not authorized and not queried"
    )
    markdown = f"""# JPL CACH4 negative-supplement metadata gate

**Decision: {decision}.** Target-catalog stage: **{target_status}**.

This final metadata filter read only released CACH4 train negatives, public ENVI header-derived centers, the safe-column MARS location view, and hash-bound already-counted MARS-Hyperspectral negative-pair metadata. It accessed no protected MARS outcome field, no released JPL test content, and no Sentinel-2/Landsat catalog or asset.

## Eligibility result

- Raw CACH4 train rows: {counts['raw_cach4_train_rows']:,}
- Resolved train negatives: {counts['resolved_negative_rows']:,}
- Within 25 km of official MARS test geography: {counts['mars_test_protected_rows']:,}
- Within 25 km of an already-counted negative source crop: {counts['prior_pair_duplicate_rows']:,}
- Excluded for either reason: {counts['excluded_rows_any_reason']:,}
- Eligible rows / flightlines: {counts['eligible_rows']:,} / {counts['eligible_flightlines']:,}
- Eligible transitive 25 km components: {counts['eligible_25km_connected_components']:,}
- Components wholly novel beyond every MARS representative location: {counts['components_novel_beyond_all_mars_25km']:,}

The frozen `minimum_nonprotected_candidate_locations >= {gate['required']}` gate counts 25 km connected components, not tiles or flightlines: **{'PASS' if gate['pass'] else 'FAIL'} ({gate['observed']:,})**.

## Distance audit

- Nearest official-test location: min {summaries['nearest_official_mars_test']['minimum']:.6f} km; median {summaries['nearest_official_mars_test']['median']:.6f} km
- Nearest counted prior-negative crop: min {summaries['nearest_counted_prior_negative_source_crop']['minimum']:.6f} km; median {summaries['nearest_counted_prior_negative_source_crop']['median']:.6f} km
- Nearest any-MARS representative: min {summaries['nearest_any_mars_representative_location']['minimum']:.6f} km; median {summaries['nearest_any_mars_representative_location']['median']:.6f} km

Detailed row-level distances and stable group IDs remain in ignored `{eligibility['output']['path']}`. The compact JSON records hashes for the protocol, resolved CACH4 rows, safe-column MARS manifest, prior Stage B report, pair catalog, mask catalog, and filtered output.

## Claim boundary

Metadata-stage PASS only authorizes the separately frozen target-catalog feasibility query. It does not establish target-pair yield, label observability, transferability, model performance, or superiority over MARS-S2L.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(markdown, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    root = EXPECTED_IGNORED_ROOT
    parser.add_argument("--resolved", type=Path, default=root / RESOLVED_NAME)
    parser.add_argument(
        "--report-json",
        type=Path,
        default=Path(
            "reports/acquisition/jpl_operational_ghg_negative_supplement_metadata.json"
        ),
    )
    parser.add_argument(
        "--report-markdown",
        type=Path,
        default=Path(
            "reports/acquisition/JPL_OPERATIONAL_GHG_NEGATIVE_SUPPLEMENT_METADATA.md"
        ),
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path(
            "configs/mars_cross_modal_negative_supplement_filter_protocol.json"
        ),
    )
    parser.add_argument(
        "--mars-manifest",
        type=Path,
        default=Path(
            ".research/source_audit_20260715/mars_s2l_current_validated_images_all.csv"
        ),
    )
    parser.add_argument(
        "--stage-b-report",
        type=Path,
        default=Path("reports/acquisition/mars_hyperspectral_transfer_stage_b.json"),
    )
    parser.add_argument(
        "--prior-pairs",
        type=Path,
        default=Path(
            ".research/mars_hyperspectral_transfer/stage_b_combined_pairs.jsonl"
        ),
    )
    parser.add_argument(
        "--prior-mask-catalog",
        type=Path,
        default=Path(
            ".research/mars_hyperspectral_transfer/train_mask_catalog.jsonl"
        ),
    )
    parser.add_argument("--output-filtered", type=Path, default=root / FILTERED_NAME)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = run_filter(
        resolved_path=args.resolved,
        initial_report_path=args.report_json,
        filter_protocol_path=args.protocol,
        mars_manifest_path=args.mars_manifest,
        stage_b_report_path=args.stage_b_report,
        prior_pairs_path=args.prior_pairs,
        prior_mask_catalog_path=args.prior_mask_catalog,
        output_filtered_path=args.output_filtered,
        output_markdown_path=args.report_markdown,
    )
    print(json.dumps(report["metadata_eligibility"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
