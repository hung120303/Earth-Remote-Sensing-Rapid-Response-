"""Validate and, only when explicitly requested, audit frozen GHGSat CSV metadata.

The default mode validates the exact protocol and frozen local inputs.  It does
not open the GHGSat CSV and cannot make a network request.  ``--audit-cached``
audits only the exact verified ignored CSV.  ``--download-and-audit`` streams
only the frozen CSV URL to ``.part``, verifies it, atomically installs it, and
then audits it.  If both explicit flags are supplied, download-and-audit wins in
a documented deterministic order: download, verify/install, then audit once.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.audit_mars_hyperspectral_train_masks import geographic_group_ids  # noqa: E402
from tools.audit_mars_hyperspectral_transfer import (  # noqa: E402
    FORBIDDEN_MARS_COLUMNS,
    SAFE_MARS_COLUMNS,
    haversine_km,
    read_mars_observations,
)
from tools.filter_jpl_cach4_metadata_eligibility import (  # noqa: E402
    load_prior_negative_coordinates,
    nearest_named_distance,
    numeric_summary,
    official_test_locations,
    within_exclusion_radius,
)

PROTOCOL_RELATIVE_PATH = "configs/mars_ghgsat_landfill_null_protocol.json"
EXPECTED_PROTOCOL = ROOT / PROTOCOL_RELATIVE_PATH
EXPECTED_PROTOCOL_SHA256 = "0943cb2a15f2106b9ee4a71f9ab36c06f563e410ad0b530ee1f3514b3aa1bcb1"
_INTEGER_RE = re.compile(r"^-?(?:0|[1-9]\d*)$")
ALWAYS_NUMERIC_COLUMNS = (
    "lat",
    "lon",
    "Q_t_per_hr",
    "wind_speed_m_per_s",
)
POSITIVE_ONLY_NUMERIC_COLUMNS = (
    "Q_error_t_per_hr",
    "IME_kg",
    "intermediate_results_L_m",
    "intermediate_results_effective_wind_speed_m_per_s",
    "conversion_ch4_ppb_to_molm2",
)
INTEGER_COLUMNS = ("year", "month", "day", "hour", "minute", "second", "sat_ID")
IDENTITY_COLUMNS = ("site_ID", "obs_ID", "date", "sat_ID")
RADIUS_KM = 25.0


class GHGSatAuditError(RuntimeError):
    """A frozen audit contract was violated."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def md5_file(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_lf_sha256(path: Path) -> str:
    payload = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(payload).hexdigest()


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    partial.write_bytes(payload)
    partial.replace(path)


def _write_json(path: Path, value: object) -> None:
    _atomic_bytes(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode())


def _write_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> dict[str, object]:
    payload = b"".join(
        (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()
        for row in rows
    )
    _atomic_bytes(path, payload)
    return {"path": path.relative_to(ROOT).as_posix(), "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest(), "rows": payload.count(b"\n")}


def _assert_protocol_contract(protocol: dict[str, Any]) -> None:
    csv_spec = protocol["authoritative_source"]["csv"]
    if csv_spec != {
        "name": "GHGSat_detected_plumes.csv",
        "url": "https://zenodo.org/api/records/16641834/files/GHGSat_detected_plumes.csv/content",
        "bytes": 299683,
        "md5": "0e3ba8fae21d6f888413148a76edc5de",
        "maximum_bytes": 1048576,
    }:
        raise GHGSatAuditError("Frozen CSV source contract mismatch")
    local = protocol["frozen_local_inputs"]["safe_mars_manifest"]
    if set(local["permitted_columns"]) != SAFE_MARS_COLUMNS:
        raise GHGSatAuditError("Frozen safe MARS-column contract mismatch")
    if set(local["forbidden_columns"]) != FORBIDDEN_MARS_COLUMNS:
        raise GHGSatAuditError("Frozen forbidden MARS-column contract mismatch")
    if protocol["site_coordinate_rule"]["maximum_pairwise_site_coordinate_span_km"] != RADIUS_KM:
        raise GHGSatAuditError("Frozen site-coordinate span mismatch")
    if protocol["geographic_eligibility"]["radius_km"] != RADIUS_KM:
        raise GHGSatAuditError("Frozen geographic radius mismatch")
    if protocol["deterministic_null_selection"]["eligible_satellites"] != [1, 2]:
        raise GHGSatAuditError("Frozen satellite selection mismatch")
    if protocol["deterministic_null_selection"]["maximum_null_observations_per_site"] != 4:
        raise GHGSatAuditError("Frozen per-site selection cap mismatch")
    expected_gates = {
        "license_permits_noncommercial_research_derivatives_with_attribution_and_share_alike": True,
        "minimum_selected_morning_null_observations": 56,
        "minimum_distinct_null_sites": 30,
        "minimum_novel_25km_connected_components": 20,
        "all_released_rows_and_observations_valid": True,
    }
    for name, required in expected_gates.items():
        if protocol["metadata_gates"].get(name) != required:
            raise GHGSatAuditError(f"Frozen metadata gate mismatch: {name}")
    expected_outputs = {
        "ignored_root": ".research/ghgsat_landfill_null",
        "ignored_csv": ".research/ghgsat_landfill_null/GHGSat_detected_plumes.csv",
        "ignored_validated_observations": ".research/ghgsat_landfill_null/validated_observations.jsonl",
        "ignored_selected_nulls": ".research/ghgsat_landfill_null/selected_morning_null_observations.jsonl",
        "ignored_eligible_nulls": ".research/ghgsat_landfill_null/eligible_morning_null_observations.jsonl",
        "compact_json": "reports/acquisition/ghgsat_landfill_null_metadata.json",
        "compact_markdown": "reports/acquisition/GHGSAT_LANDFILL_NULL_METADATA.md",
    }
    if protocol["outputs"] != expected_outputs:
        raise GHGSatAuditError("Frozen output-path contract mismatch")
    if not protocol["target_catalog_boundary"]["target_assets_forbidden_in_this_protocol"]:
        raise GHGSatAuditError("Target-asset prohibition is not frozen")


def load_protocol(path: Path = EXPECTED_PROTOCOL) -> dict[str, Any]:
    if path.resolve() != EXPECTED_PROTOCOL.resolve():
        raise GHGSatAuditError("Only the exact committed GHGSat protocol is permitted")
    if sha256_file(path) != EXPECTED_PROTOCOL_SHA256:
        raise GHGSatAuditError("Frozen GHGSat protocol SHA-256 mismatch")
    try:
        protocol = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GHGSatAuditError("Frozen GHGSat protocol is unreadable") from exc
    if not isinstance(protocol, dict):
        raise GHGSatAuditError("Frozen GHGSat protocol must be an object")
    _assert_protocol_contract(protocol)
    return protocol


def validate_frozen_local_inputs(protocol: dict[str, Any], *, root: Path = ROOT) -> dict[str, object]:
    receipts: dict[str, object] = {}
    for role, spec in protocol["frozen_local_inputs"].items():
        path = root / spec["path"]
        if not path.is_file():
            raise GHGSatAuditError(f"Frozen local input is missing: {role}")
        observed_bytes = path.stat().st_size
        if observed_bytes != int(spec["bytes"]):
            raise GHGSatAuditError(f"Frozen local-input bytes mismatch: {role}")
        kind = "normalized_lf_sha256" if "normalized_lf_sha256" in spec else "sha256"
        observed_hash = normalized_lf_sha256(path) if kind == "normalized_lf_sha256" else sha256_file(path)
        if observed_hash != spec[kind]:
            raise GHGSatAuditError(f"Frozen local-input hash mismatch: {role}")
        receipts[role] = {"path": spec["path"], "bytes": observed_bytes, "hash_kind": kind, "sha256": observed_hash}
    return receipts


def validation_plan() -> dict[str, object]:
    protocol = load_protocol()
    return {
        "mode": "validation_only",
        "protocol": {"path": PROTOCOL_RELATIVE_PATH, "sha256": EXPECTED_PROTOCOL_SHA256},
        "frozen_local_inputs": validate_frozen_local_inputs(protocol),
        "network_executed": False,
        "ghgsat_csv_opened": False,
        "ghgsat_rasters_accessed": False,
        "target_catalog_accessed": False,
        "target_assets_accessed": False,
        "protected_outcomes_accessed": False,
        "score_caches_accessed": False,
        "model_checkpoints_accessed": False,
    }


class FrozenCSVDownloader:
    """One bounded stream; observed bytes are never refunded after failure."""

    def __init__(self, session: Any) -> None:
        self.session = session
        self.bytes_received = 0

    def download(self, *, protocol: dict[str, Any], destination: Path) -> dict[str, object]:
        spec = protocol["authoritative_source"]["csv"]
        url = str(spec["url"])
        maximum = int(spec["maximum_bytes"])
        expected_bytes = int(spec["bytes"])
        frozen_destination = ROOT / protocol["outputs"]["ignored_csv"]
        if destination.resolve() != frozen_destination.resolve():
            raise GHGSatAuditError("CSV download destination is not the exact frozen ignored cache")
        partial = destination.with_name(destination.name + ".part")
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial.unlink(missing_ok=True)
        response = None
        digest = hashlib.md5(usedforsecurity=False)
        try:
            response = self.session.get(
                url,
                allow_redirects=True,
                stream=True,
                timeout=(15, 120),
                headers={
                    "Accept": "*/*",
                    "User-Agent": "ERSRR-research-metadata-audit/1.0",
                },
            )
            final_url = str(getattr(response, "url", ""))
            final_parts = urlsplit(final_url)
            if final_parts.scheme != "https" or final_parts.hostname != "zenodo.org":
                raise GHGSatAuditError("Frozen CSV redirect left the authorized Zenodo host")
            declared = response.headers.get("Content-Length")
            if declared is not None:
                try:
                    declared_bytes = int(declared)
                except ValueError as exc:
                    raise GHGSatAuditError("Invalid CSV Content-Length") from exc
                if declared_bytes < 0 or declared_bytes > maximum:
                    raise GHGSatAuditError("Frozen CSV response exceeds 1 MiB cap")
            with partial.open("wb") as target:
                for chunk in response.iter_content(chunk_size=64 * 1024):
                    if not chunk:
                        continue
                    self.bytes_received += len(chunk)
                    if self.bytes_received > maximum:
                        raise GHGSatAuditError("Streamed CSV exceeds 1 MiB cap")
                    target.write(chunk)
                    digest.update(chunk)
            if int(response.status_code) != 200:
                raise GHGSatAuditError(f"Frozen CSV HTTP status rejected: {response.status_code}")
            if self.bytes_received != expected_bytes:
                raise GHGSatAuditError("Frozen CSV exact byte count mismatch")
            if digest.hexdigest() != spec["md5"]:
                raise GHGSatAuditError("Frozen CSV MD5 mismatch")
            partial.replace(destination)
            return {
                "url": url,
                "requested_urls": [url],
                "final_url": final_url,
                "redirect_count": len(getattr(response, "history", [])),
                "bytes": self.bytes_received,
                "md5": digest.hexdigest(),
                "sha256": sha256_file(destination),
                "atomic_install": True,
            }
        except Exception:
            partial.unlink(missing_ok=True)
            raise
        finally:
            if response is not None:
                close = getattr(response, "close", None)
                if callable(close):
                    close()


def verify_cached_csv(protocol: dict[str, Any], path: Path) -> dict[str, object]:
    spec = protocol["authoritative_source"]["csv"]
    if path.resolve() != (ROOT / protocol["outputs"]["ignored_csv"]).resolve():
        raise GHGSatAuditError("Only the exact ignored GHGSat CSV cache is permitted")
    if not path.is_file() or path.stat().st_size != int(spec["bytes"]):
        raise GHGSatAuditError("Cached GHGSat CSV exact byte count mismatch")
    observed_md5 = md5_file(path)
    if observed_md5 != spec["md5"]:
        raise GHGSatAuditError("Cached GHGSat CSV MD5 mismatch")
    return {"path": protocol["outputs"]["ignored_csv"], "bytes": path.stat().st_size, "md5": observed_md5, "sha256": sha256_file(path)}


def _text(value: str | None, *, name: str) -> str:
    if value is None or not value or value.strip() != value:
        raise GHGSatAuditError(f"{name} must be a non-empty trimmed string")
    return value


def _finite(value: str | None, *, name: str) -> float:
    if value is None or value == "" or value.strip() != value:
        raise GHGSatAuditError(f"{name} must be a finite number")
    try:
        result = float(value)
    except ValueError as exc:
        raise GHGSatAuditError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise GHGSatAuditError(f"{name} must be a finite number")
    return result


def _integer(value: str | None, *, name: str) -> int:
    if value is None or _INTEGER_RE.fullmatch(value) is None:
        raise GHGSatAuditError(f"{name} must be a strict integer")
    return int(value)


def _parse_utc(value: str, *, fields: tuple[int, int, int, int, int, int]) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise GHGSatAuditError("date must be an exact UTC timestamp") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    if parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise GHGSatAuditError("date must be an exact UTC timestamp")
    observed = (parsed.year, parsed.month, parsed.day, parsed.hour, parsed.minute, parsed.second)
    if observed != fields or parsed.microsecond != 0:
        raise GHGSatAuditError("calendar fields do not reproduce date")
    return parsed


def parse_csv_rows(path: Path, protocol: dict[str, Any]) -> list[dict[str, object]]:
    expected = list(protocol["csv_contract"]["expected_columns_in_order"])
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != expected:
            raise GHGSatAuditError("GHGSat CSV header does not exactly match frozen ordered header")
        for data_row_number, source in enumerate(reader, start=1):
            if None in source or set(source) != set(expected):
                raise GHGSatAuditError(f"row {data_row_number}: malformed CSV width")
            try:
                site_id = _text(source["site_ID"], name="site_ID")
                obs_id = _text(source["obs_ID"], name="obs_ID")
                integers = {name: _integer(source[name], name=name) for name in INTEGER_COLUMNS}
                numbers: dict[str, float | None] = {
                    name: _finite(source[name], name=name)
                    for name in ALWAYS_NUMERIC_COLUMNS
                }
                if integers["sat_ID"] not in {1, 2, 3, 4, 5}:
                    raise GHGSatAuditError("sat_ID must identify GHGSat C1-C5")
                if not (-90 <= numbers["lat"] <= 90 and -180 <= numbers["lon"] <= 180):
                    raise GHGSatAuditError("coordinate outside WGS84")
                calendar = tuple(integers[name] for name in ("year", "month", "day", "hour", "minute", "second"))
                _parse_utc(_text(source["date"], name="date"), fields=calendar)  # type: ignore[arg-type]
                pinned = source["manually_pinned_sources"]
                plume_name = source["plume_tif_file_name"]
                null_identity = (
                    numbers["lat"] == 0.0
                    and numbers["lon"] == 0.0
                    and numbers["Q_t_per_hr"] == 0.0
                    and pinned == ""
                    and plume_name == ""
                )
                if null_identity:
                    nonempty = [
                        name
                        for name in POSITIVE_ONLY_NUMERIC_COLUMNS
                        if source[name] != ""
                    ]
                    if nonempty:
                        raise GHGSatAuditError(
                            "null row has non-empty positive-only measurement fields: "
                            + ", ".join(nonempty)
                        )
                    numbers.update(
                        {name: None for name in POSITIVE_ONLY_NUMERIC_COLUMNS}
                    )
                else:
                    numbers.update(
                        {
                            name: _finite(source[name], name=name)
                            for name in POSITIVE_ONLY_NUMERIC_COLUMNS
                        }
                    )
                null = null_identity
                positive = (
                    (numbers["lat"] != 0.0 or numbers["lon"] != 0.0)
                    and numbers["Q_t_per_hr"] > 0.0
                    and bool(plume_name)
                    and plume_name.strip() == plume_name
                )
                if null == positive:
                    raise GHGSatAuditError("row is neither exclusively null nor positive")
            except GHGSatAuditError as exc:
                raise GHGSatAuditError(f"row {data_row_number}: {exc}") from exc
            rows.append(
                {
                    "data_row_number": data_row_number,
                    "site_ID": site_id,
                    "obs_ID": obs_id,
                    "date": source["date"],
                    **integers,
                    **numbers,
                    "manually_pinned_sources": pinned,
                    "plume_tif_file_name": plume_name,
                    "row_state": "null" if null else "positive",
                }
            )
    return rows


def coordinate_medoid(points: list[tuple[float, float, int]]) -> tuple[float, float, int]:
    if not points:
        raise GHGSatAuditError("cannot derive a representative without positive coordinates")
    return min(
        points,
        key=lambda candidate: (
            sum(haversine_km(candidate[0], candidate[1], other[0], other[1]) for other in points),
            candidate[0],
            candidate[1],
            candidate[2],
        ),
    )


def maximum_pairwise_span_km(points: list[tuple[float, float, int]]) -> float:
    return max(
        (haversine_km(left[0], left[1], right[0], right[1]) for index, left in enumerate(points) for right in points[index + 1 :]),
        default=0.0,
    )


def validate_and_group_rows(
    rows: list[dict[str, object]], protocol: dict[str, Any], *, expectations: dict[str, object] | None = None
) -> tuple[list[dict[str, object]], dict[str, object]]:
    expected = expectations or protocol["population_reconciliation_gates"]
    groups: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    positive_points: dict[str, list[tuple[float, float, int]]] = defaultdict(list)
    observation_signatures: dict[tuple[str, str], tuple[str, int]] = {}
    for row in rows:
        groups[tuple(row[name] for name in IDENTITY_COLUMNS)].append(row)
        abbreviated_identity = (str(row["site_ID"]), str(row["obs_ID"]))
        signature = (str(row["date"]), int(row["sat_ID"]))
        previous = observation_signatures.setdefault(abbreviated_identity, signature)
        if previous != signature:
            raise GHGSatAuditError(
                f"observation {abbreviated_identity!r} has inconsistent date or satellite"
            )
        if row["row_state"] == "positive":
            positive_points[str(row["site_ID"])].append((float(row["lat"]), float(row["lon"]), int(row["data_row_number"])))
    representatives: dict[str, tuple[float, float, int]] = {}
    spans: dict[str, float] = {}
    observations: list[dict[str, object]] = []
    for identity, members in sorted(groups.items(), key=lambda item: tuple(map(str, item[0]))):
        states = {str(row["row_state"]) for row in members}
        if len(states) != 1:
            raise GHGSatAuditError(f"observation {identity!r} mixes positive and null rows")
        state = next(iter(states))
        first = members[0]
        site_id = str(first["site_ID"])
        if state == "null":
            if len(members) != 1:
                raise GHGSatAuditError(
                    f"null observation {identity!r} must have exactly one released row"
                )
            points = positive_points.get(site_id, [])
            if not points:
                raise GHGSatAuditError(f"null site {site_id!r} has no positive coordinate")
            if site_id not in representatives:
                span = maximum_pairwise_span_km(points)
                if span > float(protocol["site_coordinate_rule"]["maximum_pairwise_site_coordinate_span_km"]):
                    raise GHGSatAuditError(f"null site {site_id!r} positive-coordinate span exceeds 25 km")
                representatives[site_id] = coordinate_medoid(points)
                spans[site_id] = span
            latitude, longitude, representative_row = representatives[site_id]
        else:
            latitude = longitude = representative_row = None
        observations.append(
            {
                "site_ID": site_id,
                "obs_ID": str(first["obs_ID"]),
                "date": str(first["date"]),
                "sat_ID": int(first["sat_ID"]),
                "year": int(first["year"]),
                "observation_state": state,
                "plume_row_count": len(members) if state == "positive" else 0,
                "source_data_rows": [int(row["data_row_number"]) for row in members],
                "representative_latitude": latitude,
                "representative_longitude": longitude,
                "representative_positive_data_row": representative_row,
                "positive_coordinate_span_km": spans.get(site_id),
                "eligible_for_target_catalog": False,
            }
        )
    counts = {
        "clear_sky_observations": len(observations),
        "positive_observations": sum(row["observation_state"] == "positive" for row in observations),
        "null_observations": sum(row["observation_state"] == "null" for row in observations),
        "positive_plume_rows": sum(row["row_state"] == "positive" for row in rows),
        "distinct_sites": len({str(row["site_ID"]) for row in rows}),
        "years": sorted({int(row["year"]) for row in rows}),
        "released_rows": len(rows),
    }
    mapping = {
        "exact_clear_sky_observations": "clear_sky_observations",
        "exact_positive_observations": "positive_observations",
        "exact_null_observations": "null_observations",
        "exact_positive_plume_rows": "positive_plume_rows",
        "exact_distinct_sites": "distinct_sites",
        "exact_years": "years",
    }
    mismatches = {key: {"expected": expected[key], "observed": counts[name]} for key, name in mapping.items() if counts[name] != expected[key]}
    if mismatches:
        raise GHGSatAuditError(f"released population reconciliation mismatch: {json.dumps(mismatches, sort_keys=True)}")
    return observations, counts


def selection_rank(row: dict[str, object]) -> tuple[str, str, str]:
    site_id, obs_id, date = (str(row[name]) for name in ("site_ID", "obs_ID", "date"))
    digest = hashlib.sha256((site_id + "\0" + obs_id + "\0" + date).encode()).hexdigest()
    return digest, obs_id, date


def select_morning_nulls(observations: list[dict[str, object]], protocol: dict[str, Any]) -> list[dict[str, object]]:
    eligible_satellites = set(protocol["deterministic_null_selection"]["eligible_satellites"])
    cap = int(protocol["deterministic_null_selection"]["maximum_null_observations_per_site"])
    by_site: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in observations:
        if row["observation_state"] == "null" and row["sat_ID"] in eligible_satellites:
            by_site[str(row["site_ID"])].append(row)
    return [dict(row) for site in sorted(by_site) for row in sorted(by_site[site], key=selection_rank)[:cap]]


def read_safe_mars_points(path: Path, protocol: dict[str, Any], *, requested_columns: Iterable[str] | None = None) -> tuple[dict[str, tuple[float, float]], dict[str, tuple[float, float]]]:
    spec = protocol["frozen_local_inputs"]["safe_mars_manifest"]
    safe = set(spec["permitted_columns"])
    forbidden = set(spec["forbidden_columns"])
    requested = safe if requested_columns is None else set(requested_columns)
    if requested - safe or requested & forbidden:
        raise GHGSatAuditError("Forbidden or undeclared MARS column requested")
    observations = read_mars_observations(path)
    for row in observations:
        if set(vars(row)) & forbidden:
            raise GHGSatAuditError("Forbidden MARS column entered returned observation objects")
    return official_test_locations(observations)


def spatial_filter(
    selected: list[dict[str, object]], *, protected_mars: dict[str, tuple[float, float]], prior_negative: dict[str, tuple[float, float]], radius_km: float = RADIUS_KM
) -> tuple[list[dict[str, object]], dict[str, int], Counter[str]]:
    retained_sites: dict[str, tuple[float, float]] = {}
    exclusions: Counter[str] = Counter()
    staged: list[dict[str, object]] = []
    for source in selected:
        row = dict(source)
        latitude = float(row["representative_latitude"])
        longitude = float(row["representative_longitude"])
        mars_distance, mars_name = nearest_named_distance(latitude, longitude, protected_mars)
        prior_distance, prior_id = nearest_named_distance(latitude, longitude, prior_negative)
        mars_excluded = within_exclusion_radius(mars_distance, radius_km)
        prior_excluded = within_exclusion_radius(prior_distance, radius_km)
        if mars_excluded:
            exclusions["within_or_at_25km_of_official_mars_test_representative"] += 1
        if prior_excluded:
            exclusions["within_or_at_25km_of_prior_negative_point"] += 1
        retained = not mars_excluded and not prior_excluded
        if not retained:
            exclusions["selected_observations_excluded_by_any_protected_radius"] += 1
        if retained:
            retained_sites[str(row["site_ID"])] = (latitude, longitude)
        row.update(
            {
                "nearest_official_mars_test_km": mars_distance,
                "nearest_official_mars_test_location": mars_name,
                "nearest_prior_negative_km": prior_distance,
                "nearest_prior_negative_id": prior_id,
                "excluded_by_official_mars_test_radius": mars_excluded,
                "excluded_by_prior_negative_radius": prior_excluded,
                "passes_protected_distance_filter": retained,
                "component_id": None,
                "eligible_for_target_catalog": False,
            }
        )
        staged.append(row)
    groups = geographic_group_ids(retained_sites, radius_km)
    component_sizes = Counter(groups.values())
    for row in staged:
        if row["passes_protected_distance_filter"]:
            row["component_id"] = groups[str(row["site_ID"])]
    return staged, dict(sorted(component_sizes.items())), exclusions


def evaluate_gates(
    eligible: list[dict[str, object]], component_sizes: dict[str, int], protocol: dict[str, Any], *, all_records_valid: bool = True
) -> dict[str, dict[str, object]]:
    gates = protocol["metadata_gates"]
    observed = {
        "license_permits_noncommercial_research_derivatives_with_attribution_and_share_alike": True,
        "minimum_selected_morning_null_observations": len(eligible),
        "minimum_distinct_null_sites": len({row["site_ID"] for row in eligible}),
        "minimum_novel_25km_connected_components": len(component_sizes),
        "all_released_rows_and_observations_valid": all_records_valid,
    }
    results: dict[str, dict[str, object]] = {}
    for name, required in gates.items():
        if name == "population":
            continue
        value = observed[name]
        passed = value is True if required is True else int(value) >= int(required)
        results[name] = {"required": required, "observed": value, "pass": passed}
    return results


def _access_boundary() -> dict[str, bool]:
    return {
        "ghgsat_plume_rasters_accessed": False,
        "ghgsat_images_accessed": False,
        "sentinel2_accessed": False,
        "landsat_accessed": False,
        "target_catalog_accessed": False,
        "target_assets_accessed": False,
        "mars_fold_0_1_2_outcomes_accessed": False,
        "official_test_outcomes_accessed": False,
        "score_caches_accessed": False,
        "model_checkpoints_accessed": False,
    }


def build_failure_report(
    protocol: dict[str, Any],
    error: Exception,
    *,
    csv_receipt: dict[str, object] | None = None,
    local_receipts: dict[str, object] | None = None,
    download_receipt: dict[str, object] | None = None,
    download_bytes: int = 0,
) -> dict[str, object]:
    gates = {
        name: {"required": value, "observed": False if value is True else None, "pass": False}
        for name, value in protocol["metadata_gates"].items()
        if name != "population"
    }
    return {
        "schema_version": 1,
        "decision": "FAIL",
        "error": str(error),
        "all_released_rows_and_observations_valid": False,
        "raw_counts": None,
        "selected_counts": None,
        "exclusions": None,
        "component_sizes": None,
        "gates": gates,
        "protocol": {"path": PROTOCOL_RELATIVE_PATH, "sha256": EXPECTED_PROTOCOL_SHA256},
        "csv_receipt": csv_receipt,
        "network_bytes_received_this_execution": download_bytes,
        "receipts": {
            "protocol": {"sha256": EXPECTED_PROTOCOL_SHA256},
            "frozen_local_inputs": local_receipts,
            "download": download_receipt,
            "csv": csv_receipt,
            "network_bytes_received_this_execution": download_bytes,
        },
        "label_boundary": protocol["claim_boundary"],
        "license_boundary": protocol["license_contract"],
        "access_boundary": _access_boundary(),
    }


def _markdown(report: dict[str, object]) -> str:
    gates = report["gates"]
    lines = [
        "# GHGSat landfill-null metadata audit",
        "",
        f"Overall decision: **{report['decision']}**.",
        "",
        "## Gates",
        "",
    ]
    for name, gate in gates.items():  # type: ignore[union-attr]
        lines.append(f"- {name}: **{'PASS' if gate['pass'] else 'FAIL'}** (observed `{gate['observed']}`, required `{gate['required']}`).")
    lines.extend(
        [
            "",
            "## Counts and exclusions",
            "",
            f"- Raw counts: `{json.dumps(report.get('raw_counts'), sort_keys=True)}`.",
            f"- Selected counts: `{json.dumps(report.get('selected_counts'), sort_keys=True)}`.",
            f"- Exclusions: `{json.dumps(report.get('exclusions'), sort_keys=True)}`.",
            f"- Component sizes: `{json.dumps(report.get('component_sizes'), sort_keys=True)}`.",
            "",
            "## Boundaries and receipts",
            "",
            f"- Protocol / inputs / CSV / outputs: `{json.dumps(report.get('receipts'), sort_keys=True)}`.",
            f"- Label boundary: {report['label_boundary']}",
            f"- License boundary: `{json.dumps(report['license_boundary'], sort_keys=True)}`.",
            "- Target catalog/assets, protected outcomes, score caches, model checkpoints, and forbidden imagery/raster resources were not accessed.",
            f"- Access proof: `{json.dumps(report['access_boundary'], sort_keys=True)}`.",
            "",
        ]
    )
    if report.get("error"):
        lines.insert(4, f"Failure: `{report['error']}`.")
    return "\n".join(lines)


def _write_reports(protocol: dict[str, Any], report: dict[str, object]) -> None:
    outputs = protocol["outputs"]
    _write_json(ROOT / outputs["compact_json"], report)
    _atomic_bytes(ROOT / outputs["compact_markdown"], _markdown(report).encode())


def audit_verified_cache(
    protocol: dict[str, Any],
    *,
    local_receipts: dict[str, object] | None = None,
    download_receipt: dict[str, object] | None = None,
) -> dict[str, object]:
    outputs = protocol["outputs"]
    csv_path = ROOT / outputs["ignored_csv"]
    csv_receipt: dict[str, object] | None = None
    try:
        for name in (
            "ignored_validated_observations",
            "ignored_selected_nulls",
            "ignored_eligible_nulls",
        ):
            (ROOT / outputs[name]).unlink(missing_ok=True)
        csv_receipt = verify_cached_csv(protocol, csv_path)
        local_receipts = local_receipts or validate_frozen_local_inputs(protocol)
        rows = parse_csv_rows(csv_path, protocol)
        observations, raw_counts = validate_and_group_rows(rows, protocol)
        selected = select_morning_nulls(observations, protocol)
        eligible_satellites = set(
            protocol["deterministic_null_selection"]["eligible_satellites"]
        )
        morning_candidates = sum(
            row["observation_state"] == "null"
            and row["sat_ID"] in eligible_satellites
            for row in observations
        )
        mars_path = ROOT / protocol["frozen_local_inputs"]["safe_mars_manifest"]["path"]
        _, protected_mars = read_safe_mars_points(mars_path, protocol)
        prior_negative, prior_counts = load_prior_negative_coordinates(
            stage_b_report_path=ROOT / protocol["frozen_local_inputs"]["prior_stage_b_report"]["path"],
            pair_catalog_path=ROOT / protocol["frozen_local_inputs"]["prior_stage_b_pairs"]["path"],
            mask_catalog_path=ROOT / protocol["frozen_local_inputs"]["prior_mask_catalog"]["path"],
        )
        filtered, component_sizes, exclusions = spatial_filter(selected, protected_mars=protected_mars, prior_negative=prior_negative)
        eligible = [row for row in filtered if row["passes_protected_distance_filter"]]
        gates = evaluate_gates(eligible, component_sizes, protocol)
        decision = "PASS" if all(bool(gate["pass"]) for gate in gates.values()) else "FAIL"
        output_receipts = {
            "validated_observations": _write_jsonl(ROOT / outputs["ignored_validated_observations"], observations),
            "selected_morning_nulls": _write_jsonl(ROOT / outputs["ignored_selected_nulls"], selected),
            "eligible_morning_nulls": _write_jsonl(ROOT / outputs["ignored_eligible_nulls"], eligible),
        }
        released_nulls = int(raw_counts["null_observations"])
        report: dict[str, object] = {
            "schema_version": 1,
            "decision": decision,
            "raw_counts": raw_counts,
            "selected_counts": {
                "released_null_observations": released_nulls,
                "excluded_non_c1_c2_null_observations": released_nulls - morning_candidates,
                "c1_c2_null_candidates_before_per_site_cap": morning_candidates,
                "excluded_by_four_per_site_cap": morning_candidates - len(selected),
                "before_protected_filter": len(selected),
                "after_protected_filter": len(eligible),
                "distinct_sites": len({row["site_ID"] for row in eligible}),
                "components": len(component_sizes),
            },
            "exclusions": dict(sorted(exclusions.items())),
            "component_sizes": component_sizes,
            "gates": gates,
            "receipts": {"protocol": {"sha256": EXPECTED_PROTOCOL_SHA256}, "frozen_local_inputs": local_receipts, "download": download_receipt, "csv": csv_receipt, "outputs": output_receipts, "prior_negative_counts": prior_counts},
            "label_boundary": protocol["claim_boundary"],
            "license_boundary": protocol["license_contract"],
            "access_boundary": _access_boundary(),
        }
    except Exception as exc:
        report = build_failure_report(
            protocol,
            exc,
            csv_receipt=csv_receipt,
            local_receipts=local_receipts,
            download_receipt=download_receipt,
            download_bytes=(
                int(download_receipt["bytes"])
                if download_receipt is not None
                else 0
            ),
        )
    _write_reports(protocol, report)
    if report["decision"] == "FAIL":
        raise GHGSatAuditError(str(report.get("error", "Frozen metadata gates failed")))
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-and-audit", action="store_true", help="stream the exact frozen CSV, verify/install it, then audit once")
    parser.add_argument("--audit-cached", action="store_true", help="audit only the exact already-cached and verified ignored CSV")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        protocol = load_protocol()
        try:
            local_receipts = validate_frozen_local_inputs(protocol)
        except Exception as exc:
            if args.download_and_audit or args.audit_cached:
                _write_reports(protocol, build_failure_report(protocol, exc))
            raise
        if not args.download_and_audit and not args.audit_cached:
            print(json.dumps(validation_plan(), indent=2, sort_keys=True))
            return 0
        if args.download_and_audit:
            downloader = FrozenCSVDownloader(requests.Session())
            try:
                download_receipt = downloader.download(
                    protocol=protocol,
                    destination=ROOT / protocol["outputs"]["ignored_csv"],
                )
            except Exception as exc:
                report = build_failure_report(
                    protocol,
                    exc,
                    local_receipts=local_receipts,
                    download_bytes=downloader.bytes_received,
                )
                _write_reports(protocol, report)
                raise
        else:
            download_receipt = None
        report = audit_verified_cache(
            protocol,
            local_receipts=local_receipts,
            download_receipt=download_receipt,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    except (GHGSatAuditError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"GHGSat audit failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
