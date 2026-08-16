"""Audit the frozen NASA ORNL header bridge for JPL train backgrounds.

The safe default is validation-only and performs no network request. The
explicit authenticated execution path queries CMR only for frozen, released
train flight IDs and downloads only selected ENVI ``*_rdn_v*_img.hdr`` text
sidecars. Stage B cannot run unless the CACH4 grid bridge passes Stage A.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
import urllib.parse
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import requests
from pyproj import Transformer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.audit_jpl_cach4_train_headers import (
    EnviMapInfo,
    flight_timestamp,
    gdal_geotransform,
    parse_envi_header,
)
from tools.audit_mars_hyperspectral_transfer import (
    FORBIDDEN_MARS_COLUMNS,
    SAFE_MARS_COLUMNS,
    read_mars_observations,
)
from tools.filter_jpl_cach4_metadata_eligibility import (
    filter_rows,
    load_prior_negative_coordinates,
    normalized_jsonl_sha256,
    numeric_summary,
    official_test_locations,
)

EXPECTED_PROTOCOL = Path("configs/mars_jpl_ornl_header_bridge_protocol.json")
STAGE_A_CMR_PREFLIGHT = Path(
    ".research/jpl_operational_ghg_supplement/ornl_stage_a_cmr_preflight.json"
)
TRAIN_DEFINITION_NAME = "multicampaign_train.csv"
CMR_ENDPOINT = "https://cmr.earthdata.nasa.gov/search/granules.umm_json"
COLLECTION_CONCEPT_ID = "C2662359874-ORNL_CLOUD"
CMR_NATIVE_PREFIX = "AVIRIS-NG_L1B_radiance."
MAX_HEADER_BYTES = 262_144
GROUP_RADIUS_KM = 25.0
CAMPAIGNS = frozenset({"COVID", "Permian"})
FORBIDDEN_CAMPAIGNS = frozenset({"CACH4", "EMIT"})
FLIGHT_RE = re.compile(r"^ang\d{8}t\d{6}$")
TILE_RE = re.compile(
    r"^(?P<campaign>COVID|Permian)/(?P<flight>ang\d{8}t\d{6})_"
    r"(?P<product>ch4mf_v2[xy]1_img)_tile"
    r"(?P<width>\d+)x(?P<height>\d+)\+"
    r"(?P<sample_off>\d+)\+(?P<line_off>\d+)\.tif$"
)
ALLOWED_HEADER_HOSTS = frozenset(
    {
        "data.ornldaac.earthdata.nasa.gov",
        "ornl-cumulus-prod-protected.s3.amazonaws.com",
        "ornl-cumulus-prod-protected.s3.us-west-2.amazonaws.com",
    }
)


class BridgeError(RuntimeError):
    """Raised when the frozen bridge contract cannot be satisfied."""


class GranuleSelectionError(BridgeError):
    """Raised for missing or ambiguous CMR header granules."""


class AuthenticationRequired(BridgeError):
    """Raised when Earthdata redirects to an interactive login response."""


@dataclass(frozen=True)
class CandidateTile:
    campaign: str
    tilepath: str
    labelpath: str
    label: int
    flight: str
    product: str
    width: int
    height: int
    sample_offset: int
    line_offset: int

    @property
    def header_version(self) -> str:
        return self.product.split("_")[1]


@dataclass(frozen=True)
class HeaderGranule:
    flight: str
    native_id: str
    concept_id: str | None
    url: str
    declared_bytes: int | None
    checksum: str | None
    checksum_algorithm: str | None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    partial.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
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


def validate_protocol_path(path: Path) -> None:
    if path.resolve() != EXPECTED_PROTOCOL.resolve():
        raise ValueError("Only the committed frozen ORNL bridge protocol is permitted")


def _validate_frozen_path(
    *,
    specification: dict[str, object],
    role: str,
    hash_field: str = "sha256",
) -> dict[str, object]:
    path = Path(str(specification["path"]))
    if not path.exists():
        raise ValueError(f"Frozen {role} is missing: {path}")
    if "bytes" in specification and path.stat().st_size != int(
        specification["bytes"]
    ):
        raise ValueError(f"Frozen {role} byte count mismatch")
    observed_hash = sha256_file(path)
    if observed_hash != specification[hash_field]:
        raise ValueError(f"Frozen {role} SHA-256 mismatch")
    return {
        "path": path.as_posix(),
        "bytes": path.stat().st_size,
        "sha256": observed_hash,
    }


def validate_frozen_inputs(
    protocol_path: Path,
) -> tuple[dict[str, object], dict[str, object]]:
    validate_protocol_path(protocol_path)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    receipts: dict[str, object] = {
        "protocol": {
            "path": protocol_path.as_posix(),
            "sha256": sha256_file(protocol_path),
        }
    }
    for key, specification in protocol["frozen_inputs"].items():
        if key == "prior_stage_b_pairs":
            receipt = _validate_frozen_path(
                specification=specification,
                role=key,
                hash_field="byte_sha256",
            )
            path = Path(str(specification["path"]))
            normalized = normalized_jsonl_sha256(path)
            if normalized != specification["normalized_lf_sha256"]:
                raise ValueError("Frozen prior pair normalized-LF SHA-256 mismatch")
            receipt["byte_sha256"] = receipt.pop("sha256")
            receipt["normalized_lf_sha256"] = normalized
            receipts[key] = receipt
            continue
        receipts[key] = _validate_frozen_path(
            specification=specification,
            role=key,
        )

    safe_spec = protocol["frozen_inputs"]["safe_mars_manifest"]
    if set(safe_spec["permitted_columns"]) != SAFE_MARS_COLUMNS:
        raise ValueError("Frozen safe MARS columns differ from implementation")
    if set(safe_spec["forbidden_columns"]) != FORBIDDEN_MARS_COLUMNS:
        raise ValueError("Frozen protected MARS columns differ from implementation")
    stage_b = json.loads(
        Path(protocol["frozen_inputs"]["prior_stage_b_report"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    declared = stage_b["ignored_pair_catalog"]["sha256"]
    expected = protocol["frozen_inputs"]["prior_stage_b_pairs"][
        "normalized_lf_sha256"
    ]
    if declared != expected:
        raise ValueError("Stage B receipt does not bind the frozen normalized pair hash")
    pair_spec = protocol["frozen_inputs"]["prior_stage_b_pairs"]
    if stage_b["ignored_pair_catalog"]["path"] != pair_spec["path"]:
        raise ValueError("Stage B receipt pair path differs from frozen bridge input")
    mask_spec = protocol["frozen_inputs"]["prior_mask_catalog"]
    if stage_b["inputs"]["mask_catalog"]["path"] != mask_spec["path"]:
        raise ValueError("Stage B receipt mask path differs from frozen bridge input")
    if stage_b["inputs"]["mask_catalog"]["sha256"] != mask_spec["sha256"]:
        raise ValueError("Stage B receipt mask hash differs from frozen bridge input")
    source = protocol["authoritative_metadata_source"]
    if source["collection_concept_id"] != COLLECTION_CONCEPT_ID:
        raise ValueError("Frozen ORNL collection differs from implementation")
    boundary = protocol["network_and_content_boundary"]
    if int(boundary["maximum_header_bytes_each"]) != MAX_HEADER_BYTES:
        raise ValueError("Frozen maximum header size differs from implementation")
    if set(protocol["candidate_cohort"]["included_campaigns"]) != CAMPAIGNS:
        raise ValueError("Frozen candidate campaigns differ from implementation")
    if not FORBIDDEN_CAMPAIGNS <= set(
        protocol["candidate_cohort"]["excluded_campaigns"]
    ):
        raise ValueError("Frozen excluded campaigns differ from implementation")
    detail_root = Path(boundary["detailed_metadata_location"])
    expected_detail_root = Path(
        ".research/jpl_operational_ghg_supplement/ornl_l1b_headers"
    )
    if detail_root.resolve() != expected_detail_root.resolve():
        raise ValueError("Frozen detailed header root differs from implementation")
    for key in (
        "ignored_cmr_receipt",
        "ignored_header_manifest",
        "ignored_resolved_rows",
        "ignored_filtered_rows",
    ):
        if not Path(protocol["outputs"][key]).resolve().is_relative_to(
            Path(".research").resolve()
        ):
            raise ValueError(f"Detailed bridge output escapes ignored .research: {key}")
    return protocol, receipts


def parse_candidate_tile(row: dict[str, str]) -> CandidateTile:
    tilepath = row.get("tilepath", "")
    campaign = tilepath.split("/", 1)[0]
    if campaign in FORBIDDEN_CAMPAIGNS or campaign not in CAMPAIGNS:
        raise ValueError(f"Campaign is not authorized by the bridge: {campaign}")
    match = TILE_RE.fullmatch(tilepath)
    if not match:
        raise ValueError(f"Unexpected COVID/Permian tile path: {tilepath}")
    label = int(row.get("label", ""))
    if label not in {0, 1}:
        raise ValueError(f"Invalid released class for {tilepath}: {label}")
    return CandidateTile(
        campaign=match.group("campaign"),
        tilepath=tilepath,
        labelpath=row.get("labelpath", ""),
        label=label,
        flight=match.group("flight"),
        product=match.group("product"),
        width=int(match.group("width")),
        height=int(match.group("height")),
        sample_offset=int(match.group("sample_off")),
        line_offset=int(match.group("line_off")),
    )


def read_candidate_train_definition(
    path: Path, *, expected_sha256: str
) -> list[CandidateTile]:
    if path.name != TRAIN_DEFINITION_NAME:
        raise ValueError(
            f"Only {TRAIN_DEFINITION_NAME} is permitted; got {path.name}"
        )
    if sha256_file(path) != expected_sha256:
        raise ValueError("Released train definition SHA-256 mismatch")
    rows: list[CandidateTile] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["tilepath", "labelpath", "label"]:
            raise ValueError("Unexpected released train-definition schema")
        for source in reader:
            campaign = source.get("tilepath", "").split("/", 1)[0]
            if campaign in CAMPAIGNS:
                rows.append(parse_candidate_tile(source))
    return rows


def validate_candidate_definition_counts(
    definitions: list[CandidateTile], protocol: dict[str, object]
) -> dict[str, int]:
    known = protocol["candidate_cohort"]["known_counts"]
    counts = {
        "covid_background_rows": sum(
            tile.campaign == "COVID" and tile.label == 0 for tile in definitions
        ),
        "covid_plume_rows": sum(
            tile.campaign == "COVID" and tile.label == 1 for tile in definitions
        ),
        "covid_flightlines": len(
            {tile.flight for tile in definitions if tile.campaign == "COVID"}
        ),
        "permian_background_rows": sum(
            tile.campaign == "Permian" and tile.label == 0 for tile in definitions
        ),
        "permian_plume_rows": sum(
            tile.campaign == "Permian" and tile.label == 1 for tile in definitions
        ),
        "permian_flightlines": len(
            {tile.flight for tile in definitions if tile.campaign == "Permian"}
        ),
    }
    counts["combined_background_rows"] = (
        counts["covid_background_rows"] + counts["permian_background_rows"]
    )
    counts["combined_flightlines_before_cross_campaign_deduplication"] = (
        counts["covid_flightlines"] + counts["permian_flightlines"]
    )
    for key, value in counts.items():
        if value != int(known[key]):
            raise ValueError(f"Released candidate count mismatch for {key}: {value}")
    versions = {
        campaign: {tile.header_version for tile in definitions if tile.campaign == campaign}
        for campaign in sorted(CAMPAIGNS)
    }
    if versions != {"COVID": {"v2y1"}, "Permian": {"v2x1"}}:
        raise ValueError(f"Released campaign header versions changed: {versions}")
    return counts


def read_anchor_manifest(
    path: Path, *, expected_count: int
) -> list[dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    headers = payload.get("headers", [])
    flights = [str(record.get("flight", "")) for record in headers]
    if len(headers) != expected_count or len(set(flights)) != expected_count:
        raise ValueError("CACH4 anchor header manifest count or identity mismatch")
    if any(not FLIGHT_RE.fullmatch(flight) for flight in flights):
        raise ValueError("Invalid CACH4 anchor flight identity")
    return headers


def expected_header_native_id(flight: str, version: str) -> str:
    if not FLIGHT_RE.fullmatch(flight) or not re.fullmatch(r"v\d+[a-z]\d+", version):
        raise ValueError(f"Invalid flight/version identity: {flight}/{version}")
    return f"{flight}_rdn_{version}_img.hdr"


def expected_cmr_native_id(flight: str, version: str) -> str:
    return CMR_NATIVE_PREFIX + expected_header_native_id(flight, version)


def native_id_pattern(
    flight: str, expected_versions: dict[str, str]
) -> str:
    if flight not in expected_versions or not FLIGHT_RE.fullmatch(flight):
        raise ValueError(f"CMR query is not authorized for flight: {flight}")
    return expected_cmr_native_id(flight, expected_versions[flight])


def cmr_query_params(
    flight: str, expected_versions: dict[str, str]
) -> dict[str, object]:
    return {
        "collection_concept_id": COLLECTION_CONCEPT_ID,
        "native_id[]": native_id_pattern(flight, expected_versions),
        "options[native_id][pattern]": "true",
        "page_size": 50,
    }


def cmr_query_url(flight: str, expected_versions: dict[str, str]) -> str:
    return CMR_ENDPOINT + "?" + urllib.parse.urlencode(
        cmr_query_params(flight, expected_versions), doseq=True
    )


def _native_id(item: dict[str, object]) -> str:
    meta = item.get("meta", {})
    umm = item.get("umm", {})
    meta_id = str(meta.get("native-id", "")) if isinstance(meta, dict) else ""
    granule_id = str(umm.get("GranuleUR", "")) if isinstance(umm, dict) else ""
    if meta_id and granule_id and meta_id != granule_id:
        raise GranuleSelectionError("CMR native-id and GranuleUR disagree")
    return meta_id or granule_id


def _distribution_entry(
    umm: dict[str, object], *, cmr_native_id: str, header_name: str
) -> dict[str, object] | None:
    data_granule = umm.get("DataGranule", {})
    if not isinstance(data_granule, dict):
        return None
    entries = data_granule.get("ArchiveAndDistributionInformation", [])
    materialized = [entry for entry in entries if isinstance(entry, dict)] if isinstance(entries, list) else []
    exact = [
        entry
        for entry in materialized
        if str(entry.get("Name", "")) in {cmr_native_id, header_name}
    ]
    if len(exact) == 1:
        return exact[0]
    if len(exact) > 1:
        raise GranuleSelectionError("CMR distribution metadata is ambiguous")
    # ORNL currently publishes one file per CMR granule but records its
    # distribution Name as the literal string "Not provided". A single entry
    # remains unambiguous because the granule identity itself is exact-bound.
    if len(materialized) == 1 and str(materialized[0].get("Name", "")) in {
        "",
        "Not provided",
    }:
        return materialized[0]
    return None


def _declared_header_bytes(entry: dict[str, object] | None) -> int | None:
    if entry is None:
        return None
    if entry.get("SizeInBytes") is not None:
        return int(entry["SizeInBytes"])
    if str(entry.get("SizeUnit", "")).lower() in {"byte", "bytes"}:
        return int(float(entry["Size"]))
    return None


def _checksum(entry: dict[str, object] | None) -> tuple[str | None, str | None]:
    checksum = entry.get("Checksum") if entry is not None else None
    if checksum is None:
        return None, None
    if not isinstance(checksum, dict):
        raise GranuleSelectionError("CMR checksum metadata is not an object")
    algorithm = str(checksum.get("Algorithm", ""))
    value = str(checksum.get("Value", ""))
    if algorithm.upper() not in {"SHA-256", "SHA256"}:
        raise GranuleSelectionError(f"Unsupported CMR checksum algorithm: {algorithm}")
    if not re.fullmatch(r"[0-9a-fA-F]{64}", value):
        raise GranuleSelectionError("CMR SHA-256 value is malformed")
    return value.lower(), "SHA-256"


def _allowed_header_host(host: str | None) -> bool:
    return host in ALLOWED_HEADER_HOSTS


def validate_header_asset_url(url: str, expected_native_id: str) -> None:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or not _allowed_header_host(parsed.hostname):
        raise BridgeError(f"Header URL is outside the allowed ORNL data hosts: {url}")
    if Path(urllib.parse.unquote(parsed.path)).name != expected_native_id:
        raise BridgeError("Header URL does not name the selected ENVI header")
    if not re.fullmatch(r"ang\d{8}t\d{6}_rdn_v[^/]+_img\.hdr", expected_native_id):
        raise BridgeError("Selected asset is not an orthocorrected radiance img header")


def select_header_granule(
    flight: str, expected_version: str, response: dict[str, object]
) -> HeaderGranule:
    expected_header_name = expected_header_native_id(flight, expected_version)
    expected_native_id = expected_cmr_native_id(flight, expected_version)
    items = response.get("items", [])
    matches: list[tuple[dict[str, object], str]] = []
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        native_id = _native_id(item)
        if native_id == expected_native_id:
            matches.append((item, native_id))
    if len(matches) != 1:
        raise GranuleSelectionError(
            f"Expected exactly one {expected_native_id}; found {len(matches)}"
        )
    item, native_id = matches[0]
    umm = item.get("umm", {})
    if not isinstance(umm, dict):
        raise GranuleSelectionError("Selected CMR item has no UMM object")
    related = umm.get("RelatedUrls", [])
    urls: list[str] = []
    for record in related if isinstance(related, list) else []:
        if not isinstance(record, dict) or not record.get("URL"):
            continue
        url = str(record["URL"])
        parsed = urllib.parse.urlparse(url)
        if (
            parsed.scheme == "https"
            and _allowed_header_host(parsed.hostname)
            and Path(urllib.parse.unquote(parsed.path)).name == expected_header_name
        ):
            urls.append(url)
    urls = sorted(set(urls))
    if len(urls) != 1:
        raise GranuleSelectionError(
            f"Expected exactly one downloadable URL for {native_id}; found {len(urls)}"
        )
    validate_header_asset_url(urls[0], expected_header_name)
    distribution = _distribution_entry(
        umm,
        cmr_native_id=expected_native_id,
        header_name=expected_header_name,
    )
    declared_bytes = _declared_header_bytes(distribution)
    checksum, checksum_algorithm = _checksum(distribution)
    if declared_bytes is not None and declared_bytes > MAX_HEADER_BYTES:
        raise BridgeError(f"CMR-declared header is too large: {declared_bytes} bytes")
    meta = item.get("meta", {})
    concept_id = str(meta.get("concept-id")) if isinstance(meta, dict) and meta.get("concept-id") else None
    return HeaderGranule(
        flight=flight,
        native_id=expected_header_name,
        concept_id=concept_id,
        url=urls[0],
        declared_bytes=declared_bytes,
        checksum=checksum,
        checksum_algorithm=checksum_algorithm,
    )


def query_header_granule(
    session: requests.Session,
    *,
    flight: str,
    expected_versions: dict[str, str],
) -> tuple[HeaderGranule, str]:
    params = cmr_query_params(flight, expected_versions)
    response = session.get(
        CMR_ENDPOINT,
        params=params,
        headers={"Accept": "application/json"},
        timeout=60,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise BridgeError("CMR did not return a JSON object")
    return select_header_granule(
        flight, expected_versions[flight], payload
    ), cmr_query_url(
        flight, expected_versions
    )


def validate_header_response(
    response: requests.Response, *, expected_native_id: str
) -> None:
    response.raise_for_status()
    final = urllib.parse.urlparse(str(response.url))
    if final.hostname == "urs.earthdata.nasa.gov":
        raise AuthenticationRequired("Earthdata authentication redirected to login")
    validate_header_asset_url(str(response.url), expected_native_id)
    content_type = str(response.headers.get("Content-Type", "")).lower()
    if any(value in content_type for value in ("text/html", "text/xml", "application/xml")):
        raise AuthenticationRequired("Earthdata returned an authentication or HTML response")
    length = response.headers.get("Content-Length")
    if length is not None and int(length) > MAX_HEADER_BYTES:
        raise BridgeError("Header Content-Length exceeds frozen maximum")


def fetch_header(
    session: requests.Session, granule: HeaderGranule
) -> tuple[bytes, dict[str, object]]:
    response = session.get(
        granule.url,
        stream=True,
        allow_redirects=True,
        timeout=120,
    )
    validate_header_response(response, expected_native_id=granule.native_id)
    chunks: list[bytes] = []
    size = 0
    for chunk in response.iter_content(chunk_size=16 * 1024):
        if not chunk:
            continue
        size += len(chunk)
        if size > MAX_HEADER_BYTES:
            raise BridgeError("Downloaded header exceeds frozen maximum")
        chunks.append(chunk)
    payload = b"".join(chunks)
    if not payload.lstrip().startswith(b"ENVI"):
        raise BridgeError("Downloaded asset is not an ENVI header")
    if granule.checksum is not None and sha256_bytes(payload) != granule.checksum:
        raise BridgeError("Downloaded header differs from the CMR SHA-256")
    parse_envi_header(payload.decode("utf-8"))
    return payload, {
        "final_url": str(response.url),
        "content_type": response.headers.get("Content-Type"),
        "content_length": response.headers.get("Content-Length"),
        "redirects": [str(item.url) for item in response.history],
    }


def acquire_flight_headers(
    *,
    session: requests.Session,
    flight_versions: dict[str, str],
    allowed_versions: dict[str, str],
    header_root: Path,
    stage: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    receipts: list[dict[str, object]] = []
    headers: list[dict[str, object]] = []
    for flight in sorted(flight_versions):
        if allowed_versions.get(flight) != flight_versions[flight]:
            raise ValueError(f"Flight/version is outside the frozen query plan: {flight}")
        query_url = cmr_query_url(flight, allowed_versions)
        try:
            granule, _ = query_header_granule(
                session, flight=flight, expected_versions=allowed_versions
            )
            payload, response_metadata = fetch_header(session, granule)
        except AuthenticationRequired:
            raise
        except BridgeError as error:
            receipts.append(
                {
                    "stage": stage,
                    "flight": flight,
                    "native_id_pattern": native_id_pattern(flight, allowed_versions),
                    "query_url": query_url,
                    "status": "unresolved_fail_closed",
                    "reason": str(error),
                }
            )
            continue
        output = header_root / flight / granule.native_id
        output.parent.mkdir(parents=True, exist_ok=True)
        partial = output.with_name(output.name + ".part")
        partial.write_bytes(payload)
        partial.replace(output)
        info = parse_envi_header(payload.decode("utf-8"))
        record = {
            "stage": stage,
            "flight": flight,
            "native_id": granule.native_id,
            "concept_id": granule.concept_id,
            "source_url": granule.url,
            "path": output.as_posix(),
            "bytes": len(payload),
            "sha256": sha256_bytes(payload),
            "declared_bytes": granule.declared_bytes,
            "source_checksum": granule.checksum,
            "source_checksum_algorithm": granule.checksum_algorithm,
            "samples": info.samples,
            "lines": info.lines,
            "epsg": info.epsg,
            "rotation_degrees": info.rotation_degrees,
        }
        headers.append(record)
        receipts.append(
            {
                "stage": stage,
                "flight": flight,
                "native_id_pattern": native_id_pattern(flight, allowed_versions),
                "query_url": query_url,
                "status": "resolved_header_only",
                "selected": record,
                "response": response_metadata,
            }
        )
    return receipts, headers


def preflight_header_granules(
    *,
    session: requests.Session,
    flight_versions: dict[str, str],
    allowed_versions: dict[str, str],
    stage: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Resolve frozen CMR identities without requesting any header content."""

    receipts: list[dict[str, object]] = []
    granules: list[dict[str, object]] = []
    for flight in sorted(flight_versions):
        if allowed_versions.get(flight) != flight_versions[flight]:
            raise ValueError(f"Flight/version is outside the frozen query plan: {flight}")
        query_url = cmr_query_url(flight, allowed_versions)
        try:
            granule, _ = query_header_granule(
                session, flight=flight, expected_versions=allowed_versions
            )
        except (BridgeError, requests.RequestException, ValueError) as error:
            receipts.append(
                {
                    "stage": stage,
                    "flight": flight,
                    "native_id_pattern": native_id_pattern(
                        flight, allowed_versions
                    ),
                    "query_url": query_url,
                    "status": "unresolved_fail_closed",
                    "reason": str(error),
                }
            )
            continue
        record = {
            "stage": stage,
            "flight": flight,
            "native_id": granule.native_id,
            "concept_id": granule.concept_id,
            "source_url": granule.url,
            "declared_bytes": granule.declared_bytes,
            "source_checksum": granule.checksum,
            "source_checksum_algorithm": granule.checksum_algorithm,
        }
        granules.append(record)
        receipts.append(
            {
                "stage": stage,
                "flight": flight,
                "native_id_pattern": native_id_pattern(flight, allowed_versions),
                "query_url": query_url,
                "status": "resolved_cmr_metadata_only",
                "selected": record,
            }
        )
    return receipts, granules


def _pixel_vector_lengths(info: EnviMapInfo) -> tuple[float, float]:
    gt = gdal_geotransform(info)
    return math.hypot(gt[1], gt[4]), math.hypot(gt[2], gt[5])


def _projected_point(info: EnviMapInfo, x: float, y: float) -> tuple[float, float]:
    gt = gdal_geotransform(info)
    return gt[0] + x * gt[1] + y * gt[2], gt[3] + x * gt[4] + y * gt[5]


def _reference_pixel_delta(
    reference: EnviMapInfo, delta_x: float, delta_y: float
) -> tuple[float, float]:
    gt = gdal_geotransform(reference)
    determinant = gt[1] * gt[5] - gt[2] * gt[4]
    if abs(determinant) < 1e-12:
        raise ValueError("Degenerate ENVI affine")
    pixel_x = (delta_x * gt[5] - gt[2] * delta_y) / determinant
    pixel_y = (gt[1] * delta_y - delta_x * gt[4]) / determinant
    return pixel_x, pixel_y


def compare_grids(
    jpl: EnviMapInfo,
    nasa: EnviMapInfo,
    *,
    relative_pixel_tolerance: float = 1e-6,
    maximum_discrepancy_pixels: float = 0.25,
) -> dict[str, object]:
    dimensions_match = jpl.samples == nasa.samples and jpl.lines == nasa.lines
    crs_match = jpl.epsg == nasa.epsg
    jpl_vectors = _pixel_vector_lengths(jpl)
    nasa_vectors = _pixel_vector_lengths(nasa)
    pixel_size_match = all(
        math.isclose(left, right, rel_tol=relative_pixel_tolerance, abs_tol=0.0)
        for left, right in zip(jpl_vectors, nasa_vectors, strict=True)
    )
    points = (
        (0.5, 0.5),
        (jpl.samples - 0.5, 0.5),
        (0.5, jpl.lines - 0.5),
        (jpl.samples - 0.5, jpl.lines - 0.5),
        (jpl.samples / 2.0, jpl.lines / 2.0),
    )
    discrepancies: list[float] = []
    if dimensions_match and crs_match:
        for x, y in points:
            jpl_x, jpl_y = _projected_point(jpl, x, y)
            nasa_x, nasa_y = _projected_point(nasa, x, y)
            dx, dy = _reference_pixel_delta(jpl, nasa_x - jpl_x, nasa_y - jpl_y)
            discrepancies.append(math.hypot(dx, dy))
    maximum = max(discrepancies, default=math.inf)
    mapping_match = maximum <= maximum_discrepancy_pixels
    return {
        "samples_and_lines_exact": dimensions_match,
        "projected_crs_exact": crs_match,
        "pixel_vector_lengths_relative_tolerance": pixel_size_match,
        "reference_pixel_convention": "ENVI_one_based_1_1",
        "points_compared": len(discrepancies),
        "maximum_discrepancy_pixels": maximum,
        "pixel_index_mapping_pass": mapping_match,
        "pass": dimensions_match and crs_match and pixel_size_match and mapping_match,
    }


def stage_a_decision(
    *,
    total_anchors: int,
    resolved_anchors: int,
    mismatch_count: int,
    minimum_resolved: int,
    minimum_fraction: float,
) -> dict[str, object]:
    fraction = resolved_anchors / total_anchors if total_anchors else 0.0
    gates = {
        "minimum_resolved_anchor_flightlines": resolved_anchors >= minimum_resolved,
        "minimum_resolved_anchor_fraction": fraction >= minimum_fraction,
        "zero_grid_mismatches": mismatch_count == 0,
    }
    return {
        "total_anchor_flightlines": total_anchors,
        "resolved_anchor_flightlines": resolved_anchors,
        "resolved_anchor_fraction": fraction,
        "grid_mismatches": mismatch_count,
        "gates": gates,
        "pass": all(gates.values()),
    }


def _read_verified_header(record: dict[str, object]) -> EnviMapInfo:
    path = Path(str(record["path"]))
    if path.suffix.lower() != ".hdr" or path.stat().st_size > MAX_HEADER_BYTES:
        raise ValueError(f"Unsafe header artifact: {path}")
    if sha256_file(path) != record["sha256"]:
        raise ValueError(f"Header SHA-256 mismatch: {path}")
    return parse_envi_header(path.read_text(encoding="utf-8"))


def evaluate_stage_a(
    *,
    anchor_records: list[dict[str, object]],
    nasa_records: list[dict[str, object]],
    minimum_resolved: int,
    minimum_fraction: float,
    relative_pixel_tolerance: float,
    maximum_discrepancy_pixels: float,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    nasa_by_flight: dict[str, dict[str, object]] = {}
    for record in nasa_records:
        if record.get("stage") != "stage_a_cach4_grid_bridge":
            continue
        flight = str(record["flight"])
        if flight in nasa_by_flight:
            raise ValueError(f"Ambiguous NASA header records for anchor {flight}")
        nasa_by_flight[flight] = record
    comparisons: list[dict[str, object]] = []
    resolved = 0
    mismatches = 0
    for anchor in sorted(anchor_records, key=lambda row: str(row["flight"])):
        flight = str(anchor["flight"])
        nasa_record = nasa_by_flight.get(flight)
        if nasa_record is None:
            comparisons.append({"flight": flight, "status": "unresolved"})
            continue
        resolved += 1
        jpl = _read_verified_header(anchor)
        nasa = _read_verified_header(nasa_record)
        comparison = compare_grids(
            jpl,
            nasa,
            relative_pixel_tolerance=relative_pixel_tolerance,
            maximum_discrepancy_pixels=maximum_discrepancy_pixels,
        )
        if not comparison["pass"]:
            mismatches += 1
        comparisons.append(
            {
                "flight": flight,
                "status": "match" if comparison["pass"] else "mismatch",
                "comparison": comparison,
            }
        )
    result = stage_a_decision(
        total_anchors=len(anchor_records),
        resolved_anchors=resolved,
        mismatch_count=mismatches,
        minimum_resolved=minimum_resolved,
        minimum_fraction=minimum_fraction,
    )
    result["unresolved_anchor_flightlines"] = len(anchor_records) - resolved
    return result, comparisons


def ensure_stage_a_pass(stage_a: dict[str, object]) -> None:
    if stage_a.get("pass") is not True:
        raise BridgeError(
            "Stage A grid bridge did not pass; COVID/Permian acquisition is forbidden"
        )


def _candidate_center_utm(
    tile: CandidateTile, info: EnviMapInfo
) -> tuple[float, float, int, int]:
    center_x = tile.sample_offset + tile.width / 2.0
    center_y = tile.line_offset + tile.height / 2.0
    if not 0.0 <= center_x < info.samples or not 0.0 <= center_y < info.lines:
        raise ValueError(f"Crop center is outside the NASA grid: {tile.tilepath}")
    gt = gdal_geotransform(info)
    easting = gt[0] + center_x * gt[1] + center_y * gt[2]
    northing = gt[3] + center_x * gt[4] + center_y * gt[5]
    return (
        easting,
        northing,
        max(0, tile.sample_offset + tile.width - info.samples),
        max(0, tile.line_offset + tile.height - info.lines),
    )


def resolve_candidate_negatives(
    *,
    definitions: list[CandidateTile],
    nasa_records: list[dict[str, object]],
) -> tuple[list[dict[str, object]], dict[str, int]]:
    if any(tile.campaign not in CAMPAIGNS for tile in definitions):
        raise ValueError("Candidate definitions contain a forbidden campaign")
    records_by_flight: dict[str, dict[str, object]] = {}
    info_by_flight: dict[str, EnviMapInfo] = {}
    for record in nasa_records:
        if record.get("stage") != "stage_b_covid_permian_metadata":
            continue
        flight = str(record["flight"])
        if flight in records_by_flight:
            raise ValueError(f"Ambiguous NASA candidate headers for {flight}")
        records_by_flight[flight] = record
        info_by_flight[flight] = _read_verified_header(record)

    negatives = [tile for tile in definitions if tile.label == 0]
    resolved: list[dict[str, object]] = []
    missing_header = 0
    invalid_center = 0
    for tile in negatives:
        header = records_by_flight.get(tile.flight)
        if header is None:
            missing_header += 1
            continue
        info = info_by_flight[tile.flight]
        try:
            easting, northing, sample_overhang, line_overhang = _candidate_center_utm(
                tile, info
            )
        except ValueError:
            invalid_center += 1
            continue
        transformer = Transformer.from_crs(info.epsg, 4326, always_xy=True)
        longitude, latitude = transformer.transform(easting, northing)
        if not (
            math.isfinite(longitude)
            and math.isfinite(latitude)
            and -180.0 <= longitude <= 180.0
            and -90.0 <= latitude <= 90.0
        ):
            invalid_center += 1
            continue
        resolved.append(
            {
                "sample_id": Path(tile.tilepath).stem,
                "sensor": "AVIRIS-NG",
                "tile": tile.flight,
                "timestamp": flight_timestamp(tile.flight),
                "longitude": longitude,
                "latitude": latitude,
                "label_state": "NO_PLUME",
                "coordinate_resolved": True,
                "eligible_for_target_catalog": False,
                "eligibility_status": "pending_frozen_25km_filter",
                "country": "United States",
                "group_id": None,
                "novel_beyond_all_mars_25km": None,
                "published_split": "train",
                "source_campaign": tile.campaign,
                "source_tile_product": tile.product,
                "source_tile_path": tile.tilepath,
                "source_label_path": tile.labelpath,
                "crop_width": tile.width,
                "crop_height": tile.height,
                "sample_offset": tile.sample_offset,
                "line_offset": tile.line_offset,
                "sample_axis_overhang_pixels": sample_overhang,
                "line_axis_overhang_pixels": line_overhang,
                "source_crs_epsg": info.epsg,
                "center_easting": easting,
                "center_northing": northing,
                "header_native_id": header["native_id"],
                "header_sha256": header["sha256"],
                "coordinate_source": "NASA_ORNL_AVIRIS_NG_L1B_ENVI_header",
            }
        )
    return resolved, {
        "released_negative_rows": len(negatives),
        "resolved_negative_rows": len(resolved),
        "rows_missing_header": missing_header,
        "rows_with_invalid_center": invalid_center,
    }


def summarize_stage_b(
    *,
    resolved: list[dict[str, object]],
    filtered: list[dict[str, object]],
    resolution_counts: dict[str, int],
    minimum_rows: int,
    minimum_components: int,
) -> dict[str, object]:
    if any(row.get("source_campaign") not in CAMPAIGNS for row in filtered):
        raise AssertionError("Stage B output contains CACH4, EMIT, or another campaign")
    eligible = [row for row in filtered if row["eligible_for_target_catalog"]]
    groups = {str(row["group_id"]) for row in eligible}
    exact = all(
        row.get("timestamp")
        and math.isfinite(float(row["latitude"]))
        and math.isfinite(float(row["longitude"]))
        for row in resolved
    )
    gates = {
        "minimum_resolved_train_background_rows": len(resolved) >= minimum_rows,
        "exact_utc_and_finite_wgs84_center_for_every_retained_row": exact,
        "minimum_eligible_25km_connected_components": len(groups)
        >= minimum_components,
    }
    counts = {
        **resolution_counts,
        "resolved_flightlines": len({str(row["tile"]) for row in resolved}),
        "protected_rows": sum(bool(row["mars_test_protected"]) for row in filtered),
        "prior_pair_duplicate_rows": sum(
            bool(row["prior_pair_duplicate_25km"]) for row in filtered
        ),
        "eligible_rows": len(eligible),
        "eligible_flightlines": len({str(row["tile"]) for row in eligible}),
        "eligible_25km_components": len(groups),
        "campaign_rows": dict(
            sorted(Counter(str(row["source_campaign"]) for row in resolved).items())
        ),
    }
    return {
        "counts": counts,
        "distance_summaries_km": {
            "nearest_official_mars_test": numeric_summary(
                float(row["nearest_mars_test_km"]) for row in filtered
            ),
            "nearest_prior_negative": numeric_summary(
                float(row["nearest_prior_negative_pair_km"]) for row in filtered
            ),
            "nearest_any_mars": numeric_summary(
                float(row["nearest_any_mars_km"]) for row in filtered
            ),
        },
        "gates": gates,
        "pass": all(gates.values()),
    }


def _load_header_manifest(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return list(payload.get("headers", []))


def _save_header_manifest(path: Path, headers: list[dict[str, object]]) -> None:
    identities = [(str(row["stage"]), str(row["flight"])) for row in headers]
    if len(identities) != len(set(identities)):
        raise ValueError("Duplicate stage/flight identities in ORNL header manifest")
    write_json(
        path,
        {
            "schema_version": 1,
            "collection_concept_id": COLLECTION_CONCEPT_ID,
            "headers": sorted(headers, key=lambda row: (str(row["stage"]), str(row["flight"]))),
        },
    )


def _compact_report(
    *,
    protocol: dict[str, object],
    input_receipts: dict[str, object],
    stage_a: dict[str, object],
    stage_b: dict[str, object] | None,
    output_hashes: dict[str, object],
) -> dict[str, object]:
    final_pass = bool(stage_a["pass"]) and bool(stage_b and stage_b["pass"])
    return {
        "schema_version": 1,
        "scope": "NASA_ORNL_L1B_headers_only_no_target_catalog_or_assets",
        "decision": "PASS" if final_pass else "FAIL",
        "stage_a_cach4_grid_bridge": stage_a,
        "stage_b_covid_permian_metadata": stage_b,
        "inputs": input_receipts,
        "outputs": output_hashes,
        "security_boundary": {
            "cmr_queries_limited_to_explicit_released_train_flight_ids": True,
            "downloaded_content_type": "ENVI_radiance_img_headers_only",
            "maximum_header_bytes_each": MAX_HEADER_BYTES,
            "jpl_test_or_emit_accessed": False,
            "mars_protected_outcomes_accessed": [],
            "target_catalog_queried": False,
            "target_asset_downloaded": False,
        },
        "claim_boundary": protocol["claim_boundary"],
    }


def write_markdown(report: dict[str, object], path: Path) -> None:
    stage_a = report["stage_a_cach4_grid_bridge"]
    stage_b = report.get("stage_b_covid_permian_metadata")
    lines = [
        "# NASA ORNL AVIRIS-NG header bridge",
        "",
        f"**Decision: {report['decision']}.**",
        "",
        "This audit used only hash-bound released train definitions, public JPL/ORNL ENVI header metadata, safe-column MARS locations, and prior-pair receipts. It did not access JPL test or EMIT content, protected MARS outcomes, target catalogs, or target assets.",
        "",
        "## Stage A — CACH4 grid bridge",
        "",
        f"- Resolved anchors: {stage_a['resolved_anchor_flightlines']:,} / {stage_a['total_anchor_flightlines']:,}",
        f"- Grid mismatches: {stage_a['grid_mismatches']:,}",
        f"- Decision: **{'PASS' if stage_a['pass'] else 'FAIL'}**",
    ]
    if stage_b is not None:
        counts = stage_b["counts"]
        lines.extend(
            [
                "",
                "## Stage B — COVID + Permian train backgrounds",
                "",
                f"- Resolved negatives: {counts['resolved_negative_rows']:,}",
                f"- Eligible rows / flightlines: {counts['eligible_rows']:,} / {counts['eligible_flightlines']:,}",
                f"- Eligible 25 km components: {counts['eligible_25km_components']:,}",
                f"- Decision: **{'PASS' if stage_b['pass'] else 'FAIL'}**",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "Stage B was not authorized because Stage A did not pass.",
            ]
        )
    lines.extend(
        [
            "",
            "## Claim boundary",
            "",
            str(report["claim_boundary"]),
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def execute_authenticated_bridge(
    *,
    protocol_path: Path,
    session: requests.Session,
) -> dict[str, object]:
    protocol, input_receipts = validate_frozen_inputs(protocol_path)
    frozen = protocol["frozen_inputs"]
    definitions = read_candidate_train_definition(
        Path(frozen["jpl_train_definition"]["path"]),
        expected_sha256=str(frozen["jpl_train_definition"]["sha256"]),
    )
    validate_candidate_definition_counts(definitions, protocol)
    anchor_records = read_anchor_manifest(
        Path(frozen["cach4_public_header_manifest"]["path"]),
        expected_count=int(frozen["cach4_public_header_manifest"]["anchor_flightlines"]),
    )
    anchor_flights = {str(record["flight"]) for record in anchor_records}
    anchor_versions = {flight: "v2t1" for flight in anchor_flights}
    candidate_versions: dict[str, str] = {}
    for tile in definitions:
        if tile.label != 0:
            continue
        previous = candidate_versions.setdefault(tile.flight, tile.header_version)
        if previous != tile.header_version:
            raise ValueError(f"Conflicting released versions for {tile.flight}")
    allowed_versions = dict(anchor_versions)
    for flight, version in candidate_versions.items():
        previous = allowed_versions.setdefault(flight, version)
        if previous != version:
            raise ValueError(f"Cross-stage version conflict for {flight}")
    outputs = protocol["outputs"]
    header_root = Path(protocol["network_and_content_boundary"]["detailed_metadata_location"])
    receipt_path = Path(outputs["ignored_cmr_receipt"])
    manifest_path = Path(outputs["ignored_header_manifest"])

    stage_a_receipts, stage_a_headers = acquire_flight_headers(
        session=session,
        flight_versions=anchor_versions,
        allowed_versions=allowed_versions,
        header_root=header_root,
        stage="stage_a_cach4_grid_bridge",
    )
    _save_header_manifest(manifest_path, stage_a_headers)
    write_json(receipt_path, {"schema_version": 1, "queries": stage_a_receipts})
    comparison = protocol["stage_a_cach4_grid_bridge"]["grid_comparison"]
    stage_a, details = evaluate_stage_a(
        anchor_records=anchor_records,
        nasa_records=stage_a_headers,
        minimum_resolved=int(
            protocol["stage_a_cach4_grid_bridge"]["minimum_resolved_anchor_flightlines"]
        ),
        minimum_fraction=float(
            protocol["stage_a_cach4_grid_bridge"]["minimum_resolved_anchor_fraction"]
        ),
        relative_pixel_tolerance=1e-6,
        maximum_discrepancy_pixels=float(comparison["maximum_discrepancy_pixels"]),
    )
    write_json(
        header_root / "stage_a_grid_comparisons.json",
        {"schema_version": 1, "comparisons": details},
    )

    stage_b: dict[str, object] | None = None
    all_headers = list(stage_a_headers)
    all_receipts = list(stage_a_receipts)
    if stage_a["pass"]:
        ensure_stage_a_pass(stage_a)
        stage_b_receipts, stage_b_headers = acquire_flight_headers(
            session=session,
            flight_versions=candidate_versions,
            allowed_versions=allowed_versions,
            header_root=header_root,
            stage="stage_b_covid_permian_metadata",
        )
        all_headers.extend(stage_b_headers)
        all_receipts.extend(stage_b_receipts)
        _save_header_manifest(manifest_path, all_headers)
        write_json(receipt_path, {"schema_version": 1, "queries": all_receipts})
        resolved, resolution_counts = resolve_candidate_negatives(
            definitions=definitions, nasa_records=stage_b_headers
        )
        resolved_path = Path(outputs["ignored_resolved_rows"])
        filtered_path = Path(outputs["ignored_filtered_rows"])
        write_jsonl(resolved_path, resolved)
        prior_coordinates, _ = load_prior_negative_coordinates(
            stage_b_report_path=Path(frozen["prior_stage_b_report"]["path"]),
            pair_catalog_path=Path(frozen["prior_stage_b_pairs"]["path"]),
            mask_catalog_path=Path(frozen["prior_mask_catalog"]["path"]),
        )
        mars = read_mars_observations(Path(frozen["safe_mars_manifest"]["path"]))
        all_mars, protected_mars = official_test_locations(mars)
        filtered = filter_rows(
            rows=resolved,
            all_mars_locations=all_mars,
            protected_mars_locations=protected_mars,
            prior_negative_coordinates=prior_coordinates,
            radius_km=GROUP_RADIUS_KM,
        )
        write_jsonl(filtered_path, filtered)
        gates = protocol["stage_b_covid_permian_metadata"]["metadata_gates"]
        stage_b = summarize_stage_b(
            resolved=resolved,
            filtered=filtered,
            resolution_counts=resolution_counts,
            minimum_rows=int(gates["minimum_resolved_train_background_rows"]),
            minimum_components=int(gates["minimum_eligible_25km_connected_components"]),
        )

    output_hashes: dict[str, object] = {}
    for key, value in outputs.items():
        path = Path(value)
        if path.exists() and key not in {"compact_json", "compact_markdown"}:
            output_hashes[key] = {
                "path": path.as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
    comparison_path = header_root / "stage_a_grid_comparisons.json"
    output_hashes["ignored_stage_a_grid_comparisons"] = {
        "path": comparison_path.as_posix(),
        "bytes": comparison_path.stat().st_size,
        "sha256": sha256_file(comparison_path),
    }
    report = _compact_report(
        protocol=protocol,
        input_receipts=input_receipts,
        stage_a=stage_a,
        stage_b=stage_b,
        output_hashes=output_hashes,
    )
    compact_json = Path(outputs["compact_json"])
    compact_markdown = Path(outputs["compact_markdown"])
    write_json(compact_json, report)
    write_markdown(report, compact_markdown)
    return report


def execute_stage_a_cmr_preflight(
    *, protocol_path: Path, session: requests.Session
) -> dict[str, object]:
    """Query only public CMR metadata for the frozen CACH4 anchors."""

    protocol, input_receipts = validate_frozen_inputs(protocol_path)
    frozen = protocol["frozen_inputs"]
    anchors = read_anchor_manifest(
        Path(frozen["cach4_public_header_manifest"]["path"]),
        expected_count=int(
            frozen["cach4_public_header_manifest"]["anchor_flightlines"]
        ),
    )
    anchor_versions = {str(record["flight"]): "v2t1" for record in anchors}
    receipts, granules = preflight_header_granules(
        session=session,
        flight_versions=anchor_versions,
        allowed_versions=anchor_versions,
        stage="stage_a_cach4_cmr_preflight",
    )
    resolved = len(granules)
    total = len(anchor_versions)
    gate = protocol["stage_a_cach4_grid_bridge"]
    minimum = int(gate["minimum_resolved_anchor_flightlines"])
    minimum_fraction = float(gate["minimum_resolved_anchor_fraction"])
    report = {
        "schema_version": 1,
        "scope": "public_cmr_metadata_only_no_header_or_stage_b_access",
        "protocol": input_receipts["protocol"],
        "total_anchor_flightlines": total,
        "resolved_cmr_granules": resolved,
        "unresolved_cmr_granules": total - resolved,
        "resolved_fraction": resolved / total if total else 0.0,
        "metadata_resolution_gate_would_pass": (
            resolved >= minimum and resolved / total >= minimum_fraction
        ),
        "grid_bridge_pass": False,
        "grid_bridge_status": "pending_authenticated_header_content",
        "header_content_accessed": False,
        "stage_b_covid_permian_queried": False,
        "target_catalog_queried": False,
        "granules": granules,
        "queries": receipts,
    }
    write_json(STAGE_A_CMR_PREFLIGHT, report)
    return report


def validation_plan(protocol_path: Path) -> dict[str, object]:
    protocol, receipts = validate_frozen_inputs(protocol_path)
    frozen = protocol["frozen_inputs"]
    definitions = read_candidate_train_definition(
        Path(frozen["jpl_train_definition"]["path"]),
        expected_sha256=str(frozen["jpl_train_definition"]["sha256"]),
    )
    candidate_counts = validate_candidate_definition_counts(definitions, protocol)
    anchors = read_anchor_manifest(
        Path(frozen["cach4_public_header_manifest"]["path"]),
        expected_count=int(frozen["cach4_public_header_manifest"]["anchor_flightlines"]),
    )
    candidates = [tile for tile in definitions if tile.label == 0]
    return {
        "scope": "validation_only_no_network",
        "inputs": receipts,
        "stage_a_explicit_flight_ids": len({str(row["flight"]) for row in anchors}),
        "stage_b_explicit_negative_flight_ids_if_authorized": len(
            {tile.flight for tile in candidates}
        ),
        "stage_b_negative_rows_if_authorized": len(candidates),
        "campaigns": dict(sorted(Counter(tile.campaign for tile in candidates).items())),
        "released_definition_counts": candidate_counts,
        "network_executed": False,
        "target_catalog_accessed": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=EXPECTED_PROTOCOL)
    network = parser.add_mutually_exclusive_group()
    network.add_argument(
        "--execute-authenticated-network",
        action="store_true",
        help="Explicitly query CMR and authenticated ORNL header URLs.",
    )
    network.add_argument(
        "--execute-stage-a-cmr-preflight",
        action="store_true",
        help="Query public CMR metadata only for frozen CACH4 anchor IDs.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.execute_authenticated_network:
        report = execute_authenticated_bridge(
            protocol_path=args.protocol,
            session=requests.Session(),
        )
    elif args.execute_stage_a_cmr_preflight:
        report = execute_stage_a_cmr_preflight(
            protocol_path=args.protocol,
            session=requests.Session(),
        )
    else:
        report = validation_plan(args.protocol)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
