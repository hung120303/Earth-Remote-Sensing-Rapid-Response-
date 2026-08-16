"""Resolve CACH4 train-crop centers from public ENVI headers only.

This metadata-stage tool deliberately reads only ``multicampaign_train.csv``.
It downloads small public ``.hdr`` sidecars, never retrieval imagery, labels, or
the released test definition. Detailed headers and resolved rows remain under
the ignored ``.research`` root; only compact aggregate reports are tracked.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from pyproj import Transformer


PUBLIC_CACH4_ROOT = (
    "https://popo.jpl.nasa.gov/pub/AThorpe/extract_time/output/"
    "FA18_500ppmm_150fetch_10_20_30merge/"
)
TRAIN_DEFINITION_NAME = "multicampaign_train.csv"
EXPECTED_TRAIN_SHA256 = "e24afc507a81969742199b53ad92c7bcda423de65298c13d4ac3c25b2acae1d4"
HEADER_SUFFIX = "_cmf_v2t1_img_filt_det_500_1500.hdr"
MAX_HEADER_BYTES = 64 * 1024
USER_AGENT = "ERSRR-CACH4-header-audit/1.0"
FLIGHT_RE = re.compile(r"^ang(?P<date>\d{8})t(?P<time>\d{6})$")
TILE_RE = re.compile(
    r"^CACH4/(?P<flight>ang\d{8}t\d{6})_(?P<tile_product>c(?:h4)?mf)_v2t1_img_"
    r"tile(?P<width>\d+)x(?P<height>\d+)\+"
    r"(?P<sample_off>\d+)\+(?P<line_off>\d+)\.tif$"
)


@dataclass(frozen=True)
class TileDefinition:
    tilepath: str
    labelpath: str
    label: int
    flight: str
    tile_product: str
    width: int
    height: int
    sample_offset: int
    line_offset: int


@dataclass(frozen=True)
class EnviMapInfo:
    samples: int
    lines: int
    easting: float
    northing: float
    x_size: float
    y_size: float
    zone: int
    hemisphere: str
    rotation_degrees: float

    @property
    def epsg(self) -> int:
        return (32600 if self.hemisphere == "north" else 32700) + self.zone


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def flight_timestamp(flight: str) -> str:
    match = FLIGHT_RE.fullmatch(flight)
    if not match:
        raise ValueError(f"Invalid AVIRIS-NG flight identifier: {flight}")
    parsed = datetime.strptime(
        match.group("date") + match.group("time"), "%Y%m%d%H%M%S"
    ).replace(tzinfo=timezone.utc)
    return parsed.isoformat().replace("+00:00", "Z")


def parse_tile_definition(row: dict[str, str]) -> TileDefinition:
    tilepath = row.get("tilepath", "")
    match = TILE_RE.fullmatch(tilepath)
    if not match:
        raise ValueError(f"Unexpected CACH4 tile path: {tilepath}")
    try:
        label = int(row.get("label", ""))
    except ValueError as error:
        raise ValueError(f"Invalid binary label for {tilepath}") from error
    if label not in {0, 1}:
        raise ValueError(f"Invalid binary label for {tilepath}: {label}")
    return TileDefinition(
        tilepath=tilepath,
        labelpath=row.get("labelpath", ""),
        label=label,
        flight=match.group("flight"),
        tile_product=match.group("tile_product"),
        width=int(match.group("width")),
        height=int(match.group("height")),
        sample_offset=int(match.group("sample_off")),
        line_offset=int(match.group("line_off")),
    )


def read_cach4_train_definitions(path: Path) -> list[TileDefinition]:
    # Check the identity before opening: test-split contents are never parsed.
    if path.name != TRAIN_DEFINITION_NAME:
        raise ValueError(
            f"Only released {TRAIN_DEFINITION_NAME} is permitted; got {path.name}"
        )
    if sha256_file(path) != EXPECTED_TRAIN_SHA256:
        raise ValueError("Released multicampaign_train.csv SHA-256 identity mismatch")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["tilepath", "labelpath", "label"]:
            raise ValueError(f"Unexpected released sampler schema: {reader.fieldnames}")
        return [
            parse_tile_definition(row)
            for row in reader
            if row.get("tilepath", "").startswith("CACH4/")
        ]


def header_url(flight: str) -> str:
    if not FLIGHT_RE.fullmatch(flight):
        raise ValueError(f"Invalid flight identifier: {flight}")
    url = f"{PUBLIC_CACH4_ROOT}{flight}/{flight}{HEADER_SUFFIX}"
    validate_header_url(url)
    return url


def validate_header_url(url: str) -> None:
    parsed = urllib.parse.urlparse(url)
    root = urllib.parse.urlparse(PUBLIC_CACH4_ROOT)
    if parsed.scheme != "https" or parsed.netloc != root.netloc:
        raise ValueError(f"Header download outside the public CACH4 host: {url}")
    if not url.startswith(PUBLIC_CACH4_ROOT):
        raise ValueError(f"Header download outside the public CACH4 root: {url}")
    relative = url.removeprefix(PUBLIC_CACH4_ROOT)
    parts = relative.split("/")
    if len(parts) != 2 or not FLIGHT_RE.fullmatch(parts[0]):
        raise ValueError(f"Unexpected public CACH4 header path: {url}")
    if parts[1] != f"{parts[0]}{HEADER_SUFFIX}":
        raise ValueError(f"Only the exact ENVI .hdr sidecar is permitted: {url}")


def read_small_header(url: str, *, retries: int = 5) -> bytes:
    validate_header_url(url)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                payload = response.read(MAX_HEADER_BYTES + 1)
            if len(payload) > MAX_HEADER_BYTES:
                raise ValueError(f"Refusing oversized ENVI header at {url}")
            if not payload.lstrip().startswith(b"ENVI"):
                raise ValueError(f"Response is not an ENVI header: {url}")
            return payload
        except urllib.error.HTTPError as error:
            if error.code < 500 and error.code != 429:
                raise RuntimeError(f"Failed to acquire {url}: HTTP {error.code}") from error
            if attempt + 1 == retries:
                raise RuntimeError(f"Failed to acquire {url}") from error
            retry_after = error.headers.get("Retry-After")
            delay = float(retry_after) if retry_after else min(30.0, 2**attempt)
            time.sleep(delay)
        except (urllib.error.URLError, TimeoutError) as error:
            if attempt + 1 == retries:
                raise RuntimeError(f"Failed to acquire {url}") from error
            time.sleep(min(30.0, 2**attempt))
    raise AssertionError("unreachable")


def acquire_header(flight: str, header_dir: Path) -> Path:
    output = header_dir / f"{flight}{HEADER_SUFFIX}"
    if output.exists():
        payload = output.read_bytes()
        if len(payload) <= MAX_HEADER_BYTES and payload.lstrip().startswith(b"ENVI"):
            return output
        raise ValueError(f"Invalid cached header: {output}")
    payload = read_small_header(header_url(flight))
    header_dir.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name + ".part")
    partial.write_bytes(payload)
    partial.replace(output)
    return output


def _header_fields(text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    pending = ""
    for raw_line in text.replace("\r", "").split("\n"):
        line = raw_line.strip()
        if not line or line.upper() == "ENVI":
            continue
        pending = f"{pending} {line}".strip() if pending else line
        if pending.count("{") > pending.count("}"):
            continue
        if "=" not in pending:
            raise ValueError(f"Malformed ENVI header statement: {pending}")
        key, value = pending.split("=", 1)
        fields[key.strip().lower()] = value.strip()
        pending = ""
    if pending:
        raise ValueError("Unterminated ENVI header value")
    return fields


def parse_envi_header(text: str) -> EnviMapInfo:
    fields = _header_fields(text)
    try:
        samples = int(fields["samples"])
        lines = int(fields["lines"])
        raw_map = fields["map info"]
    except (KeyError, ValueError) as error:
        raise ValueError("ENVI header lacks valid samples, lines, or map info") from error
    if not (raw_map.startswith("{") and raw_map.endswith("}")):
        raise ValueError("Malformed ENVI map info")
    parts = [part.strip() for part in raw_map[1:-1].split(",")]
    if len(parts) < 10 or parts[0].upper() != "UTM":
        raise ValueError(f"Unsupported ENVI map info: {raw_map}")
    if float(parts[1]) != 1.0 or float(parts[2]) != 1.0:
        raise ValueError("Only the source's (1,1) ENVI reference pixel is supported")
    hemisphere = parts[8].lower()
    if hemisphere not in {"north", "south"}:
        raise ValueError(f"Unsupported UTM hemisphere: {parts[8]}")
    if parts[9].upper().replace("_", "-") not in {"WGS-84", "WGS84"}:
        raise ValueError(f"Unsupported ENVI datum: {parts[9]}")
    options: dict[str, str] = {}
    for part in parts[10:]:
        if "=" in part:
            key, value = part.split("=", 1)
            options[key.strip().lower()] = value.strip()
    if options.get("units", "Meters").lower() != "meters":
        raise ValueError(f"Unsupported ENVI units: {options.get('units')}")
    zone = int(parts[7])
    if not 1 <= zone <= 60:
        raise ValueError(f"Invalid UTM zone: {zone}")
    return EnviMapInfo(
        samples=samples,
        lines=lines,
        easting=float(parts[3]),
        northing=float(parts[4]),
        x_size=float(parts[5]),
        y_size=float(parts[6]),
        zone=zone,
        hemisphere=hemisphere,
        rotation_degrees=float(options.get("rotation", "0")),
    )


def gdal_geotransform(map_info: EnviMapInfo) -> tuple[float, ...]:
    # Exact GDAL ENVI-driver convention: rotation radians = -degrees*pi/180.
    rotation = -map_info.rotation_degrees * math.pi / 180.0
    return (
        map_info.easting,
        math.cos(rotation) * map_info.x_size,
        -math.sin(rotation) * map_info.x_size,
        map_info.northing,
        -math.sin(rotation) * map_info.y_size,
        -math.cos(rotation) * map_info.y_size,
    )


def tile_overhang(tile: TileDefinition, map_info: EnviMapInfo) -> tuple[int, int]:
    """Return source-window overhang; the released sampler pads these edges."""
    return (
        max(0, tile.sample_offset + tile.width - map_info.samples),
        max(0, tile.line_offset + tile.height - map_info.lines),
    )


def validate_tile_center_bounds(tile: TileDefinition, map_info: EnviMapInfo) -> None:
    # Empirically verified for this CMF sampler: +A+B is sample/column then
    # line/row, despite the source array-axis naming used by the sampler.
    center_sample = tile.sample_offset + tile.width / 2.0
    center_line = tile.line_offset + tile.height / 2.0
    if not 0.0 <= center_sample < map_info.samples:
        raise ValueError(
            f"Tile center exceeds ENVI samples for {tile.tilepath}: "
            f"{center_sample} not in [0,{map_info.samples})"
        )
    if not 0.0 <= center_line < map_info.lines:
        raise ValueError(
            f"Tile center exceeds ENVI lines for {tile.tilepath}: "
            f"{center_line} not in [0,{map_info.lines})"
        )


def crop_center_utm(
    tile: TileDefinition, map_info: EnviMapInfo
) -> tuple[float, float]:
    validate_tile_center_bounds(tile, map_info)
    pixel_x = tile.sample_offset + tile.width / 2.0
    pixel_y = tile.line_offset + tile.height / 2.0
    gt = gdal_geotransform(map_info)
    easting = gt[0] + pixel_x * gt[1] + pixel_y * gt[2]
    northing = gt[3] + pixel_x * gt[4] + pixel_y * gt[5]
    return easting, northing


def resolved_negative_row(
    tile: TileDefinition, map_info: EnviMapInfo, *, header_sha256: str
) -> dict[str, object]:
    if tile.label != 0:
        raise ValueError("Only released train background rows can be resolved")
    easting, northing = crop_center_utm(tile, map_info)
    transformer = Transformer.from_crs(map_info.epsg, 4326, always_xy=True)
    longitude, latitude = transformer.transform(easting, northing)
    if not (
        math.isfinite(longitude)
        and math.isfinite(latitude)
        and -180.0 <= longitude <= 180.0
        and -90.0 <= latitude <= 90.0
    ):
        raise ValueError(f"Non-finite WGS84 center for {tile.tilepath}")
    return {
        "sample_id": Path(tile.tilepath).stem,
        "sensor": "AVIRIS-NG",
        "tile": tile.flight,
        "timestamp": flight_timestamp(tile.flight),
        "longitude": longitude,
        "latitude": latitude,
        "label_state": "NO_PLUME",
        "coordinate_resolved": True,
        "eligible_for_target_catalog": False,
        "eligibility_status": "pending_25km_protected_site_and_duplicate_filter",
        "country": "United States",
        "group_id": None,
        "novel_beyond_all_mars_25km": None,
        "published_split": "train",
        "source_campaign": "CACH4",
        "source_tile_product": tile.tile_product,
        "source_tile_path": tile.tilepath,
        "source_label_path": tile.labelpath,
        "crop_width": tile.width,
        "crop_height": tile.height,
        "sample_offset": tile.sample_offset,
        "line_offset": tile.line_offset,
        "source_crs_epsg": map_info.epsg,
        "center_easting": easting,
        "center_northing": northing,
        "header_sha256": header_sha256,
    }


def _license_summary(record_path: Path) -> dict[str, object]:
    record = json.loads(record_path.read_text(encoding="utf-8"))
    license_data = record.get("metadata", {}).get("license")
    license_id = license_data.get("id") if isinstance(license_data, dict) else None
    return {
        "metadata": license_data,
        "explicit_research_use_license": license_id == "cc-by-4.0",
        "record_id": record.get("id"),
        "doi": record.get("doi"),
    }


def _assert_header_cache_boundary(header_dir: Path, flights: set[str]) -> None:
    if not header_dir.exists():
        return
    allowed = {f"{flight}{HEADER_SUFFIX}" for flight in flights}
    unexpected = [path.name for path in header_dir.iterdir() if path.name not in allowed]
    if unexpected:
        raise ValueError(f"Forbidden file in header-only cache: {sorted(unexpected)}")


def _write_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            count += 1
    return count


def run_audit(
    *,
    train_csv: Path,
    record_json: Path,
    ignored_root: Path,
    report_json: Path,
    report_markdown: Path,
    workers: int = 4,
) -> dict[str, object]:
    expected_ignored_root = Path(
        ".research/jpl_operational_ghg_supplement"
    ).resolve()
    if ignored_root.resolve() != expected_ignored_root:
        raise ValueError(
            "Detailed CACH4 metadata must remain under "
            ".research/jpl_operational_ghg_supplement"
        )
    if not 1 <= workers <= 8:
        raise ValueError("workers must be in [1, 8]")
    definitions = read_cach4_train_definitions(train_csv)
    if not definitions:
        raise ValueError("No CACH4 rows found in released train definition")
    flights = sorted({row.flight for row in definitions})
    header_dir = ignored_root / "cach4_headers"
    _assert_header_cache_boundary(header_dir, set(flights))
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        paths = list(executor.map(lambda flight: acquire_header(flight, header_dir), flights))

    headers: dict[str, EnviMapInfo] = {}
    header_records: list[dict[str, object]] = []
    for flight, path in zip(flights, paths, strict=True):
        payload = path.read_bytes()
        parsed = parse_envi_header(payload.decode("utf-8"))
        headers[flight] = parsed
        header_records.append(
            {
                "flight": flight,
                "url": header_url(flight),
                "path": path.as_posix(),
                "bytes": len(payload),
                "sha256": sha256_bytes(payload),
                "samples": parsed.samples,
                "lines": parsed.lines,
                "epsg": parsed.epsg,
                "rotation_degrees": parsed.rotation_degrees,
            }
        )

    # The released sampler pads edge windows, so full nominal tile dimensions
    # may overhang. The frozen metadata gate requires the exact crop center;
    # validate that center against the source dimensions for both classes.
    for tile in definitions:
        validate_tile_center_bounds(tile, headers[tile.flight])
    overhangs = [tile_overhang(tile, headers[tile.flight]) for tile in definitions]
    sample_overhang_count = sum(sample > 0 for sample, _line in overhangs)
    line_overhang_count = sum(line > 0 for _sample, line in overhangs)
    any_overhang_count = sum(sample > 0 or line > 0 for sample, line in overhangs)
    max_sample_overhang = max(sample for sample, _line in overhangs)
    max_line_overhang = max(line for _sample, line in overhangs)
    header_hashes = {item["flight"]: item["sha256"] for item in header_records}
    negative_rows = [row for row in definitions if row.label == 0]
    resolved = [
        resolved_negative_row(
            tile,
            headers[tile.flight],
            header_sha256=str(header_hashes[tile.flight]),
        )
        for tile in negative_rows
    ]
    resolved_path = ignored_root / "cach4_train_negative_resolved_rows.jsonl"
    resolved_count = _write_jsonl(resolved_path, resolved)
    header_manifest_path = ignored_root / "cach4_header_manifest.json"
    write_json(header_manifest_path, {"schema_version": 1, "headers": header_records})

    positive_rows = sum(row.label == 1 for row in definitions)
    negative_flights = {row.flight for row in negative_rows}
    license_summary = _license_summary(record_json)
    coordinate_gate = (
        resolved_count == len(negative_rows)
        and all(row["timestamp"] and row["coordinate_resolved"] for row in resolved)
    )
    report: dict[str, object] = {
        "schema_version": 1,
        "scope": "released_CACH4_train_definitions_and_public_ENVI_headers_only",
        "interpretive_basis": (
            "Public JPL .hdr sidecars are geospatial metadata matching the frozen "
            "protocol's safe field 'crop georeferencing'; no retrieval, GeoTIFF, "
            "label raster, target catalog, or released test content was opened."
        ),
        "sources": {
            "train_definition": train_csv.as_posix(),
            "train_definition_sha256": sha256_file(train_csv),
            "public_header_root": PUBLIC_CACH4_ROOT,
            "zenodo_record": record_json.as_posix(),
        },
        "license": license_summary,
        "counts": {
            "cach4_train_rows": len(definitions),
            "cach4_train_positive_rows": positive_rows,
            "cach4_train_negative_rows": len(negative_rows),
            "cach4_train_flightlines": len(flights),
            "cach4_train_negative_flightlines": len(negative_flights),
            "public_headers_resolved": len(header_records),
            "negative_rows_with_exact_center_and_utc": resolved_count,
            "padded_edge_tiles_sample_axis": sample_overhang_count,
            "padded_edge_tiles_line_axis": line_overhang_count,
            "padded_edge_tiles_any_axis": any_overhang_count,
            "maximum_sample_axis_overhang_pixels": max_sample_overhang,
            "maximum_line_axis_overhang_pixels": max_line_overhang,
        },
        "flight_ids": flights,
        "gates": {
            "explicit_dataset_research_use_license_or_written_permission": (
                license_summary["explicit_research_use_license"]
            ),
            "minimum_metadata_resolved_train_background_tiles": {
                "required": 100,
                "observed": resolved_count,
                "pass": resolved_count >= 100,
            },
            "exact_utc_and_crop_center_available_without_bulk_download": coordinate_gate,
            "minimum_nonprotected_candidate_locations": {
                "status": "PENDING",
                "reason": "25km protected-site and prior-pair filtering not run",
            },
            "metadata_stage_decision": "PENDING_PROTECTED_AND_DUPLICATE_FILTERING",
        },
        "padded_edge_semantics": {
            "source_behavior": (
                "The released imagesampler uses extract_window with a fill value, "
                "so nominal 256-pixel windows can extend beyond the source edge."
            ),
            "validation": (
                "Every crop center satisfies 0 <= A+width/2 < samples and "
                "0 <= B+height/2 < lines; full-window overhang is recorded, not "
                "mistaken for a center-georeferencing failure."
            ),
        },
        "eligibility_boundary": {
            "coordinate_resolved": True,
            "eligible_for_target_catalog": False,
            "novel_beyond_all_mars_25km": None,
            "target_catalog_queried": False,
        },
        "ignored_artifacts": {
            "headers": header_dir.as_posix(),
            "header_manifest": header_manifest_path.as_posix(),
            "resolved_rows_jsonl": resolved_path.as_posix(),
            "resolved_rows_sha256": sha256_file(resolved_path),
        },
    }
    write_json(report_json, report)
    markdown = f"""# JPL CACH4 train-header metadata audit

Status: **PENDING protected-site and duplicate filtering**. This is not a metadata-stage pass and no target Sentinel-2/Landsat catalog was queried.

## Scope and interpretation

Only released `multicampaign_train.csv` definitions and {len(header_records)} tiny public JPL ENVI `.hdr` sidecars were read. The headers are geospatial metadata matching the frozen protocol's permitted crop-georeferencing field; no retrieval raster, GeoTIFF, label raster, target asset, or released test content was opened.

The JPL CMF suffix `+A+B` is interpreted as ENVI sample/column `A`, then line/row `B`. All {len(definitions):,} train crop centers passed `0 <= A + width/2 < samples` and `0 <= B + height/2 < lines`. The released sampler pads edge windows: {sample_overhang_count:,} rows overhang the sample axis, {line_overhang_count:,} overhang the line axis, and {any_overhang_count:,} overhang either axis (maximum {max_sample_overhang} sample pixels and {max_line_overhang} line pixels). This does not move the crop center outside the source image. Crop centers use the GDAL ENVI affine convention with rotation radians `-degrees*pi/180` before conversion from UTM WGS84 to EPSG:4326.

## Compact result

- Dataset license metadata: `{license_summary['metadata']}`
- CACH4 train rows: {len(definitions):,} ({len(negative_rows):,} background; {positive_rows:,} plume)
- Train flightlines listed and header-resolved: {len(flights)}
- Background rows with exact UTC and WGS84 crop center: {resolved_count:,}
- Coordinate-resolution gate: **{'PASS' if coordinate_gate else 'FAIL'}**
- Target-catalog eligibility: **FALSE pending 25 km protected-site and duplicate filtering**

Detailed headers and row-level coordinates remain beneath ignored `{ignored_root.as_posix()}`. The compact JSON lists all train flight IDs and exposes aggregate gates without publishing the detailed location catalog.
"""
    report_markdown.parent.mkdir(parents=True, exist_ok=True)
    report_markdown.write_text(markdown, encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(".research/jpl_operational_ghg_supplement")
    parser.add_argument("--train-csv", type=Path, default=root / TRAIN_DEFINITION_NAME)
    parser.add_argument("--record-json", type=Path, default=root / "zenodo_record.json")
    parser.add_argument("--ignored-root", type=Path, default=root)
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
    parser.add_argument("--workers", type=int, default=4)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = run_audit(
        train_csv=args.train_csv,
        record_json=args.record_json,
        ignored_root=args.ignored_root,
        report_json=args.report_json,
        report_markdown=args.report_markdown,
        workers=args.workers,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
