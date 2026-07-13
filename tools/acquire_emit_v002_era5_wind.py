#!/usr/bin/env python3
"""Acquire, hash, and extract ERA5-Land wind for frozen external candidates."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from acquire_v002_pilot import parse_datetime

DEFAULT_INPUT = Path("reports/acquisition/emit_v002_era5_wind_requests.json")
DEFAULT_OUTPUT = Path(
    "EarthRemoteSensingRapidResponse/Data Collection/s2_emit_pairs/"
    "emit-v002-external-l1c-2026-07/era5_land"
)
DEFAULT_JSON = Path("reports/acquisition/emit_v002_era5_wind_acquisition.json")
DEFAULT_MARKDOWN = Path("reports/acquisition/EMIT_V002_ERA5_WIND_ACQUISITION.md")
TIME_ALIASES = ("valid_time", "validity_time", "datetime", "time", "date")
U_ALIASES = ("u10", "10m_u_component_of_wind", "u_component_of_wind_10m")
V_ALIASES = ("v10", "10m_v_component_of_wind", "v_component_of_wind_10m")
LAT_ALIASES = ("latitude", "lat")
LON_ALIASES = ("longitude", "lon")


def repo_root() -> Path:
    value = subprocess.check_output(
        ["git", "rev-parse", "--show-toplevel"], text=True
    ).strip()
    return Path(value).resolve()


def cds_credentials_available(
    environment: dict[str, str] | None = None, home: Path | None = None
) -> bool:
    """Check the credential locations supported by pinned cdsapi without reading secrets."""
    values = os.environ if environment is None else environment
    if values.get("CDSAPI_URL") and values.get("CDSAPI_KEY"):
        return True
    default = (Path.home() if home is None else home) / ".cdsapirc"
    path = Path(values.get("CDSAPI_RC", str(default))).expanduser()
    return path.is_file() and path.stat().st_size > 0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_path(root: Path, value: str) -> Path:
    result = (root / value).resolve()
    if result != root and root not in result.parents:
        raise ValueError("Path must resolve beneath the repository root")
    return result


def normalized(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def column(fieldnames: list[str], aliases: tuple[str, ...]) -> str:
    by_normalized = {normalized(item): item for item in fieldnames}
    for alias in aliases:
        if normalized(alias) in by_normalized:
            return by_normalized[normalized(alias)]
    raise ValueError(f"Missing required CSV column; expected one of {aliases}, got {fieldnames}")


def parse_wind_csv(path: Path, validity_times: dict[str, str]) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            raise ValueError(f"ERA5 CSV has no header: {path}")
        fields = list(reader.fieldnames)
        time_key = column(fields, TIME_ALIASES)
        u_key = column(fields, U_ALIASES)
        v_key = column(fields, V_ALIASES)
        lat_key = column(fields, LAT_ALIASES)
        lon_key = column(fields, LON_ALIASES)
        rows = list(reader)
    parsed: dict[str, dict[str, Any]] = {}
    for row in rows:
        timestamp = parse_datetime(row[time_key]).isoformat().replace("+00:00", "Z")
        parsed[timestamp] = {
            "validity_time": timestamp,
            "wind_u_m_s": float(row[u_key]),
            "wind_v_m_s": float(row[v_key]),
            "grid_latitude": float(row[lat_key]),
            "grid_longitude": float(row[lon_key]),
        }
    selected: dict[str, Any] = {}
    for role in ("previous", "nearest", "following"):
        expected = validity_times[role]
        if expected not in parsed:
            raise ValueError(f"ERA5 CSV lacks {role} validity time {expected}: {path}")
        selected[role] = parsed[expected]
    return {"rows": len(rows), "selected": selected}


def normalize_csv_download(path: Path) -> None:
    """Replace a CDS ZIP response in place with its single CSV member."""
    if not zipfile.is_zipfile(path):
        return
    extracted = path.with_suffix(path.suffix + ".extract.tmp")
    if extracted.exists():
        extracted.unlink()
    with zipfile.ZipFile(path) as archive:
        members = sorted(
            (item for item in archive.infolist() if not item.is_dir() and item.filename.lower().endswith(".csv")),
            key=lambda item: item.filename,
        )
        if len(members) != 1:
            raise ValueError(
                f"CDS ZIP must contain exactly one CSV, found {len(members)}: {path}"
            )
        with archive.open(members[0]) as source, extracted.open("xb") as destination:
            shutil.copyfileobj(source, destination)
    if extracted.stat().st_size == 0:
        extracted.unlink()
        raise ValueError(f"CDS ZIP contains an empty CSV: {path}")
    os.replace(extracted, path)


def verify_record(root: Path, record: dict[str, Any]) -> Path:
    path = safe_path(root, record["path"])
    if not path.is_file() or path.stat().st_size != int(record["bytes"]):
        raise ValueError(f"Missing or size-mismatched ERA5 asset: {path}")
    if sha256(path) != record["sha256"]:
        raise ValueError(f"ERA5 SHA-256 mismatch: {path}")
    return path


def acquire_one(
    client: Any,
    root: Path,
    output_root: Path,
    dataset: str,
    item: dict[str, Any],
    *,
    overwrite: bool,
    verify_only: bool,
) -> dict[str, Any]:
    csv_path = output_root / f"{item['group_id']}.csv"
    manifest_path = output_root / f"{item['group_id']}.manifest.json"
    if manifest_path.is_file() and not overwrite:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        asset = verify_record(root, manifest["asset"])
        parsed = parse_wind_csv(asset, item["hourly_validity_times"])
        if parsed["selected"] != manifest["selected"]:
            raise ValueError(f"Parsed ERA5 values changed for {item['group_id']}")
        return manifest
    if verify_only:
        raise FileNotFoundError(manifest_path)
    output_root.mkdir(parents=True, exist_ok=True)
    temporary = csv_path.with_suffix(".csv.tmp")
    if temporary.exists():
        temporary.unlink()
    client.retrieve(dataset, item["cds_request"], str(temporary))
    if not temporary.is_file() or temporary.stat().st_size == 0:
        raise ValueError(f"CDS produced no CSV for {item['group_id']}")
    normalize_csv_download(temporary)
    os.replace(temporary, csv_path)
    parsed = parse_wind_csv(csv_path, item["hourly_validity_times"])
    manifest = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "group_id": item["group_id"],
        "granule_id": item["granule_id"],
        "sentinel2_target_scene_id": item["sentinel2_target_scene_id"],
        "sentinel2_target_datetime": item["sentinel2_target_datetime"],
        "dataset": dataset,
        "cds_request_sha256": item["cds_request_sha256"],
        "rows": parsed["rows"],
        "selected": parsed["selected"],
        "asset": {
            "path": csv_path.relative_to(root).as_posix(),
            "bytes": csv_path.stat().st_size,
            "sha256": sha256(csv_path),
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def compact_report(source: dict[str, Any], manifests: list[dict[str, Any]]) -> dict[str, Any]:
    records = []
    for manifest in sorted(manifests, key=lambda item: item["group_id"]):
        nearest = manifest["selected"]["nearest"]
        records.append(
            {
                "group_id": manifest["group_id"],
                "granule_id": manifest["granule_id"],
                "sentinel2_target_scene_id": manifest["sentinel2_target_scene_id"],
                "sentinel2_target_datetime": manifest["sentinel2_target_datetime"],
                "validity_time": nearest["validity_time"],
                "wind_u_m_s": nearest["wind_u_m_s"],
                "wind_v_m_s": nearest["wind_v_m_s"],
                "grid_latitude": nearest["grid_latitude"],
                "grid_longitude": nearest["grid_longitude"],
                "previous": manifest["selected"]["previous"],
                "following": manifest["selected"]["following"],
                "raw_csv_bytes": manifest["asset"]["bytes"],
                "raw_csv_sha256": manifest["asset"]["sha256"],
                "cds_request_sha256": manifest["cds_request_sha256"],
            }
        )
    return {
        "contract": source["contract"],
        "summary": {
            "requested": source["summary"]["requests"],
            "acquired": len(records),
            "complete": len(records) == source["summary"]["requests"],
            "total_raw_csv_bytes": sum(item["raw_csv_bytes"] for item in records),
            "hash_verified": len(records),
        },
        "records": records,
    }


def markdown(result: dict[str, Any]) -> str:
    summary = result["summary"]
    return f"""# EMIT V002 external ERA5-Land wind acquisition

## Result

- Acquired: **{summary['acquired']}/{summary['requested']}** frozen requests.
- Complete: **{summary['complete']}**.
- Hash-verified raw CSVs: **{summary['hash_verified']}**.
- Ignored raw bytes: **{summary['total_raw_csv_bytes']:,}**.

The committed report retains the nearest-hour wind and the predeclared adjacent-hour sensitivity values. Raw CDS CSVs and credentials remain outside version control.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_INPUT.as_posix())
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT.as_posix())
    parser.add_argument("--output-json", default=DEFAULT_JSON.as_posix())
    parser.add_argument("--output-markdown", default=DEFAULT_MARKDOWN.as_posix())
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    root = repo_root()
    source = json.loads(safe_path(root, args.input).read_text(encoding="utf-8"))
    items = source["requests"][: args.limit]
    output_root = safe_path(root, args.output_dir)
    try:
        import cdsapi
    except ImportError as exc:
        raise RuntimeError('Install the pinned dependency with pip install "cdsapi>=0.7.7"') from exc
    if not args.verify_only and not cds_credentials_available():
        raise RuntimeError(
            "Copernicus CDS credentials are not configured. Accept the ERA5-Land terms, "
            "then follow https://cds.climate.copernicus.eu/how-to-api and create "
            "$HOME/.cdsapirc outside the repository. Earthdata credentials do not apply."
        )
    client = None if args.verify_only else cdsapi.Client()

    def run(item: dict[str, Any]) -> dict[str, Any]:
        return acquire_one(
            client,
            root,
            output_root,
            source["contract"]["dataset"],
            item,
            overwrite=args.overwrite,
            verify_only=args.verify_only,
        )

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        manifests = list(executor.map(run, items))
    result = compact_report(source, manifests)
    json_path = safe_path(root, args.output_json)
    markdown_path = safe_path(root, args.output_markdown)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(markdown(result), encoding="utf-8")
    print(json.dumps(result["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
