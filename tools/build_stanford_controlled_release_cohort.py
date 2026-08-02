#!/usr/bin/env python3
"""Build a provenance-first Sentinel-2/Landsat controlled-release cohort."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
from collections import Counter
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Any

import requests


SOURCE_URL = (
    "https://raw.githubusercontent.com/sahar-elabbadi/"
    "SU-Controlled-Releases-2022/publish/"
    "Satellite_overpasses_with_release_rates_20230404.csv"
)
ZENODO_RECORD = "https://doi.org/10.5281/zenodo.10149991"
PAPER_URL = "https://doi.org/10.5194/amt-17-765-2024"
SITE_LON = -111.7857730
SITE_LAT = 32.8218205
EARTH_SEARCH = "https://earth-search.aws.element84.com/v1/search"
LANDSAT_SEARCH = "https://landsatlook.usgs.gov/stac-server/search"
DEFAULT_SOURCE = Path(".research/stanford_controlled_release_2022/overpasses.csv")
DEFAULT_MANIFEST = Path(
    ".research/stanford_controlled_release_2022/cohort_manifest.jsonl"
)
DEFAULT_JSON = Path("reports/acquisition/stanford_controlled_release_cohort.json")
DEFAULT_MARKDOWN = Path("reports/acquisition/STANFORD_CONTROLLED_RELEASE_COHORT.md")
DEFAULT_MARS_METADATA = Path(
    "EarthRemoteSensingRapidResponse/Data Collection/s2_emit_pairs/publication-v1/"
    "external/MARS-S2L-paper-source/validated_images_all_20251129.csv"
)
S2_ASSETS = {
    "B02": "blue",
    "B03": "green",
    "B04": "red",
    "B08": "nir",
    "B11": "swir16",
    "B12": "swir22",
}
LANDSAT_ASSETS = {
    "B2": "blue",
    "B3": "green",
    "B4": "red",
    "B5": "nir08",
    "B6": "swir16",
    "B7": "swir22",
    "QA_PIXEL": "qa_pixel",
}


def repo_root() -> Path:
    return Path(
        subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip()
    ).resolve()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_datetime(row: dict[str, str]) -> datetime:
    value = f"{row['Date']}T{row['Timestamp (UTC)']}"
    return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)


def methane_rate(row: dict[str, str]) -> float:
    value = row.get("ch4_kgh_mean", "").strip()
    if value:
        return float(value)
    gas = row.get("gas_kgh_mean", "").strip()
    if gas and float(gas) == 0.0:
        return 0.0
    raise ValueError(f"Missing methane rate for nonzero overpass: {row}")


def truth_stratum(rate_kgh: float) -> str:
    """Return a visibility-aware label without calling sub-threshold gas a plume."""
    if rate_kgh <= 10.0:
        return "primary_negative"
    if rate_kgh >= 1000.0:
        return "primary_positive"
    return "subthreshold_challenge"


def post_json(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    response = requests.post(
        url,
        json=payload,
        timeout=60,
        headers={"Accept": "application/json", "User-Agent": "ERSRR-research/1.0"},
    )
    response.raise_for_status()
    result = response.json()
    if not isinstance(result, dict):
        raise ValueError(f"Expected JSON object from {url}")
    return result


def compact_assets(item: dict[str, Any], mapping: dict[str, str]) -> dict[str, str]:
    assets = item.get("assets", {})
    missing = sorted(set(mapping.values()) - set(assets))
    if missing:
        raise ValueError(f"STAC item {item.get('id')} lacks assets {missing}")
    return {band: str(assets[name]["href"]) for band, name in mapping.items()}


def resolve_item(
    sensor: str,
    observed: datetime,
    post: Any = post_json,
) -> dict[str, Any]:
    if sensor == "Sentinel-2":
        url = EARTH_SEARCH
        collection = "sentinel-2-l1c"
        mapping = S2_ASSETS
        allowed = lambda item_id: item_id.startswith(("S2A_", "S2B_", "S2C_"))
    elif sensor == "Landsat":
        url = LANDSAT_SEARCH
        collection = "landsat-c2l1"
        mapping = LANDSAT_ASSETS
        allowed = lambda item_id: item_id.startswith(("LC08_", "LC09_"))
    else:
        raise ValueError(f"Unsupported sensor: {sensor}")

    start = observed.strftime("%Y-%m-%dT00:00:00Z")
    end = observed.strftime("%Y-%m-%dT23:59:59Z")
    result = post(
        url,
        {
            "collections": [collection],
            "bbox": [SITE_LON - 0.001, SITE_LAT - 0.001, SITE_LON + 0.001, SITE_LAT + 0.001],
            "datetime": f"{start}/{end}",
            "limit": 100,
        },
    )
    candidates = []
    for item in result.get("features", []):
        item_id = str(item.get("id", ""))
        if not allowed(item_id):
            continue
        acquired = datetime.fromisoformat(
            str(item["properties"]["datetime"]).replace("Z", "+00:00")
        )
        candidates.append((abs((acquired - observed).total_seconds()), acquired, item))
    if not candidates:
        return {"status": "unresolved", "reason": "no_exact_sensor_item_on_date"}
    delta, acquired, item = min(candidates, key=lambda value: value[0])
    if delta > 180.0:
        return {
            "status": "unresolved",
            "reason": "nearest_item_exceeds_180_seconds",
            "nearest_delta_seconds": delta,
            "nearest_id": item.get("id"),
        }
    return {
        "status": "resolved",
        "id": str(item["id"]),
        "datetime": acquired.isoformat(),
        "scheduled_delta_seconds": delta,
        "cloud_cover": item.get("properties", {}).get("eo:cloud_cover"),
        "assets": compact_assets(item, mapping),
    }


def parse_rows(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source in csv.DictReader(StringIO(text)):
        sensor = source.get("Satellite", "")
        if sensor not in {"Sentinel-2", "LandSat"}:
            continue
        observed = parse_datetime(source)
        rate = methane_rate(source)
        rows.append(
            {
                "sensor": "Landsat" if sensor == "LandSat" else sensor,
                "observed_at_utc": observed.isoformat(),
                "metered_ch4_kgh": rate,
                "truth_stratum": truth_stratum(rate),
            }
        )
    rows.sort(key=lambda row: (row["observed_at_utc"], row["sensor"]))
    return rows


def haversine_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    radius = 6371.0088
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    value = (
        math.sin(dphi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    )
    return 2.0 * radius * math.asin(math.sqrt(value))


def mars_overlap(path: Path, product_ids: set[str]) -> dict[str, Any]:
    nearest = math.inf
    exact_products = 0
    rows = 0
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        for row in csv.DictReader(source):
            rows += 1
            tile = row.get("tile", "")
            if tile in product_ids:
                exact_products += 1
            try:
                distance = haversine_km(
                    SITE_LON,
                    SITE_LAT,
                    float(row["lon"]),
                    float(row["lat"]),
                )
            except (KeyError, TypeError, ValueError):
                continue
            nearest = min(nearest, distance)
    return {
        "rows_checked": rows,
        "exact_target_product_matches": exact_products,
        "nearest_location_km": None if math.isinf(nearest) else nearest,
        "site_disjoint_at_25km": bool(nearest > 25.0),
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(records),
        "sensors": dict(sorted(Counter(row["sensor"] for row in records).items())),
        "truth_strata": dict(
            sorted(Counter(row["truth_stratum"] for row in records).items())
        ),
        "resolution": dict(
            sorted(Counter(row["target"]["status"] for row in records).items())
        ),
        "sensor_truth": dict(
            sorted(
                Counter(
                    f"{row['sensor']}:{row['truth_stratum']}" for row in records
                ).items()
            )
        ),
    }


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    summary = report["summary"]
    lines = [
        "# Stanford 2022 controlled-release cohort audit",
        "",
        f"Generated: {report['generated_at_utc']}.",
        "",
        "| Contract | Count |",
        "|---|---:|",
        f"| Exact S2/Landsat overpasses | {summary['rows']} |",
        f"| Primary positives (at least 1,000 kg CH4/h) | {summary['truth_strata'].get('primary_positive', 0)} |",
        f"| Primary negatives (at most 10 kg CH4/h) | {summary['truth_strata'].get('primary_negative', 0)} |",
        f"| Sub-threshold challenge scenes | {summary['truth_strata'].get('subthreshold_challenge', 0)} |",
        f"| Resolved exact products | {summary['resolution'].get('resolved', 0)} |",
        "",
        "The fixed-location single-blind campaign supplies genuine metered zero-release negatives and high-rate positives. It is a valuable external operating-point stress test, but all observations belong to one physical site. It therefore cannot provide an independent site-block bootstrap claim or replace the official MARS-S2L benchmark.",
        "",
        "The three intermediate-rate observations are reported separately and never silently relabeled. The 4.95 kg/h Sentinel-2 event is a primary negative because the paper explicitly treats it as more than two orders of magnitude below Sentinel-2 detectability.",
        "",
        f"MARS exact-product overlap: {report['mars_overlap']['exact_target_product_matches']}; nearest MARS location: {report['mars_overlap']['nearest_location_km']:.2f} km.",
        "",
        "Bulk source data and future image crops remain under `.research/` and are excluded from Git.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=DEFAULT_SOURCE.as_posix())
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST.as_posix())
    parser.add_argument("--mars-metadata", default=DEFAULT_MARS_METADATA.as_posix())
    parser.add_argument("--output-json", default=DEFAULT_JSON.as_posix())
    parser.add_argument("--output-markdown", default=DEFAULT_MARKDOWN.as_posix())
    args = parser.parse_args()

    root = repo_root()
    source_path = root / args.source
    source_path.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(
        SOURCE_URL, timeout=60, headers={"User-Agent": "ERSRR-research/1.0"}
    )
    response.raise_for_status()
    source_path.write_bytes(response.content)
    rows = parse_rows(response.text)
    if len(rows) != 20:
        raise ValueError(f"Expected 20 S2/Landsat overpasses, found {len(rows)}")

    records = []
    for row in rows:
        observed = datetime.fromisoformat(row["observed_at_utc"])
        records.append({**row, "target": resolve_item(row["sensor"], observed)})

    manifest_path = root / args.manifest
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in records),
        encoding="utf-8",
    )
    product_ids = {
        row["target"]["id"]
        for row in records
        if row["target"]["status"] == "resolved"
    }
    mars_path = root / args.mars_metadata
    report = {
        "schema_version": 1,
        "status": "metadata audited; no model outcome accessed",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "paper": PAPER_URL,
            "zenodo": ZENODO_RECORD,
            "table_url": SOURCE_URL,
            "table_path": args.source,
            "table_sha256": sha256(source_path),
            "license": "CC-BY-4.0",
            "site": {"longitude": SITE_LON, "latitude": SITE_LAT},
        },
        "label_contract": {
            "primary_negative": "metered CH4 <= 10 kg/h",
            "primary_positive": "metered CH4 >= 1000 kg/h",
            "subthreshold_challenge": "10 < metered CH4 < 1000 kg/h",
            "rationale": (
                "Use only metered absence/negligible release and rates at or above the paper's "
                "approximately 1 t/h best-case S2/Landsat detectability boundary for primary metrics."
            ),
        },
        "summary": summarize(records),
        "mars_overlap": mars_overlap(mars_path, product_ids),
        "manifest": {
            "path": args.manifest,
            "sha256": sha256(manifest_path),
            "tracked": False,
        },
        "claim_boundary": (
            "One-site external stress test only; no site-bootstrap superiority claim and no "
            "replacement for the official MARS-S2L full/test-only views."
        ),
        "provenance": {
            "script": Path(__file__).resolve().relative_to(root).as_posix(),
            "script_sha256": sha256(Path(__file__).resolve()),
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True
            ).strip(),
        },
    }
    output_json = root / args.output_json
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_markdown(root / args.output_markdown, report)
    print(json.dumps({"ok": True, **report["summary"], "mars_overlap": report["mars_overlap"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
