#!/usr/bin/env python3
"""Build the public Stanford 2024-2025 S2/Landsat truth cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import xml.etree.ElementTree as ET
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from io import BytesIO
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import requests

from build_stanford_controlled_release_cohort import resolve_item


DATA_DOI = "https://doi.org/10.25740/qh001qt3946"
PAPER_DOI = "https://doi.org/10.21203/rs.3.rs-9110475/v1"
PURL = "https://purl.stanford.edu/qh001qt3946"
WORKBOOK_URL = (
    "https://stacks.stanford.edu/file/qh001qt3946/Code%20scripts/"
    "source_data_Reuland_2026_07162026.xlsx"
)
DEFAULT_WORKBOOK = Path(
    ".research/stanford_controlled_release_2024_2025/"
    "source_data_Reuland_2026_07162026.xlsx"
)
DEFAULT_SOURCE_ROWS = Path(
    ".research/stanford_controlled_release_2024_2025/clean_events.jsonl"
)
DEFAULT_MANIFEST = Path(
    ".research/stanford_controlled_release_2024_2025/cohort_manifest.jsonl"
)
DEFAULT_JSON = Path(
    "reports/acquisition/stanford_large_controlled_release_cohort.json"
)
DEFAULT_MARKDOWN = Path(
    "reports/acquisition/STANFORD_LARGE_CONTROLLED_RELEASE_COHORT.md"
)
DEFAULT_MARS_METADATA = Path(
    "EarthRemoteSensingRapidResponse/Data Collection/s2_emit_pairs/publication-v1/"
    "external/MARS-S2L-paper-source/validated_images_all_20251129.csv"
)
MAIN_SHEET = "xl/worksheets/sheet1.xml"
XML_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
EVENT_FIELDS = (
    "release_ID",
    "date",
    "time_UTC",
    "location",
    "lat",
    "lon",
    "ch4_kgh_mean",
    "ch4_kgh_sigma",
    "SatelliteCode",
    "SatellitePlotName",
    "Acquisition status",
    "QC_ExperimentTeam",
    "Phase",
)


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


def truth_stratum(rate_kgh: float) -> str:
    if rate_kgh == 0.0:
        return "primary_negative"
    if rate_kgh >= 1000.0:
        return "primary_positive"
    return "subthreshold_challenge"


def column_index(reference: str) -> int:
    letters = re.match(r"[A-Z]+", reference)
    if letters is None:
        raise ValueError(f"Invalid cell reference: {reference}")
    result = 0
    for char in letters.group(0):
        result = result * 26 + ord(char) - ord("A") + 1
    return result - 1


def shared_strings(archive: ZipFile) -> list[str]:
    try:
        root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    except KeyError:
        return []
    return [
        "".join(node.text or "" for node in item.iter(f"{XML_NS}t"))
        for item in root.findall(f"{XML_NS}si")
    ]


def cell_value(cell: ET.Element, strings: list[str]) -> Any:
    cell_type = cell.attrib.get("t", "n")
    if cell_type == "inlineStr":
        return "".join(node.text or "" for node in cell.iter(f"{XML_NS}t"))
    value = cell.find(f"{XML_NS}v")
    if value is None or value.text is None:
        return None
    if cell_type == "s":
        return strings[int(value.text)]
    if cell_type == "b":
        return value.text == "1"
    if cell_type in {"str", "e"}:
        return value.text
    number = float(value.text)
    return int(number) if number.is_integer() else number


def parse_workbook(content: bytes) -> list[dict[str, Any]]:
    with ZipFile(BytesIO(content)) as archive:
        strings = shared_strings(archive)
        root = ET.fromstring(archive.read(MAIN_SHEET))
    matrix: list[list[Any]] = []
    for row in root.iter(f"{XML_NS}row"):
        cells: dict[int, Any] = {}
        for cell in row.findall(f"{XML_NS}c"):
            cells[column_index(cell.attrib["r"])] = cell_value(cell, strings)
        width = max(cells, default=-1) + 1
        matrix.append([cells.get(index) for index in range(width)])
    if not matrix:
        raise ValueError("Workbook contains no rows")
    headers = [str(value or "") for value in matrix[0]]
    required = set(EVENT_FIELDS)
    missing = sorted(required - set(headers))
    if missing:
        raise ValueError(f"Workbook lacks required columns: {missing}")
    rows: list[dict[str, Any]] = []
    for values in matrix[1:]:
        padded = values + [None] * (len(headers) - len(values))
        rows.append(dict(zip(headers, padded)))
    return rows


def excel_datetime(date_serial: Any, time_serial: Any) -> datetime:
    return datetime(1899, 12, 30, tzinfo=timezone.utc) + timedelta(
        days=float(date_serial) + float(time_serial)
    )


def select_events(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    events: dict[str, dict[str, Any]] = {}
    for row in rows:
        platform = str(row.get("SatellitePlotName") or "")
        if platform not in {"Sentinel-2", "Landsat"}:
            continue
        release_id = str(row.get("release_ID") or "")
        if not release_id:
            continue
        metadata = {name: row.get(name) for name in EVENT_FIELDS}
        if release_id in events and events[release_id] != metadata:
            raise ValueError(f"Inconsistent event metadata for {release_id}")
        events[release_id] = metadata

    selected = []
    for event in events.values():
        if event["QC_ExperimentTeam"] != "OK":
            continue
        if event["Acquisition status"] == "acquisition failed":
            continue
        observed = excel_datetime(event["date"], event["time_UTC"])
        rate = float(event["ch4_kgh_mean"] or 0.0)
        selected.append(
            {
                "release_id": event["release_ID"],
                "sensor": event["SatellitePlotName"],
                "platform_code": event["SatelliteCode"],
                "observed_at_utc": observed.isoformat(),
                "location": event["location"],
                "latitude": float(event["lat"]),
                "longitude": float(event["lon"]),
                "metered_ch4_kgh": rate,
                "metered_ch4_sigma": float(event["ch4_kgh_sigma"] or 0.0),
                "truth_stratum": truth_stratum(rate),
                "phase": int(event["Phase"]),
            }
        )
    selected.sort(key=lambda row: (row["observed_at_utc"], row["platform_code"]))
    return selected


def mars_overlap(path: Path, records: list[dict[str, Any]]) -> dict[str, Any]:
    import csv

    target_by_id = {
        row["target"]["id"]: row
        for row in records
        if row["target"]["status"] == "resolved"
    }
    exact_matches: list[dict[str, Any]] = []
    locations: dict[str, dict[str, Any]] = {
        name: {
            "longitude": group[0]["longitude"],
            "latitude": group[0]["latitude"],
            "same_site_rows": 0,
            "same_site_split_label": Counter(),
            "location_ids": set(),
        }
        for name in sorted({row["location"] for row in records})
        for group in [[row for row in records if row["location"] == name]]
    }
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        for row in csv.DictReader(source):
            tile = row.get("tile", "")
            if tile in target_by_id:
                target = target_by_id[tile]
                exact_matches.append(
                    {
                        "release_id": target["release_id"],
                        "product_id": tile,
                        "mars_split": row.get("split_name", ""),
                        "mars_label": row.get("isplume", ""),
                        "mars_location_id": row.get("id_location", ""),
                    }
                )
            try:
                lon, lat = float(row["lon"]), float(row["lat"])
            except (KeyError, TypeError, ValueError):
                continue
            for item in locations.values():
                if abs(lon - item["longitude"]) <= 0.002 and abs(lat - item["latitude"]) <= 0.002:
                    item["same_site_rows"] += 1
                    item["same_site_split_label"][
                        f"{row.get('split_name', '')}:{row.get('isplume', '')}"
                    ] += 1
                    item["location_ids"].add(str(row.get("id_location", "")))
    return {
        "exact_target_product_matches": len(exact_matches),
        "exact_matches": exact_matches,
        "locations": {
            name: {
                **{
                    key: value
                    for key, value in item.items()
                    if key not in {"same_site_split_label", "location_ids"}
                },
                "same_site_split_label": dict(
                    sorted(item["same_site_split_label"].items())
                ),
                "location_ids": sorted(item["location_ids"]),
            }
            for name, item in locations.items()
        },
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(records),
        "locations": dict(sorted(Counter(row["location"] for row in records).items())),
        "sensors": dict(sorted(Counter(row["sensor"] for row in records).items())),
        "truth_strata": dict(
            sorted(Counter(row["truth_stratum"] for row in records).items())
        ),
        "sensor_truth": dict(
            sorted(
                Counter(
                    f"{row['sensor']}:{row['truth_stratum']}" for row in records
                ).items()
            )
        ),
        "resolution": dict(
            sorted(Counter(row["target"]["status"] for row in records).items())
        ),
    }


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    summary = report["summary"]
    overlap = report["mars_overlap"]
    lines = [
        "# Stanford 2024-2025 large controlled-release cohort audit",
        "",
        f"Generated: {report['generated_at_utc']}.",
        "",
        "| Contract | Count |",
        "|---|---:|",
        f"| QC-valid S2/Landsat overpasses | {summary['rows']} |",
        f"| Locations | {len(summary['locations'])} |",
        f"| Metered blank controls | {summary['truth_strata'].get('primary_negative', 0)} |",
        f"| Primary positives (at least 1,000 kg CH4/h) | {summary['truth_strata'].get('primary_positive', 0)} |",
        f"| Sub-threshold challenge scenes | {summary['truth_strata'].get('subthreshold_challenge', 0)} |",
        f"| Resolved exact L1 products | {summary['resolution'].get('resolved', 0)} |",
        "",
        "This public CC BY 4.0 cohort is the strongest controlled no-release stress test found so far. Labels come from metered gas flow, not inferred catalog absence or temporal background imagery.",
        "",
        "The authoritative source workbook contains release and non-release events. The per-release summary files omit blank controls and are therefore not used as the cohort universe.",
        "",
        "The QC-valid paper source-workbook cohort contains one physical site (Casa Grande), already represented in excluded upstream MARS metadata. It is a temporally new operating-point stress test, not a source-disjoint geographic benchmark. Intermediate nonzero releases remain a separate challenge stratum.",
        "",
        f"Target-product matches anywhere in upstream MARS metadata: {overlap['exact_target_product_matches']}. Match details are retained in the JSON audit because a shared satellite tile need not mean the same crop location.",
        "",
        "No ERSRR model score was accessed while building this audit. The compact source workbook and future image crops remain under `.research/` and outside Git.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workbook", default=DEFAULT_WORKBOOK.as_posix())
    parser.add_argument("--source-rows", default=DEFAULT_SOURCE_ROWS.as_posix())
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST.as_posix())
    parser.add_argument("--mars-metadata", default=DEFAULT_MARS_METADATA.as_posix())
    parser.add_argument("--output-json", default=DEFAULT_JSON.as_posix())
    parser.add_argument("--output-markdown", default=DEFAULT_MARKDOWN.as_posix())
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()

    root = repo_root()
    response = requests.get(
        WORKBOOK_URL, timeout=90, headers={"User-Agent": "ERSRR-research/1.0"}
    )
    response.raise_for_status()
    workbook_path = root / args.workbook
    workbook_path.parent.mkdir(parents=True, exist_ok=True)
    workbook_path.write_bytes(response.content)
    rows = select_events(parse_workbook(response.content))
    if len(rows) != 262:
        raise ValueError(f"Expected 262 QC-valid S2/Landsat events, found {len(rows)}")

    source_path = root / args.source_rows
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )

    def resolve(row: dict[str, Any]) -> dict[str, Any]:
        return {
            **row,
            "target": resolve_item(
                row["sensor"],
                datetime.fromisoformat(row["observed_at_utc"]),
                longitude=row["longitude"],
                latitude=row["latitude"],
            ),
        }

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        records = list(pool.map(resolve, rows))
    records.sort(key=lambda row: (row["observed_at_utc"], row["platform_code"]))
    manifest_path = root / args.manifest
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in records),
        encoding="utf-8",
    )
    overlap = mars_overlap(root / args.mars_metadata, records)
    report = {
        "schema_version": 2,
        "status": "metadata audited; no ERSRR model outcome accessed",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "paper": PAPER_DOI,
            "data": DATA_DOI,
            "purl": PURL,
            "workbook_url": WORKBOOK_URL,
            "workbook_path": args.workbook,
            "workbook_sha256": sha256(workbook_path),
            "license": "CC-BY-4.0",
            "clean_event_rows": args.source_rows,
            "clean_event_rows_sha256": sha256(source_path),
        },
        "metadata_only_amendment": {
            "reason": (
                "The initial build used per-release summaries, which omit intentional blank "
                "controls. The paper source workbook contains both releases and non-releases."
            ),
            "outcome_accessed": False,
            "contract_change": (
                "Replace the incomplete summary-file universe with the paper's QC-valid source "
                "workbook rows; label thresholds and scoring sequence are unchanged."
            ),
        },
        "label_contract": {
            "primary_negative": "metered CH4 == 0 kg/h",
            "primary_positive": "metered CH4 >= 1000 kg/h",
            "subthreshold_challenge": "0 < metered CH4 < 1000 kg/h",
        },
        "summary": summarize(records),
        "mars_overlap": overlap,
        "manifest": {
            "path": args.manifest,
            "sha256": sha256(manifest_path),
            "tracked": False,
        },
        "claim_boundary": (
            "One-site controlled operating-point stress test; Casa Grande occurs in excluded "
            "upstream MARS metadata, so this is temporally new but not source-disjoint, cannot "
            "support site-bootstrap inference, and is not a replacement for official MARS-S2L views."
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
    print(json.dumps({"ok": True, **report["summary"], "mars_overlap": overlap}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
