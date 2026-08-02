#!/usr/bin/env python3
"""Build the public Stanford 2024-2025 S2/Landsat truth cohort."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Any
from urllib.parse import unquote

import requests

from build_stanford_controlled_release_cohort import resolve_item


DATA_DOI = "https://doi.org/10.25740/qh001qt3946"
PAPER_DOI = "https://doi.org/10.21203/rs.3.rs-9110475/v1"
PURL = "https://purl.stanford.edu/qh001qt3946"
EMBED_URL = f"https://embed.stanford.edu/iframe?url={PURL}"
SUMMARY_RE = re.compile(
    r'href="(?P<url>https://stacks\.stanford\.edu/file/qh001qt3946/[^\"]+_summary\.csv)"',
    re.IGNORECASE,
)
TARGET_FOLDER_RE = re.compile(r"/\d{8}_(?P<code>S2[A-C]?|LS[89]?)/", re.IGNORECASE)
DEFAULT_SOURCE = Path(".research/stanford_controlled_release_2024_2025/summaries.jsonl")
DEFAULT_MANIFEST = Path(".research/stanford_controlled_release_2024_2025/cohort_manifest.jsonl")
DEFAULT_JSON = Path("reports/acquisition/stanford_large_controlled_release_cohort.json")
DEFAULT_MARKDOWN = Path("reports/acquisition/STANFORD_LARGE_CONTROLLED_RELEASE_COHORT.md")
DEFAULT_MARS_METADATA = Path(
    "EarthRemoteSensingRapidResponse/Data Collection/s2_emit_pairs/publication-v1/"
    "external/MARS-S2L-paper-source/validated_images_all_20251129.csv"
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


def extract_summary_urls(html: str) -> list[str]:
    urls = sorted({match.group("url") for match in SUMMARY_RE.finditer(html)})
    return [url for url in urls if TARGET_FOLDER_RE.search(unquote(url))]


def parse_summary(text: str, source_url: str) -> dict[str, Any]:
    rows = list(csv.DictReader(StringIO(text)))
    if len(rows) != 1:
        raise ValueError(f"Expected one row in {source_url}, found {len(rows)}")
    row = rows[0]
    folder = TARGET_FOLDER_RE.search(unquote(source_url))
    if folder is None:
        raise ValueError(f"Cannot infer sensor from {source_url}")
    code = folder.group("code").upper()
    sensor = "Sentinel-2" if code.startswith("S2") else "Landsat"
    observed = datetime.fromisoformat(f"{row['date']}T{row['time_UTC']}").replace(
        tzinfo=timezone.utc
    )
    rate = float(row["ch4_kgh_mean"])
    return {
        "release_id": row["release_ID"],
        "sensor": sensor,
        "platform_code": code,
        "observed_at_utc": observed.isoformat(),
        "location": row["location"],
        "latitude": float(row["lat"]),
        "longitude": float(row["lon"]),
        "metered_ch4_kgh": rate,
        "metered_ch4_sigma": float(row["ch4_kgh_sigma"]),
        "truth_stratum": truth_stratum(rate),
        "source_url": source_url,
    }


def get_text(session: requests.Session, url: str) -> str:
    response = session.get(url, timeout=90)
    response.raise_for_status()
    return response.text


def fetch_rows(session: requests.Session, workers: int = 12) -> list[dict[str, Any]]:
    html = get_text(session, EMBED_URL)
    urls = extract_summary_urls(html)
    if not urls:
        raise ValueError("No Sentinel-2/Landsat summary tables found")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        texts = list(pool.map(lambda url: get_text(session, url), urls))
    rows = [parse_summary(text, url) for text, url in zip(texts, urls)]
    rows.sort(key=lambda row: (row["observed_at_utc"], row["platform_code"]))
    return rows


def mars_overlap(path: Path, records: list[dict[str, Any]]) -> dict[str, Any]:
    target_ids = {
        row["target"]["id"]
        for row in records
        if row["target"]["status"] == "resolved"
    }
    exact: Counter[str] = Counter()
    locations: dict[str, dict[str, Any]] = {
        name: {
            "longitude": rows[0]["longitude"],
            "latitude": rows[0]["latitude"],
            "same_site_rows": 0,
            "same_site_split_label": Counter(),
            "location_ids": set(),
        }
        for name in sorted({row["location"] for row in records})
        for rows in [[row for row in records if row["location"] == name]]
    }
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        for row in csv.DictReader(source):
            if row.get("tile", "") in target_ids:
                exact[f"{row.get('split_name', '')}:{row.get('isplume', '')}"] += 1
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
        "exact_target_product_matches": sum(exact.values()),
        "exact_target_product_split_label": dict(sorted(exact.items())),
        "locations": {
            name: {
                **{k: v for k, v in item.items() if k not in {"same_site_split_label", "location_ids"}},
                "same_site_split_label": dict(sorted(item["same_site_split_label"].items())),
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
        "truth_strata": dict(sorted(Counter(row["truth_stratum"] for row in records).items())),
        "sensor_truth": dict(
            sorted(Counter(f"{row['sensor']}:{row['truth_stratum']}" for row in records).items())
        ),
        "resolution": dict(sorted(Counter(row["target"]["status"] for row in records).items())),
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
        f"| S2/Landsat overpasses | {summary['rows']} |",
        f"| Locations | {len(summary['locations'])} |",
        f"| Metered blank controls | {summary['truth_strata'].get('primary_negative', 0)} |",
        f"| Primary positives (at least 1,000 kg CH4/h) | {summary['truth_strata'].get('primary_positive', 0)} |",
        f"| Sub-threshold challenge scenes | {summary['truth_strata'].get('subthreshold_challenge', 0)} |",
        f"| Resolved exact L1 products | {summary['resolution'].get('resolved', 0)} |",
        "",
        "This public CC BY 4.0 cohort is the strongest controlled no-release stress test found so far. Labels come from metered gas flow, not inferred catalog absence or temporal background imagery.",
        "",
        "It contains only two physical sites, so it cannot by itself establish broad geographic generalization. Intermediate nonzero releases remain a separate challenge stratum and cannot be relabeled after scoring.",
        "",
        f"Exact-product matches in upstream MARS metadata: {overlap['exact_target_product_matches']}.",
        "",
        "No model score was accessed while building this audit. Bulk source tables and future image crops remain under `.research/` and outside Git.",
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
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()

    root = repo_root()
    session = requests.Session()
    session.headers.update({"User-Agent": "ERSRR-research/1.0"})
    rows = fetch_rows(session, args.workers)
    if len(rows) != 160:
        raise ValueError(f"Expected 160 S2/Landsat overpasses, found {len(rows)}")

    source_path = root / args.source
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
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
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in records), encoding="utf-8"
    )
    overlap = mars_overlap(root / args.mars_metadata, records)
    report = {
        "schema_version": 1,
        "status": "metadata audited; no ERSRR model outcome accessed",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "paper": PAPER_DOI,
            "data": DATA_DOI,
            "purl": PURL,
            "license": "CC-BY-4.0",
            "source_table_manifest": args.source,
            "source_table_manifest_sha256": sha256(source_path),
        },
        "label_contract": {
            "primary_negative": "metered CH4 == 0 kg/h",
            "primary_positive": "metered CH4 >= 1000 kg/h",
            "subthreshold_challenge": "0 < metered CH4 < 1000 kg/h",
        },
        "summary": summarize(records),
        "mars_overlap": overlap,
        "manifest": {"path": args.manifest, "sha256": sha256(manifest_path), "tracked": False},
        "claim_boundary": (
            "Two-site controlled operating-point stress test; not enough independent sites for a "
            "broad geographic superiority claim and not a replacement for official MARS-S2L views."
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
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(root / args.output_markdown, report)
    print(json.dumps({"ok": True, **report["summary"], "mars_overlap": overlap}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
