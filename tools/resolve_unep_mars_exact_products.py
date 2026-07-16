#!/usr/bin/env python3
"""Resolve exact public L1 assets for nonsealed UNEP MARS samples."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import requests


DEFAULT_INPUT = Path(".research/unep_mars_post2024/eligible_manifest.jsonl")
DEFAULT_OUTPUT = Path(".research/unep_mars_post2024/nonsealed_exact_assets.jsonl")
DEFAULT_JSON = Path("reports/acquisition/unep_mars_post2024_asset_resolution.json")
DEFAULT_MARKDOWN = Path("reports/acquisition/UNEP_MARS_POST2024_ASSET_RESOLUTION.md")

EARTH_SEARCH = "https://earth-search.aws.element84.com/v1/search"
LANDSAT_ITEM = (
    "https://landsatlook.usgs.gov/stac-server/collections/landsat-c2l1/items/{product}"
)
S2_COLLECTION = "sentinel-2-l1c"
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
S2_PRODUCT = re.compile(r"^S2[ABC]_MSIL1C_(\d{8})T\d{6}_.*_T\d{2}[A-Z]{3}_")


def repo_root() -> Path:
    value = subprocess.check_output(
        ["git", "rev-parse", "--show-toplevel"], text=True
    ).strip()
    if not value:
        raise RuntimeError("Could not resolve repository root")
    return Path(value).resolve()


def safe_path(root: Path, value: str) -> Path:
    path = (root / value).resolve()
    if path != root and root not in path.parents:
        raise ValueError(f"Path must remain beneath repository root: {value}")
    return path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_json(url: str, *, params: dict[str, Any] | None = None) -> dict[str, Any]:
    error: Exception | None = None
    for attempt in range(3):
        try:
            response = requests.get(
                url,
                params=params,
                headers={"Accept": "application/json", "User-Agent": "ERSRR-research/1.0"},
                timeout=60,
            )
            if response.status_code == 404:
                return {"_http_status": 404}
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object from {url}")
            return payload
        except (requests.RequestException, ValueError) as exc:
            error = exc
            if attempt < 2:
                time.sleep(float(2**attempt))
    assert error is not None
    raise error


def post_json(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    error: Exception | None = None
    for attempt in range(3):
        try:
            response = requests.post(
                url,
                json=payload,
                headers={"Accept": "application/json", "User-Agent": "ERSRR-research/1.0"},
                timeout=60,
            )
            response.raise_for_status()
            result = response.json()
            if not isinstance(result, dict):
                raise ValueError(f"Expected JSON object from {url}")
            return result
        except (requests.RequestException, ValueError) as exc:
            error = exc
            if attempt < 2:
                time.sleep(float(2**attempt))
    assert error is not None
    raise error


def contains(item: dict[str, Any], center: list[float]) -> bool:
    bbox = item.get("bbox")
    return bool(
        bbox
        and len(bbox) >= 4
        and float(bbox[0]) <= center[0] <= float(bbox[2])
        and float(bbox[1]) <= center[1] <= float(bbox[3])
    )


def s2_date(product: str) -> datetime:
    match = S2_PRODUCT.match(product)
    if match is None:
        raise ValueError(f"Unexpected Sentinel-2 L1C product: {product}")
    return datetime.strptime(match.group(1), "%Y%m%d").replace(tzinfo=timezone.utc)


def official_s2_href(href: str) -> str:
    prefixes = (
        "s3://sentinel-s2-l1c/",
        "s3://sentinel-s2-l2a/",
        "https://sentinel-s2-l1c.s3.amazonaws.com/",
        "https://sentinel-s2-l2a.s3.amazonaws.com/",
        "https://roda.sentinel-hub.com/sentinel-s2-l1c/",
    )
    for prefix in prefixes:
        if href.startswith(prefix):
            return "https://sentinel-s2-l1c.s3.amazonaws.com/" + href[len(prefix) :]
    raise ValueError(f"Unsupported Sentinel-2 asset URL: {href}")


def compact_assets(
    item: dict[str, Any], roles: dict[str, str], *, s2: bool
) -> dict[str, str]:
    assets = item.get("assets", {})
    missing = sorted(set(roles.values()) - set(assets))
    if missing:
        raise ValueError(f"STAC item lacks required assets: {missing}")
    result = {band: str(assets[role]["href"]) for band, role in roles.items()}
    if s2:
        result = {band: official_s2_href(href) for band, href in result.items()}
    return result


def resolve_s2(product: str, center: list[float]) -> dict[str, Any]:
    observed = s2_date(product)
    response = post_json(
        EARTH_SEARCH,
        {
            "collections": [S2_COLLECTION],
            "bbox": [center[0] - 0.001, center[1] - 0.001, center[0] + 0.001, center[1] + 0.001],
            "datetime": (
                f"{observed.isoformat().replace('+00:00', 'Z')}/"
                f"{(observed + timedelta(days=1)).isoformat().replace('+00:00', 'Z')}"
            ),
            "limit": 100,
        },
    )
    expected = product + ".SAFE"
    matches = [
        item
        for item in response.get("features", [])
        if item.get("properties", {}).get("s2:product_uri") == expected
        and contains(item, center)
    ]
    if len(matches) != 1:
        return {
            "status": "unavailable_exact_product",
            "product": product,
            "matches": len(matches),
        }
    item = matches[0]
    return {
        "status": "resolved",
        "product": product,
        "provider": "Element 84 Earth Search / Sentinel-2 L1C public bucket",
        "stac_item_id": item["id"],
        "stac_item": next(
            (link["href"] for link in item.get("links", []) if link.get("rel") == "self"),
            None,
        ),
        "assets": compact_assets(item, S2_ASSETS, s2=True),
    }


def resolve_landsat(product: str, center: list[float]) -> dict[str, Any]:
    item = get_json(LANDSAT_ITEM.format(product=product))
    if item.get("_http_status") == 404:
        return {"status": "unavailable_exact_product", "product": product}
    observed_product = str(
        item.get("properties", {}).get("landsat:product_id") or item.get("id") or ""
    )
    if observed_product != product or not contains(item, center):
        return {
            "status": "identity_or_coverage_mismatch",
            "product": product,
            "observed_product": observed_product,
        }
    return {
        "status": "resolved",
        "product": product,
        "provider": "USGS LandsatLook Collection-2 Level-1 STAC",
        "stac_item_id": item["id"],
        "stac_item": next(
            (link["href"] for link in item.get("links", []) if link.get("rel") == "self"),
            None,
        ),
        "assets": compact_assets(item, LANDSAT_ASSETS, s2=False),
    }


def resolve_one(sample: dict[str, Any]) -> dict[str, Any]:
    resolver = resolve_s2 if sample["sensor_family"] == "Sentinel-2" else resolve_landsat
    center = list(map(float, sample["source_center"]))
    try:
        target = resolver(sample["target_product"], center)
        reference = resolver(sample["background_product"], center)
        status = (
            "resolved"
            if target["status"] == "resolved" and reference["status"] == "resolved"
            else "unresolved"
        )
        return {
            "sample_id": sample["sample_id"],
            "research_role": sample["research_role"],
            "sensor_family": sample["sensor_family"],
            "group_id": sample["group_id"],
            "source_center": sample["source_center"],
            "status": status,
            "target": target,
            "reference": reference,
        }
    except Exception as exc:
        return {
            "sample_id": sample["sample_id"],
            "research_role": sample["research_role"],
            "sensor_family": sample["sensor_family"],
            "group_id": sample["group_id"],
            "status": "query_error",
            "error_type": type(exc).__name__,
            "error": str(exc)[:500],
        }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    status = Counter(record["status"] for record in records)
    by_sensor: dict[str, dict[str, int]] = {}
    by_role: dict[str, dict[str, int]] = {}
    for field, output in (("sensor_family", by_sensor), ("research_role", by_role)):
        for value in sorted({record[field] for record in records}):
            selected = [record for record in records if record[field] == value]
            output[value] = dict(sorted(Counter(record["status"] for record in selected).items()))
            output[value]["total"] = len(selected)
    unresolved_products = Counter()
    for record in records:
        for side in ("target", "reference"):
            value = record.get(side, {})
            if value.get("status") != "resolved":
                unresolved_products[f"{record['sensor_family']}:{side}:{value.get('status', record['status'])}"] += 1
    return {
        "samples": len(records),
        "status": dict(sorted(status.items())),
        "by_sensor": by_sensor,
        "by_role": by_role,
        "unresolved_product_sides": dict(sorted(unresolved_products.items())),
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    summary = report["summary"]
    lines = [
        "# UNEP MARS post-2024 exact-product resolution",
        "",
        f"Generated: {report['generated_at_utc']}.",
        "",
        "Only auxiliary-training and development samples were resolved. Sealed-external rows were excluded.",
        "",
        "## Result",
        "",
        f"- Nonsealed samples: **{summary['samples']:,}**.",
        f"- Fully resolved exact target/reference pairs: **{summary['status'].get('resolved', 0):,}**.",
        f"- Unresolved pairs: **{summary['status'].get('unresolved', 0):,}**.",
        f"- Query errors: **{summary['status'].get('query_error', 0):,}**.",
        "",
        "## By sensor",
        "",
    ]
    for sensor, values in summary["by_sensor"].items():
        lines.append(
            f"- {sensor}: {values.get('resolved', 0):,} resolved / {values['total']:,} total."
        )
    lines.extend(["", "## Integrity", ""])
    lines.extend(
        [
            "- Sentinel-2 identities must match the exact UNEP product URI and cover the source center.",
            "- Landsat identities must match the exact USGS Collection-2 Level-1 product ID and cover the source center.",
            "- Missing real-time Landsat products are reported unavailable; later tier products are not substituted.",
            "- Resolved spectral assets are the six released MARS-S2L bands; Landsat also retains QA_PIXEL.",
            f"- Ignored row-level resolver SHA-256: `{report['output']['sha256']}`.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_INPUT.as_posix())
    parser.add_argument("--output", default=DEFAULT_OUTPUT.as_posix())
    parser.add_argument("--output-json", default=DEFAULT_JSON.as_posix())
    parser.add_argument("--output-markdown", default=DEFAULT_MARKDOWN.as_posix())
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 32:
        parser.error("--workers must be between 1 and 32")

    root = repo_root()
    input_path = safe_path(root, args.input)
    output_path = safe_path(root, args.output)
    output_json = safe_path(root, args.output_json)
    output_markdown = safe_path(root, args.output_markdown)
    samples = []
    with input_path.open("r", encoding="utf-8") as source:
        for line in source:
            sample = json.loads(line)
            if sample["research_role"] != "sealed_external":
                samples.append(sample)
    if not samples or any(sample["research_role"] == "sealed_external" for sample in samples):
        raise ValueError("Resolver input must contain nonsealed samples only")

    records: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(resolve_one, sample): sample["sample_id"] for sample in samples}
        for completed, future in enumerate(as_completed(futures), start=1):
            records.append(future.result())
            if completed % 25 == 0 or completed == len(futures):
                print(f"resolved {completed}/{len(futures)}", flush=True)
    records.sort(key=lambda record: record["sample_id"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as target:
        for record in records:
            target.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
    report = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "nonsealed exact-product assets resolved; imagery not downloaded",
        "source_manifest": {"path": args.input, "sha256": sha256(input_path)},
        "summary": summarize(records),
        "output": {
            "path": args.output,
            "bytes": output_path.stat().st_size,
            "sha256": sha256(output_path),
        },
        "providers": {
            "sentinel2": "Element 84 Earth Search sentinel-2-l1c backed by public Sentinel-2 L1C assets",
            "landsat": "USGS LandsatLook landsat-c2l1 STAC",
        },
        "substitution_policy": "exact product only; no RT-to-tier or L1C-to-L2A substitution",
        "sealed_external_accessed": False,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(report, output_markdown)
    print(json.dumps(report["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
