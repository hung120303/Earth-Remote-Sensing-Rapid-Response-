"""Query public CDSE metadata for preregistered near-simultaneous S2 L1C scenes."""

from __future__ import annotations

import argparse
import json
import math
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import timedelta, timezone
from pathlib import Path

from tools.audit_mars_hyperspectral_transfer import (
    nearest_distance_km,
    parse_datetime,
    read_hsi_samples,
    read_mars_observations,
    representative_locations,
)


STAC_ENDPOINT = "https://stac.dataspace.copernicus.eu/v1/search"


@dataclass(frozen=True)
class CatalogQuery:
    latitude: float
    longitude: float
    sample_ids: tuple[str, ...]
    timestamp_iso: str


def eligible_queries(
    *, metadata_root: Path, mars_manifest: Path, protocol_path: Path
) -> list[CatalogQuery]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    filters = protocol["stage_b_label_and_catalog_audit"]["filters"]
    samples = read_hsi_samples(metadata_root)
    mars = read_mars_observations(mars_manifest)
    mars_locations = representative_locations(mars)
    protected_names = {
        row.location_name for row in mars if row.split_name.lower().startswith("test")
    }
    protected_coords = [mars_locations[name] for name in protected_names]
    grouped: dict[tuple[float, float, str], list[str]] = {}
    for sample in samples:
        if sample.published_split != "train":
            continue
        if sample.percentage_clear < float(filters["minimum_hyperspectral_percentage_clear"]):
            continue
        coords = (
            (sample.latitude, sample.longitude)
            if sample.latitude is not None and sample.longitude is not None
            else mars_locations.get(sample.location_name)
        )
        if coords is None:
            continue
        latitude, longitude = coords
        if sample.location_name in protected_names:
            continue
        if nearest_distance_km(
            latitude, longitude, protected_coords
        ) <= float(filters["mars_protected_exclusion_radius_km"]):
            continue
        timestamp_iso = sample.timestamp.astimezone(timezone.utc).isoformat()
        key = (round(latitude, 7), round(longitude, 7), timestamp_iso)
        grouped.setdefault(key, []).append(sample.sample_id)
    return [
        CatalogQuery(
            latitude=key[0],
            longitude=key[1],
            timestamp_iso=key[2],
            sample_ids=tuple(sorted(sample_ids)),
        )
        for key, sample_ids in sorted(grouped.items())
    ]


def query_cdse(
    query: CatalogQuery,
    *,
    retries: int = 7,
    request_delay_seconds: float = 0.0,
) -> dict[str, object]:
    center = parse_datetime(query.timestamp_iso).astimezone(timezone.utc)
    start = (center - timedelta(hours=6)).isoformat().replace("+00:00", "Z")
    end = (center + timedelta(hours=6)).isoformat().replace("+00:00", "Z")
    body = {
        "collections": ["sentinel-2-l1c"],
        "intersects": {
            "type": "Point",
            "coordinates": [query.longitude, query.latitude],
        },
        "datetime": f"{start}/{end}",
        "limit": 100,
        "fields": {
            "include": [
                "id",
                "collection",
                "bbox",
                "properties.datetime",
                "properties.eo:cloud_cover",
            ],
            "exclude": ["geometry", "assets", "links"],
        },
    }
    payload = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        STAC_ENDPOINT,
        data=payload,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "User-Agent": "ERSRR-research-audit/1.0",
        },
    )
    for attempt in range(retries):
        if request_delay_seconds:
            time.sleep(request_delay_seconds)
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                result = json.loads(response.read().decode("utf-8"))
            features = []
            for feature in result.get("features", []):
                properties = feature.get("properties", {})
                item_time = parse_datetime(properties["datetime"])
                delta_hours = abs((item_time - center).total_seconds()) / 3600.0
                if delta_hours > 6.0 + 1e-9:
                    continue
                features.append(
                    {
                        "id": feature["id"],
                        "collection": feature.get("collection"),
                        "datetime": properties["datetime"],
                        "cloud_cover": properties.get("eo:cloud_cover"),
                        "bbox": feature.get("bbox"),
                        "offset_hours": delta_hours,
                    }
                )
            features.sort(key=lambda value: (value["offset_hours"], value["id"]))
            return {
                "sample_ids": list(query.sample_ids),
                "hsi_datetime": query.timestamp_iso,
                "latitude": query.latitude,
                "longitude": query.longitude,
                "products": features,
            }
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
            if attempt + 1 == retries:
                raise RuntimeError(
                    f"CDSE query failed after {retries} attempts for {query.sample_ids[0]}"
                ) from error
            retry_after = 0.0
            if isinstance(error, urllib.error.HTTPError) and error.code == 429:
                try:
                    retry_after = float(error.headers.get("Retry-After", "0"))
                except (TypeError, ValueError):
                    retry_after = 0.0
            time.sleep(max(retry_after, min(60.0, 5.0 * (2**attempt))))
    raise AssertionError("unreachable")


def read_records(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def run_queries(
    queries: list[CatalogQuery],
    *,
    workers: int,
    checkpoint_path: Path,
    request_delay_seconds: float,
) -> list[dict[str, object]]:
    records = read_records(checkpoint_path)
    completed = {
        tuple(record["sample_ids"])
        for record in records
    }
    pending = [query for query in queries if query.sample_ids not in completed]
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                query_cdse,
                query,
                request_delay_seconds=request_delay_seconds,
            ): query
            for query in pending
        }
        for future in as_completed(futures):
            record = future.result()
            records.append(record)
            with checkpoint_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
    records.sort(key=lambda value: value["sample_ids"][0])
    return records


def summarize(records: list[dict[str, object]], query_count: int) -> dict[str, object]:
    sample_offsets: dict[str, float] = {}
    unique_products: set[str] = set()
    for record in records:
        products = record["products"]
        if not products:
            continue
        minimum = min(float(product["offset_hours"]) for product in products)
        unique_products.update(str(product["id"]) for product in products)
        for sample_id in record["sample_ids"]:
            sample_offsets[sample_id] = minimum
    return {
        "schema_version": 1,
        "endpoint": STAC_ENDPOINT,
        "collection": "sentinel-2-l1c",
        "scope": "metadata_only_no_target_assets",
        "eligible_query_groups": query_count,
        "eligible_hsi_samples": sum(len(record["sample_ids"]) for record in records),
        "hsi_samples_with_candidate": len(sample_offsets),
        "unique_sentinel_products": len(unique_products),
        "within_15_minutes": sum(offset <= 0.25 for offset in sample_offsets.values()),
        "within_1_hour": sum(offset <= 1.0 for offset in sample_offsets.values()),
        "within_6_hours": sum(offset <= 6.0 for offset in sample_offsets.values()),
        "minimum_offset_hours": min(sample_offsets.values(), default=math.inf),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metadata-root",
        type=Path,
        default=Path(".research/mars_hyperspectral_transfer"),
    )
    parser.add_argument(
        "--mars-manifest",
        type=Path,
        default=Path(
            ".research/source_audit_20260715/mars_s2l_current_validated_images_all.csv"
        ),
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("configs/mars_hyperspectral_transfer_acquisition_protocol.json"),
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=Path(
            ".research/mars_hyperspectral_transfer/cdse_s2_l1c_candidates.jsonl"
        ),
    )
    parser.add_argument(
        "--output-summary",
        type=Path,
        default=Path(
            ".research/mars_hyperspectral_transfer/cdse_s2_l1c_summary.json"
        ),
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--request-delay-seconds", type=float, default=0.75)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            ".research/mars_hyperspectral_transfer/cdse_s2_l1c_candidates.partial.jsonl"
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.workers < 1 or args.workers > 16:
        raise ValueError("workers must be in [1, 16]")
    queries = eligible_queries(
        metadata_root=args.metadata_root,
        mars_manifest=args.mars_manifest,
        protocol_path=args.protocol,
    )
    records = run_queries(
        queries,
        workers=args.workers,
        checkpoint_path=args.checkpoint,
        request_delay_seconds=args.request_delay_seconds,
    )
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.output_jsonl.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    summary = summarize(records, len(queries))
    args.output_summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
