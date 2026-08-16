"""Query USGS LandsatLook for Stage B Landsat Collection 2 L1 candidates."""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timedelta, timezone
from pathlib import Path

from tools.audit_mars_hyperspectral_transfer import parse_datetime
from tools.query_mars_hyperspectral_stage_b_cdse import (
    CatalogGroup,
    STAC_LIMIT,
    build_query_groups,
    jsonl_payload,
    point_in_bbox,
    read_jsonl,
)


LANDSAT_ENDPOINT = "https://landsatlook.usgs.gov/stac-server/search"
LANDSAT_COLLECTION = "landsat-c2l1"
LANDSAT_PREFIXES = ("LC08_", "LC09_")


def query_landsat_group(
    group: CatalogGroup,
    *,
    retries: int = 7,
    request_delay_seconds: float = 0.0,
) -> dict[str, object]:
    center = parse_datetime(group.timestamp_iso).astimezone(timezone.utc)
    start = (center - timedelta(hours=6)).isoformat().replace("+00:00", "Z")
    end = (center + timedelta(hours=6)).isoformat().replace("+00:00", "Z")
    coordinates = [[point.longitude, point.latitude] for point in group.points]
    geometry: dict[str, object] = (
        {"type": "Point", "coordinates": coordinates[0]}
        if len(coordinates) == 1
        else {"type": "MultiPoint", "coordinates": coordinates}
    )
    body = {
        "collections": [LANDSAT_COLLECTION],
        "intersects": geometry,
        "datetime": f"{start}/{end}",
        "limit": STAC_LIMIT,
    }
    request = urllib.request.Request(
        LANDSAT_ENDPOINT,
        data=json.dumps(body).encode("utf-8"),
        method="POST",
        headers={
            "Accept": "application/geo+json,application/json",
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
            if len(result.get("features", [])) >= STAC_LIMIT:
                raise RuntimeError(
                    f"USGS result reached the {STAC_LIMIT}-item ceiling: {group.group_id}"
                )
            products: list[dict[str, object]] = []
            for feature in result.get("features", []):
                item_id = str(feature.get("id", ""))
                if not item_id.startswith(LANDSAT_PREFIXES):
                    continue
                properties = feature.get("properties", {})
                item_time = parse_datetime(properties["datetime"])
                offset_hours = abs((item_time - center).total_seconds()) / 3600.0
                if offset_hours > 6.0 + 1e-9:
                    continue
                bbox = feature.get("bbox")
                if not isinstance(bbox, list):
                    continue
                covered = [
                    point.sample_id
                    for point in group.points
                    if point_in_bbox(point.longitude, point.latitude, bbox)
                ]
                if not covered:
                    continue
                products.append(
                    {
                        "id": item_id,
                        "datetime": properties["datetime"],
                        "offset_hours": offset_hours,
                        "cloud_cover": properties.get("eo:cloud_cover"),
                        "bbox": bbox,
                        "covered_sample_ids": covered,
                    }
                )
            products.sort(key=lambda item: (item["offset_hours"], item["id"]))
            return {
                "group_id": group.group_id,
                "target_sensor": "landsat",
                "endpoint": LANDSAT_ENDPOINT,
                "collection": LANDSAT_COLLECTION,
                "sensor": group.sensor,
                "hsi_tile": group.hsi_tile,
                "hsi_datetime": group.timestamp_iso,
                "sample_ids": [point.sample_id for point in group.points],
                "products": products,
            }
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
            if attempt + 1 == retries:
                raise RuntimeError(
                    f"USGS Stage B query failed after {retries} attempts: {group.group_id}"
                ) from error
            retry_after = 0.0
            if isinstance(error, urllib.error.HTTPError) and error.code == 429:
                try:
                    retry_after = float(error.headers.get("Retry-After", "0"))
                except (TypeError, ValueError):
                    retry_after = 0.0
            time.sleep(max(retry_after, min(60.0, 5.0 * (2**attempt))))
    raise AssertionError("unreachable")


def run_queries(
    groups: list[CatalogGroup],
    *,
    checkpoint_path: Path,
    workers: int,
    request_delay_seconds: float,
) -> list[dict[str, object]]:
    records = read_jsonl(checkpoint_path)
    completed = {str(record["group_id"]) for record in records}
    pending = [group for group in groups if group.group_id not in completed]
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                query_landsat_group,
                group,
                request_delay_seconds=request_delay_seconds,
            ): group
            for group in pending
        }
        for future in as_completed(futures):
            record = future.result()
            records.append(record)
            with checkpoint_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
    records.sort(key=lambda item: str(item["group_id"]))
    return records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mask-catalog",
        type=Path,
        default=Path(
            ".research/mars_hyperspectral_transfer/train_mask_catalog.jsonl"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            ".research/mars_hyperspectral_transfer/stage_b_landsat.partial.jsonl"
        ),
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=Path(
            ".research/mars_hyperspectral_transfer/stage_b_landsat.jsonl"
        ),
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--request-delay-seconds", type=float, default=0.5)
    parser.add_argument(
        "--max-groups",
        type=int,
        default=None,
        help="Optional smoke-test cap; use a separate checkpoint when set.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.workers < 1 or args.workers > 8:
        raise ValueError("workers must be in [1, 8]")
    mask_records = read_jsonl(args.mask_catalog)
    groups = build_query_groups(mask_records)
    if args.max_groups is not None:
        if args.max_groups < 1:
            raise ValueError("max-groups must be positive")
        groups = groups[: args.max_groups]
    query_records = run_queries(
        groups,
        checkpoint_path=args.checkpoint,
        workers=args.workers,
        request_delay_seconds=args.request_delay_seconds,
    )
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.output_jsonl.write_text(jsonl_payload(query_records), encoding="utf-8")
    print(
        json.dumps(
            {
                "endpoint": LANDSAT_ENDPOINT,
                "collection": LANDSAT_COLLECTION,
                "query_groups": len(groups),
                "groups_with_products": sum(
                    bool(record["products"]) for record in query_records
                ),
                "products": sum(len(record["products"]) for record in query_records),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
