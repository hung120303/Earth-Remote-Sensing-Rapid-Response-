"""Query CDSE for mask-resolved MARS-Hyperspectral Stage B candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import timedelta, timezone
from pathlib import Path

from tools.audit_mars_hyperspectral_transfer import parse_datetime
from tools.query_mars_hyperspectral_cdse import STAC_ENDPOINT


MAX_POINTS_PER_QUERY = 20
STAC_LIMIT = 100


@dataclass(frozen=True)
class SamplePoint:
    sample_id: str
    longitude: float
    latitude: float


@dataclass(frozen=True)
class CatalogGroup:
    group_id: str
    sensor: str
    hsi_tile: str
    timestamp_iso: str
    points: tuple[SamplePoint, ...]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def jsonl_payload(records: list[dict[str, object]]) -> str:
    return "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)


def jsonl_sha256(records: list[dict[str, object]]) -> str:
    return hashlib.sha256(jsonl_payload(records).encode("utf-8")).hexdigest()


def build_query_groups(records: list[dict[str, object]]) -> list[CatalogGroup]:
    grouped: dict[tuple[str, str, str], list[SamplePoint]] = defaultdict(list)
    for record in records:
        if not record["eligible_for_target_catalog"]:
            continue
        key = (
            str(record["sensor"]),
            str(record["tile"]),
            str(record["timestamp"]),
        )
        grouped[key].append(
            SamplePoint(
                sample_id=str(record["sample_id"]),
                longitude=float(record["longitude"]),
                latitude=float(record["latitude"]),
            )
        )
    result: list[CatalogGroup] = []
    for (sensor, tile, timestamp), points in sorted(grouped.items()):
        ordered = sorted(points, key=lambda value: value.sample_id)
        for start in range(0, len(ordered), MAX_POINTS_PER_QUERY):
            chunk = tuple(ordered[start : start + MAX_POINTS_PER_QUERY])
            identity_payload = "\0".join(
                [sensor, tile, timestamp, *(point.sample_id for point in chunk)]
            )
            identity = hashlib.sha256(identity_payload.encode("utf-8")).hexdigest()[:20]
            result.append(
                CatalogGroup(
                    group_id=f"hsi_{identity}",
                    sensor=sensor,
                    hsi_tile=tile,
                    timestamp_iso=timestamp,
                    points=chunk,
                )
            )
    return result


def point_in_bbox(longitude: float, latitude: float, bbox: list[float]) -> bool:
    if len(bbox) < 4:
        return False
    west, south, east, north = map(float, bbox[:4])
    latitude_inside = south - 1e-9 <= latitude <= north + 1e-9
    longitude_inside = (
        west - 1e-9 <= longitude <= east + 1e-9
        if west <= east
        else longitude >= west - 1e-9 or longitude <= east + 1e-9
    )
    return latitude_inside and longitude_inside


def query_cdse_group(
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
        "collections": ["sentinel-2-l1c"],
        "intersects": geometry,
        "datetime": f"{start}/{end}",
        "limit": STAC_LIMIT,
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
    request = urllib.request.Request(
        STAC_ENDPOINT,
        data=json.dumps(body).encode("utf-8"),
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
            if len(result.get("features", [])) >= STAC_LIMIT:
                raise RuntimeError(
                    f"CDSE result reached the {STAC_LIMIT}-item ceiling: {group.group_id}"
                )
            products: list[dict[str, object]] = []
            for feature in result.get("features", []):
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
                        "id": feature["id"],
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
                "sensor": group.sensor,
                "hsi_tile": group.hsi_tile,
                "hsi_datetime": group.timestamp_iso,
                "sample_ids": [point.sample_id for point in group.points],
                "products": products,
            }
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
            if attempt + 1 == retries:
                raise RuntimeError(
                    f"CDSE Stage B query failed after {retries} attempts: {group.group_id}"
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
                query_cdse_group,
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


def _deduplicated_pairs(
    mask_records: list[dict[str, object]],
    query_records: list[dict[str, object]],
) -> list[dict[str, object]]:
    by_sample = {str(record["sample_id"]): record for record in mask_records}
    candidate_by_key: dict[tuple[str, str], dict[str, object]] = {}
    for query_record in query_records:
        for product in query_record["products"]:
            for sample_id in product["covered_sample_ids"]:
                source = by_sample[str(sample_id)]
                label = str(source["label_state"])
                offset = float(product["offset_hours"])
                if label == "NO_PLUME" and offset > 1.0:
                    continue
                if label == "PLUME" and offset > 6.0:
                    continue
                key = (str(sample_id), str(product["datetime"]))
                cloud = product.get("cloud_cover")
                cloud_sort = math.inf if cloud is None else float(cloud)
                candidate = {
                    "sample_id": str(sample_id),
                    "label_state": label,
                    "source_sensor": source["sensor"],
                    "source_tile": source["tile"],
                    "source_datetime": source["timestamp"],
                    "target_product_id": product["id"],
                    "target_datetime": product["datetime"],
                    "offset_hours": offset,
                    "target_tile_cloud_cover": cloud,
                    "country": source["country"],
                    "group_id": source["group_id"],
                    "novel_beyond_all_mars_25km": source[
                        "novel_beyond_all_mars_25km"
                    ],
                    "dense_reprojection_candidate": offset <= 0.25,
                    "high_confidence_scene_pair": offset <= 1.0,
                    "scene_supervision": (
                        "presence_high_confidence"
                        if label == "PLUME" and offset <= 1.0
                        else (
                            "presence_exploratory"
                            if label == "PLUME"
                            else "absence_high_confidence"
                        )
                    ),
                    "_sort": (cloud_sort, str(product["id"])),
                }
                previous = candidate_by_key.get(key)
                if previous is None or candidate["_sort"] < previous["_sort"]:
                    candidate_by_key[key] = candidate
    result = []
    for candidate in candidate_by_key.values():
        candidate.pop("_sort")
        result.append(candidate)
    result.sort(key=lambda item: (str(item["sample_id"]), str(item["target_datetime"])))
    return result


def summarize_stage_b(
    *,
    mask_records: list[dict[str, object]],
    query_records: list[dict[str, object]],
    query_group_count: int,
    protocol_path: Path,
    mask_catalog_path: Path,
    query_catalog_path: Path,
    output_pairs_path: Path,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    pairs = _deduplicated_pairs(mask_records, query_records)
    positive_pairs = [pair for pair in pairs if pair["label_state"] == "PLUME"]
    negative_pairs = [pair for pair in pairs if pair["label_state"] == "NO_PLUME"]
    high_confidence = [pair for pair in pairs if pair["high_confidence_scene_pair"]]
    dense_candidates = [pair for pair in pairs if pair["dense_reprojection_candidate"]]
    pair_samples = {str(pair["sample_id"]) for pair in pairs}
    novel_groups = {
        str(pair["group_id"])
        for pair in pairs
        if pair["novel_beyond_all_mars_25km"]
    }
    countries = {str(pair["country"]) for pair in pairs if pair["country"]}
    gates_config = protocol["stage_b_label_and_catalog_audit"]["gates"]
    metrics = {
        "mask_catalog_samples": len(mask_records),
        "eligible_catalog_samples": sum(
            bool(record["eligible_for_target_catalog"]) for record in mask_records
        ),
        "catalog_query_groups": query_group_count,
        "catalog_query_groups_with_products": sum(
            bool(record["products"]) for record in query_records
        ),
        "samples_with_usable_pair": len(pair_samples),
        "scene_pairs_after_same_acquisition_deduplication": len(pairs),
        "positive_scene_pairs_within_6_hours": len(positive_pairs),
        "positive_scene_pairs_within_1_hour": sum(
            float(pair["offset_hours"]) <= 1.0 for pair in positive_pairs
        ),
        "negative_scene_pairs_within_1_hour": len(negative_pairs),
        "high_confidence_pairs_within_1_hour": len(high_confidence),
        "dense_reprojection_candidates_within_15_minutes": len(dense_candidates),
        "novel_25km_groups_beyond_all_mars_locations": len(novel_groups),
        "countries": len(countries),
        "by_source_sensor": dict(
            sorted(Counter(str(pair["source_sensor"]) for pair in pairs).items())
        ),
        "by_scene_supervision": dict(
            sorted(Counter(str(pair["scene_supervision"]) for pair in pairs).items())
        ),
        "unique_target_products": len(
            {str(pair["target_product_id"]) for pair in pairs}
        ),
    }
    gates = {
        "minimum_positive_scene_pairs": metrics[
            "positive_scene_pairs_within_6_hours"
        ]
        >= int(gates_config["minimum_positive_scene_pairs"]),
        "minimum_negative_scene_pairs": metrics[
            "negative_scene_pairs_within_1_hour"
        ]
        >= int(gates_config["minimum_negative_scene_pairs"]),
        "minimum_novel_25km_groups": metrics[
            "novel_25km_groups_beyond_all_mars_locations"
        ]
        >= int(gates_config["minimum_novel_25km_groups"]),
        "minimum_countries": metrics["countries"]
        >= int(gates_config["minimum_countries"]),
        "minimum_high_confidence_pairs_within_1_hour": metrics[
            "high_confidence_pairs_within_1_hour"
        ]
        >= int(gates_config["minimum_high_confidence_pairs_within_1_hour"]),
    }
    report = {
        "schema_version": 1,
        "decision": "PASS" if all(gates.values()) else "FAIL",
        "scope": "train_mask_truth_and_public_target_catalog_metadata_only",
        "source_revision": protocol["source"]["revision"],
        "target_catalog": {
            "endpoint": STAC_ENDPOINT,
            "collection": "sentinel-2-l1c",
            "scope": "metadata_only_no_target_assets",
        },
        "metrics": metrics,
        "gates": gates,
        "pass": all(gates.values()),
        "inputs": {
            "protocol": {
                "path": protocol_path.as_posix(),
                "sha256": sha256_file(protocol_path),
            },
            "mask_catalog": {
                "path": mask_catalog_path.as_posix(),
                "sha256": sha256_file(mask_catalog_path),
            },
        },
        "ignored_pair_catalog": {
            "path": output_pairs_path.as_posix(),
            "pairs": len(pairs),
            "sha256": jsonl_sha256(pairs),
        },
        "ignored_query_catalog": {
            "path": query_catalog_path.as_posix(),
            "groups": len(query_records),
            "sha256": jsonl_sha256(query_records),
        },
        "claim_boundary": (
            "PASS establishes enough leakage-safe catalog candidates to preregister "
            "target-band acquisition and modeling. Target crop observability, dense "
            "reprojection validity, complementarity, and model improvement remain unproven."
        ),
    }
    return pairs, report


def write_markdown(report: dict[str, object], path: Path) -> None:
    metrics = report["metrics"]
    lines = [
        "# MARS-Hyperspectral transfer Stage B",
        "",
        f"**Decision:** {report['decision']}",
        "",
        "Authoritative labels came only from published train-split `plumemask.tif` pixels. Validation, test, full-tile, retrieval, and official MARS-S2L outcomes remained unopened.",
        "",
        "## Leakage-safe target candidates",
        "",
        f"- Positive scene pairs within 6 hours: {metrics['positive_scene_pairs_within_6_hours']:,}",
        f"- Positive pairs within 1 hour: {metrics['positive_scene_pairs_within_1_hour']:,}",
        f"- Reviewed negative pairs within 1 hour: {metrics['negative_scene_pairs_within_1_hour']:,}",
        f"- All high-confidence pairs within 1 hour: {metrics['high_confidence_pairs_within_1_hour']:,}",
        f"- Dense reprojection candidates within 15 minutes: {metrics['dense_reprojection_candidates_within_15_minutes']:,}",
        f"- Novel 25 km groups beyond every MARS-S2L location: {metrics['novel_25km_groups_beyond_all_mars_locations']:,}",
        f"- Countries: {metrics['countries']:,}",
        f"- Unique Sentinel-2 products: {metrics['unique_target_products']:,}",
        "",
        "## Frozen gates",
        "",
    ]
    lines.extend(
        f"- {'PASS' if passed else 'FAIL'} `{name}`"
        for name, passed in report["gates"].items()
    )
    lines.extend(["", "## Claim boundary", "", report["claim_boundary"], ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


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
        "--protocol",
        type=Path,
        default=Path("configs/mars_hyperspectral_transfer_acquisition_protocol.json"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            ".research/mars_hyperspectral_transfer/stage_b_cdse.partial.jsonl"
        ),
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=Path(
            ".research/mars_hyperspectral_transfer/stage_b_cdse.jsonl"
        ),
    )
    parser.add_argument(
        "--output-pairs",
        type=Path,
        default=Path(
            ".research/mars_hyperspectral_transfer/stage_b_pairs.jsonl"
        ),
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=Path("reports/acquisition/mars_hyperspectral_transfer_stage_b.json"),
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=Path("reports/acquisition/MARS_HYPERSPECTRAL_TRANSFER_STAGE_B.md"),
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--request-delay-seconds", type=float, default=0.75)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.workers < 1 or args.workers > 8:
        raise ValueError("workers must be in [1, 8]")
    mask_records = read_jsonl(args.mask_catalog)
    groups = build_query_groups(mask_records)
    query_records = run_queries(
        groups,
        checkpoint_path=args.checkpoint,
        workers=args.workers,
        request_delay_seconds=args.request_delay_seconds,
    )
    args.output_jsonl.write_text(jsonl_payload(query_records), encoding="utf-8")
    pairs, report = summarize_stage_b(
        mask_records=mask_records,
        query_records=query_records,
        query_group_count=len(groups),
        protocol_path=args.protocol,
        mask_catalog_path=args.mask_catalog,
        query_catalog_path=args.output_jsonl,
        output_pairs_path=args.output_pairs,
    )
    args.output_pairs.write_text(jsonl_payload(pairs), encoding="utf-8")
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_markdown(report, args.output_markdown)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
