"""Combine preregistered Sentinel-2 and Landsat Stage B feasibility results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.query_mars_hyperspectral_cdse import STAC_ENDPOINT
from tools.query_mars_hyperspectral_stage_b_cdse import (
    jsonl_payload,
    read_jsonl,
    sha256_file,
    summarize_stage_b,
    write_markdown,
)
from tools.query_mars_hyperspectral_stage_b_landsat import (
    LANDSAT_COLLECTION,
    LANDSAT_ENDPOINT,
)


def annotate_sentinel(records: list[dict[str, object]]) -> list[dict[str, object]]:
    result = []
    for source in records:
        record = dict(source)
        record["target_sensor"] = "sentinel2"
        record["endpoint"] = STAC_ENDPOINT
        record["collection"] = "sentinel-2-l1c"
        result.append(record)
    return result


def combine(
    *,
    mask_catalog_path: Path,
    sentinel_query_path: Path,
    landsat_query_path: Path,
    protocol_path: Path,
    combined_query_path: Path,
    output_pairs_path: Path,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    mask_records = read_jsonl(mask_catalog_path)
    sentinel_records = annotate_sentinel(read_jsonl(sentinel_query_path))
    landsat_records = read_jsonl(landsat_query_path)
    combined_records = sorted(
        [*sentinel_records, *landsat_records],
        key=lambda record: (str(record["target_sensor"]), str(record["group_id"])),
    )
    combined_query_path.parent.mkdir(parents=True, exist_ok=True)
    combined_query_path.write_text(jsonl_payload(combined_records), encoding="utf-8")
    pairs, report = summarize_stage_b(
        mask_records=mask_records,
        query_records=combined_records,
        query_group_count=len(combined_records),
        protocol_path=protocol_path,
        mask_catalog_path=mask_catalog_path,
        query_catalog_path=combined_query_path,
        output_pairs_path=output_pairs_path,
        target_catalogs=[
            {
                "sensor": "sentinel2",
                "endpoint": STAC_ENDPOINT,
                "collection": "sentinel-2-l1c",
                "scope": "metadata_only_no_target_assets",
            },
            {
                "sensor": "landsat",
                "endpoint": LANDSAT_ENDPOINT,
                "collection": LANDSAT_COLLECTION,
                "scope": "metadata_only_no_target_assets",
            },
        ],
    )
    report["source_query_catalogs"] = [
        {
            "sensor": "sentinel2",
            "path": sentinel_query_path.as_posix(),
            "groups": len(sentinel_records),
            "sha256": sha256_file(sentinel_query_path),
        },
        {
            "sensor": "landsat",
            "path": landsat_query_path.as_posix(),
            "groups": len(landsat_records),
            "sha256": sha256_file(landsat_query_path),
        },
    ]
    return pairs, report


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
        "--sentinel-query",
        type=Path,
        default=Path(
            ".research/mars_hyperspectral_transfer/stage_b_cdse.jsonl"
        ),
    )
    parser.add_argument(
        "--landsat-query",
        type=Path,
        default=Path(
            ".research/mars_hyperspectral_transfer/stage_b_landsat.jsonl"
        ),
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("configs/mars_hyperspectral_transfer_acquisition_protocol.json"),
    )
    parser.add_argument(
        "--combined-query",
        type=Path,
        default=Path(
            ".research/mars_hyperspectral_transfer/stage_b_combined_queries.jsonl"
        ),
    )
    parser.add_argument(
        "--output-pairs",
        type=Path,
        default=Path(
            ".research/mars_hyperspectral_transfer/stage_b_combined_pairs.jsonl"
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
    return parser


def main() -> None:
    args = build_parser().parse_args()
    pairs, report = combine(
        mask_catalog_path=args.mask_catalog,
        sentinel_query_path=args.sentinel_query,
        landsat_query_path=args.landsat_query,
        protocol_path=args.protocol,
        combined_query_path=args.combined_query,
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
