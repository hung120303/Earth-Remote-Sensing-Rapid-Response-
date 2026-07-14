#!/usr/bin/env python3
"""Audit scene labels against raw and observable MARS-S2L plume masks."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import rasterio

from acquire_mars_metadata import DEFAULT_OUTPUT, repo_root, sha256
from train_mars_paper_residual import DEFAULT_MANIFEST, DEFAULT_PROTOCOL
from mars_s2l_adapter import iter_development_manifest, role_paths, safe_asset_path

DEFAULT_JSON = Path("reports/acquisition/mars_s2l_development_label_audit.json")
DEFAULT_MARKDOWN = Path("reports/acquisition/MARS_S2L_DEVELOPMENT_LABEL_AUDIT.md")


def empty_stratum() -> dict[str, int]:
    return {
        "positive_scenes": 0,
        "raw_empty_masks": 0,
        "observable_empty_masks": 0,
        "raw_positive_pixels": 0,
        "observable_positive_pixels": 0,
    }


def add_stratum(target: dict[str, int], raw_pixels: int, observable_pixels: int) -> None:
    target["positive_scenes"] += 1
    target["raw_empty_masks"] += int(raw_pixels == 0)
    target["observable_empty_masks"] += int(observable_pixels == 0)
    target["raw_positive_pixels"] += raw_pixels
    target["observable_positive_pixels"] += observable_pixels


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    overall = report["overall"]
    lines = [
        "# MARS-S2L development label audit",
        "",
        "This audit covers development labels only; the sealed paper test was not loaded.",
        "",
        f"- Positive scenes: {overall['positive_scenes']:,}",
        f"- Raw empty positive masks: {overall['raw_empty_masks']:,}",
        f"- Positive masks empty after cloud/radiometric observability: {overall['observable_empty_masks']:,}",
        f"- Raw / observable positive pixels: {overall['raw_positive_pixels']:,} / {overall['observable_positive_pixels']:,}",
        "",
        "| Stratum | Positive | Raw empty | Observable empty |",
        "|---|---:|---:|---:|",
    ]
    for name, value in report["strata"].items():
        lines.append(
            f"| {name} | {value['positive_scenes']:,} | "
            f"{value['raw_empty_masks']:,} | {value['observable_empty_masks']:,} |"
        )
    lines.extend(["", "## Exceptions", ""])
    if report["exceptions"]:
        for item in report["exceptions"]:
            lines.append(
                f"- `{item['sample_id']}` ({item['split']}, fold {item['fold']}, "
                f"{item['sensor_family']}): raw={item['raw_positive_pixels']}, "
                f"observable={item['observable_positive_pixels']} pixels."
            )
    else:
        lines.append("None.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-dir", default=DEFAULT_OUTPUT.as_posix())
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST.as_posix())
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL.as_posix())
    parser.add_argument("--output-json", default=DEFAULT_JSON.as_posix())
    parser.add_argument("--output-markdown", default=DEFAULT_MARKDOWN.as_posix())
    args = parser.parse_args()

    root = repo_root()
    metadata_dir = (root / args.metadata_dir).resolve()
    manifest = (root / args.manifest).resolve()
    protocol_path = (root / args.protocol).resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if sha256(manifest) != protocol["development_manifest_sha256"]:
        raise ValueError("Development manifest differs from the frozen protocol")
    group_to_fold = {
        str(item["group_id"]): int(item["fold"])
        for item in protocol["assignments"]
    }

    overall = empty_stratum()
    strata: defaultdict[str, dict[str, int]] = defaultdict(empty_stratum)
    exceptions: list[dict[str, Any]] = []
    for record in iter_development_manifest(manifest):
        if record["label_state"] != "PLUME":
            continue
        paths = role_paths(record)
        with rasterio.open(safe_asset_path(metadata_dir, paths["plume_mask"])) as source:
            mask = source.read(1).astype(bool)
        with rasterio.open(safe_asset_path(metadata_dir, paths["cloud_mask"])) as source:
            clear = source.read(1) == 0
        with rasterio.open(safe_asset_path(metadata_dir, paths["image"])) as source:
            radiometric = np.all(source.read() != 0, axis=0)
        observable_mask = mask & clear & radiometric
        raw_pixels = int(np.count_nonzero(mask))
        observable_pixels = int(np.count_nonzero(observable_mask))
        fold = group_to_fold[str(record["group_id"])]
        add_stratum(overall, raw_pixels, observable_pixels)
        for name in (
            f"split:{record['split']}",
            f"fold:{fold}",
            f"sensor:{record['sensor_family']}",
        ):
            add_stratum(strata[name], raw_pixels, observable_pixels)
        if raw_pixels == 0 or observable_pixels == 0:
            exceptions.append(
                {
                    "sample_id": record["sample_id"],
                    "group_id": record["group_id"],
                    "split": record["split"],
                    "fold": fold,
                    "sensor_family": record["sensor_family"],
                    "observability": record["observability"],
                    "raw_positive_pixels": raw_pixels,
                    "observable_positive_pixels": observable_pixels,
                }
            )

    report = {
        "schema_version": 1,
        "scope": "development labels only; sealed paper test not loaded",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "development_manifest_sha256": sha256(manifest),
        "protocol_sha256": sha256(protocol_path),
        "overall": overall,
        "strata": dict(sorted(strata.items())),
        "exceptions": sorted(exceptions, key=lambda item: item["sample_id"]),
    }
    write_json((root / args.output_json).resolve(), report)
    write_markdown((root / args.output_markdown).resolve(), report)
    print(json.dumps({"ok": True, "overall": overall, "exceptions": len(exceptions)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
