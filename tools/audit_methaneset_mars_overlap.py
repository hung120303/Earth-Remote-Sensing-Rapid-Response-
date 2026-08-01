#!/usr/bin/env python3
"""Audit MethaneSET sample identity overlap with the MARS-S2L paper corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_MARS_METADATA = Path(
    "EarthRemoteSensingRapidResponse/Data Collection/s2_emit_pairs/publication-v1/"
    "external/MARS-S2L-paper-source/validated_images_all_20251129.csv"
)
DEFAULT_JSON = Path("reports/acquisition/methaneset_mars_overlap_audit.json")
DEFAULT_MARKDOWN = Path("reports/acquisition/METHANESET_MARS_OVERLAP_AUDIT.md")
METHANESET_REVISION = "f7dc01e166d7a2f4ae659a7046b2c5a3c49ab7be"
DATASETS = {
    "methaneset-s2-pretraining": Path(
        ".research/methaneset_multisensor_audit/methaneset-s2-pretraining"
    ),
    "methaneset-s2-finetune": Path(
        ".research/methaneset_s2_finetune_audit/methaneset-s2-finetune"
    ),
    "methaneset-l89-pretraining": Path(
        ".research/methaneset_multisensor_audit/methaneset-l89-pretraining"
    ),
    "methaneset-l89-finetune": Path(
        ".research/methaneset_multisensor_audit/methaneset-l89-finetune"
    ),
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


def audit_dataset(
    root: Path,
    name: str,
    relative: Path,
    mars_split_by_id: dict[str, str],
) -> dict[str, Any]:
    parquet = root / relative / ".tacocat" / "level0.parquet"
    readme = root / relative / "README.md"
    frame = pd.read_parquet(parquet, columns=["id", "split"])
    if frame["id"].duplicated().any():
        raise ValueError(f"Duplicate MethaneSET ids in {name}")
    identifiers = frame["id"].astype(str)
    matched = identifiers.map(mars_split_by_id)
    source_split = Counter(str(value) for value in frame["split"])
    paper_split = Counter(str(value) for value in matched.dropna())
    crosswalk = Counter(
        f"{source}->{paper}"
        for source, paper in zip(frame["split"], matched)
        if pd.notna(paper)
    )
    return {
        "rows": len(frame),
        "unique_ids": int(identifiers.nunique()),
        "exact_mars_id_overlap": int(matched.notna().sum()),
        "not_in_mars_metadata": int(matched.isna().sum()),
        "source_split": dict(sorted(source_split.items())),
        "mars_split_of_exact_matches": dict(sorted(paper_split.items())),
        "split_crosswalk": dict(sorted(crosswalk.items())),
        "official_test_exact_matches": int((matched == "test_2023").sum()),
        "metadata": {
            "path": parquet.relative_to(root).as_posix(),
            "bytes": parquet.stat().st_size,
            "sha256": sha256(parquet),
        },
        "readme": {
            "path": readme.relative_to(root).as_posix(),
            "bytes": readme.stat().st_size,
            "sha256": sha256(readme),
        },
    }


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# MethaneSET / MARS-S2L identity-overlap audit",
        "",
        f"Generated: {report['generated_at_utc']}.",
        "",
        "| MethaneSET subset | Rows | Exact MARS IDs | Exact official-test IDs |",
        "|---|---:|---:|---:|",
    ]
    for name, result in report["datasets"].items():
        lines.append(
            f"| {name} | {result['rows']:,} | {result['exact_mars_id_overlap']:,} | {result['official_test_exact_matches']:,} |"
        )
    lines.extend(
        [
            "",
            "MethaneSET is a valuable repackaging of the MARS corpus, but these four multispectral subsets are not new independent supervision for the exact MARS-S2L paper benchmark. Downloading the imagery would either duplicate existing training rows or leak exact validation/test observations. Only metadata was downloaded; no MethaneSET imagery was acquired or used.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mars-metadata", default=DEFAULT_MARS_METADATA.as_posix())
    parser.add_argument("--output-json", default=DEFAULT_JSON.as_posix())
    parser.add_argument("--output-markdown", default=DEFAULT_MARKDOWN.as_posix())
    args = parser.parse_args()
    root = repo_root()
    mars_path = root / args.mars_metadata
    mars = pd.read_csv(mars_path, usecols=["id_loc_image", "split_name"], low_memory=False)
    if mars["id_loc_image"].duplicated().any():
        raise ValueError("MARS image ids are not unique")
    split_by_id = dict(zip(mars["id_loc_image"].astype(str), mars["split_name"].astype(str)))
    datasets = {
        name: audit_dataset(root, name, relative, split_by_id)
        for name, relative in DATASETS.items()
    }
    report = {
        "schema_version": 1,
        "status": "complete; metadata-only acquisition; all imagery rejected as duplicate/leaking",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "repo_id": "tacofoundation/methaneset",
            "revision": METHANESET_REVISION,
            "url": "https://huggingface.co/datasets/tacofoundation/methaneset",
            "image_files_downloaded": 0,
        },
        "mars": {
            "metadata_path": args.mars_metadata,
            "metadata_sha256": sha256(mars_path),
            "columns_accessed": ["id_loc_image", "split_name"],
            "labels_accessed": False,
            "rows": len(mars),
        },
        "datasets": datasets,
        "decision": (
            "Do not acquire or train on MethaneSET multispectral imagery for the exact MARS "
            "successor campaign. It is a repackaging of the comparator corpus, not an "
            "independent external cohort."
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
    print(
        json.dumps(
            {
                "ok": True,
                "datasets": {
                    name: {
                        "rows": result["rows"],
                        "exact_mars_id_overlap": result["exact_mars_id_overlap"],
                        "official_test_exact_matches": result[
                            "official_test_exact_matches"
                        ],
                    }
                    for name, result in datasets.items()
                },
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
