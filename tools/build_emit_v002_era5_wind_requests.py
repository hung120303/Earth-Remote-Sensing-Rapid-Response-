#!/usr/bin/env python3
"""Freeze prediction-blind ERA5-Land requests for the external EMIT cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path
from typing import Any

import requests

from acquire_v002_pilot import parse_datetime

DEFAULT_INPUT = Path("reports/acquisition/emit_v002_l1c_pairs.json")
DEFAULT_JSON = Path("reports/acquisition/emit_v002_era5_wind_requests.json")
DEFAULT_MARKDOWN = Path("reports/acquisition/EMIT_V002_ERA5_WIND_REQUESTS.md")
DATASET = "reanalysis-era5-land-timeseries"
VARIABLES = ("10m_u_component_of_wind", "10m_v_component_of_wind")
DATASET_URL = (
    "https://cds.climate.copernicus.eu/datasets/"
    "reanalysis-era5-land-timeseries?tab=download"
)
COSTING_URL = (
    "https://cds.climate.copernicus.eu/api/retrieve/v1/processes/"
    f"{DATASET}/costing"
)
RAW_ROOT = (
    "EarthRemoteSensingRapidResponse/Data Collection/s2_emit_pairs/"
    "emit-v002-external-l1c-2026-07/era5_land"
)


def repo_root() -> Path:
    value = subprocess.check_output(
        ["git", "rev-parse", "--show-toplevel"], text=True
    ).strip()
    return Path(value).resolve()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_output(root: Path, value: str) -> Path:
    result = (root / value).resolve()
    if root not in result.parents:
        raise ValueError("Output must resolve beneath the repository root")
    return result


def hourly_bracket(value: str) -> tuple[str, str, str]:
    """Return the previous, nearest, and next UTC validity times.

    Nearest-hour ties are resolved toward the later validity time. The adjacent
    hours are retained for a predeclared temporal-sensitivity analysis.
    """

    timestamp = parse_datetime(value)
    previous = timestamp.replace(minute=0, second=0, microsecond=0)
    following = previous + timedelta(hours=1)
    nearest = following if timestamp - previous >= timedelta(minutes=30) else previous
    return tuple(item.isoformat().replace("+00:00", "Z") for item in (previous, nearest, following))


def cds_request(center: list[float], target_datetime: str) -> dict[str, Any]:
    if len(center) != 2:
        raise ValueError("Expected [longitude, latitude] center")
    longitude, latitude = map(float, center)
    if not -180.0 <= longitude <= 180.0 or not -90.0 <= latitude <= 90.0:
        raise ValueError("Candidate center is outside longitude/latitude bounds")
    target = parse_datetime(target_datetime)
    start = (target - timedelta(days=1)).date().isoformat()
    end = (target + timedelta(days=1)).date().isoformat()
    return {
        "variable": list(VARIABLES),
        "date": [start, end],
        "location": {
            "latitude": round(latitude, 6),
            "longitude": round(longitude, 6),
        },
        "data_format": "csv",
    }


def build_manifest(source: dict[str, Any], input_path: Path) -> dict[str, Any]:
    requests: list[dict[str, Any]] = []
    for pair in source.get("pairs", []):
        if pair.get("status") != "paired":
            continue
        group_id = str(pair["group_id"])
        target_datetime = str(pair["target"]["datetime"])
        payload = cds_request(pair["center"], target_datetime)
        payload_sha = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        previous, nearest, following = hourly_bracket(target_datetime)
        requests.append(
            {
                "group_id": group_id,
                "granule_id": pair["granule_id"],
                "sentinel2_target_scene_id": pair["target"]["l1c_scene_id"],
                "sentinel2_target_datetime": target_datetime,
                "candidate_center": pair["center"],
                "hourly_validity_times": {
                    "previous": previous,
                    "nearest": nearest,
                    "following": following,
                },
                "cds_request": payload,
                "cds_request_sha256": payload_sha,
                "ignored_raw_csv": f"{RAW_ROOT}/{group_id}.csv",
            }
        )
    requests.sort(key=lambda item: item["group_id"])
    groups = [item["group_id"] for item in requests]
    if len(groups) != len(set(groups)):
        raise ValueError("ERA5 request manifest contains duplicate groups")
    return {
        "contract": {
            "input_manifest": input_path.as_posix(),
            "input_manifest_sha256": sha256(input_path),
            "dataset": DATASET,
            "dataset_url": DATASET_URL,
            "license": "CC-BY-4.0",
            "variables": list(VARIABLES),
            "units": "m s-1",
            "spatial_policy": "CDS nearest native 0.1-degree ERA5-Land grid point",
            "temporal_policy": (
                "nearest UTC hourly validity time to Sentinel-2 target acquisition; "
                "ties choose the later hour"
            ),
            "sensitivity_policy": "also retain previous and following hourly values",
            "selection_inputs": "frozen public metadata only; no detector predictions",
            "credentials": (
                "CDS personal access token in $HOME/.cdsapirc outside the repository; "
                "dataset CC-BY terms must be accepted manually"
            ),
            "raw_storage": RAW_ROOT,
        },
        "summary": {
            "requests": len(requests),
            "unique_groups": len(set(groups)),
            "unique_target_scenes": len(
                {item["sentinel2_target_scene_id"] for item in requests}
            ),
            "authentication_state": "required_before_download",
            "official_costing_api_validation": "not_run",
        },
        "requests": requests,
    }


def validate_costing(requests_to_check: list[dict[str, Any]], workers: int) -> dict[str, Any]:
    """Validate request shapes against the public official CDS costing endpoint."""

    def check(item: dict[str, Any]) -> dict[str, Any]:
        error = ""
        for attempt in range(3):
            try:
                response = requests.post(
                    COSTING_URL,
                    json={"inputs": item["cds_request"]},
                    timeout=60,
                    headers={"User-Agent": "ERSRR-research/1.0"},
                )
                if response.status_code == 200:
                    return {"group_id": item["group_id"], "status": "accepted"}
                error = f"HTTP {response.status_code}: {response.text[:200]}"
            except requests.RequestException as exc:
                error = str(exc)
            if attempt < 2:
                time.sleep(float(2**attempt))
        return {"group_id": item["group_id"], "status": "error", "error": error}

    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(check, requests_to_check))
    failures = [item for item in results if item["status"] != "accepted"]
    if failures:
        raise RuntimeError(f"CDS costing validation failed: {failures[:5]}")
    return {"checked": len(results), "accepted": len(results), "endpoint": COSTING_URL}


def markdown(result: dict[str, Any]) -> str:
    summary = result["summary"]
    contract = result["contract"]
    return f"""# EMIT V002 external ERA5-Land wind requests

## Result

- Frozen requests: **{summary['requests']}** across **{summary['unique_groups']}** independent groups.
- Unique Sentinel-2 targets: **{summary['unique_target_scenes']}**.
- Download state: **CDS authentication and licence acceptance required**.
- Official public CDS request-shape validation: **{summary['official_costing_api_validation']}**.
- Selection remains prediction-blind; no model output was consulted.

## Contract

- Dataset: [{contract['dataset']}]({contract['dataset_url']}) under CC-BY-4.0.
- Variables: 10-m eastward (`u`) and northward (`v`) wind, in m/s.
- Space: the CDS-selected nearest 0.1-degree ERA5-Land grid point to the frozen plume center.
- Time: nearest hourly validity time to the Sentinel-2 target acquisition; exact half-hour ties choose the later hour.
- Sensitivity: the immediately previous and following hourly values are retained before model evaluation.
- Each request spans one day on either side of the target so the bracket is available across UTC date boundaries.
- Raw CSVs are written beneath `{contract['raw_storage']}`, which is ignored by Git.

## Authentication boundary

The official CDS service requires a personal account, manual acceptance of the dataset CC-BY terms, and a personal access token stored in `$HOME/.cdsapirc`. Credentials must never be placed in this repository. Earthdata authentication is unrelated and cannot authorize this download. The compact request payloads and their SHA-256 identities are frozen here before any external model prediction.

## Exact handoff and execution

1. Sign in or register at [Copernicus CDS](https://cds.climate.copernicus.eu/).
2. Open the [ERA5-Land time-series dataset]({contract['dataset_url']}) and accept its terms.
3. Follow the [official API setup](https://cds.climate.copernicus.eu/how-to-api) and create
   `/home/joshu/.cdsapirc` inside WSL. Do not put the token anywhere under the repository.
4. From the repository root in WSL, acquire the frozen requests:

   ```bash
   .venv/bin/python tools/acquire_emit_v002_era5_wind.py --workers 1
   ```

5. Verify every downloaded CSV, parsed hourly bracket, file size, and SHA-256 receipt without
   redownloading:

   ```bash
   .venv/bin/python tools/acquire_emit_v002_era5_wind.py --verify-only
   ```

The downloader writes raw CSVs only to the ignored directory declared above and writes compact,
token-free acquisition evidence to `reports/acquisition/emit_v002_era5_wind_acquisition.json` and
`reports/acquisition/EMIT_V002_ERA5_WIND_ACQUISITION.md`. After verification, the research workflow
runs `tools/evaluate_emit_v002_external.py` exactly once against the sealed 55-group cohort.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_INPUT.as_posix())
    parser.add_argument("--output-json", default=DEFAULT_JSON.as_posix())
    parser.add_argument("--output-markdown", default=DEFAULT_MARKDOWN.as_posix())
    parser.add_argument("--skip-cost-validation", action="store_true")
    parser.add_argument("--costing-workers", type=int, default=8)
    args = parser.parse_args()
    root = repo_root()
    input_path = safe_output(root, args.input)
    output_json = safe_output(root, args.output_json)
    output_markdown = safe_output(root, args.output_markdown)
    source = json.loads(input_path.read_text(encoding="utf-8"))
    result = build_manifest(source, input_path.relative_to(root))
    if not args.skip_cost_validation:
        validation = validate_costing(result["requests"], args.costing_workers)
        result["summary"]["official_costing_api_validation"] = (
            f"{validation['accepted']}/{validation['checked']} accepted"
        )
        result["contract"]["costing_validation"] = validation
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_markdown.write_text(markdown(result), encoding="utf-8")
    print(json.dumps(result["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
