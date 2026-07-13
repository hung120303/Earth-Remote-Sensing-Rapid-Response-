# EMIT V002 external ERA5-Land wind requests

## Result

- Frozen requests: **70** across **70** independent groups.
- Unique Sentinel-2 targets: **70**.
- Download state: **CDS authentication and licence acceptance required**.
- Official public CDS request-shape validation: **70/70 accepted**.
- Selection remains prediction-blind; no model output was consulted.

## Contract

- Dataset: [reanalysis-era5-land-timeseries](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land-timeseries?tab=download) under CC-BY-4.0.
- Variables: 10-m eastward (`u`) and northward (`v`) wind, in m/s.
- Space: the CDS-selected nearest 0.1-degree ERA5-Land grid point to the frozen plume center.
- Time: nearest hourly validity time to the Sentinel-2 target acquisition; exact half-hour ties choose the later hour.
- Sensitivity: the immediately previous and following hourly values are retained before model evaluation.
- Each request spans one day on either side of the target so the bracket is available across UTC date boundaries.
- Raw CSVs are written beneath `EarthRemoteSensingRapidResponse/Data Collection/s2_emit_pairs/emit-v002-external-l1c-2026-07/era5_land`, which is ignored by Git.

## Authentication boundary

The official CDS service requires a personal account, manual acceptance of the dataset CC-BY terms, and a personal access token stored in `$HOME/.cdsapirc`. Credentials must never be placed in this repository. Earthdata authentication is unrelated and cannot authorize this download. The compact request payloads and their SHA-256 identities are frozen here before any external model prediction.

## Exact handoff and execution

1. Sign in or register at [Copernicus CDS](https://cds.climate.copernicus.eu/).
2. Open the [ERA5-Land time-series dataset](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land-timeseries?tab=download) and accept its terms.
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
