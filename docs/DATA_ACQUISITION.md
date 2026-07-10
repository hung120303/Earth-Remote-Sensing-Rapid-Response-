# ERSRR data acquisition contract

ERSRR needs paired Sentinel-2 imagery and methane-plume labels. Acquisition must preserve source product, time, grid, validity, and provenance; a plume polygon must never be presented as a concentration raster.

## Supported sources

### EMIT methane labels

- **EMIT L2B CH4PLM V002** is the preferred source for new work: <https://www.earthdata.nasa.gov/data/catalog/lpcloud-emitl2bch4plm-002>
- DOI: <https://doi.org/10.5067/EMIT/EMITL2BCH4PLM.002>
- Direct Earthdata Search collection: <https://search.earthdata.nasa.gov/search/granules?p=C3242707413-LPCLOUD>
- NASA CMR UMM metadata is public and contains the authoritative V002 plume footprint.
- The per-pixel concentration COG and companion metadata are protected by Earthdata Login. Do not automate credentials into the repository.

Google Earth Engine currently exposes EMIT CH4PLM V001, not V002: <https://developers.google.com/earth-engine/datasets/catalog/NASA_EMIT_L2B_CH4PLM>. Keep V001 and V002 provenance explicit.

### Sentinel-2 inputs

- New pilot collection uses public Element 84 `sentinel-2-l2a` COGs (surface reflectance).
- The legacy curated ERSRR tiles came from `COPERNICUS/S2_HARMONIZED` (L1C/TOA): <https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_HARMONIZED>
- Future Earth Engine L2A work should use `COPERNICUS/S2_SR_HARMONIZED`: <https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED>

Never silently combine L1C/TOA and L2A/surface-reflectance values. Either train separate artifacts or implement and validate an explicit domain-harmonization step.

## Reproducible public V002 pilot

From the repository root:

```bash
python tools/acquire_v002_pilot.py \
  EMIT_L2B_CH4PLM_002_20250922T204933_003374 \
  --batch emit-v002-2026-07 \
  --window-days 14 \
  --max-cloud 20 \
  --temporal-mode bracketing
```

The collector:

1. resolves the exact V002 granule through NASA CMR collection `C3242707413-LPCLOUD`;
2. saves public UMM metadata, the physical plume GeoJSON, and the public browse image;
3. deterministically ranks Sentinel-2 L2A scenes by absolute time offset, scene cloud cover, and scene ID;
4. selects the closest before/after scenes when both exist;
5. writes co-registered 256 x 256, 20 m stacks in canonical `B2, B3, B4, B11, B12` order;
6. rasterizes the same V002 plume footprint on the shared before/after grid;
7. records cloud/clear percentages, timestamps, offsets, CRS, transform, source URLs, and SHA-256 hashes in `manifest.json`.

Outputs are local and ignored:

- `EarthRemoteSensingRapidResponse/Data Collection/EMIT_Plumes/<batch>/`
- `EarthRemoteSensingRapidResponse/Data Collection/s2_emit_pairs/<batch>/<granule>/`

The July 2026 research batch contains 12 V002 granules, 24 L2A image stacks, and 24 masks (11.31 MB). A containment audit found six plume polygons clipped by the fixed 5.12 km frame; four of those also occupied more than half of the tile. All six clipped groups are excluded, leaving only six independent groups for a feasibility benchmark. This batch is too small and too positive-heavy for architecture or operational accuracy claims.

## Label contracts

### Physical polygon-mask contract

- task: binary plume segmentation;
- label: public NASA CMR V002 plume footprint;
- values: `0` background and `1` plume;
- before/after masks may be identical because the label is tied to the EMIT observation, not the Sentinel-2 acquisition;
- must not be inserted as band 6 of the legacy concentration dataset.

### Concentration-raster contract

- task: methane enhancement regression or threshold-derived segmentation;
- label: authenticated EMIT concentration COG with original units and nodata mask;
- target validity and physical calibration must be independent of the prediction;
- protected data may be stored locally but must not be committed.

## Quality gates before promotion

A batch can move into a curated dataset only after it has:

- an immutable manifest with hashes and product level;
- a valid-mask definition and corrected nodata metadata;
- a bounded temporal gap (the current legacy research filter is `|gap| <= 7 days`);
- a documented cloud/clear threshold;
- geographic/source group IDs for leakage-free splitting;
- true negative/background examples from comparable geography and season;
- inspection for duplicate scenes, implausibly broad masks, and train/validation group overlap.
- proof that the full physical plume geometry is contained in the raster or a documented tiling strategy that preserves it.

Run the audit after any curated-dataset change:

```bash
python tools/ersrr.py status
python tools/ersrr.py audit
```

## Immediate acquisition priorities

1. Obtain authenticated V002 concentration COGs through a user-managed Earthdata session and preserve their nodata/unit metadata.
2. Add at least 200 hard negatives matched by geography, season, surface type, and cloud regime.
3. Grow a geographically/source-disjoint held-out validation set to at least 50 independent groups.
4. Collect tighter-time L2A pairs and keep before/after stacks on one grid.
5. Consider synthetic plume injection only as augmentation; retain a real-data-only test set.
