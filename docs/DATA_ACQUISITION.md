# ERSRR Data Acquisition Guide

This project learns a mapping from Sentinel-2 multispectral imagery to EMIT methane plume targets. More model work will not help much until the dataset grows beyond the current ~100 paired tiles.

## Primary data sources

1. EMIT methane plume complexes in Google Earth Engine
   - Dataset ID: `NASA/EMIT/L2B/CH4PLM`
   - Catalog: https://developers.google.com/earth-engine/datasets/catalog/NASA_EMIT_L2B_CH4PLM
   - Use this when you want the fastest path from plume geometries to paired Sentinel-2 exports.

2. EMIT L2B CH4PLM V002 from NASA Earthdata / LP DAAC
   - Catalog: https://www.earthdata.nasa.gov/data/catalog/lpcloud-emitl2bch4plm-002
   - DOI: https://doi.org/10.5067/EMIT/EMITL2BCH4PLM.002
   - Use this when you want original COG + GeoJSON granules locally.
   - Earthdata account required for downloads: https://urs.earthdata.nasa.gov/
   - Earthdata Search: https://search.earthdata.nasa.gov/

3. Sentinel-2 L2A / harmonized imagery
   - Google Earth Engine collection used by the repo: `COPERNICUS/S2_HARMONIZED`
   - Copernicus Browser for manual downloads/checks: https://browser.dataspace.copernicus.eu/
   - Copernicus Sentinel-2 docs: https://dataspace.copernicus.eu/data-collections/copernicus-sentinel-missions/sentinel-2

## What to collect

Prioritize EMIT plume scenes that have:
- non-trivial valid CH4 plume coverage;
- visible plume concentration above background;
- usable Sentinel-2 imagery within a small time window;
- low cloud cover;
- diverse regions/surfaces, not just repeated neighboring tiles.

Also collect negatives:
- Sentinel-2 tiles from similar geographies and seasons with no known methane plume;
- target CH4 band should be zero with an explicit valid mask or reliable nodata metadata.

## Manual download workflow if you need to grab data yourself

1. Create/sign into NASA Earthdata:
   https://urs.earthdata.nasa.gov/

2. Open the EMIT CH4PLM V002 catalog:
   https://www.earthdata.nasa.gov/data/catalog/lpcloud-emitl2bch4plm-002

3. Click the Earthdata Search / data access option for the product.

4. In Earthdata Search:
   - search for `EMITL2BCH4PLM` or use the product page's direct search link;
   - filter dates from 2022-08 onward;
   - select plume granules in regions with likely Sentinel-2 coverage;
   - download both the `.tif` COG and the companion `.json` / GeoJSON metadata when available.

5. Put each batch under:
   `EarthRemoteSensingRapidResponse/Data Collection/EMIT_Plumes/<batch-name>/`

6. Authenticate Google Earth Engine locally:
   `earthengine authenticate`

7. Use the repo's pairing/processing workflow or future CLI collection command to export matching Sentinel-2 tiles, then run:
   `python tools/ersrr.py audit`

## Recommended next targets

- Grow validation from 3 files to at least 50 held-out tiles.
- Keep held-out validation geographically/source separated from training.
- Add 200+ negative/background Sentinel-2 samples before investing in heavier architectures.
- Track every pair in the audit manifest; do not train on unlabeled mystery TIFFs.
