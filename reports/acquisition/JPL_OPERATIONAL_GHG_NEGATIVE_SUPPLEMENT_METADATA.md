# JPL CACH4 train-header metadata audit

Status: **PENDING protected-site and duplicate filtering**. This is not a metadata-stage pass and no target Sentinel-2/Landsat catalog was queried.

## Scope and interpretation

Only released `multicampaign_train.csv` definitions and 124 tiny public JPL ENVI `.hdr` sidecars were read. The headers are geospatial metadata matching the frozen protocol's permitted crop-georeferencing field; no retrieval raster, GeoTIFF, label raster, target asset, or released test content was opened.

The JPL CMF suffix `+A+B` is interpreted as ENVI sample/column `A`, then line/row `B`. All 3,332 train crop centers passed `0 <= A + width/2 < samples` and `0 <= B + height/2 < lines`. The released sampler pads edge windows: 1,015 rows overhang the sample axis, 159 overhang the line axis, and 1,131 overhang either axis (maximum 127 sample pixels and 127 line pixels). This does not move the crop center outside the source image. Crop centers use the GDAL ENVI affine convention with rotation radians `-degrees*pi/180` before conversion from UTM WGS84 to EPSG:4326.

## Compact result

- Dataset license metadata: `{'id': 'cc-by-4.0'}`
- CACH4 train rows: 3,332 (3,149 background; 183 plume)
- Train flightlines listed and header-resolved: 124
- Background rows with exact UTC and WGS84 crop center: 3,149
- Coordinate-resolution gate: **PASS**
- Target-catalog eligibility: **FALSE pending 25 km protected-site and duplicate filtering**

Detailed headers and row-level coordinates remain beneath ignored `.research/jpl_operational_ghg_supplement`. The compact JSON lists all train flight IDs and exposes aggregate gates without publishing the detailed location catalog.
