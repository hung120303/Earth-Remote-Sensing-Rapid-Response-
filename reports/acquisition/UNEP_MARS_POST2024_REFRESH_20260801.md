# UNEP MARS post-2024 catalog refresh — 2026-08-01

The current official UNEP/IMEO detected-plume catalog was downloaded and audited with the already frozen exact-product and paper-test spatial-exclusion rules. Bulk archives, extracted files, and the eligible manifest remain under `.research/unep_mars_post2024_refresh_20260801/` and are not committed.

## Provenance

- Official catalog page: <https://data.unep.org/methane>
- CSV archive: <https://unepazeconomyadlsstorage.blob.core.windows.net/public/unep_methanedata_detected_plumes_csv.zip>
- GeoJSON archive: <https://unepazeconomyadlsstorage.blob.core.windows.net/public/unep_methanedata_detected_plumes_geojson.zip>
- CSV archive: 9,735,367 bytes; SHA-256 `431c1e979554f2771a3e588b1815d66a79c614ba210111d42b1d22e21d3cbfee`
- GeoJSON archive: 49,797,154 bytes; SHA-256 `cc539e494671727ad27ff629bddd4cbe1a60dd25b237cdad84eb510cb0f933f3`

## Audited result

The 27,403-row catalog yielded 269 exact-product positive samples from 43 physical groups and 88 sources: 178 Sentinel-2 and 91 Landsat. Every eligible sample remains outside the official paper-test exclusion; the minimum distance to an official test location is 25.631477 km.

| Frozen role | July catalog | Current catalog | Change |
|---|---:|---:|---:|
| Auxiliary training | 215 | 216 | +1 |
| Development | 9 | 40 | +31 |
| Sealed external | 13 | 13 | 0 |
| Total | 237 | 269 | +32 |

## Research decision

This is a genuine catalog refresh but not a meaningful training-set expansion: 31 of 32 additions fall in the already frozen development role. Do not revise roles after seeing these counts. The new development rows may support a future independently frozen confirmation only after a candidate passes the existing MARS fold-3/fold-4 gates.
