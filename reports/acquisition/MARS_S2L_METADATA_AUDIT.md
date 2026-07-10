# MARS-S2L pinned metadata audit

- Source: `UNEP-IMEO/MARS-S2L`
- Revision: `c26b1d7e31a0c5241fa37c9140802622c215eb32`
- Generated: `2026-07-10T18:52:20.258520+00:00`
- Metadata files: 7 / 188,857,049 bytes verified

## Official split summary

| Split | Rows | Plume | No plume | Plume % | Locations | S2 L1C | Recommended S2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 38,345 | 3,512 | 34,833 | 9.16% | 618 | 31,662 | 29,708 |
| val | 6,018 | 299 | 5,719 | 4.97% | 89 | 5,756 | 5,527 |
| test | 43,524 | 1,832 | 41,692 | 4.21% | 1,289 | 25,438 | 21,317 |

The official split union contains 87,887 unique image-location items. The full metadata table contains 93,538; 5,651 rows are outside the released train/validation/test CSVs and must not be silently added.

## What the metadata proves

- Explicit real negatives: 82,244 official-split rows.
- Positive images: 5,643; validated plume table: 5,964 plume records on 5,855 images.
- Plume-table linkage is incomplete for 197 official positive images; keep those out of object/flux analyses until reconciled.
- Sentinel-2 inputs are MSI L1C; Landsat Collection-2 L1 products are a separate domain.
- Exact acquisition-scene overlap across official splits is zero.
- Physical location overlap is not zero: train/validation 89, train/test 592, validation/test 84.
- Test-only physical locations: 697; these contain 15,655 rows (229 plume, 15,426 no plume).
- A 25 km location graph yields 318 components; 135 cross official splits.

## Architecture and evaluation consequences

1. Start with the Sentinel-2 L1C target/reference cohort only; do not mix Landsat or the existing L2A pilot.
2. Preserve the official test set for published comparability, but report the test-only-location subset as the primary geographic-transfer result.
3. Rebuild group IDs as connected components of physical location and 25 km proximity before any ERSRR cross-validation.
4. Train on reviewed negatives and calibrate no-plume thresholds on validation only.
5. Exclude non-clear, low-clear, or missing-reference rows from the first model; evaluate them later as observability/abstention cases.
6. Download only assets referenced by the selected Sentinel-2 cohort rather than mirroring the full mixed-sensor repository.

## Integrity findings

- Duplicate item IDs in official splits: 0
- Exact scenes crossing official splits: 0
- Official items missing from the full table: 0
- Positive asset-contract violations: 0
- Negative rows carrying positive assets: 0
- Full-table positive images missing plume-table records: 199

The metadata gate passes for a selective Sentinel-2 asset import. It does not authorize a model claim; image-band and raster-grid contracts must be verified on downloaded assets next.
