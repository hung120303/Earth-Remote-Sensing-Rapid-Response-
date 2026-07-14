# MARS-S2L paper-v3 mixed-sensor cohort

- Remote assets: 187,014 / 82.909 GiB
- Manifest SHA-256: `52e4c9220a11e6b8b09d2204988583ef7fb3764e16ea48d23fbf31ed992a14bf`
- Catalog SHA-256: `363616a38f8e78e89a7b0f355d5697d9b07a1c27ead6eac687e22175ebeec292`

| Research role | Rows | Plume | No plume | Sites | Sentinel-2 | Landsat |
|---|---:|---:|---:|---:|---:|---:|
| development_training | 38,345 | 3,512 | 34,833 | 618 | 31,662 | 6,683 |
| development_validation | 6,018 | 299 | 5,719 | 89 | 5,756 | 262 |
| sealed_paper_test | 43,524 | 1,811 | 41,713 | 1,289 | 25,438 | 18,086 |

The exact paper archive covers 43,529 test scenes. Current released raster paths remain available for 43,524; the 5 unavailable historical scenes are preserved in the benchmark lock and scored adversarially for the candidate. Paper-era targets override later public labels on sealed test scenes.

Development code may read only training and validation roles. The sealed paper-test role is downloaded for eventual one-shot inference but cannot be used for architecture, checkpoint, calibration, threshold, or postprocessing selection.

The released checkpoint config names an internal `az://imeodata/models/data/marss2l_20250302/validated_images_all.csv` snapshot (38,366 training / 5,921 validation rows), while paper Table S3 reports 38,366 / 6,034. The private snapshot is not in the public release, whose pinned files contain 38,345 / 6,018. Successor development uses those complete public splits, and the exact archived paper test remains the comparison target.

Acquisition: `python tools/acquire_mars_cohort.py --catalog-file paper_v3_mixed_remote_catalog.jsonl --workers 8 --receipt reports/acquisition/mars_s2l_paper_v3_mixed_download.json`.
