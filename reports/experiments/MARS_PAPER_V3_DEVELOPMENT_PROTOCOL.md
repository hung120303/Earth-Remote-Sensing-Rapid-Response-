# MARS paper-v3 successor development protocol

- Development scenes: 44,363
- Physical-location groups: 618
- Assignment SHA-256: `013f784a28eaa88e20ee62ae8cb232c8d0c3e7c6265f1193a9b5e524fc264363`
- Development manifest SHA-256: `31ba92e791ba07be781dd700ff1e720b8cd686357b9bec38ebfe41bbaa207e8e`

| Fold | Sites | Scenes | Plume | No plume | Sentinel-2 | Landsat |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 107 | 8,987 | 745 | 8,242 | 7,599 | 1,388 |
| 1 | 108 | 8,798 | 745 | 8,053 | 7,412 | 1,386 |
| 2 | 153 | 8,833 | 797 | 8,036 | 7,444 | 1,389 |
| 3 | 146 | 8,799 | 758 | 8,041 | 7,403 | 1,396 |
| 4 | 104 | 8,946 | 766 | 8,180 | 7,560 | 1,386 |

Architecture work uses site-held fold 0 and must confirm on fold 1. The five-model campaign then produces exactly one out-of-fold prediction per development site. Calibration and thresholds are cross-fitted across those predictions.

The sealed paper-test manifest is not read by this script. It may be opened once only after the complete candidate artifact and evaluator hashes are frozen.
