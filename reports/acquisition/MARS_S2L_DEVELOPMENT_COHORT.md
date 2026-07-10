# MARS-S2L development cohort

- Samples: 1,731 (451 plume / 1,280 no plume)
- Groups: 272; locations: 486
- Assets: 4,364; exact size: 1,834,308,393 bytes (1.708 GiB)
- Sample-manifest SHA-256: `e59985b592a4bd1cf0561717c680116242266bf19b86069160a00debd72de3d3`
- Asset-catalog SHA-256: `03e023bcdfbabf08e985418d39ae7245d2218cfcd4d73670ed5b88319bb16ee3`

| Role | Label | Rows | Groups | Locations |
|---|---|---:|---:|---:|
| internal_training | NO_PLUME | 512 | 97 | 164 |
| internal_training | PLUME | 256 | 42 | 86 |
| internal_validation | NO_PLUME | 256 | 24 | 57 |
| internal_validation | PLUME | 128 | 8 | 34 |
| strict_spatial_test | NO_PLUME | 512 | 150 | 218 |
| strict_spatial_test | PLUME | 67 | 19 | 20 |

This is a group-diverse development tranche for baseline and pipeline iteration. It preserves the frozen internal train/validation and strict spatial test roles, but it is deliberately class-enriched and is not the paper's prevalence-representative final cohort.

Download and verify only this tranche with:

```bash
python tools/acquire_mars_cohort.py --catalog-file publication_dev_remote_catalog.jsonl
python tools/acquire_mars_cohort.py --catalog-file publication_dev_remote_catalog.jsonl --verify-only
```
