# MARS-S2L minimum v3 training transfer

Frozen image/cloud/plume-mask corpus for clean ERSRR v3 training and validation. Methane-enhancement rasters are intentionally excluded because the detector losses do not consume them.

- Samples: 29,708 (23,763 fit / 5,945 validation)
- Positives: 2,512; reviewed negatives: 27,196
- Assets: 61,928; exact size: 30,366,803,325 bytes (28.281 GiB)
- Already verified in the development tranche: 2,688 assets / 1,167,947,077 bytes
- Remaining transfer: 59,240 assets / 29,198,856,248 bytes (27.194 GiB)
- Sample manifest SHA-256: `edd3c1da3d706b109798dcf86bc3db284f639d401d4953f3ced175c27b27d566`
- Asset catalog SHA-256: `3183e126512e660bf2279d25ec974efec36acdd7708c7affd1380aaf26aba9a1`

## Acquire

No Hugging Face account or token is required. From the repository root:

```powershell
python tools/acquire_mars_cohort.py --catalog-file publication_v3_training_remote_catalog.jsonl --dry-run
python tools/acquire_mars_cohort.py --catalog-file publication_v3_training_remote_catalog.jsonl --receipt reports/acquisition/mars_s2l_v3_training_download.json
python tools/acquire_mars_cohort.py --catalog-file publication_v3_training_remote_catalog.jsonl --verify-only --receipt reports/acquisition/mars_s2l_v3_training_download.json
```

Raw files remain under `EarthRemoteSensingRapidResponse/Data Collection/s2_emit_pairs/publication-v1/external/MARS-S2L` and are ignored by Git.
