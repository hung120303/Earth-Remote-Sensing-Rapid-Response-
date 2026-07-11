# MARS-S2L full strict-spatial v3 evaluation transfer

Frozen prevalence-representative strict-spatial evaluation corpus. It includes every official-test scene whose 25 km group contains no official-training scene; methane-enhancement rasters are excluded because detection evaluation does not consume them.

- Samples: 4,401 (67 plume / 4,334 reviewed no-plume)
- Frozen 25 km groups: 150
- Assets: 8,869; exact size: 4,489,575,260 bytes (4.181 GiB)
- Already verified in the development tranche: 1,225 assets / 577,629,770 bytes
- Remaining transfer: 7,644 assets / 3,911,945,490 bytes (3.643 GiB)
- Sample manifest SHA-256: `6e959ae0af50c5a309247cbe674fe01b31a36f8b149f9be3292acebed3e5f906`
- Asset catalog SHA-256: `ad8be1441d5a7a0585bc4c8b66305cc53524f48d3837137e92e27881abd3ef28`

```powershell
python tools/acquire_mars_cohort.py --catalog-file publication_v3_strict_remote_catalog.jsonl --dry-run
python tools/acquire_mars_cohort.py --catalog-file publication_v3_strict_remote_catalog.jsonl --receipt reports/acquisition/mars_s2l_v3_strict_download.json
python tools/acquire_mars_cohort.py --catalog-file publication_v3_strict_remote_catalog.jsonl --verify-only --receipt reports/acquisition/mars_s2l_v3_strict_download.json
```

Raw files remain under `EarthRemoteSensingRapidResponse/Data Collection/s2_emit_pairs/publication-v1/external/MARS-S2L` and are ignored by Git.
