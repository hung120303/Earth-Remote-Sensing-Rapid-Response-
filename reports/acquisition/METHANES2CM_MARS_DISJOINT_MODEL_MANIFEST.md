# MethaneS2CM / MARS spatial-disjoint auxiliary cohort

Generated: 2026-08-01T00:56:29.486512+00:00.

- MARS v3 paper-test physical sites (coordinates only): **1,289**.
- MethaneS2CM source locations: **3,460** / **80,217** crops.
- Locations farther than 25 km from every MARS paper-test site: **1,347**.
- Auxiliary-training crops: **14,859** across **147** frozen 25 km groups.
- Held source-development crops: **11,478** across **48** frozen 25 km groups.
- Selected positive / no-plume crops: **13,277 / 13,060**.

The exclusion reads no MARS model targets or pixel labels. It uses only the published test split name, physical-location identifier, latitude, and longitude. Roles preserve MethaneS2CM's pre-existing 25 km group-held fitting/development boundary. The large loader manifests and packed HDF5 stay ignored; this tracked receipt binds them by SHA-256.
