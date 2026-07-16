# UNEP MARS post-2024 model manifest

Generated: 2026-07-16T03:48:30.037826+00:00.

## Result

- Loader-compatible exact Sentinel-2 positives: **139**.
- Auxiliary training: **135** rows across **27** groups.
- Development confirmation: **4** rows across **4** groups.
- Sealed-external crop directories and assets accessed: **0**.

## Compatibility contract

- Image, polygon mask, and CloudSEN12 sidecars are byte- and SHA-256-verified before inclusion.
- The released twelve-band ordering and reflectance scaling are unchanged.
- Catalog flux, wind, and polygon geometry are not model features. Wind channels are explicitly zero-filled only to preserve the released 16-channel shape.
- The output manifests remain ignored bulk metadata; this report records their identities.
