# UNEP MARS post-2024 catalog audit

Generated: 2026-07-16T02:34:42.285556+00:00.

## Result

- Catalog plume rows: **26,762**.
- Eligible exact-product plume records: **237**.
- Source-crop samples after merging: **237**.
- Independent 25 km groups: **42**.
- Pixel-truth samples with polygon geometry: **237**.
- Sentinel-2 / Landsat samples: **153 / 84**.

## Fixed roles

- auxiliary_training: **215** samples.
- development: **9** samples.
- sealed_external: **13** samples.

## Sequential exclusions

- before_cutoff: 13,525
- disallowed_satellite: 9,246
- missing_required_field: 76
- not_expert_actionable: 929
- within_paper_test_exclusion: 2,749

## Integrity

- Pinned paper test SHA-256: `add125547e0e0066216070ed61a8544e76e84f062f636390be5d2ef1808dbfaa`.
- Minimum accepted distance from a paper-test location: **25.631 km**.
- Eligible manifest SHA-256: `13a19dd7daad36b40261f7acb0332b0418cb37b1b9407d708e864b3847976e74`.
- No catalog absence was interpreted as a no-plume label.
- The sealed-external role remains positive-only and must stay unread during model selection.
- Bulk catalogs and the row-level manifest remain under ignored `.research/` storage.

Source: UNEP IMEO Eye on Methane MARS sources and plumes, CC BY-NC-SA 4.0 (non-commercial).
