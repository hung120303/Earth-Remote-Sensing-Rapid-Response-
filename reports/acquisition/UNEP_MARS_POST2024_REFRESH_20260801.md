# UNEP MARS post-2024 catalog refresh — 2026-08-01

The current official UNEP/IMEO detected-plume catalog was downloaded and audited with the already frozen exact-product and paper-test spatial-exclusion rules. Bulk archives, imagery, masks, and manifests remain ignored under `.research/unep_mars_post2024_refresh_20260801/`.

## Provenance

- Official catalog page: <https://data.unep.org/methane>
- CSV archive: 9,735,367 bytes; SHA-256 `431c1e979554f2771a3e588b1815d66a79c614ba210111d42b1d22e21d3cbfee`
- GeoJSON archive: 49,797,154 bytes; SHA-256 `cc539e494671727ad27ff629bddd4cbe1a60dd25b237cdad84eb510cb0f933f3`
- The 27,403-row catalog yields 269 exact-product positive samples from 43 physical groups: 178 Sentinel-2 and 91 Landsat.
- Every eligible location remains at least 25.631477 km from the official paper-test locations. No official-test imagery or labels were accessed.

## Append-only role correction

The first raw refresh recomputed connected-component hashes from scratch. Catalog growth changed 36 old group identifiers and would have moved 18 already frozen July identities from auxiliary training into development. Those raw 216/40/13 counts are superseded.

The append-only reconciler treats the July manifest as immutable: an expanded component inherits its one prior group and role, a new component receives a prospective role, and a merge involving multiple prior groups is quarantined. The actual refresh has zero prior identity changes and zero quarantines.

| Frozen role | July catalog | Reconciled current catalog | Change |
|---|---:|---:|---:|
| Auxiliary training | 215 | 247 | +32 |
| Development | 9 | 9 | 0 |
| Sealed external | 13 | 13 | 0 |
| Total | 237 | 269 | +32 |

Reconciled manifest SHA-256: `82a1f5ff0b20b15127fa987b67e9a08df83588ca577fa4de21763ea5d371a663`.

## Exact acquisition result

After removing the 135 identities already present in the model-ready auxiliary manifest, 52 new resolved rows remained: 26 Sentinel-2 and 26 Landsat across 15 groups.

- All 26 exact Sentinel-2 L1C crops were acquired, totaling 16,954,844 raster bytes.
- The frozen CloudSEN12 observability pass retained 25 scenes and rejected one cloudy scene, with zero processing errors.
- The resulting combined auxiliary model manifest contains 160 real plume scenes across 28 physical groups, up from 135 scenes across 27 groups. Its SHA-256 is `8c42b4350ccc0abbd2fec727abba435fca2af8f29289e837d52e7151553d861b`.
- All 26 exact Landsat Collection 2 Level-1 attempts reached LandsatLook but were redirected to USGS EROS authentication. They remain pending; no L2, TOA, or alternate-product substitution was made.

## Research decision

Use the 25 newly verified Sentinel-2 auxiliary positives in the next development-only experiment. Preserve the 26 Landsat identities as pending exact acquisition, keep all bulk pixels ignored, and never revise frozen roles based on model outcomes.
