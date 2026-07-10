# ERSRR data acquisition contract

ERSRR needs paired Sentinel-2 imagery and methane-plume labels. Acquisition must preserve source product, time, grid, validity, and provenance; a plume polygon must never be presented as a concentration raster.

## Supported sources

### MARS-S2L primary training corpus

- Public dataset: <https://huggingface.co/datasets/UNEP-IMEO/MARS-S2L>
- Companion implementation: <https://github.com/UNEP-IMEO-MARS/marss2l>
- Pinned revision: `c26b1d7e31a0c5241fa37c9140802622c215eb32`
- License: CC BY-NC-SA 4.0

Use MARS-S2L as the primary source of real plume positives and reviewed no-plume examples. Keep
its Sentinel-2 MSI L1C domain separate from ERSRR's Element 84 L2A pilot and legacy Earth Engine
L1C data. All raw MARS files and local manifests belong under this ignored path:

```text
EarthRemoteSensingRapidResponse/Data Collection/s2_emit_pairs/
  publication-v1/external/MARS-S2L/
```

From the repository root, acquire and audit the 188,857,049-byte metadata tranche:

```bash
python tools/acquire_mars_metadata.py
python tools/acquire_mars_metadata.py --verify-only
python tools/audit_mars_metadata.py
```

Acquire or verify the deterministic 18-sample contract pilot:

```bash
python tools/acquire_mars_pilot.py
python tools/acquire_mars_pilot.py --verify-only
```

The raster audit requires the repository's Linux environment because it includes Rasterio:

```bash
# Run from the repository root inside WSL:
.venv/bin/python tools/audit_mars_pilot.py
```

The verified native S2 contract is a 200 x 200, 10 m, 12-band `uint16` stack. Target and
background each contain `B02,B03,B04,B08,B11,B12`. Positive examples add a binary plume mask and
a `float64` enhancement raster; negatives intentionally contain only the image and cloud mask.
Enhancement units are unresolved: populated TIFF descriptions say `DeltaCH4(ppm)`, while the
pinned MARS-S2L README says ppb. Preserve the raw values but do not train a unit-bearing regression
or make flux claims until UNEP-IMEO reconciles the product metadata. Ancillary descriptions are
not universally populated, and cloud nodata can overlap the clear class, so resolve semantic roles
from the pinned manifest and interpret cloud classes explicitly.

The larger development tranche additionally proves that some cloud rasters contain encoded value
`4`, consistent with a nodata/invalid class omitted from the README's documented 0-3 table. The
adapter treats only `0` as clear; values `1-3` are cloud/shadow and `4` is invalid/unobservable.

Do not download the full mixed-sensor repository by default. The proposed first cohort is limited
to official-split Sentinel-2 L1C rows with `observability=clear`, at least 80% clear coverage, and a
background reference: 56,552 samples (3,826 plume / 52,726 no plume). Generate and freeze its
asset manifest and total byte estimate before authorizing the large transfer.

The frozen inventory resolves 120,756 unique assets totaling exactly 58,455,597,233 bytes
(58.456 GB / 54.441 GiB). It can be rebuilt from the pinned API or verified entirely from the
ignored local catalog:

```bash
python tools/build_mars_cohort.py
python tools/build_mars_cohort.py --offline
```

No Hugging Face account is required. The authoritative download base is
<https://huggingface.co/datasets/UNEP-IMEO/MARS-S2L>; every asset URL is pinned and recorded in the
ignored `publication_s2_cohort.jsonl`. Do not begin the 54.441 GiB transfer until storage approval.

After approval, use the resumable downloader and independently verify the finished corpus:

```bash
python tools/acquire_mars_cohort.py --dry-run
python tools/acquire_mars_cohort.py
python tools/acquire_mars_cohort.py --verify-only
```

The downloader keeps `.part` files for interrupted transfers, checks a 5 GiB free-space reserve,
and verifies LFS SHA-256 or Git blob SHA-1 identities from the frozen catalog. `--max-assets N` is
only for downloader smoke tests and must never be treated as a complete research cohort.

For ERSRR v3, prefer the smaller frozen fit+validation transfer rather than the all-split catalog.
It retains image, cloud-mask, and positive plume-mask assets, but excludes methane-enhancement
rasters unused by the detector loss. It contains 29,708 group-disjoint fit/validation samples and
61,928 assets totaling exactly 30,366,803,325 bytes (28.281 GiB). The verified development tranche
already supplies 2,688 assets / 1,167,947,077 bytes, so the remaining transfer is exactly
29,198,856,248 bytes (27.193 GiB). No Hugging Face account is required:

```bash
python tools/build_mars_v3_training_cohort.py
python tools/acquire_mars_cohort.py --catalog-file publication_v3_training_remote_catalog.jsonl --dry-run
python tools/acquire_mars_cohort.py --catalog-file publication_v3_training_remote_catalog.jsonl --receipt reports/acquisition/mars_s2l_v3_training_download.json
python tools/acquire_mars_cohort.py --catalog-file publication_v3_training_remote_catalog.jsonl --verify-only --receipt reports/acquisition/mars_s2l_v3_training_download.json
```

The exact destination remains
`EarthRemoteSensingRapidResponse/Data Collection/s2_emit_pairs/publication-v1/external/MARS-S2L/`.
The large manifests and rasters remain ignored; commit only the compact cohort report and final
verification receipt.

Freeze or verify the publication evaluation roles before training:

```bash
python tools/build_mars_protocol.py
python tools/build_mars_protocol.py --verify-only
```

The machine-readable specification is `configs/mars_publication_protocol.json`; detailed ignored
assignments are `publication_protocol_assignments.jsonl` under the MARS directory. The internal
train/validation split has zero 25 km group overlap. The primary strict spatial test contains
4,401 official-test rows (67 plume / 4,334 no plume) across 150 groups that have zero overlap with
official training. Released validation and overlapping official test rows are comparability-only.

## Verified development tranche

For architecture and baseline iteration before the full 54.441 GiB transfer, the deterministic
development tranche contains 1,731 scenes / 4,364 assets / exactly 1,834,308,393 bytes (1.708 GiB):

- internal training: 256 plume + 512 no plume;
- group-disjoint internal validation: 128 plume + 256 no plume;
- strict spatial benchmark: all 67 available clear-S2 plume rows + 512 group-diverse no-plume rows.

The sample-manifest identity is
`e59985b592a4bd1cf0561717c680116242266bf19b86069160a00debd72de3d3`; the asset-catalog identity
is `03e023bcdfbabf08e985418d39ae7245d2218cfcd4d73670ed5b88319bb16ee3`. All 4,364 assets have
been independently verified; the compact receipt is
`reports/acquisition/mars_s2l_development_download.json`.

Rebuild and verify:

```bash
python tools/build_mars_dev_cohort.py
python tools/build_mars_dev_cohort.py --verify-only
python tools/acquire_mars_cohort.py --catalog-file publication_dev_remote_catalog.jsonl --verify-only
```

The development tranche is deliberately class-enriched. It is suitable for pipeline and baseline
screening, not a prevalence or final-paper estimate.

### EMIT methane labels

- **EMIT L2B CH4PLM V002** is the preferred source for new work: <https://www.earthdata.nasa.gov/data/catalog/lpcloud-emitl2bch4plm-002>
- DOI: <https://doi.org/10.5067/EMIT/EMITL2BCH4PLM.002>
- Direct Earthdata Search collection: <https://search.earthdata.nasa.gov/search/granules?p=C3242707413-LPCLOUD>
- NASA CMR UMM metadata is public and contains the authoritative V002 plume footprint.
- The per-pixel concentration COG and companion metadata are protected by Earthdata Login. Do not automate credentials into the repository.

Google Earth Engine currently exposes EMIT CH4PLM V001, not V002: <https://developers.google.com/earth-engine/datasets/catalog/NASA_EMIT_L2B_CH4PLM>. Keep V001 and V002 provenance explicit.

### Sentinel-2 inputs

- New pilot collection uses public Element 84 `sentinel-2-l2a` COGs (surface reflectance).
- The legacy curated ERSRR tiles came from `COPERNICUS/S2_HARMONIZED` (L1C/TOA): <https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_HARMONIZED>
- Future Earth Engine L2A work should use `COPERNICUS/S2_SR_HARMONIZED`: <https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED>

Never silently combine L1C/TOA and L2A/surface-reflectance values. Either train separate artifacts or implement and validate an explicit domain-harmonization step.

## Reproducible public V002 pilot

From the repository root:

```bash
python tools/acquire_v002_pilot.py \
  EMIT_L2B_CH4PLM_002_20250922T204933_003374 \
  --batch emit-v002-2026-07 \
  --window-days 14 \
  --max-cloud 20 \
  --temporal-mode bracketing
```

The collector:

1. resolves the exact V002 granule through NASA CMR collection `C3242707413-LPCLOUD`;
2. saves public UMM metadata, the physical plume GeoJSON, and the public browse image;
3. deterministically ranks Sentinel-2 L2A scenes by absolute time offset, scene cloud cover, and scene ID;
4. selects the closest before/after scenes when both exist;
5. writes co-registered 256 x 256, 20 m stacks in canonical `B2, B3, B4, B11, B12` order;
6. rasterizes the same V002 plume footprint on the shared before/after grid;
7. records cloud/clear percentages, timestamps, offsets, CRS, transform, source URLs, and SHA-256 hashes in `manifest.json`.

Outputs are local and ignored:

- `EarthRemoteSensingRapidResponse/Data Collection/EMIT_Plumes/<batch>/`
- `EarthRemoteSensingRapidResponse/Data Collection/s2_emit_pairs/<batch>/<granule>/`

The July 2026 research batch contains 12 V002 granules, 24 L2A image stacks, and 24 masks (11.31 MB). A containment audit found six plume polygons clipped by the fixed 5.12 km frame; four of those also occupied more than half of the tile. All six clipped groups are excluded, leaving only six independent groups for a feasibility benchmark. This batch is too small and too positive-heavy for architecture or operational accuracy claims.

## Label contracts

### Physical polygon-mask contract

- task: binary plume segmentation;
- label: public NASA CMR V002 plume footprint;
- values: `0` background and `1` plume;
- before/after masks may be identical because the label is tied to the EMIT observation, not the Sentinel-2 acquisition;
- must not be inserted as band 6 of the legacy concentration dataset.

### Concentration-raster contract

- task: methane enhancement regression or threshold-derived segmentation;
- label: authenticated EMIT concentration COG with original units and nodata mask;
- target validity and physical calibration must be independent of the prediction;
- protected data may be stored locally but must not be committed.

## Quality gates before promotion

A batch can move into a curated dataset only after it has:

- an immutable manifest with hashes and product level;
- a valid-mask definition and corrected nodata metadata;
- a bounded temporal gap (the current legacy research filter is `|gap| <= 7 days`);
- a documented cloud/clear threshold;
- geographic/source group IDs for leakage-free splitting;
- true negative/background examples from comparable geography and season;
- inspection for duplicate scenes, implausibly broad masks, and train/validation group overlap.
- proof that the full physical plume geometry is contained in the raster or a documented tiling strategy that preserves it.

Run the audit after any curated-dataset change:

```bash
python tools/ersrr.py status
python tools/ersrr.py audit
```

## Immediate acquisition priorities

1. Freeze and byte-estimate the selective 56,552-item MARS-S2L S2 cohort before its large transfer.
2. Obtain authenticated V002 concentration COGs through a user-managed Earthdata session and preserve their nodata/unit metadata.
3. Use MARS reviewed negatives for training, then mine hard negatives by geography, season, surface type, and cloud regime.
4. Grow a geographically/source-disjoint held-out validation set to at least 50 independent groups.
5. Collect tighter-time L2A pairs and keep before/after stacks on one grid.
6. Consider synthetic plume injection only as augmentation; retain a real-data-only test set.
