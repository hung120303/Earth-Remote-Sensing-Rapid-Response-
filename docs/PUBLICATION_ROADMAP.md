# ERSRR publication roadmap: reliable plume detection and no-plume rejection

- Status: execution active; frozen MARS-S2L protocol, baselines, and joint MIL v2 recorded
- Date: 2026-07-10
- Starting revision: `3cab591405cf59287a76249c12e08ca4e82f6855`
- Current artifact: `research_only`; it is an engineering baseline, not the paper model

## Executive decision

The project should stop optimizing the existing positive-only legacy cohort. Its calibrated
artifact has specificity `0.00127`, so it cannot make a credible no-plume decision. The next
research system should be built around:

1. the public MARS-S2L dataset as the primary real Sentinel-2/Landsat training and benchmark
   corpus;
2. the current compact ERSRR model, MBMP, CH4Net, and the released MARS-S2L model as explicit
   baselines;
3. a dual-temporal, physics-guided detector with separate scene-presence and pixel-segmentation
   heads;
4. hard-negative mining and validation-only risk calibration;
5. an independent EMIT V002 concentration/sensitivity/uncertainty cohort for external validation;
6. a three-state prediction contract: `PLUME`, `NO_PLUME`, or
   `UNOBSERVABLE_OR_UNCERTAIN`.

The paper should not claim that lack of a catalog entry proves methane absence. `NO_PLUME` is
allowed only when the scene is observable, passes the quality contract, and has a reviewed
negative label. Cloudy, invalid, temporally ambiguous, or low-sensitivity scenes must abstain.

## Proposed paper

Working title:

> Coverage-aware negative sampling and risk-controlled physics-guided learning for methane
> plume detection in Sentinel-2 imagery

Primary research question:

> Can a dual-temporal detector trained with real hard negatives reduce false alarms at matched
> plume recall, while retaining pixel-level plume delineation and generalizing to geographically
> and sensor-disjoint observations?

Planned contributions:

1. A reproducible label contract that separates plume, reviewed no-plume, uncertain, and
   unobservable scenes.
2. A source- and geography-disjoint benchmark with explicit negative examples and a frozen
   external EMIT V002 evaluation cohort.
3. A physics-guided target/reference model with scene-presence, segmentation, and observability
   outputs.
4. A predeclared operating point and selective prediction rule that controls false positives
   instead of maximizing F1 after seeing the test set.
5. A full error analysis by region, surface type, cloud regime, plume size, temporal gap, and
   source-site novelty.

Atmospheric Measurement Techniques (AMT) is the primary venue because the paper is a retrieval,
validation, and data-processing method. Remote Sensing of Environment is a stretch venue if the
external validation and generalization results are unusually strong. AMT requires a data/code
availability section and recommends DOI-backed FAIR repositories for the code and evidence:
<https://www.atmospheric-measurement-techniques.net/submission.html>.

## Current evidence and why the plan changes

- The legacy ERSRR data contains 102 positive-centered tiles and no verified true negatives.
- Only 65 legacy scenes in 32 connected split groups pass the current validity/time-gap filter.
- Raw logistic regression outranks the compact raw ResUNet on both tested legacy thresholds.
- The packaged ResUNet predicts almost everywhere positive at its calibrated threshold.
- The public V002 pilot proves the acquisition path but leaves only six complete plume groups.
- Non-contemporaneous EMIT polygons cannot be copied onto nearby Sentinel-2 dates and treated as
  time-correct plume truth.
- The native MARS adapter smoke test runs end to end on 18 balanced samples, but validation-tuned
  MBMP and pixel-logistic rules each produced two false alarms among three test negatives. This
  pilot is too small for an accuracy estimate, yet it confirms that a segmentation threshold alone
  is not a credible no-plume decision rule.
- A verified 1,731-scene development tranche now exercises 768 internal-training, 384
  group-disjoint validation, and 579 strict-spatial benchmark scenes. At validation FPR <= 0.05,
  MBMP, raw logistic, physics logistic, and physics gradient boosting achieved only 0.8%-14.1%
  validation recall. The validation-selected raw logistic reached 7.5% recall / 96.3% specificity
  on the spatial benchmark. Aggregate scene statistics are therefore inadequate; the next model
  must learn spatial plume morphology jointly with presence and observability.
- The spatial baseline closes the remaining classical loophole. Validation-selected pixel
  logistic achieved 3.9% recall at 3.1% FPR and then 0% recall on the spatial benchmark; MBMP
  achieved 0.8% validation recall and 1.5% benchmark recall. Both have effectively zero pixel IoU.
  Connected-component filtering cannot rescue a classifier without a learned spatial receptive
  field.
- Joint model v1 verifies that learned spatial context helps masks but not yet operational scene
  decisions. It raises strict-spatial pixel AP to 0.0355 and IoU to 0.0517, but its
  validation-selected presence rule transfers at only 1.5% recall / 91.4% specificity. The best
  checkpoint occurs at epoch 1 because global-average scene pooling quickly overfits. The next
  frozen change was therefore top-k multiple-instance plume evidence plus stronger presence loss,
  not a wider/deeper encoder.
- Joint MIL v2 validates that architecture call. It reaches 39.8% validation recall at 4.7% FPR,
  versus 4.7% recall for v1 at a comparable validation constraint. With its checkpoint and
  thresholds frozen before test evaluation, it reaches 14.9% strict-spatial recall, 98.8%
  specificity, 0.752 AUROC, and 0.0611 pixel AP. This is a 10x recall improvement over v1 while
  reducing v1's strict FPR from 8.6% to 1.2%. However, the group-bootstrap recall 95% interval is
  only 4.4%-35.0%, so the promotion gate still fails. The next decision must come from the released
  MARS-S2L baseline and validation-only error analysis, not from further inspection of this frozen
  benchmark.
- The pinned released MARS-S2L checkpoint establishes the credible scale baseline. Using the
  authors' fixed >0.5 / 100-pixel connected-component rule without ERSRR recalibration, it reaches
  64.2% recall, 92.2% specificity, 0.822 AUROC, 0.494 pixel AP, and 0.530 pixel Dice on the same 579
  strict-spatial scenes. The group-bootstrap recall interval is 48.0%-86.1% and FPR is 7.8%, so it
  also fails the ERSRR research gate. It nevertheless outperforms MIL v2 decisively and shows that
  the next candidate must inherit full-resolution U-Net capacity and substantially broader
  training data while adding a separately calibrated high-specificity presence/abstention head.
- The released single-date CH4Net checkpoint confirms that the gain is not generic U-Net capacity:
  on the same strict cohort it reaches only 16.4% recall, 91.2% specificity, 0.597 AUROC, and 0.0069
  pixel AP. Target/reference pairing, MBMP/wind/cloud context, and the MARS data program are therefore
  essential ablations. Phase 2 baseline reproduction is complete.
- The internal-validation-only MIL-v2 audit explains why another pooling adjustment is not the
  answer. Presence score is coupled to segmentation top-1% confidence at Spearman rho=0.909. The
  smallest plume quartile reaches only 9.4% presence recall even though the mask rule proposes a
  component in 59.4% of those scenes; median true-positive plume area is 1,515 pixels versus 643
  for false negatives. All 12 false positives have saturated mask proposals, and six occur in one
  Kazakhstan stratum. MBMP top-1% strength is higher for false positives than for ordinary true
  negatives but is nearly identical between true and missed plumes. The next head must classify
  connected proposals using morphology, multiscale context, wind alignment, and geography-balanced
  hard negatives rather than treating top-k mask confidence or MBMP magnitude as presence.
- V3 is implemented as a 14,268,915-parameter full-resolution GroupNorm U-Net trained from scratch,
  with proposal descriptors combining deep global context, decoder features at high-evidence
  pixels through a learned 16-channel component embedding, soft area/centroid/covariance/compactness,
  and along-/cross-wind variance. Positive
  presence loss upweights small plumes by valid mask area. A 64-train/32-validation, group-disjoint
  one-epoch smoke run completed the enhancement-free adapter, augmentation-aware wind rotation,
  GPU training, checkpoint, and validation-rule pipeline; it explicitly did not load the strict
  benchmark and its metrics are not accuracy evidence. Full v3 training now depends on the frozen
  27.193 GiB remaining fit/validation transfer.

The current shared core and artifact contract remain useful engineering infrastructure. They do
not establish a useful detector.

## Data program

### 1. Primary corpus: MARS-S2L

Authoritative dataset:
<https://huggingface.co/datasets/UNEP-IMEO/MARS-S2L>

Companion code:
<https://github.com/UNEP-IMEO-MARS/marss2l>

Reference paper:
<https://arxiv.org/abs/2511.21777>

The dataset card reports approximately 87,000 target/background Sentinel-2 and Landsat image
pairs, more than 5,600 manually verified plumes, explicit cloud masks, plume masks and methane
enhancement for positive cases, and about 100 GB total storage. It includes official train,
validation, and test splits. The license is CC BY-NC-SA 4.0. As checked on 2026-07-10, the
repository is public and ungated.

The pinned local audit now provides exact counts. The official split union contains 87,887 image
items: 5,643 plume positives and 82,244 reviewed negatives. A first S2-only, clear,
`percentage_clear >= 80`, background-present cohort contains 56,552 items (3,826 positive and
52,726 negative). It has 29,708 train, 5,527 validation, and 21,317 test items. The official
splits have no exact scene overlap, but they do share physical locations: 89 train/validation,
592 train/test, and 84 validation/test. The final protocol is stricter than physical-site novelty:
the primary geographic-transfer result uses only official-test rows whose connected 25 km group
contains no official-train row. The test-only-location view is secondary, while the full official
test is retained for released-benchmark comparison.
See `reports/acquisition/MARS_S2L_METADATA_AUDIT.md` for the reproducible evidence.

Pin the initial import to repository revision:

```text
c26b1d7e31a0c5241fa37c9140802622c215eb32
```

Local destination, already beneath an ignored acquisition root:

```text
EarthRemoteSensingRapidResponse/Data Collection/s2_emit_pairs/
  publication-v1/external/MARS-S2L/
```

Download and verify only the pinned metadata first (188,857,049 bytes):

```bash
python tools/acquire_mars_metadata.py
python tools/acquire_mars_metadata.py --verify-only
python tools/audit_mars_metadata.py
```

The audit gate passed for selective S2 ingestion. Do not mirror the full mixed-sensor 100 GB
repository: the first paper experiment neither needs Landsat nor the 5,651 rows outside the
official splits. Before a large transfer, generate a cohort manifest from the criteria above,
freeze its sample IDs and asset URLs, estimate its byte size, obtain storage approval, and fetch
only its image/cloud-mask assets plus positive plume/enhancement assets.

That inventory is now frozen at 56,552 samples and 120,756 unique assets totaling exactly
58,455,597,233 bytes (58.456 GB / 54.441 GiB). Rebuild or verify it without downloading imagery:

```bash
python tools/build_mars_cohort.py
python tools/build_mars_cohort.py --offline
```

The detailed ignored manifest and catalog are stored as
`publication_s2_cohort.jsonl` and `publication_s2_remote_catalog.jsonl` under the local MARS path.
Their hashes and the compact evidence are in
`reports/acquisition/MARS_S2L_PUBLICATION_COHORT.md`.

After explicitly accepting the 54.441 GiB transfer, run the pinned resumable downloader from the
repository root:

```bash
python tools/acquire_mars_cohort.py --dry-run
python tools/acquire_mars_cohort.py
python tools/acquire_mars_cohort.py --verify-only
```

It verifies every LFS asset with the repository-declared SHA-256, verifies small assets with their
Git blob SHA-1, resumes `.part` files, rejects path escape, reserves 5 GiB of free space, and never
writes outside the ignored MARS acquisition directory. No Hugging Face account or token is needed.

The full all-split inventory is no longer the recommended next transfer. The validation error audit
shows that v3 first needs only the leakage-resistant internal fit and validation roles. A frozen
minimum v3 corpus excludes unused methane-enhancement rasters and contains 29,708 samples / 61,928
assets / exactly 30,366,803,325 bytes (28.281 GiB). The verified development tranche reuses 2,688
of those assets, leaving exactly 59,240 assets / 29,198,856,248 bytes (27.193 GiB) to download.
Build and acquire that exact catalog with:

```bash
python tools/build_mars_v3_training_cohort.py
python tools/acquire_mars_cohort.py --catalog-file publication_v3_training_remote_catalog.jsonl --dry-run
python tools/acquire_mars_cohort.py --catalog-file publication_v3_training_remote_catalog.jsonl --receipt reports/acquisition/mars_s2l_v3_training_download.json
python tools/acquire_mars_cohort.py --catalog-file publication_v3_training_remote_catalog.jsonl --verify-only --receipt reports/acquisition/mars_s2l_v3_training_download.json
```

The manifest/catalog SHA-256 values and zero group-overlap evidence are recorded in
`reports/acquisition/MARS_S2L_V3_TRAINING_COHORT.md`. Use the full 54.441 GiB catalog only later if
official-split comparability or additional test-side error analysis is explicitly required.

The deterministic 18-sample contract pilot is already available locally and can be independently
verified and audited with:

```bash
python tools/acquire_mars_pilot.py --verify-only
# Run from the repository root inside WSL:
.venv/bin/python tools/audit_mars_pilot.py
```

The pilot verifies 54 files / 19,835,687 bytes across 18 balanced samples with zero contract
violations. Its native image is a 200 x 200, 10 m, 12-band `uint16` stack: target and background
each contain `B02,B03,B04,B08,B11,B12`. Positive samples additionally contain a binary plume mask
and an enhancement raster; negatives intentionally omit both. The enhancement unit metadata is
internally inconsistent: populated TIFF descriptions say `DeltaCH4(ppm)`, while the pinned MARS
README says ppb. Preserve raw values but prohibit quantitative regression/flux claims until the
producer resolves the unit. Some ancillary rasters also lack band
descriptions, and cloud-mask nodata metadata can overlap the clear class, so the adapter must use
manifest roles and explicit mask values rather than GDAL validity masks. See
`reports/acquisition/MARS_S2L_CONTRACT_PILOT.md`.

Do not copy this corpus into the tracked `Dataset/` directory. Track only a derived manifest,
license record, pinned revision, checksums, split/group audit, and experiment summaries.

### 2. External validation: EMIT CH4ENH and CH4PLM V002

EMIT V002 enhancement is particularly valuable because it contains all captured scenes,
including scenes without a detected plume complex, and provides enhancement, uncertainty, and
sensitivity COGs. NASA describes the enhancement in `ppm m`, at nominal 60 m resolution:

- CH4ENH V002 collection: <https://search.earthdata.nasa.gov/search/granules?p=C3242680113-LPCLOUD>
- CH4PLM V002 collection: <https://search.earthdata.nasa.gov/search/granules?p=C3242707413-LPCLOUD>
- V002 product guide: <https://lpdaac.usgs.gov/documents/2250/EMIT_L2B_GHG_User_Guide_V2.pdf>
- Earthdata account registration: <https://urs.earthdata.nasa.gov/users/new>

Earthdata metadata and browse imagery are public, but the science COGs require a free Earthdata
Login. Credentials must never be placed in this repository or pasted into a task. Use a browser
session or user-managed Earthdata credential store.

Local destination:

```text
EarthRemoteSensingRapidResponse/Data Collection/EMIT_Plumes/publication-v1/
  CH4ENH/<enhancement-scene-id>/
  CH4PLM/<plume-complex-id>/
```

For each CH4ENH scene, download all four protected files:

```text
EMIT_L2B_CH4ENH_002_<timestamp>_<orbit>_<scene>.tif
EMIT_L2B_CH4UNCERT_002_<timestamp>_<orbit>_<scene>.tif
EMIT_L2B_CH4SENS_002_<timestamp>_<orbit>_<scene>.tif
EMIT_L2B_CH4ENH_002_<timestamp>_<orbit>_<scene>.cmr.json
```

For each CH4PLM complex, download the plume COG and companion plume metadata JSON. Preserve the
NASA filenames unchanged.

The first authenticated pilot should cover the 12 public plume complexes already collected and
their exact source enhancement scenes:

| CH4PLM V002 complex | Source CH4ENH V002 scene |
|---|---|
| `EMIT_L2B_CH4PLM_002_20241017T083946_003674` | `EMIT_L2B_CH4ENH_002_20241017T083946_2429106_009` |
| `EMIT_L2B_CH4PLM_002_20241019T101302_003676` | `EMIT_L2B_CH4ENH_002_20241019T101302_2429307_021` |
| `EMIT_L2B_CH4PLM_002_20241020T075129_003680` | `EMIT_L2B_CH4ENH_002_20241020T075129_2429405_003` |
| `EMIT_L2B_CH4PLM_002_20241020T092144_003679` | `EMIT_L2B_CH4ENH_002_20241020T092144_2429406_003` |
| `EMIT_L2B_CH4PLM_002_20241020T170504_003677` | `EMIT_L2B_CH4ENH_002_20241020T170504_2429411_003` |
| `EMIT_L2B_CH4PLM_002_20241022T074913_003701` | `EMIT_L2B_CH4ENH_002_20241022T074913_2429605_016` |
| `EMIT_L2B_CH4PLM_002_20241023T070038_003703` | `EMIT_L2B_CH4ENH_002_20241023T070038_2429705_008` |
| `EMIT_L2B_CH4PLM_002_20241023T083741_003699` | `EMIT_L2B_CH4ENH_002_20241023T083741_2429706_026` |
| `EMIT_L2B_CH4PLM_002_20241026T061152_003688` | `EMIT_L2B_CH4ENH_002_20241026T061152_2430004_002` |
| `EMIT_L2B_CH4PLM_002_20241026T172133_003687` | `EMIT_L2B_CH4ENH_002_20241026T172133_2430011_019` |
| `EMIT_L2B_CH4PLM_002_20241130T180310_003723` | `EMIT_L2B_CH4ENH_002_20241130T180310_2433512_007` |
| `EMIT_L2B_CH4PLM_002_20250922T204933_003374` | `EMIT_L2B_CH4ENH_002_20250922T204933_2526514_006` |

Example protected links for the final pilot scene:

- [enhancement COG](https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2BCH4ENH.002/EMIT_L2B_CH4ENH_002_20250922T204933_2526514_006/EMIT_L2B_CH4ENH_002_20250922T204933_2526514_006.tif)
- [uncertainty COG](https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2BCH4ENH.002/EMIT_L2B_CH4ENH_002_20250922T204933_2526514_006/EMIT_L2B_CH4UNCERT_002_20250922T204933_2526514_006.tif)
- [sensitivity COG](https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2BCH4ENH.002/EMIT_L2B_CH4ENH_002_20250922T204933_2526514_006/EMIT_L2B_CH4SENS_002_20250922T204933_2526514_006.tif)

This pilot is for validating units, nodata, sensitivity correction, raster alignment, and label
construction. It is not large enough for the final paper.

### 3. Optional external corroboration

Carbon Mapper provides public plume/source metadata and imagery through its portal, STAC, and API.
API registration is available at <https://data.carbonmapper.org/register>; documentation is at
<https://api.carbonmapper.org/api/v1/docs>. Its current terms are non-commercial and require
attribution. If used, store exports under:

```text
EarthRemoteSensingRapidResponse/Data Collection/s2_emit_pairs/
  publication-v1/external/carbon-mapper/
```

Carbon Mapper is a corroborating catalog or external test source, not proof that an unlisted scene
contains no plume.

STARCOP and GeoCH4PlumeNet are useful optional ablations, not primary truth for this Sentinel-2
paper:

- STARCOP real hyperspectral benchmark: <https://doi.org/10.5281/zenodo.7863343>
- GeoCH4PlumeNet synthetic Sentinel-2 benchmark: <https://zenodo.org/records/16813369>

## Label and observability contract

Each sample must have one of four mutually exclusive labels:

| State | Required evidence | Permitted model output |
|---|---|---|
| `PLUME` | Reviewed S2 plume mask or sufficiently time-aligned high-confidence external detection | plume probability and mask |
| `NO_PLUME` | Observable reviewed negative from the same data domain | zero mask and calibrated no-plume confidence |
| `UNCERTAIN` | Conflicting, weak, temporally ambiguous, or single-annotator evidence | abstain |
| `UNOBSERVABLE` | Cloud, invalid coverage, low sensitivity, missing reference, or severe artifact | abstain |

An EMIT detection at time `t` is not automatically a Sentinel-2 plume label at `t +/- 1 day`.
External positive transfer should target `|delta t| <= 6 h`; `6-24 h` is exploratory and must be
reported separately; larger offsets cannot enter the locked test. Negative transfer is even more
strict: no EMIT plume at a different time cannot prove no plume in the Sentinel-2 image.

For new manually reviewed S2 labels:

- require two independent annotators for the locked test;
- preserve each original mask and adjudication record;
- report inter-annotator IoU and presence agreement;
- include hard negative categories: thin cloud, cloud shadow, water edges, bright surfaces,
  agricultural patterns, burn scars, sensor striping, and heterogeneous SWIR backgrounds;
- never convert an unlabeled image into a negative.

## Dataset and split contract

The tracked manifest should contain, at minimum:

```text
sample_id, source_dataset, source_revision, license, sensor, product_level,
target_scene_id, reference_scene_id, target_datetime, reference_datetime,
band_order, radiometric_scale, crs, transform, valid_mask, cloud_fraction,
label_state, label_source, label_confidence, plume_id, physical_source_id,
region_id, enhancement_units, sensitivity_path, uncertainty_path,
split, group_id, source_urls, checksums
```

Split rules:

1. Preserve the full official MARS-S2L validation/test sets for comparability; never tune on either.
2. Group by connected components sharing physical source, a 25 km geographic neighborhood,
   acquisition sequence, or near-duplicate imagery.
3. Split the released training rows into a deterministic internal train/validation partition by
   complete 25 km group. Fit normalization, class weights, thresholds, component area, and
   calibrators on those internal roles only.
4. Use as the primary test only official-test groups with no official-train member. Report the
   full official test and test-only-location views as explicitly weaker secondary comparisons.
5. Keep EMIT V002 and Carbon Mapper external tests untouched until the model and operating point
   are frozen.
6. Deduplicate against the current ERSRR corpus and all external catalogs before splitting.

The frozen machine-readable protocol is `configs/mars_publication_protocol.json`. Its ignored
56,552-row assignment manifest has SHA-256
`49d48669c765f06555f90a9fb94647e4983cbce13a983805e5fa440310c11671`. Exact roles are:

- internal training: 23,763 rows / 98 groups / 2,007 plume;
- internal validation: 5,945 rows / 24 groups / 505 plume;
- strict 25 km test: 4,401 rows / 150 groups / 67 plume / 4,334 no plume;
- official validation and the remaining 16,916 overlapping official-test rows: comparability only.

Both enforced group-overlap invariants are zero. See
`reports/acquisition/MARS_S2L_EVALUATION_PROTOCOL.md`.

The predeclared classical/physics baseline ladder may inspect the MARS strict-spatial benchmark,
but candidate architecture and threshold selection remain internal-validation-only. The final
independent confirmation is the untouched time-aligned EMIT V002 cohort, evaluated only after the
candidate architecture and operating rule are frozen.

## Model architecture hypothesis

The current compact raw ResUNet remains a five-band single-time baseline. The released-checkpoint
result rules out another compact shared-encoder iteration as the primary path. The next paper
candidate is a full-resolution dual-temporal selective U-Net initialized from the released model:

```text
target + reference + MBMP + wind + cloud --> full U-Net --> segmentation decoder --> plume mask
                                                   |       multi-scale top-k --> plume/no-plume
                                                   |       quality features --> observable/abstain
validity mask -------------------------------------|---------------------------> masked losses
```

Initial input contract:

- native MARS target and reference Sentinel-2 observations on one 200 x 200, 10 m grid;
- six declared bands per date: B02/B03/B04/B08/B11/B12, plus separately derived MBMP/retrieval
  channels;
- cloud/validity masks and temporal gap as explicit metadata;
- manifest-declared asset roles and explicit cloud classes; never infer clear/invalid solely from
  ancillary-raster nodata metadata;
- no silent L1C/L2A mixing;
- S2-only experiments first; Landsat is a later cross-sensor extension.

Heads and training:

- retain the released 13.6M-parameter full-resolution U-Net as the capacity baseline instead of
  widening the undertrained 2.75M-parameter shared encoder;
- connected-proposal presence head trained on balanced positives and real hard negatives;
- proposal evidence combines decoder features, component shape/area, surrounding multiscale
  context, target/reference change, and wind alignment; top-k/max mask confidence is retained only
  as an input, not the decision statistic;
- segmentation loss evaluated only where the target and annotation are valid;
- observability/quality head trained to reject invalid scenes;
- online hard-negative mining after the first baseline pass;
- class-balanced sampling by source and region, not by pixel;
- deep ensemble or repeated-seed ensemble for epistemic uncertainty;
- validation-only calibration with separate low/high thresholds for no-plume, abstain, and plume.

The native adapter follows the released implementation at inspected commit
`f7d264c2c845dfba1cb27f76ef6026275f8d8758`: divide TOA integers by 5,000, clip to `[0,2]`, and
compute MBMP from separately median-normalized B12/B11 target and background ratios. ERSRR also
computes a validity-aware MBMP variant whose medians exclude cloud and radiometrically invalid
pixels. Both must remain explicit ablations.

The released MARS-S2L model is now reproduced and establishes that capacity plus data scale is the
dominant gap. The next experiment should therefore warm-start that exact U-Net and add selective
heads; Prithvi, SatMAE, DOFA, or SegFormer remains a later ablation only if the warm-started model
fails under the same expanded-data contract.

## Experiment ladder

| ID | Experiment | Purpose | Promotion gate |
|---|---|---|---|
| E0 | Prior, raw logistic, MBMP threshold | Detect leakage and metric bugs | Above prevalence/chance on frozen validation |
| E1 | Existing compact single-time raw ResUNet | Retain current baseline | Reproduce recorded grouped result |
| E2 | Released CH4Net and MARS-S2L checkpoints | Establish credible state of the art | Reproduce published evaluation protocol within tolerance |
| E3 | Dual-temporal raw model | Test temporal information | Beat E1 on site-blocked AUPRC and false-positive rate |
| E4 | Dual-temporal raw + MBMP physics channel | Test physics guidance | Beat E3 at matched recall |
| E5 | Presence head + hard-negative mining | Improve no-plume rejection | At least 25% relative FPR reduction without worse recall CI |
| E6 | Selective calibration / ensemble | Control operational risk | Meet locked recall/FPR and calibration gates |
| E7 | Leave-one-region-out and EMIT V002 external test | Test generalization | Improvement survives unseen sites/regions |
| E8 | Optional pretrained backbone | Test scale/transfer | Added value exceeds variance and compute cost |

Every learned experiment should use at least five fixed seeds. Report all seeds, not only the best.

## Predeclared metrics and success gates

The primary endpoint is scene-level plume recall at a fixed no-plume false-positive rate. Pixel F1
alone is not sufficient.

Required metrics:

- scene: precision, recall, specificity, negative predictive value, AUPRC, AUROC, Brier score,
  expected calibration error, and false positives per 100 observable scenes;
- plume object: precision/recall, detection probability by plume area/enhancement/flux bucket, and
  source-localization error;
- segmentation: IoU, Dice/F1, pixel average precision, false-positive area, and boundary error;
- selective prediction: coverage versus risk, abstention rate, and errors among accepted
  `NO_PLUME` decisions;
- stratified: unseen site, region, surface heterogeneity, cloud fraction, temporal gap, sensor,
  plume size, and enhancement strength.

Statistics:

- 95% confidence intervals from a block bootstrap over physical source sites, not pixels;
- paired comparisons on identical test scenes;
- at least five fixed training seeds for learned models;
- no test-set threshold selection;
- publish failures and per-region results.

Minimum research promotion gate:

- lower 95% confidence bound for scene recall at least 0.75 at scene FPR no greater than 0.05;
- no-plume specificity at least 0.95 on representative reviewed negatives;
- at least 25% relative FPR reduction versus the strongest reproduced baseline at non-inferior
  recall, or a clearly novel external-benchmark contribution if that model lift is not achieved;
- positive performance above prevalence and simple retrieval baselines in every primary test;
- a usable calibration curve and explicit abstention behavior;
- no unresolved source/geographic leakage.

Stretch target: at least 0.85 scene recall at no more than 0.03 FPR on unseen sites. These are
targets, not current claims. For context, the current MARS-S2L paper reports 78% plume recall and
8% false-positive rate at 697 previously unseen sites.

## Execution phases

### Phase 0 - freeze the protocol (2-3 days)

- [x] Convert this plan into a machine-readable experiment specification.
- [x] Define the four label states and primary metric before training.
- [x] Record dataset licenses and immutable source revisions.

Exit: signed-off protocol and no ambiguity about what `NO_PLUME` means.

### Phase 1 - ingest and audit MARS-S2L (1 week)

- Download and verify pinned metadata, then build a frozen selective S2 cohort manifest.
- Estimate and approve the selective transfer before downloading its assets; do not mirror the
  full mixed-sensor corpus.
- Build a read-only dataset adapter; do not rewrite 100 GB into another raw copy.
- Audit class balance, sites, regions, product levels, bands, temporal pairs, clouds, duplicates,
  official splits, and missing assets.
- Generate a small tracked manifest and integrity report.

Exit: every sample resolves to a state, group, split, checksum, and product contract.

### Phase 2 - reproduce baselines (1-2 weeks)

- Implement MBMP under the same grid and masks.
- Run CH4Net and released MARS-S2L checkpoints.
- Run the current compact ERSRR model as a controlled baseline.
- Reproduce official split metrics before designing a new network.

Exit: credible baseline table and documented discrepancies from published values.

Current development status: native MBMP, raw/physics scene and pixel-logistic, joint v1/MIL-v2,
released CH4Net, and released MARS-S2L baselines are complete and frozen. Released MARS-S2L is the
strongest detector but still fails the recall-confidence and FPR gates; CH4Net is substantially
weaker. Phase 2 is complete. Any subsequent ERSRR change must be selected on internal validation
or new external data, not on the already reported strict-spatial development benchmark.

### Phase 3 - hard-negative and architecture experiments (2-4 weeks)

- Add the target/reference data path.
- Run E3-E6 in order, stopping variants that fail gates.
- Mine and categorize validation false positives without touching test data.
- Freeze architecture, seeds, and operating thresholds.

Exit: one selected model with a fixed artifact contract and no test-set tuning.

Current status: v3 code and unit/integration smoke tests are complete. The full from-scratch run is
waiting only for the minimum v3 asset catalog documented above; released weights will not be used
because they would leak official-train samples into ERSRR internal validation.

### Phase 4 - independent EMIT V002 validation (2-3 weeks)

- Validate the 12 authenticated pilot bundles and build the enhancement/sensitivity/uncertainty
  reader.
- Use public CMR metadata to select a larger, geography-balanced candidate set before downloading.
- Retain only time-aligned, observable examples; label ambiguous cases as uncertain.
- Have two annotators review the locked external cohort.
- Run the frozen model exactly once for primary reporting.

Exit: at least 50 independent external source/region groups, or an explicit feasibility result if
the strict time-alignment gate yields fewer cases.

### Phase 5 - paper and release (2-3 weeks)

- Run five seeds and site-block bootstrap intervals.
- Produce error-atlas, calibration, PR, geography, and ablation figures.
- Archive code/config/manifests/derived labels with a DOI; do not redistribute restricted raw data.
- Write AMT-format Methods, Results, limitations, code availability, and data availability.
- Release a model card that says exactly which sensors/products and operating thresholds are valid.

Exit: a fully reproducible manuscript package, not just a model checkpoint.

Expected duration: roughly 8-13 focused weeks after the full data is available. The schedule is
data- and gate-dependent; failed ablations should shorten it rather than trigger architecture
shopping.

## Immediate handoff

### User action required

1. Download the public minimum MARS-S2L v3 transfer: 59,240 remaining assets / exactly
   29,198,856,248 bytes (27.193 GiB). No account is required; use the four commands above and keep
   the generated receipt. Source: <https://huggingface.co/datasets/UNEP-IMEO/MARS-S2L>.
2. Create or activate a free NASA Earthdata Login:
   <https://urs.earthdata.nasa.gov/users/new>.
3. In Earthdata Search, download the 12 CH4ENH scene bundles and matching 12 CH4PLM bundles listed
   above.
4. Put them in the exact `publication-v1/CH4ENH/...` and `publication-v1/CH4PLM/...` paths. Do not
   rename files and do not add them to Git.
5. Do not send credentials. When the files are present, report only that the download is complete.

### Work that does not require user credentials

- Pinned metadata, the 1.708 GiB development tranche, and both released/MIL-v2 checkpoint baselines
  are downloaded, verified, audited, and committed as compact evidence. The 27.193 GiB remaining
  v3 training transfer awaits the user's download; do not perform the full ~100 GB mirror.
- The MARS adapter, group/leakage protocol, classical baselines, released checkpoint reproduction,
  and validation-only MIL failure audit are complete.
- Generate the exact larger EMIT candidate list from public CMR metadata before requesting more
  authenticated downloads.
- Keep raw data and trained models ignored; commit only manifests, hashes, code, configs, and
  results.
