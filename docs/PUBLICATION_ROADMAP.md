# ERSRR publication roadmap: reliable plume detection and no-plume rejection

- Status: proposed execution plan
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

Pin the initial import to repository revision:

```text
c26b1d7e31a0c5241fa37c9140802622c215eb32
```

Local destination, already beneath an ignored acquisition root:

```text
EarthRemoteSensingRapidResponse/Data Collection/s2_emit_pairs/
  publication-v1/external/MARS-S2L/
```

Install the public Hugging Face downloader in the project environment:

```bash
python -m pip install --upgrade "huggingface_hub[cli]"
```

Download only metadata first (roughly 190 MB):

```bash
hf download UNEP-IMEO/MARS-S2L \
  README.md train.csv val.csv test.csv validated_images_all.csv \
  validated_images_plumes.csv location_name_mapping.json \
  --repo-type dataset \
  --revision c26b1d7e31a0c5241fa37c9140802622c215eb32 \
  --local-dir "EarthRemoteSensingRapidResponse/Data Collection/s2_emit_pairs/publication-v1/external/MARS-S2L"
```

After the metadata audit confirms product level, bands, site identifiers, split integrity, and
license obligations, download the full pinned repository:

```bash
hf download UNEP-IMEO/MARS-S2L \
  --repo-type dataset \
  --revision c26b1d7e31a0c5241fa37c9140802622c215eb32 \
  --local-dir "EarthRemoteSensingRapidResponse/Data Collection/s2_emit_pairs/publication-v1/external/MARS-S2L"
```

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

1. Preserve the official MARS-S2L test set as frozen external evidence; never tune on it.
2. Group by connected components sharing physical source, a 25 km geographic neighborhood,
   acquisition sequence, or near-duplicate imagery.
3. Keep complete regions/sites out of training for the geographic-transfer test.
4. Fit normalization and class weights on training groups only.
5. Choose thresholds and calibrators on validation groups only.
6. Keep EMIT V002 and Carbon Mapper external tests untouched until the model and operating point
   are frozen.
7. Deduplicate against the current ERSRR corpus and all external catalogs before splitting.

## Model architecture hypothesis

The current compact raw ResUNet remains a five-band single-time baseline. The paper candidate is a
dual-temporal selective detector:

```text
target S2 bands ----- shared encoder ----\
                                      change fusion --> segmentation decoder --> plume mask
reference S2 bands -- shared encoder ----/        |--> presence head ---------> plume/no-plume
MBMP / retrieval channels ------------------------|--> quality head ----------> observable/abstain
cloud + validity masks ---------------------------/
```

Initial input contract:

- target and reference Sentinel-2 observations on one 20 m grid;
- canonical B2/B3/B4/B11/B12 raw values, plus a separately declared MBMP/retrieval channel;
- cloud/validity masks and temporal gap as explicit metadata;
- no silent L1C/L2A mixing;
- S2-only experiments first; Landsat is a later cross-sensor extension.

Heads and training:

- scene-presence head trained on balanced positives and real hard negatives;
- segmentation loss evaluated only where the target and annotation are valid;
- observability/quality head trained to reject invalid scenes;
- online hard-negative mining after the first baseline pass;
- class-balanced sampling by source and region, not by pixel;
- deep ensemble or repeated-seed ensemble for epistemic uncertainty;
- validation-only calibration with separate low/high thresholds for no-plume, abstain, and plume.

Do not add a larger backbone until the released MARS-S2L model is reproduced. A pretrained
Prithvi, SatMAE, DOFA, or SegFormer encoder is an ablation only after the data adapter, baselines,
and evaluation gates pass.

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

- Convert this plan into a machine-readable experiment specification.
- Define the four label states and primary metric before training.
- Record dataset licenses and immutable source revisions.

Exit: signed-off protocol and no ambiguity about what `NO_PLUME` means.

### Phase 1 - ingest and audit MARS-S2L (1 week)

- Download pinned metadata, then the full corpus after the metadata audit.
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

### Phase 3 - hard-negative and architecture experiments (2-4 weeks)

- Add the target/reference data path.
- Run E3-E6 in order, stopping variants that fail gates.
- Mine and categorize validation false positives without touching test data.
- Freeze architecture, seeds, and operating thresholds.

Exit: one selected model with a fixed artifact contract and no test-set tuning.

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

1. Create or activate a free NASA Earthdata Login:
   <https://urs.earthdata.nasa.gov/users/new>.
2. In Earthdata Search, download the 12 CH4ENH scene bundles and matching 12 CH4PLM bundles listed
   above.
3. Put them in the exact `publication-v1/CH4ENH/...` and `publication-v1/CH4PLM/...` paths. Do not
   rename files and do not add them to Git.
4. Do not send credentials. When the files are present, report only that the download is complete.

### Work that does not require user credentials

- Download and audit the pinned MARS-S2L metadata and, after storage approval, the full ~100 GB
  public corpus.
- Build the MARS adapter, group/leakage audit, benchmark specification, and baseline runner.
- Generate the exact larger EMIT candidate list from public CMR metadata before requesting more
  authenticated downloads.
- Keep raw data and trained models ignored; commit only manifests, hashes, code, configs, and
  results.
