# ERSRR MARS-S2L successor paper outline

## Working title

**Baseline-preserving temporal-physics correction for mixed-sensor methane-plume detection**

Alternative benchmark title:

**Beyond MARS-S2L: leakage-resistant mixed Sentinel-2/Landsat methane-plume detection on the authors' official test**

## Central question

Can a successor trained only on public, site-separated MARS-S2L development data improve scene ranking, dense plume overlap, and recall without increasing false-positive rate relative to the released MARS-S2L system on the authors' exact official 2024 benchmark?

The endpoints are binary scene detection and dense binary segmentation. The study does not claim methane concentration, emission flux, source attribution, operational readiness, or prevalence-representative precision.

## Frozen comparator and primary endpoints

Primary source: [MARS-S2L paper v3](https://arxiv.org/html/2511.21777v3), revised 2026-04-24. Official code is pinned at `UNEP-IMEO-MARS/marss2l@f7d264c2c845dfba1cb27f76ef6026275f8d8758`.

| View | Scenes | Plume | Sites | Comparator AP | Recall | FPR | Pixel IoU |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full official test | 43,529 | 1,813 | 1,289 | 0.641020 | 0.791506 | 0.070692 | 0.324365 |
| Test-only sites | 15,655 | 227 | 697 | 0.450274 | 0.775330 | 0.075512 | 0.171560 |

The reconstructed archive values above are the primary numerical comparator because they are slightly stronger than the rounded paper tables. A successful model must also exceed the published Table S5/S6 values.

Primary superiority endpoints on both views:

1. higher average precision;
2. higher fixed-rule pixel IoU;
3. higher recall;
4. no higher false-positive rate; and
5. paired site-block bootstrap support, with the lower 95% bounds of AP and IoU deltas above zero.

Unavailable historical scenes and missing pixel truth receive adversarial candidate outcomes. This prevents missing data from creating a favorable claim.

## Contribution set

1. An exact, versioned reconstruction of the paper-v3 cohort, scene rule, onshore/offshore checkpoint routing, and published metrics.
2. A mixed Sentinel-2/Landsat 16-channel adapter with audited native-band semantics, immutable 82.9 GiB catalog identities, and physically separated development/test manifests.
3. Five deterministic site-block development folds with primary architecture selection on fold 0, independent confirmation on fold 1, five-fold out-of-fold calibration, and one-shot test access.
4. A forward-equivalent released U-Net embedded as a frozen teacher, plus a zero-initialized temporal-physics correction using MBMP, target/reference differences, normalized differences, log ratios, wind, cloud, sensor identity, and teacher logits.
5. One-sided non-regression constraints that discourage new responses on known no-plume scenes and discourage suppression inside annotated plume masks.
6. Exact connected-component evaluation, sensor-stratified model promotion, multi-seed/fold uncertainty, and adversarial treatment of the five unavailable official scenes.

## Data contract

- Public development: 44,363 scenes, 3,811 plumes, 618 physical sites; 37,418 Sentinel-2 and 6,945 Landsat scenes.
- Paper test: 43,529 archived assignments, 1,813 plumes, 1,289 sites; held sealed until ensemble and calibration hashes are frozen.
- Development assets: 96,348 files / 45.54 GB selected from the pinned remote catalog.
- Full mixed cohort: 187,014 assets / 82.91 GiB; bulk files and model artifacts remain Git-ignored.
- The released checkpoint refers to a private March-2025 Azure label snapshot that is unavailable. The successor therefore uses the complete pinned public train/validation labels and states this limitation explicitly.

## Methods structure

### Released teacher

Recreate the authors' `UnetOriginal` exactly and verify bitwise logit equality against the independent implementation for both sensor identities. Keep its parameters and batch-normalization state frozen during correction-only fitting.

### Temporal-physics correction

Use a compact residual encoder-decoder over explicit temporal features and teacher logits. Initialize the final correction convolution to zero. Include trainable sensor-specific teacher scale and bias, initialized to identity/zero.

### Objectives

- hard-negative pixel BCE and positive-scene soft Dice;
- top-k scene-presence BCE;
- upward-logit penalty on fully labeled no-plume scenes;
- downward-logit penalty inside observable annotated plume pixels; and
- correction-magnitude regularization.

All weights are command-line parameters stored in checkpoints and reports. The primary configuration is frozen before fold-1 confirmation.

### Sampling and augmentation

Sample equally across plume/no-plume and Sentinel-2/Landsat strata. Apply deterministic seeded rotations and flips, rotating wind vectors consistently. Later ablations may add physically simulated plumes only from fitting-fold source masks and fitting-fold negative backgrounds.

### Evaluation

Use the paper's scene score: the maximum probability threshold supporting at least 100 8-connected pixels. Use probability 0.5 and the same 100-pixel component filter for fixed-rule masks. Report overall and per-sensor AP, recall at the comparator FPR, FPR, precision, AUROC, Dice, and IoU.

## Experiment sequence

1. Full fold-0/fold-1 released-baseline evaluation and zero-residual equivalence audit.
2. Fold-0 correction-only primary fit.
3. If any primary point gate fails, ablate loss weights, temporal feature families, correction capacity, scene surrogate, and hard-negative curriculum without reading fold 1.
4. Freeze the best fold-0 architecture and confirm once on fold 1.
5. Fit all five site folds and collect exactly one out-of-fold prediction per development scene.
6. Cross-fit any calibration or ensemble weighting; compare multi-seed variance.
7. Freeze code, checkpoints, thresholds, missing-scene policy, and evaluator hashes.
8. Open the official test once and compute 10,000 paired site-block bootstrap replicates on full and test-only-site views.
9. If the gate fails, report the failure without tuning on test; improvements require new prospective development evidence or a newly sealed confirmation cohort.

## Required ablations

| Ablation | Question |
|---|---|
| Released model only | Does the evaluator reproduce the frozen teacher? |
| No temporal normalized/log-ratio features | Do explicit temporal physics features add value? |
| No sensor identity/affine correction | Is mixed-sensor adaptation necessary? |
| No no-plume upward constraint | Does the constraint reduce false alarms? |
| No plume-pixel teacher floor | Does the floor protect recall/IoU? |
| No scene loss | Is scene-level supervision useful beyond masks? |
| Reduced correction capacity | Are gains architectural or simply parametric? |
| No simulation / physics simulation | Does augmentation improve rare morphology generalization? |

## Statistical analysis contract

- Experimental unit: physical site, not scene or pixel.
- Architecture selection: fold 0 only; confirmation: fold 1 only.
- Final inference: 10,000 paired nonparametric site-block bootstrap resamples with a committed seed.
- Point estimates and two-sided percentile intervals are reported for model metrics and paired deltas.
- AP and IoU superiority require lower 95% paired-delta bounds above zero.
- Recall must be higher and FPR no worse at the frozen paper operating rule on both official views.
- Sensor-stratum regressions greater than 0.01 AP or IoU block development promotion.
- No model, seed, threshold, component rule, ensemble weight, or postprocessing parameter may be chosen from the paper test.

## Planned tables

1. Paper-v3 comparator reproduction and residual differences from Tables S5/S6.
2. Public/private split-count differences and data availability.
3. Mixed-sensor input semantics and asset provenance.
4. Fold/site/label/sensor balance.
5. Fold-0 ablations and promotion checks.
6. Independent fold-1 confirmation.
7. Five-fold/multi-seed out-of-fold results and calibration.
8. One-shot full official and test-only-site results versus reconstructed and published MARS-S2L.
9. Paired site-bootstrap deltas and superiority decisions.
10. Sensor, geography, plume-size, cloud, wind, and onshore/offshore error strata.

The development ablation table must include the bounded-reproducibility
Gaussian-ViT + fixed-DOFA ensemble: AP +0.002449 with paired-site 95% interval
[+0.000489,+0.004068], identical operating counts, and fixed dense IoU evidence.
It must also show the rejected exact replay and explain why bounded stochastic
replicate gates were frozen before retaining logits.

## Planned figures

1. Data boundary, site-fold, ensemble-freeze, and one-shot test flow.
2. Released teacher plus temporal-physics correction architecture.
3. Development PR curves and fixed operating points for folds 0 and 1.
4. Pixel masks for deterministically chosen improvement/regression examples.
5. Full/test-only paired bootstrap distributions.
6. Sensor-stratified calibration and performance.

## Claim language

Until the one-shot gate passes, permissible wording is: “The paper comparator and leakage-resistant successor protocol were reproduced; paper-test superiority has not yet been evaluated.”

Only if every primary gate passes may the abstract say that ERSRR outperformed released MARS-S2L on the authors' official benchmark. Point-estimate improvements without paired interval support are not “without doubt” and must be described as inconclusive.

## Limitations and ethics

- Public fitting labels differ from the unavailable private checkpoint-training snapshot.
- Five official scenes lack current rasters and four positive scenes lack current pixel truth.
- Site-blocking reduces, but cannot eliminate, all forms of geographic or operator dependence.
- Benchmark precision is not operational positive predictive value.
- Outputs must not be presented as emission-rate estimates or regulatory findings.
- MARS-S2L is CC-BY-NC-SA-4.0; redistribution and publication must follow its license and attribution requirements.

## Reproducibility bundle

Release code, exact commits, environment versions, compact protocols, hashes, commands, metric JSON, bootstrap seeds, reports, and the final HTML dossier. Exclude third-party bulk imagery, credentials, signed URLs, model checkpoints, and prediction caches unless redistribution rights and size justify inclusion.

## 2026-08-02 calibration result for the manuscript

An honest fold-3/fold-4 threshold-transfer experiment tests group-level
conformal risk control at a preregistered 7.5% target. Fold 4 -> 3 transfers at
7.15% crop FPR and 4.27% group-balanced FPR, but fold 3 -> 4 rises to 16.52%
crop FPR and 8.31% group-balanced FPR. Pooled recall remains 97.44%, so the
failure is calibration transport rather than complete detector collapse.

Paper framing: CRC supplies an exchangeability-scoped expected-risk curve, not
a distribution-free guarantee under geographic or product shift. Directional
results must be shown rather than hidden behind the pooled 5.95% group FPR.
This negative result strengthens the data-diversity thesis and prevents an
overclaim present in the external recommendation.

## 2026-08-02 controlled-release data-provenance result

Stanford's 2022 single-blind campaign supplies the strongest available
no-plume semantics—metered zero release—but is not an independent MARS source.
The audit found 677 same-site upstream MARS rows and eight exact target-product
matches, all excluded as `Not Used`. The paper may use a future one-shot score
only as a fixed-site diagnostic and must not count it as geographic confirmation.
The methods section should also state explicitly that MARS temporal background
selection does not imply methane absence and therefore was never used to create
negative labels.
