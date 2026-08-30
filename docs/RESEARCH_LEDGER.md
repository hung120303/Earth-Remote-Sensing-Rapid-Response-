# ERSRR research ledger

This ledger records paper-relevant decisions, outcomes, and interpretation boundaries. Compact JSON
reports are authoritative for numbers and hashes; this document is the human-readable study map.

## Current study: tri-temporal v5.1 (2026-07-13)

### Research question and claim boundary

The current question is whether a tri-temporal, physics-aware ERSRR model can accurately rank and
segment methane-plume crops while rejecting no-plume crops at geographically isolated locations,
and how that result compares with the frozen ERSRR v4.3 ensemble and released MARS-S2L checkpoint.
The primary confirmation cohort is the pinned MethaneS2CM L2A location test. It is approximately
balanced by crop construction, so its precision is benchmark precision rather than operational
positive predictive value.

V5.1 was trained on MethaneS2CM L2A. V4.3 and released MARS-S2L were trained on MARS-S2L L1C and
are zero-shot comparators on this test. This is a same-cohort performance comparison, not an
architecture-only causal experiment. Published MARS-S2L metrics remain different-cohort context.

### Data and seal

- Source: `H1deaki/MethaneS2CM`, revision
  `ee9a96d4994ca6bc45725c1e92d7a06258131eaf`, CC-BY-NC-4.0.
- Location training metadata: 80,217 crops, 40,425 positives, and 3,460 exact locations.
- Internal fitting: 64,759 crops in 193 frozen 25 km groups.
- Internal development: 15,458 crops in 64 disjoint 25 km groups.
- Sealed location test: 20,789 crops, 10,453 positives, 10,336 negatives, 816 exact locations, and
  100 test-only 25 km components; exact train/test coordinate overlap is zero.
- Six source archives total 16,050,584,895 compressed bytes. The ignored train pack is
  4,249,375,485 bytes; the ignored test pack is 1,075,369,707 bytes.
- The test pack SHA-256 is
  `7e0c7d06cdf6fde8eb81c6feea179f9cb6b1e6d887797a5ebadf99037670caca`.
- Architecture, seed reports, checkpoint hashes, empirical-CDF calibrators, scene thresholds,
  pixel threshold, acquisition code, and comparison code were committed before any test TIFF was
  decoded. The one-shot evaluator ran from commit `4076e690`.

Bulk archives, HDF5 packs, checkpoints, and prediction caches remain ignored. Compact protocols,
source identities, code, and result reports are tracked.

### Architecture decision

V5.1 uses a 9,358,256-parameter model per seed:

1. Sentinel-2 L2A frames at T, T-90, and T-365 using B02/B03/B04/B08/B11/B12;
2. shared encoder weights across all three frames;
3. two scale-invariant MBMP channels and seven-term temporal fusion at five scales;
4. a full-resolution U-Net decoder with a learned physics prior;
5. a scene logit fixed to 35% mask-derived evidence and 65% of a small context head; and
6. a three-seed ensemble using per-seed empirical-CDF scene percentiles and equal dense-probability
   averaging.

The context head was a controlled response to the released mask geometry: the positive crop has a
median 226/1,024 mask pixels, but 2,552 positive crops are at least half masked, 758 are at least
90% masked, and 103 are fully masked. A purely mask-derived scene bottleneck was therefore too
restrictive for this benchmark. The best frozen physics-only scene baseline achieved AP 0.5509 and
at most 0.0564 recall at approximately 5% FPR; learned v5/v5.1 performance is not explained by an
MBMP threshold alone.

### Internal development confirmation

| Estimate | Scene AP | AUROC | Recall at 5% target | Realized FPR | Pixel AP | Dice | IoU |
|---|---:|---:|---:|---:|---:|---:|---:|
| V5.1 seed mean | 0.8481 | 0.8524 | 0.4125 | ~0.0500 | 0.3028 | 0.4330 | 0.2763 |
| Five-fold 25 km group-held calibration | 0.8647 | 0.8622 | 0.4687 | 0.0519 | — | 0.4389 | 0.2811 |
| Final all-development rule | 0.8658 | 0.8655 | 0.4671 | 0.0499 | 0.3090 | 0.4424 | 0.2840 |

The group-held 2,000-resample bootstrap gave AP 0.8647 (95% CI 0.8232–0.9020), AUROC 0.8622
(0.8124–0.9110), recall 0.4687 (0.3825–0.5545), FPR 0.0519 (0.0401–0.0650), Dice 0.4389
(0.3675–0.5200), and IoU 0.2811 (0.2251–0.3514). Checkpoints were still selected on this
development cohort, so these were confirmation/freeze values rather than an external estimate.

### One-shot location-test result

| Frozen model/rule | Scene AP | AUROC | Recall | FPR | Precision* | Pixel AP | Dice | IoU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ERSRR v5.1 three-seed | 0.8180 | 0.8276 | 0.3778 | 0.0607 | 0.8630 | 0.2083 | 0.3125 | 0.1852 |
| ERSRR v4.3 zero-shot | 0.5911 | 0.5950 | 0.0000 | 0.0000 | 0.0000 | 0.1561 | 0.0000 | 0.0000 |
| Released MARS-S2L zero-shot | 0.5252 | 0.5126 | 0.0052 | 0.0033 | 0.6136 | 0.1691 | 0.0045 | 0.0023 |

*Precision is not operational PPV on this approximately balanced benchmark.

At its frozen primary rule, v5.1 detected 3,949/10,453 plume crops and falsely flagged
627/10,336 no-plume crops. Released MARS-S2L rejected almost every no-plume crop but detected only
54/10,453 plume crops. V4.3 made no positive scene or pixel prediction at its frozen rules.

V5.1 minus released MARS-S2L point deltas were +0.2928 AP, +0.3149 AUROC, +0.3726 recall,
+0.0574 FPR, +0.3080 Dice, and +0.1829 IoU. Paired 2,000-resample 25 km group intervals supported
the AP delta [0.2256, 0.3824], AUROC delta [0.2495, 0.4061], recall delta [0.2886, 0.5036], Dice
delta [0.2079, 0.4805], and IoU delta [0.1163, 0.3183]. They also established that the FPR delta
was higher, not lower: [0.0319, 0.0914].

The strongest permissible claim is therefore: **v5.1 materially improves plume ranking and dense
localization over the frozen zero-shot comparators on the MethaneS2CM location test, but does not
establish across-the-board MARS-S2L superiority because the no-plume FPR criterion fails.**

### Frozen post-hoc calibration audit

After the primary report was committed, the four thresholds already selected on development were
applied without choosing any new rule:

| Development FPR target | Frozen threshold | Dev recall | Test recall | Dev FPR | Test FPR |
|---:|---:|---:|---:|---:|---:|
| 2.0% | 0.827382 | 0.2740 | 0.2436 | 0.0200 | 0.0244 |
| 5.0% | 0.733579 | 0.4671 | 0.3778 | 0.0499 | 0.0607 |
| 8.0% | 0.663238 | 0.5957 | 0.4912 | 0.0799 | 0.1035 |
| 9.5% | 0.636391 | 0.6379 | 0.5345 | 0.0950 | 0.1242 |

Every frozen point transferred to higher FPR and lower recall. This supports calibration/domain
transfer as a future hypothesis. It does not authorize selecting a threshold from test labels.

### Relationship to MARS-S2L benchmarks

On the separate strict MARS cohort, v4.3 achieved AP 0.3903, AUROC 0.8644, recall 0.5224, FPR
0.0277, Dice 0.1422, and IoU 0.0766. Released MARS-S2L on those same scenes achieved AP 0.3521,
AUROC 0.8175, recall 0.6418, FPR 0.0948, Dice 0.2346, and IoU 0.1329. V4.3 improved ranking and
FPR but not recall or overlap; only its FPR delta was bootstrap-conclusive.

A later candidate-specific post-test replay applied the already frozen Gaussian+DOFA score to all
4,401 strict rows (67 positives, 4,334 negatives, 150 spatial components). Relative to a fresh
released-checkpoint connected-component replay, Gaussian+DOFA raised AP from 0.3521 to 0.3749 and
reduced FPR from 0.0951 to 0.0570, but recall fell from 0.6418 to 0.6119. Paired component intervals
crossed zero for AP `[-0.0132, 0.0605]`, fixed recall `[-0.1698, 0.0282]`, and matched-FPR recall
`[-0.0370, 0.1087]`; only the FPR delta `[-0.0453, -0.0309]` was conclusive. Gaussian+DOFA and its
current-v3 base made identical fixed decisions; the added endpoints improved ranking AP by 0.0103
with a positive interval `[0.0027, 0.0202]` but no operational recall gain. The frozen superiority
gate failed, so Gaussian+DOFA is retired as a superiority candidate. This is not a fresh
project-level holdout because earlier ERSRR candidates had already opened the exact-paper view.

The MARS-S2L paper reports full-test AP 0.6408, recall 0.7915, FPR 0.0713, and IoU 0.3224, and
test-only-site AP 0.4496, recall 0.7753, and FPR 0.0763. Those are different-cohort context. V5.1’s
MethaneS2CM AP is numerically higher, but its recall and IoU are lower than the paper full-test
numbers; none of those cross-table differences is a paired superiority estimate.

### V5.2 research path fixed from the evidence

1. Fit spatially group-held calibration or conformal/risk-control rules on new calibration groups.
   The current test may audit transfer but may not fit the rule.
2. Train product-aware L1C/L2A harmonization with explicit product tokens, missing-frame handling,
   and shared development folds across MARS-S2L and MethaneS2CM.
3. Improve dense boundaries using mask-quality weighting, plume-scale sampling, and hard-negative
   scene balance while retaining a high-capacity ranking head.
4. Evaluate calibration by geography and prevalence under a preregistered coverage-risk contract,
   not a single pooled threshold.
5. Acquire a new, geographically isolated and prevalence-aware confirmation cohort before v5.2 is
   finalized. Seal it before model selection and open it exactly once.

Authoritative current artifacts:

- `reports/experiments/methanes2cm_v5_1_ensemble_validation.json`;
- `reports/experiments/methanes2cm_v5_1_location_test.json`;
- `reports/experiments/methanes2cm_v5_1_location_test_posthoc.json`;
- `reports/experiments/mars_v4_3_strict_comparison.json`;
- `reports/experiments/mars_gaussian_dofa_strict_spatial_one_shot.json`;
- `reports/ERSRR_RESEARCH_REPORT.html`.

## Retired v3 study question and frozen decision rule

The primary question was whether the 14.27 M-parameter, 16-channel ERSRR v3 detector could reduce
false alarms on observable no-plume Sentinel-2 scenes while preserving plume recall relative to the
released MARS-S2L checkpoint at geographically isolated sites.

Before opening the strict cohort, five seeds (101, 202, 303, 404, and 505) were fixed. Each seed
selected its checkpoint and operating threshold on internal validation only by maximizing recall
subject to FPR <= 0.05, followed by AP and positive-mask Dice. The strict promotion gate required:

1. mean FPR <= 0.05 and mean specificity >= 0.95;
2. the lower 95% recall interval >= 0.75;
3. the lower paired recall-delta interval >= 0;
4. the lower relative-FPR-reduction interval >= 0.25.

The unit of paired resampling was the frozen 25 km group. The campaign used 2,000 bootstrap
replicates and resampled both groups and the five training seeds.

## Cohorts and artifact discipline

- Internal training: 23,763 scenes in 98 groups.
- Internal validation: 5,945 scenes in 24 disjoint groups.
- Strict spatial test: 4,401 scenes in 150 groups, with 67 plume and 4,334 no-plume scenes.
- Strict-manifest SHA-256: `6e959ae0af50c5a309247cbe674fe01b31a36f8b149f9be3292acebed3e5f906`.
- Training/validation MARS transfer: 61,928 assets and 30,366,803,325 bytes, kept outside Git.
- Detector-only strict transfer: 8,869 assets and 4,489,575,260 bytes, kept outside Git.
- Checkpoints, proposal artifacts, prediction caches, imagery, and logs remain ignored. Only compact
  reports, manifests, receipts, code, and documentation are committed.

The strict baseline initially stopped before scoring its first scene because the released evaluator
requested an unused methane-enhancement raster that the deliberately detector-only strict transfer
does not contain. The adapter already supported detector-only samples. Commit `cfce7f45` added the
missing `require_enhancement=False` call and a regression test. No prediction had been produced, no
cohort membership or model input changed, and the repaired evaluation restarted from a clean
worktree.

## Internal-validation evidence

| Seed | Recall | FPR | AP |
|---:|---:|---:|---:|
| 101 | 0.8337 | 0.0489 | 0.8220 |
| 202 | 0.8317 | 0.0487 | 0.7964 |
| 303 | 0.8257 | 0.0491 | 0.8118 |
| 404 | 0.8475 | 0.0487 | 0.8280 |
| 505 | 0.8198 | 0.0489 | 0.7938 |
| Mean | 0.8317 | 0.0489 | 0.8104 |

Seed 505 ultimately selected epoch 27, not the provisional epoch 8 discussed during training. Its
final checkpoint achieved validation recall 0.8198, FPR 0.0489, AP 0.7938, AUROC 0.9270, and pixel
Dice 0.5488. Proposal calibration selected neural weight 1.0; proposal-only scoring slightly reduced
FPR but lost four plume detections and substantially reduced AP.

## Once-only strict results

| Model | Recall | FPR | AP | AUROC | Pixel IoU | Pixel Dice |
|---|---:|---:|---:|---:|---:|---:|
| Released MARS-S2L | 0.6418 | 0.0948 | 0.3521 | 0.8175 | 0.1329 | 0.2346 |
| ERSRR seed 101 | 0.3582 | 0.0510 | 0.1178 | 0.7624 | 0.0819 | 0.1513 |
| ERSRR seed 202 | 0.2985 | 0.0298 | 0.1948 | 0.7436 | 0.0966 | 0.1763 |
| ERSRR seed 303 | 0.2687 | 0.0242 | 0.1116 | 0.7523 | 0.0794 | 0.1471 |
| ERSRR seed 404 | 0.3881 | 0.0438 | 0.1968 | 0.7326 | 0.0653 | 0.1226 |
| ERSRR seed 505 | 0.2836 | 0.0346 | 0.0964 | 0.7269 | 0.1062 | 0.1919 |
| ERSRR seed mean | 0.3194 | 0.0367 | 0.1435 | 0.7436 | 0.0859 | 0.1578 |

The paired campaign found:

- recall delta: -0.3146, 95% CI [-0.4220, -0.1644];
- FPR delta: -0.0580, 95% CI [-0.0771, -0.0392];
- relative FPR reduction: 0.6115, 95% CI [0.4671, 0.7278];
- AP delta: -0.2161, 95% CI [-0.4261, -0.0645];
- AUROC delta: -0.0874, 95% CI [-0.2142, -0.0109].

ERSRR passed the mean-FPR, mean-specificity, and relative-FPR-reduction criteria. It failed the
absolute recall and recall-noninferiority criteria. The promotion gate therefore failed.

## Relationship to published MARS-S2L numbers

The MARS-S2L paper reports recall 0.7915, FPR 0.0713, AP 0.6408, and pixel IoU 0.3224 on its full
official test; its unseen/test-only-sites subset reports recall 0.7753, FPR 0.0763, and AP 0.4496.
Those values are context, not paired comparators. On the ERSRR strict cohort, the same released
checkpoint achieved recall 0.6418, FPR 0.0948, AP 0.3521, and IoU 0.1329. Both systems therefore
show substantial spatial-transfer degradation, but MARS-S2L remains stronger on plume recovery and
ranking on the paired cohort.

## Supported interpretation

1. ERSRR v3 is a conservative detector: it materially reduces false alarms but rejects too many
   true plumes for a useful monitoring operating point.
2. Random-seed internal validation was optimistic for geographically unseen groups. Mean recall
   fell by 51.2 percentage points and mean AP by 66.7 points from validation to strict evaluation.
3. The gap is not a single unlucky seed; every seed exhibits the same direction of failure.
4. The released MARS-S2L checkpoint also suffers a hard-cohort transfer gap, so the strict cohort is
   meaningfully more difficult than its paper-wide test, but this does not rescue ERSRR's claim.
5. Positive-only EMIT confirmation cannot repair the failed no-plume/recall comparison. It remains a
   distinct sensor-transfer stress test and must not be used to report FPR or specificity.

## Interpretation boundaries

- Do not select seed 404 as the model because it had the best strict recall; all five seeds are the
  estimand and strict behavior cannot select a seed.
- Do not lower thresholds, alter proposal weights, or tune postprocessing from these results.
- Do not compare the ERSRR strict point estimate directly to MARS-S2L's different-cohort paper table
  as if it were paired evidence.
- Do not claim operational methane concentration or flux retrieval.
- Any subgroup analysis of strict predictions is exploratory and must be labeled post hoc.
- Any v4 model informed by this campaign needs a newly untouched final test cohort.

## Frozen post-hoc diagnostic

`reports/experiments/mars_v3_strict_posthoc_diagnostic.json` describes the already-frozen scene
decisions and is explicitly exploratory. It does not select a new threshold, seed, model, or rule.

- 36/67 plumes were missed by all five ERSRR seeds; only 13/67 were detected by all five.
- 15 plumes were detected by released MARS-S2L but missed by every ERSRR seed; 21 were missed by
  both the released baseline and every ERSRR seed.
- 451/4,334 no-plume scenes were flagged by at least one ERSRR seed, but only 14 were flagged by all
  five. Most ERSRR false alarms are therefore seed-unstable rather than consensus errors.
- Mean ERSRR recall was 0.304 for the smallest plume-area third, 0.145 for the middle third, and
  0.509 for the largest third. MARS-S2L achieved 0.522, 0.545, and 0.864, respectively. The
  non-monotonic ERSRR middle-bin result is descriptive and may reflect geography/site confounding.
- ERSRR recall was highest in the middle wind-speed third (0.464) and lower in the low/high thirds
  (0.243/0.255). This is a hypothesis for data stratification, not a powered wind-effect claim.
- False alarms rose with longer target/reference intervals: ERSRR mean FPR increased from 0.029 in
  the shortest third to 0.046 in the longest, while MARS-S2L increased from 0.076 to 0.126.
- Several of the largest consensus false negatives were clear, baseline-detected plumes for which
  the ERSRR five-seed mean score was near zero. That pattern cannot be repaired by a modest threshold
  change and supports prioritizing representation/domain generalization over further postprocessing.

The diagnostic JSON also freezes deterministic sample ids for consensus false negatives, consensus
true positives, persistent false positives, and baseline-only true positives. Selection is based on
plume area, seed-hit count, frozen score, and sample-id tie-breaking—never visual preference.

## Evidence-based v4 research path

The next study should address spatial generalization before increasing architecture complexity:

1. Replace the single internal validation role with nested geographic group folds. Generate
   out-of-fold predictions for threshold selection and calibration, and report fold dispersion.
2. Build training batches and losses around site/region balance, plume-size balance, and hard
   negative taxonomy rather than scene count alone. Preserve uncertain and unobservable states.
3. Run post-hoc diagnostics on the retired v3 strict predictions by plume size, geography,
   target/reference interval, cloud/observable fraction, wind speed, and score calibration. Use
   these only to define hypotheses and data-acquisition strata for v4.
4. Compare a frozen released MARS-S2L encoder/head, a pretrained Sentinel-2 foundation encoder, and
   the current U-Net under the same nested-fold protocol. Architecture choice must be made from
   development folds, not this strict cohort.
5. Separate ranking from the final operating policy: optimize/calibrate scene presence out of fold,
   evaluate segmentation conditionally, and treat quality abstention as a prespecified coverage-risk
   curve rather than a rescue threshold.
6. Acquire a new, geographically isolated no-plume/plume confirmation cohort before v4 training is
   finalized. Seal it prediction-blind and use it exactly once.

The goal for v4 is not merely lower FPR. It is to move the entire recall-FPR/PR frontier outward and
demonstrate that movement with paired geographic uncertainty on an untouched cohort.

## Paper asset map and remaining work

- Primary aggregate: `reports/experiments/mars_v3_strict_campaign.json`.
- Released same-cohort baseline: `reports/experiments/mars_released_model_full_strict_baseline.json`.
- Per-seed strict reports: `reports/experiments/mars_v3_seed{101,202,303,404,505}_strict_evaluation.json`.
- Frozen exploratory diagnostic: `reports/experiments/mars_v3_strict_posthoc_diagnostic.json`.
- Claim and manuscript structure: `docs/PAPER_OUTLINE.md`.
- Architecture decision and evaluation contract: `docs/ARCHITECTURE_DECISION_V3.md`.
- Data/acquisition chronology: `docs/PUBLICATION_ROADMAP.md` and `reports/acquisition/`.
- Verification: 50/50 tests passed in the project WSL environment on 2026-07-12; the report's
  inline script also passed a standalone JavaScript parse check.

Still required for the final package:

1. complete ERA5-Land wind retrieval after a user-managed Copernicus CDS token is configured;
2. run the sealed 55-scene EMIT positive-only confirmation exactly once;
3. convert the deterministic error-atlas selections into manuscript figures while retaining their
   post-hoc labels;
4. visually inspect the generated HTML in a local browser (automated browser preview was blocked
   by the local-file URL policy; structural and content checks are automated);
5. run the final requirement-by-requirement audit.

## 2026-07-31: physics-contrast Gaussian ViT full-bank audit

The successor path now uses a 10.58 M-parameter Vision Transformer with a convolutional decoder.
Its external contract remains the exact 16-channel mixed Sentinel-2/Landsat MARS-S2L input. Inside
the network, fixed formulas add MBMP evidence, six temporal log changes, all 15 temporal log-ratio
changes, wind, cloud, observability, and a learned sensor embedding. No cohort or test statistics are
used by those formulas.

Synthetic positives are not Gaussian noise images. A physically parameterized anisotropic Gaussian
plume is injected into real MARS backgrounds through the affected spectral channels. Every positive
has a no-plume twin with the same source background and augmentation. Independent deterministic
random streams control background selection, plume parameters, and augmentation; a paired sampler
keeps twins adjacent while covering each template once per epoch. A direct audit confirmed identical
backgrounds, zero protected-channel change, changes only in channels 0, 5, and 6, a nonempty positive
mask, and an empty negative mask.

After two deliberately small learnability studies, the schedule was frozen by a full-bank audit:

- 16,000 training templates, producing 32,000 matched positive/negative requests per epoch;
- 2,000 separately seeded validation templates, producing 4,000 requests;
- 10 epochs, batch size 24, AdamW learning rate 0.001, BF16;
- 320,000 total training requests;
- checkpoint selection by validation mask IoU, then AP, then earliest epoch;
- gates fixed at AP >= 0.80, IoU >= 0.25, and pixel recall >= 0.30.

Epoch 9 was selected with dense-evidence AP 0.927611, AUROC 0.911105, mask IoU 0.261063, and pixel
recall 0.579037. Epoch 10 was lower on the primary selection metric (IoU 0.241686). All three frozen
gates passed. Peak CUDA allocation was 4.30 GB. The compact result is
`reports/experiments/mars_gaussian_contrast_full_bank.json` with SHA-256
`689b806228abda20e433af180f25a2e722bd71a1a2c213a9fb51c0f082f162f3`.

This result establishes synthetic representation learnability only. It is not a comparison with
MARS-S2L and cannot support a paper-performance claim. The selected nine-epoch pretraining schedule
must now be applied independently to the two authorized real development endpoints (held folds 3
and 4), followed by the already-frozen paired site bootstrap gates. Folds 0, 1, 2 and the official
2024 test remain unavailable for architecture selection.

## 2026-08-01: physics-contrast Gaussian ViT real-development cross-fit

The nine-epoch full-bank schedule was applied independently in both authorized development
directions. The model holding fold 3 fit only fold 4 (8,946 real scenes; 7,872 eligible synthetic
backgrounds; seed 20268303). The model holding fold 4 fit only fold 3 (8,799 real scenes; 7,728
eligible backgrounds; seed 20268304). Each endpoint consumed 288,000 paired synthetic pretraining
requests and 12,288 fixed joint real/synthetic requests. Final pretraining losses were 1.274510 and
1.348589; final joint losses were 1.827474 and 1.833775.

The first execution attempt stopped after endpoint-1 pretraining and before any held prediction
because a fully unobservable real crop exposed a numerical defect in the scene reducer: masked
pixels used BF16's finite minimum value, which overflowed the scene head. The reducer was changed to
use negative infinity, discard non-finite top-k entries, and divide by the valid selected count.
The exact repair, zero-observability regression test, lack of outcome exposure, and new hashes were
recorded in the frozen protocol and committed before restarting. Seven focused tests and the full
smoke passed. A later desktop-wrapper timeout killed a second attempt during pretraining, also before
joint training or held prediction; the final clean run started from scratch in an independent WSL
session and reused no partial state.

The clean 17,745-scene cross-fit rejected the architecture. The current spatial-Prithvi baseline had
AP 0.904076, recall 0.961942 at FPR 0.071266, and pixel IoU 0.595146. The predeclared rank selected
fusion strength 0.05, which produced:

- AP 0.904345, delta +0.000269, paired-site 95% CI [-0.000467, +0.000797];
- recall 0.961286 at the same FPR, delta -0.000656;
- Landsat AP delta -0.000029 and Sentinel-2 AP delta +0.000412;
- pixel IoU 0.596406, delta +0.001260, paired-site 95% CI
  [-0.000248, +0.002782].

Strength 0.10 was the most useful diagnostic candidate: it preserved pooled recall and FPR and
achieved pixel-IoU delta +0.001806 with a strictly positive paired-site lower bound (+0.000251), but
its AP delta was only +0.000326 and its paired AP lower bound was -0.001077. Strengths 0.25, 0.50,
and 1.00 progressively degraded AP; the two strongest also degraded IoU. The selected candidate
failed the minimum AP, matched-recall, each-sensor AP, paired AP, and paired IoU gates. No model
artifact was written, and fold 2, folds 0/1, and the official test stayed closed.

This experiment supports a narrow conclusion: physics-contrast Gaussian pretraining transfers a
small amount of dense localization information to real scenes, but its learned scene head is not
complementary enough to the high-performing spatial-Prithvi ranking baseline. The next development
path should preserve the validated dense branch while replacing scene fusion with real-scene,
site-balanced hard-negative and error-correcting evidence. More Gaussian exposure or stronger
fusion is contradicted by the observed strength curve.

Compact evidence: `reports/experiments/mars_gaussian_contrast_crossfit.json` (SHA-256
`728e9a1f69a608cf09816114ed50fc785953f6dc5937ddae370e87f990d85f55`).

## 2026-08-01: robust temporal site templates rejected

The next unseen-site hypothesis replaced the earlier label-free pixelwise site mean with per-site
q25, median, and q75 maps. Two fixed feature contracts used either an IQR-normalized median anomaly
or explicit original/median/residual maps, followed by the established compact spatial CNN. Four
cross-fitted rankers and five small blends were frozen at commit `37c7de00` before fold-3/fold-4
outcomes were computed. The experiment covered 17,745 scenes and 250 physical sites; the median site
history contained 13 observations (range 1-714).

No candidate supplied complementary ranking. The largest pooled AP change anywhere in the fixed
grid was only +0.000019, but its matched-FPR recall changed -0.002625 and its paired-site AP interval
lower bound was -0.000710. The preregistered worst-fold-first selector chose the group-weighted
IQR-normalized model at blend 0.05; it changed AP -0.000099, recall -0.001969, and had paired-site AP
interval lower bound -0.000787. Increasing blend strength worsened AP monotonically, reaching
-0.016126 in the strongest site-cell median candidate. No artifact was written; fold 2, folds 0/1,
external-development outcomes, and official-test outcomes were not accessed.

This result retires robust same-site median/IQR templating as an isolated scene-ranking lever. Both
mean and robust site histories can improve or preserve spatial context, but neither produces the
large, site-general anomaly ordering needed for the official test-only AP/recall confidence gates.
Compact evidence: `reports/experiments/mars_robust_site_template_ranker.json` (SHA-256
`16a3928c9ea7dcc3f25e2ba3c62fededc231655e0436c3d35d6eb2e4006e8a88`).

## 2026-08-01: current UNEP catalog refresh and append-only correction

The official UNEP/IMEO detected-plume CSV and GeoJSON archives were downloaded again and audited
with the already frozen exact-product and 25 km paper-test exclusion rules. The 27,403-row catalog
contains 269 eligible positives from 43 physical groups: 178 Sentinel-2 and 91 Landsat. The closest
eligible location remains 25.631477 km from an official paper-test location.

Relative to the July audit, eligible rows increase from 237 to 269. A raw full-catalog regrouping
initially reported 216 auxiliary / 40 development / 13 sealed, but catalog growth had changed 36
component hashes and would have reassigned 18 immutable July auxiliary identities. An append-only
reconciler now freezes every prior identity, group, and role, inherits the one prior role for an
expanded component, and quarantines ambiguous multi-group merges. The corrected split is 247
auxiliary / 9 development / 13 sealed: all 32 additions are auxiliary, with zero prior identity
changes and zero quarantines.

Exact resolution produced 52 new auxiliary candidates (26 Sentinel-2 and 26 Landsat). All 26
Sentinel-2 crops were acquired; CloudSEN12 retained 25 and rejected one. The verified real-positive
training manifest therefore grows from 135 scenes / 27 groups to 160 scenes / 28 groups, SHA-256
`8c42b4350ccc0abbd2fec727abba435fca2af8f29289e837d52e7151553d861b`. The 26 Landsat scenes remain
pending USGS EROS authentication, with no product substitution. No imagery was added to Git.
Compact provenance and archive hashes are recorded in
`reports/acquisition/unep_mars_post2024_refresh_20260801.json`.

## 2026-08-01: expanded UNEP controlled transfer

The 160-row UNEP manifest was substituted into the previously frozen protected bi-sensor
fine-tune with every model, fold, seed, training, sampling, fusion, bootstrap, and gate setting
unchanged. On all 17,745 fold-3/fold-4 rows, selected strength 0.50 changes AP +0.000941,
matched-FPR recall/FPR exactly zero, and IoU +0.006395. Paired-site intervals cross zero for AP
[-0.000034,+0.002236] and IoU [-0.002502,+0.013529]. The conservative strength 0.10 has positive
AP and IoU lower bounds but only +0.000257 AP, well below the fixed +0.002 floor. The branch is
rejected with no artifact. Compact result SHA-256:
`3e70513986192dda3b7aadc8feed6a5b12ce6e11eaaef2691e9deb7859efc96c`.

## 2026-08-01: protected independent-signal ensemble passes selection

A preregistered six-candidate local-logit ensemble combines the previously deployment-safe
DOFA-v2 residual with the expanded-UNEP anchored-U-Net residual above a final 0.25 protection
gate. Selected anchored strength 0.50 × multiplier 1.0 improves folds-3/4 AP +0.002230 with
paired-site 95% interval [+0.000678,+0.004151]. Fold AP changes +0.001143/+0.001682 and
Sentinel-2/Landsat changes +0.002246/+0.001396. Operating recall/FPR counts remain exact. The
separately fixed strength-0.10 dense path improves IoU +0.002631 with paired lower +0.000369 and
positive changes in both folds. Every selection gate passes; only a fixed fold-2 confirmation is
authorized. Compact result SHA-256:
`85d5959061cc77d4b2a32f12609c7852af545ff8df6923273d6ae245d020eec1`.

## 2026-08-01: weight-anchored released-U-Net fine-tune rejected

The next external-transfer experiment tested a mechanism not covered by the two frozen-teacher
NDMI adapters: direct movement of all 13,579,393 released-U-Net convolutional parameters. An
immutable released teacher supplied exact baseline logits; the student began at the identical
checkpoint, retained all released BatchNorm affine values and running state, used discriminative
1e-5/5e-5 learning rates, and incurred normalized L2-SP plus logit-direction penalties. Every
endpoint mixed 75% opposite-fold MARS requests with 12.5% exact-L1C UNEP positives and 12.5%
CloudSEN negatives. BF16, three epochs, four interpolation strengths, and all gates were committed
at `8733b8d0` before held scoring.

Both cross-fit endpoints completed, covering 17,745 scenes. At the selected weakest strength 0.05,
the candidate changed AP -0.000448, matched-FPR recall -0.001312, and pixel IoU +0.001431. The
paired-site 95% intervals were [-0.001151, +0.000003] for AP and [+0.000223, +0.002352] for IoU.
Fold-3 AP/recall/IoU changed +0.000063/+0.001319/+0.002506, while fold 4 changed
-0.000554/-0.002611/+0.000629. No artifact was written.

The strength curve separates the tasks. Sentinel-2-only AP improved monotonically from +0.000180
to +0.000925, and pooled IoU improved from +0.001431 to +0.006284. Nevertheless pooled AP worsened
monotonically to -0.004205 because modifying only Sentinel-2 absolute scores disturbed their
calibration against unchanged Landsat scores; fold-4 recall also remained adverse. The experiment
therefore supports direct external fine-tuning as dense localization evidence, not as the missing
complementary scene-ranking solution. Any reuse must preserve cross-sensor score calibration and
pair the dense branch with stronger site-general scene evidence. Compact result:
`reports/experiments/mars_anchored_full_finetune_pilot.json` (SHA-256
`1d870c08e27111859e03d10c0479bf15297e19203c4837f687027dd4b7f0cba1`).

## 2026-08-01: protected bi-sensor anchored fusion rejected

The parent result showed positive within-Sentinel ranking and dense evidence but adverse pooled
calibration. A single mechanistic follow-up retained the same model, data, losses, and parent seeds,
then changed only scene fusion. Scores below 0.25 remained exact current-score identities; scores
above the gate were locally reranked and mapped back above 0.25. Teacher-student evidence applied
to both Sentinel-2 and Landsat. This structurally preserves the development operating confusion
counts. Four strengths were frozen at commit `060859fd` before the reused folds were rescored.

The routing correction works but is insufficient. Every strength has positive AP on both folds and
both sensors with exactly unchanged recall and FPR. At strength 0.10, AP changes +0.000237 with
paired-site interval [+0.000022, +0.000542], while IoU changes +0.002698 with interval
[+0.000410, +0.004497]; it passes every gate except the preregistered +0.002 AP point floor.
The selector chooses strength 0.50 by its worst-fold-first rank: AP +0.000881, recall 0, and IoU
+0.006130, but both paired intervals cross zero and fold-4 IoU changes -0.000190. Strength 1.0
raises AP to +0.001220 but remains below the point floor, has AP interval
[-0.000436, +0.003151], and changes IoU -0.000642.

Thus protected bi-sensor routing resolves the parent calibration reversal and establishes a small,
direction-stable complementary scene signal. It does not supply the magnitude or joint uncertainty
needed alone. No artifact was written. A future ensemble may use only independently confirmed
high-confidence ranking evidence and a separately conservative dense strength; simply increasing
this branch is contradicted by the IoU curve. Compact result:
`reports/experiments/mars_protected_bisensor_finetune_pilot.json` (SHA-256
`0c3e2f2253316c714f1b58f39092bf6cc9e5b445823a46be27a5b1e1ebd818af`).

## 2026-08-01: protected independent-signal ensemble fails fold-2 confirmation

The exact strength-0.50 anchored residual plus fixed DOFA ensemble was frozen and committed before
a new endpoint fit folds 3+4 and scored all 8,833 fold-2 scenes. AP improved only +0.000313 over
the current spatial-Prithvi comparator, with paired-site 95% interval
[-0.000541,+0.001131]. The complete operating confusion matrix was unchanged. The independent
dense strength-0.10 path did generalize, improving IoU +0.004367 with paired lower +0.002352.
Scene-ranking magnitude and certainty failed, so no model was promoted and the official test was
not reopened. Compact result SHA-256:
`4a55afb2dad0306f56ebad82289a06157f3855be18a22a4a3e567395c60fa138`.

## 2026-08-01: Gaussian scene-supervision ablation

The earlier full-bank Gaussian ViT trained dense masks but disabled scene loss throughout its
288,000-request synthetic phase. A controlled same-seed ablation enabled paired synthetic scene
BCE and partial-AUC while retaining the 16,000-template bank, 10,575,102-parameter mixed-sensor
ViT-U-Net, nine pretraining epochs, three joint epochs, and real-only folds-3/4 evaluation.
Synthetic scene BCE fell 0.639803 to 0.379344 and 0.656790 to 0.412188 in the two directions.

Strength 0.25 improved pooled AP +0.002507, fold AP +0.002487/+0.001137, and
Sentinel-2/Landsat AP +0.003459/+0.001538. Pixel IoU improved +0.002996. It was rejected because
paired-site intervals crossed zero for AP [-0.001290,+0.005322] and IoU
[-0.000915,+0.005762], and fold-4 matched-FPR recall lost one positive. Conservative strengths
0.05/0.10 retained positive paired-site AP lower bounds but produced only +0.000949/+0.001597 AP,
below the fixed +0.002 floor. The finding supports conservative Gaussian scene evidence as an
ensemble component, not a standalone higher-weight successor. Compact result SHA-256:
`6963e5ac8832011a4c39899d3097ff23afb3222ab6f1d2bbbc67cdc325190358`.

## 2026-08-01: exact replay rejected; stochastic reproducibility made explicit

Two executions of the exact raw-logit replay produced no accepted cache. The
captured diagnostic run completed both endpoints in 4,866.9 seconds, preserved
matched-FPR recall within 1e-9, but failed 1e-9 equality for pooled and sensor
AP at strength 0.05. Endpoint training losses also differed slightly from the
reference, establishing CUDA/BF16/worker nondeterminism. The original frozen
protocol is not loosened; its rejection is recorded in
`reports/experiments/mars_gaussian_scene_aligned_cache_replay_failure.json`.

A separate bounded-reproducibility protocol now freezes a 0.001 AP-delta
tolerance before its cache exists and requires positive pooled, fold, and
sensor AP plus non-worse recall per strength. The protected ensemble verifies
both the compact replicate receipt and its cache hash, and scores only strengths
marked eligible. No fold-2 or official-test outcome was accessed.

## 2026-08-01: conservative Gaussian-ViT + DOFA ensemble frozen before cache

The next development-only test combines the fixed DOFA residual with Gaussian
scene strengths 0.05/0.10 above an exact 0.25 protection gate. These are the
only Gaussian strengths whose completed standalone result had unchanged pooled
recall and a positive paired-site AP lower bound. The exact-replay protocol is
`configs/mars_gaussian_scene_aligned_cache_replay_protocol.json`; it repeats
the deterministic folds-3/4 training path and must reproduce the committed
standalone deltas within 1e-9 before retaining raw scene logits. The ensemble
protocol is `configs/mars_dofa_gaussian_protected_ensemble_protocol.json`.

Selection requires AP delta at least +0.002, unchanged operating counts,
positive AP in each fold and sensor, positive paired-site AP lower bound, and
the already fixed positive dense IoU evidence at anchored strength 0.10. Both
protocols precede the future ignored cache. Fold 2, the separate fold-0/fold-1
files, external outcomes, and the official test are not selection inputs.

## 2026-08-01: bounded Gaussian replicate passes reproducibility gates

The 17,745-row replicate retained both fixed strengths. Strengths 0.05/0.10
improve AP +0.000689/+0.001154, preserve matched-FPR recall exactly, and have
positive AP in both folds and both sensors. Their pooled AP deviations from the
reference are -0.000261/-0.000443, within the frozen 0.001 tolerance. Raw
logits remain ignored; cache SHA-256 is
`68aaf9d5b0e6d650bfc70d8ea511f5a4d3838bbc353811ce11462214f96faa02`.
The compact receipt and cache hashes are bound into the already preregistered
Gaussian+DOFA evaluator before selection.

## 2026-08-01: protected Gaussian-ViT + DOFA ensemble passes development

Selected Gaussian strength 0.10 plus the fixed DOFA residual improves
folds-3/4 AP +0.002449 with paired-site interval
[+0.000489,+0.004068]. Fold changes are +0.001722/+0.001968 and
Sentinel-2/Landsat changes +0.002942/+0.001692. Operating counts are identical.
Versus released MARS-S2L, AP is +0.031241 and matched-FPR recall +0.009843.
The fixed dense path retains positive IoU point/lower bounds on development and
the earlier fold-2 dense confirmation. All gates pass; compact result SHA-256:
`9473bc039e43bfc15d97b8eca0228fce66b31dffc8c487f8b3a4351c3280e1b5`.
Only a separately frozen external/new-cohort confirmation is authorized.

## V6 architecture decision and preregistered viability test

The first v6 implementation uses separate rank-8 Prithvi-Tiny-TL adapters for
scene ranking and dense segmentation. A fixed source-specific transform first
restores physical reflectance, after which bounded learned L1C/L2A/Landsat
corrections, product/sensor embeddings, and explicit missing-reference flags
handle product semantics. Separate adapters and harmonizers prevent the
validated Gaussian dense objective from changing the scene representation.

The unified cohort audit binds 33,131 authorized training rows and a 48-group,
11,478-row untouched MethaneS2CM source-development resource. That resource is
prepartitioned into 24 calibration and 24 confirmation groups. The risk method
controls the expected within-group negative-crop FPR under exchangeability via
the finite-sample CRC correction; shifted-product/geography results are
transport tests, not guarantees.

Before dense-v6 training, a lower-cost scene viability gate compares the new
residual directly with the passed Gaussian+DOFA champion on cross-fit folds
3/4. It must improve AP by at least 0.003 with positive fold, sensor, and paired
site-bootstrap evidence and non-worse matched-FPR recall. A pass requires a
second seed; a failure retires this schedule without opening any external or
official outcome.

## 2026-08-01: v6 unified scene schedule rejected at the viability gate

The frozen one-seed cross-fit completed both four-epoch endpoints. Training
loss declined in both directions, but the learned scene residual added only
+0.000182 AP to the Gaussian+DOFA champion at the selected strength 0.05.
Fold-3/fold-4 AP changes were +0.000161/+0.000007 and Sentinel-2/Landsat
changes were +0.000295/+0.000064. Matched-FPR recall and the full operating
confusion matrix were unchanged. The 10,000-replicate paired 25 km group
interval was [-0.000107,+0.000427].

The result fails both the frozen +0.003 AP floor and the strictly positive
paired-site lower-bound gate. No second seed, dense-v6 training, external
confirmation outcome, fold 2, or official-test outcome is authorized by this
experiment. The specific product-aware rank-8 scene-adapter schedule is
retired. This negative result strengthens the conclusion that simply adding
the auxiliary cohorts to a protected residual learner does not break the
current scene-ranking ceiling.

## 2026-08-01: v6.1 error-correcting residual pilot frozen

The v6 rejection revealed that its MARS training path itself bounded the
candidate to roughly +/-0.2 local log-odds at the fixed strength 0.10. Focal
and partial-AUC losses therefore optimized only tiny changes around an already
strong champion. V6.1 keeps the identical product-aware rank-8 Prithvi scene
architecture and cohorts but fits an unconstrained additive residual to the
champion logit. The exact 0.25 protection gate is applied only at inference:
scores below it remain bitwise identical and changed scores map back above it.

The one-seed protocol searches protected strengths 0.10/0.25/0.50/1.00 and
retains the same +0.003 AP, positive fold/sensor, positive paired-site lower
bound, and non-worse matched-FPR recall gates. A real-checkpoint mixed-source
smoke is finite (loss 0.254646, gradient norm 1.622571, peak CUDA 214,051,840
bytes) and accessed no held outcome. Trainer, dependency, input, and smoke
hashes are frozen before the new cross-fit.

## 2026-08-01: v6.1 unconstrained error correction rejected

The corrected objective converged and produced substantially larger residuals,
but every preregistered inference strength worsened AP. At strength 0.10, AP
changed -0.001376 with paired-site interval [-0.002920,-0.000085]; fold-3 and
fold-4 changes were -0.000145/-0.000609, and both Sentinel-2 and Landsat
regressed. Strengths 0.25/0.50/1.00 degraded monotonically by
-0.003785/-0.008486/-0.021014 AP. The protection gate preserved matched-FPR
recall exactly at all strengths.

This falsifies the explanation that the earlier v6 gain was limited only by
its bounded training gradient. On this cohort, stronger product-aware adapter
corrections are anti-complementary to the Gaussian+DOFA champion. V6.1 is
retired without a second seed, dense training, or external outcome access.
The ignored raw-logit cache is retained by SHA-256 in the compact report for
reproducibility and future non-outcome diagnostics.

## 2026-08-01: 100M Prithvi representation path acquired and frozen

After rejecting both v6 scene objectives, the next test changes the foundation
representation rather than the residual head. The official
Prithvi-EO-2.0-100M-TL checkpoint was downloaded at immutable revision
`2c84e383194986040f883cc43d7869002c425e1b`. Its 454,660,610-byte checkpoint
strictly loads 112,639,492 parameters (86,237,186 encoder parameters) with no
missing or unexpected keys. All model files remain ignored.

The frozen development extractor uses the same two-frame physical-reflectance
contract as the tiny model, retains only block 3/6/9/12 CLS tokens, and emits
3,072 float16 features per scene. A 64-row real-scene smoke passed. Full
extraction is restricted to folds 3/4 and is label-independent; the MARS
broad-NIR to HLS narrow-NIR mismatch remains explicitly disclosed. Candidate
probe/fusion choices will be frozen separately after the cache identity exists
and before any model score is computed.

## 2026-08-01: 100M Prithvi CLS probe frozen before outcomes

The completed 102,807,297-byte cache contains exactly 17,745 unique rows and
3,072 finite features; identities, labels, sensors, folds, and groups align
one-to-one with the Gaussian+DOFA champion. Cache SHA-256 is
`9758449fd8ca580a31771a1929314d99eabe2ad0822c902ad48ae96226f3944b`.

The probe receives only frozen 100M CLS features. It uses equal label mass and
equal physical-site mass within label, separate source and unlabeled-target
feature moments, and L2 logistic regression. The frozen grid is C
0.001/0.003/0.01 crossed with champion logit blends
0.025/0.05/0.10/0.20. Selection prioritizes worst-fold AP before pooled AP;
promotion requires +0.003 pooled AP, positive AP in both folds and sensors, a
positive 10,000-replicate paired-site lower bound, and non-worse matched-FPR
recall. A pass authorizes only separate folds-0/1 extraction/confirmation.

## 2026-08-01: 100M Prithvi CLS probe rejected

All six cross-fold logistic fits converged within 203 iterations. Every one of
the 12 fixed fusions reduced pooled AP, and degradation increased with blend
strength. The selected C=0.001, blend=0.025 candidate changed AP -0.000288,
matched-FPR recall -0.000656 (one positive), fold-3/fold-4 AP
-0.000084/-0.000433, Sentinel-2 AP -0.000451, and Landsat AP +0.000114. Its
paired-site interval was [-0.000666,+0.000083].

The 100M representation therefore fails every promotion gate. Larger frozen
foundation capacity does not add complementary scene-ranking information to
the champion under this two-frame HLS-transfer contract. Folds 0/1, fold 2,
external held cohorts, and the official test remain unopened. The cache is
retained as an ignored negative-result artifact; no classifier artifact is
promoted.

## 2026-08-11: fixed-split group-CRC calibration transport rejected

The retired v6 scene family was not retrained. Instead, the already frozen
MethaneS2CM source-development partition was used for a calibration-only test:
24 fixed 25 km groups (5,050 crops) fit a group-balanced conformal-risk
threshold, and the disjoint 24 groups (6,428 crops) were evaluated once. This
is candidate-specific development evidence, not untouched confirmation,
because the underlying v5.1 development labels were opened previously and the
confirmation label marginals were exposed during prerequisite alignment.

At primary alpha 0.075, CRC froze threshold 0.787820 with corrected calibration
risk 0.074981. On the fixed evaluation groups it achieved crop FPR 0.0336 and
recall 0.3022, versus 0.0531 FPR and 0.4310 recall for the pooled 5% calibration
threshold. Group-balanced FPR improved by -0.0210 with paired 95% interval
[-0.0308,-0.0126], but group-balanced recall fell -0.1661 with interval
[-0.2084,-0.1255]. The frozen 0.35 absolute-recall and -0.05 paired-recall
noninferiority gates both failed.

Group CRC is therefore rejected as operationally useful under this fixed split.
Secondary alpha values may not be selected from these outcomes. No model
weights, ranking, dense masks, opened MethaneS2CM location-test outcomes, MARS
test outcomes, or other prohibited evidence changed or entered the experiment.
The machine-readable result is
`reports/experiments/methanes2cm_v5_1_group_crc_transport.json`.

## 2026-08-16: released-sensitivity rescue is feasible but low precision

The frozen Stanford controlled-release failures motivated a development-only
complementarity audit on MARS folds 3/4. At independently matched 7.13% FPR,
Gaussian+DOFA uniquely detects 33 positives and released MARS uniquely detects
nine; both unique cells contain 220 negatives. The released paper rule identifies
11 champion-missed positives but also 446 negatives. This rules out an unconditional
OR while proving that the released model retains sensitivity absent from the current
champion.

Within that rescue region, frozen model-aligned features separate true from false
rescues: the largest absolute standardized contrast is 1.384, above the predeclared
0.25 feasibility floor. All three diagnostic gates pass, authorizing a constrained
dual-expert pilot that may only raise champion scores when both released and primary
dense evidence agree. This is folds-3/4 architecture evidence, not external or
official confirmation. Result:
`reports/experiments/mars_recall_anchor_diagnostic.json`.

The separately frozen deterministic follow-up tested nine bounded consensus
rescues and rejected all of them. The safest candidate raised 19 positives but
562 negatives, reducing AP by 0.000347 and matched-FPR recall by 0.004593; its
paired group AP interval was [-0.001139,+0.000689]. Both folds regressed. This
rejection means released sensitivity cannot be restored by an unconditional
OR/max rule; any continuation must explicitly verify rescue candidates using a
cross-fitted, model-aligned gate or change the representation itself.

The separately preregistered cross-fitted verifier now rejects the first of
those continuations. A small transformer trained on nine full-spatial evidence
maps reached held-proposal AP 0.915--0.966, yet its conservative rescue raised
42 negatives and zero positives. Pooled AP changed -0.000024, matched-FPR recall
was unchanged, both folds and both sensors regressed, and the paired 25 km-group
AP interval was [-0.000055,-0.000006]. The stronger rescue raised 98 negatives
and zero positives. No external or official replay was authorized.

The released-proposal branch is therefore closed. Another ranker or verifier on
the same frozen folds would repeat a saturated experiment family; a defensible
next attempt must add independently labeled geographic diversity or materially
new observational information rather than reweighting existing scores.

## 2026-08-16: independent-data audit redirects the architecture to reference sets

The official Zhao et al. ACP 2025 archive was downloaded and hash-verified, but
all 1,627 rendered images come from six sites overlapping both MARS development
and official-test geography within 0.216 km. They also lack the georeferenced
six-band/dense-mask contract. Zhao DSAN is retained as product-alignment
literature only; zero images are admitted to model development.

Project Eucalyptus provides a stronger, non-outcome architecture clue. Its
controlled-release discussion identifies methane-contaminated or mismatched
reference observations as a false-negative mechanism and recommends multiple
recent references, similarity selection, or learned attention. Its released
24-channel model is incompatible with MARS and is not loaded. The adopted path
instead changes the information supplied to the trusted released MARS model.

An outcome-blind folds-3/4 selector found exact-grid, strictly prior Sentinel-2
references for 14,569/14,963 rows (97.37%) and five references for 13,733
(91.78%). The original all-rows gate remains a recorded FAIL because 112 rows
with a same-key prior have only shifted grids. A separately preregistered
fallback-aware adjudication passes: reference coverage 97.3668%, five-reference
coverage 91.7797%, grid-exclusion fraction 0.7629%, with exact original-pair
fallback and no outcomes accessed.

The next experiment is therefore staged rather than assumed. First, the pinned
released U-Net scores the original plus five frozen prior-reference views while
holding target, wind, and target cloud fixed. A four-row real smoke reproduces
the original connected score within 0.000041 maximum absolute error with exact
paper-rule decisions. Only if a separately frozen folds-3/4 diagnostic finds
complementary positive signal will a small permutation-invariant reference-set
temporal transformer be cross-fitted. This avoids committing to another large
architecture before the new information source proves useful.

## 2026-08-16: reference-set temporal path rejected at the information gate

The hash-pinned released U-Net completed 17,745 original-plus-prior reference
sets without outcomes. Original-view parity passed on every row (maximum score
drift 0.001186, mean 0.0000475, exact paper decisions), so the subsequent
negative result is not an input-contract artifact.

After a separately committed protocol opened folds-3/4 outcomes, every fixed
prior-reference aggregation degraded sharply. Similarity weighting was least
harmful but still changed AP -0.0552 and matched-FPR recall -0.0617; nearest,
median, top-two, and maximum changed AP by -0.0951/-0.0633/-0.0728/-0.1044.
All fold/Sentinel-2 effects were negative and all paired 25 km-group AP
intervals were strictly below zero.

Max-over-priors recovered 25 new champion-missed positives across both folds
and 11 groups, but raised 3,009 negatives (0.824% precision). No fixed
aggregation improved AP, so large TP/FP feature contrasts cannot authorize a
learned verifier under the frozen rule. The temporal set transformer is retired
before training. This falsifies the hypothesis that selecting multiple visible/
NIR-matched prior scenes can break the current ranking ceiling through the
released detector; future work must add genuinely new labeled geography or a
different observation modality rather than another MARS-only reference head.

## 2026-08-16: MARS-Hyperspectral transfer acquisition fails one frozen gate

The pinned UNEP-IMEO MARS-Hyperspectral train archive was audited without
opening validation, test, full-tile, retrieval, or official MARS-S2L outcomes.
All 7,041 authoritative `plumemask.tif` files were retained under ignored
research storage (16.06 MB with optional metadata), and every mask was strictly
binary. Pixel truth yields 3,349 plume and 3,692 reviewed no-plume crops. Raster
georeferencing agrees with the 1,295 published crop coordinates to 0.012 km
median and 0.021 km maximum, validating mask-derived EMIT centers.

After the frozen >=75% clear and 25 km official-test exclusion, 2,345 crops
remain: 1,008 plume and 1,337 no-plume across 59 countries and 390 connected
25 km groups; 380 groups are beyond every MARS-S2L location. Public metadata
queries used exact Sentinel-2 L1C and Landsat 8/9 Collection 2 Level-1 products.
Same-sensor tiles from one acquisition were deduplicated per crop. Sentinel-2
alone provides 286 positive and 161 negative pairs; the combined catalogs
provide 500 positive, 272 reviewed negative, 460 <=1-hour, and 137 <=15-minute
pairs across 204 novel groups and 49 countries.

Four of five Stage B gates pass, but the preregistered minimum of 300 reviewed
negative pairs fails by 28. Landsat 7 is not admitted because the exact
MARS-S2L operational contract is LC08/LC09. The failed gate is not lowered:
no target bands, hyperspectral retrievals, model architecture, threshold, or
MARS outcome was opened. This cross-modal path is retired under the frozen
protocol until an independent reviewed-negative source supplies the missing
evidence. Compact results are in
`reports/acquisition/mars_hyperspectral_transfer_stage_b.json` and the
Sentinel-only ablation beside it.

## 2026-08-16: JPL CACH4 reviewed-negative supplement fails the unchanged location gate

The JPL operational-GHG release contributes 3,149 released CACH4 train
background tiles across 124 AVIRIS-NG flightlines. All 124 public JPL CMF ENVI
headers were retrieved without a raster download, and every tile has an exact
flight UTC and finite WGS84 crop center. The released sampler's edge padding is
preserved explicitly: 1,131 tiles overhang at least one source-image boundary,
but every crop center is in bounds. Filename offsets map to ENVI sample then
line, and centers use the GDAL rotated-ENVI affine convention.

The separately frozen eligibility filter excluded 390 rows within 25 km of
official MARS-S2L test geography and 194 rows within 25 km of an already-counted
MARS-Hyperspectral negative source crop. This leaves 2,565 rows and 104
flightlines, but only 15 transitive 25 km components. The unchanged requirement
was at least 20 independently located components, so the metadata stage fails.
No Sentinel-2/Landsat catalog, target asset, released JPL test content, or
protected MARS outcome was accessed. The source label must be described as
"no confidently detectable plume after expert review," not physical zero
methane. Authoritative result:
`reports/acquisition/jpl_operational_ghg_negative_supplement_metadata.json`.

## 2026-08-16: NASA header bridge frozen for untouched COVID/Permian cohorts

NASA ORNL DAAC collection `AVIRIS-NG_L1B_radiance_2095` (DOI
10.3334/ORNLDAAC/2095, CMR concept `C2662359874-ORNL_CLOUD`) publishes each
orthocorrected L1B product as an ENVI binary plus text header. CMR resolves the
known CACH4 anchor `ang20180821t184959` to the 14,074-byte
`ang20180821t184959_rdn_v2t1_img.hdr` granule and reports exact flight UTC,
WGS84 geometry, SHA-256, and the protected ORNL download URL.

Before opening that header, a new protocol froze an all-or-nothing geometric
bridge: at least 100 of 124 CACH4 anchors and at least 80% overall must resolve,
and every resolved NASA grid must match the already verified JPL CMF grid in
dimensions, WGS84 UTM CRS, pixel-vector length, ENVI reference convention, and
five control-point centers within 0.25 pixel. Only a PASS may geolocate the
13,444 untouched COVID/Permian train backgrounds. CACH4 is excluded entirely
from their unchanged >=20-component gate, preventing post-failure pooling.
Target satellite catalogs remain forbidden until a separate committed
protocol. The bridge is currently pending Earthdata authorization, not a pass
or fail. Frozen contract:
`configs/mars_jpl_ornl_header_bridge_protocol.json`.

## 2026-08-16: NASA CMR preflight resolves every frozen grid anchor

Commit `3c01e76a` added a public-metadata-only preflight before the batch
query. NASA CMR then resolved all 124/124 exact CACH4 L1B radiance-header
granules. Every selected granule has a distinct concept ID, declared byte
count, and source SHA-256; the protected header content totals only 1,745,203
bytes. No header content, COVID/Permian granule, target catalog, target asset,
JPL test data, or protected MARS outcome was accessed.

This clears only the Stage-A metadata-resolution sub-gate. The geometric grid
bridge remains pending until the authenticated headers are downloaded and all
resolved NASA grids match the public JPL grids with zero mismatches. Stage B
remains unauthorized. Compact receipt:
`reports/acquisition/jpl_operational_ghg_ornl_stage_a_cmr_preflight.json`.

## 2026-08-16: independent reviewed-negative source order frozen

The 28-pair MARS-Hyperspectral shortfall is a diversity problem, not a raw-row
problem. Four independent real-negative sources were compared before opening a
new manifest. STARCOP was selected first because its released train split has
author-refined no-plume windows, exact AVIRIS-NG flight UTC, per-chip zero-mask
GeoTIFFs, an open CC BY-NC 4.0 research license, and a 1 MB hash-bound manifest
that can be audited without downloading its 60.8 GB imagery release. Commit
`241f07d0` freezes a two-stage sparse protocol: select at most four train
negatives per flightline by SHA-256 of the ID before coordinates are known,
then range-read only their `labelbinary.tif` members. STARCOP alone must retain
at least 20 independent 25 km components and later provide at least 28
Sentinel-2/Landsat pairs within one hour. It cannot be pooled with any failed
candidate to pass either gate. A released no-plume label means no
annotated/refined plume in that benchmark window, not physical zero methane.

Carbon Mapper Tanager is the preferred fallback if STARCOP fails. Carbon
Mapper's primary product guide defines a candidate source-level null detect as
an assessed scene with at most 25% cloud that covers a known source without a
detection above the sensor limit. Its API exposes assessed scenes, exact UTC,
bounds, Tanager instrument identity, cloud assessment, plume counts, and source
observation/null-detection associations. A future protocol must restrict the
source to Tanager so existing EMIT/AVIRIS cohorts are not recycled, bind the API
schema and query before enumeration, and join scenes to reviewed source points
rather than label arbitrary scene centers. No Carbon Mapper catalog query has
been made in this path.

GHGSat's global-landfill table is technically attractive (434 published null
observations across a 47-country survey), but the deposit states All Rights
Reserved; it is ineligible for training unless written permission is obtained.
Two much larger direct Sentinel-2/Landsat negative sets were deprioritized:
He et al. Dryad contains duplicated annotations/augmentations concentrated in
Turkmenistan, and CH4Net contains 9,121 visible-no-plume scenes at only 23
Turkmenistan sites under CC BY-NC-ND 4.0. Their raw counts do not compensate for
probable MARS overlap, weak geographic independence, and—in CH4Net's case—an
uncertain derivative-work license. Controlled-release and atmospheric-baseline
collections were rejected because they cannot provide 28 distinct satellite
overpasses or do not establish crop-level column-plume absence.

## 2026-08-16: STARCOP train-manifest gate passes

The exact 1,038,970-byte Zenodo `train.csv` matched its frozen MD5 and contains
3,425 released train rows: 1,713 no-plume and 1,712 plume. All 256 flightlines
contain at least one negative. The preregistered outcome-independent rule chose
1,009 negatives, never more than four per flightline, by the smallest
`(SHA-256(id), id)` values. All three Stage-A gates pass by wide margins:
1,713 >= 1,000 negatives, 256 >= 50 negative-bearing flightlines, and an
observed maximum of four selected rows per flightline.

Two released-schema distinctions were verified against the pinned STARCOP
generator rather than treated as data errors. The ID records the source-image
window, while `window_*` columns are reset to `0,0,512,512` after each cached
chip is materialized. Source windows may be 151x151 for plume examples and one
edge-padded positive begins at source row -29; all released cached chips remain
512x512. Every no-plume ID uses a 512x512 source window. These corrections do
not alter the frozen negative selection or use `qplume`.

No train archive, label mask, imagery product, test manifest, target catalog,
MARS outcome, or coordinate was accessed. Stage A therefore establishes only
manifest feasibility and authorizes the already-preregistered sparse Stage-B
zero-mask georeferencing audit. The compact receipt is
`reports/acquisition/starcop_negative_supplement_stage_a.json` (SHA-256
`a80189174bf7bbe795720c0757000733e5d020fc56f4e85b2b4801449d368747`).

## 2026-08-16: STARCOP sparse Stage B fails the frozen geography gate

The Stage-B audit reconstructed all 1,009 deterministically selected train
`labelbinary.tif` members across 256 AVIRIS-NG flightlines. Every member is a
single-band 512x512 projected GeoTIFF, every affine and WGS84 center is finite,
and every pixel is zero. The 1,009 unique reconstructed masks occupy 1,935,966
bytes in ignored resumable storage and are bound to their authoritative ZIP
central-directory uncompressed sizes and CRC-32 values. No full archive,
source imagery, positive member, test content, target catalog, target asset, or
protected MARS outcome column was accessed. Record-declared archive MD5 values
are reported but not claimed as verified because the full archives were never
downloaded.

The leakage filter is decisive. Of 1,009 resolved negatives, 984 are within
25 km of an official MARS-S2L test representative; none is within 25 km of an
already-counted MARS-Hyperspectral negative source crop. Only 25 rows from 14
flightlines survive, and they collapse to eight transitive 25 km components.
All eight are also beyond every MARS-S2L representative, but the frozen gate
required at least 20 components. Stage B therefore **FAILS**. The requirement
is not lowered, failed sources are not pooled, and no Sentinel-2/Landsat target
query is authorized. STARCOP is retired as the 28-negative supplement.

The sparse transport failed closed several times before the final report. The
first archive parse exposed standard ZIP data-descriptor flags; the second
exposed a deterministic archive-root packaging prefix. Both source-format
corrections were tested and committed before selected label access. A later
transient non-206 response motivated bounded retry, shared pacing, and a
CRC-bound ignored cache. One resumable run reached 988 masks before the
15-minute process limit; the next completed the remaining 21 before a local
absolute/relative path comparison failed. The final report was then produced
from all verified cached masks. Across the documented attempts, exact central-
directory range traffic was 78,605,306 bytes, below the 512 MiB cap. Exact
cumulative label-range traffic from pre-resume failures is unavailable because
those early processes did not persist failure receipts; the report explicitly
scopes its byte totals to the final successful execution rather than presenting
a misleading zero as lifetime acquisition. This provenance limitation does not
affect mask identity, counts, coordinates, or the failed component gate.

Authoritative compact result:
`reports/acquisition/starcop_negative_supplement_stage_b.json` (SHA-256
`c5e8e431ec66113ebf72a7de8064459b129af64d4198f6e161c05ab491f08ce7`).
The next independent source is the preregistered fallback direction: a separate
metadata-only protocol for Carbon Mapper Tanager assessed null-detect scenes.
No Carbon Mapper catalog has yet been enumerated.

## 2026-08-16: Carbon Mapper Tanager null-detect metadata protocol frozen

Official Carbon Mapper Product Guide v1.1.6 defines a null detect as an
absence of detection under optimal observing conditions and considers a scene
a candidate when it has less than 25% cloud and intersects a known source.
The hash-pinned API v1.0.0 schema provides a stronger source-scene record:
`counts_as_null_detection`, `counts_toward_daily_emissions`, `has_detection`,
and `has_non_null_emission` for every scene in each source's daily breakdown.
It also exposes a public annotated-scene endpoint with Tanager identity,
mission phase, timestamp, assessed cloud, bounds, and publication status.

Protocol `configs/mars_carbon_mapper_tanager_null_protocol.json` (SHA-256
`06b2058a05ecf748dac25cb7389bb62e9fe88e36859dded389fc03b13d5b0ad0`)
is frozen before catalog enumeration. It retains only production-phase
Tanager CH4 scenes explicitly counted as source-level null detections, with
no detection/non-null emission at that source, assessed cloud <=25%, and a
source point inside the published scene bounds. Scene-wide plume count is not
used because a valid null source crop may coexist with a plume elsewhere in
the scene. At most four null scenes per source are chosen by a fixed SHA-256
order before geography is opened.

The metadata gate requires at least 50 leakage-safe source-scene pairs, 28
distinct Tanager scenes, 30 source points, and 20 connected components whose
every member is more than 25 km from every MARS-S2L representative. Carbon
Mapper must later provide 28 <=1-hour target pairs by itself; it may not pool a
failed source to pass. The current Terms permit non-commercial research and
derivatives in furtherance of emissions mitigation with attribution and
share-alike obligations. Any checkpoint materially trained on these labels
must carry those restrictions. The label claim is no Carbon Mapper detection
above applicable Tanager sensitivity at the reviewed source and time, never
physical zero methane. No catalog or image asset has been queried.

## 2026-08-16: Carbon Mapper Tanager metadata route fails the frozen response cap

The guarded metadata execution failed before the public source catalog could
be enumerated. Carbon Mapper's exact preregistered `sources.geojson` response
streamed past the 8,388,608-byte per-response ceiling; the reproduced capped
attempt observed at least 8,414,189 response bytes before aborting. No complete
source response was cached, no population row was parsed, and the frozen gates
for pair, scene, source, and component counts were therefore not evaluated.
This is an acquisition-contract failure, not evidence that the population
would or would not have passed those gates.

The first live attempt exposed that streamed bytes on an exception were not
persisted in a compact report. The auditor was corrected to treat every chunk
as non-refundable traffic and to emit a fail-closed artifact, its 17 focused
tests passed, and that correction was committed before the reproduced attempt.
The original threshold was not enlarged and the exact endpoint was not
replaced by an unregistered query. Carbon Mapper is therefore retired under
the frozen failure action. No Carbon Mapper image asset, Sentinel-2/Landsat
catalog, protected MARS outcome, or target-pair query was accessed, and every
target eligibility claim remains false.

Authoritative compact result:
`reports/acquisition/carbon_mapper_tanager_null_metadata.json` (SHA-256
`730048bf557a9024ea97282d6bd042d3093cd29fec983cffe6ce7142694a088b`).
The next research decision must use an independently preregistered source or
return to architecture work supported by cohorts that have already passed;
failed STARCOP, CACH4, and Carbon Mapper candidates may not be pooled.

## 2026-08-16: corrected domain-adaptive Prithvi v2 fails seed one

The v2 protocol ran to its conditional pilot boundary on the frozen git state
`5bb79d4a56349b8be2db92fe5e4e6817c315844c`. It trained two independent
folds-3/4 endpoints; each endpoint's self-supervised and supervised inputs came
only from the opposite development fold plus the fixed MARS-disjoint auxiliary
cohort. No fold 0/1/2 or official-test input was loaded.

The least harmful registered strength, 0.25, reached AP 0.904501 and changed
-0.002024 AP and -0.000656 matched-FPR recall versus the Gaussian+DOFA
champion. Its 250-group, 10,000-replicate paired-site interval was
[-0.004297, +0.000679]. Fold AP changes disagreed (-0.001481 on fold 3,
+0.000655 on fold 4), while sensor AP changed -0.003565 for Sentinel-2 and
-0.000097 for Landsat. Every-fold AP, every-sensor AP, pooled AP, pooled
recall, and paired-site lower-bound gates failed.

The fixed strength sweep degraded monotonically: 0.50 changed -0.004743 AP
and 1.00 changed -0.012790 AP. The strength-1 paired-site interval was
[-0.024466, -0.001631]. Seed two was therefore skipped exactly as registered;
no artifact was promoted and no external or official evaluation occurred.
Ignored checkpoints are retained only for reproducibility and are not eligible
for reuse as a promoted model.

Interpretation: the observability-weighted MAE optimized stably but did not
learn a correction complementary to the champion. The failure is strongest on
Sentinel-2 and is geographically discordant. This retires further scalar
Prithvi corrections and post-hoc weight search. A defensible next candidate
must introduce new methane-specific measurement supervision—subject to a
separate frozen data bridge—or a representation whose evidence is not already
captured by Gaussian, DOFA, and spatial-Prithvi components. Authoritative
compact result: `reports/experiments/mars_prithvi_domain_adaptive_v2.json`.

## 2026-08-16: cross-modal literature supports targets, not unrestricted features

Primary-source review after the Prithvi rejection found one directly relevant
methane transfer precedent: PlumeBed composes hyperspectral Carbon Mapper
plumes with Sentinel-2 backgrounds and applies domain-adversarial learning.
General remote-sensing work supports learning with a privileged modality, but
only for classification benchmarks. The large synthetic Sentinel-2 methane
ViT/U-Net study explicitly treats asynchronous airborne comparison as
indirect, and the EMIT V002 product contract provides enhancement,
uncertainty, and sensitivity for scenes while reserving multi-scientist review
for identified plume complexes.

The conditional architecture call is therefore conservative. If the frozen
300-negative acquisition gate passes, PSF-matched enhancement/masks and their
uncertainty/sensitivity become primary training-only targets. Unrestricted
hyperspectral hidden-feature matching is not the default. Dense labels remain
limited to <=15 minutes, scene labels to <=1 hour, and unobservable coverage
is never converted to no plume. Sentinel-2 and Landsat use separate students
or stems and must each pass the sensor AP gate. No model protocol or imagery
download is authorized by this review. Detailed note:
`reports/research/HYPERSPECTRAL_PRIVILEGED_SUPERVISION_NOTE.md`.

A follow-up source-only Hermes sweep found no public cohort that already closes
the 28-pair negative deficit. Local receipts independently confirm that CACH4
and STARCOP fail their spatial-component gates; Carbon Mapper null detections
are source-specific; and COVID/Permian remain behind the authenticated
NASA/ORNL grid bridge. Temporary source downloads created by the scout in the
repository root were removed and never staged. Future Hermes use is therefore
bounded to discovery, with local primary-source verification and diff review
before any claim or protocol change.

## 2026-08-16: domain-adaptive Prithvi experiment frozen before outcomes

Architecture work resumed without pooling any failed acquisition source. The
new protocol uses only MARS folds 3/4 and the already qualified, spatially
disjoint MethaneS2CM auxiliary cohort. It corrects the old Prithvi transfer
path's factor-of-two MARS radiometric scale, withholds geographic coordinates,
and cross-fits both self-supervised pixels and supervised labels: each endpoint
uses only the opposite fold.

The extended-pretraining stage uses all-block rank-8 attention LoRA, a fully
trainable patch embedder and final encoder block, and a two-block pretrained
MAE decoder. Its 75%-masked reconstruction loss normalizes correlated
visible/NIR/SWIR groups independently and adds a fixed temporal-change term.
The downstream patch-change residual is exactly zero-initialized and bounded
around the existing Gaussian+DOFA champion. A dedicated MARS unlabeled reader
opens only image and cloud assets; a dedicated MethaneS2CM reader opens only
reflectance arrays and sample IDs.

Ten focused CPU tests and one CUDA adapter-placement regression pass. The
final real-data smoke produced finite gradients/losses with 8,043,650
trainable pretraining parameters and 422,994 scene parameters, without
computing an AP, recall, bootstrap, or protected-fold outcome. The protocol is
now frozen. Seed two is conditional on a strict seed-one +0.002 AP pilot; final
promotion requires +0.005 AP, positive fold/sensor deltas, nonnegative
matched-FPR recall in both folds, a positive paired-site lower bound, and an
independently positive replication. Folds 0/1/2 and official data are excluded
from every declared input.

Frozen protocol: `configs/mars_prithvi_domain_adaptive_protocol.json`,
SHA-256 `6b5adaabf785dcfc6226c984429c26cdc942feb5fb1678e5496f779671af6658`.

## 2026-08-16: v1 stopped before outcome; corrected v2 frozen

A read-only independent Hermes audit identified two verified contract defects
while v1 was still below the outcome boundary: the seed-two code enforced a
weaker gate than the prose, and invalid pixels entered MAE normalization/loss.
The running process was terminated before pooled AP, recall, bootstrap, or
strength selection. Only the fold-3 endpoint had completed; its ignored
checkpoints cannot initialize v2. Compact supersession report:
`reports/experiments/mars_prithvi_domain_adaptive_v1_superseded.json`, SHA-256
`18b8070d638dce61bdd51e2295ceac3d79cd9d93b27f2d432ac165c32ccc6b22`.

V2 excludes invalid pixels from patch/group statistics and both loss terms,
requires seed two to independently pass all pilot fold/sensor/recall/bootstrap
checks, declares the future external four-member mean-correction rule, uses
new seeds, and writes to a new checkpoint directory. Thirteen focused tests
and a new real-data finite-gradient smoke pass with no outcome. Frozen v2
protocol SHA-256:
`86c350b2b61732c643dc4207b74cd50baf5cb5f4037ec0d462f6cb5b3d5fa934`.

The research claim is deliberately limited to an ExPLoRA-inspired,
MAESTRO-inspired integrated system. It is not presented as a faithful
reproduction or as a causal ablation of either published method.

## 2026-08-29: NASA/ORNL COVID+Permian bridge rejected at frozen component gate

Authenticated header-only acquisition completed for the exact 124 CACH4 grid
anchors and 280 COVID/Permian released-train flightlines. All 404 ENVI headers
matched NASA CMR byte counts and SHA-256 values. Stage A resolved every anchor
and found zero grid mismatch, proving the JPL crop-index and NASA orthocorrected
header grids geometrically interchangeable under the frozen comparison.

Stage B resolved all 13,444 released background rows with exact UTC and finite
WGS84 centers. Frozen geographic protection excluded 9,772 rows; the remaining
3,672 rows across 133 flightlines formed only 13 independent 25 km components,
below the preregistered minimum of 20. The overall decision is **FAIL**. No
Sentinel-2/Landsat catalog or asset was queried, and no protected MARS outcome
was opened. COVID/Permian cannot be pooled with any previously failed source
to rescue the gate. The compact result is
`reports/acquisition/jpl_operational_ghg_ornl_header_bridge.json`.

## 2026-08-29: independent GHGSat landfill-null audit preregistered

Hermes identified the Dogniaux et al. GHGSat global-landfill release as the
only defensible new public candidate after the ORNL failure. Codex independently
verified the primary Zenodo and Nature sources before making the protocol call.
The released population contains 1,447 clear-sky site observations at 151
waste sites: 1,013 positive observations represented by 1,085 plume rows and
434 reviewed null observations. Null means no localized plume detected at the
reviewed site above applicable GHGSat sensitivity. It does not mean zero
emissions, below-threshold absence, or a scene-wide negative.

Zenodo shows both GHGSat's retained copyright and the governing CC BY-NC-SA
4.0 dataset license. Non-commercial use, attribution, share-alike handling,
and a constrained checkpoint-distribution statement are now explicit parts of
the contract. The peer-reviewed GHGSat platform table reports C1-C2 at 10:30
local descending-node time and C3-C5 at 13:00. The metadata feasibility audit
therefore selects only C1/C2 null observations; exact cross-sensor time offsets
remain forbidden until a later, separately committed target-catalog protocol.

Before the full CSV is downloaded, the frozen metadata gates require exact
paper-count reconciliation, validity of every released observation, at least
56 deterministically selected morning nulls, at least 30 distinct null sites,
and at least 20 MARS/prior-negative-disjoint 25 km components. Null coordinates
are zero sentinels in the release, so each site's representative is frozen as
the medoid of its validated positive-row coordinates, with a 25 km within-site
span sanity bound. Failure retires this source without threshold changes or
pooling. Protocol SHA-256:
`0943cb2a15f2106b9ee4a71f9ab36c06f563e410ad0b530ee1f3514b3aa1bcb1`.

## 2026-08-29: GHGSat independent negative-source gate passes

The exact hash-bound CSV was downloaded after protocol freeze. Its 299,683
bytes match the released MD5 and SHA-256
`b383b457db5790cac9d99c36d01a22599c53b03bf67075dbb87501c31997896a`.
An initial 88-byte HTTP 406 response led only to a browser-compatible request
header fix; the exact URL, file identity, cap, and every scientific gate stayed
unchanged. A later parse failure correctly identified empty positive-only
measurement fields on null rows. The Zenodo description states that no
emission rate is calculated for non-detections, so the implementation now
requires those fields to be exactly empty for nulls and finite for positives.
That schema correction was committed before the cached audit was rerun.

The audit exactly reconciled 1,447 observations, 1,013 positive observations,
434 null observations, 1,085 plume rows, 151 sites, and the 2021-2022 years.
There were 329 C1/C2 null candidates. Deterministic selection capped them at
199 across sites; five selected observations were within 25 km of protected
official MARS test geography and 18 were within 25 km of a counted prior
negative point. The remaining 176 observations span 66 distinct sites and 64
independent 25 km components. Gates pass by margins of +120 observations,
+36 sites, and +44 components.

No target catalog or imagery was accessed. This pass authorizes only a new,
committed Sentinel-2/Landsat pairing protocol. It does not authorize model
training, create dense-negative masks, or establish benchmark improvement.
Authoritative compact artifacts are
`reports/acquisition/ghgsat_landfill_null_metadata.json` and
`reports/acquisition/GHGSAT_LANDFILL_NULL_METADATA.md`.

The exact metadata-only target-catalog protocol is frozen before its first
query. It uses the official CDSE `sentinel-2-l1c` and USGS `landsat-c2l1`
STAC collections over the closed +/-1-hour window for the 176 hash-bound
source observations. Landsat must be LC08/LC09 L1TP Tier 1; Sentinel-2 must be
S2A/S2B L1C. Full geometry containment is mandatory, asset fields are excluded,
and catalog cloud cover cannot decide eligibility. At most one candidate per
source observation per target sensor survives deterministic offset/cloud/ID
selection. The gate counts at least 28 distinct source observations, not merely
28 potentially correlated source-sensor pairs, plus 20 sites, 20 components,
and 20 target item IDs. Protocol:
`configs/mars_ghgsat_target_catalog_protocol.json`, SHA-256
`486e78bdb41b4aa9dfcb6bc6943eb80caf0cf8c7c5899bce1cdb6cc7a06967ff`.

## 2026-08-29: frozen GHGSat target-catalog audit passes

The independently reviewed and corrected auditor completed all 352 frozen
logical queries. It returned 109 candidates and deterministically selected 79
source-sensor pairs: 47 Sentinel-2 L1C and 32 Landsat C2 L1. Those selections
represent 66 distinct source observations, 44 sites, 44 independent 25 km
components, and 78 target item IDs. Every frozen acquisition-feasibility gate
passes.

The ignored candidate output has SHA-256
`cabe8b0c7055de8844f203daa87657c694b197b89540120cb411dc3ecbf34cba`;
the ignored selected-pairs output has SHA-256
`8c7942f0ac7bc07e250603e25ff23bb345e7bafbd86394e4ce0e42d41a33f6a8`.
The authoritative compact report is
`reports/acquisition/ghgsat_landfill_target_catalog.json`.

This result establishes only catalog-level acquisition feasibility under the
frozen point, time, sensor, geometry, and deterministic-selection contract. No
assets, references, protected outcomes, or model artifacts were accessed. It
does not establish raster-level observability, create dense all-negative
labels, authorize a model protocol, or claim benchmark or model improvement.

## 2026-08-29: frozen GHGSat reference-catalog audit passes

The frozen auditor completed 79 primary logical queries and 9 seasonal logical
queries, 88 total. It made 89 HTTP attempts: 88 returned status 200 and one
handled status 429. The audit retained 270 valid candidates and selected 76
target/reference pairs: 47 Sentinel-2 L1C and 29 Landsat C2 L1. Of those
selections, 70 came from the primary window and 6 from the seasonal fallback.
They represent 64 distinct source observations, 43 sites, 43 independent 25 km
components, and 75 distinct reference item IDs. All frozen gates passed.

The candidate JSONL SHA-256 is
`d160fe663f646ed8a2e0798954517e077c63f7f405a107a4c7dea9edc5cc4c90`;
the pair JSONL SHA-256 is
`ebd2cd3e4f4a201905f441c86eeabd1a50798e57c3818a3bd46d4fb11709f6a2`.
The compact report is
`reports/acquisition/ghgsat_landfill_reference_catalog.json`.

No asset URL, item detail, raster, protected outcome, or model artifact was
accessed. This result is strictly catalog feasibility; it does not establish
local observability, dense no-plume truth, training value, or benchmark
improvement.
