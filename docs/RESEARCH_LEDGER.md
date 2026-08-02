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
