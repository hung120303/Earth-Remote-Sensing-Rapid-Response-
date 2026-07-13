# ERSRR research ledger

This ledger records paper-relevant decisions, outcomes, and interpretation boundaries. Compact JSON
reports are authoritative for numbers and hashes; this document is the human-readable study map.

## Study question and frozen decision rule

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
