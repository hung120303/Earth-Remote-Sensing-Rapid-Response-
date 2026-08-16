# ERSRR publication outline and claim contract

## Working title

**Tri-temporal physics-aware methane-plume detection in Sentinel-2 imagery under spatially isolated evaluation**

Alternative benchmark framing:

**What transfers in methane-plume detection? A sealed, product-aware comparison of ERSRR and MARS-S2L**

## Central research question

Can a shared-weight tri-temporal Sentinel-2 detector improve methane-plume ranking and dense
localization while correctly rejecting no-plume crops at geographically isolated locations, and
does that performance exceed frozen ERSRR v4.3 and released MARS-S2L baselines under the same test?

The study concerns binary scene detection and dense mask segmentation. It does not claim methane
concentration, emission flux, operational deployment, or prevalence-representative positive
predictive value.

## Primary contribution set

1. A pinned MethaneS2CM L2A contract with three temporal frames, exact 12-page TIFF semantics,
   binary-mask validation, immutable train/test metadata, and Git-safe HDF5 packing.
2. A leakage-resistant development protocol: exact-location-disjoint official split plus 25 km
   connected-component fitting/development groups, immutable manifests, and a one-shot test seal.
3. A 9.36 M-parameter shared tri-temporal U-Net with two MBMP physics maps, seven-term fusion at five
   scales, dense segmentation, and a controlled small context-head ablation.
4. A fixed three-seed ensemble with empirical-CDF scene calibration, predeclared FPR thresholds,
   equal dense-probability averaging, and 2,000 spatial group-bootstrap replicates.
5. A same-cohort comparison with zero-shot v4.3 and released MARS-S2L, alongside a separate paired
   MARS strict comparison and clearly separated published-MARS context.
6. A fully audited negative claim: ranking/localization improve strongly, but calibration transfer
   misses the no-plume criterion, so across-the-board superiority is rejected.

## Frozen result and permissible abstract language

On the 20,789-crop MethaneS2CM location test, ERSRR v5.1 achieved scene AP 0.8180, AUROC 0.8276,
recall 0.3778, FPR 0.0607, pixel AP 0.2083, Dice 0.3125, and IoU 0.1852. Released MARS-S2L achieved
AP 0.5252, AUROC 0.5126, recall 0.0052, FPR 0.0033, pixel AP 0.1691, Dice 0.0045, and IoU 0.0023.
V4.3 made no positive prediction at its frozen rule.

Paired 2,000-resample 25 km group intervals supported positive v5.1-minus-MARS deltas for AP,
AUROC, recall, Dice, and IoU. The v5.1 FPR delta was also conclusively positive. The abstract may
therefore say:

> ERSRR v5.1 substantially improved plume ranking, recall, and dense localization over frozen
> zero-shot comparators on a spatially isolated MethaneS2CM test, but its development-frozen
> threshold transferred to a higher false-positive rate; full MARS-S2L superiority was not
> established.

It may not say that v5.1 categorically outperforms MARS-S2L, solves no-plume rejection, or is ready
for operational monitoring.

## Comparison boundaries

| Evidence | Valid use |
|---|---|
| V5.1, v4.3, and released MARS-S2L on MethaneS2CM | Same-test performance comparison; note L2A in-domain versus L1C zero-shot training difference. |
| V4.3 and released MARS-S2L on strict MARS cohort | Paired L1C comparison; v4.3 improves AP/AUROC/FPR point estimates but loses recall/overlap. |
| MARS-S2L full official and test-only-site tables | Different-cohort context only; no paired delta or superiority inference. |
| Frozen MethaneS2CM post-hoc thresholds | Calibration-transfer diagnostic only; cannot select a new test operating point. |

## Manuscript structure

1. **Introduction** — point-source methane monitoring, false-alarm cost, temporal context, spatial
   leakage, product shift, and the need to separate ranking from operating-policy transfer.
2. **Related work** — MBMP/multipass retrieval, CH4Net, MARS-S2L, MethaneS2CM/MEECNet, dense
   segmentation, selective prediction, and risk-controlling calibration.
3. **Datasets and contracts** — MethaneS2CM revision/license/splits, MARS-S2L comparators, TIFF and
   band semantics, label geometry, observability, missing wind/cloud metadata, and bulk-data policy.
4. **Leakage-resistant protocol** — exact locations, 25 km components, fitting/development/test
   boundaries, campaign freeze, one-shot unlock, seeds, hashes, and preregistered metrics.
5. **Methods** — v5/v5.1 architecture, physics features, losses, augmentation, context-head
   ablation, ensemble calibration, and frozen operating points.
6. **Results** — physics baselines, single-seed ablation, three-seed development confirmation,
   one-shot test, paired uncertainty, zero-shot baselines, and the strict MARS context table.
7. **Calibration transfer** — development-to-test AP/recall/FPR/overlap gaps and the explicitly
   post-hoc four-threshold audit; no test-selected threshold.
8. **Discussion** — why rank transfer is stronger than policy transfer, product mismatch, broad
   mask geometry, no-plume implications, and the new calibration/domain-generalization study.
9. **Limitations and ethics** — balanced prevalence, missing timestamps/wind/cloud mask,
   non-operational status, source attribution, license limits, and monitoring misuse.
10. **Reproducibility statement** — commits, environment, manifests, seeds, checksums, ignored bulk
    artifacts, commands, and deterministic/non-deterministic runtime boundaries.

## Primary tables

1. Data funnel: train/fitting/development/test crops, positives/negatives, exact locations, 25 km
   groups, overlap checks, archive bytes, and licenses.
2. Input/product contract: L2A/L1C product, bands, temporal frames, reflectance scale, wind/cloud
   availability, patch size, and comparator imputations.
3. Physics baselines and v5 mask-derived seed-1101 reference.
4. V5.1 seeds 1101/2202/3303: selected epoch, AP, AUROC, recall at target FPR, FPR, pixel AP, Dice,
   IoU, and artifact hashes.
5. V5.1 ensemble group-held and all-development freeze values with confidence intervals.
6. One-shot same-test results for v5.1, v4.3, and released MARS-S2L.
7. V5.1-minus-baseline paired 25 km group-bootstrap deltas.
8. Separate MARS strict comparison and published-MARS different-cohort context.
9. Frozen four-threshold calibration-transfer audit, labeled post hoc.

## Primary figures

1. Acquisition, grouping, development, freeze, and one-shot unlock flow.
2. Tri-temporal shared encoder and seven-term fusion diagram.
3. Development and location-test PR/ROC curves with frozen operating points.
4. Paired group-bootstrap distributions for AP, recall, FPR, Dice, and IoU deltas.
5. Development-to-test calibration transfer across all four predeclared thresholds.
6. Deterministically selected true positive, false negative, false positive, and no-plume examples;
   selection rule declared before viewing and labeled as test error analysis.
7. Product/domain matrix showing L2A in-domain and L1C zero-shot comparisons.

## Statistical analysis contract

- Primary external unit: frozen 25 km MethaneS2CM test component, not crops or pixels.
- Replicates: 2,000 paired nonparametric group resamples with committed seed 20,260,713.
- Primary scene metrics: AP, AUROC, recall, FPR, and precision at each frozen rule.
- Primary dense metrics: all-observable-pixel AP, Dice, and IoU at each frozen rule.
- Report point estimates, two-sided percentile intervals, and absolute paired deltas.
- Treat balanced-benchmark precision as descriptive, never operational PPV.
- Keep the primary 5%-development target and test-realized FPR distinct.
- Never choose a model, seed, context weight, scene threshold, pixel threshold, or postprocessing rule
  from the location-test result.
- Published MARS-S2L tables and ERSRR same-cohort tables remain separate.

## Ablations and negative results to retain

- Physics-only MBMP evidence is weak (best scene AP 0.5509 and recall 0.0564 at approximately 5%
  FPR), supporting learned temporal/context features.
- Mask-only v5 seed 1101 achieved AP 0.8225 and recall 0.3421; v5.1 seed 1101 achieved AP 0.8542
  and recall 0.4321 while retaining similar dense overlap.
- The rejected v4.2 multi-seed campaign and v4 cascade remain documented negative experiments.
- V4.3’s strict MARS result improves FPR/AP/AUROC point estimates but not recall or segmentation;
  only the FPR improvement is bootstrap-conclusive.
- V4.3 and released MARS-S2L collapse under the MethaneS2CM product/domain shift at their fixed
  rules. This cannot be presented as proof that their architectures are intrinsically inferior.
- The preregistered MARS-Hyperspectral cross-modal acquisition is a data-gate negative result,
  not a model ablation: exact Sentinel-2 L1C plus Landsat 8/9 L1 metadata produce 500 positive
  pairs, 460 pairs within one hour, 204 novel 25 km groups, and 49 countries, but only 272/300
  required reviewed negatives. No target bands or model were opened; retain this as evidence for
  conservative cohort qualification rather than implying cross-modal learning was evaluated.

## V5.2 preregistration requirements

1. Define new spatial calibration groups before training and fit risk control only there.
2. Specify whether the primary endpoint is a fixed target-FPR rule, a coverage-risk curve, or both.
3. Add product-aware L1C/L2A harmonization and missing-temporal-frame handling to the development
   comparison, with identical folds and compute budgets.
4. Freeze mask-quality weighting, plume-scale sampling, negative sampling, and architecture choice
   from development evidence only.
5. Acquire a new prevalence-aware, geographically isolated confirmation cohort with defensible
   positive and negative labels.
6. Commit acquisition/evaluator code and all identities before opening the cohort once.

## Reproducibility and release bundle

Release code, configs, compact protocols, result JSON, checksums, test commands, HTML dossier, and
paper documents. Keep third-party bulk imagery, HDF5 packs, model checkpoints, prediction caches,
credentials, Earthdata cookies, and temporary signed URLs out of Git. MethaneS2CM is
CC-BY-NC-4.0; redistribution and publication use must respect its license and attribution terms.

The current runs fix Python/NumPy/PyTorch seeds but do not promise bitwise CUDA determinism. The
paper should describe the three seeds as stochastic replicates and disclose the runtime versions
recorded in the primary JSON.

The v6 manuscript extension should add a methods subsection separating fixed
physical radiometry from learned product correction, a dual-adapter diagram
showing disjoint scene/dense gradients, and a calibration subsection stating
the exchangeable-group CRC estimand exactly. The data table must distinguish
the 33,131 authorized training rows, 5,050 MethaneS2CM risk-calibration rows,
and 6,428 untouched confirmation rows. The Gaussian+DOFA result remains the
scene comparator; it is not replaced by the weaker spatial-Prithvi baseline.

The first unified-cohort scene-adapter pilot is a negative-result ablation,
not the proposed main method: +0.000182 AP with paired-site interval
[-0.000107,+0.000427] versus Gaussian+DOFA, failing the frozen +0.003 gate.
Retain it to show that cohort scaling plus product tokens alone was
insufficient; do not imply that its unrun dense branch or CRC stage was
evaluated.

Pair that ablation with the v6.1 objective control: removing the bounded
training correction reversed the small gain to -0.001376 AP at the weakest
protected inference strength, with an entirely negative paired-site interval.
Together these runs support an information/complementarity ceiling, not an
optimizer-capacity explanation.

Add the frozen-encoder scale ablation: Prithvi-EO-2.0-100M-TL at 3,072
multiscale CLS features produced -0.000288 AP at its weakest fusion and no
passing candidate. This closes the simple claim that the 5M Prithvi encoder
alone caused the plateau; it does not compare dense decoders or full
fine-tuning of the 100M checkpoint.

## Candidate venues

The present evidence best fits a remote-sensing methods/benchmark paper emphasizing spatial
transfer and calibrated operating policies: *Remote Sensing of Environment*, *Atmospheric
Measurement Techniques*, or *IEEE Transactions on Geoscience and Remote Sensing*. A categorical
superiority paper requires the v5.2 preregistered confirmation cycle, not reinterpretation of this
test.
