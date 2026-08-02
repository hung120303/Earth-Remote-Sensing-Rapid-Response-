# ERSRR v6 product-aware multi-cohort design

Status: architecture decision recorded before v6 outcome scoring.

## Decision

The next major branch is a product-harmonized, multi-cohort Prithvi/UNet, but
the external recommendation is not implemented verbatim. The repository's
actual foundation checkpoint is `Prithvi-EO-2.0-tiny-TL`, a 5M-parameter
temporal/location encoder pretrained on 4.2 million six-band HLS V2 samples at
30 m. The proposed 23M/100M description and 30-channel count are incorrect.

V6 uses a fixed physical-radiometry adapter followed by learned near-identity
product corrections. The fixed conversion is source specific:

- MARS-S2L L1C/Landsat tensors are stored by the released loader as raw DN / 5,000,
  so they are multiplied by 0.5 to recover DN / 10,000 physical reflectance.
- MethaneS2CM L2A tensors already use raw DN / 10,000 and remain unchanged.
- Sentinel-2 L1C, Sentinel-2 L2A, and Landsat retain distinct product identities.
- The broad Sentinel-2 B08 input remains an explicit mismatch with Prithvi's
  HLS narrow-NIR pretraining slot; the paper must not describe it as exact.

## Architecture

The fixed canonical sample contains three six-band frames
(`reference90`, `reference365`, `target`), an observability mask, two reference
availability flags, a product identity, and a sensor identity. Missing frames
initially copy the target and receive a bounded learned product-specific
imputation residual. Product identity is not treated as a substitute for
radiometric harmonization.

The learned system has two independent branches initialized from the same
Prithvi checkpoint:

1. **Dense branch.** A rank-8 LoRA Prithvi pair encoder processes
   reference90/target and reference365/target. A full-resolution convolutional
   path consumes a declared 47-channel physics tensor: three absolute frames,
   two signed differences, two absolute differences, two scale-invariant MBMP
   maps, observability, and two reference-availability maps. Product and sensor
   embeddings add 12 more learned channels. Four down/up stages fuse Prithvi
   token maps, and MBMP evidence enters every decoder stage through an
   identity-initialized gated residual.
2. **Scene branch.** A separate rank-8 LoRA Prithvi pair encoder uses learned
   patch evidence, top-k pooling, mean pooling, CLS tokens, product/sensor
   embeddings, and reference availability. On MARS development it learns only
   a bounded residual around the frozen Gaussian+DOFA champion, with exact
   identity below a protection gate. Synthetic data never supervises this head.

Separate encoders, LoRA adapters, product harmonizers, and embeddings make the
optimization decoupling literal. Dense and scene batches are alternating
phases with disjoint trainable parameter sets, rather than two losses hidden in
one weighted sum.

## Authorized data and boundaries

Training may use only:

- cross-fit MARS folds 3 and 4, fitting each endpoint on the opposite fold;
- the 14,859-crop, 769-location, 147-group MethaneS2CM auxiliary partition that
  is more than 25 km from every official MARS test location;
- the 160 nonsealed UNEP/IMEO positive auxiliary scenes for dense/hard-positive
  work under their existing roles;
- existing CloudSEN training negatives, never the already-opened fresh test;
- the frozen analytical Gaussian bank for dense-only initialization.

The 11,478-crop MethaneS2CM source-development partition, MARS fold 2, MARS
folds 0/1, official MARS test, UNEP sealed positives, EMIT V002 outcomes, and
already-opened CloudSEN test remain outside model selection. MethaneSET is
excluded as a MARS repackaging/leakage risk. MethaneS2CM is CC-BY-NC-4.0; bulk
files and derived checkpoints stay ignored.

## Training and evaluation sequence

1. Dense-only initialization uses the already validated Gaussian bank endpoint
   (epoch 9: dense-evidence AP 0.927611, mask IoU 0.261063). This is an
   initialization/ablation result, not evidence of real-scene superiority.
2. Dense-only real training alternates group-balanced MARS, MethaneS2CM, and
   authorized auxiliary positives. The scene adapter is frozen.
3. Scene-only real training uses group- and label-balanced batches plus hard
   negatives. The dense branch is frozen and no synthetic label reaches the
   scene head.
4. MARS folds 3/4 are cross-fit. Promotion requires at least +0.005 AP over the
   current champion, positive AP in both folds and sensors, non-worse pooled
   matched-FPR recall/FPR, positive dense IoU, and strictly positive paired
   25 km-group bootstrap lower bounds.
5. Only a passing multiseed candidate may be frozen for one-shot evaluation on
   the untouched MethaneS2CM source-development partition. This is a product-
   transfer confirmation, not a replacement for the exact official MARS claim.

## Risk-control claim boundary

The proposed negative-score quantile is not, by itself, a geographic-transfer
guarantee. Conformal risk control bounds the expected value of a monotone loss
under its exchangeability assumptions. V6 will report:

- an ordinary target-FPR coverage-risk curve;
- a finite-sample negative-class upper confidence bound calibrated at the
  **25 km group level**, not the crop level;
- Mondrian diagnostics by product/sensor and preregistered broad geography only
  where calibration group counts are adequate;
- realized FPR and recall on every genuinely held cohort.

The paper will state the guarantee only for exchangeable future groups. Results
under product or geographic shift are empirical transport tests, not
distribution-free certificates.

## Evidence motivating this branch

- The current spatial-Prithvi champion already beats released MARS-S2L on the
  full official replay, while test-only AP/recall uncertainty remains
  inconclusive.
- The protected Gaussian+DOFA residual passes folds 3/4 at AP 0.906525,
  +0.002449 with paired-site interval [+0.000489,+0.004068], but this is too
  small to resolve the final same-domain uncertainty by itself.
- Dense improvements repeatedly have positive paired-site support, including
  +0.004367 IoU on the already-opened fold-2 confirmation.
- A prior 640 m MARS/MethaneS2CM physical-transfer detector failed scene
  promotion, showing that simple resampling and source-adversarial alignment
  are insufficient.
- MethaneS2CM threshold transfer and the strict MARS spatial test both show that
  ranking can improve while a development-selected operating point fails.

Primary sources: the MARS-S2L manuscript (arXiv:2408.04745), the
Prithvi-EO-2.0 report (arXiv:2412.02732), the pinned IBM/NASA Tiny-TL model
card, the MethaneS2CM KDD 2025 dataset card/paper citation, and Conformal Risk
Control (ICLR 2024; arXiv:2208.02814).
