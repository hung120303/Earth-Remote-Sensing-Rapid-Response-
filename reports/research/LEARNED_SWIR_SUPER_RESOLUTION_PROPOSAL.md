# Learned Sentinel-2 SWIR super-resolution proposal

**Status:** proposed and documented on 2026-09-03; not implemented, selected, trained, or
evaluated. This document records hypotheses and a possible preregistration. It does not
authorize access to protected or official outcomes.

## Decision summary

Retain learned 20 m→10 m SWIR sharpening as a bounded research path, but do not describe
its output as recovered 10 m SWIR. The most defensible first candidate is the pretrained
`SEN2SRLite Reference_RSWIR_x2` model. It should first be audited without detector
retraining and compared with the exact current MARS-S2L preprocessing contract: native
20 m B11/B12 exported on the 10 m grid, restored by nearest downsampling to the native
lattice and bilinear upsampling to 10 m.

The expected upside is modest improvement in dense plume boundaries. A large scene-ranking
gain is less plausible because the new high-frequency detail is inferred primarily from
10 m RGB/NIR land-cover edges, not newly measured methane absorption. The central risk is
that inferred facility, field, cloud, water, or road texture becomes false B11/B12 ratio
structure.

For that reason, any later detector integration should keep native/bilinear B11/B12 and
native MBMP evidence as authoritative inputs. Super-resolved bands, high-frequency
residuals, low-resolution closure error, and transform/tile-offset uncertainty should enter
only through an optional gated auxiliary branch. The scene and dense heads must be able to
ignore that branch independently.

## Evidence boundary

- [SEN2SR](https://doi.org/10.1016/j.rse.2025.115222) provides a directly applicable
  pretrained 20 m→10 m reference model for B5–B7, B8A, B11, and B12. Its Fourier
  low-frequency hard constraint is designed to preserve the native observation after
  downsampling. The [implementation](https://github.com/ESAOpenSR/SEN2SR) and published
  weights are CC0-1.0.
- SEN2SR's methane case study reported higher normalized MBMP SNR for its learned outputs,
  but did not establish significantly better methane detectability. The paper also notes
  that the Wald 40 m→20 m training protocol lacks quantitative proof that it extrapolates
  faithfully to real 20 m→10 m SWIR. This is mechanism evidence, not evidence of higher
  scene AP, recall, FPR, plume-object recall, or dense IoU.
- [DSen2](https://doi.org/10.1016/j.isprsjprs.2018.09.018) is the established learned
  Wald-protocol baseline. Its GPL-3.0 implementation uses an old TensorFlow/Keras stack, so
  it is a scientific comparator rather than the preferred integration target.
- [OpenSR-Test](https://github.com/ESAOpenSR/opensr-test) supplies consistency,
  hallucination, omission, and improvement measures that are more relevant than PSNR or
  SSIM alone.
- [VegiSR](https://openreview.net/forum?id=h3lBLmgIjj) is a different task: it predicts an
  EMIT-like 285-band hyperspectral representation from Sentinel-2 at a common 60 m grid.
  It is not evidence for spatial B11/B12 sharpening. Its patch-random split also requires
  an independent site-disjoint validation before its performance can support a geographic
  generalization claim.
- [S2Sharp](https://doi.org/10.1109/TGRS.2019.2906048) is a useful non-neural inverse-model
  comparator. Its MATLAB repository has no clear reusable license; resolve that before
  distributing adapted code or artifacts.

## Proposed preregistered experiment ladder

### Stage 0 — freeze identities and scope

Use development folds 3/4 only, grouped by the existing canonical 25 km site components.
Keep every target/reference acquisition from a site in one fold. Freeze hashes for the
manifest, input products, SR weights, preprocessing kernels, detector, thresholds, and
evaluation code before reading downstream outcomes. Record spacecraft, processing baseline,
L1C/L2A product, sensor response revision, cloud state, biome, and brightness strata. Do not
open folds 0/1/2, protected, external, controlled-release, or official MARS-S2L outcomes.

### Stage 1 — radiometric and hallucination audit

Run no detector training. Compare:

1. current MARS nearest-to-native plus bilinear restoration;
2. bicubic interpolation;
3. DSen2;
4. SEN2SRLite `Reference_RSWIR_x2`;
5. S2Sharp only if license and reproducibility are resolved.

Measure the Wald 40 m→20 m reconstruction task; 20 m round-trip closure after
super-resolution; B11/B12 bias, MAE, RMSE, spectral angle, and alignment; OpenSR-Test
hallucination/omission/improvement metrics; MBMP background mean and variance; plume SNR;
and stability across crop offsets, flips, and overlap choices.

Predeclare calibration-derived tolerances. Reject a method before downstream evaluation if
it worsens low-frequency closure, no-plume MBMP variance, alignment, or hallucination rate
relative to the best conventional resampler.

### Stage 2 — required negative controls

- Identical target/reference acquisitions must yield zero temporal evidence.
- Swapping target and reference must approximately invert signed temporal evidence.
- Spatially shuffle or shift the 10 m guide bands while retaining native SWIR; plume
  evidence must not follow displaced RGB/NIR edges.
- Use blurred and constant guide bands to isolate information supplied by native SWIR.
- Audit urban, industrial, agricultural, cloud-edge, water, and bright-surface negatives.
- Inject instrument-convolved synthetic methane into SWIR only and run a separate
  RGB-edge-only sham injection.
- Compare "super-resolve each pass, then compute MBMP" with "compute native MBMP, then
  interpolate."
- Downsample every claimed 10 m plume back to 20 m and require the physical evidence to
  remain present.

### Stage 3 — frozen-detector downstream evaluation

Feed each surviving preprocessing variant into an unchanged frozen detector. Do not tune
the detector, scene threshold, mask threshold, or postprocessing per variant. Compare with
the current conventional resampling contract and the native-resolution evidence.

Promotion requires all of the following:

- pooled scene AP delta at least +0.003;
- 10,000-replicate paired 25 km bootstrap AP lower 95% bound greater than zero;
- positive AP delta in each fold and each prespecified sensor/product stratum;
- nonnegative mean matched-FPR recall delta across 0.5%, 1%, 2%, 5%, and 10% FPR;
- no-plume FPR increase no greater than 0.2 percentage points;
- dense IoU delta greater than zero with paired-bootstrap lower bound greater than zero;
- no increase in false-positive plume area; and
- every Stage 1 radiometric/hallucination safety gate passed.

A method may be promoted for the dense branch alone if its localization gates pass while
its scene-ranking gates do not.

### Stage 4 — uncertainty-gated dual-stream fusion

Only if Stage 3 passes, build a small out-of-fold residual adapter. Preserve raw B11/B12 and
native MBMP; add SR B11/B12, high-frequency residuals, round-trip error, and transform or
tile-offset variance. Keep scene and dense objectives separate. Require the same gates and
at least two confirmation seeds before any protected evaluation.

### Stage 5 — VegiSR-inspired spectral translation

Keep this last and separate. Do not begin by reconstructing 285 bands as if they were
observed. A safer derivative is an NDVI-conditioned regularizer or compact teacher that
predicts an EMIT matched-filter/evidence score. It would require source-site-disjoint
Sentinel-2/EMIT pairs and real controlled-release confirmation.

## Artifact and claim policy

Commit only protocols, source/model hashes, compact metrics, and fixed representative plots.
Keep generated image banks and third-party weights ignored. Until real higher-resolution
SWIR truth exists, publication language should say **"10 m-grid spectrally constrained
sharpening"**, not "recovered 10 m SWIR." No MARS-S2L superiority claim follows from SR
image-quality metrics; only the existing task-level, site-disjoint promotion protocol can
support that claim.
