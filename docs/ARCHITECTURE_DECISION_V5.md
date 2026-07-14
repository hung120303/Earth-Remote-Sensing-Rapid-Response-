# Architecture decision record: ERSRR v5/v5.1

## Status

**Frozen and externally evaluated. Do not retune from the MethaneS2CM location test.**

The v5.1 architecture, seeds, calibration, scene thresholds, pixel threshold, acquisition code, and
comparison evaluator were committed before the location-test imagery was opened. The primary result
is immutable in `reports/experiments/methanes2cm_v5_1_location_test.json`.

## Decision

Adopt a shared-weight tri-temporal segmentation model with a controlled small context head:

- frames: T, T-90, and T-365;
- bands per frame: B02, B03, B04, B08, B11, and B12;
- physics: MBMP(T,T-90) and MBMP(T,T-365), centered within the model;
- encoder: one shared encoder applied independently to all three times;
- fusion: target, both references, signed differences, absolute differences, and resized MBMP at
  five scales;
- output: full-resolution segmentation probability;
- mask-derived scene logit: top 1% observable mask logits with a 15% maximum contribution;
- v5.1 context logit: 128-unit MLP over mean/max fused bottleneck plus mean/max/min centered MBMP;
- final scene logit: 35% mask-derived plus 65% context;
- parameters: 9,358,256 per seed; and
- ensemble: seeds 1101, 2202, and 3303 with equal per-seed empirical-CDF scene averaging and equal
  dense-probability averaging.

## Why v5 replaced the earlier direction

V3 demonstrated that low false-positive rate alone was not useful when recall collapsed across
geography. V4.2 and the cascade did not clear development gates. V4.3 recovered much stronger
ranking and a lower strict-MARS FPR, but remained bitemporal, product-specific, and weaker than
released MARS-S2L on strict-cohort recall and overlap.

MethaneS2CM adds a third temporal frame, dense binary masks, balanced plume/no-plume crops, and an
official exact-location test. The v5 direction therefore prioritized:

1. temporal consistency beyond one reference;
2. weight sharing rather than three independent encoders;
3. physics channels available throughout the network rather than an external threshold;
4. a dense output retained as a first-class endpoint; and
5. spatially grouped, sealed evaluation.

## Why v5.1 added context

The original v5 scene score was entirely mask-derived. A source-mask audit found:

- positive median mask area: 226/1,024 pixels;
- positive mean: 378.6 pixels;
- 2,552 positive crops at least 50% masked;
- 758 at least 90% masked; and
- 103 fully masked.

This geometry means scene identity can be broad and contextual rather than a compact connected
plume. A small context head was the narrowest controlled change that addressed that mismatch while
retaining dense segmentation. The 65/35 fusion was frozen before the multi-seed campaign.

Seed 1101 improved from mask-only v5 to v5.1 as follows:

| Metric | V5 | V5.1 | Delta |
|---|---:|---:|---:|
| Scene AP | 0.8225 | 0.8542 | +0.0318 |
| AUROC | 0.8341 | 0.8572 | +0.0231 |
| Recall at ~5% FPR | 0.3421 | 0.4321 | +0.0900 |
| Pixel Dice | 0.4318 | 0.4366 | +0.0048 |
| Pixel IoU | 0.2753 | 0.2792 | +0.0039 |

Pixel AP fell by 0.0056, so the context head is not presented as a universal dense-metric
improvement. It improved the primary scene bottleneck without materially sacrificing overlap.

## Evidence that learning, not a threshold, drives the result

The strongest physics-only development baseline achieved scene AP 0.5509, AUROC 0.5773, recall
0.0479 at 0.0499 FPR, and pixel AP 0.1692. V5 seed 1101 achieved AP 0.8225, AUROC 0.8341, recall
0.3421, and pixel AP 0.3171. The learned model therefore uses temporal and spatial context beyond a
simple high/low MBMP rule.

## Frozen ensemble evidence

The v5.1 seed mean achieved AP 0.8481, AUROC 0.8524, recall 0.4125 at approximately 5% FPR, pixel
AP 0.3028, Dice 0.4330, and IoU 0.2763. The final all-development ensemble achieved AP 0.8658,
AUROC 0.8655, recall 0.4671 at 0.0499 FPR, pixel AP 0.3090, Dice 0.4424, and IoU 0.2840.

On the sealed location test, it achieved AP 0.8180, AUROC 0.8276, recall 0.3778, FPR 0.0607, pixel
AP 0.2083, Dice 0.3125, and IoU 0.1852. Ranking transferred better than the operating threshold and
dense boundary quality.

## MARS-S2L decision

V5.1 strongly exceeds the released MARS-S2L zero-shot comparator on MethaneS2CM AP, AUROC, recall,
pixel AP, Dice, and IoU. However, its FPR is 0.0607 versus 0.0033 for MARS-S2L. The released model's
recall is only 0.0052, but the predeclared claim still required lower no-plume error. V5.1 therefore
does not establish across-the-board superiority.

This result does not invalidate the architecture. It distinguishes two problems:

- **representation/ranking:** materially improved; and
- **operating-policy transfer:** not yet controlled across spatial/product shift.

## Rejected actions

- Do not raise the scene threshold from test labels.
- Do not choose the frozen 2% development rule because it looks safer on test.
- Do not replace a seed or alter the 65/35 context fusion from test behavior.
- Do not tune connected components or pixel threshold from test masks.
- Do not treat MARS's near-zero test recall as permission to ignore v5.1 FPR.
- Do not report the balanced-crop precision as deployment PPV.

## V5.2 architecture/research call

Keep the tri-temporal shared encoder as the reference architecture. The next controlled campaign
should test three changes, in this order:

1. **Prospective risk control:** group-held calibration or conformal FPR control fitted only on new
   calibration groups, with coverage-risk curves and geography-stratified diagnostics.
2. **Product-aware domain generalization:** L1C/L2A harmonization, a product token or normalization
   layer, and missing-frame handling so MARS-S2L and MethaneS2CM can share a development protocol.
3. **Dense-boundary quality:** mask-quality weighting, plume-scale sampling, and boundary-aware loss
   evaluated without weakening no-plume scene balance.

Only development evidence may select among these. V5.2 requires a new geographically isolated,
prevalence-aware confirmation cohort sealed before final model selection.
