# ADR: full-resolution proposal-aware methane detector v3

- Status: accepted for the next leakage-safe research experiment; not approved for deployment
- Date: 2026-07-10
- Decision evidence through revision: `26871b20ee3f52107b8c7ab85844383b7f31e9fe`
- Supersedes: the compact primary-research choice in `ARCHITECTURE_DECISION.md`

## Decision

Train `ersrr_mars_full_unet_proposal_v3` from scratch on the frozen MARS-S2L internal fit groups.
Use a 14,268,915-parameter full-resolution GroupNorm U-Net with three outputs:

1. a dense plume segmentation map;
2. a proposal-aware scene-presence score for plume / abstain / no-plume calibration;
3. an observability score that is not promoted until cloudy and invalid scenes are represented.

The presence descriptor combines deep average/max context, a learned 16-channel component embedding
gathered at high-evidence pixels, and differentiable area, centroid, covariance, compactness, total
variation, and along-/cross-wind shape features. Mask top-k confidence is only an input to this
descriptor, not the decision rule. The same compact dense embedding supports a separately calibrated
connected-component classifier without exporting the full 128-channel decoder tensor.

Do not initialize the primary v3 experiment from the released MARS-S2L checkpoint. That checkpoint
trained on the official training rows from which ERSRR's internal validation groups are drawn, so
warm-starting it would contaminate validation. Released weights remain a fixed external baseline.

## Input and label contract

The input has 16 ordered channels on the native 200 x 200, 10 m MARS grid:

```text
release MBMP
target B02 B03 B04 B08 B11 B12
reference B02 B03 B04 B08 B11 B12
wind_u / 8
wind_v / 8
binary cloud/invalid indicator
```

Sentinel-2 integers are divided by 5,000 and clipped to `[0,2]`, matching the released loader.
Only cloud class `0` and nonzero radiometry are observable. Positive masks are evaluated only on
observable pixels. Methane-enhancement rasters are deliberately absent from detector training: their
units are inconsistent upstream and none of the v3 losses consume them.

Geometric augmentation rotates/flips the wind vector with the imagery. Input channel order,
normalization, model metadata, thresholds, checkpoint identity, source manifest identity, and Git
revision are part of the artifact contract.

## Evidence behind the decision

All figures below use the same 579-scene, 150-group strict-spatial development cohort. The released
checkpoints use their authors' fixed `>0.5` / 100-connected-pixel rule; ERSRR thresholds were selected
on internal validation before their one-time strict evaluation.

| Model | Scene recall | Specificity | FPR | AUROC | Pixel AP | Pixel Dice |
|---|---:|---:|---:|---:|---:|---:|
| MBMP | 0.015 | 0.990 | 0.010 | n/a | 0.0165 | effectively 0 |
| Pixel logistic | 0.000 | 0.988 | 0.012 | n/a | 0.0092 | effectively 0 |
| Joint v1 | 0.015 | 0.914 | 0.086 | n/a | 0.0355 | 0.0983 |
| Joint MIL v2 | 0.149 | 0.988 | 0.012 | 0.752 | 0.0611 | 0.1295 |
| Released CH4Net | 0.164 | 0.912 | 0.088 | 0.597 | 0.0069 | 0.0150 |
| Released MARS-S2L | **0.642** | **0.922** | 0.078 | **0.822** | **0.4943** | **0.5303** |

The released MARS-S2L result proves that full-resolution capacity, bitemporal context, and training
scale matter. CH4Net's failure at nearly the same parameter count proves that capacity alone does not.

The MIL-v2 internal-validation audit shows why v3 must change the presence mechanism:

- presence versus segmentation top-1% confidence has Spearman rho `0.909`;
- the smallest plume quartile has only `9.4%` presence recall while its mask proposal recall is `59.4%`;
- true-positive median plume area is `1,515` pixels versus `643` for false negatives;
- all 12 false-positive scenes have a saturated mask proposal;
- six of 12 false positives occur in the sampled Kazakhstan stratum;
- MBMP top-1% magnitude is higher for false positives than ordinary true negatives, but nearly equal
  between true and missed plumes.

V3 therefore uses explicit proposal morphology/context, small-plume loss weighting, and group/class
balanced sampling. It does not add another global/top-k pooling variant.

## Leakage-resistant training and evaluation

- Fit: 23,763 samples / 98 frozen 25 km groups / 2,007 positives.
- Internal validation: 5,945 samples / 24 disjoint groups / 505 positives.
- Strict-spatial development benchmark: 4,401 samples / 150 groups, retained only for a frozen-candidate evaluation.
- Independent confirmation: untouched time-aligned EMIT V002 groups after the architecture and thresholds are frozen.

The minimum v3 corpus contains image, cloud-mask, and positive plume-mask assets only. It totals
61,928 assets / 30,366,803,325 bytes (28.281 GiB); the verified development tranche reuses 2,688
assets, leaving 29,198,856,248 bytes (27.193 GiB) to download.

Checkpoint selection maximizes validation recall subject to observed FPR <= 0.05, then validation AP
and positive-mask Dice. Small positive examples receive bounded inverse-square-root area weighting.
No strict-test result may change the architecture, loss, threshold, or component rule.

## Promotion boundary

V3 is not an operational detector until all of the following hold:

1. group-bootstrap lower 95% recall bound at least 0.75;
2. FPR at most 0.05 and specificity at least 0.95 on representative negatives;
3. at least 25% relative FPR reduction versus the strongest reproduced baseline at non-inferior recall;
4. calibrated plume/no-plume thresholds with explicit abstention and accepted no-plume NPV;
5. five fixed seeds and site-block confidence intervals;
6. unchanged performance claims on an untouched, time-aligned EMIT V002 confirmation cohort;
7. a model card limiting supported sensor, product level, preprocessing, and operating thresholds.

Until those gates pass, the web application must continue to label model output as research-only and
must not silently replace the packaged legacy artifact with v3.

## Reproducibility record

- Model: `EarthRemoteSensingRapidResponse/mars_v3_model.py`
- Trainer: `tools/train_mars_v3.py`
- Unit tests: `tests/test_mars_v3_model.py`
- Minimum-corpus builder: `tools/build_mars_v3_training_cohort.py`
- Frozen cohort evidence: `reports/acquisition/MARS_S2L_V3_TRAINING_COHORT.md`
- Pipeline smoke evidence: `reports/experiments/MARS_V3_SMOKE.md`
- Publication protocol: `configs/mars_publication_protocol.json`
- Full plan and authenticated data handoff: `docs/PUBLICATION_ROADMAP.md`
