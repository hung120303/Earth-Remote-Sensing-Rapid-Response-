# Balanced-request physics-guided adapter v2

Status: **rejected before full cross-fit or external scoring**.

V2 preserved the released MARS-S2L logits exactly at initialization and trained the
same 1,293,888-parameter physics adapter as v1. The only distributional change gave
exactly 0.25 request mass to each PLUME/NO_PLUME x Sentinel-2/Landsat stratum.
Across four fixed 32,768-sample epochs, successful simulation was
24.16-24.54% of all samples, close to the 25% theoretical ceiling and more than
twice v1's 10.4-10.8%.

Training was numerically healthy: total loss decreased from 0.318170 to 0.257249,
and focal, Dice, scene, and pairwise terms all decreased. The fixed strength-0.25
endpoint nevertheless regressed on the reused fold-2 development audit:

- average precision: 0.872230 versus 0.873419 (delta -0.001189);
- matched-FPR recall: 0.920954 versus 0.919699 (delta +0.001255, one extra TP);
- fixed-threshold IoU: 0.560083 versus 0.561886 (delta -0.001803);
- Landsat AP delta: +0.000116;
- Sentinel-2 AP delta: -0.000671;
- 10,000-replicate paired-site AP interval: [-0.005348, +0.002351].

The endpoint therefore failed the AP floor, Sentinel-2 direction, IoU, and
strictly-positive paired-site confidence gates. Equalized request sampling fixed
the measured simulation-frequency mismatch but did not improve generalization.
Increasing synthetic exposure is retired as an isolated architecture lever; future
simulation work must separate real and synthetic domains or use a curriculum rather
than treating them as exchangeable examples.

No artifact was written. No fresh CloudSEN or exact MARS-S2L paper input was
accessed. Full metrics, input identities, and provenance are in
`mars_physics_guided_teacher_balanced_pilot.json`.
