# Instance-guided physics teacher pilot

Status: **rejected before full cross-fit or external scoring**.

The 1,551,795-parameter model preserved the exact released MARS-S2L logits at
initialization, then trained a physics-guided feature adapter, plume-occupancy head,
component-center head, and object-gated bounded pixel residual. Synthetic examples
supervised pixel/object representation at half weight; only real examples drove
scene ranking, pairwise ordering, and teacher-direction penalties.

Training was healthy across three fixed 32,768-sample epochs. Total loss fell from
0.338580 to 0.268222; focal, Dice, object, real-scene, pairwise, and negative-upward
terms all improved. Successful simulation remained 12.14-12.25%, and the mean
object gate tightened from 0.0599 to 0.0452.

The fixed strength-0.25 endpoint produced the first useful new ranking signal since
the spatial-Prithvi ensemble:

- average precision: 0.876636 versus 0.873419 (delta +0.003147);
- matched-FPR recall: unchanged at 0.919699;
- Landsat AP delta: +0.000638;
- Sentinel-2 AP delta: +0.004058;
- 10,000-replicate paired-site AP interval: [-0.003005, +0.006813].

However, the same correction expanded predicted-positive pixels from 1,487,426 to
1,592,866 while adding only 29,942 intersecting truth pixels. Fixed-threshold IoU
therefore fell from 0.561886 to 0.555041 (delta -0.006802). The confidence interval
also crossed zero, so the pilot failed the IoU and paired-site gates.

The representation is not promoted as a standalone segmenter. Its ranking signal
will be tested only in a separately frozen, conservative two-output ensemble against
the stronger current development ranker; mask output will use the already
cross-fitted scene-gated sensor-threshold rule. No artifact was written. No fresh
CloudSEN or exact MARS-S2L paper input was accessed. Full metrics and provenance
are in `mars_instance_guided_teacher_pilot.json`.

