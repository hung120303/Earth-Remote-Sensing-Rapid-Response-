# Physics-guided released-U-Net adapter pilot

Status: **rejected before full cross-fit or external scoring**.

The adapter preserved the released MARS-S2L logits exactly at initialization and
trained 1,293,888 parameters on folds 3/4 for the fixed three-epoch endpoint. All
losses were finite. The inherited sampler realized only 10.4-10.8% simulated scenes
because simulation was attempted for half of the comparatively rare positive
requests, not for half of all requests.

The endpoint showed a promising but statistically insufficient fold-2 signal. At
strength 0.25, AP improved +0.003174, matched-FPR recall +0.003764, and fixed-mask
IoU +0.009601. Landsat and Sentinel-2 AP improved +0.000247 and +0.005036. However,
the 10,000-replicate paired-site AP interval was [-0.004576, +0.007349], failing
the preregistered strictly positive lower-bound gate.

Strength 0.125 improved AP/recall/IoU by +0.001646/+0.001255/+0.005373 but also
failed the AP floor and confidence gate. Strength 0.5 improved recall +0.008783 and
IoU +0.015584 but slightly reduced Landsat AP. Strength 1.0 overcorrected, reducing
AP and IoU. No strength passed all gates.

No artifact was written. No fresh CloudSEN or exact MARS-S2L paper input was
accessed. Full metrics and provenance are in
`mars_physics_guided_teacher_pilot.json`.
