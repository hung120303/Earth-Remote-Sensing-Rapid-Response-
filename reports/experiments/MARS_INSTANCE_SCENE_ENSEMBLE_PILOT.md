# Conservative instance-scene ensemble pilot

Status: **rejected before full cross-fit or external scoring**.

An independently seeded instance-guided model completed three fixed
32,768-sample epochs with finite losses. Total loss fell from 0.359443 to
0.275545, the object gate tightened from 0.0640 to 0.0462, and successful
simulation remained 12.27-12.42%. The live held-fold rows aligned exactly with the
frozen current-score and pixel-count caches by label, sensor, and physical site.

The independently reproduced connected signal improved over the released U-Net
(AP 0.876105 versus 0.873419) but did not complement the much stronger current
cross-fitted ranker (AP 0.916603). All twelve preregistered blends failed:

- best: scene-head weight 0.025, AP delta -0.000004;
- its paired-site AP interval: [-0.001317, +0.001008];
- matched-FPR recall delta: -0.001255;
- minimum sensor AP delta: -0.000293;
- larger weights produced progressively larger AP losses.

Standalone signal AP was 0.876105 for connected correction, 0.867291 for the
real-scene head, and 0.671773 for proposal objectness. The frozen conservative mask
rule improved fold-2 point IoU by +0.010805, but the fold-2-only paired-site lower
bound was -0.002249; its earlier positive evidence applies to the combined
selection/confirmation cohorts, not this isolated fold.

The instance/object signal family is retired: it is useful relative to the released
model but redundant or harmful relative to the spatial-Prithvi ranker. No artifact
or score cache was written. No fresh CloudSEN or exact MARS-S2L paper input was
accessed. Full candidates and provenance are in
`mars_instance_scene_ensemble_pilot.json`.

