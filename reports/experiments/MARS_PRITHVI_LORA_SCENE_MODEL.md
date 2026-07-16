# Patch-supervised Prithvi LoRA pilot

Status: **rejected before external scoring**.

The checksum-frozen seed-20261800 pilot trained 130,754 parameters: rank-4 LoRA
residuals in Prithvi's final four attention blocks plus patch and scene heads. All
training losses were finite and decreased, but the learned score did not improve
unseen-location ranking.

At the selected minimum blend of 0.05, selection AP changed by -0.000489 with a
10,000-replicate paired-site interval of [-0.001637, +0.000990]. Matched-FPR recall
changed by -0.000431. Fold AP deltas were -0.001691, +0.000203, and +0.000603 for
folds 2/3/4; fold 2 recall changed by -0.003764. The reused fold-0/1 audit also
failed: AP changed by -0.000514 with interval [-0.001526, +0.000361], and both fold
recalls changed by -0.001342.

Increasing the candidate weight made selection AP progressively worse: -0.001455,
-0.005474, -0.013184, and -0.052206 at weights 0.10, 0.20, 0.30, and 0.50. This is
evidence against promoting or merely reweighting this scene-score family, not a
case for searching a smaller numeric blend on exposed folds.

No model artifact or development score cache was written. No fresh CloudSEN or
exact MARS-S2L paper input was accessed, and the required second seed was not run.
Full machine-readable metrics and provenance are in
`mars_prithvi_lora_scene_model.json`.
