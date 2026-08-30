# Codex binding decision: sensor-aware ordinal pilot

## Scope

Run exactly two development-only outer endpoints:

- `H3`: fit on fold 4, select a checkpoint using an inner split carved only from fold 4, then score held fold 3 once.
- `H4`: fit on fold 3, select a checkpoint using an inner split carved only from fold 3, then score held fold 4 once.

Nothing outside folds 3/4 participates. No extra data, folds, external pretraining, pseudo-labels, auxiliary models, score blending, calibration model, synthetic Gaussian data, Prithvi, DOFA, released-MARS responses, rescue verification, reference bank, coordinates, prevalence, protected scores, or champion logits enter the candidate model. The frozen Gaussian+DOFA comparator is aligned only after both candidate endpoint predictions are immutable.

Enhancement values are used as ordinal producer-supplied ordering only. No physical-unit or quantitative-regression claim is permitted because repository evidence conflicts between ppm and ppb.

## Input contract

The model input is exactly 14 channels, in order:

1. Six target reflectance bands.
2. Six paired reference reflectance bands.
3. `radiometric_valid_mask`.
4. Binary cloud indicator (`1` for not clear, `0` for clear).

The spectral band order remains the native MARS six-band target/reference contract. Reflectance is clamped to `[0, 1.5]` and linearly scaled to `[-1, 1]`; support channels remain binary. `observable_mask` is support for loss and evaluation only and is never a model input.

## Architecture

Use separate sensor-specific stems with identical layouts but independent parameters:

- `Conv3x3 14->24`, `GroupNorm(6)`, `SiLU`.
- `Conv3x3 24->24`, `GroupNorm(6)`, `SiLU`.

Both stems feed one shared compact full-resolution U-Net:

- encoder 1: `24->32->32`
- encoder 2: `32->48->48`
- encoder 3: `48->64->64`
- bottleneck: `64->96->96`
- decoder 3: `160->64->64`
- decoder 2: `112->48->48`
- decoder 1: `80->32->32`

All convolutions are `3x3`, stride 1, same padding, GroupNorm, and SiLU. Downsampling is `2x2` max-pooling. Upsampling is bilinear x2 followed by skip concatenation. There is no attention, transformer, deep supervision, or pretrained component.

Heads from the final 32-channel full-resolution feature map:

- binary dense head: `Conv3x3 32->16`, `GN(4)`, `SiLU`, `Conv1x1 16->1`
- ordinal head: `Conv3x3 32->16`, `GN(4)`, `SiLU`, `Conv1x1 16->4`

The ordinal head emits `(a1,d2,d3,d4)` and cumulative logits are monotone by construction:

- `z1 = a1`
- `z2 = z1 - softplus(d2)`
- `z3 = z2 - softplus(d3)`
- `z4 = z3 - softplus(d4)`

`sigmoid(zk) = P(level >= k)` for `k=1..4`. Background is level 0. Hard ordinal class for artifacts is `sum(sigmoid(zk) >= 0.5)`.

## Branch-local ordinal targets

For each outer endpoint, first freeze the deterministic inner 25 km group split. Collect all finite producer enhancement values at `plume_mask & observable_mask` pixels from that endpoint's inner-training groups only. Fit `q25`, `q50`, and `q75` with `numpy.quantile(..., method="linear")`. These three values are frozen per endpoint before inner validation or held-fold inference and are applied unchanged to inner training, inner validation, and held-fold pixels.

Exact assignment:

- level 0: background
- level 1: positive enhancement `<= q25`
- level 2: `(q25, q50]`
- level 3: `(q50, q75]`
- level 4: `> q75`

Within-scene ranks are prohibited. Held-fold and inner-validation enhancement statistics never determine cutpoints. A positive pixel without a finite enhancement value is excluded from ordinal supervision; invalid pixels have zero dense and ordinal loss weight.

## Standalone scene head

Scene gradients are isolated from the stems, shared U-Net, binary head, and ordinal head.

Dense field:

- `p_dense = stopgrad(sigmoid(binary_logit))`
- weights `valid * p_dense`

Ordinal field:

- `m_ord = stopgrad((p1+p2+p3+p4)/4)`
- weights `valid * stopgrad(p1)`

For each field, pool weighted `mean, q50, q75, q90, q97` over valid soft-plume pixels. If total branch weight is zero, all five values are zero.

For the global feature, detach the 96-channel bottleneck, apply `Conv1x1 96->16`, `GN(4)`, `SiLU`, then valid-masked global average pooling. Concatenate 10 dense/ordinal scalars and the 16-dimensional global feature. The scene MLP is `26->32->16->1` with SiLU and no dropout. Only this projection and MLP receive scene gradients.

## Inner split and checkpoint rule

Global seed is `26082917`. Set Python, NumPy, and PyTorch RNGs, require deterministic algorithms, and run no additional seed.

The canonical MARS `group_id` is the frozen 25 km connected component; do not reconstruct groups from coordinates. Inside each outer fitting fold:

1. Mark a group positive if it contains any positive scene.
2. Split positive and negative groups separately.
3. Sort each class by `sha1(f"26082917:{group_id}")`.
4. Assign every fifth group (`index % 5 == 0`) to inner validation and the remainder to inner training.
5. If inner validation lacks a positive or negative scene, promote the earliest missing-class group from training.

Rank epoch checkpoints lexicographically on inner validation by:

1. higher scene AP;
2. higher aggregate dense IoU at probability threshold 0.40;
3. lower scene BCE.

Keep exactly one checkpoint per outer endpoint.

## Training

Dense/ordinal loader:

- batch 16 crops, targeting 8 positive sites and 8 negative sites when available;
- one scene/crop per sampled site;
- positive crop: 256x256 centered on a valid positive pixel;
- negative crop: random 256x256 crop with valid fraction at least 0.70;
- zero-pad outside bounds;
- only shared horizontal flip, vertical flip, and 90-degree rotation.

Scene loader:

- batch 4 native full images, targeting 2 positive and 2 negative sites;
- one scene per site;
- pad spatial dimensions to a multiple of 8;
- no resize, tiling, TTA, or synthetic data.

Schedule:

- epochs 1-4: 600 dense/ordinal steps only;
- epochs 5-24: 600 dense/ordinal steps, then 150 scene-only steps.

Pixel optimizer: AdamW, LR `3e-4`, betas `(0.9,0.999)`, weight decay `1e-4`, two-epoch linear warmup, cosine decay to `3e-5`.

Scene optimizer: AdamW, LR `1e-3`, betas `(0.9,0.999)`, weight decay `1e-4`, one-epoch warmup beginning at epoch 5, cosine decay to `1e-4`.

Losses:

- `L_bin = 0.7 * masked BCEWithLogits + 0.3 * masked soft Dice`
- `L_ord = mean` of four observable/ordinal-supported BCE losses against cumulative targets
- `L_pixel = L_bin + 0.5 * L_ord`
- `L_scene = BCEWithLogits` with balanced batches and no extra class weighting

Pixel and scene use separate forward/backward/update passes. There is no joint scalar loss and no scene gradient into the pixel model.

## Evaluation and promotion

Held inference occurs once per held fold using the selected checkpoint. Candidate scene score is `sigmoid(scene_logit)`. Dense prediction is `sigmoid(binary_logit) >= 0.40`. Scene threshold 0.50 is metadata-only and does not affect AP.

Align candidate and comparator by exact sample identity, fold, sensor, and canonical 25 km group. The scene comparator is frozen Gaussian+DOFA cache SHA-256 `988b98c92a1a5fa1fe52d7052b9159352f0fadd876b400fce1c8c879c94ea424`. The dense comparator is its already frozen Gaussian dense endpoint/rule reconstructed from the frozen dense state/evidence; the scene-only NPZ is insufficient for dense comparison.

Metrics:

- exact `sklearn.metrics.average_precision_score` for pooled, per-fold, and per-sensor scene AP;
- matched-FPR recall delta is the mean over FPR targets `{0.005,0.01,0.02,0.05,0.10}` of `max recall candidate(FPR<=f) - max recall comparator(FPR<=f)`;
- dense IoU is aggregate observable-pixel IoU: compute per-scene TP/FP/FN, sum counts, then `TP/(TP+FP+FN)`.

All gates are required:

- pooled AP delta at least `+0.003`;
- AP delta strictly positive in held folds 3 and 4;
- AP delta strictly positive for pooled Sentinel-2 and Landsat;
- matched-FPR recall delta at least zero;
- 10,000-replicate paired 25 km group AP-delta lower bound strictly above zero;
- aggregate dense IoU delta strictly above zero;
- 10,000-replicate paired 25 km group dense-IoU-delta lower bound strictly above zero.

AP bootstrap: paired, stratified within outer fold, 10,000 group replicates, seed `26082918`, 2.5th percentile lower bound. Dense bootstrap: paired, stratified within outer fold, 10,000 group replicates, seed `26082919`, reaggregate TP/FP/FN in every draw, 2.5th percentile lower bound.

If seed `26082917` fails any gate, stop with no second seed, no retuning, no replication, and no protected, external, fold-0/1/2, or official outcome access. If it passes, report and stop for Codex review.

## Artifact rules

Bulk checkpoints, cutpoint caches, prediction arrays, and dense comparator reconstruction stay under ignored `.research/`. Never commit them. Commit only reproducible code/config/tests and compact JSON/Markdown reports. No output may overwrite an existing one-shot result.

## Frozen constants checklist

- outer endpoints: fold 4 -> 3 and fold 3 -> 4
- held scoring: once per endpoint
- seed: `26082917`
- input: exact 14 channels
- stems: separate sensor `14->24->24`
- U-Net widths: `32,48,64,96,64,48,32`
- dense/ordinal hidden widths: 16/16
- ordinal: four monotone cumulative logits
- cutpoints: inner-training-group-only q25/q50/q75, NumPy linear quantiles
- scene descriptor/MLP: `26`, then `26->32->16->1`
- scene gradient into pixel model: none
- inner validation: hash-stratified one-fifth canonical groups
- crop/batches: 256, dense 16, scene 4
- epochs/steps: 24; dense 600/epoch; scene 150/epoch after epoch 4
- pixel optimizer: AdamW `3e-4`, wd `1e-4`
- scene optimizer: AdamW `1e-3`, wd `1e-4`
- pixel loss: `0.7 BCE + 0.3 Dice + 0.5 ordinal BCE`
- dense threshold: 0.40
- scene artifact threshold: 0.50
- matched-FPR grid: 0.005, 0.01, 0.02, 0.05, 0.10
- bootstrap: 10,000; seeds `26082918`, `26082919`
- AP gate: pooled `>= +0.003`, each fold/sensor `> 0`, AP lower `> 0`
- recall gate: matched-FPR mean delta `>= 0`
- dense gate: aggregate IoU delta `> 0`, lower `> 0`
- retained checkpoints: one per endpoint under `.research`
- reruns after failure: none
