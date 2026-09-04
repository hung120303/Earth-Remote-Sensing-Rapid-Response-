# ERSRR Research Improvement Plan

> **Goal:** turn this capstone into a publishable research contribution on methane plume
> detection from Sentinel-2, with EMIT CH4 as the target. This document is a code-grounded
> audit of the current methodology, a literature scan of the state of the art, and a
> prioritized roadmap. Every claim about the current code cites `file:line`; every external
> recommendation cites a real paper.
>
> **Generated:** 2026-06-17. Based on code at commit `b3b6ba79` and the dataset audit at
> `reports/dataset_audit/` (generated 2026-06-08).

> **Status (2026-07-10): historical planning snapshot, not the final result.** The
> leakage grouping, physical-mask V002 pilot, shared model/serving contract, artifact
> validation, and repository-hygiene work proposed below have since been implemented.
> The synthetic-plume, foundation-model, MBMP, and flux-quantification sections remain
> research hypotheses, not measured improvements. Use
> `docs/ARCHITECTURE_DECISION.md` and `reports/ERSRR_RESEARCH_REPORT.html` for the current
> evidence-backed decision and quantitative results.

---

## TL;DR — Executive summary

Three findings dominate everything else:

1. **The single highest-leverage change is synthetic plume generation.** With ~100 paired
   tiles (and effectively ~92 independent scenes), training a ViT from scratch is
   data-starved. The closest published analog to this exact project — Rouet-Leduc &
   Hulbert, *Nature Communications* (2024) — got its ~10× detection improvement not from a
   cleverer backbone but from training on physics-shaped synthetic plumes injected onto
   plume-free S2 tiles. This is the contribution that makes the paper publishable.

2. **The current evaluation is not defensible as-is.** The train/test split is a random
   per-tile shuffle (`ERSRR_Model.py:517–524`) with no grouping by S2 scene or EMIT
   overpass, so near-identical scenes leak across the split. The validation folder (3
   tiles, only 1 EMIT-paired) is unused for metrics. The segmentation label is a per-tile
   top-decile of methane, not a physical plume mask. **Fix this before claiming any
   improvement number** — reviewers in remote sensing will reject patch-based DL results
   without a leakage-proof split.

3. **There is a clean, publishable minimum-viable paper within reach of this team:**
   implement the Varon et al. (2021) MBMP physics baseline, fine-tune a pretrained S2
   backbone (Prithvi-EO-2.0 or SatMAE++) on a fused EMIT + synthetic-augmented dataset,
   report detection-vs-flux curves under a spatial-blocked CV protocol, and add IME-to-flux
   quantification (NIST IR 8575). That is a methods contribution that fills a real gap.

The rest of this document backs these up with code references, data characterization, a
literature review, and a tiered roadmap with effort estimates.

---

## Table of contents

1. [Current methodology audit (code-grounded)](#1-current-methodology-audit-code-grounded)
2. [Dataset characterization](#2-dataset-characterization)
3. [Literature review — the state of the art](#3-literature-review--the-state-of-the-art)
4. [Prioritized improvement roadmap](#4-prioritized-improvement-roadmap)
5. [Cross-cutting additions](#5-cross-cutting-additions-cheap-fold-into-the-above)
6. [Recommended execution plan](#6-recommended-execution-plan-capstone-feasible)
7. [Repo-hygiene prerequisites](#7-repo-hygiene-prerequisites-do-these-first)
8. [References](#8-references)

---

## 1. Current methodology audit (code-grounded)

### 1.1 Architecture (`create_vit_encoder_decoder`, `ERSRR_Model.py:185–260`)

A Vision Transformer encoder + 3-stage convolutional decoder, multi-task with two heads:

- **Patch embedding** (`ERSRR_Model.py:80–107`): `extract_patches(size=8)` flattens each
  8×8×5 region into a 320-dim token → 1024 tokens for a 256×256 input.
- **PatchEncoder** (`:113–133`): `Dense(64)` projection + learned positional embedding
  (`Embedding(1024, 64)`). No `[CLS]` token.
- **Transformer encoder** (`:149–158`, `:197–200`): 8 blocks (post-norm),
  `MultiHeadAttention(num_heads=4, key_dim=64, dropout=0.1)`, MLP `[128, 64]` with GELU +
  dropout 0.1. Hardcoded `projection_dim=64`.
- **Skip connections** (`:196–200`): tokens at depths `[1, 4, 7]` reshaped to 32×32 and
  fed into the decoder. ⚠️ Hardcoded — breaks with `--tl < 8` (IndexError at `:223`).
- **Decoder** (`:169–178`, `:220–223`): three `Conv2DTranspose` blocks (filters 64 → 32
  → 16), each upsampling 2× then concatenating the bilinearly-resized skip. Functionally a
  loose U-Net over the 32×32 ViT feature map.
- **Two heads** (`:239–249`), sharing the final 256×256 feature map:
  - `regression_output`: `Conv2D(1, 1, sigmoid)` → methane intensity in [0, 1].
  - `mask_output`: `Conv2D(1, 1, sigmoid)` → binary plume mask.

Input `(N, 256, 256, 5)`, two outputs `(N, 256, 256, 1)`. Small model (low single-digit
millions of params — `projection_dim=64` is tiny vs a real ViT-S).

### 1.2 Losses

- **Regression** — `weighted_masked_mse` (`:284–297`): `weights = 1.0 + 4.0·y_true`,
  normalized by `sum(weights·valid_mask)`. NaN-aware (EMIT nodata −9999 masked). The `1+4y`
  weighting is mild in practice because it's normalized by weight-sum dominated by
  background. Hardcoded, not tunable.
- **Segmentation** — `plume_segmentation_loss` (`:337–338`): `0.5·masked_bce + 0.5·masked_dice_loss`,
  both weights hardcoded.
- **Multi-task balancing** (`:592–595`): `loss_weights = {regression: 1.0, mask: 1.0}`,
  equal, hardcoded. No dynamic balancing (GradNorm/uncertainty weighting).

### 1.3 Targets and preprocessing

- **EMIT normalization** — `normalize_emit_local` (`:344–378`): per-tile, `log1p` then
  2nd/98th-percentile clip to [0, 1]. ⚠️ **Strictly per-image** — each tile's target stats
  are recomputed independently.
- **Plume mask** — `create_plume_mask` (`:405–444`): ⚠️ the label is the **top-decile of
  log-methane per tile** (`percentile(valid_log, 90)` at `:431–432`). **Every tile is
  forced to be 10% "plume," including plume-free ones.** This is a relative proxy, not a
  physical plume mask — a fundamental issue (see §4.4).
- **Input normalization** — `process_dataset` (`:500–510`): a **single global max-abs
  scalar** across all 5 bands. ⚠️ The audit data shows SWIR bands (B11/B12) have ~10× the
  dynamic range of visible bands (B2/B3/B4); dividing all bands by one scalar (dominated
  by B12's tail) squeezes visible bands into a narrow range. Per-band normalization was
  present in the prototype `VisionTransformer.py:45–57` (`Normalization().adapt()`) but
  **dropped** in production.
- **Eval-side leak** — `errsr_model_prediction` (`:703–723`) denormalizes the prediction
  using the **test tile's own ground-truth EMIT percentiles**. Reported physical-unit
  numbers are partly circular.

### 1.4 Train/val split — the biggest methodological problem

`process_dataset` (`:517–524`):
```python
np.random.seed(42); np.random.shuffle(indices)   # pure random per-tile 80/20
```
- **No grouping** by S2 tile, EMIT overpass, geography, or date.
- The directory split (`Dataset/train_test` vs `Dataset/validation`) is **ignored** by
  training — only `train_test/` is walked (`:64`, `:456`). The 3 validation tiles are used
  only for a qualitative plot (`errsr_model_prediction(test_image)`, `:66`).
- `model.fit(validation_split=0.1)` (`:616`) holds out an internal 10% from the same
  shuffled pool.
- **Result:** the audit finds 51 unique S2 tiles among 99 train files (tile `T39SWV` alone
  contributes 10 files). Random shuffling guarantees same-scene tiles in both train and
  test. **Direct scene leakage is essentially certain.**

### 1.5 Evaluation metrics — thin

- During training (`:597–599`): regression reports `masked_mae`, `masked_mse`; mask head
  reports **only `masked_bce`**. **No IoU, F1, precision, or recall tracked.** Dice is in
  the loss but not reported.
- ModelCheckpoint monitors `val_regression_output_masked_mse` (`:605–606`) — **the
  segmentation head never influences checkpoint selection.**
- Final `model.evaluate` (`:621`) re-runs the same compiled metrics. **No segmentation
  F1/IoU computed on the test set anywhere.**
- CAFO matching (`Plume_Classifier.py:15–17`): `THRESHOLD=0.15` (absolute, on
  denormalized values), `DISTANCE=1.0 km`, greedy first-match (not Hungarian). Runs on a
  single tile, never aggregated, no PR curve, no confidence sweep.

### 1.6 Regularization, augmentation, training

- **Augmentation: none** in production (`process_dataset` reads tiles, stacks into numpy,
  `model.fit` directly). Flips/rotations exist only in the unused `VisionTransformer.py:45–54`.
- **Regularization:** dropout 0.1 in transformer blocks; AdamW `weight_decay=0.1` (`:582–584`,
  high). No early stopping, no LR schedule, no gradient clipping, no mixed precision. No
  dropout in the decoder or heads.
- **Data pipeline:** entire dataset loaded into RAM as numpy (`:500–502`); no `tf.data`,
  no lazy loading, no prefetch. Fine at 100 tiles; won't scale.
- `batch_size=4` default (`:50`) — ~63 train batches/epoch after the internal val split.

### 1.7 Bugs / correctness risks

1. **`--mode` silently ignored** (`:31, :41, :838–841`): `MODEL_SETTING="train"` is
   hardcoded and is the only thing `__main__` reads. `--mode predict` does nothing.
2. **Skip-index assumption** (`:199`): `if depth in [1, 4, 7]` IndexErrors when
   `--tl < 8`.
3. **Checkpoint filename divergence:** training writes `checkpoint.weights.h5` (`:65`);
   the web server reads `checkpoint_new.weights.h5` (`ERSRR_Website/server.py:31`);
   `convert_checkpoint.py` emits `checkpoint_k3.weights.h5`. **None match.** The server is
   loading a different (possibly stale) weights file than training produces.
4. **Two divergent model copies:** the full architecture is re-declared inline in
   `ERSRR_Website/server.py` (~lines 280–1040) rather than imported. Any architecture
   change must touch both files.

---

## 2. Dataset characterization

Source: `reports/dataset_audit/{SUMMARY.md,summary.json,pairings_manifest.csv}` (generated
2026-06-08), `docs/DATA_ACQUISITION.md`, and a Glob of the Dataset folders.

### 2.1 Counts

| Split | Files | Notes |
|---|---|---|
| `train_test` | 99 | the only split training reads |
| `validation` | 3 | only 1 is EMIT-paired; 2 are orphans (no EMIT id) |
| **Total** | **102** | all Glob/CSV/summary.json agree |

### 2.2 Effective independent sample size (the real n)

- **99 distinct EMIT plume ids** in train (zero plume-id duplicates) → effective n ≈ 99 at
  the plume level.
- Collapsing to **EMIT overpass granularity**: 3 exact-granule pairs + ~5 same-day
  12-second-offset pairs share one EMIT overpass across two adjacent S2 tiles → **~92
  distinct EMIT scenes** for train. This is the truest independence ceiling, not 99.
- **Validation effective n ≈ 1** (only 1 EMIT-paired tile; the other 2 have no EMIT source).

### 2.3 Leakage (concrete)

- S2 tile **`T34RET` appears in BOTH splits**: 2× in train (plumes `002181`, `003549`) and
  1× in validation (plume `003244`), same geographic tile in Egypt, validation date
  bracketed by the two train dates. This directly violates `DATA_ACQUISITION.md`'s own
  guidance ("keep held-out validation geographically/source separated from training").
- The 2 orphan validation tiles (`T14RMU`, `T13RFQ`) have no EMIT pairing — "unlabeled
  mystery TIFFs" the doc itself warns against.

### 2.4 Class balance — there are zero negatives

- `target_positive_pct_of_valid` minimum across all 102 files = **37.84%** (mean 75.95%).
  Every tile is plume-positive. **102 positive : 0 negative.**
- A binary plume/no-plume classifier cannot be trained/validated on this split as-is — the
  model never sees the true null class. Only a regression/enhancement task is currently
  supported.

### 2.5 Geographic spread

**Global, not Iowa** — despite the Iowa CAFO matching code, the training tiles span known
oil/gas basins worldwide: USA (California, Permian, others), Mexico, North Africa
(Egypt/Algeria/Libya), Middle East/West Asia (Turkmenistan, Uzbekistan, Iran, Saudi
Arabia), East/South Asia (China, Pakistan, Indonesia). Heavy clustering: `T39SWV`×10,
`T41SMC`×6, `T11SNR`×5 (top-10 tiles = 46% of files).

> **Implication:** the CAFO point-source evaluation is mismatched to the data — the model
> is trained on global oil/gas super-emitters but evaluated against Iowa CAFOs. Pick one
> domain or build both.

### 2.6 Temporal misalignment

- S2 dates span Aug 2022 – Oct 2024 (~26 months, uneven coverage).
- `days_between_s2_emit` (negative = S2 after EMIT): mean −5.95, median −2.92, **worst
  case −53 days**. Plumes dissipate in hours–days, so a 53-day gap decouples the S2 input
  from the EMIT target. Only ~5% of pairs are within ±0.13 days.
- Validation's only paired tile: −0.80 days (tight).

### 2.7 The doc's own stated gaps (`DATA_ACQUISITION.md`)

All currently unmet:
- Grow validation 3 → **50+ held-out tiles.**
- Keep held-out validation geographically/source separated from training (violated today).
- Add **200+ negative/background** S2 samples before heavier architectures.
- Track every pair in the audit manifest; don't train on unlabeled mystery TIFFs
  (2 orphan val tiles violate this).
- `target_has_metadata_mismatch = True` for **all 102 files** (EMIT −9999 nodata
  inconsistent with GeoTIFF metadata).

---

## 3. Literature review — the state of the art

### 3.1 Sentinel-2 methane detection (the baselines you must beat or cite)

- **Varon et al. 2021, *AMT* 14, 2771** — the foundational physics-based S2 retrieval.
  Three strategies using SWIR bands 11/12: **SBMP** (single-band multi-pass), **MBMP**
  (multi-band multi-pass, uses B11 as a surface proxy to correct B12), and a hybrid.
  Detection limit ~3–5 t/hr at optimum pixels. **This is the baseline every S2 methane DL
  paper is measured against.** Replicate it.
- **Rouet-Leduc & Hulbert 2024, *Nature Communications* 15, 3801** — **the closest analog
  to this project.** A ViT on S2, trained on **physics-shaped synthetic plumes** injected
  onto plume-free tiles, with a two-timestep stacked input. Reported ~10× detection
  improvement over MBMP, detecting plumes down to ~200–300 kg/hr. Code capsule published.
  **This is your template.**
- **CH4Net (Allen et al. 2024, *AMT* 17, 2583)** — U-Net for S2 super-emitter monitoring,
  trained on ~23 known sites, open-source. Your closest DL peer; cite and ideally
  re-baseline against it.
- **Robust Small Methane Plume Segmentation, arXiv:2508.16282 (2025)** — F1 = 78.39% on
  small plumes that Varon misses. Anchors your "SOTA ledger."
- **MEQNet (NeurIPS 2025 CCAI Workshop)** — joint segmentation + flux quantification.
  Closest to your multi-task design.

### 3.2 EMIT mission and products (your label source)

- **Thorpe et al. 2023, *Science Advances* 9, eadh2391** — the EMIT flagship paper. ISS
  imaging spectrometer, 60 m resolution, 52°N–52°S, matched-filter CH4 retrieval. First 30
  days: plumes 0.3–73 t/hr. **EMIT plume products are free COG + GeoJSON from LP DAAC —
  your ground-truth-ish label source.**
- **NIST IR 8575 (2025)** — community standard for IME-to-flux conversion and uncertainty.
  Cite whenever you report kg/hr.
- **Liu et al. 2024, *AMT* 17, 1633** — blind controlled-release evaluation of 10
  quantification systems. The gold-standard validation protocol; use it to frame your
  accuracy claims.
- **Tiemann et al. 2024, arXiv:2408.15122** — best taxonomy of ML methane methods
  (retrieval vs. segmentation vs. quantification).

### 3.3 Benchmarks and metrics

- **No standardized benchmark exists yet** — this is itself a publishable contribution.
  Candidate sources to unify: CH4Net's 23-site set, MPDataset (~4,172 EMIT samples),
  AI4CH4 (IBM/Climate Change AI, paired with Prithvi), the EMIT CH4PLM product.
- **De facto metric set:** pixel-level IoU / F1 / precision / recall; detection-at-flux-
  threshold; flux RMSE/MAE in kg/hr with wind uncertainty bands. Report all three layers.

### 3.4 Foundation models / pretrained S2 backbones

With ~100 tiles, training a ViT from scratch is nearly hopeless — **transfer learning is
the realistic path.**

- **Prithvi-EO-2.0 (NASA-IBM, 2024)** — MAE pretrained on HLS (Harmonized Landsat-S2),
  ~300M params, multiscale. HLS-native → strongest off-the-shelf S2 backbone for this task.
  Used by the AI4CH4 challenge. Weights/code permissive.
- **SatMAE / SatMAE++ (NeurIPS 2022 / arXiv 2403.05419)** — handles multi-band and
  multi-temporal input natively; SatMAE++ adds multi-scale pretraining, good for
  segmentation.
- **Scale-MAE (ICCV 2023)** — encodes ground-sample-distance explicitly; relevant for the
  EMIT (60 m) vs S2 (10/20 m) resolution gap.
- **Clay (Development Seed)** — open ViT-MAE across sensors; good "does a generic FM beat a
  methane-specific ViT?" baseline.
- **DOFA (Dynamic One-For-All)** — natively handles arbitrary band configs (your 5-band
  single-pass stack is non-standard; DOFA fits it cleanly).

⚠️ **Framework caveat:** all of these are **PyTorch** (HuggingFace / torchgeo). Your stack
is **Keras 3 / TensorFlow.** Options: (a) port to PyTorch (significant rewrite of
`ERSRR_Model.py` + `server.py`), (b) reimplement encoder in Keras + convert weights
(brittle), (c) **self-supervised MAE pretraining within your own Keras stack** on unlabeled
S2 tiles (zero framework friction, lower prestige). For a capstone, route (c) is pragmatic;
route (a) only if a teammate is fluent in PyTorch.

### 3.5 Synthetic plume augmentation

- **S2MetNet (Radman et al. 2023)** — WRF-LES-generated synthetic plumes composited onto S2
  tiles; shows synthetic-only training transfers to real plumes.
- **Rouet-Leduc & Hulbert (Nature Communications 2024)** — about 20,000 analytical
  Gaussian plumes with 2-D colored turbulence noise, embedded into real Sentinel-2 L1C
  backgrounds with Beer-Lambert attenuation; a ViT-U-Net is trained with unchanged real
  negatives. This is the directly relevant Gaussian-synthesis precedent.
- **Groshenry et al. (arXiv:2211.15429)** — transposes high-resolution PRISMA plume
  retrievals into Sentinel-2 scenes; it is a real-plume transfer method, not the source of
  the Gaussian generator previously attributed to it here.
- **Hybrid (field best practice):** WRF-LES for shape realism + Gaussian for wind sweeps →
  controllable flux labels + addresses class imbalance. **Most cost-effective path to a
  large labeled training set** — scale to thousands of synthetic chips per flux bucket.

### 3.6 Cross-sensor fusion (EMIT labels ↔ S2 input)

- **SEN2SR (Remote Sensing of Environment 334, 2026)** — provides a pretrained,
  radiometrically constrained `Reference_RSWIR_x2` model for sharpening native 20 m
  B11/B12 and red-edge bands onto a 10 m grid. This is the directly relevant learned-SWIR
  preprocessing proposal. It must be evaluated as constrained sharpening, not as measured
  10 m SWIR truth, because real 10 m SWIR ground truth is unavailable and the published
  methane case study did not establish significantly better detectability.
- **VegiSR (AAAI 2026)** — reconstructs an EMIT-like 285-band spectrum from Sentinel-2 at
  a shared 60 m grid. It is spectral translation, not spatial B11/B12 super-resolution.
  Its vegetation-aware regularization is a possible later research direction, but it must
  not be cited as evidence that 20 m→10 m SWIR sharpening improves methane detection.
- The preregisterable experiment ladder, hallucination controls, downstream gates, and
  source audit are recorded in
  `reports/research/LEARNED_SWIR_SUPER_RESOLUTION_PROPOSAL.md`. Status: **proposed; not
  implemented or evaluated**.

### 3.7 Temporal methods

- Methane detection on S2 classically uses **multi-pass (temporal) differencing** (Varon
  MBMP). The current model is single-image. Rouet-Leduc & Hulbert used a **two-timestep
  stacked input**. There is real value in (a) feeding a pre/post S2 pair and (b) reporting
  MBMP as the baseline — it converts a reviewer weakness ("you should compare to
  multi-pass") into a controlled comparison.

---

## 4. Prioritized improvement roadmap

Ranked by **impact-to-effort ratio** for a small capstone team. Effort is rough
person-weeks for one developer.

### Tier 1 — Do these first

#### 4.1 Synthetic plume generation (the core contribution)
Inject physics-shaped Gaussian/WRF-LES plumes onto plume-free S2 tiles to generate
effectively unlimited supervised data, then fine-tune on the ~100 real tiles. This breaks
the data-starvation bottleneck and gives a clean "we beat MBMP" headline. **Effort: medium
(~2–3 wks). Expected lift: order-of-magnitude sensitivity gain.** This is what makes the
paper publishable.

**Concrete first step:** reproduce the Rouet-Leduc/Hulbert design principles with an
analytical wind-aligned Gaussian column plume, 2-D colored turbulence noise, and real
unchanged negatives. For MARS-S2L, attenuate B11/B12 through the released sensor-specific
MODTRAN lookup for Sentinel-2 and Landsat rather than additive blending. Because the public
MARS enhancement assets have a documented ppm/ppb metadata conflict, do not claim physical
flux labels until units are reconciled; parameterize spectral strength in LUT input units and
freeze its range from authorized fit-fold statistics. Keep template banks disjoint by split.

#### 4.2 Leakage-proof evaluation protocol (gates every other result)
Before claiming any improvement: restructure the split to **prevent spatial/temporal
leakage.** Tiles from the same S2 scene or EMIT overpass must not span train/test. Use
**grouped k-fold CV keyed on EMIT overpass + S2 tile + region**, and report:
- per-scene and per-region metrics (not just global mean — tiny plumes dominate global IoU
  with noise),
- error bars: mean ± std across ≥5 folds and ≥3 seeds,
- detection metrics (P/R/F1 at a fixed flux threshold) **in addition to** segmentation IoU.

**Effort: low (~1 wk). Expected lift: no modeling lift, but mandatory for a credible
"we beat baseline X" claim** — the difference between accept and desk-reject.

**Concrete first step:** add a `group_split.py` utility that reads
`reports/dataset_audit/pairings_manifest.csv`, assigns each file a group key
`f"{s2_tile}|{emit_time[:8]}"` (and optionally a region bucket), and emits train/val/test
indices with `GroupKFold`/`GroupShuffleSplit`. Wire it into `process_dataset` to replace
the random shuffle at `ERSRR_Model.py:517–524`.

#### 4.3 Loss-function upgrade
- **Mask head:** replace `0.5·BCE + 0.5·Dice` with **Unified Focal** (or Focal Tversky
  with δ > γ to penalize false negatives — missed plumes are the failure mode). Add
  **Lovász-Softmax** as a second-stage fine-tune loss for direct IoU optimization.
- **Regression head:** switch to **log-space Huber** — predict `log1p(methane)` to let
  small plumes contribute meaningfully, and use Huber to resist the long tail from nodata
  and saturated bright plumes. The current `1+4y` weighting is a brittle approximation of
  this.

**Effort: low (~days). Expected lift: single-digit-point IoU/F1 gain on small plumes +
stable training.**

#### 4.4 Fix the segmentation label (physical plume mask, not top-decile)
`create_plume_mask` currently labels the top 10% of methane pixels per tile as "plume"
(`ERSRR_Model.py:431–432`). This forces every tile to be 10% plume, including plume-free
ones, and produces a label with no physical meaning. Replace with an **absolute
enhancement threshold** above background (e.g. EMIT CH4 enhancement > X ppb·m, validated
against the EMIT CH4PLM plume polygons from LP DAAC). Better: **use the EMIT CH4PLM
polygon product directly as the mask** rather than deriving a mask from the enhancement
field — the polygons are the canonical plume delineation.

**Effort: low–medium (~1 wk, mostly data work). Expected lift: turns the segmentation task
into a real, evaluable problem.**

### Tier 2 — High value, medium effort

#### 4.5 Foundation-model backbone (transfer learning)
Replace the from-scratch ViT encoder with a **pretrained S2 backbone** (Prithvi-EO-2.0 for
HLS-native, or DOFA for arbitrary-band), drop the existing decoder + two heads on top,
fine-tune. Given the framework caveat (§3.4), the pragmatic route is either a PyTorch port
(high effort, high prestige) or **in-stack self-supervised MAE pretraining** on unlabeled
S2 tiles (low effort, zero framework friction).

**Effort: high (PyTorch port, ~4 wks) or low (self-pretrain, ~1–2 wks). Expected lift:
~2–5 mIoU on S2 benchmarks vs from-scratch; larger upside on this tiny dataset.**

#### 4.6 Two-timestep input + explicit MBMP baseline
Add a `(T, T−Δt)` stacked input so the network learns change directly (Rouet-Leduc &
Hulbert's design). Implement or call the **Varon MBMP** retrieval as the physics baseline
and report detection-vs-flux curves. Converts "single-image DL vs classical multi-pass"
from a weakness into a controlled comparison.

**Effort: medium (~2–3 wks; cloud-free pairing across dates is the real constraint).
Expected lift: temporal robustness + a defensible benchmark table.**

#### 4.7 CH4Net as the direct DL comparable
Run CH4Net (or cite its published numbers) as your closest DL peer. Reviewers in this niche
expect it. **Effort: low–medium (run their repo) or low (cite numbers).**

### Tier 3 — Stretch goals (high prestige, high effort)

#### 4.8 Flux quantification (IME-to-flux with GEOS-FP winds)
Convert a plume mask to a flux estimate: `Q = IME · U_eff / L`, where IME is the integrated
methane enhancement over the mask, `U_eff` is the effective 10-m wind from GEOS-FP/ERA5,
and `L = √(mask area)` (Varon 2018; standardized by NIST IR 8575). Report flux with a
wind-uncertainty band; mask out low-wind (<2 m/s) cases where the parameterization blows
up. This converts the work from a "detection method" paper to an "emissions quantification"
paper — a higher-impact venue. **Realistic as a stretch goal or a follow-up paper.**

**Effort: high (~3–4 wks; GEOS-FP ingestion + unit conversions + uncertainty propagation).
Expected lift: different, higher-impact publication tier.**

#### 4.9 Cross-sensor fusion architecture
Keep two hypotheses separate: (1) conservative SEN2SRLite-style B11/B12 sharpening as an
auxiliary 10 m-grid input, and (2) VegiSR-inspired EMIT-like spectral translation at 60 m.
The first is a bounded preprocessing/dual-stream experiment; the second is a higher-risk
cross-sensor teacher or regularizer. Neither has been implemented or evaluated in ERSRR.

---

## 5. Cross-cutting additions (cheap — fold into the above)

- **Uncertainty quantification:** enable **MC dropout** at inference (keep existing dropout
  active, run ~20 stochastic passes, report mean + variance). Almost free, gives an
  uncertainty map reviewers like, and justifies log-space/Huber regression. Pair with
  seed-error bars from §4.2 for a complete UQ story.
- **Augmentation rules of thumb:** geometric flips/rotations by **90° steps** are safe.
  **Arbitrary rotation is risky** — plume orientation is physically tied to wind direction,
  so free rotation decorouples plume shape from the regression target. Use rotation only
  with a correlated wind feature. CutMix/MixUp are fine for the mask head; **synthetic
  plume injection (§4.1) supersedes them** for this task.
- **Per-band input normalization:** replace the single global max-abs scalar
  (`ERSRR_Model.py:504–509`) with per-band standardization. The audit data quantifies why —
  SWIR bands are ~10× visible bands. Re-add the `Normalization().adapt()` the prototype
  had (`VisionTransformer.py:45–57`). Low effort, real lift.
- **Mixed precision + larger batches:** add `keras.mixed_precision.set_global_policy("mixed_float16")`
  and a LR warmup + cosine schedule + gradient clipping. Stabilizes training, speeds it
  ~2×, lets you raise `batch_size` from 4. Low effort.

---

## 6. Recommended execution plan (capstone-feasible)

| Phase | Weeks | Work |
|---|---|---|
| **1. Hygiene + eval** | 1–2 | §7 (repo hygiene), §4.2 (grouped CV split + metrics), §4.4 (physical mask label) — gates everything |
| **2. Quick modeling wins** | 2–3 | §4.3 (losses), §5 (per-band norm, mixed precision, MC dropout), fix the `--mode`/checkpoint bugs |
| **3. Core contribution** | 3–7 | §4.1 (synthetic plume generation + fine-tune) — the publishable result |
| **4. Baselines + framing** | 7–9 | §4.6 (MBMP baseline + two-timestep input), §4.7 (CH4Net comparison), benchmark table |
| **5. Stretch** | 9+ | §4.5 (foundation backbone) if PyTorch port feasible; §4.8 (flux) as a second paper |

**Headline result to aim for:** *"Physics-informed synthetic plume training lets a ViT
detect S2 methane point sources below the MBMP threshold, validated under a
spatial-leakage-proof CV protocol, with calibrated uncertainty."* That is a publishable
methods contribution achievable with this team's resources.

**Target venues:** *Atmospheric Measurement Techniques* (AMT — open access, the home of
Varon/CH4Net), *Remote Sensing of Environment*, NeurIPS/ICML Climate Change AI Workshop,
or a methods track at IGARSS.

---

## 7. Repo-hygiene prerequisites (do these first)

These are cheap and prevent real correctness bugs. None change the science; all make the
science reproducible.

1. **Wire up `--mode` or remove it.** `ERSRR_Model.py:31` parses `--mode` but `:41`/`:838–841`
   ignore it. Either make `--mode predict` work or delete the flag. Same in `server.py`.
2. **Unify the checkpoint filename.** Training writes `checkpoint.weights.h5` (`:65`), the
   server reads `checkpoint_new.weights.h5` (`server.py:31`), the converter emits
   `checkpoint_k3.weights.h5`. Pick one, use it everywhere, and add a sha256 check at load
   time so staleness is detectable.
3. **Stop maintaining two model copies.** `ERSRR_Website/server.py` re-declares the whole
   architecture inline (~280–1040). Extract the model into an importable module that both
   `ERSRR_Model.py` and `server.py` import. Otherwise every architecture change silently
   desyncs training and serving.
4. **Validate `patch_size` divides `image_size`** (`:53–54`) and that `--tl ≥ 8` (or make
   the skip depths `[1,4,7]` at `:199` adaptive to `transformer_layers`).
5. **Fix the eval-side denormalization leak** (`:703–723`): don't use the test tile's own
   ground-truth EMIT percentiles to invert predictions for reported metrics.
6. **Run `git rm -r --cached ERSRR_Website/node_modules`** (~186 tracked files) and resolve
   the `ERSRR_Website/` vs `ERSRR_website/` casing duplicate. The `.gitignore` rules for
   `*.weights.h5`, `reports/dataset_audit/`, etc. were added after those files were
   committed, so they remain tracked — audit and untrack as appropriate.
7. **Version-control the dataset manifest** (`reports/dataset_audit/`) and record the
   exact commit + audit timestamp with every reported metric, so results are reproducible.

---

## 8. References

**Sentinel-2 methane detection**
- Varon, D.J. et al. (2021). *High-frequency monitoring of anomalous methane point sources with Sentinel-2.* AMT 14, 2771. https://amt.copernicus.org/articles/14/2771/2021/
- Rouet-Leduc, B. & Hulbert, C. (2024). *Automatic detection of methane emissions in multispectral satellite imagery using a vision transformer.* Nature Communications 15, 3801. https://www.nature.com/articles/s41467-024-47754-y
- Allen, A. et al. (2024). *CH4Net: Deep learning for Sentinel-2 methane plume detection.* AMT 17, 2583. https://amt.copernicus.org/articles/17/2583/2024/
- *Robust Small Methane Plume Segmentation* (2025), arXiv:2508.16282. https://arxiv.org/abs/2508.16282
- Groshenry et al. arXiv:2211.15429 (real PRISMA plume transposition into Sentinel-2 imagery; not Gaussian synthesis).

**EMIT and flux quantification**
- Thorpe, A.K. et al. (2023). *Attribution of individual methane and carbon dioxide emission sources using EMIT observations from space.* Science Advances 9, eadh2391. https://www.science.org/doi/10.1126/sciadv.adh2391
- Varon, D.J. et al. (2018). *Quantifying time-variant methane emissions from point sources.* AMT 11, 5673 (IME-to-flux). https://amt.copernicus.org/articles/11/5673/2018/
- NIST IR 8575 (2025). *Common Practices for Quantifying Methane Emissions from Plumes.* https://nvlpubs.nist.gov/nistpubs/ir/2025/NIST.IR.8575.pdf
- Liu, M. et al. (2024). *Blind controlled-release assessment of 10 methane point-source quantification systems.* AMT 17, 1633. https://amt.copernicus.org/articles/17/1633/2024/
- Tiemann et al. (2024). ML methane methods taxonomy, arXiv:2408.15122. https://arxiv.org/html/2408.15122v1

**Foundation models / backbones**
- Prithvi-EO-2.0 (NASA-IBM, 2024), arXiv:2412.02732. https://arxiv.org/abs/2412.02732
- SatMAE (Cong et al., NeurIPS 2022), arXiv:2207.08051. https://sustainlab-group.github.io/SatMAE/
- SatMAE++ (2024), arXiv:2403.05419.
- Scale-MAE (Reed et al., ICCV 2023). https://ai-climate.berkeley.edu/scale-mae-website/
- Clay Foundation Model (Development Seed). https://developmentseed.org/projects/clay/
- DOFA (Xiong et al., 2024), arXiv:2403.15356. https://huggingface.co/earthflow/DOFA

**Cross-sensor / super-resolution**
- VegiSR (AAAI 2026). https://openreview.net/forum?id=h3lBLmgIjj
- Aybar et al., SEN2SR, *Remote Sensing of Environment* 334 (2026), 115222.
  https://doi.org/10.1016/j.rse.2025.115222
- Lanaras et al., DSen2, *ISPRS Journal of Photogrammetry and Remote Sensing* 146
  (2018), 305–319. https://doi.org/10.1016/j.isprsjprs.2018.09.018
- ESA OpenSR-Test. https://github.com/ESAOpenSR/opensr-test

**Synthetic plumes**
- Radman, A. et al. (2023). *S2MetNet.* https://www.varon.org/papers/radman_etal_2023.pdf

**Evaluation / leakage**
- Ploton et al. and the spatial-CV literature; Remote Sensing review on patch-based DL
  evaluation bias (RS 17(8):1373, 2025).

---

*This document is a living plan. Re-run `python tools/ersrr.py audit` after any data change
and update §2; revisit §4 as baselines land.*
