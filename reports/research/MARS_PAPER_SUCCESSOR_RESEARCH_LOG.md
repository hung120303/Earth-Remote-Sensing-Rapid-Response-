# MARS-S2L paper successor research log

This log is the narrative spine for the planned paper and final HTML report. Numerical artifacts remain authoritative; this document records hypotheses, architecture decisions, negative results, and interpretation boundaries.

## Research question

Can a mixed Sentinel-2/Landsat methane-plume detector improve both scene retrieval and pixel segmentation over the released MARS-S2L system on the authors' exact 2024 evaluation cohort, including the full 43,529-scene view and the 15,655-scene test-only-site view, without increasing false-positive rate?

## Frozen comparator

Primary source: [MARS-S2L paper v3](https://arxiv.org/html/2511.21777v3), revised 2026-04-24. Code source: [UNEP-IMEO-MARS/marss2l](https://github.com/UNEP-IMEO-MARS/marss2l), pinned locally at `f7d264c2c845dfba1cb27f76ef6026275f8d8758`.

| View | Scenes | Plume | Sites | Paper AP | Recall | FPR | Pixel IoU |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full official test | 43,529 | 1,813 | 1,289 | 0.6408 | 0.7915 | 0.0713 | 0.3224 |
| Test-only sites | 15,655 | 227 | 697 | 0.4496 | 0.7753 | 0.0763 | not reported |

The archived per-scene reconstruction is a slightly stronger comparator than the rounded table: full AP 0.64102, recall 0.791506, FPR 0.070692, and IoU 0.324365; test-only AP 0.450274, recall 0.775330, and FPR 0.075512. Exact identities and hashes are in `reports/acquisition/mars_s2l_paper_v3_benchmark.json`.

## Evidence and data boundary

- Public successor development uses the pinned Hugging Face `train.csv` and `val.csv`: 44,363 scenes across 618 physical sites, with 37,418 Sentinel-2 and 6,945 Landsat scenes.
- The released checkpoint names a private Azure March-2025 CSV that is not in the public release. Its counts, paper Table S3, and the later public split differ. This prevents exact reconstruction of the authors' fitting labels, but not exact reconstruction of their test assignment and outputs.
- Five unavailable historical test rasters and every missing pixel target receive adversarial candidate outcomes.
- Development and sealed-test rows are in physically separate ignored manifests. The development loader rejects `sealed_paper_test` rows.
- Five deterministic site-block folds produce exactly one out-of-fold development prediction per site. Calibration is cross-fitted; test access is one-shot after model and evaluator hashes are frozen.
- Bulk imagery, manifests, checkpoints, and prediction caches remain ignored. Only compact protocols, hashes, metrics, and reports are committed.

## Architecture decisions

1. Preserve the released detector as the starting point. The successor embeds an exact forward-equivalent released U-Net and loads checkpoint SHA-256 `be634fb9e24dc4877f44c1ff9f69972e6f0453e30d70c0dc03677876340ef246`.
2. Learn only a zero-initialized correction first. At initialization, independent and successor logits agree bit-for-bit for Sentinel-2 and Landsat identities; any later change is attributable to training.
3. Give the correction branch explicit centered MBMP, raw and normalized target-reference differences, log temporal ratios, wind, cloud, sensor identity, and released logits.
4. Penalize upward logit movement on known no-plume scenes. This encodes the requirement to mark no plume when none exists and makes false-positive regressions costly.
5. Evaluate with the authors' connected-component scene score and 0.5 probability/100-pixel mask rule on identical rows. Candidate-specific calibration may be considered only from cross-fitted development predictions and must be frozen before paper-test access.
6. Use one-sided released-model constraints during correction fitting: do not increase logits on known no-plume scenes, and do not decrease released logits inside annotated plume pixels. These constraints protect the two failure modes named in the research question while leaving the residual free to suppress background artifacts and add missed plume structure.

## Original-paper method audit

The [paper-v3 Methods](https://arxiv.org/html/2511.21777v3) establish several controls that matter when interpreting successor experiments:

- the released system is a 16-channel U-Net trained with pixelwise binary cross entropy, with positive pixels weighted by the retrieved per-pixel methane enhancement;
- the scene score is the minimum probability threshold supporting at least 100 connected pixels;
- training uses location/label-stratified sampling and physics-based injection of real training-plume enhancement fields into plume-free scenes;
- a synthetic plume is used about half the time on average, with much higher simulation probability at sites having few real plumes;
- source and target wind speeds must be within 1.5 m/s, target wind above 9 m/s disables simulation, and the plume is rotated to the target wind direction;
- the released model was trained for 170 epochs, 682 steps per epoch, batch size 96, Adam learning rate 0.0005, weight decay 0.000001, early stopping, and AP-based validation selection; offshore adaptation adds one real-data epoch.

The frozen primary ERSRR run learns only a 1.28 M-parameter correction over a frozen released teacher, so its smaller 393,216-sample budget is a deliberate efficient first test rather than a compute-matched retraining claim. If its fold-0 gate fails, the first source-aligned follow-up is enhancement-weighted residual fitting plus wind-consistent physics simulation using fitting-fold plume fields and fitting-fold no-plume backgrounds. That experiment must remain fold-0-only until its configuration is frozen.

Implementation audit: the earlier `mars_v4_simulation.py` wrapper admitted only LUT keys beginning with `S2`, even though the pinned integrated-transmittance LUT also contains `LC08`, `LC09`, `LE07`, `LT04`, and `LT05` entries with the same band-key schema. That silent Sentinel-2-only limitation is now removed and an `LC08` attenuation test passes. No simulation result is claimed yet; a mixed-sensor experiment still requires a frozen configuration and sensor-stratified reporting.

## Experiment ledger

| Date (UTC) | Experiment | Evidence | Decision |
|---|---|---|---|
| 2026-07-13 | ERSRR v4.3 three-checkpoint ensemble on the previously opened strict cohort | AP 0.3903 vs released 0.3521, but recall 0.5224 vs 0.6418 and IoU 0.0766 vs 0.1329 | Rejected; it did not outperform on recall or segmentation. |
| 2026-07-14 | Paper residual initialization audit | Maximum absolute logit delta 0 for both sensor identities | Passed; establishes a non-regressing starting architecture. |
| 2026-07-14 | Mixed-sensor residual smoke | 64 fit / 56 held-out scenes; AP delta approximately 0, recall delta 0, IoU delta -0.0011 after one tiny epoch | Pipeline-only success; no promotion or scientific claim. |
| 2026-07-14 | One-sided teacher-floor smoke | 64 fit / 64 held-out scenes; AP and recall deltas 0, IoU delta -0.000006 after one 128-sample epoch; both sensor-stratum regressions stayed within 0.0002 | Pipeline and constraint audit passed; the tiny, saturated smoke cohort cannot select the architecture. |
| 2026-07-14 | Full mixed-sensor development acquisition | 96,348/96,348 selected assets and 45,540,221,188 bytes independently hash-verified; zero missing or partial assets; receipt bound to manifest `31ba92e…e8e` | Passed; full fold evaluation and training are authorized, while the paper test remains sealed. |
| 2026-07-14 | Released-checkpoint reproduction on frozen development folds | Fold 0: 8,987 scenes, AP 0.88645, recall 0.95705 at FPR 0.07122, IoU 0.50863. Fold 1: 8,798 scenes, AP 0.85975, recall 0.93691 at FPR 0.07128, IoU 0.46937. Overall and Sentinel-2/Landsat wrapper deltas were exactly zero. | Passed; the exact connected-component evaluator and mixed-sensor wrapper are authorized for the preregistered fold-0 run. |
| 2026-07-14 | Complete development label audit | 6/3,811 positive scenes have producer-supplied raw empty masks; all six are `bad_retrieval` rows from one fold-4 site (4 Sentinel-2, 2 Landsat). No additional positive mask becomes empty only after observability filtering. | Preserve upstream semantics explicitly: scene label remains positive and pixel target remains empty; do not relabel or drop rows. |
| 2026-07-14 | Preregistered correction-only fold 0, seed 606 | Best epoch 7: AP 0.88599 vs 0.88645 (Δ −0.00046), recall 0.95570 vs 0.95705 (Δ −0.00134), IoU 0.49884 vs 0.50863 (Δ −0.00979). Landsat AP/IoU Δ −0.00118/−0.01385; Sentinel-2 +0.00002/−0.00943. | Rejected all promotion checks; do not access fold 1 with this architecture. Preserve artifact `b94880d8…c7d49` for fold-0 trust-region analysis. |
| 2026-07-14 | Frozen fold-0 correction trust region | Selected α=0.5: AP 0.88891 (Δ +0.00246), recall 0.95705 (Δ 0), IoU 0.51897 (Δ +0.01034). Landsat AP/IoU Δ +0.00031/+0.00526; Sentinel-2 +0.00361/+0.01073. Alpha zero reproduced the stored baseline exactly. | Rejected because recall was equal rather than strictly higher; do not access fold 1. Proceed to source-aligned fitting. |
| 2026-07-14 | Source-aligned residual smoke | 128 sampled crops, 27.34% simulated overall; CH₄-weighted BCE and complete mixed-sensor validation executed. Tiny-cohort AP Δ −0.00095, recall Δ 0, IoU Δ +0.00553. Twenty-four focused tests pass. | Pipeline-only pass; freeze a full fold-0 configuration before reading full results. |
| 2026-07-14 | Preregistered source-aligned residual fold 0, seed 707 | Selected epoch 4: AP 0.88651 (Δ +0.00006), recall 0.95570 (Δ −0.00134), IoU 0.49576 (Δ −0.01286). Epoch 3 was complementary rather than promotable: AP 0.88710 (Δ +0.00065), recall 0.95839 (Δ +0.00134; one additional true positive), IoU 0.49005 (Δ −0.01857). | Rejected; fold 1 remains unread. Preserve the endpoint result and recover the deterministic epoch-3 checkpoint solely for a preregistered interpolation with the already frozen alpha-0.5 correction model. |
| 2026-07-14 | Retained endpoint interpolation, primary alpha 0.5 to source epoch 4 | Exact endpoint assertions passed. Selected beta 1/32: AP Δ +0.00256, recall Δ 0, IoU Δ +0.01010; Sentinel-2 AP/IoU Δ +0.00379/+0.01057 and Landsat +0.00035/+0.00499. No interior beta raised recall. | Rejected on the strict recall gate; fold 1 remains unread. Stop spending experiments on this segmentation-only direction and introduce a separately trained scene-ranking head while retaining the stronger alpha-0.5 mask endpoint. |
| 2026-07-14 | Scene-ranking head inner selection, train folds 3-4 / validate fold 2 | Weighted logistic C=0.1 at head blend 0.25: AP 0.88733 vs 0.88060 (Δ +0.00673), recall 0.92723 vs 0.92597 (Δ +0.00125) at identical FPR 0.07118. Sentinel-2/Landsat AP Δ +0.00598/+0.00112. | Passed all inner gates. Refit on folds 2-4 and freeze ignored artifact `98ce79c6…a336c`; authorize one fold-0 extraction/evaluation after its evaluator is committed. |
| 2026-07-14 | Frozen scene-ranking head, one-shot fold 0 | Cache identity passed. AP 0.89752 vs 0.88645 (Δ +0.01107), recall 0.95570 vs 0.95705 (Δ −0.00134; one fewer true positive), FPR unchanged at 0.07122, IoU unchanged from the stronger mask endpoint at 0.51897 (Δ +0.01034). Sentinel-2/Landsat AP Δ +0.01445/+0.00249. | Rejected only on recall; fold 1 remains unread. The broad AP gain validates decoupling, but the next inner search must explicitly optimize hard-positive ordering and require a multi-TP recall margin before another fold-0 evaluation. |
| 2026-07-14 | Hard-example scene-head inner search | Best robust-ranked candidate (C=0.1, hard-positive ×2, hard-negative ×2, blend 0.25) retained AP Δ +0.00650 and sensor AP gains but improved fold-2 recall by only one TP (Δ +0.00125). No grid point achieved the required three-TP margin. | Rejected before fold 0; no model artifact created. Hard-example reweighting changes calibration more than ordering, so add label-free site-sequence context before further evaluation. |
| 2026-07-14 | Site-context scene-head inner search | HGB (31 leaves, min leaf 50, L2 10), blend 0.5: AP 0.89357 vs 0.88060 (Δ +0.01297); recall 0.93977 vs 0.92597 at identical FPR 0.07118 (Δ +0.01380; 11 extra TPs). Sentinel-2/Landsat AP Δ +0.01727/+0.00007. | Passed every gate and the three-TP robustness margin. Freeze folds-2/3/4 artifact `8334c7b5…fb284` and preregister evaluation on the existing fold-0 cache. |
| 2026-07-14 | Frozen site-context scene head, one-shot fold 0 | AP 0.90343 vs 0.88645 (Δ +0.01698), recall 0.95436 vs 0.95705 (Δ −0.00268; two fewer TPs), FPR unchanged at 0.07122, IoU Δ +0.01034. Sentinel-2/Landsat AP Δ +0.02287/+0.00604. | Rejected only on recall; fold 1 remains unread. The 11-TP fold-2 gain did not transfer, so replace single-inner-fold selection with three-fold out-of-fold recall stability. |
| 2026-07-14 | Three-fold OOF site-context selection on folds 2/3/4 | HGB (31 leaves, min leaf 20, L2 10), blend 0.625: pooled AP Δ +0.01437, recall Δ +0.00732 (17 extra TPs), Sentinel-2/Landsat AP Δ +0.01516/+0.00780. Recall Δ by held-out fold: +0.01129/+0.00923/+0.00261; AP Δ +0.01177/+0.01880/+0.02235. | Passed every OOF stability gate. Freeze refit artifact `2d014f54…c370` and preregister a new fold-0 evaluation on the unchanged cache. |
| 2026-07-14 | OOF-stable context head at selected blend 0.625, fold 0 | AP 0.90272 vs 0.88645 (Δ +0.01627), recall 0.95570 vs 0.95705 (Δ −0.00134; one fewer TP), FPR unchanged, IoU Δ +0.01034. Sentinel-2/Landsat AP Δ +0.02260/+0.00456. | Rejected only on recall. The model family transfers AP but the maximum-ranked blend over-intervenes at the discrete cutoff; freeze a minimum-intervention OOF trust region that selects the smallest blend satisfying stability. |
| 2026-07-14 | Minimum-intervention OOF context trust region | Smallest fully stable blend is 0.25: pooled AP Δ +0.01373, recall Δ +0.00302 (7 extra TPs), Sentinel-2/Landsat AP Δ +0.01448/+0.00783. Fold-2/3/4 recall Δ +0.00753/+0.00528/+0.00392 and AP Δ +0.01231/+0.01423/+0.02146. | Passed every OOF gate. Freeze blend 0.25 with the unchanged HGB artifact for one fold-0 evaluation. |
| 2026-07-14 | Minimum-intervention OOF context head, one-shot fold 0 | AP 0.90092 vs 0.88645 (Δ +0.01447), recall 0.95973 vs 0.95705 (Δ +0.00268; two extra TPs), FPR unchanged at 0.07122, IoU 0.51897 vs 0.50863 (Δ +0.01034). Sentinel-2 AP/IoU Δ +0.01905/+0.01073; Landsat +0.00516/+0.00526. | Passed every fold-0 gate. Freeze the architecture; authorize an untouched fold-1 confirmation using a fold-1 residual trained without fold-1 labels and a fixed epoch inherited from fold 0. |

## Frozen primary correction run

The first full architecture-selection run was frozen before the complete development download or full-fold baseline results were available.

- Code commit: `8f372969617c57cf745cb498df911342b818b648`
- Training script SHA-256: `df18d7b38f0f75ca1b5554ad78043fc5af3b99f4b0adefbbab179d0c0e167c8a`
- Model SHA-256: `73699f0263b264f83293351125b40e30a210f2b3b1150fb08e04465109974926`
- Fold protocol SHA-256: `6862182bdc1a14ec4a36cc33f318ba3b49a927789f3ff5be7801dc5162051873`
- Primary fold / seed: 0 / 606
- Epochs / samples per epoch / batch: 12 / 32,768 / 16
- Optimizer: AdamW, learning rate 0.0002, weight decay 0.0001, cosine schedule, patience 4
- Loss weights: scene 0.25, no-plume upward 0.25, plume-pixel downward 0.10, correction L2 0.002
- Selection rank: maximize the worse of AP and IoU deltas, then their sum, then recall delta at no more than 7.13% FPR.
- Promotion: AP, IoU, and recall deltas must all be positive; neither sensor may regress more than 0.01 AP or IoU. Fold 1 is not read for architecture decisions before this gate passes.

The frozen command is:

```text
python tools/train_mars_paper_residual.py --fold 0 --seed 606 --epochs 12 --samples-per-epoch 32768 --batch-size 16 --workers 8 --learning-rate 0.0002 --patience 4 --artifact EarthRemoteSensingRapidResponse/artifacts/mars_paper_residual_fold0_seed606.pt --output-json reports/experiments/mars_paper_residual_fold0_seed606.json --output-markdown reports/experiments/MARS_PAPER_RESIDUAL_FOLD0_SEED606.md
```

### Pre-result benchmark-integrity amendment

The first complete fold-0 equivalence audit stopped before emitting any fold
metric because the zero-initialized wrapper was not bitwise identical under
CUDA autocast. The released backbone emitted float16 logits, while float32
identity calibration parameters promoted the successor logits before sigmoid.
Commit `feb0c176b0fdb7ca6b1b1e7d6a2d595d7b511ac0` casts learned calibration to
the released-logit dtype and adds a CUDA autocast bitwise-equivalence test. No
metric, label, split, loss, selection rule, or training hyperparameter changed.
The corrected model SHA-256 is
`f9cc65b3bb764ed83a4aa67209cccdbad2b6c80826b0915711392ca8b10a0486`.
This amendment was recorded before rerunning the complete fold audit and before
starting the primary fold-0 training run.

The first primary launch then stopped in its data loader before the first
optimizer update when it sampled one of the public split's empty-mask positive
rows. A complete development-only audit found 6/3,811 such positives, all at
one fold-4 site and all marked `bad_retrieval`; upstream MARS-S2L loads these as
`isplume=1` with an all-zero pixel target. Commit
`0d5bb400220cdbbabaa5c9cd58bb672f412793d4` preserves that source behavior only
when the paper-cohort dataset opts in, while the general adapter remains strict
by default. No row was dropped or relabeled and no metric was observed. The
amended training-script SHA-256 is
`951fb2e0812c579496816b5031ce4a0c520f233c79a1d853c1883b7e1ad36b22`; the
adapter SHA-256 is
`60087bc668109e5146a42f96f996a14e34ccbff3164124cfeb3ff367b93625a3`.

## Frozen correction trust region

The correction-only run was rejected before fold 1, but its epoch-7 checkpoint
was close to the released baseline and later epochs showed overshoot. Before
evaluating any blend strength, the follow-up was frozen as
`released_logits + alpha * (trained_logits - released_logits)` on fold 0 only.

- Input artifact SHA-256: `b94880d858e1e7791591eeb5f7d0da9be84b99a324e980437ebe83cfae6c7d49`
- Evaluator commit: `34956c01fefda1ef8364387725411a84b6ac130c`
- Evaluator SHA-256: `02b94256fa7d9ac7cfc6693e25b013ddf2ad96485ed638b12b1cc9904ff83961`
- Alpha grid: 0, 1/32, 1/16, 1/8, 1/4, 3/8, 1/2, 3/4, 1
- Alpha 0 is an excluded-from-selection identity control and must reproduce the stored released baseline exactly overall and by sensor.
- Positive alphas use the primary run's balanced rank and identical AP, recall, IoU, and sensor non-regression promotion gates.
- Fold 1 remains unread unless a positive alpha passes every fold-0 gate.

The frozen command is:

```text
python tools/evaluate_mars_residual_trust_region.py
```

## Frozen source-aligned residual run

The first source-aligned follow-up starts from the fold-0 trust-region model at
alpha 0.5 and keeps the released backbone frozen. It changes the fitting signal,
not the held-out evaluator: real and simulated plume pixels use the authors'
CH₄-weighted BCE contract (positive weight 10; enhancement clamped to 100–2,000
ppb and divided by 1,000), and simulation rotates real fit-fold enhancement
fields onto clear, onshore fit-fold no-plume targets with source/target wind
speeds within 1.5 m/s and target speed no more than 9 m/s. Training crops are
192 pixels, as in the released recipe. The asymmetric released-teacher terms
remain at lower weights to protect no-plume precision and annotated plume
support.

- Code commit: `fb289435c238faa63d44ab02984a5a9dfa322312`
- Script SHA-256: `a0a10e4c911b153731aefa34cf79f9af3c9557f73770317c2dce744f97c95a82`
- Parent artifact SHA-256: `b94880d858e1e7791591eeb5f7d0da9be84b99a324e980437ebe83cfae6c7d49`
- Fold / seed: 0 / 707
- Epochs / samples per epoch / batch: 8 / 32,768 / 16
- Learning rate / weight decay / patience: 0.00005 / 0.000001 / 3
- Initial residual strength / simulation fraction on positive requests: 0.5 / 0.5
- Loss weights: scene 0.05, no-plume upward 0.10, plume-pixel downward 0.10, correction L2 0.001
- Site/label/sensor cells receive equal sampling mass; rows are sampled within cells with replacement.
- Selection and promotion gates are unchanged. Fold 1 remains unread unless every fold-0 gate passes.

The frozen command is:

```text
python tools/train_mars_source_aligned_residual.py
```

### Fold-0 result and checkpoint-recovery amendment

The preregistered run selected epoch 4 by its frozen balanced rank, but failed
the IoU, recall, and sensor-protection gates. Its ignored artifact has SHA-256
`8da4abe2bdbbbe3f3b8ca9ab189c59c701f57a8186c4d89bdf2337c11e551629`.
Epoch 3 was not selected and was therefore not retained by the original saver,
yet its stored validation record showed the complementary error profile needed
for the next conservative experiment: AP and recall improved while IoU
regressed. No fold-1 or sealed-test data were read.

The next code change may add optional epoch snapshots without changing the
optimizer, schedule, data order, losses, evaluator, selection rule, or original
result. The exact seed-707 command will be replayed through epoch 3. Recovery is
valid only if epochs 1-3 reproduce their stored validation metrics exactly; the
recovered checkpoint is not itself a promoted model. Before any new fold-0
metric is read, an interpolation grid between the frozen alpha-0.5 correction
endpoint and recovered source-aligned epoch 3 must be committed. Endpoint
identity checks must reproduce both previously observed metrics exactly.

The replay was stopped after epoch 1 because its balanced rank differed from
the archived run by approximately 7e-6, violating the exact-recovery rule.
No recovery checkpoint was accepted. The next experiment therefore uses only
the retained alpha-0.5 primary artifact and retained source-aligned epoch-4
artifact. Before evaluation, beta is frozen to 0, 1/64, 1/32, 3/64, 1/16,
3/32, 1/8, 3/16, 1/4, 3/8, 1/2, 5/8, 3/4, 7/8, and 1. Endpoints must exactly
reproduce their archived metrics and are excluded from selection. If any
interior beta passes every promotion gate, selection maximizes balanced rank
among passing values; otherwise it maximizes balanced rank among all interior
values and the experiment is rejected.

Pre-result arithmetic amendment: the first endpoint-blend execution aborted at
the primary identity assertion and wrote no report. Contracting correction
parameters by 0.5 is numerically close to, but not bitwise identical with, the
frozen trust-region definition `baseline + 0.5 * (trained - baseline)` computed
in float32 and cast back to the released-logit dtype. The evaluator now uses
that original arithmetic directly. Artifacts, rows, beta grid, metrics,
selection, and promotion gates are unchanged.

## Frozen scene-ranking head inner search

The next architecture preserves the alpha-0.5 segmentation logits exactly and
adds a separate scene-ranking head. This makes the already positive mask-IoU
delta invariant while allowing the retrieval objective to use scene-level
spectral, temporal, morphology, wind, cloud, and sensor evidence. The compact
feature cache contains folds 2-4 only: 26,578 rows by 108 float features,
9,383,939 bytes, SHA-256
`01d8587e283c1179d61a7c789eb514b3f699d3e7a75bf8c50e4baff3f1698b89`.

Inner selection trains on folds 3-4 and validates on fold 2. It compares five
weighted logistic models (C 0.01, 0.03, 0.1, 0.3, 1.0) and eight deterministic
histogram-gradient-boosting models: 15 or 31 leaves, 20 or 50 minimum samples
per leaf, and L2 1 or 10. Site/label/sensor cells receive equal total fitting
weight. Each head is logit-blended with the frozen connected-component score at
weights 1/8 through 1 in eighth-step increments. A candidate must improve fold-2
AP and recall at no more than 7.13% FPR while keeping each sensor AP within 0.01
of the alpha-0.5 endpoint. Passing candidates outrank all failing candidates;
balanced AP/recall delta breaks ties. Only an inner pass authorizes refitting
the chosen specification on folds 2-4 and one fold-0 extraction/evaluation.
Fold 1 and the paper test remain unread.

Inner selection passed with weighted logistic regression at C=0.1 and scene-head
blend weight 0.25. The refit folds-2/3/4 artifact is 5,018 bytes with SHA-256
`98ce79c62b6af0c97155acdf4255ee4a721f5ef3d5412203bf6a04d2512a336c`.
The authoritative inner report SHA-256 is
`20c82ea36024dbaa5f1fd56673a139a63945b07c1ade4a706acbd3cda2f5fcec`.
The fold-0 evaluator and its exact artifact/report/cache contracts must be
committed before extracting or evaluating fold-0 scene-head features.

The fold-0 decision is frozen as a single evaluation of that artifact. The new
feature cache must contain fold 0 only, reproduce the stored alpha-0.5 overall
AP/recall/FPR and both sensor AP values exactly, match the fitting cache's
artifact/manifest/protocol provenance, and use the identical 108-column schema.
The scene head is compared directly with the released model. Promotion requires
strictly higher AP, recall at no more than 7.13% FPR, and pixel IoU; candidate
FPR must be no worse than the released operating point; and neither sensor may
regress more than 0.01 AP or IoU. Pixel masks are definitionally unchanged from
the frozen alpha-0.5 endpoint, whose exact stored pixel counts and IoU are
inherited rather than recomputed. Only a complete pass authorizes fold 1.

## Frozen hard-example scene-head inner search

The broad scene-head AP gain with a one-TP fold-0 recall loss motivates a
recall-focused inner experiment, not fold-0 threshold tuning. On folds 3-4,
primary positives below the primary 7.13%-FPR threshold and primary negatives
above it are identified as hard examples. Site/label/sensor-balanced weights
are multiplied by 2, 4, 8, or 16 for hard positives and by 1 or 2 for hard
negatives. Weighted logistic C is 0.03, 0.1, or 0.3; frozen-head blend weights
are 1/8, 1/4, 3/8, 1/2, or 5/8. Selection remains on fold 2 only.

Unlike the first inner search, authorization now requires at least three extra
fold-2 true positives at no more than 7.13% FPR, in addition to higher AP,
higher recall, and both sensor AP deltas above -0.01. This robustness margin is
intended to prevent another one-example sign reversal. Only an inner robust
pass permits a folds-2/3/4 refit and a separately preregistered evaluation of
the already extracted fold-0 cache. Fold 1 and the paper test remain unread.

## Frozen site-context scene-head inner search

The next representation adds label-free context from other observations of the
same physical site. Ten primary/released, MBMP, coverage, and residual summary
features each receive group mean, standard deviation, maximum, 90th percentile,
leave-one-out maximum, and within-site rank, plus log site sequence length.
Because the protocol assigns whole sites to folds, these 61 context values are
computed independently inside folds 3-4 and fold 2 without site leakage.

Four logistic models (C 0.01, 0.03, 0.1, 0.3) and four deterministic histogram
gradient boosters (15/31 leaves, 20/50 minimum leaf size, L2 10) are blended at
1/8, 1/4, 3/8, 1/2, or 5/8. Site/label/sensor-balanced fitting weights and the
three-extra-TP fold-2 authorization margin are unchanged. Only an all-gate
inner pass may produce a folds-2/3/4 artifact and a preregistered fold-0 run.

The site-context inner gate passed with 11 additional fold-2 true positives.
The refit artifact is 539,595 bytes with SHA-256
`8334c7b5da880c794dad949dc886b81322579e151933a546b0a63018c93fb284`;
the authoritative inner report SHA-256 is
`c247c1326bbd621d0148d4fffb2045f1fe4f132c134b449f9c9238f4bec23bfa`.
Its fold-0 evaluator must be committed before loading the existing fold-0 cache.

The site-context fold-0 evaluation is frozen to cache SHA-256
`372e152734db1314417ed385b099af54acd182bf758b1d2eabcedfeb64a709e7`,
artifact/report hashes above, the exact HGB and 0.5 blend selected on fold 2,
and the prior primary endpoint. It inherits the unchanged alpha-0.5 masks and
uses the same strict AP/recall/FPR/IoU and sensor non-regression gates as the
first scene-head evaluation. No context statistic uses labels; all are computed
within the evaluation site's own image sequence. This is a one-shot result.

## Frozen three-fold OOF context-head search

Single-fold inner recall did not transfer to fold 0, so the same eight context
model specifications and five blend weights are now selected by complete
out-of-fold predictions across folds 2, 3, and 4. For each held-out fold the
model is trained only on the other two, while label-free context remains within
site. Authorization requires pooled AP and recall gains, at least six extra
pooled true positives, no recall regression on any inner fold, recall gains on
at least two of three folds, no fold AP loss beyond 0.005, and pooled sensor AP
protection. Only a stable pass may be refit on all three folds and receive a
new preregistered fold-0 evaluation. Fold 1 and the paper test remain unread.

The OOF-stability gate passed on all three held-out folds. The refit artifact
SHA-256 is
`2d014f54918f68726d2ca4da19f35a1f29cb1b622fe7c32b56afc554ec27c370`;
the authoritative OOF report SHA-256 is
`a125830c41d1d592a7d3a52ee2343ad5faa883061869e4638113c9df09f421e0`.
A new evaluator must pin these hashes and the unchanged fold-0 cache before it
may read predictions.

The new fold-0 run is frozen to the unchanged cache SHA-256
`372e152734db1314417ed385b099af54acd182bf758b1d2eabcedfeb64a709e7`,
the OOF artifact/report hashes above, HGB specification, 0.625 blend, and prior
primary endpoint. The evaluator requires the complete OOF stability record and
uses the same strict released-model AP/recall/FPR/IoU and sensor gates. It is a
one-shot evaluation; fold 1 remains unread unless every gate passes.

## Frozen minimum-intervention OOF context trust region

The HGB family and all fitting data remain fixed. OOF predictions are recomputed
for blends 1/64, 1/32, 3/64, 1/16, 3/32, 1/8, 5/32, 3/16, 7/32, and 1/4.
The selection rule is the smallest blend satisfying every previously frozen
three-fold stability gate, including at least six pooled extra true positives.
This directly addresses the observed failure of the maximum-ranked 0.625 blend
without inspecting another fold-0 score. Only a full OOF pass authorizes a new
fold-0 evaluator; fold 1 and the paper test remain unread.

The selected minimum blend is 0.25. Its authoritative report SHA-256 is
`8fc190bf4cac9d3abb24979cd20678930f09143302a69ddbfb944a7959951b0b`.
The next evaluator must pin this report, the unchanged HGB artifact, the
unchanged fold-0 cache, and the primary endpoint before loading predictions.

The fold-0 minimum-intervention run is frozen to blend 0.25, selection SHA-256
`8fc190bf4cac9d3abb24979cd20678930f09143302a69ddbfb944a7959951b0b`,
the unchanged HGB artifact/cache/primary hashes, and the same strict promotion
gates. The evaluator rejects any selection other than the exact OOF-stable 0.25
value. It is a one-shot result; fold 1 remains unread unless all gates pass.

Fold 0 passed every strict gate. The authoritative result SHA-256 is
`4eec74a60109e6ed48d10358bde3c077755cfb15fa166dbec20d655ec2c492ce`.
Independent confirmation must not reuse the fold-0 residual because that model
fit fold-1 scenes. A fold-1 residual must train on all non-fold-1 development
sites with the already frozen primary recipe, alpha 0.5, and epoch 7 inherited
from fold 0. Fold-1 labels may be read once after training, not used for epoch
selection or early stopping. The scene-head spec and blend remain unchanged.

## Frozen fold-1 residual confirmation training

The independent residual uses the primary fold-0 recipe without architecture or
hyperparameter selection: fold 1, seed 606, 32,768 samples per epoch, batch 16,
AdamW learning rate 0.0002 and weight decay 0.0001, original 12-epoch cosine
schedule, and exactly seven executed optimizer epochs because epoch 7 was fixed
by fold 0. Loss weights remain scene 0.25, no-plume upward 0.25, plume-pixel
downward 0.10, and correction L2 0.002.

Confirmation mode never iterates the fold-1 validation loader. It saves raw
epoch-7 correction weights with a machine-readable deferred-validation marker;
all seven history rows contain training losses only. The command is:

```text
python tools/train_mars_paper_residual.py --fold 1 --seed 606 --epochs 12 --fixed-confirmation-epoch 7 --samples-per-epoch 32768 --batch-size 16 --workers 8 --learning-rate 0.0002 --patience 4 --artifact EarthRemoteSensingRapidResponse/artifacts/mars_paper_residual_fold1_seed606_epoch7.pt --output-json reports/experiments/mars_paper_residual_fold1_seed606_epoch7_training.json --output-markdown reports/experiments/MARS_PAPER_RESIDUAL_FOLD1_SEED606_EPOCH7_TRAINING.md
```

After training, a confirmation-specific feature extractor/evaluator must pin the
artifact hash, apply alpha 0.5 with the exact float32-logit arithmetic, keep the
scene head and blend 0.25 unchanged, and consume fold-1 labels only once.

Training completed all seven fixed epochs. The compact report contains seven
training-only history rows, zero validation-bearing rows, and
`validation_reads=0`. The ignored 5,135,391-byte artifact has SHA-256
`f6054d0fc8f17d661bce2a17b3947de0e6e566976730aa88f5bc1b6bed347e12`
and embeds fold 1, seed 606, epoch 7, protocol
`6862182bdc1a14ec4a36cc33f318ba3b49a927789f3ff5be7801dc5162051873`,
and the deferred-confirmation marker. Fold-1 outcomes remain sealed.

## Frozen single-read fold-1 confirmation

The confirmation evaluator pins residual
`f6054d0fc8f17d661bce2a17b3947de0e6e566976730aa88f5bc1b6bed347e12`,
scene head `2d014f54918f68726d2ca4da19f35a1f29cb1b622fe7c32b56afc554ec27c370`,
minimum-blend report
`8fc190bf4cac9d3abb24979cd20678930f09143302a69ddbfb944a7959951b0b`,
and released fold report
`4085d6e1e3683dfe4f73d25fd1bc0906a756b7433cc850ed64392db08f1f7935`.
It loads fold-1 images once, writes no reusable feature cache, asserts exact
released AP/recall/FPR/IoU and both sensor AP/IoU identities, applies exact
alpha-0.5 arithmetic, derives the same 108 base and 61 label-free site-context
features, and uses the fixed HGB head at blend 0.25. Promotion requires strict
AP, recall, and IoU gains, no worse FPR, and sensor AP/IoU protection. A failure
forbids paper-test access; a pass authorizes final protocol freeze only.

Pre-result identity amendment: the first confirmation execution completed
inference but aborted at the released-baseline identity assertion before writing
a report or revealing successor metrics. The evaluator now reports labeled
actual/expected values only for failed released-identity fields; it still aborts
before scoring the scene head or emitting candidate results. No model, hash,
row, feature, threshold, metric definition, or promotion gate changed.

The diagnostic rerun showed exact released recall, FPR, overall IoU, and both
sensor IoUs; only overall and sensor AP differed. Cause: the confirmation path
passed raw Python float64 connected scores to AP, whereas the frozen 108-column
feature contract stores the two connected scores as float32 before all scene
metrics. The evaluator now uses those already-produced float32 schema columns
for released identity, primary score, and final blend. This is a pre-result
dtype amendment; no successor metric was revealed and no scientific choice
changed.

## Fold-1 confirmation result: retrieval passes, segmentation fails

The single authorized fold-1 read failed the joint promotion gate, so the paper
test remains sealed. The released model and frozen successor respectively
scored AP 0.85974835 and 0.88038834 (delta +0.02063999), recall 0.93691275
and 0.95033557 at the same 0.07127778 FPR (delta +0.01342282, ten additional
true positives), and pixel IoU 0.46937442 and 0.46026097 (delta -0.00911346).
The AP gains were positive for both Landsat (+0.00320553) and Sentinel-2
(+0.02432732), while IoU regressed for both (-0.00239279 and -0.00933680).

This is evidence for a split architecture decision, not a near pass. The
label-free site-context scene head generalized independently and should remain
the retrieval candidate. Alpha-0.5 residual masks did not generalize and must
not remain coupled to that scene-score path. Subsequent work will construct
fully cross-fitted mask predictions on development folds, select a separate
mask trust rule using only out-of-fold evidence, and require per-fold and
per-sensor IoU stability before freezing a final model. Fold 1 is now
development evidence; it cannot be described as an independent confirmation
again. The authoritative result JSON SHA-256 is
`eff232d7aaf08b87ea54c6d7e0929c9ed6f3e5bc6a7c9e10d0dcb4e51e99ab5d0`.

Fold 1 is now development data, so a dense post-hoc alpha sweep tested whether
the segmentation failure was merely excessive correction strength. It was not.
The smallest nonzero alpha, 1/256, already reduced overall IoU by 0.00004164;
alpha 1/128 reduced it by 0.00002378, alpha 0.5 by 0.00911346, and alpha 1.0
by 0.05097897. Tiny corrections occasionally helped Landsat but consistently
hurt Sentinel-2, which dominates the cohort. The learned residual direction is
therefore rejected for dense masks at every nonzero trust strength. Its scene
features remain eligible because the separately trained label-free context head
already generalized. The authoritative sweep JSON SHA-256 is
`6d06fde3790cc5e0cc1f0039f256ec5dd2acaeca86ae990bee5679b701f6efeb`.

The next mask path is a lower-variance calibration of the released logits:
select a single connected-mask probability threshold on folds 2--4, ranking
rules first by their worst fold IoU delta, then worst sensor delta, then pooled
delta. Only a threshold that improves every selection fold and both sensors may
advance to folds 0--1. This changes no scene score and cannot leak labels into
inference.

The seven-point folds-2--4 search selected threshold 0.70. Pooled IoU rose from
0.52726015 to 0.56413023 (delta +0.03687008). Fold deltas were +0.02664675,
+0.03909668, and +0.04452736; pooled Sentinel-2 and Landsat deltas were
+0.04106910 and +0.01228303. Thresholds 0.75 and 0.80 had marginally higher
pooled IoU but smaller worst-fold gains, so the preregistered robustness rank
correctly preferred 0.70. The authoritative selection JSON SHA-256 is
`9d4dd8140908489e48be849c0e89100a8f663c3368db9354c1b30ed8670865be`.
The confirmation run is restricted to thresholds 0.50 and 0.70 on folds 0 and
1, with the same 100-pixel connected-component rule. It must improve both folds
and both pooled sensor strata; no threshold refinement is permitted from the
confirmation result.

Threshold 0.70 passed that confirmation. Pooled folds-0--1 IoU rose from
0.48871519 to 0.53703413 (delta +0.04831894). Fold-0 and fold-1 deltas were
+0.04916728 and +0.04737035; pooled Sentinel-2 and Landsat deltas were
+0.05415067 and +0.00909880. The threshold-0.50 fold identities were exactly
0.5086263440 and 0.4693744219, matching the pinned released evaluations.
The authoritative confirmation JSON SHA-256 is
`b3d77820ce1fc2a81c95e479f4412f545310ae8146cbd25f97f5c398b33fc7fe`.

The frozen integrated candidate therefore separates responsibilities: the
alpha-0.5 temporal residual and 0.25-blend OOF-stable context head produce only
the scene ranking, while the released U-Net logits at threshold 0.70 and the
unchanged 100-pixel connected rule produce the dense mask. This candidate has
strict AP/recall gains at matched FPR on both development confirmation folds
and strict mask-IoU gains on all five development folds and both sensors. The
paper test remains sealed pending an integrated evaluator, artifact/hash freeze,
fixed operational scene threshold, and paired site-bootstrap implementation.

## Frozen official paper-test evaluator

The integrated architecture is now preregistered in
`configs/mars_successor_paper_test_v1.json` (SHA-256
`b6ada879e727be8de73f287c707bb05d5fb699473cee339faa18f402b6d60926`).
The fixed operational scene threshold is 0.5240683426, the conservative maximum
of the two independent confirmation-fold target-FPR thresholds; AP and recall
are additionally evaluated at the comparator's matched FPR without changing
the ranking. The mask threshold is 0.70 with a 100-pixel connected minimum.

The one-shot evaluator SHA-256 is
`740b27cdbf089c4307f713bfd0946bb9b6e3c4a855de5be8360d58a061d12735`.
It pins every model/report hash and the 43,524-row sealed manifest hash, then
reconstructs the paper's hybrid general/offshore 43,529-row comparator from the
authors' archived per-scene files. The five historical scenes without current
rasters receive adversarial candidate scene and 200x200 pixel outcomes. The two
available positive scenes without pixel masks receive zero candidate
intersection, archived truth pixels as false negatives, and all predicted
pixels as false positives.

For both the full 43,529-image view and the 15,655-image test-only-site view,
the evaluator runs 10,000 paired physical-site bootstrap replicates. Promotion
requires lower 95% bounds above zero for AP, pixel IoU, and matched-FPR recall,
plus an upper 95% bound at or below zero for the frozen operational FPR delta,
and point estimates above both the exact reconstructed comparator and all
published paper metrics that exist for the view. No test manifest row has been
loaded by the evaluator; test assets may now be acquired byte-for-byte, after
which this exact code is the only authorized result-producing path.

The architecture/evaluator freeze commit is `9c4ca8a3`. Any subsequent code,
configuration, model, threshold, or gate change invalidates the one-shot
authorization and must be documented as post-test work rather than silently
rerun. Byte acquisition and a verification receipt do not change the model.

The sealed test acquisition is complete. All 90,666 selected assets totaling
43,483,074,992 bytes passed their pinned Git-blob or LFS hash and exact-size
checks; no partial or missing bytes remain. The full verification receipt
SHA-256 is
`eefc7b9cdbb7b950bf109bd7fa79c8d88f94c716eeca7d97a247395e95fc5895`.
The compact resumable-transfer receipt SHA-256 is
`8ecaa5094048db32cf025b829d0eadb0d94f6b1978b2415588b0fe7a0134b20d`.

## Frozen official paper-test result: retrieval improves, joint superiority fails

The one-shot evaluator was executed exactly once from freeze commit
`9c4ca8a3`, after complete byte verification, without changing any model,
threshold, configuration, hash, or promotion gate. The authoritative JSON has
SHA-256
`589210e313fd1c6e93daf83e22db2582223ad065162e98cb93be5627f3934119`.
This evaluation ends the independently sealed status of the public 2024 paper
test. Every later analysis using its labels is post-test development and must
not be described as a fresh one-shot confirmation.

On the full 43,529-image view, average precision increased from 0.64101960 to
0.64693347 (delta +0.00591387), and recall at the exact comparator FPR of
0.07069230 increased from 0.79150579 to 0.81687810 (delta +0.02537231).
The paired physical-site bootstrap lower 95% bound for the recall delta was
+0.01603202, but the AP-delta interval was [-0.00898100, +0.01991029]. The
fixed operational threshold reduced FPR to 0.03317672, but also reduced recall
to 0.74296746. Pixel IoU fell from 0.32436515 to 0.24943580 (delta
-0.07492935; paired 95% interval [-0.09977347, -0.04798498]).

On the 15,655-image test-only-site view, average precision increased from
0.45027380 to 0.46710370 (delta +0.01682990), and matched-FPR recall increased
from 0.77533040 to 0.80176211 (delta +0.02643172) at FPR 0.07551206. The
recall-delta lower 95% bound was +0.00621022, while the AP-delta interval was
[-0.00984156, +0.04321676]. Fixed-threshold FPR fell to 0.03415867. Pixel IoU
was nearly tied but lower, 0.17156194 versus 0.16869549 (delta -0.00286645;
paired 95% interval [-0.01793803, +0.01039576]).

The point AP and matched-FPR recall estimates exceed the exact reconstructed
paper model in both official views, with statistically supported recall gains
and much lower operational false-positive rates. They do not establish AP
superiority because both AP confidence intervals cross zero. More decisively,
the candidate fails the segmentation requirement, especially on the portion
outside the test-only-site view. There, relative to the hybrid paper
comparator, it loses approximately 181,414 true-positive pixels and adds
962,692 false-positive pixels. This is consistent with a failure to match the
paper's specialized offshore/training-site mask branch, not with a globally
transferable benefit from the development-selected 0.70 probability threshold.

The frozen candidate therefore fails the preregistered joint superiority gate.
It must never be reported as unambiguously outperforming MARS-S2L. The scene
retrieval path remains promising, while segmentation requires a separately
cross-fitted offshore/domain-specialized architecture. A future confirmatory
paper claim also requires a genuinely independent external or future holdout;
the now-open public paper test may be used only for transparently labeled
diagnosis and post-test comparison.

## Post-test diagnosis: the paper's offshore specialization is necessary, not sufficient

### Pre-rerun metric correction amendment

Before any architecture or threshold experiment, audit of the failed result
found that the frozen evaluator's candidate pixel counts were not mutually
exclusive. For rows with available truth it computed false positives as every
predicted observable pixel, including the intersection pixels already counted
as true positives. The correct definition is predicted, observable, and not
truth. This is a deterministic evaluator defect: the model, logits, masks,
thresholds, scene scores, cohort, and bootstrap design are unchanged.

From the already stored aggregate counts alone, subtracting candidate TP from
the erroneous FP changes the full candidate IoU from 0.24943580 to 0.33233106
(delta versus comparator +0.00796591), and the test-only candidate IoU from
0.16869549 to 0.20292864 (delta +0.03136670). Site-block confidence intervals
cannot be repaired from aggregates, so the unchanged frozen model will be run
again after committing the corrected count definition and regression tests.
The original result JSON SHA-256
`589210e313fd1c6e93daf83e22db2582223ad065162e98cb93be5627f3934119`
remains the immutable superseded-result identity. The correction is post-result
and will be reported as such; it is not a new independently sealed one-shot.

The committed correction was then executed on the unchanged frozen model and
cohort. The corrected JSON SHA-256 is
`7e3bc6f45af2b116d654f6e3bbec82a78c3d54b1f1d9d1219e6d6201ac094a01`.
All scene scores and scene metrics are identical to the superseded run. Full
pixel IoU is 0.33233106 versus 0.32436515 (delta +0.00796591), with paired
site-bootstrap 95% interval [-0.00646972, +0.02184039]. Test-only pixel IoU is
0.20292864 versus 0.17156194 (delta +0.03136670), with interval
[+0.01877352, +0.04403263].

The corrected candidate therefore beats the exact and published comparator on
every point estimate in both official views. It passes both recall-confidence
gates, both no-worse-FPR gates, and the test-only IoU-confidence gate. It still
fails the predeclared unambiguous-superiority standard because the AP-delta
intervals cross zero in both views (full [-0.00898100, +0.01991029]; test-only
[-0.00984156, +0.04321676]) and the full IoU-delta interval crosses zero. The
remaining research problem is consequently narrower but not complete: produce
larger, site-consistent ranking gains and stabilize full-view segmentation,
without sacrificing the already confirmed recall, operational FPR, or
test-only IoU improvements.

The two author-released prediction archives permit an exact controlled
comparison of the general checkpoint and the unpublished offshore fine-tuned
checkpoint on identical paper-v3 scenes. This analysis is explicitly post-test
and does not rerun or modify the frozen ERSRR result. The authors' hybrid routing
replaces the general output on 2,185 offshore scenes, including 583 test-only-site
scenes.

On all 2,185 offshore rows, the extra real-data epoch raises AP from 0.39099 to
0.52876 (delta +0.13777) and pixel IoU from 0.08929 to 0.22685 (delta
+0.13756). Offshore Sentinel-2 and Landsat IoU improve by +0.08483 and
+0.15059 respectively. Across 55 offshore sites, site-level IoU improves at
eight, regresses at one, and ties at 46; the many ties mainly reflect sites
without positive pixel support.

Substituting the fine-tuned model only offshore raises full-benchmark IoU from
0.30255 to the exact comparator's 0.32437 (delta +0.02182), raises full AP by
+0.00192, and removes 274 scene false positives. On the test-only-site view,
all 583 offshore rows are no-plume scenes: substitution removes 89 scene false
positives and 150,485 false-positive pixels, raising global pixel IoU by
+0.01874, while AP changes by -0.00188 because the score redistribution among
negatives slightly alters their ordering around positives from other regions.

This establishes an architecture constraint. Reproducing an offshore-specific
branch is necessary, but its +0.02182 full-IoU contribution is far smaller than
the frozen candidate's -0.07493 official full-IoU deficit. The development-wide
0.70 threshold is therefore rejected as a universal mask policy. Future work
must independently validate a domain-routed segmentation path, retain the
general or specialized 0.50 mask where appropriate, and improve segmentation
beyond merely cloning the paper's extra epoch. The compact diagnostic JSON and
Markdown are generated by `tools/analyze_mars_offshore_finetune.py`.

## Predeclared next experiments

1. Reproduce the released model on complete held-out folds 0 and 1 under the exact connected-component evaluator.
2. Train correction-only fold 0 with label-sensor balanced sampling, hard-negative segmentation loss, scene MIL loss, and asymmetric no-plume non-regression.
3. Advance only if fold 0 improves AP, recall at no more than 7.13% FPR, and fixed-rule pixel IoU; independently confirm on fold 1.
4. Ablate sensor identity, temporal normalized/log-ratio features, no-plume penalty, scene loss, and correction capacity.
5. If confirmed, run all five fold models, generate out-of-fold predictions, cross-fit calibration, and freeze an ensemble.
6. Open the paper test once. Require paired site-block bootstrap lower 95% bounds above zero for AP and IoU deltas, higher recall, and no worse FPR on both official views.

## Current claim boundary

The paper comparator is reproduced and the research protocol is publication-grade, but ERSRR has not yet demonstrated paper-test superiority. No wording should imply otherwise until every frozen superiority gate passes.

## Post-test development: sensor-specific dense-mask calibration

The corrected one-shot result narrowed the remaining segmentation problem to the
full-view confidence interval. A label-free scene-confidence routing experiment
tested whether high-confidence scenes should retain a broader 0.50 mask while
other scenes used the development-selected 0.70 mask. This hypothesis was
rejected on folds 2/3/4: every tested routing cutoff reduced IoU. The least
harmful cutoff, 0.975, still changed IoU by -0.00296 versus global 0.70, with a
paired physical-group 95% lower bound of -0.00853. The negative result and code
are retained so this path is not repeated.

The global threshold surface instead showed a repeatable sensor interaction.
On selection folds 2/3/4, moving from 0.70 to 0.80 improved Sentinel-2 IoU by
+0.00654 but reduced Landsat IoU by -0.01106. A separate extended pass on folds
0/1 confirmed the direction: Sentinel-2 improved by +0.01341 and Landsat
regressed by -0.01070. This motivated a frozen, physically interpretable rule:
Sentinel-2 uses threshold 0.80 and Landsat retains 0.70; the connected-component
minimum remains 100 pixels.

The rule was then audited once across all 44,363 development scenes with 10,000
paired physical-group bootstrap replicates. Relative to universal 0.70, IoU
improved by +0.00815 on selection folds 2/3/4 (95% interval lower +0.00070),
+0.01431 on confirmation folds 0/1 (lower +0.00691), and +0.01066 across all
five folds (lower +0.00479). Every individual fold improved; Sentinel-2 gained
+0.00941 over all folds and Landsat was exactly unchanged by construction.

This rule was developed after the official paper test was opened. It may be
reported only as transparent post-test development, never as a replacement for
the original one-shot result. Configuration
`configs/mars_successor_paper_test_v2_posttest_sensor_mask.json` preserves all
scene scores and thresholds from v1 and changes only dense-mask calibration.

The versioned post-test run was committed before evaluation at `905606b9` and
produced JSON SHA-256
`1986ad46fa3de4630d7676769df81e982d5a9b1e7a7cec83589ca5f9b314b9c0`.
Full-view candidate IoU rose from 0.33233 to 0.33747, a +0.00514 gain over v1
and +0.01311 over the exact hybrid paper comparator. Its paired site-bootstrap
95% interval versus the comparator narrowed to [-0.00326, +0.02968], so the
full-view segmentation confidence gate still fails. Test-only IoU rose from
0.20293 to 0.21809, +0.04653 over the comparator, with interval
[+0.03049, +0.06281].

As required, all scene metrics are unchanged: full/test-only AP remain
0.64693/0.46710 with deltas +0.00591/+0.01683 and lower confidence bounds
-0.00898/-0.00984. Matched-FPR recall and operational FPR gates remain passed.
The sensor rule is a confirmed improvement, but the overall superiority claim
still fails both AP-confidence gates and the full-view IoU-confidence gate.
Further threshold-only work is unlikely to solve the dominant scene-ranking
uncertainty; the next research stage prioritizes stronger cross-fitted scene
ensembles and domain-specialized segmentation.

## Post-test development: stronger cross-fitted scene head

A development-only search compared HistGradientBoosting and ExtraTrees heads
over the frozen 169-dimensional scene-plus-site-context feature schema. Every
fold-2/3/4 prediction was generated by a head trained on the other two folds.
The search explicitly compared the original site/label/sensor cell weighting,
group-balanced weighting, and uniform scene weighting. Folds 0/1 and the paper
test were not loaded.

The selected head is a 400-tree uniform-weight ExtraTrees classifier with
minimum leaf size 5, 50% feature subsampling, and a 0.625 logit-space blend with
the frozen primary scene score. Pooled OOF AP rose from 0.87373 to 0.90331
(delta +0.02957), and recall at no more than 7.13% FPR rose by +0.01206.
Fold AP deltas were +0.03600, +0.03314, and +0.03155 on folds 2, 3, and 4;
both sensors improved (+0.03611 Sentinel-2, +0.00830 Landsat). The paired
physical-group 10,000-replicate AP interval was [+0.01905, +0.04190].

This substantially exceeds the current HGB head's development AP delta
(+0.01437). Artifact SHA-256 is
`9e6fa18b83ef065ac24c94a06a510057a0c382cecf1efa3b54e818566a45c9ac`.
The artifact and selection report are frozen before one-shot fold-0 evaluation;
fold 1 remains untouched at this stage.

The frozen-head fold-0 result passed every gate. AP increased from 0.88645 to
0.91429 (delta +0.02784), recall at no more than 7.13% FPR increased by
+0.00403, and the paired 107-site bootstrap AP interval was
[+0.00966, +0.04546]. Both sensor non-regression and the unchanged primary
segmentation gates passed. Result JSON SHA-256 is
`89440711e43fd89ccc7eb67814b7f10659f93b87c732402a458896647006f43b`.

Fold 1 has been used by earlier architecture confirmation in this research
history, so it is not globally untouched. It remains label-independent for
this newly frozen head, however: the head and blend were selected without
fold-1 data. The next evaluation is therefore described as a fixed-head
held-fold confirmation, not as a new pristine one-shot claim. Its image
features will be generated with the independently trained fold-1 residual,
which recorded zero fold-1 validation reads during training.

The first fold-1 execution of the generic evaluator returned a false overall
status despite AP +0.03686, recall +0.00940, and a paired AP interval of
[+0.01875, +0.05828]. Audit showed the only failing field was the inherited
`pixel_iou_higher` check on the fold-1 residual endpoint. That check belongs to
an older coupled architecture and is irrelevant to this scene-head experiment:
the official successor's mask branch uses released logits with separately
confirmed sensor thresholds. The evaluator contract was therefore corrected
before committing a fold-1 result: AP, recall, FPR, sensor non-regression, and
paired AP confidence are promotion gates; residual pixel IoU remains a reported
non-promotion diagnostic. No scores, model parameters, or blend weights changed.

Under the corrected, architecture-aligned contract, both fixed-head held folds
pass. Fold 0 remains AP 0.91429 (delta +0.02784), recall delta +0.00403,
bootstrap interval [+0.00966, +0.04546]. Fold 1 reaches AP 0.89661 versus
0.85975 (delta +0.03686), recall delta +0.00940, and interval
[+0.01875, +0.05828]. The fold-0/fold-1 operating thresholds are 0.15312 and
0.28188; the conservative fixed operational threshold for a post-test v3
architecture is therefore 0.28187603894788654. Result hashes are
`4827c6a380010bb6a234d85298427e35ce2583051869300cbcabda3f4cb3bd06`
(fold 0) and
`8380d079039ca6c9513ca125934665351c49988429336c63287e28e4cd7d0d1c`
(fold 1).

The v3 architecture was committed at `d85d1c48` before its exact-paper run.
On the full 43,529-scene view, AP reached 0.67380 versus the exact comparator's
0.64102 (delta +0.03278), with paired site-bootstrap interval
[+0.01424, +0.04684]. This is the first statistically positive full-view AP
result. Matched-FPR recall also increased by +0.03199 with lower bound
+0.01892, while fixed-threshold FPR was 0.03303.

On the 15,655-scene test-only-site view, AP was 0.46550 versus 0.45027 (delta
+0.01523), but its interval [-0.01874, +0.04639] still crossed zero.
Matched-FPR recall increased by +0.02203, but its lower bound was -0.00901, so
that confidence gate also regressed from v2. The v2 dense masks reproduced
exactly: full IoU 0.33747 retained interval [-0.00326, +0.02968], while
test-only IoU 0.21809 retained [+0.03049, +0.06281]. V3 therefore clears the
full AP gate but still fails full IoU confidence plus test-only AP and recall
confidence. Result JSON SHA-256 is
`6086f09d9948fde846b83a01c4ce6bdba2080d43a2f4cb409fa2e1de470a065a`.

## Post-test domain-routing hypothesis and rejection

The ignored v3 diagnostic cache was used to ask whether sensor-specific
blending of the legacy HGB head and stronger ExtraTrees head, plus an offshore
logit correction, could explain the remaining paper-test scene errors. The
post-test rule (0.7 new-head weight for Sentinel-2, 1.0 for Landsat, and a
-4.0 offshore logit shift) would clear both paper views' AP and recall
confidence gates. Because that rule was discovered after opening paper-test
labels, it was not promoted directly.

Instead, the unchanged rule was reverse-validated on development data. Scores
on folds 2/3/4 were cross-fitted, while the previously frozen heads scored
held folds 0 and 1. The rule passed fold 0, but failed the pooled folds-2/3/4
contract: AP improved by only +0.00895 with paired physical-group interval
[-0.01942, +0.02685], and recall declined by -0.00215. The inner development
partition contains 102 positive scenes from offshore groups, demonstrating
that the hard offshore penalty is not a generally defensible domain rule.
Fold 1 improved AP by +0.02907, but was 0.00728 below the stronger head and
therefore also missed the conservative non-regression check.

The rule is rejected and will not be represented as confirmation evidence.
Its full negative report is retained, while the 1.09 MB row-score cache is
ignored by Git (SHA-256
`fd955b78b26a3b2a5165b4abab02180ccf4dad433511bf4da7afbff44275c1c7`)
for a development-only, leakage-safe meta-model search.

## Cross-fitted scene stacking: negative experiment

A 48-candidate meta-model search combined the frozen primary score, legacy HGB
head, and stronger ExtraTrees head. Candidate families were regularized
logistic regression and HistGradientBoosting; weighting modes were uniform,
physical-group balanced, and site/label/sensor-cell balanced. Sensor and
offshore interaction features were available. Every folds-2/3/4 selection
score remained cross-fitted, and the paper cache was not loaded during model
selection or held-fold confirmation.

The selected group-weighted logistic stacker improved the primary model on the
inner partition by +0.02121 AP and +0.01206 recall; its paired AP lower bound
versus primary was +0.01211. It also passed both held folds, improving primary
AP by +0.02550 (fold 0) and +0.03802 (fold 1), with small AP gains over the
stronger head of +0.00012 and +0.00167. However, on the actual cross-fitted
selection partition it was 0.00836 AP below the stronger head, with paired
interval lower bound -0.01252. It therefore failed the predeclared inner
non-regression gate and is rejected.

For diagnosis only, the rejected stacker was then scored from the ignored
exact-paper cache. It preserved the full-view superiority result (AP delta
+0.03637, lower bound +0.01780; matched-recall lower bound +0.02429), but did
not solve the test-only view: AP lower bound -0.01744 and matched-recall lower
bound -0.00538. This confirms that calibration/stacking of the existing heads
is not enough to close the remaining site-novel ranking uncertainty.

## Offshore mask threshold: reverse-validation rejection

The post-test diagnostic suggested using a 0.90 component-mask threshold on
offshore scenes, over the confirmed Sentinel-2 0.80 / Landsat 0.70 rule. A
single released-model inference pass computed all three thresholds on 44,363
development scenes, followed by 10,000-replicate physical-group bootstraps.

The rule passed the folds-0/1 confirmation partition, which contains 48
offshore negatives and no offshore positive scenes. It failed where genuine
offshore plumes are present. On folds 2/3/4, 102 offshore positives occur among
830 offshore scenes; offshore IoU fell from 0.59830 to 0.55159. Pooled
selection IoU delta was -0.00232 with interval [-0.00728, +0.00039], and the
all-fold delta was -0.00136 with lower bound -0.00404. The rule is rejected.
This is convergent evidence with the scene-routing failure: hard offshore
suppression exploits the paper test's domain mix but is unsafe for real
offshore plume detection.

## Frozen multi-scale encoder probe: negative experiment

Recent spatial-spectral and scale-aware remote-sensing representation work,
together with methane-specific attention models, motivated testing richer
internal representations before attempting another end-to-end detector. The
released MARS U-Net was frozen and used to extract masked mean, standard
deviation, and maximum activations from encoder levels 3, 4, and 5. This
produced 3,840 deterministic features for every one of 44,363 development
scenes. The 316 MB compressed feature cache is ignored by Git; SHA-256 is
`397501c6230436cb677047abb8b2895f3bafb9916150087370c9586a47559e1f`.

Seven low-capacity probes were cross-fitted on folds 2/3/4: linear probes over
all three scales and 64-unit MLPs over the deepest scale, with the established
108 scalar scene features appended. Uniform, physical-group, and
site/label/sensor-cell weighting were compared, plus one all-scale MLP. Every
standalone probe was worse than the existing primary score. The least-bad
site-cell-weighted level-5 MLP had inner AP delta -0.03136 versus primary and
-0.06093 versus the stronger ExtraTrees head. Held-fold degradation was larger:
-0.07970 AP on fold 0 and -0.08566 on fold 1 versus primary.

The probe is rejected before any paper-test feature extraction. Global channel
moments discard the spatial plume morphology needed to distinguish compact
methane structure from high-response background artifacts. If deep spatial
features are revisited, the next justified design is an explicitly spatial,
physics-guided morphology classifier over released logits and temporal SWIR
contrast maps, not a larger classifier over pooled moments.

## Physics-guided spatial morphology classifier

The promoted follow-up preserves spatial structure at 64×64 resolution. A
frozen released U-Net supplies mean- and max-pooled methane probability maps;
the input also contains centered MBMP, target-minus-reference B11/B12,
normalized B11/B12 temporal differences, cloud fraction, and observable
fraction. The nine-channel float16 development cache is 3.27 GB and remains
ignored by Git. Image SHA-256 is
`6530fa2d07d94bd57ba1ac757039dedd18745227e7a476fb0d69f78f996134a5`;
metadata SHA-256 is
`0160ed93371396a487b819dd170a682f55756d285f0ba94d0f0d093ba51a8d01`.

The classifier is a small residual CNN with learned spatial attention pooling,
global average/max pooling, and an explicit sensor embedding. Four models were
cross-fitted on folds 2/3/4: full physics inputs with uniform, group, or
site/label/sensor-cell weighting, plus a probability-map-only group-weighted
ablation. Each raw morphology score was tested only as a predeclared
logit-space complement to the already frozen stronger scene head.

The selected model uses full physics inputs, site-cell weighting, eight epochs,
and a 0.10 spatial blend. Inner AP improves by +0.03018 versus primary and
+0.00060 versus the stronger head; recall improves by +0.01336/+0.00129.
The paired physical-group AP interval versus primary is
[+0.01997, +0.04190], while the interval versus the stronger head is
[-0.00101, +0.00163]. Every pooled/per-fold point stability gate passes. The
probability-only ablation remained 0.00041 AP below the stronger head, providing
evidence that temporal SWIR/MBMP physics contributes beyond morphology alone.

The architecture then passed both fixed held folds. Fold-0 AP delta is
+0.02564 versus primary and +0.00025 versus the stronger head, recall delta
+0.00537, and paired AP lower bound +0.01049. Fold-1 AP delta is +0.03735
versus primary and +0.00101 versus the stronger head, recall delta +0.00671,
and lower bound +0.01986. Sentinel-2 and Landsat AP improve on both folds. The
conservative operational threshold is 0.30274470404433573. Artifact SHA-256 is
`36135b7c8f9538f3ce7b896df0c2b767ee85b81d57e2de3eede2cf33384730c3`;
selection report SHA-256 is
`dfeba0d4e8dde28ae880077c0db2010ccda5c57f58d293f4d3f834b541605c22`.
The candidate is frozen before any paper-test spatial features are extracted.

The label-independent sealed-cohort spatial cache contains 43,524 available
scenes (the exact five missing scenes retain v3 scores). Image SHA-256 is
`7cce444a552c6c873c05ae3a972f62a8081fde89d31ac63d94303dbfac3b1b94`;
metadata SHA-256 is
`b2b58eabf4478b46912d3c3437c1ee0b039841ee283bd7fe0164775b9d448022`.
The frozen exact evaluator was committed before scoring.

On the exact paper rows, the spatial complement preserves statistically
positive full-view performance: AP 0.67210, delta +0.03108, interval lower
+0.01254; matched-FPR recall delta +0.03034, lower +0.01893. Test-only AP rises
from v3's 0.46550 to 0.46774, but its delta +0.01747 retains lower bound
-0.01431. Test-only matched-recall delta is +0.01762 with lower bound -0.01117.
The unchanged v2 masks retain the full-view IoU lower bound near -0.00331 and
the positive test-only IoU result. Thus the ordinary-BCE morphology model is a
real but insufficient complement: it does not clear site-novel scene
confidence or full-view segmentation confidence. Result JSON SHA-256 is
`77d6f3592571341c6b18b8f0a266c7696f5f6d0dbd5faedb7ce41a7b3430edf3`.

## Hard-negative pairwise spatial ranking

Ordinary scene BCE was redirected toward the actual retrieval boundary using
balanced positive/negative batches, stronger-head-weighted hard-negative
sampling, and a paired softplus ranking loss. Five fixed variants compared
site-cell versus group loss weighting, 75% versus 100% hard-negative sampling,
and pairwise weights 0.25, 0.50, and 1.00. Each used the same frozen spatial
cache, CNN, augmentations, and 0.10-to-1.00 blend grid as the BCE experiment.

The strongest pooled configuration was group-weighted (AP +0.00132 versus the
stronger head), but selection prioritizes worst-fold stability. The frozen
choice therefore uses site-cell weighting, 100% hard-negative sampling, and
pairwise weight 0.50. It improves inner AP by +0.02984 versus primary and
+0.00027 versus the stronger head; recall improves by +0.01293 versus primary.
The paired AP lower bounds are +0.01977 versus primary and -0.00137 versus the
stronger head, satisfying the predeclared no-material-regression interval.

Both held folds pass unchanged gates. Fold-0 AP improves +0.02574 versus
primary and +0.00036 versus the stronger head, with recall +0.00537 and paired
AP lower bound +0.01051. Fold-1 AP improves +0.03748/+0.00114, recall +0.00805,
and lower bound +0.02079. The frozen operational threshold is
0.2965460234436805. Artifact SHA-256 is
`3d362b8dc6b9244abeb949d5f2dc34c2f3101e2ef3849d62ab5ef3c85d4fcb85`;
selection report SHA-256 is
`55726830efa06f98339ecc5df662a124f7bc8a504c34d974589734b54842435e`.

The exact paper evaluation rejects this objective as the next successor. On
the full view it reaches AP 0.67286 (delta +0.03184, paired site-bootstrap
lower bound +0.01319) and matched-FPR recall delta +0.03751 (lower bound
+0.02336), but the unchanged sensor-conditioned masks retain IoU delta
+0.01311 with lower bound -0.00331. On test-only sites it reaches AP 0.46598
(delta +0.01570, lower bound -0.01701) and recall delta +0.02203 (lower bound
-0.00830); both are slightly worse than the ordinary-BCE spatial classifier.
Test-only IoU remains a confirmed improvement. The pairwise loss therefore
improves development ranking without improving site-novel transfer, and must
not replace the ordinary spatial model. Result JSON SHA-256 is
`f9cbe235c7f05fa830001df264e8d1942c61ecc57aea2ace661855139aa9b577`.

## Scene-conditioned dense-mask suppression

The released dense mask and the stronger scene head were next coupled without
retraining either model. Sentinel-2 retains its confirmed 0.80 mask threshold,
Landsat retains 0.70, and the mask is set to empty when the cross-fitted v3
scene probability is below a cutoff. This addresses the paper contract's
pixel false positives on all no-plume scenes while leaving scene AP, recall,
and FPR unchanged.

A fixed 0.025-spaced cutoff grid was selected only on folds 2/3/4. The chosen
0.75 cutoff raises IoU from 0.57228 to 0.59512, delta +0.02284 with paired
physical-group interval [+0.01248, +0.03937]. It retains 97.32% of baseline
true-positive pixels while removing 30.12% of false-positive pixels; every
selection fold and both sensors improve. On untouched confirmation folds 0/1,
IoU rises from 0.55135 to 0.57258, delta +0.02123 with interval
[+0.01276, +0.03427]. Fold-0 and fold-1 deltas are +0.01461 and +0.02768;
Landsat and Sentinel-2 deltas are +0.00623 and +0.02359. Across all five
folds, the gain is +0.02221 with lower bound +0.01490, retaining 97.26% of
true-positive pixels and removing 28.37% of false-positive pixels.

The rule therefore passes selection, held-fold, sensor-domain, bootstrap, and
true-positive-retention gates before paper-cache replay. Analysis script
SHA-256 is
`da451898021b66580a907cb74e481cff0cfbe46773149f36cc35ad7cb357ff3d`;
selection report SHA-256 is
`c1e5a1497abebba80d42898a8165b30fd255ff252478a0ee1fd90fd32456a51c`.

The frozen paper-cache replay makes segmentation superiority unambiguous. On
the full 43,529-scene view, gated-mask IoU is 0.37996 versus 0.32437, delta
+0.05560 with paired site-bootstrap interval [+0.03492, +0.07788]. It retains
94.85% of ungated true-positive pixels and removes 43.64% of false-positive
pixels. Combined with the ordinary spatial scene classifier, the full view now
passes every AP, matched-FPR recall, FPR, and IoU gate.

On the 15,655-scene test-only-site view, IoU is 0.29246 versus 0.17156, delta
+0.12090 with interval [+0.08591, +0.15621]. The gate retains 93.94% of
true-positive pixels and removes 52.80% of false-positive pixels. Test-only
segmentation therefore passes decisively, but overall promotion remains
blocked by scene AP delta +0.01747 (lower -0.01431) and matched-FPR recall
delta +0.01762 (lower -0.01117). Future work should target only site-novel
scene ranking; further mask tuning is not justified. Exact replay script
SHA-256 is
`3d2561b6835abaf99141710dda166f1ebff85ec214f52fc1374d14dae5758030`;
result JSON SHA-256 is
`bc05836a18b1f1cd795a7bd3223849290d0b20a3d3f0fad7b1e30a53731fd5d0`.

## All-development ExtraTrees refit: negative post-test diagnostic

After the v2 ExtraTrees scene-head specification and 0.625 logit blend passed
the original development selection and confirmation checks, that fixed
specification was refit once on all 44,363 authorized development scenes
(3,811 positives across 618 physical groups). No architecture or
hyperparameter changed. The resulting in-sample AP of 0.97492 is recorded only
as a fit audit and is not evidence of generalization. Artifact SHA-256 is
`7dd81a2f1d9b30b88500eeceb086664c4a3fb1cad21810a10783b2ce72c4ab1a`.

A transparent paper-cache replay did not justify promotion. Full-view AP was
0.67295 versus the released comparator's 0.64102, delta +0.03193 with paired
site-bootstrap lower bound +0.01425; matched-FPR recall lower bound was
+0.01834. Test-only-site AP was only 0.46006 versus 0.45027, delta +0.00979
with lower bound -0.02298; matched-FPR recall lower bound was -0.01550. This is
worse than both the cross-fitted v3 head (0.46550) and the ordinary spatial
complement (0.46774) on the site-novel view. The refit is rejected. Fitting
every development row improves apparent training separation while removing
the cross-fitted diversity that mattered for new-site ranking; it is not the
final deployment head. Exact replay script SHA-256 is
`ccc734acbf2c2fc0bbad82c38dbb56e4ac93b5189c0d937206be239fad80e1da`;
result JSON SHA-256 is
`d101752cd3764da055746e5738ed58a9bfa28366575690bc5a66a8d8cd648b71`.

## MethaneS2CM v5.1 frozen context transfer: negative experiment

The independently trained MethaneS2CM v5.1 checkpoint (seed 1101, checkpoint
SHA-256
`7b648548cc62ca3f6d428df2cf427e373fba5a7bdcf03aabada68bf6f1cfc446`)
was tested as a frozen methane-specific representation. Its context branch,
direct mask statistics, and intermediate pooled features produced 401
features for all 44,363 development scenes. The ignored float16 cache has
SHA-256
`6f4a9adedbcf9c389fb4a49d5304f2ed4de92fef61538988e14905063f076b96`.

Zero-shot scores transferred poorly. On inner folds 2/3/4, direct mask,
context, and fused AP were 0.113, 0.079, and 0.093; fold-0 values were 0.148,
0.077, and 0.099; fold-1 values were 0.120, 0.064, and 0.079. A fixed
200-tree ExtraTrees probe over the 401 features plus the established scene
features also regressed: raw AP was 0.86436 versus the current head's 0.90331
on inner folds, 0.88919 versus 0.91429 on fold 0, and 0.87692 versus 0.89661
on fold 1. Logit blends did not recover the inner or fold-0 loss; fold 1 gained
only +0.00029 at a 0.10 blend. The branch is rejected before paper-feature
extraction. Methane-task specificity alone did not compensate for its domain
and input-contract mismatch with mixed MARS Sentinel-2/Landsat pairs. Feature
extractor SHA-256 is
`fed0c05b75ed7b721ba59b2463a944094a770f9196774b77fbc6605a5aea3fe2`.

## Prithvi-EO-2.0 tiny temporal transfer: frozen protocol

The next site-novel representation experiment uses the official
IBM/NASA Prithvi-EO-2.0 tiny temporal/location encoder from
`ibm-nasa-geospatial/Prithvi-EO-2.0-tiny-TL`, pinned at repository revision
`335eadc2c45ad5abe7bd307223e1c48c5b60c41b`. The Apache-2.0 checkpoint SHA-256
is `d47326db9bad502b611f73e3e3f3a0e68b7b82640d67c22c795417f7209f8d70`.
Official model and paper sources are
<https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-tiny-TL>,
<https://github.com/NASA-IMPACT/Prithvi-EO-2.0>, and
<https://arxiv.org/abs/2412.02732>.

Prithvi was pretrained on 4.2 million global HLS V2 time-series samples at
30 m using blue, green, red, Narrow NIR, SWIR1, and SWIR2 plus temporal and
location coordinates. Five spectral channels closely correspond to MARS.
The HLS Narrow NIR slot receives broad Sentinel-2 B08 or Landsat B05 in this
transfer experiment; that declared spectral-response mismatch prevents any
claim of an exact band contract. MARS's 2×2 km, 200×200 Sentinel-2 crop is resized to
128x128, placing a 16-pixel token near the HLS pretraining footprint. Landsat
remains a separately reported sensor stratum.

For every chronological reference/target pair, the frozen 5M-parameter
encoder emits CLS tokens from blocks 3, 6, 9, and 12 plus last-layer token
mean, standard deviation, and maximum for reference, target, signed temporal
difference, and absolute temporal difference: 3,072 float16 features. MARS
omits the reference product ID for 871 of 44,363 development rows and its
GeoTIFF metadata contains no replacement timestamp. Those rows use the target
time coordinate for both frames, a neutral zero-separation encoding recorded
in the cache. Feature values never use labels; labels and physical groups are
stored only for verified downstream alignment. Per-fold caches and the 129 MB
foundation checkpoint remain ignored by Git.

Before any Prithvi scene score was computed, the probe search was frozen to
three views: four-block CLS plus the established 108 scene features;
signed/absolute temporal-change statistics plus those scene features; and all
3,072 Prithvi features plus the scene features. Linear probes use uniform,
physical-group, or site/label/sensor-cell weighting at weight decay 0.001;
uniform linear probes also test weight decay 0.01. Two bounded nonlinear
ablations use 64 hidden units on temporal-change features or 128 on all
features. Each is cross-fitted on folds 2/3/4 and tested at seven predeclared
logit blends from 0.05 to 1.00 with the current stronger head. Ranking favors
worst-fold stability before pooled AP. The single selected probe must pass
paired physical-group AP, recall, per-fold, and sensor gates on selection and
then independently on both folds 0 and 1 before any paper-test Prithvi feature
is extracted.

The five restartable fold shards were extracted with the committed protocol,
then schema-, provenance-, fold-, and duplicate-checked before merging. The
ignored 250,854,616-byte cache contains 44,363 unique rows by 3,072 finite
float16 features and has SHA-256
`e3e52a9453426e5e048cd753daf2597d59cbe820a18ae584c61a2de7ae405f23`.
Fold row counts are 8,987 / 8,798 / 8,833 / 8,799 / 8,946 and positive counts
are 745 / 745 / 797 / 758 / 766. The recorded fallback count is exactly 871.
The compact acquisition/merge receipt includes every shard hash and has
SHA-256
`92995743aadcbaf8c93c2b80421da66bcd6d5b4a5365b8613f990875be0f1535`.

The frozen development probe technically passes its predeclared promotion
rule, with an important qualification. The selected model is a uniformly
weighted linear probe over four Prithvi CLS tokens plus established scene
features, weight decay 0.01, blended at only 0.05 with the current stronger
head. On cross-fitted folds 2/3/4, AP improves +0.02960 versus the original
primary score (paired physical-group lower bound +0.01975) but only +0.000026
versus the current stronger head (lower bound -0.00183). This shows safe
compatibility, not a material foundation-model gain.

The single frozen model then passes both held development folds. Fold-0 AP
delta is +0.02506 versus primary with lower bound +0.00955 and recall delta
+0.00537, but AP is -0.00033 versus the current head. Fold-1 AP delta is
+0.03765 with lower bound +0.02019 and recall delta +0.00537; incremental AP
versus the current head is +0.00130. Both sensor strata pass the allowed
regression bounds. The operational threshold is 0.2788401679788501 and the
ignored artifact SHA-256 is
`e0ac070549f3864af1407aabd64eedf653e5f48e7205320a8303dce097bc2fab`.

Because the frozen rule required statistically positive gains versus primary
and no material regression (rather than significant improvement) versus the
current head, it authorizes one transparent exact-paper replay. The tiny,
mixed incremental effect makes a large test-only-site improvement unlikely;
paper feature extraction must not be interpreted as a new untouched test.
Probe script SHA-256 is
`e7e302e7d32aa5fc4b1cc4cdf812f7fdc4ba45bd11c1835e51de52a3fcef60fb`;
result JSON SHA-256 is
`7b8c935bba06feaefd38559175f4e25474687490547eaa3340ec04b502d2750d`.
The reporting-only deterministic rerun retained the identical artifact and
metrics while reducing the tracked JSON from 1.37 MB to 43.7 KB.

Before paper-cohort feature extraction, the transparent replay pipeline was
frozen separately. It verifies the exact 43,524-scene available sealed
manifest and completed acquisition receipt, extracts only the selected 768 CLS
features into five restartable contiguous shards, and forbids a `labels` array
in every shard and merged cache. The exact five unavailable paper scenes keep
their existing v3 score. The evaluator keeps the already-confirmed dense-mask
gate driven by the unchanged v3 controller at 0.75; using the new blended
score would silently alter the mask gate's calibrated score semantics.

Seven focused Prithvi extraction/merge tests pass together, and a one-row
sealed-cohort smoke produced finite features with no labels. Paper extractor,
merger, and exact evaluator SHA-256 values are respectively
`f3d0fc1c65c6312384e53523c231e33f133c975a7ac1ffc49755a54fa00345b7`,
`0a118513123df51ef6bb7be6392bbf81b5d3c35118a4991a5b9232ce80fb57e4`,
and
`4b9e0feb69c718d3e5d2fd273adab3ff2f91c5a4503a92a16593a6be5c3d951b`.

The first evaluator launch stopped before metric computation because PyTorch
2.6's weights-only loader does not allow the NumPy arrays in the locally
generated fitted-probe payload. The evaluator now loads that payload only
after verifying its frozen SHA-256 identity, using the trusted-pickle path.
This compatibility correction changes no cache, model, score, threshold,
bootstrap seed, or promotion gate and was committed before metric replay.

The frozen exact-paper replay rejects the Prithvi complement as the final
successor. On all 43,529 paper rows, it reaches AP 0.673384 versus 0.641020
(delta +0.032365; paired site-bootstrap lower bound +0.014808), matched-FPR
recall delta +0.030888 (lower +0.018574), FPR delta -0.036820, and the already
confirmed gated-mask IoU delta +0.055598 (lower +0.034816). That view passes.

On the more important 15,655 test-only-site view, AP is 0.467791 versus
0.450274 (delta +0.017517) but its paired lower bound is -0.017805. Matched-FPR
recall improves +0.022026 at the point estimate but its lower bound is
-0.008698. FPR improves -0.041677 and gated-mask IoU improves +0.120900 with
lower bound +0.085986. The AP and recall uncertainty gates therefore fail.
The result is also only +0.000051 AP above the previously tested ordinary
spatial scene head, consistent with the negligible incremental development
effect. No architecture or threshold was adapted to this outcome. The exact
result JSON SHA-256 is
`21f905944c24c119b5412fe6d2e626353f07381d3866811607d92492c25bcc89`.

The exact-paper Prithvi extraction completed without retries across all five
predeclared contiguous shards. Their row counts are 8,704 / 8,705 / 8,705 /
8,705 / 8,705 and their missing-reference-time fallback counts are 428 / 409 /
423 / 405 / 412, totaling the expected 2,077. The merged ignored cache is a
63,339,053-byte, 43,524-by-768 finite float16 array with 43,524 unique sample
IDs, no `labels` field, and SHA-256
`d3d9bfb6423fe9ac6bf53185ea408a476491ca8fa31e941e1782a8c85a016795`.
Its compact tracked receipt binds the sealed manifest, acquisition receipt,
foundation revision/checkpoint, input and NIR-transfer contracts, shard
ranges, and every shard hash. This provenance checkpoint was committed before
the frozen evaluator was allowed to inspect any paper-cohort labels or scores.

## Site-relative spatial background model: frozen protocol

The remaining exact-paper failure is restricted to scene ranking and recall
uncertainty on new physical sites; both mask views, full-view scene metrics,
and false-positive-rate gates already pass. A development-only diagnostic
first tested direct within-site rank, mean, median, z-score, and maximum-gap
normalization of the current scene score. None improved AP. The least harmful
inner transform, a 0.025 maximum-gap logit blend, changed AP by -0.000223 and
also slightly reduced AP on both held folds. Scalar score normalization is
therefore rejected rather than tuned against the paper cache.

The next model uses spatial site context that previous architectures did not
represent. Of 3,811 development positives, 3,807 occur at mixed sites that
also contain negative observations. For each scene and each of the nine
64-by-64 probability/MBMP/SWIR/coverage maps, a label-free pixelwise site mean
is computed from every *other* observation of that physical site. Singleton
sites use the scene as their template and therefore have a zero residual.
Sites are assigned wholly to folds, so no fitted parameter or label crosses a
fold boundary; held-site templates use only unlabeled observations from that
same held site, matching the intended historical-monitoring deployment mode.

Before training, the search is frozen to four residual CNNs: original plus
leave-one-out residual maps, or original plus template plus residual maps;
each uses either physical-group or site/label/sensor-cell weighting. All use
the established residual CNN, eight epochs, AdamW at 0.0003, weight decay
0.001, dropout 0.2, and fixed logit blends 0.05 / 0.10 / 0.20 / 0.30 / 0.40 /
0.50 / 0.625 / 0.75 / 1.00 with the current stronger head. Selection is fully
cross-fitted on folds 2/3/4. Promotion requires positive AP and recall point
gains versus the current head, nonnegative AP gain on every selection fold,
a positive paired-site AP lower bound versus both primary and current heads,
positive held-fold point gains, and a positive pooled folds-0/1 paired-site AP
lower bound versus the current head. Only a complete pass can authorize a
transparent exact-paper replay. The frozen training script SHA-256 is
`cef9460b2d82bffc033aa9049564949b2477155ce3bef428ba4666b4661e8347`;
two focused leave-one-out/template-schema tests pass.

The four-model run passed the complete frozen protocol. The pooled-AP leader
used original, template, and residual maps (+0.003524 AP versus the current
head), but the preregistered worst-fold-first selector chose the more stable
group-weighted original-plus-residual model at blend 0.10. Its worst selection
fold AP gain is +0.000653 versus +0.000557 for the pooled leader, preventing a
post-result preference for the larger pooled number.

Across cross-fitted folds 2/3/4, the selected model improves current-head AP by
+0.002662 and recall at 7.13% FPR by +0.001293. Both sensors improve. The
10,000-replicate paired physical-site AP interval versus the current head is
[+0.000615, +0.005112], while the interval versus primary is
[+0.020750, +0.045670]. Fold-0 AP/recall gains versus current are
+0.001304/+0.002685; fold-1 gains are +0.001562/+0.004027. The pooled held
folds-0/1 AP interval versus current is [+0.000910, +0.002096]. Both held
sensor strata improve and every frozen gate passes.

The operational threshold is 0.30299312593404687. The ignored 3.91 MB artifact
SHA-256 is
`9401678aa4f38fb3b54a914318dc5c2a553b39fa8b309d1a127b9d21abbfc496`;
the compact development result SHA-256 is
`c2b6695dbe24f0b5bc097d89c6fa6332c2af93441079cd55bec99e77aeeb820f`.
This statistically positive incremental evidence authorizes exactly one
transparent paper-cache replay after its label-independent extraction and
evaluator are frozen and committed.

The exact-paper evaluator is frozen to the existing label-independent
43,524-by-9-by-64-by-64 paper spatial cache (image SHA-256
`7cce444a552c6c873c05ae3a972f62a8081fde89d31ac63d94303dbfac3b1b94`;
metadata SHA-256
`b2b58eabf4478b46912d3c3437c1ee0b039841ee283bd7fe0164775b9d448022`),
the development artifact/report hashes above, the exact v3 diagnostic cache,
and the confirmed 0.75 dense-mask gate report. It verifies that paper spatial
metadata contains no labels, computes every site template and raw CNN score
before opening the diagnostic cache containing paper labels/comparator scores,
and leaves the exact five unavailable scenes at their v3 score. Dense masks
remain driven by the unchanged v3 controller, not the new blended score. All
metric and paired 10,000-site-bootstrap gates are identical to the prior exact
replays. Evaluator SHA-256 is
`1203950a85ed11dde2885c79b9a534bd9cd8f59ba99a6a0be31d3f0fe7939c6c`.

The frozen replay rejects the site-relative model as the final successor.
Full-view AP reaches 0.675110 versus 0.641020 (delta +0.034091, paired-site
lower bound +0.015764), matched-FPR recall improves +0.030336 (lower
+0.018433), FPR improves -0.038067, and gated-mask IoU retains its confirmed
+0.055598 gain (lower +0.035020). Every full-view gate passes.

Test-only-site AP is 0.467125 versus 0.450274 (delta +0.016851), with lower
bound -0.017692. Matched-FPR recall improves +0.017621 at the point estimate
but has lower bound -0.011050. FPR and IoU remain decisively better, yet the AP
and recall confidence gates fail. The score is below both the ordinary spatial
model (0.467740) and Prithvi complement (0.467791) on this view. Label-free
site residuals therefore improve development and the mixed full cohort without
solving new-site transfer. No site-context rule will be tuned to this paper
outcome. Exact result JSON SHA-256 is
`7d99569bb915baa456741055a1309aceb550313662a747612c1c65ce28b3d19e`.

## Crossfold-bagged scene head: frozen protocol

The stronger ExtraTrees scene head used for v3 was selected by cross-fitting
folds 2/3/4, then fitted once on those three folds. A later single refit on all
five development folds degraded new-site paper AP, suggesting that its useful
generalization came partly from fold/subset diversity rather than merely more
training rows. Crossfold bagging tests that hypothesis without changing the
proven tree family or fitting hyperparameters.

For each held development fold, four 400-tree uniform-weight ExtraTrees
members are trained on distinct three-fold subsets of the other four folds.
Thus every OOF member excludes the held labels and one additional complete
physical-site fold. The fixed aggregations are probability mean, logit mean,
and probability median; each is tested at primary/head logit blends 0.25 /
0.50 / 0.625 / 0.75 / 0.875 / 1.00. Selection prioritizes all-fold stability
before pooled AP. Promotion requires positive pooled AP and recall versus the
current head, nonnegative AP on every fold, bounded per-fold recall and sensor
regressions, and a positive 10,000-site-bootstrap AP lower bound versus both
primary and current heads.

If promoted, the deployable artifact contains five members, each fitted on
four of five complete development folds; their aggregation and blend remain
the OOF-selected values. No paper feature, score, image, or label is loaded by
this experiment. Frozen script SHA-256 is
`ac3e58d54477b67219c66200180a715591e75f25baebe7350edc61b7af3253b3`;
two focused split-exclusion and aggregation tests pass.

The first crossfold-bagging protocol is rejected before paper scoring despite
a materially larger AP signal. The worst-fold-first selector chose median
probability aggregation blended 0.875 with primary. OOF AP improves +0.008198
versus the current head, with paired physical-site interval
[+0.004470, +0.011569]; recall improves +0.002362 overall. AP improves on all
five folds (+0.004181 / +0.006145 / +0.009181 / +0.015603 / +0.006564), and
both sensor strata improve.

The predeclared per-fold recall floor fails: fold 0 loses -0.004027 recall
(three positives) and fold 4 loses -0.002611 (two positives), exceeding the
-0.002 allowance. The other three folds improve. The 51.19 MB ignored final
five-member artifact SHA-256 is
`c9efecc2315305bb306b6deb84037f60096bdfae94164e3e9e646a2612fdcafb`;
its models were fitted independently of paper data and remain eligible as
fixed components of a separately preregistered development-only trust-region
experiment. Result JSON SHA-256 is
`d9d8da26df1912ac62ae71c32ba8bab108d28f5367feed59656fb8a0d118760e`.

The follow-up trust region is frozen before its rerun. It reuses the identical
nested OOF member construction and hash-pinned five-member final artifact, but
blends each probability/logit-mean or probability-median bag score around the
current v3 stronger head rather than around primary. Bag weights are fixed at
0.05 / 0.10 / 0.20 / 0.30 / 0.40 / 0.50 / 0.625 / 0.75 / 0.875 / 1.00.
Authorization requires at least +0.002 pooled AP versus current, positive
pooled recall, strictly positive AP on every fold, nonnegative recall on every
fold, no sensor AP regression, and positive paired-site AP lower bounds versus
both current and primary. Ranking prioritizes worst-fold recall and AP before
pooled AP, implementing a minimum-risk recall repair rather than maximizing
the already observed pooled gain. Paper data remains excluded. Frozen script
SHA-256 is
`a9f6297cea26dbaba57bcd0ae20d6ce24bfc68fea77f32deb5f422913793b1e7`.

The frozen trust region is also rejected before paper scoring. Mean-logit
aggregation at bag weight 0.10 preserves matched-FPR recall exactly on four
folds and improves it by one positive on fold 3. AP improves on every fold and
by +0.001520 pooled, with paired-site interval [+0.001008, +0.002007], but
misses the preregistered +0.002 pooled materiality floor; Landsat AP also
changes by -0.000332 on fold 4. Raising bag weight to 0.20 yields +0.002896 AP
and positive pooled recall but loses one positive on one fold. The continuous
score trust region therefore cannot simultaneously clear the strict AP,
all-fold recall, and sensor gates. Result JSON SHA-256 is
`e52b74527eba291c9ac0e94be7a4efe483d5c4714738dce062bc8a41627e12d2`.

The next frozen experiment moves recall protection into fitting. Within every
nested member's training folds, positives below that training subset's current
head threshold at 7.13% FPR receive weight 2, 4, or 8; all other scenes retain
unit weight. The hard definition uses only OOF development scores and training
labels. Each multiplier is evaluated with the same three bag aggregations and
bag weights 0.10 / 0.20 / 0.30 / 0.40 / 0.50 / 0.625 / 0.75 / 0.875 / 1.00
around current. Promotion retains the +0.002 pooled AP floor, positive pooled
recall, strictly positive per-fold AP, nonnegative per-fold recall, no sensor
AP regression, and positive paired-site AP lower bounds versus current and
primary. A promoted final artifact would refit five hard-positive-weighted
members, each omitting one physical-site fold. Paper data remains excluded.
Frozen script SHA-256 is
`1e7d21727a99d0d1c43de289035eb04d90245baa53e71a46a280d8ce47703c42`;
the focused hard-positive weighting test passes.

Hard-positive weighting is rejected before paper scoring. The safest result
uses multiplier 2, mean-logit aggregation, and bag weight 0.10. It improves AP
by +0.001521 versus current with paired-site interval
[+0.001024, +0.002009], improves one pooled positive (+0.000262 recall), and
has nonnegative recall plus positive AP on every fold. It nevertheless misses
the frozen +0.002 pooled AP floor, and fold-4 Landsat AP changes -0.000268.
Weights 4 and 8 do not improve the stability frontier; at bag weights large
enough to exceed the AP floor, at least one fold again loses two positives.
Recall-focused sample weighting therefore does not resolve the crossfold
bag's AP/recall tradeoff. Result JSON SHA-256 is
`f3a67223b3d0012f3be14ab8ba21df1910662822313c64a876e09f9c5e8535f7`.

## Unsupervised target-density weighting: frozen protocol

The next branch addresses the consistent new-site distribution shift rather
than adding another source-only score complement. Importance weighting under
covariate shift estimates the target/source density ratio and reweights labeled
source fitting; modern work also warns that unconstrained importance or
test-time adaptation can harm under support mismatch. Relevant primary sources
are <https://arxiv.org/abs/2006.04662>,
<https://openaccess.thecvf.com/content/CVPR2023/html/Chen_Improved_Test-Time_Adaptation_for_Domain_Generalization_CVPR_2023_paper.html>,
and the satellite spatial-generalization expert framework at
<https://openaccess.thecvf.com/content/CVPR2025W/EarthVision/html/Kuriyal_CoDEx_Combining_Domain_Expertise_for_Spatial_Generalization_in_Satellite_Image_CVPRW_2025_paper.html>.

The leakage-safe simulation treats each development fold in turn as an
unlabeled target. A balanced regularized logistic domain classifier receives
the 169 scene/context features plus sensor identity from source and target,
but never target methane labels. Its target/source odds are clipped and used
as ExtraTrees fitting weights. Three fixed contracts compare square-root or
full ratios with [0.25, 4] clipping, plus square-root ratios with [0.1, 10]
clipping. Each target-weighted OOF head is blended 0.10 / 0.20 / 0.30 / 0.40 /
0.50 / 0.625 / 0.75 / 0.875 / 1.00 around current.

Promotion requires at least +0.002 pooled AP, positive pooled recall, strictly
positive AP and nonnegative recall on every fold, no sensor AP regression, and
positive paired-site AP lower bounds versus current and primary. A pass would
freeze the weighting contract before extracting a separate label-free paper
feature cache; only then could one all-development ExtraTrees model be fitted
with paper-feature density weights. Frozen script SHA-256 is
`3e95b711e64a00aeb5389b15a089d51f551cf81bbf4e38457cdf5fcc700564ce`;
the synthetic density-direction/normalization test passes.

The frozen target-density experiment is rejected before paper adaptation.
The selected conservative contract uses square-root density odds clipped to
[0.25, 4] and a 0.30 target-weighted-head blend. It improves pooled AP by
+0.004408 versus current, with paired physical-site interval
[+0.002488, +0.006234], and improves pooled matched-FPR recall by +0.001050.
All five folds improve AP (+0.002050 / +0.003398 / +0.004511 / +0.008843 /
+0.004198), while Landsat and Sentinel-2 improve +0.000745 and +0.006103 AP.

The frozen every-fold recall gate nevertheless fails: fold 4 changes
-0.001305 (one positive at its fold size), while fold 0 ties and folds 1--3
improve. The highly separable fold-1 domain audit (AUC 0.998313) also confirms
that the experiment is addressing a substantial feature-distribution shift,
not merely tree randomness. Because the sole failure is a predeclared
stability condition, no paper feature is loaded and the result remains a
development-only negative. Result JSON SHA-256 is
`1926e16e912e8e3aecc9df5fc1fc6b29d3b2b6af4e0d1d3e9ccd6252999af6a3`.

## Regularized XGBoost scene head: frozen protocol

The existing development program has already compared sklearn histogram
gradient boosting, logistic/HGB stacking, ExtraTrees bagging and weighting,
and spatial CNN ranking objectives. The next experiment therefore changes the
learner family rather than adding another post-hoc score rule. XGBoost's
regularized additive-tree formulation and histogram training are described in
the primary system paper <https://arxiv.org/abs/1603.02754>; the installed
research environment uses XGBoost 3.2.0. The 131.7 MB wheel is confined to the
ignored project virtual environment.

Three fixed CPU-histogram specifications use depths 3 / 4 / 5, learning rates
0.04 / 0.03 / 0.025, 600 / 800 / 1000 trees, and minimum child weights 10 /
10 / 20. Every specification fixes 0.8 row subsampling, 0.7 feature
subsampling, L1 0.1, L2 10, and binary logistic loss. No held-fold early
stopping is permitted. Each model is cross-fitted across all five complete
physical-site folds and blended 0.10 / 0.20 / 0.30 / 0.40 / 0.50 / 0.625 /
0.75 / 0.875 / 1.00 around the current v3 head in logit space.

Promotion requires at least +0.002 pooled AP, positive pooled matched-FPR
recall, strictly positive AP and nonnegative recall on every fold, no pooled
sensor AP regression, superiority to primary, and positive 10,000-replicate
paired physical-site AP lower bounds versus both current and primary. Only a
passing choice is refitted on all development folds and serialized; otherwise
no deployable artifact is produced. Paper features, images, scores, and labels
remain excluded until a passing artifact and a separate exact evaluator are
frozen. Frozen script SHA-256 is
`0ba06e9d2abd2ddbb5b3ce7ae6add30ab2f04d789615843932141268cbc7bf2e`;
two focused contract tests pass.

The frozen XGBoost protocol passes every development gate. Worst-fold-first
selection chooses the shallow depth-3, 600-tree specification at a conservative
0.10 logit blend around current. Pooled AP improves +0.002455, with paired
physical-site interval [+0.001508, +0.003484], and matched-FPR recall improves
+0.000787. Fold AP improves +0.002082 / +0.002206 / +0.002418 / +0.002894 /
+0.002161; recall ties on folds 0 and 4 and improves on folds 1--3. Both
sensor strata improve (Landsat +0.000567; Sentinel-2 +0.003309 AP), including
positive sensor deltas within every fold.

The final all-development artifact is only 129,942 bytes and remains ignored;
SHA-256 is
`e383b9e4e0c3879aa1db4b33d12a823a396a66c8c7abd86ad6813bb44c56fb4b`.
Compact result JSON SHA-256 is
`f79fb3c8b0a0ff2832d6d7c9d1ae945b6bf6fc6d795e661e49bcaba0910da1db`.
This authorizes freezing one exact-paper evaluator, but is not itself evidence
about either paper view.

Before exact scoring, a dedicated extractor will copy only paper sample IDs,
site groups, 108 base scene features, feature names, and aligned frozen-v3
scores from the hash-pinned diagnostic archive. NumPy NPZ members are lazy, so
the extractor does not access label, test-only-site, baseline, or pixel-truth
members. The ignored output is schema-checked against forbidden outcome names
and receives a compact acquisition receipt. This additional boundary does not
restore pristineness to the historically opened paper test; it ensures only
that the new model score is computed without outcome arrays in its input
container. Frozen extractor SHA-256 is
`cb79752666b6f02d3c88cc9fc91c97dff16bcd0d4d2932a2649817aa3d5fd2de`;
its synthetic alignment/schema test passes.

The extractor sealed 43,524 available scenes from 1,289 sites with 108 base
features. The ignored label-free cache SHA-256 is
`8a35e60e7c396e58639f940239020adb36def885124841e0b20901e10db52f33`;
its committed receipt enumerates the five allowed arrays and their forbidden
outcome-name audit.

The exact evaluator is frozen to the XGBoost artifact, development report,
label-free cache and receipt, exact diagnostic comparator, and confirmed
scene-gated-mask report by SHA-256. Available rows receive the fixed 0.10
current/XGBoost logit blend; five missing rasters fall back to v3. Its fixed
operational threshold is the conservative maximum of the candidate thresholds
across all five development OOF folds. Dense masks remain the released model's
sensor-thresholded masks gated at 0.75 by the unchanged v3 score. Both exact
paper views must independently clear paired-site AP, matched-FPR recall,
fixed-FPR, and pixel-IoU point and confidence gates.
Frozen evaluator SHA-256 is
`b4abd3e37c9798877af9014dec988e9d31d55d1bfa3e949308d241b25d676db5`;
its conservative-threshold contract tests pass.

The exact replay rejects the development-promoted XGBoost complement. On the
full 43,529-scene view it reaches AP 0.675991, delta +0.034971 with paired-site
lower bound +0.015654. Matched-FPR recall improves +0.035301 (lower
+0.020871), fixed FPR improves -0.037540, and the confirmed masks retain IoU
delta +0.055598 (lower +0.034992); every full-view gate passes.

On 15,655 test-only-site scenes, AP is 0.464003, below v3 and the spatial and
Prithvi complements. Its delta over the exact comparator is +0.013729 but the
paired lower bound is -0.022625. Matched-FPR recall improves +0.022026 at the
point estimate but has lower bound -0.009132. FPR and IoU remain decisive.
The divergence between uniformly positive five-fold development results and
worse new-site AP is further evidence that source-only scalar model capacity
is not the limiting factor. No XGBoost parameter or blend will be tuned to the
paper outcome. Exact result JSON SHA-256 is
`4d1b802fe6f322e738aca2e74f396bb6f771c773c4ddb7e5ef39242004ba9e3d`.

## Target-adapted XGBoost consensus: frozen protocol

The density-weighted ExtraTrees branch produced the largest recent
development AP gain (+0.004408) but lost one fold-4 positive; regularized
XGBoost produced smaller gains on every fold while tying or improving fold
recall. Their complementary development failure modes motivate a direct
convex logit consensus. Each held fold remains unlabeled to its density-ratio
estimator, and both component models exclude that fold's labels.

The target component is fixed to square-root density odds clipped to [0.25,
4]; XGBoost is fixed to the depth-3, 600-tree specification. Target weights
0.20 / 0.30 / 0.40 / 0.50 and XGBoost weights 0.05 / 0.10 / 0.20 leave the
remaining weight on current. Promotion retains the +0.002 AP floor, positive
pooled recall, strictly positive per-fold AP, nonnegative per-fold recall and
sensor AP, and positive paired-site AP lower bounds versus current and
primary. Only a development pass may authorize a separately frozen
label-free paper adaptation. Frozen script SHA-256 is
`bf64dab45ec6192e406f7e30283e7fd1a3340c16b1c048439515102023cb713f`;
two focused logit-consensus tests pass.

The frozen consensus is rejected before paper adaptation. Selection chooses
current / target / XGBoost logit weights 0.65 / 0.30 / 0.05. It improves
pooled AP +0.005435 with paired-site interval [+0.003180, +0.007958] and
recall +0.002624. Fold AP improves +0.002906 / +0.004432 / +0.005643 /
+0.009926 / +0.004948; both sensors improve in every fold.

Fold 4 nevertheless loses -0.001305 recall (one positive at its fold size),
while fold 0 ties and folds 1--3 improve. No candidate in the frozen grid has
nonnegative recall on every fold. This reproduces the target-weighting
boundary failure despite an independent recall-safe component and closes the
continuous scalar-ensemble path. No paper feature was loaded. Result JSON
SHA-256 is
`bdd8c1f0ccb7472b0ab3c7f54ab48d720cd39ece212e475438849c981f3a6212`.

## Adaptive Prithvi moment alignment: frozen protocol

The next branch changes representation geometry rather than scalar weights.
AdaBN (<https://arxiv.org/abs/1603.04779>) and CORAL
(<https://arxiv.org/abs/1612.01939>) motivate estimating feature moments from
unlabeled targets. For each held physical-site fold, source and target receive
independent per-feature mean/variance normalization; only source labels fit a
regularized logistic probe. The target labels never enter normalization,
fitting, or selection inputs.

The fixed search compares 768 four-block Prithvi CLS features alone or with
the established 169 scene/context features, C 0.001 / 0.01 / 0.1, and current
logit blends 0.05 / 0.10 / 0.20 / 0.30 / 0.40 / 0.50. Positive source rows
receive the square-root imbalance weight. Promotion retains the +0.002 AP,
positive pooled recall, all-fold AP/recall, sensor, and paired-site confidence
gates. A pass would authorize a separately frozen label-free paper alignment;
otherwise the paper cache remains unused. Frozen script SHA-256 is
`658e56607d9df031c935b9fc00da2b9e1fcb843d7f62b36d67b4271a94085af3`;
the independent-moment test passes.

The adaptive Prithvi protocol passes every frozen development gate. It selects
CLS plus 169 scene/context features, C=0.01, and a 0.10 blend. AP improves
+0.004342 versus current with paired-site interval [+0.002495, +0.006361];
matched-FPR recall improves +0.001312. Fold AP deltas are +0.002288 /
+0.001244 / +0.003090 / +0.004702 / +0.001855, and every fold gains one or
two matched-FPR positives (+0.001342 / +0.001342 / +0.001255 / +0.002639 /
+0.002611). Pooled Landsat and Sentinel-2 AP improve +0.001097 and +0.005594;
fold-1 Landsat changes -0.000228 but the predeclared pooled sensor gate passes.

The ignored control artifact SHA-256 is
`a38be1acc8ca425ef5307c0ad2b253274fd36d4c3e6b5ed63ce1bf205f6fb0d5`;
compact result JSON SHA-256 is
`3a0478d9abdb94a76815744cefcf7ac181765c20e2969a7a40f310f272a03a8b`.
This authorizes label-free paper alignment but is not paper-view evidence.

The authorized adaptation step is separated from evaluation. It hash-pins the
development/control artifacts, combines the sealed paper Prithvi CLS cache
with the separate label-free 108-feature paper cache, reconstructs the same
169 label-free site-context schema, estimates independent source/target
moments, fits only development labels, and writes only paper IDs and scores.
The outcome-bearing diagnostic archive is not an input to this step. Frozen
adaptation script SHA-256 is
`d95b77cb1246acd04125f1e982430e2605e290b30262787a35d78cd1ac2c442d`.

The label-free adaptation produced 43,524 finite paper scores without opening
outcomes. The 1.48 MB ignored score cache SHA-256 is
`37aa4eb2e14bd7265df95a0cf55fc805e2a6e1ee8a2af7aa3e84a23165fe0059`;
the committed receipt SHA-256 is
`1b76927c30b3614c0c08c4139f25fb4e66f92b517ac4a5fc76d3e4cc06cb5f9f`.

The exact evaluator is frozen to the label-free score cache and receipt,
development report, exact diagnostic comparator, and confirmed mask gate. It
uses the conservative maximum development OOF threshold, v3 fallback for five
missing rows, and unchanged v3-driven dense masks. Both paper views retain all
seven AP/recall/FPR/IoU point and paired-site confidence gates. Frozen
evaluator SHA-256 is
`7850a1c53a646000faaf56e977461863c097b55f76334dbad5d915bfeb978cd9`.

The exact replay rejects adaptive Prithvi. Full-view AP reaches 0.677001,
delta +0.035981 with paired-site lower bound +0.016325; matched-FPR recall
improves +0.034197 (lower +0.020586), and FPR/IoU gates pass. This is a valid
full-cohort improvement.

Test-only-site AP is 0.465504, delta +0.015231 with lower bound -0.026300.
Matched-FPR recall gains six positives (+0.026432), one more than v3, but its
lower bound remains -0.006711. FPR and gated-mask IoU remain decisive. Global
target moment alignment therefore improves recall slightly without producing
site-distributed AP/recall confidence and is rejected. No normalization or
regularization will be tuned to this outcome. Exact result JSON SHA-256 is
`9a37b84e410943953f3959a4ca8ae2c3d79db5cd90cb494bd1e695db30aca5f6`.

## Multi-environment V-REx spatial representation: frozen protocol

The next branch changes the learned representation. V-REx
(<https://arxiv.org/abs/2003.00688>) penalizes variance in risk across source
environments, while group-DRO evidence
(<https://arxiv.org/abs/1911.08731>) motivates stronger regularization for
worst-group generalization. Each physical-site fold is an environment. The
existing nine-channel physics/morphology residual CNN is trained on four
source folds with mean environment risk plus beta times risk variance.

Two fixed beta values (0.5 and 2.0), two independent seeds, eight epochs,
dropout 0.3, AdamW 2e-4 and weight decay 0.003 are frozen. Site/label/sensor
cell row weights and square-root positive weighting remain. Each seed produces
five leave-one-fold-out models; candidates average both seeds and test blends
0.05 / 0.10 / 0.20 / 0.30 / 0.40 around current. Promotion requires the
usual pooled, all-fold, sensor and paired-site gates plus nonnegative fold
recall for each seed separately. A passing deployable artifact is the ten
crossfold members, avoiding an all-data refit that previously harmed new-site
transfer. Frozen script SHA-256 is
`142b86e7d27fe9a78fabe347381cf1bc1b7e694999900f5c00b6169d4af52a41`;
the environment-risk penalty test passes.

The frozen two-seed V-REx sweep is rejected before paper scoring. Its selected
candidate is beta=0.5 with a 0.05 blend around current. Development AP changes
only +0.000077 and its paired-site interval is [-0.000297, +0.000429];
matched-FPR recall changes -0.000787. Folds 2 and 4 each lose one positive at
the operating point, so the pooled recall, all-fold recall, bootstrap, and
per-seed stability gates fail. The representation does improve substantially
over the original primary model (+0.029646 AP), but it adds essentially no
independent signal beyond the current ensemble. This closes environment-risk
variance regularization of the existing nine-channel spatial CNN; the paper
cache was not loaded. Compact result JSON SHA-256 is
`008df042388e601740f3c24088efda9356da38dea5eb542b8d7ce52409230d37`.

## Spatial Prithvi patch representation: frozen extraction protocol

The next branch adds spatial foundation information rather than another loss
over the existing representation. Prithvi-EO-2.0 is a multi-temporal ViT/MAE
whose non-overlapping 3-D patch embeddings retain temporal and spatial token
geometry (<https://arxiv.org/abs/2412.02732>; official model card
<https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-tiny-TL>). The
earlier probe compressed these tokens to CLS and global statistics, while the
existing spatial CNN used only nine pooled probability/physics maps.

The frozen development extraction therefore keeps signed target-minus-
reference patch tokens from encoder blocks 3, 6, 9, and 12. At the fixed
128-pixel input and 16-pixel patch size this produces a 768 x 8 x 8 float16
map per scene. Plume and observability fractions are stored separately at the
same grid for auxiliary training; they cannot enter paper feature extraction.
The cache is resumable, hash-pinned, and ignored by Git. A 64-row real-data
CUDA smoke run completed with the expected shape and finite values. Frozen
extractor SHA-256 is
`7f6bb99befcffa96f18e58eb7431dccfa2b60d051c461f6c36f48c5323cf9f35`;
the two tensor-contract tests pass.

The full development extraction completed 44,363 scenes in 1:04:46. The
feature tensor is 4,361,060,480 bytes with SHA-256
`c0bd358da7563bec1e6e3cd706aeec86c6558b5bde6181c66bab9c5b30621925`;
the separate 11,357,056-byte target tensor SHA-256 is
`6aca9c26dd72d56d3a8c95b7ccc2ef55331f2c8f376a06e4fc2298099744c3c9`;
metadata SHA-256 is
`de89016e60ab5b6b71cf3f2c6b5d4b9885a1d7257b0cf34e09158396fc0fe813`.
Sample IDs, labels, sensors, and groups exactly match the established physics
cache. The bulk files remain ignored.

## Spatial Prithvi patch head: frozen development protocol

The fixed 1.52M-parameter head projects each of the four 192-channel encoder
depths separately, fuses them with the existing nine physics/probability maps
pooled to 8 x 8, and uses two local residual blocks. Observable-weighted
attention, mean, and maximum pooling feed the scene classifier. Training uses
site-cell weights, square-root positive weighting capped at four, AdamW at
3e-4 with weight decay 0.01, dropout 0.3, eight epochs, and scene BCE plus
0.25 times observable fractional-patch BCE/Dice. Positive rows without pixel
truth are excluded only from the patch term, not scene supervision.

Two fixed seeds (20261900 and 20262000) each produce five geographic
leave-one-fold-out members. Their averaged OOF score is tested only at blends
0.05 / 0.10 / 0.20 / 0.30 / 0.40 around current. Promotion requires at least
+0.002 pooled AP, positive pooled recall, positive AP and nonnegative recall
in every fold, no pooled sensor AP loss, positive paired-site lower bounds,
and independently positive AP plus nonnegative fold recall for each seed. A
pass preserves all ten crossfit members; a rejection never opens the paper
cache. Frozen trainer SHA-256 is
`26e83caef2915eea73bda006ac7af87e65e6a6683a0559c58fc34ed88b1a765f`;
its forward and missing-mask loss tests pass.

The frozen spatial Prithvi head is rejected before paper scoring. At the
minimum 0.05 blend it changes pooled AP -0.000654 and matched-FPR recall
-0.002886; the paired-site AP interval is [-0.001489, +0.000106]. Landsat AP
changes +0.000034 but Sentinel-2 changes -0.000932. Fold AP deltas are
+0.000155 / +0.000044 / -0.001883 / +0.000319 / +0.000385; fold 4 loses two
matched-FPR positives (-0.002611). The multi-depth tokens optimize stably but
do not add a sufficiently general rank signal beyond current, especially on
fold-2 Sentinel-2. No paper feature was extracted. Compact result JSON
SHA-256 is
`77507f57de075a7b47adf36831d93830900db51062484ee48765094dea1a3f2f`.

## MethaneS2CM v5.1 end-to-end transfer: frozen pilot

The earlier MethaneS2CM experiment froze zero-shot v5.1 context features,
which were not transferable. End-to-end adaptation remains distinct and is
tested first on one inner development fold before authorizing a costly full
campaign. The source checkpoint is the independently validated v5.1 seed-1101
model (SHA-256
`7b648548cc62ca3f6d428df2cf427e373fba5a7bdcf03aabada68bf6f1cfc446`),
which achieved 0.818 scene AP on its sealed MethaneS2CM location test.

MARS provides one reference rather than v5.1's 90- and 365-day references, so
the single reference and MBMP map are duplicated into both pretrained slots.
All weights are adapted on physical-site folds 3 and 4 at 64 x 64, using the
v5.1 segmentation-first BCE/hard-negative/Dice/scene loss, balanced
label-sensor sampling, AdamW 1e-4 with weight decay 0.01, seed 20262200, and
exactly three epochs. Fold 2 is scored once at current-logit blends 0.05 /
0.10 / 0.20 / 0.30. The pilot must gain at least +0.003 AP, retain recall and
both sensor APs, and have a positive paired-site AP lower bound. Only a pass
can authorize two-seed five-fold transfer; paper data is excluded. Frozen
script SHA-256 is
`08b9c96b3dbe2b11f2ae6ea7151b1b82587b80118306fa410113fd0b5fddb4f5`;
the input-contract test passes.

The frozen end-to-end transfer pilot is rejected before a full campaign.
Three-epoch adaptation reduces mean scene BCE from 0.519 to 0.331. At the
selected 0.10 current-logit blend, fold-2 AP improves +0.001369, recall is
unchanged, and Landsat/Sentinel-2 AP improve +0.000155/+0.001504. However the
gain misses the frozen +0.003 pilot floor and its paired-site interval is
[-0.001341, +0.003668]. MethaneS2CM pretraining transfers weakly after full
adaptation, but not decisively enough to justify ten crossfit runs. The paper
cache remains unused. Compact result JSON SHA-256 is
`2c08a97c91923d6522fdef10371170b821cf3839396625baebdc5071e995a226`.

## Geographic neighboring-site prior: frozen protocol

The exact new-site benchmark excludes exact training facilities but not whole
emission basins. Neither released MARS-S2L nor the current scene/context head
uses neighboring sites' historical labels. The next deterministic branch
tests whether this operational information improves unseen-facility ranking
without using any held-site label.

For each held physical-site fold, source sites receive Jeffreys-smoothed scene
prevalence from source labels. Each held site receives a haversine k-nearest
estimate weighted by exp(-distance/scale) and square-root source observation
count, with one global-prior pseudo-observation. The fixed grid is k=5 / 20 /
50, scale=100 / 500 / 2,000 km, and current-logit blend 0.05 / 0.10 / 0.20 /
0.30 / 0.40. Promotion retains the +0.002 pooled AP, positive recall,
all-fold AP/recall, sensor, and paired-site gates. A pass would store only the
selected deterministic specification and use all pre-2024 development sites
as the historical source for a label-free paper scoring step. Frozen script
SHA-256 is
`9a689b20f6564e45a52c91c0d6962deb7db68e20b187ed56f772cfe366282bb5`;
the distance and held-label exclusion tests pass.

The geographic prior is rejected before paper scoring. The safest result uses
20 neighbors, a 2,000 km scale, and a 0.05 blend. Pooled AP changes -0.000052
with paired-site interval [-0.000573, +0.000589]; recall gains one positive
(+0.000262). Fold AP changes -0.000379 / -0.000354 / -0.000075 / -0.000395 /
+0.000829, so four all-fold AP gates fail. Neighboring historical prevalence
does not add useful ranking information beyond the existing scene/site
context. No paper score was computed. Result JSON SHA-256 is
`f199c636653f471353c8455211b396d5b10e1f948505618dbbf7a1fa51dc024f`.

## UNEP MARS post-2024 cohort: frozen acquisition protocol

The source-only, invariance, frozen/spatial foundation-model, end-to-end
cross-dataset transfer, and geographic-prior branches are now exhausted
without a development result strong enough to justify another paper scoring
step. The next hypothesis is genuinely new, independently validated examples
under the exact MARS-S2L target/reference contract.

UNEP IMEO's Eye on Methane MARS export provides the exact target product,
exact multitemporal background product, timestamp, source coordinates,
satellite, plume/source identity, and expert-validation status for each
detection. Before downloading the catalog, the post-2024 protocol freezes
Sentinel-2/Landsat-only selection, `actionable=YES`, exact product IDs, a 25 km
exclusion around every pinned paper-test location, exact paper-test product
exclusion, source-disjoint hash roles, and positive-only label semantics.
Only polygonal GeoJSON may supply pixel truth; catalog absence is never a
negative. The sealed external bucket cannot influence architecture or
threshold selection. Bulk data remains ignored under
`.research/unep_mars_post2024/`.

The first schema-only catalog audit found 26,762 rows spanning 2020-10-29 to
2026-06-15. Before computing any eligible-cohort result, the protocol received
one transparent schema correction: UNEP uses the aggregate display value
`Landsat - NASA/USGS`, not spacecraft-specific display names. The selection
therefore recognizes that value while still requiring exact `LC08` or `LC09`
product prefixes. No date, validation, spatial-exclusion, grouping, or
promotion rule changed.

The GeoJSON schema audit found 25,555 MultiPolygon plume geometries and 1,207
records without geometry. Before outcome filtering, deduplication was also
clarified from whole-product identity to exact product plus source-centered
crop. A 100+ km satellite product can contain multiple distant 2×2 km samples;
only plume polygons for the same target/source crop are merged. This preserves
the released MARS-S2L sample contract and does not inspect labels or model
scores.

The frozen catalog audit yields 237 exact-product source-crop samples across
42 independent 25 km groups: 153 Sentinel-2 and 84 Landsat. Every sample has
expert-actionable positive evidence and MultiPolygon pixel truth. The fixed
hash assigns 215 samples to auxiliary training, 9 to development, and 13 to
sealed external confirmation. All accepted samples are at least 25.631 km
from every paper-test location; none were selected by model score. The compact
audit records the two source archive hashes, the 26,762-row catalog identity,
the 13,393 exact paper-test targets and 1,289 paper-test locations excluded,
and eligible manifest SHA-256
`13a19dd7daad36b40261f7acb0332b0418cb37b1b9407d708e864b3847976e74`.
This cohort is large enough to authorize exact-product asset resolution.

The frozen nonsealed resolver finds 167 fully available exact target/reference
pairs out of 224: 141/143 Sentinel-2 and 26/81 Landsat. Auxiliary training has
162 resolved samples and development has 5. There are zero query errors.
Unresolved sides are 38 Landsat targets, 26 Landsat references, one Sentinel-2
target, and one Sentinel-2 reference; the Landsat losses are dominated by
retired real-time products. No later tier product, L2 product, or otherwise
similar acquisition is substituted. Sealed-external products were not
resolved. The ignored exact-asset manifest SHA-256 is
`027022bf72ea7f17e68bf69c887f5e408cf9ca5aa485ef4c59c0560b5e25d269`.

Before new pixel acquisition, a released 12-band MARS raster was directly
inspected: it is 200×200 at 10 m in the target UTM CRS, a 2×2 km footprint.
Earlier prose calling the crop 4 km is corrected. All prior training and
benchmark evaluation used the actual stored 200×200 rasters, so no numerical
result changes.

Exact crop acquisition succeeds for all 141 resolved Sentinel-2 samples (136
auxiliary training, 5 development). Every 12-band 200×200 uint16 target/
reference crop has a nonempty UNEP plume mask and passes the frozen 0.80
radiometry gate; compact ignored raster storage is 87,674,839 bytes. All 26
resolved Landsat attempts fail before artifact acceptance because the public
USGS STAC pixel links redirect to the separate EROS login page. This is an
access gate, not an image-quality rejection. No Earthdata credential applies
to EROS, no tier/L2 substitute is used, and the exact Landsat metadata remain
available for a later authenticated extension. Sentinel-2 proceeds to the
frozen CloudSEN12+ observability acquisition.

CloudSEN12+ acquisition completes for all 141 exact Sentinel-2 crops with zero
errors using the already pinned `UNetMobV2_V2.pt` weights (SHA-256
`218fa69aa3c7212d4e690b48af88ac6f3c976fc50d07f275b8fd623909183d7a`).
At the frozen 0.80 scene/plume clear gates, 139 pass: 135 auxiliary-training
and 4 isolated development samples. One sample in each role fails and remains
documented but excluded from model fitting/scoring. Median scene and
plume-polygon clear fractions are both 1.000. This produces a same-sensor,
same-acquisition, exact-product positive cohort more than twice the size of
the 55-sample time-offset EMIT confirmation and with direct polygon truth.

The model-compatibility build verifies every accepted image, plume mask, and
CloudSEN12 sidecar by byte count and SHA-256, then successfully loads all 139
examples through the released MARS adapter. The ignored auxiliary manifest
contains 135 positives across 27 independent source groups (SHA-256
`2971ebad317c7f709a677e5c6431a75804fce4026bbcf28fa50ce5bb4cc89300`);
the isolated development manifest contains 4 positives across 4 groups
(SHA-256
`5a17c24cb7a78941ed8586cbf99813c3f4772040d5c4bfb9377b2d3a675e1741`).
Catalog flux, wind, and polygon coordinates are not model features. Wind is
explicitly zero-filled only to retain the released 16-channel tensor shape,
and polygon geometry supplies target pixels only. No sealed crop asset was
opened.

## UNEP MARS post-2024 positive baseline

Before inference, the evaluation froze two released-logit endpoints: the
paper's 0.5 pixel cutoff and the current controller's independently confirmed
0.8 Sentinel-2 cutoff, each with the 100-pixel connected-component rule. The
positive-only cohort is permitted to estimate detection recall and mask
overlap, never AP, FPR, precision, specificity, or AUROC. Uncertainty uses
10,000 bootstrap replicates over the frozen 25 km source groups.

On 135 auxiliary-training positives, released 0.5 detects 87/135 (recall
0.6444, group-bootstrap 95% CI 0.5030--0.8061) with aggregate pixel IoU 0.4440
(0.3743--0.5199). The current 0.8 Sentinel-2 threshold detects 78/135 (recall
0.5778, 0.4279--0.7576) with IoU 0.3945 (0.3154--0.4806). On the four isolated
development groups, both detect 3/4; IoU is 0.4308 at 0.5 and 0.3235 at 0.8.
The development intervals are necessarily wide and did not select an
endpoint. The consistent direction shows that merely tightening the released
mask sacrifices true-plume sensitivity on the independently acquired cohort.
It does not authorize loosening the production threshold because positive-only
data provide no false-positive evidence. The next experiment must improve
hard-positive representation while preserving the existing negative replay
and frozen paper operating point.

## UNEP-positive augmented scene head

The first architecture use of the new cohort is scene-only: the already
successful regularized depth-3 XGBoost representation is refit with all
original MARS negatives/positives plus the 135 source-disjoint UNEP auxiliary
positives. Dense masks, sensor mask thresholds, and the mask gate remain
unchanged. The protocol froze auxiliary multipliers 1/2/4/8 and small logit
blends 0.025/0.05/0.10/0.20 before fitting. Fold 2 selected while folds 0 and
1 and the four UNEP development groups remained unread.

The selected setting is the most conservative auxiliary multiplier, 1.0, with
a 0.20 complement blend. On fold 2 it improves AP by +0.00412 and recall at
7.13% FPR by +0.00251 over the current stronger scene head. With that setting
frozen, independent fold 0 improves AP by +0.00392 and recall by +0.00134;
its paired group-bootstrap AP interval is [+0.00052, +0.00808]. Fold 1
improves AP by +0.00325 and recall by +0.00537, with AP interval
[+0.00067, +0.00754]. Both Sentinel-2 and Landsat AP pass noninferiority on
both folds. Only after those gates passed were the four isolated UNEP
development features opened: candidate and current positive recall are both
3/4 at the frozen conservative threshold. The final ignored artifact SHA-256
is `f570a2ffa93486e9d4d5e3b52c4f4723f384f42c6c2692eb0f632ce7de4cbfcb`.
This is the first new-data architecture to pass selection, two independent
original-development confirmations, paired uncertainty, and external-domain
noninferiority. It is therefore eligible for an exact transparent paper-cache
replay; no paper row entered fitting or selection.

The frozen exact-paper replay rejects the branch despite its clean development
promotion. On the full 43,529-row view, it remains strong: AP 0.67429 (delta
+0.03327, 95% site interval [+0.01409, +0.04795]), matched-FPR recall delta
+0.03309 ([+0.01961, +0.04624]), and unchanged promoted-mask IoU 0.37996
(delta +0.05560, lower bound +0.03531). On the 15,655-row test-only-site view,
however, AP falls to 0.44823, delta -0.00204 with interval
[-0.04284, +0.03440]; matched-FPR recall rises by +0.01762 but its lower bound
is -0.01149. Mask IoU remains decisively higher at 0.29246 (delta +0.12090,
lower bound +0.08528). This is evidence of unseen-facility domain shift, not a
reason to post-hoc tune the auxiliary weight or blend on the paper cache. The
branch is frozen and rejected. The next data path must add contemporaneous,
source-disjoint negative evidence or a demonstrably invariant ranking signal;
positive-only augmentation by itself is insufficient.

## CloudSEN12+ source-disjoint negative augmentation

The rejected positive-only branch motivated a primary-source audit for
explicitly labeled negative backgrounds. The current MARS-S2L repository at
revision `c26b1d7e31a0c5241fa37c9140802622c215eb32` was checked first. Its
93,538-row `validated_images_all.csv` is byte-identical to the frozen local
paper source (SHA-256
`799fa3272be6c313534c5d974894883db9f97874adb617eeaace1c8a4f9dc9b2`),
so the July repository activity does not provide a new MARS label cohort.

MethaneSET's June 2026 Sentinel-2 pretraining release is genuinely useful as
a format and pretraining resource: it contains 57,291 explicitly plume-free
target/reference crops. It is not new label evidence for this benchmark,
however. Its 19,572 unique target products all occur in the existing MARS-S2L
table; multiple 2 km facility crops explain the larger row count. MethaneSET
is therefore not counted as an independent negative confirmation cohort and
cannot be used to relabel or leak the paper test split.

The MARS paper's CloudSEN12+ false-positive cohort is independent of the 1,315
emitter-site archive and provides the missing negative domain. UNEP publishes
10,435 clear-scene metadata rows from 10,150 ROI groups across 181 countries,
with disjoint ROI identities in the published train/validation/test splits.
The companion label-free statistics table contains the exact 10,434 scenes
used in the paper: 9,804 training negatives, 256 validation negatives, and 374
sealed test negatives; one metadata-only training row lacks statistics. Every
row is explicitly `isplume=false`.

Before fitting, a protocol froze 32 operationally available features: wind,
global MBMP moments/range, global moments/range for the six shared spectral
bands, and clear/non-clear pixel counts. Country, coordinates, ROI, sector,
offshore state, flux, plume geometry, notification state, and split identity
are prohibited model inputs. Two small regularized XGBoost specifications,
three CloudSEN12 negative weights, and five conservative logit blends are
predeclared. Selection remains on cross-fitted MARS folds 2/3/4, fixed-family
confirmation remains on folds 0/1, and the CloudSEN12 test partition remains
sealed until all original-development and 256-row negative-development gates
pass.

The label-free extractor maps all 44,363 authorized MARS development scenes
to the common statistic schema and extracts only the 10,060 nonsealed
CloudSEN12 rows. Ignored caches are
`mars_cloudsen12_common_stats_development.npz` (SHA-256
`99ce2c6c39b3d184ef8821290d32a8759291adf1e825a1d1ec421602154a8a4e`)
and `cloudsen12_common_stats_nonsealed.npz` (SHA-256
`404bd6569949017014966d8fc04b2f94e07654df98120a30eb1b3d380fa9bcaf`).
No paper-test label or CloudSEN12 sealed-test feature is present. This branch
is now ready for its preregistered development-only model fit.

The common-statistic branch is rejected at its first selection gate. The
frozen choice is the shallower depth-2 head, CloudSEN12 negative multiplier
4.0, and minimum 0.025 logit blend. Pooled folds-2/3/4 AP changes only
+0.000083, recall ties, and the paired 403-group AP interval is
[-0.000419, +0.000629]. Fold 2 regresses AP by -0.000135 and recall by one
positive (-0.001255), so the per-fold stability gate also fails. In contrast,
the 256 source-disjoint validation negatives receive low raw-head scores
(p95 0.03145, p99 0.05154 versus MARS held-negative p95 0.25155), confirming
that the external data and negative label contract behave as intended.

No fold-0/fold-1 confirmation, CloudSEN12 sealed-test feature, or paper cache
was opened. Global spectral/MBMP moments can recognize generic clear
backgrounds but do not supply the localized, model-aligned morphology needed
to reorder difficult MARS scenes. The justified continuation retains the
CloudSEN12 source but acquires a bounded target/reference crop pilot so the
released detector and the frozen spatial feature extractor can generate
hard-negative representations. A full 10,434-scene imagery download is not
authorized unless that pilot first shows transferable development benefit.

The bounded spatial pilot is frozen at 512 exact Sentinel-2 target/reference
pairs: 384 auxiliary-training negatives and 128 development negatives. Every
selected scene has exactly 40,000 published clear pixels and zero non-clear
pixels, is at least 25 km from every MARS emitter location, and belongs to the
published CloudSEN12 train or validation partition. Half of each partition is
selected for high MBMP variance/range and half by deterministic country
round-robin. The result spans 506 ROI groups and 176 countries; the closest
selected crop is 27.64 km from a MARS emitter. Country and location metadata
are sampling-only, not model features. The ignored selection manifest SHA-256
is recorded in `cloudsen12_spatial_pilot_selection.json`; zero published test
rows were selected or opened.

Exact-product resolution succeeded for 492 of the 512 frozen pilot rows: 367
auxiliary-training negatives and 125 development negatives. The remaining 20
rows lack at least one exact target/reference product in the public Sentinel-2
L1C catalog and were retained as unavailable without substituting another
processing level or acquisition. The ignored resolver manifest has SHA-256
`d2af7b9b60a98bf2708cc2cb54e08e3ed85e6533dc85c94cc40b8d499ce35c93`.
All published CloudSEN12+ test rows remain sealed. Crop acquisition is allowed
only for these resolved nonsealed identities and now enforces label-aware
geometry: a clear-scene negative must have an exactly zero plume mask, whereas
the pre-existing positive contract still requires nonempty rasterized truth.

The frozen metadata also names producer-side 200x200 CloudSEN12+ image and
cloud-mask paths, which would have been the closest and most efficient data
source. A recursive audit of the pinned public MARS-S2L tree found that those
rasters are not published: only the metadata and statistics CSVs are present,
and a direct named-crop request returns HTTP 404. This rules out a missing-login
explanation. The pilot consequently reconstructs the exact grid from the
resolved public L1C target/reference identities; the raw path is slower but
does not substitute products or processing levels.

A first-row grid audit stopped that reconstruction before model use. A
center-derived grid began at easting 408500, while the frozen producer affine
for the same scene begins at 408510: an otherwise easy-to-miss one-pixel
offset. The selection contract now carries the published CRS, affine, width,
and height for every row, and the cropper uses that grid without recomputing
it from the rounded longitude/latitude center. Selection membership and all
source/product identities are unchanged. The corrected ignored selection
manifest SHA-256 is
`98eaacc62e94fb72cbdc05aa74172283165b1f2f235babde4606a06881813908`;
the resolver output remains byte-identical at
`d2af7b9b60a98bf2708cc2cb54e08e3ed85e6533dc85c94cc40b8d499ce35c93`.
An overwrite smoke test confirms exact equality among producer, manifest, and
written-raster transforms for the audited row. No feature or outcome was
computed from the offset crops.

The full 512-row geometry diagnostic shows this was systematic: only 18
rounded-center grids exactly matched the producer affine. Another 413 differed
by one pixel on both axes, 46 differed on the vertical axis, and 35 differed
on the horizontal axis. Preserving the producer affine is therefore a material
architecture/data-contract correction rather than a cosmetic one-row fix.

The remaining Sentinel-2 resampling contract was resolved from the public
`marss2l` 0.2.4 and `georeader-spaceml` 1.5.9 wheels. The official pipeline
does not directly bilinear-warp B11/B12 from their 20 m source grids. Earth
Engine first exports them at 10 m with nearest neighbor, after which
`interpolate_20mbands_s2ee` downsamples to 20 m with nearest neighbor and
restores 10 m with bilinear interpolation and uint16 rounding. The cropper now
implements that published rule. On the audited row, the exact producer affine
and product already reproduce the published B02/B03/B04/B08 mean, standard
deviation, minimum, and maximum to numerical precision. The official SWIR
interpolation was frozen from source code rather than selected against a model
metric. Package hashes and code paths are recorded in
`MARSS2L_SENTINEL2_PREPROCESSING_AUDIT.md`.

A pre-feature-extraction hash check corrected one metadata-only transcription
in the spatial-pilot protocol: the pinned released `best_epoch` checkpoint is
SHA-256
`be634fb9e24dc4877f44c1ff9f69972e6f0453e30d70c0dc03677876340ef246`,
matching the local file and every earlier verified training/evaluation receipt.
The mistyped protocol value began `be634d`; no different checkpoint was loaded,
and no data partition, architecture, fitted weight, or outcome was changed.

All 16 deterministic acquisition shards completed and jointly reconcile to
492/492 resolved nonsealed crops, 492 pre-cloud geometry/radiometry passes,
and zero errors: 367 auxiliary-training negatives and 125 isolated-development
negatives. Ignored image and zero-plume-mask storage is 287,840,064 bytes. The
canonical unsharded verifier then rechecked every cached identity, current
preprocessing contract, byte count, and asset SHA-256 without redownloading.

The one-row producer-statistics audit is intentionally precise about replay
equivalence. Public exact-product JP2 data on the published producer grid
reproduce B02/B03/B04/B08 mean, standard deviation, minimum, and maximum to
numerical precision. B11/B12 follow the released Earth Engine nearest export
plus `georeader` nearest-down/bilinear-up algorithm, but are not bitwise equal
to the unpublished producer GEE TIFF: for the audited target, reconstructed
B11 mean/std are 0.535728/0.049399 versus 0.537828/0.047414 published, and B12
mean/std are 0.425124/0.064243 versus 0.427001/0.063348. This is recorded as an
upstream pixel-source limitation, not hidden as exact TIFF recovery and not
tuned against a model outcome.

The loader-manifest build hash-verifies every 12-band image and zero plume
mask, materializes exact-grid all-clear cloud masks from the published
40,000-clear/0-nonclear label, and retains all 492 successful rows. The
auxiliary manifest spans 362 groups (SHA-256
`eab58d320d53d1b1937b148d9efc2c5b4f0f17618dfe60e13f5351481bafb377`);
development spans 124 groups (SHA-256
`9652b546b5f0db933955e4f9049804e7af571216d37fb110912681d61cda841a`).
The 20 frozen exclusions are exactly the unresolved products; zero sealed-test
rows were acquired or materialized.

The first auxiliary feature pass stopped before writing a cache because the
loader correctly propagated non-finite wind metadata. The failure was not a
raster defect or mixed-precision overflow: 103 accepted CloudSEN12+ rows have
one or more missing published wind components, which made the broadcast wind
channels and downstream logits non-finite in both float16 and float32. The
manifest builder now retains every row but zero-fills only missing components,
matching the already documented UNEP external-cohort compatibility convention;
it records row/component imputation counts and writes strict JSON with
`allow_nan=false`. Feature extraction must remain stopped until rebuilt
manifest hashes and the feature protocol are recommitted.

The corrected strict-JSON manifests retain all 492 rows and record 103 rows
with 206 zero-filled components: 99 auxiliary and four development rows, in
every case both `u` and `v`. Their final hashes are the ones reported above;
they supersede the pre-diagnostic manifests that contained nonstandard JSON
`NaN` values. No feature cache existed from the failed pass.

The recommitted protocol then completed model-aligned extraction on the RTX
5070. The physically separate auxiliary cache contains 367 unique negative
rows by 108 finite features across 362 groups (SHA-256
`9d1f5116f10088a953aea66233e5896ef06be8252f3565f5b4b1d768d8e2ac0b`);
the development cache contains 125 unique negative rows across 124 groups
(SHA-256
`a7f5c4100ecd2d4172bccd550f6a17736e4008e551f8bce2a935ce3caed6dcb4`).
Both caches have label sum zero and no non-finite value. Extraction used the
hash-pinned released checkpoint plus frozen alpha-0.5 residual representation,
performed no fitting or selection, and accessed neither the MARS paper test nor
the 374-row published CloudSEN12+ test partition.

The frozen spatial-negative XGBoost experiment selected auxiliary weight 1.0
and complement blend 0.20. Cross-fitted folds 2/3/4 pass: AP improves
+0.002286, matched-FPR recall +0.005170 (12 additional positives), both sensor
AP strata improve, and the paired 403-group AP interval is
[+0.000233, +0.005250]. Each selection fold has positive AP and nonnegative
recall change.

The frozen setting then reached both independent original confirmation folds.
Fold 1 passes with AP +0.00295, recall +0.00537, and paired AP interval
[+0.00079, +0.00692]. Fold 0 improves AP +0.00337 but recall is exactly tied
and its paired AP interval is [-0.00025, +0.00711], failing both strict
confirmation requirements. The branch is therefore rejected before opening
the 125-row CloudSEN development cache, before any paper-cache replay, and
without writing an artifact. The direction is useful evidence that localized
source-disjoint negatives transfer, but it is not yet robust enough alone.

The next model is frozen as a joint external-data head before fitting. It adds
both the 135 source-disjoint UNEP positives and 367 source-disjoint CloudSEN
negatives to the same regularized depth-3 XGBoost fit. The candidate family is
deliberately four settings: positive per-row weight 1.0 or 367/135=2.7185,
negative weight 1.0, and complement blend 0.10 or 0.20. Weight 1.0 is the
independently selected value in both one-domain experiments; 2.7185 equalizes
total external positive and negative sample mass. Selection is cross-fitted on
folds 2/3/4 and ranks passing candidates by worst-fold AP first. Both original
confirmation folds must have AP delta at least 0.002, strictly higher recall,
and positive paired-group AP lower bounds. Only afterward may the four UNEP
development positives and 125 CloudSEN development negatives be opened, and
both must pass their independent recall/false-positive gates before paper
replay. Dense masks remain unchanged.

The joint experiment selects class-mass-balanced positive weight 2.7185,
negative weight 1.0, and blend 0.20. It passes every original-development
gate. Cross-fitted folds 2/3/4 improve AP +0.003448 and matched-FPR recall
+0.003016, with paired-group interval [+0.000936, +0.006275] and improvements
in both sensor strata. Fold 0 improves AP +0.003958 and recall +0.001342, with
interval [+0.000547, +0.007880]; fold 1 improves AP +0.003287 and recall
+0.006711, with interval [+0.000742, +0.007871]. This is the first new-data
scene head to pass the pooled selection and both strict original confirmations.

Only then were both external development caches opened. UNEP positive recall
ties current at 3/4 using separate frozen current/candidate thresholds. On 125
CloudSEN negatives, current and candidate both produce exactly one false
positive, but the candidate p95 probability margin to threshold is -0.22081
versus current -0.26096. The predeclared margin-noninferiority gate therefore
fails. No artifact is written and the paper cache remains unopened. Because
the CloudSEN development partition is now observed, subsequent negative-weight
calibration must label it as reused development evidence and reserve the still
sealed 374-row CloudSEN published test partition for final external safety.

A narrow follow-up is frozen before fitting. It retains the jointly successful
positive weight 2.7185 and blend 0.20, varies only CloudSEN auxiliary-negative
weight 2.0 versus 4.0, and selects by the same original folds with worst-fold
AP priority. The opened UNEP/CloudSEN development rows remain post-selection
gates but are explicitly labeled reused tuning evidence. The exact paper cache
and 374-row published CloudSEN test remain unopened; the latter is now the only
eligible fresh external-negative confirmation.

The stronger-negative run selects weight 4.0 and again passes every original
gate: folds-2/3/4 AP/recall deltas are +0.003632/+0.003447 with interval
[+0.000946, +0.006739]; fold 0 is +0.003738/+0.001342 with interval
[+0.000214, +0.007394]; fold 1 is +0.003815/+0.004027 with interval
[+0.001093, +0.008623]. UNEP development recall remains 3/4. CloudSEN
development again ties false positives at 1/125 but fails the frozen raw
probability-subtraction margin (-0.22302 versus -0.26096), so this run is also
rejected as specified, with no artifact or paper access.

A reporting-only scale audit shows why stronger negative weighting did not
move that gate despite suppressing negatives. Candidate p95 negative score is
0.01129 versus current 0.02091, and its p95 log-odds distance from the separate
operating threshold is -3.28835 versus current -2.91114; both are safer. The
failed quantity subtracts probabilities from thresholds on differently
calibrated scales (candidate threshold 0.23431 versus current 0.28188), which
reverses the safety ordering. This diagnostic does not retroactively pass the
experiment. A new fixed-model safety contract may use false-positive count,
raw negative-score quantiles, and log-odds threshold distance, but must freeze
before refitting and must reserve the 374-row CloudSEN test for fresh evidence.

The calibration-aware finalization contract is frozen around exactly the
weight-4 candidate and final-fit seed 20260917. It hash-binds the rejected
source report and does not rerun or reinterpret its original-fold gates. The
reused development safety checks are fixed to positive recall, false-positive
count, raw negative-score p95, and log-odds distance from each model's own
frozen threshold. Passing may create an ignored artifact and authorize fresh
CloudSEN-test acquisition; it does not itself authorize or inspect paper data.

Fixed-candidate finalization passes. UNEP development recall remains 3/4.
CloudSEN development keeps false positives at 1/125 while lowering raw-score
p95 from 0.02091 to 0.01129 and making logit distance from threshold more
negative, -3.28937 versus -2.91287. The ignored fixed artifact is 129,486
bytes with SHA-256
`e1305d4f1199891b8f2f2cae27ee27219af819de72f42f95e329ffd3fa6c15b1`
and operational threshold 0.2343112561. This authorizes acquisition and
evaluation of the still-sealed 374-row CloudSEN published test as fresh
external-negative evidence. It also makes the fixed head eligible for a
separately frozen exact-paper evaluator, but paper access should follow the
CloudSEN test safety check.

Before fresh-test row materialization, a dedicated CloudSEN protocol freezes
the expected 374 published test identities, all-clear/no-plume truth contract,
producer-grid requirements, exact-product/no-substitution rule, fixed artifact
hash, operational threshold, and safety metrics. Missing exact products must be
reported and receive an adversarial candidate-false-positive bound. The exact
MARS paper cache remains outside this workflow.
### 2026-07-15: Fresh CloudSEN test metadata contract correction

The first post-freeze cohort materialization stopped before imagery access because the
published test split is not uniformly all-clear. All 374 identities are no-plume
Sentinel-2 scenes, but 42 contain published non-clear pixels (including one fully
non-clear scene): 14,835,412 clear and 124,588 non-clear pixels in total. The frozen
protocol was corrected to assert this exact composition. No identity was excluded,
the fixed model was not changed, and the exact MARS paper cache remained unopened.
This makes the safety test harder and more representative of operational negatives.
### 2026-07-16: Fresh CloudSEN test identities materialized

After the fixed joint scene artifact was finalized, the corrected frozen protocol
materialized all 374 published CloudSEN12+ test identities (361 ROI groups across
55 countries; cohort SHA-256 `be085ac1e934aede89d3a3a9b69a4915d0d15d8097fe4401fe8e89918f2e76a4`).
Five rows have missing published wind components and will use the already frozen,
explicit zero-fill policy. Exact-product resolution had not yet begun and the exact
MARS paper cache remained unopened.
### 2026-07-16: Fresh CloudSEN exact products resolved

Exact published target/reference identities resolved for 368 of 374 fresh-test
scenes (resolver manifest SHA-256
`120cfac9d364c8c75ab44687307645ca5de353a24a7a4b18192f1c93cd948988`).
Six scenes remain unavailable without substitution: one lacks a published reference
identity, four have unavailable exact references, and five have unavailable exact
targets (some scenes miss both sides). There were zero query errors. These six stay
in the declared full-cohort adversarial false-positive bound.
### 2026-07-16: Fresh CloudSEN exact crops acquired

All 368 available exact target/reference pairs were acquired and independently
reverified on their published 200x200, 10 m grids. Every no-plume crop passed the
pre-cloud radiometry/geometry gate and there were zero errors. The 235,585,490 bytes
of rasters remain ignored. Public JP2 data reproduce the audited native-10 m band
statistics; B11/B12 follow the released conversion algorithm but cannot be claimed
bitwise-identical to the unpublished producer Earth Engine TIFFs. The paper cache
remained unopened.
### 2026-07-16: Fresh-test loader and cloud proxy frozen

Before feature extraction, the loader contract was frozen for all 368 acquired rows.
The producer's spatial CloudSEN12+ mask TIFFs are unavailable; therefore both frozen
scene heads receive the same exact-grid all-zero cloud proxy. This proxy is explicitly
not claimed as published cloud truth. Hash-bound aggregate statistics preserve a
separate all-clear (327 rows) versus non-clear (41 rows) stratification, with 84,588
published non-clear pixels among available scenes. The six unavailable identities
remain outside the loader manifest and inside the adversarial full-cohort bound.
### 2026-07-16: Fresh-test loader manifest materialized

The frozen loader builder emitted 368 unique rows across 355 groups (manifest
SHA-256 `c6cabb8b21c9573b5de9386327ec1800faab11225c8819e1fc3c5cad35d27d9a`).
It reconciled all 41 available non-clear rows and all 84,588 published non-clear
pixels, explicitly zero-filled both missing wind components in five rows, and kept
all six unavailable identities in the receipt. No scoring or paper-test access
occurred.
### 2026-07-16: Fresh-test feature protocol frozen

The one-shot feature extraction protocol now binds the 368-row loader manifest,
released checkpoint, fold-0 residual representation, 108-feature schema, and output
path by hash. Published cloud-composition fields are carried only for prespecified
stratification and are prohibited model features. Neither scene head can be refit or
selected during extraction, and the exact paper cache remains sealed.
### 2026-07-16: Fresh-test features extracted

The RTX 5070 extraction completed all 368 rows/355 groups with 108 finite frozen
features per row (cache SHA-256
`4104ed4bbc1c2289943d1b76f251b3a8739d946cb67e09fce1f8301baf9afe3c`).
The cache retains all-clear/non-clear metadata only as evaluation strata. No model
fit, threshold change, paper score, or paper label was accessed.
### 2026-07-16: One-shot fresh-scene safety evaluation frozen

Before reading any fresh scene score, the comparison froze both artifacts and
operational thresholds, all 368 feature rows, three published-cloud strata, score
quantiles, logit-to-threshold margins, false-positive gates, and symmetric
adversarial bounds for six unavailable identities. Passing requires no higher false
positive count in the full available, all-clear, and non-clear strata; no higher
overall raw p95; no higher p95 logit margin in all three strata; and no worse
symmetric full-cohort bound. Failure blocks paper-cache access.
### 2026-07-16: Fixed candidate passes fresh no-plume safety test

All predeclared gates passed. On 368 available fresh scenes, the current head made
4 false positives (1.08696%) and the fixed candidate made 2 (0.54348%); both
candidate errors were shared with the current head, so there were zero candidate-only
false positives. Both heads made 0/41 errors in the published non-clear stratum.
The candidate also lowered overall raw p95 (0.02231 to 0.01392) and p95
logit-to-threshold margin (-2.84564 to -3.07652). Counting all six unavailable rows
as errors for both heads gives symmetric worst-case full-cohort FPR bounds of
2.67380% current versus 2.13904% candidate. This authorizes exact paper replay.
### 2026-07-16: Fixed joint exact-paper replay frozen

After fresh-safety authorization, the exact-paper replay was frozen against the
official [MARS-S2L v3 paper](https://arxiv.org/html/2511.21777v3) and the hash-bound
benchmark receipt. The evaluator must assert that its live baseline exactly matches
the reconstructed 43,529-row full and 15,655-row test-only-site AP/recall/FPR/IoU
values, which are slightly stronger than the rounded published tables. Candidate
scores are computed before outcome-cache loading. Both views require strictly
positive paired site-bootstrap lower bounds for AP, matched-FPR recall, and IoU,
plus a nonpositive upper bound for fixed-FPR delta. No partial pass is promotable.
### 2026-07-16: Fixed joint scene head rejected on exact paper replay

The evaluator exactly reproduced the official v3 comparator and the candidate passed
every full-view gate: AP 0.67416079 versus 0.64101960 (paired lower delta +0.01298),
matched-FPR recall delta +0.03365 (lower +0.01973), FPR delta -0.03915, and IoU
0.37996356 versus 0.32436515 (lower delta +0.03506). It failed the required
test-only-site scene gates: AP 0.44810937 versus 0.45027380 (delta -0.00216,
lower -0.04445) and matched-FPR recall delta +0.01762 with lower -0.01081.
Test-only IoU remains decisive at 0.29246225 versus 0.17156194 (lower delta
+0.08597), and FPR improves by -0.04226. Because both views must pass, the joint
scene head is rejected for final promotion; the dense-mask result remains valid.
### 2026-07-16: Temporal site-prior development experiment frozen

Repeated exact-paper failures diagnose unseen-site transfer, while simple temporal
site evidence remains label-free at inference. A new candidate family therefore
adds each physical site's top-k mean current-scene logit to every scene logit. The
18 top-k/weight combinations, selection folds 2/3/4, independent confirmation folds
0/1, paired site-bootstrap gates, recall tolerances, rank rule, and artifact policy
were frozen before development scoring. The family is transparently motivated by
post-test diagnosis, but its parameter selection and confirmation use only authorized
development folds; this experiment does not load the paper cache.
### 2026-07-16: Deterministic temporal site prior rejected

The frozen search selected top-1/weight-0.1, but it failed the preregistered evidence
gates and wrote no artifact. Selection AP improved only +0.00044 with paired lower
-0.00162, and fold 4 regressed -0.00119. Combined fold-0/1 confirmation improved AP
+0.00093 and recall +0.00201, but its AP lower bound was -0.00020. The direction is
promising but not sufficiently stable. Post-failure cohort analysis clarifies the
domain shift: development sites average 57-86 scenes/fold and roughly 31% contain a
positive, while the exact test-only cohort averages 22.5 scenes/site and only 59/697
(8.5%) sites contain a positive. A learned site-level risk model with one equal-weight
row per physical site is the next research path.
### 2026-07-16: Cross-site risk-prior experiment frozen

The next architecture addresses the measured site-prevalence shift directly. Each
physical site contributes one equally weighted classifier row built from 32
label-free temporal aggregates of the frozen current and primary scene scores.
Four small site models and five scene-logit weights (20 candidates) are frozen.
Selection uses nested site-held-out predictions on folds 2/3/4; the selected model
is then fit on those folds and confirmed once on folds 0/1. Both stages require
positive paired site-bootstrap AP evidence, improved recall, and per-fold stability.
Coordinates, country, site identity, and all inference-time labels are prohibited;
the exact paper cache is not loaded.
### 2026-07-16: Learned cross-site risk prior rejected

The 20-candidate experiment selected ExtraTrees depth-3/weight-0.1 but failed its
frozen gates and wrote no artifact. Selection AP delta was +0.00025 with paired lower
-0.00054; confirmation delta was +0.00034 with lower -0.00058, and fold 0 regressed
-0.00038. Equal site weighting alone does not resolve the transfer gap. A constrained
post-failure diagnostic found that temporal priors have opposite behavior by history
length: they hurt sparse sites but materially improve sites with roughly 30 or more
observations. The next candidate family will freeze history-length routing and select
its cutoff/top-k/weight only on authorized development folds.
### 2026-07-16: Large-history temporal routing frozen

The history-length hypothesis is now a formal development-only experiment. Forty-five
minimum-history/top-k/weight combinations are frozen. A candidate leaves sparse sites
mathematically unchanged and adds top-k temporal evidence only at sites meeting its
minimum scene count. Selection remains folds 2/3/4 with positive paired-bootstrap AP
evidence and per-fold stability; the fixed winner must repeat those properties on
folds 0/1. No paper or fresh-test cache is loaded during parameter selection.
### 2026-07-16: Large-history temporal routing rejected

The 45-route search selected minimum history 20, top-1, weight 0.25, but failed
selection and confirmation gates and wrote no artifact. Folds 2/3 improved AP by
+0.00485/+0.00562, while fold 4 regressed -0.00436; the combined selection AP
delta was effectively zero with lower -0.00515. Confirmation AP improved +0.00110
but recall changed -0.00067 and its AP lower bound was -0.00177. History length
does not make a bidirectional prior stable. The next rule is one-sided: it may only
suppress scenes at sites whose top-k evidence is below a confidence cutoff, so it
cannot create new high-score false positives at a fixed threshold.
### 2026-07-16: One-sided temporal suppression frozen

The new candidate family is mathematically conservative: it can only lower a scene
score. For each site, a penalty applies when its top-k current logits remain below
a confidence cutoff; sites above the cutoff are unchanged. Thirty-six top-k/cutoff/
weight candidates and all evidence gates are frozen. Selection uses folds 2/3/4,
confirmation uses folds 0/1, and every candidate is runtime-asserted never to raise
any score. The paper and fresh-test caches remain out of scope.

### 2026-07-16: One-sided temporal suppression rejected

The frozen search selected top-1, confidence cutoff 0.90, and penalty weight 0.50.
It passed the selection-stage direction checks with AP delta +0.000526 and paired
site-bootstrap lower bound +0.000026, and every selection fold improved. It did not
replicate on independent folds 0/1: combined confirmation AP changed -0.000053
with lower bound -0.000844, including fold-0 delta -0.000495 (fold 1 improved
+0.000316). Confirmation recall improved +0.002013, but the AP failure is decisive.
No artifact was written. The next architecture will route temporal evidence only
for sites absent from the training cohort and will select parameters on whole-site
low-prevalence development strata that more closely represent the declared unseen-
site target domain; prevalence labels define evaluation strata only, never routing.

### 2026-07-16: Unseen low-prevalence temporal router frozen

The next experiment formalizes the transfer target without using target labels at
inference. Entire development sites with scene-positive rate at most 5% define the
selection and confirmation evaluation strata; this definition is evaluation-only.
The deployed route uses only whether a physical site appears in the frozen training
set, its group identity, and current scene scores. Known training sites remain
exactly unchanged. Forty-five minimum-history/top-k/weight candidates, selection
folds 2/3/4, confirmation folds 0/1, 10,000-replicate paired site bootstraps, and
whole-fold safeguards are frozen before scoring. The paper cache remains prohibited.

### 2026-07-16: Unseen low-prevalence temporal router rejected

The frozen search selected minimum history 20, top-1, weight 0.25. It was strong
on selection folds 2/3/4: low-prevalence AP improved +0.017486 with paired site-
bootstrap interval [+0.003218, +0.036668]. The fixed route did not replicate on
folds 0/1. Combined low-prevalence AP improved only +0.000203 with interval
[-0.013497, +0.013972]; fold 0 improved +0.003467 while fold 1 regressed
-0.005999. Recall was unchanged. Whole-fold AP changed +0.001100 and recall
-0.000671, within safeguards, but the primary confirmation interval failure is
decisive. No artifact was written, and neither fresh safety nor paper replay is
authorized for this route.

### 2026-07-16: Permutation-equivariant site modeling research path

The next path builds on the already validated site-relative spatial classifier,
which improved every development fold and raised exact test-only point AP to
0.46712 but lacked a positive bootstrap lower bound. Physical-site observations
form unordered sets. Deep Sets establishes the sum-decomposition family for
permutation-invariant/equivariant functions
(https://proceedings.neurips.cc/paper/2017/hash/f22e4747da1aa27e363d86d40ff442fe-Abstract.html),
while attention-based multiple-instance learning provides a learned, interpretable
way to emphasize rare informative instances within a bag
(https://proceedings.mlr.press/v80/ilse18a.html). The planned meta-head therefore
combines the stronger site-relative spatial scene score with label-free set context,
using site-held-out development evaluation. First, a hash-bound builder will
reproduce all leakage-controlled site-relative development scores and assert the
published internal metrics before any new model selection.

### 2026-07-16: Exact site-relative OOF reproduction rejected; deterministic rebuild frozen

The hash-bound reproduction attempt completed all three inner fits but rejected its
output: rebuilt combined inner AP delta differed from the old report by 0.00106,
far above the frozen 1e-7 tolerance. Inputs, architecture, code revision, and seeds
matched. The historical trainer seeded CUDA but did not require deterministic
kernels, so seeded training alone is insufficient for bitwise research provenance.
No score cache or success report was written. A separately declared v2 rebuild is
now frozen with deterministic PyTorch/cuDNN algorithms. It does not claim to recreate
the old stochastic OOF scores. Instead, it requires positive AP delta in every inner
fold, bounded recall changes, limited AP drift, and tight reproduction of fold-0/1
metrics produced by the already frozen artifact. This distinction will be disclosed
in the methods and reproducibility limitations.

### 2026-07-16: Deterministic site-relative score rebuild rejected

The deterministic v2 rebuild improved combined inner AP by +0.003080 and matched-
FPR recall by +0.003016; all three inner folds improved AP. Frozen-artifact inference
also exactly reproduced fold-0 AP/recall deltas (+0.001304/+0.002685) and fold-1
deltas (+0.001562/+0.004027). Nevertheless, at least one inner fold exceeded the
preregistered -0.002 recall tolerance. The wrapper deleted the 1.75 MB cache and
raised failure, so these scores are not inputs to subsequent selection. Rather than
relax a seen gate, the next model will learn a permutation-equivariant site context
directly from the existing deterministic 108-feature OOF caches and frozen current
scores. This removes the stochastic spatial-crossfit dependency entirely.

### 2026-07-16: Low-prevalence set-context head frozen

The literature-motivated site-set path is now a concrete, development-only
experiment. The model consumes the deterministic 108 scene features, existing
label-free site-context features, and nine current-score set statistics including
leave-one-out maximum and within-site rank. Two strongly regularized shallow
gradient-boosted models and five current-score blends (10 candidates) are frozen.
Selection predictions are cross-fitted across folds 2/3/4. The fixed winner is then
fit on low-prevalence sites in those folds and confirmed once on folds 0/1. The
primary strata require positive paired-site AP lower bounds and nondecreasing recall;
whole-fold and per-fold safeguards are also required. The 5% site-positive threshold
defines supervised development domains only and is absent from inference. Known
training sites remain on current v3; the head routes only unseen sites if promoted.

### 2026-07-16: Low-prevalence set-context head rejected

The frozen search selected the depth-2 model with a 0.10 blend. Every selection
fold improved low-prevalence AP (+0.006362, +0.007537, +0.006756), but combined
selection AP delta was +0.005471 with paired interval [-0.003393, +0.014600], and
pooled matched-FPR recall changed -0.005556. Independent confirmation improved
combined low-prevalence AP +0.003820 with interval [-0.001282, +0.009054] and
unchanged recall; fold 0 AP regressed -0.000638 while fold 1 improved +0.000561.
Whole-fold confirmation AP was effectively unchanged (+0.000074) and recall changed
-0.000671. The inferential failures are decisive, no artifact was written, and the
paper cache remains unauthorized for this candidate.

### 2026-07-16: Site-relative deterministic feature-cache build frozen

The earlier deterministic rebuild incorrectly coupled feature acquisition to model-
promotion recall gates. A new v3 contract cleanly separates those roles. It accepts
or rejects the spatial OOF cache only on leakage control, deterministic execution,
alignment, tight frozen-artifact fold-0/1 reproduction, and broad drift detection.
All predictive deltas are diagnostics, not acceptance criteria. Downstream temporal-
ensemble selection and independent confirmation will carry the actual AP/recall
promotion gates. This protocol was frozen before rebuilding the cache.

### 2026-07-16: Site-relative deterministic feature cache built

The correctness-only build passed and wrote an ignored 1,748,406-byte cache with
SHA-256 `186ab5946790798a4a81274130ba5756ab3e5316b7b42238c2772d36805152a1`.
The deterministic inner spatial score improved AP +0.003080 and recall +0.003016
versus current; frozen-artifact fold-0/1 deltas reproduced exactly. These numbers
remain diagnostic. The cache is now authorized only as a leakage-controlled input
to a separately frozen temporal-spatial ensemble experiment.

### 2026-07-16: Temporal-spatial unseen-site ensemble frozen

The strongest validated spatial component and strongest observed unseen-site
mechanism are now combined in a leakage-controlled development experiment. Twenty-
four minimum-history/top-k/weight routes include a zero-temporal control. Each held
development fold is unseen to its deterministic spatial model. Selection uses folds
2/3/4; the fixed route is confirmed once on folds 0/1. Both the <=5% site-positive
target stratum and the complete view require positive AP point deltas, positive
10,000-replicate paired-site bootstrap lower bounds, and nondecreasing matched-FPR
recall, with per-fold safeguards. If promoted, the spatial component scores every
site; temporal corroboration applies only to groups absent from the frozen training-
site list. No inference-time labels or paper-cache data enter this experiment.

### 2026-07-16: Temporal-spatial unseen-site ensemble rejected

The frozen search selected minimum history 20, top-1, weight 0.50. Selection target-
stratum AP improved +0.026561 with paired interval [+0.004207, +0.052496], but
recall changed -0.005556; the natural whole-view AP delta was only +0.000361 with
interval [-0.010983, +0.014434]. Independent confirmation failed decisively:
target AP changed -0.003325 with interval [-0.027510, +0.022416], driven by fold 1
(-0.017670; fold 0 was +0.006735). Whole confirmation AP changed +0.000412 with
interval [-0.004948, +0.005988], and fold-0 recall regressed -0.006711. No artifact
was written. The repeated split-dependent behavior shows that selecting individual
sites by <=5% scene-positive rate is not an adequate transport model for the paper's
rare-site mixture; it conflates site-level rarity with within-positive-site plume
persistence.

### 2026-07-16: Target-mixture transport simulation frozen

The next experiment separates site-level rarity from plume persistence. Development
folds contain 19%-45% positive sites, versus approximately 8.5% in the exact test-
only diagnostic. Each simulation replicate therefore samples, per fold, 4 positive
and 46 negative physical sites without replacement (8%), then uniformly caps each
site at 23 scenes, close to the target's 22.5-scene mean. Site class controls
evaluation sampling only; within-site scene sampling is label-blind, and no such
label enters inference. Twenty-four temporal-spatial routes are frozen. Two hundred
common-random-number replicates select on folds 2/3/4; the fixed route receives
1,000 independent replicates on folds 0/1. Promotion requires a positive 2.5th
percentile for simulated AP delta, positive per-fold AP medians, nonnegative combined
recall median, and natural-cohort noninferiority. The architecture-family motivation
is transparently post-test, but all parameter selection uses development labels only.

### 2026-07-16: Target-mixture v1 protocol rejected; v2 metric rule frozen

The v1 run was stopped during selection because uniformly sampling 23 scenes from
a site known to be positive over its full history can yield a fold-level replicate
with no sampled positive scene. Scikit-learn correctly warned that AP/recall were
undefined, but v1 attempted to rank those values. No report or artifact was written,
and partial candidate outputs are discarded. V2 preserves the exact site/scene
sampling, candidates, seeds, and gates. It now excludes an undefined replicate only
from that fold's metric distribution, requires every combined replicate to contain
both classes, reports valid fold-replicate coverage, and requires at least 50%
coverage in every fold. This analysis rule was frozen before restarting.

### 2026-07-16: Target-mixture transport simulation rejected

V2 selected minimum history 20, top-1, weight 0.10. Across 200 selection
simulations, median AP delta was +0.011099 but its interval was
[-0.006482, +0.040466]; all three fold medians were positive and natural-cohort AP/
recall improved +0.003526/+0.002585. The fixed route did not obtain inferential
confirmation: across 1,000 fold-0/1 simulations, median AP delta was +0.002545 with
interval [-0.010218, +0.017912], and fold medians were effectively zero. Natural
confirmation AP/recall improved +0.002347/+0.002013. Valid fold-metric coverage was
99.5%-100%, so missing-class handling is not the explanation. No artifact was
written. Prevalence/history standardization alone cannot make an unconstrained site
offset reliable. The next temporal mechanism will be one-sided and attention-like:
only high-confidence large sites may receive a boost, while every other scene remains
exactly on the validated spatial score.

### 2026-07-16: Gated temporal-spatial boost frozen

The next candidate is an attention-like one-sided rule. Only sites meeting a minimum
history can change, and only when their top-k spatial evidence exceeds a confidence
cutoff. The excess evidence raises every site scene logit; it can never suppress a
scene. Sparse and below-cutoff sites remain bitwise on the validated spatial score.
Forty-eight minimum-history/top-k/cutoff/weight combinations, selection folds 2/3/4,
confirmation folds 0/1, and paired-site gates on both low-prevalence and complete
views are frozen. The rule uses no labels at inference, and the paper cache remains
out of scope.

### 2026-07-16: Gated temporal-spatial boost rejected

The search selected minimum history 20, top-1, cutoff 0.10, weight 1.0. Selection
target AP improved +0.027425 but its paired interval crossed zero
[-0.002733, +0.061129], recall changed -0.005556, and the whole-view AP regressed
-0.013863. Independent confirmation was uniformly adverse: target AP/recall changed
-0.024858/-0.020979, while whole AP/recall changed -0.011317/-0.008725; the whole-
view AP interval was entirely negative [-0.023321, -0.000113]. No artifact was
written. Both bidirectional and one-sided temporal families are now retired. The
next path is a non-temporal ensemble of two independently development-validated,
test-complementary representations: site-relative spatial residuals and adaptive
Prithvi foundation features.

### 2026-07-16: Adaptive Prithvi development-score reproduction frozen

The selected adaptive Prithvi control (`cls_plus_base`, C=0.01, current-score blend
0.10) previously passed its five-fold development gates and independently reached
test-only point AP 0.46550. A hash-bound builder is frozen to reproduce exactly that
OOF vector from the 44,363-row Prithvi feature cache, align it by sample identity,
and assert its original AP/recall deltas within 1e-10. No model or blend search is
reopened, and the paper cache remains out of scope.

### 2026-07-16: Adaptive Prithvi development scores reproduced

All five OOF probes completed and reproduced the frozen AP/recall deltas exactly:
+0.00434163/+0.00131199 versus current. The ignored 2,082,539-byte score cache has
SHA-256 `672ab43155280de950ac529f197b5084bb0b0ab76f17361b60ac99fc65ba1208`
and is aligned by 44,363 unique sample identities. It is authorized only as an input
to a separately frozen representation ensemble.

### 2026-07-16: Spatial-Prithvi representation ensemble frozen

Two non-temporal components with independent positive development evidence are now
combined: the site-relative spatial residual CNN and adaptive Prithvi-EO-2.0 probe.
Their OOF score caches reproduce the original component metrics and align by sample
identity. Five logit-space Prithvi weights are frozen. Selection uses folds 2/3/4;
the fixed weight is confirmed once on folds 0/1. Both stages require positive paired-
site AP lower bounds, nondecreasing matched-FPR recall, per-fold stability, and sensor
noninferiority. No paper data are loaded during this experiment.

### 2026-07-16: Spatial-Prithvi representation ensemble passes development

All frozen gates passed. The selected logit ensemble assigns 0.75 weight to adaptive
Prithvi and 0.25 to site-relative spatial scores. Selection AP/recall improved
+0.004911/+0.002585 with paired-site AP interval [+0.002437, +0.007277]; every fold
and both sensors improved. Independent fold-0/1 confirmation improved AP/recall
+0.002248/+0.000671 with interval [+0.001135, +0.003462]. Fold AP deltas were
+0.002173 and +0.001480, and both sensor deltas were positive. The ignored 408-byte
ensemble artifact has SHA-256
`0a3be48f5eda659aaaa7cf91c259acd230f81340677e5a6fc5fab83ba048420b`.
This authorizes a newly frozen fresh CloudSEN representation extraction and safety
test; exact paper replay remains blocked until that safety gate passes.

### 2026-07-16: Fresh spatial-Prithvi extraction frozen

The passing ensemble requires representations not present in the original 108-
feature fresh cache. A no-fit extractor is frozen against the previously acquired
368 exact CloudSEN products, residual/released checkpoints, and pinned Prithvi-EO-2.0
receipt. It will compute a 9x64x64 spatial tensor and 768 CLS features for the same
identity-ordered rows in one pass. The six unavailable identities remain in the
adversarial bound. No safety metric or paper data are accessed during extraction.

The v1 extractor exited at module import because `CLS_WIDTH` was imported from the
feature extractor rather than defined locally. No model or scene was loaded and no
output was written. V2 fixes the constant at its frozen 768-feature contract and is
refrozen under a new script hash before inference.

The corrected extraction completed all 368 rows with 355 groups. The ignored
19,163,578-byte representation cache has SHA-256
`a1f46a7dd9c4d579ad141122b8997aeb1713f27bd3c9a08a53a9ce8f7e9f08fa`;
its shapes are exactly 368x9x64x64 spatial and 368x768 Prithvi. No paper cache or
labels beyond the fixed all-negative contract were accessed.

### 2026-07-16: Spatial-Prithvi fresh safety evaluation frozen

The one-shot evaluator is frozen against both fresh representation caches, all
component/ensemble artifacts, adaptive source features, the conservative 0.281876
threshold for both heads, published clear/non-clear strata, and the six-row
adversarial bound. The adaptive Prithvi source probe may be refit under its already
selected C/blend and use unlabeled fresh feature moments; no fresh label, calibration,
or selection enters scoring. The same false-positive-count, raw-p95, logit-margin,
and symmetric unavailable-row gates used for the prior fresh test are required.

### 2026-07-16: Uncalibrated spatial-Prithvi fresh safety test rejected

The ensemble produced exactly the same four false positives as current on 368 rows,
with four shared errors, zero candidate-only errors, and zero errors in the 41-row
published non-clear stratum. The symmetric unavailable-row bound also passed. It
failed every preregistered p95 distribution gate: overall candidate p95 was 0.023892
versus current 0.022308, all-clear 0.026156 versus 0.023027, and non-clear 0.018496
versus 0.017337. Thus this is a calibration rejection, not an error-count rejection;
exact paper replay remains blocked. A monotone logit-offset calibration will be fit
only from development negatives. Such an offset preserves every scene ranking, AP,
and matched-FPR recall exactly while reducing fixed-threshold scores.

### 2026-07-16: Development-only spatial-Prithvi calibration frozen

The calibration experiment is checksum-bound before execution. For each selection
fold (2/3/4), it computes the current model's negative-scene logit p95 minus the raw
ensemble's negative-scene logit p95. The single offset is the most conservative
(minimum) fold difference minus a fixed 0.05-logit margin. It must be negative and
strictly lower every score. Selection and one-shot fold-0/1 confirmation each
require every fold's negative logit p95 and false-positive count at the fixed
0.281876 threshold to be no higher than current, while raw-ensemble AP is preserved
within 1e-12. The calibrator cannot load fresh CloudSEN or paper-cache inputs. If it
passes, the resulting artifact authorizes a transparently post-test replay of the
same fresh safety cohort; that replay cannot be described as a new untouched test.

### 2026-07-16: P95-derived spatial-Prithvi calibration rejected

The frozen rule produced a -0.052065-logit offset. It preserved AP exactly and made
negative p95 lower than current in all five folds. Confirmation folds also reduced
fixed-threshold false positives (343 vs 357 and 544 vs 574). However, selection fold
2 retained 226 false positives versus current's 224, so the predeclared all-fold
gate failed even though folds 3/4 improved (246 vs 249 and 514 vs 546). No artifact
was written and the fresh cohort was not replayed. A single p95 statistic does not
guarantee fixed-threshold tail control; any replacement calibration must derive its
tail constraint from development data and receive a new frozen confirmation test.

### 2026-07-16: Exact-tail spatial-Prithvi calibration frozen

The replacement monotone calibration controls both the negative logit p95 and the
exact fixed-threshold tail. Within each selection fold, the candidate score at the
first rank beyond current's false-positive budget defines an upper offset limit.
The frozen offset is the minimum of all p95 and exact-tail limits over folds 2/3/4,
minus the same fixed 0.05-logit margin. This is a deterministic constraint, not an
offset search. It loads no fresh or paper data and preserves every ranking metric.
Because folds 0/1 were exposed by the p95-only attempt, their evaluation is now
explicitly a reused holdout audit rather than independent confirmation. Passing
development permits only a transparently post-test fresh safety replay.

### 2026-07-16: Exact-tail spatial-Prithvi calibration passes development

The binding selection constraint was fold 2's first disallowed negative, yielding
an offset limit of -0.055610 and a final conservative offset of -0.105610 logits.
All selection gates passed: calibrated false-positive counts were 215/242/499 versus
current's 224/249/546 on folds 2/3/4, every negative p95 was lower, and AP was
bitwise unchanged. The reused fold-0/1 audit also passed (334/531 false positives
versus 357/574), but is not independent evidence. The ignored 454-byte calibration
artifact has SHA-256
`fbb69fc987220cda88e04532f74bf610ab4eaecb2cb9d18594a985851af7387b`.
This authorizes a post-test fresh replay under the already frozen safety gates.

### 2026-07-16: Tail-calibrated fresh replay frozen

A separate evaluator is checksum-bound to the passing -0.105610-logit artifact and
all original fresh inputs. It rescales the raw ensemble only after both components
are reproduced, then applies the exact same 0.281876 thresholds, clear/non-clear
strata, false-positive counts, p95 comparisons, logit-margin checks, and symmetric
six-row unavailable bound. No gate has been relaxed. The report must state that the
cohort was previously inspected and therefore supplies a post-test safety audit,
not independent external confirmation. The exact paper cache remains out of scope.

### 2026-07-16: Tail-calibrated fresh replay rejected

The -0.105610-logit candidate retained exactly the same four false positives as
current, with zero candidate-only errors and zero errors in the 41 non-clear rows.
Overall p95 improved to 0.021549 versus 0.022308, and the available/non-clear logit-
margin gates passed. The sole failure was the published all-clear logit-margin p95:
-2.787786 versus current's -2.812609, a 0.024823-logit excess. No paper input was
opened. Because this is already a post-test cohort, that gap cannot select another
numeric margin. The next development rule will instead require full negative
empirical-CDF dominance in every selection fold, a stronger label-controlled
calibration property that simultaneously constrains every observed negative
quantile and fixed threshold.

### 2026-07-16: Negative empirical-CDF calibration frozen

The stronger calibration is frozen without using the observed 0.024823 fresh gap.
For each selection fold, current and raw-ensemble negative scores are independently
sorted. Every aligned order statistic supplies an offset limit; the fold limit is
the minimum over all ranks, and the global offset is the minimum fold limit minus a
fixed 0.05 logits. This enforces first-order empirical-CDF dominance for the entire
observed development-negative distribution, hence every empirical quantile and
fixed-threshold exceedance count, while leaving all rankings unchanged. Folds 0/1
remain a reused audit. No fresh or paper input is accessible to the calibrator.

### 2026-07-16: Negative empirical-CDF calibration passes development

The binding selection rank was fold 2's 338th ascending negative (empirical quantile
0.04194), with current/raw scores 0.00004327/0.00006816. Its -0.454476 limit plus
the frozen margin produced a -0.504476-logit offset. Full negative empirical-CDF
dominance passed on folds 2/3/4 and in the reused fold-0/1 audit. Fixed-threshold
false positives fell to 157/199/416 from current's 224/249/546 on selection and to
270/421 from 357/574 in the audit; AP remained exactly invariant. The ignored
501-byte artifact has SHA-256
`ce33bae550f20ad53076338e8412aef97d26a4b1c6c9c6e201086ccd208a1c37`.
This authorizes one more explicitly post-test fresh replay under unchanged gates.

### 2026-07-16: Negative-CDF fresh replay frozen

The previously frozen calibrated fresh evaluator is reused byte-for-byte under a
new checksum-bound protocol. Only the calibration artifact changes, to the passing
-0.504476-logit negative-CDF control. All inputs, thresholds, row contracts, strata,
p95 and false-positive gates, and the unavailable-row bound remain identical. This
is still post-test reuse; a pass is a safety consistency check, not new external
evidence. Paper inputs remain inaccessible to the evaluator.

### 2026-07-16: Negative-CDF fresh safety replay passes

Every unchanged safety gate passed. Current and candidate each produced four false
positives, all shared, with zero candidate-only errors and zero non-clear errors.
Available p95 fell from 0.022308 to 0.014564; all-clear p95 fell from 0.023027 to
0.015959; non-clear p95 fell from 0.017337 to 0.011251. All logit-margin gates and
the symmetric six-unavailable-row bound passed. This is post-test safety consistency,
not independent external confirmation, but it authorizes the already planned exact
MARS-S2L v3 paper replay. The fresh evaluator accessed no paper input.

### 2026-07-16: Exact calibrated spatial-Prithvi paper replay frozen

The final evaluator is checksum-bound to the official MARS-S2L paper v3 receipt and
paper URL. It must reproduce the exact full 43,529-row/1,289-site comparator (AP
0.6410196024, recall 0.7915057915, FPR 0.0706923003, IoU 0.3243651486) and exact
15,655-row/697-site test-only comparator (AP 0.4502738001, recall 0.7753303965,
FPR 0.0755120560, IoU 0.1715619429) before candidate evaluation. All 43,524
available ensemble scores are built and persisted from label-free current, spatial,
and Prithvi inputs before the diagnostic labels open; five unavailable rows retain
the adversarial missing policy. The dense mask is the independently validated
released-probability gate. Both views require positive point and 10,000-replicate
paired-site lower bounds for AP, matched-FPR recall, and IoU, plus no-worse fixed
FPR. Failure on any gate resumes architecture research.

### 2026-07-16: Calibrated spatial-Prithvi ensemble rejected on exact v3 test-only uncertainty

The evaluator reproduced every exact comparator identity field and built the
2,339,911-byte label-free ensemble cache (SHA-256
`b58895b44091e02e0161842abbe2becf2da4c2060952b804320755d0a30e92ed`)
before labels opened. The full view passed decisively: AP 0.676102 versus 0.641020
(delta +0.035082, paired-site interval [+0.016224,+0.049491]); matched-FPR recall
delta +0.031440 [+0.019979,+0.048110]; IoU 0.379964 versus 0.324365 (delta
+0.055598 [+0.034660,+0.077298]). Test-only point metrics improved—AP 0.467027
versus 0.450274, matched-FPR recall +0.022026, IoU 0.292462 versus 0.171562—but
AP and matched-recall intervals crossed zero ([-0.023301,+0.052032] and
[-0.008066,+0.051119]). The test-only view therefore failed. This architecture is
not the final successor, and no claim of unequivocal superiority is permitted.
The next architecture must create a materially stronger unseen-site ranking signal,
not another calibration or blend of these highly correlated component scores.

### 2026-07-16: Robust Prithvi hard-pair ranker frozen

The next architecture targets unseen-site ordering directly. A compact 128-hidden
residual MLP consumes frozen Prithvi CLS, augmented scene/context, and OOF spatial
features. Its BCE objective is a smooth worst-domain loss over source-fold x sensor
x class groups with physical-site balancing; a frozen three-value pairwise term
compares positives against the ranker's hardest source negatives each epoch. This draws
on regularized GroupDRO (`https://openreview.net/pdf?id=ryxGuJrFvS`), unlabeled
target moment alignment (`https://arxiv.org/abs/1607.01719`), and direct AUC
optimization for imbalance (`https://proceedings.mlr.press/v119/guo20f.html`).
Only pair weights 0/0.25/0.5 and current blends 0.05/0.10/0.20/0.30 are searched.
Selection cross-fits folds 2/3/4 using only the other selection folds; the fixed
winner is audited on already-exposed folds 0/1. Every fold, both sensors, matched-
FPR recall, and paired-site AP must pass before any fresh or paper scoring.

### 2026-07-16: Robust Prithvi hard-pair ranker rejected

The selected pair weight was 0.50 with a 0.10 blend. Selection AP improved
+0.001993 with paired-site interval [+0.000725,+0.003275], but fold-2 and fold-4
matched-FPR recall each regressed about 0.0013. More importantly, the reused
fold-0/1 audit improved AP only +0.000658 with interval
[-0.000524,+0.001675], and fold-0 recall regressed 0.00134. No artifact or score
cache was written; fresh and paper data were never loaded. A nonlinear head over
frozen embeddings remains too correlated with current errors. Frozen-feature head
experiments are now retired. The next representation experiment must adapt internal
foundation-model features under strong parameter and group-robust regularization.

### 2026-07-16: Patch-supervised Prithvi LoRA pilot frozen

The first internal-representation adaptation is now frozen after a successful
16-scene GPU smoke test. It adds rank-4 low-rank residuals only to attention qkv
and output projections in Prithvi's final four encoder blocks; all other foundation
weights remain fixed. A shared target-frame 8x8 patch head receives observable-mask
supervision, while learned patch attention, top-4 pooling, and the CLS token feed a
192-hidden scene head. Scene BCE is augmented by fixed 0.25 patch and 0.10 hard-pair
terms. The complete model has 130,754 trainable parameters. The smoke run completed
one epoch with finite total, scene, patch, and pair losses and produced no report,
candidate choice, checkpoint, or score cache.

Selection is physical-location cross-fitting on folds 2/3/4, with only the other
two selection folds used for each prediction. Current-score blends are restricted
to 0.05/0.10/0.20/0.30/0.50, and fixed every-fold, sensor, matched-FPR recall, and
10,000-replicate paired-site AP gates apply. Folds 0/1 are explicitly a reused audit,
not independent confirmation. A passing seed 20261800 must be reproduced by a
separately frozen second seed before any fresh or exact-paper scoring. No fresh or
paper input is available to this trainer.

### 2026-07-16: Pre-result parallel-loader amendment frozen

The initial full command was interrupted before its first epoch or any metric after
measured single-worker TIFF throughput left the GPU near 2% utilization. No candidate
result was observed. The only amendment uses four deterministic data-loader workers,
two-batch prefetch per worker, and pinned host transfer for training and scoring.
Samples, replacement sampler, augmentation rules and seeds, architecture, optimizer,
losses, epochs, folds, blends, and gates are unchanged. A repeat 16-scene GPU smoke
test passed with finite losses, 16/16 finite scores, and the same 130,754 trainable
parameters. The amended trainer hash and loader settings were frozen before restart.

### 2026-07-16: Patch-supervised Prithvi LoRA scene pilot rejected

The seed-20261800 pilot completed all three selection cross-fits and the reused
fold-0/1 audit without numerical or data failures, but its ranking signal was
anti-helpful. The selected minimum 0.05 blend reduced selection AP by 0.000489
(paired-site interval [-0.001637,+0.000990]) and matched-FPR recall by 0.000431.
Fold 2 regressed most: AP -0.001691 and recall -0.003764. Folds 3/4 had small AP
gains, but could not satisfy the every-fold gate. The reused audit also regressed
AP by 0.000514 (interval [-0.001526,+0.000361]); folds 0 and 1 both lost recall.
Larger blends worsened selection AP monotonically, reaching -0.052206 at 0.50.
This rules out second-seed replication and external scoring for this score family.
No artifact or score cache was written, and no fresh or paper input was opened.

### 2026-07-16: Physics-guided released-U-Net adapter pilot frozen

The next representation restores the ingredient most associated with MARS-S2L's
unseen-site gain: its wind-matched physics simulation. The exact v3 paper reports
test-only AP 0.4496 with simulation versus 0.2725 without it
(<https://arxiv.org/html/2511.21777v3>). A June 2026 methane-segmentation study
independently motivates multi-scale feature-guided fusion of semantic context and
an explicit methane-enhancement branch (<https://arxiv.org/abs/2606.26416>).

The frozen pilot keeps the released U-Net entirely fixed. A 25-channel physics
pyramid (MBMP, target/reference changes and ratios, wind, cloud, sensor identity,
and released logits) supplies bounded scale-and-shift gates to all five teacher
encoder levels; the frozen teacher decoder maps the adapted features back to plume
logits. Every gate is zero-initialized, and the GPU smoke test reproduced released
logits bitwise (`max_abs_delta=0`). Its 1,293,888 trainable parameters then completed
one 64-sample physics-simulated epoch with finite focal, Dice, scene, pairwise, and
teacher-conservation terms.

Training is fixed to folds 3/4, seed 20262300, three 32,768-sample epochs, 50%
requested wind-consistent simulation, and no held-fold epoch selection. The scene
surrogate combines global top-100 evidence with a local 10x10 mean, aligning much
more closely with the paper's 100-connected-pixel score than the rejected top-4
Prithvi patches. Fold 2 is a reused architecture-selection pilot. Four frozen
adapter strengths must deliver at least +0.003 AP, nonnegative matched-FPR recall,
strictly higher IoU, nonnegative AP for both sensors, and a positive 10,000-replicate
paired-site AP lower bound. A pass authorizes only a new multi-seed cross-fit; no
fresh or exact-paper input is available to this trainer.

### 2026-07-16: Physics-guided teacher adapter v1 rejected with positive point signal

The fixed endpoint completed 98,304 optimizer samples without a runtime or numerical
failure. At adapter strength 0.25 it passed every point gate on reused selection fold
2: AP +0.003174, matched-FPR recall +0.003764 (three additional positives), IoU
+0.009601, Landsat AP +0.000247, and Sentinel-2 AP +0.005036. Its paired-site AP
interval nevertheless crossed zero [-0.004576,+0.007349], so the preregistered
confidence gate failed and no full cross-fit is authorized. Strength 0.5 raised
recall and IoU further but regressed Landsat AP; strength 1.0 overcorrected.

The run also exposed a training-distribution mismatch without changing any metric:
the inherited site/label/sensor-cell sampler generated comparatively few positive
requests, and simulation is attempted only for positive requests. Consequently the
declared 0.50 attempt probability realized just 10.4-10.8% simulated scenes. The
paper instead samples a binary plume indicator within locations and reports roughly
half of plume examples as simulated. A distinct v2 may correct the sampler to equal
positive/negative and sensor request mass, retain the same architecture and gates,
use the single v1 point-safe strength 0.25, and require the same positive site-
bootstrap lower bound. V1 itself is rejected: no artifact was written and no fresh
or paper input was opened.

### 2026-07-16: Balanced-request physics-guided adapter v2 frozen

V2 changes no model layer, loss weight, optimizer setting, metric, or evidence gate.
Its sampler first gives exactly 0.25 total request mass to each PLUME/NO_PLUME x
Sentinel-2/Landsat stratum, then equalizes occupied physical-site cells within the
stratum. Positive requests therefore carry exactly 0.50 mass before the unchanged
0.50 simulation attempt and wind/visibility filters. The single adapter strength is
fixed to 0.25 because it was the only v1 strength that passed every point gate; no
strength search is reopened. A fourth fixed endpoint epoch is justified by the still-
declining training objective and is frozen without intermediate held-fold scoring.

The GPU smoke reproduced released logits exactly, verified all four request masses
at 0.25, realized 17.2% successful simulations in its 64-sample draw, and completed
finite optimization with the unchanged 1,293,888 trainable parameters. The full
pilot retains the +0.003 AP, nonnegative recall and two-sensor AP, positive IoU, and
strictly positive paired-site lower-bound gates. Fold 2 is transparently reused
selection evidence. Failure cannot create an artifact or authorize external access.

### 2026-07-30: Balanced-request physics-guided adapter v2 rejected

The checksum-bound restart completed four fixed 32,768-sample epochs after an
earlier process interruption produced no report. All losses were finite and total
loss fell from 0.318170 to 0.257249. The sampler correction worked as designed:
successful simulation stayed between 24.16% and 24.54%, versus 10.4-10.8% in v1.
The fixed strength-0.25 endpoint nevertheless reduced fold-2 AP by 0.001189 and IoU
by 0.001803. It recovered one additional positive at matched FPR (recall delta
+0.001255), but Sentinel-2 AP fell 0.000671 and the paired-site AP interval was
[-0.005348,+0.002351]. It failed four promotion gates and wrote no artifact.

This is a clean negative ablation: correcting simulation frequency did not improve
the same architecture, so additional synthetic exposure is retired as an isolated
lever. Future simulation must distinguish real from synthetic domains through a
curriculum, domain-conditioned normalization, or separate losses rather than assume
exchangeability. Fresh and exact-paper inputs remained inaccessible.

### 2026-07-30: Instance-aware physics architecture selected as the next branch

Two primary 2026 results motivate replacing the current pixel-only correction with
explicit object structure. A MethaneSAT study reports Mask R-CNN outperforming U-Net
on both MethaneAIR and MethaneSAT, followed by further gains from physics-informed
morphological filtering and proximity merging
(<https://arxiv.org/abs/2605.24273>). CVPR 2026 work shows that neighbor-critical
segmentation penalties can improve connected-component topology while remaining
compatible with focal and Dice losses
(<https://openaccess.thecvf.com/content/CVPR2026/papers/Valverde_Towards_High-Quality_Image_Segmentation_Improving_Topology_Accuracy_by_Penalizing_Neighbor_CVPR_2026_paper.pdf>).

The next local experiment will therefore retain the exact released model as an
identity-safe floor but add a compact objectness/proposal branch over physics-guided
features. Supervision will be derived from the existing connected plume masks:
component centers, component extent, and a differentiable occupancy map aligned to
the paper's 100-connected-pixel scene rule. A consistency gate will permit pixel
corrections only where objectness and methane enhancement agree. Real and synthetic
losses will be reported separately, with synthetic examples used for representation
learning but real examples controlling the scene-ranking objective. This branch
targets the two failed quantities directly—unseen-site AP and component IoU—without
reopening score calibration or synthetic frequency.

### 2026-07-30: Instance-guided teacher pilot frozen after GPU smoke

The implementation adds a half-resolution plume-occupancy and component-center head
over concatenated physics and released-teacher features. Upsampled objectness gates
the adapted-teacher correction plus a bounded, zero-initialized pixel residual. The
training scene surrogate combines the paper-aligned connected-pixel approximation
with proposal occupancy and center evidence. Synthetic examples receive half weight
for pixel/object representation learning; only real examples drive scene BCE,
real-positive/negative pairs, and teacher-direction penalties. Balanced request
mass is retained, while simulation attempt probability is reduced to 0.25 so
synthetic exposure is no longer the dominant change.

The 64-sample GPU smoke reproduced released logits exactly (`max_abs_delta=0`),
verified 0.25 mass in every label x sensor stratum, produced nonempty center targets
for all eight positive inspection rows, and completed finite optimization for all
loss terms. The architecture has 1,551,795 trainable parameters and realized 14.1%
simulation in the smoke draw. The full pilot is fixed to seed 20263000, three
32,768-sample epochs, strength 0.25, and the unchanged AP/recall/IoU/two-sensor/site-
bootstrap gates on reused fold 2. It cannot access fresh or exact-paper inputs.

### 2026-07-30: Instance-guided teacher rejected, ranking branch retained

Three fixed epochs completed without numerical failure. Total loss fell from
0.338580 to 0.268222, object and real-scene losses improved, simulation stayed near
12.2%, and mean object gate probability tightened from 0.0599 to 0.0452. At the
fixed strength, fold-2 AP improved +0.003147, meeting the point floor; Landsat and
Sentinel-2 AP improved +0.000638 and +0.004058, and matched-FPR recall was unchanged.
The paired-site AP interval still crossed zero [-0.003005,+0.006813].

Pixel behavior explains the rejection. The candidate added 105,440 predicted-
positive pixels but only 29,942 intersecting truth pixels, reducing IoU by 0.006802.
No checkpoint was written. The object representation is therefore rejected as a
standalone mask corrector but retained as a candidate scene-ranking signal.

### 2026-07-30: Conservative two-output instance ensemble selected

Scene ranking and dense segmentation will now be separated, as already supported
by the project's multi-task architecture and prior development evidence. The
instance signal must improve the stronger cross-fitted spatial/Prithvi scene score,
not merely the released U-Net. The dense output will reuse the independently
confirmed development rule: released probabilities thresholded at 0.70 for Landsat
and 0.80 for Sentinel-2, then set empty below the fixed cross-fitted scene cutoff
0.75. That rule improved development IoU +0.022211 overall, +0.010805 on fold 2,
and passed every sensor, fold, retained-TP, and paired-site gate.

A separately seeded rerun will expose three label-free instance signals—connected
corrected probability, the real-scene head, and proposal objectness. Only small
predeclared blends with the current `inner_new` score will be tested. Promotion
requires a material, site-bootstrap-positive AP gain over the current ranker,
nonnegative matched-FPR recall and per-sensor AP, plus the already frozen positive
mask-IoU evidence. Failure writes no model or score cache and cannot authorize any
external evaluation.

### 2026-07-30: Conservative instance-scene ensemble pilot frozen

The second-seed protocol is fixed at seed 20263100 and the same three-epoch
instance-training recipe. At the endpoint it will create three label-free fold-2
signals: connected corrected probability at strength 0.25, sigmoid real-scene head,
and sigmoid proposal objectness. For each signal, only weights
0.025/0.05/0.10/0.20 are allowed in a convex blend with the stronger current
`inner_new` score. This is twelve candidates total. Promotion requires at least
+0.001 AP over current, nonnegative matched-FPR recall, nonnegative AP for both
sensors, and a positive 10,000-replicate paired-site AP lower bound.

The mask is not searched: it remains the confirmed sensor threshold plus current
scene cutoff rule, and must reproduce positive point, sensor, retained-TP, and
paired-site IoU gates from frozen development caches. Score and pixel caches are
checksum-bound and must align exactly to live evaluation labels, sensors, and
physical sites. The GPU smoke reproduced released logits exactly, completed finite
optimization with 1,551,795 trainable parameters, and wrote no output. No fresh or
exact-paper input is available to the protocol.

### 2026-07-30: Conservative instance-scene ensemble rejected

The independent seed completed cleanly and live fold-2 rows aligned exactly to both
frozen caches. Its connected signal again improved over released AP (0.876105 versus
0.873419), demonstrating repeatability of the small object-aware gain. It did not
improve the current `inner_new` ranker at AP 0.916603. The best of twelve frozen
blends was scene-head weight 0.025 with AP delta -0.000004, recall delta -0.001255,
minimum sensor AP delta -0.000293, and paired-site interval
[-0.001317,+0.001008]. All larger weights worsened AP monotonically. Standalone
proposal AP was only 0.671773.

The conservative mask reproduced +0.010805 point IoU on fold 2, but the isolated
fold's paired-site lower bound was -0.002249; the prior positive mask interval was
for the larger selection and confirmation cohorts. No model or score cache was
written. This closes the instance/object signal family: it is reproducible against
the released model but redundant relative to current spatial/Prithvi errors.

### 2026-07-30: Dense foundation-token fusion selected next

The next branch will use Prithvi where previous experiments did not: dense spatial
tokens will be injected into the released U-Net hierarchy, rather than reducing the
foundation model to a frozen CLS vector or adapting late transformer blocks for a
standalone scene head. The exact released segmentation remains an identity-safe
floor. Frozen multi-layer Prithvi token maps will be projected into the teacher
bottleneck and decoder with zero-initialized cross-scale residuals, while the
physics branch supplies spectral-change cues. Training will retain real-only scene
ranking and conservative correction penalties.

This architecture is justified by two surviving observations: frozen Prithvi
features materially improved the current cross-fitted ensemble, while explicit
physics/object branches can produce repeatable gains over the released model.
Dense token fusion is intended to combine semantic generalization with local plume
geometry before connected-component scoring. It is a representation change, not
another calibration or scalar-feature head.

### 2026-07-30: Dense Prithvi/U-Net pilot frozen

The cache audit verified 44,363 development rows with exact alignment across sample
identifier, physical-site fold, label, sensor, and the current cross-fitted score.
Each scene carries four 192-channel target-minus-reference token maps from frozen
Prithvi-EO-2.0 blocks 3/6/9/12 on an 8x8 grid. The 4.36 GB cache remains ignored;
its SHA-256, the Prithvi checkpoint identity, the development manifest, and the
fold protocol are bound into the new protocol.

The fusion model projects the four token depths and combines them with explicit
spectral-change physics at released-U-Net levels 3, 4, and 5. Zero-initialized
bounded feature gates make the dense output exactly the released U-Net at
initialization. A separate zero-initialized scene residual is stacked on the
current cross-fitted spatial/Prithvi score, so scene ranking also starts at the
stronger current system rather than at the paper model. Auxiliary 8x8 plume
occupancy, pixel focal/Dice, scene BCE, and pair-ranking losses jointly supervise
the representation.

The full-batch GPU smoke used the production batch size of eight, reproduced both
floors exactly (`pixel_max_abs=0`, `scene_delta_max_abs=0`), assigned exactly 0.25
sampling mass to every label x sensor stratum, and completed finite optimization
with 5,470,867 trainable parameters. The fixed pilot uses folds 3/4 for fitting,
reused fold 2 for architecture selection, seed 20263200, three 24,576-sample
epochs, and only residual strengths 0.25/0.50/1.00. Promotion requires at least
+0.001 AP over the current ranker, nonnegative matched-FPR recall and both sensor
AP deltas, a positive paired-site AP lower bound, and higher dense-mask IoU under
the already frozen sensor-threshold/scene-gate rule. No fresh or exact-paper input
is accessible to this experiment.

### 2026-07-30: Dense fusion splits the mask and scene decisions

All three fixed epochs completed cleanly. Total loss fell from 0.327994 to
0.221126; focal loss fell from 0.068986 to 0.029523, Dice loss from 0.266612 to
0.187245, patch loss from 0.137736 to 0.061773, scene BCE from 0.165925 to
0.137560, and the real-negative upward penalty from 0.052949 to 0.012075. The
representation learned meaningful local and scene structure without numerical or
memory failure.

The joint candidate is rejected because the learned scene residual does not improve
the already-strong current ranker. At selected strength 0.25, fold-2 AP changes
-0.000644, matched-FPR recall is unchanged, Sentinel-2 AP changes -0.000937, and
the paired-site AP interval is [-0.001659,+0.000297]. Strengths 0.50 and 1.00
regress AP further. No artifact was written under the frozen joint-promotion rule.

The dense-mask result is materially different. At strength 0.25, IoU improves
from 0.577726 to 0.584476 (+0.006750) with a positive paired-site interval
[+0.001740,+0.010598]. The preregistered strength 0.50 is stronger still:
IoU 0.588789, delta +0.011063, interval [+0.003561,+0.016703], while increasing
true-positive pixels from 801,164 to 826,403. Strength 1.00 becomes over-aggressive
and loses interval support. Dense Prithvi/U-Net fusion is therefore retained as a
mask architecture at strength 0.50, while its scene residual is retired.

Because the joint failure protocol correctly wrote no checkpoint, a separate
mask-only preservation protocol must reproduce the fixed seed/endpoint and save
the adapter only if strength 0.50 reproduces positive point, sensor, retained-TP,
and paired-site IoU gates with the current scene score left exactly unchanged.
That follow-up is development-only and will require independent folds before any
paper-cache use. Scene ranking will proceed as a separate research branch using
features from the learned dense representation rather than its BCE residual.

### 2026-07-30: Mask preservation replication closes initialization gap

The joint trainer set seed 20263200 immediately before optimization but constructed
random token projections, physics blocks, and scene layers just before that call.
Thus its sampling and optimization were seed-bound while its random module
initialization was not. The result remains valid as one completed pilot, but its
checkpoint could not have been exactly reconstructed from the nominal seed. This
provenance gap was found before promoting or preserving any artifact.

The mask-only replication now calls the same seeding function before model
construction and again at optimization start. It otherwise retains the exact
architecture, folds, 24,576-sample endpoint, batch size, sampler, augmentations,
loss, and optimizer. Strength 0.50 is fixed from the completed joint result; scene
strength is exactly zero, so the current cross-fitted score and 0.75 gate cannot
change. Promotion requires positive pooled and paired-site IoU, nonnegative IoU
for both sensors, at least 100% of the current rule's aggregate true-positive
pixels, and scene-score identity within 1e-6.

The full-batch GPU smoke reproduced zero pixel and scene residuals at
initialization, exact 0.25 request mass in all four label x sensor strata, all
44,363 cache identities, and finite optimization with the initialization-bound
seed. No exact-paper or fresh external input is accessible. The protocol is frozen
before its full reproduction run.

### 2026-07-30: Deterministic dense mask replication passes

The initialization-bound run closely reproduced the original learning trajectory.
At epoch 3, total loss was 0.221123 versus 0.221126 previously; focal, Dice, patch,
scene, and pair losses remained similarly close. The largest absolute difference
among all recorded history fields was 0.02425, arising in correction L2 scale,
well below the descriptive 0.20 bound. Exact scene-score identity held to
5.96e-8.

At the fixed mask strength 0.50, fold-2 IoU improved from 0.577726 to 0.587695,
a +0.009969 gain with paired-site interval [+0.002480,+0.016141]. Sentinel-2
improved +0.010014 and Landsat +0.009504. True-positive pixels rose from 801,164
to 823,208 (1.027515x) while recall increased from 0.644922 to 0.662667. The
precision tradeoff was 0.847206 to 0.838567, but aggregate IoU and both sensor
domains improved.

All preservation gates passed. The ignored 21,925,323-byte adapter was saved as
`EarthRemoteSensingRapidResponse/artifacts/mars_dense_prithvi_mask_pilot.pt` with
SHA-256 `f5304281ed85e1eb997da33905bb047673e8e5c0176ac99feb76b6a7105c6cea`.
This is architecture-selection evidence on reused fold 2, not final confirmation.
The next mask step is a preregistered multi-seed cross-fit over all development
folds, with pooled and per-sensor paired-site IoU gates. The scene branch remains
the unchanged current ranker until a separately validated representation-level
ranking improvement is found.

### 2026-07-30: Frozen dense scene-representation extraction

The preserved mask adapter will now be used as a frozen feature generator, not as
a scene classifier. For every development scene, the extractor records the
585-dimensional pre-MLP scene pooling vector, its 64-dimensional learned
embedding, the rejected scalar scene residual as an explicitly labeled feature,
and 20 observable-region statistics from released probabilities, strength-0.50
adapted probabilities, logit corrections, and 8x8 patch logits. The unchanged
current cross-fitted score is included as the exact residual floor. This produces
671 float16 features per scene.

The extractor binds the 21.9 MB adapter, released checkpoint, 4.36 GB Prithvi
token cache, current score cache, manifest, and fold protocol by SHA-256. Its
16-row CUDA smoke produced a finite 16x671 matrix with exact sample/fold alignment.
Extraction is label-free; labels are copied only into ignored development metadata
for the later ranker. The bulk cache will remain ignored and only its compact
receipt will be committed. No exact-paper or fresh external input is in scope.

### 2026-07-30: Dense scene cache completed and independently finalized

All 44,363 rows and 671 features were extracted without a geometry or finiteness
failure. A transient WSL DrvFs visibility race occurred after the atomic rename:
the extractor successfully created and hashed both final cache files, but its
immediate `stat()` for receipt formatting briefly returned not-found. It therefore
failed closed and wrote no receipt.

A separately committed finalizer verified the historical generator commit and raw
extractor blob, re-hashed every frozen input, scanned the complete float16 matrix
for finiteness, checked the 44,363x671 shape, and matched every sample identifier,
physical site, and fold back to the manifest and site protocol. The final feature
cache is 59,535,274 bytes with SHA-256
`cb44c6957be7fb2df9b95652dcf8f099775e244d87ddfea91d15679cc62400cc`;
metadata is 1,537,400 bytes with SHA-256
`533300f8ec7760b645b7e30c23a569dcc6c5522d6bc1c64ee7ad485623cffb32`.
Both remain ignored. The compact receipt records generator commit
`8e3d0807650fc21215c19f43a9d702e692930212`, finalizer commit
`3f06447e1f140f16104ab08fa5ae45560cf27c0b`, and no external input access.

### 2026-07-30: AP-focused dense residual selected internally

The rejected end-to-end BCE scene residual was excluded. A new 67k-parameter
zero-initialized MLP instead learns a bounded residual around the exact float64
current score from 670 frozen features. Preprocessing uses only each fit split's
median and normalized IQR; absolute sentinel values >=1000 are treated as missing
and replaced with the fit median. The objective directly combines a differentiable
SmoothAP listwise surrogate with current-difficulty-weighted pair ranking, light
BCE, and residual L2.

Before any fold-2 evaluation, two seeds trained on fold 3 and predicted fold 4,
and two seeds trained on fold 4 and predicted fold 3. All five predeclared residual
strengths passed the internal point/domain/site gates. Improvements increased
through strength 0.80, which is now frozen: pooled AP +0.007277, matched-FPR recall
+0.012467, Sentinel-2 AP +0.004609, Landsat AP +0.005862, fold-3 AP +0.009844,
fold-4 AP +0.007054, and paired-site AP interval lower +0.002105.

These folds also trained the frozen representation, so this is architecture
selection evidence rather than unbiased confirmation. Fold 2 has not been scored.
The trainer, two final seeds, 1,500-step endpoint, strength 0.80, feature hashes,
exact current-score cache, and +0.001 held-fold AP floor are now committed before
fitting on folds 3+4 and evaluating the reused architecture fold once. Failure
writes no artifact and cannot authorize external scoring.

### 2026-07-30: AP residual rejected by true representation holdout

The frozen internal selection recomputed exactly, then two final seeds trained on
folds 3+4 and strength 0.80 was evaluated once on fold 2. It failed decisively:
AP changed -0.015748, matched-FPR recall -0.007528, Sentinel-2 AP -0.021159,
Landsat AP -0.004460, and the paired-site AP interval was
[-0.032805,-0.003079]. No artifact was written and no external input was loaded.

This large reversal is informative. The ranker heads cross-predicted folds 3 and 4,
but the frozen dense adapter that generated both sets of features had itself been
trained on both folds. Head-level cross-fitting therefore did not remove
representation-label leakage. Fold 2 was the first genuinely out-of-fit
representation and exposed the distribution shift. SmoothAP optimization is not
rejected; training it on in-sample learned representations is.

Any next dense scene experiment must generate fold-3 features from an adapter that
did not train on fold 3 and fold-4 features from an adapter that did not train on
fold 4. Only those honest cross-fitted representations may select a residual head
for the existing folds-3+4 -> fold-2 adapter. This correction is also the required
design for later five-fold paper-grade training.

### 2026-07-30: Honest dense-representation cross-fit frozen

The next experiment corrects the representation leakage directly. One adapter is
fit only on fold 4 and will encode fold 3; an independently initialized adapter is
fit only on fold 3 and will encode fold 4. Folds 0, 1, and 2 remain completely
outside both jobs. Each endpoint uses the same fixed three 24,576-sample epochs as
the preserved dense adapter and is saved only as an ignored intermediate
representation artifact. Descriptive mask metrics do not promote or reject these
endpoints; their purpose is to create honest out-of-fold ViT representations.

The two-job CUDA smoke passed before freezing. Both branches reproduced the
released model exactly at initialization (maximum absolute pixel and scene-score
differences both 0.0), completed a finite training epoch, and sampled all four
label-by-sensor strata with equal requested mass. The frozen protocol binds the
trainer and every model, manifest, fold, checkpoint, feature, and score input by
SHA-256. The full run may now train the two endpoints, after which feature
extraction and AP-head selection must use only each adapter's held-out fold.

### 2026-07-30: Primary-literature architecture cross-check

The June 2026 multimodal methane-segmentation study
<https://arxiv.org/abs/2606.26416> uses intermediate DINOv3 transformer blocks
2/5/8/11, a separate ResNet-18 methane-enhancement encoder, multiplicative
methane-guided fusion at four semantic scales, and a SegFormer decoder. On its
4,172-scene EMIT-derived MPDataset it reports 82.15% mean IoU, 90.07% mean
precision, and 76.93% plume recall, improving its MPSUNet comparator by
0.92/0.87/1.01 percentage points respectively. Its benchmark and metrics are not
comparable to MARS-S2L, but its ablation-motivated pattern independently supports
the present architecture: intermediate Prithvi blocks 3/6/9/12 provide semantic
context, the spectral-change physics pyramid provides methane evidence, and
zero-initialized scale/shift gates fuse them at three deep released-U-Net levels.

The May 2026 MethaneSAT study <https://arxiv.org/abs/2605.24273> reports that Mask
R-CNN improved pixel F1 over U-Net by 10.49 points on MethaneAIR and 5.48 points on
MethaneSAT. Its physics-informed morphological filtering and proximity merging
then trade sensitivity for precision. This supports keeping object consistency
and post-processing in the research space, but the local controlled object branch
already proved redundant for scene AP and harmful to pixel IoU. The current
decision therefore remains evidence-led: retain transformer/physics fusion for
the mask, and require a separately cross-fitted ranker to improve plume/no-plume
ordering. Neither external paper supplied data or labels to the running endpoint.

### 2026-07-30: Honest dense-representation endpoints completed

Both checksum-bound jobs completed all three fixed 24,576-sample epochs with
finite parameters and exact released-model identity at initialization. The
fold-4-only endpoint's loss fell 0.321839 -> 0.228765 -> 0.207340; the fold-3-only
endpoint fell 0.259684 -> 0.184310 -> 0.156027. Their ignored 21,926,411-byte
artifacts have SHA-256 values
`8fae4288edeae4aa74438356385992860c4a8e2af2e340cf1b3155f6503313e8`
and
`b35cef6d577286376f0ed5630c6967cd91db0dc33cd4d24c7e1a2d228ea388f2`
for held folds 3 and 4 respectively.

The descriptive strength-0.50 mask transfer was negative: IoU changed -0.007476
on fold 3 and -0.005995 on fold 4, with both paired-site lower bounds negative.
This contrasts with the +0.009969 fold-2 gain from the folds-3+4 adapter and shows
that a one-source-fold mask adapter is not itself a transferable mask model. It
does not invalidate the preregistered representation purpose: every endpoint is
retained unconditionally so its held fold can be encoded without representation
label leakage. No endpoint is promoted for segmentation. The next inexpensive
gate is label-free extraction followed by honest fold-3/4 AP-head cross-prediction;
failure there will retire the dense scene-representation branch before another
held-fold or external evaluation. No fresh or exact-paper input was accessed.

### 2026-07-30: Honest dense-feature extraction frozen

The extractor maps each fold-3 scene through only the fold-4-trained endpoint and
each fold-4 scene through only the fold-3-trained endpoint. It exports the same
671-feature schema as the preserved adapter cache, separately retains the exact
float64 current score as the residual floor, and binds both artifact schemas,
their training protocol, the endpoint report, released checkpoint, Prithvi token
cache, manifest, and fold assignments by SHA-256.

The two-artifact CUDA smoke encoded eight scenes from each held fold into a finite
16x671 matrix. Both artifact hashes and fold contracts matched, the float16 score
feature agreed with the exact floor under the frozen tolerance, and all 16 sample
identities mapped to the 44,363-row Prithvi cache. The full 17,745-row extraction
is label-free; labels are copied only into ignored development metadata for the
subsequent ranker. No fold 0/1/2, fresh, or exact-paper row is selected.

### 2026-07-30: Honest dense-feature cache completed and audited

All 17,745 selected scenes were encoded: 8,799 from fold 3 and 8,946 from fold 4.
The 17,745x671 float16 matrix is 23,813,918 bytes with SHA-256
`f45de57bda41ba083b4a981cbbcafa3c1e537ff7178eedae2c2ae2567987f78e`;
its 706,101-byte metadata has SHA-256
`5c4c9930b9e4fdbd295ffbd17d621db23f2aa33f98e88f87b2496c0f93d4805d`.
Both remain ignored.

An independent full scan found all 11,906,895 cached values finite, verified the
17,745 unique sample identifiers and exact fold counts, and confirmed 16,221
negative/1,524 positive scenes across 14,963 Sentinel-2 and 2,782 Landsat scenes.
The separately stored exact float64 score floor is finite over
[0.0000166513, 0.9999570152] and matches the float16 feature copy under the frozen
tolerance. The compact receipt binds generator commit
`b206ebb4af6c13a904da127aa46812877fbab009`, both endpoint hashes, output hashes,
and `external_inputs_accessed=false`.

### 2026-07-30: Honest-crossfit AP ranker frozen

The AP-focused head is unchanged in capacity and objective from the rejected
in-sample-representation experiment: 670 inputs, 96/32 hidden units, bounded
zero-initialized logit residual, SmoothAP plus hard-pair/BCE/L2 losses, two seeds,
and a fixed 1,500-step endpoint. The material correction is the input contract.
Internal fold-3/4 training now uses only individually out-of-fit representation
features; a later fold-2 prediction, if authorized, will use the preserved
folds-3+4 adapter. The exact float64 current score remains the identity floor.

Seven residual strengths (0.025/0.05/0.10/0.20/0.40/0.80/1.00) and all promotion
gates were fixed before internal training. The CUDA smoke used a balanced
32-row fold/label/sensor grid, trained both internal cross-prediction directions
and the cross-domain final-fit path for two steps each, and produced finite
32-row outputs with 670 features. Every model passed its exact-zero residual
initialization assertion. Internal selection may use only folds 3 and 4; fold 2,
fresh data, and exact-paper data remain unavailable to it.

### 2026-07-30: Honest dense AP ranker rejected internally

All four fixed 1,500-step heads completed, but none of the seven preregistered
strengths passed. Strength 0.025 was safest: pooled AP +0.000989 and paired-site
lower bound +0.000055, but matched-FPR recall fell 0.000656, Landsat AP fell
0.000130, and fold-3 AP fell 0.000002. The largest pooled AP gain was +0.006242
at strength 0.40, but both within-fold AP values, matched-FPR recall, Landsat AP,
and the paired-site lower bound were negative.

This is a Simpson's-paradox failure: pooled AP improves because the head shifts
score levels between folds, while ordering within each fold does not improve at
any strength. The honest representation correction therefore changed the prior
apparent +0.007277 gain into evidence that the dense scene head is learning domain
offsets rather than transferable plume evidence. Fold 2 was not evaluated, no
artifact was written, and no fresh or exact-paper input was accessed. The dense
adapter remains the independently supported mask branch, but this scene-ranking
family is retired.

### 2026-07-30: Invariant sparse dense-signal audit frozen

Before discarding the 669 non-floor dense features entirely, a final cheap audit
tests them one at a time under a transformation that cannot exploit raw
fold/sensor offsets. Each feature is empirical-rank normalized and inverse-normal
transformed separately within each fold x sensor domain. Missing sentinels use
only the unlabeled domain median. Its plume direction is estimated on the
opposite fold and must agree in both folds before any residual strength is tested.

Seven small fixed logit strengths are evaluated against the exact current score.
Promotion requires +0.001 pooled AP, nonnegative AP within both folds and both
sensors, nonnegative matched-FPR recall, and a positive paired-site AP lower bound.
Only the top 20 point-gate survivors may be bootstrapped. The eight-feature smoke
found five direction-agreeing features, exercised the no-survivor path cleanly,
and wrote no output. This is an exploratory development audit; any discovery
would require a separately committed fold-2 protocol.

### 2026-07-30: Invariant univariate audit finds sparse transferable signal

Of 669 features, 345 had plume directions that agreed across folds after
fold x sensor rank normalization. Twelve feature/strength candidates passed all
point gates and were bootstrapped; six retained a positive paired-site AP lower
bound. The gate-ranked winner is `scene_head_input_49` (matrix column 51) at
strength 0.10. It improved pooled AP +0.001362, fold-3/fold-4 AP
+0.000403/+0.000786, Sentinel-2/Landsat AP +0.001738/+0.000321, and left
matched-FPR recall unchanged. Its paired-site AP interval is
[+0.000298,+0.002181].

This proves that the dense representation contains a small transferable signal
once raw domain offsets are removed, but the magnitude is not yet sufficient for
the paper test-only uncertainty target. Five unique features survived all gates:
columns 51, 47, 644, 335, and 156. Before opening fold 2, the next experiment will
freeze simple equal-weight top-k ensembles over only these survivors and add a
strict low-prevalence-site gate. No fold-2, fresh, or exact-paper input was
accessed.

### 2026-07-30: Sparse invariant ensemble protocol frozen

The next development experiment is limited to equal-weight prefixes of the five
univariate survivors, ordered by descending individual paired-site AP lower
bound: columns 51, 47, 644, 335, and 156. It searches top-k values 1 through 5
and logit-residual strengths 0.025, 0.05, 0.10, 0.20, and 0.40. Every feature is
rank/inverse-normal normalized within its own fold x sensor domain and oriented
using only the opposite fold.

Promotion now requires the earlier whole-development AP, fold, sensor, and
matched-FPR recall gates plus a deliberately demanding low-prevalence proxy:
11,344 rows, 118 positives, and 199 physical locations whose folds-3/4 site
positive rate is at most 5%. The low-prevalence pooled AP gain must be at least
+0.005, with no fold, sensor, or matched-FPR recall regression. Both the whole
and low-prevalence paired-site AP 95% lower bounds must be strictly positive
across 10,000 replicates. A smoke run exercised all four point candidates and
both bootstrap paths without writing an output. Fold 2 and external inputs
remain inaccessible to this protocol.

### 2026-07-30: Sparse invariant ensemble rejected

Only one of 25 frozen candidates cleared every point gate:
`scene_head_input_49` alone at strength 0.20. It improved whole-development AP
+0.002118 and matched-FPR recall +0.000656, with positive AP deltas in both folds
and both sensors. On the low-prevalence view it improved pooled AP +0.006841,
fold AP +0.006674/+0.007683, Sentinel-2/Landsat AP +0.006870/+0.000844, and
left matched-FPR recall unchanged.

The paired-site evidence rejected it. The whole-development AP interval was
[-0.000170,+0.003612], and the low-prevalence interval was
[-0.004742,+0.019381]. No multi-feature equal-weight prefix survived the point
gates. This separates two findings: the invariant feature itself has useful
rare-site signal, but equal-weight aggregation does not reduce its between-site
variance. No fold-2, fresh, or exact-paper input was accessed. The next path is
a cross-fitted, strongly regularized linear residual ranker over domain-rank
normalized dense features, with equal physical-site weighting and a separately
rank-normalized held-fold output. This replaces the failed equal-weight
assumption while retaining the invariant representation contract.

### 2026-07-30: Invariant site-balanced linear ranker frozen

The learned replacement is deliberately sparse and cross-fitted. All 669
authorized dense representation columns are rank/inverse-normal normalized
independently within each fold x sensor. For each candidate, an L1 logistic model
fit only on fold 4 predicts fold 3 and a separate model fit only on fold 3 predicts
fold 4. Each held decision function is rank normalized again within its fold x
sensor before it can perturb the exact current-score logit.

The frozen search contains five regularization strengths, three physical-site
weighting rules, and six residual strengths (90 candidates). The weighting rules
give every site equal total training influence, optionally equalizing label or
sensor x label cells within each site. The 5% low-prevalence cohort is used only
for evaluation gates, never as a feature or route. Promotion retains the same
+0.005 low-prevalence AP minimum, all fold/sensor/recall safeguards, and strict
positive whole and low-prevalence paired-site AP lower bounds over 10,000
replicates. A 32-feature smoke run exercised both honest fits, both candidate
strengths, and both bootstrap views without warnings or output artifacts. Fold 2
and all external evaluation inputs remain untouched.

### 2026-07-30: Invariant site-balanced linear ranker rejected

None of 90 frozen candidates cleared the point gates, so no bootstrap or artifact
was produced. The best-screened model was site-equal L1 logistic regression at
C=0.01 and residual strength 0.20. Its low-prevalence fold AP deltas were
+0.011540/+0.003415 and recall improved +0.008475, but pooled low-prevalence AP
changed -0.004393 because Sentinel-2 AP regressed -0.007083. Whole-development
AP regressed -0.008885 and recall -0.001969.

The coefficient diagnostics identify the failure mechanism. Across the two
opposite-fold fits, the best model selected 34 unique features but only five in
both fits; union sign agreement was 11.8% and coefficient cosine similarity was
0.159. Most other regularization/weighting combinations had zero to 17 shared
features and near-zero sign agreement. Domain-rank normalization removed marginal
offsets, but it could not make unconstrained multivariate feature axes stable.
No fold-2 or external input was accessed. The next experiment will therefore
exclude all direction-unstable columns and use only the five univariate survivors,
testing label-free nonlinear residual gates rather than learning high-dimensional
supervised weights.

### 2026-07-30: Direction-stable nonlinear residual search frozen

The next experiment holds the representation and feature selection fixed and
changes only where the invariant residual is allowed to act. Each of the five
all-gate univariate survivors is tested alone under eight label-free transforms:
symmetric, positive-only, negative-only, uncertainty-weighted, low-score-weighted,
high-score-weighted, tanh-bounded, and winsorized. Non-symmetric transforms are
mean-zero/unit-variance standardized within each fold x sensor using no labels.
Six frozen strengths produce 240 candidates.

This tests a specific causal hypothesis from the preceding failures: useful
direction-stable evidence may be concentrated in positive, negative, uncertain,
low-score, or high-score scenes, while global perturbations create the observed
between-site variance. All whole/low-prevalence point and paired-site gates remain
unchanged. A one-feature/two-transform smoke run exercised four candidates and
both bootstrap views. Fold 2 and external evaluation data remain prohibited.

### 2026-07-30: Single-feature nonlinear residuals rejected, complementary pair identified

Four of 240 candidates passed all point gates, but none passed both paired-site
intervals. The strongest anchor was column 644 (`scene_embedding_57`) with a tanh
transform at strength 0.20: whole AP improved +0.003258 with interval
[+0.000010,+0.005919], both folds and sensors improved, and recall was unchanged.
Its low-prevalence AP improved +0.008006 and recall +0.008475, but the low-site
interval remained [-0.002562,+0.019664].

The most important near-survivor was independently direction-stable column 156
(`scene_head_input_154`) with high-score weighting. At strength 0.10 it improved
low-prevalence AP +0.018270, including fold deltas +0.020036/+0.016834 and
Sentinel-2/Landsat deltas +0.028378/+0.002230, with unchanged recall. Whole AP
improved +0.002013 and both sensors improved, but fold 3 regressed -0.000833 while
fold 4 improved +0.002019. At strength 0.05 its whole fold deltas were both
positive, but low-prevalence Landsat AP was -0.001597.

These residuals have complementary failures: the column-644 tanh anchor supplies
stable whole-view and Landsat evidence, while the column-156 high-score residual
supplies the missing rare-site magnitude. The next experiment will freeze a
two-component additive logit residual using only these already screened transforms
and small independent strengths. It remains development exploration requiring a
separate fold-2 confirmation; no protected input was accessed.

### 2026-07-30: Complementary anchor-plus-booster protocol frozen

The complementary experiment fixes the anchor to column 644 with tanh and the
booster to column 156 with high-score weighting. It searches four small anchor
strengths (0.10-0.25) and four booster strengths (0.025-0.10), for 16 additive
logit residuals. No feature, transform, routing rule, or gate is reopened.

The one-candidate smoke path (anchor 0.10, booster 0.025) cleared every point gate:
whole AP +0.002722 with fold deltas +0.000307/+0.001782 and
Sentinel-2/Landsat deltas +0.003671/+0.000586; low-prevalence AP +0.010664 with
fold deltas +0.018570/+0.009332 and sensor deltas +0.011916/+0.004392. Recall was
unchanged in both views. Its 100-replicate smoke lower bounds were +0.000746
whole and +0.000430 low-prevalence. These smoke intervals are not inferential
evidence. The committed full protocol requires the unchanged 10,000-replicate
paired-site gates before fold 2 can be considered.

### 2026-07-30: Complementary invariant residual pair passes development

Eight of 16 candidates passed every point gate and six retained strictly positive
paired-site AP lower bounds in both views. The gate-ranked selection is anchor
strength 0.20 plus booster strength 0.05. On all 17,745 development rows, AP
improved +0.004336, fold-3/fold-4 AP +0.000166/+0.002748,
Sentinel-2/Landsat AP +0.005804/+0.001158, and matched-FPR recall +0.001312.
The 10,000-replicate physical-site AP interval was
[+0.000555,+0.007336].

On the 11,344-row low-prevalence stress view, AP improved +0.018325,
fold AP +0.024956/+0.017260, Sentinel-2/Landsat AP
+0.018623/+0.009407, and matched-FPR recall +0.008475. Its paired-site AP
interval was [+0.001558,+0.036592]. This is the first dense-scene architecture
in the current sequence to clear every preregistered whole and rare-site
development gate with publication-strength resampling.

The result authorizes only a separately committed, one-shot fold-2 confirmation.
Before any fold-2 representation or label access, a new dense adapter must be
trained on folds 3+4, and the fold-2 extractor, fixed component signs/transforms/
strengths, low-prevalence definition, sensor/recall gates, and paired-site
bootstrap must be frozen. External and exact-paper inputs remain untouched.

### 2026-07-30: Fold-2-excluding dense adapter training frozen

A fit-only adapter trainer is frozen for the one-shot confirmation path. It selects
only folds 3+4, uses the identical three-epoch/24,576-sample dense training
endpoint and balanced label x sensor sampler, records held fold 2 in the artifact
contract, and never constructs a fold-2 dataset, loader, mask evaluation, or
metric. The adapter is an ignored intermediate representation endpoint, not a
promoted prediction model.

The 16-row CUDA smoke used only folds 3+4, reproduced exact zero pixel and scene
corrections at initialization, maintained the required 0.25 request mass in each
label x sensor cell, completed a finite training epoch, and wrote no artifact.
The full trainer and its hashes will be committed before the endpoint is fit.

### 2026-07-30: Fold-2-excluding dense adapter trained

The frozen fit-only job completed on 17,745 folds-3/4 rows. Epoch loss decreased
monotonically 0.328333 -> 0.249191 -> 0.220884. The ignored 21,926,411-byte
artifact has SHA-256
`8855cb52d48c197d3831351b129ca49718f9e9caa3a3bc6b06cd17565c633d73`.
Independent loading verified 122 finite state tensors and the embedded contract:
kind `mars_dense_prithvi_crossfit_representation`, fit folds [3,4], held fold 2,
seed 20265900, mask strength 0.5, scene strength 0.0, and training-protocol
SHA-256
`506a210c4de33d8911be4827d073586e7dc3ef38650ae5a2e68191530e904385`.

The trainer constructed zero held-fold records and reports no fold-2 label, fresh,
or exact-paper access. This endpoint can now be hash-bound into a separately
committed fold-2 extractor and fixed-candidate evaluator.

### 2026-07-30: One-shot fold-2 extraction and evaluator frozen

The fold-2 extractor is hash-bound to the folds-3+4 adapter and will write an
ignored 671-column cache plus a compact receipt. The evaluator is committed
before extraction and verifies the future cache paths and hashes through that
receipt's frozen extractor-protocol identity. It fixes column 644 direction -1
with tanh at strength 0.20 and column 156 direction +1 with high-score weighting
at strength 0.05. Rank normalization and transform standardization are unlabeled
within fold 2 x sensor.

The first development-reproduction smoke correctly rejected a draft that had
encoded the anchor sign as +1; the committed development report shows column 644
has negative effects in both folds. This was corrected before any fold-2 cache
existed. The final smoke reproduced the selected folds-3/4 whole AP
0.9084121516710161, low-prevalence AP 0.7265114771853518, and both recall deltas
within 1e-12. The full fold-2 gates require AP gains >=0.001 whole and >=0.005
low-prevalence, nonnegative sensor AP and matched-FPR recall, no actual FPR
increase, and strictly positive 10,000-replicate paired-site AP lower bounds in
both views. The evaluator refuses to overwrite or repeat its report.

### 2026-07-30: Complementary pair rejected on one-shot fold 2

The frozen extractor encoded 8,833 unique fold-2 rows with the folds-3+4 adapter.
The ignored feature and metadata caches have SHA-256
`9f57ac352fa794f3e21ba7e28cb95cc61fa2e7d0289feb7599f28e4e2d5297c7`
and
`d3db1430dde0771e23d37026bdc37cfd02400492323960f1945b903783c714e3`.
The evaluator then ran exactly once.

The complete fold failed: AP changed 0.916603 -> 0.914441
(-0.002162), Sentinel-2/Landsat AP changed -0.002572/-0.000484, and
matched-FPR recall changed 0.956085 -> 0.949812 (-0.006274) at identical FPR
0.071180. The paired-site AP interval was [-0.006129,+0.000513].

The preregistered low-prevalence view replicated positively on 5,816 rows, 62
positives, and 137 sites. AP changed 0.678808 -> 0.686947 (+0.008138),
Sentinel-2/Landsat AP improved +0.008321/+0.015713, recall was unchanged at
0.951613, FPR was unchanged at 0.071255, and the paired-site AP interval was
[+0.002192,+0.016814]. Thus the rare-site mechanism is independently real, but
applying it to every unseen fold-2 site is unsafe.

The fixed pair is retired before fresh or exact-paper access, and fold 2 will not
be reused for model selection. The failure motivates a narrower deployment
architecture: preserve the validated current score bitwise for known training
sites and route the invariant pair only to label-free unseen-site cohorts whose
score-history statistics match the rare official test-only mixture. Any such
router must be selected again on folds 3+4 and confirmed on still-unused folds
0/1 before external access.

### 2026-07-30: Hard unseen-site router rejected

Only one of 42 candidates passed: the control with maximum size 1,000,000 and
maximum predicted-positive rate 1.0. It routes 100% of rows and sites and is
therefore exactly the already retired fixed pair, not a selective router. The
separate adjudication report denies promotion because independent fold 2 showed
that same all-site candidate regressed whole AP -0.002162 and recall -0.006274.

Every genuinely selective rate-only gate retained the rare-site benefit but
regressed fold-3 whole AP; adding a hard history limit routed only 8%-20% of rows,
missed the >=50% rare-site coverage requirement, and produced rare-site AP gains
below +0.003. No folds-0/1 or external access is authorized. The next router will
replace discontinuous site inclusion with a smooth label-free trust weight based
on current-score predicted-positive rate, continuously shrinking the fixed
residual as a site looks less like the rare target mixture.

### 2026-07-30: Smooth unseen-site trust protocol frozen

The smooth router preserves the known-site current score and, for unseen sites,
maps the current-score predicted-positive rate to a residual multiplier in [0,1].
It tests linear cutoff, exponential decay, and inverse-decay families over 16
frozen scales. A weight of zero is exactly the current score; one is the complete
fixed pair. No all-sites trust-one control is included.

Promotion requires mean trust >=0.5 on low-prevalence rows, at least half of those
rows receiving trust >=0.5, all prior whole/rare-site AP/fold/sensor/recall
gates, and positive paired-site AP lower bounds in both views. The two-candidate
smoke exercised smooth weighting, exact logit residual scaling, trust checks, and
both bootstrap paths. Fold 2 remains report-only and folds 0/1 remain untouched.

### 2026-07-30: Smooth unseen-site trust rejected

None of 16 smooth trust candidates cleared the point gates. The strongest
candidate, linear scale 1.0, retained 95.1% mean trust on low-prevalence rows and
improved low-prevalence AP +0.016909, but fold-3 whole AP regressed -0.004319
while fold 4 improved +0.002841. Inverse scale 0.4 improved pooled whole AP
+0.001792 and low-prevalence AP +0.015042 but still regressed fold 3 -0.002505.
All site-varying weights produced the same split-specific cross-site calibration
failure, so both hard and smooth site-conditioned routing are retired.

The next regularization path is uniform global shrinkage of the fixed pair's
logit residual. This preserves all cross-site ordering induced by attenuation,
excludes the failed full-strength control, and tests whether the independently
observed rare-site signal survives at a safer magnitude. Selection remains
folds 3+4; any pass must be confirmed on untouched folds 0/1.

### 2026-07-30: Label-free unseen-site router frozen

The next development experiment keeps the failed fold-2 pair entirely fixed and
adds only an inference-time route. Known training physical locations remain
bitwise on the current score. For an unseen site, the router computes two
label-free statistics from current scores: number of available scenes and the
fraction exceeding the already frozen folds-3/4 matched-FPR threshold
0.1672813993. The pair applies only when both a maximum-history and
maximum-predicted-positive-rate gate pass.

Six maximum sizes and seven predicted-positive-rate limits define 42 candidates.
True <=5% site prevalence remains evaluation-only. Candidates must route at least
10% of all rows and 50% of low-prevalence rows/sites, retain all earlier whole/
rare-site fold, sensor, AP, and recall gates, and obtain positive paired-site AP
lower bounds in both views. The four-candidate smoke exercised label-free routing,
coverage checks, candidate scoring, and both bootstrap paths. Fold 2 is referenced
only through its committed failure report and is not reloaded; folds 0/1 and
external inputs remain untouched.

### 2026-07-30: Uniform global shrinkage protocol frozen

The next experiment removes site-conditioned weighting entirely and applies one
global scalar to the fixed pair's logit residual. The frozen grid is 0.125, 0.25,
0.375, 0.50, 0.625, and 0.75. Zero is excluded because it is the unchanged
current-score control; one is excluded because the full-strength pair failed the
independent whole fold-2 view. Selection maximizes the smaller of the whole and
low-prevalence paired-site AP lower bounds after all point gates pass.

The two-value smoke exercised hash validation, exact fixed-pair reconstruction,
uniform logit shrinkage, fold/sensor/recall gates, and paired site bootstraps.
At shrinkage 0.25, development AP improved +0.001476 overall and +0.005908 on
the low-prevalence proxy. Every fold and sensor delta was positive, matched-FPR
recall changed +0.000656/0.000000, and the preliminary 100-replicate site
interval lower bounds were +0.000315/+0.000475. These smoke numbers are not
promotion evidence; the committed protocol requires 10,000 replicates over all
six candidates. Fold 2 remains report-only, and folds 0/1 and external inputs
remain untouched.

### 2026-07-30: Uniform global shrinkage passes development gates

Five of six frozen shrinkages passed every point and 10,000-replicate paired-site
bootstrap gate. The preregistered selection rule chose shrinkage 0.50, equivalent
to anchor strength 0.10 and booster strength 0.025.

On all 17,745 development rows, AP changed 0.904076 -> 0.906798
(+0.002722). Fold-3/fold-4 AP changed +0.000307/+0.001782 and
Sentinel-2/Landsat AP changed +0.003671/+0.000586. Matched-FPR recall was
unchanged at 0.961942 with FPR 0.071266. The physical-site paired AP interval
was [+0.000761,+0.004360].

On the 11,344-row low-prevalence proxy, AP changed 0.708187 -> 0.718851
(+0.010664). Fold-3/fold-4 AP changed +0.018570/+0.009332 and
Sentinel-2/Landsat AP changed +0.011916/+0.004392. Matched-FPR recall was
unchanged at 0.906780 with FPR 0.071263. The physical-site paired AP interval
was [+0.001255,+0.021951].

The compact report SHA-256 is
`f3228425fe595c2363d5b3d17831aee103b022f775088bfd8ac0c4d96075b57f`.
This pass authorizes a separately frozen folds-0/1-excluding adapter, label-free
feature extraction, and one-shot confirmation of exactly shrinkage 0.50. It does
not authorize fold-2 reuse or external/exact-paper access.

### 2026-07-30: Folds-0/1-excluding representation fit frozen

The one-shot confirmation representation will use a new dense-Prithvi adapter
trained only on folds 3 and 4. Folds 0 and 1 are the held confirmation set, and
fold 2 is explicitly forbidden because its labels already influenced the
architecture decision. The fixed endpoint remains three 24,576-sample epochs,
batch size 8, mask strength 0.5, exact zero scene/pixel correction at
initialization, and balanced 0.25 request mass for each plume x sensor stratum.

The 32-sample GPU smoke constructed only folds-3/4 records, reported exact
identity maxima 0.0/0.0, balanced all four request strata at 0.25, completed one
finite epoch, and did not create an artifact or report. The frozen artifact will
record fit_folds=[3,4], held_folds=[0,1], and forbidden_folds=[0,1,2].
Folds 0/1/2, fresh inputs, and exact-paper inputs remain unopened.

### 2026-07-30: Folds-0/1-excluding dense adapter endpoint recorded

The frozen RTX 5070 fit completed all three epochs on 17,745 folds-3/4 rows.
Loss changed 0.331513 -> 0.242505 -> 0.227087. Exact initialization
corrections were 0.0 for both pixel and scene outputs, all 122 saved tensors are
finite, and the four plume x sensor request strata each retained 0.25 mass.
No held-fold record was constructed.

The ignored artifact is 21,926,603 bytes with SHA-256
`e8dde7a07fd93db3185891d63955e1c9c6a9681fe44c98208ec53661d5ced460`.
Its embedded contract records fit_folds=[3,4], held_folds=[0,1],
forbidden_folds=[0,1,2], seed 20266800, mask strength 0.5, scene strength
0.0, and training-protocol SHA-256
`55bb893d2d4949490707079828f01d7a7b8066ba9f13c1096f12667fcacfd229`.
The compact fit report SHA-256 is
`a435292b1eb52f2efb94ab543e0fb4ec0903767d601f5de7c77b08baa331f702`.
Folds 0/1/2 and external inputs remain unopened.

### 2026-07-30: One-shot folds-0/1 extraction and evaluator frozen

The extractor binds both held folds to the same folds-3/4 adapter and enforces
the artifact's held_folds=[0,1], forbidden_folds=[0,1,2], fit_folds=[3,4],
protocol identity, mask strength 0.5, and exact scene-score floor. Its smoke
validated both job contracts and the 671-feature schema while constructing zero
held records and accessing zero held labels.

The evaluator fixes column 644 direction -1 with tanh, column 156 direction +1
with high-score weighting, full strengths 0.20/0.05, and global shrinkage 0.50.
Its development-only smoke exactly reproduced whole AP 0.9067981487096828,
low-prevalence AP 0.7188508777014572, both recall deltas, the candidate key, and
effective strengths 0.10/0.025 within 1e-12.

The one-shot gates require pooled AP gains >=0.001 whole and >=0.005
low-prevalence, nonnegative fold and sensor AP deltas, nondecreasing matched-FPR
recall, no actual FPR increase, and strictly positive 10,000-replicate paired
physical-site AP lower bounds in both views. The evaluator refuses to repeat or
overwrite its report. Fold 2 is not reaccessed, and fresh/exact-paper inputs
remain inaccessible.

### 2026-07-30: Global shrinkage rejected on one-shot folds 0/1

The frozen extractor encoded all 17,785 unique held rows (8,987 fold 0 and
8,798 fold 1) into the ignored 671-column cache. Feature and metadata SHA-256
values are
`5683cb10a535ad1f4d83fb8108502f897b19a48bcd47a42667c4a13f0d630ad6`
and
`2041dae9edf3aa203b6dfdce21938e8b93f35716b290ce995752c3dc1addb09e`.
The compact receipt SHA-256 is
`534aa3f1f117f708c8d1d68820f5c2787e407093fc74061b6099bf2b43a703d4`.
The fixed evaluator then ran exactly once.

On the complete held view, AP changed 0.903158 -> 0.903086
(-0.000072). Fold-0/fold-1 deltas were -0.000047/+0.000102 and
Sentinel-2/Landsat deltas were -0.000239/+0.000228. Matched-FPR recall and FPR
were unchanged; the paired physical-site AP interval was
[-0.000781,+0.000897].

On the 11,529-row, 143-positive, 164-site low-prevalence view, AP changed
0.691129 -> 0.690931 (-0.000198). Fold-0/fold-1 deltas were
+0.003171/-0.008468, while both sensor deltas remained slightly positive.
Recall and FPR were unchanged; the paired-site interval was
[-0.008543,+0.005033].

The evaluator report SHA-256 is
`7e921fa2f7134bd5bd82477ab76c345eefbccc4053eadf2815b265f3754a63eb`.
Uniform shrinkage and the entire two-feature dense-residual family are retired.
The development gains did not generalize across independently trained adapter
seeds and physical-site folds. Folds 0, 1, and 2 may not be reused for selection;
fresh and exact-paper inputs remain inaccessible.

### 2026-07-31: Counterfactual temporal-physics representation frozen

The dense residual failure retires calibration on adapter-specific embeddings.
The next hypothesis instead changes the observable evidence: genuine methane
attenuation should be directional in time, whereas persistent facility texture
and many land-cover artifacts should remain under a same-scene control. A new
label-free extractor therefore scores each fold-3/4 scene with the frozen
released MARS-S2L U-Net in four views: factual target/reference, swapped
reference/target, target/target, and reference/reference. It records the spatial
response of every view plus explicit factual-minus-control maps.

The representation also adds two-way low-frequency Fourier-amplitude alignment
on pooled bitemporal reflectance, high-pass SWIR change, gas-specific B12-minus-
B11 change, and nuisance magnitudes. This is a fixed physical input transform
inspired by FeaSpect (CVPR 2025), whose central finding is that aligning
low-frequency amplitude can suppress environment-driven pseudo-change; it is
not presented as a reproduction of that paper's trainable latent module. The
released 16-channel mixed Sentinel-2/Landsat contract remains intact upstream.

The frozen extractor emits 28x64x64 float16 maps. Its four-row CUDA smoke covered
both labels and sensors, produced finite 4x28x64x64 values, and did not write a
cache. The full extraction is restricted to folds 3 and 4; folds 0/1/2,
exact-paper data, and fresh external data are forbidden. The eventual ranker
must improve the already frozen spatial-Prithvi ensemble, not an obsolete weaker
scene head. Bulk outputs remain ignored and only a compact hashed receipt will
be committed.

The first full attempt was stopped before 800 rows because batch 8 left most of
the 12 GB GPU unused and projected to several hours. The ignored partial memmap
was deleted without producing a receipt. A label-free 128-row benchmark then
processed batch 64 at 1,561 rows/minute with 5.67 GiB peak CUDA allocation. The
runtime-only contract is re-frozen at batch 64 and eight loader workers; the
eligible folds, transforms, model, and representation are unchanged.

### 2026-07-31: Counterfactual cache finalized and ranker selection frozen

The accelerated extraction completed all 17,745 fold-3/4 rows, but the process
encountered the known WSL DrvFS visibility race after atomically renaming the
final files: Windows exposed both outputs while the process's immediate
`stat()` failed, so it wrote no receipt. A separately committed finalizer bound
the generator commit, re-hashed every frozen input, scanned every value, and
matched every sample, physical site, label, sensor, and fold to manifest order.

The ignored 17,745x28x64x64 float16 cache is 4,070,277,248 bytes with SHA-256
`f2698d01f034a8836afe82f27857da5bdc9657a269d933dde4c5c5cb08a9d27a`;
metadata SHA-256 is
`8fcc864ae81ba48e2057ffd4f167f73ed11310fc6a9f2db274818806cb069463`.
All values are finite in [-3.099609375, 9.0], folds contain 8,799/8,946 rows,
and the labels contain 16,221 no-plume and 1,524 plume scenes. The compact
receipt SHA-256 is
`19b21e05e9bbde4cc70ececcd7ef2440ee59843787f8d21a25a15b8594b07955`.

Ranker selection is now frozen before any full optimization. Three scientific
ablations isolate directional frozen-teacher evidence, the complete
counterfactual-physics representation, and physics without teacher responses.
Each uses a compact residual spatial encoder, site-cell weighting, BCE plus
hard-pair ranking, two fixed seeds, and honest fold-3-to-4/fold-4-to-3 scoring.
Only four predeclared logit-blend weights may supplement the current strongest
spatial-Prithvi score (exact AP 0.676102 full and 0.467027 test-only).

Promotion requires positive paired-site AP interval lower bounds in both the
whole and <=5% site-prevalence views, >=0.001/>=0.005 point AP gains,
nonnegative matched-FPR recall, and nonnegative AP in every fold and sensor.
The 16-row GPU smoke covered all four label x sensor strata, completed a finite
epoch, and produced finite scores. Folds 0/1/2, exact-paper inputs, and fresh
external inputs remain inaccessible.

### 2026-07-31: Observational counterfactual ranker rejected

No frozen ablation passed. At its least harmful 0.05 blend, directional-teacher
evidence improved whole-view AP by +0.000909 but reduced matched-FPR recall by
-0.001969; its paired-site interval was [-0.000159,+0.001647]. Fold 3 and
Landsat AP were slightly negative. More importantly, the low-prevalence proxy
changed -0.000458 AP with interval [-0.006864,+0.006634].

The complete counterfactual-physics model changed whole AP +0.000745 and rare
AP -0.002243; physics without teacher responses changed +0.000741/-0.002407.
Larger blends increased whole AP to as much as +0.002725 but progressively
harmed rare-site AP (down to -0.010195), so attenuation cannot repair the target
domain. Report SHA-256 is
`faae5ec16a0caa740564073ca5f7cdbbe8d65a1692dec552662e95c5cfa3b76f`;
the ignored cross-fit score cache SHA-256 is
`1fcf35fa2e61976bac038ff7797ac5ce4f502c3fdb74698e2c960751ced41f40`.
No artifact was written and folds 0/1/2 or external data were not accessed.

This rejects observational counterfactual features as sufficient, but not the
causal hypothesis. The missing intervention is a paired positive on the same
no-plume background. The next experiment will use the paper-compatible,
wind-matched MODTRAN-LUT simulator to inject fit-fold plume fields into fit-fold
no-plume scenes, then train the directional ranker with those same-background
positive interventions. Each cross-fit branch must keep both source plumes and
backgrounds inside its fitting fold. This targets the rare-site failure directly
by breaking the association between facility appearance and plume labels.

### 2026-07-31: Same-background methane interventions frozen

The simulator cache is fixed at 4,096 positives: 1,024 Sentinel-2 and 1,024
Landsat interventions from each of folds 3 and 4. Every intervention uses a
clear, onshore, <=9 m/s no-plume background and a real CH4 field from the same
fitting fold. Source and target wind speed must differ by <=1.5 m/s with no
nearest-neighbor fallback; the plume is rotated to target wind, scaled uniformly
0.5-1.5x, and injected into B11/B12 through the released integrated-
transmittance LUT. At least half of injected plume pixels must remain observable.

The four-row CUDA smoke accepted one example per fold x sensor stratum. Maximum
wind-speed mismatch was 1.4029 m/s and minimum visible fraction was 1.0. All
four simulations passed through the same frozen U-Net and 28-channel
counterfactual transform as real data. One attempted sample failed closed on an
expected simulation exception and was replaced before inference. The generator,
all code/data/LUT hashes, sample counts, and deterministic seeds are now frozen
before the full cache is created. Folds 0/1/2 and external inputs remain
inaccessible.

### 2026-07-31: Causal intervention cache completed; paired training frozen

All 4,096 same-background interventions completed in 10.8 minutes. Each fitting
fold contributes exactly 2,048 examples, split equally between Sentinel-2 and
Landsat. The maximum source-target wind-speed mismatch is 1.499951 m/s,
minimum visible plume fraction is 0.503289, and realized scale range is
[0.500021,1.499948]. Forty-nine attempts failed the wind/visibility gate and 20
failed closed on simulation exceptions; accepted rows never relaxed a gate.

The ignored 939,524,224-byte image cache SHA-256 is
`1486c272422454431bea712bd3214426d7e7356784f0f3c5f0301e070a49cf35`;
metadata SHA-256 is
`7b44c99b2dc851cd453d67bf1b4f276263bd624a933b1125911235d96c533ece`;
receipt SHA-256 is
`79a599bfd611a2a4fbf6376445f650f85b7d24994a7f4f33393ea4bdb1c079fe`.

The paired trainer fixes two intervention strengths. Both retain the previously
best directional-teacher feature set and actual-data hard ranking. `causal_weak`
uses 96 actual/32 simulated examples per step with 0.25 synthetic BCE and pair
weights; `causal_balanced` uses 64/64 with 0.50/0.50 weights. Every simulated
positive is compared against its exact real no-plume background after the same
spatial transform with a +0.5 logit margin. Two fixed seeds are mean-logit
ensembled. The GPU smoke completed finite actual, simulation, and causal-pair
losses and finite scores while rechecking the strict 1.499951 m/s cache bound.
Full selection remains restricted to folds 3/4 and the prior whole/rare gates.

### 2026-07-31: Standalone causal classifier rejected; causal signal retained

Neither intervention strength passed the joint gates. At blend 0.05,
`causal_weak` changed whole AP -0.000085 and rare-site AP +0.003243. The rare
gain was consistent by fold (+0.002150/+0.004289) and sensor
(+0.003609/+0.003247), and rare matched-FPR recall improved +0.008475, but its
paired-site interval still crossed zero [-0.000363,+0.007216]. Whole matched-FPR
recall fell -0.001312 and the whole interval was
[-0.000653,+0.000500]. Raising the blend to 0.10 passed the rare point floor at
+0.005441 but further harmed whole ranking and recall.

`causal_balanced` changed whole/rare AP +0.000478/+0.000451 at blend 0.05,
with both intervals crossing zero. No artifact was written. Report SHA-256 is
`2a3c5ade34816d88567bfe7c467e817fef7dff3ee33cbfb6fe38199290209b6f`;
the ignored cross-fit scores have SHA-256
`3faa53ac49c4eb21e47b0033de6a97a4d353fae9766487a08780446b42302bcc`.

The intervention is the first new representation to improve rare AP, recall,
both folds, and both sensors together, so the causal signal is not retired. The
failure is architectural: treating it as a complete probability classifier and
logit-blending it with the much stronger current score discards hard-boundary
ordering. The next controlled model will be zero-initialized and learn only a
bounded residual around each OOF current logit, while the same-background pairs
supervise the residual direction. This preserves the current ranker as the
explicit floor instead of asking a weaker standalone head to relearn it.

### 2026-07-31: Baseline-preserving causal residual frozen

The residual architecture reuses only the empirically supported `causal_weak`
intervention schedule and directional-teacher channels. Its final linear layer
is exactly zero-initialized, so before optimization candidate scores equal the
current spatial-Prithvi OOF scores bit-for-bit in mathematical precision. Actual
training BCE/hard-pair loss operates on `logit(current)+residual`; simulation
positives and exact backgrounds supervise residual direction with the prior
0.25/0.25 weights and +0.5 causal margin. A 0.01 residual L2 penalty makes
preservation explicit.

Two fixed seeds will produce honest fold-3-to-4 and fold-4-to-3 residuals. Only
four uniform strengths [0.10,0.25,0.50,1.00] are eligible; no site router or
prevalence feature exists. The smoke verified exact zero initialization, finite
post-update residuals, finite actual/simulation/pair/L2 losses, and no artifact
creation. The full run retains the strict whole and rare-site gates and cannot
access folds 0/1/2 or external inputs.

### 2026-07-31: Baseline-preserving causal residual rejected

The two-seed honest fold-3-to-4/fold-4-to-3 run rejects the residual family. At
the minimum strength 0.10, whole-view AP changed -0.000222, matched-FPR recall
changed -0.001312, and the paired-site interval was
[-0.000685,+0.000033]. Both held folds and both sensor strata lost AP. The
rare-site view changed -0.001485 AP with tied recall and interval
[-0.004714,+0.002408].

Increasing the residual strength made the ordering progressively worse. At
strengths 0.25/0.50/1.00, whole AP changed
-0.000636/-0.001622/-0.004444, and rare AP changed
-0.003154/-0.009761/-0.026820. The whole-view paired interval was strictly
negative by strength 0.50. No candidate passed, no artifact was written, and
folds 0/1/2, external inputs, and exact-paper inputs remained inaccessible.

The compact report SHA-256 is
`8c816b02401f234f4b3b30cc15ee94383daa9acc92d1fe7ff3dbc0cdf93e74d2`;
the ignored score-cache SHA-256 is
`08900f3868661fe24ecf368e9955db74d1b2b8a58119a8742d0eceaf6abcc605`.
The result retires zero-initialized post-hoc causal residual ranking. Same-
background interventions remain useful mechanistic evidence, but their learned
score does not improve the already strong spatial-Prithvi ordering. Any further
use must alter representation learning upstream or add genuinely new label
support; another scalar blend or bounded correction is not justified.

### 2026-07-31: Physical-radiometry Prithvi correction frozen

The upstream-representation audit found a concrete preprocessing mismatch in
all earlier Prithvi transfer experiments. The released MARS loader converts its
uint16 reflectance digital numbers to U-Net inputs by dividing by 5,000. The
pinned Prithvi configuration, however, supplies means and standard deviations
in reflectance-times-10,000 digital-number units. The earlier transfer extractor
multiplied the MARS tensor by 10,000, which reconstructed twice the original
digital number. The physically correct inverse conversion is multiplication by
5,000. This conclusion follows directly from the pinned local MARS loader,
upstream MARS-S2L loader, Prithvi configuration, and raw-data unit contract; it
does not use labels or performance outcomes.

A versioned extractor now changes only that multiplier. It keeps the frozen
Prithvi-EO-2.0 tiny temporal ViT, reference-then-target chronology, 128-pixel
resize, HLS band slots, temporal/location coordinates, invalid-pixel policy,
and 3,072-feature schema unchanged. The Sentinel-2 B08 versus HLS narrow-NIR
transfer mismatch remains explicit. Earlier caches and reports are preserved
as valid evidence for the legacy 2x-radiometry transfer contract rather than
silently rewritten.

Selection is frozen to a two-way fold-3/fold-4 physical-site cross-fit before
the outcome-bearing cache is generated. Three predeclared feature views (CLS,
temporal change, and all Prithvi features), three logistic regularizations, and
five small logit blends may supplement the current spatial-Prithvi score.
Promotion requires at least +0.001 pooled AP, nonnegative pooled recall and
per-sensor AP, positive AP on both folds, and a positive lower bound from a
10,000-replicate paired-site AP bootstrap. A four-row CUDA smoke produced
finite 3,072-wide features; two focused tests and static checks pass. Folds
0/1/2, external data, and the paper test remain excluded. The complete frozen
contract is `configs/mars_prithvi_physical_scene_probe_protocol.json`.

### 2026-07-31: Physical-radiometry Prithvi cache finalized

The corrected extractor completed all 17,745 fold-3/4 rows in 1,502 seconds
with a single-process loader. Two faster multiprocess launches failed before
writing output because WSL's 16 GB host-memory limit killed duplicated raster
workers; the safe run changed only batch/loader execution settings and used the
identical frozen feature transform. A 512-row cross-check had already shown
byte-identical output between worker configurations.

The ignored 17,745x3,072 float16 cache is 99,828,149 bytes with SHA-256
`fefbe31fabda56fac01d6acefe0b3d8d8034c98496f6312d70e22dd3c6518108`.
Every feature is finite in [-21.0625, 22.3125]. Independent finalization matches
all 17,745 sample IDs, physical groups, labels, sensors, and folds to the
canonical manifest in exact order. Fold counts are 8,799/8,946; label counts
are 16,221 no-plume and 1,524 plume; sensor counts are 14,963 Sentinel-2 and
2,782 Landsat. The compact acquisition receipt is committed before any probe
metric is computed. Folds 0/1/2 and paper-test outcomes remain excluded.

### 2026-07-31: Physical-radiometry Prithvi probe rejected

None of the 45 predeclared candidates passes the fold-3/4 promotion gates. The
least harmful candidate uses temporal-change features, C=0.01, and the minimum
0.025 logit blend. It changes pooled AP versus the current spatial-Prithvi
score by only +0.000091 and leaves matched-FPR recall unchanged. Its paired-site
AP interval is [-0.000299,+0.000601], so the effect is not distinguishable from
zero. Fold 3 AP changes -0.000050 while fold 4 changes +0.000106; pooled
Sentinel-2/Landsat AP changes are both positive but only
+0.000094/+0.000117. The candidate therefore fails the +0.001 pooled point
floor, the both-fold sign gate, and the positive bootstrap lower-bound gate.

A reporting-only replay of that exact selected specification against the
legacy 2x-radiometry cache gives +0.000238 AP but -0.000656 recall, with fold 3
again negative. Correct physical scaling removes the small recall loss but does
not unlock a useful representation. This preserves an important paper caveat:
the earlier Prithvi results used an erroneous radiometric transfer, yet fixing
it does not alter the architectural conclusion because the probe remains almost
perfectly redundant with the current score.

The compact selection report SHA-256 is
`f12a36af99d6b399f2292d45d68cd93e446220388aedea0f9936315bf76e2bcd`.
No candidate is promoted and fold 2 is not extracted. The next representation
branch can now test a wavelength-conditioned foundation model rather than
another Prithvi scalar head.

### 2026-07-31: Exact DOFA-v2 sensor-aware representation frozen

The next branch changes the upstream representation rather than fitting another
head over Prithvi. DOFA's peer-reviewed design uses a wavelength-conditioned
hypernetwork to generate a patch embedding for arbitrary channel sets and was
jointly pretrained across five Earth-observation modalities/sensors. The paper
reports transfer to unseen sensors (<https://arxiv.org/abs/2403.15356>), which
directly targets the mixed Sentinel-2/Landsat weakness. Landsat was not a native
pretraining sensor in the published five-source mixture, so this remains an
explicit unseen-sensor hypothesis rather than an assumed in-domain advantage.

The official 421,811,730-byte DOFA-v2 ViT-base checkpoint is pinned at model
revision `7a5219e48d2f8848511b0fabea7920a8836bc480`, with SHA-256
`e1be9d50fb3e4e3640e337d098b92d67797eaf2a579de3b7a1e363095885314d`.
The public v2 architecture had been removed from the current source tree, so it
was recovered from official commit `c850a1623413d4c1fc1134202641e3059fe4ab50`
and paired with the corrected wavelength layer at current pinned commit
`0cfb7e1099f4d4c4022946ff7862c7cd7b8411b9`. The local 105,433,856-parameter
implementation loads all 196 checkpoint tensors strictly with no missing or
unexpected keys. On a fixed six-channel tensor, all four 768x16x16 maps are
bit-identical to the official historical source. Code is MIT licensed; the
checkpoint model card is CC-BY-4.0.

Each target/reference frame is converted from the MARS U-Net scale back to
physical reflectance, then to the official DOFA 0-255 normalization scale.
Sentinel-2 and Landsat receive distinct six-band center-wavelength lists; the
full upstream 16-channel contract remains unchanged. The 10,752 frozen scene
features contain final reference/target context and four-depth temporal-change
magnitudes, dispersion, and signed extremes. Three predeclared feature views
are mapped to 2,048 dimensions by fixed sparse random projections and scored by
regularized linear probes in an honest fold-3/fold-4 cross-fit. Promotion is
against the current spatial-Prithvi score with the same strict point, fold,
sensor, recall, and paired-site interval gates.

The official-equivalence audit, four focused tests, and static checks pass. A
four-row CUDA smoke covers every label x sensor cell and both folds with finite
features. Batch 256 with a single loader process is frozen after using 5.40 GiB
peak CUDA allocation; larger batch 384 was slower and used 8.30 GiB. The full
protocol is `configs/mars_dofa_v2_scene_probe_protocol.json`. Folds 0/1/2,
external outcomes, and paper-test outcomes remain inaccessible.

### 2026-07-31: DOFA-v2 folds-3/4 cache finalized

The frozen sensor-aware extractor completed all 17,745 authorized rows in
1,966.63 seconds with batch 256 and a single loader process. Mixed-sensor peak
CUDA allocation was 6.31 GiB and peak host use was 5,138,332 KiB. The ignored
17,745x10,752 float16 cache is 352,234,214 bytes with SHA-256
`2c43352e92f96549d64188c78318fe606d55552aa9587aace1a195e8dfba66fe`.

Independent finalization scans every value as finite in [-722.0,720.5] and
matches all sample IDs, physical groups, labels, sensors, and folds to the
canonical manifest in exact order. Counts remain 8,799/8,946 by fold,
16,221/1,524 by no-plume/plume label, and 14,963/2,782 by Sentinel-2/Landsat.
The cache embeds the pinned checkpoint, foundation revision/receipt, manifest,
fold protocol, wavelength lists, and radiometric multiplier. Its compact
receipt is committed before any projected feature or outcome metric is
computed. Folds 0/1/2 and paper-test outcomes remain excluded.

### 2026-07-31: DOFA-v2 point gates pass; projection stability confirmation frozen

The first frozen DOFA-v2 selection evaluates 45 predeclared feature-family,
regularization, and blend combinations. Its selected `change_extreme`, C=0.01,
0.05-blend probe is the first new representation in this research phase to pass
every pooled point, fold, sensor, and recall gate. Relative to the current
spatial-Prithvi score, pooled AP rises +0.001087 with unchanged matched-FPR
recall. AP changes are +0.000910/+0.000964 on folds 3/4 and
+0.001266/+0.000887 on Landsat/Sentinel-2. The paired-site AP estimate is
+0.001113, but its 95% interval [-0.000223,+0.002588] crosses zero. The result
is therefore promising but not sufficient to touch fold 2.

One variance-control confirmation is frozen before observing any further model
outcomes. It holds the selected feature family, C, 0.05 blend, 2,048-dimensional
sparse projection, cross-fit, and all gates fixed. Five entirely new projection
seeds are averaged in log-odds space; the favorable seed used during selection
is deliberately excluded. Individual seeds are diagnostics only, and exactly
one five-seed aggregate is eligible. Passing still requires every original
point/fold/sensor/recall gate plus a strictly positive lower bound from 10,000
paired physical-site bootstrap replicates. Failure rejects the branch without
extracting fold 2. The frozen contract is
`configs/mars_dofa_v2_projection_ensemble_protocol.json`.

### 2026-07-31: DOFA-v2 projection ensemble rejected on pooled recall

The fixed five-new-seed mean-logit ensemble makes the DOFA AP effect stronger
and statistically distinguishable from zero. Pooled AP changes +0.001251 versus
the current spatial-Prithvi score; the paired-site estimate is +0.001240 with a
95% interval of [+0.000216,+0.002410]. AP remains positive on folds 3/4
(+0.000516/+0.001314) and both sensors (Landsat +0.000965, Sentinel-2
+0.001084). This is the first DOFA result to pass the strict AP uncertainty
gate, and it is +0.030042 AP above the released primary scene score.

It nevertheless fails the preregistered no-recall-regression gate. At the
pooled FPR=0.0713 operating point, recall falls by 0.001312, exactly two of the
1,524 positive scenes, despite fold-local recall increasing on both fold 3 and
fold 4. Four of five individual new projections also lose between one and four
pooled positives. This pattern exposes a cross-fit score-calibration/ranking
problem rather than evidence for a uniformly better decision rule. The branch
is rejected as frozen; fold 2 remains untouched. The JSON result was written
successfully at experimental commit `f69e8107` before a Markdown-only field-path
error stopped the command. The formatter is corrected without recomputing or
altering the recorded outcome; result SHA-256 is
`540c5d509b1340ae0a07bd0b89a07bca70b3ce85b8d8e9d865066ab122efdb96`.

### 2026-07-31: Recall-protected DOFA-v2 fusion frozen

The new-seed ensemble establishes that DOFA contributes a real AP signal but
also shows why an unconstrained global blend is unsafe: it perturbs examples at
the no-plume decision boundary. The next architecture changes the fusion rule,
not the representation or labels. Below a fixed current-model confidence gate,
the output is exactly the current score. Above the gate, the current score is
rescaled locally, blended with the fixed five-seed DOFA probability in log-odds
space, and mapped back above the same gate. This is a sample-wise deployable
function; no batch ranks, test distribution, or labels enter inference.

Three fixed gates (0.25, 0.50, 0.75) and three fixed DOFA weights (0.05, 0.10,
0.20) define nine eligible development candidates. Every gate must remain at
least 0.05 above the current fold-3/4 matched-FPR threshold. Promotion requires
the original point/fold/sensor/bootstrap gates plus exact preservation of the
pooled operating confusion counts. All candidates reuse the five new projection
seeds and must first reproduce the prior unconstrained result to 1e-14. The
frozen contract is `configs/mars_dofa_v2_protected_fusion_protocol.json`; folds
0/1/2 and external/paper-test outcomes remain untouched.

### 2026-07-31: Protected fusion passes development gates; held for normalization audit

The selected gate=0.50, DOFA-weight=0.05 protected fusion passes every frozen
fold-3/4 numerical gate. It improves AP by +0.001208 over the current score,
with paired-site mean +0.001215 and 95% interval
[+0.000362,+0.002204]. Fold AP changes are +0.000543/+0.001331 and sensor AP
changes are +0.000933 Landsat/+0.001084 Sentinel-2. The pooled and both
fold-local operating confusion matrices are identical to the current model, so
matched-FPR recall changes exactly zero. Six of nine predeclared candidates
pass; the fixed stability-first rule selects 0.50/0.05 by the highest bootstrap
lower bound. The result report SHA-256 is
`95f292353d6f9e4d0d2416e78d46aaf3a7003e9c3d0ec23a2d499fbfa2b0a044`.

A deployment-equivalence audit before opening fold 2 finds that the inherited
DOFA probe normalizes each held cohort with that cohort's own unlabeled mean and
standard deviation, both before and after sparse projection. This is label-free
but transductive test-cohort adaptation: a single production scene cannot be
scored without a cohort distribution, and an official-test result would depend
on test-set composition. The protected fusion itself is sample-wise, but its
upstream DOFA probability is not yet. Therefore this passing result is retained
as a transductive ablation and does not authorize fold-2 access. A train-fitted
normalization confirmation must pass the same gates first.

### 2026-07-31: Train-fitted DOFA-v2 normalization confirmation frozen

The deployment audit produces one new two-candidate protocol before outcomes.
Both candidates hold the selected change-extreme features, C=0.01, five new
projection seeds, mean-logit aggregation, gate=0.50, and DOFA weight=0.05 fixed.
The only choice is whether fit-fold statistics are global or stratified by the
known sensor family. At both the raw-feature and projected-feature stages, the
mean and standard deviation are learned exclusively from the fit fold and
applied unchanged to the held fold. Neither held feature statistics nor held
labels enter fitting.

Promotion repeats every point/fold/sensor/bootstrap and operating-count gate.
The selection rule prioritizes a passing paired-site lower bound, with global
normalization preferred only as the final tie-break. This is the last required
deployment-equivalence check before any DOFA fold-2 access. The frozen contract
is `configs/mars_dofa_v2_train_fitted_normalization_protocol.json`.

### 2026-07-31: Train-fitted DOFA-v2 normalization passes

Both deployment-safe normalization candidates pass every frozen fold-3/4 gate.
The selected global train-fitted mode improves pooled AP by +0.001412 with
unchanged pooled and fold-local operating recall/counts. Its paired-site mean is
+0.001421 with 95% interval [+0.000526,+0.002462]. AP changes are
+0.000591/+0.001414 on folds 3/4 and +0.000949/+0.001400 on
Landsat/Sentinel-2. Sensor-train-fitted scaling independently passes at
+0.001225 AP with interval [+0.000347,+0.002109], making the conclusion robust
to the deployment-safe normalization choice.

Global scaling wins the preregistered stability ranking and is now the frozen
candidate: change-extreme DOFA-v2 features, C=0.01, five new projection seeds,
mean-logit aggregation, gate=0.50, weight=0.05, with every scaler fit on
training rows only. Result SHA-256 is
`23e9124a5f0b5a9a788a46a6fd07753501a8b83c880b0a4e3f24e680c92074b2`.
This pass authorizes a separately frozen one-shot fold-2 extraction/evaluation;
fold 2 remains unaccessed at this point.

### 2026-07-31: One-shot DOFA-v2 fold-2 package frozen

The exact global-train-fitted candidate is now bound into a one-shot fold-2
protocol before any fold-2 DOFA feature or outcome access. A dedicated wrapper
invokes the already verified sensor-aware extractor for fold 2 only, validates
the 10,752-feature schema and all provenance invariants, and writes a compact
hash receipt while leaving the bulk cache ignored. Extraction computations do
not use labels; labels copied for later alignment remain inaccessible until the
evaluator is frozen.

The separately hashed evaluator first has to recompute the fixed folds-3/4
candidate and match its pooled, fold, and sensor deltas to 1e-14. Only then may
it consume the receipt, fit five projection/scaler/probe stacks on folds 3+4,
and score fold 2 once. Promotion requires at least +0.001 AP, nonnegative AP on
both sensors, identical operating confusion counts, higher AP and recall than
the released primary score, and a strictly positive paired-site AP lower bound.
It refuses to overwrite an existing result. The package is frozen in
`configs/mars_dofa_v2_fold2_protocol.json`; fold 2 is still untouched pending a
successful development-only reproduction smoke.

### 2026-07-31: Fold-2 DOFA-v2 features finalized

The pre-access evaluator smoke reproduces all six frozen pooled, sensor, and
fold development deltas exactly. The authorized fold-2 extractor then processes
8,833 scenes from 153 physical groups in 929.7 seconds. The cache contains 797
plume and 8,036 no-plume rows, split across 7,444 Sentinel-2 and 1,389 Landsat
scenes. All 8,833x10,752 float16 features are finite in [-725.0,721.5]. Peak
CUDA allocation is 6.20 GiB.

The 175,383,631-byte cache remains ignored and has SHA-256
`9ead8028d98112bd9a9d494a08104700fa580b2258c7c2c15bb4a38a0d6723ab`.
Its compact receipt is committed before outcome evaluation. No fold-2 label was
used by feature extraction or model fitting; the already-frozen evaluator may
now perform the single authorized outcome read.

### 2026-07-31: Protected DOFA-v2 fusion rejected on one-shot fold 2

The frozen train-fitted candidate preserves the fold-2 operating point exactly
but does not transport its development AP gain. Candidate AP is 0.916801 versus
current 0.916603, a delta of only +0.000198; the 153-site paired estimate is
+0.000203 with 95% interval [-0.000343,+0.000933]. The effect is positive but
fails both the +0.001 point floor and positive-lower-bound gate. Recall remains
0.956085 with the identical 762 TP, 35 FN, 572 FP, and 7,464 TN. Sensor AP
deltas are also merely +0.000067 Landsat and +0.000276 Sentinel-2.

The candidate still exceeds the released primary scene score on fold 2 by
+0.036202 AP and +0.030113 matched-FPR recall, but the comparison that governs
promotion is the stronger current spatial-Prithvi score. The fixed branch is
retired before external or official-test evaluation, and fold 2 will not be
reused for DOFA selection. Result SHA-256 is
`b6bd52fed2cf8444ac9fccb05011f8daddc285db706e791ee93f3fd9c5b505dd`.

Together with earlier dense and causal failures, this rules out another scalar
scene-probe iteration as the next best use of compute. The subsequent branch
must learn a transferable spatial plume representation or add genuinely new,
source-disjoint supervision rather than post-hoc blending another frozen scene
summary.

### 2026-07-31: Spatial DINOv3 methane-gated branch selected for development

The fold-2 DOFA result closes the scalar-probe branch. The next experiment is a
spatial, identity-safe adapter whose semantic stream is a frozen DINOv3
ViT-S/16 and whose methane stream retains the MARS temporal/spectral physics.
Target and reference visible bands will be encoded separately at transformer
blocks 2, 5, 8, and 11. Signed and absolute temporal feature differences keep
the 16x16 token geometry; a small methane-enhancement network will gate those
features before a residual segmentation and scene head. Both heads must be
zero-initialized so an untrained candidate reproduces the current segmentation
logits and cross-fitted scene score exactly.

This choice is grounded in three complementary primary sources. DINOv3 reports
dense pretrained visual features and exposes intermediate transformer maps
([Siméoni et al., 2025](https://arxiv.org/abs/2508.10104)). Quintero et al.'s
multimodal methane segmentation model combines intermediate DINOv3 features
with a learned methane-enhancement gate and a SegFormer-style decoder
([Quintero et al., 2026](https://arxiv.org/abs/2606.26416)). The scene objective
will emphasize the benchmark's low-FPR region using a pairwise partial-AUC
surrogate, following the distributionally robust pAUC formulation of
[Zhu et al., 2022](https://proceedings.mlr.press/v162/zhu22g.html). These papers
motivate the components; they do not establish performance on the MARS-S2L
contract, so promotion still requires the preregistered physical-site gates.

The exact `vit_small_patch16_dinov3.lvd1689m` checkpoint was acquired from the
official timm Hugging Face repository at immutable revision
`3bf4720a82ec2066db88137180ff1f83a675cef0`. The 86,362,376-byte safetensors
file has SHA-256
`2a1ec16ae28ffa07bc0ead0241ee7df9fc26451fe6f9f839b7b3afa0a906b040`.
A strict load accounts for all 162 tensors and 21,586,944 parameters; blocks
2/5/8/11 each produce a finite 384x16x16 map for a 256x256 input. The checkpoint
and its support files remain under ignored `.research/`; only the acquisition
script and compact receipt are tracked. The DINOv3 license permits this use and
requires acknowledgment in any publication, which is now an explicit paper
requirement. No fold-2, external, or sealed official-test label is authorized
for this architecture's selection.

### 2026-07-31: DINOv3 methane-fusion pilot frozen before held-fold outcomes

The implemented pilot preserves the complete 16-channel mixed-sensor contract.
It encodes shared-contrast-stretched target/reference RGB with frozen DINOv3
blocks 2/5/8/11, then represents signed change, absolute change, and temporal
context at each depth. A second spatial stream consumes the already frozen
28-channel counterfactual cache: factual/swapped/self released-model maps,
MBMP and SWIR changes, frequency-aligned residuals, cloud, and observability.
A learned methane gate modulates the semantic stream before three deep
identity gates inject it into the released U-Net. The scene head is also an
identity-initialized residual around the current spatial-Prithvi score. Scores
below 0.50 are exactly untouched, while higher scores are adjusted in local
log-odds coordinates that cannot cross below 0.50.

Two GPU smokes were used only to establish feasibility, never to inspect a
held-fold outcome. At the frozen batch size of 8, two balanced fold-3-only
batches reproduce both released pixel logits and current scene scores with
maximum absolute error 0.0, complete a finite backward/optimizer step, and use
2,228,345,344 bytes peak CUDA memory. The adapter has 8,540,787 trainable
parameters; both DINOv3 and the released U-Net remain frozen.

The outcome protocol is now immutable in
`configs/mars_dinov3_methane_fusion_pilot_protocol.json`. It trains two
cross-fit endpoints (fold 3 to predict fold 4 and vice versa), each for exactly
two epochs of 4,096 site/label/sensor-balanced requests. The scene objective
adds a top-negative partial-AUC pair loss. Only residual strengths
0.10/0.25/0.50 are eligible. Promotion requires at least +0.001 pooled AP,
nonnegative AP on both sensors and both folds, unchanged operating confusion
counts, positive pooled and per-fold IoU changes, and strictly positive
physical-site bootstrap lower bounds for both AP and IoU. Failure cannot create
an artifact or authorize fold-2/external scoring.

### 2026-07-31: First DINOv3 fusion run exposes a zero-gradient scene bug

The frozen two-fold run completes without numerical or memory failure, but it
cannot pass promotion because all three candidate strengths reproduce the
current scene scores exactly: AP delta 0.0, recall delta 0.0, and paired-site AP
interval [0.0,0.0]. The spatial branch is nevertheless active. At strength
0.10 it improves pooled mask IoU by +0.003292 with paired-site 95% interval
[+0.000981,+0.005181], positive changes on both folds
(+0.005633/+0.001536), and unchanged scene operating counts. Strength 0.25
raises pooled IoU by +0.006070 with interval [+0.000841,+0.010278], also
positive on both folds, while 0.50 loses uncertainty support. No artifact is
written; result SHA-256 is
`81da27897942b0f78f3234ca64892bb4dde83e78a1774956ed4446298b342149`.

An implementation audit explains the exact-zero scene result. The protected
score function used `where(delta == 0, base, adjusted)` to guarantee bit-exact
identity. Because the zero-initialized scene head satisfies that condition for
every row, autograd follows the constant `base` branch and supplies exactly
zero gradient to the scene head. This is not caused by an empty protected
region: folds 3/4 contain 1,890 scores at or above 0.50, including 1,401 plume
scenes. The run therefore evaluates the spatial branch but is not a valid test
of the intended scene architecture.

The correction is algebraic and does not use the observed labels to change a
scientific choice: compute `base + (1-gate) * (sigmoid(logit(local)+delta) -
sigmoid(logit(local)))`. The two sigmoid terms are bit-identical at delta zero,
so initialization remains exact, while the first term retains its derivative.
All folds, inputs, sampling, epochs, losses, strengths, and promotion gates will
remain fixed in a separately hashed rerun. Fold 2 and external/test data remain
unauthorized.

### 2026-07-31: Scene-gradient correction frozen reproducibly

The corrected protected-score expression retains bit-exact identity for pixel
logits, scene deltas, and scene scores (all maximum errors 0.0) while producing
a nonzero scene-output gradient of 0.265380859375 in three repeated GPU probes.
The training smoke moves the scene residual away from zero and remains finite.
Model initialization, worker state, sampling, and dropout are now seeded before
construction for each cross-fit endpoint; the prior implementation seeded only
after random adapter weights had been created. Peak smoke allocation is 3.16
GB. The corrected trainer/model hashes and unchanged scientific contract are
frozen in the same versioned protocol path; outputs use the distinct
`*_gradient_fix` names so the failed result cannot be overwritten.

### 2026-07-31: Corrected protected-scene pilot rejected; spatial gain repeats

The seeded, differentiable rerun confirms that the implementation correction
worked: scene-delta L2 is 0.0071-0.0116 across endpoints/epochs instead of
exactly zero. The selected strength 0.10 nevertheless improves pooled AP by
only +0.000086, below the +0.001 floor, with paired-site 95% interval
[-0.000016,+0.000188]. Recall and all operating confusion counts remain exactly
unchanged. Sentinel-2 AP rises +0.000129, but Landsat changes -0.000020; fold 3
rises +0.000022 while fold 4 changes -0.000013. Larger strengths increase the
pooled point estimate monotonically (+0.000191/+0.000396) but worsen the same
transport asymmetry and still cross zero under site bootstrap. Result SHA-256
is `e60c8bd49d3309ac8592958d295a3f633fe765b610a6adf79c1f4ba29b174949`.

The spatial conclusion is independently repeatable. Strength 0.10 raises mask
IoU +0.002874 with paired-site interval [+0.000938,+0.004254] and positive fold
changes (+0.004610/+0.001576). Strength 0.25 raises IoU +0.005740 with interval
[+0.001372,+0.008621], again positive on both folds. No artifact is written
because the joint AP/IoU gate fails.

The diagnostic is now architectural: protected inference is appropriate for
the low-FPR operating guarantee, but using that protected score inside the
training losses removes all scene supervision below 0.50. The next bounded
variant will train the residual representation against the raw additive scene
logit for every development row while still applying the unchanged protected
mapping at evaluation and deployment. This uses no new outcome-dependent
threshold, strength, fold, or loss weight; it corrects the mismatch between the
training objective and inference safety rule. Fold 2 and all external/test
assets remain untouched.

### 2026-07-31: All-score scene objective frozen

The next bounded variant changes only the coordinate system used by the scene
losses. Scene BCE and the top-negative partial-AUC pair loss now see the raw
`base_logit + residual` for every training row; deployment and evaluation still
use the same 0.50-protected mapping. Pixel loss, architecture tensors, folds,
seeds, sampling, epochs, loss weights, residual strengths, and promotion gates
are unchanged.

The frozen GPU smoke again gives exact 0.0 pixel/scene identity and finite
optimization. Crucially, the raw scene objective produces a scene-output
gradient both for a base score below the protection gate (0.203125 at 0.05) and
above it (0.444091796875 at 0.75). Peak CUDA allocation is 3.16 GB. The output
paths are distinct `*_all_score_scene` artifacts/reports, and no held-fold
outcome was read before hashing this contract.

### 2026-07-31: All-score DINOv3 scene objective rejected

Training on raw additive logits increases the learned scene residual by more
than an order of magnitude (epoch-2 L2 0.1628/0.1092), but it does not improve
protected cross-fit ranking. The selected strength 0.10 changes pooled AP
-0.000113 with paired-site interval [-0.000437,+0.000319], including fold
changes -0.000065/-0.000229 and sensor changes -0.000027 Landsat/-0.000226
Sentinel-2. Recall and operating counts remain identical. Higher strengths
worsen pooled and fold AP. Result SHA-256 is
`e43151c2ce8df8192a3acc1828204084615ef36b8ebf5df01b0a9ba94df1dc08`;
no artifact is written.

The mask result repeats a third time: strength 0.10 improves pooled IoU
+0.002786 with paired-site interval [+0.000789,+0.004293] and positive fold
changes (+0.004575/+0.001449). Strength 0.25 again gives the larger point gain
(+0.004920) with a positive lower bound (+0.000508), although the protocol's
joint rank correctly selects 0.10 because all scene variants fail. The DINOv3
methane-gated branch is therefore retained as a validated segmentation research
direction, while its learned scene head is retired. It will not be forced into
the current scene score or evaluated on fold 2/external/test data without a new
independent scene rationale.

### 2026-07-31: Segmentation-only dense-surrogate branch frozen

The next experiment decouples the repeatable spatial gain from the retired
learned scene head. Scene BCE, learned pair ranking, and scene-residual penalties
are all zero-weighted; DINOv3 fusion is trained only by pixel focal/Dice,
8x8 occupancy, asymmetric no-plume/plume preservation, and correction L2.
At evaluation, the same top-100/global plus local-window MIL statistic is
computed on released and corrected logits. Their difference is bounded by
`2*tanh`, multiplied by the fixed 0.25 evidence weight, and applied only in the
already frozen >=0.50 protected scene coordinates. Mask gating continues to use
the unchanged current score, isolating dense-map quality from scene reranking.

The full-batch smoke reproduces pixel logits and current scene scores exactly,
then completes finite segmentation optimization with nonzero correction L2 and
an unchanged zero learned-scene residual. Peak CUDA allocation is 2.23 GB. Data,
folds, endpoint seeds, epochs, sampling, pixel losses, three residual strengths,
and all joint AP/IoU gates remain fixed. The contract was hashed before any new
held-fold outcome.

### 2026-07-31: Dense-surrogate pilot is directionally positive but rejected

The deterministic dense-MIL residual is the first DINO scene mechanism to
improve pooled AP without a learned scene head. At strength 0.50, AP changes
+0.000739 with unchanged recall and positive fold changes
(+0.000544/+0.000227), but it misses the +0.001 point floor and its paired-site
interval [-0.000155,+0.002062] crosses zero. Sentinel-2 AP improves +0.001275;
Landsat changes -0.000725 and fails the sensor gate. Mask IoU changes +0.005569,
but its lower bound -0.000842 also crosses zero at this largest strength.

Strength 0.25 gives +0.000701 AP and +0.005316 IoU; its IoU interval is
strictly positive [+0.001374,+0.008109], but fold-3 AP is effectively zero
(-0.000011), Landsat AP is negative, and AP uncertainty crosses zero. Strength
0.10 retains strong IoU evidence but weaker AP. No candidate passes and no
artifact is written. Result SHA-256 is
`7f7f2fdb7aa80d35d0ec5fb7a08d1b579155efc82b05007c8d694b795343e72b`.

The direction justifies one variance-control confirmation, not a free numeric
search. Three entirely new endpoint seeds will average only the deterministic
dense-scene evidence; the first seed supplies the fixed mask endpoint. Because
all three learned DINO scene variants and the dense surrogate consistently show
the only AP regression on Landsat, whose resolution/spectral transfer differs
from Sentinel-2, the confirmation will leave Landsat scene scores exactly on
the current model while retaining the mixed-sensor mask correction. The scene
evidence weight is fixed at 0.50 (twice the deliberately conservative pilot
weight) before new outcomes. The same three pixel strengths and all AP/IoU/
recall/site gates remain; folds 0/1/2 and external/test inputs stay excluded.

### 2026-07-31: Multi-seed Sentinel-2 confirmation frozen

The variance-control confirmation is now immutable before development-outcome
access. It trains six fixed endpoints: three entirely new seed bases
(`20263500`, `20263550`, and `20263600`) in both directions across development
folds 3/4. Scene evidence is the arithmetic mean of the three dense-MIL
residuals. It is applied at predeclared weight 0.50 only to Sentinel-2 scores
inside the existing >=0.50 protected coordinate system; every Landsat scene
score is asserted to remain exactly equal to the current spatial-Prithvi
ranker. The first seed supplies the single fixed mixed-sensor mask endpoint.

The GPU smoke proves exact zero-initialized pixel and scene identity, exact
synthetic Landsat routing identity, a positive change only for an eligible
Sentinel-2 synthetic score, and finite segmentation-only optimization. Peak
CUDA allocation is 2.23 GB for 8,540,787 trainable adapter parameters. The
trainer SHA-256 is
`cf102754c3fe2a53a5b43dfc59e7bd2b75edd9e434035db88b699d830b686f84`.
The three residual strengths (0.10/0.25/0.50), epoch-2 endpoint, sampling,
loss, bootstrap, and promotion gates are unchanged. Fold 2, source-disjoint
inputs, and official-test inputs and labels remain inaccessible to this run.

The first execution was stopped before its first epoch completed because the
shell wrapper had detached the still-running WSL trainer and its loader workers,
leaving no observable output. Process inspection showed that most elapsed time
was input hashing and only about eight minutes was training; no fold metric or
prediction report had been produced. A logging-only wrapper now emits progress
every 64 batches. It does not query RNG state or alter loader order, samples,
tensors, optimization, seeds, endpoints, or evaluation. The trainer is
re-hashed as
`904e3b1e16b9eb9092cda8cf241e2070712cefa9e519ef2b22e8349ee04048cc`
and recommitted before restarting the same frozen experiment.

### 2026-07-31: Multi-seed DINO scene confirmation rejected

All six predeclared endpoints completed and aligned across 17,745 development
rows (fold 3: 8,799; fold 4: 8,946). The routing assertion proves every
Landsat candidate score is exactly the current spatial-Prithvi score. Despite
that protection, the three-seed averaged Sentinel-2 evidence fails decisively.
At selected strength 0.10, pooled AP changes -0.001832 with paired-site 95%
interval [-0.003701,-0.000176]. Sentinel-2 AP changes -0.000623, Landsat is
exactly 0.0, and fold AP changes are -0.000344/-0.002304. Recall and operating
confusion counts are unchanged. Strengths 0.25 and 0.50 worsen AP further
(-0.004248 and -0.005877), with entirely negative paired-site intervals.

The first-seed mask endpoint again transports. Strength 0.10 improves pooled
IoU +0.002462 with paired-site interval [+0.000765,+0.003941] and positive fold
changes (+0.004324/+0.001060). Strength 0.25 improves IoU +0.004742 with
interval [+0.001059,+0.007708] and positive fold changes, while 0.50 has a
larger point gain but an interval crossing zero. No scene candidate passes, no
artifact is written, and fold 2, source-disjoint inputs, and official-test data
remain untouched. Result SHA-256 is
`38e0718c51fb6025fba6a733a3d7e86f11d9a329c39a956b2185dab6d07c098f`.

This exhausts the one promised variance-control confirmation. DINOv3 dense
features are retained only as a reproducible mask-improvement candidate; their
dense-MIL statistic is retired for scene ranking. The next scene architecture
must obtain a transportable image-level signal independently of this mask
surrogate, rather than applying another weight, route, seed, or strength search
to the same failed evidence.
### 2026-07-31: DINO scene ranking retired; physics-guided bi-temporal branch selected

The frozen three-seed DINOv3 confirmation rejected the transformer-derived
scene signal: at the selected 0.1 residual strength, average precision changed
by -0.001832 with a paired-site 95% interval of [-0.003701,-0.000176]. Dense
mask IoU still improved by +0.002462 [0.000765,0.003941], so the result does not
show that foundation features are intrinsically useless; it specifically rules
out this DINOv3 scene-ranking family under its predeclared contract. DINO is
therefore retired from the scene path and remains only a possible mask-side
ablation.

The next architecture is fixed conceptually before outcome scoring. It uses a
single shared six-band encoder for target and reference observations; explicit
target/reference NDMI, NDMI change, MBMP, normalized band changes, and temporal
log-ratios; multi-scale cross-date fusion; and bounded change-guided feature
modulation. Its dense path modulates the frozen released MARS-S2L U-Net through
zero-initialized gates, giving exact released-mask identity before training. An
independent 8x8 patch/MIL head is supervised directly for scene presence rather
than deriving every scene decision from a post-hoc top-k mask statistic. Initial
scene experiments route learned evidence only to Sentinel-2 because both new
source-disjoint cohorts are Sentinel-2; Landsat scene scores remain exact
identities while its dense path remains supported and trained on MARS.

This design is inspired by, but is not a reproduction of, Xu et al.'s 2026
N-BPMSNet. That paper reports an NDMI-guided shared-weight dual-date encoder,
cross-channel fusion, and change-guided attention, with F1 0.8858 and AUC
0.9856 on its own 11,494-sample, 44-site Sentinel-2 dataset. Those numbers are
not directly comparable to MARS-S2L's exact 43,529-image benchmark; the present
work will continue to use the MARS-S2L v3 comparator, site-block bootstrap, and
mixed-sensor views as its only superiority contract. The acquired 135 UNEP
positive scenes and 367 CloudSEN12+ clear negatives are auxiliary training data;
their separate development cohorts remain unopened until the original MARS
development gates pass.

Primary literature: [Xu et al., *IEEE TGRS*
2026](https://doi.org/10.1109/TGRS.2026.3689118);
[Kumar et al., *MethaneMapper*, CVPR
2023](https://openaccess.thecvf.com/content/CVPR2023/html/Kumar_MethaneMapper_Spectral_Absorption_Aware_Hyperspectral_Transformer_for_Methane_Detection_CVPR_2023_paper.html);
[Quintero et al., *Methane-Plume Segmentation From Hyperspectral Satellite
Imagery Via Multimodal Deep Learning*](https://arxiv.org/abs/2606.26416).

### 2026-07-31: NDMI bi-temporal context-scene pilot rejected, dense path retained

The fixed fold-3/fold-4 cross-fit trained two 4.61M-parameter endpoints on
MARS plus the source-disjoint auxiliary cohorts. Each training request stream
used exactly 75% MARS, 12.5% UNEP positives, and 12.5% CloudSEN12+ negatives,
with balanced labels and physical sites. The held external development cohorts,
fold 2, and official test stayed closed. Both endpoints were finite and all
scene, partial-AUC, patch, and dense losses decreased through the fixed third
epoch.

At the selected strength 0.10, the candidate improves pooled AP +0.000756,
matched-FPR recall +0.000656, and IoU +0.002426. Both held folds improve AP
(+0.001328/+0.001037), recall (+0.001319/+0.001305), and IoU
(+0.003352/+0.001738); Sentinel-2 AP improves +0.000923 and Landsat remains an
exact scene-score identity. Mask transport is statistically positive, with a
paired-site IoU interval of [+0.000465,+0.003671]. Scene evidence is not:
paired-site AP spans [-0.000725,+0.003139], and the point gain is below the
predeclared +0.002 minimum. Strength 0.25 raises recall +0.002625 and IoU
+0.005075, but AP is only +0.000534 with a wider interval; strength 0.50 lowers
AP. The architecture is therefore rejected and writes no artifact. Result
SHA-256 is `0c8bebeace840acbb0a27450919d47e852cb3166ea171d9344701e2ba90d3177`.

The result separates two hypotheses. Explicit NDMI/change fusion consistently
helps the dense mask, whereas the global attentive/mean/max context head adds
only weak scene ranking and may exploit source/background context. This agrees
with N-BPMSNet's ablation discussion: cross-channel fusion mainly improves
pixel metrics and can slightly reduce scene F1/recall, while NDMI-guided change
localization carries the strongest dense benefit. N-BPMSNet uses BCE plus
Lovasz-hinge (weight 0.75), 500 epochs, and best-validation-IoU selection on its
own 160x160 Sentinel-2 benchmark; those training numbers are documented as a
future dense-loss ablation, not imported post hoc into this failed protocol.
The next bounded hypothesis removes the global context classifier and makes
scene evidence a deterministic top-patch MIL aggregation of directly
mask-supervised patch logits. It must use new seeds and the same closed-data
boundary.

### 2026-07-31: NDMI patch-MIL scene pilot rejected; mask conclusion replicated

The precommitted patch-local variant removed the learned global scene context
head and computed its Sentinel-2 scene correction only from the top eight
directly mask-supervised patch logits. It used new seeds and otherwise retained
the same cross-fit, source-balanced auxiliary data, three-epoch budget, routing,
strength grid, and 10,000-replicate paired-site promotion contract. External
development, fold 2, and the official test remained unopened.

At selected strength 0.10, pooled AP changes only +0.000694 and matched-FPR
recall changes -0.002625. The paired-site AP interval is
[-0.000680,+0.001764]. Fold 3 improves AP +0.002024, while fold 4 changes
-0.000294; Sentinel-2 AP changes +0.001488 and Landsat remains an exact
identity. Strength 0.25 increases pooled AP by only +0.000721 and has a wider
interval crossing zero; strength 0.50 lowers AP. The candidate therefore fails
the minimum AP, recall, each-fold, and paired-site AP gates and writes no model
artifact.

The dense result again transports: strength 0.10 improves pooled IoU +0.002552
with paired-site interval [+0.000509,+0.004217], and both held folds improve.
Strength 0.25 raises IoU +0.005488 with a still-positive interval but does not
rescue scene ranking. Two independently specified architectures now reproduce
the same separation: explicit NDMI/change features provide a statistically
positive mask-side signal, while deriving an image-level correction from the
dense representation does not provide stable site-general scene ranking. More
patch aggregation is therefore retired. The next scene experiment must first
expand or improve source-disjoint positive-site supervision and explicitly
address domain/site invariance. Result SHA-256 is
`ab76b2f66d6a3f96a23aff6b54e22653a566520b0cfd460b706d5f86d23d9b08`.

### 2026-07-31: external-data provenance audit and spatially disjoint MethaneS2CM cohort

The newly released MethaneSET collection initially appeared to supply 3,612
verified Sentinel-2 plume scenes plus 57,291 Sentinel-2 and 21,926 Landsat
plume-free scenes. A pinned metadata-only audit against the exact MARS corpus
shows that it is a repackaging rather than independent supervision. Exact MARS
image identifiers account for 3,612/3,612 S2 finetune rows, 57,290/57,291 S2
pretraining rows, 1,547/1,548 Landsat finetune rows, and 21,924/21,926 Landsat
pretraining rows. Those matches include respectively 1,053, 23,250, 714, and
16,311 exact official-test observations. No MethaneSET image archives were
downloaded, and none may be used in this campaign. This prevents a large but
subtle test-leakage error while preserving MethaneSET as a useful distribution
format citation.

The already acquired MethaneS2CM L2A location-split training corpus is not an
exact MARS repackaging, but its CarbonMapper sites overlap MARS geography. A
label-blind spatial audit used only the 1,289 official MARS test location IDs
and centroids. Of 3,460 MethaneS2CM exact locations, 2,113 lie within 25 km and
are excluded. The remaining 1,347 locations are strictly farther than 25 km
(minimum 26.099 km), yielding 26,337 crops: 13,277 plume and 13,060 no-plume.
The existing MethaneS2CM 25 km group-held roles are preserved, leaving 14,859
auxiliary-training crops across 769 exact locations/147 groups and 11,478 held
source-development crops across 578 locations/48 groups. The 4.25 GB HDF5 and
generated manifests remain ignored; compact checksummed receipts are tracked.

This changes the next experiment materially. The scene bottleneck will be
tested with a shared physical-scale local detector: direct 32x32, 20 m
MethaneS2CM crops provide source-disjoint positive/no-plume supervision, while
approximately 640 m windows from MARS are resampled to the same physical field
of view. Training must mix source-balanced batches and discourage source-domain
separation; whole-scene output remains a bounded residual on the frozen current
score and Landsat remains identity until independent Landsat-positive
supervision exists. The held MethaneS2CM development images, MARS fold 2, and
the official test remain closed during the first cross-fit.

Metadata-only review also found the June 2026 MethaneUnion upload (168,832
rows in its original-scale geographic split and 764 GB of imagery). It has no
paper, acquisition timestamps, source crosswalk, or provenance discussion in
its dataset card. Its imagery is therefore not acquired or used; scale alone is
not acceptable evidence of independence.

Primary data sources: [MethaneSET](https://huggingface.co/datasets/tacofoundation/methaneset),
[MethaneS2CM](https://huggingface.co/datasets/H1deaki/MethaneS2CM), and
[MethaneUnion](https://huggingface.co/datasets/yuyao42/MethaneUnion).

### 2026-07-31: physical-scale cross-domain detector protocol frozen

The next pilot is frozen before any held-fold outcome is computed. Its common
field of view is 640 m: each native 32x32 MethaneS2CM crop at 20 m is compared
with a 64x64 MARS crop at 10 m that is area-pooled to 32x32. MARS reflectance
is pooled before MBMP is recomputed; plume masks are max-pooled, while clear
and observable masks require both contributing pixels. This avoids the earlier
error of treating equal pixel dimensions as equal plume physics.

The 9,450,995-parameter detector uses one shared six-band encoder across the
target, 90-day reference, and 365-day reference. MARS has one historical frame,
so that frame and its recomputed MBMP are duplicated explicitly rather than
inventing a second observation. Five fusion scales consume signed/absolute
change, two MBMP maps, wind, cloud, observability, and sensor identity. A U-Net
decoder produces the local plume mask. Local scene evidence is a fixed equal
blend of mask-top-1% evidence and a small bottleneck context head. At full-scene
inference, 36 overlapping crops cover every 200x200 MARS pixel and scene evidence
is the predeclared `0.75*mean(top four)+0.25*maximum` tile-logit aggregation.

To discourage L2A/L1C source formatting from becoming the classifier, a
gradient-reversal head predicts MARS versus MethaneS2CM from pooled Sentinel-2
representations. Training first uses two fixed epochs of the 14,859 disjoint
MethaneS2CM auxiliary crops, then three fixed joint epochs with 50% MARS and
50% MethaneS2CM request mass. MARS mass remains equal across label by sensor;
MethaneS2CM mass is equal by label and then by frozen 25 km group. MARS plume
crops are plume-centered with fixed probability 0.85 and otherwise random,
so the local classifier also sees within-positive-scene background.

The real-data smoke covered two mixed-source optimization batches and two full
MARS scenes without reporting any candidate metric. All losses and both
released/stiched 200x200 outputs were finite. Peak CUDA allocation was
754,310,144 bytes on the RTX 5070. Unit tests prove the 20-channel transform,
edge-aligned full-scene coverage, source-routing identity, output shapes, and
gradient-reversal sign. The protocol pins every outcome-affecting code/input
hash, seeds 20263903/20263904 for the fold-3/fold-4 endpoints, strengths
0.10/0.25/0.50, and the existing 10,000-replicate physical-site bootstrap
gates. MethaneS2CM development, MARS fold 2, folds 0/1, and official-test
outcomes remain closed.

### 2026-07-31: physical-scale cross-domain detector rejected

Both fixed endpoints completed the two external-pretraining and three joint
epochs, then scored all 17,745 held fold-3/fold-4 scenes through 36 overlapping
physical tiles. Training was finite and the source adversary remained active,
but the new signal did not improve the stronger current spatial-Prithvi ranker.

At the selected minimum strength 0.10, pooled AP changes -0.000172 and
matched-FPR recall changes -0.001312. The paired-site AP interval is
[-0.000501,+0.000171]. Sentinel-2 AP changes only +0.000023 while Landsat
remains an exact scene-score identity; cross-sensor pooling accounts for the
slightly negative overall AP. Fold AP changes are -0.000004/-0.000042, and
fold 4 loses one matched-FPR positive. Strengths 0.25 and 0.50 worsen AP
monotonically (-0.000490/-0.001226).

The mask change is small and uncertain. Strength 0.10 raises aggregate IoU
+0.000284 with positive point changes on both folds, but its paired-site
interval [-0.000238,+0.000690] crosses zero. Larger strengths raise the point
gain to +0.000543/+0.000842 while widening negative uncertainty. This contrasts
with the consistently positive NDMI and frozen-transformer dense adapters:
matching physical crop scale alone does not make the MethaneS2CM representation
transport to MARS L1C masks or scene ordering.

No artifact is written. MethaneS2CM development, MARS fold 2, folds 0/1, and
the official paper test remain closed. Result SHA-256 is
`4a0425da9c6186a7b803c3543bb94d1c3bdb286baedc874e7f9be1aa0888e117`.
The result retires direct S2CM/MARS local-domain alignment as an isolated lever.
Any further external-data path must use exact MARS-domain radiometry or a causal
physics generator rather than another L2A-to-L1C feature alignment objective.

### 2026-07-31: Gaussian-plume ViT source audit and fit-fold parameterization

The next causal representation branch is grounded in the final published
methods of Rouet-Leduc and Hulbert rather than the earlier roadmap's incorrect
attribution to Groshenry/MethaNet. The Nature Communications study generates
about 20,000 analytical Gaussian plumes, adds two-dimensional colored noise,
embeds them in Sentinel-2 B12 with Beer-Lambert attenuation, and keeps the
template train/validation/test sets disjoint. Half of its image pairs are
synthetic positives and half are unchanged real negatives. Its ViT-B/16
encoder and convolutional U-Net decoder are trained from scratch on 1,235,000
two-time Sentinel-2 examples for ten epochs. The paper explicitly identifies
large-scale synthetic exposure, temporal reference imagery, and transformer
long-range context as the three main performance mechanisms. Primary source:
[Rouet-Leduc & Hulbert 2024](https://www.nature.com/articles/s41467-024-47754-y).

The MARS adaptation preserves those mechanisms but not the single-sensor
shortcut. It generates a vertically integrated, wind-aligned Gaussian column
field with widening crosswind variance, exponential finite residence length,
and correlated log-normal turbulence. The exact released MARS integrated-
transmittance LUT attenuates B11 and B12 separately for S2A/S2B/S2C and the
supported Landsat platforms. It never reports synthetic flux: public MARS
enhancement assets retain a documented ppm/ppb metadata conflict. Instead,
peak enhancement is sampled in LUT input units and audited against unitless
same-scene MBMP contrast.

Only development folds 3 and 4 parameterized the morphology. Among 1,524
scene-positive rows, 1,518 have non-empty masks and six producer-positive rows
have empty pixel masks; the latter are retained for scene learning but excluded
from geometry. The non-empty cohort covers 81 physical sites, 1,139 Sentinel-2
and 379 Landsat scenes, all at 10 m. Real q05/q50/q95 anchors are 327/1,008/3,603
mask pixels, 314/704/1,303 m major-axis four-sigma extent, 124/206/492 m
minor-axis extent, and 1.52/2.81/4.71 moment aspect ratio. Median major-axis
alignment with published wind is 0.952, while its q05 is 0.335; this supports a
predeclared heavy-tailed wind-direction mismatch rather than unrealistically
perfect alignment. Real median MBMP absorption contrast is 0.00501 and robust
SNR is 1.218, with q05/q95 contrast 0.00097/0.01868.

The deterministic bank reserves 20,000 indexed templates: 16,000 training,
2,000 validation, and 2,000 diagnostic, with no index overlap. A 5,000-template
audit passed every frozen distribution gate. The common family contributes 88%
and a broad morphology tail 12%; all generated masks are exactly one connected
component. Across q05/q25/q50/q75/q95, the maximum relative discrepancy is at
most 20% for area, major/minor extent, and aspect ratio, while wind alignment is
within 0.05 absolute. Peak LUT input strength is log-uniform from 500 to 10,000;
an independent fit-fold radiometric anchor sweep maps roughly 2,000 to the real
median contrast and roughly 8,000 to its upper tail.

### 2026-07-31: full-scene Gaussian-pretrained ViT-U-Net smoke passed

The candidate is an actual vision transformer rather than a transformer-feature
probe or convolution-only patch detector. It keeps the exact 16-channel MARS
target/reference/MBMP/wind/cloud input. A 16-pixel patch embedding feeds eight
256-dimensional, eight-head transformer blocks; four convolutional skip scales
feed a U-Net decoder. Learned sensor embeddings condition Sentinel-2 versus
Landsat. Training uses 160x160 crops (1.6 km at 10 m), while interpolated
position embeddings and reflection padding preserve native 200x200 inference.
The 9,907,262-parameter model returns both a dense mask correction and a scene
correction derived from mean/max global tokens plus top-1% dense evidence.

The mixed smoke exercised both same-background Gaussian positive/unchanged-
negative pairs and genuine MARS crops with the protected spatial-Prithvi
residual objective. All segmentation, scene, partial-AUC, residual, and
direction losses were finite. A real batch activated the residual BCE and both
correction-direction penalties. Full 200x200 mixed-sensor inference returned
finite `[2,1,200,200]` masks and two scene logits. An actual 24-example training
batch peaked at 3,831,370,752 CUDA bytes on the RTX 5070. The held-fold outcome
grid is still unopened; hashes, endpoints, strengths, and promotion gates must
be committed before cross-fit scoring.

The first full launch stopped after 64 synthetic-pretraining batches and before
any held-fold scoring because one indexed plume had zero observable target
pixels. The deterministic data path was tightened to search fit-fold background
crops with at least 90% observable support and to accept a generated plume only
when at least half of its mask is observable. A 1,024-positive-template stress
test across both sensors then produced 1,024 non-empty masks. The trainer hash
was refrozen and committed before the cross-fit restart; no metric, checkpoint,
or candidate outcome from the aborted launch exists.

### 2026-07-31: short-exposure Gaussian-pretrained ViT-U-Net rejected

The frozen fold-3/fold-4 cross-fit completed both analytical-Gaussian
pretraining endpoints, three joint real/synthetic epochs, and all 17,745 held
development scenes. It did not promote. At the selected minimum residual
strength 0.10, AP changes -0.000010 versus the current spatial-Prithvi score,
matched-FPR recall is unchanged, and dense-mask IoU changes -0.000098. The
10,000-replicate paired-site intervals are [-0.000123,+0.000153] for AP and
[-0.001242,+0.000632] for IoU. Strengths 0.25, 0.50, and 1.00 worsen AP
monotonically; the result is therefore an inert representation rather than a
fusion-strength miss. No checkpoint is written, and fold 2, folds 0/1, external
development outcomes, and the official test remain closed.

This pilot is not a negative test of the published large-scale Gaussian-data
hypothesis. Each endpoint received only 8,192 synthetic requests before joint
training, whereas Rouet-Leduc and Hulbert report 1,235,000 training pairs for
ten epochs. Synthetic scene BCE remained near chance (0.6977 to 0.6951 and
0.6975 to 0.6943 across the two endpoints), and Dice loss remained near 0.91,
showing that the 9.9M-parameter model had not learned the synthetic task before
real-data fusion. One joint epoch aggregate on the fold-3 endpoint contained a
transient non-finite batch; mixed-precision scaling skipped the affected update
and subsequent training/evaluation were finite, but any reuse must add an
explicit finite-batch assertion and identify the offending input path.

The next Gaussian decision is therefore gated by a synthetic-only learnability
audit: first prove that the exact architecture can memorize a small fixed bank
and improve on an index-disjoint synthetic validation bank, then estimate the
exposure curve before committing to a paper-scale pretraining run. Result
SHA-256 is
`cba9a42cf1c5b10bf8969cbf7f56cf3785cc6f9864947f7f05a6d1dfa8834ff0`.

The first synthetic-only audit launch stopped before its first optimizer update
and before writing any outcome report. All cached inputs, forward outputs, and
the scalar loss were finite, but the default FP16 GradScaler initial scale of
65,536 overflowed the first backward pass. The audit now freezes an initial
scale of 1,024 with a 1,000-step growth interval and retains its strict
non-finite loss and gradient assertions. This is a numerical-stability repair;
the bank, model, optimizer, epochs, checkpoints, and learnability gates are
unchanged and were not scored by the aborted launch.

### 2026-07-31: raw-input Gaussian ViT learnability audit fails

The numerically stabilized audit completed all fixed checkpoints without a
non-finite tensor, loss, or gradient. The exact raw-input architecture cannot
memorize 64 positive/unchanged-negative template pairs in 20 epochs (160
updates): train scene AP changes only 0.5004 to 0.5042 and train mask IoU at
0.5 changes 0.0522 to 0.0530. The 64 index-disjoint templates likewise remain
near chance: validation AP changes 0.5010 to 0.5129 and IoU 0.0437 to 0.0456.
Both frozen gates fail, and no checkpoint is retained. Result SHA-256 is
`9ac1ce127ad273f46caf100ff4f58bc02dae7baf234e2c5cdf465083110fcd7b`.

A direct paired-tensor audit rules out an injection no-op. Across 16 sampled
pairs, the median in-mask signed MBMP change is -0.00551; B11 and B12 target
channels change while all reference, wind, and cloud channels remain exact.
The median maximum absolute MBMP change is 0.03968. A one-pair optimization
probe then reaches 0.918 mask IoU and 0.974 scene-logit separation by update
50, proving that the computation graph can learn the signal but needs far more
exposure across heterogeneous backgrounds. A later backward overflow at update
88 also confirms that FP16 is an unnecessarily fragile format for this branch.

The final Nature methods use batch 64, an initial learning rate of 1e-3, ten
epochs, and best-validation checkpointing over 1,235,000 unique training
samples. The associated primary preprint describes inputs as normalized
temporal differences of band ratios. The present raw 16-channel patch stream,
2e-4 learning rate, and 342 synthetic-pretraining updates therefore test a much
weaker training regime, not the paper mechanism at credible scale. The next
architecture will retain the exact external MARS 16-channel interface but
derive normalized temporal log-ratio and relative-change channels internally,
train in BF16, and first repeat the disjoint learnability gate before any real
held-fold score is opened.

### 2026-07-31: physics-contrast ViT learns dense transfer but fails scene gate

The frozen normalized-contrast audit completes in BF16 with no numerical
failure. At the same 64-train/64-disjoint-template, 160-update budget, the
learned scene head still fails: train AP changes 0.5073 to 0.5265 and validation
AP 0.5063 to 0.5159. The strict memorization and validation gates therefore
remain false.

The dense result is qualitatively different from the raw-input model. Train
mask IoU rises from 0.0202 to 0.1353 and index-disjoint validation IoU from
0.0216 to 0.1240. The close train/validation values show that fixed MBMP,
per-band log change, and all 15 temporal log-ratio changes expose a transferable
synthetic plume representation; the raw model ended at only 0.0530/0.0456.
This is a 2.55x train and 2.72x validation final-IoU improvement without using
another scene, template index, or held label. Result SHA-256 is
`b19f3e348105b4d1c681c0a06edb5208c1e3646c7c99120374ffbcaba8bdfb56`.

The outcome changes the next objective rather than relaxing the failed gate.
The primary paper trains a plume mask and derives detection from its dense
output; it does not require a separately supervised global token head during
synthetic pretraining. A new fixed exposure curve will therefore train the same
contrast architecture only on dense positive/hard-negative/Dice losses, record
top-mask evidence AP, and extend to 100 epochs (800 cached updates). Real scene
ranking, development folds, and the current spatial-Prithvi residual remain
closed until dense synthetic transfer is demonstrated.

### 2026-07-31: dense exposure curve rejected at fixed final checkpoint

The 100-epoch dense-only BF16 curve is finite and establishes that the
paper-aligned dense mechanism is learnable. At the frozen epoch-100 endpoint,
train dense-evidence AP is 0.9664 and train IoU 0.2836, while index-disjoint
validation AP is 0.7294 and IoU 0.1271. The final endpoint misses the train-IoU
and validation-IoU gates, so both promotion booleans are false and no checkpoint
is retained. Result SHA-256 is
`8dd879acac0411257363dfc905bb37aab726bb892be53a8a0ba8e60ca299a6cf`.

The fixed curve reveals classical small-bank overfitting rather than failure to
learn. Epoch 50, which was a predeclared diagnostic checkpoint but not an
eligible endpoint, has train AP/IoU 0.9134/0.3450 and disjoint validation
AP/IoU 0.7578/0.2301; those values exceed every numerical gate. By epoch 100,
train AP rises while validation IoU collapses. Epoch 50 is not promoted post
hoc. Instead, the next confirmation adopts the final paper's stated
best-validation-model rule before rerunning, uses entirely new template ranges,
and increases both train and validation banks fourfold. Checkpoint selection is
by highest synthetic-validation IoU with dense AP as a tie-break; all selected
checkpoint AP/IoU gates remain mandatory. No real held outcome is involved.

### 2026-07-31: new-bank dense Gaussian-ViT confirmation passes

The fourfold larger confirmation uses 256 entirely new training templates and
256 new validation templates, with no index overlap with the earlier 64-pair
audits. All 80 BF16 epochs and nine predeclared checkpoints are finite. The
paper-aligned selection rule chooses epoch 70 by maximum validation pixel IoU,
with dense-evidence AP and earliest epoch as tie-breaks.

Both frozen gates pass. Selected train dense-evidence AP is 0.9195 and mask IoU
is 0.3204. On the new index-disjoint validation bank, dense-evidence AP is
0.8092 and mask IoU is 0.2711, up from initialization IoU 0.0521. The selected
validation pixel recall is 0.3382. Peak CUDA allocation is 4,368,115,200 bytes.
The result independently confirms that the normalized temporal-contrast ViT
learns a transferable Gaussian plume mask and that synthetic validation can
identify overfitting without any real held label. Result SHA-256 is
`20c6b05f4cec5b17b516522bb4037c9eaaa4706d6e3c28b16534b4b3483c9667`.

This pass authorizes full-bank scaling only. It does not promote a real MARS
candidate, open fold 2, or justify official-test access. The next protocol must
use all 16,000 reserved training templates and the 2,000 reserved validation
templates, retain BF16 and the physics-contrast front end, and prove full-bank
dense transfer before a separately frozen fold-3/fold-4 real cross-fit.

The first full-bank launch was stopped before epoch 1 completed and before any
validation checkpoint or outcome report existed. Direct throughput measurement
on the Windows-mounted source tiles projected more than four hours for ten
epochs, exceeding the frozen process window. The data path selected each paired
positive and negative from the same background/crop but shuffled their requests
independently, causing duplicate file reads and allowing the positive's plume
RNG consumption to alter its augmentation orientation.

A deterministic performance/invariance repair is refrozen before restart.
Templates are shuffled while their positive/unchanged twins remain adjacent;
each worker holds a bounded 64-sample LRU cache, making the second twin a cache
hit. Background/crop, plume, and augmentation now use independent deterministic
RNG streams, so paired twins receive exactly the same geometric transform and
wind-vector transform. The optimizer still sees one random template order per
epoch, every index exactly once. Removing an unused returned contrast tensor
reduces validation memory without changing forward values or gradients. Twelve
focused tests pass, including pair adjacency and full epoch coverage. No bank,
loss, model parameter, epoch count, gate, or outcome-dependent setting changes.

### 2026-07-31: full-bank Gaussian contrast pretraining passes

The optimized paired loader completed all 16,000 training templates (32,000
matched positive/unchanged-negative requests per epoch) and 2,000 disjoint
validation templates for ten BF16 epochs. Synthetic-validation mask IoU selected
epoch 9. Its dense-evidence AP is 0.927611, mask IoU 0.261063, and pixel recall
0.579037, clearing the frozen 0.80/0.25/0.30 gates. This establishes large-bank
synthetic learnability, not real-scene superiority. Compact result SHA-256 is
`689b806228abda20e433af180f25a2e722bd71a1a2c213a9fb51c0f082f162f3`.

### 2026-08-01: Gaussian contrast real cross-fit rejected

Both authorized endpoints completed nine full-bank pretraining epochs and three
joint real/synthetic epochs, then scored all 17,745 fold-3/fold-4 scenes. At the
selected 0.05 residual strength, AP changes +0.000269 with paired-site interval
[-0.000467,+0.000797], recall changes -0.000656 at unchanged FPR, and IoU
changes +0.001260 with interval [-0.000248,+0.002782]. Strength 0.10 preserves
pooled recall and yields a significant +0.001806 IoU change (paired lower
+0.000251), but AP remains only +0.000326 and uncertain. Stronger fusion harms
both tasks. No candidate passes and no artifact is written. Result SHA-256 is
`728e9a1f69a608cf09816114ed50fc785953f6dc5937ddae370e87f990d85f55`.

### 2026-08-01: robust temporal site-template ranker rejected

The unseen-site architecture replaced the earlier label-free pixelwise mean with
per-site q25, median, and q75 spatial maps. Two predeclared feature contracts fed
either an IQR-normalized median anomaly or explicit original/median/residual maps
to the established residual spatial CNN. Four models (two feature contracts by
group or site-cell weighting), five current-score blends, deterministic seeds,
and every promotion gate were committed at `37c7de00` before held scoring.

The 17,745-scene, 250-site fold-3/fold-4 cross-fit rejects the family. The best
pooled AP point change anywhere in the fixed grid is only +0.000019, accompanied
by recall change -0.002625 and paired-site AP lower bound -0.000710. The frozen
worst-fold-first selector chooses the group-weighted IQR-normalized model at
blend 0.05: AP changes -0.000099, matched-FPR recall -0.001969, and its paired
AP lower bound is -0.000787. Stronger blends monotonically worsen AP, reaching
-0.016126. No artifact is written and fold 2, folds 0/1, external-development,
and official-test outcomes remain unopened.

Together with the earlier site-mean replay, this closes scalar and spatial
same-site template aggregation as a sufficient solution. The missing signal is
not a more robust temporal background statistic; the next architecture must
learn a stronger product-matched representation or add genuinely independent,
contemporaneous positive supervision. Result SHA-256 is
`16a3928c9ea7dcc3f25e2ba3c62fededc231655e0436c3d35d6eb2e4006e8a88`.

### 2026-08-01: official UNEP refresh — raw role recomputation corrected

Fresh official UNEP/IMEO CSV and GeoJSON archives contain 27,403 plume rows and
yield 269 exact-product, paper-test-disjoint positives under the frozen audit,
32 more than July. The first raw rerun reported roles 216 auxiliary / 40
development / 13 sealed, but this was not a valid frozen-role comparison:
catalog growth changed 36 connected-component hashes and would have moved 18
previously frozen auxiliary identities into development.

An append-only reconciliation now treats every July identity, group, and role
as immutable. Expanded components inherit their one prior group and role; new
components keep their prospective assignment; merges of multiple prior groups
are quarantined. The audit reports zero old identity changes and zero
quarantines. Correct roles are 247 auxiliary / 9 development / 13 sealed, so
all 32 additions are auxiliary and none alter the frozen development or sealed
cohorts.

Exact-product resolution found 52 new auxiliary identities absent from the old
135-row model manifest: 26 Sentinel-2 and 26 Landsat across 15 groups. All 26
Sentinel-2 crops were acquired; the frozen CloudSEN12 gate retained 25 and
rejected one, producing a combined 160-row, 28-group real-positive training
manifest with SHA-256
`8c42b4350ccc0abbd2fec727abba435fca2af8f29289e837d52e7151553d861b`.
The 26 LandsatLook requests redirected to USGS EROS authentication and remain
pending without product substitution. Bulk data remain ignored. Corrected
receipt: `reports/acquisition/unep_mars_post2024_refresh_20260801.json`.

### 2026-08-01: anchored full released-U-Net transfer rejected

A new 13.58M-parameter candidate directly fine-tuned every released-U-Net
convolution while freezing all released BatchNorm state and retaining an
immutable released teacher. The student began at exact logit identity and used
BF16, discriminative 1e-5/5e-5 learning rates, normalized L2-SP anchoring,
three fixed epochs, and the same 75/12.5/12.5 MARS/UNEP/CloudSEN request mass.
Its protocol and four strengths were frozen at `8733b8d0` before held scoring.

At selected strength 0.05, AP changes -0.000448, matched-FPR recall -0.001312,
and IoU +0.001431. Paired-site 95% intervals are
[-0.001151,+0.000003] for AP and [+0.000223,+0.002352] for IoU. Fold 3 changes
+0.000063 AP, +0.001319 recall, and +0.002506 IoU; fold 4 changes -0.000554,
-0.002611, and +0.000629 respectively. All stronger strengths worsen pooled AP,
reaching -0.004205, while IoU rises to +0.006284.

Within Sentinel-2, AP actually improves monotonically from +0.000180 to
+0.000925; Landsat is exact scene-score identity. The pooled reversal therefore
identifies cross-sensor calibration damage from moving only Sentinel-2 absolute
scores. Full-model external transfer is useful dense evidence but not a
standalone scene-ranker solution. No artifact is written and all unopened
cohorts remain closed. Result SHA-256 is
`1d870c08e27111859e03d10c0479bf15297e19203c4837f687027dd4b7f0cba1`.

### 2026-08-01: protected bi-sensor routing reveals a small stable signal

One same-seed mechanistic replay changes only the rejected parent's scene
fusion. Current scores below 0.25 stay bit-identical; higher scores are reranked
in local logit space, mapped back above 0.25, and receive teacher-student evidence
for both sensors. The fixed gate therefore preserves all operating confusion
counts. Four strengths and unchanged gates were frozen at `060859fd`.

All strengths improve AP on both folds and both sensors with exactly zero recall
and FPR change. Strength 0.10 changes AP +0.000237 with paired-site interval
[+0.000022,+0.000542], and IoU +0.002698 with interval
[+0.000410,+0.004497]; only the +0.002 AP floor fails. The frozen selector picks
0.50: AP +0.000881, recall 0, and IoU +0.006130, but AP and IoU intervals cross
zero and fold-4 IoU is -0.000190. Strength 1.0 reaches +0.001220 AP but remains
uncertain and changes IoU -0.000642.

This confirms that the parent pooled reversal was partly a sensor-calibration
artifact. The corrected ranking signal is direction-stable but too small to
promote alone, and stronger dense interpolation overfits fold 4. No artifact is
written. Result SHA-256 is
`0c3e2f2253316c714f1b58f39092bf6cc9e5b445823a46be27a5b1e1ebd818af`.

### 2026-08-01: expanded-real-data protected transfer frozen

The append-only UNEP refresh increases the verified real-positive auxiliary
manifest from 135 scenes / 27 groups to 160 scenes / 28 groups. A controlled
repeat will isolate that data change: architecture, released initialization,
folds 3/4, deterministic endpoint seeds, three epochs, source masses, optimizer,
losses, protection gate, four interpolation strengths, bootstrap, and promotion
gates are identical to the preceding protected pilot. Group-balanced sampling
means repeated temporal views add within-site variation rather than extra site
mass; the one genuinely new group adds one site unit.

The mixed-source CUDA smoke begins at exact pixel-logit identity, completes two
BF16 optimization batches with finite gradients, moves the anchored student,
and peaks at 6,781,294,592 allocated bytes. No held-fold outcome was computed.
The protocol is frozen at
`configs/mars_protected_bisensor_expanded_unep_protocol.json`; UNEP/CloudSEN
development rows, MARS fold 2, folds 0/1, and all official-test inputs remain
closed.

### 2026-08-01: expanded-real-data protected transfer rejected

The controlled 160-row UNEP repeat completed both endpoints and all 17,745
fold-3/fold-4 scores from frozen commit `ee5ca224`. It moves every candidate in
the expected positive direction but does not create a promotion-scale effect.
The selector again chooses strength 0.50: AP changes +0.000941, matched-FPR
recall and FPR are exact identities, and IoU changes +0.006395. Paired-site
intervals are [-0.000034,+0.002236] for AP and
[-0.002502,+0.013529] for IoU. Fold-3/fold-4 AP changes
+0.000773/+0.000577; IoU changes +0.014878/+0.000093.

The conservative strength 0.10 remains the statistically stable point: AP
+0.000257 with lower bound +0.000026 and IoU +0.002631 with lower bound
+0.000369. Relative to the 135-row run, those changes were +0.000237 and
+0.002698, so the new real scenes slightly improve ranking but do not enlarge
the effect materially. Strength 1.0 reaches AP +0.001301 yet remains uncertain
and nearly eliminates the dense gain. No candidate reaches the fixed +0.002 AP
floor; no artifact is written. Result SHA-256 is
`3e70513986192dda3b7aadc8feed6a5b12ce6e11eaaef2691e9deb7859efc96c`.

This controlled null result is still useful: additional temporal positives in
the same representation do not solve the site-general ranking gap. The stable
protected signal should next be combined with an independently learned
foundation representation while decoupling scene strength from the already
positive strength-0.10 dense interpolation.

### 2026-08-01: independent protected ensemble preregistered

The next candidate combines the deployment-safe train-fitted DOFA-v2 scene
residual (+0.001412 AP on folds 3/4) with the independently learned expanded-
UNEP anchored residual. Both are expressed in local-logit coordinates above a
final 0.25 gate; scores below 0.25 remain exact, so the operating confusion
counts cannot change. DOFA remains fixed at its prior gate 0.50 / weight 0.05.
Only three already evaluated anchored source strengths (0.25/0.50/1.0) and two
residual multipliers (0.5/1.0) form the six-candidate grid.

Scene selection requires AP +0.002, a strictly positive paired-site lower
bound, positive AP in both folds and both sensors, and exact operating counts.
Dense prediction is not retuned: anchored strength 0.10 is fixed because its
IoU change is +0.002631 with paired lower +0.000369 and positive fold changes
+0.004456/+0.001269. A passing result authorizes only a new fold-2 confirmation.
The grid and evaluator are committed before the anchored score cache exists;
after generation, only the cache SHA-256 and frozen status may be filled.

The same-seed cache replication completed from preregistration commit
`dcf8c2af`. Its isolated selected signal remains rejected (AP +0.000987,
recall exact, IoU +0.006091), so it does not alter the ensemble grid. The
pickle-free scene cache contains 17,745 aligned rows and four candidate vectors,
is 932,951 bytes, and has SHA-256
`c872ba9e92f2af17fa83a1fdac1d161f56a0278ca309ad2db3cfce5bfbe3dadb`.
The research-only two-endpoint state cache is 108,855,455 bytes with SHA-256
`657a79945f788439f9ccd312dedee6d77931439c83e56489e0453fc9b6e369ac`.
Both are ignored. Schema, row count, and protocol binding were inspected; no
combined DOFA+anchored metric was computed before this hash-only freeze.

The first ensemble command failed closed during input verification because the
anchored-result SHA had one omitted `e` in its manual transcription. No feature
or score array was loaded and no candidate metric was computed. Only the
63-to-64-character receipt correction was made before retrying; the candidate
grid, formulas, seeds, gates, and cache hashes are unchanged.

### 2026-08-01: protected DOFA + anchored ensemble passes folds 3/4

The frozen six-candidate ensemble selects anchored source strength 0.50 with
multiplier 1.0 beside the unchanged train-fitted DOFA residual. AP changes
+0.002230 with paired-site mean +0.002257 and 95% interval
[+0.000678,+0.004151], clearing the fixed +0.002 floor. Fold-3/fold-4 AP
changes are +0.001143/+0.001682 and Sentinel-2/Landsat changes are
+0.002246/+0.001396. The final 0.25 protection gate preserves the complete
operating confusion matrix, so matched-FPR recall and FPR remain exact.

Dense evidence is independently fixed at anchored strength 0.10: IoU changes
+0.002631 with paired lower +0.000369 and fold changes
+0.004456/+0.001269. Thus every preregistered scene and dense selection gate
passes. The result is mechanistically stronger than either component alone:
DOFA contributes wavelength-conditioned global representation while the
anchored U-Net contributes supervised spatial plume evidence. This pass
authorizes exactly one fold-2 confirmation of strength 0.50 / multiplier 1.0;
it does not authorize fold 0/1, external, or official-test evaluation. Result
SHA-256 is
`85d5959061cc77d4b2a32f12609c7852af545ff8df6923273d6ae245d020eec1`.

### 2026-08-01: protected ensemble fold-2 confirmation preregistered

The passing folds-3/4 ensemble is now sealed as one exact fold-2 candidate:
anchored scene strength 0.50 with multiplier 1.0, fixed train-fitted DOFA at
gate 0.50 / weight 0.05, and a final identity gate at 0.25. Its independently
scored dense path remains anchored strength 0.10. The anchored endpoint uses a
new seed, fits only the union of folds 3 and 4, and will score fold 2 once. The
DOFA probes likewise fit folds 3 and 4 and reuse their already frozen five
projection seeds. There is no fold-2 grid or endpoint selection.

The confirmation requires AP improvement of at least +0.001 versus the current
spatial-Prithvi system, a strictly positive 10,000-replicate paired-site AP
lower bound, positive AP changes for Sentinel-2 and Landsat, unchanged operating
confusion counts and no worse recall/FPR, positive dense IoU with a positive
paired-site lower bound, and positive AP/recall versus the released primary
model. Evaluator, dependencies, inputs, bootstrap seed 20264330, formula, and
gates are recorded in
`configs/mars_dofa_anchored_protected_fold2_protocol.json` before either future
anchored score or result file exists. After generation, only those two exact
SHA-256 placeholders and the frozen status may change before the one-shot
evaluation. Folds 0/1, external outcomes, and official-test inputs remain out
of scope.

### 2026-08-01: protected ensemble fold-2 confirmation rejected

The fixed candidate was evaluated once from hash-binding commit `3c12fca2` on
all 8,833 fold-2 scenes (797 positives, 153 physical-site groups). Candidate
AP is 0.916916, only +0.000313 above the current spatial-Prithvi score. The
10,000-replicate paired-site interval is [-0.000541,+0.001131], so both the
fixed +0.001 minimum and positive-lower-bound gates fail. Sentinel-2 and
Landsat AP changes are positive but very small (+0.000145/+0.000166).
The complete operating confusion matrix is preserved: recall remains 0.956085
at FPR 0.071180. Relative to the released primary model, AP is +0.036317 and
matched-FPR recall +0.030113, but that does not rescue the failed improvement
over the stronger current model.

The independent dense path does generalize: anchored strength 0.10 improves
pixel IoU by +0.004367 with paired-site lower bound +0.002352. The scene path,
not segmentation, is therefore the failed mechanism. No artifact is promoted,
and folds 0/1, external outcomes, and the official test are not scored. Compact
result SHA-256 is
`4a55afb2dad0306f56ebad82289a06157f3855be18a22a4a3e567395c60fa138`.

Post-run access audit: the common `load_development` helper opened the existing
fold-0 and fold-1 feature-cache files while constructing a shared schema/index,
even though the estimator then masked all training arrays to folds 3/4 and all
prediction, label, and metric arrays to fold 2. Thus the generated report's
`folds_0_1_accessed: false` field must be interpreted as *not used for fitting
or scoring*, not as *files never opened*. This does not change any metric or
candidate decision, but future confirmation evaluators must use a restricted
fold loader so the field can be literal. The next architecture may be selected
on folds 3/4, but it must use a fresh untouched confirmation fold rather than
retuning against this failed fold-2 outcome.

### 2026-08-01: scene-aligned Gaussian ViT correction frozen

The prior Gaussian-ViT cross-fit exposed a specific mismatch with the published
large-scale synthetic task: all 288,000 synthetic requests per endpoint trained
the dense mask path with scene loss disabled. Its scene head was trained only
during the later 12,288-request joint phase. That experiment therefore proved
dense synthetic transfer but was not a strong test of synthetic scene-ranking
pretraining. The next controlled run keeps the same 16,000-template bank,
positive/unchanged-negative pairing, model capacity, optimizer, nine-epoch
exposure, endpoint seeds, and three real-joint epochs, while enabling scene BCE
and partial-AUC throughout synthetic pretraining. A fixed 0.25 protection gate
leaves low-score operating decisions exact.

The CUDA smoke gives direct evidence that the corrected path is active:
synthetic scene BCE is 0.709939, synthetic partial-AUC loss 1.012344, and the
scene-head parameter update has maximum absolute magnitude 0.001020. The real
joint step and native 200x200 mask/scene outputs are finite; peak allocation is
1,461,616,640 bytes for the 10,575,102-parameter ViT-U-Net. Five focused tests
pass, including exact identity below the protection gate. No held metric or
future result/artifact exists. The full folds-3/4 cross-fit is frozen at
`configs/mars_gaussian_scene_aligned_crossfit_protocol.json`; fold 2 will not be
reused, and folds 0/1, external outcomes, and the official test remain outside
this selection run.

### 2026-08-01: scene-aligned Gaussian ViT improves ranking but is rejected

Both fixed endpoints completed 288,000 paired synthetic requests and 12,288
joint requests. Synthetic scene learning is real rather than nominal: scene BCE
fell 0.639803 to 0.379344 for the model holding fold 3 and 0.656790 to 0.412188
for the model holding fold 4; partial-AUC loss fell 0.932021 to 0.550931 and
0.957253 to 0.586192. This corrects the prior dense-only pretraining omission.

At strength 0.25, pooled AP improves +0.002507, fold-3/fold-4 AP
+0.002487/+0.001137, and Sentinel-2/Landsat AP +0.003459/+0.001538. Pixel IoU
improves +0.002996. The candidate nevertheless fails: paired-site AP interval
[-0.001290,+0.005322] and IoU interval [-0.000915,+0.005762] cross zero, and
fold-4 matched-FPR recall loses one true positive (-0.001305). No artifact is
written. Compact result SHA-256 is
`6963e5ac8832011a4c39899d3097ff23afb3222ab6f1d2bbbc67cdc325190358`.

The strength curve supplies a narrower positive finding. Strengths 0.05 and
0.10 have strictly positive paired-site AP lower bounds (+0.000149 and
+0.000082) with unchanged pooled recall, but their point gains (+0.000949 and
+0.001597) remain below the preregistered +0.002 floor. Increasing Gaussian
weight is therefore retired: it trades stable small evidence for between-site
variance. The next development experiment may combine only these conservative
Gaussian residuals with independently learned protected DOFA/anchored evidence,
using a final local-logit gate that mathematically keeps every affected score
above 0.25. That combination must be preregistered before any per-scene Gaussian
cache is generated.

### 2026-08-01: exact Gaussian replay rejected; bounded replicate frozen

The frozen raw-logit replay completed both 80-minute cross-fit directions but
failed its deliberately strict 1e-9 validation at Gaussian strength 0.05.
Matched-FPR recall reproduced exactly; pooled AP and sensor AP did not. The
training trace independently confirms non-bitwise execution: endpoint-4 final
synthetic scene BCE was 0.410157 in the diagnostic attempt versus 0.412188 in
the committed reference. No cache, state, or success receipt was written, and
the exact protocol remains rejected rather than being relaxed after seeing the
failure. The compact audit is
`reports/experiments/mars_gaussian_scene_aligned_cache_replay_failure.json`.

The next attempt is therefore framed as a stochastic reproducibility replicate,
not an exact replay. Before its outputs exist, the tolerance is frozen at
0.001 AP-delta difference (half the downstream +0.002 promotion floor). Each
strength must independently retain positive pooled AP, positive AP in both
folds and both sensors, non-worse matched-FPR recall, and pooled/sensor AP
within that tolerance of the reference. A raw cache may be retained as ignored
research evidence, but the ensemble evaluator cryptographically binds its
receipt and rejects any strength not explicitly marked eligible. The ensemble
protocol was updated before the new cache exists; fold 2 and the official test
remain closed.

### 2026-08-01: conservative Gaussian-ViT plus DOFA ensemble preregistered

The next experiment tests whether two independently learned, individually
stable scene-ranking residuals are complementary. The first is the fixed DOFA
change/extreme probe previously selected with fit-fold-only normalization
(five fixed random projections, C=0.01, gate 0.50, weight 0.05). The second is
the scene-aligned 10,575,102-parameter physics-contrast ViT trained with
288,000 analytical Gaussian positive/unchanged-negative requests and 12,288
real/synthetic joint requests per cross-fit endpoint. Only Gaussian strengths
0.05 and 0.10 are eligible because both retained unchanged pooled recall and a
positive paired-site AP lower bound in the completed standalone experiment.

Both residuals are added in local-logit coordinates only where the current
score is at least 0.25; lower scores remain bitwise identical and every changed
score remains above the protection gate. The dense mask remains the separately
fixed anchored strength-0.10 path, which has positive point and paired-lower
IoU changes on folds 3/4 and the earlier one-shot fold 2. Promotion requires at
least +0.002 AP, unchanged operating confusion counts, positive AP on both
folds and sensors, a positive 10,000-replicate paired-site AP lower bound, and
the fixed dense evidence.

The raw Gaussian logits were not retained after the rejected standalone run,
so an exact deterministic replay is frozen first. It must reproduce the
committed pooled AP, recall, and per-sensor AP deltas within 1e-9 before its
ignored cache is accepted. The ensemble grid, code hashes, components,
bootstrap seed, and gates are frozen before that cache exists. The restricted
evaluator uses only folds 3/4 from the shared folds-2/3/4 cache and does not open
the separate fold-0/fold-1 files. Fold 2 and the official test remain closed.

### 2026-08-01: bounded Gaussian reproducibility replicate passes

The separately frozen stochastic replicate completed all 17,745 folds-3/4
scenes. Both conservative strengths pass every predeclared reproducibility
gate. Strength 0.05 improves AP +0.000689 with fold changes
+0.000526/+0.000621 and Sentinel-2/Landsat changes
+0.000919/+0.000474; strength 0.10 improves AP +0.001154 with fold changes
+0.001155/+0.000847 and sensor changes +0.001566/+0.000776. Matched-FPR
recall remains exact for both. Relative to the committed reference, pooled AP
differences are -0.000261 and -0.000443, within the frozen 0.001 tolerance.

The ignored 17,745-row raw-logit cache is 719,360 bytes with SHA-256
`68aaf9d5b0e6d650bfc70d8ea511f5a4d3838bbc353811ce11462214f96faa02`.
The compact receipt SHA-256 is
`0ad67fdf5d21e4d109b0fbec6e5f964a449f594812bfc872c752db84f32d0de1`.
Both hashes, and only the predeclared hash/status placeholders, are now bound
into the protected Gaussian+DOFA ensemble protocol before its one-shot
two-candidate evaluation.

### 2026-08-01: protected Gaussian-ViT plus DOFA ensemble passes development

The one-shot two-candidate evaluator selected Gaussian strength 0.10 combined
with the fixed DOFA change/extreme residual. On all 17,745 folds-3/4 scenes,
candidate AP is 0.906525, a +0.002449 improvement over the current
spatial-Prithvi score. The 10,000-replicate paired-site AP interval is
[+0.000489,+0.004068]. Fold-3/fold-4 AP changes are
+0.001722/+0.001968; Sentinel-2/Landsat changes are
+0.002942/+0.001692. The complete operating confusion matrix is preserved
(1,466 TP, 1,156 FP, 15,065 TN, 58 FN), so matched-FPR recall remains
0.961942 and FPR 0.071266.

Relative to the released MARS-S2L primary score, AP is +0.031241 and
matched-FPR recall +0.009843. The independently fixed dense path also remains
supported: IoU delta/lower bound are +0.002631/+0.000369 on development and
+0.004367/+0.002352 on the already completed fold-2 dense confirmation. Every
predeclared gate passes. Compact result SHA-256 is
`9473bc039e43bfc15d97b8eca0228fce66b31dffc8c487f8b3a4351c3280e1b5`.
This authorizes only a separately frozen external/new-cohort confirmation; it
does not authorize reopening fold 2 or the official test.

### 2026-08-01: external multi-cohort recommendation audited

The proposed unified-cohort direction is retained, but four claims were
corrected before implementation. The local foundation is the 5M-parameter
Prithvi-EO-2.0-tiny-TL checkpoint, not a 23M model. Its six-band HLS V2
pretraining uses 4.2 million 30 m samples, and the MARS broad B08 input is not
an exact match to the HLS narrow-NIR slot. The proposed 30-channel count was
internally inconsistent. Finally, a negative-score quantile does not guarantee
geographic-transfer FPR: conformal risk control bounds expected monotone loss
only under its exchangeability assumptions.

The audited training boundary contains 33,131 rows: 17,745 MARS folds-3/4
scenes, 14,859 MARS-test-disjoint MethaneS2CM crops, 160 UNEP positives, and
367 CloudSEN negatives. The untouched MethaneS2CM source-development partition
contains 11,478 rows and 48 groups. Before any v6 score exists, its groups are
hash-sorted into 24 risk-calibration groups (5,050 rows) and 24 confirmation
groups (6,428 rows). The partition, cohort counts, source hashes, licenses, and
prohibited resources are bound in
`reports/acquisition/mars_v6_unified_cohort.json`.

### 2026-08-01: product-aware dual-adapter Prithvi v6 implemented

V6 converts every product to physical DN/10,000 reflectance before applying a
bounded near-identity product correction. MARS tensors require a fixed 0.5
conversion because the released loader stores DN/5,000; MethaneS2CM L2A is
already DN/10,000. The canonical input contains target, 90-day reference,
365-day reference, observability, and two explicit availability flags.
Missing frames use a learned bounded product-specific imputation residual.

Scene and dense branches use separate rank-8 LoRA Prithvi encoders, product
harmonizers, and embeddings. The dense path uses a declared 47-channel physics
contract, four down/up stages, token fusion, and identity-initialized MBMP
gates. The scene path uses CLS, mean, and learned top-k patch pooling and adds a
bounded residual around the frozen champion. This makes objective decoupling
literal: synthetic data can update only the dense branch. Ten focused tests
cover physical equivalence, missing-frame behavior, channel count, branch
shapes, phase isolation, frozen-base preservation, protected identity, and
group-level risk calibration.

The real-checkpoint smoke found and corrected a phase-toggle bug that would
have unfrozen the Prithvi base. The corrected 14,193,203-parameter model has
338,502 trainable scene parameters and 2,586,601 trainable dense parameters.
Four real MARS/MethaneS2CM scene/dense updates are finite, exact below the 0.25
protection gate, and peak at 185,997,312 CUDA bytes. A separate mixed-source
scene smoke is finite at 214,051,840 bytes. No held-fold outcome was computed.

### 2026-08-01: v6 scene viability pilot frozen

Before any v6 held score exists, the one-seed scene pilot is frozen at
`configs/mars_v6_scene_pilot_protocol.json`. Each endpoint fits only the
opposite MARS fold plus fixed MethaneS2CM, UNEP-positive, and CloudSEN-negative
auxiliary cohorts. Four fixed epochs use 4,096 MARS-compatible and 4,096
MethaneS2CM requests each; the last epoch doubles the top-decile champion hard
negatives. Dense parameters and synthetic scene supervision are disabled.

The comparator is not the older spatial-Prithvi score. The already-selected
Gaussian-strength-0.10 plus DOFA champion was deterministically materialized
as a 17,745-row ignored cache with SHA-256
`988b98c92a1a5fa1fe52d7052b9159352f0fadd876b400fce1c8c879c94ea424`.
The pilot searches only residual strengths 0.025, 0.05, and 0.10 and requires
+0.003 AP, positive fold/sensor changes, non-worse matched-FPR recall, and a
strictly positive 10,000-replicate paired-site lower bound. Only a pass may
authorize a second seed and dense-v6 investment. MARS fold 2, folds 0/1,
official test, and held external outcomes remain closed.

### 2026-08-01: v6 scene viability pilot rejected

The frozen run completed in 1,619.6 seconds. Both endpoint losses decreased
over four epochs, confirming that the 338,502-parameter scene adapters were
trainable. The selected protected residual strength 0.05 reached AP 0.906707,
only +0.000182 over the Gaussian+DOFA comparator. The AP change was positive
but negligible on fold 3 (+0.000161), fold 4 (+0.000007), Sentinel-2
(+0.000295), and Landsat (+0.000064). Matched-FPR recall was unchanged.

The paired 25 km group-bootstrap interval was
[-0.000107,+0.000427], so the lower bound was not positive; the point change
also missed the preregistered +0.003 floor by an order of magnitude. The
experiment is rejected without a second seed or dense-v6 phase. It accessed no
external held outcome and did not reopen MARS fold 2, folds 0/1, or the
official test.

### 2026-08-01: unconstrained error-correction objective frozen

Post-run code inspection identified a mechanistic limitation in the rejected
v6 schedule: the MARS training loss saw only a strength-0.10 bounded protected
correction, limiting its effective local-logit change while also reducing the
pairwise gradient. The follow-up changes the fitting objective, not the held
data: protected MARS rows optimize `logit(champion) + residual`, while UNEP,
CloudSEN, and MethaneS2CM auxiliary rows retain direct scene logits. A small
L2 term regularizes only protected residuals.

At evaluation the residual is again bounded and applied only above the exact
0.25 gate. The four strengths and all promotion gates are fixed before
cross-fit scoring. Twelve focused tests pass, and the real mixed-source smoke
is finite at 214,051,840 peak CUDA bytes with no held outcome accessed.

### 2026-08-01: unconstrained error-correction pilot rejected

The frozen two-endpoint run completed in 1,645.3 seconds. Losses were finite
and the residual magnitude grew enough to test the mechanistic hypothesis.
Nevertheless, the AP strength curve was monotonically adverse: -0.001376,
-0.003785, -0.008486, and -0.021014 for strengths 0.10, 0.25, 0.50, and 1.00.
At the selected smallest strength the paired 25 km group interval was entirely
negative, [-0.002920,-0.000085], and both folds and both sensors regressed.
The exact protection gate retained unchanged matched-FPR recall.

The architecture is rejected before replication, dense-v6 work, and external
confirmation. No MARS fold 2, fold 0/1, official-test, or external held
outcome was accessed. Compact result SHA-256:
`1da707289767e9a4f840f86629770089406813dd2819a86ad6b2d1be35c4672a`;
ignored raw-logit cache SHA-256:
`986bcbf2a3a920a4ae5bb617c2a696863ab66a591c2cb70735b5c182d79e71c2`.

### 2026-08-01: official Prithvi-100M representation acquired and preregistered

The official IBM/NASA 100M temporal/location checkpoint was acquired at
immutable Hugging Face revision
`2c84e383194986040f883cc43d7869002c425e1b`. Its 454,660,610-byte state
strictly loads 112,639,492 parameters, including 86,237,186 encoder
parameters, with no missing or unexpected keys. The Apache-2.0 source files,
checkpoint identity, and model configuration are recorded in
`reports/acquisition/prithvi_eo_2_100m_tl.json`; bulk files remain ignored.

Frozen extraction completed all 17,745 folds-3/4 scenes in 1,535.8 seconds.
The compact four-block CLS contract yields 3,072 float16 features per scene
in a 102,807,297-byte ignored cache. An independent audit confirms finite
values and exact champion identity/label/sensor/fold/group alignment.

Before probe outcomes, three L2 values and four conservative champion blend
weights are fixed, together with the worst-fold-first rank and +0.003 AP
promotion gate. Target-fold normalization uses no target labels; training is
site/label balanced. The finite probe smoke accesses no held metric.

### 2026-08-01: Prithvi-100M CLS probe rejected

All six cross-fold logistic fits converged. The complete 12-candidate grid was
monotone-degrading with blend strength. The worst-fold-first selector chose
C=0.001 and blend 0.025, but AP changed -0.000288, matched-FPR recall lost one
positive (-0.000656), and both folds regressed. Sentinel-2 changed -0.000451
AP while Landsat changed +0.000114. The 10,000-replicate paired-site interval
[-0.000666,+0.000083] crosses zero.

No promotion gate passes, so no folds-0/1 extraction, external scoring, or
official replay is authorized. Compact result SHA-256:
`d5cd5ad6457d0234f186ac42b638c61886df77c3c9003eb566a7f4e1897ad6e9`.
This negative result separates representation scale from the remaining
ranking ceiling: a 20x larger nominal Prithvi encoder does not improve the
champion under the same two-frame physical contract.

### 2026-08-02: honest group conformal-risk transfer audit rejected

The external recommendation's calibration claim was tested in its corrected
form rather than as a negative-score quantile. For each direction, bounded
within-25 km-group false-positive losses calibrated a threshold on one cross-fit
fold and that fixed threshold was evaluated on the other. The finite-sample
CRC correction was `(n * empirical risk + 1) / (n + 1)`; its claim remains
limited to expected group-balanced risk for future exchangeable groups.

At the preregistered alpha 0.075 target, fold 4 -> fold 3 transferred: held
crop FPR was 0.071509, group-balanced FPR 0.042708, and recall 0.972296.
Fold 3 -> fold 4 did not: held crop FPR rose to 0.165159 and group-balanced FPR
to 0.083104, despite a calibration bound of 0.074963; Sentinel-2 crop FPR was
0.182782 while Landsat was 0.062552. The pooled cross-fit group FPR of
0.059485 hides that directional failure, while pooled crop FPR is 0.118735 and
recall 0.974409.

Group CRC is therefore retained only for exchangeability-scoped curves and
empirical transport diagnostics. It is rejected as an operational substitute
for the current threshold and cannot be described as a geographic-transfer
FPR guarantee. This is direct evidence that product/geographic data coverage,
not a threshold formula, remains the primary bottleneck. The result used only
folds 3/4 and accessed no fold 2, folds 0/1, external held outcome, or official
test. Compact result SHA-256 is
`aa717d6500efeddcecfb05c6db3ee4ec6d3875ec8c92c66b971a1e7a67cf449c`.

### 2026-08-02: Earthdata/USGS acquisition boundary verified

The user's authenticated Earthdata browser session is valid, but the exact
Landsat Collection 2 Level-1 LandsatLook asset redirects independently to the
USGS EROS Registration System. The public USGS STAC item confirms the exact
same product and its requester-pays `s3://usgs-landsat` alternate, but this
machine has no AWS account/profile. An EROS sign-in tab is left for user
handoff. No alternate Landsat product, tier, or processing level is used.

### 2026-08-02: controlled-release negative source audited before scoring

The Stanford 2022 single-blind satellite campaign is the first candidate found
with metered zero-release Sentinel-2/Landsat observations rather than negatives
inferred from catalog absence. Its public CC-BY-4.0 table contains 20 relevant
overpasses at the Casa Grande, Arizona release site: eight primary positives at
or above 1,000 kg CH4/h, nine primary negatives at or below 10 kg CH4/h, and
three intermediate-rate challenge observations. Nineteen exact L1 targets
resolve through public STAC; one Sentinel-2 acquisition is absent from the
current catalog.

The mandatory overlap audit then found the site already represented by 677
rows in upstream MARS metadata, including eight exact target products. Every
same-site and exact-match row is in the excluded `Not Used` split, so none was
part of the paper train/development/test contract. The cohort remains eligible
as a previously unscored, post-freeze fixed-site diagnostic, but it is neither
source-disjoint nor geographically independent and cannot provide a paired-site
superiority claim. No model score or crop outcome was accessed.

MARS `background_image_tile` must not be converted into a no-plume label. The
authors' selector defaults `query_only_images_without_plumes` to false and may
accept an acquisition containing a catalogued plume when its wind direction is
not similar to the target plume. Background products remain temporal references
only unless an independent absence label exists. This rules out a tempting but
invalid way to manufacture thousands of negatives.

### 2026-08-02: large controlled-release source workbook audited before scoring

The April 2026 Stanford preprint and its public CC-BY-4.0 data repository were
audited as a stronger source of intentional no-release controls. The compact
paper source workbook, not the per-release summary files, is the authoritative
cohort universe: the summary files omit blank controls. Applying the paper's
experiment-team QC and acquisition filters yields 262 unique 2025 observations
at Casa Grande: 174 Sentinel-2 and 88 Landsat. The frozen visibility-aware
contract contains 136 exact zero-release negatives, 13 primary positives at or
above 1,000 kg CH4/h, and 113 nonzero sub-threshold challenge observations.

Public STAC resolves 257 exact Level-1 products; five remain unresolved. No
exact 2025 target product appears in the upstream MARS metadata. The Casa Grande
site itself is present in 677 upstream MARS rows, all under the excluded `Not
Used` split (669 negative, eight positive). The cohort is therefore temporally
new but neither geographically nor source-disjoint. With one physical site it
cannot support site-block inference, but it is a strong post-freeze operating-
point and false-positive stress test. No ERSRR model score was accessed.

The raw workbook, clean event rows, and resolved manifest remain ignored under
`.research/stanford_controlled_release_2024_2025/`. The tracked audit is
`reports/acquisition/stanford_large_controlled_release_cohort.json`.

### 2026-08-16: released-detector recall-anchor feasibility established

The Stanford one-shot results exposed a repeated sensitivity failure: released
MARS detected the only >=1,000 kg CH4/h event at both Casa Grande and Evanston,
while Gaussian+DOFA and Spatial-Prithvi missed both at scores far below their
frozen thresholds. A committed, development-only diagnostic therefore tested
whether the released detector contains complementary sensitivity on the still
authorized MARS folds 3/4. It did not fit or select a candidate and did not open
folds 0/1/2, any external outcome, or the official paper cache.

At separately matched 7.13% FPR, Gaussian+DOFA reached AP 0.906525 and recall
0.961942, while released MARS reached AP 0.871465 and recall 0.946194. The
champion uniquely recovered 33 positives and the released detector uniquely
recovered nine; each unique-decision cell also contained 220 negatives. Under
the released paper rule (`score > 0.5`), the released detector identified 11
champion-missed positives among 457 rescue-eligible rows. The region is therefore
low precision but genuinely complementary rather than redundant.

The rescue true positives and false positives are distinguishable by existing
model-aligned evidence. The strongest standardized contrasts include primary
area above 0.9 (+1.384), target-channel top-100 mean (-1.201), primary connected
score (+0.851), primary top-200 mean (+0.841), and primary/released logit-delta
mean (+0.819). Because these quantities were already extracted without outcome
access and the maximum absolute contrast exceeds the frozen 0.25 feasibility
floor, all preregistered diagnostic checks pass. This authorizes a separate,
constrained dual-expert architecture pilot; it does not authorize a holdout replay
or a threshold change. The authoritative result is
`reports/experiments/mars_recall_anchor_diagnostic.json`.

### 2026-08-16: deterministic dual-teacher rescue rejected

The first frozen rescue pilot deliberately avoided fitting another scene head. It
retained Gaussian+DOFA everywhere except the low-champion/high-released route and
could only raise scores using the minimum, geometric mean, or topology-weighted
consensus of released and primary dense evidence. Three semantically anchored
rescue strengths produced nine fixed candidates. Promotion required at least
+0.005 AP, strictly positive pooled matched-FPR recall, nonnegative fold/sensor
effects, and a positive paired 25 km-group AP lower bound.

Every candidate was adverse. The least harmful setting used the conservative
minimum anchor at weight 1/3, raised 19 positives and 562 negatives, changed AP
-0.000347, and changed matched-FPR recall -0.004593. Both folds regressed and the
paired AP interval was [-0.001139,+0.000689]. Stronger rescue increased the number
of promoted false positives and worsened recall. The low-precision released-only
region therefore cannot be used through an unconditional bounded OR, even when
the two dense teachers agree. The deterministic family is retired without any
external or official replay. Result:
`reports/experiments/mars_dual_teacher_rescue_folds34.json`.

### 2026-08-16: selective proposal-verifier transformer rejected

A preregistered cross-fitted verifier then tested whether the deterministic
rescue failed only because it lacked local spatial context. The 122,051-parameter
transformer received nine frozen, label-independent 64x64 evidence maps and was
trained only on released-positive proposals. Fold 3 was predicted by fold-4
fits and vice versa; two fixed seeds were averaged. The verifier could only
raise Gaussian+DOFA inside the frozen low-champion/high-released route.

The verifier learned the proposal labels in isolation (held-route AP ranged
from 0.915 to 0.966 across endpoints), but it did not identify any useful
champion error. At the least aggressive rescue weight it raised 42 rows, all
negative, changed pooled AP by -0.000024, and left matched-FPR recall unchanged.
Both folds and both sensors lost AP, and the paired 25 km-group interval was
strictly adverse at [-0.000055,-0.000006]. The stronger fixed weight raised 98
negatives and no positives. All promotion gates failed, so no state, score,
external-cohort, or official-test artifact was produced.

This closes the released-proposal rescue branch: an unconditional consensus
rule and a learned spatial verifier both fail to add complementary ranking to
the folds-3/4 champion. Further scene-head variations on the same frozen cohort
are not evidence-based. The next architecture must change the information
available to the model or train on new geographically independent real events.
Result: `reports/experiments/mars_selective_proposal_transformer_folds34.json`.

### 2026-08-03: Casa Grande controlled-release one-shot rejected for superiority

The frozen label-free scorer completed all 169 exact Sentinel-2 L1C pairs before
outcomes were joined once. The primary view contained 86 metered-zero negatives
and eight releases at or above 1,000 kg CH4/h across 87 UTC-date blocks.
Released MARS achieved AP 0.09559, recall 1/8, and FPR 13/86. Gaussian+DOFA
achieved AP 0.08724, recall 0/8, and FPR 7/86; Spatial-Prithvi achieved AP
0.08730, recall 0/8, and FPR 1/86.

The candidate FPR reductions were paired-date conclusive, but both candidates
lost the released model's only primary true positive. Their AP intervals crossed
zero and recall-delta intervals were [-0.5, 0.0]. Both superiority gates failed.
Casa Grande remains one-site temporal evidence and cannot establish geographic
transfer. Compact report SHA-256:
`6b148e85a9d25ef9627a169ef2c7c2d89c963418a6c693aeb16a35a90a7e53b0`.

### 2026-08-03: independent Evanston one-shot completed without negative controls

Nine outcome-blind Evanston targets were resolved to exact official Sentinel-2
L1C products, paired with prior-only same-MGRS references, cropped to six-band
256x256 uint16 tensors, and scored once with the unchanged Casa scorer and
thresholds. The immutable score bundle was validated before the outcome adapter,
nine official summary-file identities, `ch4_kgh_mean` strata, metrics, and
uncertainty were frozen. All nine public sources then matched Stanford repository
size/SHA-1/MD5 metadata and joined exactly once.

The outcome cohort contained eight nonzero sub-threshold challenge releases, one
release at or above 1,000 kg CH4/h, and zero no-release controls. Released MARS
detected the sole primary positive and 2/8 challenges. Gaussian+DOFA detected
0/1 and 1/8; Spatial-Prithvi detected 0/1 and 0/8. AP, AUROC, and FPR are
undefined because the primary view has no negative row. Exact McNemar p=1.0 for
each candidate comparison, and both superiority gates fail.

Evanston is valid independent-site sensitivity/challenge evidence, but it is not
an independent negative cohort and cannot support broad geographic
generalization. No threshold or model may be retuned from this outcome. Compact
report SHA-256:
`372e963fa335b9e0506770352480db25a224518a67ca220b3bca6f45368b5917`.
The cross-site interpretation and reproducibility bindings are in
`reports/research/STANFORD_CONTROLLED_RELEASE_EVIDENCE_SYNTHESIS.md`.

### 2026-08-16: Zhao DSAN public cohort acquired and excluded for benchmark leakage

The official Science Data Bank archive associated with Zhao et al. (ACP 2025)
was acquired from `https://doi.org/10.57760/sciencedb.15792` and verified against
the publisher's MD5. The 70,398,286-byte archive contains 1,627 rendered RGBA
PNGs (508 plume and 1,119 no-plume) from six named sites but no georeferenced
six-band tensors or dense plume masks. The paper and its official Tables 2--3
remain useful product-alignment/DSAN literature.

Every public site lies within 0--0.216 km of geography already represented in
both MARS development and the official paper-test contract. The cohort therefore
contributes zero eligible model rows: using it for training, calibration, or
confirmation would leak benchmark geography and its rendered PNG contract cannot
reproduce the MARS multispectral input. Bulk files remain ignored. The compact
audit is `reports/acquisition/zhao_dsan_2025_dataset_audit.json`.

### 2026-08-16: Project Eucalyptus identifies reference choice as new information

The official Project Eucalyptus repository was pinned at revision
`46f719b7003eeb9ed604b8f1dd4616aa1ba1def0`. Its Sentinel-2 pipeline uses two
cloud-filtered prior references, documents false negatives caused by methane-
contaminated or surface-mismatched references in controlled releases, and
proposes averaging the last 5/10 overpasses, similarity selection, or learned
attention. This supports a materially new information path rather than another
scene head over MARS's fixed target/reference pair.

The released Eucalyptus model is not loaded: its 24-channel input requires B05,
B07, and B8A, which are absent from MARS's six-band contract, and its whole-object
PyTorch pickle is not a weights-only artifact. The research path uses only the
documented failure mechanism and continues with the trusted released MARS model.

### 2026-08-16: strictly prior reference bank is feasible with explicit fallback

Before any model score or outcome was opened, a selector scanned only folds 3/4
and used B02/B03/B04/B08 texture and radiometry to choose up to five strictly
earlier, >=95%-clear, same-site/sensor/MGRS, exact-grid Sentinel-2 observations.
No B11/B12 value, label, mask, flux, prediction, or model score influenced the
bank. Bulk descriptors and the 17,745-row selection manifest remain ignored.

The original preregistered audit correctly records a formal FAIL: although
14,569/14,963 Sentinel-2 rows (97.37%) have an exact-grid reference and
13,733 (91.78%) have five, its all-or-nothing gate required all 14,681 rows with
any same-key prior to have an exact-grid prior. Small grid-origin shifts excluded
112 rows. That result remains immutable in
`reports/acquisition/mars_prior_reference_bank_folds34.json`.

A separate post-feasibility engineering adjudication disclosed those coverage
facts before freeze and asked only whether exact filtering plus exact original-
pair fallback is representative enough for label-free inference. It passed the
frozen >=95% any-reference, >=90% five-reference, <=1% grid-exclusion, and
outcome-blind gates: 97.3668%, 91.7797%, 0.7629%, and true, respectively. This
authorizes only a separately frozen released-model score extraction; it is not a
performance result. The adjudication is
`reports/acquisition/mars_prior_reference_bank_adjudication.json`.

### 2026-08-16: alternate-reference evidence extractor frozen

The dedicated label-free extractor keeps each original target, wind vector, and
target cloud mask fixed; substitutes only a selected candidate's first six
target bands (never its background bands); recomputes MBMP; and scores the
original plus five prior-reference views with the hash-pinned released MARS
U-Net. It stores scalar connected/dense summaries and ignored 32x32 mean/max
probability maps for a later frozen aggregation diagnostic. It cannot read plume
masks, enhancement rasters, labels, champion scores, external outcomes, or any
official-test value.

A real outcome-blind four-row smoke covered two five-reference Sentinel-2 rows,
one Sentinel-2 fallback, and one Landsat fallback. Recomputed original-reference
scores matched the frozen released-score cache with maximum/mean absolute drift
0.0000409/0.0000139 and identical paper-rule decisions. Nine focused tests passed.
The full folds-3/4 extraction was frozen in
`configs/mars_prior_reference_score_extraction_protocol.json` before execution.

### 2026-08-16: prior-reference temporal architecture rejected before fitting

The full label-free extraction completed all 17,745 folds-3/4 rows. It reproduced
the original released connected score with maximum/mean absolute differences
0.001186/0.0000475 and identical `>0.5` paper decisions. The ignored artifacts
contain six-view scalar evidence (5.23 MB) and 32x32 probability maps (436.10 MB);
the compact receipt is
`reports/acquisition/mars_prior_reference_released_scores_folds34.json`.

After a separate protocol and synthetic tests were committed, the already-opened
folds-3/4 outcomes were joined once. All five outcome-independent aggregations
were strongly adverse versus the original released view. AP deltas were -0.0951
(nearest), -0.0633 (median), -0.0728 (top-two mean), -0.0552 (similarity-
weighted), and -0.1044 (maximum); matched-FPR recall deltas ranged from -0.0617
to -0.0958. Every fold and Sentinel-2 AP delta was negative, and all five paired
25 km-group 95% AP intervals were strictly below zero.

Max-over-priors did recover 25 champion-missed positives across 11 groups and
both folds, but also 3,009 negatives: 0.824% precision, below the frozen 3%
feasibility floor. Although TP/FP scalar contrasts were large (maximum absolute
standardized difference 1.398), no fixed aggregation improved AP at all. Both
independent continuation routes therefore fail. The reference-set temporal
transformer is retired before fitting; no model, fusion weight, threshold,
external cohort, or official cache was opened or promoted. Authoritative result:
`reports/experiments/mars_prior_reference_complementarity_folds34.json`.

### 2026-08-16: train-only MARS-Hyperspectral transfer feasibility retired

The acquisition protocol was committed before mask access at pinned dataset
revision `74b3d3132d135fee1761df1dadb7d662a4b5245b`. Stage A used only safe
identity, split, time, country, sector, clear-fraction, and coordinate fields;
it passed after a complete public Copernicus metadata query found 122 eligible
Sentinel-2 candidates among the initially coordinate-resolved subset.

Stage B then downloaded only published train-split `plumemask.tif` and optional
`info.json` files. The manifest records 7,041/7,041 masks, 6,899 optional info
records, 16,056,560 bytes, and per-file SHA-256 values; validation/test/full-tile
masks and all hyperspectral retrieval/RGB products remained sealed. Mask pixels,
never CSV `isplume` or flux fields, define 3,349 positives and 3,692 reviewed
negatives. All masks have either `{0}` or `{0,1}` values. Mask-derived centers
match 1,295 source coordinates to 0.01199 km median and 0.02095 km maximum.

The frozen clear/protected-site filters leave 2,345 catalog-eligible train crops
(1,008 positive, 1,337 negative), 390 connected 25 km components, 380 components
more than 25 km from every MARS-S2L location, and 59 countries. Rate-limited,
checkpointed metadata queries covered 1,466 grouped crop requests in each target
catalog. Sentinel-2 L1C alone yields 447 deduplicated scene pairs: 286 positive
within six hours, 161 negative within one hour, 270 total within one hour, 81
within 15 minutes, 155 novel groups, and 41 countries.

The preregistered Landsat extension used the exact USGS `landsat-c2l1` catalog
and only LC08/LC09 items. Combined with Sentinel-2, it yields 772 scene pairs,
500 positives, 272 negatives, 460 total within one hour, 137 dense-reprojection
candidates within 15 minutes, 204 novel groups, and 49 countries. The positive,
high-confidence, geography, and country gates pass; the frozen 300-negative gate
fails by 28. The result therefore does not authorize target-band download or
modeling. No gate was lowered, Landsat 7 was not added post hoc, and no protected
model outcome was opened. Detailed catalogs remain ignored; compact authoritative
receipts are `reports/acquisition/mars_hyperspectral_transfer_stage_b.json` and
`reports/acquisition/mars_hyperspectral_transfer_stage_b_sentinel2.json`.

### 2026-08-16: JPL CACH4 metadata supplement rejected conservatively

The released JPL operational-GHG train definitions contain 3,332 CACH4 rows:
3,149 background and 183 plume crops on 124 AVIRIS-NG flightlines. Header-only
acquisition retrieved 124 public CMF ENVI sidecars totaling 33,011 bytes. It did
not retrieve source rasters, labels, the released JPL test definition, or any
target-satellite asset. All 3,149 background rows have exact flight UTC and
finite WGS84 centers. The source sampler pads edge windows, explaining 1,015
sample-axis overhangs, 159 line-axis overhangs, and 1,131 rows overhanging either
axis; all crop centers remain in bounds.

After the hash-bound safe-column filter, 390 rows are official-test protected,
194 duplicate already-counted MARS-Hyperspectral negative geography, and 2,565
are row-eligible. Those rows collapse to only 15 connected 25 km physical
components despite spanning 104 flightlines. The preregistered minimum is 20,
so the result is FAIL and target-catalog querying was never authorized. Counts,
distance summaries, protocol identities, and ignored-output hashes are recorded
in `reports/acquisition/jpl_operational_ghg_negative_supplement_metadata.json`.

### 2026-08-16: NASA ORNL header bridge preregistered, authentication pending

Primary NASA documentation for DOI 10.3334/ORNLDAAC/2095 states that each
flightline's L1B `img` is orthocorrected into a rotated UTM grid and accompanied
by a text ENVI header containing samples, lines, datum, map reference, pixel
size, and units. CMR collection `C2662359874-ORNL_CLOUD` independently resolves
the known CACH4 anchor to
`AVIRIS-NG_L1B_radiance.ang20180821t184959_rdn_v2t1_img.hdr`, 14,074 bytes,
SHA-256 `67b842bc81d95db3104ce04adde1b8e52124d332e2ba5182d9a470430f3b766b`,
with acquisition beginning 2018-08-21T18:49:59Z.

Commit `6aa596e1` freezes the bridge before any flight-specific header download.
At least 100 of the 124 CACH4 anchors must resolve, at least 80% overall, and
every resolved NASA header must match the public JPL CMF dimensions/CRS and five
control-point centers within 0.25 pixel. A pass alone would authorize metadata
resolution for 3,317 COVID plus 10,127 Permian train backgrounds. CACH4 is not
pooled into the new cohort; COVID/Permian alone must retain at least 20 eligible
25 km components under identical official-test and prior-pair exclusions.

The public CMR query works, and JPL's public directory contains only the 2018
CACH4 product family. NASA routes the actual header through Earthdata Login;
the connected browser session currently presents empty login fields, and no
command-line `.netrc` exists. Therefore the bridge remains pending explicit
Earthdata authorization. No credential, header, COVID/Permian coordinate,
target catalog, or protected outcome has been accessed yet.

### 2026-08-16: public CMR anchor preflight passes metadata resolution

A separately committed preflight queried only the 124 frozen CACH4 anchor IDs
in the preregistered NASA collection. All 124 exact `v2t1` L1B radiance-header
granules resolved, all 124 carried source SHA-256 values and declared byte
counts, and the total declared protected content was 1,745,203 bytes. The
ignored detailed receipt is bound by SHA-256 in
`reports/acquisition/jpl_operational_ghg_ornl_stage_a_cmr_preflight.json`.

This result does not pass the grid bridge: no header content was accessed, no
grid was compared, and no COVID/Permian or target-catalog query was authorized.
It removes CMR identity/availability as a risk while leaving Earthdata
authentication and the zero-mismatch geometric comparison as the next gate.

### 2026-08-16: STARCOP chosen for a sparse independent-negative audit

The next candidate was chosen on label provenance, exact time/georeferencing,
license, geography, and sparse-access feasibility rather than raw negative-row
count. STARCOP (Zenodo 7863343; paper DOI 10.1038/s41598-023-44918-6) is the
first route. Its authors construct no-plume training windows from regions that
do not intersect known plume annotations, explicitly repair a set of missed
plume windows, and release a `labelbinary.tif` for each 512x512 chip. The
negative claim is therefore limited to no annotated/refined plume in a released
train window. It is not a physical-zero claim.

Commit `241f07d0` was made before `train.csv` or any archive member was opened.
The protocol pins the 1,038,970-byte manifest and six train-only archive hashes,
forbids all test content, and selects at most four negatives per flightline by
the lexicographically smallest `(SHA-256(id), id)` values. Only after Stage A
passes may ZIP end/central-directory ranges and exact selected zero-mask members
be read. STARCOP must independently clear 100 resolved rows, 50 flightlines, 20
leakage-safe 25 km components, and later 28 <=1-hour target pairs. Bulk imagery,
positive members, target catalogs, and MARS outcomes remain forbidden.

The ranked fallback is Carbon Mapper's Tanager-only assessed-scene catalog.
Primary documentation confirms that an optimal-observation null-detect
candidate is a <=25%-cloud scene covering a known source with no detected plume
above the sensor limit. The API exposes `assessment_status`, Tanager instrument,
UTC, bounds, assessed cloud percentage, plume counts, and source-level
null-detection scene associations. Any use requires a separate frozen query and
must attach a zero scene to a reviewed source coordinate; a scene centroid is
not a no-plume label. GHGSat landfill nulls are globally stronger but excluded
pending written permission because the public deposit is All Rights Reserved.
Turkmenistan-only He et al. and CH4Net datasets are deferred because their many
rows likely collapse to already represented geography and CH4Net's ND license
makes downstream training/publication unclear. No fallback catalog or archive
was queried during this ranking.

### 2026-08-16: STARCOP Stage A passes before archive access

After the guarded auditor and tests were committed, the exact hash-bound
`train.csv` was downloaded from Zenodo. It contains 3,425 rows split almost
evenly between 1,713 author-labeled no-plume and 1,712 plume chips. All 256
flightlines carry negatives. The frozen blind hash rule selected 1,009 negative
IDs, with an observed maximum of four per flightline. The negative-row,
negative-flightline, and per-flight cap gates all pass; `qplume`, coordinates,
geography, and any model output were not used for selection.

The first parse exposed two documented generator semantics. STARCOP's `id`
retains the original source-window offsets and dimensions, whereas the released
cached chip's `window_*` columns are reset to 0/0/512/512. Positive source
windows can be 151x151 and one padded positive has source row -29. The pinned
official `WindowDataset.cache` code performs this reset, every negative source
window is 512x512, and every cached row declares a 512x512 local chip. The
auditor was corrected and recommitted without changing selection or thresholds.

The compact receipt SHA-256 is
`a80189174bf7bbe795720c0757000733e5d020fc56f4e85b2b4801449d368747`;
the ignored selected JSONL contains 1,009 rows and SHA-256
`28589941ad79d1c8cb1baac38923e9c4027950ce70618c83bab99fe357b36d19`.
No ZIP, label mask, imagery, test content, target catalog, coordinate, or MARS
outcome was opened. Stage B is now authorized under the existing sparse
zero-mask protocol, but target-catalog access remains forbidden.

### 2026-08-16: STARCOP zero masks validate, but independent geography fails

The guarded ZIP64 range reader resolved the six exact train archives and only
the 1,009 blind-hash-selected `labelbinary.tif` members. All 1,009 masks are
valid 512x512 projected single-band GeoTIFFs, all pixels are zero, and all chip
centers transform to finite WGS84 coordinates. The selected cohort spans all
256 negative-bearing flightlines, so the row, flightline, and grid-validity
gates pass.

The unchanged leakage audit rejects the source. There are 984 rows within 25
km of official MARS-S2L test geography, leaving 25 rows on 14 flightlines in
only eight 25 km connected components. The requirement was 20. None of the
rows duplicates a counted MARS-Hyperspectral negative within 25 km. STARCOP is
therefore retired without a target-catalog query, threshold change, or pooling
with CACH4/NASA candidates. A released STARCOP zero mask remains an
author-refined benchmark no-plume label, not proof of physical zero methane.

The full acquisition chronology is retained in `docs/RESEARCH_LEDGER.md`.
Notably, standard ZIP data descriptors and deterministic archive-root prefixes
were source-format corrections made before selected label access; later
transport was paced, bounded, and resumable. The final compact report scopes
range totals to its successful execution and separately discloses that exact
label-range bytes from earlier pre-resume failures were not persisted. The
1,009 unique masks (1,935,966 bytes) remain ignored; tracked artifacts contain
only compact counts, gates, hashes, and limitations.

Research implication: STARCOP confirms that additional raw negative rows do
not solve the project bottleneck when their geography is already represented
by protected MARS sites. The next source decision stays at the data layer, not
the architecture layer. A Tanager-only Carbon Mapper assessed-null metadata
protocol is the preferred fallback; it must first demonstrate independently
reviewed source-level null associations, exact scene UTC/bounds, adequate
license terms, and at least 20 leakage-safe 25 km components before any target
asset or model training is allowed.

### 2026-08-16: Tanager source-level null protocol preregistered

The preferred fallback is now frozen before any Carbon Mapper catalog row is
opened. Hash-pinned official documentation supplies the precise label: a
source response must mark the Tanager scene `counts_as_null_detection=true`
and `counts_toward_daily_emissions=true`, while both `has_detection` and
`has_non_null_emission` are false. The scene must be public, production-phase,
annotated, <=25% assessed cloud, and geometrically contain the source point.
This is stronger and more local than filtering whole scenes to zero plumes.

The preregistered source must independently retain 50 null source-scene pairs,
28 scenes, 30 sources, and 20 25 km components novel beyond all MARS-S2L
locations before a separate target-catalog protocol can exist. The applicable
license is non-commercial, attribution-required, and share-alike; downstream
checkpoint and data-release terms are recorded now rather than deferred until
publication. The protocol SHA-256 is
`06b2058a05ecf748dac25cb7389bb62e9fe88e36859dded389fc03b13d5b0ad0`.

### 2026-08-16: Tanager metadata acquisition fails closed

The first exact public `sources.geojson` request exceeded the preregistered
8 MiB per-response cap before enumeration completed. The failure-accounting
path was then corrected and committed—without changing the endpoint, cap,
selection, or population gates—and the reproduced attempt recorded at least
8,414,189 streamed bytes against an 8,388,608-byte maximum. The response was
not cached as complete, so no source feature or population gate was evaluated.

Result: Carbon Mapper **FAILS** at metadata acquisition and is retired under
the frozen protocol. No image, target-satellite catalog, protected MARS
outcome, or target pair was accessed. The tracked compact report is
`reports/acquisition/carbon_mapper_tanager_null_metadata.json`, SHA-256
`730048bf557a9024ea97282d6bd042d3093cd29fec983cffe6ce7142694a088b`.
This is not a negative finding about the underlying Tanager null population;
it is a disciplined refusal to alter a preregistered transport bound after
seeing live response behavior.

Research implication: three independent negative-source candidates now show
that data acquisition and geographic novelty—not encoder capacity—are the
binding constraints. Failed sources remain non-poolable. The next path should
either finish the separately frozen authenticated NASA grid bridge or design
the next architecture experiment around only the already qualified
MARS-Hyperspectral transfer cohort.

### 2026-08-16: spectral-temporal extended Prithvi pretraining frozen

The next model branch changes the representation before supervised fitting.
Prithvi-EO-2.0 is already a multi-temporal HLS masked autoencoder, but its
source domain is 30 m harmonized surface reflectance with narrow NIR. MARS
uses mixed Sentinel-2 L1C and Landsat TOA crops at finer native resolution,
including Sentinel-2 wide B08. A local audit also found that the earlier
Prithvi feature path multiplied the adapter's DN/5,000 tensor by 10,000,
feeding twice the physical producer scale. The HLS channel names B05/B06/B07
still mean narrow-NIR/SWIR1/SWIR2, so this is not described as a band-order
bug; the verified issues are product, resolution, NIR-response, and scale
shift.

The frozen experiment continues the MAE objective separately for each held
fold. A fold-3 endpoint sees only fold 4 pixels; a fold-4 endpoint sees only
fold 3. Each request combines equal MARS and fixed MARS-disjoint
MethaneS2CM auxiliary counts. Dedicated unlabeled readers open reflectance and
cloud arrays only: no scene label, plume mask, or plume asset is read. Inputs
are restored to producer DN/10,000 exactly, location is a constant zero
sentinel, and only acquisition time remains as metadata.

The encoder adapts its patch embedder, all-block rank-8 attention LoRA, and
final transformer block. The disposable MAE decoder retains two pretrained
blocks. The 75% masking objective is MAESTRO-inspired patch-group-normalized
L1 over visible / NIR / SWIR groups plus 0.2 temporal-difference L1. This
combines ExPLoRA's parameter-efficient extended-pretraining principle with a
spectral prior matched to methane change detection. After LoRA is merged, a
fresh last-four-block LoRA residual head fuses reference, target, signed
change, and absolute change patch tokens. Its final layer is initialized to
exactly zero and its correction is bounded, so an untrained candidate exactly
reproduces the frozen Gaussian+DOFA champion.

The real-data smoke opened no outcome metric and passed after catching and
fixing one CPU/CUDA adapter-placement defect. With the final two-block decoder,
8,043,650 parameters are trainable during MAE adaptation and 422,994 during
scene fitting. Reconstruction, temporal-change, patch, pair, and scene losses
were finite on real folds-3/4 and auxiliary tensors. The production protocol
uses 1,200 balanced steps and four scene epochs. Seed one must first add at
least +0.002 AP with both folds/sensors positive, nonnegative matched-FPR
recall, and a positive paired-site lower bound. Only then may the frozen second
seed run. Final promotion requires +0.005 AP, the same strict subgroup/recall
gates, a positive paired-site lower bound, and an independently positive
second seed. Folds 0/1/2 and the official test do not appear in the trainer's
input contract.

Primary research basis: Prithvi-EO-2.0
(<https://github.com/NASA-IMPACT/Prithvi-EO-2.0>), ExPLoRA
(<https://proceedings.mlr.press/v267/khanna25a.html>), and MAESTRO
(<https://openaccess.thecvf.com/content/WACV2026/html/Labatie_MAESTRO_Masked_AutoEncoders_for_Multimodal_Multitemporal_and_Multispectral_Earth_Observation_WACV_2026_paper.html>).
Frozen protocol SHA-256:
`6b5adaabf785dcfc6226c984429c26cdc942feb5fb1678e5496f779671af6658`.

### 2026-08-16: independent audit supersedes v1 before outcome; v2 frozen

At the user's request, the already configured local Hermes agent performed a
read-only independent protocol/literature audit while v1 was running. It did
not edit the repository, run training, or open a score cache. Two findings
survived local code verification. First, v1's seed-two standalone check tested
pooled/fold AP and pooled recall but omitted sensor AP, per-fold recall, and a
seed-two paired-site lower bound required by the written contract. Second,
invalid pixels were zero-filled in the encoder input but still contributed to
patch normalization and reconstruction/temporal loss. The future four-member
external aggregation rule was also undefined.

V1 was terminated before `evaluate_candidate` ran: no pooled AP, recall,
bootstrap, strength selection, fold-4 endpoint, or tracked outcome exists. Its
completed fold-3 encoder/scene checkpoints remain ignored and are forbidden as
v2 initialization. This is a supersession, not an architecture result. Compact
record SHA-256:
`18b8070d638dce61bdd51e2295ceac3d79cd9d93b27f2d432ac165c32ccc6b22`.

V2 uses new seeds and an isolated checkpoint directory. Valid pixels alone now
define each patch/group mean, variance, reconstruction numerator/denominator,
and temporal-change numerator/denominator; completely invalid masked patches
are skipped. Seed two must independently pass every pilot check, including
both sensors, both-fold recall, and paired-site uncertainty. Future external
scoring is fixed as: average the two endpoint corrections within each seed,
then average across seeds, then add the selected strength times that
four-member mean to the champion logit.

Thirteen focused tests pass, including an invalid-pixel invariance test and an
exact seed-two gate test. The corrected real-data smoke produced finite losses
and gradients without outcome metrics. V2 protocol SHA-256 is
`86c350b2b61732c643dc4207b74cd50baf5cb5f4037ec0d462f6cb5b3d5fa934`.

Paper-language constraint from the audit: this is an integrated,
ExPLoRA-inspired and MAESTRO-inspired candidate, not a faithful reproduction
or causal demonstration of either method. ExPLoRA's published parameter
placement and MAESTRO's structured masking differ from this implementation.
Any positive result is attributed to the full preregistered system unless a
same-head untouched-Prithvi ablation is separately frozen and run.

### 2026-08-16: domain-adaptive Prithvi v2 rejected at seed-one gate

The corrected v2 production run completed both strictly cross-fitted fold
endpoints and opened only folds 3/4. The preregistered 0.25 residual was the
least harmful candidate: AP was 0.904501, a -0.002024 change from the frozen
Gaussian+DOFA champion, matched-FPR recall changed -0.000656, and the paired
250-site/10,000-replicate AP interval was [-0.004297, +0.000679]. Fold 3
regressed -0.001481 AP while fold 4 gained only +0.000655. Pooled Sentinel-2
AP regressed -0.003565; Landsat changed -0.000097.

Increasing the correction did not reveal an underweighted signal. Frozen
strengths 0.50 and 1.00 produced AP deltas -0.004743 and -0.012790; the 1.00
paired-site interval was entirely negative [-0.024466, -0.001631]. The harm is
therefore monotonic over the registered sweep and concentrated in the shifted
Sentinel-2 domain. No post-hoc smaller weight will be tried.

Both masked objectives optimized normally—the endpoint total losses fell from
3.999/4.139 at step 100 to 0.820/0.845 at step 1,200—so stable reconstruction
is not sufficient for complementary methane ranking. The result falsifies the
complete spectral-temporal domain-adaptive Prithvi residual path under this
protocol. Seed two, external cohorts, folds 0/1/2, and official evaluation
were never authorized. The next architecture should add genuinely new
methane-specific supervision or measurement evidence rather than another
Prithvi-derived scalar correction. Compact result:
`reports/experiments/mars_prithvi_domain_adaptive_v2.json`.

### 2026-08-16: privileged hyperspectral supervision evidence narrowed

A bounded Hermes source sweep was independently checked against four primary
sources. Remote-sensing modality distillation supports training-only privileged
modalities, but its published benchmark is classification rather than methane.
PlumeBed is the strongest methane-specific transfer precedent: it combines
Carbon Mapper hyperspectral plumes with Sentinel-2 backgrounds and
domain-adversarial training, but does not validate temporally paired hidden
feature distillation. The large synthetic Sentinel-2 ViT/U-Net study describes
its airborne comparison as indirect because of acquisition timing, and NASA's
EMIT documentation distinguishes scene-wide enhancement/uncertainty/sensitivity
from plume complexes confirmed by three scientists.

Architecture consequence: if the frozen cross-modal acquisition gate later
passes, use PSF-matched, uncertainty-weighted enhancement/mask targets as the
primary privileged signal. Hidden-feature distillation is a low-weight
ablation. Sentinel-2 and Landsat require separate students or stems and
independent sensor gates because no cited methane source validates Landsat
transfer. Temporal intermittency is the principal validity threat. Full
source audit and conditional design boundary:
`reports/research/HYPERSPECTRAL_PRIVILEGED_SUPERVISION_NOTE.md`.

A second read-only Hermes scout was restricted to the missing negative-data
gate. After independent source and compact-receipt checks, no candidate already
clears the frozen requirement. CACH4 remains below the component gate,
STARCOP remains retired, Carbon Mapper null detections are source-specific, and
COVID/Permian remain conditional on the authenticated NASA/ORNL grid bridge.
Hermes created temporary root-level source downloads despite the read-only
instruction; they were identified from the before/after Git status and removed
without entering version control. Hermes remains a discovery scout, while
source verification, protocol decisions, and repository writes stay local.

### 2026-08-29: authenticated NASA/ORNL bridge passes geometry but fails geography

The frozen header-only bridge was completed through the authenticated Edge
session without exporting credentials. All 404 exact ORNL ENVI header assets
(124 CACH4 anchors plus 280 COVID/Permian train-background flightlines) were
checked against CMR-declared byte counts and SHA-256 values. The 5,880,697
downloaded bytes remain under ignored research storage; no AVIRIS image,
target-satellite catalog, target asset, JPL test/EMIT content, or protected
MARS outcome was accessed. Ten focused bridge tests pass.

Stage A passed completely: 124/124 anchors resolved, with zero geometric grid
mismatches. Stage B then resolved all 13,444 released COVID/Permian background
rows across all 280 flightlines, with no missing header and no invalid crop
center. After the frozen MARS-test and prior-pair exclusions, 3,672 rows in 133
flightlines remained, but they formed only 13 connected 25 km components. The
unchanged gate required 20. The bridge therefore **FAILS**, and its target
catalog remains forbidden.

This retires COVID/Permian as the missing-negative supplement under the frozen
contract. Its 13 components may not be pooled with the failed CACH4, STARCOP,
or Carbon Mapper candidates, and the 300-negative privileged-supervision gate
is still unmet. Authoritative compact artifacts are
`reports/acquisition/jpl_operational_ghg_ornl_header_bridge.json` and
`reports/acquisition/JPL_OPERATIONAL_GHG_ORNL_HEADER_BRIDGE.md`; the compact
JSON SHA-256 is
`32ecb2c905cc2ac81433174465de70feaa6a186e23eb9b37aa543eb4ee2725e2`.

### 2026-08-29: GHGSat landfill-null metadata audit frozen

After the ORNL rejection, a bounded Hermes source audit proposed one genuinely
independent candidate: the released GHGSat global-landfill survey associated
with Dogniaux et al. Independent checks against the Zenodo record and Nature
paper confirm 1,447 clear-sky site observations across 151 landfills, including
1,013 positive observations and 434 reviewed non-detections. The null label is
strictly a site-level failure to detect a localized plume under that GHGSat
observation and sensitivity; it is not physical zero methane or scene-wide
absence.

The apparent rights discrepancy is resolved on the record itself. GHGSat
retains copyright, while the dataset carries an explicit CC BY-NC-SA 4.0
license. Any research use must therefore remain non-commercial, attributed,
and share-alike. A peer-reviewed GHGSat platform audit also corrected the
initial timing estimate: C1-C2 use a 10:30 local descending node, whereas
C3-C5 use about 13:00. Only C1/C2 nulls may contribute to the frozen
morning-pair headroom gate for Sentinel-2/Landsat.

The metadata protocol was committed before downloading the complete CSV. It
requires exact reproduction of the paper population, deterministic site
coordinates from released positive rows for zero-coordinate null records,
at least 56 selected C1/C2 null observations, 30 distinct null sites, and 20
novel 25 km components after the unchanged protected-geography exclusions.
This is twice the eventual 28-pair requirement, not an assumed pairing rate.
No GHGSat raster, target-satellite catalog, target image, or protected MARS
outcome is authorized at this stage. Frozen protocol:
`configs/mars_ghgsat_landfill_null_protocol.json`, SHA-256
`0943cb2a15f2106b9ee4a71f9ab36c06f563e410ad0b530ee1f3514b3aa1bcb1`.

### 2026-08-29: GHGSat landfill-null metadata gate passes with headroom

The exact frozen 299,683-byte CSV was acquired from Zenodo and matched MD5
`0e3ba8fae21d6f888413148a76edc5de` and SHA-256
`b383b457db5790cac9d99c36d01a22599c53b03bf67075dbb87501c31997896a`.
The first request received an 88-byte HTTP 406 response because the client
advertised an unnecessarily narrow content type. The browser-compatible
header correction was committed without changing the URL, hash, byte cap,
population, or gate before the successful retry.

The first verified-cache parse then exposed the release's documented null-row
convention: positive-only emission uncertainty and IME intermediate fields are
empty for non-detections because no emission rate is calculated. The parser
was corrected to require those fields to be exactly empty for nulls and finite
for positives; all thresholds and label rules were unchanged, the correction
was committed, and 50 focused/helper tests pass.

The final cached audit exactly reproduces all published counts: 1,447
clear-sky observations at 151 sites in 2021-2022, 1,013 positive observations,
434 null observations, and 1,085 positive plume rows. Of 434 nulls, 329 were
C1/C2 morning candidates; the deterministic four-per-site cap retained 199.
The protected MARS/prior-negative filters excluded 23, leaving 176 reviewed
null observations at 66 sites in 64 connected 25 km components. Every frozen
gate passes versus requirements of 56 observations, 30 sites, and 20
components.

This is a metadata-source pass only. The label remains a GHGSat site-level
non-detection above applicable sensitivity, not zero methane or a dense
all-negative scene. No GHGSat raster, Sentinel-2/Landsat catalog or asset,
protected outcome, score cache, checkpoint, or official benchmark result was
accessed. The next authorized action is a separately committed exact
target-catalog pairing protocol. Compact result:
`reports/acquisition/ghgsat_landfill_null_metadata.json`.

The next catalog audit is frozen before any target query. It will issue one
metadata-only point/time request per qualified source observation to the
official CDSE Sentinel-2 L1C and USGS Landsat Collection 2 Level-1 STAC
collections. The window remains exactly +/-1 hour. Landsat is restricted to
LC08/LC09 L1TP Tier 1; Sentinel-2 to S2A/S2B L1C. Full item geometry must
contain the site point, assets are excluded, and cloud cover is only a
deterministic tie-breaker because local observability requires raster QA later.
At most one item per source observation and target sensor is retained. A pass
requires at least 28 distinct source observations, 20 sites, 20 novel 25 km
components, and 20 distinct target item IDs; dual-sensor matches cannot inflate
the source-observation count. Protocol SHA-256:
`486e78bdb41b4aa9dfcb6bc6943eb80caf0cf8c7c5899bce1cdb6cc7a06967ff`.

### 2026-08-29: frozen GHGSat target-catalog acquisition-feasibility gate passes

The independently reviewed and corrected auditor completed 352 logical queries
under the frozen contract. It found 109 candidates and selected 79
source-sensor pairs: 47 Sentinel-2 L1C and 32 Landsat C2 L1. The selected set
contains 66 distinct source observations, 44 sites, 44 independent 25 km
components, and 78 target item IDs. All frozen gates pass.

The ignored candidates SHA-256 is
`cabe8b0c7055de8844f203daa87657c694b197b89540120cb411dc3ecbf34cba`,
and the ignored selected-pairs SHA-256 is
`8c7942f0ac7bc07e250603e25ff23bb345e7bafbd86394e4ce0e42d41a33f6a8`.
The compact result is
`reports/acquisition/ghgsat_landfill_target_catalog.json`.

This is strictly an acquisition-feasibility result under the frozen catalog
contract. No assets, references, protected outcomes, or model artifacts were
accessed. The pass does not establish target-raster observability, convert a
GHGSat site-level non-detection into a dense all-negative label, authorize a
model protocol, or provide evidence of benchmark or model improvement.

### 2026-08-29: frozen GHGSat reference-catalog feasibility gate passes

The frozen auditor completed 79 primary logical queries and 9 seasonal logical
queries, 88 total. It made 89 HTTP attempts: 88 returned status 200 and one
handled status 429. The audit retained 270 valid candidates and selected 76
target/reference pairs: 47 Sentinel-2 L1C and 29 Landsat C2 L1. Seventy
selections came from the primary window and 6 from the seasonal fallback. The
selected set contains 64 distinct source observations, 43 sites, 43
independent 25 km components, and 75 distinct reference item IDs. All frozen
gates passed.

The candidate JSONL SHA-256 is
`d160fe663f646ed8a2e0798954517e077c63f7f405a107a4c7dea9edc5cc4c90`,
and the pair JSONL SHA-256 is
`ebd2cd3e4f4a201905f441c86eeabd1a50798e57c3818a3bd46d4fb11709f6a2`.
The compact report is
`reports/acquisition/ghgsat_landfill_reference_catalog.json`.

No asset URL, item detail, raster, protected outcome, or model artifact was
accessed. This result is strictly catalog feasibility; it does not establish
local observability, dense no-plume truth, training value, or benchmark
improvement.

### 2026-08-29: frozen GHGSat windowed-observability gate fails

The exact audit attempted all 76 frozen target/reference pairs. None became an
observable training row: 47/47 Sentinel-2 pairs failed exact asset resolution,
and all 29 resolved Landsat pairs failed local raster processing. Observable
pairs, source observations, sites, and independent 25 km components are all
zero. No crop or dense/zero mask was retained.

Codex reproduced the leading diagnostic in each path. The strict first
Sentinel-2 replay returned an empty feature list for the frozen +/-1-second
Earth Search query (`No exact Earth Search mirror`). The first exact Landsat C2
L1 Tier-1 asset returned HTTP 302 to `ers.cr.usgs.gov` authentication, leaving
GDAL with a non-TIFF response. Neither diagnostic permits a source, mirror,
product, cohort, crop, reference, or threshold amendment after outcome.

The authoritative decision is **FAIL**. GHGSat is retired as the independent
missing-negative source; it cannot be pooled with prior failed sources. The
conditional privileged-supervision branch remains unauthorized, and no model
or protected outcome was run. Compact result:
`reports/acquisition/ghgsat_landfill_windowed_observability.json`.

### 2026-08-30: sensor-aware ordinal held attempt fails in infrastructure

The one authorized unchanged attempt at commit
`c6f987e75f170b0c0ddfb607b746173c67b16642` ran for 14,199 seconds and then
raised `torch.AcceleratorError: CUDA error: unknown error` during fold-3
endpoint fitting after epoch 9. Captured inner-validation progress for epochs
1-9 is preserved in the compact failure receipt. The endpoint did not complete;
its provisional in-memory checkpoint rank selected epoch 6, but no endpoint
state or candidate prediction was written.

Independent review confirmed that no candidate result, comparator access, gate
evaluation, endpoint-state artifact, compact result, protected artifact, or
official artifact exists. WSL kernel diagnostics repeatedly reported DXG
allocation-bridge failures (`ffffffb5`, `Ioctl -75`); Windows System logs
contained no thermal or WHEA event, and GPU temperatures were normal. The
attempt is therefore recorded as **infrastructure failure, not scientific
rejection**. The frozen trainer, model, protocol, thresholds, comparators, and
all seven gates remain unchanged. No retry was run and bulk output remains
absent. Comparator files were streamed only as opaque bytes during protocol
SHA-256 preflight; no comparator values were decoded and no semantic comparator
outcome was accessed. Compact receipt:
`reports/experiments/mars_sensor_ordinal_held_attempt_failure.json`.

### 2026-08-30: native-Windows exact-recovery amendment frozen

The next scientific attempt is constrained to native Windows CUDA on the exact
registered RTX 5070/driver/dependency runtime. This bypasses the failed WSL DXG
bridge and WSL `p9` I/O path. Because the WSL failure persisted no recovery
checkpoint, the retry must begin at epoch 1 and must not import the provisional
epoch-6 state or tune from epochs 1-9.

The infrastructure-only amendment adds exact complete-epoch recovery, strict
identity/runtime/access-ledger validation, immutable endpoint state before held
access, atomic read-only per-held-fold parts with verified reuse, and semantic
comparator decoding only after both parts are immutable and merged into the
final immutable candidate. Data, folds, seed, architecture, precision, batches,
schedules, losses, checkpoint rank, thresholds, comparators, bootstrap, and all
seven gates are unchanged. The Windows environment, native CUDA recovery smoke,
and held experiment were not run. Freeze note:
`reports/research/MARS_SENSOR_ORDINAL_NATIVE_WINDOWS_RECOVERY.md`.
