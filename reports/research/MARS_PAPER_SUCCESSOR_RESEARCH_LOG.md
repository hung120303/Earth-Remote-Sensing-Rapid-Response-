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
claim of an exact band contract. MARS's 4 km Sentinel-2 crop is resized to
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
