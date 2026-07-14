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

## Predeclared next experiments

1. Reproduce the released model on complete held-out folds 0 and 1 under the exact connected-component evaluator.
2. Train correction-only fold 0 with label-sensor balanced sampling, hard-negative segmentation loss, scene MIL loss, and asymmetric no-plume non-regression.
3. Advance only if fold 0 improves AP, recall at no more than 7.13% FPR, and fixed-rule pixel IoU; independently confirm on fold 1.
4. Ablate sensor identity, temporal normalized/log-ratio features, no-plume penalty, scene loss, and correction capacity.
5. If confirmed, run all five fold models, generate out-of-fold predictions, cross-fit calibration, and freeze an ensemble.
6. Open the paper test once. Require paired site-block bootstrap lower 95% bounds above zero for AP and IoU deltas, higher recall, and no worse FPR on both official views.

## Current claim boundary

The paper comparator is reproduced and the research protocol is publication-grade, but ERSRR has not yet demonstrated paper-test superiority. No wording should imply otherwise until every frozen superiority gate passes.
