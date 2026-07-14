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

## Predeclared next experiments

1. Reproduce the released model on complete held-out folds 0 and 1 under the exact connected-component evaluator.
2. Train correction-only fold 0 with label-sensor balanced sampling, hard-negative segmentation loss, scene MIL loss, and asymmetric no-plume non-regression.
3. Advance only if fold 0 improves AP, recall at no more than 7.13% FPR, and fixed-rule pixel IoU; independently confirm on fold 1.
4. Ablate sensor identity, temporal normalized/log-ratio features, no-plume penalty, scene loss, and correction capacity.
5. If confirmed, run all five fold models, generate out-of-fold predictions, cross-fit calibration, and freeze an ensemble.
6. Open the paper test once. Require paired site-block bootstrap lower 95% bounds above zero for AP and IoU deltas, higher recall, and no worse FPR on both official views.

## Current claim boundary

The paper comparator is reproduced and the research protocol is publication-grade, but ERSRR has not yet demonstrated paper-test superiority. No wording should imply otherwise until every frozen superiority gate passes.
