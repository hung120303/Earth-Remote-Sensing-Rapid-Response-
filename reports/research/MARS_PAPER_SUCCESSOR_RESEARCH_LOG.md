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

## Experiment ledger

| Date (UTC) | Experiment | Evidence | Decision |
|---|---|---|---|
| 2026-07-13 | ERSRR v4.3 three-checkpoint ensemble on the previously opened strict cohort | AP 0.3903 vs released 0.3521, but recall 0.5224 vs 0.6418 and IoU 0.0766 vs 0.1329 | Rejected; it did not outperform on recall or segmentation. |
| 2026-07-14 | Paper residual initialization audit | Maximum absolute logit delta 0 for both sensor identities | Passed; establishes a non-regressing starting architecture. |
| 2026-07-14 | Mixed-sensor residual smoke | 64 fit / 56 held-out scenes; AP delta approximately 0, recall delta 0, IoU delta -0.0011 after one tiny epoch | Pipeline-only success; no promotion or scientific claim. |

## Predeclared next experiments

1. Reproduce the released model on complete held-out folds 0 and 1 under the exact connected-component evaluator.
2. Train correction-only fold 0 with label-sensor balanced sampling, hard-negative segmentation loss, scene MIL loss, and asymmetric no-plume non-regression.
3. Advance only if fold 0 improves AP, recall at no more than 7.13% FPR, and fixed-rule pixel IoU; independently confirm on fold 1.
4. Ablate sensor identity, temporal normalized/log-ratio features, no-plume penalty, scene loss, and correction capacity.
5. If confirmed, run all five fold models, generate out-of-fold predictions, cross-fit calibration, and freeze an ensemble.
6. Open the paper test once. Require paired site-block bootstrap lower 95% bounds above zero for AP and IoU deltas, higher recall, and no worse FPR on both official views.

## Current claim boundary

The paper comparator is reproduced and the research protocol is publication-grade, but ERSRR has not yet demonstrated paper-test superiority. No wording should imply otherwise until every frozen superiority gate passes.
