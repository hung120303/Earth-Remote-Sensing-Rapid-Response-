# ADR: segmentation-first research architecture

- Status: accepted for research, not approved for operational deployment
- Date: 2026-07-09
- Evidence revision: `6bf2059213c77c521416b0e460fb0f49de8d4fd7`

## Decision

Use a compact residual U-Net for **binary plume segmentation**, with a shared training/serving contract in `EarthRemoteSensingRapidResponse/ersrr_core.py`:

- canonical Sentinel-2 bands: `B2, B3, B4, B11, B12`;
- five `log1p`-transformed raw bands—physics-derived ratios are not part of the selected artifact;
- 144,433 trainable parameters with `base_filters=8`;
- normalization statistics fitted on training groups only;
- an explicit target-validity channel in the loss and metrics;
- five group-disjoint outer folds, with disjoint inner groups for early stopping and decision-threshold calibration;
- a versioned artifact config containing product level, feature schema, normalization, threshold, model hash, dataset cohort, and limitations;
- one model implementation shared by experiments, artifact packaging, and Flask serving.

The packaged artifact is labeled `research_only`. Physical methane regression is disabled until a target-independent calibration and trustworthy concentration-raster contract exist. Revision `6bf20592` consolidated the tracked web assets into canonical `ERSRR_Website/` and removed the lowercase path.

## Evidence

The audit found 102 legacy tiles but only 41.64% valid target pixels, no true negatives, a mean Sentinel-2/EMIT offset of 5.95 days (worst 53.08), and geographic group overlap in the original random split. Filtering to at least 10% valid target and at most seven days absolute gap leaves 65 scenes. Leakage-safe connected components—linking scenes that share an MGRS tile, Sentinel-2 acquisition, or EMIT granule—reduce this to 32 split groups; this grouping prevents known scene crossover but does not prove geographic independence.

All clean legacy comparisons use the same 128 × 128 (~20 m) grid, five outer group-held-out folds, disjoint inner groups for threshold calibration, and 2,048 sampled valid pixels per scene. The U-Net runs use `base_filters=8` and `positive_weight=1`.

At the moderate `>300 ppm·m` label:

| Model | Features | AUPRC | AUROC | Scene F1 | Scene IoU |
|---|---:|---:|---:|---:|---:|
| Prior dummy | 5 | 0.3548 | 0.5000 | 0.4988 | **0.3548** |
| Raw logistic | 5 | **0.4481** | **0.5995** | 0.4781 | 0.3398 |
| Physics logistic | 11 | 0.4350 | 0.5805 | 0.4606 | 0.3268 |
| Raw ResUNet | 5 | 0.4168 | 0.5659 | 0.4988 | 0.3546 |
| Physics ResUNet | 11 | 0.4079 | 0.5556 | 0.4939 | 0.3502 |

At the high `>1000 ppm·m` label:

| Model | Features | AUPRC | AUROC | Scene F1 | Scene IoU |
|---|---:|---:|---:|---:|---:|
| Prior dummy | 5 | 0.0601 | 0.5000 | **0.2234** | **0.2133** |
| Raw logistic | 5 | **0.1243** | **0.6384** | 0.1614 | 0.1266 |
| Raw ResUNet | 5 | 0.0953 | 0.5793 | 0.1203 | 0.1016 |
| Physics ResUNet | 11 | 0.0767 | 0.5504 | 0.0868 | 0.0622 |

The raw ResUNet is selected because it is the stronger neural architecture at both thresholds, avoids an unsupported feature expansion, and preserves a simple train/serve contract. It does not beat raw logistic ranking, and its scene F1 is approximately the calibrated prior at the moderate threshold. These results justify a compact research baseline, not an accuracy or deployment claim.

## Packaged artifact diagnostic

The ignored model artifact is documented by the tracked snapshot `reports/artifacts/compact_resunet_v1_config.json`. It uses 50 fit scenes in 26 leakage-safe groups and 15 calibration scenes in six groups, stopped after five of 12 requested epochs, and selected a decision threshold of `0.01`. On 2,048 sampled calibration pixels per scene it reports AUPRC 0.4754, AUROC 0.6162, F1 0.5275, and IoU 0.3582.

Those calibration metrics are not independent test results. More importantly, recall is 0.9984 while specificity is only 0.00127: the calibrated operating point predicts almost everywhere positive. The artifact proves that the versioned raw-band train/serve path can be packaged and verified; it is not a deployable detector.

## V002 feasibility boundary

The V002 acquisition produced 12 EMIT granules, 24 Sentinel-2 L2A stacks, and 11.31 MB of ignored local data with zero declared-file, SHA-256, or grid mismatches. Six plume polygons are clipped by their fixed tiles; four of those also cover more than 50% of a tile. Requiring complete, non-clipped polygons leaves only six evaluation groups.

| Model | Features | AUPRC | AUROC | Scene F1 | Scene IoU |
|---|---:|---:|---:|---:|---:|
| Prior dummy | 5 | 0.1708 | 0.5000 | 0.2819 | 0.1708 |
| Raw single-time logistic | 5 | 0.2410 | 0.6367 | 0.2294 | 0.1406 |
| Physics single-time logistic | 11 | **0.2415** | **0.6397** | 0.2292 | 0.1404 |
| Physics single-time HistGB | 11 | 0.1941 | 0.5861 | **0.2939** | **0.1777** |
| Bitemporal logistic | 33 | 0.2271 | 0.6212 | 0.1917 | 0.1152 |
| Bitemporal HistGB | 33 | 0.2070 | 0.5995 | 0.2673 | 0.1635 |

Raw and physics single-time logistic are effectively tied; bitemporal features are not proven. Both non-contemporaneous Sentinel-2 observations receive the same EMIT-time polygon, so a bitemporal model may learn persistent land-cover or source context rather than plume change. With six groups, nested folds leave only three or four fit groups. The pilot therefore establishes a reproducible data path and exposes benchmark defects; it does not estimate generalization.

## Consequences

- The web API can no longer silently change band order, normalization, model topology, or threshold relative to training.
- L1C/TOA legacy data and L2A/surface-reflectance pilot data remain separate artifact domains.
- Physics-derived features remain an experimental ablation, not the selected model input.
- Bitemporal L2A remains a research question until contemporaneous labels and enough complete independent groups exist.
- Accuracy work should focus on data validity, temporal alignment, complete polygons, true negatives, and physical labels before increasing model size.
- Historical checkpoints and target-derived prediction denormalization are not part of the supported path.
- Repository-hygiene commit `4e9d8787` removed generated dependencies, predictions, checkpoints, and raw acquisition outputs from the index while preserving local files; revision `6bf20592` then consolidated tracked frontend assets into `ERSRR_Website/` and removed the lowercase path.

## Reproducibility record

- Clean evidence commit: `6bf2059213c77c521416b0e460fb0f49de8d4fd7`
- V002 batch identity: `81410698329cdab6b0dabe5f39e1252617aaa1e4c539d51d382e30016d49f0fd`
- V002 summary script: `7edc5ac1ed0018719abce25473030e9afc40ec352a28f634a689f70c381ef11c`
- Legacy baseline input: `9b3b12e97d74c9ea65ad3b95eb3dafc2b1ec770bc7fdc9f0c8a216dcc0048c83`
- V002 baseline input: `f98fb4b70978bba4a14593c274a7967f3fb76d4c106db2762bead05f3df7ffc9`
- Baseline script: `ed5764d222567887d94241ab4ff55a175a81bc2a2a2d3ba526223f52ace7ae77`
- Neural cohort: `b93fee1bd96de7300b2f92e137e32d83640f626050357f5c4d2447a8761b1964`
- Neural experiment script: `e455cc2f5e6c66f2a992af9fc20398b934044f78736973a7acb6445e2f79e5cb`
- Packaged raw ResUNet model: `d73dd7728626a6a2e584f1ff7757087ed84f26b41fd05596db53c952a84a48e6`
- Tracked artifact snapshot: `reports/artifacts/compact_resunet_v1_config.json`

## Promotion gates

Operational status requires all of the following:

1. at least 50 independent geographic/source groups in a locked real-data test set;
2. representative hard negatives and calibrated precision/recall at a predeclared operating point;
3. L2A-trained artifact evaluated on L2A inputs without domain mixing;
4. authenticated V002 concentration rasters or an equivalent physically meaningful target contract;
5. confidence intervals and repeated grouped trials showing improvement over prevalence and classical baselines;
6. no outer-test or serving target information used in preprocessing, normalization, threshold selection, or output calibration, and no target-derived inference denormalization.
