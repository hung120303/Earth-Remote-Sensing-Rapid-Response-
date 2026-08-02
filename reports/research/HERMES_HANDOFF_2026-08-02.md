# ERSRR research handoff to Hermes

Prepared 2026-08-02 for continuation from commit `f5165411` on branch
`local/research-audit-and-existing-work`.

## Mission

Develop and validate a publication-grade ERSRR successor that unambiguously
outperforms the released MARS-S2L paper model under the authors' exact official
benchmark contract. The target is not a favorable point estimate: require
paired 25 km site-block evidence for higher scene average precision and dense
IoU, plus higher recall at no worse FPR, on both the 43,529-image full official
test and the 15,655-image test-only-site view. Preserve research provenance,
keep bulk data/checkpoints ignored, commit reproducible code and compact
artifacts, and update the paper-ready documentation and HTML report only when
the evidence is honest.

The primary paper is Rouet-Leduc and Hulbert, *Automatic detection of methane
emissions in multispectral satellite imagery using a vision transformer*,
Nature Communications 15, 3801 (2024):
https://www.nature.com/articles/s41467-024-47754-y. The benchmark used for this
project is the authors' revised MARS-S2L v3 contract:
https://arxiv.org/html/2511.21777v3. Official code is pinned at
`UNEP-IMEO-MARS/marss2l@f7d264c2c845dfba1cb27f76ef6026275f8d8758`.

## Start here

1. Work in `C:\Users\joshu\PROJECTS\REMOTE-SENSING\Earth-Remote-Sensing-Rapid-Response-`.
2. Read `C:\Users\joshu\.agents\skills\ersrr\SKILL.md` and its
   `references/gotchas.md` before editing.
3. Read these tracked records before making an architecture call:
   - `reports/research/MARS_PAPER_SUCCESSOR_RESEARCH_LOG.md`
   - `docs/MARS_PAPER_SUCCESSOR_OUTLINE.md`
   - `docs/MARS_V6_UNIFIED_DESIGN.md`
   - `docs/RESEARCH_LEDGER.md`
   - `reports/experiments/MARS_SPATIAL_PRITHVI_ENSEMBLE_PAPER_POSTTEST.md`
   - `reports/experiments/mars_dofa_gaussian_protected_ensemble.json`
   - `reports/acquisition/STANFORD_LARGE_CONTROLLED_RELEASE_COHORT.md`
4. Verify `git status --short` is clean and inspect the latest commits. Do not
   delete or commit ignored `.research/` data or model checkpoints.
5. GPU work runs in WSL using the repo `.venv`; the machine has an RTX 5070
   with 12 GB VRAM. Typical invocation:
   `wsl bash -lc "cd '/mnt/c/Users/joshu/PROJECTS/REMOTE-SENSING/Earth-Remote-Sensing-Rapid-Response-' && source .venv/bin/activate && ..."`.

## Exact benchmark state

Published MARS-S2L v3 context is AP 0.6408, recall 0.7915, FPR 0.0713, and
IoU 0.3224 on the full official test. The locally reconstructed checkpoint,
which is the exact paired comparator, measures:

| View | AP | Recall at matched FPR | FPR | Pixel IoU |
|---|---:|---:|---:|---:|
| Full official | 0.641020 | 0.791506 | 0.070692 | 0.324365 |
| Test-only sites | 0.450274 | 0.775330 | 0.075512 | 0.171562 |

The strongest official post-test architecture evaluated so far is the
spatial-Prithvi ensemble:

| View | Candidate AP | AP delta (95% paired-site CI) | Recall delta (95% CI) | Candidate IoU | IoU delta (95% CI) | Gate |
|---|---:|---:|---:|---:|---:|---|
| Full official | 0.676102 | +0.035082 ([+0.016224,+0.049491]) | +0.031440 ([+0.019979,+0.048110]) | 0.379964 | +0.055598 ([+0.034660,+0.077298]) | pass |
| Test-only sites | 0.467027 | +0.016753 ([-0.023301,+0.052032]) | +0.022026 ([-0.008066,+0.051119]) | 0.292462 | +0.120900 ([+0.085772,+0.155426]) | fail |

Therefore the project does **not** yet outperform MARS-S2L without doubt. The
full view passes, but the test-only-site AP and recall intervals still cross
zero.

The strongest development-only candidate is the Gaussian scene-aligned ViT at
strength 0.10 plus the fixed DOFA protected fusion, evaluated on folds 3/4
(17,745 scenes; 250 25 km groups): AP 0.906525, +0.002449 over the prior
spatial-Prithvi development champion, paired-site AP CI
[+0.000489,+0.004068]. Both folds and both sensors improve; matched-FPR counts
are unchanged (TP 1466, FP 1156, TN 15065, FN 58). Dense evidence also passed:
development IoU +0.002631 with lower bound +0.000369 and fold-2 IoU +0.004367
with lower bound +0.002352. This candidate has **not** been replayed on the
official test. Its ignored cache is
`.research`-backed through
`outputs/mars_dofa_gaussian_champion_folds34_scores.npz`, SHA-256
`988b98c92a1a5fa1fe52d7052b9159352f0fadd876b400fce1c8c879c94ea424`.

## Model and synthetic-data answer

Yes, this research uses vision transformers. The released comparator is a ViT;
the current line uses Prithvi/DOFA remote-sensing transformer representations
and a small scene-aligned Gaussian-contrast ViT residual. Gaussian plume
sampling was used only where evidence supported it: dense/auxiliary plume
learning and a bounded protected residual. It must not be presented as real
external evidence, and synthetic data must not supervise a scene head as if it
were a real no-plume/plume cohort.

## Architecture conclusions already established

Architecture/head search on folds 3/4 is near its information ceiling. Do not
repeat these rejected paths without a materially new hypothesis:

- Product-aware dual-Prithvi v6 scene pilot: rejected, AP +0.000182.
- Unconstrained v6 error-correction residual: rejected; AP deltas became more
  negative as strength increased.
- Official Prithvi-EO-2.0 100M CLS probe: rejected, AP -0.000288 and one TP
  lost; paired interval crossed zero.
- DOFA-v2, DINOv3 fusion, Prithvi LoRA, physics-guided adapters, Gaussian
  scene-head cross-fit, anchored U-Net, robust site templates, and bi-sensor
  protected fusion all failed scene-ranking promotion gates or gave only
  dense improvements.
- Group conformal risk control did not transfer directionally. At alpha 0.075,
  fold 3 -> 4 produced crop FPR 0.1652 and group FPR 0.0831. Do not claim a
  geographic FPR guarantee from conformal calibration under violated
  exchangeability.

The sound part of the earlier unified-v6 recommendation is the diagnosis:
new, independently labeled real observations and robust calibration are now
more valuable than another marginal head. The proposed specific v6
architecture and conformal guarantee were experimentally contradicted.

## Holdout and claim discipline

- The official test has already been opened once by predecessor/post-test
  architecture analyses. Any future official result is transparently
  post-test; independence cannot be recovered by relabeling it.
- Fold 2 has already been used as a one-shot dense confirmation for several
  candidates. Folds 0/1 were used earlier. Do not reopen them casually.
- Current fresh selection is limited to folds 3/4.
- Never tune on Stanford controlled-release outcomes. Freeze checkpoint,
  threshold, crop geometry, and temporal-reference rule before scoring.
- MARS `background_image_tile` is **not** a verified negative. Upstream code
  permits backgrounds containing plumes with dissimilar wind direction.
- CloudSEN12 “clear” means cloud-free, not methane-free. Do not use it as a
  no-plume truth source without independent methane labels.
- Preserve the exact mixed Sentinel-2/Landsat 16-channel paper input contract;
  do not substitute L2A for L1C or Landsat L2 for Collection 2 Level 1.

## Newly acquired controlled-release cohort

Commit `f5165411` adds a reproducible metadata audit for the April 2026
Stanford preprint *Unlocking credible space-based methane sensing through a
year-long single-blind test* and its public CC-BY-4.0 repository:

- Paper: https://doi.org/10.21203/rs.3.rs-9110475/v1
- Data: https://doi.org/10.25740/qh001qt3946
- Builder: `tools/build_stanford_large_controlled_release_cohort.py`
- Protocol: `configs/stanford_large_controlled_release_protocol.json`
- Audit: `reports/acquisition/stanford_large_controlled_release_cohort.json`

The authoritative compact workbook is 517,597 bytes with SHA-256
`8539cdf39dae5fe12be3e5a5b98c556701d29f6719b52d5551c6d0d47a546fd8`.
It remains ignored at
`.research/stanford_controlled_release_2024_2025/source_data_Reuland_2026_07162026.xlsx`.

After paper QC and acquisition filters there are 262 unique 2025 Casa Grande
events: 174 Sentinel-2 and 88 Landsat; 136 exact blanks, 13 primary positives
at or above 1,000 kg CH4/h, and 113 nonzero sub-threshold challenges. STAC
resolves 257 exact L1 products: all 88 Landsat, 169 Sentinel-2, with five S2
unresolved. Among resolved products are 133 blanks, all 13 primary positives,
and 111 challenges. No ERSRR model score has been accessed.

This cohort is temporally new but not geographically/source-disjoint: Casa
Grande occurs in 677 upstream MARS rows, all excluded as `Not Used`. With one
site it cannot provide site-bootstrap geographic superiority, but its 133
resolved blank controls are the strongest available false-positive stress test.

Recommended next move: freeze an external inference protocol, then acquire
only the 169 public S2 L1C target crops and required temporal-reference crops
using STAC/COG window reads. Do not bulk-download full products. Landsat L1
downloads are blocked by a separate USGS EROS login despite valid Earthdata
authentication; the user must finish EROS authentication and say `USGS ready`,
or provide an AWS requester-pays profile. Do not substitute Landsat L2.

## Research assets and commits

Important recent commits:

- `f5165411` audit large controlled-release truth cohort.
- `3635b422` freeze large controlled-release protocol.
- `c271b787` audit 2022 controlled-release provenance.
- `49d8cb78` record rejected group CRC transfer.
- `cbffb758` record rejected Prithvi-100M probe.
- `5f9aef15` record rejected v6 error-correction pilot.
- `a369bc17` record rejected unified v6 scene pilot.
- `e0dfe6ad` pass protected Gaussian+DOFA development gates.

Bulk datasets, caches, and checkpoints are intentionally ignored. Never add
them to Git merely to simplify a handoff.

## Proposed resume sequence

1. Audit the handoff and reproduce the 262-row Stanford metadata result; do
   not score models yet.
2. Freeze an external S2-only acquisition/inference protocol. Specify exact
   target/ref selection, 256x256 crop geometry, band order/scaling, cloud/QC
   handling, candidate/checkpoint hashes, and one fixed threshold before
   reading outcomes.
3. Acquire only windowed S2 L1C crops for 169 resolved events into ignored
   storage and audit every crop. Keep all 133 resolved blanks in the primary
   false-positive analysis; use the 13 ≥1 t/h events for primary recall; report
   111 lower-rate events separately.
4. Score the released MARS-S2L comparator and the frozen ERSRR candidate once.
   Report every event, not just aggregates. Treat results as one-site temporal
   stress evidence, not geographic proof.
5. In parallel only if necessary, solve USGS EROS/AWS access for exact Landsat
   L1 windows and extend the already-frozen protocol without changing labels.
6. Seek genuinely independent multi-site real cohorts. New groups, especially
   verified no-release observations, are the most plausible route to tighter
   test-only-site evidence. Do not call catalog absence a negative.
7. Only after new real evidence, decide whether a new training architecture is
   justified. Prefer explicit error analysis and preregistered ablations over a
   broad model sweep.
8. Update the paper outline, research log, and `reports/ERSRR_RESEARCH_REPORT.html`
   with exact numbers, uncertainty, limitations, and architecture only after
   the relevant results are frozen and committed.

## Ready-to-paste Hermes goal

```text
/goal Continue the publication-grade ERSRR methane-plume research in C:\Users\joshu\PROJECTS\REMOTE-SENSING\Earth-Remote-Sensing-Rapid-Response- from the current HEAD of branch local/research-audit-and-existing-work; the frozen large-cohort audit is commit f5165411. First read reports/research/HERMES_HANDOFF_2026-08-02.md, the ERSRR skill and gotchas, reports/research/MARS_PAPER_SUCCESSOR_RESEARCH_LOG.md, docs/MARS_PAPER_SUCCESSOR_OUTLINE.md, and the frozen protocols. Preserve all holdout and provenance rules. The objective remains to outperform the exact released MARS-S2L v3 comparator without doubt on both official full and test-only-site views, using paired 25 km site-block evidence for higher AP and IoU plus higher recall at no worse FPR. Be transparent that the official test is already post-test and that the current spatial-Prithvi candidate passes the full view but not test-only-site AP/recall uncertainty. Do not repeat rejected v6, Prithvi-100M, or conformal-risk paths without a materially new hypothesis. The immediate priority is new real evidence: reproduce the frozen Stanford 2025 controlled-release audit (262 QC-valid Casa Grande S2/Landsat events; 136 blanks, 13 >=1000 kg/h primary positives, 113 lower-rate challenges; 257 exact L1 products resolved; no model scores accessed), freeze checkpoint/threshold/crop/reference rules, acquire only windowed Sentinel-2 L1C crops into ignored storage, and conduct a one-shot released-MARS-versus-frozen-ERSRR false-positive/recall stress test. Keep Landsat L1 pending exact USGS EROS authentication; never substitute L2. Treat this one-site cohort as temporal operating-point evidence, not geographic proof. Continue seeking independent multi-site verified no-plume data, run preregistered experiments only when they answer a specific failure mode, keep bulk data/checkpoints ignored, commit reproducible code and compact reports, maintain paper-ready research documentation, and update the final HTML report with exact benchmark comparisons, confidence intervals, architecture, data provenance, negative results, and limitations. Do not declare success until every superiority gate genuinely passes; otherwise document the remaining uncertainty precisely.
```
