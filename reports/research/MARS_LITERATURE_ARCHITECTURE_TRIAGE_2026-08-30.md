# Literature-to-repository architecture triage

Date: 2026-08-30  
Scope: contingency planning for the frozen sensor-aware ordinal experiment  
Decision authority: Codex architecture review; no held outcome was inspected for this triage

## Decision

The current sensor-aware ordinal experiment remains the only active architecture
path. It introduces producer-enhancement supervision while keeping scene and dense
outputs decoupled, which is a materially different information source from the
previous score-fusion, NDMI, instance/objectness, Gaussian, foundation-model, and
threshold-calibration families.

If the frozen ordinal experiment fails any preregistered development gate, do not
start another same-cohort architecture search. The evidence-based successor is to
obtain genuinely new geographically disjoint labeled data, or obtain author-released
training data/code that makes a faithful external replication possible. Repackaging
the already-rejected NDMI, patch-MIL, instance/objectness, Gaussian, Prithvi, or
conformal paths is not a defensible publication experiment.

Post-review source discovery identified MethaneUnion as the first concrete candidate
for that successor. The metadata-only audit in
`reports/research/METHANEUNION_METADATA_AUDIT_2026-08-30.md` finds 52 positive and
56 negative released-training Sentinel-2 query coordinates beyond 25 km from both
the pinned MARS and MethaneS2CM coordinates. They remain candidates—not approved
training data—until dense masks, observability, and archive membership are verified.

## Primary-source findings and repository mapping

### MARS-S2L (the exact benchmark)

Primary source: [MARS-S2L paper, arXiv v3](https://arxiv.org/html/2511.21777v3).

The paper's model uses current and reference Sentinel-2/Landsat imagery, cloud
masks, wind information, MBMP evidence, a compact U-Net, and Gaussian plume
simulation. Its scene decision is derived from connected predicted plume pixels.
The exact paper values used by this repository remain Table S6 full-evaluation AP
0.6408, recall 0.7915, FPR 0.0713, and IoU 0.3224; Table S5 test-only AP 0.4496,
recall 0.7753, and FPR 0.0763. The repository also preserves higher-precision
reconstructions and evaluates candidate deltas with paired 25 km site bootstrap.

Implication: comparisons to datasets with random image splits, different sensors,
or no scene-level FPR are useful for architecture hypotheses but cannot establish
superiority to MARS-S2L.

### AttMetNet

Primary source: [AttMetNet paper](https://arxiv.org/html/2512.02751v1).

AttMetNet reports an attention U-Net using twelve Sentinel-2 bands plus NDMI on
6,114 128-by-128 images. Its reported scene recall is 0.86, FPR is 0.09, and mIoU
is 0.66. The inspected paper describes an 80/10/10 image split; it does not establish
the geographically disjoint MARS-S2L protocol or the same benchmark population.

Repository evidence: the NDMI-guided bi-temporal fusion pilot produced only
+0.000756 AP, with paired-site AP interval [-0.000725, +0.003139], while dense IoU
improved +0.002426. The NDMI patch-MIL pilot produced +0.000694 AP with negative
matched-FPR recall and +0.002552 IoU. Both failed promotion. The dense signal is
useful evidence, but the scene-ranking hypothesis has already been tested under a
stricter spatial protocol and was not confirmed.

### N-BPMSNet

Primary source inspected: [N-BPMSNet author-uploaded paper](https://www.researchgate.net/publication/404791662_N-BPMSNet_An_NDMI-Guided_Bitemporal_Network_for_Methane_Plume_Detection_and_Segmentation_From_Sentinel-2_Multispectral_Observations).

N-BPMSNet uses bitemporal Sentinel-2 L1C imagery, an NDMI branch, cross-channel
fusion, and a change-guidance module. The paper describes 11,494 samples from 44
sites and reports temporal IoU 0.6858 and site-disjoint IoU 0.6419, with headline
scene F1 0.8858 and AUC 0.9856. These metrics and population are not an exact
MARS-S2L comparison.

Repository evidence: the two NDMI bitemporal pilots above directly cover the core
scene-ranking hypothesis. Their small positive dense deltas support retaining NDMI
as a possible decoder feature once new data exist, but not reopening the retired
same-cohort scene architecture search.

### Robust Small Methane Plume detection

Primary source: [Robust Small Methane Plume Detection paper](https://arxiv.org/html/2508.16282v1).

This study uses a ResNet-34 U-Net, pseudo-RGB methane representations, and synthetic
no-plume construction at one Australian oil-and-gas facility. It reports validation
F1 0.7839 and IoU 0.6503. Because it is a single-facility validation rather than the
MARS geographic benchmark, the reported values are not direct evidence of MARS-S2L
superiority.

Repository evidence: Gaussian/synthetic and physics-guided representation families
have already been extensively audited. Gaussian training improved dense localization
but was not complementary to the spatial-Prithvi scene ranker. Synthetic supervision
therefore remains dense-only evidence, not a reason to relaunch a synthetic scene
head.

### MethaneSAT instance segmentation

Primary source: [MethaneSAT instance-detection paper](https://arxiv.org/html/2605.24273v1).

On MethaneSAT XCH4 imagery, a Mask R-CNN ResNet-50 increased F1 by 5.48 points over
a U-Net and increased precision by 6.30 points with a recall tradeoff; the reported
ViT variants were weaker. This supports testing object-aware inference in its own
sensor domain, but it does not transfer the numerical claim to Sentinel-2 MARS-S2L.

Repository evidence: the instance-guided teacher pilot improved AP by +0.003147,
but its paired-site interval [-0.003005, +0.006813] crossed zero and dense IoU fell
-0.006802. The follow-on connected-signal ensemble found no complementarity to the
stronger ranker (best blend AP delta approximately -0.000004). The instance/object
family is therefore retired for this same cohort.

## Code and data availability boundary

The inspected primary-source pages did not expose a credible author code or dataset
link for AttMetNet, N-BPMSNet, or the robust-small-plume study, and targeted source
discovery did not locate one. This is a bounded statement about those three sources
as inspected on 2026-08-30, not a claim that no private or later release exists.
MethaneUnion/MethaneFuse is a separate, newly discovered public release and is
audited independently. Any future author release should be captured with a versioned
acquisition receipt before replication.

## Publication claim boundary

The literature values above must remain contextual comparisons. Only the exact
MARS-S2L paper protocol, the reconstructed official population, and preregistered
paired-site uncertainty can support the project's superiority claim. Development
success authorizes a separately frozen protected/official confirmation; it is not
itself a paper-level win.
