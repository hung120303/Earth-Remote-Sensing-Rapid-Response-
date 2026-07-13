# MethaneS2CM v5 development and sealed-test protocol

Status: frozen before any L2A location-test image was extracted or opened.

## Cohort

- Source revision: `ee9a96d4994ca6bc45725c1e92d7a06258131eaf`
- Train metadata: 80,217 crops / 3,460 exact locations
- Sealed test metadata: 20,789 crops / 816 exact locations
- Exact train/test coordinate overlap: 0
- Product: L2A surface reflectance at 20 m; balanced crop benchmark, not operational prevalence

## Internal development boundary

- Fitting: 64,759 crops / 193 frozen 25 km groups
- Development: 15,458 crops / 64 frozen 25 km groups
- Fitting/development 25 km group overlap: 0
- Architecture selection, augmentation, epochs, thresholds, and seed selection may use only this internal development partition.

## Predeclared v5 direction

- Shared-weight tri-temporal encoder for T, T-90, and T-365 using B02/B03/B04/B08/B11/B12.
- Two scale-invariant MBMP channels (T vs T-90 and T vs T-365).
- Segmentation-first output; scene presence must be derived from dense plume evidence rather than a free classifier.
- Primary selection: scene AP, AUROC, recall at no more than 5% FPR, and positive-pixel Dice, all on frozen internal development groups.
- Final evidence: a fixed multi-seed ensemble evaluated once on the sealed location test together with zero-shot v4.3 and released MARS-S2L.

## Test seal

Do not extract or open any l2a_location_split_32x32 test imagery until the v5 architecture, selected checkpoints, ensemble, pixel threshold, scene calibration, and comparison code are checksum-bound in a clean tracked commit. Then run one test campaign; never retune from its outcomes.

The test is balanced by crop construction and lacks acquisition time, wind, and per-pixel cloud masks. Precision is therefore not an operational positive predictive value, and the comparison is a cross-product (L2A versus MARS L1C) robustness benchmark rather than a substitute for a prevalence-representative deployment trial.
