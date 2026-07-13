# MethaneS2CM v5.1 confirmation campaign protocol

Frozen before seeds 2202 and 3303 and before any location-test imagery is opened.

- Architecture: 9.36M-parameter tri-temporal v5.1 with fixed 65% bottleneck-context / 35% mask-derived scene logit.
- Seeds: 1101, 2202, 3303; from-scratch initialization with identical optimization and augmentation.
- Checkpoint selection: lexicographic scene AP, AUROC, recall at <=5% FPR, then pixel Dice on the 64 frozen development groups.
- Scene ensemble: per-seed empirical-CDF calibration on development, then equal mean.
- Pixel ensemble: equal mean probability; threshold selected on development from the frozen 0.05-step grid.
- Uncertainty: 2,000 bootstrap resamples of the frozen 25 km groups.
- Test unlock: only after checkpoint hashes, calibrators, thresholds, and comparison code are committed.
- One-shot test comparators: v5.1 ensemble, frozen v4.3 ensemble, and released MARS-S2L.

The 20,789-image location test remains sealed. Its outcomes cannot be used for architecture, threshold, calibration, or model selection.
