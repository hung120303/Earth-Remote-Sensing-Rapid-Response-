# Unsupervised target-weighted MARS scene head

Every held fold supplies features and sensor identity but no labels to its density-ratio estimator.

- Weight specification: `clip_lower-0.25_clip_upper-4.0_gamma-0.5`
- Target-weighted head blend: 0.300
- AP delta vs current: +0.00441
- Recall delta vs current: +0.00105
- Paired-site AP interval vs current: [+0.00249, +0.00623]
- Fold AP deltas: +0.00205 / +0.00340 / +0.00451 / +0.00884 / +0.00420
- Fold recall deltas: +0.00000 / +0.00134 / +0.00502 / +0.00264 / -0.00131
- Sensor AP deltas: Landsat +0.00074; Sentinel-2 +0.00610

Reject unsupervised target weighting before paper adaptation. Fold 4 loses
matched-FPR recall, violating the frozen nonnegative-recall gate; fold 0 ties.
