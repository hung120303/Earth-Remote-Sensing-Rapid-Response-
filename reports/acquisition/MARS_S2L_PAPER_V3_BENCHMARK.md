# MARS-S2L paper-v3 benchmark lock

The paper's general model is used on onshore scenes and its fine-tuned model on offshore scenes. The historical per-scene archives, not the later public metadata labels, define the evaluated cohort.

| View | Scenes | Plume | Sites | AP | Recall | FPR | Pixel IoU |
|---|---:|---:|---:|---:|---:|---:|---:|
| Reconstructed full | 43,529 | 1,813 | 1,289 | 0.6410 | 0.7915 | 0.0707 | 0.3244 |
| Paper Table S6 | 43,529 | 1,813 | 1,289 | 0.6408 | 0.7915 | 0.0713 | 0.3224 |
| Reconstructed test-only | 15,655 | 227 | 697 | 0.4503 | 0.7753 | 0.0755 | 0.1716 |
| Paper Table S5 | 15,655 | 227 | 697 | 0.4496 | 0.7753 | 0.0763 | not reported |

- Assignment SHA-256: `e3e7296c9837123b6d43870f7c5323cea5cc9b667f4c2b00fbc6e4ec8a87d797`
- Paper-identity metadata coverage: 43,527 / 43,529; current released raster coverage: 43,524 / 43,529.
- Unavailable current scene rasters: 5; positive scenes without current pixel truth: 4.
- Paper-era targets disagree with the July 2025 public metadata label on 23 available scenes.
- Exact scene, positive, site, unseen-site, and recall counts reproduce. Small residual AP/FPR/IoU differences are retained explicitly and never rounded into an exact reproduction claim.

## Superiority gate

A successor must beat both the paper table and the reconstructed per-scene comparator. On full and test-only views, paired site-bootstrap 95% confidence intervals must show AP and IoU improvements; recall must improve while FPR is no worse. All historical scenes without current rasters are scored adversarially for the candidate (positive as a miss, negative as a false alarm, and worst-case pixel error); missing pixel truth is handled adversarially for IoU. Test outputs remain sealed until architecture, ensemble, and thresholds are frozen.
