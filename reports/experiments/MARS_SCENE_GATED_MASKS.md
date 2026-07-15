# Development-only scene-gated MARS masks

The paper test was not loaded. Released-model masks use the confirmed sensor thresholds (Sentinel-2 0.80; Landsat 0.70), then are suppressed when the cross-fitted v3 scene probability is below a development-selected cutoff.

| Partition | Cutoff | Baseline IoU | Gated IoU | Delta | 95% CI | TP pixels retained | FP pixels removed | Gates |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Selection folds 2/3/4 | 0.750 | 0.57228 | 0.59512 | +0.02284 | [+0.01248, +0.03937] | 97.32% | 30.12% | PASS |
| Confirmation folds 0/1 | 0.750 | 0.55135 | 0.57258 | +0.02123 | [+0.01276, +0.03427] | 97.17% | 26.05% | PASS |
| All five folds | 0.750 | 0.56404 | 0.58625 | +0.02221 | [+0.01490, +0.03224] | 97.26% | 28.37% | PASS |

Freeze the scene-gated mask rule for one transparent post-test paper-cache replay.
