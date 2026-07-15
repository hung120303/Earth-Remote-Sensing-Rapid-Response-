# Stronger OOF MARS scene ensemble: one-shot fold 0

The head was frozen on folds 2/3/4. This evaluates fold 1; the paper test was not loaded.

| Model | AP | Recall at <=7.13% FPR | FPR | Pixel IoU (diagnostic) |
|---|---:|---:|---:|---:|
| Released MARS-S2L | 0.85975 | 0.93691 | 0.07128 | 0.46937 |
| OOF scene ensemble v2 | 0.89661 | 0.94631 | 0.07128 | 0.46026 |

AP delta: +0.03686; paired site-bootstrap 95% CI [+0.01875, +0.05828].

Promotion is scene-only: final dense masks use released logits with separately confirmed sensor thresholds, not this residual endpoint.

Advance the stronger OOF scene ensemble to the next confirmation stage.
