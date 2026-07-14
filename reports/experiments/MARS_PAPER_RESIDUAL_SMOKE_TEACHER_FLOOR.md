# MARS paper residual fold 0 validation

- Scope: smoke only
- Fit / held-out scenes: 64 / 64
- Best epoch: 1

| Model | AP | Recall at <=7.13% FPR | Pixel IoU at 0.5 |
|---|---:|---:|---:|
| Released MARS-S2L | 0.9890 | 0.9375 | 0.7180 |
| ERSRR residual | 0.9890 | 0.9375 | 0.7180 |

Deltas (ERSRR minus released): AP +0.0000, recall +0.0000, pixel IoU -0.0000.

Smoke execution only; this result cannot promote an architecture.
