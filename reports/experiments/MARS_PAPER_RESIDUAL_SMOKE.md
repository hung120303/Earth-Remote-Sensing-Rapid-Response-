# MARS paper residual fold 0 validation

- Scope: smoke only
- Fit / held-out scenes: 64 / 56
- Best epoch: 1

| Model | AP | Recall at <=7.13% FPR | Pixel IoU at 0.5 |
|---|---:|---:|---:|
| Released MARS-S2L | 0.9917 | 0.9583 | 0.6844 |
| ERSRR residual | 0.9917 | 0.9583 | 0.6832 |

Deltas (ERSRR minus released): AP -0.0000, recall +0.0000, pixel IoU -0.0011.

Smoke execution only; this result cannot promote an architecture.
