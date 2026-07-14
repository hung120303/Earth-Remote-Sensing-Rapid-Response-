# MARS paper residual fold 0 validation

- Scope: frozen site-held development
- Fit / held-out scenes: 35,376 / 8,987
- Best epoch: 7

| Model | AP | Recall at <=7.13% FPR | Pixel IoU at 0.5 |
|---|---:|---:|---:|
| Released MARS-S2L | 0.8865 | 0.9570 | 0.5086 |
| ERSRR residual | 0.8860 | 0.9557 | 0.4988 |

Deltas (ERSRR minus released): AP -0.0005, recall -0.0013, pixel IoU -0.0098.

Do not advance: at least one predeclared primary-fold point gate failed.
