# Stronger OOF MARS scene ensemble: one-shot fold 0

The head was frozen on folds 2/3/4. Fold 1 and the paper test were not loaded.

| Model | AP | Recall at <=7.13% FPR | FPR | Pixel IoU |
|---|---:|---:|---:|---:|
| Released MARS-S2L | 0.88645 | 0.95705 | 0.07122 | 0.50863 |
| OOF scene ensemble v2 | 0.91429 | 0.96107 | 0.07122 | 0.51897 |

AP delta: +0.02784; paired site-bootstrap 95% CI [+0.00966, +0.04546].

Advance the stronger OOF scene ensemble to independent fold-1 confirmation.
