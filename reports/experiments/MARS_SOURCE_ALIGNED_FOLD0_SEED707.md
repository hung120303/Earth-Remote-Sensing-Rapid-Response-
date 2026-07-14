# Source-aligned MARS residual fold-0 validation

- Scope: frozen site-held development
- Best epoch: 4

| Model | AP | Recall at <=7.13% FPR | Pixel IoU |
|---|---:|---:|---:|
| Released MARS-S2L | 0.88645 | 0.95705 | 0.50863 |
| Source-aligned residual | 0.88651 | 0.95570 | 0.49576 |

Deltas: AP +0.00006, recall -0.00134, IoU -0.01286.

Reject source-aligned residual on fold 0.
