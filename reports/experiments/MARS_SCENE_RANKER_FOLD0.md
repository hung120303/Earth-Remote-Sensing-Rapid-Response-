# Frozen MARS scene ranker on fold 0

The scene head was frozen from folds 2-4. Fold 1 and the paper test were not loaded.

| Model | AP | Recall at <=7.13% FPR | FPR | Pixel IoU |
|---|---:|---:|---:|---:|
| Released MARS-S2L | 0.88645 | 0.95705 | 0.07122 | 0.50863 |
| Frozen segmentation + scene head | 0.89752 | 0.95570 | 0.07122 | 0.51897 |

Fold-0 gate failed.

Reject the frozen scene-head architecture on fold 0.
