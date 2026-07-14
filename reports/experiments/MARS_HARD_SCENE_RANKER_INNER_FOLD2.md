# Hard-example MARS scene ranker inner selection

Train folds 3-4, validate fold 2; folds 0/1 and the paper test were not loaded.

- Logistic C: 0.1
- Hard-positive multiplier: 2.0
- Hard-negative multiplier: 2.0
- Head blend weight: 0.25
- AP delta: +0.00650
- Recall delta: +0.00125
- Robust inner gate: fail

Reject hard-example scene ranking before another fold-0 evaluation.
