# MARS scene ranker inner selection

The ranker was trained on folds 3-4 and selected on fold 2. Folds 0, 1, and the paper test were not loaded.

- Selected model: `C-0.1_family-logistic`
- Selected head weight: 0.250
- Fold-2 AP delta: +0.00673
- Fold-2 recall delta at <=7.13% FPR: +0.00125
- All inner gates pass: yes

Freeze this scene head, refit it on folds 2-4, then evaluate fold 0 once.
