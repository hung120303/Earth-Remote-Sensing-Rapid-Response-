# Site-context MARS scene ranker inner selection

Train folds 3-4, validate fold 2; folds 0/1 and the paper test were not loaded.

- Selected model: `family-hist_gradient_boosting_l2_regularization-10.0_max_leaf_nodes-31_min_samples_leaf-50`
- Head blend weight: 0.5
- AP delta: +0.01297
- Recall delta: +0.01380
- Robust three-TP gate: pass

Refit the context head on folds 2-4 and freeze its fold-0 evaluation.
