# Three-fold OOF site-context MARS ranker

Every fold-2/3/4 prediction comes from a model trained on the other two folds. Folds 0/1 and the paper test were not loaded.

- Selected model: `family-hist_gradient_boosting_l2_regularization-10.0_max_leaf_nodes-31_min_samples_leaf-20`
- Head blend weight: 0.625
- Pooled AP delta: +0.01437
- Pooled recall delta: +0.00732
- Stable authorization gate: pass

Refit the OOF-stable context head on folds 2-4 and freeze fold-0 evaluation.
