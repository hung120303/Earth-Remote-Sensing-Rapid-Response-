# Stronger three-fold OOF MARS scene-head search

Every fold-2/3/4 score is produced by a head trained on the other two folds. Folds 0/1 and the paper test were not loaded.

- Model: `family-extra_trees_max_features-0.5_min_samples_leaf-5_n_estimators-400_weighting-uniform`
- Blend: 0.625
- Pooled AP delta: +0.02957
- Worst-fold AP delta: +0.03155
- Pooled recall delta: +0.01206
- Paired site-bootstrap AP delta: [+0.01905, +0.04190]

Refit the selected head on folds 2-4 and advance it to untouched fold-0 evaluation.
