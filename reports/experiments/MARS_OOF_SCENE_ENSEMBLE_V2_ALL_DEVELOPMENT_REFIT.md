# Frozen MARS scene head: all-development refit

The ExtraTrees specification, feature schema, and 0.625 logit blend were selected and confirmed before this refit. All five authorized development folds are now used for the final fit; paper imagery and labels were not loaded.

- Rows / positive scenes / sites: 44,363 / 3,811 / 618
- Frozen specification: `{'family': 'extra_trees', 'weighting': 'uniform', 'n_estimators': 400, 'min_samples_leaf': 5, 'max_features': 0.5}`
- Development-calibrated threshold at <=7.13% FPR: 0.094263858
- In-sample fit AP (audit only): 0.97492
- Artifact SHA-256: `7dd81a2f1d9b30b88500eeceb086664c4a3fb1cad21810a10783b2ce72c4ab1a`

Freeze this all-development refit for one transparent exact-paper cache evaluation.
