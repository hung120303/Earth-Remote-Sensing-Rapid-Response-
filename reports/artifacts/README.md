# Reproducible artifact records

`compact_resunet_v1_config.json` is a tracked copy of the final local artifact contract. It records the exact model SHA-256, cohort SHA-256, feature and radiometric contracts, train/calibration split, normalization statistics, calibrated threshold, limitations, and dependency provenance.

The corresponding `model.keras` is intentionally ignored and is reproduced with:

```bash
CUDA_VISIBLE_DEVICES=-1 python tools/train_compact_model.py \
  --architecture raw_resunet --image-size 128 --base-filters 8 \
  --threshold 300 --positive-weight 1 --epochs 12 \
  --batch-size 4 --seed 42
```

This is a `research_only` artifact. Its calibration threshold is `0.01`, with sampled calibration specificity `0.00127`; that operating point is effectively an almost-everywhere positive mask. It is evidence that the current positive-centered cohort cannot calibrate an operational alert threshold, not evidence of deployment readiness.
