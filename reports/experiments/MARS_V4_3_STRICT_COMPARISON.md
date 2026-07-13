# ERSRR v4.3 frozen strict MARS comparison

Development benchmark on the already-opened ERSRR strict cohort. All v4.3 calibrators and thresholds were frozen on internal validation.

| Model | Recall | FPR | Precision | AP | AUROC | Pixel IoU | Pixel Dice |
|---|---:|---:|---:|---:|---:|---:|---:|
| ERSRR v4.3 ensemble | 0.5224 | 0.0277 | 0.2258 | 0.3903 | 0.8644 | 0.0766 | 0.1422 |
| Released MARS-S2L | 0.6418 | 0.0948 | 0.0947 | 0.3521 | 0.8175 | 0.1329 | 0.2346 |

## Same-cohort deltas (ERSRR - released MARS-S2L)

- Recall: -0.1194 (paired group-bootstrap 95% CI -0.2315 to +0.0606)
- FPR: -0.0671 (95% CI -0.0854 to -0.0490)
- AP: +0.0382 (95% CI -0.0845 to +0.1885)
- AUROC: +0.0469 (95% CI -0.0385 to +0.0978)
- Pixel IoU / Dice: -0.0563 / -0.0924

## Interpretation

ERSRR v4.3 does not outperform the released MARS-S2L checkpoint on every same-cohort point metric; failed comparisons: recall_higher, pixel_iou_higher, pixel_dice_higher. Preserve this result and do not retune from strict behavior.

The official MARS-S2L paper benchmarks use different, much larger test cohorts and are contextual only; they are not substituted for this paired comparison.
