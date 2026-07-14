# MARS-S2L offshore fine-tune diagnostic

This is a post-test diagnostic of the authors' two archived prediction files. It does not change or rerun the frozen ERSRR one-shot result.

## Effect of substituting the fine-tuned model only offshore

| View | Offshore rows | General AP | Hybrid AP | ΔAP | General IoU | Hybrid IoU | ΔIoU |
|---|---:|---:|---:|---:|---:|---:|---:|
| full | 2,185 | 0.63910 | 0.64102 | +0.00192 | 0.30255 | 0.32437 | +0.02182 |
| test_only_sites | 583 | 0.45215 | 0.45027 | -0.00188 | 0.15282 | 0.17156 | +0.01874 |

## Offshore-only strata

| Stratum | Rows | Plume | General AP | Fine-tuned AP | ΔAP | General IoU | Fine-tuned IoU | ΔIoU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| all_offshore | 2,185 | 39 | 0.39099 | 0.52876 | +0.13777 | 0.08929 | 0.22685 | +0.13756 |
| seen_site_offshore | 1,602 | 39 | 0.43756 | 0.56020 | +0.12263 | 0.12436 | 0.26148 | +0.13713 |
| sentinel2_offshore | 1,116 | 14 | 0.21514 | 0.42508 | +0.20994 | 0.03988 | 0.12471 | +0.08483 |
| landsat_offshore | 1,069 | 25 | 0.49656 | 0.59583 | +0.09927 | 0.12929 | 0.27989 | +0.15059 |
| seen_site_sentinel2_offshore | 806 | 14 | 0.24054 | 0.44581 | +0.20527 | 0.05686 | 0.14272 | +0.08586 |
| seen_site_landsat_offshore | 796 | 25 | 0.54498 | 0.62670 | +0.08172 | 0.17676 | 0.32385 | +0.14709 |

Across 55 offshore sites, fine-tuning improved site-level pixel IoU at 8, regressed at 1, and tied at 46.

The paper specifies one extra full-model epoch on offshore real data. The public release includes the general checkpoint and both prediction archives, but not the offshore checkpoint, so exact weight reproduction requires retraining.
