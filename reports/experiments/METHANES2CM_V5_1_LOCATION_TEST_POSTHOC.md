# MethaneS2CM v5.1 frozen-test post-hoc diagnostic

Exploratory calibration-transfer audit only. It does not select a test threshold, change a model, or authorize retuning.

| Development FPR target | Frozen threshold | Dev recall | Test recall | Dev FPR | Test FPR |
|---:|---:|---:|---:|---:|---:|
| 2.0% | 0.827382 | 0.2740 | 0.2436 | 0.0200 | 0.0244 |
| 5.0% | 0.733579 | 0.4671 | 0.3778 | 0.0499 | 0.0607 |
| 8.0% | 0.663238 | 0.5957 | 0.4912 | 0.0799 | 0.1035 |
| 9.5% | 0.636391 | 0.6379 | 0.5345 | 0.0950 | 0.1242 |

## Interpretation boundary

Every predeclared development threshold produced higher FPR on the location test, while recall also fell relative to development. The primary 5% target moved from 4.99% development FPR / 46.71% recall to 6.07% test FPR / 37.78% recall. This supports calibration/domain-transfer risk as a future hypothesis, not a test-driven threshold adjustment.

The next study must fit any calibration or risk-control layer on new development/calibration groups and use a newly untouched confirmation cohort. These test labels may not choose that layer.
