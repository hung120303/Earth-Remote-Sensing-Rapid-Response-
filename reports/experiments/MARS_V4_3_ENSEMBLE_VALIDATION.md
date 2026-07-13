# ERSRR v4.3 frozen ensemble validation

Predeclared internal-development evaluation; the strict cohort was not loaded.

| Estimate | AP | AUROC | Recall @ <=5% FPR | FPR | Pixel Dice |
|---|---:|---:|---:|---:|---:|
| Five-fold 25 km group-held | 0.8558 | 0.9620 | 0.8337 | 0.0474 | 0.7235 |
| Final all-validation rule | 0.8126 | 0.9536 | 0.8337 | 0.0500 | 0.7235 |
| v3 five-seed mean | 0.8103 | 0.9288 | 0.8313 | <=0.0500 | 0.5770 |

- Selected ensemble pixel threshold: 0.8
- Strict evaluation authorized: true

## Decision

Promote the frozen v4.3 ensemble to one evaluation on the already-opened strict MARS cohort using only these validation calibrators, scene thresholds, and pixel threshold. Treat the comparison as development evidence, not a new untouched paper test.

These internal values are not directly comparable to the MARS-S2L strict or paper benchmarks.
