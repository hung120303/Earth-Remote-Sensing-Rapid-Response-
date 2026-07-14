# MethaneS2CM v5.1 one-shot location test

The sealed location test was opened once after architecture, checkpoints, calibration, thresholds, acquisition, and evaluator code were frozen.

| Frozen model/rule | Scene AP | AUROC | Recall | FPR | Precision | Pixel AP | Dice | IoU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ERSRR v5.1 (three-seed) | 0.8180 | 0.8276 | 0.3778 | 0.0607 | 0.8630 | 0.2083 | 0.3125 | 0.1852 |
| ERSRR v4.3 zero-shot | 0.5911 | 0.5950 | 0.0000 | 0.0000 | 0.0000 | 0.1561 | 0.0000 | 0.0000 |
| Released MARS-S2L zero-shot | 0.5252 | 0.5126 | 0.0052 | 0.0033 | 0.6136 | 0.1691 | 0.0045 | 0.0023 |

## Interpretation

ERSRR v5.1 does not establish across-the-board superiority over released MARS-S2L on the frozen MethaneS2CM location test. Preserve the partial result and do not retune from this test.

- All same-cohort point checks versus released MARS-S2L pass: false
- All predeclared group-bootstrap checks pass: false
- Precision is benchmark precision on a roughly balanced crop set, not deployment positive predictive value.
- V5.1 was trained on MethaneS2CM L2A; v4.3 and released MARS-S2L are L1C-trained zero-shot comparators, so this is not an architecture-only causal comparison.
- The test result is frozen evidence. No retuning from it is permitted.
