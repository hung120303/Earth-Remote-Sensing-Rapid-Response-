# Joint external-data successor: exact MARS-S2L paper benchmark

Transparent post-test replay, not an untouched confirmation. Candidate scores were computed from the label-free cache before comparator outcomes were opened; dense masks remain the unchanged promoted v3 branch.

| View | AP | AP delta (95% CI) | Matched-FPR recall delta (95% CI) | FPR delta | IoU delta (95% CI) | Gates |
|---|---:|---:|---:|---:|---:|---|
| full | 0.67416 | +0.03314 ([+0.01298, +0.04843]) | +0.03365 ([+0.01973, +0.04660]) | -0.03915 | +0.05560 ([+0.03506, +0.07802]) | PASS |
| test_only_sites | 0.44811 | -0.00216 ([-0.04445, +0.03587]) | +0.01762 ([-0.01081, +0.04590]) | -0.04226 | +0.12090 ([+0.08597, +0.15545]) | FAIL |

At least one exact MARS-S2L v3 paper gate remains unresolved.
