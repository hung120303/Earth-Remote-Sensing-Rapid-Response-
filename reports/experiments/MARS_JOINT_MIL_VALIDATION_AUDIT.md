# MIL-v2 internal-validation error audit

Validation-only audit; the strict-spatial benchmark was not loaded or scored.

- Cohort: 384 scenes / 24 frozen groups
- Presence outcomes: 51 TP / 77 FN / 12 FP / 244 TN
- Selective decisions: 63 plume / 224 no-plume / 97 abstain

## Plume-size sensitivity

| Stratum | Pixel area | n | Presence recall | Mask scene recall | Median score |
|---|---:|---:|---:|---:|---:|
| smallest | 74-491 | 32 | 0.094 | 0.594 | 0.413 |
| small | 491-844 | 32 | 0.312 | 0.625 | 0.410 |
| large | 844-1768 | 32 | 0.656 | 0.781 | 0.828 |
| largest | 1768-6492 | 32 | 0.531 | 0.875 | 0.793 |

## Error signatures

- TP median plume area: 1515.000 pixels; FN median: 643.000.
- FP median segmentation top-1% probability: 1.000; TN median: 0.925.
- FP median MBMP top-1% signal: 0.066; TN median: 0.038.

## Decision

Use these validation-only error strata to define the expanded hard-negative/positive sampling plan. Do not modify a threshold or architecture from strict-test behavior.
