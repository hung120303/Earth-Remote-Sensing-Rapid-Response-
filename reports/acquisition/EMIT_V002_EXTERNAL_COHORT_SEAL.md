# EMIT V002 external cohort seal

## Result

- Preliminary SCL/radiometry/containment gate: **60**.
- Final exact-model-input gate: **55 pass / 5 fail**.
- Minimum 50-independent-group goal: **pass**.
- No methane-detector prediction was computed or consulted.

The final seal requires at least 70% valid target/reference radiometry and CloudSEN12-clear support across both the full detector window and the EMIT plume mask. This resolves a prediction-blind observability disagreement between Sentinel-2 L2A SCL and the exact cloud model used by MARS.

## Exclusions

- `model_local_observable`: 5
- `model_plume_support_observable`: 5

## Frozen records

| Group | Model observable | Plume observable | Gate |
|---|---:|---:|---:|
| `emit25km-0001` | 100.0% | 100.0% | pass |
| `emit25km-0003` | 100.0% | 100.0% | pass |
| `emit25km-0004` | 100.0% | 100.0% | pass |
| `emit25km-0006` | 100.0% | 100.0% | pass |
| `emit25km-0007` | 100.0% | 100.0% | pass |
| `emit25km-0008` | 100.0% | 100.0% | pass |
| `emit25km-0009` | 100.0% | 100.0% | pass |
| `emit25km-0010` | 100.0% | 100.0% | pass |
| `emit25km-0011` | 97.0% | 100.0% | pass |
| `emit25km-0012` | 100.0% | 100.0% | pass |
| `emit25km-0013` | 98.6% | 100.0% | pass |
| `emit25km-0014` | 100.0% | 100.0% | pass |
| `emit25km-0015` | 100.0% | 100.0% | pass |
| `emit25km-0016` | 100.0% | 100.0% | pass |
| `emit25km-0017` | 100.0% | 100.0% | pass |
| `emit25km-0018` | 100.0% | 100.0% | pass |
| `emit25km-0021` | 93.7% | 100.0% | pass |
| `emit25km-0022` | 100.0% | 100.0% | pass |
| `emit25km-0023` | 100.0% | 100.0% | pass |
| `emit25km-0024` | 100.0% | 100.0% | pass |
| `emit25km-0025` | 100.0% | 100.0% | pass |
| `emit25km-0026` | 100.0% | 100.0% | pass |
| `emit25km-0027` | 100.0% | 100.0% | pass |
| `emit25km-0028` | 100.0% | 100.0% | pass |
| `emit25km-0029` | 100.0% | 100.0% | pass |
| `emit25km-0030` | 100.0% | 100.0% | pass |
| `emit25km-0031` | 0.0% | 0.0% | fail |
| `emit25km-0032` | 94.2% | 100.0% | pass |
| `emit25km-0033` | 100.0% | 100.0% | pass |
| `emit25km-0034` | 95.6% | 100.0% | pass |
| `emit25km-0035` | 100.0% | 100.0% | pass |
| `emit25km-0036` | 100.0% | 100.0% | pass |
| `emit25km-0037` | 100.0% | 100.0% | pass |
| `emit25km-0038` | 97.5% | 99.0% | pass |
| `emit25km-0039` | 95.2% | 100.0% | pass |
| `emit25km-0041` | 0.0% | 0.0% | fail |
| `emit25km-0042` | 100.0% | 100.0% | pass |
| `emit25km-0043` | 4.6% | 0.0% | fail |
| `emit25km-0044` | 100.0% | 100.0% | pass |
| `emit25km-0046` | 100.0% | 100.0% | pass |
| `emit25km-0047` | 100.0% | 100.0% | pass |
| `emit25km-0049` | 100.0% | 100.0% | pass |
| `emit25km-0052` | 61.7% | 55.6% | fail |
| `emit25km-0053` | 9.7% | 11.9% | fail |
| `emit25km-0054` | 100.0% | 100.0% | pass |
| `emit25km-0055` | 100.0% | 100.0% | pass |
| `emit25km-0056` | 100.0% | 100.0% | pass |
| `emit25km-0057` | 100.0% | 100.0% | pass |
| `emit25km-0058` | 100.0% | 100.0% | pass |
| `emit25km-0059` | 100.0% | 100.0% | pass |
| `emit25km-0060` | 100.0% | 100.0% | pass |
| `emit25km-0061` | 100.0% | 100.0% | pass |
| `emit25km-0062` | 100.0% | 100.0% | pass |
| `emit25km-0063` | 100.0% | 100.0% | pass |
| `emit25km-0064` | 100.0% | 100.0% | pass |
| `emit25km-0065` | 100.0% | 100.0% | pass |
| `emit25km-0066` | 70.3% | 82.2% | pass |
| `emit25km-0067` | 100.0% | 100.0% | pass |
| `emit25km-0068` | 100.0% | 100.0% | pass |
| `emit25km-0069` | 100.0% | 100.0% | pass |

This is an independent positive-confirmation cohort. It does not estimate no-plume false-positive rate; the sealed MARS-S2L strict cohort provides that benchmark.
