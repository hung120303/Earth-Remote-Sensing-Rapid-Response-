# EMIT external L1C raster-quality gate

- Requested pairs: 70
- Verified local pairs: 70
- Gate-pass pairs: 60
- Gate-fail pairs: 10
- Acquisition errors: 0
- Ignored cropped assets: 350 / 45,931,204 bytes
- Minimum 50-group acquisition goal: `pass`

The gate uses public six-band L1C target/reference radiometry on the native 200x200, 10 m detector grid and co-temporal L2A SCL only for observability. Selection and quality filtering remain prediction-blind.

| Group | Target clear | Reference clear | Plume clear | Margin px | Gate |
|---|---:|---:|---:|---:|---:|
| `emit25km-0001` | 100.0% | 100.0% | 100.0% | 21.7 | pass |
| `emit25km-0002` | 40.1% | 100.0% | 38.9% | 0.1 | fail |
| `emit25km-0003` | 100.0% | 99.6% | 100.0% | 8.4 | pass |
| `emit25km-0004` | 100.0% | 100.0% | 100.0% | 18.6 | pass |
| `emit25km-0005` | 100.0% | 64.4% | 100.0% | 57.7 | fail |
| `emit25km-0006` | 100.0% | 100.0% | 100.0% | 9.2 | pass |
| `emit25km-0007` | 100.0% | 100.0% | 100.0% | 60.0 | pass |
| `emit25km-0008` | 100.0% | 100.0% | 100.0% | 35.6 | pass |
| `emit25km-0009` | 100.0% | 100.0% | 100.0% | 63.2 | pass |
| `emit25km-0010` | 100.0% | 100.0% | 100.0% | 23.9 | pass |
| `emit25km-0011` | 97.0% | 96.6% | 100.0% | 38.6 | pass |
| `emit25km-0012` | 100.0% | 100.0% | 100.0% | 63.5 | pass |
| `emit25km-0013` | 100.0% | 100.0% | 100.0% | 16.3 | pass |
| `emit25km-0014` | 100.0% | 100.0% | 100.0% | 54.1 | pass |
| `emit25km-0015` | 99.9% | 99.2% | 100.0% | 34.3 | pass |
| `emit25km-0016` | 100.0% | 100.0% | 100.0% | 44.6 | pass |
| `emit25km-0017` | 100.0% | 100.0% | 100.0% | 47.7 | pass |
| `emit25km-0018` | 100.0% | 100.0% | 100.0% | 39.1 | pass |
| `emit25km-0019` | 59.7% | 100.0% | 62.5% | 26.9 | fail |
| `emit25km-0020` | 23.8% | 100.0% | 22.6% | 25.5 | fail |
| `emit25km-0021` | 92.8% | 100.0% | 98.5% | 24.1 | pass |
| `emit25km-0022` | 100.0% | 100.0% | 100.0% | 6.2 | pass |
| `emit25km-0023` | 100.0% | 100.0% | 100.0% | 45.7 | pass |
| `emit25km-0024` | 100.0% | 100.0% | 100.0% | 7.5 | pass |
| `emit25km-0025` | 100.0% | 100.0% | 100.0% | 9.1 | pass |
| `emit25km-0026` | 100.0% | 100.0% | 100.0% | 41.9 | pass |
| `emit25km-0027` | 100.0% | 100.0% | 100.0% | 32.8 | pass |
| `emit25km-0028` | 100.0% | 100.0% | 100.0% | 36.0 | pass |
| `emit25km-0029` | 100.0% | 100.0% | 100.0% | 5.8 | pass |
| `emit25km-0030` | 100.0% | 100.0% | 100.0% | 18.2 | pass |
| `emit25km-0031` | 100.0% | 100.0% | 100.0% | 11.0 | pass |
| `emit25km-0032` | 94.8% | 100.0% | 100.0% | 41.8 | pass |
| `emit25km-0033` | 100.0% | 100.0% | 100.0% | 21.5 | pass |
| `emit25km-0034` | 100.0% | 100.0% | 100.0% | 36.4 | pass |
| `emit25km-0035` | 100.0% | 100.0% | 100.0% | 0.4 | pass |
| `emit25km-0036` | 100.0% | 100.0% | 100.0% | 4.4 | pass |
| `emit25km-0037` | 100.0% | 100.0% | 100.0% | 55.5 | pass |
| `emit25km-0038` | 100.0% | 88.6% | 100.0% | 48.2 | pass |
| `emit25km-0039` | 98.4% | 100.0% | 100.0% | 57.7 | pass |
| `emit25km-0040` | 92.4% | 100.0% | 57.4% | 77.1 | fail |
| `emit25km-0041` | 100.0% | 100.0% | 100.0% | 17.8 | pass |
| `emit25km-0042` | 100.0% | 99.9% | 100.0% | 36.7 | pass |
| `emit25km-0043` | 100.0% | 100.0% | 100.0% | 41.6 | pass |
| `emit25km-0044` | 99.5% | 99.0% | 99.2% | 24.9 | pass |
| `emit25km-0045` | 60.2% | 100.0% | 55.1% | 17.8 | fail |
| `emit25km-0046` | 100.0% | 100.0% | 100.0% | 3.7 | pass |
| `emit25km-0047` | 100.0% | 100.0% | 100.0% | 57.5 | pass |
| `emit25km-0048` | 100.0% | 34.8% | 100.0% | 2.0 | fail |
| `emit25km-0049` | 100.0% | 100.0% | 100.0% | 12.5 | pass |
| `emit25km-0050` | 62.7% | 91.7% | 53.4% | 0.9 | fail |
| `emit25km-0051` | 98.1% | 67.9% | 99.7% | 4.8 | fail |
| `emit25km-0052` | 100.0% | 100.0% | 100.0% | 15.6 | pass |
| `emit25km-0053` | 100.0% | 100.0% | 100.0% | 52.9 | pass |
| `emit25km-0054` | 100.0% | 100.0% | 100.0% | 3.3 | pass |
| `emit25km-0055` | 100.0% | 100.0% | 100.0% | 2.5 | pass |
| `emit25km-0056` | 100.0% | 100.0% | 100.0% | 45.4 | pass |
| `emit25km-0057` | 100.0% | 100.0% | 100.0% | 54.4 | pass |
| `emit25km-0058` | 100.0% | 100.0% | 100.0% | 24.7 | pass |
| `emit25km-0059` | 100.0% | 100.0% | 100.0% | 2.8 | pass |
| `emit25km-0060` | 100.0% | 100.0% | 100.0% | 9.2 | pass |
| `emit25km-0061` | 100.0% | 100.0% | 100.0% | 12.4 | pass |
| `emit25km-0062` | 92.8% | 100.0% | 98.2% | 51.7 | pass |
| `emit25km-0063` | 100.0% | 100.0% | 100.0% | 18.0 | pass |
| `emit25km-0064` | 100.0% | 99.8% | 100.0% | 28.0 | pass |
| `emit25km-0065` | 100.0% | 100.0% | 100.0% | 43.3 | pass |
| `emit25km-0066` | 93.8% | 100.0% | 95.6% | 37.0 | pass |
| `emit25km-0067` | 100.0% | 100.0% | 100.0% | 31.5 | pass |
| `emit25km-0068` | 100.0% | 100.0% | 100.0% | 6.2 | pass |
| `emit25km-0069` | 100.0% | 100.0% | 100.0% | 28.3 | pass |
| `emit25km-0070` | 55.3% | 100.0% | 38.3% | 12.2 | fail |

## Exclusion counts

- `reference_local_clear`: 3
- `target_local_clear`: 6
- `target_plume_support_clear`: 7

Passing this raster gate does not create a locked paper label. Protected EMIT enhancement/uncertainty/sensitivity acquisition, wind reanalysis, source deduplication, and two-annotator review remain required before the one-time external evaluation.
