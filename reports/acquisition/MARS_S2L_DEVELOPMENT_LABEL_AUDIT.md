# MARS-S2L development label audit

This audit covers development labels only; the sealed paper test was not loaded.

- Positive scenes: 3,811
- Raw empty positive masks: 6
- Positive masks empty after cloud/radiometric observability: 6
- Raw / observable positive pixels: 5,377,303 / 5,369,718

| Stratum | Positive | Raw empty | Observable empty |
|---|---:|---:|---:|
| fold:0 | 745 | 0 | 0 |
| fold:1 | 745 | 0 | 0 |
| fold:2 | 797 | 0 | 0 |
| fold:3 | 758 | 0 | 0 |
| fold:4 | 766 | 6 | 6 |
| sensor:Landsat | 946 | 2 | 2 |
| sensor:Sentinel-2 | 2,865 | 4 | 4 |
| split:development_training | 3,512 | 6 | 6 |
| split:development_validation | 299 | 0 | 0 |

## Exceptions

- `1f8549f9-93f5-4a49-acf8-985dd67f7509` (development_training, fold 4, Sentinel-2): raw=0, observable=0 pixels.
- `30866940-9f13-4629-a7d2-a51c27337d3f` (development_training, fold 4, Sentinel-2): raw=0, observable=0 pixels.
- `3af54601-2eb0-4b03-af23-fbc8b2f02ab0` (development_training, fold 4, Landsat): raw=0, observable=0 pixels.
- `b0f0570d-48e8-4789-9451-1bfef59ec4ba` (development_training, fold 4, Landsat): raw=0, observable=0 pixels.
- `c246c1f5-ffd9-4a2c-888c-96d762634c1a` (development_training, fold 4, Sentinel-2): raw=0, observable=0 pixels.
- `f03af98f-dff1-4269-9a4c-2a683f804090` (development_training, fold 4, Sentinel-2): raw=0, observable=0 pixels.
