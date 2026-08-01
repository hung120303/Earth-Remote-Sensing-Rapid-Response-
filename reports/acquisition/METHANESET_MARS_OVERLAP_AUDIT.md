# MethaneSET / MARS-S2L identity-overlap audit

Generated: 2026-08-01T00:55:03.527852+00:00.

| MethaneSET subset | Rows | Exact MARS IDs | Exact official-test IDs |
|---|---:|---:|---:|
| methaneset-s2-pretraining | 57,291 | 57,290 | 23,250 |
| methaneset-s2-finetune | 3,612 | 3,612 | 1,053 |
| methaneset-l89-pretraining | 21,926 | 21,924 | 16,311 |
| methaneset-l89-finetune | 1,548 | 1,547 | 714 |

MethaneSET is a valuable repackaging of the MARS corpus, but these four multispectral subsets are not new independent supervision for the exact MARS-S2L paper benchmark. Downloading the imagery would either duplicate existing training rows or leak exact validation/test observations. Only metadata was downloaded; no MethaneSET imagery was acquired or used.
