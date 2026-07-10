# MARS-S2L frozen publication cohort

- Source revision: `c26b1d7e31a0c5241fa37c9140802622c215eb32`
- Samples: 56,552 (3,826 plume / 52,726 no plume)
- Unique remote assets: 120,756
- Exact transfer size: 58,455,597,233 bytes (58.456 GB / 54.441 GiB)
- Detailed manifest SHA-256: `d5d41635f79a93fd79f8678b75b788c45f41ba5f3812166d00cc5558d91dc8a3`
- Remote catalog SHA-256: `14ebf04712220d658c517bcc92dd6041d6b208b315ed7843f7ebbef2f73096ac`

## Selection

Official split rows only; Sentinel-2 MSI L1C; `observability=clear`; at least 80% clear; background reference required.

| Split | Plume | No plume |
|---|---:|---:|
| train | 2,512 | 27,196 |
| val | 264 | 5,263 |
| test | 1,050 | 20,267 |

## Split and grouping warning

The cohort contains 1,208 physical locations and 272 connected 25 km groups. 107 groups cross official split boundaries. The official validation locations are not isolated from training, so create a group-disjoint internal validation split for model selection. Preserve the official validation/test results for comparison, use the 621 test-only locations (7,953 samples: 134 plume / 7,819 no plume) for the primary geographic-transfer claim, and never tune thresholds on either test view.

## Transfer decision

This manifest is the proposed maximum first S2 scope. It inventories files only and does not download the raster corpus. A large transfer should proceed only after the exact byte total above is accepted. Raw files and both detailed JSONL manifests remain ignored by Git.
