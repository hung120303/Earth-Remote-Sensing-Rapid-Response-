# Stanford 2022 controlled-release cohort audit

Generated: 2026-08-02T06:22:30.514705+00:00.

| Contract | Count |
|---|---:|
| Exact S2/Landsat overpasses | 20 |
| Primary positives (at least 1,000 kg CH4/h) | 8 |
| Primary negatives (at most 10 kg CH4/h) | 9 |
| Sub-threshold challenge scenes | 3 |
| Resolved exact products | 19 |

The fixed-location single-blind campaign supplies genuine metered zero-release negatives and high-rate positives. It is a valuable previously unscored operating-point stress test, but all observations belong to one physical site already present in excluded upstream MARS metadata. It therefore cannot provide an independent site-block bootstrap claim or replace the official MARS-S2L benchmark.

The three intermediate-rate observations are reported separately and never silently relabeled. The 4.95 kg/h Sentinel-2 event is a primary negative because the paper explicitly treats it as more than two orders of magnitude below Sentinel-2 detectability.

MARS exact-product overlap: 8; same-site upstream rows: 677; nearest MARS location: 0.00 km.

All overlapping targets and same-site rows are in MARS's excluded `Not Used` split. This makes the cohort eligible only as a previously unscored, fixed-site diagnostic after model freeze—not as source-disjoint or geographic confirmation.

Bulk source data and future image crops remain under `.research/` and are excluded from Git.
