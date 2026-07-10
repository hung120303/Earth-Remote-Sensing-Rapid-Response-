# MARS-S2L frozen evaluation protocol

- Cohort: 56,552 pinned Sentinel-2 L1C samples
- Assignment identity: `49d48669c765f06555f90a9fb94647e4983cbce13a983805e5fa440310c11671`
- Internal split seed: 20260710; validation groups: 24 / 122
- Status: frozen before full-cohort training

| Research role | Rows | Plume | No plume | Groups | Locations |
|---|---:|---:|---:|---:|---:|
| internal_training | 23,763 | 2,007 | 21,756 | 98 | 397 |
| internal_validation | 5,945 | 505 | 5,440 | 24 | 163 |
| official_test_overlap_comparability_only | 16,916 | 983 | 15,933 | 107 | 801 |
| official_validation_comparability_only | 5,527 | 264 | 5,263 | 22 | 86 |
| strict_spatial_test | 4,401 | 67 | 4,334 | 150 | 373 |

## Leakage result

- Internal train/validation 25 km group overlap: 0.
- Official-train/strict-test 25 km group overlap: 0.
- Released validation and full test remain comparability-only because their physical/proximity groups overlap training.
- The strict spatial test is the primary test. The test-only-location view remains a weaker secondary analysis.

## Frozen operating contract

Thresholds, normalization, component area, calibration, and abstention rules are selected on the internal group-disjoint validation set only. The primary endpoint is scene recall at FPR <= 0.05 among observable accepted scenes. `NO_PLUME` requires observability and probability below the lower threshold; intermediate or invalid scenes abstain.

Promotion requires lower 95% recall CI >= 0.75, FPR <= 0.05, specificity >= 0.95, and at least 25% relative FPR reduction versus the strongest reproduced baseline without inferior recall. Five fixed seeds and 2,000 group-bootstrap replicates are mandatory.

This protocol defines evaluation; it does not claim that any current model passes the gate.
