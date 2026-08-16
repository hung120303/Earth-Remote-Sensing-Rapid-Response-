# MARS prior-reference-bank audit: folds 3/4

Generated: 2026-08-16T16:22:12.902578+00:00.

## Outcome-blind feasibility

The frozen selector found at least one strictly prior, >=95% clear, exact-grid Sentinel-2 reference for 14,569/14,963 rows (97.37%).

A full five-reference set is available for 13,733 Sentinel-2 rows. Landsat remains an exact champion identity in the proposed pilot.

| Fold / sensor | Rows | Any selected reference | Five references |
|---|---:|---:|---:|
| fold 3 / Sentinel-2 | 7,403 | 7,170 | 6,730 |
| fold 3 / Landsat | 1,396 | 0 | 0 |
| fold 4 / Sentinel-2 | 7,560 | 7,399 | 7,003 |
| fold 4 / Landsat | 1,386 | 0 | 0 |

No label, plume mask, flux, model score, prediction, or held outcome was read. The bank remains an ignored data artifact and is not yet authorized for model selection.

## Research basis

Project Eucalyptus identifies poor or methane-contaminated reference scenes as a false-negative mechanism and explicitly recommends averages over the last 5/10 overpasses, similarity-based selection, or learned attention. This audit tests only whether the first two ingredients are locally feasible; it makes no performance claim.

## Decision

FAIL: do not run alternate-reference model inference.
