# GHGSat windowed observability audit

**Decision: FAIL**

- Frozen pairs attempted: 76
- Sentinel-2 exact-asset resolution failures: 47/47
- Landsat local-raster processing failures: 29/29 resolved pairs
- Observable scene-head negatives: 0
- Observable source observations / sites / 25 km components: 0 / 0 / 0
- Crops retained: False
- Dense/zero masks created: False

The first strict Sentinel-2 replay returned an empty feature list for the frozen
+/-1-second Earth Search mirror query, so there was no exact mirror. The first
exact Landsat Collection 2 Level-1 Tier-1 asset returned HTTP 302 to
`ers.cr.usgs.gov` authentication; GDAL therefore received a non-TIFF response.
These diagnostics explain the two failure classes but do not amend the frozen
source, product, cohort, crop, or observability contract.

GHGSat is retired as the missing-negative training source under this protocol.
No model run, protected outcome, dense supervision, threshold reduction,
product substitution, or failed-source pooling is authorized.

A PASS establishes a geographically novel, locally observable, scene-head-only reviewed-negative acquisition cohort. It does not establish physical-zero methane, dense no-plume truth, model complementarity, calibration, generalization, or superiority over MARS-S2L.
