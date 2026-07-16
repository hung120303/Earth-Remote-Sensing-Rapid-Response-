# UNEP MARS post-2024 acquisition protocol

Status: **frozen with one pre-outcome schema correction** on 2026-07-16 UTC.

The initial header/category audit, performed before any eligibility outcome was
computed, showed that UNEP encodes both Landsat spacecraft under
`Landsat - NASA/USGS`. The frozen placeholder names were corrected to that
catalog value; exact `LC08`/`LC09` prefixes still identify and constrain the
spacecraft. This is a schema repair, not a selection change.

The GeoJSON schema audit also established that records describe full satellite
products plus source coordinates. Because one full product can contain several
distant source-centered crops, the deduplication key is clarified as exact target
product plus source-centered crop. Only multiple plume polygons for that same
sample are merged; distant sources on one product remain separate samples.

Before pixel acquisition, direct inspection of a released MARS image fixed the
grid contract at 200×200 pixels with 10 m spacing: a 2×2 km footprint. Earlier
text calling this a 4 km crop was incorrect; prior benchmark computations used
the actual stored rasters and are unaffected.

## Why this cohort

UNEP IMEO's Eye on Methane export identifies each MARS plume with the exact
target satellite product, the exact background product used for multitemporal
retrieval, UTC acquisition time, source coordinates, satellite, flux/wind
metadata, and expert-validation status. Sentinel-2 and Landsat rows therefore
match the released MARS-S2L target/reference input contract without the
cross-sensor time offset that limited the EMIT external cohort.

Primary sources:

- [Download page](https://methanedata.unep.org/download-dataset)
- [MARS plume data dictionary](https://methanedata.unep.org/dict-mars-plumes)
- [UNEP IMEO satellite detection and quantification methodology](https://wedocs.unep.org/rest/api/core/bitstreams/e2c82c4c-b3bd-4a18-97bd-8e544a46d88a/content)

The export is CC BY-NC-SA 4.0 and may not be used commercially. All derived
research artifacts must credit UNEP IMEO and retain compatible terms.

## Frozen selection

Use only observations at or after 2025-01-01 UTC from Sentinel-2 or UNEP's
aggregate Landsat category, constrained to exact `LC08`/`LC09` products.
Require `actionable=YES` and nonempty plume ID, source ID,
satellite, timestamp, target product, background product, latitude, and
longitude. Target and background must be distinct, sensor-consistent exact
product IDs. Multiple plume records on one target and one source-centered crop
are merged rather than duplicating that sample; one full product may validly
yield multiple distant source crops.

Every accepted source must be at least 25 km from every location in the pinned
MARS-S2L paper test CSV, and every exact paper-test target product is excluded.
The pinned paper-test CSV SHA-256 is
`add125547e0e0066216070ed61a8544e76e84f062f636390be5d2ef1808dbfaa`.
This stronger exclusion prevents auxiliary labels from exposing either an
exact paper-test acquisition or its facility context.

GeoJSON Polygon/MultiPolygon geometry may supervise pixels after intersection,
rasterization, observability, and nonempty-mask checks. Point geometry supplies
scene-positive evidence only. Flux, wind, geometry, validation status, and
source identity are label/provenance fields and never model inputs. Catalog
absence is never a no-plume label.

## Fixed roles

Canonical source/25-km groups are assigned with
`sha256("ERSRR-UNEP-MARS-v1|" + group_id) mod 10`: buckets 0-7 are auxiliary
training, bucket 8 is development, and bucket 9 is sealed external
confirmation. No group can cross roles. The sealed bucket stays unread until
the architecture, calibration, and thresholds are frozen.

The external cohort is positive-only unless a separately validated negative
source is added under a new preregistration. It can therefore support positive
recall and mask-IoU claims, not AP or false-positive-rate claims.

## Storage and acquisition

Catalog ZIPs, exact product imagery, masks, and checkpoints belong under
`.research/unep_mars_post2024/`, which is already ignored. Git receives only
code, this protocol, compact checksum/count manifests, acquisition audits, and
experiment summaries. Sentinel-2 must remain MSI L1C and Landsat must remain
USGS Collection-2 Level-1; product substitution is prohibited.

The machine-readable frozen contract is
`configs/unep_mars_post2024_protocol.json`.
