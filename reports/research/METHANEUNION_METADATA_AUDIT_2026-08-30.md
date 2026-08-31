# MethaneUnion metadata and overlap audit

Date: 2026-08-30  
Status: candidate auxiliary-data source; bulk imagery not downloaded  
Scope: metadata-only audit performed while the frozen sensor-aware ordinal run continued

## Decision

MethaneUnion is a credible source of some genuinely new Sentinel-2 geography, but
it is not a drop-in benchmark and its full release must not be downloaded as one
bulk operation on this workstation. If the frozen ordinal experiment fails, the
next research path is a preregistered, training-only acquisition of the novel
MethaneUnion Sentinel-2 subset, followed by mask and observability adjudication.

The release's own geographic split is contextual evidence only. It is exact-coordinate
disjoint, but it is not disjoint under this project's 25 km connected-site contract.
Its negative queries are event-centered background crops around Carbon Mapper plume
scenes rather than independently sampled no-plume sites, so every candidate negative
must have a zero dense mask and pass observability checks before training.

## Pinned sources

- [MethaneUnion repository](https://github.com/yuyao-wang/MethaneUnion), commit
  `84009297dc2847dd5436695087eb2ea71d04d68e`.
- [MethaneFuse repository](https://github.com/yuyao-wang/MethaneFuse), commit
  `060e3caeced2b0557f1dc0a54f63c2aea5b9e8b6`.
- [MethaneUnion release](https://huggingface.co/datasets/yuyao42/MethaneUnion),
  CC BY-NC 4.0.
- Geo-split 480 m train manifest: 5,656,552 bytes, SHA-256
  `4333a0edb9ff49bb19cdedf0e77489816daf3bc3a568725146301a8619132d46`.
- Geo-split 480 m test manifest: 1,345,740 bytes, SHA-256
  `7e4fd2586029641f862f28f27146c84e47791b9095cfc58d8823d2008c93c65f`.

The source repositories describe an ICDM 2026-accepted MethaneFuse release built
from Carbon Mapper plume reports and matched Sentinel-2, Landsat 8/9, EMIT, and
Sentinel-5P observations. The release card reports 3,211 valid Sentinel-2 events
and 8,981 observable multi-sensor events. These are author-release facts; this
audit independently checks only the downloaded manifests and pinned source metadata.

## Released-manifest audit

The 480 m geographic split contains 31,587 rows:

| Split | Rows | Positive | Negative | Unique coordinates |
| --- | ---: | ---: | ---: | ---: |
| Train | 24,794 | 13,338 | 11,456 | 1,133 |
| Test | 6,793 | 3,436 | 3,357 | 402 |

Train/test exact-coordinate overlap is zero. Distance auditing shows, however:

- 29 of 402 test coordinates are within 1 km of a train coordinate;
- 243 of 402 are within 25 km;
- 159 of 402 are beyond 25 km;
- median nearest-train distance is 14.83 km.

This split therefore does not meet the ERSRR 25 km site-disjoint protocol and must
not be described as equivalent to it.

## Novelty relative to existing ERSRR data

The pinned MethaneUnion source table contains 4,066 rows and 3,989 Sentinel-2 source
coordinates. Relative to all 4,276 MethaneS2CM locations, 1,662 are exact matches,
3,466 are within 25 km, and 523 are beyond 25 km.

On the released 480 m manifests, focusing on rows with Sentinel-2 available:

| Label | Unique query coordinates | Beyond 25 km from MethaneS2CM | Beyond 25 km from MethaneS2CM and all pinned MARS train/strict coordinates |
| --- | ---: | ---: | ---: |
| Negative | 499 | 77 | 58 |
| Positive | 474 | 73 | 54 |

The novel coordinates are concentrated in the released training split:

- train: 56 negative and 52 positive Sentinel-2 coordinates beyond 25 km from both
  existing cohorts;
- test: only 2 negative and 2 positive coordinates beyond the same boundary.

The training subset is therefore the plausible auxiliary-data contribution. The
released test is mostly overlapping geography and is not a strong independent
confirmation resource for this project.

## Negative-label boundary

The pinned generation code samples negative query windows away from the plume
center in the same event-centered scene. One pipeline explicitly rejects windows
containing any plume-mask pixel; another legacy query routine requires that the
center not be contained. Consequently, a `label=0` manifest value alone is not
sufficient for ERSRR's no-plume contract. Acquisition must verify the released
dense mask is identically zero and should record distance from the plume footprint.

These examples can become hard near-event backgrounds. They cannot replace the
independent CloudSEN12/no-emitter negative cohort needed for deployment FPR claims.

## Storage and acquisition contract

The Hugging Face release contains 35 compressed archives totaling 156,173,577,972
bytes (145.45 GiB). At audit time the C: drive had only 126.63 GiB free, before
extraction. A full download is therefore forbidden.

If authorized by ordinal rejection, acquisition must:

1. freeze the 52 positive and 56 negative training coordinates and their exact row
   identities before pixel access;
2. exclude every row within 25 km of any pinned MARS or MethaneS2CM coordinate;
3. process at most one approximately 4.5 GB release archive at a time, extracting
   only selected Sentinel-2 target/reference/mask members and deleting the archive
   only after hash and member receipts are durable;
4. reject negative rows with nonzero mask pixels, missing frames, invalid bands, or
   failed radiometric/observability checks;
5. retain imagery under ignored research storage and commit only compact manifests,
   checksums, aggregate audits, protocol, code, and results;
6. keep the released test split sealed during architecture development.

No bulk archive, image tensor, model outcome, or held MARS outcome was opened by
this metadata audit.
