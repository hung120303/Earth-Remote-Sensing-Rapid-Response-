# Hyperspectral privileged-supervision research note

Date: 2026-08-16

Status: architecture evidence only; no model protocol or target-band acquisition
is authorized until the frozen MARS-Hyperspectral 300-negative gate passes.

## Question

If the NASA/ORNL COVID+Permian bridge closes the 28-negative shortfall, what
use of hyperspectral information is defensible for a deployable
Sentinel-2/Landsat student?

## Primary-source evidence

1. Pande et al., *An Adversarial Approach to Discriminative Modality
   Distillation for Remote Sensing Image Classification* (ICCV Workshops,
   2019), <https://openaccess.thecvf.com/content_ICCVW_2019/html/CROMOL/Pande_An_Adversarial_Approach_to_Discriminative_Modality_Distillation_for_Remote_Sensing_ICCVW_2019_paper.html>.
   The paper explicitly trains a teacher with privileged modalities and a
   student that hallucinates unavailable descriptors at inference. Its evidence
   is remote-sensing classification on hyperspectral and
   multispectral/panchromatic benchmarks, not methane segmentation.

2. Zhao et al., *PlumeBed: A Multispectral Satellite Methane Plume Detector
   Enabled by Transfer Learning of a Multi-Source Hyperspectral Data Set*
   (JGR Atmospheres, 2025), <https://doi.org/10.1029/2024JD042795>.
   PlumeBed composites Carbon Mapper hyperspectral methane plumes with
   Sentinel-2 backgrounds and uses domain-adversarial training. It reports mean
   macro-F1 0.86 and finds 14 super-emitters in an unseen Turkmenistan region.
   This is direct methane-specific support for hyperspectral-to-Sentinel-2
   transfer, but it is synthetic compositing rather than temporally paired
   feature distillation.

3. Rouet-Leduc and Hulbert, *Automatic detection of methane emissions in
   multispectral satellite imagery using a vision transformer* (Nature
   Communications, 2024), <https://doi.org/10.1038/s41467-024-47754-y>.
   The deployable model is a bi-temporal Sentinel-2 ViT/U-Net trained on about
   1.235 million samples with synthetic Gaussian plumes. The authors describe
   comparison with airborne detections as indirect because acquisitions occur
   on different days. Landsat support is discussed as future work, not
   validated by the reported model.

4. NASA/JPL, *EMIT Level 2B Greenhouse Gas Data Product User Guide, Second
   Release* (2025),
   <https://lpdaac.usgs.gov/documents/2250/EMIT_L2B_GHG_User_Guide_V2.pdf>.
   EMIT CH4 enhancement, uncertainty, and sensitivity layers exist on the
   60 m scene grid. High-confidence plume complexes are manually identified
   and require confirmation by three scientists. Therefore, enhancement can
   be soft privileged supervision, but absence of a published plume complex is
   not by itself an exhaustively reviewed no-plume label.

Hermes performed the bounded primary-source discovery. Each claim above was
then checked against the publisher, CVF, Nature, or NASA/JPL primary page.

## Acquisition-source triage

A second bounded Hermes sweep searched specifically for a defensible source of
the 28 missing reviewed negative pairs. Independent checks did not identify a
source that clears the frozen contract today:

- the JPL operational-GHG Zenodo API records CC BY 4.0 and publishes COVID,
  CACH4, and Permian train definitions, but the completed CACH4 audit retained
  2,565 rows in only 15 novel 25 km components, below the frozen 20-component
  gate;
- the NASA/ORNL preflight resolves exact L1B header granules for all 124 CACH4
  anchors, but the authenticated geometric grid bridge and the untouched
  COVID/Permian component count remain pending;
- the STARCOP sparse audit retained only 25 rows in 8 eligible components and
  was retired without querying a target catalog; and
- Carbon Mapper defines a null detect at an emission source under suitable
  observing conditions, not as scene-wide proof that no plume exists anywhere.

The acquisition call is therefore unchanged: finish the frozen NASA/ORNL
header bridge. Only if its metadata gates pass may a separately committed
target-catalog protocol measure the actual <=1-hour Sentinel-2/Landsat yield.
No alternative source is pooled post hoc to rescue a failed gate.
Compact receipts are
`reports/acquisition/jpl_operational_ghg_negative_supplement_metadata.json`,
`reports/acquisition/jpl_operational_ghg_ornl_stage_a_cmr_preflight.json`, and
`reports/acquisition/starcop_negative_supplement_stage_b.json`; Carbon Mapper's
definition is in <https://carbonmapper.org/articles/product-guide>.

## Architecture implication

If and only if the acquisition gate later passes, the first modeling protocol
should prefer physically interpretable privileged targets over unrestricted
teacher-feature imitation:

- separate Sentinel-2 and Landsat stems/students; a shared plume-object latent
  is an ablation, not an assumption of sensor interchangeability;
- reproject and PSF-match hyperspectral enhancement/mask evidence to the
  target-sensor grid;
- use uncertainty and sensitivity to weight a soft dense enhancement loss;
- allow dense supervision only at <=15 minutes and scene presence/absence only
  at <=1 hour, exactly as frozen in the acquisition protocol;
- treat cloud, missing coverage, or inadequate sensitivity as unobservable;
- remove every hyperspectral input and teacher branch at deployment;
- compare label-only transfer against a low-weight privileged-feature
  distillation ablation so any gain is not automatically attributed to hidden
  representations.

The largest validity threat is methane intermittency across the source/target
time offset. It creates structured label error, not ordinary independent
noise. Spectral-response/PSF mismatch and source-site selection bias are the
next threats. Landsat transfer is specifically unproven by the cited methane
papers and must pass its own sensor-level development gate.

This note does not preregister an architecture, authorize imagery downloads,
or claim expected improvement. The frozen sequence remains: close the data
gate, freeze a target-band manifest, acquire only that manifest, then freeze a
model protocol before any development outcome is opened.

## 2026-08-29 acquisition outcome

The pending NASA/ORNL bridge has now been executed under its frozen contract.
All 124 CACH4 anchors passed the grid-equivalence test with zero mismatch, and
all 13,444 COVID/Permian background rows resolved from 280 exact NASA headers.
After the unchanged geographic exclusions, however, 3,672 eligible rows formed
only 13 connected 25 km components versus the required 20. The supplement is
therefore retired without a target-catalog query.

This result does not weaken or reinterpret the 300-negative requirement. The
conditional privileged-supervision architecture remains unauthorized, and the
13 failed components cannot be pooled with CACH4, STARCOP, or Carbon Mapper.
Any future cross-modal path requires a genuinely independent preregistered
source that clears the existing gate on its own.

## 2026-08-29 independent replacement-source protocol

The next candidate is the licensed GHGSat global-landfill release, not a pool
of any retired source. Its 434 released null observations are reviewed
site-level non-detections in clear-sky GHGSat acquisitions. They are valid only
as sensitivity-qualified source-site claims and cannot become dense all-zero
masks or physical-zero labels. The dataset is CC BY-NC-SA 4.0; model research
and any materially trained redistributed checkpoint must preserve the
non-commercial, attribution, and share-alike boundary.

Only C1/C2 observations enter the frozen metadata headroom gate because their
10:30 local descending-node time is aligned in principle with Sentinel-2 and
Landsat daytime acquisitions; C3-C5 are approximately 13:00. The audit must
independently clear 56 selected morning nulls, 30 sites, and 20 novel 25 km
components before a target catalog can be queried. Passing that audit would
still authorize only a new pairing protocol, not the privileged-student model.
The full CSV, GHGSat rasters, and target imagery remain unopened at protocol
freeze. See `configs/mars_ghgsat_landfill_null_protocol.json`.

## 2026-08-29 GHGSat metadata outcome

The frozen metadata audit passes: 176 protected-geography-safe C1/C2 reviewed
null observations remain after deterministic selection and exclusions, across
66 sites and 64 independent 25 km components. Requirements were 56, 30, and
20, respectively. The released paper counts also reconcile exactly. This
closes the independent-source metadata gate with substantial headroom, without
pooling a failed source.

The conditional privileged-supervision architecture is still not authorized.
The next step must freeze an exact Sentinel-2 L1C/Landsat C2 L1 catalog-pairing
contract and demonstrate at least 28 <=1-hour pairs from GHGSat alone. Until
that separate gate passes and the exact target manifest is committed, no
target imagery or model protocol may be opened. GHGSat nulls remain scene-head
supervision only; they cannot supervise an all-zero dense mask.
