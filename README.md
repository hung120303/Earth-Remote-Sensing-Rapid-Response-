# Earth Remote Sensing Rapid Response (ERSRR)

ERSRR is a research system for detecting and segmenting methane plumes in multi-temporal Sentinel-2 imagery. The current publication candidate is **v5.1**, a 9,358,256-parameter shared tri-temporal U-Net using T, T-90, and T-365 imagery, two MBMP physics maps, dense segmentation, and a fixed small context head. It is a research model, not an operational detector or physical concentration/flux estimator.

On the sealed 20,789-crop MethaneS2CM location test, v5.1 achieved scene AP 0.8180, AUROC 0.8276, recall 0.3778, false-positive rate 0.0607, pixel AP 0.2083, Dice 0.3125, and IoU 0.1852. It substantially exceeded released MARS-S2L's zero-shot ranking, recall, and dense overlap on the same test, with paired 25 km bootstrap support, but MARS-S2L had lower FPR at an almost-zero 0.0052 recall. Across-the-board superiority is therefore **not** established, and no threshold may be retuned from the test result. See the [research dossier](reports/ERSRR_RESEARCH_REPORT.html), [research ledger](docs/RESEARCH_LEDGER.md), and [paper outline](docs/PAPER_OUTLINE.md).

The retired v3 campaign remains an important negative result: on its separate geographically isolated MARS cohort, ERSRR reduced mean FPR from 9.48% for released MARS-S2L to 3.67%, but mean recall fell from 64.18% to 31.94%. V4.3 later improved strict-cohort AP/AUROC/FPR point estimates but still lost recall and overlap. These studies are preserved rather than overwritten.

The original capstone was created by Eduardo Gonon, Kincaid Larson, Kevin Nguyen, and Hung-Nghi Vu for Dr. Cenek, advised by Dr. Nuxoll.

## Publication architecture and evidence

- `EarthRemoteSensingRapidResponse/methanes2cm_v5_model.py` defines the shared tri-temporal v5/v5.1 architecture.
- `EarthRemoteSensingRapidResponse/methanes2cm_adapter.py` enforces the 12-page TIFF, band, reflectance, mask, and comparator contracts.
- `tools/train_methanes2cm_v5.py` implements the frozen fitting/development and three-seed campaign.
- `tools/aggregate_methanes2cm_v5_1.py` freezes ensemble calibration, thresholds, dense averaging, spatial cross-fitting, and uncertainty.
- `tools/acquire_methanes2cm_v5_test.py` and `tools/evaluate_methanes2cm_v5_1_test.py` implement the precommitted one-shot location-test boundary.
- `tools/analyze_methanes2cm_v5_1_test_posthoc.py` applies only already-frozen thresholds and cannot select a test operating rule.
- `tools/evaluate_mars_v4_3_strict.py` preserves the paired v4.3 versus released MARS-S2L strict comparison.
- `tools/build_research_report.py` regenerates the self-contained HTML dossier from committed machine-readable evidence.

V5.1 inputs are two MBMP maps plus six Sentinel-2 L2A bands for each of T, T-90, and T-365. MethaneS2CM lacks wind and per-pixel cloud masks; frozen L1C comparators therefore use documented wind imputation and an unavailable-cloud zero channel. L1C/MARS, L2A/MethaneS2CM, and EMIT studies remain explicitly separated.

## Setup

Python 3.11 is recommended. TensorFlow 2.16+ ships Keras 3; native Windows GPU support is unavailable, so use WSL2 for GPU work or run CPU-only.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

On Windows PowerShell, activate with `.\.venv\Scripts\Activate.ps1` if the environment was created natively on Windows.

## Reproduce the frozen report and checks

```bash
python tools/ersrr.py status
python tools/ersrr.py audit
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python tools/build_research_report.py
```

The one-shot test must not be rerun as a tuning loop. Its immutable result is `reports/experiments/methanes2cm_v5_1_location_test.json`; the post-hoc diagnostic verifies frozen report/cache identities before rewriting its explicitly exploratory report. Checkpoints, HDF5 packs, imagery, and prediction caches are intentionally ignored by Git.

## Acquire an EMIT V002 pilot pair

The collector uses public NASA CMR V002 plume geometry and public Element 84 Sentinel-2 L2A COGs. It writes bracketing 256 x 256 image stacks, physical plume masks, SHA-256 hashes, and a provenance manifest into ignored acquisition directories.

```bash
python tools/acquire_v002_pilot.py \
  EMIT_L2B_CH4PLM_002_20250922T204933_003374 \
  --batch emit-v002-2026-07 --temporal-mode bracketing
```

The protected EMIT concentration COG still requires an Earthdata login. The public polygon is a segmentation label, not a concentration raster, so the collector never manufactures a legacy six-band regression tile. See `docs/DATA_ACQUISITION.md` for the full contract.

Create the tracked, token-free batch integrity record with:

```bash
python tools/summarize_v002_batch.py
```

## Run the web API

Create the local research artifact first, then configure Earth Engine credentials outside the repository and run:

```bash
python ERSRR_Website/server.py
```

`GET /health` reports artifact and Earth Engine readiness. Inference uses JSON `POST /sentinel`; prediction routes return an explicit `503` when either dependency is unavailable.

## Data and Git policy

- The curated legacy dataset remains tracked for reproducibility.
- Raw acquisition batches, generated predictions, temporary rasters, model artifacts, audit output, and frontend dependencies are ignored.
- Do not commit credentials, Earthdata cookies, protected download URLs containing tokens, or bulk imagery.
- Add a manifest and dataset contract before promoting any new batch into a curated split.

The complete research narrative and quantitative results are generated at `reports/ERSRR_RESEARCH_REPORT.html`. Human-readable paper planning stays in `docs/RESEARCH_LEDGER.md` and `docs/PAPER_OUTLINE.md`; those notes distinguish preregistered decisions, frozen primary results, and post-hoc hypotheses.
