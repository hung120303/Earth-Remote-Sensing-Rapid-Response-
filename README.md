# Earth Remote Sensing Rapid Response (ERSRR)

ERSRR is a research system for detecting and segmenting methane plumes in paired Sentinel-2 imagery. The current publication candidate is a **14,268,915-parameter, 16-channel, three-head U-Net** with scene presence, pixel segmentation, and quality/abstention outputs. It is a research model, not an operational detector or physical concentration/flux estimator.

The frozen v3 result is a useful negative result: on the same geographically isolated 4,401-scene cohort, ERSRR reduced mean false-positive rate from 9.48% for released MARS-S2L to 3.67%, but mean recall fell from 64.18% to 31.94%. The preregistered promotion gate failed, so v3 must not be retuned from strict-test behavior. See the [research dossier](reports/ERSRR_RESEARCH_REPORT.html), [research ledger](docs/RESEARCH_LEDGER.md), and [paper outline](docs/PAPER_OUTLINE.md).

The original capstone was created by Eduardo Gonon, Kincaid Larson, Kevin Nguyen, and Hung-Nghi Vu for Dr. Cenek, advised by Dr. Nuxoll.

## Publication architecture and evidence

- `EarthRemoteSensingRapidResponse/mars_v3_model.py` defines the 16-channel U-Net and its three heads.
- `EarthRemoteSensingRapidResponse/mars_v3_proposals.py` defines deterministic connected-plume proposals and descriptors.
- `tools/train_mars_v3.py` and `tools/train_mars_v3_proposals.py` implement the frozen five-seed development protocol.
- `tools/evaluate_mars_v3.py` and `tools/evaluate_released_marss2l.py` evaluate ERSRR and the released baseline on the same strict cohort.
- `tools/aggregate_mars_v3_strict.py` performs the paired 2,000-replicate group-and-seed bootstrap.
- `tools/analyze_mars_v3_strict_posthoc.py` creates explicitly exploratory failure strata and a deterministic error atlas.
- `tools/build_research_report.py` regenerates the self-contained HTML dossier from committed machine-readable evidence.

The v3 inputs are release-compatible MBMP, six target and six reference Sentinel-2 L1C bands, ERA5-Land u/v wind, and a CloudSEN12 observability mask. Legacy L1C/TOA and EMIT V002 L2A/surface-reflectance experiments remain explicitly separated.

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
.venv/bin/python tools/aggregate_mars_v3_strict.py
.venv/bin/python tools/analyze_mars_v3_strict_posthoc.py
.venv/bin/python tools/build_research_report.py
```

The aggregation and diagnostic commands verify frozen prediction-cache identities before rewriting reports. Experiment summaries are stored under `reports/experiments/`; checkpoints, proposal classifiers, and prediction caches under `EarthRemoteSensingRapidResponse/artifacts/` are intentionally ignored by Git.

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
