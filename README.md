# Earth Remote Sensing Rapid Response (ERSRR)

ERSRR is a research system for locating methane-plume pixels from Sentinel-2 imagery and displaying predictions on a web map. The selected architecture is a **research-only, 144,433-parameter raw-band residual U-Net** with an explicit preprocessing/artifact contract. It is not yet accurate enough for operational methane detection or physical concentration estimation.

The original capstone was created by Eduardo Gonon, Kincaid Larson, Kevin Nguyen, and Hung-Nghi Vu for Dr. Cenek, advised by Dr. Nuxoll.

## Current architecture

- `EarthRemoteSensingRapidResponse/ersrr_core.py` is the single source of truth for canonical band order, optional physics features, normalization, model topology, masked loss/metrics, and artifact validation.
- `tools/run_research_baselines.py` evaluates grouped classical baselines on the legacy concentration cohort and the new EMIT V002 physical-mask cohort.
- `tools/run_unet_experiment.py` runs group-disjoint nested validation for raw and physics-feature compact residual U-Nets.
- `tools/train_compact_model.py` packages the selected research model as an ignored local artifact.
- `ERSRR_Website/server.py` loads that artifact lazily and never authenticates Earth Engine at import time.
- `EarthRemoteSensingRapidResponse/ERSRR_Model.py` is retained only as a historical baseline. Its regression workflow and old checkpoints must not be used for current claims.

The canonical Sentinel-2 input order is `B2, B3, B4, B11, B12`. Legacy curated tiles are L1C/TOA; the new V002 pilot is L2A/surface reflectance. These product levels are deliberately kept separate.

## Setup

Python 3.11 is recommended. TensorFlow 2.16+ ships Keras 3; native Windows GPU support is unavailable, so use WSL2 for GPU work or run CPU-only.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

On Windows PowerShell, activate with `.\.venv\Scripts\Activate.ps1` if the environment was created natively on Windows.

## Reproduce the research checks

```bash
python tools/ersrr.py status
python tools/ersrr.py audit
python tools/run_research_baselines.py --image-size 128 --pixels-per-scene 2048
CUDA_VISIBLE_DEVICES=-1 python tools/run_unet_experiment.py \
  --architectures raw_resunet physics_resunet --image-size 128 \
  --base-filters 8 --sampled-pixels 2048 --positive-weight 1
CUDA_VISIBLE_DEVICES=-1 python tools/train_compact_model.py \
  --architecture raw_resunet --image-size 128 --base-filters 8 \
  --threshold 300 --positive-weight 1
```

Experiment summaries are stored under `reports/experiments/`. The packaged `.keras` model and its `config.json` are generated under `EarthRemoteSensingRapidResponse/artifacts/` and intentionally ignored by Git.

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

The complete research narrative and quantitative results are in `reports/ERSRR_RESEARCH_REPORT.html`.
