# Earth-Remote-Sensing-Rapid-Response-
The Earth Remote Sensing Rapid Response is a Capstone Project made by a team of 4 UP students (Eduardo Gonon, Kincaid Larson, Kevin Nguyen, Hung-Nghi Vu) for a client (Dr. Cenek), and is advised under Faculty Advisor Dr. Nuxoll. The purpose of the project is to predict methane production using machine learning represented by an interactive heat map

To test the program, run ERSRR_Model.py in your IDE or through CLI.
Requires Python 3.11 or higher.

## Python setup

The project uses **Keras 3** and **TensorFlow 2.16+**.

### Create and activate a virtual environment

From the repository root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate          # Linux / WSL
# OR on Windows PowerShell:
# .\.venv\Scripts\Activate.ps1
```

### Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> **Note:** TensorFlow dropped native Windows GPU support after 2.10.
> For GPU training, run inside WSL2 where CUDA works normally.
> CPU-only inference and prediction work on both Windows and WSL.

### Run the model

Train the model (default mode):

```bash
python EarthRemoteSensingRapidResponse/ERSRR_Model.py
```

To switch to prediction mode, change `MODEL_SETTING` in `ERSRR_Model.py` from `"train"` to `"predict"`, then run again.

Hyperparameters can be tuned via CLI flags:

```bash
python EarthRemoteSensingRapidResponse/ERSRR_Model.py --lr 0.001 --ep 50 --bs 8
```

### Google Earth Engine (for data collection)

Authenticate once inside the environment:

```bash
earthengine authenticate
```

Then run the data collection scripts in `EarthRemoteSensingRapidResponse/Data Collection/`.

## Agent-friendly data CLI

A small local CLI is available for repeatable dataset checks and data acquisition handoff steps:

```bash
python tools/ersrr.py status
python tools/ersrr.py audit
python tools/ersrr.py guide --output docs/DATA_ACQUISITION.md
python tools/ersrr.py init-batch emit-v002-YYYY-MM
```

Useful outputs:

- `tools/ersrr.py status` reports dataset counts, Earth Engine credential presence, and repo hygiene checks.
- `tools/ersrr.py audit` writes `reports/dataset_audit/pairings_manifest.csv`, `summary.json`, and `SUMMARY.md`.
- `tools/ersrr.py guide` prints the exact EMIT/Sentinel-2 data source links and manual download workflow.
- `tools/ersrr.py init-batch <name>` creates local folders for a new EMIT plume batch.

The audit is read-only with respect to source imagery.

## Old checkpoints

Existing `.weights.h5` files in `tmp/` were saved with Keras 2 and are **not compatible** with Keras 3. You will need to retrain the model after upgrading. A conversion helper is available:

```bash
python convert_checkpoint.py
```
