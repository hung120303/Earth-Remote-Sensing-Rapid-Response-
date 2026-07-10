# ERSRR research map

The Vite/OpenLayers frontend displays Sentinel-2 imagery and the current research-only plume-probability mask. Inference is intentionally deferred until zoom level 14 so a country-scale map movement cannot trigger an oversized Earth Engine download.

## API prerequisites

From the repository root, first create the ignored local artifact:

```bash
CUDA_VISIBLE_DEVICES=-1 python tools/train_compact_model.py
```

Provide non-interactive Google Earth Engine credentials outside the repository, then run:

```bash
python ERSRR_Website/server.py
```

Useful environment variables:

- `ERSRR_ARTIFACT_DIR`: artifact directory (defaults to `EarthRemoteSensingRapidResponse/artifacts/compact_resunet_v1`)
- `ERSRR_EE_PROJECT`: Earth Engine project
- `ERSRR_DEFAULT_START` / `ERSRR_DEFAULT_END`: default date window
- `ERSRR_MAX_CLOUD`: maximum scene cloud percentage
- `ERSRR_MAX_ROI_SPAN_DEGREES`: maximum longitude or latitude span per request (default `0.25`)
- `ERSRR_CORS_ORIGINS`: comma-separated allowed frontend origins (default `http://localhost:5173`)

`GET /health` reports artifact and Earth Engine readiness. Inference uses a JSON `POST /sentinel`, which is restricted by the configured CORS allowlist; the server never opens an interactive authentication flow.

## Frontend

```bash
cd ERSRR_Website
npm ci
npm start
```

Set `VITE_API_BASE_URL` if the Flask API is not at `http://localhost:5000`. Build with `npm run build`; generated dependencies and `dist/` are ignored by Git.
