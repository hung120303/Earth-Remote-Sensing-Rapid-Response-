# MARS sensor-aware ordinal native-Windows recovery freeze

Status: frozen pre-outcome infrastructure amendment; stop for Codex review before execution.

## Why the retry moves to native Windows

Commit `908f63133bed1e31ade56c38473beebb22dfdaf2` records the failed WSL attempt. The WSL run completed fitting epochs 1 through 9 for the endpoint held on fold 3, then failed in epoch 10 with repeated WSL DXG allocation-bridge errors and `cudaErrorUnknown`. It never produced a held-fold prediction, decoded comparator value, gate decision, result, endpoint state, or recovery checkpoint. The provisional epoch-6 best rank and all incomplete-run metrics are non-authoritative and must not be imported or used for tuning.

The next scientific attempt therefore starts at epoch 1 on native Windows CUDA. Native Windows bypasses both the WSL DXG bridge implicated by the failure and WSL access to the project/data through the `p9` filesystem path. No scientific setting changes: data, folds, seed, architecture, precision, batch sizes, schedules, losses, checkpoint ranking, thresholds, comparators, bootstrap, and all seven gates remain frozen.

## Exact required runtime

- OS/runtime: native Windows (WSL is rejected)
- GPU: NVIDIA GeForce RTX 5070
- NVIDIA driver: `595.79`
- PyTorch: `2.11.0+cu128`
- NumPy: `2.4.4`
- rasterio: `1.4.4`
- scikit-learn: `1.9.0`
- SciPy: `1.17.1`
- `PYTORCH_ALLOC_CONF=expandable_segments:True`
- `CUDA_MODULE_LOADING=LAZY`

The runtime signature is part of every recoverable generation and must match exactly before resume. The Windows environment is intentionally not created by this amendment.

## Recovery boundary and contents

The only recoverable epoch boundary is after all updates for the epoch, inner validation, best-rank/best-epoch update, and complete history append, and before the next epoch begins. A crash inside an epoch discards that partial epoch; the last complete generation is restored and the partial epoch is replayed.

Each ignored recovery generation binds and persists the live model separately from `best_state`; both AdamW optimizer states including parameter groups and current learning rates; best rank and epoch; complete history; cutpoints; completed and next epoch; held and fit fold; the independent `SiteBalancedBatcher` generator state; Python, legacy NumPy, Torch CPU, and all CUDA RNG states; protocol identity and registered hashes; a scientific-settings digest; trainer/model/dependency/manifest/fold and ordered record/split hashes; seed and schedule digest; exact runtime signature; and the access ledger. Construction and state loading occur first; global RNG restoration occurs last.

Generations are written under ignored `.research/` storage with unique names and atomically published metadata. Resume accepts only a fully validated generation. A corrupt newest generation falls back to the prior valid generation. Identity, runtime, or forbidden-access mismatch refuses resume.

## Outer one-shot durability

Endpoint state and metadata become immutable before the corresponding held fold is opened. The one allowed held evaluation is immediately written as an atomic, read-only per-endpoint candidate part. A restart validates and reuses an existing immutable part rather than evaluating that held fold again. This prevents a crash during the second endpoint from spending the first held fold twice.

Comparator preflight may stream opaque bytes only to verify frozen SHA-256 values and records `comparator_integrity_bytes_hashed=true` while `comparator_values_decoded=false`. Before the final candidate is immutable, comparator files must not be passed to `np.load` or `torch.load`, and no comparator score inspection, metric computation, or semantic outcome access is permitted. Comparator values may be decoded only after both held parts are immutable, validated, merged, and published as the final immutable candidate. Existing one-shot refusal remains in force for scientific outputs.

## Authorized next verification step

After Codex reviews the commits, the first executable step is the native-Windows fitting-only recovery runtime smoke. It uses fold-3 fitting evidence only, the production batch-16 path, and a checkpoint round trip. It does not compute inner-validation outcomes and does not open held data, semantic comparator values, folds 0/1/2, external evidence, or official evidence. Its receipt must attest exact sample/group identities and equality of the next training step across uninterrupted and restored paths.

This amendment does not authorize creation of the Windows environment, the native CUDA smoke, or any held-fold run.