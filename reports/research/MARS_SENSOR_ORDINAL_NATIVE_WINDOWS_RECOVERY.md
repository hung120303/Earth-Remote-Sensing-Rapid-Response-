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

Endpoint state and metadata become immutable before the corresponding held fold is opened. Immediately before each one allowed held evaluation, the runner atomically seals a read-only access-start receipt bound to the protocol and scientific digests, endpoint identity, ordered held records and sample IDs, fold, and prior access ledger. A start receipt without a validated immutable part is an explicit incomplete-held-access failure: restart refuses automatic reevaluation rather than risk spending the fold twice. After evaluation, the part is atomically published read-only and a completion receipt is sealed with the start-receipt and part hashes. Restart validates the receipt/part pair and reuses it without evaluation.

Both completed parts must validate and merge in exact fold/sample order to the same arrays, identities, bindings, and receipt/part hashes before the final candidate is sealed or accepted. Comparator preflight still streams opaque bytes only for frozen SHA-256 verification; no comparator container is semantically decoded before candidate immutability. After that boundary, comparator reconstruction and metric computation may safely repeat because the candidate is fixed. Endpoint states, exact JSON, and exact Markdown are then seal-or-validated in order, so a crash after any write resumes without overwrite, endpoint retraining, or held reevaluation. `verify_protocol` admits only a consistent immutable recovery prefix bound to the current protocol and science; mutable, inconsistent, partial, wrong-protocol, wrong-science, and unrelated outputs remain hard failures.

## Pre-smoke native-Windows portability amendment

The exact native-Windows runtime imports passed at commit `73385ee76871a7f9ce4201f3bde69a56347f9ce1`: Windows, PyTorch `2.11.0+cu128`, CUDA `12.8`, NVIDIA GeForce RTX 5070, NumPy `2.4.4`, rasterio `1.4.4`, scikit-learn `1.9.0`, and SciPy `1.17.1`. No runtime smoke and no held data were run. The 42-test native suite had 17 failures from two portability defects only.

First, a clean Windows checkout converted protocol-bound text to CRLF, changing byte hashes while leaving scientific content unchanged. Repository attributes now force LF for protocol-relevant text, including Python, JSON/JSONL, Markdown, TOML, YAML, shell, and ordinary text/tabular files. Binary model, NumPy, Torch, TIFF, image, archive, and related artifacts are explicitly exempt from normalization. A clean-checkout audit with Windows `core.autocrlf=true` must preserve the exact trainer, model, and protocol bytes and their frozen identities.

Second, Windows rejects `fsync` on a descriptor reopened read-only. Every atomic NumPy, Torch, text, durable endpoint, recovery generation, receipt, and pointer path now writes directly through a writable handle, flushes and fsyncs that same handle, then applies read-only mode where required, atomically replaces the destination, and fsyncs the directory only where supported. Atomicity, immutable one-shot outputs, two-generation recovery, and refusal to repeat held access are unchanged.

This is a pre-smoke infrastructure-only amendment. The scientific-settings digest remains `25773453cdf37062260d8d76bca01e5d46dc17bcc513ce372708a02806991dca`; data, folds, split, seed, architecture, precision, batches, schedules, losses, ranking, thresholds, comparators, bootstrap, and all seven gates remain unchanged. The amendment does not authorize creation or modification of the Windows environment, native runtime smoke, or held-data execution.

## First native checkpoint smoke failure

The single authorized smoke at native commit `676fb75871d4aed2c4a3e94b559d4aede15f7c59` failed before JSON output. `RecoveryStore` loaded the whole checkpoint with CUDA `map_location`, which relocated `rng_state["torch_cpu"]` to a CUDA ByteTensor. `restore_rng_state` passed it to `torch.set_rng_state`, which raised `TypeError: RNG state must be a torch.ByteTensor`. This is a serializer/device-placement infrastructure defect, not scientific rejection.

The run emitted `expandable_segments not supported on this platform`. Native Windows accepted `PYTORCH_ALLOC_CONF=expandable_segments:True`, but that allocator option was unsupported and inert; it did not cause the failure. No inner-validation outcome, held data or outcome, semantic comparator value, fold 0/1/2, external evidence, protected evidence, or official evidence was opened. No JSON output, persistent smoke checkpoint, held part, candidate prediction, endpoint state, experiment JSON/Markdown, gate, or promotion decision exists. The compact exact-traceback receipt is `reports/research/mars_sensor_ordinal_native_checkpoint_smoke_failure.json`.

## Pre-smoke serializer amendment and authorized next step

Recovery generations now load completely on CPU. `restore_rng_state` validates that Torch CPU and per-device CUDA RNG entries are one-dimensional `uint8` tensors, converts device-relocated entries back to contiguous CPU tensors without changing their bytes, and passes only CPU ByteTensors to `torch.set_rng_state` and `torch.cuda.set_rng_state_all`. Live and best model states therefore load safely from CPU, while `model.load_state_dict` copies live/best tensors to the model device and `Optimizer.load_state_dict` applies PyTorch's parameter-device policy to AdamW state. Focused continuation tests require exact next-step model and optimizer equality after a CPU-loaded CUDA restore.

After Codex review, exactly one corrected native-Windows fitting-only `--checkpoint-roundtrip-smoke` was authorized. It uses fold-3 fitting evidence only, the production batch-16 path, and a checkpoint round trip. It must not compute inner-validation outcomes or open held data, semantic comparator values, folds 0/1/2, external evidence, protected evidence, or official evidence. No held-fold run was authorized. That invocation was later consumed by the host-wrapper failure recorded below.

## Corrected invocation stopped by the host wrapper

The corrected invocation at clean native commit `a8dce79cbc6a773dcd007944b175ada9a76c5a98` is recorded only as a host-wrapper infrastructure failure. Python emitted only the expected inert `expandable_segments not supported on this platform` warning before Windows PowerShell with `ErrorActionPreference Stop` promoted native stderr to terminating `NativeCommandError`. The wrapper reported exit code 1, but the actual Python process exit code was not observed. No JSON output or persistent smoke artifact exists, and no inner-validation outcome, held data or outcome, semantic comparator, fold 0/1/2, external, protected, or official evidence was opened. The exact captured output is preserved in `reports/research/mars_sensor_ordinal_corrected_smoke_host_wrapper_failure.json`.

Exactly one wrapper-corrected checkpoint-roundtrip smoke is authorized. The retry must use native `cmd.exe` or PowerShell without `ErrorActionPreference Stop`, redirect stdout and stderr separately, and judge success only by the actual Python process exit code. The exact Python command, environment, trainer, model, data, batch, seed, and scientific settings remain unchanged. No held-fold run is authorized.

## Wrapper-corrected smoke allocator failure and bounded amendment

The one wrapper-corrected smoke at clean native commit `26ef0523be3ff5dedca42c7298250e601e7ab797` used native `cmd.exe`, separate stdout/stderr files, and the actual Python exit code. Python exited 1 with empty stdout. The exact batch-16 pixel backward completed, but the implicit CUDA foreach implementation of `clip_grad_norm_` requested TensorList workspace at peak backward memory and raised `cudaErrorMemoryAllocation` before the first optimizer step or JSON output. This is an allocator infrastructure failure, not scientific rejection. The exact output and absence attestations are preserved in `reports/research/mars_sensor_ordinal_native_bounded_allocator_failure.json`.

The bounded allocator amendment retains the frozen global L2 gradient clipping threshold of 2.0 for both pixel and scene updates while passing `foreach=False`. The Codex-reviewed correction also passes `foreach=False` to every pixel and scene AdamW construction, including recovery and smoke reconstruction, so neither clipping nor the immediately following optimizer update can select a CUDA TensorList path at peak memory. Model, losses, optimizer hyperparameters, batch and crop sizes, precision, seed, schedule, data, folds, comparators, thresholds, bootstrap, all seven gates, and scientific digest remain unchanged. No runtime smoke or held data was run by either amendment, and no further execution is authorized before independent review.
