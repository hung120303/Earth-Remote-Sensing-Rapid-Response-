# Gaussian scene-aligned exact replay rejected

- Frozen protocol: `a981d848681b6a62fc228ea1cd6a9531a325cdd391b0976e81dd1cd0bfef49a4`
- Diagnostic runtime: **4,866.9 seconds**
- Strength 0.05 recall reproduced within **1e-9**.
- Pooled and sensor AP did **not** reproduce within **1e-9**.
- No raw cache, endpoint state, or success receipt was written.
- Fold 2 and the official test were not accessed.

The fixed-seed CUDA/BF16/worker execution is not bitwise reproducible. The
exact protocol remains rejected; it is not relaxed after observing this
failure. A separately frozen bounded stochastic replicate will measure and
gate the variation explicitly before any Gaussian strength can enter the
protected ensemble.
