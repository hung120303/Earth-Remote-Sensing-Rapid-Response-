# CloudSEN12+ spatial-pilot model manifest

Generated: 2026-07-16T15:57:34.557520+00:00.

- Loader-compatible clear-scene negatives: **492**.
- Auxiliary training: **367**.
- Development confirmation: **125**.
- Radiometry or availability exclusions: **20**.
- Rows with one or more missing wind components explicitly zero-filled: **103**.
- Published CloudSEN12+ test rows accessed: **0**.

Each output row binds the exact 12-band crop and an identically gridded zero cloud mask by hash. The zero mask is valid only because every frozen source row has exactly 40,000 published clear pixels and zero non-clear pixels.
