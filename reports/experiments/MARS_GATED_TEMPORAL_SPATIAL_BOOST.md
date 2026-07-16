# Gated temporal-spatial boost

- Selected minimum history: **20**; top-k: **1**; cutoff: **0.10**; weight: **1.00**.
- Confirmation target AP/recall deltas: **-0.02486 / -0.02098**.
- Confirmation whole AP/recall deltas: **-0.01132 / -0.00872**.
- All promotion gates pass: **false**.

The temporal rule is mathematically one-sided: candidate scores never fall below the validated spatial score.
