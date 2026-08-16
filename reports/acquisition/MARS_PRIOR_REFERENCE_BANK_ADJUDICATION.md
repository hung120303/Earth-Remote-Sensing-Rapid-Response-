# MARS prior-reference-bank alignment adjudication

Generated: 2026-08-16T16:26:44.638096+00:00.

This post-feasibility engineering adjudication preserves the original FAIL. It asks whether exact-grid filtering plus exact original-pair fallback makes a separate label-free inference diagnostic safe and sufficiently representative.

| Check | Observed | Required | Pass |
|---|---:|---:|:---:|
| Sentinel-2 reference coverage | 97.3668% | >= 95.00% | yes |
| Five-reference coverage | 91.7797% | >= 90.00% | yes |
| Grid-excluded among prior-candidate rows | 0.7629% | <= 1.00% | yes |
| Label/model access absent | n/a | required | yes |

**Decision:** PASS: authorize a separately frozen, label-free alternate-reference score extraction on folds 3/4 with original-pair fallback.

Passing authorizes only the separately preregistered alternate-reference score extraction on folds 3/4. It does not authorize training, threshold selection, external replay, official-test access, or a performance claim.
