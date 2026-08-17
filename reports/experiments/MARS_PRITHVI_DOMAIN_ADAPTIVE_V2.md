# Spectral-temporal domain-adaptive Prithvi

This experiment continued Prithvi masked-autoencoder pretraining on only the opposite MARS development fold and a fixed external auxiliary cohort. It never loaded MARS folds 0/1/2 or the official test.

- Decision: **REJECT**
- Pilot passed: no
- Pilot strength: 0.250
- Pilot AP delta: -0.002024
- Pilot recall delta at matched FPR: -0.000656
- Pilot paired-site AP interval: [-0.004297, +0.000679]

The pilot failed its preregistered development gate; replication and all external/official evaluation were skipped.

## Frozen strength sweep

| Residual strength | Pooled AP | AP delta | Recall delta at matched FPR | Paired-site AP 95% interval |
| ---: | ---: | ---: | ---: | ---: |
| 0.25 | 0.904501 | -0.002024 | -0.000656 | [-0.004297, +0.000679] |
| 0.50 | 0.901781 | -0.004743 | -0.002625 | [-0.009502, +0.000629] |
| 1.00 | 0.893735 | -0.012790 | -0.003937 | [-0.024466, -0.001631] |

The harm increases monotonically with residual strength. The smallest frozen
strength was therefore the least harmful candidate, not a promising optimum.
There is no basis for an unregistered weaker-strength rescue.

## Discordance and sensor result

At strength 0.25, fold 3 AP changed -0.001481 while fold 4 changed +0.000655.
Sentinel-2 AP changed -0.003565 in the pooled result; Landsat changed only
-0.000097. Both the every-fold and every-sensor AP gates failed. This pattern
is consistent with a correction that is not geographically stable or
complementary to the frozen champion, especially for the shifted Sentinel-2
wide-NIR/L1C input domain.

This experiment does show that the corrected observability-weighted masked
objective trained stably: the two endpoints' total pretraining losses fell
from 3.999/4.139 at step 100 to 0.820/0.845 at step 1,200. That is an
optimization result, not evidence of improved methane ranking.

## Scope and publication constraint

Only MARS-S2L development folds 3/4 and the frozen MARS-disjoint auxiliary
cohort were available to the trainer. Folds 0/1/2 and the official test were
not loaded. Because the pilot failed, seed two, external transfer, and official
confirmation were not authorized. The result falsifies this complete
ExPLoRA-inspired/MAESTRO-inspired Prithvi residual system; it does not isolate
either published method causally and is not a reproduction of either paper.

Provenance: git `5bb79d4a56349b8be2db92fe5e4e6817c315844c`, protocol SHA-256
`86c350b2b61732c643dc4207b74cd50baf5cb5f4037ec0d462f6cb5b3d5fa934`,
NVIDIA GeForce RTX 5070, PyTorch 2.11.0+cu128. Checkpoints remain ignored;
the compact JSON contains the complete 10,000-replicate receipts.
