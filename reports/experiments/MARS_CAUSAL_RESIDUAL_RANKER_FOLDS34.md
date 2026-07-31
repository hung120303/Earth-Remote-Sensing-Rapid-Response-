# Baseline-preserving causal residual scene ranker

The zero-initialized model learns a correction around the frozen spatial-Prithvi logit and uses fit-fold-only same-background interventions.

No residual strength passed every preregistered gate. The smallest correction was already harmful:

| Strength | Whole AP delta | Whole recall delta | Whole paired-site 95% interval | Rare-site AP delta | Rare-site recall delta | Rare-site 95% interval |
|---:|---:|---:|---:|---:|---:|---:|
| 0.10 | -0.000222 | -0.001312 | [-0.000685, +0.000033] | -0.001485 | 0.000000 | [-0.004714, +0.002408] |
| 0.25 | -0.000636 | -0.003281 | [-0.001767, +0.000005] | -0.003154 | 0.000000 | [-0.011335, +0.007920] |
| 0.50 | -0.001622 | -0.003937 | [-0.004061, -0.000407] | -0.009761 | +0.008475 | [-0.023481, +0.007152] |
| 1.00 | -0.004444 | -0.005906 | [-0.009807, -0.001842] | -0.026820 | +0.008475 | [-0.050748, +0.002153] |

At strength 0.10, AP regressed on both held folds and both sensor strata. Increasing residual strength worsened both whole- and rare-site ranking; by strength 0.50 the whole-view AP interval was strictly negative.

Reject baseline-preserving causal residual ranking before fold-0/1/2 or external access.
