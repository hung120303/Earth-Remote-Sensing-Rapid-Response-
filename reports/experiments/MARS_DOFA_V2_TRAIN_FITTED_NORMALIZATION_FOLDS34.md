# Train-fitted DOFA-v2 normalization confirmation - folds 3/4

All feature normalization statistics are fit on the training fold and applied unchanged to the held fold. The protected gate, DOFA weight, projection seeds, and downstream promotion gates are fixed.

- Selected mode: `global_train_fitted`
- AP delta vs current: +0.001412
- Recall delta at FPR 0.0713: +0.000000
- Paired-site AP 95% interval: [+0.000526, +0.002462]
- Operating confusion counts preserved: yes

| Normalization | AP delta | Recall delta | AP CI lower | Promoted |
|---|---:|---:|---:|:---:|
| `global_train_fitted` | +0.001412 | +0.000000 | +0.000526 | yes |
| `sensor_train_fitted` | +0.001225 | +0.000000 | +0.000347 | yes |

Freeze the selected train-fitted DOFA-v2 fusion for one-shot fold-2 extraction.
