# CloudSEN12-negative augmented MARS scene head

The candidate was selected on cross-fitted MARS folds 2/3/4 with only CloudSEN12 train negatives. Folds 0/1 and CloudSEN12 validation were evaluated only after selection.

- Selected model: `depth2_lr003`
- CloudSEN12 negative weight: 4.00
- Current/candidate logit blend: 0.025
- Inner AP delta: +0.00008
- Inner recall delta: +0.00000

| Confirmation | AP delta | Recall delta | AP lower 95% CI |
|---|---:|---:|---:|

CloudSEN12 validation raw-head p95/p99: 0.03145 / 0.05154.

Reject the CloudSEN12-negative branch before any sealed-negative or paper replay.
