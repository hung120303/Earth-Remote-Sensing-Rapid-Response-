# Domain-adaptive Prithvi v1 supersession

V1 was stopped before any pooled AP, recall, bootstrap, candidate-strength
selection, or protected-fold outcome. One endpoint encoder and scene
checkpoint had completed, but those ignored artifacts are not eligible for
reuse.

The independent protocol audit found two contract-level defects: seed two did
not independently enforce all written sensor/fold-recall/paired-site gates,
and the masked reconstruction loss did not exclude cloud/nodata pixels from
normalization and loss. External four-member score aggregation was also not
explicitly frozen.

V1 is therefore **superseded before outcome**, not accepted or rejected. V2
uses new seeds and checkpoints, observability-weighted reconstruction, exact
seed-two pilot gates, and a declared four-member mean-correction rule.
