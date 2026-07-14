# MARS paper successor initialization audit

The untrained successor exactly reproduces the independently implemented released MARS-S2L logits for both Sentinel-2 and Landsat sensor identities.

- Checkpoint SHA-256: `be634fb9e24dc4877f44c1ff9f69972e6f0453e30d70c0dc03677876340ef246`
- Maximum absolute logit delta: `0.0`
- Maximum absolute correction: `0.0`
- Total parameters: 14,865,638
- Initially trainable correction parameters: 1,278,053

Any subsequent gain or regression is therefore attributable to the successor training and frozen decision rule, not a weaker random initialization.
