# ERSRR MARS joint-model development result

Single-seed development experiment under the frozen group-disjoint protocol; not a final paper claim.

- Model: `ersrr_mars_joint_v1` / 2,677,803 parameters
- Best epoch: 1 / 6
- Checkpoint SHA-256: `38df62c9e450962df27642ad2ad8e5ccb3fd4e93f9ff3ac16a5e456e411d4ebc`
- Validation-selected presence thresholds: no-plume `0.4962158203125`, plume `0.520752`
- Strict-spatial scene recall: 0.015; specificity: 0.914; FPR: 0.086
- Group-bootstrap recall 95% CI: 0.000-0.071
- Segmentation pixel AP: 0.0355; IoU: 0.0517
- Selective weighted coverage: 0.513; accepted no-plume NPV: 0.996

## Decision

The joint model does not yet clear the promotion gate. Use validation-only error analysis and hard-negative mining before any backbone expansion; do not tune on the strict benchmark.

The quality head is trained on the predeclared >=99% observable label within the clear-S2 tranche. Non-clear/unobservable scenes are still required before treating learned quality as an operational abstention guarantee.
