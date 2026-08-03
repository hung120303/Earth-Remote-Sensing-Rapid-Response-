# Stanford Evanston Independent-Site One-Shot Evaluation

Status: **completed independent-site one-shot; thresholds unchanged**

## Cohort

- Frozen events: 9
- Primary negatives: 0
- Primary positives (>=1,000 kg/h): 1
- Challenge events (0-1,000 kg/h): 8
- Location values in source: Evanston

## Models

### released_mars_v3

- AP: None
- AUROC (descriptive): None
- Recall: 1.0 (exact 95%: [0.025000000000000022, 1.0])
- FPR: None (exact 95%: [None, None])
- Precision: 1.0 (exact 95%: [0.025000000000000022, 1.0])
- Confusion: {'tp': 1, 'tn': 0, 'fp': 0, 'fn': 0}
- Challenge detection: {'rows': 8, 'detected': 2, 'detection_fraction': 0.25, 'exact_clopper_pearson_95': [0.031854026249944246, 0.6508557944128242]}

### gaussian_dofa

- AP: None
- AUROC (descriptive): None
- Recall: 0.0 (exact 95%: [0.0, 0.975])
- FPR: None (exact 95%: [None, None])
- Precision: None (exact 95%: [None, None])
- Confusion: {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 1}
- Challenge detection: {'rows': 8, 'detected': 1, 'detection_fraction': 0.125, 'exact_clopper_pearson_95': [0.003159723531251909, 0.5265096708752065]}

### spatial_prithvi_posttest

- AP: None
- AUROC (descriptive): None
- Recall: 0.0 (exact 95%: [0.0, 0.975])
- FPR: None (exact 95%: [None, None])
- Precision: None (exact 95%: [None, None])
- Confusion: {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 1}
- Challenge detection: {'rows': 8, 'detected': 0, 'detection_fraction': 0.0, 'exact_clopper_pearson_95': [0.0, 0.3694166475528192]}

## Interpretation Boundary

This is a post-test, nine-event evaluation at one independent geographic site. Thresholds were unchanged. It is separate from the Casa Grande one-shot result and cannot by itself establish broad geographic generalization or superiority.

## License and Attribution

Stanford source dataset: Reuland et al., Large-Scale Controlled Methane Releases for Satellite-Based Detection and Emission Quantification of Point-Sources, Stanford Digital Repository, CC BY 4.0, DOI 10.25740/qh001qt3946.
