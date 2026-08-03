# Stanford Controlled-Release Cross-Site Evidence Synthesis

## Purpose

This note combines two separately frozen, one-shot Sentinel-2 L1C controlled-release evaluations of released MARS-S2L v3 and the ERSRR successor candidates. It does not pool the sites, retune any threshold, or convert sub-threshold releases into negatives.

- **Casa Grande, Arizona:** temporally new but same known physical source site; strong metered-zero false-positive stress test.
- **Evanston, Wyoming (Rawhide Leasing Yard):** independent physical site; sensitivity/challenge stress test, but not an independent negative test because its nine frozen summaries contain no zero-release event.

The compact JSON reports are authoritative. This synthesis states the cross-site interpretation and gate status.

## Outcome-blind execution

For Evanston, the nine target dates, exact Sentinel-2 L1C products, prior-only same-MGRS references, six-band 256×256 crops, scorer/model hashes, thresholds, output paths, public outcome files, outcome field, strata, metrics, uncertainty, and evaluator hash were frozen before their corresponding access stages.

The immutable nine-row score bundle was validated before any official summary CSV was opened. The outcome evaluator then downloaded exactly nine protocol-listed `*_summary.csv` files, verified each against Stanford repository size/SHA-1/MD5 metadata, required one exact-schema row per `release_ID`, and joined once. No threshold was selected from either site's outcomes.

## Frozen cohorts

| Site | Frozen rows | Primary negatives | Primary positives (≥1,000 kg/h) | Challenge releases (0–1,000 kg/h) | Interpretation |
|---|---:|---:|---:|---:|---|
| Casa Grande | 169 | 86 | 8 | 75 | One-site temporal operating-point/FPR stress test |
| Evanston | 9 | 0 | 1 | 8 | Independent-site sensitivity/challenge stress test only |

Evanston therefore does **not** fill the independent-negative evidence gap. Its AP, AUROC, and FPR are undefined under the frozen primary view because that view contains no negative row.

## Casa Grande one-shot result

Primary view: 94 rows (86 metered-zero negatives and 8 releases ≥1,000 kg/h), 87 UTC-date blocks.

| Frozen model/rule | AP | AUROC | Recall | FPR | Confusion (TP/TN/FP/FN) |
|---|---:|---:|---:|---:|---:|
| Released MARS-S2L v3 | 0.09559 | 0.47238 | 0.125 (1/8) | 0.15116 (13/86) | 1 / 73 / 13 / 7 |
| Gaussian+DOFA | 0.08724 | 0.46221 | 0.000 (0/8) | 0.08140 (7/86) | 0 / 79 / 7 / 8 |
| Spatial-Prithvi post-test | 0.08730 | 0.46366 | 0.000 (0/8) | 0.01163 (1/86) | 0 / 85 / 1 / 8 |

Gaussian+DOFA reduced FPR relative to released MARS by 0.06977; its 10,000-replicate paired-date interval was [-0.12791, -0.02299]. Spatial-Prithvi reduced FPR by 0.13953, interval [-0.21795, -0.07059]. Both candidates nevertheless lost the released model's only high-rate true positive. Their AP-delta intervals crossed zero and recall-delta intervals were [-0.5, 0.0]. Both preregistered superiority gates failed.

On 75 sub-threshold challenge releases, released MARS detected 8 (10.67%), Gaussian+DOFA detected 2 (2.67%), and Spatial-Prithvi detected 0.

## Evanston one-shot result

Primary view: one release ≥1,000 kg/h and no negatives.

| Frozen model/rule | High-rate detection | Recall | Exact Clopper–Pearson 95% | Challenge detections | Exact challenge 95% |
|---|---:|---:|---:|---:|---:|
| Released MARS-S2L v3 | 1/1 | 1.0 | [0.0250, 1.0] | 2/8 (0.25) | [0.03185, 0.65086] |
| Gaussian+DOFA | 0/1 | 0.0 | [0.0, 0.975] | 1/8 (0.125) | [0.00316, 0.52651] |
| Spatial-Prithvi post-test | 0/1 | 0.0 | [0.0, 0.975] | 0/8 (0.0) | [0.0, 0.36942] |

AP, AUROC, FPR, and their paired deltas are null by design. Each candidate differs from released MARS on the single primary decision; the two-sided exact McNemar p-value is 1.0. The superiority gates fail.

## Evidence gates

| Gate | Verdict | Reason |
|---|---|---|
| Outcome-blind scorer/model/threshold binding | **Passed** | Scores were immutable and validated before outcome access |
| Exact public-source provenance and licensing | **Passed** | Stanford repository v17; all nine source digests verified; CC BY 4.0 attribution retained |
| Casa Grande temporal false-positive stress test | **Passed as diagnostic** | 86 metered-zero controls; candidates reduced FPR but lost recall |
| Independent geographic-site execution | **Passed** | Nine frozen Evanston products at Rawhide Leasing Yard were scored without retuning |
| Independent-site negative/FPR validation | **Failed / unavailable** | Evanston contains zero negative events |
| Ranking superiority | **Failed / unavailable** | Casa AP deltas inconclusive; Evanston ranking undefined |
| Recall noninferiority | **Failed** | Both candidates missed all frozen ≥1,000 kg/h positives across these site-specific primary views |
| Across-the-board superiority over released MARS | **Failed** | FPR gains do not compensate for failed AP/recall gates |
| Broad geographic generalization | **Not established** | One independent site with one primary positive and no negatives is insufficient |
| Pixel localization/IoU | **Unavailable** | Official summary sources contain flow outcomes, not plume masks |

## Strongest permissible claim

On the Casa Grande temporal controlled-release stress test, Gaussian+DOFA and Spatial-Prithvi reduced fixed-threshold false positives relative to released MARS-S2L, but both missed all eight primary high-rate releases and did not improve ranking. At the independent Evanston site, released MARS detected the sole primary high-rate release while both candidates missed it; the cohort had no zero-release observations, so independent-site FPR and ranking could not be evaluated. These results reject superiority and do not establish broad geographic generalization.

## What evidence is still required

A new sealed confirmation cohort must contain, at minimum:

1. metered zero-release observations at physical sites not represented in model fitting or excluded source metadata;
2. enough ≥1,000 kg/h releases and zero-release controls across multiple sites to support site-block uncertainty;
3. frozen exact L1 product/crop/reference rules and prior-only references;
4. sufficient positives and negatives for AP/AUROC, recall, FPR, and paired superiority gates; and
5. plume masks or another validated spatial reference if an IoU/localization claim is desired.

The Evanston outcomes must not be used to retune the current models or thresholds. Future cohorts must remain separate confirmations rather than extensions selected from these results.

## Reproducibility and immutable identities

### Casa Grande

- Scoring protocol: `configs/stanford_large_controlled_release_scoring_protocol.json`, SHA-256 `658f0d51b9b710fb6aa9deb1e83290f86e5e5b1bb53f22ab7349db0851e4ec70`
- Crop manifest: `.research/stanford_controlled_release_2024_2025/l1c_stress/crop_manifest.json`, SHA-256 `500246526f908b0de13553d61c0ea20d6cd676b4467309c965454eb5aa32eed3`
- One-shot report: `reports/experiments/stanford_large_controlled_release_one_shot.json`, SHA-256 `6b148e85a9d25ef9627a169ef2c7c2d89c963418a6c693aeb16a35a90a7e53b0`

### Evanston

- Scoring protocol: `configs/stanford_evanston_label_free_scoring_protocol.json`, SHA-256 `182b5b7065cdd14ce7868f4945dccaceac4da1ab802de66280c150aae9ff41fb`
- Outcome protocol: `configs/stanford_evanston_outcome_evaluation_protocol.json`, SHA-256 `4e8fc757e815530702dcdd30605192c4a4b68e661e6498ac7b268cdac55dc76f`
- Crop manifest: `.research/stanford_controlled_release_2024_2025/evanston_l1c_stress/crop_manifest.json`, SHA-256 `0ed63b0199673b8932c2825476b0885abed10ee9fe996ae9612ef0872faac23f`
- Label-free scores: `.research/stanford_controlled_release_2024_2025/evanston_l1c_stress/scores/label_free_scores.npz`, SHA-256 `89e887309ba15ed85ba38a24fee653ca830c1a0a1d8a9c0c075224655b7aa6f8`
- Source manifest: `.research/stanford_controlled_release_2024_2025/evanston_outcomes/source_manifest.json`, SHA-256 `a2f746c86cbea1184e5b22bee6fcf3b05c69bd7eea865e2e2f1c04e1211e7a4b`
- Joined one-shot rows: `.research/stanford_controlled_release_2024_2025/evanston_l1c_stress/scores/one_shot_joined.jsonl`, SHA-256 `3e9fc874d9721b16aaab5d285fb34701341b0caeab96f30f7ae8eb54d2e3307d`
- One-shot report: `reports/experiments/stanford_evanston_one_shot.json`, SHA-256 `372e963fa335b9e0506770352480db25a224518a67ca220b3bca6f45368b5917`
- Base scorer: `tools/score_stanford_large_controlled_release_label_free.py`, SHA-256 `adfbe547d4108cc1865026ec300a8ccfa6fccf82cc7c824a936dcf05f42f5239`
- Fixed Evanston launcher: `tools/score_stanford_evanston_label_free.py`, SHA-256 `4e9ec4be6f677a21e1bb0bd6edff33d9ae901ba096162d19cc4db1bfbe5e8d41`
- Evanston evaluator: `tools/evaluate_stanford_evanston_scores.py`, SHA-256 `26d34b9a5fbd01804855608f95a5e3e1ec20a3730b4b40cd6f19059255099958`

The scoring and evaluation commands refuse overwrite and therefore must be run only in a fresh artifact location after all ignored dependencies are restored and hash-verified.

## Source attribution

Reuland et al., *Large-Scale Controlled Methane Releases for Satellite-Based Detection and Emission Quantification of Point-Sources*, Stanford Digital Repository, DOI `10.25740/qh001qt3946`, CC BY 4.0. The outcome field is the repository-defined `ch4_kgh_mean`, calculated over the five-minute window preceding the approximate satellite overpass.
