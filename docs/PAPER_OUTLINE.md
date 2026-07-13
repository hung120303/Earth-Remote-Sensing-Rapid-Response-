# ERSRR publication outline and claim contract

## Working title

**Physics-guided selective methane-plume detection in Sentinel-2 imagery with spatially isolated evaluation**

## Central research question

Can a dual-temporal, physics-guided detector reduce false alarms on observable no-plume
Sentinel-2 scenes while preserving plume recall at geographically isolated sites, relative to the
released MARS-S2L checkpoint?

The paper is a detection and segmentation study. It does not claim emission-rate retrieval,
continuous methane concentration estimation, or operational deployment.

## Contributions that may be claimed from methods alone

1. A pinned MARS-S2L data contract with explicit `PLUME`, `NO_PLUME`, `UNCERTAIN`, and
   `UNOBSERVABLE` states; target/reference identity; wind and cloud provenance; and SHA-256-bound
   raw artifacts kept outside Git.
2. A 25 km connected-component split that separates 23,763 training scenes, 5,945 internal
   validation scenes, and a 4,401-scene strict test whose groups do not intersect official training.
3. A 14.27 M-parameter full-resolution U-Net with target/reference fusion, release-compatible MBMP,
   wind, CloudSEN12 observability, segmentation, scene-presence, and quality/abstention outputs.
4. A predeclared five-seed operating-point protocol: maximize validation recall subject to FPR
   <= 0.05, then AP and positive-mask Dice; never tune from strict-test behavior.
5. A prediction-blind external-confirmation funnel linking EMIT V002 plumes to product-matched
   Sentinel-2 L1C target/reference scenes and retaining 55 independent scenes after exact-input
   radiometry and CloudSEN12 gates.

## Result-dependent claim ladder

The strongest permissible claim is selected only after the sealed campaign.

| Evidence | Permissible wording |
|---|---|
| Five-seed strict gate and paired baseline gate both pass | ERSRR outperformed the released MARS-S2L checkpoint on the same spatially isolated cohort at the frozen operating rules. |
| ERSRR gate passes but paired superiority is uncertain | ERSRR met its predeclared operating target; superiority to MARS-S2L was not established. |
| Mean improves but confidence interval or a gate fails | ERSRR showed a directional improvement that requires confirmatory evaluation. |
| Recall or false-alarm gate fails | The proposed architecture did not meet the promotion boundary; report the negative result without retuning. |

MARS-S2L paper metrics and same-cohort checkpoint metrics must remain in separate tables. The
official paper test uses a different cohort and cannot support a paired superiority statement.

## Frozen campaign outcome (2026-07-12)

The final row of the claim ladder applies. ERSRR v3 did **not** meet the promotion boundary and did
not outperform the released MARS-S2L checkpoint overall. The five-seed ERSRR mean reduced the
same-cohort false-positive rate from 0.0948 to 0.0367 (61.3% relative reduction), but recall fell
from 0.6418 to 0.3194. In the paired 2,000-replicate seed-and-25-km-group bootstrap, the relative
FPR reduction was 61.2% (95% CI 46.7% to 72.8%) and the recall delta was -31.5 percentage points
(95% CI -42.2 to -16.4). AP, AUROC, and pixel segmentation were also inferior.

The paper must therefore be framed as a rigorous spatial-transfer and false-alarm trade-off study,
not as an architecture-superiority paper. The central empirical finding is that strong random-seed
internal validation (mean recall 0.8317 at mean FPR 0.0489) did not predict performance on unseen
25 km groups (mean recall 0.3194 at mean FPR 0.0367). The released MARS-S2L checkpoint also
transferred poorly relative to its different-cohort paper values, but remained materially stronger
than ERSRR on recall, AP, AUROC, and segmentation on this paired cohort.

No v3 threshold, loss, architecture, or postprocessing rule may be changed in response to these
strict results. Any v4 system informed by this result requires a newly untouched final cohort.
The authoritative campaign artifact is `reports/experiments/mars_v3_strict_campaign.json`; the
chronology, interpretation boundaries, and next-study decisions are maintained in
`docs/RESEARCH_LEDGER.md`.

## Manuscript structure

1. **Introduction** — methane point-source monitoring, no-plume rejection, spatial leakage, and the
   research question.
2. **Related work** — MBMP/multipass retrieval, CH4Net, MARS-S2L, small-plume segmentation,
   EMIT confirmation, and selective prediction.
3. **Data** — MARS-S2L revision and cohort construction; label states; negative provenance; EMIT
   V002/Sentinel-2 confirmation funnel; licenses and exclusions.
4. **Methods** — exact 16-channel input contract, architecture, loss terms, group-balanced sampling,
   proposal ablation, quality head, and three-state decision policy.
5. **Evaluation protocol** — frozen seeds, thresholds, strict spatial split, author-fixed released
   baseline, paired block bootstrap, metrics, and prediction-blind external seal.
6. **Results** — internal validation variability, strict paired results, segmentation results,
   calibration/selective results, error strata, and external positive confirmation.
7. **Discussion** — practical false-alarm implications, small-plume behavior, domain shift,
   determinism, and why positive-only EMIT evidence cannot measure FPR.
8. **Limitations and ethics** — label uncertainty, source attribution, monitoring misuse, geographic
   coverage, meteorological dependence, and non-operational status.
9. **Reproducibility statement** — commit, environment, manifests, hashes, ignored bulk data,
   commands, seed policy, and artifact availability.

## Required primary tables

1. Dataset funnel and exclusions, with plume/no-plume counts and spatial groups.
2. MARS-S2L published targets, clearly labeled as different-cohort context.
3. Five internal-validation seeds with selected epoch, threshold, proposal weight, recall, FPR, AP,
   AUROC, calibration, and Dice.
4. Same-cohort strict results for released MARS-S2L and every ERSRR seed.
5. Seed-aggregated paired deltas with 95% 25 km group-bootstrap intervals.
6. Selective decision metrics: coverage, abstention, accepted no-plume error, and NPV.
7. EMIT positive-confirmation recall and descriptive mask overlap for ERSRR and released MARS-S2L;
   no FPR, specificity, or AP columns.

## Required figures

1. Cohort construction and prediction-blind sealing flow.
2. Architecture/input-contract diagram.
3. Strict-test precision-recall or operating-point plot with per-seed dispersion.
4. Paired group-bootstrap delta distributions for recall and FPR.
5. Recall by plume-size and observability strata.
6. Representative true positive, false negative, false positive, and abstained scenes selected by a
   declared rule rather than visual preference.

## Statistical analysis contract

- Unit of resampling: frozen 25 km group, never individual pixels or temporally related scenes.
- Replicates: 2,000 with the committed bootstrap seed.
- Primary endpoint: scene recall at the validation-frozen operating point under FPR <= 0.05.
- Report point estimates, two-sided percentile intervals, and absolute paired deltas.
- Report relative FPR reduction only when the baseline FPR is nonzero, with the absolute delta beside it.
- Report every fixed seed and failures; do not choose a best seed from strict results.
- Treat EMIT as positive confirmation only. An absent catalog plume is not a no-plume label.
- Do not infer physical concentration or flux from the unresolved MARS enhancement units.

## Ablations and secondary analyses

The current campaign may report only ablations selected before strict evaluation: neural presence
versus connected-proposal blends, released MBMP/MARS-S2L/CH4Net baselines, and the recorded legacy
compact model. Any new architecture, loss, threshold, backbone, or postprocessing idea is future work
and requires a new sealed test cohort or a preregistered confirmatory split.

Secondary analyses should include plume-size strata, geography, cloud/observable fraction,
target-reference interval, wind speed, and score calibration. Sparse strata are descriptive and must
not be presented as powered subgroup claims.

## Reproducibility and release bundle

Release code, configs, cohort builders, compact manifests, aggregate metrics, environment pins,
report HTML, and cryptographic artifact receipts. Do not release or commit bulk imagery, protected
Earthdata URLs with temporary signatures, credentials, checkpoints without a deliberate model-card
decision, or third-party data that the licenses do not permit redistributing.

The training runs fix Python/NumPy/PyTorch seeds but do not currently request deterministic CUDA
algorithms. The manuscript must describe the five seeds as stochastic replicates, not promise bitwise
reproduction. A later confirmatory rerun should enable deterministic kernels where supported and
record any operations that cannot be made deterministic.

## Candidate venues

The method/data-evaluation emphasis is suitable for *Remote Sensing of Environment*, *Atmospheric
Measurement Techniques*, or *IEEE Transactions on Geoscience and Remote Sensing*. Venue choice
should follow the final evidence: a strong paired and external result supports a methods paper; a
failed promotion gate is better framed as a rigorous benchmark/evaluation study.

Given the frozen negative promotion result, the present manuscript should target the benchmark,
evaluation, or methods-validation framing. A later superiority manuscript requires a genuinely new
v4 development cycle and untouched confirmation cohort rather than reinterpretation of v3.
