# EMIT V002 external positive confirmation

This prediction-blind sealed cohort contains positives only; it cannot estimate false-positive rate or average precision.

- Cohort: 55 scenes / 55 independent groups
- ERSRR five-seed mean recall: 0.011
- Released MARS-S2L recall: 0.018
- Recall delta: -0.007
- Paired recall-delta 95% CI: -0.055 to +0.029
- ERSRR five-seed mean EMIT-mask IoU: 0.001
- Released MARS-S2L EMIT-mask IoU: 0.000

EMIT/Sentinel-2 offsets of up to six hours make this a confirmation and stress test, not exact simultaneous Sentinel-2 ground truth. The sealed MARS strict campaign remains the primary no-plume and same-distribution benchmark.

Provenance note: the original JSON recorded the tracked worktree as dirty because its WSL Git
check omitted the repository's Windows line-ending setting. Host Git and
`git -c core.autocrlf=true status --porcelain --untracked-files=no` both reported a clean tracked
worktree. The JSON retains the original value and report SHA-256 in an explicit amendment;
predictions and metrics were not rerun or changed.
