# EMIT V002 external CloudSEN12 acquisition

## Result

- Exact MARS Sentinel-2 cloud model: **UNetMobV2_V2**.
- Prediction-blind gate-pass scenes processed: **60**.
- Hash-verified masks: **60**.
- Ignored raw mask bytes: **71,992**.

Eligibility was fixed from independent L2A SCL, radiometric validity, and plume containment. CloudSEN12 output was generated only afterward as a model input, so it cannot alter cohort membership. The 13-band order, input scaling, model repository, and exact weight hash are frozen in the JSON report.
