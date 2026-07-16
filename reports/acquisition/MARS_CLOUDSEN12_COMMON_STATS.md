# MARS / CloudSEN12+ common-statistic extraction

The extracted caches contain only preregistered summaries of operational input channels. No paper-test label or CloudSEN12 test feature is present.

- MARS development rows: **44,363**.
- CloudSEN12 training rows: **9,804**.
- CloudSEN12 validation rows: **256**.
- CloudSEN12 sealed test rows retained only in source metadata: **374**.
- Published clear metadata rows without a statistics row: **1**.
- Allowed feature width: **32**.

CloudSEN12 ROI identities are disjoint across train, validation, and sealed test partitions. Every metadata and statistics row is explicitly no-plume.
