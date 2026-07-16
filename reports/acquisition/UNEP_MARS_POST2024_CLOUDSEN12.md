# UNEP MARS post-2024 CloudSEN12+ acquisition

Generated: 2026-07-16T03:41:47.002238+00:00.

## Result

- Exact Sentinel-2 crops processed: **141**.
- Cloud/radiometry/geometry gate pass: **139**.
- Gate failures retained in the audit: **2**.
- Acquisition errors: **0**.
- Median scene/plume clear fraction: **1.000 / 1.000**.
- Auxiliary-training gate pass: **135**.
- Development gate pass: **4**.

## Frozen contract

- Model: `isp-uv-es/cloudsen12_models/UNetMobV2_V2.pt` (`218fa69aa3c7212d4e690b48af88ac6f3c976fc50d07f275b8fd623909183d7a`).
- Thirteen Sentinel-2 L1C bands in the published CloudSEN12 order, scaled by 1/10,000.
- Scene and plume clear-fraction gates: **0.80 / 0.80**.
- Cloud classes are 0 clear, 1 thick cloud, 2 thin cloud, 3 shadow, and 4 invalid.
- Cloud predictions affect observability only and are not methane labels or model inputs beyond the released cloud channel.
- Sealed-external samples were excluded.
