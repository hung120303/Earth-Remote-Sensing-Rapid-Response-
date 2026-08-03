# Stanford controlled-release Sentinel-2 L1C windowed crops

- Status: `acquisition_complete`
- Requested/verified pairs: 169 / 169
- Acquisition errors: 0
- Ignored crop assets: 338
- Ignored crop bytes: 183,395,980
- Target all-band nonzero fraction, min/median: 1.0 / 1.0
- Reference all-band nonzero fraction, min/median: 1.0 / 1.0
- Ignored crop manifest: `.research/stanford_controlled_release_2024_2025/l1c_stress/crop_manifest.json`
- Crop manifest SHA-256: `500246526f908b0de13553d61c0ea20d6cd676b4467309c965454eb5aa32eed3`

Only six-band 256x256 Sentinel-2 Level-1C windows were read and retained. No full product, Level-2A, SCL, cloud product, release label, release rate, or detector output was accessed.

Ten-meter bands use nearest/native target-grid sampling; B11/B12 use bilinear resampling. Raw uint16 L1C DNs are preserved without scale or processing-offset correction.

Landsat remains pending exact USGS EROS Collection 2 Level-1 authentication. No Landsat Level-2 substitute is permitted.
