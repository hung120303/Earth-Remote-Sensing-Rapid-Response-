# Stanford controlled-release Sentinel-2 L1C windowed crops

- Status: `acquisition_complete`
- Requested/verified pairs: 9 / 9
- Acquisition errors: 0
- Ignored crop assets: 18
- Ignored crop bytes: 9,551,062
- Target all-band nonzero fraction, min/median: 1.0 / 1.0
- Reference all-band nonzero fraction, min/median: 1.0 / 1.0
- Ignored crop manifest: `.research/stanford_controlled_release_2024_2025/evanston_l1c_stress/crop_manifest.json`
- Crop manifest SHA-256: `0ed63b0199673b8932c2825476b0885abed10ee9fe996ae9612ef0872faac23f`

Only six-band 256x256 Sentinel-2 Level-1C windows were read and retained. No full product, Level-2A, SCL, cloud product, release label, release rate, or detector output was accessed.

Ten-meter bands use nearest/native target-grid sampling; B11/B12 use bilinear resampling. Raw uint16 L1C DNs are preserved without scale or processing-offset correction.

Landsat remains pending exact USGS EROS Collection 2 Level-1 authentication. No Landsat Level-2 substitute is permitted.
