# CloudSEN12+ producer-asset availability audit

The frozen `cloudsen12_clear_images.csv` names producer-side 200x200 `s2.tif`
and `cloudmask.tif` paths. Those paths are metadata references, not downloadable
files in the public MARS-S2L revision used by this study.

The recursive Hugging Face tree for MARS-S2L revision
`c26b1d7e31a0c5241fa37c9140802622c215eb32` contains 586 files totaling
305,262,101 bytes. Its only CloudSEN12+ files are:

- `cloudsen12_clear_images.csv` (11,695,682 bytes)
- `cloudsen12_data/cloudsen12_stats_dataset.csv` (8,615,755 bytes)

A direct request for the development example
`ROI_16217__20190618T101029_..._s2.tif` returned HTTP 404. The ignored API
tree response has SHA-256
`7186a28dc152706bd84b66e7804c79306131e4b159a104efde34fd61e831e59e`.

Therefore the experiment reconstructs the exact 2 km target/reference crop
from the public Sentinel-2 L1C product identities. It does not silently replace
the missing producer crop, change processing level, or use a nearby acquisition.
