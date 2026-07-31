# DOFA-v2 scene features — folds 3/4

The frozen sensor-aware extractor completed all 17,745 authorized development
rows. The ignored float16 cache is 17,745 x 10,752, 352,234,214 bytes, and has
SHA-256 `2c43352e92f96549d64188c78318fe606d55552aa9587aace1a195e8dfba66fe`.

Every value is finite. Sample IDs, physical-site groups, labels, sensors, and
fold assignments match the canonical manifest in exact order. Counts are
8,799/8,946 rows for folds 3/4, 16,221/1,524 no-plume/plume rows, and
14,963/2,782 Sentinel-2/Landsat rows. The mixed-sensor run used 6.31 GiB peak
CUDA allocation. Folds 0/1/2 and the paper test were not loaded; the bulk cache
remains ignored.
