# MARS-S2L Sentinel-2 preprocessing audit

The public `marss2l` package was inspected to resolve the producer's exact
Sentinel-2 resampling contract before spatial features were extracted.

- `marss2l==0.2.4` wheel SHA-256:
  `d286aa54908f93c7c3a01a4cf80ad81846007729546ad11e0d895382aa526ac1`
- `georeader-spaceml==1.5.9` wheel SHA-256:
  `9541e5c4b0b2835254a3bf23699e33e49952c1000f3f329e2f39f8d4f858a683`

`marss2l.mars_sentinel2.s2lutils` requests the exact product from Earth
Engine's `COPERNICUS/S2_HARMONIZED` collection on the requested 10 m grid.
Earth Engine exports the native 20 m Sentinel-2 bands at 10 m using nearest
neighbor. `georeader.readers.ee_image.interpolate_20mbands_s2ee` then restores
the intended interpolation by nearest-neighbor downsampling to 20 m followed
by bilinear upsampling to 10 m, with uint16 rounding.

The local exact-product cropper now implements that published algorithm for
B11/B12. B02/B03/B04/B08 stay on their native 10 m grid. On the audited
CloudSEN12+ row, the exact producer affine plus public L1C product reproduces
all four published summary statistics for every native 10 m band to numerical
precision. This cross-check was completed before any model feature or outcome
was computed from the pilot.
