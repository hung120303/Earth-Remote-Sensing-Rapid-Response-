# Zhao et al. (2025) DSAN dataset audit

Generated: 2026-08-16T15:58:05.831679+00:00.

## Outcome

All six Zhao sites overlap MARS development or official-test geography within the frozen 25 km boundary. The archive is research evidence only and contributes zero training, calibration, selection, or independent-evaluation rows.

## Archive integrity and composition

The official Science Data Bank archive passed its frozen byte-size, MD5, SHA-256, path-safety, and ZIP CRC checks. It contains 1,627 class-organized retrieval-map PNGs and no dense mask files. No archive extraction or pixel-array decoding was performed.

Observed raster dimensions: `{"430x430": 1627}`.

| Dataset | Field | Country | Rows | Min dev distance (km) | Min official-test distance (km) | Eligible |
|---:|---|---|---:|---:|---:|:---:|
| 1 | Korpeje | Turkmenistan | 242 | 0.000 | 0.000 | no |
| 2 | Gamyshlja Gunorta | Turkmenistan | 259 | 0.000 | 0.000 | no |
| 3 | Keymir | Turkmenistan | 252 | 0.000 | 0.000 | no |
| 4 | Hassi Messaoud | Algeria | 325 | 0.014 | 0.014 | no |
| 5 | Hassi Messaoud | Algeria | 309 | 0.014 | 0.014 | no |
| 6 | Permian basin | USA | 240 | 0.216 | 0.216 | no |

## Claim boundary

This audit can establish integrity, composition, and development eligibility only. It cannot establish model improvement, independent generalization, or superiority to MARS-S2L.

Primary sources: [ACP paper](https://acp.copernicus.org/articles/25/4035/2025/) and [Science Data Bank record](https://doi.org/10.57760/sciencedb.15792).
