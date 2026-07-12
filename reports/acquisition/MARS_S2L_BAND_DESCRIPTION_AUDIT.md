# MARS-S2L image band-description audit

- Images inspected: 34,109
- Embedded 12-band descriptions: 34,108
- All descriptions absent; exact frozen manifest declaration used: 1
- Contract failures: 0

The fallback accepts only the producer omission where all 12 TIFF descriptions are absent and the hash-bound manifest declares the exact expected order. Partial, mixed, or conflicting labels remain fatal.

## Manifest-fallback samples

- `8f92ea70-e87c-4ef5-9101-7791a68d37a1` — internal_validation / Algeria
