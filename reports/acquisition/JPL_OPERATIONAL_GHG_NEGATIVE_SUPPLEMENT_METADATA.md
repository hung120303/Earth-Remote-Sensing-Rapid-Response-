# JPL CACH4 negative-supplement metadata gate

**Decision: FAIL.** Target-catalog stage: **not authorized and not queried**.

This final metadata filter read only released CACH4 train negatives, public ENVI header-derived centers, the safe-column MARS location view, and hash-bound already-counted MARS-Hyperspectral negative-pair metadata. It accessed no protected MARS outcome field, no released JPL test content, and no Sentinel-2/Landsat catalog or asset.

## Eligibility result

- Raw CACH4 train rows: 3,332
- Resolved train negatives: 3,149
- Within 25 km of official MARS test geography: 390
- Within 25 km of an already-counted negative source crop: 194
- Excluded for either reason: 584
- Eligible rows / flightlines: 2,565 / 104
- Eligible transitive 25 km components: 15
- Components wholly novel beyond every MARS representative location: 15

The frozen `minimum_nonprotected_candidate_locations >= 20` gate counts 25 km connected components, not tiles or flightlines: **FAIL (15)**.

## Distance audit

- Nearest official-test location: min 4.836929 km; median 274.400742 km
- Nearest counted prior-negative crop: min 4.832943 km; median 140.333998 km
- Nearest any-MARS representative: min 4.836929 km; median 274.400742 km

Detailed row-level distances and stable group IDs remain in ignored `.research/jpl_operational_ghg_supplement/cach4_train_negative_eligible_rows.jsonl`. The compact JSON records hashes for the protocol, resolved CACH4 rows, safe-column MARS manifest, prior Stage B report, pair catalog, mask catalog, and filtered output.

## Claim boundary

Metadata-stage PASS only authorizes the separately frozen target-catalog feasibility query. It does not establish target-pair yield, label observability, transferability, model performance, or superiority over MARS-S2L.
