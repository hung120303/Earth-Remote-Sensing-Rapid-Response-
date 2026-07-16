# Label-free temporal site-prior scene architecture

- Selected top-k: **1**; logit prior weight: **0.10**.
- Selection AP delta: **+0.00044**.
- Confirmation AP delta: **+0.00093**.
- All promotion gates pass: **false**.

The prior uses no labels at inference: it is the mean of each physical site's top-k current scene logits, added to every scene at that site.
