# Label-free temporal site-prior scene architecture

- Selected minimum site history: **20** scenes; top-k: **1**; logit prior weight: **0.25**.
- Selection AP delta: **+0.00004**.
- Confirmation AP delta: **+0.00110**.
- All promotion gates pass: **false**.

The prior uses no labels at inference: it is the mean of each physical site's top-k current scene logits, added to every scene at that site.
