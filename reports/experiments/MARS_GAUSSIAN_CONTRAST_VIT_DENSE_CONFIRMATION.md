# Physics-contrast Gaussian-ViT dense transfer confirmation

- Selected epoch: **70**
- Memorization gate: **True**
- Disjoint-validation gate: **True**
- Train scene AP: **0.4997 -> 0.3797**
- Validation scene AP: **0.4979 -> 0.3985**
- Train dense-evidence AP: **0.4990 -> 0.9195**
- Validation dense-evidence AP: **0.5000 -> 0.8092**
- Train pixel IoU: **0.0534 -> 0.3204**
- Validation pixel IoU: **0.0521 -> 0.2711**

Both gates pass; a separately frozen exposure-scaling experiment is justified.
