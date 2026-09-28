# Leakage audit (extended for R1.6)

## Threshold sensitivity

| pHash Hamming | test~train | test~val | test flagged (union) | % of test | clean n | val~train |
|---|---|---|---|---|---|---|
| <= 0 | 2 | 0 | 2 | 0.51 % | 387 | 4 |
| <= 2 | 12 | 8 | 20 | 5.14 % | 369 | 11 |
| <= 5 | 37 | 13 | 45 | 11.57 % | 344 | 29 |
| <= 8 | 96 | 35 | 108 | 27.76 % | 281 | 70 |
| <= 10 | 133 | 54 | 148 | 38.05 % | 241 | 100 |

## Source attribution at Hamming <= 5

| source | flagged | test images | flagged rate |
|---|---|---|---|
| AquaSurf-Malnad-223 | 5 | 33 | 15.2 % |
| AquaTrash | 0 | 48 | 0.0 % |
| FloW-Img | 40 | 308 | 13.0 % |

Exact byte-identical images across splits: **1**.
