# R1.6 - every detector re-scored on the leakage-cleaned subset

## Table 8 (paper threshold, Hamming <= 5)

| detector | full (n=all) | clean | delta | flagged-only |
|---|---|---|---|---|
| yolov8m | 0.334 | 0.327 | -0.006 | 0.974 |
| frcnn | 0.467 | 0.462 | -0.005 | 0.947 |
| avn15 | 0.314 | 0.306 | -0.008 | 0.855 |
| dino_plain | 0.640 | 0.637 | -0.003 | 0.990 |
| dino_tta | 0.688 | 0.684 | -0.004 | 0.995 |

## Threshold sensitivity of the cleaned score

| detector | h<=0 | h<=2 | h<=5 | h<=8 | h<=10 |
|---|---|---|---|---|---|
| yolov8m | 0.334 | 0.331 | 0.327 | 0.328 | 0.322 |
| frcnn | 0.467 | 0.465 | 0.462 | 0.460 | 0.452 |
| avn15 | 0.314 | 0.311 | 0.306 | 0.309 | 0.302 |
| dino_plain | 0.640 | 0.638 | 0.637 | 0.638 | 0.634 |
| dino_tta | 0.688 | 0.686 | 0.684 | 0.686 | 0.685 |

A ranking that survives every column is a ranking the leakage did not create. A ranking that flips somewhere must be reported as flipping.
