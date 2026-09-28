# Measured results, as reported in the paper

Every value below was measured on the AquaTax-4 test split (n = 389 images) and is the value printed
in the paper. Use this file to check a recomputation. All scores use pycocotools with maxDets = 100.
Intervals are 95 % image-level bootstrap intervals, B = 1000, seed 20260729.

The files these numbers were computed from are in `results/preds/` (per-image predictions),
`results/stats/` (bootstrap and paired-bootstrap outputs) and `results/leakage/` (the pHash audit),
once the release is built. With them, every interval and comparison below can be recomputed on a CPU
with `scripts/revision_analysis.py`.

Headings follow the paper's Tables 2, 6, 7, 8 and 9; the lettered sub-tables (6b, 6c, 7b, 8b, 8c) are
breakdowns that the paper gives in its text and figures.

A note on reproducing the training runs: two independent trainings of the same Faster R-CNN recipe
on identical data differed by 0.007 mAP@0.5. A retrained model should land within about that
distance of the values below, not on them exactly.

## Table 2/3. Corpus composition (counted, not inferred)

| split | images | with boxes | negatives | boxes | plastic | paper | metal | glass |
|---|---|---|---|---|---|---|---|---|
| train | 1814 | 1676 | 138 | 4040 | 3837 | 87 | 83 | 33 |
| val | 389 | 359 | 30 | 907 | 866 | 17 | 17 | 7 |
| test | 389 | 360 | 29 | 878 | 835 | 19 | 18 | 6 |
| **total** | **2592** | **2395** | **197** | **5825** | **5538** | **123** | **118** | **46** |

Plastic-to-glass ratio 120:1. The 197 negatives are the verified debris-free images of
AquaSurf-Malnad-223, distributed across all three splits and retained as background.

## Table 6. All detectors, test split, one evaluator

pycocotools, maxDets = 100, 389 images, 878 boxes. Bracketed values are 95 % image-level bootstrap
intervals, B = 1000, seed 20260729.

| detector | params | protocol | mAP@0.5 | mAP@[.5:.95] | AR@100 | img/s |
|---|---|---|---|---|---|---|
| YOLOv8-m | 25.9 M | matched | 0.334 [0.282, 0.411] | 0.155 | 0.300 | 62.2 |
| Faster R-CNN R50-FPN | 41.1 M | matched | 0.467 [0.370, 0.563] | 0.235 | 0.346 | 38.5 |
| AquaVisionNet | 123.6 M | matched | 0.314 [0.265, 0.414] | 0.137 | **0.390** | 37.0 |
| DINO-Swin-B | 108.6 M | enhanced | 0.640 [0.510, 0.762] | 0.307 | **0.556** | 3.8 |
| DINO-Swin-B + TTA/WBF | 108.6 M | enhanced | 0.688 [0.534, 0.800] | 0.314 | 0.486 | 3.8 |

DINO-Swin-B size and speed measured 2026-09-28 on the row-B checkpoint
(`work_dirs/ablB/epoch_18.pth`), which is the same architecture as the reported configuration —
copy-paste is a training-data augmentation and changes neither. 108,597,936 parameters; 200 images
in 52.59 s at batch 1 on the RTX 4000 Ada. The DINO throughput is measured at its own multi-scale
test resolution, not at 512 px, so it is not directly comparable with the matched-protocol rows.
Note that the highest-scoring configuration is **smaller** than AquaVisionNet (108.6 M vs 123.6 M):
parameter count does not predict accuracy anywhere in this study.

Matched protocol: same corpus build, same split, 512 px, 15 epochs, scored with pycocotools.
Enhanced: multi-scale to 1280 px, 18 epochs, SAM copy-paste, optional TTA+WBF. The two blocks are
not directly comparable and are never mixed in a single claim.

## Table 6b. Per-class AP@0.5

| detector | plastic (835) | paper (19) | metal (18) | glass (6) |
|---|---|---|---|---|
| YOLOv8-m | 0.818 | 0.194 | 0.323 | 0.000 |
| Faster R-CNN | 0.808 | 0.512 | 0.545 | 0.002 |
| AquaVisionNet | 0.638 | 0.219 | 0.389 | 0.012 |
| DINO-Swin-B | 0.897 | 0.594 | 0.610 | 0.457 |
| DINO-Swin-B + TTA/WBF | 0.898 | 0.605 | 0.522 | 0.726 |

## Table 6c. Scale-wise (Figure 3 source)

| detector | AP_s | AP_m | AP_l | AR_s | AR_m | AR_l |
|---|---|---|---|---|---|---|
| YOLOv8-m | 0.306 | 0.282 | 0.184 | 0.397 | 0.401 | 0.348 |
| Faster R-CNN | 0.293 | 0.273 | 0.296 | 0.380 | 0.293 | 0.417 |
| AquaVisionNet | 0.165 | 0.234 | 0.180 | 0.275 | 0.376 | 0.446 |
| DINO-Swin-B | 0.385 | 0.357 | 0.341 | 0.534 | 0.545 | 0.589 |
| DINO-Swin-B + TTA/WBF | 0.368 | 0.340 | 0.351 | 0.479 | 0.484 | 0.513 |

## Table 7. Controlled ablation of the enhanced configuration

Each row differs from the one above it in exactly one variable. The resolved configurations differ
in 4 of 655 leaf values, and the diffs are in `configs/dino/`.

| row | backbone init | copy-paste | sampler | TTA+WBF | mAP@0.5 | Δ | attributable to |
|---|---|---|---|---|---|---|---|
| A | Swin-B ImageNet-22k | – | – | – | 0.536 | — | — |
| B | Swin-B Objects365 | – | – | – | 0.538 | **+0.002** [−0.085, 0.110] | detection pretraining (n.s.) |
| C | Swin-B Objects365 | ✓ | inert | – | 0.640 | **+0.102** [−0.075, 0.224] | SAM copy-paste (n.s.) |
| D | Swin-B Objects365 | ✓ | inert | ✓ | 0.688 | +0.048 [−0.023, 0.117] | TTA+WBF (n.s.) |

Every step is a single training run, and no step is separable from zero at 389 test images. The
steps differ in magnitude by a factor of fifty, and only the copy-paste step exceeds the measured
run-to-run floor of 0.007, but we do not present any of them as established.

Per-class Δ:

| step | plastic | paper | metal | glass |
|---|---|---|---|---|
| B − A (pretraining) | +0.003 | −0.050 | −0.070 | +0.126 |
| C − B (copy-paste) | +0.000 | **+0.162** | **+0.141** | **+0.104** |
| D − C (TTA+WBF) | +0.001 | +0.011 | **−0.088** | +0.269 |

The class-balanced sampler used `oversample_thr = 1e-3`, but the least frequent class appears in
3.66 % of training images, so every repeat factor equals 1.000 and the wrapper never resamples
anything. Confirmed arithmetically and by the run's own iteration count (1080 iterations at batch 2
over a 2159-image corpus).

## Table 7b. Validation curves (Figure 5 source)

| epoch | 3 | 6 | 9 | 12 | 15 | 18 |
|---|---|---|---|---|---|---|
| row A | 0.211 | 0.290 | 0.261 | 0.392 | 0.469 | 0.516 |
| row C (reported) | 0.255 | 0.355 | 0.382 | 0.465 | 0.481 | 0.486 |

Row B reaches 0.514 at epoch 18. Row C's validation-to-test gap is +0.154 while rows A and B sit at
+0.020 and +0.024, so the large gap the submitted manuscript flagged is specific to the run trained
with synthetic images and is not a property of the architecture.

## Table 8. Leakage-cleaned re-scoring

pHash Hamming ≤ 5, union of test-to-train and test-to-val near-duplicates: 45 of 389 images
(11.57 %), clean subset n = 344.

| detector | full (389) | clean (344) | Δ | flagged-only, plastic AP (45 img, 110 boxes) |
|---|---|---|---|---|
| YOLOv8-m | 0.334 | 0.327 | −0.006 | 0.974 |
| Faster R-CNN | 0.467 | 0.462 | −0.005 | 0.947 |
| AquaVisionNet | 0.314 | 0.306 | −0.008 | 0.855 |
| DINO-Swin-B | 0.640 | 0.637 | −0.003 | 0.990 |
| DINO-Swin-B + TTA/WBF | 0.688 | 0.684 | −0.004 | 0.995 |

The flagged images contain 110 boxes, all of class plastic, so the final column is single-class AP
and is not comparable to the four-class columns beside it.

## Table 8b. Threshold sensitivity of the cleaned score

| Hamming ≤ | 0 | 2 | 5 | 8 | 10 |
|---|---|---|---|---|---|
| flagged images | 2 | 20 | 45 | 108 | 148 |
| clean n | 387 | 369 | 344 | 281 | 241 |
| YOLOv8-m | 0.334 | 0.331 | 0.327 | 0.328 | 0.322 |
| Faster R-CNN | 0.467 | 0.465 | 0.462 | 0.460 | 0.452 |
| AquaVisionNet | 0.314 | 0.311 | 0.306 | 0.309 | 0.302 |
| DINO-Swin-B | 0.640 | 0.638 | 0.637 | 0.638 | 0.634 |
| DINO-Swin-B + TTA/WBF | 0.688 | 0.686 | 0.684 | 0.686 | 0.685 |

The detector ordering is identical in every column.

## Table 8c. Leakage by source and direction

| source | flagged | test images | rate | 95 % CI |
|---|---|---|---|---|
| FloW-Img | 40 | 308 | 13.0 % | [9.7, 17.2] |
| AquaSurf-Malnad-223 | 5 | 33 | 15.2 % | [6.7, 30.9] |
| AquaTrash | 0 | 48 | **0.0 %** | [0.0, 7.4] |

FloW-Img and Malnad are statistically indistinguishable (Fisher exact, p = 0.79). Both differ from
AquaTrash (p = 0.003 and p = 0.009). If AquaTrash leaked at the pooled rate of the other two, the
probability of observing zero of 48 is 0.0013.

Directions at Hamming ≤ 5: test-to-train 37, test-to-val 13 (union 45), val-to-train 29 (7.5 % of
the validation split). Exactly one image is byte-identical across splits (SHA-256, train:479 =
test:491).

## Table 9. Detection metrics to debris counts

Operating threshold chosen on validation by minimum per-image count error, then applied to test.

| detector | threshold | count MAE | mean bias | plastic | paper | metal | glass |
|---|---|---|---|---|---|---|---|
| Faster R-CNN | 0.65 | **0.458** | −0.185 | 0.93 | 0.84 | 0.61 | 0.33 |
| YOLOv8-m | 0.25 | 0.517 | −0.152 | **0.98** | 0.16 | 0.06 | 0.00 |
| AquaVisionNet | 0.35 | 1.154 | −0.440 | 0.71 | 2.47 | 1.78 | 5.67 |

Ratios are predicted count divided by true count.

## Statistical comparisons

Paired image-level bootstrap, B = 1000, seed 20260729.

| comparison | Δ mAP@0.5 | 95 % CI | P(Δ ≤ 0) | verdict |
|---|---|---|---|---|
| Faster R-CNN − AquaVisionNet | +0.152 | [0.055, 0.213] | 0.002 | significant |
| YOLOv8-m − AquaVisionNet | +0.019 | [−0.057, 0.070] | 0.382 | not distinguishable |
| DINO TTA+WBF − DINO plain | +0.048 | [−0.023, 0.117] | 0.113 | not significant |
| DINO TTA+WBF − plain, metal only | −0.088 | [−0.172, −0.013] | 0.996 | significantly worse |
| DINO plain − Faster R-CNN | +0.173 | [0.028, 0.322] | 0.008 | significant |
| DINO plain − Faster R-CNN, plastic only | +0.089 | [0.061, 0.116] | 0.000 | significant |
| DINO plain − Faster R-CNN, paper only | +0.082 | [−0.138, 0.295] | 0.216 | not distinguishable |
| DINO plain − Faster R-CNN, metal only | +0.065 | [−0.215, 0.379] | 0.365 | not distinguishable |
| DINO plain − Faster R-CNN, glass only | +0.455 | [0.061, 0.802] | 0.000 | significant |
| **C − B (copy-paste)** | **+0.102** | **[−0.075, 0.224]** | **0.142** | **not significant** |
| C − B, plastic only | +0.000 | [−0.013, 0.012] | 0.484 | no effect |
| C − B, paper only | +0.162 | [−0.050, 0.346] | 0.075 | not significant |
| C − B, metal only | +0.141 | [−0.020, 0.268] | 0.052 | not significant |
| C − B, glass only | +0.104 | [−0.482, 0.598] | 0.399 | not significant |
| **B − A (pretraining)** | **+0.002** | **[−0.085, 0.110]** | **0.428** | **not significant** |
| B − A, plastic only | +0.003 | [−0.009, 0.017] | 0.274 | no effect |
| B − A, paper only | −0.050 | [−0.223, 0.158] | 0.631 | not significant |
| B − A, metal only | −0.070 | [−0.231, 0.108] | 0.784 | not significant |

Two independent trainings of the identical Faster R-CNN recipe on identical data differ by 0.007
mAP@0.5. That is the empirical run-to-run floor for this corpus, and no difference below roughly
0.01 should be interpreted.

