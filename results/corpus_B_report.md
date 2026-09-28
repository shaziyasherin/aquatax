# Corpus of record (Build B), verified

Source: `D:\aquavisionnet\data\aquatax10\annotations`  ->  working copy: `D:\aquavisionnet\revision\corpus_B`

## Split composition

| split | images | unique images | images with boxes | negatives | boxes | plastic | paper | metal | glass |
|---|---|---|---|---|---|---|---|---|---|
| train | 1814 | 1814 | 1676 | 138 | 4040 | 3837 | 87 | 83 | 33 |
| val | 389 | 389 | 359 | 30 | 907 | 866 | 17 | 17 | 7 |
| test | 389 | 389 | 360 | 29 | 878 | 835 | 19 | 18 | 6 |
| **total** | 2592 | | | | 5825 | 5538 | 123 | 118 | 46 |

Every count above is read from the annotation files. None is obtained by subtraction (R4.5).

## Byte-level duplicate check

| split A | id | file | split B | id | file |
|---|---|---|---|---|---|
| train | 479 | `malnad_0110.jpg` | test | 491 | `malnad_0122.jpg` |

Within-split duplicates: train 0, val 0, test 0.
