# AquaTax-4: a leakage-audited, multi-source benchmark for floating macro-debris detection

Benchmark manifest, rebuild script, annotations, leakage-audit and evaluation code for:

> Shaziya Banu S and Pallavi G. B. *Floating Macro-Debris Detection for Freshwater Hazard
> Monitoring: A Leakage-Audited Multi-Source Benchmark and a Controlled Study of Detection
> Pretraining.* Journal of Hazardous Materials Advances (under review, manuscript HAZADV-D-26-02212).

Archived release: **DOI [https://doi.org/10.5281/zenodo.23023328]**.

AquaTax-4 pools 2,592 images and 5,825 bounding boxes in four material classes (plastic 5,538,
paper 123, metal 118, glass 46) from three sources, split 70/15/15 into 1,814 / 389 / 389 images
with seed 20260729.

## What is here, and what is not

**No image is redistributed.** One of the three sources, FloW-Img, is distributed by its authors
for non-commercial research use and must be requested from them. Instead of images, this
repository publishes everything needed to rebuild the exact corpus from your own copies of the
sources, and to check every number in the paper:

| path | contents |
|---|---|
| `manifest/AquaTax4_manifest.csv` | one row per image: split, image id, source, original file path, SHA-256, size, box count per class |
| `splits/{train,val,test}.txt` | the split, as SHA-256 + source + original path |
| `annotations/aquatax4_boxes_ccby.json` | all boxes of the two CC BY 4.0 sources (AquaTrash, AquaSurf-Malnad-223) |
| `scripts/rebuild_corpus.py` | rebuilds the corpus from your copies of the three sources |
| `scripts/flow_convention.json` | how FloW-Img's own Pascal VOC boxes are converted, verified to reproduce the corpus exactly |
| `scripts/` | the leakage audit, the evaluation and bootstrap statistics, the matched-protocol baselines, and the figures |
| `configs/dino/` | the reported DINO configuration and the ablation configurations, with their field-by-field diffs |
| `results/MEASURED_RESULTS.md` | every number the paper reports, for checking a recomputation |
| `results/preds/`, `results/stats/`, `results/leakage/` | per-image predictions of all five detectors, bootstrap outputs, and the pHash audit |

**Available from the corresponding author on reasonable request:** the trained checkpoints, the
AquaVisionNet implementation and the SAM-assisted copy-paste script. The copy-paste procedure is
described step by step in the paper's Supplementary Material S1. The hyperparameter search belongs to
separate work; every value it selected is listed in Supplementary Material S2, so the reported runs
can be reproduced without it.

## Rebuild the corpus

1. Obtain the three sources:
   - **AquaTrash**: Panwar et al. (2020), <https://doi.org/10.1016/j.cscee.2020.100026>
   - **FloW-Img**: Cheng et al. (2021), <https://doi.org/10.1109/ICCV48922.2021.01077>, on request
     from its authors
   - **AquaSurf-Malnad-223**: <https://doi.org/10.5281/zenodo.22945111> (also IEEE DataPort,
     <https://doi.org/10.21227/k8d4-5n64>)
2. Run, with Python 3.8 or later and no other dependency:

   ```
   python scripts/rebuild_corpus.py --roots <AquaTrash folder> <FloW-Img folder> <AquaSurf-Malnad-223 folder> --out rebuilt
   ```

The script hashes every image under the three folders, finds each of the 2,592 manifest entries by
its SHA-256, and writes `rebuilt/{train,val,test}.json` in COCO format with the image and category
ids used in the paper. Anything it cannot find is listed in `rebuilt/missing.csv`. Before release,
this procedure was run against the original downloads and reproduced the corpus image by image;
`RELEASE_REPORT.md` in the Zenodo record records the check.

## Check the paper's numbers without a GPU

The per-image predictions in `results/preds/` let every confidence interval and paired comparison
be recomputed on a CPU. For example, the Faster R-CNN versus AquaVisionNet difference:

```
python scripts/revision_analysis.py compare --gt rebuilt/test.json --dets results/preds/frcnn_test.json --dets2 results/preds/avn15_test.json --B 1000 --seed 20260729
```

| paper | what | script |
|---|---|---|
| Table 2 | exact counts per split and class | `15_prepare_corpus_B.py`, `01_count_boxes.py` |
| Tables 3, 6 | scores of all five detectors from their predictions | `06_coco_summary.py` |
| Tables 3, 5-8 intervals | image-level bootstrap and paired bootstrap, B = 1000, seed 20260729 | `revision_analysis.py bootstrap` / `compare` |
| Table 8 | pHash leakage audit, Hamming 0-10 sweep, per-source rates | `13_leakage_audit_plus.py audit` / `curve`; `revision_analysis.py clean` |
| Table 9 | count error at a validation-selected threshold | `revision_analysis.py counts` |
| Figures | all result figures | `17_make_figures.py`, `25_figures_3_6.py` (measured values are literals; no data needed) |

`run_all.py` drives the evaluation stages in order from a paths file
(`revision_paths.template.json`).

## Retraining

| model | how |
|---|---|
| YOLOv8-m, Faster R-CNN (matched protocol: 512 px letterbox, 15 epochs) | `yolov8_baseline_standalone.py`, `fasterrcnn_baseline_standalone.py`; predictions with `02_export_yolo.py`, `10_export_matched.py` |
| DINO-Swin-B, ablation rows A and B | `16_mmdet_run.py train` with `configs/dino/dino_ablation_{A,B}.py` |
| DINO-Swin-B, reported configuration (row C) | `configs/dino/dino_reported_rowC.py`; its training set needs the copy-paste images (code on request) |
| AquaVisionNet | on request |

Two independent trainings of the same Faster R-CNN recipe on identical data differed by 0.007
mAP@0.5, so a retrained model should land within about that distance of the reported values, not
on them exactly.

Absolute paths in script docstrings (`D:\aquavisionnet\...`) are those of the authors'
workstation; every path is a command-line argument. **`avn_out` in any docstring refers to an
earlier, defective build of the corpus that is disclosed in Section 3.2 of the paper and was not
used for any reported number.** Use the rebuilt corpus instead.

## Environments

Two conda environments were used, both on one NVIDIA RTX 4000 Ada Generation GPU (20 GB):

| env | used for | versions |
|---|---|---|
| `yolo` | the matched-protocol detectors, all statistics | torch 2.5.1+cu121, torchvision 0.20.1, ultralytics 8.4.163, pycocotools |
| `aquavision` | DINO-Swin-B and its ablation | torch 2.1.0+cu121, torchvision 0.16.0, mmdet 3.3.0, mmengine 0.10.7 |

`scripts/check_env.py` prints what an environment provides. The rebuild script and the statistics
need only Python, NumPy and pycocotools.

## Licences

- Code (`scripts/`, `configs/`): MIT, see `LICENSE`.
- Manifest, splits, released annotations and results: CC BY 4.0, see `LICENSE-DATA.md`.
- Source images remain under their own licences and are not part of this release.

## Citation

See `CITATION.cff`. Please also cite the three source datasets.

