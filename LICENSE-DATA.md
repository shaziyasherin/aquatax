# Data licences

| component | licence |
|---|---|
| `manifest/`, `splits/` | CC BY 4.0 |
| `annotations/aquatax4_boxes_ccby.json` (AquaTrash and AquaSurf-Malnad-223 boxes) | CC BY 4.0, as are both source datasets |
| `results/` (predictions, statistics, audit outputs) | CC BY 4.0 |

CC BY 4.0: <https://creativecommons.org/licenses/by/4.0/>

**FloW-Img.** No FloW-Img image or annotation file is redistributed here. Its boxes are regenerated
by `scripts/rebuild_corpus.py` from the files its authors distribute for non-commercial research
use (Cheng et al., 2021, <https://doi.org/10.1109/ICCV48922.2021.01077>).

**Not part of this release:** trained checkpoints, the AquaVisionNet implementation, the
whale-optimisation search and the SAM-assisted copy-paste script. They are available from the
corresponding author on reasonable request.

**Source images** of all three datasets remain under their own licences and are not part of this
release.
