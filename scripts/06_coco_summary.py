"""
06_coco_summary.py - full pycocotools summary (12 COCO numbers incl. small/medium/large AP and AR)
for one prediction file; writes JSON used by 07_make_figures.py (Figure 3) and prints the table.

  python scripts\\06_coco_summary.py --gt test.json --dets results\\preds\\avn15_test.json --name avn15 ^
      --out results\\stats\\summary_avn15.json
"""
import argparse, contextlib, io, json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
NAMES = ['mAP', 'mAP50', 'mAP75', 'AP_small', 'AP_medium', 'AP_large',
         'AR1', 'AR10', 'AR100', 'AR_small', 'AR_medium', 'AR_large']
ap = argparse.ArgumentParser()
ap.add_argument('--gt', required=True); ap.add_argument('--dets', required=True)
ap.add_argument('--name', required=True); ap.add_argument('--out', required=True)
a = ap.parse_args()
with contextlib.redirect_stdout(io.StringIO()) as buf:
    g = COCO(a.gt); d = g.loadRes(a.dets); E = COCOeval(g, d, 'bbox'); E.evaluate(); E.accumulate(); E.summarize()
rec = {'name': a.name, 'dets': a.dets, 'gt': a.gt, **{k: float(v) for k, v in zip(NAMES, E.stats)}}
json.dump(rec, open(a.out, 'w'), indent=1)
print(buf.getvalue()); print(json.dumps(rec, indent=1))
