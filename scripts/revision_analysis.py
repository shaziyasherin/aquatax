#!/usr/bin/env python3
"""
revision_analysis.py  --  statistics for the HAZADV-D-26-02212 revision.

Answers reviewer items that need numbers but NO retraining:
  R1.6  clean     : re-score any detector on the leakage-cleaned test subset
  R1.9  bootstrap : image-level bootstrap 95% CIs for mAP@0.5, mAP@[.5:.95], per-class AP@0.5
        compare   : paired bootstrap of the difference between two detectors
  R1.10 counts    : per-image debris-count error at an operating threshold chosen on val

All inputs are standard COCO files:
  --gt    COCO ground-truth json (test split, 389 images)
  --dets  COCO results json: [{"image_id", "category_id", "bbox":[x,y,w,h], "score"}, ...]
  --flagged  text file, one test image_id per line (the 45 pHash-flagged images)

Every number this script prints comes from the files you give it. Nothing is estimated.

Examples
  python revision_analysis.py clean     --gt test.json --dets frcnn_test.json --flagged flagged_ids.txt
  python revision_analysis.py bootstrap --gt test.json --dets dino_tta_test.json --B 1000 --seed 20260729
  python revision_analysis.py bootstrap --gt test.json --dets frcnn_test.json --flagged flagged_ids.txt --subset clean
  python revision_analysis.py compare   --gt test.json --dets frcnn_test.json --dets2 avn_test.json --B 1000
  python revision_analysis.py counts    --gt test.json --dets dino_test.json --val-gt val.json --val-dets dino_val.json
"""
import argparse, contextlib, io, json, math, sys
from collections import defaultdict

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


# ---------------------------------------------------------------- io helpers
def load_json(p):
    with open(p) as f:
        return json.load(f)


def load_ids(p):
    with open(p) as f:
        return {int(x.strip()) for x in f if x.strip()}


def subset(gt, dets, keep_ids):
    keep_ids = set(keep_ids)
    g = dict(gt)
    g["images"] = [im for im in gt["images"] if im["id"] in keep_ids]
    g["annotations"] = [a for a in gt["annotations"] if a["image_id"] in keep_ids]
    d = [x for x in dets if x["image_id"] in keep_ids]
    return g, d


# ---------------------------------------------------------------- evaluation
def coco_eval(gt_dict, dets):
    """Return mAP@0.5, mAP@[.5:.95] and per-class AP@0.5 (NaN if a class has no GT)."""
    names = {c["id"]: c["name"] for c in gt_dict["categories"]}
    out = {"mAP50": float("nan"), "mAP": float("nan")}
    out.update({n: float("nan") for n in names.values()})
    if not dets:
        return out
    with contextlib.redirect_stdout(io.StringIO()):
        g = COCO()
        g.dataset = gt_dict
        g.createIndex()
        d = g.loadRes([dict(x) for x in dets])
        E = COCOeval(g, d, "bbox")
        E.evaluate(); E.accumulate(); E.summarize()
    out["mAP"], out["mAP50"] = float(E.stats[0]), float(E.stats[1])
    prec = E.eval["precision"]                      # [T, R, K, A, M]
    for k, cid in enumerate(E.params.catIds):
        p = prec[0, :, k, 0, 2]                     # IoU=0.50, all areas, maxDets=100
        p = p[p > -1]
        out[names[cid]] = float(p.mean()) if p.size else float("nan")
    return out


# ---------------------------------------------------------------- bootstrap
class Resampler:
    """Image-level bootstrap. Duplicated images get fresh ids so COCOeval treats them as distinct."""

    def __init__(self, gt, det_lists):
        self.gt = gt
        self.images = gt["images"]
        self.ann_by = defaultdict(list)
        for a in gt["annotations"]:
            self.ann_by[a["image_id"]].append(a)
        self.det_by = []
        for dets in det_lists:
            m = defaultdict(list)
            for x in dets:
                m[x["image_id"]].append(x)
            self.det_by.append(m)

    def draw(self, rng):
        n = len(self.images)
        pick = rng.integers(0, n, size=n)
        imgs, anns = [], []
        dets = [[] for _ in self.det_by]
        aid = 1
        for new_id, idx in enumerate(pick, start=1):
            im = dict(self.images[idx]); old = im["id"]; im["id"] = new_id
            imgs.append(im)
            for a in self.ann_by[old]:
                b = dict(a); b["id"] = aid; b["image_id"] = new_id; aid += 1
                anns.append(b)
            for j, m in enumerate(self.det_by):
                for x in m[old]:
                    e = dict(x); e["image_id"] = new_id; e.pop("id", None)
                    dets[j].append(e)
        g = {"images": imgs, "annotations": anns, "categories": self.gt["categories"]}
        return g, dets


def ci(vals):
    v = np.asarray([x for x in vals if not math.isnan(x)])
    if v.size == 0:
        return float("nan"), float("nan"), 0
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)), int(v.size)


def fmt(x):
    return "  nan" if math.isnan(x) else f"{x:.3f}"


def cmd_bootstrap(a):
    gt, dets = load_json(a.gt), load_json(a.dets)
    if a.subset == "clean":
        flagged = load_ids(a.flagged)
        gt, dets = subset(gt, dets, {im["id"] for im in gt["images"]} - flagged)
    point = coco_eval(gt, dets)
    rs, rng = Resampler(gt, [dets]), np.random.default_rng(a.seed)
    draws = defaultdict(list)
    for b in range(a.B):
        g, (d,) = rs.draw(rng)
        for k, v in coco_eval(g, d).items():
            draws[k].append(v)
        if (b + 1) % 100 == 0:
            print(f"  ... {b + 1}/{a.B}", file=sys.stderr)
    print(f"\nImage-level bootstrap, B={a.B}, seed={a.seed}, n_images={len(gt['images'])}, subset={a.subset}")
    print(f"{'metric':<10}{'point':>8}{'95% CI':>22}{'valid draws':>14}")
    rec = {}
    for k, v in point.items():
        lo, hi, nv = ci(draws[k])
        rec[k] = {'point': v, 'lo': lo, 'hi': hi, 'valid_draws': nv}
        print(f"{k:<10}{fmt(v):>8}   [{fmt(lo)}, {fmt(hi)}]{nv:>12}/{a.B}")
    if a.json_out:
        json.dump({'dets': a.dets, 'gt': a.gt, 'subset': a.subset, 'B': a.B, 'seed': a.seed,
                   'n_images': len(gt['images']), 'metrics': rec}, open(a.json_out, 'w'), indent=1)
    print("\nA class whose 'valid draws' < B had zero GT boxes in some resamples; "
          "report its CI with that caveat (expected for glass, n=7).")


def cmd_compare(a):
    gt, d1, d2 = load_json(a.gt), load_json(a.dets), load_json(a.dets2)
    if a.subset == "clean":
        keep = {im["id"] for im in gt["images"]} - load_ids(a.flagged)
        gt, d1 = subset(gt, d1, keep)
        _, d2 = subset(load_json(a.gt), d2, keep)
    p1, p2 = coco_eval(gt, d1), coco_eval(gt, d2)
    rs, rng = Resampler(gt, [d1, d2]), np.random.default_rng(a.seed)
    diffs = defaultdict(list)
    for _ in range(a.B):
        g, (x1, x2) = rs.draw(rng)
        e1, e2 = coco_eval(g, x1), coco_eval(g, x2)
        for k in e1:
            diffs[k].append(e1[k] - e2[k])
    print(f"\nPaired bootstrap (A = {a.dets}, B = {a.dets2}), B={a.B}")
    print(f"{'metric':<10}{'A-B':>8}{'95% CI of diff':>24}{'P(diff<=0)':>13}")
    rec = {}
    for k in p1:
        lo, hi, nv = ci(diffs[k])
        v = np.asarray([x for x in diffs[k] if not math.isnan(x)])
        ple0 = float((v <= 0).mean()) if v.size else float("nan")
        rec[k] = {'diff': p1[k] - p2[k], 'lo': lo, 'hi': hi, 'p_le0': ple0}
        print(f"{k:<10}{fmt(p1[k] - p2[k]):>8}     [{fmt(lo)}, {fmt(hi)}]{fmt(ple0):>11}")
    if a.json_out:
        json.dump({'A': a.dets, 'B': a.dets2, 'subset': a.subset, 'B_resamples': a.B, 'seed': a.seed,
                   'metrics': rec}, open(a.json_out, 'w'), indent=1)


def cmd_clean(a):
    gt, dets = load_json(a.gt), load_json(a.dets)
    flagged = load_ids(a.flagged)
    all_ids = {im["id"] for im in gt["images"]}
    missing = flagged - all_ids
    if missing:
        sys.exit(f"flagged ids not in GT: {sorted(missing)[:10]} ...")
    full = coco_eval(gt, dets)
    g_c, d_c = subset(gt, dets, all_ids - flagged)
    g_f, d_f = subset(gt, dets, flagged)
    clean, flag = coco_eval(g_c, d_c), coco_eval(g_f, d_f)
    print(f"\nfull n={len(all_ids)} | clean n={len(all_ids - flagged)} | flagged n={len(flagged)}")
    print(f"{'metric':<10}{'full':>8}{'clean':>8}{'delta':>8}{'flagged-only':>14}")
    for k in full:
        dl = clean[k] - full[k]
        print(f"{k:<10}{fmt(full[k]):>8}{fmt(clean[k]):>8}{fmt(dl):>8}{fmt(flag[k]):>14}")
    print("\n'flagged-only' is the effect-size check: if near-duplicates inflate scores, "
          "AP on the flagged images should sit well above AP on the clean images.")


# ---------------------------------------------------------------- counting (R1.10)
def per_image_counts(gt, dets, thr):
    cats = [c["id"] for c in gt["categories"]]
    ids = [im["id"] for im in gt["images"]]
    G = {i: defaultdict(int) for i in ids}
    P = {i: defaultdict(int) for i in ids}
    for x in gt["annotations"]:
        G[x["image_id"]][x["category_id"]] += 1
    for x in dets:
        if x["score"] >= thr and x["image_id"] in P:
            P[x["image_id"]][x["category_id"]] += 1
    return ids, cats, G, P


def count_errors(gt, dets, thr):
    ids, cats, G, P = per_image_counts(gt, dets, thr)
    tot_err = [sum(P[i].values()) - sum(G[i].values()) for i in ids]
    res = {"MAE_total": float(np.mean(np.abs(tot_err))), "bias_total": float(np.mean(tot_err))}
    for c in cats:
        gsum = sum(G[i][c] for i in ids); psum = sum(P[i][c] for i in ids)
        res[c] = (gsum, psum)
    return res


def cmd_counts(a):
    gt, dets = load_json(a.gt), load_json(a.dets)
    thr = a.thr
    if thr is None:
        vg, vd = load_json(a.val_gt), load_json(a.val_dets)
        grid = np.round(np.arange(0.05, 0.96, 0.05), 2)
        maes = [count_errors(vg, vd, t)["MAE_total"] for t in grid]
        thr = float(grid[int(np.argmin(maes))])
        print(f"operating threshold chosen on VAL by min per-image count MAE: {thr}")
    r = count_errors(gt, dets, thr)
    names = {c["id"]: c["name"] for c in gt["categories"]}
    print(f"\nTest split, score >= {thr}")
    print(f"per-image total-count MAE = {r['MAE_total']:.3f} items; mean signed bias = {r['bias_total']:+.3f} items/image")
    print(f"{'class':<10}{'GT total':>10}{'pred total':>12}{'ratio':>8}")
    for c, n in names.items():
        g, p = r[c]
        print(f"{n:<10}{g:>10}{p:>12}{(p / g if g else float('nan')):>8.2f}")
    print("\nratio < 1 = systematic undercount (missed detections dominate); > 1 = overcount (false positives dominate).")


# ---------------------------------------------------------------- cli
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)

    def common(p, flagged_required=False):
        p.add_argument("--gt", required=True)
        p.add_argument("--dets", required=True)
        p.add_argument("--flagged", required=flagged_required)
        p.add_argument("--json-out", default=None, help="also write the numbers as JSON")

    p = sp.add_parser("clean"); common(p, True); p.set_defaults(f=cmd_clean)
    for name, fn in (("bootstrap", cmd_bootstrap), ("compare", cmd_compare)):
        p = sp.add_parser(name); common(p)
        p.add_argument("--B", type=int, default=1000)
        p.add_argument("--seed", type=int, default=20260729)
        p.add_argument("--subset", choices=["full", "clean"], default="full")
        if name == "compare":
            p.add_argument("--dets2", required=True)
        p.set_defaults(f=fn)
    p = sp.add_parser("counts"); common(p)
    p.add_argument("--thr", type=float, default=None)
    p.add_argument("--val-gt"); p.add_argument("--val-dets")
    p.set_defaults(f=cmd_counts)

    a = ap.parse_args()
    if getattr(a, "subset", "full") == "clean" and not a.flagged:
        ap.error("--subset clean needs --flagged")
    if a.cmd == "counts" and a.thr is None and not (a.val_gt and a.val_dets):
        ap.error("counts needs --thr or both --val-gt and --val-dets")
    a.f(a)


if __name__ == "__main__":
    main()
