#!/usr/bin/env python3
"""
13_leakage_audit_plus.py - Reviewer 1 comment 6, answered completely.

R1.6: "The leakage audit is useful but incomplete ... the leakage-cleaned evaluation was applied
only to the DINO-Swin-B extension. The earlier AquaVisionNet, YOLOv8-m, and Faster R-CNN results
were not re-scored ... For a paper centered on leakage auditing, the authors should re-run these
models on the cleaned subset or clearly downgrade the strength of any cross-model conclusion."

The original audit (e1_leakage_audit.py + e1b_clean_eval.py) did one threshold, one direction
(val/test vs train, plus a test-vs-val check inside e1b), and one model. This extends it on every
axis that a reviewer can still poke at, and it reproduces the published numbers exactly rather than
replacing them.

What it adds
  1. All three leakage directions in one place: test->train, test->val, val->train.
  2. A threshold sweep (Hamming 0,2,5,8,10) instead of a single cutoff, so 11.6% is shown not to be
     an artefact of choosing 5.
  3. Exact SHA-256 duplicate detection - threshold-free, and stronger evidence than pHash if it
     fires.
  4. Per-source attribution, which is the part that turns this from a patch into a finding: if the
     leakage concentrates in FloW-Img, the mechanism is video-frame adjacency, and the fix is a
     group-aware split rather than a bigger test set.
  5. flagged id lists at every threshold, using the SAME union rule the paper used
     (near-dup in train OR near-dup in val), so `audit` reproduces the published 45 / 389 = 11.6%.

  `curve` then re-scores any set of detectors on the clean subset at every threshold, which is the
  actual answer to R1.6 and gives Table 8 plus a figure.

Usage
  python scripts\\13_leakage_audit_plus.py audit --ann-dir D:\\aquavisionnet\\data\\aquatax10\\annotations ^
      --out results\\leakage
  python scripts\\13_leakage_audit_plus.py curve --gt <test.json> --preds results\\preds ^
      --models yolov8m frcnn avn15 dino_plain dino_tta --audit results\\leakage --out results\\leakage

Needs: pillow, imagehash, numpy, pycocotools  (all already in the aquavision env).
"""
import argparse, csv, hashlib, json, os, sys, time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

THRESHOLDS = [0, 2, 5, 8, 10]
PAPER_THRESHOLD = 5
POPCNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)

SOURCE_RULES = [
    ("FloW-Img", ("flow_img", "flow-img", "flowimg")),
    ("AquaTrash", ("aquatrash", "aqua_trash", "aqua-trash")),
    ("AquaSurf-Malnad-223", ("malnad", "aquasurf")),
]


def source_of(im):
    """Prefer the annotation file's own 'source' field; fall back to the path."""
    s = (im.get("source") or "").strip().lower()
    if s:
        return {"flow_img": "FloW-Img", "aquatrash": "AquaTrash",
                "malnad": "AquaSurf-Malnad-223"}.get(s, s)
    low = str(im.get("file_name", "")).replace("/", "\\").lower()
    for name, keys in SOURCE_RULES:
        if any(k in low for k in keys):
            return name
    return "UNKNOWN"


def hamming(q, refs):
    """q: uint64 scalar. refs: uint64 array. -> uint8 distances."""
    x = np.bitwise_xor(refs, q)
    return POPCNT[x.view(np.uint8).reshape(-1, 8)].sum(1)


# ------------------------------------------------------------------ hashing
def load_split(p):
    d = json.loads(Path(p).read_text(encoding="utf-8"))
    return {im["id"]: im for im in d["images"]}


def hash_split(images, label, cache):
    """Returns ids(list), phash(uint64 array), sha(list), meta(list of dict). Uses a cache keyed
    by path+mtime+size so re-runs at other thresholds are instant."""
    from PIL import Image
    import imagehash
    ids, ph, sha, meta = [], [], [], []
    miss = 0
    t0 = time.time()
    for k, (iid, im) in enumerate(sorted(images.items())):
        fp = Path(im["file_name"])
        if not fp.exists():
            miss += 1
            continue
        try:
            st = fp.stat()
        except OSError:
            miss += 1
            continue
        key = f"{fp}|{int(st.st_mtime)}|{st.st_size}"
        ent = cache.get(key)
        if ent is None:
            try:
                with Image.open(fp) as img:
                    h = imagehash.phash(img, hash_size=8)
                d = hashlib.sha256()
                with open(fp, "rb") as fh:
                    for b in iter(lambda: fh.read(1 << 20), b""):
                        d.update(b)
                ent = {"p": str(h), "s": d.hexdigest()}
                cache[key] = ent
            except Exception:
                miss += 1
                continue
        ids.append(iid)
        ph.append(np.uint64(int(ent["p"], 16)))
        sha.append(ent["s"])
        meta.append({"id": iid, "source": source_of(im), "file": im["file_name"]})
        if (k + 1) % 250 == 0:
            print(f"    {label}: {k + 1}/{len(images)}  ({time.time() - t0:.0f}s)")
    print(f"  {label}: hashed {len(ids)}/{len(images)}"
          + (f"  ({miss} missing/unreadable)" if miss else ""))
    return ids, np.array(ph, dtype=np.uint64), sha, meta


def nearest(q_ids, q_ph, q_meta, r_ids, r_ph, r_meta):
    """Nearest reference for every query, by Hamming distance on the 64-bit pHash."""
    rows = []
    if len(r_ph) == 0:
        return rows
    for i, qid in enumerate(q_ids):
        d = hamming(q_ph[i], r_ph)
        j = int(np.argmin(d))
        rows.append({"query_id": qid, "query_source": q_meta[i]["source"],
                     "query_file": q_meta[i]["file"], "ref_id": r_ids[j],
                     "ref_source": r_meta[j]["source"], "ref_file": r_meta[j]["file"],
                     "hamming": int(d[j])})
    return rows


# ------------------------------------------------------------------ audit
def cmd_audit(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cache_p = out / "phash_cache.json"
    cache = json.loads(cache_p.read_text(encoding="utf-8")) if cache_p.exists() else {}

    ann = Path(a.ann_dir)
    splits = {s: load_split(ann / f"{s}.json") for s in ("train", "val", "test")}
    print("hashing (cached after the first run):")
    H = {s: hash_split(v, s, cache) for s, v in splits.items()}
    cache_p.write_text(json.dumps(cache), encoding="utf-8")

    tr_ids, tr_ph, tr_sha, tr_meta = H["train"]
    va_ids, va_ph, va_sha, va_meta = H["val"]
    te_ids, te_ph, te_sha, te_meta = H["test"]

    print("\nnearest-neighbour search:")
    pairs = {
        "test_vs_train": nearest(te_ids, te_ph, te_meta, tr_ids, tr_ph, tr_meta),
        "test_vs_val": nearest(te_ids, te_ph, te_meta, va_ids, va_ph, va_meta),
        "val_vs_train": nearest(va_ids, va_ph, va_meta, tr_ids, tr_ph, tr_meta),
    }
    for k, v in pairs.items():
        print(f"  {k}: {len(v)} queries")

    # ---- threshold sweep, using the paper's union rule for the test split
    print(f"\n{'threshold':>10}{'test~train':>12}{'test~val':>10}{'test UNION':>12}"
          f"{'% of test':>11}{'val~train':>11}")
    sweep = {}
    for t in THRESHOLDS:
        a_ids = {r["query_id"] for r in pairs["test_vs_train"] if r["hamming"] <= t}
        b_ids = {r["query_id"] for r in pairs["test_vs_val"] if r["hamming"] <= t}
        u = a_ids | b_ids
        v_ids = {r["query_id"] for r in pairs["val_vs_train"] if r["hamming"] <= t}
        n_test = len(te_ids)
        sweep[t] = {"test_vs_train": len(a_ids), "test_vs_val": len(b_ids),
                    "test_union": len(u), "test_pct": round(100 * len(u) / n_test, 2),
                    "val_vs_train": len(v_ids), "n_test": n_test, "n_val": len(va_ids),
                    "clean_n": n_test - len(u)}
        (out / f"flagged_ids_h{t}.txt").write_text("\n".join(str(i) for i in sorted(u)),
                                                   encoding="utf-8")
        print(f"{t:>10}{len(a_ids):>12}{len(b_ids):>10}{len(u):>12}"
              f"{sweep[t]['test_pct']:>10.1f}%{len(v_ids):>11}")

    paper = sweep[PAPER_THRESHOLD]
    print(f"\npublished figure to reproduce: 45 / 389 test images flagged = 11.6 %")
    print(f"this run at Hamming <= {PAPER_THRESHOLD}: {paper['test_union']} / {paper['n_test']} "
          f"= {paper['test_pct']} %   -> clean subset n = {paper['clean_n']}")
    if paper["test_union"] != 45:
        print("  !! does not match 45. Check that --ann-dir is the same annotation set the audit")
        print("     originally used, and that no images have moved. Report the new number, not 45.")
    # backward-compatible name used by run_all.py / revision_analysis.py
    (Path(a.flagged_out) if a.flagged_out else out.parent / "flagged_ids.txt").write_text(
        "\n".join(str(i) for i in sorted(
            {r["query_id"] for r in pairs["test_vs_train"] if r["hamming"] <= PAPER_THRESHOLD} |
            {r["query_id"] for r in pairs["test_vs_val"] if r["hamming"] <= PAPER_THRESHOLD})),
        encoding="utf-8")

    # ---- per-source attribution at the paper threshold
    print(f"\nper-source attribution at Hamming <= {PAPER_THRESHOLD} (test images):")
    bysrc_tot = Counter(m["source"] for m in te_meta)
    flagged_u = {int(x) for x in (out / f"flagged_ids_h{PAPER_THRESHOLD}.txt")
                 .read_text(encoding="utf-8").split()}
    bysrc_fl = Counter(m["source"] for m in te_meta if m["id"] in flagged_u)
    print(f"{'source':<24}{'flagged':>9}{'in test':>9}{'rate':>9}")
    attribution = {}
    for s in sorted(bysrc_tot):
        r = 100 * bysrc_fl[s] / bysrc_tot[s] if bysrc_tot[s] else float("nan")
        attribution[s] = {"flagged": bysrc_fl[s], "total": bysrc_tot[s], "rate_pct": round(r, 1)}
        print(f"{s:<24}{bysrc_fl[s]:>9}{bysrc_tot[s]:>9}{r:>8.1f}%")
    print("  If one source dominates this column, name the mechanism in Section 4.7: video-derived")
    print("  frames are adjacent in time, so a random split cannot separate them. The fix is a")
    print("  group-aware split, and that is a contribution, not an apology.")

    # ---- exact duplicates (threshold-free)
    print("\nexact SHA-256 duplicates across splits:")
    owner = defaultdict(set)
    where = defaultdict(list)
    for split, (ids, _ph, sha, meta) in (("train", H["train"]), ("val", H["val"]), ("test", H["test"])):
        for i, s in enumerate(sha):
            owner[s].add(split)
            where[s].append(f"{split}:{ids[i]}")
    cross = {s: sorted(v) for s, v in owner.items() if len(v) > 1}
    print(f"  {len(cross)} images are byte-identical across two or more splits")
    if cross:
        with (out / "exact_cross_split_duplicates.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["sha256", "splits", "occurrences"])
            for s, v in cross.items():
                w.writerow([s, "+".join(v), " | ".join(where[s])])
        print(f"  -> {out / 'exact_cross_split_duplicates.csv'}")
        print("  This is HARD leakage: identical bytes, no threshold involved. It belongs in the")
        print("  abstract if the count is non-trivial.")
    else:
        print("  none - the leakage is near-duplicate only, which is worth stating explicitly.")

    # ---- distance distribution, for the reviewer who asks 'why 5?'
    dist = {k: Counter(r["hamming"] for r in v) for k, v in pairs.items()}

    for name, rows in pairs.items():
        with (out / f"pairs_{name}.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(sorted(rows, key=lambda r: r["hamming"]))

    report = {"thresholds": THRESHOLDS, "paper_threshold": PAPER_THRESHOLD, "sweep": sweep,
              "attribution_at_paper_threshold": attribution,
              "exact_cross_split_duplicates": len(cross),
              "n_train": len(tr_ids), "n_val": len(va_ids), "n_test": len(te_ids),
              "distance_histogram": {k: {str(d): c for d, c in sorted(v.items())}
                                     for k, v in dist.items()}}
    (out / "leakage_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")

    md = out / "leakage_table.md"
    with md.open("w", encoding="utf-8") as fh:
        fh.write("# Leakage audit (extended for R1.6)\n\n")
        fh.write("## Threshold sensitivity\n\n")
        fh.write("| pHash Hamming | test~train | test~val | test flagged (union) | % of test | "
                 "clean n | val~train |\n|---|---|---|---|---|---|---|\n")
        for t in THRESHOLDS:
            s = sweep[t]
            fh.write(f"| <= {t} | {s['test_vs_train']} | {s['test_vs_val']} | {s['test_union']} | "
                     f"{s['test_pct']} % | {s['clean_n']} | {s['val_vs_train']} |\n")
        fh.write(f"\n## Source attribution at Hamming <= {PAPER_THRESHOLD}\n\n")
        fh.write("| source | flagged | test images | flagged rate |\n|---|---|---|---|\n")
        for s, v in sorted(attribution.items()):
            fh.write(f"| {s} | {v['flagged']} | {v['total']} | {v['rate_pct']} % |\n")
        fh.write(f"\nExact byte-identical images across splits: **{len(cross)}**.\n")
    print(f"\nwrote {md} and {out / 'leakage_report.json'}")
    print("Next: python scripts\\13_leakage_audit_plus.py curve ...")


# ------------------------------------------------------------------ curve
def coco_eval(gt_dict, dets):
    import contextlib, io
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
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
    prec = E.eval["precision"]
    for k, cid in enumerate(E.params.catIds):
        p = prec[0, :, k, 0, 2]
        p = p[p > -1]
        out[names[cid]] = float(p.mean()) if p.size else float("nan")
    return out


def subset(gt, dets, keep):
    keep = set(keep)
    g = dict(gt)
    g["images"] = [i for i in gt["images"] if i["id"] in keep]
    g["annotations"] = [x for x in gt["annotations"] if x["image_id"] in keep]
    return g, [x for x in dets if x["image_id"] in keep]


def cmd_curve(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    gt = json.loads(Path(a.gt).read_text(encoding="utf-8"))
    all_ids = {i["id"] for i in gt["images"]}
    rows = []
    for m in a.models:
        p = Path(a.preds) / f"{m}_test.json"
        if not p.exists():
            print(f"  [skip] {m}: {p} not found")
            continue
        dets = json.loads(p.read_text(encoding="utf-8"))
        full = coco_eval(gt, dets)
        for t in THRESHOLDS:
            fp = Path(a.audit) / f"flagged_ids_h{t}.txt"
            if not fp.exists():
                continue
            fl = {int(x) for x in fp.read_text(encoding="utf-8").split()} & all_ids
            gc, dc = subset(gt, dets, all_ids - fl)
            gf, df = subset(gt, dets, fl)
            clean = coco_eval(gc, dc)
            flag = coco_eval(gf, df) if fl else {"mAP50": float("nan")}
            rows.append({"model": m, "threshold": t, "n_flagged": len(fl),
                         "n_clean": len(all_ids) - len(fl),
                         "full_mAP50": full["mAP50"], "clean_mAP50": clean["mAP50"],
                         "delta": clean["mAP50"] - full["mAP50"],
                         "flagged_only_mAP50": flag["mAP50"]})
            print(f"  {m:<11} h<={t:<3} clean n={len(all_ids) - len(fl):<4} "
                  f"full {full['mAP50']:.3f} -> clean {clean['mAP50']:.3f} "
                  f"({clean['mAP50'] - full['mAP50']:+.3f})  flagged-only {flag['mAP50']:.3f}")
    if not rows:
        sys.exit("no predictions found - run the exports first")
    csv_p = out / "leakage_curve.csv"
    with csv_p.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    md = out / "leakage_curve.md"
    with md.open("w", encoding="utf-8") as fh:
        fh.write("# R1.6 - every detector re-scored on the leakage-cleaned subset\n\n")
        fh.write(f"## Table 8 (paper threshold, Hamming <= {PAPER_THRESHOLD})\n\n")
        fh.write("| detector | full (n=all) | clean | delta | flagged-only |\n|---|---|---|---|---|\n")
        for r in rows:
            if r["threshold"] != PAPER_THRESHOLD:
                continue
            fh.write(f"| {r['model']} | {r['full_mAP50']:.3f} | {r['clean_mAP50']:.3f} | "
                     f"{r['delta']:+.3f} | {r['flagged_only_mAP50']:.3f} |\n")
        fh.write("\n## Threshold sensitivity of the cleaned score\n\n")
        fh.write("| detector | " + " | ".join(f"h<={t}" for t in THRESHOLDS) + " |\n")
        fh.write("|---" * (len(THRESHOLDS) + 1) + "|\n")
        for m in a.models:
            mr = {r["threshold"]: r for r in rows if r["model"] == m}
            if not mr:
                continue
            fh.write(f"| {m} | " + " | ".join(
                f"{mr[t]['clean_mAP50']:.3f}" if t in mr else "-" for t in THRESHOLDS) + " |\n")
        fh.write("\nA ranking that survives every column is a ranking the leakage did not create. "
                 "A ranking that flips somewhere must be reported as flipping.\n")
    print(f"\nwrote {csv_p} and {md}")

    # the check that actually answers the reviewer
    at5 = [r for r in rows if r["threshold"] == PAPER_THRESHOLD]
    if at5:
        order_full = [r["model"] for r in sorted(at5, key=lambda r: -r["full_mAP50"])]
        order_clean = [r["model"] for r in sorted(at5, key=lambda r: -r["clean_mAP50"])]
        print(f"\nranking on the full test set : {' > '.join(order_full)}")
        print(f"ranking on the clean subset  : {' > '.join(order_clean)}")
        print("SAME ORDER - the cross-model conclusion survives the leakage correction."
              if order_full == order_clean else
              "*** ORDER CHANGES. The cross-model conclusion must be downgraded, exactly as R1.6 "
              "warned. Say so plainly in Section 4.7 and in the letter.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("audit")
    p.add_argument("--ann-dir", required=True, help="folder with train.json / val.json / test.json")
    p.add_argument("--out", default=r"results\leakage")
    p.add_argument("--flagged-out", default=None,
                   help="where to also write the paper-threshold list (default results\\flagged_ids.txt)")
    p.set_defaults(f=cmd_audit)
    p = sp.add_parser("curve")
    p.add_argument("--gt", required=True)
    p.add_argument("--preds", default=r"results\preds")
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--audit", default=r"results\leakage")
    p.add_argument("--out", default=r"results\leakage")
    p.set_defaults(f=cmd_curve)
    a = ap.parse_args()
    a.f(a)


if __name__ == "__main__":
    main()
