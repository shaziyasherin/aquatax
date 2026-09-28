#!/usr/bin/env python3
"""
15_prepare_corpus_B.py - make a writable, verified copy of the corpus of record (Build B) and
produce the exact per-class counts the manuscript needs.

Why a copy: `avn_full_testsplit_standalone.py` saves its checkpoint next to `--data`, and we never
write into the dataset directory. Everything else about the training runs stays identical; only
`--data` changes.

What it verifies, before anything is trained on it:
  * image and box counts per split
  * every image unique within its split (Build A failed this: 392 duplicate training images)
  * no byte-identical image shared between train and val/test, and it names any that exist
  * exact per-class box counts per split -> Table 2 / Table 3, which answers R4.5's objection that
    the plastic count "~5,538" looks like it was obtained by subtraction

  python scripts\\15_prepare_corpus_B.py --src D:\\aquavisionnet\\data\\aquatax10\\annotations ^
      --dst D:\\aquavisionnet\\revision\\corpus_B --cache results\\leakage\\phash_cache.json ^
      --out results\\corpus_B_report.md
"""
import argparse, hashlib, json, os, shutil
from collections import Counter, defaultdict
from pathlib import Path

SPLITS = ("train", "val", "test")


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_cache(p):
    if not p or not Path(p).exists():
        return {}
    raw = json.loads(Path(p).read_text(encoding="utf-8"))
    out = {}
    for k, v in raw.items():
        path = k.rsplit("|", 2)[0]
        if v.get("s"):
            out[os.path.normcase(os.path.abspath(path))] = v["s"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--cache", default=r"results\leakage\phash_cache.json")
    ap.add_argument("--out", default=r"results\corpus_B_report.md")
    a = ap.parse_args()

    src, dst = Path(a.src), Path(a.dst)
    dst.mkdir(parents=True, exist_ok=True)
    cache = load_cache(a.cache)
    print(f"reusing {len(cache)} cached hashes")

    data, shas, ok = {}, {}, True
    for s in SPLITS:
        sp = src / f"{s}.json"
        if not sp.exists():
            raise SystemExit(f"missing {sp}")
        dp = dst / f"{s}.json"
        shutil.copy2(sp, dp)
        d = json.loads(dp.read_text(encoding="utf-8"))
        data[s] = d
        h = []
        miss = 0
        for im in d["images"]:
            fp = Path(im["file_name"])
            key = os.path.normcase(os.path.abspath(str(fp)))
            v = cache.get(key)
            if v is None:
                if not fp.exists():
                    miss += 1
                    continue
                v = sha256(fp)
            h.append(v)
        shas[s] = h
        dup = len(h) - len(set(h))
        flag = "" if dup == 0 else f"   *** {dup} DUPLICATE IMAGES INSIDE THIS SPLIT"
        if dup or miss:
            ok = False
        print(f"  {s:<6} {len(d['images']):>5} images  {len(set(h)):>5} unique  "
              f"{len(d['annotations']):>5} boxes"
              + (f"  {miss} missing on disk" if miss else "") + flag)

    print("\ncross-split byte-identical images:")
    pairs = []
    for x, y in (("train", "val"), ("train", "test"), ("val", "test")):
        sx, sy = set(shas[x]), set(shas[y])
        common = sx & sy
        print(f"  {x} and {y}: {len(common)}")
        for c in common:
            ix = [im for im, s in zip(data[x]["images"], shas[x]) if s == c]
            iy = [im for im, s in zip(data[y]["images"], shas[y]) if s == c]
            for p in ix:
                for q in iy:
                    pairs.append((x, p["id"], p["file_name"], y, q["id"], q["file_name"], c))
    if pairs:
        print(f"  -> {len(pairs)} pair(s); listed in the report. Report these in Section 4.7 "
              f"rather than dropping them silently.")

    names = {c["id"]: c["name"] for c in data["train"]["categories"]}
    order = [names[k] for k in sorted(names)]
    per = {s: Counter(names[x["category_id"]] for x in data[s]["annotations"]) for s in SPLITS}
    imgs_with = {s: len({x["image_id"] for x in data[s]["annotations"]}) for s in SPLITS}

    print(f"\nper-class boxes (counted, not inferred):")
    print(f"{'split':<8}{'images':>8}{'with boxes':>12}{'boxes':>8}" + "".join(f"{c:>10}" for c in order))
    tot = Counter()
    for s in SPLITS:
        tot += per[s]
        print(f"{s:<8}{len(data[s]['images']):>8}{imgs_with[s]:>12}"
              f"{len(data[s]['annotations']):>8}" + "".join(f"{per[s][c]:>10}" for c in order))
    total_boxes = sum(len(data[s]["annotations"]) for s in SPLITS)
    print(f"{'TOTAL':<8}{sum(len(data[s]['images']) for s in SPLITS):>8}{'':>12}"
          f"{total_boxes:>8}" + "".join(f"{tot[c]:>10}" for c in order))

    negatives = {s: len(data[s]["images"]) - imgs_with[s] for s in SPLITS}
    print(f"\nimages with no boxes (negatives, kept as background): "
          + ", ".join(f"{s} {negatives[s]}" for s in SPLITS))

    rep = Path(a.out)
    rep.parent.mkdir(parents=True, exist_ok=True)
    with rep.open("w", encoding="utf-8") as fh:
        fh.write("# Corpus of record (Build B), verified\n\n")
        fh.write(f"Source: `{src}`  ->  working copy: `{dst}`\n\n")
        fh.write("## Split composition\n\n")
        fh.write("| split | images | unique images | images with boxes | negatives | boxes | "
                 + " | ".join(order) + " |\n")
        fh.write("|" + "---|" * (6 + len(order)) + "\n")
        for s in SPLITS:
            fh.write(f"| {s} | {len(data[s]['images'])} | {len(set(shas[s]))} | {imgs_with[s]} | "
                     f"{negatives[s]} | {len(data[s]['annotations'])} | "
                     + " | ".join(str(per[s][c]) for c in order) + " |\n")
        fh.write(f"| **total** | {sum(len(data[s]['images']) for s in SPLITS)} | | | | "
                 f"{total_boxes} | " + " | ".join(str(tot[c]) for c in order) + " |\n")
        fh.write("\nEvery count above is read from the annotation files. None is obtained by "
                 "subtraction (R4.5).\n")
        fh.write("\n## Byte-level duplicate check\n\n")
        if pairs:
            fh.write("| split A | id | file | split B | id | file |\n|---|---|---|---|---|---|\n")
            for x, i, f1, y, j, f2, _ in pairs:
                fh.write(f"| {x} | {i} | `{Path(f1).name}` | {y} | {j} | `{Path(f2).name}` |\n")
        else:
            fh.write("No image is byte-identical across splits.\n")
        fh.write("\nWithin-split duplicates: "
                 + ", ".join(f"{s} {len(shas[s]) - len(set(shas[s]))}" for s in SPLITS) + ".\n")
    print(f"\nwrote {rep}")
    print("\nOK - safe to train on this copy." if ok else
          "\n*** Problems above. Do not train until they are understood.")


if __name__ == "__main__":
    main()
