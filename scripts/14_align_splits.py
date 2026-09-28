#!/usr/bin/env python3
"""
14_align_splits.py - reconcile the two annotation sets before anything is put in Table 8.

The matched-protocol detectors (Faster R-CNN, AquaVisionNet, YOLOv8-m) were trained and evaluated
against D:\\aquavisionnet\\avn_out\\{train,val,test}.json. DINO-Swin-B and the leakage audit used
D:\\aquavisionnet\\data\\aquatax10\\annotations\\{...}.json. Both test files hold 389 images, but
their image_id numbering does not agree, so predictions from one cannot be scored against the other
and the flagged-id list from the audit does not apply to the matched detectors as-is.

This decides, from the image bytes rather than from ids or names, whether the two files describe the
SAME 389 images under different ids (fixable: remap) or DIFFERENT images (not fixable: the paper has
two different test splits and Table 6 cannot mix them).

  diagnose  python scripts\\14_align_splits.py diagnose ^
                --a D:\\aquavisionnet\\avn_out\\test.json ^
                --b D:\\aquavisionnet\\data\\aquatax10\\annotations\\test.json ^
                --out results\\align --cache results\\leakage\\phash_cache.json

  remap     python scripts\\14_align_splits.py remap --map results\\align\\idmap_a_to_b.json ^
                --preds results\\preds\\frcnn_test.json --out results\\preds\\frcnn_test_bids.json

  ids       python scripts\\14_align_splits.py ids --map results\\align\\idmap_b_to_a.json ^
                --in results\\flagged_ids.txt --out results\\flagged_ids_avn.txt

`diagnose` reuses the SHA-256 values already computed by 13_leakage_audit_plus.py when you point
--cache at its phash_cache.json, so it costs seconds rather than re-hashing 778 images.
"""
import argparse, hashlib, json, os
from collections import Counter
from pathlib import Path


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_cache(p):
    """phash_cache.json from 13_leakage_audit_plus.py: {"path|mtime|size": {"p":..., "s": sha}}"""
    if not p or not Path(p).exists():
        return {}
    raw = json.loads(Path(p).read_text(encoding="utf-8"))
    out = {}
    for k, v in raw.items():
        path = k.rsplit("|", 2)[0]
        out[os.path.normcase(os.path.abspath(path))] = v.get("s")
    return {k: v for k, v in out.items() if v}


def hash_images(coco, cache, label):
    rows, missing = [], 0
    for im in coco["images"]:
        fp = Path(im["file_name"])
        key = os.path.normcase(os.path.abspath(str(fp)))
        s = cache.get(key)
        if s is None:
            if not fp.exists():
                missing += 1
                continue
            s = sha256(fp)
            cache[key] = s
        rows.append({"id": im["id"], "sha": s, "path": str(fp),
                     "base": Path(im["file_name"]).name.lower(),
                     "source": im.get("source", "")})
    print(f"  {label}: {len(rows)} images hashed"
          + (f", {missing} missing on disk" if missing else ""))
    return rows


def cmd_diagnose(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    A = json.loads(Path(a.a).read_text(encoding="utf-8"))
    B = json.loads(Path(a.b).read_text(encoding="utf-8"))
    cache = load_cache(a.cache)
    print(f"reusing {len(cache)} cached hashes" if cache else "no cache - hashing from scratch")
    ra = hash_images(A, cache, "A " + Path(a.a).name)
    rb = hash_images(B, cache, "B " + Path(a.b).name)

    sa = {r["sha"]: r for r in ra}
    sb = {r["sha"]: r for r in rb}
    both = set(sa) & set(sb)
    only_a, only_b = set(sa) - set(sb), set(sb) - set(sa)

    dup_a = [s for s, c in Counter(r["sha"] for r in ra).items() if c > 1]
    dup_b = [s for s, c in Counter(r["sha"] for r in rb).items() if c > 1]

    print(f"\nA: {len(ra)} images, {len(A['annotations'])} boxes")
    print(f"B: {len(rb)} images, {len(B['annotations'])} boxes")
    print(f"\nsame image bytes in both : {len(both)}")
    print(f"only in A                : {len(only_a)}")
    print(f"only in B                : {len(only_b)}")
    if dup_a or dup_b:
        print(f"!! duplicate bytes within A: {len(dup_a)}, within B: {len(dup_b)} "
              f"- an image appearing twice in one split makes the id map ambiguous")

    same_ids = sum(1 for s in both if sa[s]["id"] == sb[s]["id"])
    print(f"of the shared images, {same_ids} already carry the same image_id")

    verdict = None
    if len(both) == len(ra) == len(rb):
        verdict = "SAME_IMAGES"
        print("\nVERDICT: the two files describe the SAME test images under different ids.")
        print("         This is a bookkeeping difference. Remap and Table 8 is valid.")
    elif len(both) == 0:
        verdict = "DISJOINT"
        print("\nVERDICT: the two files share NO images. These are different test splits.")
        print("         Table 6 and Table 8 cannot mix detectors across them. Stop and decide")
        print("         which split the paper reports, then re-score everything on that one.")
    else:
        verdict = "PARTIAL"
        print(f"\nVERDICT: PARTIAL overlap ({len(both)} of {max(len(ra), len(rb))}).")
        print("         The splits are not the same. Any cross-model table built from both is")
        print("         invalid. Decide which split is authoritative before going further.")

    ab = {sa[s]["id"]: sb[s]["id"] for s in both}
    ba = {sb[s]["id"]: sa[s]["id"] for s in both}
    (out / "idmap_a_to_b.json").write_text(json.dumps(ab, indent=0), encoding="utf-8")
    (out / "idmap_b_to_a.json").write_text(json.dumps(ba, indent=0), encoding="utf-8")

    rep = {"verdict": verdict, "a": str(a.a), "b": str(a.b),
           "n_a": len(ra), "n_b": len(rb), "boxes_a": len(A["annotations"]),
           "boxes_b": len(B["annotations"]), "shared": len(both),
           "only_a": len(only_a), "only_b": len(only_b),
           "ids_already_equal": same_ids,
           "dup_bytes_within_a": len(dup_a), "dup_bytes_within_b": len(dup_b)}
    (out / "align_report.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")

    with (out / "unmatched.txt").open("w", encoding="utf-8") as fh:
        for s in sorted(only_a):
            fh.write(f"A-only\t{sa[s]['id']}\t{sa[s]['path']}\n")
        for s in sorted(only_b):
            fh.write(f"B-only\t{sb[s]['id']}\t{sb[s]['path']}\n")
    print(f"\nwrote {out / 'idmap_a_to_b.json'}, {out / 'idmap_b_to_a.json'}, "
          f"{out / 'align_report.json'}, {out / 'unmatched.txt'}")

    # box-level agreement, for the shared images only
    if both:
        ann_a, ann_b = Counter(), Counter()
        for x in A["annotations"]:
            ann_a[x["image_id"]] += 1
        for x in B["annotations"]:
            ann_b[x["image_id"]] += 1
        diff = [s for s in both if ann_a[sa[s]["id"]] != ann_b[sb[s]["id"]]]
        print(f"\nshared images whose box COUNT differs between A and B: {len(diff)}")
        if diff:
            print("  the two files disagree about the annotations, not only the ids -")
            print("  check which annotation version the paper's numbers came from")


def cmd_matrix(a):
    """Cross-tabulate two corpus builds by image bytes: which split of A corresponds to which
    split of B, whether either build leaks train into its own test, and - decisively - whether
    one build's TRAIN contains the other build's TEST (which would make cross-scoring impossible)."""
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cache = load_cache(a.cache)
    print(f"reusing {len(cache)} cached hashes")
    sets, boxes, nimg, ndup = {}, {}, {}, {}
    for tag, d in (("A", a.a_dir), ("B", a.b_dir)):
        for sp in ("train", "val", "test"):
            p = Path(d) / f"{sp}.json"
            if not p.exists():
                print(f"  missing {p}")
                continue
            coco = json.loads(p.read_text(encoding="utf-8"))
            rows = hash_images(coco, cache, f"{tag}/{sp}")
            key = f"{tag}/{sp}"
            sets[key] = {r["sha"] for r in rows}
            nimg[key] = len(rows)
            ndup[key] = len(rows) - len(sets[key])
            boxes[key] = len(coco["annotations"])

    keys = list(sets)
    print(f"\n{'file':<10}{'images':>8}{'unique':>8}{'dup':>6}{'boxes':>8}")
    for k in keys:
        print(f"{k:<10}{nimg[k]:>8}{len(sets[k]):>8}{ndup[k]:>6}{boxes[k]:>8}")

    ak = [k for k in keys if k.startswith("A/")]
    bk = [k for k in keys if k.startswith("B/")]
    print("\noverlap by image bytes (rows = A build, cols = B build):")
    print(f"{'':<10}" + "".join(f"{k:>12}" for k in bk))
    for i in ak:
        print(f"{i:<10}" + "".join(f"{len(sets[i] & sets[j]):>12}" for j in bk))

    print("\nwithin-build train/test contamination (should be 0):")
    for tag in ("A", "B"):
        for pair in (("train", "test"), ("train", "val")):
            x, y = f"{tag}/{pair[0]}", f"{tag}/{pair[1]}"
            if x in sets and y in sets:
                n = len(sets[x] & sets[y])
                print(f"  {x} and {y} = {n}" + ("   *** SELF-LEAKAGE" if n else ""))

    print("\ncross-build contamination - decides whether either build can score the other's models:")
    for x, y in (("A/train", "B/test"), ("B/train", "A/test")):
        if x in sets and y in sets:
            n = len(sets[x] & sets[y])
            pct = 100 * n / max(1, len(sets[y]))
            print(f"  {x} and {y} = {n}  ({pct:.1f}% of {y})"
                  + ("   *** models trained on " + x + " CANNOT be evaluated on " + y if n else "   clean"))

    rep = {"images": nimg, "unique": {k: len(v) for k, v in sets.items()},
           "within_file_byte_duplicates": ndup, "boxes": boxes,
           "overlap": {f"{i}|{j}": len(sets[i] & sets[j]) for i in ak for j in bk},
           "self": {f"{t}/train|{t}/{s}": len(sets.get(f"{t}/train", set()) & sets.get(f"{t}/{s}", set()))
                    for t in ("A", "B") for s in ("val", "test")}}
    (out / "corpus_matrix.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(f"\nwrote {out / 'corpus_matrix.json'}")
    print("\nRead the matrix like this: a large number off the diagonal means A's split X is")
    print("B's split Y under a different name. A large A/train and B/test means the matched")
    print("detectors saw B's test images during training, so they cannot be scored on B.")


def cmd_remap(a):
    m = {int(k): int(v) for k, v in json.loads(Path(a.map).read_text(encoding="utf-8")).items()}
    preds = json.loads(Path(a.preds).read_text(encoding="utf-8"))
    kept, dropped = [], 0
    for p in preds:
        nid = m.get(int(p["image_id"]))
        if nid is None:
            dropped += 1
            continue
        q = dict(p)
        q["image_id"] = nid
        kept.append(q)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(kept), encoding="utf-8")
    print(f"remapped {len(kept)} detections -> {a.out}")
    if dropped:
        print(f"!! dropped {dropped} detections whose image_id had no counterpart. "
              f"Those images are not in both splits; do not report this file until that is resolved.")


def cmd_ids(a):
    m = {int(k): int(v) for k, v in json.loads(Path(a.map).read_text(encoding="utf-8")).items()}
    src = [int(x) for x in Path(getattr(a, "in")).read_text(encoding="utf-8").split()]
    outv = [m[i] for i in src if i in m]
    Path(a.out).write_text("\n".join(str(i) for i in sorted(outv)), encoding="utf-8")
    print(f"translated {len(outv)}/{len(src)} ids -> {a.out}")
    if len(outv) != len(src):
        print(f"!! {len(src) - len(outv)} flagged ids have no counterpart in the target split. "
              f"The cleaned subset would not be the same 344 images. Resolve before using.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("diagnose")
    p.add_argument("--a", required=True); p.add_argument("--b", required=True)
    p.add_argument("--out", default=r"results\align")
    p.add_argument("--cache", default=r"results\leakage\phash_cache.json")
    p.set_defaults(f=cmd_diagnose)
    p = sp.add_parser("matrix")
    p.add_argument("--a-dir", required=True); p.add_argument("--b-dir", required=True)
    p.add_argument("--out", default=r"resultslign")
    p.add_argument("--cache", default=r"results\leakage\phash_cache.json")
    p.set_defaults(f=cmd_matrix)
    p = sp.add_parser("remap")
    p.add_argument("--map", required=True); p.add_argument("--preds", required=True)
    p.add_argument("--out", required=True); p.set_defaults(f=cmd_remap)
    p = sp.add_parser("ids")
    p.add_argument("--map", required=True); p.add_argument("--in", required=True)
    p.add_argument("--out", required=True); p.set_defaults(f=cmd_ids)
    a = ap.parse_args()
    a.f(a)


if __name__ == "__main__":
    main()
