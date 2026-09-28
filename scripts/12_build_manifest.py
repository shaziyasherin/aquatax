#!/usr/bin/env python3
"""
12_build_manifest.py - the reproducibility artifact Reviewer 4 (comment 3) asked for.

R4.3 says: "Either release a build script that reconstructs the corpus from independently obtained
sources, with per-image manifests and hashes, and state plainly that FloW must be obtained
separately, or drop the reproducibility claim."

This is that build script. It redistributes nothing. It publishes, for every image in AquaTax-4,
the SHA-256 of the image bytes, its source dataset, its path relative to that dataset's own root,
its dimensions, its split, and its box count. Anyone who obtains AquaTrash, FloW-Img and
AquaSurf-Malnad-223 from their original providers can then run `verify` and rebuild the exact
corpus, or find out precisely which files they are missing.

  build   python scripts\\12_build_manifest.py build --data D:\\aquavisionnet\\avn_out ^
              --out results\\manifest
  verify  python scripts\\12_build_manifest.py verify --manifest results\\manifest\\AquaTax4_manifest.csv ^
              --roots D:\\downloads\\AquaTrash D:\\downloads\\FloW_IMG D:\\downloads\\Malnad223
  splits  python scripts\\12_build_manifest.py splits --manifest results\\manifest\\AquaTax4_manifest.csv ^
              --found results\\manifest\\verify_found.csv --out results\\manifest\\rebuilt

`splits` writes COCO jsons that point at the verifier's own copies of the images, so a third party
reproduces the train/val/test split byte-for-byte without ever receiving an image from us.
"""
import argparse, csv, hashlib, json, os, sys
from collections import Counter, defaultdict
from pathlib import Path

# Substrings that identify which source dataset an image came from. Order matters: first hit wins.
SOURCE_RULES = [
    ("FloW-Img", ("flow_img", "flow-img", "flowimg", "flow_", "/flow/", "\\flow\\")),
    ("AquaTrash", ("aquatrash", "aqua_trash", "aqua-trash")),
    ("AquaSurf-Malnad-223", ("malnad", "aquasurf")),
]
CHUNK = 1 << 20


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(CHUNK), b""):
            h.update(b)
    return h.hexdigest()


def classify(path):
    low = str(path).replace("/", "\\").lower()
    for name, keys in SOURCE_RULES:
        if any(k.replace("/", "\\") in low for k in keys):
            return name
    return "UNKNOWN"


def relpath_within_source(path, source):
    """Path relative to the source dataset's own root, so it is meaningful to someone who
    downloaded that dataset themselves. Falls back to the basename."""
    parts = Path(path).parts
    low = [p.lower() for p in parts]
    keys = dict(SOURCE_RULES).get(source, ())
    for i, p in enumerate(low):
        if any(k.strip("/\\") in p for k in keys):
            return "/".join(parts[i + 1:]) or Path(path).name
    return Path(path).name


# ------------------------------------------------------------------ build
def cmd_build(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, missing, per_source = [], [], Counter()
    for split in ("train", "val", "test"):
        p = Path(a.data) / f"{split}.json"
        if not p.exists():
            sys.exit(f"missing {p}")
        d = json.loads(p.read_text(encoding="utf-8"))
        nbox = Counter(x["image_id"] for x in d["annotations"])
        cats = {c["id"]: c["name"] for c in d["categories"]}
        percls = defaultdict(Counter)
        for x in d["annotations"]:
            percls[x["image_id"]][cats[x["category_id"]]] += 1
        print(f"{split}: {len(d['images'])} images, {len(d['annotations'])} boxes")
        for k, im in enumerate(d["images"]):
            f = Path(im["file_name"])
            if not f.is_absolute():
                f = Path(a.images or a.data) / f
            if not f.exists():
                missing.append(str(f))
                continue
            src = classify(f)
            per_source[src] += 1
            c = percls[im["id"]]
            rows.append({
                "split": split, "image_id": im["id"], "source": src,
                "relpath": relpath_within_source(f, src), "basename": f.name,
                "sha256": sha256(f), "bytes": f.stat().st_size,
                "width": im.get("width", ""), "height": im.get("height", ""),
                "n_boxes": nbox.get(im["id"], 0),
                "plastic": c.get("plastic", 0), "paper": c.get("paper", 0),
                "metal": c.get("metal", 0), "glass": c.get("glass", 0),
            })
            if (k + 1) % 200 == 0:
                print(f"  ... {k + 1}/{len(d['images'])}")

    man = out / "AquaTax4_manifest.csv"
    with man.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {len(rows)} rows -> {man}")

    dup = Counter(r["sha256"] for r in rows)
    repeated = {h: n for h, n in dup.items() if n > 1}
    if repeated:
        print(f"!! {len(repeated)} sha256 values appear more than once "
              f"({sum(repeated.values())} images). These are EXACT duplicates, a stronger finding "
              f"than the pHash near-duplicates - list them in the leakage section.")
        with (out / "exact_duplicates.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["sha256", "count", "split:image_id:relpath"])
            for h, n in sorted(repeated.items(), key=lambda x: -x[1]):
                same = [r for r in rows if r["sha256"] == h]
                w.writerow([h, n, " | ".join(f"{r['split']}:{r['image_id']}:{r['relpath']}" for r in same)])
        print(f"   -> {out / 'exact_duplicates.csv'}")
    else:
        print("no exact byte-level duplicates across the three splits.")

    cross = defaultdict(set)
    for r in rows:
        cross[r["sha256"]].add(r["split"])
    leak = [h for h, s in cross.items() if len(s) > 1]
    if leak:
        print(f"!! {len(leak)} images are byte-identical ACROSS splits - that is hard train/test "
              f"leakage and must be reported in Section 4.7.")

    if missing:
        print(f"\n!! {len(missing)} file_name entries do not exist on disk; first few: {missing[:5]}")

    summary = out / "manifest_summary.md"
    bysrc = defaultdict(Counter)
    for r in rows:
        bysrc[r["source"]][r["split"]] += 1
    with summary.open("w", encoding="utf-8") as fh:
        fh.write("# AquaTax-4 composition (counted from the manifest, not from the paper)\n\n")
        fh.write("| source | train | val | test | total | share |\n|---|---|---|---|---|---|\n")
        tot = len(rows)
        for s, c in sorted(bysrc.items()):
            n = sum(c.values())
            fh.write(f"| {s} | {c['train']} | {c['val']} | {c['test']} | {n} | {100 * n / tot:.1f} % |\n")
        fh.write(f"| **total** | {bysrc and sum(c['train'] for c in bysrc.values())} | "
                 f"{sum(c['val'] for c in bysrc.values())} | {sum(c['test'] for c in bysrc.values())} | "
                 f"{tot} | 100 % |\n")
        fh.write(f"\nExact byte-level duplicates: {len(repeated)} hash groups. "
                 f"Byte-identical across splits: {len(leak)}.\n")
    print(f"wrote {summary}")
    print(open(summary, encoding="utf-8").read())


# ------------------------------------------------------------------ verify
def cmd_verify(a):
    man = list(csv.DictReader(open(a.manifest, encoding="utf-8")))
    want = {r["sha256"]: r for r in man}
    print(f"manifest: {len(man)} images, {len(want)} distinct hashes")
    found, seen = {}, 0
    for root in a.roots:
        for dp, _dn, fns in os.walk(root):
            for fn in fns:
                if Path(fn).suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
                    continue
                p = Path(dp) / fn
                seen += 1
                h = sha256(p)
                if h in want and h not in found:
                    found[h] = str(p)
                if seen % 500 == 0:
                    print(f"  ... hashed {seen}, matched {len(found)}")
    out = Path(a.manifest).parent / "verify_found.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["sha256", "local_path"])
        for h, p in found.items():
            w.writerow([h, p])
    bysrc_t, bysrc_f = Counter(), Counter()
    for r in man:
        bysrc_t[r["source"]] += 1
        if r["sha256"] in found:
            bysrc_f[r["source"]] += 1
    print(f"\nhashed {seen} local files; matched {len(found)}/{len(want)} manifest hashes")
    print(f"{'source':<24}{'found':>8}{'needed':>8}")
    for s in sorted(bysrc_t):
        print(f"{s:<24}{bysrc_f[s]:>8}{bysrc_t[s]:>8}")
    print(f"\nwrote {out}")
    short = [s for s in sorted(bysrc_t) if bysrc_f[s] < bysrc_t[s]]
    if short:
        print("incomplete sources: " + ", ".join(f"{s} ({bysrc_t[s] - bysrc_f[s]} missing)" for s in short))
        if any("FloW" in s for s in short):
            print("FloW-Img is non-commercial and must be requested from its authors; it cannot be "
                  "redistributed with this paper.")
    else:
        print("every manifest image was located locally: the corpus is fully reconstructed.")


# ------------------------------------------------------------------ splits
def cmd_splits(a):
    man = list(csv.DictReader(open(a.manifest, encoding="utf-8")))
    loc = {r["sha256"]: r["local_path"] for r in csv.DictReader(open(a.found, encoding="utf-8"))}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cats = [{"id": i, "name": n} for i, n in enumerate(["plastic", "paper", "metal", "glass"])]
    for split in ("train", "val", "test"):
        rows = [r for r in man if r["split"] == split]
        have = [r for r in rows if r["sha256"] in loc]
        imgs = [{"id": int(r["image_id"]), "file_name": loc[r["sha256"]],
                 "width": int(r["width"] or 0), "height": int(r["height"] or 0)} for r in have]
        json.dump({"images": imgs, "annotations": [], "categories": cats},
                  open(out / f"{split}_images.json", "w"), indent=1)
        print(f"{split}: {len(have)}/{len(rows)} images resolvable -> {out / (split + '_images.json')}")
    print("\nThese carry images and ids only. Merge the released annotation file onto them by "
          "image_id to get the full COCO jsons.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("build"); p.add_argument("--data", required=True)
    p.add_argument("--images", default=None, help="root for relative file_name entries")
    p.add_argument("--out", default=r"results\manifest"); p.set_defaults(f=cmd_build)
    p = sp.add_parser("verify"); p.add_argument("--manifest", required=True)
    p.add_argument("--roots", nargs="+", required=True); p.set_defaults(f=cmd_verify)
    p = sp.add_parser("splits"); p.add_argument("--manifest", required=True)
    p.add_argument("--found", required=True); p.add_argument("--out", required=True)
    p.set_defaults(f=cmd_splits)
    a = ap.parse_args()
    a.f(a)


if __name__ == "__main__":
    main()
