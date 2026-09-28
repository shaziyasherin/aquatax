#!/usr/bin/env python3
"""
rebuild_corpus.py - reconstruct AquaTax-4 from your own copies of its three source datasets.

No image is redistributed with this release. What is released is, for every one of the 2,592
images, its SHA-256 hash, source, split and box counts (manifest/AquaTax4_manifest.csv), and the
boxes of the two CC BY 4.0 sources (annotations/aquatax4_boxes_ccby.json). FloW-Img boxes are not
redistributed: they are regenerated from the Pascal VOC files that come with FloW-Img, using the
conversion recorded in scripts/flow_convention.json.

Obtain the three sources first:
    AquaTrash            Panwar et al. (2020), https://doi.org/10.1016/j.cscee.2020.100026
    FloW-Img             Cheng et al. (2021), on request from its authors (non-commercial use)
    AquaSurf-Malnad-223  https://doi.org/10.5281/zenodo.22945111

Then:
    python scripts/rebuild_corpus.py --roots PATH_TO_AquaTrash PATH_TO_FloW_IMG PATH_TO_Malnad223 \
        --out rebuilt

It hashes every image under the roots, finds each manifest entry by its hash, and writes
rebuilt/{train,val,test}.json in COCO format with absolute file paths, the same image ids and the
same category ids as the corpus used in the paper. Any image it cannot find is listed in
rebuilt/missing.csv; the rebuild still completes for the rest.

    --compare-to DIR   compare the rebuilt splits with a reference COCO directory, image by image
                       (used once, by the authors, to prove the round trip is exact)

Standard library only.
"""
import argparse
import csv
import hashlib
import json
import os
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
IMG_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
SPLITS = ('train', 'val', 'test')
FLOW = 'FloW-Img'

# Pascal VOC stores inclusive pixel corners. Converters differ in how they turn those into a COCO
# [x, y, width, height] box; the one used for AquaTax-4 is named in flow_convention.json and was
# chosen by reproducing the paper's corpus exactly, not by assumption.
CONVENTIONS = {
    'c0': lambda x0, y0, x1, y1: [x0, y0, x1 - x0, y1 - y0],
    'c1': lambda x0, y0, x1, y1: [x0 - 1, y0 - 1, x1 - x0 + 1, y1 - y0 + 1],
    'c2': lambda x0, y0, x1, y1: [x0, y0, x1 - x0 + 1, y1 - y0 + 1],
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def hash_roots(roots, cache=None):
    """sha256 -> first path found, over every image file under the given roots."""
    seen = {}
    if cache and Path(cache).exists():
        seen = json.loads(Path(cache).read_text(encoding='utf-8'))
    index, n = {}, 0
    for root in roots:
        for dirpath, _, files in os.walk(root):
            for fn in files:
                if Path(fn).suffix.lower() not in IMG_EXT:
                    continue
                p = os.path.abspath(os.path.join(dirpath, fn))
                st = os.stat(p)
                key = '%s|%d|%d' % (p, st.st_size, int(st.st_mtime))
                h = seen.get(key) or sha256(p)
                seen[key] = h
                index.setdefault(h, p)
                n += 1
                if n % 500 == 0:
                    print('  hashed %d images' % n)
    if cache:
        Path(cache).write_text(json.dumps(seen), encoding='utf-8')
    print('  %d image files hashed under %d root(s), %d distinct' % (n, len(roots), len(index)))
    return index


def xml_index(roots):
    idx = defaultdict(list)
    for root in roots:
        for dirpath, _, files in os.walk(root):
            for fn in files:
                if fn.lower().endswith('.xml'):
                    idx[Path(fn).stem].append(os.path.join(dirpath, fn))
    return idx


def find_xml(img_path, idx):
    cands = idx.get(Path(img_path).stem, [])
    if len(cands) <= 1:
        return cands[0] if cands else None
    # Same stem in several folders (e.g. training/ and test/): take the closest by path.
    img = os.path.normcase(os.path.abspath(img_path))
    return max(cands, key=lambda c: len(os.path.commonprefix([img, os.path.normcase(os.path.abspath(c))])))


def voc_boxes(xml_path, names, conv):
    out = []
    for obj in ET.parse(xml_path).getroot().iter('object'):
        name = (obj.findtext('name') or '').strip().lower()
        if name not in names:
            continue
        bb = obj.find('bndbox')
        x0, y0, x1, y1 = (float(bb.findtext(k)) for k in ('xmin', 'ymin', 'xmax', 'ymax'))
        out.append(CONVENTIONS[conv](x0, y0, x1, y1))
    return out


def rebuild(manifest, boxes_json, convention_json, roots, out, cache=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(manifest, newline='', encoding='utf-8')))
    released = json.loads(Path(boxes_json).read_text(encoding='utf-8'))
    conv = json.loads(Path(convention_json).read_text(encoding='utf-8'))
    cats = released['categories']
    cat_id = {c['name']: c['id'] for c in cats}
    by_key = defaultdict(list)
    for a in released['annotations']:
        by_key[(a['split'], int(a['image_id']))].append(a)

    print('hashing source images ...')
    found = hash_roots(roots, cache)
    xidx = xml_index(roots)
    names = {n.lower() for n in conv['voc_names']}

    missing = []
    per_src = Counter()
    coco = {s: {'images': [], 'annotations': [], 'categories': cats} for s in SPLITS}
    for r in rows:
        split, iid, src = r['split'], int(r['image_id']), r['source']
        path = found.get(r['sha256'])
        if path is None:
            missing.append(r)
            continue
        per_src[src] += 1
        coco[split]['images'].append({'id': iid, 'file_name': path, 'width': int(r['width']),
                                      'height': int(r['height']), 'source': src,
                                      'sha256': r['sha256']})
        if src == FLOW:
            x = find_xml(path, xidx)
            bbs = voc_boxes(x, names, conv['convention']) if x else []
            if x is None and int(r['n_boxes']):
                print('  no VOC file for %s' % path)
            anns = [{'category_id': cat_id[conv['maps_to']], 'bbox': b} for b in bbs]
        else:
            anns = [{'category_id': a['category_id'], 'bbox': a['bbox']} for a in by_key[(split, iid)]]
        for a in anns:
            a.update(image_id=iid, area=a['bbox'][2] * a['bbox'][3], iscrowd=0)
            coco[split]['annotations'].append(a)

    for s in SPLITS:
        for k, a in enumerate(coco[s]['annotations'], 1):
            a['id'] = k
        (out / ('%s.json' % s)).write_text(json.dumps(coco[s]), encoding='utf-8')
        print('%-5s %4d images, %4d boxes -> %s' % (s, len(coco[s]['images']),
                                                   len(coco[s]['annotations']), out / ('%s.json' % s)))
    if missing:
        with open(out / 'missing.csv', 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(missing[0].keys()))
            w.writeheader()
            w.writerows(missing)
        by = Counter(r['source'] for r in missing)
        print('!! %d of %d images not found (%s) -> %s' % (
            len(missing), len(rows), ', '.join('%s %d' % kv for kv in sorted(by.items())),
            out / 'missing.csv'))
    else:
        print('all %d images found' % len(rows))
    return len(missing)


def box_key(b):
    return tuple(round(float(v), 2) for v in b)


def compare(rebuilt, reference):
    """Image-by-image comparison of two COCO split directories. Returns number of differences."""
    diffs = 0
    for s in SPLITS:
        a = json.loads((Path(rebuilt) / ('%s.json' % s)).read_text(encoding='utf-8'))
        b = json.loads((Path(reference) / ('%s.json' % s)).read_text(encoding='utf-8'))
        cname_a = {c['id']: c['name'] for c in a['categories']}
        cname_b = {c['id']: c['name'] for c in b['categories']}

        def per_image(d, cname):
            m = defaultdict(Counter)
            for x in d['annotations']:
                m[x['image_id']][(cname[x['category_id']], box_key(x['bbox']))] += 1
            return m
        pa, pb = per_image(a, cname_a), per_image(b, cname_b)
        ids_a = {im['id'] for im in a['images']}
        ids_b = {im['id'] for im in b['images']}
        bad = [i for i in ids_b if i not in ids_a or pa.get(i, Counter()) != pb.get(i, Counter())]
        diffs += len(bad) + len(ids_a - ids_b)
        print('%-5s images %d/%d present, %d with boxes differing from the reference%s' % (
            s, len(ids_a & ids_b), len(ids_b), len([i for i in bad if i in ids_a]),
            '' if not bad else ' (first: %s)' % sorted(bad)[:5]))
    print('ROUND TRIP %s' % ('EXACT' if diffs == 0 else 'NOT EXACT: %d differences' % diffs))
    return diffs


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--roots', nargs='+', required=True, help='your copies of the three sources')
    ap.add_argument('--out', default='rebuilt')
    ap.add_argument('--manifest', default=str(REPO / 'manifest' / 'AquaTax4_manifest.csv'))
    ap.add_argument('--boxes', default=str(REPO / 'annotations' / 'aquatax4_boxes_ccby.json'))
    ap.add_argument('--convention', default=str(HERE / 'flow_convention.json'))
    ap.add_argument('--cache', default=None, help='optional hash cache file, speeds up reruns')
    ap.add_argument('--compare-to', default=None)
    a = ap.parse_args()
    missing = rebuild(a.manifest, a.boxes, a.convention, a.roots, a.out, a.cache)
    if a.compare_to:
        sys.exit(1 if compare(a.out, a.compare_to) else 0)
    sys.exit(1 if missing else 0)


if __name__ == '__main__':
    main()
