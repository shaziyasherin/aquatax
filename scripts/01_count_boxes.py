"""
01_count_boxes.py  -  exact image and box counts per class for each split (Table 2, plastic count).

Usage:
  python scripts\\01_count_boxes.py --split train=PATH\\train.json --split val=PATH\\val.json ^
      --split test=PATH\\test.json [--split train_aug=PATH\\train_aug.json] --out results\\stats\\box_counts.md
"""
import argparse, json
from collections import Counter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', action='append', required=True, help='name=path/to/coco.json')
    ap.add_argument('--out', default=r'results\stats\box_counts.md')
    a = ap.parse_args()
    rows, cats_all = [], None
    for s in a.split:
        name, path = s.split('=', 1)
        d = json.load(open(path, encoding='utf-8'))
        names = {c['id']: c['name'] for c in d['categories']}
        cats_all = cats_all or [names[k] for k in sorted(names)]
        cnt = Counter(names[x['category_id']] for x in d['annotations'])
        imgs_with = len({x['image_id'] for x in d['annotations']})
        rows.append((name, path, len(d['images']), imgs_with, len(d['annotations']), cnt))
    with open(a.out, 'w', encoding='utf-8') as o:
        o.write('# Box counts read from COCO files\n\n')
        o.write('| split | images | images with boxes | boxes | ' + ' | '.join(cats_all) + ' | file |\n')
        o.write('|' + '---|' * (5 + len(cats_all)) + '\n')
        tot = Counter(); ti = tb = 0
        for name, path, ni, nw, nb, cnt in rows:
            o.write(f'| {name} | {ni} | {nw} | {nb} | ' + ' | '.join(str(cnt.get(c, 0)) for c in cats_all) + f' | {path} |\n')
            if name in ('train', 'val', 'test'):
                tot += cnt; ti += ni; tb += nb
        o.write(f'| **train+val+test** | {ti} | | {tb} | ' + ' | '.join(str(tot.get(c, 0)) for c in cats_all) + ' | |\n')
    print(open(a.out, encoding='utf-8').read())


if __name__ == '__main__':
    main()
