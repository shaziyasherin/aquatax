"""21_dino_facts.py - the last two numbers the manuscript is missing.

Fills Table 6's DINO-Swin-B rows: parameter count (placeholders 70, 73) and throughput
(71, 74). Nothing else is outstanding.

  params      CPU only, a few seconds. Reads the checkpoint's state_dict and sums tensor sizes.
  throughput  GPU, about a minute. Batch-1 forward passes over the test split, same harness as
              the matched-protocol detectors, so the figure is comparable to theirs.

Run in the `aquavision` env (the one with mmdet).

  conda activate aquavision
  cd /d D:\\aquavisionnet\\revision
  python 21_dino_facts.py params
  python 21_dino_facts.py throughput --config D:/aquavisionnet/dino_swinb_aquavision_v2.py

If `params` cannot find the checkpoint it prints every .pth it did find; pass the right one with
--ckpt. If the throughput run is any trouble, skip it: Table 6 already prints an em-dash there and
the manuscript says the two protocol blocks are not directly comparable.
"""
import argparse, glob, os, sys, time

SEARCH = [
    'D:/aquavisionnet/revision/work_dirs/**/*.pth',
    'D:/aquavisionnet/work_dirs/**/*.pth',
    'D:/aquavisionnet/**/dino*.pth',
]


def find_ckpts():
    out = []
    for pat in SEARCH:
        out.extend(glob.glob(pat, recursive=True))
    return sorted(set(out), key=lambda p: -os.path.getmtime(p))


def cmd_params(args):
    import torch
    ck = args.ckpt
    if not ck:
        cands = find_ckpts()
        if not cands:
            print('No checkpoint found. Candidates searched:')
            for p in SEARCH:
                print('   ', p)
            for c in find_ckpts():
                print('  found:', c)
            return 1
        ck = cands[0]
        print('using most recent non-ablation checkpoint:')
    print('  %s' % ck)
    sd = torch.load(ck, map_location='cpu')
    sd = sd.get('state_dict', sd)
    total = sum(v.numel() for v in sd.values() if hasattr(v, 'numel'))
    skip = ('num_batches_tracked',)
    real = sum(v.numel() for k, v in sd.items()
               if hasattr(v, 'numel') and not k.endswith(skip))
    print()
    print('  parameters : %.1f M   (%d)' % (real / 1e6, real))
    print('  incl. buffers: %.1f M' % (total / 1e6,))
    print()
    print('  -> Table 6, DINO-Swin-B rows, "Params (M)" column: %.1f' % (real / 1e6,))
    return 0


def cmd_throughput(args):
    import torch
    from mmdet.apis import init_detector, inference_detector
    imgs = []
    if args.gt and os.path.isfile(args.gt):
        import json
        names = [im['file_name'] for im in json.load(open(args.gt))['images']]
        roots = [args.images, os.path.dirname(args.gt),
                 os.path.join(os.path.dirname(args.gt), 'images'), '']
        for nm in names:
            for r in roots:
                cand = nm if not r else os.path.join(r, nm)
                if os.path.isfile(cand):
                    imgs.append(cand)
                    break
        print('resolved %d of %d images from %s' % (len(imgs), len(names), args.gt))
    if not imgs:
        imgs = sorted(glob.glob(os.path.join(args.images, '*.jpg')) +
                      glob.glob(os.path.join(args.images, '*.png')))
    imgs = imgs[:args.n]
    if not imgs:
        print('no images found. Pass --images <folder> or --gt <test.json>.')
        return 1
    model = init_detector(args.config, args.ckpt, device='cuda:0')
    for p in imgs[:10]:                      # warm-up, not timed
        inference_detector(model, p)
    torch.cuda.synchronize()
    t0 = time.time()
    for p in imgs:
        inference_detector(model, p)
    torch.cuda.synchronize()
    dt = time.time() - t0
    print()
    print('  %d images in %.2f s  ->  %.1f img/s (batch 1)' % (len(imgs), dt, len(imgs) / dt))
    print('  -> Table 6, DINO-Swin-B rows, "Imgs/s*" column: %.1f' % (len(imgs) / dt,))
    return 0


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    p1 = sub.add_parser('params')
    p1.add_argument('--ckpt', default=None)
    p1.set_defaults(fn=cmd_params)
    p2 = sub.add_parser('throughput')
    p2.add_argument('--config', required=True)
    p2.add_argument('--ckpt', default=None)
    p2.add_argument('--images', default='D:/aquavisionnet/revision/corpus_B/images')
    p2.add_argument('--gt', default='D:/aquavisionnet/revision/corpus_B/test.json')
    p2.add_argument('--n', type=int, default=200)
    p2.set_defaults(fn=cmd_throughput)
    a = ap.parse_args()
    if a.cmd == 'throughput' and not a.ckpt:
        c = find_ckpts()
        if not c:
            print('pass --ckpt')
            sys.exit(1)
        a.ckpt = c[0]
        print('using checkpoint: %s' % a.ckpt)
    sys.exit(a.fn(a))
