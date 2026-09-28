"""
05_dino_export.py - COCO-format predictions of the reported DINO-Swin-B checkpoint (epoch 18) on TEST and
on VAL with the SAME pipeline (item S5), via mmdetection's tools\\test.py. Shell-independent (works from
Anaconda Prompt, Git Bash or PowerShell).

  python scripts\\05_dino_export.py --mmdet D:\\aquavisionnet\\mmdetection --cfg PATH\\dino_config.py ^
      --ckpt PATH\\epoch_18.pth --val-ann PATH\\val.json --img-root PATH\\images --split test
  (then again with --split val)

mmdet 3.x CocoMetric writes <outfile_prefix>.bbox.json; this script renames it to
results\\preds\\dino_plain_<split>.json. If the config uses data_root, pass --val-ann relative to data_root
and --img-root relative too (check the printed config). TTA+WBF uses Shaz's separate script: run it
unchanged on test and val and save dino_tta_test.json / dino_tta_val.json in results\\preds.
Throughput: add --benchmark to run tools\\analysis_tools\\benchmark.py (check its --help first).
"""
import argparse, os, shutil, subprocess, sys

ap = argparse.ArgumentParser()
ap.add_argument('--mmdet', required=True); ap.add_argument('--cfg', required=True); ap.add_argument('--ckpt', required=True)
ap.add_argument('--split', choices=['test', 'val'], required=True)
ap.add_argument('--val-ann'); ap.add_argument('--img-root')
ap.add_argument('--out-dir', default=r'results\preds'); ap.add_argument('--benchmark', action='store_true')
a = ap.parse_args()
os.makedirs(a.out_dir, exist_ok=True)
prefix = os.path.abspath(os.path.join(a.out_dir, f'dino_plain_{a.split}')).replace('\\', '/')
opts = [f'test_evaluator.outfile_prefix={prefix}']
if a.split == 'val':
    if not (a.val_ann and a.img_root): sys.exit('--split val needs --val-ann and --img-root')
    ann = a.val_ann.replace('\\', '/'); img = a.img_root.replace('\\', '/').rstrip('/') + '/'
    opts += [f'test_dataloader.dataset.ann_file={ann}', f'test_dataloader.dataset.data_prefix.img={img}',
             f'test_evaluator.ann_file={ann}']
cmd = [sys.executable, os.path.join(a.mmdet, 'tools', 'test.py'), a.cfg, a.ckpt,
       '--work-dir', os.path.join(a.out_dir, f'_mmdet_{a.split}'), '--cfg-options', *opts]
print('RUN:', ' '.join(cmd)); r = subprocess.run(cmd)
if r.returncode: sys.exit(f'test.py failed with code {r.returncode}')
src = prefix + '.bbox.json'; dst = prefix + '.json'
shutil.move(src, dst); print('saved', dst)
if a.benchmark:
    b = [sys.executable, os.path.join(a.mmdet, 'tools', 'analysis_tools', 'benchmark.py'), a.cfg,
         '--checkpoint', a.ckpt, '--task', 'inference', '--max-iter', '200']
    print('RUN:', ' '.join(b)); out = subprocess.run(b, capture_output=True, text=True)
    open(os.path.join(a.out_dir, 'dino_plain_throughput.txt'), 'w').write(out.stdout + out.stderr); print(out.stdout[-1500:])
