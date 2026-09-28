"""
09_collect_results.py - gather every measured result into results\\FILL_SHEET.md, organised by the
manuscript table/figure each value belongs to. It only copies numbers from result files; if a file is
missing it says MISSING. Paste FILL_SHEET.md into the Claude chat to have the manuscript placeholders filled.

  python scripts\\09_collect_results.py --results results
"""
import argparse, glob, json, os, time
ap = argparse.ArgumentParser(); ap.add_argument('--results', default='results'); a = ap.parse_args()
R = a.results; S = os.path.join(R, 'stats'); out = []
def j(p):
    p = os.path.join(S, p); return json.load(open(p)) if os.path.exists(p) else None
def f3(x): return 'nan' if x is None or x != x else f'{x:.3f}'
def ci(m, key='mAP50'):
    b = j(f'boot_{m}.json')
    if not b: return 'MISSING'
    r = b['metrics'][key]; return f"{f3(r['point'])} [{f3(r['lo'])}-{f3(r['hi'])}] (valid {r['valid_draws']}/{b['B']})"
out.append(f'# FILL SHEET  (generated {time.strftime("%Y-%m-%d %H:%M")})\n')
out.append('## Tables 3 / 5 / 6 / 7: mAP@0.5 and per-class AP@0.5 with 95% bootstrap CI\n')
for m in ('yolov8m', 'frcnn', 'avn15', 'dino_plain', 'dino_tta', 'ablA', 'ablB'):
    out.append(f'- **{m}** mAP@0.5 {ci(m)} | mAP@[.5:.95] {ci(m, "mAP")} | ' +
               ' | '.join(f'{c} {ci(m, c)}' for c in ('plastic', 'paper', 'metal', 'glass')))
out.append('\n## Paired differences (compare_*.json)\n')
for p in sorted(glob.glob(os.path.join(S, 'compare_*.json'))):
    d = json.load(open(p)); r = d['metrics']['mAP50']
    out.append(f"- {os.path.basename(p)}: A-B mAP@0.5 = {f3(r['diff'])} [{f3(r['lo'])}-{f3(r['hi'])}], P(diff<=0) = {f3(r['p_le0'])}")
out.append('\n## Table 8 and other text outputs (clean / counts / validation re-score)\n')
for p in sorted(glob.glob(os.path.join(S, '*.txt'))):
    out.append(f'### {os.path.basename(p)}\n```\n' + open(p, encoding='utf-8', errors='ignore').read().strip() + '\n```')
out.append('\n## Figure 3 inputs (scale-wise)\n')
for m in ('yolov8m', 'frcnn', 'avn15'):
    s = j(f'summary_{m}.json')
    out.append(f'- {m}: ' + (', '.join(f'{k} {f3(s[k])}' for k in ('AP_small', 'AP_medium', 'AP_large', 'AR_small', 'AR_medium', 'AR_large')) if s else 'MISSING'))
out.append('\n## Throughput (Table 6)\n')
for p in sorted(glob.glob(os.path.join(R, 'preds', '*throughput*'))):
    out.append(f'- {os.path.basename(p)}: ' + open(p, encoding='utf-8', errors='ignore').read().strip().replace('\n', ' ')[:300])
for name in ('box_counts.md',):
    p = os.path.join(S, name)
    out.append(f'\n## Table 2 box counts\n' + (open(p, encoding='utf-8').read() if os.path.exists(p) else 'MISSING'))
p = os.path.join(R, 'FACTS.md')
out.append('\n## Facts from code/logs (F2-F6)\n' + (open(p, encoding='utf-8').read() if os.path.exists(p) else 'MISSING'))
p = os.path.join(R, 'RUN_LOG.md')
out.append('\n## Run log\n' + (open(p, encoding='utf-8').read() if os.path.exists(p) else 'MISSING'))
open(os.path.join(R, 'FILL_SHEET.md'), 'w', encoding='utf-8').write('\n'.join(out))
print('wrote', os.path.join(R, 'FILL_SHEET.md'))
