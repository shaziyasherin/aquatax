"""
02_export_yolo.py  -  run a trained Ultralytics YOLOv8 model on a COCO split and save COCO-format predictions,
so YOLOv8-m is scored with pycocotools like every other detector.

Classes are matched to COCO category ids BY NAME (model.names vs gt categories); the script stops if a
name does not match, instead of guessing.

Usage:
  python scripts\\02_export_yolo.py --weights PATH\\best.pt --gt PATH\\test.json --images PATH\\images_root ^
      --out results\\preds\\yolov8m_test.json --imgsz 512 --time
"""
import argparse, json, os, time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', required=True)
    ap.add_argument('--gt', required=True)
    ap.add_argument('--images', required=True, help='folder that file_name entries are relative to')
    ap.add_argument('--out', required=True)
    ap.add_argument('--imgsz', type=int, default=512)
    ap.add_argument('--conf', type=float, default=0.001)
    ap.add_argument('--iou', type=float, default=0.7)
    ap.add_argument('--max-det', type=int, default=100)
    ap.add_argument('--device', default='0')
    ap.add_argument('--time', action='store_true', help='record images/s at batch 1 (incl. pre/post-processing)')
    a = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(a.weights)
    gt = json.load(open(a.gt, encoding='utf-8'))
    name2cid = {c['name'].strip().lower(): c['id'] for c in gt['categories']}
    idx2cid = {}
    for i, n in model.names.items():
        key = n.strip().lower()
        if key not in name2cid:
            raise SystemExit(f'YOLO class "{n}" not among COCO categories {list(name2cid)}; fix the mapping explicitly.')
        idx2cid[int(i)] = name2cid[key]
    print('class map (yolo idx -> coco id):', idx2cid)

    dets, times = [], []
    imgs = gt['images']
    for k, im in enumerate(imgs):
        path = Path(a.images) / im['file_name']
        if not path.exists():
            raise SystemExit(f'missing image: {path}')
        t0 = time.perf_counter()
        r = model.predict(str(path), imgsz=a.imgsz, conf=a.conf, iou=a.iou, max_det=a.max_det,
                          device=a.device, verbose=False)[0]
        dt = time.perf_counter() - t0
        if k >= 10: times.append(dt)          # skip warm-up
        b = r.boxes
        for (x1, y1, x2, y2), s, c in zip(b.xyxy.cpu().tolist(), b.conf.cpu().tolist(), b.cls.cpu().tolist()):
            dets.append({'image_id': im['id'], 'category_id': idx2cid[int(c)],
                         'bbox': [round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)],
                         'score': round(float(s), 5)})
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    json.dump(dets, open(a.out, 'w'))
    print(f'wrote {len(dets)} detections for {len(imgs)} images -> {a.out}')
    if a.time and times:
        ips = len(times) / sum(times)
        rec = {'model': 'yolov8m', 'images_per_s_batch1': round(ips, 2), 'n_timed': len(times), 'imgsz': a.imgsz}
        json.dump(rec, open(a.out.replace('.json', '_throughput.json'), 'w'), indent=1)
        print('throughput:', rec)


if __name__ == '__main__':
    main()
