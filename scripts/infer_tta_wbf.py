"""
infer_tta_wbf.py
-----------------
Multi-scale + horizontal-flip test-time augmentation, combined via
Weighted Box Fusion (WBF), producing predictions.bbox.json in the same
COCO-results format eval_stratified.py already expects.

Requires:
    pip install ensemble-boxes

Run (after training + picking the best checkpoint):
    python infer_tta_wbf.py ^
        --config D:/aquavisionnet/dino_swinb_aquavision_v2.py ^
        --checkpoint D:/aquavisionnet/avn_out/dino_swinb_v2/best_coco_bbox_mAP_50_epoch_XX.pth ^
        --test-json D:/aquavisionnet/data/aquatax10/annotations/test.json ^
        --out D:/aquavisionnet/avn_out/dino_swinb_v2/predictions_tta.bbox.json

Then evaluate exactly as before:
    python eval_stratified.py --gt <test.json> --pred predictions_tta.bbox.json --score-thr 0.30
"""
import argparse
import json

import cv2
import numpy as np
from mmdet.apis import init_detector, inference_detector
from pycocotools.coco import COCO
from ensemble_boxes import weighted_boxes_fusion

SCALES = [800, 1024, 1280]   # short-side scales for TTA
SCORE_THR = 0.05             # keep low so WBF has enough candidates to fuse


def run_one(model, img_bgr, short_side, flip):
    h, w = img_bgr.shape[:2]
    scale = short_side / min(h, w)
    long_side_cap = int(1400 * (short_side / 800))
    new_w, new_h = int(w * scale), int(h * scale)
    if max(new_w, new_h) > long_side_cap:
        scale = long_side_cap / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
    new_w, new_h = max(new_w, 1), max(new_h, 1)
    resized = cv2.resize(img_bgr, (new_w, new_h))
    if flip:
        resized = cv2.flip(resized, 1)

    result = inference_detector(model, resized)
    pred = result.pred_instances
    boxes = pred.bboxes.cpu().numpy()
    scores = pred.scores.cpu().numpy()
    labels = pred.labels.cpu().numpy()

    if len(boxes) == 0:
        return boxes, scores, labels

    boxes = boxes / np.array([new_w, new_h, new_w, new_h])
    boxes = np.clip(boxes, 0.0, 1.0)
    if flip:
        boxes[:, [0, 2]] = 1.0 - boxes[:, [2, 0]]  # x0' = 1-x1, x1' = 1-x0 (keeps x0<x1)
    return boxes, scores, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--test-json", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    model = init_detector(args.config, args.checkpoint, device="cuda:0")
    coco = COCO(args.test_json)

    results = []
    img_ids = coco.getImgIds()
    for n, img_id in enumerate(img_ids):
        info = coco.loadImgs(img_id)[0]
        img_bgr = cv2.imread(info["file_name"])
        if img_bgr is None:
            continue
        h, w = img_bgr.shape[:2]

        all_boxes, all_scores, all_labels = [], [], []
        for s in SCALES:
            for flip in (False, True):
                b, sc, lb = run_one(model, img_bgr, s, flip)
                keep = sc >= SCORE_THR
                if keep.sum() == 0:
                    continue
                all_boxes.append(b[keep].tolist())
                all_scores.append(sc[keep].tolist())
                all_labels.append(lb[keep].tolist())

        if not all_boxes:
            continue

        fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
            all_boxes, all_scores, all_labels,
            weights=None, iou_thr=0.55, skip_box_thr=0.001)

        for (x0, y0, x1, y1), sc, lb in zip(fused_boxes, fused_scores, fused_labels):
            X0, Y0, X1, Y1 = x0 * w, y0 * h, x1 * w, y1 * h
            results.append({
                "image_id": img_id,
                "category_id": int(lb),
                "bbox": [float(X0), float(Y0), float(X1 - X0), float(Y1 - Y0)],
                "score": float(sc),
            })

        if (n + 1) % 50 == 0:
            print(f"  {n + 1}/{len(img_ids)} images done")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f)
    print(f"wrote {len(results)} detections to {args.out}")


if __name__ == "__main__":
    main()
