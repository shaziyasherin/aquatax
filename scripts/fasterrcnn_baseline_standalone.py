"""
Faster R-CNN (ResNet-50 FPN) baseline — standalone (Paper 1, Section 4.7).
Self-contained: one file, no other avn_*.py imports needed.
Trains + evaluates directly on the existing COCO corpus (train.json/val.json/test.json
at --data), letterboxed to the same 512x512 input AquaVisionNet uses, and evaluates with
pycocotools (same evaluator family as the rest of the paper) for a fair comparison.
Uses torchvision's standard fine-tuning recipe (SGD, lr=0.005) — deliberately NOT the
AquaOpt-tuned recipe, since Section 4.7 is a "standard recipe per architecture" baseline.

Usage:
    python fasterrcnn_baseline_standalone.py --data D:\\aquavisionnet\\avn_out --epochs 15 --workers 0
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # avoid OMP Error #15 (duplicate OpenMP runtime on Windows)

import argparse, json, time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

CLASSES = ["plastic", "paper", "metal", "glass"]

# ══════════════════════════════════════════════════════════════════════
# Data (letterboxed to `size`, raw [0,1] — Faster R-CNN normalizes internally)
# ══════════════════════════════════════════════════════════════════════
class FRCNNDet(Dataset):
    def __init__(self, json_path, size=512, train=True):
        d = json.load(open(json_path))
        self.size, self.train = size, train
        self.imgs = {i["id"]: i for i in d["images"]}
        self.by_img = {i: [] for i in self.imgs}
        for a in d["annotations"]:
            self.by_img[a["image_id"]].append(a)
        self.ids = list(self.imgs.keys())

    def __len__(self):
        return len(self.ids)

    def scale_of(self, idx):
        info = self.imgs[self.ids[idx]]
        return self.size / max(info["width"], info["height"])

    def _letterbox(self, img, boxes):
        w, h = img.size
        s = self.size / max(h, w)
        nw, nh = int(round(w * s)), int(round(h * s))
        img = img.resize((nw, nh), Image.BILINEAR)
        canvas = np.full((self.size, self.size, 3), 114, np.uint8)
        canvas[:nh, :nw] = np.asarray(img)
        if len(boxes):
            boxes = boxes * s
        return canvas, boxes

    def __getitem__(self, idx):
        info = self.imgs[self.ids[idx]]
        img = Image.open(info["file_name"]).convert("RGB")
        anns = self.by_img[self.ids[idx]]
        boxes = np.array([[a["bbox"][0], a["bbox"][1],
                           a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]]
                          for a in anns], np.float32).reshape(-1, 4)
        labels = np.array([a["category_id"] + 1 for a in anns], np.int64).reshape(-1)  # +1: 0 = background
        img, boxes = self._letterbox(img, boxes)

        if self.train and len(boxes):
            import random
            if random.random() < 0.5:
                img = img[:, ::-1, :].copy()
                x1 = boxes[:, 0].copy(); x2 = boxes[:, 2].copy()
                boxes[:, 0] = self.size - x2
                boxes[:, 2] = self.size - x1

        t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.
        if len(boxes):
            keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
            boxes, labels = boxes[keep], labels[keep]
        target = {"boxes": torch.from_numpy(boxes).float(),
                  "labels": torch.from_numpy(labels).long(),
                  "image_id": torch.tensor([self.ids[idx]])}
        return t, target

def collate(batch):
    return [b[0] for b in batch], [b[1] for b in batch]

# ══════════════════════════════════════════════════════════════════════
# Model
# ══════════════════════════════════════════════════════════════════════
def build_model(device, num_classes=len(CLASSES) + 1, size=512):
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights, min_size=size, max_size=size, box_score_thresh=0.05)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model.to(device)

# ══════════════════════════════════════════════════════════════════════
# Train
# ══════════════════════════════════════════════════════════════════════
def train(cfg, device):
    tr = FRCNNDet(f"{cfg.data}/train.json", size=cfg.size, train=True)
    dl = DataLoader(tr, batch_size=cfg.batch, shuffle=True, num_workers=cfg.workers,
                    collate_fn=collate, drop_last=True)
    model = build_model(device, size=cfg.size)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, cfg.epochs // 2), gamma=0.1)
    n_par = sum(p.numel() for p in params) / 1e6
    print(f"  Faster R-CNN (ResNet-50 FPN)  trainable params = {n_par:.1f}M")
    for ep in range(cfg.epochs):
        model.train()
        t0 = time.time()
        run_loss, n_batches = 0.0, 0
        for imgs, targets in dl:
            imgs = [im.to(device) for im in imgs]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(imgs, targets)
            loss = sum(loss_dict.values())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 10.0)
            opt.step()
            run_loss += float(loss); n_batches += 1
        sched.step()
        print(f"    e{ep}  {time.time()-t0:.0f}s  train_loss={run_loss/max(1,n_batches):.3f}")
    return model, n_par

# ══════════════════════════════════════════════════════════════════════
# Evaluate (pycocotools — same evaluator family as the rest of the paper)
# ══════════════════════════════════════════════════════════════════════
@torch.no_grad()
def evaluate(model, cfg, device, split="test"):
    model.eval()
    ds = FRCNNDet(f"{cfg.data}/{split}.json", size=cfg.size, train=False)
    coco_gt = COCO(f"{cfg.data}/{split}.json")
    results = []
    for idx in range(len(ds)):
        img_t, _ = ds[idx]
        iid = ds.ids[idx]
        s = ds.scale_of(idx)
        out = model([img_t.to(device)])[0]
        boxes = out["boxes"].cpu().numpy()
        scores = out["scores"].cpu().numpy()
        labels = out["labels"].cpu().numpy()
        keep = scores > 0.05
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
        boxes = boxes / s
        for b, sc, lb in zip(boxes, scores, labels):
            x1, y1, x2, y2 = b
            results.append({"image_id": int(iid), "category_id": int(lb) - 1,
                            "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                            "score": float(sc)})
    if not results:
        print("  !! no detections above score threshold — skipping COCOeval")
        return None
    coco_dt = coco_gt.loadRes(results)
    ev = COCOeval(coco_gt, coco_dt, "bbox")
    ev.evaluate(); ev.accumulate(); ev.summarize()
    return ev.stats  # [AP@[.5:.95], AP@.5, AP@.75, APs, APm, APl, AR1, AR10, AR100, ARs, ARm, ARl]

def main(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("="*70 + f"\n  PAPER 1 SEC.4.7 . FASTER R-CNN (ResNet-50 FPN) BASELINE  device={device}\n" + "="*70)
    model, n_par = train(cfg, device)
    print("\n" + "="*70 + "\n  TEST-SPLIT EVALUATION (pycocotools)\n" + "="*70)
    stats = evaluate(model, cfg, device, split="test")
    if stats is not None:
        print(f"\n  Table 5 row (paste into paper):")
        print(f"  Faster R-CNN (ResNet-50 FPN)   params={n_par:.1f}M   mAP@0.5={stats[1]:.3f}   mAP@[.5:.95]={stats[0]:.3f}")
    torch.save(model.state_dict(), f"{cfg.out}/fasterrcnn_baseline.pth")
    print(f"\n  Saved checkpoint to {cfg.out}/fasterrcnn_baseline.pth")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="D:/aquavisionnet/avn_out")
    p.add_argument("--out", default="D:/aquavisionnet/avn_out")
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--lr", type=float, default=0.005)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    main(p.parse_args())
