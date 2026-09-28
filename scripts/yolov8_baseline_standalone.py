"""
YOLOv8-m baseline — standalone (Paper 1, Section 4.7 cross-architecture comparison).
Converts the existing COCO corpus (train.json/val.json/test.json at --data) into YOLO
format once, then trains + evaluates YOLOv8-m at the same 512x512 input size used for
AquaVisionNet, using Ultralytics' standard default recipe (no AquaOpt tuning — this is
the "standard recipe per architecture" baseline, kept separate on purpose for a fair
cross-architecture comparison).

Prerequisite (one-time):
    pip install ultralytics

Usage:
    python yolov8_baseline_standalone.py --data D:\\aquavisionnet\\avn_out --epochs 15 --workers 0
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # avoid OMP Error #15 (opencv-python vs torch OpenMP DLL clash)

import argparse, json, shutil
from pathlib import Path

CLASSES = ["plastic", "paper", "metal", "glass"]

def build_yolo_dataset(data_root, yolo_root, classes=CLASSES):
    for split in ("train", "val", "test"):
        json_path = f"{data_root}/{split}.json"
        d = json.load(open(json_path))
        imgs = {i["id"]: i for i in d["images"]}
        by_img = {i: [] for i in imgs}
        for a in d["annotations"]:
            by_img[a["image_id"]].append(a)

        img_out = Path(yolo_root) / "images" / split
        lbl_out = Path(yolo_root) / "labels" / split
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        n = 0
        for iid, info in imgs.items():
            src = info["file_name"]
            ext = os.path.splitext(src)[1] or ".jpg"
            stem = f"{split}_{iid}"
            dst = img_out / f"{stem}{ext}"
            if not dst.exists():
                shutil.copy2(src, dst)
            W, H = info["width"], info["height"]
            lines = []
            for a in by_img[iid]:
                x, y, w, h = a["bbox"]
                cx, cy = (x + w / 2) / W, (y + h / 2) / H
                nw, nh = w / W, h / H
                lines.append(f"{a['category_id']} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
            (lbl_out / f"{stem}.txt").write_text("\n".join(lines))
            n += 1
        print(f"  [{split}] {n} images -> {img_out}")

    yaml_path = Path(yolo_root) / "data.yaml"
    names_str = "[" + ", ".join(f"'{c}'" for c in classes) + "]"
    yaml_path.write_text(
        f"path: {yolo_root}\ntrain: images/train\nval: images/val\ntest: images/test\n"
        f"nc: {len(classes)}\nnames: {names_str}\n"
    )
    return str(yaml_path)

def main(cfg):
    yaml_path = f"{cfg.yolo_out}/data.yaml"
    if cfg.rebuild or not os.path.isfile(yaml_path):
        print("="*70 + "\n  Building YOLO-format dataset (one-time)\n" + "="*70)
        yaml_path = build_yolo_dataset(cfg.data, cfg.yolo_out)
    else:
        print(f"Reusing existing YOLO dataset at {yaml_path} (pass --rebuild to redo)")

    from ultralytics import YOLO
    model = YOLO(cfg.model)
    n_par = sum(p.numel() for p in model.model.parameters()) / 1e6

    print("\n" + "="*70 + f"\n  PAPER 1 SEC.4.7 . YOLOv8-m BASELINE  params={n_par:.1f}M\n" + "="*70)
    model.train(data=yaml_path, epochs=cfg.epochs, imgsz=cfg.imgsz, batch=cfg.batch,
                device=0, workers=cfg.workers, seed=0, project=cfg.yolo_out, name="run",
                exist_ok=True, verbose=True)

    print("\n" + "="*70 + "\n  TEST-SPLIT EVALUATION\n" + "="*70)
    metrics = model.val(data=yaml_path, split="test", imgsz=cfg.imgsz, batch=cfg.batch, device=0)

    print(f"\n  Table 5 row (paste into paper):")
    print(f"  YOLOv8-m   params={n_par:.1f}M   mAP@0.5={metrics.box.map50:.3f}   mAP@[.5:.95]={metrics.box.map:.3f}")
    print(f"\n  Per-class AP@0.5 (index order = {CLASSES}):")
    if hasattr(metrics.box, "ap50") and metrics.box.ap50 is not None:
        for c, ap in zip(CLASSES, metrics.box.ap50):
            print(f"    {c:10s} AP@0.5={ap:.3f}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="D:/aquavisionnet/avn_out")
    p.add_argument("--yolo_out", default="D:/aquavisionnet/yolo_data")
    p.add_argument("--model", default="yolov8m.pt")
    p.add_argument("--imgsz", type=int, default=512)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--rebuild", action="store_true")
    main(p.parse_args())
