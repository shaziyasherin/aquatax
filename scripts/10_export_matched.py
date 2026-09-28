"""
10_export_matched.py - dump COCO-format predictions for the two matched-protocol detectors
(Faster R-CNN and the 15-epoch AquaVisionNet) from their SAVED checkpoints, with NO retraining.

Why this file exists
--------------------
It does not re-implement anything. It imports the two standalone scripts that already produced
the paper's numbers and calls their own Dataset / model / decode code verbatim:

    fasterrcnn_baseline_standalone.py  ->  FRCNNDet, build_model   (letterbox 512, label = cat_id + 1)
    avn_full_testsplit_standalone.py   ->  CocoDet, AquaVisionNet, decode  (letterbox 512, NMS 0.6)

The only thing added is: write the per-image detections to a COCO results .json instead of
throwing them away after COCOeval, plus optional batch-1 throughput.

Do NOT use the kit's 03_export_frcnn.py (it resizes with torchvision's min_size/max_size, which is
NOT the letterbox these weights were trained with) or 04_export_avn.py (stub, raises NotImplementedError).

Usage (Anaconda Prompt, env aquavision)
---------------------------------------
  python scripts\10_export_matched.py --model frcnn --split test ^
      --src D:\aquavisionnet\code --data D:\aquavisionnet\avn_out ^
      --ckpt D:\aquavisionnet\avn_out\fasterrcnn_baseline.pth ^
      --out results\preds\frcnn_test.json --time

  python scripts\10_export_matched.py --model avn15 --split test ^
      --src D:\aquavisionnet\code --data D:\aquavisionnet\avn_out ^
      --ckpt D:\aquavisionnet\avn_out\avn_full_testsplit.pth ^
      --out results\preds\avn15_test.json --time

--src is the folder that holds the two standalone .py files.
--data is the folder holding train.json / val.json / test.json (image file_name entries are absolute).
"""
import argparse, importlib.util, json, os, sys, time
from pathlib import Path

FILES = {"frcnn": "fasterrcnn_baseline_standalone.py",
         "avn15": "avn_full_testsplit_standalone.py"}


def load_module(path):
    path = Path(path)
    if not path.exists():
        sys.exit(f"not found: {path}")
    spec = importlib.util.spec_from_file_location(path.stem, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = mod
    spec.loader.exec_module(mod)          # safe: training is guarded by __main__
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(FILES), required=True)
    ap.add_argument("--split", choices=["test", "val"], required=True)
    ap.add_argument("--src", required=True, help="folder containing the standalone .py scripts")
    ap.add_argument("--data", required=True, help="folder containing train/val/test.json")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--score-thr", type=float, default=0.05, help="same 0.05 the paper's evaluate() used")
    ap.add_argument("--time", action="store_true")
    a = ap.parse_args()

    import torch
    M = load_module(Path(a.src) / FILES[a.model])
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ann = f"{a.data}/{a.split}.json"
    print(f"model={a.model} split={a.split} device={dev}\n  ckpt={a.ckpt}\n  ann ={ann}")

    if a.model == "frcnn":
        model = M.build_model(dev, num_classes=len(M.CLASSES) + 1, size=a.size)
        ds = M.FRCNNDet(ann, size=a.size, train=False)
    else:
        model = M.AquaVisionNet(pretrained=False).to(dev)
        ds = M.CocoDet(ann, size=a.size, train=False)
    sd = torch.load(a.ckpt, map_location=dev)
    sd = sd.get("state_dict", sd.get("model", sd)) if isinstance(sd, dict) else sd
    model.load_state_dict(sd, strict=True)
    model.eval()
    print(f"  loaded state_dict ok ({sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params)")

    dets, times = [], []
    with torch.no_grad():
        for idx in range(len(ds)):
            iid, s = ds.ids[idx], ds.scale_of(idx)
            if dev == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            item = ds[idx]
            img_t = item[0]
            if a.model == "frcnn":
                out = model([img_t.to(dev)])[0]
                boxes = out["boxes"].cpu().numpy()
                scores = out["scores"].cpu().numpy()
                labels = out["labels"].cpu().numpy()
                keep = scores > a.score_thr
                boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
                boxes = boxes / s
                rows = [(b, sc, int(lb) - 1) for b, sc, lb in zip(boxes, scores, labels)]
            else:
                out = model(img_t.unsqueeze(0).to(dev))
                bx, sc, cl = M.decode(out, a.size, conf=a.score_thr)[0]
                if bx.shape[0]:
                    from torchvision.ops import batched_nms
                    keep = batched_nms(bx, sc, cl, 0.6)
                    bx, sc, cl = bx[keep].cpu(), sc[keep].cpu(), cl[keep].cpu()
                bx = bx / s
                rows = [(b, float(v), int(c)) for b, v, c in zip(bx.numpy(), sc.numpy(), cl.numpy())]
            if dev == "cuda":
                torch.cuda.synchronize()
            if idx >= 10:
                times.append(time.perf_counter() - t0)
            for b, score, cid in rows:
                x1, y1, x2, y2 = [float(v) for v in b]
                dets.append({"image_id": int(iid), "category_id": int(cid),
                             "bbox": [round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)],
                             "score": round(float(score), 5)})
            if (idx + 1) % 50 == 0:
                print(f"  ... {idx + 1}/{len(ds)}")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(dets, open(a.out, "w"))
    print(f"wrote {len(dets)} detections for {len(ds)} images -> {a.out}")

    if a.time and times:
        rec = {"model": a.model, "split": a.split,
               "images_per_s_batch1": round(len(times) / sum(times), 2),
               "ms_per_image": round(1000 * sum(times) / len(times), 2),
               "n_timed": len(times), "size": a.size, "device": dev,
               "note": "image decode + letterbox + normalise + forward + decode/NMS, batch 1"}
        p = a.out.replace(".json", "_throughput.json")
        json.dump(rec, open(p, "w"), indent=1)
        print("throughput:", rec, "->", p)


if __name__ == "__main__":
    main()
