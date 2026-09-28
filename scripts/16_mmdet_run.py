#!/usr/bin/env python3
"""
16_mmdet_run.py - train or test an mmdet 3.x config without the mmdetection source tree.

mmdet is installed here as a pip package (3.3.0), so `tools/train.py` and `tools/test.py` are not
on disk. This is the same thing those two scripts do: build a Runner from the config and call
train() or test(). Nothing about the model, data or schedule is changed.

  train   python scripts\\16_mmdet_run.py train --config configs\\dino_ablation_B.py
  test    python scripts\\16_mmdet_run.py test --config configs\\dino_ablation_B.py ^
              --ckpt <best epoch .pth> --out results\\preds\\ablB_test.json
  dump    python scripts\\16_mmdet_run.py dump --config configs\\dino_ablation_B.py ^
              --out results\\ablB_config_dump.py

`dump` writes the fully-resolved config so you can diff an ablation row against the reported run
and prove only one thing changed. Always do that before spending seven hours on a training run.

`test` points CocoMetric's outfile_prefix at your chosen path, so mmdet writes
<out without .json>.bbox.json, which this script then renames to exactly --out.
"""
import argparse, os, shutil, sys
from pathlib import Path


def build(cfg_path, work_dir=None, cfg_options=None):
    from mmengine.config import Config, DictAction  # noqa: F401
    cfg = Config.fromfile(cfg_path)
    if cfg_options:
        cfg.merge_from_dict(cfg_options)
    if work_dir:
        cfg.work_dir = work_dir
    elif cfg.get("work_dir") is None:
        cfg.work_dir = str(Path("work_dirs") / Path(cfg_path).stem)
    return cfg


def register():
    """mmdet 3.x registers its modules through the default scope; importing is enough in most
    setups, but be explicit so a missing default_scope in the config cannot bite."""
    import mmdet  # noqa: F401
    try:
        from mmdet.utils import register_all_modules
        register_all_modules(init_default_scope=True)
    except Exception:
        pass


def parse_options(pairs):
    """--cfg-options a.b=1 c=xyz  -> nested dict, with ints/floats/bools/None converted."""
    if not pairs:
        return None
    def cast(v):
        for f in (int, float):
            try:
                return f(v)
            except ValueError:
                pass
        return {"True": True, "False": False, "None": None}.get(v, v)
    out = {}
    for p in pairs:
        if "=" not in p:
            sys.exit(f"--cfg-options needs key=value, got {p!r}")
        k, v = p.split("=", 1)
        out[k] = cast(v)
    return out


def cmd_train(a):
    register()
    from mmengine.runner import Runner
    cfg = build(a.config, a.work_dir, parse_options(a.cfg_options))
    print(f"config   : {a.config}")
    print(f"work_dir : {cfg.work_dir}")
    print(f"epochs   : {cfg.get('max_epochs', cfg.train_cfg.get('max_epochs'))}")
    os.makedirs(cfg.work_dir, exist_ok=True)
    Runner.from_cfg(cfg).train()


def cmd_test(a):
    register()
    from mmengine.runner import Runner
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prefix = str(out.with_suffix("")).replace("\\", "/")
    opts = parse_options(a.cfg_options) or {}
    opts["test_evaluator.outfile_prefix"] = prefix
    cfg = build(a.config, a.work_dir or str(out.parent / f"_mmdet_{out.stem}"), opts)
    cfg.load_from = a.ckpt
    cfg.resume = False
    print(f"config   : {a.config}\nckpt     : {a.ckpt}\npredictions -> {out}")
    Runner.from_cfg(cfg).test()
    src = Path(prefix + ".bbox.json")
    if src.exists():
        shutil.move(str(src), str(out))
        print(f"saved {out}")
    else:
        print(f"!! expected {src}; look in {out.parent} for what CocoMetric actually wrote")


def cmd_dump(a):
    register()
    cfg = build(a.config, a.work_dir, parse_options(a.cfg_options))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(cfg.pretty_text, encoding="utf-8")
    print(f"wrote {a.out}  ({len(cfg.pretty_text.splitlines())} lines)")
    print("diff this against the reported config's dump; only the intended field may differ.")


def _flatten(d, prefix=""):
    out = {}
    if isinstance(d, dict):
        if not d:
            out[prefix] = "{}"
        for k, v in d.items():
            out.update(_flatten(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(d, (list, tuple)):
        if not d:
            out[prefix] = "[]"
        for i, v in enumerate(d):
            out.update(_flatten(v, f"{prefix}[{i}]"))
    else:
        out[prefix] = repr(d)
    return out


WRAPPERS = ("ClassBalancedDataset", "RepeatDataset", "ConcatDataset", "MultiImageMixDataset")


def _unwrap_dataset(cfg):
    """Replace a wrapper dataset with the dataset it wraps, recording the wrapper separately.

    Without this, removing a ClassBalancedDataset re-nests every key beneath it and the diff
    reports dozens of spurious changes. With it, the diff shows the one field that really changed
    plus an explicit note that the wrapper is gone.
    """
    for split in ("train_dataloader", "val_dataloader", "test_dataloader"):
        dl = cfg.get(split)
        if not isinstance(dl, dict):
            continue
        ds = dl.get("dataset")
        removed = []
        while isinstance(ds, dict) and ds.get("type") in WRAPPERS and isinstance(ds.get("dataset"), dict):
            removed.append({k: v for k, v in ds.items() if k != "dataset"})
            ds = ds["dataset"]
        if removed:
            dl["dataset"] = ds
            dl["_wrappers_removed_for_diff"] = removed


def cmd_diff(a):
    """Structural, key-by-key diff of two resolved configs. `fc` is useless here because removing
    a dataset wrapper re-indents everything below it; this compares meaning, not text."""
    register()
    from mmengine.config import Config
    da, db = Config.fromfile(a.a).to_dict(), Config.fromfile(a.b).to_dict()
    if not a.no_unwrap:
        for d in (da, db):
            _unwrap_dataset(d)
    A = _flatten(da)
    B = _flatten(db)
    keys = sorted(set(A) | set(B))
    rows = [(k, A.get(k, "<absent>"), B.get(k, "<absent>")) for k in keys
            if A.get(k, "<absent>") != B.get(k, "<absent>")]
    lines = ["A = " + str(a.a), "B = " + str(a.b),
             "%d leaf values in A, %d in B, %d differ" % (len(A), len(B), len(rows)), ""]
    for k, va, vb in rows:
        lines.append(k)
        lines.append("    A: " + va[:200])
        lines.append("    B: " + vb[:200])
    lines.append("")
    lines.append("For a controlled ablation row, every line above must be one you intended.")
    nl = chr(10)
    text = nl.join(lines)
    print(text)
    if a.out:
        open(a.out, "w", encoding="utf-8").write(text + nl)
        print("wrote " + str(a.out))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("diff")
    p.add_argument("--a", required=True, help="reference config (the reported run)")
    p.add_argument("--b", required=True, help="ablation config")
    p.add_argument("--out", default=None)
    p.add_argument("--no-unwrap", action="store_true",
                   help="compare raw nesting instead of unwrapping ClassBalancedDataset etc.")
    p.set_defaults(f=cmd_diff)
    for name, fn in (("train", cmd_train), ("test", cmd_test), ("dump", cmd_dump)):
        p = sp.add_parser(name)
        p.add_argument("--config", required=True)
        p.add_argument("--work-dir", default=None)
        p.add_argument("--cfg-options", nargs="+", default=None)
        if name == "test":
            p.add_argument("--ckpt", required=True)
            p.add_argument("--out", required=True)
        if name == "dump":
            p.add_argument("--out", required=True)
        p.set_defaults(f=fn)
    a = ap.parse_args()
    a.f(a)


if __name__ == "__main__":
    main()
