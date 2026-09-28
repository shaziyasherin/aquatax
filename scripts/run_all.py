#!/usr/bin/env python3
"""
run_all.py - one driver for the whole HAZADV-D-26-02212 revision measurement run.

It does not implement any science. It calls the existing, already-tested scripts in the right
order, skips work whose output already exists, captures every stdout to results\\logs\\, and
appends one row per step to results\\RUN_LOG.md. A step that fails is reported and the stage
continues, so one missing checkpoint never kills the whole run.

Setup (once):
    copy revision_paths.template.json revision_paths.json
    notepad revision_paths.json          <- fill in the paths the inventory found
    python scripts\\run_all.py --stage check

Then:
    python scripts\\run_all.py --stage all

Stages, in order:
    check    paths exist, env is sane, the two test.json files are the same split   (seconds)
    facts    exact per-class box counts (Table 3 plastic; R4.5)                      (seconds)
    leakage  R1.6 extended pHash audit: sweep, 3 directions, per-source, exact dups (~10 min)
    preds    COCO predictions on test+val for every detector that has a checkpoint  (~20 min GPU)
    summary  full pycocotools summary per model (scale-wise AP/AR for Figure 3)     (minutes)
    stats    R1.6 clean, R1.9 bootstrap + paired compare, R1.10 counts              (1-2 h CPU)
    collect  results\\FILL_SHEET.md                                                  (seconds)

Useful flags:
    --stage preds --only frcnn,avn15     run one stage for a subset of models
    --force                              redo steps whose output already exists
    --dry-run                            print the commands without running them
    --B 200                              smaller bootstrap for a fast rehearsal (final run must be 1000)
"""
import argparse, datetime, json, os, shlex, shutil, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                      # D:\aquavisionnet\revision
RESULTS = ROOT / "results"
PREDS = RESULTS / "preds"
STATS = RESULTS / "stats"
LOGS = RESULTS / "logs"
RUN_LOG = RESULTS / "RUN_LOG.md"

MATCHED = ["yolov8m", "frcnn", "avn15"]
DINO = ["dino_plain", "dino_tta"]
ABL = ["ablA", "ablB", "ablB2"]
ALL_MODELS = MATCHED + DINO + ABL

C_OK, C_SKIP, C_FAIL = "  OK  ", " SKIP ", " FAIL "


# ----------------------------------------------------------------- infrastructure
class Ctx:
    def __init__(self, a):
        self.a = a
        self.cfg = json.loads(Path(a.config).read_text(encoding="utf-8"))
        self.py = self.cfg.get("python") or sys.executable
        self.only = set(x.strip() for x in a.only.split(",")) if a.only else None
        self.rows = []
        for d in (RESULTS, PREDS, STATS, LOGS):
            d.mkdir(parents=True, exist_ok=True)

    def p(self, key, required=False):
        v = (self.cfg.get(key) or "").strip()
        if required and not v:
            raise KeyError(f"revision_paths.json is missing '{key}'")
        return v

    def want(self, model):
        return self.only is None or model in self.only

    def log(self, step, cmd, outputs, status, note=""):
        self.rows.append((step, cmd, outputs, status, note))
        stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        line = f"| {stamp} | {step} | `{cmd}` | {outputs} | {status.strip()} | {note} |\n"
        if not RUN_LOG.exists():
            RUN_LOG.write_text("# RUN LOG\n\n| date | step | command | output | status | note |\n"
                               "|---|---|---|---|---|---|\n", encoding="utf-8")
        with RUN_LOG.open("a", encoding="utf-8") as f:
            f.write(line)


def run(ctx, step, cmd, outputs, stdout_to=None, cwd=None):
    """Run one command. Skip if every output already exists. Never raises."""
    outs = [Path(o) for o in (outputs if isinstance(outputs, (list, tuple)) else [outputs])]
    if outs and all(o.exists() and o.stat().st_size > 0 for o in outs) and not ctx.a.force:
        print(f"[{C_SKIP}] {step}  (output exists; --force to redo)")
        ctx.log(step, "skipped", ", ".join(o.name for o in outs), C_SKIP, "output already present")
        return True
    printable = " ".join(shlex.quote(str(c)) for c in cmd)
    print(f"[ run  ] {step}\n         {printable}")
    if ctx.a.dry_run:
        return True
    logf = LOGS / f"{step}.log"
    try:
        with logf.open("w", encoding="utf-8", errors="replace") as lf:
            proc = subprocess.run([str(c) for c in cmd], cwd=str(cwd or ROOT),
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  text=True, encoding="utf-8", errors="replace")
            lf.write(proc.stdout or "")
        if stdout_to:
            Path(stdout_to).write_text(proc.stdout or "", encoding="utf-8")
        tail = "\n".join((proc.stdout or "").strip().splitlines()[-12:])
        if proc.returncode != 0:
            print(f"[{C_FAIL}] {step}  exit {proc.returncode}   full log: {logf}")
            print("         " + tail.replace("\n", "\n         "))
            ctx.log(step, printable, ", ".join(o.name for o in outs), C_FAIL, f"exit {proc.returncode}")
            return False
        print("         " + tail.replace("\n", "\n         "))
        missing = [o.name for o in outs if not o.exists()]
        if missing:
            print(f"[{C_FAIL}] {step}  ran but did not produce: {missing}")
            ctx.log(step, printable, ", ".join(o.name for o in outs), C_FAIL, f"missing {missing}")
            return False
        print(f"[{C_OK}] {step}")
        ctx.log(step, printable, ", ".join(o.name for o in outs), C_OK)
        return True
    except Exception as e:
        print(f"[{C_FAIL}] {step}  {type(e).__name__}: {e}")
        ctx.log(step, printable, "-", C_FAIL, f"{type(e).__name__}: {e}")
        return False


# ----------------------------------------------------------------- stages
def stage_check(ctx):
    print("\n=== check =========================================================")
    ok = True
    run(ctx, "check_env", [ctx.py, HERE / "check_env.py"], [], stdout_to=None)

    need = {"gt_test": True, "gt_val": True, "src": False, "mmdet": False,
            "ckpt_frcnn": False, "ckpt_avn": False, "ckpt_yolo": False,
            "ckpt_dino": False, "dino_cfg": False, "flagged_src": False,
            "tta_test_json": False, "tta_val_json": False, "dino_ann_test": False}
    print("\npath check:")
    for k, required in need.items():
        v = ctx.cfg.get(k, "")
        if not v:
            print(f"  {'MISSING ' if required else 'not set '} {k}")
            ok = ok and not required
        elif Path(v).exists():
            print(f"  ok       {k}  = {v}")
        else:
            print(f"  NOT FOUND {k} = {v}")
            ok = False

    # the single most important consistency check in the whole revision
    gt_test, dino_ann = ctx.p("gt_test", True), ctx.p("dino_ann_test")
    if dino_ann and Path(dino_ann).exists():
        a = {i["id"] for i in json.loads(Path(gt_test).read_text(encoding="utf-8"))["images"]}
        b = {i["id"] for i in json.loads(Path(dino_ann).read_text(encoding="utf-8"))["images"]}
        same = a == b
        print(f"\nsplit identity: matched n={len(a)}  dino n={len(b)}  overlap={len(a & b)}  identical={same}")
        if not same:
            print("  *** STOP. The matched detectors and DINO were scored on different test sets.")
            print("  *** No cross-model row in Tables 6 or 8 is valid until this is resolved.")
            ok = False
        ctx.log("split_identity", "python inline", "-", C_OK if same else C_FAIL,
                f"matched {len(a)}, dino {len(b)}, identical {same}")
    else:
        print("\nsplit identity: SKIPPED (dino_ann_test not set) - do this before trusting Table 6/8")
    print("\ncheck:", "PASS" if ok else "PROBLEMS ABOVE")
    return ok


def stage_facts(ctx):
    print("\n=== facts =========================================================")
    data = ctx.p("data", True)
    splits = [f"train={data}\\train.json", f"val={data}\\val.json", f"test={data}\\test.json"]
    aug = ctx.p("train_aug_json")
    if aug:
        splits.append(f"train_aug={aug}")
    cmd = [ctx.py, HERE / "01_count_boxes.py"]
    for s in splits:
        cmd += ["--split", s]
    cmd += ["--out", STATS / "box_counts.md"]
    run(ctx, "box_counts", cmd, [STATS / "box_counts.md"])
    print("  -> this is the exact plastic count for Table 3; R4.5 flagged '~5,538' as obtained by subtraction")

    validate_flagged(ctx)


def validate_flagged(ctx):
    flagged = RESULTS / "flagged_ids.txt"
    if not flagged.exists():
        print(f"[{C_SKIP}] flagged_ids  (not built yet - run --stage leakage)")
        return False
    f = {int(x) for x in flagged.read_text(encoding="utf-8").split()}
    g = {i["id"] for i in json.loads(Path(ctx.p("gt_test", True)).read_text(encoding="utf-8"))["images"]}
    ok = len(f) == len(f & g) and len(f) > 0
    print(f"  flagged={len(f)}  present in test={len(f & g)}  (published figure: 45 of 389)")
    if not ok:
        print("  *** flagged ids are not all inside test.json - the two annotation sets disagree")
    elif len(f) != 45:
        print(f"  note: this run flags {len(f)}, the paper says 45. Report the number this run")
        print("        produces, and say in the letter that the audit was re-run.")
    ctx.log("flagged_ids", "validate", "flagged_ids.txt", C_OK if ok else C_FAIL,
            f"{len(f)} ids, {len(f & g)} in test")
    return ok


def stage_leakage(ctx):
    print("\n=== leakage (R1.6) ================================================")
    ann = ctx.p("ann_dir") or ctx.p("data", True)
    out = RESULTS / "leakage"
    ok = run(ctx, "leakage_audit",
             [ctx.py, HERE / "13_leakage_audit_plus.py", "audit", "--ann-dir", ann,
              "--out", out, "--flagged-out", RESULTS / "flagged_ids.txt"],
             [out / "leakage_report.json", RESULTS / "flagged_ids.txt"])
    if ok:
        print("  -> threshold sweep, all three leakage directions, per-source attribution,")
        print("     and an exact SHA-256 cross-split duplicate check.")
        validate_flagged(ctx)


def stage_manifest(ctx):
    print("\n=== manifest (R4.3 reproducibility artifact) ======================")
    out = RESULTS / "manifest"
    cmd = [ctx.py, HERE / "12_build_manifest.py", "build", "--data", ctx.p("data", True), "--out", out]
    ok = run(ctx, "manifest_build", cmd, [out / "AquaTax4_manifest.csv", out / "manifest_summary.md"])
    if ok:
        print("  -> per-image SHA-256 manifest + measured source shares. This replaces the")
        print("     'FloW-Img is 77% of the corpus' figure with a counted one, and it is the")
        print("     artifact R4.3 asked for. It also reports exact byte-level duplicates, which")
        print("     is a stronger leakage statement than pHash alone.")


def stage_preds(ctx):
    print("\n=== preds =========================================================")
    data, src = ctx.p("data", True), ctx.p("src")
    gt = {"test": ctx.p("gt_test", True), "val": ctx.p("gt_val", True)}

    for model, ckpt_key in (("frcnn", "ckpt_frcnn"), ("avn15", "ckpt_avn")):
        ck = ctx.p(ckpt_key)
        if not ctx.want(model):
            continue
        if not (ck and Path(ck).exists() and src):
            print(f"[{C_SKIP}] {model}  (checkpoint or src not set)")
            continue
        for split in ("test", "val"):
            out = PREDS / f"{model}_{split}.json"
            cmd = [ctx.py, HERE / "10_export_matched.py", "--model", model, "--split", split,
                   "--src", src, "--data", data, "--ckpt", ck, "--out", out]
            if split == "test":
                cmd.append("--time")
            run(ctx, f"export_{model}_{split}", cmd, [out])

    if ctx.want("yolov8m"):
        ck = ctx.p("ckpt_yolo")
        if ck and Path(ck).exists():
            for split in ("test", "val"):
                out = PREDS / f"yolov8m_{split}.json"
                cmd = [ctx.py, HERE / "02_export_yolo.py", "--weights", ck, "--gt", gt[split],
                       "--images", ctx.p("image_root") or ".", "--out", out, "--imgsz", "512"]
                if split == "test":
                    cmd.append("--time")
                run(ctx, f"export_yolov8m_{split}", cmd, [out])
        else:
            print(f"[{C_SKIP}] yolov8m  (ckpt_yolo not set)")

    if ctx.want("dino_plain"):
        cfg, ck, mm = ctx.p("dino_cfg"), ctx.p("ckpt_dino"), ctx.p("mmdet")
        if cfg and ck and mm:
            run(ctx, "export_dino_plain_test",
                [ctx.py, HERE / "05_dino_export.py", "--mmdet", mm, "--cfg", cfg, "--ckpt", ck,
                 "--split", "test", "--benchmark"], [PREDS / "dino_plain_test.json"])
            if ctx.p("dino_ann_val") and ctx.p("dino_img_root"):
                run(ctx, "export_dino_plain_val",
                    [ctx.py, HERE / "05_dino_export.py", "--mmdet", mm, "--cfg", cfg, "--ckpt", ck,
                     "--split", "val", "--val-ann", ctx.p("dino_ann_val"),
                     "--img-root", ctx.p("dino_img_root")], [PREDS / "dino_plain_val.json"])
        else:
            print(f"[{C_SKIP}] dino_plain  (dino_cfg / ckpt_dino / mmdet not set)")

    # TTA+WBF is produced by your own script; this only copies the result into place.
    for split in ("test", "val"):
        srcj, dst = ctx.p(f"tta_{split}_json"), PREDS / f"dino_tta_{split}.json"
        if not ctx.want("dino_tta"):
            continue
        if dst.exists() and not ctx.a.force:
            print(f"[{C_SKIP}] dino_tta_{split}  (already in results\\preds)")
        elif srcj and Path(srcj).exists():
            shutil.copy2(srcj, dst)
            print(f"[{C_OK}] dino_tta_{split}  copied from {srcj}")
            ctx.log(f"copy_dino_tta_{split}", f"copy {srcj}", dst.name, C_OK)
        else:
            print(f"[{C_SKIP}] dino_tta_{split}  - run your TTA+WBF script unchanged and put the "
                  f"COCO results json at {dst}")

    for m in ABL:
        if (PREDS / f"{m}_test.json").exists():
            print(f"[{C_OK}] {m}_test.json present (ablation row exported)")


def present(models):
    return [m for m in models if (PREDS / f"{m}_test.json").exists()]


def stage_summary(ctx):
    print("\n=== summary =======================================================")
    gt = ctx.p("gt_test", True)
    for m in present(ALL_MODELS):
        if not ctx.want(m):
            continue
        out = STATS / f"summary_{m}.json"
        run(ctx, f"summary_{m}",
            [ctx.py, HERE / "06_coco_summary.py", "--gt", gt, "--dets", PREDS / f"{m}_test.json",
             "--name", m, "--out", out], [out])
    print("\n  sanity vs the submitted paper: frcnn 0.275 | avn15 0.184 | dino_plain 0.640 | "
          "dino_tta 0.688")
    print("  yolov8m WILL differ - 0.178 came from the Ultralytics validator, this is pycocotools.")
    print("  Anything else that disagrees: keep the new number, do not tune to match.")


def stage_stats(ctx):
    print("\n=== stats =========================================================")
    gt, gtv = ctx.p("gt_test", True), ctx.p("gt_val", True)
    ra = HERE / "revision_analysis.py"
    flagged = RESULTS / "flagged_ids.txt"
    B, seed = str(ctx.a.B), str(ctx.a.seed)
    have = [m for m in present(ALL_MODELS) if ctx.want(m)]
    if not have:
        print("  no prediction files yet - run --stage preds first")
        return
    print(f"  models with predictions: {', '.join(have)}")

    # R1.6 clean-subset curve across every pHash threshold ---------------
    lk = RESULTS / "leakage"
    if (lk / "flagged_ids_h5.txt").exists():
        run(ctx, "leakage_curve",
            [ctx.py, HERE / "13_leakage_audit_plus.py", "curve", "--gt", gt,
             "--preds", PREDS, "--models", *have, "--audit", lk, "--out", lk],
            [lk / "leakage_curve.csv", lk / "leakage_curve.md"])
        print("  -> R1.6: Table 8 plus the threshold-sensitivity table, and an explicit")
        print("     check of whether the cross-model ranking survives the cleaning.")

    # R1.6 -------------------------------------------------------------
    if flagged.exists():
        for m in have:
            out = STATS / f"clean_{m}.txt"
            run(ctx, f"clean_{m}", [ctx.py, ra, "clean", "--gt", gt, "--dets",
                                    PREDS / f"{m}_test.json", "--flagged", flagged],
                [out], stdout_to=out)
        print("  -> R1.6: full / clean / flagged-only mAP for every detector (Table 8)")
    else:
        print(f"[{C_SKIP}] R1.6 clean  (results\\flagged_ids.txt missing - run --stage facts)")

    # R1.9 -------------------------------------------------------------
    for m in have:
        out, js = STATS / f"boot_{m}.txt", STATS / f"boot_{m}.json"
        run(ctx, f"boot_{m}", [ctx.py, ra, "bootstrap", "--gt", gt, "--dets",
                               PREDS / f"{m}_test.json", "--B", B, "--seed", seed,
                               "--json-out", js], [js], stdout_to=out)
    if flagged.exists():
        for m in [x for x in have if x in ("frcnn", "avn15", "dino_tta")]:
            out, js = STATS / f"bootclean_{m}.txt", STATS / f"bootclean_{m}.json"
            run(ctx, f"bootclean_{m}", [ctx.py, ra, "bootstrap", "--gt", gt, "--dets",
                                        PREDS / f"{m}_test.json", "--B", B, "--seed", seed,
                                        "--subset", "clean", "--flagged", flagged,
                                        "--json-out", js], [js], stdout_to=out)

    pairs = [("frcnn", "avn15"), ("yolov8m", "avn15"), ("frcnn", "yolov8m"),
             ("dino_tta", "dino_plain"),                 # D - C : TTA+WBF
             ("ablB", "ablA"),                           # B - A : detection pretraining
             ("ablB2", "ablB"),                          # B2 - B : SAM copy-paste
             ("dino_plain", "ablB2")]                    # C - B2 : class-balanced sampler
    for x, y in pairs:
        if not ((PREDS / f"{x}_test.json").exists() and (PREDS / f"{y}_test.json").exists()):
            continue
        out = STATS / f"compare_{x}_vs_{y}.txt"
        js = STATS / f"compare_{x}_vs_{y}.json"
        run(ctx, f"compare_{x}_vs_{y}",
            [ctx.py, ra, "compare", "--gt", gt, "--dets", PREDS / f"{x}_test.json",
             "--dets2", PREDS / f"{y}_test.json", "--B", B, "--seed", seed,
             "--json-out", js], [js], stdout_to=out)
    print("  -> R1.9: bootstrap CIs and paired differences (and the R1.8 ablation deltas)")

    # R1.10 ------------------------------------------------------------
    for m in ("dino_tta", "frcnn", "avn15"):
        if m not in have or not (PREDS / f"{m}_val.json").exists():
            continue
        out = STATS / f"counts_{m}.txt"
        run(ctx, f"counts_{m}", [ctx.py, ra, "counts", "--gt", gt, "--dets",
                                 PREDS / f"{m}_test.json", "--val-gt", gtv,
                                 "--val-dets", PREDS / f"{m}_val.json"], [out], stdout_to=out)
    print("  -> R1.10: per-image count MAE, signed bias, per-class over/under-count ratio (Table 9)")


def stage_collect(ctx):
    print("\n=== collect =======================================================")
    run(ctx, "collect", [ctx.py, HERE / "09_collect_results.py", "--results", RESULTS],
        [RESULTS / "FILL_SHEET.md"])
    print(f"\n  open {RESULTS / 'FILL_SHEET.md'} and bring it back to the chat")


STAGES = {"check": stage_check, "facts": stage_facts, "manifest": stage_manifest,
          "leakage": stage_leakage, "preds": stage_preds, "summary": stage_summary,
          "stats": stage_stats, "collect": stage_collect}
ORDER = ["check", "facts", "manifest", "leakage", "preds", "summary", "stats", "collect"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "revision_paths.json"))
    ap.add_argument("--stage", default="all", help="all, or a comma-separated subset of " + ", ".join(ORDER))
    ap.add_argument("--only", default=None, help="comma-separated model names")
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20260729)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    if not Path(a.config).exists():
        sys.exit(f"missing {a.config}\n  copy revision_paths.template.json to it and fill in the paths")
    ctx = Ctx(a)
    stages = ORDER if a.stage == "all" else [s.strip() for s in a.stage.split(",")]
    bad = [s for s in stages if s not in STAGES]
    if bad:
        sys.exit(f"unknown stage(s): {bad}; choose from {ORDER}")

    if a.B != 1000:
        print(f"!! B={a.B}, not 1000. Fine for a rehearsal; the submitted numbers must use B=1000.\n")

    for s in stages:
        if s == "check":
            if not STAGES[s](ctx) and a.stage == "all":
                print("\nStopping: fix the problems above, then re-run. "
                      "(To push past deliberately: --stage facts,preds,summary,stats,collect)")
                return
        else:
            STAGES[s](ctx)

    print("\n=== summary of this run ===========================================")
    for step, _cmd, outs, status, note in ctx.rows:
        print(f"[{status}] {step:32s} {outs} {note}")
    fails = [r for r in ctx.rows if r[3] == C_FAIL]
    print(f"\n{len(ctx.rows)} steps, {len(fails)} failed. Full log: {RUN_LOG}")
    if fails:
        print("failed steps:", ", ".join(r[0] for r in fails))


if __name__ == "__main__":
    main()
