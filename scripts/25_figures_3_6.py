#!/usr/bin/env python3
"""
25_figures_3_6.py - manuscript Figures 3 and 6 for HAZADV-D-26-02212.

Figure 3  scale-wise AP@[.5:.95] and AR@100 of the three matched-protocol detectors (Table 6c).
Figure 6  test mAP@0.5 of all five detectors with 95% image-level bootstrap intervals (Table 4).

Both replace images generated from runs that no longer exist in the paper: Figure 3's old image
came from the removed 24-epoch run and Figure 6 carried a 0.6 reference line the reviewer objected
to. Every value below is a measurement listed in REVISED_PAPER_MATERIAL.md; nothing is estimated.
Design tokens, colours and markers are imported from 17_make_figures.py so the whole figure set
stays one system.

    python 25_figures_3_6.py --out figures
"""
import argparse
import importlib.util
from pathlib import Path

import numpy as np

_spec = importlib.util.spec_from_file_location(
    "mk17", Path(__file__).with_name("17_make_figures.py"))
mk = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mk)

plt = mk.plt
SURFACE, INK, INK_2, MUTED = mk.SURFACE, mk.INK, mk.INK_2, mk.MUTED
DCOLOR, DMARK, MM, W2 = mk.DCOLOR, mk.DMARK, mk.MM, mk.W2
style, save = mk.style, mk.save

MATCHED = ["YOLOv8-m", "Faster R-CNN", "AquaVisionNet"]
SCALES = ["small\n(area < 32²)", "medium\n(32² to 96²)", "large\n(> 96²)"]

# Table 6c: AP@[.5:.95] then AR@100, each small / medium / large.
AP_SCALE = {
    "YOLOv8-m":      [0.306, 0.282, 0.184],
    "Faster R-CNN":  [0.293, 0.273, 0.296],
    "AquaVisionNet": [0.165, 0.234, 0.180],
}
AR_SCALE = {
    "YOLOv8-m":      [0.397, 0.401, 0.348],
    "Faster R-CNN":  [0.380, 0.293, 0.417],
    "AquaVisionNet": [0.275, 0.376, 0.446],
}

# Figure 6: protocol each detector was trained under.
PROTOCOL = {
    "YOLOv8-m": "matched", "Faster R-CNN": "matched", "AquaVisionNet": "matched",
    "DINO-Swin-B": "enhanced", "DINO + TTA/WBF": "enhanced",
}


def fig3(out):
    """Scale-wise precision and recall of the three matched-protocol detectors."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(W2, 70 * MM),
                                   gridspec_kw={"wspace": 0.22})
    x = np.arange(len(SCALES))
    w = 0.26
    for ax, data, ylab, title in (
            (ax1, AP_SCALE, "AP@[.5:.95]",
             "(a)  Weakest where it was designed to be strongest"),
            (ax2, AR_SCALE, "AR@100",
             "(b)  AquaVisionNet's recall advantage is confined to large objects")):
        for i, name in enumerate(MATCHED):
            bars = ax.bar(x + (i - 1) * w, data[name], width=w * 0.92, label=name,
                          color=DCOLOR[name], edgecolor=SURFACE, linewidth=0.8, zorder=3)
            for b, v in zip(bars, data[name]):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.008, f"{v:.3f}",
                        ha="center", va="bottom", fontsize=6.5, color=INK_2, zorder=4)
        ax.set_xticks(x)
        ax.set_xticklabels(SCALES, color=INK)
        ax.set_ylabel(ylab)
        ax.set_ylim(0, 0.50)
        ax.set_title(title, loc="left", color=INK, pad=6)
        style(ax)
    ax1.legend(loc="upper right", handlelength=1.1, borderpad=0.2, labelspacing=0.3)
    save(fig, out, "fig3_scalewise")


def fig6(out):
    """Test mAP@0.5 of every detector, with its 95% bootstrap interval."""
    order = ["AquaVisionNet", "YOLOv8-m", "Faster R-CNN", "DINO-Swin-B", "DINO + TTA/WBF"]
    fig, ax = plt.subplots(figsize=(W2 * 0.62, 66 * MM))
    y = np.arange(len(order))[::-1]
    for yi, name in zip(y, order):
        val, lo, hi = mk.MAP50_CI[name]
        enhanced = PROTOCOL[name] == "enhanced"
        ax.barh(yi, val, height=0.5, color=DCOLOR[name], zorder=3,
                edgecolor=SURFACE, linewidth=0.8, alpha=1.0 if enhanced else 0.55,
                hatch="" if enhanced else "///")
        ax.plot([lo, hi], [yi, yi], color=INK_2, lw=1.3, solid_capstyle="butt", zorder=5)
        for xb in (lo, hi):
            ax.plot([xb, xb], [yi - 0.1, yi + 0.1], color=INK_2, lw=1.3, zorder=5)
        ax.plot([val], [yi], marker=DMARK[name], ms=4.0, color=INK_2,
                markerfacecolor=SURFACE, markeredgewidth=1.0, zorder=6)
        ax.text(hi + 0.012, yi, f"{val:.3f}", va="center", ha="left",
                fontsize=7.5, color=INK_2)
    ax.set_yticks(y)
    ax.set_yticklabels(order, fontsize=7.5, color=INK)
    ax.set_xlim(0, 0.92)
    ax.set_xlabel("test mAP@0.5 (n = 389), with 95% bootstrap interval")
    ax.set_title("Hatched: matched protocol, 512 px and 15 epochs\nSolid: enhanced configuration",
                 loc="left", color=INK_2, fontsize=7.5, pad=6)
    style(ax, grid_axis="x")
    save(fig, out, "fig6_map_overview")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figures")
    a = ap.parse_args()
    out = Path(a.out)
    print("writing figures to", out.resolve())
    fig3(out)
    fig6(out)
    print("done")


if __name__ == "__main__":
    main()
