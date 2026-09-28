#!/usr/bin/env python3
"""
17_make_figures.py - result figures for HAZADV-D-26-02212.

Every value below is a measurement taken on AIMLSERVER between 26 and 28 September 2026 and is
listed in REVISED_PAPER_MATERIAL.md. Nothing here is estimated. The numbers are literals rather
than file reads so that the figures regenerate identically without the GPU box.

    python 17_make_figures.py --out figures

Writes each figure as PDF (vector, for typesetting) and PNG at 400 dpi.

Design notes
  * Categorical hues are assigned in a fixed order, one per detector, and never cycled: the same
    detector is the same colour in every figure.
  * The five-slot subset was validated for colour-vision deficiency separation and the three-slot
    subset for all-pairs use; three slots sit below 3:1 against a white surface, so every series
    carries a direct text label and a distinct marker. Identity never rests on colour alone.
  * No figure uses two y-scales.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ----------------------------------------------------------------- design tokens
SURFACE = "#ffffff"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
NEUTRAL = "#b9b8b2"

SLOT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
MARK = ["o", "s", "^", "D", "v"]

DETECTORS = ["YOLOv8-m", "Faster R-CNN", "AquaVisionNet", "DINO-Swin-B", "DINO + TTA/WBF"]
DCOLOR = dict(zip(DETECTORS, SLOT))
DMARK = dict(zip(DETECTORS, MARK))

MM = 1 / 25.4
W2 = 190 * MM          # Elsevier double column
W1 = 90 * MM           # single column

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "axes.titlesize": 8.5,
    "axes.labelsize": 8,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS,
    "axes.linewidth": 0.6,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "text.color": INK,
    "axes.labelcolor": INK_2,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "legend.frameon": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def style(ax, grid_axis="y"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, axis=grid_axis, color=GRID, linewidth=0.6, linestyle="-", zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=2.5, pad=2)


def save(fig, out, name):
    out.mkdir(parents=True, exist_ok=True)
    for ext, kw in (("pdf", {}), ("png", {"dpi": 400})):
        fig.savefig(out / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.02, **kw)
    plt.close(fig)
    print(f"  wrote {name}.pdf / .png")


# ----------------------------------------------------------------- measured data
ABL = {"A": 0.536, "B": 0.538, "C": 0.640, "D": 0.688}
ABL_STEP_CI = {                   # paired bootstrap, B=1000, seed 20260729
    "B-A": (0.002, -0.085, 0.110),    # compare_ablB_vs_ablA.txt,        P = 0.428
    "C-B": (0.102, -0.075, 0.224),    # compare_dino_plain_vs_ablB.txt,  P = 0.142
    "D-C": (0.048, -0.023, 0.117),    # compare_dino_tta_vs_plain.txt,   P = 0.113
}
ABL_CLASS = {                     # per-class AP@0.5 for each ablation row
    "A": [0.894, 0.483, 0.539, 0.228],
    "B": [0.898, 0.433, 0.468, 0.354],
    "C": [0.897, 0.594, 0.610, 0.457],
    "D": [0.898, 0.605, 0.522, 0.726],
}
CLASSES = ["plastic", "paper", "metal", "glass"]
TEST_BOXES = [835, 19, 18, 6]

HAMMING = [0, 2, 5, 8, 10]
FLAGGED = [2, 20, 45, 108, 148]
CLEAN_N = [387, 369, 344, 281, 241]
CLEAN_MAP = {
    "YOLOv8-m":      [0.334, 0.331, 0.327, 0.328, 0.322],
    "Faster R-CNN":  [0.467, 0.465, 0.462, 0.460, 0.452],
    "AquaVisionNet": [0.314, 0.311, 0.306, 0.309, 0.302],
    "DINO-Swin-B":   [0.640, 0.638, 0.637, 0.638, 0.634],
    "DINO + TTA/WBF":[0.688, 0.686, 0.684, 0.686, 0.685],
}

SOURCE_LEAK = [                    # source, flagged, n, rate %, Wilson CI
    ("AquaTrash",           0, 48,  0.0, (0.0, 7.4)),
    ("FloW-Img",           40, 308, 13.0, (9.7, 17.2)),
    ("AquaSurf-Malnad-223", 5, 33, 15.2, (6.7, 30.9)),
]

MAP50_CI = {                       # point, lo, hi  (image-level bootstrap, B=1000)
    "YOLOv8-m":       (0.334, 0.282, 0.411),
    "Faster R-CNN":   (0.467, 0.370, 0.563),
    "AquaVisionNet":  (0.314, 0.265, 0.414),
    "DINO-Swin-B":    (0.640, 0.510, 0.762),
    "DINO + TTA/WBF": (0.688, 0.534, 0.800),
}
PAIRED = [                         # label, diff, lo, hi, significant
    # compare_dino_plain_vs_frcnn.txt
    ("DINO-Swin-B\nminus Faster R-CNN",   0.173, 0.028, 0.322, True),
    # compare_frcnn_vs_avn.txt
    ("Faster R-CNN\nminus AquaVisionNet", 0.152, 0.055, 0.213, True),
    # compare_dino_tta_vs_plain.txt
    ("TTA+WBF\nminus plain DINO",         0.048, -0.023, 0.117, False),
    # compare_yolo_vs_avn.txt
    ("YOLOv8-m\nminus AquaVisionNet",     0.019, -0.057, 0.070, False),
]
NOISE_FLOOR = 0.007

PER_CLASS_AP = {
    "YOLOv8-m":       [0.818, 0.194, 0.323, 0.000],
    "Faster R-CNN":   [0.808, 0.512, 0.545, 0.002],
    "AquaVisionNet":  [0.638, 0.219, 0.389, 0.012],
    "DINO-Swin-B":    [0.897, 0.594, 0.610, 0.457],
    "DINO + TTA/WBF": [0.898, 0.605, 0.522, 0.726],
}

COUNTS = {                         # threshold, count MAE, per-class predicted/true ratio
    "Faster R-CNN":  (0.65, 0.458, [0.93, 0.84, 0.61, 0.33]),
    "YOLOv8-m":      (0.25, 0.517, [0.98, 0.16, 0.06, 0.00]),
    "AquaVisionNet": (0.35, 1.154, [0.71, 2.47, 1.78, 5.67]),
}
MAP_FOR_COUNTS = {"Faster R-CNN": 0.467, "YOLOv8-m": 0.334, "AquaVisionNet": 0.314}


# ----------------------------------------------------------------- Figure 1
def fig_ablation(out):
    """What each component of the enhanced configuration actually contributes."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(W2, 72 * MM),
                                   gridspec_kw={"width_ratios": [1, 1.25], "wspace": 0.34})

    steps = [("Detection pretraining\n(ImageNet-22k to Objects365)", "B-A"),
             ("Class-balanced sampler", None),
             ("SAM copy-paste", "C-B"),
             ("Test-time augmentation\n+ weighted box fusion", "D-C")]
    y = np.arange(len(steps))[::-1]
    for yi, (label, key) in zip(y, steps):
        if key is None:
            ax1.plot([0, 0.004], [yi, yi], color=NEUTRAL, lw=2, solid_capstyle="butt", zorder=4)
            ax1.text(0.018, yi, "0.000  never resampled", va="center", ha="left",
                     fontsize=7.5, color=INK_2)
            continue
        val, lo, hi = ABL_STEP_CI[key]
        # Every interval includes zero, so no step is drawn as established: the bar carries the
        # point estimate, the whisker the 95% paired-bootstrap interval.
        ax1.barh(yi, val, height=0.46, color=SLOT[0], zorder=3,
                 edgecolor=SURFACE, linewidth=0.8, alpha=0.5)
        ax1.plot([lo, hi], [yi, yi], color=SLOT[0], lw=1.6, solid_capstyle="butt", zorder=5)
        for xb in (lo, hi):
            ax1.plot([xb, xb], [yi - 0.11, yi + 0.11], color=SLOT[0], lw=1.6, zorder=5)
        ax1.plot([val], [yi], marker="o", ms=4.2, color=SLOT[0],
                 markeredgecolor=SURFACE, markeredgewidth=0.8, zorder=6)
        ax1.text(hi + 0.008, yi, f"+{val:.3f}", va="center", ha="left",
                 fontsize=7.5, color=INK_2)
    ax1.axvline(0, color=INK_2, lw=0.9, zorder=2)
    ax1.axvline(NOISE_FLOOR, color=MUTED, lw=0.9, ls=(0, (3, 2)), zorder=2)
    ax1.set_yticks(y); ax1.set_yticklabels([s[0] for s in steps], fontsize=7.5, color=INK)
    ax1.set_xlim(-0.125, 0.30)
    ax1.set_ylim(-1.15, len(steps) - 0.35)
    ax1.text(NOISE_FLOOR + 0.012, -0.70, "run-to-run floor, 0.007",
             fontsize=7, color=MUTED, va="center", ha="left")
    ax1.plot([NOISE_FLOOR], [-0.70], marker="o", ms=2.5, color=MUTED, zorder=3)
    ax1.set_xlabel("contribution to test mAP@0.5, with 95% interval")
    ax1.set_title("(a)  Every step's interval includes zero", loc="left", color=INK, pad=6)
    style(ax1, grid_axis="x")
    ax1.text(0.30, -1.02, f"A {ABL['A']:.3f}  →  D {ABL['D']:.3f}",
             fontsize=7.5, color=INK_2, ha="right", va="center")

    x = np.arange(len(CLASSES))
    series = [("Detection pretraining", np.array(ABL_CLASS["B"]) - np.array(ABL_CLASS["A"])),
              ("SAM copy-paste", np.array(ABL_CLASS["C"]) - np.array(ABL_CLASS["B"])),
              ("TTA + WBF", np.array(ABL_CLASS["D"]) - np.array(ABL_CLASS["C"]))]
    w = 0.26
    for i, (name, vals) in enumerate(series):
        ax2.bar(x + (i - 1) * w, vals, width=w * 0.92, label=name, color=SLOT[i],
                edgecolor=SURFACE, linewidth=0.8, zorder=3)
    ax2.axhline(0, color=AXIS, lw=0.8, zorder=2)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{c}\n{n} boxes" for c, n in zip(CLASSES, TEST_BOXES)], color=INK)
    ax2.set_ylabel("change in AP@0.5")
    ax2.set_title("(b)  Every effect is confined to the three rare classes", loc="left",
                  color=INK, pad=6)
    ax2.legend(loc="upper left", ncol=1, handlelength=1.1, borderpad=0.2, labelspacing=0.3)
    style(ax2)
    save(fig, out, "fig_ablation_decomposition")


# ----------------------------------------------------------------- Figure 2
def fig_leakage(out):
    """Near-duplicate leakage: the flagged count is threshold-dependent, the accuracy is not."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(W2, 70 * MM),
                                   gridspec_kw={"width_ratios": [1.75, 1], "wspace": 0.34})

    nudge = {"YOLOv8-m": 0.010, "AquaVisionNet": -0.010}
    xi = np.arange(len(HAMMING))
    for name in DETECTORS:
        ax1.plot(xi, CLEAN_MAP[name], color=DCOLOR[name], marker=DMARK[name], markersize=3.6,
                 lw=1.6, zorder=3, markeredgecolor=SURFACE, markeredgewidth=0.6)
        ax1.text(xi[-1] + 0.14, CLEAN_MAP[name][-1] + nudge.get(name, 0.0), name,
                 va="center", ha="left", fontsize=7.2, color=INK)
    ax1.set_xticks(xi)
    ax1.set_xticklabels([f"{h}\n{f}" for h, f in zip(HAMMING, FLAGGED)], color=INK)
    ax1.set_xlabel("pHash Hamming threshold  /  test images removed")
    ax1.set_ylabel("mAP@0.5 on the retained images")
    ax1.set_xlim(-0.25, len(HAMMING) + 3.1)
    ax1.set_ylim(0.26, 0.74)
    ax1.set_title("(a)  Removing 2 to 148 images barely moves any score", loc="left",
                  color=INK, pad=6)
    style(ax1)
    ax1.annotate("", xy=(4, 0.288), xytext=(0, 0.288),
                 arrowprops=dict(arrowstyle="<->", color=MUTED, lw=0.8))
    ax1.text(2, 0.281, "74x more images removed", fontsize=7, color=MUTED, ha="center", va="top")

    names = [s[0] for s in SOURCE_LEAK]
    rates = [s[3] for s in SOURCE_LEAK]
    los = [s[3] - s[4][0] for s in SOURCE_LEAK]
    his = [s[4][1] - s[3] for s in SOURCE_LEAK]
    yb = np.arange(len(names))[::-1]
    ax2.barh(yb, rates, height=0.5, color=SLOT[0], edgecolor=SURFACE, linewidth=0.8, zorder=3)
    ax2.errorbar(rates, yb, xerr=[los, his], fmt="none", ecolor=INK_2, elinewidth=0.9,
                 capsize=2.5, capthick=0.9, zorder=4)
    for yi, s in zip(yb, SOURCE_LEAK):
        ax2.text(s[4][1] + 1.2, yi, f"{s[1]}/{s[2]}", va="center", ha="left",
                 fontsize=7.2, color=INK_2)
    ax2.set_yticks(yb)
    ax2.set_yticklabels(names, fontsize=7.5, color=INK)
    ax2.set_xlabel("near-duplicate rate (% of test images)")
    ax2.set_xlim(0, 38)
    ax2.set_title("(b)  Only where scenes recur", loc="left", color=INK, pad=6)
    style(ax2, grid_axis="x")
    save(fig, out, "fig_leakage_audit")


# ----------------------------------------------------------------- Figure 3
def fig_uncertainty(out):
    """Every score and every ranking claim with its interval."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(W2, 72 * MM),
                                   gridspec_kw={"width_ratios": [1.15, 1], "wspace": 0.42})

    y = np.arange(len(DETECTORS))[::-1]
    for yi, name in zip(y, DETECTORS):
        p, lo, hi = MAP50_CI[name]
        ax1.plot([lo, hi], [yi, yi], color=DCOLOR[name], lw=1.8, solid_capstyle="round", zorder=3)
        ax1.plot([p], [yi], marker=DMARK[name], color=DCOLOR[name], markersize=5,
                 markeredgecolor=SURFACE, markeredgewidth=0.8, zorder=4)
        ax1.text(hi + 0.012, yi, f"{p:.3f}", va="center", ha="left", fontsize=7.2, color=INK_2)
    ax1.set_yticks(y); ax1.set_yticklabels(DETECTORS, fontsize=7.5, color=INK)
    ax1.set_xlabel("test mAP@0.5 with 95% bootstrap interval")
    ax1.set_xlim(0.22, 0.90)
    ax1.set_title("(a)  Intervals are wide at this test-set size", loc="left", color=INK, pad=6)
    style(ax1, grid_axis="x")

    y2 = np.arange(len(PAIRED))[::-1]
    for yi, (label, d, lo, hi, sig) in zip(y2, PAIRED):
        col = SLOT[0] if sig else NEUTRAL
        ax2.plot([lo, hi], [yi, yi], color=col, lw=1.8, solid_capstyle="round", zorder=3)
        ax2.plot([d], [yi], marker="o", color=SURFACE if not sig else col, markersize=5,
                 markeredgecolor=col, markeredgewidth=1.3, zorder=4)
        ax2.text(hi + 0.008, yi, f"{d:+.3f}" + ("" if sig else "  n.s."),
                 va="center", ha="left", fontsize=7.2, color=INK_2)
    ax2.axvline(0, color=INK_2, lw=0.9, zorder=2)
    ax2.set_yticks(y2); ax2.set_yticklabels([p[0] for p in PAIRED], fontsize=7.2, color=INK)
    ax2.set_xlabel("paired difference in mAP@0.5")
    ax2.set_xlim(-0.10, 0.44)
    ax2.set_ylim(-0.7, len(PAIRED) - 0.3)   # room for the two-line tick labels
    ax2.set_title("(b)  Two of four differences include zero", loc="left", color=INK, pad=6)
    style(ax2, grid_axis="x")
    save(fig, out, "fig_uncertainty")


# ----------------------------------------------------------------- Figure 4
def fig_rarity(out):
    fig, ax = plt.subplots(figsize=(W1 * 1.55, 66 * MM))
    # Three detectors land within 0.012 of zero on glass, so their end labels are staggered and
    # joined to their own line by a hairline leader rather than overprinting one another.
    LABEL_Y = {"DINO + TTA/WBF": 0.726, "DINO-Swin-B": 0.457,
               "AquaVisionNet": 0.128, "Faster R-CNN": 0.070, "YOLOv8-m": 0.012}
    x = np.arange(len(CLASSES))
    for name in DETECTORS:
        ax.plot(x, PER_CLASS_AP[name], color=DCOLOR[name], marker=DMARK[name], markersize=4,
                lw=1.6, zorder=3, markeredgecolor=SURFACE, markeredgewidth=0.6)
        y_end, y_lab = PER_CLASS_AP[name][-1], LABEL_Y[name]
        if abs(y_lab - y_end) > 0.005:
            ax.plot([x[-1] + 0.03, x[-1] + 0.22], [y_end, y_lab], color=DCOLOR[name],
                    lw=0.7, zorder=2)
        ax.text(x[-1] + 0.26, y_lab, name, va="center", ha="left", fontsize=7.2, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c}\n{n} test boxes" for c, n in zip(CLASSES, TEST_BOXES)], color=INK)
    ax.set_ylabel("AP@0.5")
    ax.set_xlim(-0.15, len(CLASSES) + 1.75)
    ax.set_ylim(-0.03, 1.0)
    ax.axhspan(-0.03, 0.05, color=GRID, alpha=0.55, zorder=1)
    ax.text(-0.08, 0.018, "undetected", fontsize=7, color=MUTED, va="center", ha="left")
    ax.set_title("Accuracy collapses with class rarity; three detectors never find glass",
                 loc="left", color=INK, pad=6)
    style(ax)
    save(fig, out, "fig_class_rarity")


# ----------------------------------------------------------------- Figure 5
def fig_counting(out):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(W2, 66 * MM),
                                   gridspec_kw={"width_ratios": [1, 1.3], "wspace": 0.3})

    for name, (thr, mae, _) in COUNTS.items():
        ax1.plot([MAP_FOR_COUNTS[name]], [mae], marker=DMARK[name], markersize=7,
                 color=DCOLOR[name], markeredgecolor=SURFACE, markeredgewidth=0.9, zorder=4)
        ax1.annotate(f"{name}\nthreshold {thr:.2f}", (MAP_FOR_COUNTS[name], mae),
                     textcoords="offset points", xytext=(0, 11), ha="center",
                     fontsize=7.2, color=INK)
    ax1.set_xlabel("test mAP@0.5")
    ax1.set_ylabel("per-image debris count error (items)")
    ax1.set_xlim(0.27, 0.53)
    ax1.set_ylim(0.30, 1.45)
    ax1.set_title("(a)  Each detector needs its own threshold", loc="left", color=INK, pad=6)
    style(ax1)

    x = np.arange(len(CLASSES))
    order = ["Faster R-CNN", "YOLOv8-m", "AquaVisionNet"]
    w = 0.26
    for i, name in enumerate(order):
        ax2.bar(x + (i - 1) * w, COUNTS[name][2], width=w * 0.92, label=name,
                color=DCOLOR[name], edgecolor=SURFACE, linewidth=0.8, zorder=3)
    ax2.axhline(1.0, color=INK_2, lw=0.9, zorder=2)
    ax2.text(-0.42, 1.12, "exact count", fontsize=7, color=INK_2, ha="left", va="bottom")
    ax2.set_xticks(x); ax2.set_xticklabels(CLASSES, color=INK)
    ax2.set_ylabel("predicted count / true count")
    ax2.set_ylim(0, 6.1)
    ax2.set_title("(b)  Per class, the ranking reverses", loc="left", color=INK, pad=6)
    ax2.legend(loc="upper left", handlelength=1.1, borderpad=0.2, labelspacing=0.3)
    style(ax2)
    save(fig, out, "fig_counting_bridge")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figures")
    a = ap.parse_args()
    out = Path(a.out)
    print("writing figures to", out.resolve())
    fig_ablation(out)
    fig_leakage(out)
    fig_uncertainty(out)
    fig_rarity(out)
    fig_counting(out)
    print("done")


if __name__ == "__main__":
    main()
