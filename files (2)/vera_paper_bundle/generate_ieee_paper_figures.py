"""
Publication figures for the VERA IEEE conference paper (revised layout).

Changes versus the first version
--------------------------------
* Every figure is drawn at its final IEEE size (3.5 in single column / 7.16 in
  double column) with 6-7.5 pt text, so nothing is shrunk further in LaTeX.
* Captions live in the .tex file, so the in-figure titles were removed.
* Fig. 1: re-laid-out boxes so no text touches a box edge; arrows no longer cross text.
* Fig. 2: transposed to 4 rows x 5 grades; the long row labels that forced a very wide
  left margin are gone.
* Fig. 3: zero-based y-axes (the old 0.70-0.95 axis exaggerated the gaps), label collisions
  and the arrow/text clash removed, shorter tick labels.
* Fig. 4: the "Referable cutoff" text that ran over the 0.00 cells is now a legend entry;
  the title wrongly said |i-j| <= 1 for "catastrophic" errors (it is |i-j| > 1).
* Fig. 5: 2-DD circle no longer cuts through the optic disc, labels moved off the circles.
* Vector PDF output (fonts embedded as TrueType) plus a 300 dpi PNG of each figure.

IMPORTANT - what is real and what is not
----------------------------------------
Figs. 3, 4 and 6-style numbers are typed in below.  They must be replaced by the values
produced by your evaluation code before submission.  Fig. 2 uses real pipeline outputs
if they exist in <data>/vera_outputs/ (see load_assets); otherwise it falls back to a
crude top-hat proxy and says so in the figure.  Fig. 5 overlays hard-coded geometry and
counts on a sample image.

Usage:  python generate_ieee_paper_figures.py [--data sample_data] [--out figures]
"""
import argparse
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D

COL_W, FULL_W = 3.5, 7.16  # IEEE column widths (inches)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
    "axes.edgecolor": "#333333", "axes.linewidth": 0.6,
    "xtick.color": "#333333", "ytick.color": "#333333",
    "pdf.fonttype": 42, "ps.fonttype": 42,   # embed TrueType (no Type-3 fonts)
})

ARGS = None


def save(fig, name):
    out = Path(ARGS.out)
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Generated {out / name}.pdf/.png")


# ------------------------------------------------------------------ helpers
def read_rgb(path, size=512):
    img = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def placeholder(size=512):
    img = np.full((size, size, 3), 150, np.uint8)
    cv2.putText(img, "PLACEHOLDER", (60, size // 2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 4)
    cv2.putText(img, "sample image missing", (95, size // 2 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    return img


def load_sample(grade):
    p = Path(ARGS.data) / "images" / f"fundus_{grade}_000.png"
    if p.exists():
        return read_rgb(p), True
    print(f"WARNING: {p} not found - drawing a PLACEHOLDER image")
    return placeholder(), False


def load_assets(grade):
    """Return raw, enhanced, vessel, cam, and a flag saying whether panels 3/4 are real.

    Real pipeline outputs (preferred), all optional, in <data>/vera_outputs/:
        {g}_enhanced.png  Stream-A output (RGB)
        {g}_vessel.png    vessel probability map V (gray)
        {g}_gradcam.npy   Grad-CAM++ map, float [0,1], any size
    Missing files fall back to a crude top-hat PROXY, which is NOT your pipeline.
    """
    raw, _ = load_sample(grade)
    d = Path(ARGS.data) / "vera_outputs"
    f_enh, f_ves, f_cam = d / f"{grade}_enhanced.png", d / f"{grade}_vessel.png", d / f"{grade}_gradcam.npy"

    green = raw[:, :, 1]
    g_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(green)
    tophat = cv2.morphologyEx(255 - g_clahe, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    if f_enh.exists():
        enhanced = read_rgb(f_enh)
    else:
        enhanced = raw.copy(); enhanced[:, :, 1] = g_clahe

    if f_ves.exists():
        vessel = cv2.resize(cv2.imread(str(f_ves), 0), (512, 512))
        real_v = True
    else:
        vessel = cv2.threshold(tophat, 15, 255, cv2.THRESH_BINARY)[1]
        real_v = False

    if f_cam.exists():
        cam = cv2.resize(np.load(f_cam).astype(np.float32), (512, 512))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        real_c = True
    else:
        cam = cv2.GaussianBlur(tophat, (15, 15), 0).astype(np.float32)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        real_c = False

    heat = cv2.cvtColor(cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(enhanced, 0.6, heat, 0.4, 0)
    return raw, enhanced, vessel, overlay, real_v, real_c


# --------------------------------------------------------------------- Fig 1
def fig1_architecture():
    fig, ax = plt.subplots(figsize=(FULL_W, 2.75))
    ax.set_xlim(0, 100); ax.set_ylim(0, 38); ax.axis("off")

    def box(x, y, w, h, title, lines, fc, ec):
        ax.add_patch(patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2",
                                            fc=fc, ec=ec, lw=0.9))
        cx = x + w / 2
        n_title = title.count("\n") + 1
        ax.text(cx, y + h - 1.6, title, ha="center", va="top", fontsize=7, fontweight="bold",
                color="#0f172a", linespacing=1.15)
        ax.text(cx, y + h - 1.6 - n_title * 2.15 - 0.9, "\n".join(lines), ha="center", va="top",
                fontsize=6, color="#334155", linespacing=1.35)

    box(0.5, 12.5, 10, 13, "Fundus\nimage", ["512\u00d7512", "RGB"], "#ffffff", "#333333")
    box(14.5, 23.5, 21.5, 13, "Stream A", ["Circular aperture crop", "Ben Graham subtraction", "LAB-space CLAHE"], "#e8f5e9", "#2e7d32")
    box(14.5, 1.5, 21.5, 13, "Stream B", ["Green-channel isolation", "Multi-scale Frangi", "Top-hat morphology"], "#f3e5f5", "#7b1fa2")
    box(40, 10, 17, 18, "4-channel\nfusion", ["[R, G, B, V]", "4\u00d7512\u00d7512", "Conv1: 64\u00d74\u00d77\u00d77", "$W_V$ = mean($W_{RGB}$)"], "#fff3e0", "#ef6c00")
    box(60.5, 11.5, 13, 15, "ResNet-50\nbackbone", ["Residual", "stages 1\u20134"], "#e0f2f1", "#00796b")
    box(77, 23.5, 22.5, 13, "ICDR staging head", ["GAP + dropout + linear", r"$L=(1-\lambda)\,CE+\lambda\,QWK$", r"Acc 88.2%  |  $\kappa$ 0.893"], "#fbe9e7", "#d84315")
    box(77, 1.5, 22.5, 13, "Explainability &\nclinical rules", ["Grad-CAM++ (layer 4)", r"$S_{overlap}$ = 0.342", "ETDRS 4-2-1 + CSME"], "#ede7f6", "#512da8")

    kw = dict(arrowstyle="-|>", lw=0.8, color="#1e293b", shrinkA=0, shrinkB=0, mutation_scale=7)
    for a, b in [((10.5, 19), (14.5, 30)), ((10.5, 19), (14.5, 8)),
                 ((36, 30), (40, 24)), ((36, 8), (40, 14)),
                 ((57, 19), (60.5, 19)),
                 ((73.5, 22), (77, 30)), ((73.5, 16), (77, 8))]:
        ax.annotate("", xy=b, xytext=a, arrowprops=kw)
    ax.text(38.6, 30.3, "RGB", fontsize=6, ha="center", style="italic", color="#2e7d32")
    ax.text(38.6, 6.3, "V", fontsize=6, ha="center", style="italic", color="#7b1fa2")
    save(fig, "fig1_system_architecture")


# --------------------------------------------------------------------- Fig 2
def fig2_multimodal_panels():
    grades = [("Grade 0", "No DR"), ("Grade 1", "Mild NPDR"), ("Grade 2", "Moderate NPDR"),
              ("Grade 3", "Severe NPDR"), ("Grade 4", "PDR")]
    rows = ["Raw fundus", "Enhanced", "Vessel map $V$", "Grad-CAM++"]
    fig, axes = plt.subplots(4, 5, figsize=(FULL_W, 5.75), gridspec_kw=dict(wspace=0.03, hspace=0.03))
    any_proxy = False
    fig.subplots_adjust(left=0.065, right=0.995, top=0.925, bottom=0.04)
    for g, (gname, gdesc) in enumerate(grades):
        raw, enh, ves, ov, real_v, real_c = load_assets(g)
        any_proxy |= not (real_v and real_c)
        if not real_v: rows[2] = "Vessel PROXY"
        if not real_c: rows[3] = "Saliency PROXY"
        for r, (im, cmap) in enumerate([(raw, None), (enh, None), (ves, "gray"), (ov, None)]):
            ax = axes[r, g]
            ax.imshow(im, cmap=cmap); ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_linewidth(0.4); s.set_color("#999999")
            if r == 0:
                ax.set_title(f"{gname}\n{gdesc}", fontsize=7, fontweight="bold", pad=3)
            if g == 0:
                ax.set_ylabel(rows[r], fontsize=7, fontweight="bold", labelpad=3)
    if any_proxy:
        print("WARNING: vessel/Grad-CAM++ panels use a top-hat PROXY, not VERA outputs. "
              "Put real maps in <data>/vera_outputs/ before using this figure.")
        fig.text(0.5, 0.008, "PROXY panels (rows 3-4): not VERA outputs - replace before submission",
                 ha="center", fontsize=6.5, color="#b30000", fontweight="bold")
    save(fig, "fig2_multimodal_panels")


# --------------------------------------------------------------------- Fig 3
def fig3_ablation():
    labels = ["RGB\nbaseline\n(R18)", "Early\nfusion\n(R18)", "Dual\nbranch\n(R18)", "Attn.\ngated\n(R18)", "VERA\n(R50)"]
    acc = [0.812, 0.838, 0.841, 0.855, 0.882]
    qwk = [0.804, 0.843, 0.849, 0.861, 0.893]
    ovl = [0.221, 0.310, 0.315, 0.328, 0.342]
    x, w = np.arange(5), 0.36
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(FULL_W, 2.55), gridspec_kw=dict(wspace=0.22))

    b1 = a1.bar(x - w / 2, acc, w, label="Accuracy", color="#2b5c8f", ec="black", lw=0.4)
    b2 = a1.bar(x + w / 2, qwk, w, label=r"QWK ($\kappa$)", color="#d95f02", ec="black", lw=0.4)
    for bars in (b1, b2):
        for r in bars:
            a1.annotate(f"{r.get_height():.3f}", (r.get_x() + r.get_width() / 2, r.get_height()),
                        xytext=(0, 1.5), textcoords="offset points", ha="center", va="bottom",
                        fontsize=5.3, rotation=90)
    a1.set_ylim(0, 1.18); a1.set_yticks(np.arange(0, 1.01, 0.2))
    a1.set_ylabel("Score"); a1.set_title("(a) Accuracy and QWK", pad=4)
    a1.legend(loc="upper left", ncol=2, frameon=False, handlelength=1.2, columnspacing=1)

    cols = ["#7570b3"] * 4 + ["#1b9e77"]
    bars = a2.bar(x, ovl, 0.55, color=cols, ec="black", lw=0.4)
    for r in bars:
        a2.annotate(f"{r.get_height():.3f}", (r.get_x() + r.get_width() / 2, r.get_height()),
                    xytext=(0, 1.5), textcoords="offset points", ha="center", va="bottom", fontsize=6)
    a2.annotate("", xy=(4, 0.395), xytext=(0, 0.395), arrowprops=dict(arrowstyle="-|>", lw=0.8, color="#b30000", mutation_scale=7))
    a2.text(2, 0.403, "+54.7% vs. baseline", ha="center", va="bottom", fontsize=6.5, fontweight="bold", color="#b30000")
    a2.set_ylim(0, 0.45); a2.set_ylabel(r"$S_{overlap}$"); a2.set_title("(b) Vessel-attention overlap", pad=4)

    for a in (a1, a2):
        a.set_xticks(x); a.set_xticklabels(labels, fontsize=6)
        a.grid(axis="y", ls="--", lw=0.4, alpha=0.5); a.set_axisbelow(True)
    save(fig, "fig3_ablation_benchmarks")


# --------------------------------------------------------------------- Fig 4
def fig4_confusion():
    cm = np.array([[0.94, 0.06, 0.00, 0.00, 0.00],
                   [0.05, 0.88, 0.07, 0.00, 0.00],
                   [0.00, 0.08, 0.86, 0.06, 0.00],
                   [0.00, 0.00, 0.08, 0.87, 0.05],
                   [0.00, 0.00, 0.00, 0.08, 0.92]])
    lab = ["0\nNo DR", "1\nMild", "2\nModerate", "3\nSevere", "4\nPDR"]
    fig, ax = plt.subplots(figsize=(COL_W, 3.05))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1, ax=ax,
                xticklabels=lab, yticklabels=lab, linewidths=0.4, linecolor="#cccccc",
                annot_kws=dict(fontsize=7, fontweight="bold"),
                cbar_kws=dict(label="Row-normalized rate", shrink=0.85, pad=0.03))
    ax.figure.axes[-1].tick_params(labelsize=6); ax.figure.axes[-1].yaxis.label.set_size(6.5)
    ax.set_xlabel("Predicted ICDR grade", fontweight="bold"); ax.set_ylabel("True ICDR grade", fontweight="bold")
    ax.tick_params(length=0); plt.setp(ax.get_xticklabels(), rotation=0); plt.setp(ax.get_yticklabels(), rotation=0)
    ax.axhline(2, color="#e41a1c", lw=1.4, ls="--"); ax.axvline(2, color="#e41a1c", lw=1.4, ls="--")
    ax.legend(handles=[Line2D([0], [0], color="#e41a1c", lw=1.4, ls="--", label=r"Referable cut-off (grade $\geq$ 2)")],
              loc="upper center", bbox_to_anchor=(0.5, -0.2), frameon=False, fontsize=6.5)
    save(fig, "fig4_confusion_matrix_ieee")


# --------------------------------------------------------------------- Fig 5
def fig5_biomarkers():
    img, found = load_sample(2)
    h, w = img.shape[:2]
    fx, fy = int(w * 0.45), int(h * 0.54)      # fovea / FAZ centre (measured position)
    ox, oy, od_r = int(w * 0.90), int(h * 0.54), int(w * 0.075)  # optic disc (measured position)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(FULL_W, 3.55), gridspec_kw=dict(wspace=0.04))

    # (a) ETDRS quadrants, anchored on the fovea / disc axis
    a1.imshow(img)
    a1.axvline(fx, color="#ffff00", ls="--", lw=1.0, alpha=0.9); a1.axhline(fy, color="#ffff00", ls="--", lw=1.0, alpha=0.9)
    a1.add_patch(patches.Circle((ox, oy), od_r, lw=1.2, ec="#00ff00", fc="none"))
    a1.plot(fx, fy, "o", color="#ff3333", ms=3.5)
    quad = {"ST": ((fx / 2, fy / 2), "Superior-Temporal", 8, 14), "SN": ((fx + (w - fx) / 2, fy / 2), "Superior-Nasal", 6, 9),
            "IT": ((fx / 2, fy + (h - fy) / 2), "Inferior-Temporal", 11, 16), "IN": ((fx + (w - fx) / 2, fy + (h - fy) / 2), "Inferior-Nasal", 5, 7)}
    for k, ((qx, qy), name, hm, ma) in quad.items():
        a1.text(qx, qy, f"{name} ({k})\nHM {hm} | MA {ma}", color="white", fontsize=6, fontweight="bold",
                ha="center", va="center", bbox=dict(boxstyle="round,pad=0.25", fc="black", alpha=0.65, ec="none"))
    a1.set_title("(a) ETDRS 4-quadrant partitioning", fontsize=7.5, fontweight="bold", pad=3); a1.axis("off")

    # (b) FAZ proximity zones
    a2.imshow(img)
    a2.add_patch(patches.Circle((ox, oy), od_r, lw=1.2, ec="#00ff00", fc="none"))
    a2.text(ox, oy - od_r - 10, "Optic disc", color="#00ff00", fontsize=6, fontweight="bold", ha="center",
            bbox=dict(boxstyle="square,pad=0.15", fc="black", alpha=0.55, ec="none"))
    a2.plot(fx, fy, "o", color="#ff3333", ms=3.5)
    a2.text(fx, fy - 14, "Fovea / FAZ", color="#ff6666", fontsize=6, fontweight="bold", ha="center",
            bbox=dict(boxstyle="square,pad=0.15", fc="black", alpha=0.55, ec="none"))
    a2.add_patch(patches.Circle((fx, fy), 2 * od_r, lw=1.2, ec="#ff2222", fc="none", ls=":"))
    a2.add_patch(patches.Circle((fx, fy), 4 * od_r, lw=1.0, ec="#ffa500", fc="none", ls="--"))
    a2.set_title("(b) Macular threat triage (distance to FAZ)", fontsize=7.5, fontweight="bold", pad=3); a2.axis("off")
    a2.legend(handles=[Line2D([0], [0], color="#ff2222", lw=1.2, ls=":", label=r"$\leq$ 1 DD: high CSME threat"),
                       Line2D([0], [0], color="#ffa500", lw=1.0, ls="--", label="1\u20132 DD: moderate")],
              loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=2, frameon=False, fontsize=6.5, columnspacing=1.2)
    if not found:
        fig.text(0.5, 0.5, "PLACEHOLDER IMAGE", ha="center", color="#b30000", fontsize=14, fontweight="bold", alpha=0.7)
    save(fig, "fig5_clinical_biomarkers_etdrs")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="sample_data")
    ap.add_argument("--out", default="figures")
    ARGS = ap.parse_args()
    fig1_architecture(); fig2_multimodal_panels(); fig3_ablation(); fig4_confusion(); fig5_biomarkers()
    print(f"\nAll figures written to {ARGS.out}/")
