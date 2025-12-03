# -*- coding: utf-8 -*-
"""
ExpNetTR schematic (clean layout) — Trend (exp mixture) + Gaussian Residual
Outputs: expnettr_clean.png / expnettr_clean.svg
Deps: matplotlib
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

BLUE = "#71b8ed"

# ---------------- helpers ----------------
def box(ax, x, y, w, h, text, fc="white", fs=11, lw=2):
    r = FancyBboxPatch((x, y), w, h,
                       boxstyle="round,pad=0.02,rounding_size=0.28",
                       linewidth=lw, edgecolor=BLUE, facecolor=fc)
    ax.add_patch(r)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center",
            fontsize=fs, color="black")
    return r

def arrow(ax, x1, y1, x2, y2, rad=0.0, lw=1.9):
    arr = FancyArrowPatch((x1, y1), (x2, y2),
                          arrowstyle="simple", color=BLUE,
                          linewidth=lw, mutation_scale=12,
                          connectionstyle=f"arc3,rad={rad}")
    ax.add_patch(arr)

# ---------------- main draw ----------------
def draw(filename_png="expnettr_clean.png",
         filename_svg="expnettr_clean.svg",
         figsize=(18, 8), dpi=300,
         N_TREND=3, N_RES=3):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 44)
    ax.set_ylim(0, 22)
    ax.axis("off")

    # Title
    ax.text(1.0, 1.0,
            "ExpNetTR — Trend (Exponential Mixture) + Gaussian Residual",
            fontsize=13, color="black")

    # -------- Left: Input & normalization --------
    q  = box(ax, 1.2, 17.6, 3.8, 2.2, "Q")
    norm = box(ax, 5.6, 17.6, 9.6, 2.2,
               "Normalize:\nĉ = (Q−c_min)/(c_max−c_min) ∈ [0,1]",
               fc="#f7fbff", fs=10)
    arrow(ax, 4.8, 18.7, 5.6, 18.7)

    # Vertical guide line position for branches input (from ĉ)
    CX = 15.6  # center x for branching arrows

    # -------- Panel A: Trend --------
    panelA = box(ax, 12.0, 6.0, 13.6, 12.6, "", fc="white")
    titleA = box(ax, 12.0, 18.8, 13.6, 2.2, "Trend  Σ αᵢ · exp(−τᵢ · ĉ)", fc="white", fs=11)
    noteA  = box(ax, 20.0, 18.8, 5.6, 2.2,
                 "α = softmax(a)\nτ = softplus(t) > 0", fs=9)

    # evenly spaced branch y-coordinates in panelA
    ysA = list(reversed([6.8 + i*(10.2/(max(1, N_TREND)-1)) for i in range(N_TREND)]))
    for y in ysA:
        f = box(ax, 12.8, y, 5.6, 2.0, "exp(−τᵢ · ĉ)", fc="#f7fbff", fs=10)
        w = box(ax, 18.8, y, 6.0, 2.0, "αᵢ · exp(−τᵢ · ĉ)", fs=10)
        arrow(ax, 15.4, 18.7, 15.4, y+1.0)     # ĉ down to left block
        arrow(ax, 18.4, y+1.0, 18.8, y+1.0)    # left→right within branch
    sumA = box(ax, 12.8, 6.0, 12.0, 1.8, "Weighted Sum  (Σ over i)", fc="#f7fbff", fs=10)

    # -------- Panel B: Residual --------
    panelB = box(ax, 27.0, 6.0, 13.6, 12.6, "", fc="white")
    titleB = box(ax, 27.0, 18.8, 13.6, 2.2, "Residual  Σ βⱼ · 𝒩(ĉ; μⱼ, σⱼ)", fc="white", fs=11)
    noteB  = box(ax, 35.0, 18.8, 5.6, 2.2,
                 "μ ∈ [0,1]\nσ = softplus(s) > 0 ; β = tanh(b)", fs=9)

    ysB = list(reversed([6.8 + i*(10.2/(max(1, N_RES)-1)) for i in range(N_RES)]))
    for y in ysB:
        g = box(ax, 27.8, y, 5.6, 2.0, "𝒩(ĉ; μⱼ, σⱼ)", fc="#f7fbff", fs=10)
        b = box(ax, 33.8, y, 6.0, 2.0, "βⱼ · 𝒩(ĉ; μⱼ, σⱼ)", fs=10)
        arrow(ax, 15.4, 18.7, 29.2, y+1.0)     # ĉ to residual left block (gentle arc)
        arrow(ax, 33.4, y+1.0, 33.8, y+1.0)    # left→right within branch
    sumB = box(ax, 27.8, 6.0, 12.0, 1.8, "Weighted Sum  (Σ over j)", fc="#f7fbff", fs=10)

    # -------- Merge Σ → linear → SOH --------
    sigma = Circle((41.6, 12.6), 0.5, edgecolor=BLUE, facecolor="white", lw=2)
    ax.add_patch(sigma); ax.text(41.6, 12.6, "Σ", ha="center", va="center", fontsize=12)

    # From panel bottoms to Σ
    arrow(ax, 24.8, 6.9, 41.2, 12.4, rad=0.07)  # Trend sum to Σ
    arrow(ax, 39.8, 6.9, 41.4, 12.4, rad=-0.07) # Residual sum to Σ

    lin = box(ax, 42.6, 11.6, 6.2, 2.0, "g(x) = a·x + b\n(+ optional clip)", fs=10)
    arrow(ax, 42.0, 12.6, 42.6, 12.6)

    out = box(ax, 49.2, 11.6, 3.0, 2.0, "SOH", fs=12)
    arrow(ax, 48.8, 12.6, 49.2, 12.6)

    # Save
    fig.savefig(filename_png, bbox_inches="tight", dpi=dpi)
    fig.savefig(filename_svg, bbox_inches="tight", dpi=dpi)
    print(f"Saved: {filename_png}\nSaved: {filename_svg}")

if __name__ == "__main__":
    draw()
