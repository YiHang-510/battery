import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

BLUE = "#71b8ed"

def box(ax, x, y, w, h, label, fc="white", fs=10):
    r = FancyBboxPatch((x,y), w,h, boxstyle="round,pad=0.02,rounding_size=0.25",
                       linewidth=2, edgecolor=BLUE, facecolor=fc)
    ax.add_patch(r)
    ax.text(x+w/2, y+h/2, label, ha="center", va="center", fontsize=fs, color="black")
    return r

def arrow(ax, x1,y1,x2,y2,rad=0.0):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='simple',color=BLUE,
                                 linewidth=1.8,mutation_scale=12,
                                 connectionstyle=f"arc3,rad={rad}"))

def render_minimal(path):
    fig, ax = plt.subplots(figsize=(14,6))
    ax.set_xlim(0,30); ax.set_ylim(0,16); ax.axis('off')
    ax.text(0.6, 0.6, "PI-Res Head (Minimal) — I-Spline Monotonic Trend + Residual", fontsize=12)

    box(ax,1.0,11.2,5.0,2.0,"t_norm ∈ [0,1]")
    box(ax,1.0, 7.6,5.0,2.0,"Feature h ∈ ℝ^d")
    box(ax,7.0,11.2,5.6,2.0,"I-Spline Basis\nB_I(t)", fc="#f7fbff"); arrow(ax,6.0,12.2,7.0,12.2)

    box(ax,13.6,12.6,4.6,1.6,"softplus → c0≥0", fs=9)
    box(ax,19.0,12.6,6.2,1.6,"Trend m = B_I·c0 + b0", fc="#f7fbff", fs=10)
    arrow(ax,12.4,12.2,13.6,13.4,rad=0.2); arrow(ax,18.4,13.4,19.0,13.4)

    box(ax,7.0, 7.6,5.6,2.0,"res_head(h) →\nsoftplus → c_h≥0", fc="#f7fbff")
    box(ax,13.6,7.6,4.6,2.0,"⊙  (elemwise)", fs=12)
    box(ax,19.0,7.6,6.2,2.0,"Residual R=(B_I⊙c_h)·1", fc="#f7fbff", fs=10)
    arrow(ax,6.0,8.6,7.0,8.6); arrow(ax,12.4,12.2,13.6,8.6,rad=-0.25)
    arrow(ax,12.6,8.6,13.6,8.6); arrow(ax,18.2,8.6,19.0,8.6)

    s = Circle((26.2,10.8),0.45,edgecolor=BLUE,facecolor="white",lw=2)
    ax.add_patch(s); ax.text(26.2,10.8,"Σ",ha="center",va="center",fontsize=12)
    arrow(ax,25.2,13.4,26.0,11.2); arrow(ax,25.2,8.6,26.0,10.4)
    box(ax,27.2,10.0,2.2,1.6,"Q",fs=11); arrow(ax,26.6,10.8,27.2,10.8)

    fig.savefig(path, bbox_inches="tight", dpi=300)

def render_detailed(path):
    fig, ax = plt.subplots(figsize=(16,7))
    ax.set_xlim(0,34); ax.set_ylim(0,18); ax.axis('off')
    ax.text(0.6, 0.8, "PI-Res Head (Detailed) — I-Spline Monotonic Trend + Controlled Residual", fontsize=12)

    box(ax,1.0,13.6,5.2,2.0,"t_norm ∈ [0,1]\n(min–max cycle index)")
    box(ax,1.0,10.2,5.2,2.0,"Feature h ∈ ℝ^d\n(from TimeMixer+MLP)")
    box(ax,7.0,13.6,6.2,2.0,"I-Spline Basis\nB_I(t) ∈ ℝ^M", fc="#f7fbff"); arrow(ax,6.0,14.6,7.0,14.6)

    box(ax,14.0,14.8,5.0,1.8,"softplus → c0≥0\n(knots denser in tail)", fs=9)
    box(ax,20.0,14.8,6.6,1.8,"Trend m = B_I·c0 + b0\nmonotonic by construction", fc="#f7fbff", fs=9)
    arrow(ax,13.0,14.6,14.0,15.7,rad=0.2); arrow(ax,19.2,15.7,20.0,15.7)

    box(ax,7.0,10.2,6.2,2.0,"res_head(h) → softplus → c_h≥0\n(λ·||c_h||² regularization)", fc="#f7fbff", fs=9)
    box(ax,14.0,10.2,5.0,2.0,"⊙  (element-wise)", fs=12)
    box(ax,20.0,10.2,6.6,2.0,"Residual R = (B_I ⊙ c_h) · 1\nsmooth, low-frequency", fc="#f7fbff", fs=9)
    arrow(ax,6.0,11.2,7.0,11.2); arrow(ax,13.0,14.6,14.0,11.2,rad=-0.25)
    arrow(ax,13.2,11.2,14.0,11.2); arrow(ax,19.2,11.2,20.0,11.2)

    s = Circle((27.6,12.8),0.45,edgecolor=BLUE,facecolor="white",lw=2)
    ax.add_patch(s); ax.text(27.6,12.8,"Σ",ha="center",va="center",fontsize=12)
    arrow(ax,26.6,15.7,27.4,13.2); arrow(ax,26.6,11.2,27.4,12.4)
    box(ax,29.0,12.0,2.4,1.6,"Q",fs=11); arrow(ax,28.2,12.8,29.0,12.8)

    box(ax,7.0,6.2,19.6,2.4,
        r"$Q(t;h)=B_I(t)\mathbf{c}_0 + b_0 + (B_I(t)\odot\mathbf{c}_h(h))\mathbf{1}$"
        "\nMonotonic direction: use t or (1−t).  Precompute $B_I$; clamp t to [0,1].", fs=9)

    fig.savefig(path, bbox_inches="tight", dpi=300)

render_minimal("pires_minimal_blue71b8ed.png")
render_detailed("pires_detailed_blue71b8ed.png")
print("Done")
