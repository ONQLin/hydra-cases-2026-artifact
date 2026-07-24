import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# Global style control (edit here later)
# ============================================================
plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 18,
    "axes.labelsize": 18,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
})

ANNOT_FONTSIZE = 12
GROUP_GAP = 1.4
BAR_W = 0.28

# ============================================================
# Raw data (percent utilization)
# ============================================================
datasets = ["ARXIV", "LW", "CHAT", "BWB"]

Bstar_BW    = np.array([49, 51, 43, 38], dtype=float)
Bstar_BW_ET = np.array([66, 68, 55, 51], dtype=float)

MT_BW       = np.array([56, 59, 51, 45], dtype=float)
MT_BW_ET    = np.array([78, 81, 61, 60], dtype=float)

# ============================================================
# Plot helper
# ============================================================
def util_panel(ax, bw, bw_et, title, legend=False):
    x = np.arange(len(datasets)) * GROUP_GAP
    off = np.array([-BAR_W/2, BAR_W/2])

    # Color map: red and blue
    colors = ["#d62728", "#1f77b4"]  # red, blue

    # Bars
    ax.bar(x + off[0], bw,    width=BAR_W, color=colors[0], label="Static Scheduling")
    ax.bar(x + off[1], bw_et, width=BAR_W, color=colors[1], label="Elastic Scheduling")

    # Labels / styling
    ax.set_title(title)
    ax.set_ylabel("Utilization (%) ↑")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylim(0, max(bw.max(), bw_et.max()) * 1.25)
    if legend:
        ax.legend(fontsize=12)

    # Arrow + improvement factor above each dataset group
    y_min, y_max = ax.get_ylim()
    y_off = 0.04 * (y_max - y_min)

    for i in range(len(datasets)):
        x0, y0 = x[i] + off[0], float(bw[i])
        x1, y1 = x[i] + off[1], float(bw_et[i])

        # Slanted dashed arrow from BW -> BW+ET
        ax.annotate(
            "", xy=(x1, y1), xytext=(x0, y0),
            arrowprops=dict(arrowstyle="->", linestyle="--", linewidth=1.0)
        )

        factor = y1 / y0 if y0 > 0 else np.nan
        txt = f"{factor:.2f}x"

        # Put text centered above the two bars
        y_text = max(y0, y1) + y_off
        ax.text(x[i], y_text, txt, ha="center", va="bottom", fontsize=ANNOT_FONTSIZE)

    return x

# ============================================================
# Figure: 1 row × 2 panels
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

x_left = util_panel(axes[0], Bstar_BW, Bstar_BW_ET, r"Compute Chiplets Utilization — $B^\star$")
x_right = util_panel(axes[1], MT_BW, MT_BW_ET, r"Compute Chiplets Utilization — $M_T$", legend=True)

# Optional: subtle gridlines
for ax in axes:
    ax.grid(axis="y", linestyle=":", linewidth=0.8)

plt.tight_layout()
plt.savefig("figs/scheds/bar_util_compute_chiplets.png", dpi=300)
