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

# ============================================================
# Data from the LaTeX table (TTFT %, Throughput %)
# ============================================================

strategies = ["BW", "BW+ET", "BW+ET+DB"]

B_star = {
    ("LLAMA3", "ARXIV"): [(111,126), (38,140), (45,163)],
    ("LLAMA3", "LW")   : [(93,195),  (77,196), (75,220)],
    ("LLAMA3", "CHAT") : [(90,105),  (54,110), (45,113)],
    ("LLAMA3", "BWB")  : [(67,229),  (93,236), (61,243)],

    ("MAMBA2", "ARXIV"): [(100,103), (69,145), (73,160)],
    ("MAMBA2", "LW")   : [(124,164), (55,141), (49,139)],
    ("MAMBA2", "CHAT") : [(98,104),  (48,159), (54,167)],
    ("MAMBA2", "BWB")  : [(104,109), (89,114), (69,120)],

    ("Nemotron-H", "ARXIV"): [(81,104), (62,129), (61,135)],
    ("Nemotron-H", "LW")   : [(95,106), (30,117), (30,132)],
    ("Nemotron-H", "CHAT") : [(101,110),(50,132), (49,136)],
    ("Nemotron-H", "BWB")  : [(81,100), (71,115), (73,132)],
}

M_T = {
    ("LLAMA3", "ARXIV"): [(80,118),  (46,184), (40,201)],
    ("LLAMA3", "LW")   : [(110,122), (107,142),(98,167)],
    ("LLAMA3", "CHAT") : [(107,115), (43,186), (45,192)],
    ("LLAMA3", "BWB")  : [(124,201), (58,183), (60,188)],

    ("MAMBA2", "ARXIV"): [(99,101),  (71,151), (70,160)],
    ("MAMBA2", "LW")   : [(99,105),  (53,148), (63,162)],
    ("MAMBA2", "CHAT") : [(104,107), (37,143), (40,150)],
    ("MAMBA2", "BWB")  : [(100,102), (64,146), (63,151)],

    ("Nemotron-H", "ARXIV"): [(80,105), (57,138), (50,147)],
    ("Nemotron-H", "LW")   : [(58,230), (30,308), (35,327)],
    ("Nemotron-H", "CHAT") : [(111,120),(75,208), (64,211)],
    ("Nemotron-H", "BWB")  : [(100,150),(74,143), (60,161)],
}

# Ordering
model_order = ["LLAMA3", "MAMBA2", "Nemotron-H"]
dataset_order = ["ARXIV", "LW", "CHAT", "BWB"]

keys = [(m, d) for m in model_order for d in dataset_order]
x_labels = dataset_order * len(model_order)  # dataset-only labels

# ============================================================
# Helpers
# ============================================================

def extract_metric(table, metric="tp"):
    out = []
    for k in keys:
        vals = table[k]
        if metric == "tp":
            out.append([v[1] for v in vals])
        else:
            out.append([v[0] for v in vals])
    return np.array(out, dtype=float)

ANNOT_FONTSIZE = 13          # smaller annotation text
GROUP_GAP = 1.1             # enlarge spacing between bar groups
ARROW_LS = "--"
ARROW_LW = 1.0

def grouped_bar(ax, Y, title, ylabel, metric_kind):
    """
    Y: shape (G, S) where S=3 strategies
    metric_kind: "tp" or "ttft"  (only affects a small vertical placement heuristic)
    """
    G, S = Y.shape
    
    special_case = metric_kind == "tp" and "TP" in title and "M_T" in title
    if special_case:
        index = 10  # special case for adjusting annotation of one specific plot
    else:
        index = -1
    # for visualization annotations free of overlapping, there is special cases needed to adjust the annoation
    
    # --- enlarge inter-group spacing ---
    x = np.arange(G) * GROUP_GAP

    bar_w = 0.22
    offsets = (np.arange(S) - (S-1)/2) * bar_w

    # Bars
    for i in range(S):
        ax.bar(x + offsets[i], Y[:, i], width=bar_w)

    # Baseline
    ax.axhline(100, linestyle="--", linewidth=1)

    # Titles/labels
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=0)
    ax.margins(x=0.01)

    # --- Slanted arrow + centered annotation for BW+ET+DB ---
    bw_i = 0       # blue
    final_i = 2    # green

    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min
    y_off = 0.03 * y_range  # vertical offset for text

    for g in range(G):
        # start from the top of BW bar
        x0 = x[g] + offsets[bw_i]
        y0 = float(Y[g, bw_i])

        # end at the top of BW+ET+DB bar
        x1 = x[g] + offsets[final_i]
        y1 = float(Y[g, final_i])

        # dashed slanted arrow from blue->green
        ax.annotate(
            "", xy=(x1, y1), xytext=(x0, y0),
            arrowprops=dict(arrowstyle="->", linestyle=ARROW_LS, linewidth=ARROW_LW)
        )

        # label = factor relative to baseline (100%)
        factor = y1 / 100.0
        factor = y1 / y0
        txt = f"{factor:.2f}x"

        # place label at center of group (x[g]), and above the group to avoid overlap
        group_top = max(Y[g, :])
        group_bot = min(Y[g, :])

        if metric_kind == "tp":
            # for throughput, place above the tallest bar in the group
            y_text = group_top + y_off
            va = "bottom"
        else:
            # for TTFT, still safest to place above the tallest bar (since tallest may be >100)
            # if you prefer below the group, swap to: y_text = group_bot - y_off; va="top"
            y_text = group_top + y_off
            va = "bottom"

        if g==index-1:
            ax.text(
                x[g]*0.97, y_text*0.9, txt,
                ha="center", va=va, fontsize=ANNOT_FONTSIZE
            )
        else:
            ax.text(
                x[g], y_text, txt,
                ha="center", va=va, fontsize=ANNOT_FONTSIZE
            )

    return x  # return x so caller can place separators correctly
# ============================================================
# Extract metrics
# ============================================================

B_tp   = extract_metric(B_star, "tp")
B_ttft = extract_metric(B_star, "ttft")
M_tp   = extract_metric(M_T, "tp")
M_ttft = extract_metric(M_T, "ttft")

# ============================================================
# Plot: 2 rows × 2 panels
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(20, 8), sharex=True)

# Row 1: Throughput
xpos_00 = grouped_bar(
    axes[0, 0], B_tp,
    title=r"Throughput (TP $\uparrow$) — $B^\star$",
    ylabel="Normalized Throughput (%) ↑",
    metric_kind="tp"
)
xpos_01 = grouped_bar(
    axes[0, 1], M_tp,
    title=r"Throughput (TP $\uparrow$) — $M_T$",
    ylabel="Normalized Throughput (%) ↑",
    metric_kind="tp"
)

# Row 2: TTFT
xpos_10 = grouped_bar(
    axes[1, 0], B_ttft,
    title=r"Time To First Token (TTFT $\downarrow$) — $B^\star$",
    ylabel="Normalized TTFT (%) ↓",
    metric_kind="ttft"
)
xpos_11 = grouped_bar(
    axes[1, 1], M_ttft,
    title=r"Time To First Token (TTFT $\downarrow$) — $M_T$",
    ylabel="Normalized TTFT (%) ↓",
    metric_kind="ttft"
)

# Consistent y-limits per row
tp_max = max(B_tp.max(), M_tp.max())
ttft_max = max(B_ttft.max(), M_ttft.max())

for ax in axes[0]:
    ax.set_ylim(0, tp_max * 1.15)
for ax in axes[1]:
    ax.set_ylim(0, ttft_max * 1.15)

# Model separators (every 4 datasets)
cuts = [4, 8]
for ax, xpos in zip(axes.flatten(), [xpos_00, xpos_01, xpos_10, xpos_11]):
    for cut in cuts:
        x_sep = (xpos[cut - 1] + xpos[cut]) / 2.0
        ax.axvline(x_sep, linestyle=":", linewidth=1)

plt.tight_layout()
plt.savefig("figs/scheds/bar_graph_Bstar_MT.png", dpi=300)
