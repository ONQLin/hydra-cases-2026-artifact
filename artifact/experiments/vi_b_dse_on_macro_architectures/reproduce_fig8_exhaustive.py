#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]

MODELS = [
    ("LLAMA3", "LLAMA3-7B"),
    ("MAMBA2", "MAMBA2-2.8B"),
    ("NEMO", "Nemotron-H-4B"),
]
DATASETS = [
    ("ARXIV", "Arxiv"),
    ("LW", "Long Writer"),
    ("CHAT", "Chat"),
    ("BWB", "BWB"),
]

SUMMARY_GLOB = "results_summary_{model}-{dataset}_bw_static_static.csv"
PARETO_GLOB = "pareto_frontier_results_summary_{model}-{dataset}_bw_static_static.csv"
NON_PARETO_GLOB = "non_pareto_results_summary_{model}-{dataset}_bw_static_static.csv"

# These limits match the manuscript Fig. 8 crop windows. They intentionally
# preserve the original Markov-inclusive view even though this artifact plot
AXIS_LIMITS = {
    ("LLAMA3", "ARXIV"): ((8.18, 31.9), (125.0, 306.0)),
    ("LLAMA3", "LW"): ((0.0, 22.3), (0.0, 3420.0)),
    ("LLAMA3", "CHAT"): ((0.0, 5.8), (0.0, 1980.0)),
    ("LLAMA3", "BWB"): ((9.83, 75.4), (0.0, 608.0)),
    ("MAMBA2", "ARXIV"): ((0.0, 45.0), (105.0, 1020.0)),
    ("MAMBA2", "LW"): ((0.0, 16.0), (0.0, 10300.0)),
    ("MAMBA2", "CHAT"): ((0.0, 13.1), (0.0, 6250.0)),
    ("MAMBA2", "BWB"): ((0.197, 54.2), (0.0, 3090.0)),
    ("NEMO", "ARXIV"): ((0.0, 73.8), (138.0, 623.0)),
    ("NEMO", "LW"): ((0.0, 25.4), (0.0, 6460.0)),
    ("NEMO", "CHAT"): ((0.0, 21.4), (0.0, 3480.0)),
    ("NEMO", "BWB"): ((3.3, 69.9), (0.0, 1320.0)),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate Fig. 8 exhaustive-search panels from prepared VI-B DSE CSV outputs."
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=REPO_ROOT / "artifact" / "run_outputs" / "vi_b_dse_on_macro_architectures" / "raw_data",
        help="Prepared raw-data root containing Summary_Reports and Pareto_Reports.",
    )
    parser.add_argument(
        "--fig-root",
        type=Path,
        default=REPO_ROOT / "artifact" / "figures" / "vi_b_dse_on_macro_architectures",
        help="Output directory for regenerated figures.",
    )
    return parser.parse_args()


def require_input_files(raw_root: Path) -> None:
    missing: list[Path] = []
    for model, _ in MODELS:
        for dataset, _ in DATASETS:
            for subdir, pattern in (
                ("Summary_Reports", SUMMARY_GLOB),
                ("Pareto_Reports", PARETO_GLOB),
                ("Pareto_Reports", NON_PARETO_GLOB),
            ):
                path = raw_root / subdir / pattern.format(model=model, dataset=dataset)
                if not path.is_file():
                    missing.append(path)
    if missing:
        formatted = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"Required VI-B CSV inputs are missing:\n{formatted}")


def sanitize_df(df: pd.DataFrame) -> pd.DataFrame:
    working = df.drop_duplicates().copy()
    working = working.dropna(subset=["TTFT(s)", "tokens_per_sec"])
    working = working[np.isfinite(working["TTFT(s)"]) & np.isfinite(working["tokens_per_sec"])].copy()
    return working.reset_index(drop=True)


def pareto_curve_xy(pareto_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    curve = pareto_df.groupby("TTFT(s)", as_index=False)["tokens_per_sec"].max().sort_values("TTFT(s)")
    return curve["TTFT(s)"].to_numpy(), curve["tokens_per_sec"].to_numpy()


def choose_focus_df(df: pd.DataFrame, pareto_df: pd.DataFrame) -> pd.DataFrame:
    if pareto_df.empty:
        return df.copy()
    x_min = float(pareto_df["TTFT(s)"].min())
    x_max = float(pareto_df["TTFT(s)"].max())
    x_span = max(x_max - x_min, max(x_max, 1.0) * 0.08)
    x_lower = max(0.0, x_min - 0.12 * x_span)
    x_upper = x_max + 0.12 * x_span
    focus_df = df[(df["TTFT(s)"] >= x_lower) & (df["TTFT(s)"] <= x_upper)].copy()
    return focus_df if not focus_df.empty else df.copy()


def choose_mean_configuration(focus_df: pd.DataFrame, pareto_df: pd.DataFrame) -> pd.Series:
    candidates = focus_df[~focus_df.get("is_pareto", False)].copy()
    if candidates.empty:
        candidates = focus_df.copy()

    target_x = float(focus_df["TTFT(s)"].median())
    target_y = float(candidates["tokens_per_sec"].median())
    if not pareto_df.empty:
        curve_x, curve_y = pareto_curve_xy(pareto_df)
        curve_at_target_x = float(np.interp(target_x, curve_x, curve_y))
        target_y = min(target_y, curve_at_target_x * 0.82)

        curve_at_candidate_x = np.interp(candidates["TTFT(s)"].to_numpy(), curve_x, curve_y)
        below_curve = candidates[candidates["tokens_per_sec"].to_numpy() <= curve_at_candidate_x * 0.97].copy()
        if not below_curve.empty:
            candidates = below_curve

    x_span = max(float(focus_df["TTFT(s)"].max() - focus_df["TTFT(s)"].min()), 1e-9)
    y_span = max(float(focus_df["tokens_per_sec"].max() - focus_df["tokens_per_sec"].min()), 1e-9)
    score = ((candidates["TTFT(s)"] - target_x).abs() / x_span) + (
        (candidates["tokens_per_sec"] - target_y).abs() / y_span
    )
    return candidates.loc[score.idxmin()]


def compute_special_points(df: pd.DataFrame, pareto_df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    focus_df = choose_focus_df(df, pareto_df)
    avg_tp = float(focus_df["tokens_per_sec"].mean())
    mean_row = choose_mean_configuration(focus_df, pareto_df)
    mt_row = focus_df.loc[focus_df["tokens_per_sec"].idxmax()]
    bstar_candidates = focus_df[focus_df["tokens_per_sec"] > avg_tp].copy()
    if bstar_candidates.empty:
        bstar_candidates = focus_df.copy()
    ratio = bstar_candidates["tokens_per_sec"] / bstar_candidates["TTFT(s)"].replace(0, np.nan)
    ratio = ratio.replace([np.inf, -np.inf], np.nan).dropna()
    bstar_row = mt_row if ratio.empty else bstar_candidates.loc[ratio.idxmax()]
    return mean_row, mt_row, bstar_row


def load_panel_data(raw_root: Path, model: str, dataset: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_path = raw_root / "Summary_Reports" / SUMMARY_GLOB.format(model=model, dataset=dataset)
    pareto_path = raw_root / "Pareto_Reports" / PARETO_GLOB.format(model=model, dataset=dataset)
    non_pareto_path = raw_root / "Pareto_Reports" / NON_PARETO_GLOB.format(model=model, dataset=dataset)
    summary = sanitize_df(pd.read_csv(summary_path))
    pareto = sanitize_df(pd.read_csv(pareto_path)).sort_values("TTFT(s)")
    non_pareto = sanitize_df(pd.read_csv(non_pareto_path))
    if "is_pareto" not in summary.columns:
        pareto_keys = set(zip(pareto["TTFT(s)"], pareto["tokens_per_sec"]))
        summary["is_pareto"] = [(x, y) in pareto_keys for x, y in zip(summary["TTFT(s)"], summary["tokens_per_sec"])]
    return summary, pareto, non_pareto


def plot_panel(
    ax: plt.Axes,
    summary: pd.DataFrame,
    pareto: pd.DataFrame,
    non_pareto: pd.DataFrame,
    model: str,
    dataset: str,
    title: str,
    row_label: str | None,
    show_ylabel: bool,
) -> None:
    mean_row, mt_row, bstar_row = compute_special_points(summary, pareto)
    xlim, ylim = AXIS_LIMITS[(model, dataset)]

    ax.scatter(
        non_pareto["TTFT(s)"],
        non_pareto["tokens_per_sec"],
        marker="x",
        color="0.45",
        alpha=0.11,
        s=13,
        linewidths=0.8,
        zorder=1,
    )
    ax.plot(
        pareto["TTFT(s)"],
        pareto["tokens_per_sec"],
        color="red",
        linewidth=1.15,
        alpha=0.72,
        zorder=4,
    )
    ax.scatter(bstar_row["TTFT(s)"], bstar_row["tokens_per_sec"], color="red", marker="*", s=72, zorder=8)
    ax.scatter(mt_row["TTFT(s)"], mt_row["tokens_per_sec"], color="red", marker="^", s=62, zorder=8)
    ax.scatter(mean_row["TTFT(s)"], mean_row["tokens_per_sec"], color="green", marker="s", s=46, zorder=7)

    ax.annotate(
        r"$B^\star$",
        xy=(bstar_row["TTFT(s)"], bstar_row["tokens_per_sec"]),
        xytext=(-12, 8),
        textcoords="offset points",
        fontsize=8,
        fontstyle="italic",
        zorder=9,
    )
    ax.annotate(
        r"$M_T$",
        xy=(mt_row["TTFT(s)"], mt_row["tokens_per_sec"]),
        xytext=(-18, 8),
        textcoords="offset points",
        fontsize=8,
        fontstyle="italic",
        zorder=9,
    )

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.set_title(title, fontsize=10, fontstyle="italic", pad=2)
    ax.set_xlabel("Time To First Token (s)", fontsize=7, fontweight="bold", labelpad=1)
    ax.set_ylabel("Throughput (tokens/second)" if show_ylabel else "", fontsize=7, fontweight="bold", labelpad=1)
    ax.tick_params(axis="both", labelsize=7, length=2, pad=1)
    if row_label:
        ax.text(-0.25, 0.5, row_label, transform=ax.transAxes, rotation=90, va="center", ha="center", fontsize=10)


def make_plots(raw_root: Path, fig_root: Path) -> int:
    fig_root.mkdir(parents=True, exist_ok=True)
    panel_root = fig_root / "panels"
    panel_root.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.linewidth": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(3, 4, figsize=(10.6, 7.4), constrained_layout=False)
    panel_count = 0

    for r, (model, model_label) in enumerate(MODELS):
        for c, (dataset, dataset_label) in enumerate(DATASETS):
            summary, pareto, non_pareto = load_panel_data(raw_root, model, dataset)
            plot_panel(
                axes[r, c],
                summary,
                pareto,
                non_pareto,
                model,
                dataset,
                dataset_label,
                model_label if c == 0 else None,
                show_ylabel=(c == 0),
            )
            panel_count += 1

            panel_fig, panel_ax = plt.subplots(figsize=(4.2, 3.1))
            plot_panel(panel_ax, summary, pareto, non_pareto, model, dataset, dataset_label, None, show_ylabel=True)
            panel_fig.tight_layout()
            panel_name = f"fig8_{model.lower()}_{dataset.lower()}_exhaustive"
            panel_fig.savefig(panel_root / f"{panel_name}.png", dpi=300)
            plt.close(panel_fig)

    handles = [
        mlines.Line2D([], [], color="red", linewidth=1.4, label="Pareto Points"),
        mlines.Line2D([], [], color="0.45", marker="x", linestyle="None", markersize=4, label="Non-Pareto Points"),
        mlines.Line2D([], [], color="red", marker="*", linestyle="None", markersize=9, label=r"$B^\star$ (exhaustive)"),
        mlines.Line2D([], [], color="red", marker="^", linestyle="None", markersize=8, label=r"$M_T$ (exhaustive)"),
        mlines.Line2D([], [], color="green", marker="s", linestyle="None", markersize=6, label="Average Point"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=5,
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, 0.025),
        columnspacing=0.9,
        handletextpad=0.35,
    )
    fig.subplots_adjust(left=0.095, right=0.995, top=0.96, bottom=0.12, wspace=0.23, hspace=0.24)
    fig.savefig(fig_root / "fig8_exhaustive_12panels.png", dpi=300)
    plt.close(fig)
    return panel_count


def main() -> None:
    args = parse_args()
    require_input_files(args.raw_root)
    panel_count = make_plots(args.raw_root, args.fig_root)
    print(f"Wrote Fig. 8 exhaustive-search reproduction to {args.fig_root}")
    print(f"Generated {panel_count} panels from prepared VI-B CSV data.")


if __name__ == "__main__":
    main()
