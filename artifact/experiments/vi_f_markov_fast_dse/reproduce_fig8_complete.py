#!/usr/bin/env python3
"""Measure filtered static candidates and reproduce the complete Fig. 8."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
VI_B_EXPERIMENT = REPO_ROOT / "artifact/experiments/vi_b_dse_on_macro_architectures"
if str(VI_B_EXPERIMENT) not in sys.path:
    sys.path.insert(0, str(VI_B_EXPERIMENT))

import reproduce_fig8_exhaustive as exhaustive


DEFAULT_VI_B_RAW_ROOT = (
    REPO_ROOT / "artifact/run_outputs/vi_b_dse_on_macro_architectures/raw_data"
)
DEFAULT_FAST_DSE_ROOT = (
    REPO_ROOT
    / "artifact/run_outputs/vi_f_markov_fast_dse/fig8_static_static"
)
DEFAULT_FIG_ROOT = (
    REPO_ROOT / "artifact/figures/vi_f_markov_fast_dse/static_static"
)
CONFIG_FIELDS = (
    "batchsize",
    "Num M",
    "Num Mp",
    "Num Md",
    "Num Ap",
    "Num Ad",
    "NoI_bw(GBps)",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vi-b-raw-root", type=Path, default=DEFAULT_VI_B_RAW_ROOT)
    parser.add_argument("--fast-dse-root", type=Path, default=DEFAULT_FAST_DSE_ROOT)
    parser.add_argument("--fig-root", type=Path, default=DEFAULT_FIG_ROOT)
    return parser.parse_args()


def load_manifest(fast_dse_root: Path) -> pd.DataFrame:
    path = fast_dse_root / "fig8_filter_candidates.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Run select_fig8_candidates.py first."
        )
    manifest = pd.read_csv(path)
    expected_cases = {
        (model, dataset)
        for model, _ in exhaustive.MODELS
        for dataset, _ in exhaustive.DATASETS
    }
    actual_cases = set(zip(manifest["model"], manifest["dataset"]))
    if actual_cases != expected_cases:
        missing = sorted(expected_cases - actual_cases)
        extra = sorted(actual_cases - expected_cases)
        raise ValueError(
            f"Incomplete Fig. 8 candidate manifest; missing={missing}, extra={extra}"
        )
    counts = manifest.groupby(["model", "dataset"]).size()
    if counts.nunique() != 1:
        raise ValueError(
            "Every model/dataset pair must use the same candidate budget; "
            f"got {counts.to_dict()}."
        )
    return manifest


def collect_measurements(
    manifest: pd.DataFrame,
    vi_b_raw_root: Path,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for model, _ in exhaustive.MODELS:
        for dataset, _ in exhaustive.DATASETS:
            candidates = manifest[
                (manifest["model"] == model)
                & (manifest["dataset"] == dataset)
            ].copy()
            summary, _, _ = exhaustive.load_panel_data(
                vi_b_raw_root,
                model,
                dataset,
            )
            measured = candidates.merge(
                summary[[*CONFIG_FIELDS, "tokens_per_sec", "TTFT(s)"]],
                on=list(CONFIG_FIELDS),
                how="left",
                validate="one_to_one",
            )
            missing = measured[
                measured[["tokens_per_sec", "TTFT(s)"]].isna().any(axis=1)
            ]
            if not missing.empty:
                configs = missing[list(CONFIG_FIELDS)].to_dict("records")
                raise RuntimeError(
                    f"{model}-{dataset}: filtered configurations are absent "
                    f"from the VI-B detailed results: {configs[:3]}"
                )

            measurement_path = str(
                (
                    vi_b_raw_root
                    / "Summary_Reports"
                    / exhaustive.SUMMARY_GLOB.format(
                        model=model,
                        dataset=dataset,
                    )
                ).relative_to(REPO_ROOT)
            )
            measured["measurement_source"] = "vi_b_detailed_simulation"
            measured["measurement_path"] = measurement_path
            frames.append(measured)
    return pd.concat(frames, ignore_index=True)


def select_operating_points(
    measurements: pd.DataFrame,
    vi_b_raw_root: Path,
) -> pd.DataFrame:
    """Choose the best measured objectives from the retained candidates."""

    rows: list[dict[str, object]] = []
    for model, _ in exhaustive.MODELS:
        for dataset, _ in exhaustive.DATASETS:
            candidates = measurements[
                (measurements["model"] == model)
                & (measurements["dataset"] == dataset)
            ].copy()
            summary, pareto, _ = exhaustive.load_panel_data(
                vi_b_raw_root,
                model,
                dataset,
            )
            focus = exhaustive.choose_focus_df(summary, pareto)
            throughput_floor = float(focus["tokens_per_sec"].mean())

            axis_xlim, _ = exhaustive.AXIS_LIMITS[(model, dataset)]
            ttft_lower = max(float(focus["TTFT(s)"].min()), axis_xlim[0])
            ttft_upper = min(float(focus["TTFT(s)"].max()), axis_xlim[1])
            scoped = candidates[
                candidates["TTFT(s)"].between(ttft_lower, ttft_upper)
            ].copy()
            if scoped.empty:
                scoped = candidates[
                    candidates["TTFT(s)"].between(*axis_xlim)
                ].copy()
            if scoped.empty:
                scoped = candidates

            mt_row = scoped.loc[scoped["tokens_per_sec"].idxmax()]
            bstar_pool = scoped[
                scoped["tokens_per_sec"] > throughput_floor
            ].copy()
            if bstar_pool.empty:
                bstar_pool = scoped
            bstar_ratio = (
                bstar_pool["tokens_per_sec"]
                / bstar_pool["TTFT(s)"]
            )
            bstar_row = bstar_pool.loc[bstar_ratio.idxmax()]

            selected_points = (
                (
                    "Bstar",
                    bstar_row,
                    "tokens_per_sec/TTFT(s)",
                    float(
                        bstar_row["tokens_per_sec"]
                        / bstar_row["TTFT(s)"]
                    ),
                ),
                (
                    "MT",
                    mt_row,
                    "tokens_per_sec",
                    float(mt_row["tokens_per_sec"]),
                ),
            )
            for objective, selected, metric, value in selected_points:
                result = selected.to_dict()
                result["objective"] = objective
                result["bstar_throughput_floor"] = throughput_floor
                result["selection_rule"] = (
                    "best_measured_objective_in_fig8_window"
                )
                result["selection_metric"] = metric
                result["selection_value"] = value
                result["selection_ttft_min_s"] = ttft_lower
                result["selection_ttft_max_s"] = ttft_upper
                rows.append(result)
    return pd.DataFrame(rows)


def plot_panel(
    ax: plt.Axes,
    summary: pd.DataFrame,
    pareto: pd.DataFrame,
    non_pareto: pd.DataFrame,
    markov: pd.DataFrame,
    model: str,
    dataset: str,
    title: str,
    row_label: str | None,
    show_ylabel: bool,
) -> None:
    mean_row, mt_row, bstar_row = exhaustive.compute_special_points(summary, pareto)
    markov_bstar = markov[markov["objective"] == "Bstar"].iloc[0]
    markov_mt = markov[markov["objective"] == "MT"].iloc[0]
    xlim, ylim = exhaustive.AXIS_LIMITS[(model, dataset)]

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
    ax.scatter(
        bstar_row["TTFT(s)"],
        bstar_row["tokens_per_sec"],
        color="red",
        marker="*",
        s=72,
        zorder=8,
    )
    ax.scatter(
        markov_bstar["TTFT(s)"],
        markov_bstar["tokens_per_sec"],
        color="blue",
        marker="*",
        s=72,
        zorder=9,
    )
    ax.scatter(
        mt_row["TTFT(s)"],
        mt_row["tokens_per_sec"],
        color="red",
        marker="^",
        s=62,
        zorder=8,
    )
    ax.scatter(
        markov_mt["TTFT(s)"],
        markov_mt["tokens_per_sec"],
        color="blue",
        marker="^",
        s=62,
        zorder=9,
    )
    ax.scatter(
        mean_row["TTFT(s)"],
        mean_row["tokens_per_sec"],
        color="green",
        marker="s",
        s=46,
        zorder=7,
    )

    ax.annotate(
        r"$B^\star$",
        xy=(bstar_row["TTFT(s)"], bstar_row["tokens_per_sec"]),
        xytext=(-12, 8),
        textcoords="offset points",
        fontsize=8,
        fontstyle="italic",
        zorder=10,
    )
    ax.annotate(
        r"$M_T$",
        xy=(mt_row["TTFT(s)"], mt_row["tokens_per_sec"]),
        xytext=(-18, 8),
        textcoords="offset points",
        fontsize=8,
        fontstyle="italic",
        zorder=10,
    )
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.set_title(title, fontsize=10, fontstyle="italic", pad=2)
    ax.set_xlabel(
        "Time To First Token (s)", fontsize=7, fontweight="bold", labelpad=1
    )
    ax.set_ylabel(
        "Throughput (tokens/second)" if show_ylabel else "",
        fontsize=7,
        fontweight="bold",
        labelpad=1,
    )
    ax.tick_params(axis="both", labelsize=7, length=2, pad=1)
    if row_label:
        ax.text(
            -0.25,
            0.5,
            row_label,
            transform=ax.transAxes,
            rotation=90,
            va="center",
            ha="center",
            fontsize=10,
        )


def make_plots(
    vi_b_raw_root: Path, measurements: pd.DataFrame, fig_root: Path
) -> int:
    fig_root.mkdir(parents=True, exist_ok=True)
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

    for row_index, (model, model_label) in enumerate(exhaustive.MODELS):
        for column_index, (dataset, dataset_label) in enumerate(exhaustive.DATASETS):
            summary, pareto, non_pareto = exhaustive.load_panel_data(
                vi_b_raw_root, model, dataset
            )
            markov = measurements[
                (measurements["model"] == model)
                & (measurements["dataset"] == dataset)
            ]
            if len(markov) != 2:
                raise RuntimeError(
                    f"Expected Bstar and MT measurements for {model}-{dataset}; got {len(markov)}."
                )
            plot_panel(
                axes[row_index, column_index],
                summary,
                pareto,
                non_pareto,
                markov,
                model,
                dataset,
                dataset_label,
                model_label if column_index == 0 else None,
                show_ylabel=(column_index == 0),
            )
            panel_count += 1

    handles = [
        mlines.Line2D([], [], color="red", linewidth=1.4, label="Pareto Points"),
        mlines.Line2D(
            [], [], color="0.45", marker="x", linestyle="None", markersize=4,
            label="Non-Pareto Points"
        ),
        mlines.Line2D(
            [], [], color="red", marker="*", linestyle="None", markersize=9,
            label=r"$B^\star$ (exhaustive)"
        ),
        mlines.Line2D(
            [], [], color="blue", marker="*", linestyle="None", markersize=9,
            label=r"$B^\star$ (Markov)"
        ),
        mlines.Line2D(
            [], [], color="red", marker="^", linestyle="None", markersize=8,
            label=r"$M_T$ (exhaustive)"
        ),
        mlines.Line2D(
            [], [], color="blue", marker="^", linestyle="None", markersize=8,
            label=r"$M_T$ (Markov)"
        ),
        mlines.Line2D(
            [], [], color="green", marker="s", linestyle="None", markersize=6,
            label="Average Point"
        ),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=7,
        frameon=False,
        fontsize=8,
        bbox_to_anchor=(0.5, 0.025),
        columnspacing=0.55,
        handletextpad=0.25,
    )
    fig.subplots_adjust(
        left=0.095, right=0.995, top=0.96, bottom=0.12, wspace=0.23, hspace=0.24
    )
    fig.savefig(fig_root / "fig8_complete_12panels.png", dpi=300)
    plt.close(fig)
    return panel_count


def main() -> None:
    args = parse_args()
    exhaustive.require_input_files(args.vi_b_raw_root)
    manifest = load_manifest(args.fast_dse_root)
    candidate_measurements = collect_measurements(manifest, args.vi_b_raw_root)
    candidate_path = (
        args.fast_dse_root / "fig8_filtered_candidate_measurements.csv"
    )
    candidate_measurements.to_csv(
        candidate_path,
        index=False,
        quoting=csv.QUOTE_MINIMAL,
        float_format="%.12g",
    )

    measurements = select_operating_points(
        candidate_measurements,
        args.vi_b_raw_root,
    )
    measurement_path = args.fast_dse_root / "fig8_markov_measurements.csv"
    measurements.to_csv(
        measurement_path,
        index=False,
        quoting=csv.QUOTE_MINIMAL,
        float_format="%.12g",
    )
    configuration_path = args.fast_dse_root / "fig8_markov_configurations.csv"
    measurements[
        ["model", "dataset", "objective", *CONFIG_FIELDS]
    ].to_csv(
        configuration_path,
        index=False,
        quoting=csv.QUOTE_MINIMAL,
    )
    panel_count = make_plots(args.vi_b_raw_root, measurements, args.fig_root)
    source = measurements["measurement_source"].iloc[0]
    print(f"Wrote {candidate_path}")
    print(f"Wrote {configuration_path}")
    print(f"Wrote {measurement_path}")
    print(
        f"Generated complete Fig. 8 with {panel_count} panels in {args.fig_root} "
        f"using {source} measurements."
    )


if __name__ == "__main__":
    main()
