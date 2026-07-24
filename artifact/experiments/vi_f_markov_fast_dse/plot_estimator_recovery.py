#!/usr/bin/env python3
"""Plot exhaustive, roofline, and Markov throughput recovery."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import PercentFormatter


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_ROOT = (
    REPO_ROOT
    / "artifact/run_outputs/vi_f_markov_fast_dse/nemo_et_db_profiles"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "artifact/figures/vi_f_markov_fast_dse"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def read_recovery(path: Path) -> float:
    frame = pd.read_csv(path)
    if len(frame) != 1:
        raise ValueError(f"Expected one recovery row in {path}.")
    row = frame.iloc[0]
    if int(row["candidate_budget_per_dataset"]) != 64:
        raise ValueError(f"{path} does not contain the 64-candidate result.")
    return float(row["average_throughput_recovery"])


def main() -> int:
    args = parse_args()
    roofline = read_recovery(
        args.data_root / "nemo_et_db_roofline_mt_recovery.csv"
    )
    markov = read_recovery(
        args.data_root / "nemo_et_db_mt_recovery.csv"
    )

    labels = ["Exhaustive", "Roofline", "Markov"]
    values = [1.0, roofline, markov]
    colors = ["#f2f2f2", "#a9bfd8", "#4e79a7"]
    hatches = ["//", "xx", ""]

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "hatch.linewidth": 0.5,
        }
    )
    figure, axis = plt.subplots(figsize=(3.4, 2.25))
    bars = axis.bar(
        labels,
        values,
        width=0.58,
        color=colors,
        edgecolor="black",
        linewidth=0.8,
    )
    for bar, hatch, value in zip(bars, hatches, values):
        bar.set_hatch(hatch)
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.025,
            f"{100.0 * value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    axis.set_ylabel(r"Average $M_T$ throughput recovery")
    axis.set_ylim(0.0, 1.13)
    axis.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    axis.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    axis.grid(axis="y", color="#d9d9d9", linewidth=0.6)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_stem = args.output_dir / "estimator_recovery_64_candidates"
    figure.tight_layout(pad=0.35)
    figure.savefig(
        output_stem.with_suffix(".png"),
        dpi=args.dpi,
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close(figure)

    print(f"Wrote {output_stem.with_suffix('.png')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
