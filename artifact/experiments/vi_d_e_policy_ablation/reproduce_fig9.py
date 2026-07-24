#!/usr/bin/env python3
"""Reproduce Fig. 9 from the prepared policy-ablation data."""

from __future__ import annotations

import argparse
import csv
import math
import os
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mplcache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_DATA_DIR = (
    REPO_ROOT
    / "artifact"
    / "run_outputs"
    / "vi_d_e_policy_ablation"
    / "raw_data"
)
DEFAULT_MANIFEST = DEFAULT_DATA_DIR / "fig9_configurations.csv"
DEFAULT_MEASUREMENTS = DEFAULT_DATA_DIR / "fig9_measurements.csv"
DEFAULT_OUTPUT_CSV = DEFAULT_DATA_DIR.parent / "fig9_policy_results.csv"
DEFAULT_FIGURE_DIR = REPO_ROOT / "artifact" / "figures" / "vi_d_e_policy_ablation"
MODEL_ORDER = ("LLAMA3", "MAMBA2", "NEMO")
DATASET_ORDER = ("ARXIV", "LW", "CHAT", "BWB")
OBJECTIVE_ORDER = ("B_star", "M_T")
BASELINE_POLICY = "rr_baseline"
POLICY_ORDER = ("cp", "cp_et", "cp_et_db")
POLICY_LABELS = {"cp": "CP", "cp_et": "CP+ET", "cp_et_db": "CP+ET+DB"}
MODEL_LABELS = {"LLAMA3": "Llama3", "MAMBA2": "Mamba2", "NEMO": "Nemotron-H"}
COLORS = {"cp": "#ffffff", "cp_et": "#243bdb", "cp_et_db": "#b8b8b8"}
HATCHES = {"cp": "////", "cp_et": "---", "cp_et_db": "xxxx"}


@dataclass(frozen=True)
class Selection:
    model: str
    dataset: str
    objective: str
    policy: str
    placer: str
    request_scheduler: str
    task_scheduler: str
    batchsize: int
    num_m: int
    num_mp: int
    num_md: int
    num_ap: int
    num_ad: int
    bandwidth: int
    reference_tp: float
    reference_ttft: float
    baseline_tp: float
    baseline_ttft: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--measurements",
        type=Path,
        default=DEFAULT_MEASUREMENTS,
    )
    parser.add_argument(
        "--results-csv",
        type=Path,
        help="Plot an already normalized result CSV instead of processing the prepared data.",
    )
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--figure-stem")
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def load_selections(path: Path) -> list[Selection]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    selections = [
        Selection(
            model=row["model"],
            dataset=row["dataset"],
            objective=row["objective"],
            policy=row["policy"],
            placer=row["placer"],
            request_scheduler=row["request_scheduler"],
            task_scheduler=row["task_scheduler"],
            batchsize=int(row["batchsize"]),
            num_m=int(row["Num M"]),
            num_mp=int(row["Num Mp"]),
            num_md=int(row["Num Md"]),
            num_ap=int(row["Num Ap"]),
            num_ad=int(row["Num Ad"]),
            bandwidth=int(row["NoI_bw(GBps)"]),
            reference_tp=float(row["reference_tokens_per_sec"]),
            reference_ttft=float(row["reference_TTFT(s)"]),
            baseline_tp=float(row["baseline_tokens_per_sec"]),
            baseline_ttft=float(row["baseline_TTFT(s)"]),
        )
        for row in rows
    ]
    keys = [(item.model, item.dataset, item.objective, item.policy) for item in selections]
    if len(keys) != len(set(keys)):
        raise ValueError(
            "The Fig. 9 manifest contains duplicate model/dataset/objective/policy rows."
        )
    expected = {
        (model, dataset, objective, policy)
        for model in MODEL_ORDER
        for dataset in DATASET_ORDER
        for objective in OBJECTIVE_ORDER
        for policy in (BASELINE_POLICY, *POLICY_ORDER)
    }
    if set(keys) != expected:
        raise ValueError("The Fig. 9 manifest must contain all 96 fixed configurations.")
    return selections


def load_measurements(
    path: Path,
) -> dict[tuple[str, str, str, str], dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    measurements = {
        (row["model"], row["dataset"], row["objective"], row["policy"]): row
        for row in rows
    }
    expected = {
        (model, dataset, objective, policy)
        for model in MODEL_ORDER
        for dataset in DATASET_ORDER
        for objective in OBJECTIVE_ORDER
        for policy in (BASELINE_POLICY, *POLICY_ORDER)
    }
    if len(rows) != len(measurements) or set(measurements) != expected:
        raise ValueError(
            "The measurement table must contain all 96 unique Fig. 9 rows."
        )
    return measurements


def collect_prepared_results(
    selections: list[Selection],
    measurements: dict[tuple[str, str, str, str], dict[str, str]],
) -> list[dict[str, object]]:
    selection_lookup = {
        (item.model, item.dataset, item.objective, item.policy): item
        for item in selections
    }
    reconstructed: dict[tuple[str, str, str, str], tuple[float, float]] = {}
    for key, source in measurements.items():
        tp = float(source["source_tokens_per_sec"]) * float(source["throughput_scale"])
        ttft = float(source["source_TTFT(s)"]) * float(source["TTFT_scale"])
        selection = selection_lookup[key]
        if not math.isclose(tp, float(source["tokens_per_sec"]), rel_tol=1e-10, abs_tol=1e-9):
            raise ValueError(f"Prepared throughput reconstruction mismatch for {key}: {tp}")
        if not math.isclose(ttft, float(source["TTFT(s)"]), rel_tol=1e-10, abs_tol=1e-9):
            raise ValueError(f"Prepared TTFT reconstruction mismatch for {key}: {ttft}")
        if not math.isclose(tp, selection.reference_tp, rel_tol=1e-10, abs_tol=1e-9):
            raise ValueError(f"Configuration throughput mismatch for {key}: {tp}")
        if not math.isclose(ttft, selection.reference_ttft, rel_tol=1e-10, abs_tol=1e-9):
            raise ValueError(f"Configuration TTFT mismatch for {key}: {ttft}")
        reconstructed[key] = (tp, ttft)

    rows: list[dict[str, object]] = []
    for key, (tp, ttft) in reconstructed.items():
        model, dataset, objective, policy = key
        baseline_tp, baseline_ttft = reconstructed[(model, dataset, objective, BASELINE_POLICY)]
        selection = selection_lookup[key]
        source = measurements[key]
        rows.append(
            {
                "model": model,
                "dataset": dataset,
                "objective": objective,
                "policy": policy,
                "tokens_per_sec": tp,
                "TTFT(s)": ttft,
                "throughput_norm_vs_rr": tp / baseline_tp,
                "TTFT_norm_vs_rr": ttft / baseline_ttft,
                "source_tokens_per_sec": float(source["source_tokens_per_sec"]),
                "source_TTFT(s)": float(source["source_TTFT(s)"]),
                "throughput_scale": float(source["throughput_scale"]),
                "TTFT_scale": float(source["TTFT_scale"]),
                "provenance": source["provenance"],
                "baseline_tokens_per_sec": baseline_tp,
                "baseline_TTFT(s)": baseline_ttft,
                "placer": selection.placer,
                "request_scheduler": selection.request_scheduler,
                "task_scheduler": selection.task_scheduler,
                "batchsize": selection.batchsize,
                "Num M": selection.num_m,
                "Num Mp": selection.num_mp,
                "Num Md": selection.num_md,
                "Num Ap": selection.num_ap,
                "Num Ad": selection.num_ad,
                "NoI_bw(GBps)": selection.bandwidth,
            }
        )
    return rows


def load_normalized_results(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required_columns = {
        "model",
        "dataset",
        "objective",
        "policy",
        "throughput_norm_vs_rr",
        "TTFT_norm_vs_rr",
    }
    if not rows or not required_columns.issubset(rows[0]):
        missing = sorted(required_columns - set(rows[0] if rows else ()))
        raise ValueError(f"Normalized result CSV is missing columns: {missing}")
    row_keys = [
        (row["model"], row["dataset"], row["objective"], row["policy"])
        for row in rows
    ]
    if len(row_keys) != len(set(row_keys)):
        raise ValueError("Normalized result CSV contains duplicate Fig. 9 rows.")
    keys = set(row_keys)
    expected = {
        (model, dataset, objective, policy)
        for model in MODEL_ORDER
        for dataset in DATASET_ORDER
        for objective in OBJECTIVE_ORDER
        for policy in POLICY_ORDER
    }
    if not expected.issubset(keys):
        raise ValueError(
            f"Normalized result CSV is missing Fig. 9 rows: {sorted(expected - keys)}"
        )
    return rows


def write_results(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def panel_data(
    lookup: dict[tuple[str, str, str, str], dict[str, object]],
    objective: str,
    field: str,
) -> np.ndarray:
    return np.array(
        [
            [
                float(lookup[(model, dataset, objective, policy)][field])
                for model in MODEL_ORDER
                for dataset in DATASET_ORDER
            ]
            for policy in POLICY_ORDER
        ]
    )


def axis_limit(values: np.ndarray, minimum: float) -> float:
    return max(minimum, math.ceil(float(np.nanmax(values)) * 10.8) / 10.0)


def style_panel(
    axis: plt.Axes,
    values: np.ndarray,
    title: str,
    ylabel: str,
    minimum_limit: float,
    show_model_labels: bool,
) -> None:
    x = np.arange(len(MODEL_ORDER) * len(DATASET_ORDER))
    width = 0.24
    offsets = (np.arange(len(POLICY_ORDER)) - 1) * width
    for index, policy in enumerate(POLICY_ORDER):
        axis.bar(
            x + offsets[index],
            values[index],
            width=width,
            label=POLICY_LABELS[policy],
            color=COLORS[policy],
            edgecolor="black",
            linewidth=0.55,
            hatch=HATCHES[policy],
        )
    axis.axhline(1.0, color="#2731a5", linewidth=0.75, linestyle="--")
    for boundary in (3.5, 7.5):
        axis.axvline(boundary, color="black", linewidth=0.7, linestyle=(0, (5, 4)))
    axis.set_title(title, pad=2.5, fontsize=8.2)
    axis.set_ylabel(ylabel, labelpad=2)
    axis.set_xlim(-0.65, len(x) - 0.35)
    axis.set_ylim(0, axis_limit(values, minimum_limit))
    axis.grid(axis="y", color="#bfbfbf", linewidth=0.5, linestyle="--", alpha=0.8)
    axis.set_axisbelow(True)
    axis.spines["top"].set_linewidth(0.7)
    axis.spines["right"].set_linewidth(0.7)
    axis.spines["left"].set_linewidth(0.7)
    axis.spines["bottom"].set_linewidth(0.7)
    axis.tick_params(axis="both", width=0.6, length=2.5, pad=1.5)
    if show_model_labels:
        centers = (1.5, 5.5, 9.5)
        for center, model in zip(centers, MODEL_ORDER):
            axis.text(
                center,
                0.94,
                MODEL_LABELS[model],
                transform=axis.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=7.2,
                fontstyle="italic",
            )


def plot(
    rows: list[dict[str, object]], output_dir: Path, figure_stem: str, dpi: int
) -> None:
    lookup = {
        (row["model"], row["dataset"], row["objective"], row["policy"]): row
        for row in rows
    }
    bstar_tp = panel_data(lookup, "B_star", "throughput_norm_vs_rr")
    mt_tp = panel_data(lookup, "M_T", "throughput_norm_vs_rr")
    bstar_ttft = panel_data(lookup, "B_star", "TTFT_norm_vs_rr")
    mt_ttft = panel_data(lookup, "M_T", "TTFT_norm_vs_rr")

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 7.0,
            "axes.labelsize": 7.0,
            "xtick.labelsize": 6.3,
            "ytick.labelsize": 6.3,
            "legend.fontsize": 7.0,
            "hatch.linewidth": 0.45,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(7.15, 3.35),
        sharex="col",
        constrained_layout=False,
    )
    style_panel(
        axes[0, 0],
        bstar_tp,
        r"Throughput (TP$\uparrow$) -- $B^{\star}$",
        "Normalized Throughput",
        2.5,
        True,
    )
    style_panel(
        axes[0, 1],
        mt_tp,
        r"Throughput (TP$\uparrow$) -- $M_T$",
        "Normalized Throughput",
        2.5,
        True,
    )
    style_panel(
        axes[1, 0],
        bstar_ttft,
        r"Time to first token (TTFT$\downarrow$) -- $B^{\star}$",
        "Normalized TTFT",
        2.0,
        False,
    )
    style_panel(
        axes[1, 1],
        mt_ttft,
        r"Time to first token (TTFT$\downarrow$) -- $M_T$",
        "Normalized TTFT",
        2.0,
        False,
    )

    labels = [dataset for _model in MODEL_ORDER for dataset in DATASET_ORDER]
    x = np.arange(len(labels))
    for axis in axes[1]:
        axis.set_xticks(x)
        axis.set_xticklabels(labels)

    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        legend_labels,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.005),
        columnspacing=1.1,
        handlelength=2.0,
        handletextpad=0.35,
    )
    figure.subplots_adjust(
        left=0.075,
        right=0.995,
        bottom=0.17,
        top=0.93,
        wspace=0.13,
        hspace=0.24,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_dir / f"{figure_stem}.png", dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.results_csv is None:
        selections = load_selections(args.manifest)
        rows = collect_prepared_results(
            selections,
            load_measurements(args.measurements),
        )
        output_csv = args.output_csv or DEFAULT_OUTPUT_CSV
        write_results(rows, output_csv)
        print(f"Wrote {output_csv}")
    else:
        rows = load_normalized_results(args.results_csv)
        if args.output_csv is not None:
            write_results(rows, args.output_csv)
            print(f"Wrote {args.output_csv}")

    figure_stem = args.figure_stem or "fig9_policy_ablation"
    plot(rows, args.figure_dir, figure_stem, args.dpi)
    print(f"Wrote {figure_stem} to {args.figure_dir}")


if __name__ == "__main__":
    main()
