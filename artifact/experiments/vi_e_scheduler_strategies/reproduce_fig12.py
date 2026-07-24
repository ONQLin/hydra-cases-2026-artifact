#!/usr/bin/env python3
"""Post-process the selected task-scheduler runs and reproduce Fig. 12."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mplcache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_MANIFEST = EXPERIMENT_DIR / "fig12_configurations.csv"
DEFAULT_RUN_ROOT = (
    REPO_ROOT / "artifact" / "run_outputs" / "vi_e_scheduler_strategies" / "fig12" / "raw_simulations"
)
DEFAULT_OUTPUT_CSV = DEFAULT_RUN_ROOT.parent / "fig12_scheduler_results.csv"
DEFAULT_FIGURE_DIR = REPO_ROOT / "artifact" / "figures" / "vi_e_scheduler_strategies"
DATASET_ORDER = ("ARXIV", "BWB", "CHAT", "LW")
OBJECTIVE_ORDER = ("B_star", "M_T")
SCHEDULER_ORDER = ("static", "fcfs", "worksteal", "elastic")
SCHEDULER_LABELS = {
    "static": "Static",
    "fcfs": "FCFS",
    "worksteal": "Work-steal",
    "elastic": "Elastic",
}
COLORS = {
    "static": "#4C566A",
    "fcfs": "#D08770",
    "worksteal": "#EBCB8B",
    "elastic": "#5E81AC",
}
HATCHES = {
    "static": "",
    "fcfs": "///",
    "worksteal": "\\\\\\",
    "elastic": "xx",
}


@dataclass(frozen=True)
class Selection:
    dataset: str
    objective: str
    scheduler: str
    batchsize: int
    num_m: int
    num_mp: int
    num_md: int
    num_ap: int
    num_ad: int
    bandwidth: int
    reference_tp: float
    reference_ttft: float
    rr_denominator: float

    @property
    def label(self) -> str:
        return (
            f"FIG12_{self.dataset}_{self.objective}_{self.scheduler}_"
            f"BW{self.bandwidth}_BS{self.batchsize}_M{self.num_m}_Ap{self.num_ap}_"
            f"Ad{self.num_ad}_Mp{self.num_mp}_Md{self.num_md}"
        )

    def run_parent(self, root: Path) -> Path:
        objective = "bstar" if self.objective == "B_star" else "mt"
        return root / self.dataset.lower() / objective / self.scheduler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def load_selections(path: Path) -> list[Selection]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    selections = [
        Selection(
            dataset=row["dataset"],
            objective=row["objective"],
            scheduler=row["scheduler"],
            batchsize=int(row["batchsize"]),
            num_m=int(row["Num M"]),
            num_mp=int(row["Num Mp"]),
            num_md=int(row["Num Md"]),
            num_ap=int(row["Num Ap"]),
            num_ad=int(row["Num Ad"]),
            bandwidth=int(row["NoI_bw(GBps)"]),
            reference_tp=float(row["reference_tokens_per_sec"]),
            reference_ttft=float(row["reference_TTFT(s)"]),
            rr_denominator=float(row["rr_denominator"]),
        )
        for row in rows
    ]
    keys = [(item.dataset, item.objective, item.scheduler) for item in selections]
    if len(keys) != len(set(keys)):
        raise ValueError("The Fig. 12 manifest contains duplicate dataset/objective/scheduler rows.")
    return selections


def suffix_float(line: str) -> float | None:
    match = re.search(r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*$", line.strip())
    return float(match.group(1)) if match else None


def parse_metrics(path: Path) -> dict[str, float | int | None]:
    result: dict[str, float | int | None] = {
        "tokens_per_sec": None,
        "TTFT(s)": None,
        "completed_requests": None,
    }
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "output tokens per second:" in line:
            result["tokens_per_sec"] = suffix_float(line)
        elif "Average time to first token:" in line:
            value = suffix_float(line)
            result["TTFT(s)"] = None if value is None else value / 1_000_000.0
        elif "Request counters - completed:" in line:
            match = re.search(r"completed:\s*(\d+)", line)
            result["completed_requests"] = int(match.group(1)) if match else None
        elif "In total" in line and "requests were processed" in line:
            match = re.search(r"In total\s+(\d+)\s+requests", line)
            result["completed_requests"] = int(match.group(1)) if match else None
    return result


def parse_alloc(value: str | dict) -> dict:
    return value if isinstance(value, dict) else ast.literal_eval(value)


def validate_config(selection: Selection, config: dict, run_dir: Path) -> None:
    alloc = parse_alloc(config["placmt_config"]["chiplet_alloc"])
    actual = (
        int(config["cluster_config"]["batch_size"]),
        int(alloc["HBM3"]),
        int(alloc["marca_p"]),
        int(alloc["marca_d"]),
        int(alloc["tscs_p"]),
        int(alloc["tscs_d"]),
        int(float(config["chips_config"]["D2D_NoI_bw"])),
        config["placmt_config"]["placer_label"],
        config["cluster_config"]["local_scheduler"],
        config["mapping_config"]["mapping_strategy"],
    )
    expected = (
        selection.batchsize,
        selection.num_m,
        selection.num_mp,
        selection.num_md,
        selection.num_ap,
        selection.num_ad,
        selection.bandwidth,
        "bw",
        "static",
        selection.scheduler,
    )
    if actual != expected:
        raise ValueError(f"Configuration mismatch in {run_dir}: expected {expected}, found {actual}")


def find_run(selection: Selection, root: Path) -> Path:
    candidates = []
    for run_dir in selection.run_parent(root).glob(f"*+{selection.label}"):
        config_path = run_dir / "config.json"
        log_path = run_dir / "log_info.txt"
        if not config_path.is_file() or not log_path.is_file():
            continue
        metrics = parse_metrics(log_path)
        if metrics["tokens_per_sec"] is None or metrics["TTFT(s)"] is None:
            continue
        candidates.append(run_dir)
    if not candidates:
        raise FileNotFoundError(
            f"No complete run found for {selection.dataset}/{selection.objective}/{selection.scheduler}"
        )
    return max(candidates, key=lambda path: (path / "log_info.txt").stat().st_mtime)


def metric(objective: str, throughput: float, ttft: float) -> float:
    return throughput / ttft if objective == "B_star" else throughput


def portable_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def collect_results(selections: list[Selection], run_root: Path) -> list[dict[str, object]]:
    rows = []
    for selection in selections:
        run_dir = find_run(selection, run_root)
        config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        validate_config(selection, config, run_dir)
        metrics = parse_metrics(run_dir / "log_info.txt")
        tp = float(metrics["tokens_per_sec"])
        ttft = float(metrics["TTFT(s)"])
        reproduced_metric = metric(selection.objective, tp, ttft)
        reference_metric = metric(selection.objective, selection.reference_tp, selection.reference_ttft)
        rows.append(
            {
                "dataset": selection.dataset,
                "objective": selection.objective,
                "scheduler": selection.scheduler,
                "metric_name": "TP_per_TTFT" if selection.objective == "B_star" else "tokens_per_sec",
                "metric": reproduced_metric,
                "norm_vs_rr": reproduced_metric / selection.rr_denominator,
                "tokens_per_sec": tp,
                "TTFT(s)": ttft,
                "completed_requests": metrics["completed_requests"],
                "reference_metric": reference_metric,
                "reference_norm_vs_rr": reference_metric / selection.rr_denominator,
                "reference_tokens_per_sec": selection.reference_tp,
                "reference_TTFT(s)": selection.reference_ttft,
                "metric_delta_percent": 100.0 * (reproduced_metric / reference_metric - 1.0),
                "rr_denominator": selection.rr_denominator,
                "batchsize": selection.batchsize,
                "Num M": selection.num_m,
                "Num Mp": selection.num_mp,
                "Num Md": selection.num_md,
                "Num Ap": selection.num_ap,
                "Num Ad": selection.num_ad,
                "NoI_bw(GBps)": selection.bandwidth,
                "run_dir": portable_path(run_dir),
            }
        )
    return rows


def write_results(rows: list[dict[str, object]], path: Path) -> None:
    fields = [
        "dataset",
        "objective",
        "scheduler",
        "metric_name",
        "metric",
        "norm_vs_rr",
        "tokens_per_sec",
        "TTFT(s)",
        "completed_requests",
        "reference_metric",
        "reference_norm_vs_rr",
        "reference_tokens_per_sec",
        "reference_TTFT(s)",
        "metric_delta_percent",
        "rr_denominator",
        "batchsize",
        "Num M",
        "Num Mp",
        "Num Md",
        "Num Ap",
        "Num Ad",
        "NoI_bw(GBps)",
        "run_dir",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def style_axes(axis: plt.Axes, ylabel: str, ylim: tuple[float, float]) -> None:
    axis.set_ylabel(ylabel, labelpad=2)
    axis.set_xticks(np.arange(len(DATASET_ORDER)))
    axis.set_xticklabels(DATASET_ORDER)
    axis.set_ylim(*ylim)
    axis.grid(axis="y", color="#D8DEE9", linewidth=0.45)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_linewidth(0.6)
    axis.spines["bottom"].set_linewidth(0.6)
    axis.tick_params(axis="both", width=0.6, length=2.5, pad=1.5)


def plot_panel(axis: plt.Axes, values: np.ndarray, title: str, ylabel: str) -> None:
    x = np.arange(len(DATASET_ORDER))
    width = 0.18
    offsets = (np.arange(len(SCHEDULER_ORDER)) - (len(SCHEDULER_ORDER) - 1) / 2.0) * width
    for index, scheduler in enumerate(SCHEDULER_ORDER):
        axis.bar(
            x + offsets[index],
            values[index],
            width=width,
            label=SCHEDULER_LABELS[scheduler],
            color=COLORS[scheduler],
            edgecolor="black",
            linewidth=0.35,
            hatch=HATCHES[scheduler],
        )
    style_axes(axis, ylabel, (0, max(1.18, float(np.nanmax(values)) * 1.12)))
    axis.text(
        0.01,
        0.94,
        title,
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=7.2,
        fontweight="bold",
    )


def plot(rows: list[dict[str, object]], output_dir: Path, dpi: int) -> None:
    lookup = {(row["dataset"], row["objective"], row["scheduler"]): row for row in rows}
    data = {}
    for objective in OBJECTIVE_ORDER:
        data[objective] = np.array(
            [
                [float(lookup[(dataset, objective, scheduler)]["norm_vs_rr"]) for dataset in DATASET_ORDER]
                for scheduler in SCHEDULER_ORDER
            ]
        )

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 7.0,
            "axes.labelsize": 7.0,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.4,
            "hatch.linewidth": 0.45,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, axes = plt.subplots(2, 1, figsize=(3.45, 2.75), sharex=True, constrained_layout=False)
    plot_panel(axes[0], data["B_star"], r"(a) $B^{\star}$: TP/TTFT", "Norm.  TP/TTFT")
    plot_panel(axes[1], data["M_T"], r"(b) $M_T$: Throughput", "Norm.  TP")

    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.52, 0.975),
        columnspacing=0.8,
        handlelength=1.45,
        handletextpad=0.35,
    )
    figure.subplots_adjust(left=0.13, right=0.995, bottom=0.095, top=0.89, hspace=0.12)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_dir / "fig12_scheduler_bars.png", dpi=dpi, bbox_inches="tight", pad_inches=0.015)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    rows = collect_results(load_selections(args.manifest), args.run_root.resolve())
    write_results(rows, args.output_csv)
    plot(rows, args.figure_dir, args.dpi)
    largest_delta = max(abs(float(row["metric_delta_percent"])) for row in rows)
    print(f"Wrote {args.output_csv}")
    print(f"Wrote Fig. 12 to {args.figure_dir}")
    print(f"Largest absolute metric difference from the paper source data: {largest_delta:.3f}%")


if __name__ == "__main__":
    main()
