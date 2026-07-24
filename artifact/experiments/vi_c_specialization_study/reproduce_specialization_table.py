#!/usr/bin/env python3
"""Reproduce the Section VI-C specialization--generality study and Table IV."""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
os.environ.setdefault("MPLCONFIGDIR", str((REPO_ROOT / ".mplcache").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


MODEL_ORDER = ["jamba", "zamba", "nemo"]
MODEL_LABELS = {"jamba": "Jamba", "zamba": "Zamba", "nemo": "Nemo"}
MODEL_ALIASES = {
    "jamba": "jamba",
    "jamba-tiny": "jamba",
    "zamba": "zamba",
    "zamba2-7b": "zamba",
    "nemo": "nemo",
    "nemotronh-4b": "nemo",
}
CONFIG_COLUMNS = ["batchsize", "Num M", "Num Mp", "Num Md", "Num Ap", "Num Ad", "NoI_bw(GBps)"]
TARGET_GROUPS = [
    ("jamba",),
    ("zamba",),
    ("nemo",),
    ("jamba", "zamba"),
    ("jamba", "nemo"),
    ("zamba", "nemo"),
    ("jamba", "zamba", "nemo"),
]
OBJECTIVES = {
    "bstar": {
        "title": r"(a) $B^{\star}$: norm. TP/TTFT $\uparrow$",
        "metric": "norm_tp_per_ttft",
        "raw_metric": "tp_per_ttft",
    },
    "mt": {
        "title": r"(b) $M_T$: norm. TP $\uparrow$",
        "metric": "norm_tp",
        "raw_metric": "tokens_per_sec",
    },
}
ROW_COLORS = {1: "#EBF5FF", 2: "#F5F5F5", 3: "#FFF5E6"}

DEFAULT_INPUT = (
    REPO_ROOT
    / "artifact"
    / "run_outputs"
    / "vi_c_specialization_study"
    / "raw_data"
    / "hybrid_llm_dse_profiles.csv"
)
DEFAULT_FIGURE = (
    REPO_ROOT
    / "artifact"
    / "figures"
    / "vi_c_specialization_study"
    / "table_iv_reproduced.png"
)


@dataclass
class ObjectiveResult:
    selected: pd.DataFrame
    evaluation: pd.DataFrame
    matrix: pd.DataFrame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-figure", type=Path, default=DEFAULT_FIGURE)
    parser.add_argument(
        "--mixed-selection",
        choices=["min", "geomean", "mean"],
        default="min",
        help="Score used to select configurations for multi-model targets.",
    )
    parser.add_argument(
        "--throughput-metric",
        choices=["tokens_per_sec", "finished_tokens_per_sec"],
        default="tokens_per_sec",
    )
    return parser.parse_args()


def normalize_model(name: str) -> str:
    return MODEL_ALIASES.get(str(name).lower(), str(name).lower())


def target_label(group: tuple[str, ...]) -> str:
    return " + ".join(MODEL_LABELS[model] for model in group)


def normalize_profiles(profile: pd.DataFrame, throughput_metric: str) -> pd.DataFrame:
    required = {"model", *CONFIG_COLUMNS, throughput_metric, "TTFT(s)"}
    missing = sorted(required - set(profile.columns))
    if missing:
        raise ValueError(f"Input profile is missing columns: {', '.join(missing)}")

    profile = profile.copy()
    profile["model"] = profile["model"].map(normalize_model)
    profile = profile[profile["model"].isin(MODEL_ORDER)].copy()
    if set(profile["model"]) != set(MODEL_ORDER):
        raise ValueError("Input profile must contain Jamba, Zamba, and Nemo rows.")
    profile["tokens_per_sec"] = profile[throughput_metric].astype(float)
    profile["tp_per_ttft"] = profile["tokens_per_sec"] / profile["TTFT(s)"].replace(0, pd.NA)
    profile["norm_tp"] = profile["tokens_per_sec"] / profile.groupby("model")["tokens_per_sec"].transform("max")
    profile["norm_tp_per_ttft"] = profile["tp_per_ttft"] / profile.groupby("model")["tp_per_ttft"].transform("max")
    return profile


def available_keys(profile: pd.DataFrame, models: tuple[str, ...]) -> set[tuple]:
    common: set[tuple] | None = None
    for model in models:
        model_keys = set(profile[profile["model"] == model][CONFIG_COLUMNS].itertuples(index=False, name=None))
        common = model_keys if common is None else common & model_keys
    return common or set()


def profile_lookup(profile: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        model: rows.set_index(CONFIG_COLUMNS, drop=False)
        for model, rows in profile.groupby("model", sort=False)
    }


def lookup_row(lookup: dict[str, pd.DataFrame], model: str, key: tuple, metric: str) -> pd.Series | None:
    if model not in lookup or key not in lookup[model].index:
        return None
    row = lookup[model].loc[key]
    if isinstance(row, pd.DataFrame):
        return row.sort_values(metric, ascending=False).iloc[0]
    return row


def mixed_score(values: list[float], policy: str) -> float:
    if len(values) == 1:
        return values[0]
    if policy == "min":
        return min(values)
    if policy == "mean":
        return sum(values) / len(values)
    if any(value <= 0 or not math.isfinite(value) for value in values):
        return 0.0
    return math.exp(sum(math.log(value) for value in values) / len(values))


def select_configurations(
    profile: pd.DataFrame,
    metric: str,
    raw_metric: str,
    policy: str,
) -> pd.DataFrame:
    lookup = profile_lookup(profile)
    selected_rows: list[dict] = []
    for group in TARGET_GROUPS:
        best: dict | None = None
        for key in sorted(available_keys(profile, group)):
            target_rows = [lookup_row(lookup, model, key, metric) for model in group]
            if any(row is None for row in target_rows):
                continue
            values = [float(row[metric]) for row in target_rows if row is not None]
            raw_values = [float(row[raw_metric]) for row in target_rows if row is not None]
            average = sum(values) / len(values)
            candidate = {
                **dict(zip(CONFIG_COLUMNS, key)),
                "DSE Target": target_label(group),
                "target_key": "+".join(group),
                "specialization_level": len(group),
                "selection_score": mixed_score(values, policy),
                "target_avg_norm_metric": average,
                "target_min_norm_metric": min(values),
                "target_avg_raw_metric": sum(raw_values) / len(raw_values),
                "num_target_models": len(group),
            }
            rank = (candidate["selection_score"], average)
            if best is None or rank > (best["selection_score"], best["target_avg_norm_metric"]):
                best = candidate
        if best is None:
            raise RuntimeError(f"No shared configuration found for {target_label(group)}.")
        selected_rows.append(best)
    return pd.DataFrame(selected_rows)


def evaluate_configurations(
    profile: pd.DataFrame,
    selected: pd.DataFrame,
    metric: str,
    raw_metric: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    lookup = profile_lookup(profile)
    rows: list[dict] = []
    for _, choice in selected.iterrows():
        key = tuple(choice[column] for column in CONFIG_COLUMNS)
        for model in MODEL_ORDER:
            result = lookup_row(lookup, model, key, metric)
            if result is None:
                continue
            rows.append(
                {
                    "DSE Target": choice["DSE Target"],
                    "target_key": choice["target_key"],
                    "eval_model": model,
                    "eval_model_label": MODEL_LABELS[model],
                    "normalized_metric": float(result[metric]),
                    "raw_metric": float(result[raw_metric]),
                    "tokens_per_sec": float(result["tokens_per_sec"]),
                    "TTFT(s)": float(result["TTFT(s)"]),
                    **{column: choice[column] for column in CONFIG_COLUMNS},
                }
            )
    evaluation = pd.DataFrame(rows)
    matrix = evaluation.pivot(index="DSE Target", columns="eval_model_label", values="normalized_metric")
    matrix = matrix.reindex(
        index=[target_label(group) for group in TARGET_GROUPS],
        columns=[MODEL_LABELS[model] for model in MODEL_ORDER],
    )
    return evaluation, matrix


def run_study(profile: pd.DataFrame, args: argparse.Namespace) -> dict[str, ObjectiveResult]:
    profile = normalize_profiles(profile, args.throughput_metric)
    results: dict[str, ObjectiveResult] = {}
    for name, spec in OBJECTIVES.items():
        selected = select_configurations(profile, spec["metric"], spec["raw_metric"], args.mixed_selection)
        evaluation, matrix = evaluate_configurations(profile, selected, spec["metric"], spec["raw_metric"])
        results[name] = ObjectiveResult(selected=selected, evaluation=evaluation, matrix=matrix)
    return results


def format_value(value: float) -> str:
    return "--" if pd.isna(value) else f"{float(value):.2f}"


def render_table_panel(axis: plt.Axes, matrix: pd.DataFrame, title: str) -> None:
    axis.axis("off")
    axis.set_title(title, fontsize=12, fontweight="bold", pad=4)
    cell_text = [[name, *(format_value(value) for value in row)] for name, row in matrix.iterrows()]
    table = axis.table(
        cellText=cell_text,
        colLabels=["DSE Target", "Jamba", "Zamba", "Nemo"],
        colWidths=[0.52, 0.16, 0.16, 0.16],
        cellLoc="center",
        loc="center",
        bbox=[0.0, 0.0, 1.0, 0.91],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    for column in range(4):
        cell = table[(0, column)]
        cell.set_facecolor("white")
        cell.set_text_props(weight="bold")
        cell.set_edgecolor("black")
        cell.set_linewidth(0.9)
    for row_index, group in enumerate(TARGET_GROUPS, start=1):
        for column in range(4):
            cell = table[(row_index, column)]
            cell.set_facecolor(ROW_COLORS[len(group)])
            cell.set_edgecolor("white")
            cell.set_linewidth(0.5)
        table[(row_index, 0)].set_text_props(ha="left")
        for model_index, model in enumerate(MODEL_ORDER, start=1):
            if model not in group:
                table[(row_index, model_index)].set_text_props(color="red")


def render_tables(matrices: dict[str, pd.DataFrame], output_path: Path, heading: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 1, figsize=(7.0, 6.0))
    figure.suptitle(heading, fontsize=13, fontweight="bold", y=0.985)
    render_table_panel(axes[0], matrices["bstar"], OBJECTIVES["bstar"]["title"])
    render_table_panel(axes[1], matrices["mt"], OBJECTIVES["mt"]["title"])
    legend = [
        Patch(facecolor=ROW_COLORS[1], edgecolor="none", label="Most-specialized"),
        Patch(facecolor=ROW_COLORS[2], edgecolor="none", label="Medium-specialized"),
        Patch(facecolor=ROW_COLORS[3], edgecolor="none", label="General"),
    ]
    figure.legend(handles=legend, loc="lower center", ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.01))
    figure.text(0.5, 0.055, "Red: off-target evaluation", color="red", ha="center", fontsize=9)
    figure.subplots_adjust(top=0.93, bottom=0.11, hspace=0.25, left=0.05, right=0.95)
    figure.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if not args.input_csv.is_file():
        raise FileNotFoundError(f"Prepared hybrid-LLM profile not found: {args.input_csv}")
    results = run_study(pd.read_csv(args.input_csv), args)
    reproduced_matrices = {name: result.matrix for name, result in results.items()}
    render_tables(
        reproduced_matrices,
        args.output_figure,
        "Specialization--Generality Tradeoff Across Hybrid LLMs",
    )
    print(f"Wrote the reproduced VI-C table to {args.output_figure}")


if __name__ == "__main__":
    main()
