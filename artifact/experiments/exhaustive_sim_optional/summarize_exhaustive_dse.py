#!/usr/bin/env python3
"""Summarize optional exhaustive simulations and derive TTFT/throughput Pareto CSVs."""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ROOT = REPO_ROOT / "output_temp_runs" / "vi_b_exhaustive_dse"
MODELS = ("LLAMA3", "MAMBA2", "NEMO")
DATASETS = ("ARXIV", "BWB", "CHAT", "LW")

TP_PATTERN = re.compile(r"output tokens per second:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")
TTFT_PATTERN = re.compile(r"Average time to first token:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")

NUMERIC_COLUMNS = [
    "batchsize",
    "Num M",
    "Num Mp",
    "Num Md",
    "Num Ap",
    "Num Ad",
    "NoI_bw(GBps)",
    "tokens_per_sec",
    "TTFT(s)",
]
METADATA_COLUMNS = [
    "placer",
    "req_scheduler",
    "task_scheduler",
    "source_kind",
    "source_label",
    "source_root",
    "source_dir",
    "simulator_dir",
    "run_dir",
]
CONFIG_KEY_COLUMNS = NUMERIC_COLUMNS[:7]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create Summary_Reports and Pareto_Reports from optional Section VI-B simulations."
    )
    parser.add_argument("--input-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Destination root; defaults to input-root.",
    )
    parser.add_argument("--models", nargs="+", choices=MODELS, default=None)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=None)
    return parser.parse_args()


def parse_metrics(log_path: Path) -> tuple[float | None, float | None]:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    tp_matches = TP_PATTERN.findall(text)
    ttft_matches = TTFT_PATTERN.findall(text)
    if not tp_matches or not ttft_matches:
        return None, None
    return float(tp_matches[-1]), float(ttft_matches[-1]) / 1_000_000.0


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_alloc(raw_alloc: str | dict) -> dict:
    return raw_alloc if isinstance(raw_alloc, dict) else ast.literal_eval(raw_alloc)


def identify_workload(config: dict) -> tuple[str, str] | None:
    model_name = str(config.get("workload_config", {}).get("model", "")).lower()
    model = {
        "llama3-8b": "LLAMA3",
        "mamba2-3b": "MAMBA2",
        "nemotronh-4b": "NEMO",
    }.get(model_name)
    dataset_name = str(config.get("workload_config", {}).get("dataset", "")).lower()
    dataset = {
        "arxiv": "ARXIV",
        "bwb": "BWB",
        "chat": "CHAT",
        "lw": "LW",
        "longwriter": "LW",
    }.get(dataset_name)
    if model is None or dataset is None:
        return None
    return model, dataset


def scrub_run_dir(name: str) -> str:
    return f"SCRUBBED_TIME+{name.split('+', 1)[1]}" if "+" in name else name


def row_from_run(config_path: Path, input_root: Path) -> tuple[tuple[str, str], dict] | None:
    run_dir = config_path.parent
    log_path = run_dir / "log_info.txt"
    if not log_path.is_file():
        return None
    tokens_per_sec, ttft_seconds = parse_metrics(log_path)
    if tokens_per_sec is None or ttft_seconds is None or not np.isfinite(ttft_seconds):
        return None

    config = load_config(config_path)
    workload = identify_workload(config)
    if workload is None:
        return None
    alloc = parse_alloc(config["placmt_config"]["chiplet_alloc"])
    cluster = config.get("cluster_config", {})
    placement = config.get("placmt_config", {})
    mapping = config.get("mapping_config", {})
    chips = config.get("chips_config", {})
    workload_dir = run_dir.parent.parent.name
    return workload, {
        "batchsize": int(cluster["batch_size"]),
        "Num M": int(alloc.get("HBM3", 0)),
        "Num Mp": int(alloc.get("marca_p", 0)),
        "Num Md": int(alloc.get("marca_d", 0)),
        "Num Ap": int(alloc.get("tscs_p", 0)),
        "Num Ad": int(alloc.get("tscs_d", 0)),
        "NoI_bw(GBps)": int(chips["D2D_NoI_bw"]),
        "tokens_per_sec": tokens_per_sec,
        "TTFT(s)": ttft_seconds,
        "placer": placement.get("placer_label", "unknown"),
        "req_scheduler": cluster.get("local_scheduler", "unknown"),
        "task_scheduler": mapping.get("mapping_strategy", "unknown"),
        "source_kind": "artifact_exhaustive",
        "source_label": "Exhaustive",
        "source_root": input_root.name,
        "source_dir": workload_dir,
        "simulator_dir": run_dir.parent.name,
        "run_dir": scrub_run_dir(run_dir.name),
    }


def pareto_partition(summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    working = summary.drop_duplicates().dropna(subset=["TTFT(s)", "tokens_per_sec"]).copy()
    working = working[
        np.isfinite(working["TTFT(s)"]) & np.isfinite(working["tokens_per_sec"]) & (working["TTFT(s)"] >= 0)
    ].copy()
    working = working.sort_values(
        ["TTFT(s)", "tokens_per_sec"], ascending=[True, False], kind="stable"
    ).reset_index(drop=True)

    is_pareto = np.zeros(len(working), dtype=bool)
    best_throughput = -np.inf
    for index, throughput in enumerate(working["tokens_per_sec"]):
        if throughput > best_throughput:
            is_pareto[index] = True
            best_throughput = throughput
    working["is_pareto"] = is_pareto
    return working[working["is_pareto"]].copy(), working[~working["is_pareto"]].copy()


def main() -> None:
    args = parse_args()
    input_root = args.input_root.resolve()
    output_root = (args.output_root or args.input_root).resolve()
    run_root = input_root / "simulator_runs"
    if not run_root.is_dir():
        raise FileNotFoundError(f"Simulation output directory not found: {run_root}")

    selected_models = set(args.models or MODELS)
    selected_datasets = set(args.datasets or DATASETS)
    grouped_rows: dict[tuple[str, str], list[dict]] = {}
    skipped = 0
    for config_path in sorted(run_root.glob("*/*/*/config.json")):
        parsed = row_from_run(config_path, input_root)
        if parsed is None:
            skipped += 1
            continue
        workload, row = parsed
        if workload[0] in selected_models and workload[1] in selected_datasets:
            grouped_rows.setdefault(workload, []).append(row)

    if not grouped_rows:
        raise RuntimeError(f"No completed simulations were found under {run_root}")

    summary_root = output_root / "Summary_Reports"
    pareto_root = output_root / "Pareto_Reports"
    summary_root.mkdir(parents=True, exist_ok=True)
    pareto_root.mkdir(parents=True, exist_ok=True)

    for (model, dataset), rows in sorted(grouped_rows.items()):
        summary = pd.DataFrame(rows)[NUMERIC_COLUMNS + METADATA_COLUMNS]
        summary = summary.sort_values(CONFIG_KEY_COLUMNS + ["run_dir"], kind="stable")
        summary = summary.drop_duplicates(subset=CONFIG_KEY_COLUMNS, keep="last").reset_index(drop=True)

        stem = f"results_summary_{model}-{dataset}_bw_static_static"
        summary_path = summary_root / f"{stem}.csv"
        pareto_path = pareto_root / f"pareto_frontier_{stem}.csv"
        non_pareto_path = pareto_root / f"non_pareto_{stem}.csv"
        summary.to_csv(summary_path, index=False)
        pareto, non_pareto = pareto_partition(summary)
        pareto.to_csv(pareto_path, index=False)
        non_pareto.to_csv(non_pareto_path, index=False)
        print(
            f"{model}-{dataset}: summary={len(summary)}, pareto={len(pareto)}, "
            f"non_pareto={len(non_pareto)}"
        )

    if skipped:
        print(f"Skipped {skipped} run directories without complete TP and TTFT metrics.")
    print(f"Wrote reports under {output_root}")


if __name__ == "__main__":
    main()
