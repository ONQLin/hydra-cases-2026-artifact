#!/usr/bin/env python3
"""Build the VI-C hybrid-LLM profile CSV from optional simulator outputs."""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ROOT = REPO_ROOT / "output_temp_runs" / "vi_c_hybrid_llm_dse"
MODEL_ALIASES = {
    "jamba-tiny": "jamba",
    "zamba2-7b": "zamba",
    "nemotronh-4b": "nemo",
}
CONFIG_COLUMNS = ["batchsize", "Num M", "Num Mp", "Num Md", "Num Ap", "Num Ad", "NoI_bw(GBps)"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_ROOT / "hybrid_llm_dse_runs")
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_ROOT / "profiles" / "hybrid_llm_dse_profiles.csv",
    )
    parser.add_argument("--min-completed", type=int, default=1)
    return parser.parse_args()


def suffix_float(line: str) -> float | None:
    match = re.search(r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*$", line.strip())
    return float(match.group(1)) if match else None


def parse_metrics(log_path: Path) -> dict[str, float | int | None]:
    metrics: dict[str, float | int | None] = {
        "tokens_per_sec": None,
        "finished_tokens_per_sec": None,
        "TTFT(s)": None,
        "completed_requests": None,
        "total_output_tokens": None,
    }
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "output tokens per second:" in line:
            metrics["tokens_per_sec"] = suffix_float(line)
        elif "finished tokens per second:" in line:
            metrics["finished_tokens_per_sec"] = suffix_float(line)
        elif "Average time to first token:" in line:
            value = suffix_float(line)
            metrics["TTFT(s)"] = None if value is None else value / 1_000_000.0
        elif "Total output tokens:" in line:
            match = re.search(r"Total output tokens:\s*(\d+)", line)
            metrics["total_output_tokens"] = int(match.group(1)) if match else None
        elif "Request counters - completed:" in line:
            match = re.search(r"completed:\s*(\d+)", line)
            metrics["completed_requests"] = int(match.group(1)) if match else None
        elif "In total" in line and "requests were processed" in line:
            match = re.search(r"In total\s+(\d+)\s+requests", line)
            metrics["completed_requests"] = int(match.group(1)) if match else None
    return metrics


def parse_alloc(raw: str | dict) -> dict:
    return raw if isinstance(raw, dict) else ast.literal_eval(raw)


def load_row(config_path: Path, min_completed: int) -> dict | None:
    run_dir = config_path.parent
    log_path = run_dir / "log_info.txt"
    if not log_path.is_file():
        return None
    config = json.loads(config_path.read_text(encoding="utf-8"))
    workload_model = str(config.get("workload_config", {}).get("model", "")).lower()
    model = MODEL_ALIASES.get(workload_model)
    if model is None:
        return None
    metrics = parse_metrics(log_path)
    if metrics["tokens_per_sec"] is None or metrics["TTFT(s)"] is None:
        return None
    completed = metrics["completed_requests"]
    if completed is not None and int(completed) < min_completed:
        return None

    alloc = parse_alloc(config["placmt_config"]["chiplet_alloc"])
    cluster = config["cluster_config"]
    placement = config["placmt_config"]
    mapping = config["mapping_config"]
    chips = config["chips_config"]
    return {
        "model": model,
        "batchsize": int(cluster["batch_size"]),
        "Num M": int(alloc.get("HBM3", 0)),
        "Num Mp": int(alloc.get("marca_p", 0)),
        "Num Md": int(alloc.get("marca_d", 0)),
        "Num Ap": int(alloc.get("tscs_p", 0)),
        "Num Ad": int(alloc.get("tscs_d", 0)),
        "NoI_bw(GBps)": int(float(chips["D2D_NoI_bw"])),
        "tokens_per_sec": metrics["tokens_per_sec"],
        "finished_tokens_per_sec": metrics["finished_tokens_per_sec"],
        "TTFT(s)": metrics["TTFT(s)"],
        "completed_requests": completed,
        "total_output_tokens": metrics["total_output_tokens"],
        "workload_model": workload_model,
        "placer": placement.get("placer_label", "unknown"),
        "req_scheduler": cluster.get("local_scheduler", "unknown"),
        "task_scheduler": mapping.get("mapping_strategy", "unknown"),
    }


def normalize_profiles(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data["tp_per_ttft"] = data["tokens_per_sec"] / data["TTFT(s)"].replace(0, pd.NA)
    data["norm_tp"] = 0.0
    data["norm_tp_per_ttft"] = 0.0
    for _, indexes in data.groupby("model").groups.items():
        model_data = data.loc[indexes]
        data.loc[indexes, "norm_tp"] = model_data["tokens_per_sec"] / model_data["tokens_per_sec"].max()
        data.loc[indexes, "norm_tp_per_ttft"] = model_data["tp_per_ttft"] / model_data["tp_per_ttft"].max()
    return data


def main() -> None:
    args = parse_args()
    input_root = args.input_root.resolve()
    rows = [
        row
        for path in sorted(input_root.rglob("config.json"))
        if (row := load_row(path, args.min_completed)) is not None
    ]
    if not rows:
        raise RuntimeError(f"No complete VI-C simulations found under {input_root}")

    raw = pd.DataFrame(rows)
    group_columns = ["model", *CONFIG_COLUMNS]
    aggregate_spec: dict[str, object] = {
        "tokens_per_sec": "mean",
        "finished_tokens_per_sec": "mean",
        "TTFT(s)": "mean",
        "completed_requests": "max",
        "total_output_tokens": "mean",
    }
    for column in ["workload_model", "placer", "req_scheduler", "task_scheduler"]:
        aggregate_spec[column] = lambda values: ",".join(sorted(set(map(str, values))))
    profiles = raw.groupby(group_columns, dropna=False).agg(aggregate_spec).reset_index()
    counts = raw.groupby(group_columns, dropna=False).size().rename("num_runs").reset_index()
    profiles = profiles.merge(counts, on=group_columns)
    ordered = [
        "model",
        *CONFIG_COLUMNS,
        "tokens_per_sec",
        "finished_tokens_per_sec",
        "TTFT(s)",
        "completed_requests",
        "total_output_tokens",
        "num_runs",
        "workload_model",
        "placer",
        "req_scheduler",
        "task_scheduler",
    ]
    profiles = normalize_profiles(profiles[ordered]).sort_values(group_columns, kind="stable")
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    profiles.to_csv(args.output_csv, index=False)
    print(f"Wrote {len(profiles)} profile rows to {args.output_csv}")
    print(profiles.groupby("model").size().to_string())


if __name__ == "__main__":
    main()
