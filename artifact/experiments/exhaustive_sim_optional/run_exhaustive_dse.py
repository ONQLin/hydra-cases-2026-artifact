#!/usr/bin/env python3
"""Run the optional exhaustive DSE that supplies the Section VI-B CSV data."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "output_temp_runs" / "vi_b_exhaustive_dse"

# Adjust this constant or pass --num-in-parallel to control concurrent simulations.
NUM_IN_PARALLEL = 32

MODEL_NAMES = {
    "LLAMA3": "llama3-8b",
    "MAMBA2": "mamba2-3b",
    "NEMO": "nemotronh-4b",
}
DATASET_ARGS = {
    "ARXIV": "arxiv",
    "BWB": "bwb",
    "CHAT": "chat",
    "LW": "longwriter",
}

# These are the traces recorded in the configs used to build the prepared Fig. 8 data.
TRACE_FILES = {
    ("LLAMA3", "ARXIV"): "dataset/arxiv/arxiv_summarization_stats_llama3.csv",
    ("LLAMA3", "BWB"): "dataset/bwb/bwb_translation_stats_llama3.csv",
    ("LLAMA3", "CHAT"): "dataset/chat/chat1m_stats_llama3.csv",
    ("LLAMA3", "LW"): "dataset/longwriter/long_writer_transformer_1.3b_6k.csv",
    ("MAMBA2", "ARXIV"): "dataset/arxiv/arxiv_summarization_stats_mamba2.csv",
    ("MAMBA2", "BWB"): "dataset/bwb/bwb_translation_stats_mamba2.csv",
    ("MAMBA2", "CHAT"): "dataset/chat/chat1m_stats_mamba2.csv",
    ("MAMBA2", "LW"): "dataset/longwriter/long_writer_mamba_6k.csv",
    ("NEMO", "ARXIV"): "dataset/arxiv/arxiv_summarization_stats_llama3.csv",
    ("NEMO", "BWB"): "dataset/bwb/bwb_translation_stats_llama3.csv",
    ("NEMO", "CHAT"): "dataset/chat/chat1m_stats_llama3.csv",
    ("NEMO", "LW"): "dataset/longwriter/long_writer_transformer_1.3b_6k.csv",
}

METRIC_PATTERNS = (
    re.compile(r"output tokens per second:\s*[-+]?\d*\.?\d+"),
    re.compile(r"Average time to first token:\s*[-+]?\d*\.?\d+"),
)


@dataclass(frozen=True)
class Architecture:
    num_m: int
    num_mp: int
    num_md: int
    num_ap: int
    num_ad: int


@dataclass(frozen=True)
class SimulationTask:
    model: str
    dataset: str
    batch_size: int
    bandwidth: int
    architecture: Architecture
    output_dir: Path

    @property
    def label(self) -> str:
        a = self.architecture
        return (
            f"{self.model}-{self.dataset}_BMPD{self.bandwidth}_{a.num_m}_"
            f"{a.num_ap}_{a.num_mp}_{a.num_ad}_{a.num_md}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the optional exhaustive bw/static/static DSE for Section VI-B (Fig. 8)."
    )
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES))
    parser.add_argument("--datasets", nargs="+", choices=DATASET_ARGS, default=list(DATASET_ARGS))
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[2, 4, 8, 16, 32, 64])
    parser.add_argument("--bandwidths", nargs="+", type=int, default=[256, 384, 512, 640])
    parser.add_argument(
        "--memory-chiplets",
        nargs="+",
        type=int,
        default=None,
        help="Restrict HBM3 counts. The full model-specific range is used by default.",
    )
    parser.add_argument("--num-in-parallel", type=int, default=NUM_IN_PARALLEL)
    parser.add_argument("--time-limit", type=int, default=100)
    parser.add_argument(
        "--command-timeout",
        type=int,
        default=0,
        help="Optional wall-clock timeout per simulation in seconds; 0 disables it.",
    )
    parser.add_argument(
        "--max-configs-per-workload",
        type=int,
        default=0,
        help="Run only the first N configurations for each model/dataset; 0 runs all.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--python-bin",
        default=sys.executable,
        help="Python interpreter used to invoke main.py.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the sweep size without launching simulations.")
    parser.add_argument(
        "--rerun-completed",
        action="store_true",
        help="Run configurations even when a completed output with the same label exists.",
    )
    return parser.parse_args()


def pure_model_architectures(memory_counts: Iterable[int]) -> list[Architecture]:
    architectures: list[Architecture] = []
    for num_m in memory_counts:
        remain = 24 - num_m
        for num_p in range(1, remain):
            architectures.append(
                Architecture(num_m=num_m, num_mp=0, num_md=0, num_ap=num_p, num_ad=remain - num_p)
            )
    return architectures


def mamba_architectures(memory_counts: Iterable[int]) -> list[Architecture]:
    architectures: list[Architecture] = []
    for num_m in memory_counts:
        remain = 24 - num_m
        for num_p in range(1, remain):
            architectures.append(
                Architecture(num_m=num_m, num_mp=num_p, num_md=remain - num_p, num_ap=0, num_ad=0)
            )
    return architectures


def nemo_architectures(memory_counts: Iterable[int]) -> list[Architecture]:
    architectures: list[Architecture] = []
    for num_m in memory_counts:
        remain = 24 - num_m
        if remain < 4:
            continue
        for num_ap in range(1, remain - 2, 2):
            for num_ad in range(1, remain - num_ap - 1, 2):
                for num_mp in range(1, remain - num_ap - num_ad, 2):
                    num_md = remain - num_ap - num_ad - num_mp
                    architectures.append(
                        Architecture(
                            num_m=num_m,
                            num_mp=num_mp,
                            num_md=num_md,
                            num_ap=num_ap,
                            num_ad=num_ad,
                        )
                    )
    return architectures


def architectures_for(model: str, requested_memory_counts: list[int] | None) -> list[Architecture]:
    if requested_memory_counts is None:
        memory_counts = range(2, 17, 2) if model == "NEMO" else range(2, 17)
    else:
        memory_counts = sorted(set(requested_memory_counts))

    invalid = [count for count in memory_counts if count < 2 or count > 16]
    if invalid:
        raise ValueError(f"HBM3 counts must be between 2 and 16; got {invalid}")
    if model == "NEMO" and any(count % 2 for count in memory_counts):
        raise ValueError("Nemotron-H uses even HBM3 counts so all four compute-chiplet counts remain odd.")

    if model == "LLAMA3":
        return pure_model_architectures(memory_counts)
    if model == "MAMBA2":
        return mamba_architectures(memory_counts)
    return nemo_architectures(memory_counts)


def build_tasks(args: argparse.Namespace) -> list[SimulationTask]:
    tasks: list[SimulationTask] = []
    run_root = args.output_root.resolve() / "simulator_runs"
    for model in args.models:
        architectures = architectures_for(model, args.memory_chiplets)
        for dataset in args.datasets:
            workload_tasks: list[SimulationTask] = []
            for batch_size in args.batch_sizes:
                for bandwidth in args.bandwidths:
                    output_dir = (
                        run_root
                        / f"{model}-{dataset}"
                        / f"simulator_output_{model}-{dataset}_bwplacement_bs{batch_size}_staticreq_statictask"
                    )
                    for architecture in architectures:
                        workload_tasks.append(
                            SimulationTask(
                                model=model,
                                dataset=dataset,
                                batch_size=batch_size,
                                bandwidth=bandwidth,
                                architecture=architecture,
                                output_dir=output_dir,
                            )
                        )
            if args.max_configs_per_workload > 0:
                workload_tasks = workload_tasks[: args.max_configs_per_workload]
            tasks.extend(workload_tasks)
    return tasks


def command_for(task: SimulationTask, args: argparse.Namespace) -> list[str]:
    a = task.architecture
    return [
        args.python_bin,
        str(REPO_ROOT / "main.py"),
        "--workload-config.model",
        MODEL_NAMES[task.model],
        "--workload-config.dataset",
        DATASET_ARGS[task.dataset],
        "--placmt-config.chiplet-alloc.marca_p",
        str(a.num_mp),
        "--placmt-config.chiplet-alloc.marca_d",
        str(a.num_md),
        "--placmt-config.chiplet-alloc.tscs_p",
        str(a.num_ap),
        "--placmt-config.chiplet-alloc.tscs_d",
        str(a.num_ad),
        "--placmt-config.chiplet-alloc.HBM3",
        str(a.num_m),
        "--arch-config.num_nodes",
        "24",
        "--arch-config.intp_width",
        "6",
        "--arch-config.intp_height",
        "4",
        "--metrics-config.label_name",
        task.label,
        "--metrics-config.output_dir",
        str(task.output_dir),
        "--chips-config.D2D_NoI_bw",
        str(task.bandwidth),
        "--chips-config.area_limit",
        "144",
        "--chips-config.Fixed_chiplet_area",
        "--workload-config.request-generator-config.trace-length-generator-config.trace_file",
        TRACE_FILES[(task.model, task.dataset)],
        "--cluster-config.batch-size",
        str(task.batch_size),
        "--cluster-config.local_scheduler",
        "static",
        "--placmt-config.placer-label",
        "bw",
        "--mapping-config.mapping_strategy",
        "static",
        "--time-limit",
        str(args.time_limit),
    ]


def output_has_metrics(run_dir: Path) -> bool:
    log_path = run_dir / "log_info.txt"
    if not log_path.is_file():
        return False
    text = log_path.read_text(encoding="utf-8", errors="replace")
    return all(pattern.search(text) for pattern in METRIC_PATTERNS)


def task_is_complete(task: SimulationTask) -> bool:
    if not task.output_dir.is_dir():
        return False
    return any(output_has_metrics(run_dir) for run_dir in task.output_dir.glob(f"*+{task.label}"))


def completed_labels(tasks: list[SimulationTask]) -> set[str]:
    expected_labels = {task.label for task in tasks}
    output_dirs = sorted({task.output_dir for task in tasks})
    completed: set[str] = set()
    for output_dir in output_dirs:
        if not output_dir.is_dir():
            continue
        for run_dir in output_dir.iterdir():
            if not run_dir.is_dir() or "+" not in run_dir.name:
                continue
            label = run_dir.name.split("+", 1)[1]
            if label in expected_labels and output_has_metrics(run_dir):
                completed.add(label)
    return completed


def run_one(task: SimulationTask, args: argparse.Namespace, env: dict[str, str]) -> dict[str, object]:
    task.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    try:
        completed = subprocess.run(
            command_for(task, args),
            cwd=REPO_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=args.command_timeout or None,
            check=False,
        )
        status = "ok" if completed.returncode == 0 else "failed"
        output = completed.stdout or ""
        return_code = completed.returncode
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout or ""
        if isinstance(output, bytes):
            output = output.decode("utf-8", errors="replace")
        status = "timeout"
        return_code = 124

    output_is_complete = all(pattern.search(output) for pattern in METRIC_PATTERNS)
    if status == "ok" and not output_is_complete and not task_is_complete(task):
        status = "incomplete"
    return {
        "label": task.label,
        "model": task.model,
        "dataset": task.dataset,
        "status": status,
        "return_code": return_code,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "output_tail": "\n".join(output.splitlines()[-20:]) if status != "ok" else "",
    }


def append_status(path: Path, result: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(result, sort_keys=True) + "\n")


def main() -> int:
    args = parse_args()
    if args.num_in_parallel < 1:
        raise ValueError("--num-in-parallel must be at least 1")
    if args.max_configs_per_workload < 0:
        raise ValueError("--max-configs-per-workload cannot be negative")

    tasks = build_tasks(args)
    by_workload: dict[str, int] = {}
    for task in tasks:
        key = f"{task.model}-{task.dataset}"
        by_workload[key] = by_workload.get(key, 0) + 1
    print(f"Generated {len(tasks)} simulation configurations:")
    for workload, count in sorted(by_workload.items()):
        print(f"  {workload}: {count}")

    if args.dry_run:
        return 0

    python_bin = Path(args.python_bin)
    if not python_bin.is_file():
        raise FileNotFoundError(f"Python interpreter not found: {python_bin}")
    missing_traces = sorted(
        {
            TRACE_FILES[(task.model, task.dataset)]
            for task in tasks
            if not (REPO_ROOT / TRACE_FILES[(task.model, task.dataset)]).is_file()
        }
    )
    if missing_traces:
        raise FileNotFoundError("Missing trace files:\n" + "\n".join(missing_traces))

    args.output_root = args.output_root.resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    status_path = args.output_root / "run_status.jsonl"
    done = set() if args.rerun_completed else completed_labels(tasks)
    pending = [task for task in tasks if task.label not in done]
    print(f"Skipping {len(tasks) - len(pending)} completed configurations; launching {len(pending)}.")
    if not pending:
        return 0

    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", str(args.output_root / ".mplcache"))
    failures = 0
    with ThreadPoolExecutor(max_workers=min(args.num_in_parallel, len(pending))) as executor:
        futures = {executor.submit(run_one, task, args, env): task for task in pending}
        for completed_count, future in enumerate(as_completed(futures), 1):
            result = future.result()
            append_status(status_path, result)
            if result["status"] != "ok":
                failures += 1
            print(
                f"[{completed_count}/{len(pending)}] {result['label']} "
                f"{result['status']} ({result['elapsed_seconds']} s)",
                flush=True,
            )

    print(f"Finished {len(pending)} configurations with {failures} non-successful runs.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
