#!/usr/bin/env python3
"""Run the optional hybrid-LLM DSE profiles used by the VI-C study."""

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


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "output_temp_runs" / "vi_c_hybrid_llm_dse"
NUM_IN_PARALLEL = 32

MODEL_CONFIG = {
    "jamba": {"model": "jamba-tiny", "scene": "JAMBA-TINY-CHAT", "placer": "rr"},
    "zamba": {"model": "zamba2-7b", "scene": "ZAMBA2-CHAT", "placer": "rr"},
    "nemo": {"model": "nemotronh-4b", "scene": "NEMO-CHAT", "placer": "bw"},
}
TRACE_FILE = "dataset/chat/chat1m_stats_nemo.csv"
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
    batch_size: int
    bandwidth: int
    architecture: Architecture
    output_dir: Path

    @property
    def label(self) -> str:
        spec = MODEL_CONFIG[self.model]
        a = self.architecture
        return (
            f"{spec['scene']}_BS{self.batch_size}_BMPD{self.bandwidth}_{a.num_m}_"
            f"{a.num_ap}_{a.num_mp}_{a.num_ad}_{a.num_md}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the optional hybrid-LLM Chat profiles for the Section VI-C study."
    )
    parser.add_argument("--models", nargs="+", choices=MODEL_CONFIG, default=list(MODEL_CONFIG))
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[2, 4, 8, 16, 32, 64])
    parser.add_argument("--bandwidths", nargs="+", type=int, default=[256, 384, 512, 640])
    parser.add_argument(
        "--memory-chiplets",
        nargs="+",
        type=int,
        default=None,
        help="Restrict even HBM3 counts; the complete range is used by default.",
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
        "--max-configs-per-model",
        type=int,
        default=0,
        help="Run only the first N configurations per model; 0 uses the complete selected space.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--rerun-completed", action="store_true")
    return parser.parse_args()


def architectures(memory_counts: list[int] | None) -> list[Architecture]:
    selected = sorted(set(memory_counts)) if memory_counts is not None else list(range(2, 17, 2))
    if any(count < 2 or count > 16 or count % 2 for count in selected):
        raise ValueError("HBM3 counts must be even values between 2 and 16.")

    result: list[Architecture] = []
    for num_m in selected:
        remain = 24 - num_m
        for num_ap in range(1, remain - 2, 2):
            for num_ad in range(1, remain - num_ap - 1, 2):
                for num_mp in range(1, remain - num_ap - num_ad, 2):
                    result.append(
                        Architecture(
                            num_m=num_m,
                            num_mp=num_mp,
                            num_md=remain - num_ap - num_ad - num_mp,
                            num_ap=num_ap,
                            num_ad=num_ad,
                        )
                    )
    return result


def build_tasks(args: argparse.Namespace) -> list[SimulationTask]:
    architecture_space = architectures(args.memory_chiplets)
    run_root = args.output_root.resolve() / "hybrid_llm_dse_runs"
    tasks: list[SimulationTask] = []
    for model in args.models:
        model_tasks = [
            SimulationTask(
                model=model,
                batch_size=batch_size,
                bandwidth=bandwidth,
                architecture=architecture,
                output_dir=run_root / model,
            )
            for batch_size in args.batch_sizes
            for bandwidth in args.bandwidths
            for architecture in architecture_space
        ]
        if args.max_configs_per_model > 0:
            model_tasks = model_tasks[: args.max_configs_per_model]
        tasks.extend(model_tasks)
    return tasks


def command_for(task: SimulationTask, args: argparse.Namespace) -> list[str]:
    spec = MODEL_CONFIG[task.model]
    a = task.architecture
    return [
        args.python_bin,
        str(REPO_ROOT / "main.py"),
        "--workload-config.model",
        spec["model"],
        "--workload-config.dataset",
        "chat",
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
        TRACE_FILE,
        "--cluster-config.batch-size",
        str(task.batch_size),
        "--cluster-config.local_scheduler",
        "static",
        "--placmt-config.placer-label",
        spec["placer"],
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
    return any(output_has_metrics(path) for path in task.output_dir.glob(f"*+{task.label}"))


def completed_labels(tasks: list[SimulationTask]) -> set[str]:
    expected = {task.label for task in tasks}
    completed: set[str] = set()
    for output_dir in sorted({task.output_dir for task in tasks}):
        if not output_dir.is_dir():
            continue
        for run_dir in output_dir.iterdir():
            if not run_dir.is_dir() or "+" not in run_dir.name:
                continue
            label = run_dir.name.split("+", 1)[1]
            if label in expected and output_has_metrics(run_dir):
                completed.add(label)
    return completed


def run_one(task: SimulationTask, args: argparse.Namespace, env: dict[str, str]) -> dict[str, object]:
    task.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    try:
        process = subprocess.run(
            command_for(task, args),
            cwd=REPO_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=args.command_timeout or None,
            check=False,
        )
        output = process.stdout or ""
        status = "ok" if process.returncode == 0 else "failed"
        return_code = process.returncode
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout or ""
        if isinstance(output, bytes):
            output = output.decode("utf-8", errors="replace")
        status = "timeout"
        return_code = 124

    output_complete = all(pattern.search(output) for pattern in METRIC_PATTERNS)
    if status == "ok" and not output_complete and not task_is_complete(task):
        status = "incomplete"
    return {
        "label": task.label,
        "model": task.model,
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
        raise ValueError("--num-in-parallel must be at least 1.")
    if args.max_configs_per_model < 0:
        raise ValueError("--max-configs-per-model cannot be negative.")

    tasks = build_tasks(args)
    print("Generated simulation configurations:")
    for model in args.models:
        print(f"  {model}: {sum(task.model == model for task in tasks)}")
    if args.dry_run:
        return 0

    if not Path(args.python_bin).is_file():
        raise FileNotFoundError(f"Python interpreter not found: {args.python_bin}")
    if not (REPO_ROOT / TRACE_FILE).is_file():
        raise FileNotFoundError(f"Trace file not found: {TRACE_FILE}")

    args.output_root = args.output_root.resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    done = set() if args.rerun_completed else completed_labels(tasks)
    pending = [task for task in tasks if task.label not in done]
    print(f"Skipping {len(tasks) - len(pending)} completed configurations; launching {len(pending)}.")
    if not pending:
        return 0

    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", str(args.output_root / ".mplcache"))
    status_path = args.output_root / "run_status.jsonl"
    failures = 0
    with ThreadPoolExecutor(max_workers=min(args.num_in_parallel, len(pending))) as executor:
        futures = {executor.submit(run_one, task, args, env): task for task in pending}
        for index, future in enumerate(as_completed(futures), 1):
            result = future.result()
            append_status(status_path, result)
            failures += result["status"] != "ok"
            print(
                f"[{index}/{len(pending)}] {result['label']} {result['status']} "
                f"({result['elapsed_seconds']} s)",
                flush=True,
            )
    print(f"Finished with {failures} non-successful runs.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
