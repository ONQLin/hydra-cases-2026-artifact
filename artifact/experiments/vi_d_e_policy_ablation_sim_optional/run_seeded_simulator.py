#!/usr/bin/env python3
"""Run a HYDRA simulator tree with deterministic placement support."""

from __future__ import annotations

import argparse
import os
import random
import runpy
import sys
from pathlib import Path


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulator-root", type=Path, required=True)
    parser.add_argument("--placement-seed", type=int, required=True)
    args, simulator_args = parser.parse_known_args()
    if simulator_args and simulator_args[0] == "--":
        simulator_args = simulator_args[1:]
    return args, simulator_args


def main() -> None:
    args, simulator_args = parse_args()
    simulator_root = args.simulator_root.resolve()
    entrypoint = simulator_root / "main.py"
    if not entrypoint.is_file():
        raise FileNotFoundError(f"No simulator entry point at {entrypoint}")
    os.chdir(simulator_root)
    sys.path.insert(0, str(simulator_root))
    random.seed(args.placement_seed)

    sys.argv = [str(entrypoint), *simulator_args]
    runpy.run_path(str(entrypoint), run_name="__main__")


if __name__ == "__main__":
    main()
