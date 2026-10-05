"""Convert recorded benchmark results: python -m integrations.workloads.prepare."""

import argparse
import json
from pathlib import Path

from integrations.workloads import BaseWorkloadImporter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark', choices=['bfcl'], default='bfcl')
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--provenance', type=Path, required=True)
    delays = parser.add_mutually_exclusive_group(required=True)
    delays.add_argument('--tool-latency-s', type=float)
    delays.add_argument('--tool-latencies', type=Path)
    parser.add_argument('--session-interval-s', type=float, default=0.01)
    parser.add_argument('--user-think-s', type=float, default=0.0)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Choose a new output path to preserve previous traces.')
    importer = BaseWorkloadImporter.create_from_name(args.benchmark)(
        json.loads(args.provenance.read_text()), tool_latency_s=args.tool_latency_s,
        tool_latencies=json.loads(args.tool_latencies.read_text()) if args.tool_latencies else None,
        session_interval_s=args.session_interval_s, user_think_s=args.user_think_s)
    workload = importer.load(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    workload.write(args.output)
    print(f'Wrote {len(workload.sessions)} sessions and {sum(len(s.steps) for s in workload.sessions)} LLM calls to {args.output}')


if __name__ == '__main__':
    main()
