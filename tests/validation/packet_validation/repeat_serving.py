#!/usr/bin/env python3
"""Repeat a small serving case sequentially across all three network levels."""

import argparse
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from tests.validation.chipsim_validation.collect_checkpoints import CheckpointRun


class RepeatedServingValidation:
    def __init__(self, args):
        self.args = args
        self.output = args.output_dir.resolve()
        self.output.mkdir(parents=True, exist_ok=True)

    def run(self):
        records = []
        reference = None
        for repeat in range(self.args.repeats):
            for backend in ('hydra_packet', 'chipsim_contended'):
                directory = self.output / f'repeat_{repeat + 1}' / backend
                command = [sys.executable, str(ROOT / 'tests/validation/chipsim_validation/run_comparison.py'),
                           '--detailed-backend', backend, '--models', 'NEMO', '--points', '1',
                           '--requests', '4', '--time-limit', str(self.args.time_limit),
                           '--workers', '1', '--output-dir', str(directory), '--resume']
                subprocess.run(command, cwd=ROOT, check=True)
                for name in ('hydra_sim', backend):
                    run = CheckpointRun(directory / 'NEMO-CHAT-P1' / name)
                    if reference is not None:
                        reference.verify_inputs(run)
                    reference = run
                    metric = run.read('metrics.json')
                    records.append(dict(repeat=repeat + 1, backend=name,
                                        directory=str(run.directory.relative_to(self.output)),
                                        wall_seconds=json.loads((run.directory / 'wall_time.json').read_text())['seconds'],
                                        metrics={k: metric[k] for k in ('tokens_per_sec', 'ttft_s', 'ttft_samples')}))
                (self.output / 'records.json').write_text(json.dumps(records, indent=2))
        summary = {'simulated_seconds': self.args.time_limit, 'requests': 4,
                   'platform': platform.platform(), 'cpu_count': os.cpu_count(),
                   'scope': 'Sequential child processes; includes startup/output; other host workloads may remain active.',
                   'backends': {}}
        for backend in ('hydra_sim', 'hydra_packet', 'chipsim_contended'):
            panel = [r for r in records if r['backend'] == backend]
            times = [r['wall_seconds'] for r in panel]
            summary['backends'][backend] = dict(
                runs=len(panel), deterministic_metrics=all(r['metrics'] == panel[0]['metrics'] for r in panel),
                metrics=panel[0]['metrics'], wall_seconds_median=statistics.median(times),
                wall_seconds_min=min(times), wall_seconds_max=max(times))
        (self.output / 'summary.json').write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2), flush=True)
        if not all(r['deterministic_metrics'] for r in summary['backends'].values()):
            raise RuntimeError('Repeated fixed-input runs produced different serving metrics.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--time-limit', type=float, default=0.25)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 2 or args.time_limit <= 0:
        parser.error('Use at least two repetitions and a positive simulation window.')
    RepeatedServingValidation(args).run()
