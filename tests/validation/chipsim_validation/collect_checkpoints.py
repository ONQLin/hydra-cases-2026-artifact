#!/usr/bin/env python3
"""Collect paired serving checkpoints from running or completed comparisons."""

import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import re
import time
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from tests.validation.chipsim_validation.run_comparison import ParetoComparison


@dataclass
class CheckpointRun:
    directory: Path

    @property
    def run_dir(self):
        paths = list(self.directory.glob('*/system_snapshot.json'))
        if len(paths) > 1:
            raise ValueError(f'Ambiguous runs in {self.directory}')
        if paths and (paths[0].parent / 'effective_workload.csv').exists():
            return paths[0].parent
        return None

    def read(self, relative_path):
        root = self.run_dir
        path = root / relative_path if root else None
        return json.loads(path.read_text()) if path and path.exists() else None

    def checkpoint(self, seconds):
        return self.read(f'checkpoints/{seconds:g}s.json')

    def status(self):
        final = self.read('metrics.json')
        network = self.read('chipsim_runtime/network_progress.json') or self.read('packet_runtime/network_progress.json')
        serving = self.read('progress.json')
        return {'finished': final is not None, 'network': network, 'serving': serving}

    def verify_inputs(self, other):
        for name in ('system_snapshot.json', 'effective_workload.csv'):
            if (self.run_dir / name).read_bytes() != (other.run_dir / name).read_bytes():
                raise ValueError(f'Backends disagree on {name}: {self.directory.parent.name}')
        # Placement and token lengths alone cannot detect different arrival
        # intervals, batch policies, seeds, or accelerator configurations.
        a, b = self.read('config.json'), other.read('config.json')
        if a is None and b is None:
            return  # Legacy fixtures may contain only the original snapshots.
        if a is None or b is None:
            raise ValueError('One run is missing its effective configuration.')
        for key in ('seed', 'cluster_config', 'arch_config', 'chips_config',
                    'placmt_config', 'workload_config', 'mapping_config'):
            left, right = (json.dumps(cfg[key], sort_keys=True) for cfg in (a, b))
            # Existing config export stringifies a few Python object instances.
            left, right = (re.sub(r' at 0x[0-9a-fA-F]+', ' at <address>', value)
                           for value in (left, right))
            if left != right:
                raise ValueError(f'Backends disagree on {key}: {self.directory.parent.name}')


class CheckpointComparison:
    def __init__(self, directory, windows, backend='chipsim_contended'):
        self.directory = directory.resolve()
        self.windows = windows
        self.backend = backend
        self.output_dir = self.directory / 'checkpoint_comparison'
        self.output_dir.mkdir(exist_ok=True)
        self.manifest = json.loads((self.directory / 'selected_points.json').read_text())

    def collect(self):
        rows = {seconds: [] for seconds in self.windows}
        statuses = {}
        for point in self.manifest:
            label = point['label']
            native = CheckpointRun(self.directory / label / 'hydra_sim')
            detailed = CheckpointRun(self.directory / label / self.backend)
            statuses[label] = {'hydra_sim': native.status(), self.backend: detailed.status()}
            if native.run_dir is None or detailed.run_dir is None:
                continue
            native.verify_inputs(detailed)
            for seconds in self.windows:
                a, b = native.checkpoint(seconds), detailed.checkpoint(seconds)
                if a is None or b is None:
                    continue
                if a['elapsed_simulation_s'] != seconds or b['elapsed_simulation_s'] != seconds:
                    raise ValueError('Checkpoint time does not match the requested window')
                model, dataset, _ = label.split('-')
                row = {'point': label, 'model': model, 'dataset': dataset, 'seconds': seconds}
                for backend, metrics in [('hydra_sim', a), (self.backend, b)]:
                    for target, source in [('tp', 'tokens_per_sec'), ('ttft_s', 'ttft_s'),
                                           ('ttft_samples', 'ttft_samples'),
                                           ('last_second_tp', 'interval_tokens_per_sec'),
                                           ('completed_requests', 'completed_requests'),
                                           ('generated_requests', 'generated_requests'),
                                           ('pending_requests', 'pending_requests'),
                                           ('running_requests', 'running_requests')]:
                        row[f'{backend}_{target}'] = metrics[source]
                row['tp_relative_error'] = b['tokens_per_sec'] / a['tokens_per_sec'] - 1 if a['tokens_per_sec'] else None
                row['ttft_relative_error'] = b['ttft_s'] / a['ttft_s'] - 1 if a['ttft_s'] and b['ttft_s'] is not None else None
                rows[seconds].append(row)
        temporary = self.output_dir / 'status.tmp'
        temporary.write_text(json.dumps(statuses, indent=2))
        temporary.replace(self.output_dir / 'status.json')
        for seconds, panel in rows.items():
            self.save_window(seconds, panel)
        return all(s[b]['finished'] for s in statuses.values() for b in ('hydra_sim', self.backend))

    def save_window(self, seconds, rows):
        directory = self.output_dir / f'{seconds:g}s'
        directory.mkdir(exist_ok=True)
        if rows:
            with (directory / 'comparison.csv').open('w') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        summary = {'simulated_seconds': seconds, 'paired_points': len(rows),
                   'expected_points': len(self.manifest),
                   'missing_points': [p['label'] for p in self.manifest if p['label'] not in {r['point'] for r in rows}],
                   'scope': 'Cumulative transient metrics from t=0; TTFT includes only requests with a first token by cutoff.',
                   'groups': {}}
        for model, dataset in sorted({(r['model'], r['dataset']) for r in rows}):
            panel = [r for r in rows if (r['model'], r['dataset']) == (model, dataset)]
            metrics = {}
            for metric, error in [('tp', 'tp_relative_error'), ('ttft_s', 'ttft_relative_error')]:
                pairs = []
                for index, first in enumerate(panel):
                    for second in panel[index + 1:]:
                        values = [r[b + '_' + metric] for r in (first, second) for b in ('hydra_sim', self.backend)]
                        if any(v is None for v in values):
                            continue
                        native_delta, detailed_delta = values[0] - values[2], values[1] - values[3]
                        if native_delta:
                            pairs.append(native_delta * detailed_delta > 0)
                errors = [abs(r[error]) for r in panel if r[error] is not None]
                metrics[metric] = {'max_absolute_relative_error': max(errors) if errors else None,
                                   'matching_pairwise_orderings': sum(pairs),
                                   'comparable_pairwise_orderings': len(pairs)}
            summary['groups'][f'{model}-{dataset}'] = metrics
        (directory / 'summary.json').write_text(json.dumps(summary, indent=2))
        plottable = [r for r in rows if r['hydra_sim_ttft_s'] is not None and r[self.backend + '_ttft_s'] is not None]
        if plottable:
            command = json.loads((self.directory / rows[0]['point'] / 'hydra_sim/command.json').read_text())
            requests = int(command[command.index('--workload-config.request-generator-config.num-requests') + 1])
            args = SimpleNamespace(output_dir=directory, detailed_backend=self.backend,
                                   models=sorted({r['model'] for r in plottable}),
                                   datasets=sorted({r['dataset'] for r in plottable}),
                                   time_limit=seconds, requests=requests)
            ParetoComparison(args).plot(plottable)

    def run(self, interval, timeout):
        start = time.monotonic()
        while True:
            finished = self.collect()
            print(f'Updated {self.output_dir}; all simulations finished: {finished}', flush=True)
            if finished or not interval or time.monotonic() - start >= timeout:
                return
            time.sleep(interval)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--detailed-backend', choices=['chipsim_contended', 'hydra_packet'], default='chipsim_contended')
    parser.add_argument('--windows', type=float, nargs='+', default=[1, 2, 5])
    parser.add_argument('--watch-seconds', type=float, default=0)
    parser.add_argument('--timeout', type=float, default=86400)
    args = parser.parse_args()
    if args.watch_seconds < 0 or args.timeout <= 0 or any(w <= 0 for w in args.windows):
        parser.error('Windows and timeout must be positive; watch interval must be nonnegative.')
    CheckpointComparison(args.directory, args.windows, args.detailed_backend).run(args.watch_seconds, args.timeout)
