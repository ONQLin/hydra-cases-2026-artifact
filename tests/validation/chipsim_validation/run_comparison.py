#!/usr/bin/env python3
"""Replay a few existing Pareto points with HYDRA and CHIPSIM feedback."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class ParetoPoint:
    model: str
    dataset: str
    index: int
    source: str
    row: dict

    @property
    def label(self):
        return f'{self.model}-{self.dataset}-P{self.index}'

    def command(self, backend, output_dir, args):
        model_names = {'LLAMA3': 'llama3-8b', 'NEMO': 'nemotronh-4b'}
        trace_names = {'LLAMA3': 'llama3', 'NEMO': 'nemo'}
        trace = f'dataset/chat/chat1m_stats_{trace_names[self.model]}.csv'
        if self.dataset == 'BWB':
            trace = f'dataset/bwb/bwb_translation_stats_{trace_names[self.model]}.csv'
        command = [sys.executable, str(REPO_ROOT / 'main.py'),
                   '--simulator-backend', backend,
                   '--workload-config.model', model_names[self.model],
                   '--workload-config.dataset', self.dataset.lower(),
                   '--workload-config.request-generator-config.trace-length-generator-config.trace-file', trace,
                   '--workload-config.request-generator-config.num-requests', str(args.requests),
                   '--mapping-config.mapping-strategy', 'static',
                   '--cluster-config.local-scheduler', 'static',
                   '--placmt-config.placer-label', 'bw',
                   '--cluster-config.batch-size', str(int(self.row['batchsize'])),
                   '--chips-config.D2D-NoI-bw', str(int(self.row['NoI_bw(GBps)'])),
                   '--time-limit', str(args.time_limit),
                   '--metrics-config.output-dir', str(output_dir),
                   '--metrics-config.label-name', self.label,
                   '--chipsim-config.virtual-channels-per-vnet', str(args.virtual_channels),
                   '--chipsim-config.timeout-seconds', str(args.transfer_timeout)]
        for name, column in [('marca_p', 'Num Mp'), ('marca_d', 'Num Md'),
                             ('tscs_p', 'Num Ap'), ('tscs_d', 'Num Ad'), ('HBM3', 'Num M')]:
            command.extend([f'--placmt-config.chiplet-alloc.{name}', str(int(self.row[column]))])
        if args.dma_pacing and backend in ('chipsim_contended', 'hydra_packet'):
            command.extend(['--chipsim-config.dma-pacing', '--chipsim-config.dma-burst-bytes', str(args.dma_burst_bytes)])
        if backend == 'hydra_packet':
            command.extend(['--packet-config.quantum-bytes', str(args.packet_quantum_bytes),
                            '--packet-config.buffer-bytes', str(args.packet_buffer_bytes)])
        return command


class ParetoComparison:
    def __init__(self, args):
        self.args = args
        self.detailed_backend = args.detailed_backend
        self.output_dir = args.output_dir.resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def runtime_directory(self):
        return 'packet_runtime' if self.detailed_backend == 'hydra_packet' else 'chipsim_runtime'

    def select_points(self):
        points = []
        for model in self.args.models:
            for dataset in self.args.datasets:
                source = Path('artifact/run_outputs/vi_b_dse_on_macro_architectures/raw_data/Pareto_Reports') / f'pareto_frontier_results_summary_{model}-{dataset}_bw_static_static.csv'
                with (REPO_ROOT / source).open() as stream:
                    rows = [r for r in csv.DictReader(stream) if int(r['batchsize']) == self.args.batch_size]
                rows.sort(key=lambda r: float(r['TTFT(s)']))
                if len(rows) < self.args.points:
                    raise ValueError(f'Not enough Pareto points in {source}.')
                indices = [round(i * (len(rows) - 1) / max(self.args.points - 1, 1))
                           for i in range(self.args.points)]
                for number, index in enumerate(indices, 1):
                    points.append(ParetoPoint(model, dataset, number, str(source), rows[index]))
        manifest = [{'label': p.label, 'source': p.source, 'row': p.row} for p in points]
        manifest_path = self.output_dir / 'selected_points.json'
        if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
            raise ValueError('Output contains a different selection; use a new --output-dir.')
        manifest_path.write_text(json.dumps(manifest, indent=2))
        return points

    def run_point(self, point):
        results = {}
        for backend in ('hydra_sim', self.detailed_backend):
            directory = self.output_dir / point.label / backend
            directory.mkdir(parents=True, exist_ok=True)
            command = point.command(backend, directory, self.args)
            command_file = directory / 'command.json'
            previous = list(directory.glob('*/metrics.json'))
            if self.args.resume and previous and command_file.exists() and json.loads(command_file.read_text()) == command:
                metric_path = max(previous, key=lambda p: p.stat().st_mtime_ns)
            else:
                if previous:
                    raise ValueError(f'{directory} already has results; use --resume or a new output directory.')
                if any(directory.iterdir()):
                    raise ValueError(f'{directory} contains an incomplete run; use a new output directory to preserve its logs and checkpoints.')
                command_file.write_text(json.dumps(command, indent=2))
                start = time.monotonic()
                with (directory / 'console.log').open('w') as log:
                    process = subprocess.Popen(command, cwd=REPO_ROOT, stdout=log,
                                               stderr=subprocess.STDOUT, start_new_session=True)
                    try:
                        returncode = process.wait(timeout=self.args.run_timeout)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGTERM)
                        process.wait()
                        raise TimeoutError(f'{point.label} {backend} timed out; see {directory}.')
                if returncode:
                    raise RuntimeError(f'{point.label} {backend} failed; see {directory / "console.log"}.')
                paths = list(directory.glob('*/metrics.json'))
                if len(paths) != 1:
                    raise RuntimeError(f'Expected one metrics.json in {directory}.')
                metric_path = paths[0]
                (directory / 'wall_time.json').write_text(json.dumps({'seconds': time.monotonic() - start}))
            result = json.loads(metric_path.read_text())
            if result['ttft_s'] is None or result['tokens_per_sec'] <= 0:
                raise RuntimeError(f'No serving metrics for {point.label} {backend}; increase --time-limit.')
            result['metrics_path'] = str(metric_path.relative_to(self.output_dir))
            timing = directory / 'wall_time.json'
            result['wall_seconds'] = json.loads(timing.read_text())['seconds'] if timing.exists() else None
            results[backend] = result
            print(f'{point.label} {backend}: TP={result["tokens_per_sec"]:.2f}, TTFT={result["ttft_s"]:.6f}s', flush=True)
        native = self.output_dir / results['hydra_sim']['metrics_path']
        detailed = self.output_dir / results[self.detailed_backend]['metrics_path']
        for filename in ('system_snapshot.json', 'effective_workload.csv'):
            a = (native.parent / filename).read_bytes()
            b = (detailed.parent / filename).read_bytes()
            if a != b:
                raise RuntimeError(f'{point.label}: backends disagree on {filename}.')
        row = {'point': point.label, 'model': point.model, 'dataset': point.dataset,
               'setup_sha256': hashlib.sha256((native.parent / 'system_snapshot.json').read_bytes()).hexdigest(),
               'workload_sha256': hashlib.sha256((native.parent / 'effective_workload.csv').read_bytes()).hexdigest()}
        for backend, result in results.items():
            row[backend + '_wall_seconds'] = result['wall_seconds']
            row[backend + '_tp'] = result['tokens_per_sec']
            row[backend + '_ttft_s'] = result['ttft_s']
            row[backend + '_ttft_samples'] = result['ttft_samples']
            row[backend + '_metrics_path'] = result['metrics_path']
        row['tp_relative_error'] = row[self.detailed_backend + '_tp'] / row['hydra_sim_tp'] - 1
        row['ttft_relative_error'] = row[self.detailed_backend + '_ttft_s'] / row['hydra_sim_ttft_s'] - 1
        return row

    def plot(self, rows):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        groups = [(model, dataset) for model in self.args.models for dataset in self.args.datasets]
        fig, axes = plt.subplots(1, len(groups), figsize=(6 * len(groups), 4.5), squeeze=False)
        for axis, (model, dataset) in zip(axes[0], groups):
            panel = sorted([r for r in rows if r['model'] == model and r['dataset'] == dataset], key=lambda r: r['point'])
            for backend, label, style in [('hydra_sim', 'HYDRA-Analytic', 'o-'),
                                           (self.detailed_backend, 'HYDRA-Packet' if self.detailed_backend == 'hydra_packet' else 'CHIPSIM', 's--')]:
                xs = [r[backend + '_ttft_s'] for r in panel]
                ys = [r[backend + '_tp'] for r in panel]
                axis.plot(xs, ys, style, label=label, markersize=7,
                          markerfacecolor='none' if backend == self.detailed_backend else None)
                if backend == 'hydra_sim':
                    for row, x, y in zip(panel, xs, ys):
                        axis.annotate(row['point'].split('-')[-1], (x, y), xytext=(4, 6), textcoords='offset points')
            axis.set(xlabel='Mean TTFT (s)', ylabel='Output tokens/s', title=f'{model} / {dataset}')
            axis.margins(x=0.08, y=0.12)
            if len(panel) == 1:
                # A smoke run has one point; avoid magnifying sub-percent
                # differences into the entire horizontal axis.
                axis.set_xlim(0, max(panel[0][b + '_ttft_s'] for b in ('hydra_sim', self.detailed_backend)) * 1.15)
                axis.set_ylim(0, max(panel[0][b + '_tp'] for b in ('hydra_sim', self.detailed_backend)) * 1.25)
            axis.grid(alpha=0.25)
            axis.legend(fontsize=9)
        fig.suptitle(f'Static batching / static mapping; {self.args.time_limit}s window from t=0')
        network_label = {'hydra_packet': 'C++ aggregated packet network', 'chipsim_contended': 'persistent shared Garnet network', 'chipsim_runtime': 'isolated Garnet bursts'}[self.detailed_backend]
        fig.text(0.5, 0.015, f'{self.args.requests} trace requests; shared runtime and compute; {network_label}',
                 ha='center', fontsize=9)
        fig.tight_layout(rect=(0, 0.05, 1, 1))
        for extension in ('png', 'pdf'):
            fig.savefig(self.output_dir / f'tp_vs_ttft.{extension}', dpi=180)
        plt.close(fig)

    def summarize(self, rows):
        groups = {}
        for model in self.args.models:
            for dataset in self.args.datasets:
                panel = [r for r in rows if r['model'] == model and r['dataset'] == dataset]
                errors = {}
                for metric, error_column in [('tp', 'tp_relative_error'),
                                              ('ttft_s', 'ttft_relative_error')]:
                    comparable = 0
                    matching = 0
                    for i, first in enumerate(panel):
                        for second in panel[i + 1:]:
                            native = first['hydra_sim_' + metric] - second['hydra_sim_' + metric]
                            detailed = first[self.detailed_backend + '_' + metric] - second[self.detailed_backend + '_' + metric]
                            if native != 0:
                                comparable += 1
                                matching += int(native * detailed > 0)
                    errors[metric] = {
                        'max_absolute_relative_error': max(abs(r[error_column]) for r in panel),
                        'matching_pairwise_orderings': matching,
                        'comparable_pairwise_orderings': comparable,
                    }
                groups[f'{model}-{dataset}'] = errors
        summary = {
            'simulated_seconds': self.args.time_limit,
            'trace_requests': self.args.requests,
            'window': 'transient window starting at t=0; no warmup exclusion',
            'detailed_backend': self.detailed_backend,
            'packet_quantum_bytes': self.args.packet_quantum_bytes if self.detailed_backend == 'hydra_packet' else None,
            'packet_buffer_bytes': self.args.packet_buffer_bytes if self.detailed_backend == 'hydra_packet' else None,
            'dma_pacing': self.args.dma_pacing,
            'dma_burst_bytes': self.args.dma_burst_bytes if self.args.dma_pacing else None,
            'fidelity_scope': ('shared HYDRA runtime/compute with persistent network contention'
                               if self.detailed_backend in ('chipsim_contended', 'hydra_packet')
                               else 'shared HYDRA runtime/compute with isolated Garnet burst feedback'),
            'groups': groups,
            'phase_diagnostics': self.phase_diagnostics(rows),
            'network_feedback': self.audit_network(rows),
        }
        (self.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2))

    def phase_diagnostics(self, rows):
        diagnostics = {}
        for row in rows:
            metrics = self.output_dir / row[self.detailed_backend + '_metrics_path']
            counts = {'compute': 0, 'hbm': 0, 'noi': 0}
            extra_noi_us = 0.0
            with (metrics.parent / self.runtime_directory / 'execution.jsonl').open() as stream:
                for line in stream:
                    result = json.loads(line)['result']
                    for phase in result.get('phases', []):
                        times = {name: phase[name + '_us'] for name in counts}
                        if max(times.values()) > 0:
                            counts[max(times, key=times.get)] += 1
                        extra_noi_us += max(0, times['noi'] - max(times['compute'], times['hbm']))
            diagnostics[row['point']] = {
                'dominant_component_phase_counts': counts,
                'sum_noi_extension_us': extra_noi_us,
                'scope': ('completed block/transfer records; unfinished blocks excluded; not elapsed serving time'
                          if self.detailed_backend in ('chipsim_contended', 'hydra_packet') else
                          'admitted phases, including work in flight at cutoff; not elapsed serving time'),
            }
        return diagnostics

    def audit_network(self, rows):
        if self.detailed_backend not in ('chipsim_contended', 'hydra_packet'):
            return {}
        diagnostics = {}
        for row in rows:
            metrics = self.output_dir / row[self.detailed_backend + '_metrics_path']
            directory = metrics.parent / self.runtime_directory
            pending = {}
            submitted = set()
            completed = set()
            previous_tick = 0
            peak_active = 0
            with (directory / 'execution.jsonl').open() as stream:
                for line in stream:
                    record = json.loads(line)
                    request, result = record['request'], record['result']
                    if request['kind'] == 'submit':
                        flow = request['flow_id']
                        if flow in submitted or result['tick'] < previous_tick:
                            raise RuntimeError('Duplicate submission or reversed network time.')
                        submitted.add(flow)
                        pending[flow] = result['tick']
                        peak_active = max(peak_active, len(pending))
                    elif request['kind'] == 'advance':
                        if not previous_tick <= result['tick'] <= request['target_tick']:
                            raise RuntimeError('Invalid network advancement in trace.')
                        previous_tick = result['tick']
                        values = result['completed']
                        if len(values) % 2:
                            raise RuntimeError('Malformed completion trace.')
                        for flow, tick in zip(values[::2], values[1::2]):
                            if flow not in pending or not pending[flow] <= tick <= previous_tick:
                                raise RuntimeError('Duplicate, unknown, or acausal network completion.')
                            del pending[flow]
                            completed.add(flow)
            summary = json.loads((directory / 'network_summary.json').read_text())
            if (summary['submitted_flows'], summary['completed_flows'], summary['inflight_at_cutoff'], summary['peak_active_flows']) != (len(submitted), len(completed), len(pending), peak_active):
                raise RuntimeError('Network summary disagrees with execution trace.')
            diagnostics[row['point']] = dict(summary, causal_trace_verified=True)
        return diagnostics

    def validate_options(self):
        if self.args.dma_pacing and self.detailed_backend not in ('chipsim_contended', 'hydra_packet'):
            raise ValueError('--dma-pacing requires --detailed-backend chipsim_contended or hydra_packet.')
        if self.args.dma_pacing and self.args.dma_burst_bytes <= 0:
            raise ValueError('--dma-burst-bytes must be positive.')
        if self.args.points < 1 or self.args.workers < 1 or self.args.batch_size < 1:
            raise ValueError('Points, workers, and batch size must be positive.')
        if self.args.time_limit <= 0 or self.args.requests < self.args.batch_size:
            raise ValueError('Use a positive time limit and at least one full batch.')

    def run(self):
        self.validate_options()
        points = self.select_points()
        rows = []
        with ThreadPoolExecutor(max_workers=self.args.workers) as executor:
            futures = [executor.submit(self.run_point, point) for point in points]
            for future in as_completed(futures):
                rows.append(future.result())
                rows.sort(key=lambda row: row['point'])
                with (self.output_dir / 'comparison.csv').open('w') as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
        self.plot(rows)
        self.summarize(rows)
        print(f'Comparison saved to {self.output_dir}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--detailed-backend', choices=['chipsim_runtime', 'chipsim_contended', 'hydra_packet'], default='chipsim_runtime')
    parser.add_argument('--packet-quantum-bytes', type=int, default=1024)
    parser.add_argument('--packet-buffer-bytes', type=int, default=16384)
    parser.add_argument('--dma-pacing', action='store_true')
    parser.add_argument('--dma-burst-bytes', type=int, default=4096)
    parser.add_argument('--models', nargs='+', choices=['LLAMA3', 'NEMO'], default=['LLAMA3', 'NEMO'])
    parser.add_argument('--datasets', nargs='+', choices=['CHAT', 'BWB'], default=['CHAT'])
    parser.add_argument('--points', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--requests', type=int, default=16)
    parser.add_argument('--time-limit', type=float, default=1)
    parser.add_argument('--virtual-channels', type=int, default=8)
    parser.add_argument('--transfer-timeout', type=int, default=120)
    parser.add_argument('--run-timeout', type=int, default=7200)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--output-dir', type=Path, default=REPO_ROOT / 'output_sanity_checks/chipsim_comparison')
    ParetoComparison(parser.parse_args()).run()
