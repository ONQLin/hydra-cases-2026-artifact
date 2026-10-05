#!/usr/bin/env python3
"""Replay explicit decoder models on the three shared HYDRA execution paths."""

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from tests.validation.chipsim_validation.collect_checkpoints import CheckpointRun
from tests.validation.chipsim_validation.run_comparison import ParetoComparison, ParetoPoint
from tests.validation.prepare_inputs import ValidationInputs


class ModernModelValidation:
    def __init__(self, args):
        self.args = args
        self.grid_override = None
        from Sim.config.package_config import PackageConfig
        self.package_config = PackageConfig(
            count=getattr(args, 'packages', 1),
            bandwidth_gbps=getattr(args, 'package_bandwidth_gbps', 25.0),
            latency_ns=getattr(args, 'package_latency_ns', 1000.0),
            efficiency=getattr(args, 'package_efficiency', 0.8))
        self.output_dir = args.output_dir.resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if args.trace is None:
            args.trace = ValidationInputs.write_trace(self.output_dir/'decoder_smoke.csv', requests=args.requests)

    def select_hardware(self):
        source = Path('artifact/run_outputs/vi_b_dse_on_macro_architectures/raw_data/Pareto_Reports/pareto_frontier_results_summary_NEMO-CHAT_bw_static_static.csv')
        with (REPO_ROOT / source).open() as stream:
            rows = [r for r in csv.DictReader(stream) if int(r['batchsize']) == 2]
        source_row = min(rows, key=lambda r: float(r['TTFT(s)']))
        row = dict(source_row)
        if self.args.memory_chiplets is not None:
            if self.args.memory_chiplets < 1:
                raise ValueError('Memory chiplet count must be positive.')
            row['Num M'] = str(self.args.memory_chiplets)
            count = sum(int(row[key]) for key in ('Num Mp','Num Md','Num Ap','Num Ad','Num M'))
            # The BW placer requires HBMs on the outer ring. Preserve that
            # constraint while choosing the least elongated feasible rectangle.
            height = max(d for d in range(1,math.isqrt(count)+1)
                         if count%d==0 and count-max(count//d-2,0)*max(d-2,0) >= self.args.memory_chiplets)
            self.grid_override = dict(nodes=count,width=count//height,height=height)
        # This is an existing Nemotron point, not an optimized Qwen design.
        point = ParetoPoint('NEMO', 'CHAT', 1, str(source), row)
        manifest = {'source': str(source), 'source_row': source_row, 'row': row,
                    'package_config': vars(self.package_config),
                    'hardware_overrides': {'memory_chiplets': self.args.memory_chiplets, 'grid': self.grid_override}, 'models': self.args.models,
                    'trace': str(self.args.trace.resolve()), 'dataset_label': self.args.dataset_label,
                    'backends': self.args.backends,
                    'arrival_interval_s': self.args.arrival_interval,
                    'require_complete': self.args.require_complete,
                    'trace_sha256': hashlib.sha256(self.args.trace.read_bytes()).hexdigest(),
                    'scope': 'Serving smoke; model metadata defines scope. Synthetic lengths are not tokenizer validation.',
                    'source_sha256': {str(path): hashlib.sha256((REPO_ROOT / path).read_bytes()).hexdigest()
                                      for path in (Path('Sim/config/modern_model_config.py'),
                                                   Path('analytic_profile/modern.py'),
                                                   Path('analytic_profile/attention.py'),
                                                   Path('analytic_profile/kda.py'),
                                                   Path('analytic_profile/mla.py'),
                                                   Path('analytic_profile/moe.py'),
                                                   Path('Sim/config/moe_config.py'),
                                                   Path('Sim/config/kimi_model_config.py'),
                                                   Path('Sim/config/deepseek_model_config.py'),
                                                   Path('Sim/config/sys_config.py'),
                                                   Path('Sim/config/package_config.py'),
                                                   Path('Sim/entities/package_system.py'),
                                                   Path('Sim/execution/package_network.py'),
                                                   Path('Sim/execution/BaseExecutionBackend.py'),
                                                   Path('Sim/simulator.py'),
                                                   Path('Sim/processing.py'),
                                                   Path('Sim/entities/static_mapper.py'),
                                                   Path('integrations/packet/network.py'),
                                                   Path('Sim/placer/bw_placer.py'),
                                                   Path('Sim/entities/expert_routing.py'),
                                                   Path('Sim/entities/operator_graph.py'),
                                                   Path('Sim/entities/execution.py'),
                                                   Path('Sim/entities/mem_sys.py'),
                                                   Path('Sim/scheduler/static.py'),
                                                   Path('Sim/execution/network.py'),
                                                   Path('Sim/execution/native.py'),
                                                   Path('integrations/chipsim/runtime_execution.py'),
                                                   Path('Sim/config/attention_operator_config.py'),
                                                   Path('Sim/config/operator_fixture_config.py'))}}

        path = self.output_dir / 'manifest.json'
        if path.exists():
            raise ValueError('Use a fresh output directory to preserve previous results.')
        path.write_text(json.dumps(manifest, indent=2)+'\n')
        return point

    def run_one(self, point, model, backend):
        directory = self.output_dir / model / backend
        directory.mkdir(parents=True)
        command = point.command(backend, directory, self.args)
        replacements = {'--workload-config.model': model,
                        '--workload-config.dataset': self.args.dataset_label,
                        '--cluster-config.batch-size': str(self.args.batch_size),
                        '--metrics-config.label-name': model,
                        '--workload-config.request-generator-config.trace-length-generator-config.trace-file': str(self.args.trace.resolve())}
        for key, value in replacements.items():
            command[command.index(key)+1] = value
        if self.grid_override:
            grid = self.grid_override
            for key,value in (('arch-config.num-nodes',grid['nodes']),
                              ('arch-config.intp-width',grid['width']),('arch-config.intp-height',grid['height']),
                              ('placmt-config.num-nodes',grid['nodes']),
                              ('placmt-config.int-width',grid['width']),('placmt-config.int-height',grid['height'])):
                command.extend(['--'+key,str(value)])
        command.extend(['--workload-config.request-generator-config.trace-interval-generator-config.interval',
                        str(self.args.arrival_interval)])
        for key, value in vars(self.package_config).items():
            command.extend(['--package-config.' + key.replace('_', '-'), str(value)])
        self.configure_command(command)
        (directory / 'command.json').write_text(json.dumps(command, indent=2)+'\n')
        started = time.monotonic()
        with (directory / 'console.log').open('w') as log:
            process = subprocess.Popen(command, cwd=REPO_ROOT, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = process.wait(timeout=self.args.run_timeout)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait()
                raise TimeoutError(f'{model} {backend} timed out; see {directory}.')
        elapsed = time.monotonic()-started
        (directory / 'wall_time.json').write_text(json.dumps({'seconds': elapsed})+'\n')
        if code:
            raise RuntimeError(f'{model} {backend} failed; see {directory / "console.log"}.')
        run = CheckpointRun(directory)
        result = run.read('metrics.json')
        if not result or result['ttft_samples'] == 0 or result['output_tokens'] == 0:
            raise RuntimeError(f'{model} {backend} produced no serving metrics; increase --time-limit.')
        if self.args.require_complete:
            self.verify_complete(run, result)
        if self.package_config.count > 1:
            package_summary = run.read('package_network_summary.json')
            if (not package_summary or package_summary['inflight_at_cutoff']
                    or not package_summary['completed_transfers']):
                raise RuntimeError('Package fabric did not drain or exercised no transfers.')
            result['package_network'] = package_summary
        if backend != 'hydra_sim':
            comparison = ParetoComparison(SimpleNamespace(**vars(self.args), detailed_backend=backend))
            row = {'point': model, backend+'_metrics_path': str((run.run_dir / 'metrics.json').relative_to(self.output_dir))}
            result['network_audit'] = comparison.audit_network([row])[model]
        result.update(model=model, backend=backend, process_wall_seconds=elapsed)
        print(f'{model} {backend}: TP={result["tokens_per_sec"]:.3f}, TTFT={result["ttft_s"]:.6f}s, wall={elapsed:.2f}s', flush=True)
        return run, result

    def configure_command(self, command):
        """Allow workload-specific runners to select a request source."""

    def verify_complete(self, run, result):
        with (run.run_dir / 'effective_workload.csv').open() as stream:
            effective = list(csv.DictReader(stream))
        with self.args.trace.open() as stream:
            source = list(csv.DictReader(stream))[:self.args.requests]
        fields = ('num_prefill_tokens', 'num_decode_tokens')
        lengths = lambda rows: [tuple(int(row[key]) for key in fields) for row in rows]
        if len(effective) != self.args.requests or lengths(effective) != lengths(source):
            raise RuntimeError('Strict replay requires all selected lengths unchanged; inspect effective_workload.csv.')
        expected_tokens = sum(int(row['num_decode_tokens'])+1 for row in effective)
        if (result['completed_requests'] != self.args.requests or result['running_requests']
                or result['pending_requests'] or result['output_tokens'] != expected_tokens):
            raise RuntimeError('Strict replay did not complete all requests/tokens; increase --time-limit.')

    def run(self):
        if (self.args.time_limit <= 0 or self.args.run_timeout <= 0 or self.args.batch_size < 1
                or self.args.requests < 1 or (self.args.requests < self.args.batch_size
                                             and not getattr(self, 'allows_partial_batches', False))):
            raise ValueError('Use positive time limits, a positive batch size, and at least one full batch.')
        if self.args.arrival_interval < 0:
            raise ValueError('Arrival interval must be nonnegative.')
        if not self.args.trace.is_file():
            raise FileNotFoundError(self.args.trace)
        if len(set(self.args.models)) != len(self.args.models) or len(set(self.args.backends)) != len(self.args.backends):
            raise ValueError('Model and backend selections must not contain duplicates.')
        point = self.select_hardware()
        results = []
        for model in self.args.models:
            reference = None
            for backend in self.args.backends:
                run, result = self.run_one(point, model, backend)
                if reference is None:
                    reference = run
                else:
                    reference.verify_inputs(run)
                results.append(result)
                (self.output_dir / 'summary.json').write_text(json.dumps(results, indent=2)+'\n')
        return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--models', nargs='+', choices=['qwen3-8b', 'qwen3.5-9b-text',
        'kda-mla-operator-fixture', 'kda-chunk-mla-operator-fixture',
        'kimi-decoder-fixture', 'kimi-linear-48b-a3b-text',
        'kimi-decoder-streaming-fixture', 'kimi-linear-48b-a3b-text-streaming',
        'deepseek-decoder-fixture', 'deepseek-v3-text'], default=['qwen3-8b', 'qwen3.5-9b-text'])
    parser.add_argument('--backends', nargs='+', choices=['hydra_sim', 'hydra_packet', 'chipsim_contended'], default=['hydra_sim', 'hydra_packet', 'chipsim_contended'])
    parser.add_argument('--dataset-label', default='chat', help='Dataset label recorded with the explicit trace.')
    parser.add_argument('--trace', type=Path, help='CSV input; omitted generates a short synthetic trace in the output directory.')
    parser.add_argument('--requests', type=int, default=2)
    parser.add_argument('--memory-chiplets', type=int,
                        help='Explicit HBM-count override for capacity studies; recorded separately from the archived point.')
    parser.add_argument('--packages', type=int, default=1,
                        help='Replicate the selected package and partition one decoder across the copies.')
    parser.add_argument('--package-bandwidth-gbps', type=float, default=25.0,
                        help='Decimal GB/s per package per direction, before efficiency.')
    parser.add_argument('--package-latency-ns', type=float, default=1000.0)
    parser.add_argument('--package-efficiency', type=float, default=0.8)
    parser.add_argument('--require-complete', action='store_true',
                        help='Require unchanged input lengths and completion of every selected request/token.')
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--arrival-interval', type=float, default=0.01)
    parser.add_argument('--time-limit', type=float, default=0.4)
    parser.add_argument('--run-timeout', type=float, default=1800)
    parser.set_defaults(virtual_channels=8, transfer_timeout=120, dma_pacing=False,
                        dma_burst_bytes=4096, packet_quantum_bytes=1024, packet_buffer_bytes=16384)
    args = parser.parse_args()
    ModernModelValidation(args).run()


if __name__ == '__main__':
    main()
