#!/usr/bin/env python3
"""Small controlled package-count and bandwidth studies on complete bursts."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tests.validation.modern_models.validate import ModernModelValidation
from tests.validation.multi_package.audit import PackageRunAudit
from tests.validation.prepare_inputs import ValidationInputs


@dataclass(frozen=True)
class StudyPoint:
    packages: int
    bandwidth_gbps: float

    @property
    def label(self):
        return f'p{self.packages}_bw{self.bandwidth_gbps:g}'


class PackageTrendStudy:
    def __init__(self, args):
        self.args = args
        self.raw_dir = args.output_dir.resolve()
        self.report_dir = (args.report_dir or self.raw_dir / 'report').resolve()
        self.trace = self.raw_dir / 'burst.csv'
        self.points = list(dict.fromkeys(
            [StudyPoint(p, args.reference_bandwidth) for p in args.package_counts]
            + [StudyPoint(args.reference_packages, b) for b in args.bandwidths]))

    def run_point(self, point):
        directory = self.raw_dir / point.label
        if not self.args.collect_only:
            args = SimpleNamespace(output_dir=directory, models=['deepseek-v3-text'], backends=['hydra_sim'],
                memory_chiplets=None, packages=point.packages, package_bandwidth_gbps=point.bandwidth_gbps,
                package_efficiency=.8, package_latency_ns=1000., dataset_label='synthetic',
                trace=self.trace, requests=4, batch_size=1,
                arrival_interval=0., require_complete=True, time_limit=120., run_timeout=1800.,
                virtual_channels=8, transfer_timeout=120, dma_pacing=False, dma_burst_bytes=4096,
                packet_quantum_bytes=1024, packet_buffer_bytes=16384)
            ModernModelValidation(args).run()
        audit = PackageRunAudit(directory / 'deepseek-v3-text' / 'hydra_sim')
        evidence = audit.verify()
        manifest = json.loads((directory / 'manifest.json').read_text())
        expected_trace = self.trace
        if (manifest['package_config']['count'] != point.packages
                or manifest['package_config']['bandwidth_gbps'] != point.bandwidth_gbps
                or manifest['package_config']['latency_ns'] != 1000.
                or manifest['package_config']['efficiency'] != .8
                or manifest['trace_sha256'] != hashlib.sha256(expected_trace.read_bytes()).hexdigest()):
            raise ValueError('Study input differs from its declared point.')
        result = self.measure(point, audit, evidence)
        (directory / 'study_result.json').write_text(json.dumps(dict(result=result, audit=evidence), indent=2) + '\n')
        print(f'{point.label}: burst={result["burst_span_s"]:.6f}s, '
              f'TP={result["burst_tp"]:.4f} tokens/s, fabric={result["fabric_service_ms"]:.3f}ms', flush=True)
        return result, evidence, manifest

    def measure(self, point, audit, evidence):
        blocks = audit.read_rows('package_blocks.jsonl')
        transfers = audit.read_rows('package_transfers.jsonl')
        starts = [b for b in blocks if b['event'] == 'start']
        ends = [b for b in blocks if b['event'] == 'complete']
        first = min(b['tick'] for b in starts)
        last = max(b['tick'] for b in ends)
        span = (last - first) / 1e6
        metrics = evidence['metrics']
        # Measure per-stage service without counting parallel blocks as elapsed
        # time. Single-request batches make request/layer pairing unambiguous.
        stage_service = [0] * point.packages
        for request in sorted({r for b in starts for r in b['request_ids']}):
            begin = [b for b in starts if request in b['request_ids']]
            finish = [b for b in ends if request in b['request_ids']]
            for a, b in zip(begin, finish):
                stage_service[a['package']] += b['tick'] - a['tick']
        return dict(label=point.label, packages=point.packages, bandwidth_gbps=point.bandwidth_gbps,
            ttft_ms=metrics['ttft_s'] * 1000, burst_span_s=span, burst_tp=metrics['output_tokens'] / span,
            output_tokens=metrics['output_tokens'], completed_requests=metrics['completed_requests'],
            fabric_bytes=evidence['fabric']['completed_bytes'], transfers=len(transfers),
            fabric_service_ms=sum(t['completed_tick'] - t['start_tick'] for t in transfers) / 1000,
            fabric_queue_ms=sum(t['start_tick'] - t['queued_tick'] for t in transfers) / 1000,
            max_effective_bandwidth_gbps=max(t['bandwidth_gbps'] for t in transfers),
            block_service_s=sum(stage_service) / 1e6,
            max_stage_service_s=max(stage_service) / 1e6,
            stage_service_s=[v / 1e6 for v in stage_service],
            process_wall_seconds=evidence['process_wall_seconds'])

    def verify_controls(self, records):
        hardware = records[0][2]['row']
        compute_service = records[0][0]['block_service_s']
        workload_hash = records[0][2]['trace_sha256']
        for result, _, manifest in records:
            if (manifest['row'] != hardware or manifest['trace_sha256'] != workload_hash
                    or manifest['arrival_interval_s'] != 0
                    or result['completed_requests'] != 4 or result['output_tokens'] != 36
                    or result['block_service_s'] != compute_service):
                raise ValueError('Package study changed per-package hardware, workload or compute service.')
        bandwidth_records = [r for r in records if r[0]['packages'] == self.args.reference_packages]
        reference = None
        for result, audit, manifest in bandwidth_records:
            from tests.validation.chipsim_validation.collect_checkpoints import CheckpointRun
            run = CheckpointRun(Path(audit['run_directory']).parent)
            setup = run.read('system_snapshot.json')
            setup['packages']['fabric'].pop('bandwidth_gbps')
            effective = (run.run_dir / 'effective_workload.csv').read_bytes()
            controlled = (setup, effective, result['fabric_bytes'], result['transfers'], result['block_service_s'])
            if reference is None:
                reference = controlled
            elif controlled != reference:
                raise ValueError('Bandwidth study changed workload, placement, traffic or compute service.')
        ordered = sorted((r[0] for r in bandwidth_records), key=lambda r: r['bandwidth_gbps'])
        if any(a['fabric_service_ms'] < b['fabric_service_ms'] for a, b in zip(ordered, ordered[1:])):
            raise ValueError('Fabric service increased with bandwidth.')
        # End-to-end queue schedules may change at discrete event boundaries;
        # do not force a desired TTFT/throughput trend or hide contrary points.

    def save(self, records):
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.verify_controls(records)
        results = [row for row, _, _ in records]
        (self.report_dir / 'results.json').write_text(json.dumps(records, indent=2) + '\n')
        with (self.report_dir / 'results.csv').open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(results[0]))
            writer.writeheader()
            writer.writerows(results)
        self.plot(results)

    def plot(self, results):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        counts = sorted((r for r in results if r['bandwidth_gbps'] == self.args.reference_bandwidth), key=lambda r: r['packages'])
        bandwidths = sorted((r for r in results if r['packages'] == self.args.reference_packages), key=lambda r: r['bandwidth_gbps'])
        fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
        for column, (series, key, xlabel) in enumerate(((counts, 'packages', 'Package count'),
                                                       (bandwidths, 'bandwidth_gbps', 'Package bandwidth (GB/s/direction)'))):
            x = [r[key] for r in series]
            axes[0, column].plot(x, [r['burst_span_s'] for r in series], 'o-', label='Burst completion span')
            axes[0, column].plot(x, [r['ttft_ms'] / 1000 for r in series], 's--', label='Mean TTFT')
            axes[0, column].set_ylabel('Time (s)')
            axes[0, column].legend()
            axes[1, column].plot(x, [r['fabric_service_ms'] for r in series], 'o-', label='Transfer service sum')
            axes[1, column].plot(x, [r['fabric_queue_ms'] for r in series], 's--', label='Transfer queue wait sum')
            axes[1, column].set_ylabel('Cumulative fabric time (ms)')
            if column:
                axes[1, column].set_yscale('symlog', linthresh=.1)
            axes[1, column].legend()
            for ax in axes[:, column]:
                ax.set_xlabel(xlabel)
                ax.grid(alpha=.25)
                if column:
                    ax.set_xscale('log')
                else:
                    ax.set_xticks(x)
        axes[0, 0].set_title(f'Package scaling at {self.args.reference_bandwidth:g} GB/s')
        axes[0, 1].set_title(f'Bandwidth sensitivity with {self.args.reference_packages} packages')
        fig.suptitle('DeepSeek-V3 / Analytic: 4 requests, each 128 prefill + 8 decode\nStatic batch 1; identical per-package hardware; alpha = 1 us, efficiency = 0.8')
        for suffix in ('png', 'pdf'):
            fig.savefig(self.report_dir / ('trends.' + suffix), dpi=180)
        plt.close(fig)

    def run(self):
        if not self.args.collect_only and self.raw_dir.exists():
            raise ValueError('Use a fresh output directory, or --collect-only for existing completed runs.')
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        if not self.trace.exists():
            ValidationInputs.write_trace(self.trace, prefill=128, decode=8, requests=4)
        with ThreadPoolExecutor(max_workers=self.args.jobs) as pool:
            records = list(pool.map(self.run_point, self.points))
        self.save(records)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--report-dir', type=Path)
    parser.add_argument('--package-counts', nargs='+', type=int, default=[4, 6, 8])
    parser.add_argument('--bandwidths', nargs='+', type=float, default=[.01, .1, 1, 25, 100, 400])
    parser.add_argument('--reference-packages', type=int, default=4)
    parser.add_argument('--reference-bandwidth', type=float, default=25.)
    parser.add_argument('--jobs', type=int, default=2)
    parser.add_argument('--collect-only', action='store_true')
    PackageTrendStudy(parser.parse_args()).run()
