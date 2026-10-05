#!/usr/bin/env python3
"""Compare matched HYDRA-Analytic, HYDRA-Packet, and CHIPSIM checkpoints."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from tests.validation.chipsim_validation.collect_checkpoints import CheckpointRun


class FidelityComparison:
    backends = [('hydra_sim', 'HYDRA-Analytic', 'o-'),
                ('hydra_packet', 'HYDRA-Packet', '^--'),
                ('chipsim_contended', 'CHIPSIM', 's:')]
    pairs = [('hydra_sim', 'chipsim_contended', 'analytic_vs_chipsim'),
             ('hydra_packet', 'chipsim_contended', 'packet_vs_chipsim'),
             ('hydra_sim', 'hydra_packet', 'analytic_vs_packet')]

    def __init__(self, args):
        self.args = args
        self.output = args.output_dir.resolve()
        self.output.mkdir(parents=True, exist_ok=True)

    def run(self):
        manifest = json.loads((self.args.packet_dir / 'selected_points.json').read_text())
        self.expected_points = len(manifest)
        if manifest != json.loads((self.args.chipsim_dir / 'selected_points.json').read_text()):
            raise ValueError('The selected design points do not match.')
        rows, missing = [], []
        for point in manifest:
            label = point['label']
            native = CheckpointRun(self.args.packet_dir / label / 'hydra_sim')
            packet = CheckpointRun(self.args.packet_dir / label / 'hydra_packet')
            garnet = CheckpointRun(self.args.chipsim_dir / label / 'chipsim_contended')
            metrics = [run.checkpoint(self.args.window) for run in (native, packet, garnet)]
            if any(item is None for item in metrics):
                missing.append(label)
                continue
            native.verify_inputs(packet)
            native.verify_inputs(garnet)
            a, b = packet.read('packet_runtime/system.json'), garnet.read('chipsim_runtime/system.json')
            for key in ('chiplet_ids', 'adjacency', 'memory_bandwidth_gbps', 'link_width_bits',
                        'network_frequency_hz', 'router_latency_cycles', 'link_latency_cycles',
                        'hbm_access_latency_ns', 'dma_pacing', 'model'):
                if a[key] != b[key]:
                    raise ValueError(f'{label}: network configurations disagree on {key}')
            if a['dma_pacing']:
                width = a['link_width_bits'] // 8
                if a['dma_burst_bytes'] != b['dma_burst_bytes'] // width * width:
                    raise ValueError('DMA burst configurations disagree')
            row = {'point': label, 'model': label.split('-')[0], 'dataset': label.split('-')[1]}
            for filename, key in [('system_snapshot.json', 'setup_sha256'),
                                  ('effective_workload.csv', 'workload_sha256')]:
                row[key] = hashlib.sha256((native.run_dir / filename).read_bytes()).hexdigest()
            row['packet_library_sha256'] = a['library_sha256']
            for (backend, _, _), run, metric in zip(self.backends, (native, packet, garnet), metrics):
                if metric['elapsed_simulation_s'] != self.args.window:
                    raise ValueError('Checkpoint time does not match the requested window')
                for key in ('tokens_per_sec', 'ttft_s', 'ttft_samples', 'completed_requests',
                            'generated_requests', 'pending_requests', 'running_requests'):
                    row[backend + '_' + key] = metric[key]
                # Checkpoint wall time starts at recorder initialization, whereas
                # completed-process wall time includes interpreter/setup/output.
                row[backend + '_checkpoint_wall_seconds'] = metric['wall_seconds']
                final = run.read('metrics.json')
                wall = run.directory / 'wall_time.json'
                matched_final = final is not None and final['elapsed_simulation_s'] == self.args.window
                row[backend + '_process_wall_seconds'] = json.loads(wall.read_text())['seconds'] if matched_final and wall.exists() else None
            for approximate, reference, label in self.pairs:
                for key in ('tokens_per_sec', 'ttft_s'):
                    base, value = row[reference + '_' + key], row[approximate + '_' + key]
                    row[label + '_' + key + '_relative_error'] = value / base - 1 if base and value is not None else None
                row[label + '_checkpoint_speedup'] = row[reference + '_checkpoint_wall_seconds'] / row[approximate + '_checkpoint_wall_seconds']
            rows.append(row)
        summary = dict(simulated_seconds=self.args.window, paired_points=len(rows),
                       expected_points=len(manifest), missing_points=missing,
                       scope='Transient window from t=0; TTFT is censored at cutoff. Network timing is refined; compute and runtime are shared.')
        summary['packet_vs_chipsim_ordering'] = self.compare_ordering(rows)
        summary['pairwise_ordering'] = {
            label: self.compare_ordering(rows, approximate, reference)
            for approximate, reference, label in self.pairs}
        summary['pairwise_bias'] = {}
        for approximate, reference, label in self.pairs:
            panel = summary['pairwise_bias'][label] = {}
            for key in ('tokens_per_sec', 'ttft_s'):
                errors = [r[label + '_' + key + '_relative_error'] for r in rows
                          if r[label + '_' + key + '_relative_error'] is not None]
                panel[key] = dict(n=len(errors), mean_signed_relative_error=sum(errors) / len(errors) if errors else None,
                                  max_absolute_relative_error=max(map(abs, errors)) if errors else None)
        summary['wall_time_scope'] = 'Single measurements under concurrent host load. Checkpoint clocks exclude some setup; process clocks appear only for a completed run of exactly this window.'
        for key in ('tokens_per_sec', 'ttft_s'):
            values = [abs(r['packet_vs_chipsim_' + key + '_relative_error']) for r in rows
                      if r['packet_vs_chipsim_' + key + '_relative_error'] is not None]
            summary['max_packet_vs_chipsim_' + key + '_relative_error'] = max(values) if values else None
        (self.output / 'summary.json').write_text(json.dumps(summary, indent=2))
        if rows:
            with (self.output / 'comparison.csv').open('w') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            self.plot(rows)
        print(json.dumps(summary, indent=2))

    def compare_ordering(self, rows, approximate='hydra_packet', reference='chipsim_contended'):
        result = {}
        for group in sorted({(r['model'], r['dataset']) for r in rows}):
            panel = [r for r in rows if (r['model'], r['dataset']) == group]
            result['-'.join(group)] = metrics = {}
            for key in ('tokens_per_sec', 'ttft_s'):
                comparisons = []
                for index, first in enumerate(panel):
                    for second in panel[index + 1:]:
                        values = [r[b + '_' + key] for r in (first, second)
                                  for b in (approximate, reference)]
                        if any(v is None for v in values):
                            continue
                        medium, detailed = values[0] - values[2], values[1] - values[3]
                        if detailed:
                            comparisons.append(medium * detailed > 0)
                metrics[key] = dict(matching=sum(comparisons), comparable=len(comparisons))
        return result

    def plot(self, rows):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        groups = sorted({(r['model'], r['dataset']) for r in rows})
        fig, axes = plt.subplots(1, len(groups), figsize=(6 * len(groups), 4.5), squeeze=False)
        for axis, group in zip(axes[0], groups):
            panel = [r for r in rows if (r['model'], r['dataset']) == group]
            for backend, label, style in self.backends:
                plotted = [r for r in panel if r[backend + '_ttft_s'] is not None]
                axis.plot([r[backend + '_ttft_s'] for r in plotted],
                          [r[backend + '_tokens_per_sec'] for r in plotted], style,
                          label=label, markerfacecolor='none', markersize=7)
                if backend == 'hydra_sim':
                    for row in plotted:
                        axis.annotate(row['point'].split('-')[-1],
                                      (row[backend + '_ttft_s'], row[backend + '_tokens_per_sec']),
                                      xytext=(4, 6), textcoords='offset points')
            axis.set(title=' / '.join(group), xlabel='Mean TTFT (s)', ylabel='Output tokens/s')
            axis.grid(alpha=0.25)
            axis.legend()
        fig.suptitle(f'Three network fidelity levels; {self.args.window:g}s transient window; '
                     f'{len(rows)}/{self.expected_points} paired points')
        fig.tight_layout()
        for extension in ('png', 'pdf'):
            fig.savefig(self.output / f'tp_vs_ttft.{extension}', dpi=180)
        plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packet-dir', type=Path, required=True)
    parser.add_argument('--chipsim-dir', type=Path, required=True)
    parser.add_argument('--window', type=float, default=1)
    parser.add_argument('--output-dir', type=Path, required=True)
    FidelityComparison(parser.parse_args()).run()
