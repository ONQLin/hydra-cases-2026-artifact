"""Joint reports must not compare clocks covering different simulation windows."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tests.validation.packet_validation.compare_levels import FidelityComparison


class FidelityComparisonTest(unittest.TestCase):
    def test_completed_process_time_is_only_used_for_matching_window(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            label = 'NEMO-CHAT-P1'
            system = dict(chiplet_ids=[0, 1], adjacency=[[0, 1], [1, 0]],
                          memory_bandwidth_gbps={'0': 1}, link_width_bits=128,
                          network_frequency_hz=1e9, router_latency_cycles=1,
                          link_latency_cycles=1, hbm_access_latency_ns=50,
                          dma_pacing=False, model='test', library_sha256='test')
            for suite in ('packet', 'garnet'):
                (root / suite).mkdir()
                (root / suite / 'selected_points.json').write_text(json.dumps([{'label': label}]))
            for backend, seconds, throughput in [('hydra_sim', 1, 12), ('hydra_packet', 2, 11),
                                                 ('chipsim_contended', 10, 10)]:
                suite = 'garnet' if backend == 'chipsim_contended' else 'packet'
                directory = root / suite / label / backend
                run = directory / 'run'
                (run / 'checkpoints').mkdir(parents=True)
                (run / 'system_snapshot.json').write_text('{}')
                (run / 'effective_workload.csv').write_text('same')
                metrics = dict(elapsed_simulation_s=3, tokens_per_sec=throughput, ttft_s=0.1,
                               ttft_samples=2, completed_requests=0, generated_requests=4,
                               pending_requests=2, running_requests=2, wall_seconds=seconds)
                (run / 'checkpoints/3s.json').write_text(json.dumps(metrics))
                # Full process clocks cover five seconds, not the reported three.
                (run / 'metrics.json').write_text(json.dumps(dict(metrics, elapsed_simulation_s=5)))
                (directory / 'wall_time.json').write_text(json.dumps({'seconds': seconds * 2}))
                if backend != 'hydra_sim':
                    runtime = run / ('packet_runtime' if backend == 'hydra_packet' else 'chipsim_runtime')
                    runtime.mkdir()
                    (runtime / 'system.json').write_text(json.dumps(system))
            args = SimpleNamespace(packet_dir=root / 'packet', chipsim_dir=root / 'garnet',
                                   window=3, output_dir=root / 'report')
            captured = []
            with patch.object(FidelityComparison, 'plot', side_effect=captured.extend), patch('builtins.print'):
                FidelityComparison(args).run()
            self.assertEqual(len(captured), 1)
            row = captured[0]
            self.assertIsNone(row['hydra_packet_process_wall_seconds'])
            self.assertEqual(row['packet_vs_chipsim_checkpoint_speedup'], 5)
            self.assertAlmostEqual(row['packet_vs_chipsim_tokens_per_sec_relative_error'], 0.1)
            summary = json.loads((args.output_dir / 'summary.json').read_text())
            self.assertEqual(summary['paired_points'], 1)
            self.assertEqual(summary['missing_points'], [])


if __name__ == '__main__':
    unittest.main()
