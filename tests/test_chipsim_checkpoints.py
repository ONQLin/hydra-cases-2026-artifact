"""Ensure partial comparison reports pair identical inputs at the same cutoff."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tests.validation.chipsim_validation.collect_checkpoints import CheckpointComparison


class CheckpointComparisonTest(unittest.TestCase):
    def test_incomplete_pairs_and_input_mismatch(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            label = 'LLAMA3-CHAT-P1'
            (root / 'selected_points.json').write_text(json.dumps([{'label': label}]))
            runs = []
            for backend in ('hydra_sim', 'chipsim_contended'):
                directory = root / label / backend
                run = directory / 'run'
                (run / 'checkpoints').mkdir(parents=True)
                (run / 'system_snapshot.json').write_text('{}')
                (run / 'effective_workload.csv').write_text('same workload')
                (directory / 'command.json').write_text(json.dumps([
                    '--workload-config.request-generator-config.num-requests', '32']))
                runs.append(run)
            record = {'elapsed_simulation_s': 1, 'tokens_per_sec': 100, 'ttft_s': 0.2,
                      'ttft_samples': 4, 'interval_tokens_per_sec': 100,
                      'completed_requests': 0, 'generated_requests': 32,
                      'pending_requests': 28, 'running_requests': 4}
            (runs[0] / 'checkpoints/1s.json').write_text(json.dumps(record))
            reporter = CheckpointComparison(root, [1, 2])
            self.assertFalse(reporter.collect())
            self.assertFalse((reporter.output_dir / '1s/comparison.csv').exists())
            record.update(tokens_per_sec=90, ttft_s=0.22)
            (runs[1] / 'checkpoints/1s.json').write_text(json.dumps(record))
            with patch('tests.validation.chipsim_validation.collect_checkpoints.ParetoComparison.plot'):
                self.assertFalse(reporter.collect())
            summary = json.loads((reporter.output_dir / '1s/summary.json').read_text())
            self.assertEqual(summary['paired_points'], 1)
            self.assertAlmostEqual(summary['groups']['LLAMA3-CHAT']['tp']['max_absolute_relative_error'], 0.1)
            self.assertFalse((reporter.output_dir / '2s/comparison.csv').exists())
            config = {key: {} for key in ('seed', 'cluster_config', 'arch_config', 'chips_config',
                                         'placmt_config', 'workload_config', 'mapping_config')}
            for run in runs:
                (run / 'config.json').write_text(json.dumps(config))
            with patch('tests.validation.chipsim_validation.collect_checkpoints.ParetoComparison.plot'):
                self.assertFalse(reporter.collect())
            config['workload_config'] = {'arrival_interval': 0.5}
            (runs[1] / 'config.json').write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, 'workload_config'):
                reporter.collect()
            (runs[1] / 'effective_workload.csv').write_text('different workload')
            with self.assertRaisesRegex(ValueError, 'Backends disagree'):
                reporter.collect()


if __name__ == '__main__':
    unittest.main()
