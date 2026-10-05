"""Check cumulative and interval metrics used by long-run comparisons."""

import json
from pathlib import Path
import tempfile
import unittest

from Sim.metrics.monitor import tokens_monitor, request_counter
from Sim.metrics.serving_recorder import ServingMetricsRecorder


class ServingRecorderTest(unittest.TestCase):
    def test_snapshots_use_actual_token_deltas_and_ttft_sample_counts(self):
        tokens_monitor.reset()
        request_counter.reset()
        self.addCleanup(tokens_monitor.reset)
        self.addCleanup(request_counter.reset)
        with tempfile.TemporaryDirectory() as folder:
            recorder = ServingMetricsRecorder(folder)
            tokens_monitor.output_tokens = 50
            tokens_monitor.time_to_first_token = [100_000, 300_000]
            recorder.sample(999_900)
            self.assertFalse((Path(folder) / 'progress.json').exists())
            recorder.sample(1_000_000)
            tokens_monitor.output_tokens = 130
            recorder.sample(2_000_000)
            data = json.loads((Path(folder) / 'progress.json').read_text())
            self.assertEqual(data['tokens_per_sec'], 65)
            self.assertEqual(data['interval_tokens_per_sec'], 80)
            self.assertEqual(data['ttft_samples'], 2)
            self.assertEqual(data['ttft_s'], 0.2)
            self.assertTrue((Path(folder) / 'checkpoints/1s.json').exists())


if __name__ == '__main__':
    unittest.main()
