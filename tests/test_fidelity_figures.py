"""Check unsigned accuracy aggregation and completed-process speedup semantics."""

import csv
import json
import math
from pathlib import Path
import tempfile
import unittest

from tests.validation.packet_validation.plot_reference_comparison import ReferenceComparison


class ReferenceComparisonTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / 'comparison.csv'
        self.rows = []
        for index, (value, wall) in enumerate(((110, 10), (80, 100)), 1):
            row = dict(point=f'MODEL-CHAT-P{index}', model='MODEL', dataset='CHAT')
            for backend in ('hydra_sim', 'hydra_packet', 'chipsim_contended'):
                row[backend+'_tokens_per_sec'] = value if backend=='hydra_sim' else 100
                row[backend+'_ttft_s'] = value/100 if backend=='hydra_sim' else 1
                row[backend+'_process_wall_seconds'] = wall if backend=='hydra_sim' else (100 if index==1 else 300)
            self.rows.append(row)
        self.write_rows()
        self.path.with_name('summary.json').write_text(json.dumps(dict(
            simulated_seconds=5, paired_points=2, expected_points=2, missing_points=[])))

    def write_rows(self):
        with self.path.open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)

    def test_mape_does_not_cancel_signed_errors_and_speedup_uses_sums(self):
        results = {r['backend']: r for r in ReferenceComparison(self.path, 5).statistics()}
        analytic = results['hydra_sim']
        self.assertAlmostEqual(analytic['tokens_per_sec_mape_percent'], 15)
        self.assertAlmostEqual(analytic['tokens_per_sec_mean_signed_shift_percent'], -5)
        self.assertAlmostEqual(analytic['tokens_per_sec_rmspe_percent'], math.sqrt(250))
        self.assertAlmostEqual(analytic['speedup_vs_chipsim'], 400/110)
        self.assertEqual(analytic['process_wall_sum_s'], 110)
        self.assertEqual(results['chipsim_contended']['ttft_s_mape_percent'], 0)
        self.assertEqual(results['chipsim_contended']['speedup_vs_chipsim'], 1)

    def test_missing_process_time_or_mismatched_window_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Window'):
            ReferenceComparison(self.path, 3)
        self.rows[0]['hydra_packet_process_wall_seconds'] = ''
        self.write_rows()
        with self.assertRaisesRegex(ValueError, 'completed-run'):
            ReferenceComparison(self.path, 5)


if __name__ == '__main__':
    unittest.main()
