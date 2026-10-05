"""Guard validation configuration, failed-run evidence, and service cleanup."""

import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from tests.validation.chipsim_validation.run_comparison import ParetoComparison
from integrations.chipsim.runtime_service import RuntimeServiceDriver


class ComparisonSafetyTest(unittest.TestCase):
    def test_isolated_backend_rejects_dma_pacing(self):
        with tempfile.TemporaryDirectory() as folder:
            comparison = ParetoComparison(SimpleNamespace(
                output_dir=Path(folder), detailed_backend='chipsim_runtime', dma_pacing=True))
            with self.assertRaisesRegex(ValueError, 'requires.*chipsim_contended'):
                comparison.validate_options()

    def test_incomplete_run_is_preserved_on_retry(self):
        with tempfile.TemporaryDirectory() as folder:
            comparison = ParetoComparison(SimpleNamespace(
                output_dir=Path(folder), detailed_backend='chipsim_contended', resume=True))
            point = SimpleNamespace(label='LLAMA3-CHAT-P1', command=Mock(return_value=['python', 'main.py']))
            directory = Path(folder) / point.label / 'hydra_sim'
            directory.mkdir(parents=True)
            log = directory / 'console.log'
            log.write_text('failure evidence')
            with patch('subprocess.Popen') as launch:
                with self.assertRaisesRegex(ValueError, 'incomplete run'):
                    comparison.run_point(point)
                launch.assert_not_called()
            self.assertEqual(log.read_text(), 'failure evidence')


class ServiceCleanupTest(unittest.TestCase):
    def test_service_closes_when_parent_disconnects(self):
        with tempfile.TemporaryDirectory() as folder:
            system = Path(folder) / 'system.json'
            system.write_text(json.dumps({'chipsim_root': folder}))
            for failure_at in (1, 2):
                with self.subTest(failure_at=failure_at):
                    service = SimpleNamespace(execute=Mock(return_value={'ok': True}), close=Mock())
                    driver = RuntimeServiceDriver()
                    output = Mock()
                    output.write.side_effect = [None] * (failure_at - 1) + [BrokenPipeError()]
                    with patch.object(driver, 'create_service', return_value=service), \
                         patch.object(sys, 'path', sys.path.copy()), \
                         patch.object(sys, 'stdin', io.StringIO('{}\n')), \
                         patch.object(sys, 'stdout', output):
                        with self.assertRaises(BrokenPipeError):
                            driver.run(system)
                    service.close.assert_called_once()


if __name__ == '__main__':
    unittest.main()
