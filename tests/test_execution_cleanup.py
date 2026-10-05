"""Backend resources must close after failed diagnostics or a dead service."""

import io
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

from Sim.config.sys_config import HPSim_Config
from Sim.execution.chipsim import ChipsimExecutionBackend
from Sim.execution.contended import ContendedExecutionBackend
from Sim.execution.packet import PacketExecutionBackend


class ExecutionCleanupTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.output_dir = Path(directory.name)

    def service(self, backend_class=ChipsimExecutionBackend):
        backend = backend_class(HPSim_Config())
        backend.output_dir = self.output_dir
        backend.log = io.StringIO()
        backend.trace = io.StringIO()
        backend.process = Mock()
        backend.process.stdin = io.StringIO()
        backend.process.stdout = io.StringIO()
        return backend

    def test_exited_service_closes_both_pipes_and_is_idempotent(self):
        backend = self.service()
        process, log, trace = backend.process, backend.log, backend.trace
        process.poll.return_value = 1
        backend.close()
        backend.close()
        self.assertTrue(all(stream.closed for stream in
                            (process.stdin, process.stdout, log, trace)))
        self.assertIsNone(backend.process)
        process.wait.assert_not_called()

    def test_broken_pipe_still_waits_and_releases_streams(self):
        backend = self.service()
        process, log, trace = backend.process, backend.log, backend.trace
        process.stdin = Mock()
        process.stdin.close.side_effect = BrokenPipeError()
        process.poll.return_value = None
        backend.close()
        process.wait.assert_called_once_with(timeout=10)
        self.assertTrue(all(stream.closed for stream in (process.stdout, log, trace)))

    def test_stalled_service_kills_process_group_and_reaps(self):
        backend = self.service()
        process = backend.process
        process.pid = 12345
        process.poll.return_value = None
        process.wait.side_effect = [subprocess.TimeoutExpired('service', 10), 0]
        with patch('Sim.execution.chipsim.os.killpg') as kill:
            backend.close()
        kill.assert_called_once()
        self.assertEqual(kill.call_args.args[0], process.pid)
        self.assertEqual(process.wait.call_count, 2)
        self.assertTrue(process.stdout.closed)

    def test_failed_network_summary_still_closes_service(self):
        backend = self.service(ContendedExecutionBackend)
        process, trace = backend.process, backend.trace
        process.poll.return_value = 0
        with patch.object(Path, 'write_text', side_effect=OSError('disk full')):
            with self.assertRaisesRegex(OSError, 'disk full'):
                backend.close()
        self.assertTrue(process.stdin.closed)
        self.assertTrue(process.stdout.closed)
        self.assertTrue(trace.closed)

    def test_packet_handle_closes_even_when_statistics_cannot_be_saved(self):
        backend = PacketExecutionBackend(HPSim_Config())
        backend.output_dir = self.output_dir
        network = backend.network = Mock()
        network.statistics.return_value = {}
        trace = backend.trace = io.StringIO()
        with patch.object(Path, 'write_text', side_effect=OSError('disk full')):
            with self.assertRaisesRegex(OSError, 'disk full'):
                backend.close()
        backend.close()
        network.statistics.assert_called_once()
        network.close.assert_called_once()
        self.assertTrue(trace.closed)
        self.assertIsNone(backend.network)


if __name__ == '__main__':
    unittest.main()
