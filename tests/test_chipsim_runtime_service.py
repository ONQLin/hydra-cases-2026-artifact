"""Contract tests for failure handling and CHIPSIM execution feedback."""

import importlib.util
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

if importlib.util.find_spec('requests') is None:
    raise unittest.SkipTest('Run service tests with .chipsim-venv/bin/python.')

ROOT = Path(__file__).resolve().parents[1]
# The service normally runs in an isolated interpreter. In this test process,
# expose both integration namespaces because the snapshot tests import HYDRA's
# integrations first, while CHIPSIM imports integrations.CIMLoop_API.
import integrations
integrations.__path__.append(str(ROOT / 'third_party/CHIPSIM/integrations'))
sys.path.insert(0, str(ROOT / 'third_party/CHIPSIM'))
from integrations.chipsim.runtime_execution import RuntimeCommunicationSimulator, RuntimeExecutionService
from src.integrations.hydra.compute_backend import HydraComputeBackend


class RuntimeServiceTest(unittest.TestCase):
    def test_dependent_phase_writes_reverse_network_endpoints(self):
        service = RuntimeExecutionService.__new__(RuntimeExecutionService)
        service.config = {'hbm_access_latency_ns': 0}
        service.compute = HydraComputeBackend()
        transfer = Mock(return_value=0)
        service.communication = SimpleNamespace(transfer=transfer, simulations=0, cache_hits=0)
        result = service.execute({'request_id': 1, 'source': 0, 'destination': 1,
                                  'bandwidth_bytes_s': 1e9,
                                  'phases': [{'compute_ns': 0, 'memory_bytes': 1000},
                                             {'compute_ns': 2000, 'memory_bytes': 0},
                                             {'compute_ns': 0, 'memory_bytes': 3000,
                                              'memory_direction': 'write'}]})
        self.assertEqual([call.args for call in transfer.call_args_list],
                         [(0, 1, 1000), (0, 1, 0), (1, 0, 3000)])
        self.assertEqual(result['latency_us'], 6)

    def test_compute_memory_overlap_and_serial_spill(self):
        service = RuntimeExecutionService.__new__(RuntimeExecutionService)
        service.config = {'hbm_access_latency_ns': 0}
        service.compute = HydraComputeBackend()
        service.communication = SimpleNamespace(transfer=Mock(side_effect=[3.0, 7.0]), simulations=2, cache_hits=0)
        result = service.execute({'request_id': 5, 'source': 0, 'destination': 1,
                                  'bandwidth_bytes_s': 1e9,
                                  'phases': [{'compute_ns': 5000, 'memory_bytes': 1000},
                                             {'compute_ns': 0, 'memory_bytes': 1000}]})
        self.assertEqual(result['latency_us'], 12.0)
        self.assertEqual(result['request_id'], 5)

    def test_rejects_incomplete_garnet_run_even_on_successful_exit(self):
        with tempfile.TemporaryDirectory() as folder:
            communication = RuntimeCommunicationSimulator.__new__(RuntimeCommunicationSimulator)
            communication.current_dir = Path(folder)
            communication.gem5_path = folder
            communication.config = {'timeout_seconds': 1}
            communication.expected_packets = 100
            communication._extract_garnet_stats = Mock(return_value={
                'packets_injected': 100, 'packets_received': 99, 'simTicks': 10000,
            })
            with patch('integrations.chipsim.runtime_execution.subprocess.run', return_value=SimpleNamespace(returncode=0)):
                with self.assertRaisesRegex(RuntimeError, 'Incomplete Garnet transfer'):
                    communication._run_garnet_simulation(['gem5'])

    def test_cache_reuses_only_equal_flit_transfers(self):
        with tempfile.TemporaryDirectory() as folder:
            communication = RuntimeCommunicationSimulator.__new__(RuntimeCommunicationSimulator)
            communication.config = {'link_width_bits': 128}
            communication.system = SimpleNamespace(num_chiplets=3)
            communication.output_dir = Path(folder)
            communication.reuse_mesh_paths = False
            communication.results = {}
            communication.cache_hits = 0
            communication.simulations = 0
            communication.simulate_communication = Mock(return_value={
                'latency': {'total_runtime_us': 2, 'packets_received': 2},
            })
            communication.transfer(0, 1, 17)
            communication.transfer(0, 1, 32)
            self.assertEqual(communication.cache_hits, 1)
            communication.transfer(0, 2, 32)
            communication.transfer(0, 1, 33)
            self.assertEqual(communication.simulate_communication.call_count, 3)

    def test_isolated_full_mesh_cache_distinguishes_path_length(self):
        with tempfile.TemporaryDirectory() as folder:
            communication = RuntimeCommunicationSimulator.__new__(RuntimeCommunicationSimulator)
            communication.config = {'link_width_bits': 128}
            communication.system = SimpleNamespace(num_chiplets=4)
            communication.output_dir = Path(folder)
            communication.mesh_columns = 2
            communication.reuse_mesh_paths = True
            communication.results = {}
            communication.cache_hits = 0
            communication.simulations = 0
            communication.simulate_communication = Mock(return_value={
                'latency': {'total_runtime_us': 2, 'packets_received': 2},
            })
            communication.transfer(0, 1, 32)
            communication.transfer(0, 2, 32)
            self.assertEqual(communication.cache_hits, 1)
            communication.transfer(0, 3, 32)
            self.assertEqual(communication.simulate_communication.call_count, 2)

    @unittest.skipUnless(os.environ.get('RUN_CHIPSIM_GARNET_TESTS') == '1',
                         'Set RUN_CHIPSIM_GARNET_TESTS=1 for real Garnet validation.')
    def test_real_router_latency_changes_execution_feedback(self):
        config = {
            'chiplet_ids': [0, 1, 2, 3],
            'adjacency': [[0, 1, 1, 0], [1, 0, 0, 1],
                          [1, 0, 0, 1], [0, 1, 1, 0]],
            'mesh_rows': 2, 'link_width_bits': 128,
            'network_frequency_hz': 1e9, 'garnet_ticks_per_cycle': 1000,
            'router_latency_cycles': 1, 'link_latency_cycles': 1,
            'virtual_channels_per_vnet': 8, 'garnet_sim_cycles': 1_000_000,
            'hbm_access_latency_ns': 50, 'timeout_seconds': 60,
            'reuse_mesh_paths': True,
        }
        request = {'request_id': 1, 'source': 0, 'destination': 3,
                   'bandwidth_bytes_s': 1e12,
                   'phases': [{'compute_ns': 100, 'memory_bytes': 16384}]}
        latencies = []
        with tempfile.TemporaryDirectory() as folder:
            for router_latency in (1, 32):
                directory = Path(folder) / str(router_latency)
                directory.mkdir()
                config['router_latency_cycles'] = router_latency
                service = RuntimeExecutionService(config.copy(), directory / 'system.json')
                result = service.execute(request)
                self.assertEqual(result['latency_us'], result['phases'][0]['noi_us'])
                latencies.append(result['latency_us'])
            self.assertGreater(latencies[1], latencies[0])


if __name__ == '__main__':
    unittest.main()
