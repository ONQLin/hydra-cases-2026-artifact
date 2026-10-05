"""Physical HBM bandwidth sharing and burst pacing in a real Garnet network."""

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(os.environ.get('RUN_CHIPSIM_GARNET_TESTS') == '1',
                     'Set RUN_CHIPSIM_GARNET_TESTS=1 for real Garnet validation.')
class DmaPacingTest(unittest.TestCase):
    def test_bandwidth_is_shared_per_hbm_and_independent_across_hbms(self):
        import integrations
        integrations.__path__.append(str(ROOT / 'third_party/CHIPSIM/integrations'))
        sys.path.insert(0, str(ROOT / 'third_party/CHIPSIM'))
        sys.path.insert(0, str(ROOT / 'integrations/chipsim'))
        from persistent_network import PersistentGarnetNetwork

        config = {
            'chiplet_ids': [0, 1, 2, 3],
            'adjacency': [[0, 1, 1, 0], [1, 0, 0, 1],
                          [1, 0, 0, 1], [0, 1, 1, 0]],
            'mesh_rows': 2, 'link_width_bits': 128,
            'network_frequency_hz': 1e9, 'garnet_ticks_per_cycle': 1000,
            'router_latency_cycles': 1, 'link_latency_cycles': 1,
            'virtual_channels_per_vnet': 8, 'garnet_sim_cycles': 1_000_000,
            'hbm_access_latency_ns': 50, 'timeout_seconds': 60,
            'dma_pacing': True, 'dma_burst_bytes': 1024,
        }
        output = os.environ.get('CHIPSIM_TEST_OUTPUT')
        if output is None:
            temporary = tempfile.TemporaryDirectory()
            self.addCleanup(temporary.cleanup)
            output = temporary.name
        root = Path(output).resolve()
        cases = {
            'single_hbm': (1, [(0, 1)]),
            'shared_hbm': (1, [(0, 1), (0, 3)]),
            'independent_hbms': (1, [(0, 1), (2, 3)]),
            'double_bandwidth': (2, [(0, 1)]),
        }
        results = {}
        for name, (bandwidth, endpoints) in cases.items():
            directory = root / name
            directory.mkdir(parents=True, exist_ok=True)
            case_config = dict(config, memory_bandwidth_gbps={str(node): bandwidth for node in range(4)})
            system = directory / 'system.json'
            system.write_text(json.dumps(case_config, indent=2))
            network = PersistentGarnetNetwork(case_config, system)
            completed = {}
            try:
                for flow, (source, destination) in enumerate(endpoints, 1):
                    network.execute({'kind': 'submit', 'flow_id': flow, 'source': source,
                                     'destination': destination, 'bytes': 16384})
                while len(completed) < len(endpoints):
                    response = network.execute({'kind': 'advance', 'target_tick': 100_000_000})
                    self.assertLess(response['tick'], 100_000_000)
                    values = response['completed']
                    completed.update(zip(values[::2], values[1::2]))
            finally:
                network.close()
            stats = network._extract_garnet_stats(str(directory / 'garnet_online/m5out/stats.txt'))
            self.assertEqual(stats['packets_received'], len(endpoints) * 1024)
            results[name] = {flow: tick / 1e6 for flow, tick in completed.items()}
        (root / 'pacing_results_us.json').write_text(json.dumps(results, indent=2))
        single = results['single_hbm'][1]
        self.assertGreaterEqual(single, 16.384 + 0.050)
        self.assertGreaterEqual(max(results['shared_hbm'].values()), 32.768 + 0.050)
        self.assertLess(abs(results['shared_hbm'][1] - results['shared_hbm'][2]), 2)
        self.assertEqual(results['independent_hbms'][1], single)
        self.assertEqual(results['independent_hbms'][2], single)
        self.assertLess(results['double_bandwidth'][1], single * 0.6)


if __name__ == '__main__':
    unittest.main()
