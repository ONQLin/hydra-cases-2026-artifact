"""Validate persistent packet contention and causally ordered completion feedback."""

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'integrations/chipsim'))


class CoSimulationTest(unittest.TestCase):
    def test_completion_preempts_next_application_event(self):
        from Sim.execution.contended import CoSimulationEnvironment, NetworkCompletion

        class Execution:
            ticks_per_hydra_tick = 1000
            network_tick = 0

            def advance(self, target):
                calls.append(target)
                if self.network_tick < 500:
                    self.network_tick = 500
                    completion.complete_at(1, 500)
                else:
                    self.network_tick = target

        execution = Execution()
        env = CoSimulationEnvironment(execution)
        calls = []
        completion = NetworkCompletion(env)
        observed = []

        def application():
            yield completion
            observed.append(env.now)
            # A dependent submission must occur before the original deadline.
            self.assertEqual(execution.network_tick, 1000)

        env.process(application())
        env.run(until=10)
        self.assertEqual(observed, [1])
        self.assertEqual(calls, [10000, 1000, 10000])

    def test_completion_rejects_time_travel_and_duplicate_delivery(self):
        import simpy
        from Sim.execution.contended import NetworkCompletion
        env = simpy.Environment(initial_time=2)
        event = NetworkCompletion(env)
        with self.assertRaises(RuntimeError):
            event.complete_at(1, 0)
        event.complete_at(3, 0)
        with self.assertRaises(RuntimeError):
            event.complete_at(3, 0)


@unittest.skipUnless(os.environ.get('RUN_CHIPSIM_GARNET_TESTS') == '1',
                     'Set RUN_CHIPSIM_GARNET_TESTS=1 for real Garnet validation.')
class PacketContentionTest(unittest.TestCase):
    def test_isolated_shared_disjoint_and_staggered_transfers(self):
        import integrations
        integrations.__path__.append(str(ROOT / 'third_party/CHIPSIM/integrations'))
        sys.path.insert(0, str(ROOT / 'third_party/CHIPSIM'))
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
        }
        folder = os.environ.get('CHIPSIM_TEST_OUTPUT')
        if folder is None:
            temporary = tempfile.TemporaryDirectory()
            self.addCleanup(temporary.cleanup)
            folder = temporary.name
        root = Path(folder).resolve()
        results = {}
        cases = {
            'isolated': [(0, 0, 3)],
            'shared_destination': [(0, 0, 3), (0, 1, 3)],
            'disjoint': [(0, 0, 1), (0, 2, 3)],
            'one_hop': [(0, 0, 1)],
            'staggered': [(0, 0, 3), (250_000, 1, 3)],
        }
        for name, arrivals in cases.items():
            directory = root / name
            directory.mkdir(parents=True, exist_ok=True)
            system = directory / 'system.json'
            system.write_text(json.dumps(config, indent=2))
            network = PersistentGarnetNetwork(config, system)
            completed = {}
            now = 0
            try:
                for flow, (arrival, source, destination) in enumerate(arrivals, 1):
                    while now < arrival:
                        response = network.execute({'kind': 'advance', 'target_tick': arrival})
                        now = response['tick']
                        values = response['completed']
                        completed.update(zip(values[::2], values[1::2]))
                    network.execute({'kind': 'submit', 'flow_id': flow,
                                     'source': source, 'destination': destination, 'bytes': 16384})
                while len(completed) < len(arrivals):
                    response = network.execute({'kind': 'advance', 'target_tick': 100_000_000})
                    self.assertLess(response['tick'], 100_000_000, 'Network failed to drain.')
                    values = response['completed']
                    completed.update(zip(values[::2], values[1::2]))
                self.assertEqual(set(completed), set(range(1, len(arrivals) + 1)))
                # Equal-time completions must not leave stale exit events that
                # stall a subsequent advance or a later application transfer.
                after = network.execute({'kind': 'advance', 'target_tick': 100_000_000})
                self.assertEqual(after['tick'], 100_000_000)
                self.assertEqual(after['completed'], [])
                results[name] = completed
            finally:
                network.close()
            stats = network._extract_garnet_stats(str(directory / 'garnet_online/m5out/stats.txt'))
            self.assertEqual(stats['packets_injected'], 1024 * len(arrivals))
            self.assertEqual(stats['packets_received'], 1024 * len(arrivals))
        (root / 'contention_results.json').write_text(json.dumps(results, indent=2))
        isolated = results['isolated'][1]
        self.assertGreater(results['shared_destination'][1], isolated * 1.5)
        self.assertGreater(results['staggered'][1], isolated)
        self.assertEqual(results['disjoint'][1], results['one_hop'][1])
        self.assertEqual(results['disjoint'][2], results['one_hop'][1])

        directory = root / 'cutoff'
        directory.mkdir(parents=True, exist_ok=True)
        network = PersistentGarnetNetwork(config, directory / 'system.json')
        try:
            network.execute({'kind': 'submit', 'flow_id': 1, 'source': 0,
                             'destination': 3, 'bytes': 16384})
            response = network.execute({'kind': 'advance', 'target_tick': 100_000})
            self.assertEqual(response['tick'], 100_000)
            self.assertEqual(response['completed'], [])
        finally:
            network.close()
        stats = network._extract_garnet_stats(str(directory / 'garnet_online/m5out/stats.txt'))
        self.assertGreater(stats['packets_received'], 0)
        self.assertLess(stats['packets_received'], 1024)
        (directory / 'result.json').write_text(json.dumps({
            'cutoff_tick': response['tick'], 'completed_flows': response['completed'],
            'expected_flits': 1024, 'received_flits': stats['packets_received'],
        }, indent=2))


if __name__ == '__main__':
    unittest.main()
