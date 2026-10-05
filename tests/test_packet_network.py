"""Physical invariants and co-simulation contract of the C++ packet network."""

from pathlib import Path
import random
import unittest

from integrations.packet.network import PacketNetwork

ROOT = Path(__file__).resolve().parents[1]


class PacketNetworkTest(unittest.TestCase):
    def config(self, quantum=1024, buffer=16384, dma=False):
        return dict(chiplet_ids=[0, 1, 2, 3], mesh_rows=2,
                    adjacency=[[0, 1, 1, 0], [1, 0, 0, 1], [1, 0, 0, 1], [0, 1, 1, 0]],
                    network_frequency_hz=1e9, link_width_bits=128,
                    router_latency_cycles=1, link_latency_cycles=1,
                    memory_bandwidth_gbps={'0': 1, '2': 1}, hbm_access_latency_ns=50,
                    packet_quantum_bytes=quantum, packet_buffer_bytes=buffer,
                    dma_pacing=dma, dma_burst_bytes=1024)

    def network(self, **kwargs):
        network = PacketNetwork(self.config(**kwargs))
        self.addCleanup(network.close)
        return network

    def submit(self, network, flow, source, destination, size=16384):
        return network.execute(dict(kind='submit', flow_id=flow, source=source,
                                    destination=destination, bytes=size))

    def advance(self, network, target):
        result = network.execute(dict(kind='advance', target_tick=target))
        self.assertLessEqual(result['tick'], target)
        return dict(zip(result['completed'][::2], result['completed'][1::2]))

    def drain(self, network):
        done = {}
        while network.pending:
            done.update(self.advance(network, 100_000_000))
            self.assertLess(network.now, 100_000_000, 'Network failed to drain')
        return done

    def test_cut_through_and_disjoint_paths(self):
        for quantum in (16, 256, 1024, 4096):
            with self.subTest(quantum=quantum):
                network = self.network(quantum=quantum)
                self.submit(network, 1, 0, 3)
                self.assertEqual(self.drain(network), {1: 1_032_000})
                network = self.network(quantum=quantum)
                self.submit(network, 1, 0, 1)
                self.submit(network, 2, 2, 3)
                self.assertEqual(self.drain(network), {1: 1_030_000, 2: 1_030_000})
                self.assertEqual(network.statistics()['received_bytes'], 32768)

    def test_contention_staggered_arrival_and_quantum_convergence(self):
        short_flow = []
        for quantum in (4096, 1024, 256):
            network = self.network(quantum=quantum)
            self.submit(network, 1, 0, 3)
            self.submit(network, 2, 1, 3)
            times = self.drain(network)
            self.assertGreater(min(times.values()), 1_500_000)
            self.assertEqual(max(times.values()), 2_054_000)
            short_flow.append(times[2])
        self.assertEqual(short_flow, sorted(short_flow))
        self.assertLess(abs(short_flow[-1] / 2_052_000 - 1), 0.01)
        network = self.network()
        self.submit(network, 1, 0, 3)
        self.assertEqual(self.advance(network, 250_000), {})
        self.submit(network, 2, 1, 3)
        times = self.drain(network)
        self.assertGreater(times[1], 1_032_000)
        self.assertGreater(times[2], times[1])

    def test_bounded_backpressure_and_conservation(self):
        network = self.network(buffer=1024)
        rng = random.Random(6)
        expected = 0
        for flow in range(32):
            a, b = rng.sample(range(4), 2)
            size = rng.randrange(1, 8192)
            expected += ((size + 15) // 16) * 16
            self.submit(network, flow, a, b, size)
        self.assertEqual(len(self.drain(network)), 32)
        stats = network.statistics()
        self.assertEqual(stats['received_bytes'], expected)
        self.assertLessEqual(stats['peak_buffer_bytes'], 1024)
        self.assertGreater(stats['blocked_checks'], 0)
        self.assertEqual(stats['submitted'], stats['completed'])

    def test_advancement_partition_does_not_change_completion(self):
        a, b = self.network(), self.network()
        for network in (a, b):
            self.submit(network, 1, 0, 3)
            self.submit(network, 2, 1, 3)
        expected = self.drain(a)
        observed = {}
        for target in range(1000, 3_000_001, 1000):
            observed.update(self.advance(b, target))
        self.assertEqual(observed, expected)

    def test_dma_bandwidth_and_independent_hbms(self):
        for shared in (False, True):
            network = self.network(dma=True)
            self.submit(network, 1, 0, 1)
            self.submit(network, 2, 0 if shared else 2, 3)
            times = self.drain(network)
            if shared:
                self.assertGreater(min(times.values()), 31_000_000)
                self.assertLess(max(times.values()) - min(times.values()), 2_000_000)
            else:
                self.assertEqual(times, {1: 16_504_000, 2: 16_504_000})

    def test_invalid_inputs_cutoff_and_lifecycle(self):
        network = self.network()
        self.submit(network, 2**40, 0, 3)
        self.assertEqual(self.advance(network, 100_000), {})
        self.assertEqual(network.statistics()['completed'], 0)
        with self.assertRaises(ValueError):
            self.advance(network, 99_000)
        with self.assertRaises(ValueError):
            self.submit(network, 2**40, 1, 3)
        with self.assertRaises(ValueError):
            self.submit(network, 2, 8, 3)
        self.assertEqual(set(self.drain(network)), {2**40})
        network.close()
        network.close()
        with self.assertRaises(RuntimeError):
            self.advance(network, 1_000_000)
        with self.assertRaises(RuntimeError):
            self.network(buffer=512)
        with self.assertRaises(ValueError):
            self.network(quantum=15)

    def test_non_divisible_link_width_rounds_aggregation_down(self):
        config = self.config()
        config['link_width_bits'] = 384 * 8
        network = PacketNetwork(config)
        self.addCleanup(network.close)
        self.assertEqual(network.config['packet_quantum_bytes'], 768)
        self.assertEqual(network.config['requested_packet_quantum_bytes'], 1024)
        self.submit(network, 1, 0, 3, 1025)
        self.drain(network)
        self.assertEqual(network.statistics()['received_bytes'], 1152)


if __name__ == '__main__':
    unittest.main()
