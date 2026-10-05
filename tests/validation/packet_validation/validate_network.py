#!/usr/bin/env python3
"""Compare the C++ packet network with real Garnet on controlled traffic."""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from integrations.packet.network import PacketNetwork


@dataclass(frozen=True)
class NetworkCase:
    name: str
    arrivals: tuple
    size: int = 16384
    dma: bool = False


class NetworkValidation:
    def __init__(self, args):
        self.args = args
        self.output = args.output_dir.resolve()
        self.output.mkdir(parents=True, exist_ok=True)

    def config(self, quantum, dma):
        return dict(chiplet_ids=[0, 1, 2, 3], mesh_rows=2,
                    adjacency=[[0, 1, 1, 0], [1, 0, 0, 1], [1, 0, 0, 1], [0, 1, 1, 0]],
                    network_frequency_hz=1e9, garnet_ticks_per_cycle=1000,
                    garnet_sim_cycles=100_000_000, link_width_bits=128,
                    router_latency_cycles=1, link_latency_cycles=1,
                    virtual_channels_per_vnet=8, reuse_mesh_paths=False,
                    memory_bandwidth_gbps={'0': 1, '2': 1},
                    hbm_access_latency_ns=50, timeout_seconds=120,
                    packet_quantum_bytes=quantum, packet_buffer_bytes=16384,
                    dma_pacing=dma, dma_burst_bytes=1024)

    def run_case(self, case, backend, quantum):
        config = self.config(quantum, case.dma)
        directory = self.output / case.name / (f'packet_{quantum}' if backend == 'hydra_packet' else 'garnet')
        directory.mkdir(parents=True, exist_ok=True)
        system = directory / 'system.json'
        system.write_text(json.dumps(config, indent=2))
        started = time.perf_counter()
        if backend == 'hydra_packet':
            network = PacketNetwork(config)
        else:
            import integrations
            integrations.__path__.append(str(ROOT / 'third_party/CHIPSIM/integrations'))
            sys.path.insert(0, str(ROOT / 'third_party/CHIPSIM'))
            sys.path.insert(0, str(ROOT / 'integrations/chipsim'))
            from persistent_network import PersistentGarnetNetwork
            network = PersistentGarnetNetwork(config, system)
        ready = time.perf_counter()
        now, completed = 0, {}

        def advance(target):
            nonlocal now
            result = network.execute(dict(kind='advance', target_tick=target))
            if not now <= result['tick'] <= target:
                raise RuntimeError('Network time escaped the requested window')
            now = result['tick']
            values = result['completed']
            for flow, tick in zip(values[::2], values[1::2]):
                if flow in completed:
                    raise RuntimeError('Duplicate network completion')
                completed[flow] = tick

        try:
            for flow, (arrival, source, destination) in enumerate(case.arrivals, 1):
                while now < arrival:
                    advance(arrival)
                network.execute(dict(kind='submit', flow_id=flow, source=source,
                                     destination=destination, bytes=case.size))
            while len(completed) < len(case.arrivals):
                advance(100_000_000_000)
                if now >= 100_000_000_000:
                    raise RuntimeError('Network did not drain before the deadline')
            finished = time.perf_counter()
            stats = network.statistics() if backend == 'hydra_packet' else None
        finally:
            network.close()
        if backend == 'hydra_packet':
            assert stats['received_bytes'] == case.size * len(case.arrivals)
        else:
            stats = network._extract_garnet_stats(str(directory / 'garnet_online/m5out/stats.txt'))
            assert stats['packets_received'] == case.size // 16 * len(case.arrivals)
        result = dict(completion_us={str(k): v / 1e6 for k, v in completed.items()},
                      startup_wall_s=ready - started, simulation_wall_s=finished - ready,
                      total_wall_s=time.perf_counter() - started)
        if backend == 'hydra_packet':
            result['statistics'] = stats
            result['library_sha256'] = network.library_sha256
        (directory / 'result.json').write_text(json.dumps(result, indent=2))
        return result

    def run(self):
        cases = [
            NetworkCase('isolated', ((0, 0, 3),)),
            NetworkCase('shared_destination', ((0, 0, 3), (0, 1, 3))),
            NetworkCase('staggered', ((0, 0, 3), (250_000, 1, 3))),
            NetworkCase('disjoint', ((0, 0, 1), (0, 2, 3))),
            NetworkCase('shared_source', ((0, 0, 1), (0, 0, 3))),
            NetworkCase('dma_single', ((0, 0, 1),), dma=True),
            NetworkCase('dma_shared', ((0, 0, 1), (0, 0, 3)), dma=True),
            NetworkCase('dma_independent', ((0, 0, 1), (0, 2, 3)), dma=True),
            NetworkCase('large_shared', ((0, 0, 3), (0, 1, 3)), size=1024**2),
        ]
        if self.args.stress:
            cases.extend([
                NetworkCase('hotspot_16', tuple((0, i % 3, 3) for i in range(16))),
                NetworkCase('bidirectional', ((0, 0, 3), (0, 3, 0), (0, 1, 2), (0, 2, 1))),
                NetworkCase('staggered_hotspot', tuple((i * 125_000, i % 3, 3) for i in range(16))),
                NetworkCase('dma_hotspot', tuple((0, 0 if i % 2 else 2, 3) for i in range(8)), dma=True),
            ])
        results = {}
        for case in cases:
            results[case.name] = panel = {}
            for quantum in self.args.quanta:
                panel[f'packet_{quantum}'] = self.run_case(case, 'hydra_packet', quantum)
            if self.args.garnet:
                panel['garnet'] = self.run_case(case, 'garnet', 1024)
                for quantum in self.args.quanta:
                    measured = panel[f'packet_{quantum}']
                    measured['max_completion_relative_error'] = max(
                        abs(value / panel['garnet']['completion_us'][flow] - 1)
                        for flow, value in measured['completion_us'].items())
                    measured['simulation_speedup_vs_garnet'] = panel['garnet']['simulation_wall_s'] / measured['simulation_wall_s']
            print(case.name, {k: v['completion_us'] for k, v in panel.items()}, flush=True)
            (self.output / 'comparison.json').write_text(json.dumps(results, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--garnet', action='store_true')
    parser.add_argument('--stress', action='store_true', help='Include sustained fan-in, bidirectional, and DMA contention.')
    parser.add_argument('--quanta', nargs='+', type=int, default=[256, 1024, 4096])
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'output_sanity_checks/packet_validation/network')
    NetworkValidation(parser.parse_args()).run()
