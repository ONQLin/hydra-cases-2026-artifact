"""CHIPSIM execution objects used by the persistent runtime service."""

import gzip
import hashlib
import json
import math
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
from src.sim.communication_simulator import CommunicationSimulator
from src.integrations.hydra.compute_backend import HydraComputeBackend


class RuntimeCommunicationSimulator(CommunicationSimulator):
    """CHIPSIM Garnet transport with isolated files and strict completion."""

    def __init__(self, config, system_path):
        self.config = config
        self.output_dir = system_path.parent
        self.current_dir = self.output_dir
        self.topology_path = self.output_dir / 'topology.yaml'
        self.simulations = 0
        self.cache_hits = 0
        self.results = {}
        system = SimpleNamespace(num_chiplets=len(self.config['chiplet_ids']),
                                 adj_matrix=np.array(self.config['adjacency']))
        self.mesh_columns = system.num_chiplets // config['mesh_rows']
        self.reuse_mesh_paths = config.get('reuse_mesh_paths', False)
        if self.reuse_mesh_paths:
            # With one flow, identical routers/links, and shortest-path routing,
            # unused mesh branches cannot affect serialization or credit return.
            # Validate the full mesh before merging equivalent endpoint pairs.
            for source in range(system.num_chiplets):
                for destination in range(system.num_chiplets):
                    adjacent = self._hops(source, destination) == 1
                    if bool(system.adj_matrix[source, destination]) != adjacent:
                        raise ValueError('Equivalent-path caching requires a full homogeneous mesh.')
        super().__init__(system, enable_dsent=False, enable_cache=False,
                         cache_file=str(self.output_dir / 'unused_cache.pkl'),
                         network_operation_frequency=self.config['network_frequency_hz'],
                         gem5_sim_cycles=self.config['garnet_sim_cycles'],
                         gem5_ticks_per_cycle=self.config['garnet_ticks_per_cycle'])

    def _convert_adj_matrix_to_yaml_topology(self, adjacency, ignored_path):
        return super()._convert_adj_matrix_to_yaml_topology(adjacency, str(self.topology_path))

    def _prepare_garnet_traffic_files(self, matrices, simulation_type):
        path = self.current_dir / 'traffic.txt'
        rows = ['# cycle i_ni i_router o_ni o_router vnet flits network_idx input_idx phase_id']
        for matrix, network_id, input_id, phase_id in matrices:
            for source, destination in zip(*np.nonzero(matrix)):
                rows.append(f'1 {source} {source} {destination + self.system.num_chiplets} {destination} 0 {int(matrix[source, destination])} {network_id} {input_id} {phase_id}')
        path.write_text('\n'.join(rows) + '\n')
        compressed = path.with_suffix('.gz')
        with gzip.open(compressed, 'wt') as stream:
            stream.write(path.read_text())
        return str(path), str(compressed)

    def _build_garnet_simulation_command(self, traffic_file, max_packets):
        self.expected_packets = max_packets
        root = Path(self.gem5_path)
        frequency = self.config['network_frequency_hz']
        # Set the actual Ruby clock and use its tick conversion in CHIPSIM.
        ticks = 1e12 / frequency
        if not math.isclose(ticks, self.config['garnet_ticks_per_cycle']):
            raise ValueError('garnet_ticks_per_cycle must equal 1e12 / network_frequency_hz.')
        return [str(root / 'build/Garnet_standalone/gem5.opt'),
                '--outdir=' + str(self.current_dir / 'm5out'),
                str(root / 'configs/example/garnet_synth_traffic.py'),
                f'--num-cpus={self.system.num_chiplets}', f'--num-dirs={self.system.num_chiplets}',
                '--topology=AnyNET_XY', '--config-file=' + str(self.topology_path),
                f'--mesh-rows={self.config["mesh_rows"]}',
                f'--link-width-bits={self.config["link_width_bits"]}',
                f'--router-latency={self.config["router_latency_cycles"]}',
                f'--link-latency={self.config["link_latency_cycles"]}',
                f'--vcs-per-vnet={self.config["virtual_channels_per_vnet"]}',
                f'--ruby-clock={frequency}Hz', f'--sys-clock={frequency}Hz',
                f'--sim-cycles={self.gem5_sim_cycles}', '--injectionrate=0',
                '--network-trace-enable', '--network-trace-file=' + traffic_file,
                f'--network-trace-max-packets={max_packets - 1}']

    def _run_garnet_simulation(self, command):
        (self.current_dir / 'command.json').write_text(json.dumps(command, indent=2))
        with (self.current_dir / 'gem5.log').open('w') as log:
            result = subprocess.run(command, cwd=self.gem5_path, stdout=log,
                                    stderr=subprocess.STDOUT,
                                    timeout=self.config['timeout_seconds'] or None)
        if result.returncode:
            raise RuntimeError(f'Garnet failed; see {self.current_dir / "gem5.log"}')
        stats = self._extract_garnet_stats(str(self.current_dir / 'm5out/stats.txt'))
        if stats.get('packets_received') != self.expected_packets or stats.get('packets_injected') != self.expected_packets:
            raise RuntimeError(f'Incomplete Garnet transfer: expected {self.expected_packets}, received {stats.get("packets_received")}. See {self.current_dir}')
        # simTicks includes drain and startup, unlike average per-packet latency.
        if stats.get('simTicks', 0) <= 0:
            raise RuntimeError('Garnet returned no elapsed simulation time.')
        return stats

    def transfer(self, source, destination, byte_count):
        if byte_count <= 0 or source == destination:
            return 0.0
        packets = math.ceil(byte_count / (self.config['link_width_bits'] / 8))
        key = (source, destination, packets)
        if self.reuse_mesh_paths:
            key = ('mesh_hops', self._hops(source, destination), packets)
        if key in self.results:
            self.cache_hits += 1
            return self.results[key]
        digest = hashlib.sha256(json.dumps(key).encode()).hexdigest()[:20]
        self.current_dir = self.output_dir / 'garnet' / digest
        self.current_dir.mkdir(parents=True, exist_ok=True)
        matrix = np.zeros((self.system.num_chiplets, self.system.num_chiplets), dtype=np.int64)
        matrix[source, destination] = packets
        stats = self.simulate_communication([(matrix, 0, 0, 0)])
        latency = stats['latency']['total_runtime_us']
        self.simulations += 1
        self.results[key] = latency
        (self.current_dir / 'result.json').write_text(json.dumps({
            'source': source, 'destination': destination, 'packets': packets,
            'latency_us': latency, 'received': stats['latency']['packets_received'],
        }, indent=2))
        return latency

    def _hops(self, source, destination):
        return (abs(source // self.mesh_columns - destination // self.mesh_columns)
                + abs(source % self.mesh_columns - destination % self.mesh_columns))

class RuntimeExecutionService:
    def close(self):
        """Isolated transfers have no persistent child process."""

    def __init__(self, config, system_path):
        self.config = config
        self.communication = RuntimeCommunicationSimulator(config, system_path)
        self.compute = HydraComputeBackend()

    def execute(self, request):
        bandwidth = request['bandwidth_bytes_s']
        if bandwidth <= 0:
            raise ValueError('Reserved bandwidth must be positive.')
        latency_us = 0.0
        phases = []
        for phase in request['phases']:
            memory_bytes = phase['memory_bytes']
            if memory_bytes < 0 or phase['compute_ns'] < 0:
                raise ValueError('Execution phases cannot have negative cost.')
            profile = {'hydra_compute_profiles': {'HYDRA': {
                'latency_us': phase['compute_ns'] / 1000,
                'energy_fj': 0, 'frequency_hz': 1e9}}}
            compute_us = self.compute.simulate(profile, request['destination'], 'HYDRA')['latency_us']
            endpoints = (request['source'], request['destination'])
            if phase.get('memory_direction', 'read') == 'write':
                endpoints = endpoints[::-1]
            noi_us = self.communication.transfer(*endpoints, memory_bytes)
            hbm_us = memory_bytes / bandwidth * 1e6
            if memory_bytes:
                hbm_us += self.config['hbm_access_latency_ns'] / 1000
            duration = max(compute_us, hbm_us, noi_us)
            latency_us += duration
            phases.append({'compute_us': compute_us, 'hbm_us': hbm_us,
                           'noi_us': noi_us, 'latency_us': duration})
        return {'request_id': request['request_id'], 'latency_us': latency_us,
                'phases': phases, 'garnet_simulations': self.communication.simulations,
                'cache_hits': self.communication.cache_hits}
