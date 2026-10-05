"""HYDRA-Packet: shared serving phases with an in-process C++ network."""

from contextlib import ExitStack
import json
from pathlib import Path

from integrations.packet.network import PacketNetwork
from Sim.execution.native import NativeExecutionBackend
from Sim.execution.network import NetworkExecutionMixin
from Sim.execution.network_system import NetworkSystemSnapshot


class PacketExecutionBackend(NetworkExecutionMixin, NativeExecutionBackend):
    @staticmethod
    def get_name():
        return 'hydra_packet'

    def __init__(self, config):
        super().__init__(config)
        self.network = None
        self.trace = None
        self.calls = 0

    def initialize(self, graph, memory_system, compute_system, mapping):
        self.output_dir = Path(self.config.metrics_config.output_dir).resolve() / 'packet_runtime'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        snapshot = NetworkSystemSnapshot(self.config).export(graph, memory_system, mapping)
        cfg, packet = self.config.chipsim_config, self.config.packet_config
        snapshot.update(
            reuse_mesh_paths=False, packet_quantum_bytes=packet.quantum_bytes,
            packet_buffer_bytes=packet.buffer_bytes, dma_pacing=cfg.dma_pacing,
            dma_burst_bytes=cfg.dma_burst_bytes,
            execution_contract='shared phases; conservative event synchronization; C++ aggregated packet network',
            network_contention='XY routing; directed link and endpoint queues; cut-through; aggregate buffer credits; no per-VC simulation')
        self.memory_bandwidth = snapshot['memory_bandwidth_gbps']
        self.network = PacketNetwork(snapshot)
        snapshot.update(self.network.config)
        snapshot['library_sha256'] = self.network.library_sha256
        (self.output_dir / 'system.json').write_text(json.dumps(snapshot, indent=2))
        self.trace = (self.output_dir / 'execution.jsonl').open('w')

    def _request(self, message):
        self.calls += 1
        message = dict(message, request_id=self.calls)
        result = self.network.execute(message)
        self.trace.write(json.dumps({'request': message, 'result': result}) + '\n')
        return result

    def close(self):
        # Register every release before writing diagnostics: a full disk must
        # not leak the C++ handle or prevent the remaining streams from closing.
        with ExitStack() as resources:
            if self.trace is not None:
                resources.callback(self.trace.close)
                self.trace = None
            network = self.network
            if self.network is not None:
                resources.callback(network.close)
                self.network = None
            resources.callback(super().close)
            if network is not None:
                (self.output_dir / 'packet_statistics.json').write_text(
                    json.dumps(network.statistics(), indent=2))
