"""Static HYDRA execution with a persistent CHIPSIM/Garnet transport."""

from Sim.execution.chipsim import ChipsimExecutionBackend
from Sim.execution.network import NetworkExecutionMixin, NetworkCompletion, CoSimulationEnvironment


class ContendedExecutionBackend(NetworkExecutionMixin, ChipsimExecutionBackend):
    service_filename = 'contended_service.py'

    @staticmethod
    def get_name():
        return 'chipsim_contended'

    def configure_system(self, snapshot):
        snapshot['reuse_mesh_paths'] = False
        snapshot['dma_pacing'] = self.config.chipsim_config.dma_pacing
        snapshot['dma_burst_bytes'] = self.config.chipsim_config.dma_burst_bytes
        if snapshot['dma_pacing'] and snapshot['dma_burst_bytes'] < snapshot['link_width_bits'] / 8:
            raise ValueError('DMA burst must hold at least one network flit.')
        snapshot['dma_contract'] = ('shared physical HBM bandwidth, round-robin bursts, one access delay per flow'
                                    if snapshot['dma_pacing'] else 'unpaced phase bursts')
        snapshot['execution_contract'] = 'persistent shared Garnet network; conservative event synchronization; shared compute/HBM phases'
        snapshot['network_contention'] = 'all admitted flows share packets, buffers, links, VCs and credits; no latency cache'
