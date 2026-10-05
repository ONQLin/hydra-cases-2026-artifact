"""HYDRA-Packet serving entry point using the shared static policy contract."""

from Sim.backends.native import NativeSimulationBackend
from Sim.backends.serving import StaticServingMixin


class PacketSimulationBackend(StaticServingMixin, NativeSimulationBackend):
    display_name = 'HYDRA-Packet'
    supports_dma_pacing = True

    @staticmethod
    def get_name():
        return 'hydra_packet'
