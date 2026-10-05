"""Static HYDRA serving with shared packet-level network contention."""

from Sim.backends.chipsim_runtime import ChipsimRuntimeBackend


class ChipsimContendedBackend(ChipsimRuntimeBackend):
    display_name = 'CHIPSIM'
    supports_dma_pacing = True

    @staticmethod
    def get_name():
        return 'chipsim_contended'
