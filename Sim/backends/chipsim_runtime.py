"""HYDRA serving runtime with isolated CHIPSIM execution feedback."""

from Sim.backends.native import NativeSimulationBackend
from Sim.backends.serving import StaticServingMixin


class ChipsimRuntimeBackend(StaticServingMixin, NativeSimulationBackend):
    display_name = 'CHIPSIM (isolated)'

    @staticmethod
    def get_name() -> str:
        return 'chipsim_runtime'
