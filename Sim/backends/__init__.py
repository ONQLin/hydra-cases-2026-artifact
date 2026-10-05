"""Simulation backend selection through HYDRA's subclass factory convention."""

from Sim.backends.BaseSimulationBackend import BaseSimulationBackend
from Sim.backends.native import NativeSimulationBackend
from Sim.backends.chipsim import ChipsimBackend
from Sim.backends.chipsim_runtime import ChipsimRuntimeBackend
from Sim.backends.chipsim_contended import ChipsimContendedBackend
from Sim.backends.packet import PacketSimulationBackend


def run_simulation(config):
    backend = BaseSimulationBackend.create_from_name(config.simulator_backend)
    return backend(config).run()
