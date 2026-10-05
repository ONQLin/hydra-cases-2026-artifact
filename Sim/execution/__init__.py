"""Register execution implementations with the HYDRA factory."""

from Sim.execution.BaseExecutionBackend import BaseExecutionBackend
from Sim.execution.native import NativeExecutionBackend
from Sim.execution.chipsim import ChipsimExecutionBackend
from Sim.execution.contended import ContendedExecutionBackend
from Sim.execution.packet import PacketExecutionBackend
