"""Execution backend contract at HYDRA's block and transfer boundaries."""

from abc import ABC, abstractmethod
import simpy
from Sim.config.utils import get_all_subclasses


class BaseExecutionBackend(ABC):
    @classmethod
    def create_from_name(cls, name: str):
        for subclass in get_all_subclasses(cls):
            if subclass.get_name() == name:
                return subclass
        raise ValueError(f"[{cls.__name__}] Invalid name: {name}")

    @staticmethod
    @abstractmethod
    def get_name() -> str:
        raise NotImplementedError

    def initialize(self, graph, memory_system, compute_system, mapping):
        """Bind the backend to the already placed HYDRA system."""

    def create_environment(self):
        return simpy.Environment(initial_time=0)

    @abstractmethod
    def start_block(self, env, *args):
        """Return a completion event and utilization for an admitted block."""
        raise NotImplementedError

    def start_transfer(self, env, *args):
        return env.timeout(int(self.transfer(*args)))

    def transfer(self, source, destination, size_mb, bandwidth, native_ticks):
        return native_ticks

    def close(self):
        """Release resources after success or partial initialization failure.

        Repeated calls must be safe; diagnostic writes must not prevent release.
        """
