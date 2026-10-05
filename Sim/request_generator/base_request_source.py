"""Factory for independent request traces and dependent session replay."""

from abc import ABC, abstractmethod
from Sim.config.utils import get_all_subclasses


class BaseRequestSource(ABC):
    @classmethod
    def create_from_name(cls, name):
        from Sim.request_generator import trace_request_generator, agent_request_generator
        for subclass in get_all_subclasses(cls):
            if subclass.get_name() == name:
                return subclass
        raise ValueError(f'[{cls.__name__}] Invalid name: {name}')

    @staticmethod
    @abstractmethod
    def get_name():
        raise NotImplementedError

    @abstractmethod
    def export_workload(self, output_dir):
        raise NotImplementedError

    def finalize(self, output_dir, tick):
        """Optionally write workload-specific completion and cutoff metrics."""

    def configure_runtime(self, memory, scheduler_config):
        """Optionally attach placed memory and runtime policy configuration."""
