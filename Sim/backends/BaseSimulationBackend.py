"""Factory for simulation entry points, following HYDRA's policy factories."""

from abc import ABC, abstractmethod
from Sim.config.utils import get_all_subclasses


class BaseSimulationBackend(ABC):
    @classmethod
    def create_from_name(cls, name: str):
        for subclass in get_all_subclasses(cls):
            if subclass.get_name() == name:
                return subclass
        raise ValueError(f"[{cls.__name__}] Invalid name: {name}")

    @classmethod
    def get_display_name(cls):
        return getattr(cls, 'display_name', cls.get_name())

    @staticmethod
    @abstractmethod
    def get_name() -> str:
        raise NotImplementedError

    @abstractmethod
    def run(self):
        raise NotImplementedError
