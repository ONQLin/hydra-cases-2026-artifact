"""Factory contract for converting recorded benchmark runs into HYDRA sessions."""

from abc import ABC, abstractmethod
from Sim.config.utils import get_all_subclasses


class BaseWorkloadImporter(ABC):
    @classmethod
    def create_from_name(cls, name):
        for subclass in get_all_subclasses(cls):
            if subclass.get_name() == name:
                return subclass
        raise ValueError(f'[{cls.__name__}] Invalid name: {name}')

    @staticmethod
    @abstractmethod
    def get_name():
        raise NotImplementedError

    @abstractmethod
    def load(self, path):
        raise NotImplementedError
