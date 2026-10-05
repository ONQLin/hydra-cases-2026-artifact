"""Shape-compatible static batches with bounded collection delay."""

from abc import ABC, abstractmethod
from Sim.config.utils import get_all_subclasses


class BaseAgentBatching(ABC):
    @classmethod
    def create_from_name(cls, name):
        for subclass in get_all_subclasses(cls):
            if subclass.get_name() == name:
                return subclass()
        raise ValueError(f'Unknown agent batching policy: {name}')

    @staticmethod
    @abstractmethod
    def get_name():
        raise NotImplementedError

    @abstractmethod
    def ready(self, requests, target_size, now, wait_ticks):
        raise NotImplementedError


class ImmediateAgentBatching(BaseAgentBatching):
    @staticmethod
    def get_name():
        return 'immediate'

    def ready(self, requests, target_size, now, wait_ticks):
        return bool(requests)


class TimeoutAgentBatching(BaseAgentBatching):
    @staticmethod
    def get_name():
        return 'timeout'

    def ready(self, requests, target_size, now, wait_ticks):
        return bool(requests) and (len(requests) >= target_size or
            now - min(request._arrived_at for request in requests) >= wait_ticks)
