"""Nonpreemptive request ordering; estimates are not measured serving times."""

from abc import ABC, abstractmethod

from Sim.config.utils import get_all_subclasses


class BaseAgentPriority(ABC):
    @classmethod
    def create_from_name(cls, name):
        for subclass in get_all_subclasses(cls):
            if subclass.get_name() == name:
                return subclass()
        raise ValueError(f'Unknown agent priority: {name}')

    @staticmethod
    @abstractmethod
    def get_name():
        raise NotImplementedError

    @abstractmethod
    def score(self, request, now, service):
        """Lower values run first; ties use arrival time and request ID."""

    def key(self, request, now, service):
        return self.score(request, now, service), request._arrived_at, request._id


class FCFSPriority(BaseAgentPriority):
    @staticmethod
    def get_name():
        return 'fcfs'

    def score(self, request, now, service):
        return request._arrived_at


class SJFRequestPriority(BaseAgentPriority):
    @staticmethod
    def get_name():
        return 'sjf'

    def score(self, request, now, service):
        return service


class HRRNRequestPriority(BaseAgentPriority):
    @staticmethod
    def get_name():
        return 'hrrn'

    def score(self, request, now, service):
        return -(1 + max(0, now - request._arrived_at) / service)


class EDFRequestPriority(BaseAgentPriority):
    @staticmethod
    def get_name():
        return 'edf'

    def score(self, request, now, service):
        return request.deadline_tick


class LeastSlackRequestPriority(BaseAgentPriority):
    @staticmethod
    def get_name():
        return 'least_slack'

    def score(self, request, now, service):
        # Only the released call is estimated, not unknown future tool/LLM work.
        return request.deadline_tick - now - service
