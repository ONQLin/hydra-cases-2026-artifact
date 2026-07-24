from abc import ABC, abstractmethod
from Sim.config.utils import get_all_subclasses
from typing import Any


# Define the BaseReqScheduler abstract base class for request scheduling 
class BaseReqScheduler(ABC):

    @classmethod
    def create_from_name(cls, name: str) -> Any:
        for subclass in get_all_subclasses(cls):
            if subclass.get_name() == name:
                return subclass
        raise ValueError(f"[{cls.__name__}] Invalid name: {name}")
    
    @abstractmethod
    def set_max_concurrent_requests(self, max_requests: int):
        """
        Sets the maximum number of concurrent requests allowed.
        Must be implemented by subclasses.
        """
        raise NotImplementedError

    @abstractmethod
    def schedule_request(self, request_data: dict) -> int:
        """
        Schedules a new request.
        Must be implemented by subclasses.
        """
        raise NotImplementedError

    @abstractmethod
    def start_request(self, request_id: int):
        """
        Initiates the processing of a scheduled request.
        Must be implemented by subclasses.
        """
        raise NotImplementedError
    
    @abstractmethod
    def get_name() -> str:
        """
        Returns the name of the scheduler.
        Must be implemented by subclasses.
        """
        raise NotImplementedError

    def __str__(self) -> str:
        # use to_dict to get a dict representation of the object
        # and convert it to a string
        class_name = self.__class__.__name__
        return f"{class_name}({str(self.to_dict())})"
