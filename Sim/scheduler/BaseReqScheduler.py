from abc import ABC, abstractmethod
from Sim.config.utils import get_all_subclasses
from typing import Any


# Define the BaseReqScheduler abstract base class for request scheduling 
class BaseReqScheduler(ABC):
    dispatch_ready_on_admission = False

    def configure_runtime(self, env, config):
        """Optional runtime policy context; existing schedulers keep defaults."""

    def create_processing(self, execution_backend):
        """Create an executor with the lifecycle required by this scheduler."""
        from Sim.processing import processing
        return processing(execution_backend)

    def write_report(self, output_dir):
        """Optional scheduler-specific execution evidence."""

    def wait_for_work(self, env):
        import Sim.common as common
        return env.timeout(common.simulation_clk)

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
