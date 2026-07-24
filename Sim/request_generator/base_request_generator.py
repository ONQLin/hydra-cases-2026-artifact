import json
from abc import ABC, abstractmethod
from typing import List

from Sim.config.sys_config import BaseRequestLengthGeneratorConfig
from Sim.entities.request import Request


class BaseRequestGenerator(ABC):

    def __init__(self, config: BaseRequestLengthGeneratorConfig):
        self.config = config

    @abstractmethod
    def generate_requests(self) -> List[Request]:
        pass

    def generate(self) -> List[Request]:
        requests = self.generate_requests()
        return requests