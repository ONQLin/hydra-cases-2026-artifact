from typing import Tuple, List
import networkx as nx
import numpy as np
from Sim.config.model_config import BaseModelConfig
from Sim.entities.infer import infer
from Sim.entities.block import block
from Sim.entities.base_entity import BaseEntity
from Sim.metrics.monitor import tokens_monitor
from Sim.entities.request import Request

class BatchOfRequests(BaseEntity):
    def __init__(self, requests: List[Request]):
        self._id = -1
        for req in requests:
            req.set_batch_id(self._id)
        self.ongoing_requests = requests
        self.completed_requests = []
        self._batch_size = len(requests)
        self._processing_target = -1

    def step_batch_id(self):
        self._id = self.generate_id()
    # @classmethod
    # def generate_id(cls):
    #     cls._id += 1
    #     return cls._id
    def remove_ongoing_request(self, req_id: int):
        self.ongoing_requests = [req for req in self.ongoing_requests if req._id != req_id]
    
    def add_request(self, request: Request):
        self.ongoing_requests.append(request)

    def remove_request(self, request: Request):
        self.ongoing_requests.remove(request)

    def update_requests(self):
        for req in self.ongoing_requests:
            if req._completed_at != 0 and req not in self.completed_requests:
                self.completed_requests.append(req)
                self.ongoing_requests.remove(req)
                
    def find_request(self, req_id: int) -> Request:
        for req in self.ongoing_requests:
            if req._id == req_id:
                return req
        raise ValueError(f"Request with ID {req_id} not found in ongoing requests.")