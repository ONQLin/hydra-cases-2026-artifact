"""Deterministic synthetic expert routes; no claim to checkpoint gate outputs."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import hashlib
import json

from Sim.config.utils import get_all_subclasses


@dataclass(frozen=True)
class ExpertRoutePlan:
    expert_ids: tuple
    num_experts: int
    top_k: int
    source: str

    def __post_init__(self):
        if self.num_experts < 1 or not 0 < self.top_k <= self.num_experts or not self.expert_ids:
            raise ValueError('Invalid expert route dimensions.')
        for row in self.expert_ids:
            if len(row) != self.top_k or len(set(row)) != self.top_k:
                raise ValueError('Every token must select top_k distinct experts.')
            if any(type(expert) is not int or not 0 <= expert < self.num_experts for expert in row):
                raise ValueError('Expert ID outside the configured range.')

    @property
    def counts(self):
        counts = [0]*self.num_experts
        for row in self.expert_ids:
            for expert in row:
                counts[expert] += 1
        return counts

    def export(self):
        return dict(source=self.source, tokens=len(self.expert_ids), top_k=self.top_k,
                    tokens_per_expert=self.counts,
                    route_sha256=hashlib.sha256(json.dumps(self.expert_ids).encode()).hexdigest())


@dataclass(frozen=True)
class GroupedExpertRoutePlan(ExpertRoutePlan):
    num_groups: int
    top_k_groups: int

    def __post_init__(self):
        super().__post_init__()
        if (type(self.num_groups) is not int or type(self.top_k_groups) is not int
                or self.num_groups < 1 or self.num_experts % self.num_groups
                or not 0 < self.top_k_groups <= self.num_groups):
            raise ValueError('Invalid grouped route dimensions.')
        width = self.num_experts//self.num_groups
        if any(len({expert//width for expert in row}) > self.top_k_groups for row in self.expert_ids):
            raise ValueError('Expert route exceeds the selected-group limit.')

    def export(self):
        return dict(super().export(),num_groups=self.num_groups,top_k_groups=self.top_k_groups)


class BaseExpertRouting(ABC):
    grouped = False
    @staticmethod
    @abstractmethod
    def get_name():
        raise NotImplementedError

    @abstractmethod
    def first_expert(self, token, batch_slot, config):
        raise NotImplementedError

    @classmethod
    def create_from_name(cls, name):
        for subtype in get_all_subclasses(cls):
            if subtype.get_name() == name:
                return subtype()
        raise ValueError(f'Unknown expert routing policy {name}.')

    def route(self, config, tokens, batch_size, token_offset):
        if min(tokens,batch_size) < 1 or token_offset < 0:
            raise ValueError('Invalid route token range.')
        rows = []
        for batch in range(batch_size):
            for token in range(token_offset,token_offset+tokens):
                rows.append(self.select_experts(token,batch,config))
        return self.create_plan(tuple(rows),config)

    def select_experts(self, token, batch_slot, config):
        first = self.first_expert(token,batch_slot,config)
        return tuple((first+rank)%config.num_experts for rank in range(config.top_k))

    def create_plan(self, rows, config):
        return ExpertRoutePlan(rows,config.num_experts,config.top_k,'synthetic-'+self.get_name())


class CyclicExpertRouting(BaseExpertRouting):
    @staticmethod
    def get_name():
        return 'cyclic'

    def first_expert(self, token, batch_slot, config):
        return (token+104729*batch_slot+config.routing_seed)*config.top_k


class HotspotExpertRouting(BaseExpertRouting):
    @staticmethod
    def get_name():
        return 'hotspot'

    def first_expert(self, token, batch_slot, config):
        return config.routing_seed%config.num_experts


class GroupedCyclicExpertRouting(CyclicExpertRouting):
    """Synthetic group-limited assignments, not inferred checkpoint gate outputs."""
    grouped = True

    @staticmethod
    def get_name():
        return 'grouped_cyclic'

    def first_expert(self, token, batch_slot, config):
        return token+104729*batch_slot+config.routing_seed

    def select_experts(self, token, batch_slot, config):
        position = self.first_expert(token,batch_slot,config)
        width = config.num_experts//config.num_groups
        first_group = position%config.num_groups
        first_local = (position//config.num_groups)%width
        return tuple(((first_group+rank%config.top_k_groups)%config.num_groups)*width
                     +(first_local+rank//config.top_k_groups)%width for rank in range(config.top_k))

    def create_plan(self, rows, config):
        return GroupedExpertRoutePlan(rows,config.num_experts,config.top_k,'synthetic-'+self.get_name(),
                                      config.num_groups,config.top_k_groups)


class GroupedHotspotExpertRouting(GroupedCyclicExpertRouting):
    @staticmethod
    def get_name():
        return 'grouped_hotspot'

    def first_expert(self, token, batch_slot, config):
        return config.routing_seed
