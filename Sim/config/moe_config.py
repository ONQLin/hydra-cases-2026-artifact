"""Explicit MoE storage/routing contract for a colocated expert baseline."""

from dataclasses import dataclass, field
import math

from Sim.config.model_config import baselayerConfig
from Sim.config.attention_operator_config import KDAConfig
from Sim.config.modern_model_config import SwiGLUConfig


@dataclass
class MoEConfig(baselayerConfig):
    embedding_dim: int = 2304
    intermediate_dim: int = 1024
    num_experts: int = 256
    top_k: int = 8
    shared_experts: int = 1
    routing_policy: str = 'cyclic'
    routing_seed: int = 0
    expert_placement: str = 'colocated'
    operator_name = 'ModernMoE'

    def __post_init__(self):
        if any(type(v) is not int or v < 1 for v in
               (self.embedding_dim,self.intermediate_dim,self.num_experts,self.top_k)):
            raise ValueError('MoE dimensions must be positive integers.')
        if self.top_k > self.num_experts or type(self.shared_experts) is not int or self.shared_experts < 0:
            raise ValueError('Invalid routed/shared expert counts.')
        if type(self.routing_seed) is not int or self.routing_seed < 0:
            raise ValueError('Routing seed must be a nonnegative integer.')
        if self.expert_placement != 'colocated':
            raise ValueError('This MoE baseline requires colocated experts; expert parallelism is not modeled.')
        from Sim.entities.expert_routing import BaseExpertRouting
        routing = BaseExpertRouting.create_from_name(self.routing_policy)
        if routing.grouped != isinstance(self, GroupedMoEConfig):
            raise ValueError('Routing policy must match the grouped/ungrouped MoE contract.')

    @property
    def parameter_count(self):
        # All experts are resident, regardless of the current routed token count.
        return (self.embedding_dim*self.num_experts+self.num_experts
                +3*self.embedding_dim*self.intermediate_dim*(self.num_experts+self.shared_experts))


@dataclass
class GroupedMoEConfig(MoEConfig):
    """DeepSeek-V3 sigmoid/correction-bias router with group-limited top-k."""
    num_groups: int = 8
    top_k_groups: int = 4
    group_score_top_k: int = 2
    routed_scaling_factor: float = 2.5
    routing_policy: str = 'grouped_cyclic'
    operator_name = 'ModernGroupedMoE'

    def __post_init__(self):
        super().__post_init__()
        if any(type(v) is not int or v < 1 for v in
               (self.num_groups,self.top_k_groups,self.group_score_top_k)):
            raise ValueError('Group dimensions must be positive integers.')
        if self.num_experts % self.num_groups or self.top_k_groups > self.num_groups:
            raise ValueError('Experts must divide evenly into valid selected groups.')
        width = self.num_experts//self.num_groups
        if self.group_score_top_k > width or self.top_k > self.top_k_groups*width:
            raise ValueError('Group selection cannot supply the configured expert top-k.')
        if not math.isfinite(self.routed_scaling_factor) or self.routed_scaling_factor <= 0:
            raise ValueError('Routing scale must be finite and positive.')


@dataclass
class DecoderLayerConfig(baselayerConfig):
    attention: baselayerConfig = field(default_factory=lambda: KDAConfig(algorithm='chunk'))
    feed_forward: baselayerConfig = field(default_factory=MoEConfig)
    operator_name = 'ModernDecoderBlock'

    def __post_init__(self):
        if self.attention.embedding_dim != self.feed_forward.embedding_dim:
            raise ValueError('Attention and FFN hidden dimensions must agree.')
        if not isinstance(self.feed_forward, (MoEConfig,SwiGLUConfig)):
            raise ValueError('Unsupported decoder feed-forward configuration.')

    @property
    def embedding_dim(self):
        return self.attention.embedding_dim

    @property
    def parameter_count(self):
        return self.attention.parameter_count+self.feed_forward.parameter_count+2*self.embedding_dim

    @property
    def cache_store(self):
        return self.attention.cache_store

    @property
    def state_size_bytes(self):
        return self.attention.state_size_bytes

    @property
    def cache_bytes_per_token(self):
        return self.attention.cache_bytes_per_token
