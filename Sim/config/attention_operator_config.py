"""Explicit KDA/MLA algorithm and state contracts, independent of full models."""

from dataclasses import dataclass, field
from Sim.config.model_config import BaseBlockConfig, baselayerConfig


@dataclass
class KDAConfig(baselayerConfig):
    operator_name = 'ModernKDA'
    cache_store = False
    embedding_dim: int = 2304
    num_heads: int = 32
    head_dim: int = 128
    conv_kernel_size: int = 4
    state_bytes: int = 4
    algorithm: str = 'recurrent'
    state_mapping: str = 'auto'
    chunk_size: int = 64
    head_tile_size: int = 1
    value_tile_size: int = 0

    def __post_init__(self):
        if any(type(v) is not int or v <= 0 for v in
               (self.embedding_dim, self.num_heads, self.head_dim, self.conv_kernel_size)):
            raise ValueError('KDA dimensions must be positive integers.')
        if self.state_bytes != 4 or self.algorithm not in ('recurrent', 'chunk'):
            raise ValueError('KDA supports FP32 state with recurrent or chunk execution.')
        if self.state_mapping not in ('auto', 'whole', 'tiled'):
            raise ValueError('Unknown KDA state mapping.')
        if any(type(v) is not int or v < 1 for v in (self.chunk_size, self.head_tile_size)):
            raise ValueError('Chunk and head tile sizes must be positive integers.')
        if type(self.value_tile_size) is not int or not 0 <= self.value_tile_size <= self.head_dim:
            raise ValueError('Value tile size must be zero (full width) or at most head_dim.')
        if self.algorithm == 'chunk' and self.state_mapping == 'whole':
            raise ValueError('Chunk KDA requires the tiled state mapping.')

    @property
    def tiled(self):
        return self.state_mapping == 'tiled' or self.algorithm == 'chunk'

    @property
    def state_size_bytes(self):
        h, k = self.num_heads, self.head_dim
        return h*k*k*self.state_bytes + 3*h*k*self.conv_kernel_size

    @property
    def parameter_count(self):
        d, h, k = self.embedding_dim, self.num_heads, self.head_dim
        # Q/K/V/O; separate low-rank decay/output gates; beta; convolution;
        # A_log, dt_bias and per-head RMSNorm scale. Storage is the W8 surrogate.
        return 4*d*h*k + 2*(d*k+k*h*k) + d*h + 3*h*k*self.conv_kernel_size + h+h*k+k


@dataclass
class MLAConfig(baselayerConfig):
    operator_name = 'ModernMLA'
    cache_store = True
    embedding_dim: int = 7168
    num_heads: int = 128
    q_lora_rank: int = 1536
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    cache_layout: str = 'absorbed'
    rope_enabled: bool = True
    algorithm: str = 'materialized'
    query_tile_size: int = 64
    key_tile_size: int = 128
    head_tile_size: int = 1

    def __post_init__(self):
        positive = (self.embedding_dim, self.num_heads, self.kv_lora_rank,
                    self.qk_nope_head_dim, self.v_head_dim)
        if any(type(v) is not int or v <= 0 for v in positive):
            raise ValueError('MLA dimensions must be positive integers.')
        if any(type(v) is not int or v < 0 for v in (self.q_lora_rank, self.qk_rope_head_dim)):
            raise ValueError('MLA optional dimensions must be nonnegative integers.')
        if self.qk_rope_head_dim % 2 or self.cache_layout not in ('absorbed', 'expanded'):
            raise ValueError('Invalid MLA positional dimension or cache layout.')
        if self.algorithm not in ('materialized', 'streaming'):
            raise ValueError('Unknown MLA algorithm.')
        if any(type(v) is not int or v <= 0 for v in
               (self.query_tile_size, self.key_tile_size, self.head_tile_size)):
            raise ValueError('MLA tile sizes must be positive integers.')

    @property
    def cache_bytes_per_token(self):
        if self.cache_layout == 'absorbed':
            return self.kv_lora_rank + self.qk_rope_head_dim
        return self.num_heads*(self.qk_nope_head_dim+self.qk_rope_head_dim+self.v_head_dim)

    @property
    def parameter_count(self):
        d, h, c = self.embedding_dim, self.num_heads, self.kv_lora_rank
        q = h*(self.qk_nope_head_dim+self.qk_rope_head_dim)
        query = (d*self.q_lora_rank+self.q_lora_rank+self.q_lora_rank*q
                 if self.q_lora_rank else d*q)
        return (query+d*(c+self.qk_rope_head_dim)+c
                +c*h*(self.qk_nope_head_dim+self.v_head_dim)+h*self.v_head_dim*d)


@dataclass(frozen=True)
class OperatorMemory:
    state_bytes: int
    activation_bytes: int
    output_bytes: int


@dataclass
class OperatorBlockConfig(BaseBlockConfig):
    operator: baselayerConfig = field(default_factory=KDAConfig)
    kernel: str = 'ModernKDA'
    execution_family: str = 'recurrent'
    type_name: str = 'kda-operator'
    memory_accounting_version: int = 3
    max_position_embeddings: int = 4096

    def __post_init__(self):
        from analytic_profile.modern import BaseOperatorProfile
        profile = BaseOperatorProfile.find(self.kernel)
        if profile is None:
            raise ValueError(f'Unknown operator {self.kernel}.')
        if not isinstance(self.operator, getattr(profile, 'config_type', type(None))):
            raise ValueError('Operator configuration does not match the profile kernel.')
        family = profile.get_execution_family(self.operator)
        if self.execution_family != family:
            raise ValueError('Operator execution family does not match the profile kernel.')
        self.embedding_dim = self.operator.embedding_dim
        self.parameter_count = self.operator.parameter_count
        self.cache_store = self.operator.cache_store
        self.states = (self.operator.cache_bytes_per_token if self.cache_store
                       else self.operator.state_size_bytes)
        self.output_mem = self.embedding_dim
        self.layers = [[self.kernel]]
        self.layer_configs = [[self.operator]]
        memory = self.memory_requirements(1, False)
        self.intermed_mem = self.peak_intermed = memory.activation_bytes

    def memory_requirements(self, context_length, is_prefill):
        from analytic_profile.modern import BaseOperatorProfile
        tokens = context_length if is_prefill else 1
        graph = BaseOperatorProfile.find(self.kernel)().graph(
            **dict(vars(self.operator), bs=tokens, batch_size=1,
                   L_seq=context_length, stage='prefill' if is_prefill else 'decode'))
        return OperatorMemory(self.states*context_length if self.cache_store else self.states,
                              graph.peak_activation_bytes(), tokens*self.embedding_dim)

    def reservation_bytes(self, context_length, is_prefill):
        memory = self.memory_requirements(context_length, is_prefill)
        return memory.state_bytes + memory.activation_bytes

    @staticmethod
    def get_name():
        return 'dependency-aware-operator-block'
