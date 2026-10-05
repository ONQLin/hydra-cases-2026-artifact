"""Explicit text-decoder shapes for modern models; no model weights are loaded.

Sources and revisions are recorded in docs/workload_modernization/model_inventory.json.
The initial execution contract uses one-byte weights/activations/KV and FP32
recurrent state. It estimates decoder blocks, excluding embedding and LM head.
"""

from dataclasses import dataclass, field

from Sim.config.model_config import BaseBlockConfig, BaseModelConfig, baselayerConfig


@dataclass
class GQAConfig(baselayerConfig):
    embedding_dim: int = 4096
    q_heads: int = 32
    kv_heads: int = 8
    head_dim: int = 128
    rotary_dim: int = 128
    output_gate: bool = False

    @property
    def parameter_count(self):
        q_dim = self.q_heads * self.head_dim
        kv_dim = self.kv_heads * self.head_dim
        return self.embedding_dim * ((2 + self.output_gate) * q_dim + 2 * kv_dim) + 2 * self.head_dim


@dataclass
class GatedDeltaNetConfig(baselayerConfig):
    embedding_dim: int = 4096
    key_heads: int = 16
    value_heads: int = 32
    key_head_dim: int = 128
    value_head_dim: int = 128
    conv_kernel_size: int = 4
    state_bytes: int = 4

    @property
    def conv_dim(self):
        return 2 * self.key_heads * self.key_head_dim + self.value_heads * self.value_head_dim

    @property
    def recurrent_elements(self):
        return self.value_heads * self.key_head_dim * self.value_head_dim

    @property
    def parameter_count(self):
        value_dim = self.value_heads * self.value_head_dim
        return (self.embedding_dim * (self.conv_dim + 2 * value_dim + 2 * self.value_heads)
                + self.conv_dim * self.conv_kernel_size + 2 * self.value_heads + self.value_head_dim)


@dataclass
class SwiGLUConfig(baselayerConfig):
    operator_name = 'ModernDenseFFN'
    embedding_dim: int = 4096
    intermediate_dim: int = 12288

    @property
    def parameter_count(self):
        return 3 * self.embedding_dim * self.intermediate_dim


@dataclass
class ModernAttentionBlockConfig(BaseBlockConfig):
    attention: GQAConfig = field(default_factory=GQAConfig)
    mlp: SwiGLUConfig = field(default_factory=SwiGLUConfig)
    max_position_embeddings: int = 40960
    memory_accounting_version: int = 2
    execution_family: str = 'attention'
    type_name: str = 'gqa-swiglu'
    cache_store: bool = True

    def __post_init__(self):
        self.embedding_dim = self.attention.embedding_dim
        self.mlp_hidden_dim = self.mlp.intermediate_dim
        self.num_q_heads = self.attention.q_heads
        self.num_kv_heads = self.attention.kv_heads
        self.states = 2 * self.num_kv_heads * self.attention.head_dim
        self.parameter_count = self.attention.parameter_count + self.mlp.parameter_count + 2 * self.embedding_dim
        self.output_mem = self.embedding_dim
        self.peak_intermed = 2 * self.embedding_dim + max(
            3 * self.mlp.intermediate_dim,
            (self.num_q_heads + 2 * self.num_kv_heads) * self.attention.head_dim)
        self.intermed_mem = self.peak_intermed
        self.layers = [['ModernGQA'], ['ModernSwiGLU']]
        self.layer_configs = [[self.attention], [self.mlp]]

    def get_accelerators(self, stage):
        return [2] if stage == 'prefill' else [3]

    @staticmethod
    def get_name():
        return 'modern-gqa-block'


@dataclass
class GatedDeltaNetBlockConfig(BaseBlockConfig):
    attention: GatedDeltaNetConfig = field(default_factory=GatedDeltaNetConfig)
    mlp: SwiGLUConfig = field(default_factory=SwiGLUConfig)
    max_position_embeddings: int = 262144
    memory_accounting_version: int = 2
    execution_family: str = 'recurrent'
    type_name: str = 'gated-delta-net-swiglu'
    cache_store: bool = False

    def __post_init__(self):
        self.embedding_dim = self.attention.embedding_dim
        self.mlp_hidden_dim = self.mlp.intermediate_dim
        # HYDRA memory units are one-byte elements under this model's contract.
        self.states = (self.attention.recurrent_elements * self.attention.state_bytes
                       + self.attention.conv_dim * self.attention.conv_kernel_size)
        self.parameter_count = self.attention.parameter_count + self.mlp.parameter_count + 2 * self.embedding_dim
        self.output_mem = self.embedding_dim
        self.peak_intermed = 2 * self.embedding_dim + max(3 * self.mlp.intermediate_dim,
                                                       2 * self.attention.conv_dim)
        self.intermed_mem = self.peak_intermed
        self.layers = [['ModernGDN'], ['ModernSwiGLU']]
        self.layer_configs = [[self.attention], [self.mlp]]

    def get_accelerators(self, stage):
        return [0] if stage == 'prefill' else [1]

    @staticmethod
    def get_name():
        return 'gated-delta-net-block'


class ModernModelMixin:
    profile_version = 'explicit-decoder-v1'
    precision_contract = 'W8-A8-KV8-recurrent-FP32'
    modeled_scope = 'decoder-blocks-only'

    def configure_metadata(self):
        # Persist the timing/precision contract in the existing config serializer.
        self.profile_version = type(self).profile_version
        self.precision_contract = type(self).precision_contract
        self.modeled_scope = type(self).modeled_scope

    def validate_request_lengths(self, context_length, prefill_length, decode_length):
        if max(context_length, prefill_length) + decode_length > self.max_position_embeddings:
            raise ValueError(f"Request exceeds context limit {self.max_position_embeddings} for {self.get_name()}")

    def validate_batch_lengths(self, lengths, batch_size):
        # The existing processor profiles one representative request per batch.
        # Until ragged/padded batch profiles exist, require equal input shapes.
        if batch_size == 1:
            return
        for start in range(0, len(lengths), batch_size):
            batch = lengths.iloc[start:start+batch_size]
            if any(batch[column].nunique() > 1 for column in ('num_prefill_tokens', 'context_length')):
                raise ValueError('Modern decoder batches require equal input/context lengths; use batch size 1 for heterogeneous traces.')

    def validate_execution(self, config):
        if config.workload_config.bytes_per_param != 1:
            raise ValueError('Modern decoder profiles require one-byte weights, activations and KV; recurrent state is FP32.')
        if config.mapping_config.task_parallelism != 'pipeline':
            raise ValueError('Modern decoder profiles currently support pipeline parallelism only.')
        if config.mapping_config.mapping_strategy != 'static' or config.cluster_config.local_scheduler not in ('static', 'agent', 'vllm_latest'):
            raise ValueError('Modern decoders require static mapping and static, agent, or vllm_latest scheduling.')


@dataclass
class Qwen3_8BModelConfig(ModernModelMixin, BaseModelConfig):
    num_layers: int = 36
    num_q_heads: int = 32
    num_kv_heads: int = 8
    embedding_dim: int = 4096
    mlp_hidden_dim: int = 12288
    max_position_embeddings: int = 40960
    vocab_size: int = 151936
    use_gated_mlp: bool = True
    activation: str = 'silu'
    norm: str = 'rms_norm'
    rope_theta: float = 1000000
    num_A_blocks: int = 36
    num_M_blocks: int = 0
    source_revision: str = 'b968826d9c46dd6066d109eabc6255188de91218'
    source_model: str = 'Qwen/Qwen3-8B'
    block_config: ModernAttentionBlockConfig = field(default_factory=ModernAttentionBlockConfig)

    def __post_init__(self):
        self.configure_metadata()
        self.hybrid_blocks = [self.block_config]
        self.block_type_sequence = [0] * self.num_layers

    @staticmethod
    def get_name():
        return 'qwen3-8b'


@dataclass
class Qwen3_5_9BModelConfig(ModernModelMixin, BaseModelConfig):
    num_layers: int = 32
    num_q_heads: int = 16
    num_kv_heads: int = 4
    embedding_dim: int = 4096
    mlp_hidden_dim: int = 12288
    max_position_embeddings: int = 262144
    vocab_size: int = 248320
    use_gated_mlp: bool = True
    activation: str = 'silu'
    norm: str = 'rms_norm'
    rope_theta: float = 10000000
    partial_rotary_factor: float = 0.25
    hybrid: bool = True
    num_A_blocks: int = 8
    num_M_blocks: int = 24
    source_revision: str = 'c202236235762e1c871ad0ccb60c8ee5ba337b9a'
    source_model: str = 'Qwen/Qwen3.5-9B'

    def __post_init__(self):
        self.configure_metadata()
        self.hybrid_blocks = [GatedDeltaNetBlockConfig(), ModernAttentionBlockConfig(
            attention=GQAConfig(q_heads=16, kv_heads=4, head_dim=256, rotary_dim=64, output_gate=True),
            max_position_embeddings=self.max_position_embeddings)]
        self.block_type_sequence = [0, 0, 0, 1] * 8

    @staticmethod
    def get_name():
        return 'qwen3.5-9b-text'
