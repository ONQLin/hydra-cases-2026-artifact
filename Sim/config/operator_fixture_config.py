"""Small runnable KDA/MLA validation fixture, not a Kimi or DeepSeek checkpoint."""

from dataclasses import dataclass, replace

from Sim.config.model_config import BaseModelConfig
from Sim.config.modern_model_config import (ModernModelMixin, Qwen3_8BModelConfig,
                                          ModernAttentionBlockConfig, GQAConfig, SwiGLUConfig)
from Sim.config.attention_operator_config import KDAConfig, MLAConfig, OperatorBlockConfig


@dataclass
class GQAPrefixFixtureModelConfig(Qwen3_8BModelConfig):
    """Small dense decoder for scheduler/cache validation, not a checkpoint."""
    num_layers: int = 2
    num_A_blocks: int = 2
    embedding_dim: int = 512
    num_q_heads: int = 8
    num_kv_heads: int = 2
    mlp_hidden_dim: int = 1024
    max_position_embeddings: int = 4096
    source_model: str = 'synthetic/gqa-prefix-fixture'
    source_revision: str = 'synthetic-v1'

    def __post_init__(self):
        self.block_config = ModernAttentionBlockConfig(
            attention=GQAConfig(embedding_dim=512, q_heads=8, kv_heads=2, head_dim=64, rotary_dim=64),
            mlp=SwiGLUConfig(embedding_dim=512, intermediate_dim=1024), max_position_embeddings=4096)
        super().__post_init__()

    @staticmethod
    def get_name():
        return 'gqa-prefix-fixture'


@dataclass
class KdaMlaFixtureModelConfig(ModernModelMixin, BaseModelConfig):
    """Two small attention operators; deliberately excludes FFN and MoE."""
    embedding_dim: int = 512
    num_layers: int = 2
    num_A_blocks: int = 1
    num_M_blocks: int = 1
    hybrid: bool = True
    max_position_embeddings: int = 4096
    profile_version = 'operator-graph-v1'
    modeled_scope = 'synthetic-KDA-MLA-operators-only'

    def __post_init__(self):
        self.configure_metadata()
        self.hybrid_blocks = [
            OperatorBlockConfig(operator=KDAConfig(embedding_dim=512, num_heads=8, head_dim=64)),
            OperatorBlockConfig(operator=MLAConfig(embedding_dim=512, num_heads=8,
                q_lora_rank=128, kv_lora_rank=64, qk_nope_head_dim=32,
                qk_rope_head_dim=16, v_head_dim=32), kernel='ModernMLA',
                execution_family='attention', type_name='mla-operator'),
        ]
        self.block_type_sequence = [0, 1]

    @staticmethod
    def get_name():
        return 'kda-mla-operator-fixture'


@dataclass
class KdaChunkMlaFixtureModelConfig(KdaMlaFixtureModelConfig):
    """Same fixture geometry with chunk prefill and tiled recurrent decode."""
    profile_version = 'operator-graph-kda-tiled-v2'

    def __post_init__(self):
        super().__post_init__()
        self.hybrid_blocks[0] = OperatorBlockConfig(operator=replace(
            self.hybrid_blocks[0].operator, algorithm='chunk', chunk_size=16,
            head_tile_size=2, value_tile_size=32))

    @staticmethod
    def get_name():
        return 'kda-chunk-mla-operator-fixture'
