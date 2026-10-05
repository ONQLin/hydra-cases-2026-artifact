"""DeepSeek-V3 decoder baseline: absorbed MLA and group-limited colocated MoE.

W8/A8/KV8 is a performance surrogate. Embedding, final norm, LM head and MTP
are excluded. The initial context limit stays within the 4096-token base window.
"""

from dataclasses import dataclass

from Sim.config.attention_operator_config import MLAConfig, OperatorBlockConfig
from Sim.config.model_config import BaseModelConfig
from Sim.config.modern_model_config import ModernModelMixin, SwiGLUConfig
from Sim.config.moe_config import DecoderLayerConfig, GroupedMoEConfig


@dataclass
class DeepSeekV3ModelConfig(ModernModelMixin, BaseModelConfig):
    embedding_dim: int = 7168
    num_layers: int = 61
    num_A_blocks: int = 61
    num_M_blocks: int = 0
    # HYDRA uses this flag for heterogeneous block types, including dense/MoE
    # FFN variants even when all attention operators have the same family.
    hybrid: bool = True
    max_position_embeddings: int = 4096
    source_context_limit: int = 163840
    source_model: str = 'deepseek-ai/DeepSeek-V3'
    source_revision: str = 'e815299b0bcbac849fa540c768ef21845365c9eb'
    profile_version = 'mla-streaming-grouped-colocated-moe-v1'
    modeled_scope = 'decoder-blocks-synthetic-grouped-routes-colocated-experts'

    def __post_init__(self):
        self.configure_metadata()
        attention = MLAConfig(algorithm='streaming',cache_layout='absorbed')
        self.hybrid_blocks = [
            OperatorBlockConfig(operator=DecoderLayerConfig(attention=attention,
                feed_forward=SwiGLUConfig(embedding_dim=7168,intermediate_dim=18432)),
                kernel='ModernDecoderBlock',execution_family='attention',type_name='mla-dense'),
            OperatorBlockConfig(operator=DecoderLayerConfig(attention=attention,
                feed_forward=GroupedMoEConfig(embedding_dim=7168,intermediate_dim=2048)),
                kernel='ModernDecoderBlock',execution_family='attention',type_name='mla-grouped-moe'),
        ]
        self.block_type_sequence = [0]*3+[1]*58

    @staticmethod
    def get_name():
        return 'deepseek-v3-text'


@dataclass
class DeepSeekDecoderFixtureModelConfig(ModernModelMixin, BaseModelConfig):
    """Two small all-attention layers exercise dense then grouped MoE FFNs."""
    embedding_dim: int = 128
    num_layers: int = 2
    num_A_blocks: int = 2
    num_M_blocks: int = 0
    hybrid: bool = True
    max_position_embeddings: int = 4096
    profile_version = 'mla-streaming-grouped-colocated-moe-v1'
    modeled_scope = 'synthetic-DeepSeek-decoder-fixture'

    def __post_init__(self):
        self.configure_metadata()
        attention = MLAConfig(embedding_dim=128,num_heads=4,q_lora_rank=32,kv_lora_rank=32,
            qk_nope_head_dim=16,qk_rope_head_dim=8,v_head_dim=16,algorithm='streaming',cache_layout='absorbed')
        self.hybrid_blocks = []
        for ffn in (SwiGLUConfig(embedding_dim=128,intermediate_dim=256),
                    GroupedMoEConfig(embedding_dim=128,intermediate_dim=64,num_experts=16,top_k=4,
                                     num_groups=4,top_k_groups=2)):
            self.hybrid_blocks.append(OperatorBlockConfig(operator=DecoderLayerConfig(attention=attention,feed_forward=ffn),
                kernel='ModernDecoderBlock',execution_family='attention',type_name='fixture-mla-ffn'))
        self.block_type_sequence = [0,1]

    @staticmethod
    def get_name():
        return 'deepseek-decoder-fixture'
