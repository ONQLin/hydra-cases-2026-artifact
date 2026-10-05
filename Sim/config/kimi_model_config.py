"""Pinned Kimi Linear text-decoder topology with a synthetic MoE route baseline.

Validation is initially limited to 4096 tokens; the checkpoint's advertised
context limit is larger. No embedding, final norm or LM head is included.
"""

from dataclasses import dataclass

from Sim.config.attention_operator_config import KDAConfig, MLAConfig, OperatorBlockConfig
from Sim.config.model_config import BaseModelConfig
from Sim.config.modern_model_config import ModernModelMixin, SwiGLUConfig
from Sim.config.moe_config import DecoderLayerConfig, MoEConfig


@dataclass
class KimiLinear48BA3BModelConfig(ModernModelMixin, BaseModelConfig):
    embedding_dim: int = 2304
    num_layers: int = 27
    num_A_blocks: int = 7
    num_M_blocks: int = 20
    hybrid: bool = True
    max_position_embeddings: int = 4096
    source_context_limit: int = 1048576
    source_model: str = 'moonshotai/Kimi-Linear-48B-A3B-Instruct'
    source_revision: str = 'e1df551a447157d4658b573f9a695d57658590e9'
    mla_algorithm = 'materialized'
    profile_version = 'kda-chunk-colocated-moe-v1'
    modeled_scope = 'decoder-blocks-synthetic-routes-colocated-experts'

    def __post_init__(self):
        self.configure_metadata()
        kda = KDAConfig(algorithm='chunk',chunk_size=64)
        mla = MLAConfig(embedding_dim=2304,num_heads=32,q_lora_rank=0,
                        cache_layout='expanded',rope_enabled=False,algorithm=self.mla_algorithm)
        self.hybrid_blocks = [
            OperatorBlockConfig(operator=DecoderLayerConfig(attention=kda,
                feed_forward=SwiGLUConfig(embedding_dim=2304,intermediate_dim=9216)),
                kernel='ModernDecoderBlock',execution_family='recurrent',type_name='kda-dense'),
            OperatorBlockConfig(operator=DecoderLayerConfig(attention=kda,feed_forward=MoEConfig()),
                kernel='ModernDecoderBlock',execution_family='recurrent',type_name='kda-moe'),
            OperatorBlockConfig(operator=DecoderLayerConfig(attention=mla,feed_forward=MoEConfig()),
                kernel='ModernDecoderBlock',execution_family='attention',type_name='mla-moe'),
        ]
        full_attention_layers = {3,7,11,15,19,23,26}
        self.block_type_sequence = [0]+[2 if layer in full_attention_layers else 1 for layer in range(1,27)]

    @staticmethod
    def get_name():
        return 'kimi-linear-48b-a3b-text'


@dataclass
class KimiDecoderFixtureModelConfig(ModernModelMixin, BaseModelConfig):
    """Three small layers exercise dense, KDA-MoE and MLA-MoE composition."""
    embedding_dim: int = 128
    num_layers: int = 3
    num_A_blocks: int = 1
    num_M_blocks: int = 2
    hybrid: bool = True
    max_position_embeddings: int = 4096
    mla_algorithm = 'materialized'
    profile_version = 'kda-chunk-colocated-moe-v1'
    modeled_scope = 'synthetic-Kimi-decoder-fixture'

    def __post_init__(self):
        self.configure_metadata()
        kda = KDAConfig(embedding_dim=128,num_heads=4,head_dim=32,algorithm='chunk',chunk_size=16)
        mla = MLAConfig(embedding_dim=128,num_heads=4,q_lora_rank=0,kv_lora_rank=32,
                        qk_nope_head_dim=16,qk_rope_head_dim=8,v_head_dim=16,
                        cache_layout='expanded',rope_enabled=False,algorithm=self.mla_algorithm)
        self.hybrid_blocks = []
        for attention,ffn in ((kda,SwiGLUConfig(embedding_dim=128,intermediate_dim=256)),
                              (kda,MoEConfig(embedding_dim=128,intermediate_dim=64,num_experts=8,top_k=2)),
                              (mla,MoEConfig(embedding_dim=128,intermediate_dim=64,num_experts=8,top_k=2))):
            self.hybrid_blocks.append(OperatorBlockConfig(operator=DecoderLayerConfig(attention=attention,feed_forward=ffn),
                kernel='ModernDecoderBlock',execution_family='attention' if attention.cache_store else 'recurrent',
                type_name='fixture-attention-ffn'))
        self.block_type_sequence = [0,1,2]

    @staticmethod
    def get_name():
        return 'kimi-decoder-fixture'


@dataclass
class KimiLinearStreamingModelConfig(KimiLinear48BA3BModelConfig):
    mla_algorithm = 'streaming'
    profile_version = 'kda-chunk-mla-streaming-colocated-moe-v2'

    @staticmethod
    def get_name():
        return 'kimi-linear-48b-a3b-text-streaming'


@dataclass
class KimiStreamingFixtureModelConfig(KimiDecoderFixtureModelConfig):
    mla_algorithm = 'streaming'
    profile_version = 'kda-chunk-mla-streaming-colocated-moe-v2'

    @staticmethod
    def get_name():
        return 'kimi-decoder-streaming-fixture'
