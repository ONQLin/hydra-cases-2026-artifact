# Adapted from
# https://github.com/microsoft/vidur/blob/main/vidur/config/model_config.py

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List

from Sim.config.base_fixed_config import BaseFixedConfig
from Sim.logger import init_logger, _setup_logger
_setup_logger()   # root logger handles console + file
logger = init_logger(__name__)  # child logger inherits handlers

class BaseBlockConfig(BaseFixedConfig):
    """
    Base class for block configurations.
    This class can be extended to define specific block configurations.
    """

    num_layers: int = 3,
    num_q_heads: int = 8,
    num_kv_heads: int = 8,
    embedding_dim: int = 512,
    mlp_hidden_dim: int = 2048,
    max_position_embeddings: int = 512,

    ssm_state_dim: int = 128      # size of internal SSM state
    conv_kernel_size: int = 3     # optional convolutional projection
    mlp_hidden_dim: int = 2048    # still applies to the FFN block
    max_position_embeddings: int = 512
    
    intermed_mem: int = 51016
    peak_intermed: int = 14336
    states: int = 2048
    output_mem: int = 4096
    cache_store: bool = True
    parameter_count: int = 230_370_112
    type_name: str = "base-block-transformer"
    memory_accounting_version: int = 1

    @property
    def execution_family(self):
        """Legacy names remain compatible; new blocks declare a capability."""
        if 'transformer' in self.type_name:
            return 'attention'
        if 'mamba' in self.type_name:
            return 'recurrent'
        raise ValueError(f"No execution family for {self.type_name}")

    @property
    def memory_group(self):
        return {'attention': 'transformer', 'recurrent': 'mamba'}[self.execution_family]

    def get_accelerators(self, stage):
        from Sim.config import utils
        families = {'attention': (utils.T_P, utils.T_D),
                    'recurrent': (utils.M_P, utils.M_D)}
        return families[self.execution_family][0 if stage == 'prefill' else 1]
    
    layers: Optional[List[str]] = field(
        default_factory=lambda: []
    )
    
    layer_configs: Optional[List[List[Any]]] = field(
        default_factory=lambda: []
    )
@dataclass
class BaseModelConfig(BaseFixedConfig):
    @classmethod
    def create_from_name(cls, name):
        # Import model extensions only after the legacy config module is loaded.
        from Sim.config import modern_model_config
        from Sim.config import operator_fixture_config
        from Sim.config import kimi_model_config
        from Sim.config import deepseek_model_config
        return super().create_from_name(name)

    def validate_execution(self, config):
        """Optional model-specific restrictions checked before simulation."""

    def validate_request_lengths(self, context_length, prefill_length, decode_length):
        """Optional context contract; legacy models retain their original behavior."""

    def validate_batch_lengths(self, lengths, batch_size):
        """Optional batch-shape contract for a prepared token-length trace."""

    num_layers: int = 12
    num_q_heads: int = 12
    num_kv_heads: int = 12
    embedding_dim: int = 768
    mlp_hidden_dim: int = 3072
    max_position_embeddings: int = 512
    use_gated_mlp: bool = False
    use_bias: bool = False
    use_qkv_bias: bool = False
    activation: str = "gelu"
    norm: str = "layer_norm"
    post_attn_norm: bool = False
    vocab_size: int = 30522
    is_neox_style: Optional[bool] = True
    rope_theta: Optional[float] = None
    rope_scaling: Optional[Dict[str, Any]] = None
    partial_rotary_factor: float = 1.0
    no_tensor_parallel: bool = False
    block_config: BaseBlockConfig = None  # This can be a specific block configuration class
    hybrid: bool = False
    num_A_blocks: int = 0
    num_M_blocks: int = 0
    block_list: Optional[List[str]] = field(
        default=None,
        metadata={"help": "List of block types to use in the model."}
    )
    hybrid_blocks: List[BaseBlockConfig] = field(
        default= lambda: [],
        metadata={"help": "List of block types to use in the hybrid model."}
    )
    block_type_sequence: List[int] = field(
        default=lambda: [],
        metadata={
            "help": "A list of integers where each integer is the index into hybrid_blocks for \
                the length must equal the total number of layers."
        }
    )

@dataclass
class Llama2ModelConfig(BaseModelConfig):
    max_position_embeddings: int = 16384
    use_gated_mlp: bool = True
    use_bias: bool = False
    use_qkv_bias: bool = False
    activation: str = "silu"
    norm: str = "rms_norm"
    post_attn_norm: bool = True
    vocab_size: int = 32768
    is_neox_style: Optional[bool] = True
    rope_theta: Optional[float] = 10000
    rope_scaling: Optional[Dict[str, Any]] = None
    partial_rotary_factor: float = 1.0
    no_tensor_parallel: bool = False

    @staticmethod
    def get_name():
        return "meta-llama/Llama-2-Config"


@dataclass
class CodeLlama34BModelConfig(Llama2ModelConfig):
    num_layers: int = 48
    num_q_heads: int = 64
    num_kv_heads: int = 8
    embedding_dim: int = 8192
    mlp_hidden_dim: int = 22016
    rope_theta: Optional[float] = 1000000

    @staticmethod
    def get_name():
        return "codellama/CodeLlama-34b-Instruct-hf"


@dataclass
class Llama2_7BModelConfig(Llama2ModelConfig):
    num_layers: int = 32
    num_q_heads: int = 32
    num_kv_heads: int = 32
    embedding_dim: int = 4096
    mlp_hidden_dim: int = 11008
    max_position_embeddings: int = 4096

    @staticmethod
    def get_name():
        return "meta-llama/Llama-2-7b-hf"


@dataclass
class Llama2_70BModelConfig(Llama2ModelConfig):
    num_layers: int = 80
    num_q_heads: int = 64
    num_kv_heads: int = 8
    embedding_dim: int = 8192
    mlp_hidden_dim: int = 28672
    max_position_embeddings: int = 4096

    @staticmethod
    def get_name():
        return "meta-llama/Llama-2-70b-hf"




@dataclass
class Llama3_70BModelConfig(Llama2ModelConfig):
    num_layers: int = 80
    num_q_heads: int = 64
    num_kv_heads: int = 8
    embedding_dim: int = 8192
    mlp_hidden_dim: int = 28672
    max_position_embeddings: int = 8192
    rope_theta: Optional[float] = 500000
    vocab_size: int = 128256

    @staticmethod
    def get_name():
        return "meta-llama/Meta-Llama-3-70B"


@dataclass
class InternLMModelConfig(Llama2ModelConfig):
    max_position_embeddings: int = 4096
    vocab_size: int = 103168


@dataclass
class InternLM_20BModelConfig(InternLMModelConfig):
    num_layers: int = 60
    num_q_heads: int = 40
    num_kv_heads: int = 40
    embedding_dim: int = 5120
    mlp_hidden_dim: int = 13824

    @staticmethod
    def get_name():
        return "internlm/internlm-20b"


@dataclass
class InternLM2ModelConfig(Llama2ModelConfig):
    max_position_embeddings: int = 32768
    vocab_size: int = 92544


@dataclass
class InternLM2_20BModelConfig(InternLM2ModelConfig):
    num_layers: int = 48
    num_q_heads: int = 48
    num_kv_heads: int = 8
    embedding_dim: int = 6144
    mlp_hidden_dim: int = 16384
    rope_theta: Optional[float] = 1000000

    @staticmethod
    def get_name():
        return "internlm/internlm2-20b"


@dataclass
class Phi2ModelConfig(Llama2ModelConfig):
    num_layers: int = 32
    num_q_heads: int = 32
    num_kv_heads: int = 32
    embedding_dim: int = 2560
    mlp_hidden_dim: int = 10240
    max_position_embeddings: int = 2048
    use_gated_mlp: bool = False
    use_bias: bool = True
    use_qkv_bias: bool = True
    activation: str = "gelu"
    norm: str = "layer_norm"
    post_attn_norm: bool = False
    vocab_size: int = 51200
    rope_scaling: Optional[Dict[str, Any]] = None
    rope_theta: Optional[float] = 10000
    partial_rotary_factor: float = 0.4
    no_tensor_parallel: bool = True

    @staticmethod
    def get_name():
        return "microsoft/phi-2"


@dataclass
class QwenModelConfig(Llama2ModelConfig):
    use_qkv_bias: bool = True
    max_position_embeddings: int = 32768
    vocab_size: int = 152064

    @staticmethod
    def get_name():
        return "Qwen/Qwen-Config"


@dataclass
class Qwen72BModelConfig(QwenModelConfig):
    num_layers: int = 80
    num_q_heads: int = 64
    num_kv_heads: int = 64
    embedding_dim: int = 8192
    mlp_hidden_dim: int = 24576
    rope_theta: Optional[float] = 1000000

    @staticmethod
    def get_name():
        return "Qwen/Qwen-72B"

# @dataclass
# class TransformerBlockConfig:
#     num_layers: int = 3,
#     num_q_heads: int = 8,
#     num_kv_heads: int = 8,
#     embedding_dim: int = 512,
#     mlp_hidden_dim: int = 2048,
#     max_position_embeddings: int = 512,

# # minor changes between mamba2 and mamba
# @dataclass
# class MambaBlockConfig:
#     num_layers: int = 6
#     embedding_dim: int = 512
#     ssm_state_dim: int = 128      # size of internal SSM state
#     conv_kernel_size: int = 3     # optional convolutional projection
#     mlp_hidden_dim: int = 2048    # still applies to the FFN block
#     max_position_embeddings: int = 512

@dataclass
class baselayerConfig:
    bs: int = 1

@dataclass
class Llama3_8B_BlockT_Config(BaseBlockConfig):
    num_layers: int = 6
    num_q_heads: int = 32
    num_kv_heads: int = 8
    embedding_dim: int = 4096
    mlp_hidden_dim: int = 14336
    max_position_embeddings: int = 4096
    rope_theta: Optional[float] = 500000
    vocab_size: int = 128256
    
    intermed_mem: int = 51016
    peak_intermed: int = 14336
    states: int = 4096*2
    output_mem: int = 4096
    cache_store: bool = True
    
    parameter_count: int = 230_370_112
    type_name: str = "llama3-8b-block-transformer"
    
    layers: List[str] = field(default_factory=lambda: [['RMS_Norm'], ['MHA'], ['RMS_Norm'], ['FC'], ['Silu'], ['FC']])
    layer_configs: List[List[baselayerConfig]] = field(default_factory=lambda: [
        [RMS_NormConfig(in_size=4096, out_size=4096, bs=1)],
        [MHA_NormConfig(embedding_dim=4096, kv_heads=8, q_heads=32)],
        [RMS_NormConfig(in_size=4096, out_size=4096, bs=1)],
        [FCConfig(f_in=4096, f_out=14336)],
        [SiluConfig(bs=1,in_size=14336, out_size=14336)],
        [FCConfig(f_in=14336, f_out=4096)]
    ])

    @staticmethod
    def get_name():
        return "llama3-8b-block-transformer"


# https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba2_simple.py
# https://huggingface.co/state-spaces/mamba-2.8b/blob/main/model.safetensors
@dataclass
class Mamba2_3B_BlockM_Config(BaseBlockConfig):
    num_layers: int = 6
    embedding_dim: int = 5120
    ssm_state_dim: int = 16
    conv_kernel_size: int = 4
    mlp_hidden_dim: int = 5120
    num_q_heads: int = 80
    max_position_embeddings: int = 50280
    
    intermed_mem: int = 43708
    peak_intermed: int = 10240 
    states: int = 327680
    output_mem: int = 2560
    cache_store: bool = False
    type_name: str = "mamba2-3b-block-mamba"
    parameter_count = 39_885_440
    # 2560*(5120*2+64*2+80) + 4*(5120+128) + 5120 + 128 +5120*2560 + 5120
    
    
    layers: List = field(default_factory=lambda: [['FC'],['Conv1D','Softplus'],['Silu'],['SSM'],['RMS_Norm'],['FC']])
    layer_configs: List[List[baselayerConfig]] = field(default_factory=lambda: [
        [FCConfig(f_in=2560, f_out=5120*2+64*2+80)], # 2xEDxdim for splitting x and z BC dt
        [Conv1DConfig(kernel_size=4, c_in=5120+2*64, c_out=5120+2*64), SoftplusConfig(in_size=80, out_size=80)],
        [SiluConfig(in_size=10240+2*64, out_size=10240+2*64)], # 2xED+2xstates z, x and BC
        [SSMConfig(state_size=16, ED=5120)],
        [RMS_NormConfig(in_size=5120, out_size=5120)],
        [FCConfig(f_in=5120, f_out=2560)]
    ])
    
    # def __post_init__(self):
    #     self.states = self.states * self.embedding_dim
        
    @staticmethod
    def get_name():
        return "mamba2-3b-block"
    
@dataclass
class Mamba2_Nemo4B_BlockM_Config(BaseBlockConfig):
    num_layers: int = 10
    embedding_dim: int = 3072*2
    ssm_state_dim: int = 128
    conv_kernel_size: int = 4
    mlp_hidden_dim: int = 5120
    num_q_heads: int = 8
    max_position_embeddings: int = 8192
    
    intermed_mem: int = 66672
    peak_intermed: int = 16496
    states: int = 786432
    output_mem: int = 2560
    cache_store: bool = False
    type_name: str = "mamba2-nemo4b-block-mamba"
    parameter_count = 148_300_000

    layers: List = field(default_factory=lambda: [['RMS_Norm'],['FC'],['Conv1D','Softplus'],['Silu'],['SSM'],['RMS_Norm'],['FC'],
                                                    ['FC'],['SquareRelu'],['FC']])
    layer_configs: List[List[baselayerConfig]] = field(default_factory=lambda: [
        # Mamba layers
        [RMS_NormConfig(in_size=3072, out_size=3072)],
        [FCConfig(f_in=3072, f_out=16496)], # 2xEDxdim for splitting x and z BC dt  (3072x2x2+128x8x2+112)
        [Conv1DConfig(kernel_size=4, c_in=3072*2+128*8*2, c_out=3072*2+128*8*2), SoftplusConfig(in_size=112, out_size=112)],
        [SiluConfig(in_size=3072*2+128*8*2, out_size=3072*2+128*8*2)], # 2xED+2xstates z, x and BC
        [SSMConfig(state_size=128, ED=3072*2)],
        [RMS_NormConfig(in_size=3072*2, out_size=3072*2)],
        [FCConfig(f_in=3072*2, f_out=3072)],
        # Feedforward layers
        [FCConfig(f_in=3072, f_out=12288)], # 2xEDxdim for splitting x and z BC dt
        [SquareReluConfig(in_size=12288, out_size=12288)], # 2xED+2xstates z, x and BC
        [FCConfig(f_in=12288, f_out=3072)],
    ])
    
    # def __post_init__(self):
    #     self.states = self.states * self.embedding_dim
        
    @staticmethod
    def get_name():
        return "mamba2-nemo4b-block-mamba"

@dataclass
class Mamba2_Nemo4B_BlockM2_Config(BaseBlockConfig):
    num_layers: int = 7
    embedding_dim: int = 3072*2
    ssm_state_dim: int = 128
    conv_kernel_size: int = 4
    mlp_hidden_dim: int = 5120
    num_q_heads: int = 8
    max_position_embeddings: int = 8192
    
    intermed_mem: int = 30924
    peak_intermed: int = 16496 
    states: int = 786432
    output_mem: int = 2560
    cache_store: bool = False
    type_name: str = "mamba2-nemo4b-block2-mamba"
    parameter_count = 69_656_320

    layers: List = field(default_factory=lambda: [['RMS_Norm'],['FC'],['Conv1D','Softplus'],['Silu'],['SSM'],['RMS_Norm'],['FC']])
    layer_configs: List[List[baselayerConfig]] = field(default_factory=lambda: [
        # Mamba layers
        [RMS_NormConfig(in_size=3072, out_size=3072)],
        [FCConfig(f_in=3072, f_out=16496)], # 2xEDxdim for splitting x and z BC dt  (3072x2x2+128x8x2+112)
        [Conv1DConfig(kernel_size=4, c_in=3072*2+128*8*2, c_out=3072*2+128*8*2), SoftplusConfig(in_size=112, out_size=112)],
        [SiluConfig(in_size=3072*2+128*8*2, out_size=3072*2+128*8*2)], # 2xED+2xstates z, x and BC
        [SSMConfig(state_size=128, ED=3072*2)],
        [RMS_NormConfig(in_size=3072*2, out_size=3072*2)],
        [FCConfig(f_in=3072*2, f_out=3072)]
    ])
    
    # def __post_init__(self):
    #     self.states = self.states * self.embedding_dim
        
    @staticmethod
    def get_name():
        return "mamba2-nemo4b-block2-mamba"

@dataclass
class Attention_Nemo4B_BlockT_Config(BaseBlockConfig):
    num_layers: int = 6
    num_q_heads: int = 32
    num_kv_heads: int = 8
    embedding_dim: int = 3072
    mlp_hidden_dim: int = 14336
    max_position_embeddings: int = 8192
    rope_theta: Optional[float] = 500000
    vocab_size: int = 128256
    
    intermed_mem: int = 36864
    peak_intermed: int = 12288
    states: int = 768*2 # 8 kv heads
    output_mem: int = 4096
    cache_store: bool = True
    
    parameter_count: int = 106_960_896
    type_name: str = "nemo4b-block-transformer"
    
    layers: List[str] = field(default_factory=lambda: [['RMS_Norm'], ['MHA'], ['RMS_Norm'], ['FC'], ['SquareRelu'], ['FC']])
    layer_configs: List[List[baselayerConfig]] = field(default_factory=lambda: [
        [RMS_NormConfig(in_size=3072, out_size=3072)],
        [MHA_NormConfig(embedding_dim=3072, kv_heads=8, q_heads=32)],
        [RMS_NormConfig(in_size=4096, out_size=4096)],
        # Feedforward layers
        [FCConfig(f_in=3072, f_out=12288)],
        [SquareReluConfig(in_size=12288, out_size=12288)],
        [FCConfig(f_in=12288, f_out=3072)]
    ])

    @staticmethod
    def get_name():
        return "nemo4b-block-transformer"



@dataclass
class Mamba2_Nemotron56B_BlockM_Config(BaseBlockConfig):
    """Mamba-style block for Nemotron-H-56B (M blocks)."""
    num_layers: int = 10
    embedding_dim: int = 8192*2
    ssm_state_dim: int = 256
    conv_kernel_size: int = 4
    mlp_hidden_dim: int = 32768
    num_q_heads: int = 64
    max_position_embeddings: int = 8192

    intermed_mem: int = 131_072 # 8192+37120+(8192*2+256*8*2)*2 + 8192*2*3 + 32768*2  
    peak_intermed: int = 32768
    states: int = 256*8192*2
    output_mem: int = 8192
    cache_store: bool = False
    parameter_count: int = 975_274_240
    # total params: 8192*37120 + 4*(8192*2+256*8*2) + 8192*2 + 256 + 8192*2*8192 + 8192*32768*2 = 975274240
    type_name: str = "mamba2-nemotron56b-block-mamba"

    # keep layer structure open for future detailed sizing
    layers: List = field(default_factory=lambda: [['RMS_Norm'],['FC'],['Conv1D','Softplus'],['Silu'],['SSM'],['RMS_Norm'],['FC'],
                                                    ['FC'],['SquareRelu'],['FC']])
    layer_configs: List[List[baselayerConfig]] = field(default_factory=lambda: [
        [RMS_NormConfig(in_size=8192, out_size=8192)],
        [FCConfig(f_in=8192, f_out=37120)],
        [Conv1DConfig(kernel_size=4, c_in=20480, c_out=20480), SoftplusConfig(in_size=112, out_size=112)],
        [SiluConfig(in_size=20480, out_size=20480)], # 3xED+3xstates z, x and BC
        [SSMConfig(state_size=256, ED=8192*2)],
        [RMS_NormConfig(in_size=8192*2, out_size=8192*2)],
        [FCConfig(f_in=8192*2, f_out=8192)],
        # Feedforward layers
        [FCConfig(f_in=8192, f_out=32768)],
        [SquareReluConfig(in_size=32768, out_size=32768)],
        [FCConfig(f_in=32768, f_out=8192)],
    ])

    @staticmethod
    def get_name():
        return "mamba2-nemotron56b-block"


@dataclass
class Mamba2_Nemotron56B_BlockM2_Config(BaseBlockConfig):
    """Mamba-style block that includes an explicit FFN/MLP (dash '-' in pattern)."""
    num_layers: int = 8
    embedding_dim: int = 8192
    ssm_state_dim: int = 256
    conv_kernel_size: int = 4
    mlp_hidden_dim: int = 32768
    num_q_heads: int = 256
    max_position_embeddings: int = 8192

    intermed_mem: int = 65536 # 8192+37120+(8192*2+256*8*2)*2 + 8192*2*3 = 65536
    peak_intermed: int = 37120
    states: int = 256*8192*2
    output_mem: int = 8192
    cache_store: bool = False
    parameter_count: int = 438_403_328
    # total params: 8192*37120 + 4*(8192+256*8*2) + 8192 + 256 + 8192*2*8192 = 438403328
    type_name: str = "mamba2-nemotron56b-block-mamba"

    # keep layer structure open for future detailed sizing
    layers: List = field(default_factory=lambda: [['RMS_Norm'],['FC'],['Conv1D','Softplus'],['Silu'],['SSM'],['RMS_Norm'],['FC'],])
    layer_configs: List[List[baselayerConfig]] = field(default_factory=lambda: [
        [RMS_NormConfig(in_size=8192, out_size=8192)],
        [FCConfig(f_in=8192, f_out=37120)],
        [Conv1DConfig(kernel_size=4, c_in=8192*2+256*8*2, c_out=8192*2+256*8*2), SoftplusConfig(in_size=112, out_size=112)],
        [SiluConfig(in_size=8192*2+256*8*2, out_size=8192*2+256*8*2)], # 3xED+3xstates z, x and BC
        [SSMConfig(state_size=256, ED=8192*2)],
        [RMS_NormConfig(in_size=8192*2, out_size=8192*2)],
        [FCConfig(f_in=8192*2, f_out=8192)],
    ])

    @staticmethod
    def get_name():
        return "mamba2-nemotron56b-block2"


@dataclass
class Attention_Nemotron56B_BlockT_Config(BaseBlockConfig):
    """Attention-style transformer block for Nemotron-H-56B (star '*' in pattern)."""
    num_layers: int = 6
    num_q_heads: int = 64
    num_kv_heads: int = 8
    embedding_dim: int = 8192
    mlp_hidden_dim: int = 32768
    max_position_embeddings: int = 8192
    rope_theta: Optional[float] = 500000
    vocab_size: int = 131072

    intermed_mem: int = 73728
    peak_intermed: int = 32768
    states: int = 1024*2 # 8 KV heads
    output_mem: int = 8192
    cache_store: bool = True

    parameter_count: int = 620_756_992
    # MHA - 83886080 + FFN - 8192*32768*2 = 620756992
    type_name: str = "nemotronh-56b-block-transformer"

    layers: List[str] = field(default_factory=lambda: [['RMS_Norm'], ['MHA'], ['RMS_Norm'], ['FC'], ['Silu'], ['FC']])
    layer_configs: List[List[baselayerConfig]] = field(default_factory=lambda: [
        [RMS_NormConfig(in_size=8192, out_size=8192)],
        [MHA_NormConfig(embedding_dim=8192, q_heads=64, kv_heads=8)],
        [RMS_NormConfig(in_size=8192, out_size=8192)],
        [FCConfig(f_in=8192, f_out=32768)],
        [SiluConfig(in_size=32768, out_size=32768)],
        [FCConfig(f_in=32768, f_out=8192)]
    ])

    @staticmethod
    def get_name():
        return "nemotronh-56b-block-transformer"

@dataclass
class Zamba2_7B_BlockM_Config(BaseBlockConfig):
    """Mamba-style block for Zyphra Zamba2-7B (M block)."""
    num_layers: int = 7
    embedding_dim: int = 3584
    ssm_state_dim: int = 64
    conv_kernel_size: int = 4
    mlp_hidden_dim: int = 4096
    num_q_heads: int = 64
    max_position_embeddings: int = 4096

    intermed_mem: int = 50656 # 14704 + 7224*2 + 3584*2*3 = 50656
    peak_intermed: int = 14704
    states: int = 64*3584*2
    output_mem: int = 3584
    cache_store: bool = False
    type_name: str = "zamba2-7b-block-mamba"
    parameter_count: int = 78_419_056
    # 112 + 7424*4 + 14704*3584 + 3584*7168 = 78419056

    layers: List = field(default_factory=lambda: [['FC'],['Conv1D','Softplus'],['Silu'],['SSM'],['RMS_Norm'],['FC']])
    layer_configs: List[List[baselayerConfig]] = field(default_factory=lambda: [
        [FCConfig(f_in=3584, f_out=14704)],
        [Conv1DConfig(kernel_size=4, c_in=7424, c_out=7424), SoftplusConfig(in_size=64, out_size=64)],
        [SiluConfig(in_size=7424, out_size=7424)],
        [SSMConfig(state_size=64, ED=3584*2)],
        [RMS_NormConfig(in_size=3584*2, out_size=3584*2)],
        [FCConfig(f_in=3584*2, f_out=3584)]
    ])

    @staticmethod
    def get_name():
        return "zamba2-7b-mamba-block"


@dataclass
class Attention_Zamba2_7B_BlockT_Config(BaseBlockConfig):
    """Attention-style transformer block for Zyphra Zamba2-7B (full attention)."""
    num_layers: int = 7
    num_q_heads: int = 32
    num_kv_heads: int = 32
    embedding_dim: int = 3584
    mlp_hidden_dim: int = 14336
    max_position_embeddings: int = 4096
    rope_theta: Optional[float] = 500000
    vocab_size: int = 65536

    intermed_mem: int = 75264 # 7168+7168+28672+14336*2 + 3584 = 75264
    peak_intermed: int = 28672
    states: int = 3584*2 # 32 KV heads
    output_mem: int = 3584
    cache_store: bool = True

    parameter_count: int = 333_971_456
    # 3584*14336 + 28672*3584 + 7168*7168*3 + 3584*7168 = 333971456
    type_name: str = "zamba2-7b-block-transformer"

    layers: List[str] = field(default_factory=lambda: [['RMS_Norm'], ['MHA'], ['RMS_Norm'], ['FC'], ['Silu'], ['FC']])
    layer_configs: List[List[baselayerConfig]] = field(default_factory=lambda: [
        [RMS_NormConfig(in_size=7168, out_size=7168)],
        [MHA_NormConfig(embedding_dim=7168, kv_heads=32, q_heads=32)],
        [RMS_NormConfig(in_size=7168, out_size=7168)],
        [FCConfig(f_in=3584, f_out=28672)],
        [SiluConfig(in_size=14336, out_size=14336)],
        [FCConfig(f_in=14336, f_out=3584)],
        [FCConfig(f_in=3584, f_out=3584)] # MLP
    ])

    @staticmethod
    def get_name():
        return "zamba2-7b-block-transformer"


@dataclass
class AttentionNoWeights_Zamba2_7B_BlockT_Config(Attention_Zamba2_7B_BlockT_Config):
    """Attention-style block variant that does not store separate attention weights (single linear for output storage).

    The sizing is identical to the full attention block; semantics differ at runtime/implementation level.
    """
    num_layers: int = 7
    num_q_heads: int = 32
    num_kv_heads: int = 32
    embedding_dim: int = 3584
    mlp_hidden_dim: int = 14336
    max_position_embeddings: int = 4096
    rope_theta: Optional[float] = 500000
    vocab_size: int = 65536

    intermed_mem: int = 75264 # 7168+7168+28672+14336*2 + 3584 = 75264
    peak_intermed: int = 28672
    states: int = 3584*2 # 32 KV heads
    output_mem: int = 3584
    cache_store: bool = True

    #  3584*3584 =  12845056
    type_name: str = "zamba2-7b-block-transformer-no-weights"
    # Slightly reduced parameter count to reflect omitted weight storage
    parameter_count: int = 12_845_056

    layers: List[str] = field(default_factory=lambda: [['RMS_Norm'], ['MHA'], ['RMS_Norm'], ['FC'], ['Silu'], ['FC']])
    layer_configs: List[List[baselayerConfig]] = field(default_factory=lambda: [
        [RMS_NormConfig(in_size=7168, out_size=7168)],
        [MHA_NormConfig(embedding_dim=7168, kv_heads=32, q_heads=32)],
        [RMS_NormConfig(in_size=7168, out_size=7168)],
        [FCConfig(f_in=3584, f_out=28672)],
        [SiluConfig(in_size=14336, out_size=14336)],
        [FCConfig(f_in=14336, f_out=3584)],
        [FCConfig(f_in=3584, f_out=3584)] # MLP
    ])


    @staticmethod
    def get_name():
        return "zamba2-7b-block-transformer-no-weights"


@dataclass
class Jamba_Tiny_BlockM_Config(BaseBlockConfig):
    """Mamba-style block for ai21labs/Jamba-tiny-dev."""
    num_layers: int = 7
    embedding_dim: int = 1024
    ssm_state_dim: int = 16
    conv_kernel_size: int = 4
    mlp_hidden_dim: int = 2048
    num_q_heads: int = 8
    max_position_embeddings: int = 262144

    intermed_mem: int = 4096
    peak_intermed: int = 2048
    states: int = 16 * 1024
    output_mem: int = 512
    cache_store: bool = False
    type_name: str = "jamba-tiny-block-mamba"
    parameter_count: int = 15_750_000

    layers: List = field(default_factory=lambda: [['FC'], ['Conv1D', 'Softplus'], ['Silu'], ['SSM'], ['RMS_Norm'], ['FC']])
    layer_configs: List[List[baselayerConfig]] = field(default_factory=lambda: [
        [FCConfig(f_in=512, f_out=1024 * 2 + 16 * 8 * 2 + 32)],
        [Conv1DConfig(kernel_size=4, c_in=1024 + 16 * 8 * 2, c_out=1024 + 16 * 8 * 2), SoftplusConfig(in_size=32, out_size=32)],
        [SiluConfig(in_size=1024 + 16 * 8 * 2, out_size=1024 + 16 * 8 * 2)],
        [SSMConfig(state_size=16, ED=1024)],
        [RMS_NormConfig(in_size=1024, out_size=1024)],
        [FCConfig(f_in=1024, f_out=512)]
    ])

    @staticmethod
    def get_name():
        return "jamba-tiny-block-mamba"


@dataclass
class Attention_Jamba_Tiny_BlockT_Config(BaseBlockConfig):
    """Attention-style block for ai21labs/Jamba-tiny-dev."""
    num_layers: int = 6
    num_q_heads: int = 8
    num_kv_heads: int = 2
    embedding_dim: int = 512
    mlp_hidden_dim: int = 2048
    max_position_embeddings: int = 262144
    vocab_size: int = 65536

    intermed_mem: int = 4096
    peak_intermed: int = 2048
    states: int = 128 * 2
    output_mem: int = 512
    cache_store: bool = True
    parameter_count: int = 15_750_000
    type_name: str = "jamba-tiny-block-transformer"

    layers: List[str] = field(default_factory=lambda: [['RMS_Norm'], ['MHA'], ['RMS_Norm'], ['FC'], ['Silu'], ['FC']])
    layer_configs: List[List[baselayerConfig]] = field(default_factory=lambda: [
        [RMS_NormConfig(in_size=512, out_size=512)],
        [MHA_NormConfig(embedding_dim=512, kv_heads=2, q_heads=8)],
        [RMS_NormConfig(in_size=512, out_size=512)],
        [FCConfig(f_in=512, f_out=2048)],
        [SiluConfig(in_size=2048, out_size=2048)],
        [FCConfig(f_in=2048, f_out=512)]
    ])

    @staticmethod
    def get_name():
        return "jamba-tiny-block-transformer"


@dataclass    
class FCConfig(baselayerConfig):
    f_in: int = 8192
    f_out: int = 8192

@dataclass
class RMS_NormConfig(baselayerConfig):
    in_size: int = 8192
    out_size: int = 8192
    bs: int = 1

@dataclass 
class SiluConfig(baselayerConfig):
    in_size: int = 8192
    out_size: int = 8192

@dataclass    
class MHA_NormConfig(baselayerConfig):
    embedding_dim: int = 8192
    kv_heads: int = 8
    q_heads: int = 32
    
@dataclass
class Conv1DConfig(baselayerConfig):
    c_in: int = 8192
    c_out: int = 8192
    f_in: int = 8192
    f_out: int = 8192
    kernel_size: int = 3

@dataclass
class SSMConfig(baselayerConfig):   
    state_size: int = 64
    ED: int = 8192   

@dataclass
class SoftplusConfig(baselayerConfig):
    in_size: int = 8192
    out_size: int = 8192
    
@dataclass
class SquareReluConfig(baselayerConfig):
    in_size: int = 8192
    out_size: int = 8192
    

@dataclass
class Llama3_8BModelConfig(Llama2ModelConfig):
    num_layers: int = 32
    num_q_heads: int = 32
    num_kv_heads: int = 8
    embedding_dim: int = 4096
    mlp_hidden_dim: int = 14336
    max_position_embeddings: int = 4096
    rope_theta: Optional[float] = 500000
    vocab_size: int = 128256
    param_scale: float = 0.4
    block_config: Llama3_8B_BlockT_Config = field(
        default_factory=Llama3_8B_BlockT_Config,
        metadata={"help": "Block configuration for Llama3 8B model."}
    )
    hybrid: bool = False
    num_A_blocks: int = 32
    num_M_blocks: int = 0
    block_list: Optional[List[str]] = field(
        default= lambda: [
            "RMS_Norm", "MHA", "RMS_Norm", "FC", "Silu", "FC"
        ],
    )
    hybrid_blocks: List = field(
        default_factory= lambda: [Mamba2_3B_BlockM_Config(),Llama3_8B_BlockT_Config()],
        metadata={"help": "List of block types to use in the hybrid model."}
    )
    
    block_type_sequence: List[int] = field(
        default_factory=lambda: [1 for _ in range(32)],
        metadata={
            "help": "A list of integers where each integer is the index into hybrid_blocks for that layer. "
                    "The length must equal the total number of layers."
        }
    )

    @staticmethod
    def get_name():
        return "llama3-8b"

@dataclass
# https://huggingface.co/nvidia/Nemotron-H-4B-Base-8K/blob/main/config.json
# "M-\M-\M-\M\*-\M-\M-\M-\M-\M\*-\M-\M-\M-\M-\M\*-\M-\M-\M-\M-\M\*-\M-\M-\M-\M-\M-\"
class Nemotron_4BModelConfig(BaseModelConfig):
    num_layers: int = 28 # 52 if all hidden layers
    num_q_heads: int = 32
    num_kv_heads: int = 8
    embedding_dim: int = 3072
    mlp_hidden_dim: int = 12288
    max_position_embeddings: int = 8192
    rope_theta: Optional[float] = 500000
    vocab_size: int = 128256
    hybrid: bool = True
    num_A_blocks: int = 4
    num_M_blocks: int = 24
    param_scale: float = 0.55
    hybrid_blocks: List[BaseBlockConfig] = field(
        # 2 M just for the cases with 
        default_factory= lambda: [Mamba2_Nemo4B_BlockM_Config(),Mamba2_Nemo4B_BlockM2_Config(),Attention_Nemo4B_BlockT_Config()],
        metadata={"help": "List of block types to use in the hybrid model."}
    )
    block_type_sequence: List[int] = field(
        default_factory=lambda: [
            0,0,0,1,2, 
            0,0,0,0,1,2,
            0,0,0,0,1,2,
            0,0,0,0,1,2,
            0,0,0,0,0
        ],
        metadata={
            "help": "A list of integers where each integer is the index into hybrid_blocks for that layer. "
                    "The length must equal the total number of layers."
        }
    )
    @staticmethod
    def get_name():
        return "nemotronh-4b"

# https://huggingface.co/nvidia/Nemotron-H-56B-Base-8K/blob/main/config.json
# M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M-
@dataclass
class Nemotron_56BModelConfig(BaseModelConfig):
    """Top-level model config for Nvidia Nemotron-H-56B (hybrid Mamba/Attention/FFN).

    Notes/assumptions:
    - Uses the published Nemotron-H-56B hyperparameters from the HuggingFace repo
      (hidden_size=8192, intermediate_size=32768, num_hidden_layers=118, vocab_size=131072).
    - The exact hybrid pattern is represented loosely here. If you want an exact
      per-layer pattern, we can replace `block_type_sequence` with the precise
      mapping from the `hybrid_override_pattern` string in the upstream config.
    """
    num_layers: int = 64 # 118
    num_q_heads: int = 64
    num_kv_heads: int = 8
    embedding_dim: int = 8192
    mlp_hidden_dim: int = 32768
    max_position_embeddings: int = 8192
    rope_theta: Optional[float] = 500000
    vocab_size: int = 131072
    hybrid: bool = True
    num_A_blocks: int = 10
    num_M_blocks: int = 54
    hybrid_blocks: List[BaseBlockConfig] = field(
        default_factory=lambda: [Mamba2_Nemotron56B_BlockM_Config(), Mamba2_Nemotron56B_BlockM2_Config(), Attention_Nemotron56B_BlockT_Config()],
        metadata={"help": "List of block types to use in the hybrid model."}
    )

    # A conservative pattern: repeat 4 M blocks then 1 attention block, repeated.
    # This is an assumption to create a valid sequence of length `num_layers`.
    block_type_sequence: List[int] = field(
        default_factory=lambda: [
            0,0,0,1,2,
            0,0,0,0,1,2,
            0,0,0,0,1,2,
            0,0,0,0,1,2,
            0,0,0,0,1,2,
            0,0,0,0,1,2,
            0,0,0,0,1,2,
            0,0,0,0,1,2,
            0,0,0,0,1,2,
            0,0,0,0,1,2,
            0,0,0,0,0
        ],
        metadata={
            "help": "Default block-type mapping (0=Mamba,1=Mamba+FFN,2=Attention). Replace with exact mapping if available."
        }
    )

    @staticmethod
    def get_name():
        return "nemotronh-56b"

# https://huggingface.co/Zyphra/Zamba2-7B/blob/main/config.json
@dataclass
class Zyphra_Zamba2_7BModelConfig(BaseModelConfig):
    """Top-level model config for Zyphra Zamba2-7B, composed as a hybrid of Mamba and Attention blocks.

    This follows the pattern used in `Nemotron_4BModelConfig` where multiple block types are
    available and a per-layer `block_type_sequence` selects the block implementation.
    """
    num_layers: int = 94
    num_q_heads: int = 32
    num_kv_heads: int = 32
    embedding_dim: int = 3584
    mlp_hidden_dim: int = 14336
    max_position_embeddings: int = 4096
    rope_theta: Optional[float] = 500000
    vocab_size: int = 65536
    hybrid: bool = True
    num_A_blocks: int = 13
    num_M_blocks: int = 81
    hybrid_blocks: List[BaseBlockConfig] = field(
        default_factory=lambda: [
            Zamba2_7B_BlockM_Config(),
            Attention_Zamba2_7B_BlockT_Config(),
            AttentionNoWeights_Zamba2_7B_BlockT_Config(),
        ],
        metadata={"help": "List of block types to use in the hybrid model."}
    )

    # Default conservative pattern: groups of three M blocks, then one attention-no-weights, repeating.
    block_type_sequence: List[int] = field(
        default_factory=lambda: [
            0,0,0,0,0,0,1,0,
            0,0,0,0,1,0,
            0,0,0,0,0,2,0,
            0,0,0,0,0,2,0,
            0,0,0,0,0,2,0,
            0,0,0,0,0,2,0,
            0,0,0,0,0,2,0,
            0,0,0,0,0,2,0,
            0,0,0,0,0,2,0,
            0,0,0,0,0,2,0,
            0,0,0,0,0,2,0,
            0,0,0,0,0,2,0,
            0,0,0,0,0,2,0,
            0,0,0
        ],
        metadata={"help": "Mapping from layer index to hybrid_blocks index. Length must equal num_layers."}
    )

    @staticmethod
    def get_name():
        return "zamba2-7b"

#TODO
@dataclass
class Jamba_MiniModelConfig(BaseModelConfig):
    num_layers: int = 32
    num_q_heads: int = 32
    num_kv_heads: int = 8
    embedding_dim: int = 4096
    mlp_hidden_dim: int = 14336
    max_position_embeddings: int = 256000
    rope_theta: Optional[float] = 500000
    vocab_size: int = 128256
    hybrid: bool = True
    num_A_blocks: int = 8
    num_M_blocks: int = 24
    param_scale: float = 0.5
    hybrid_blocks: List[BaseBlockConfig] = field(
        default_factory= lambda: [Mamba2_3B_BlockM_Config(),Llama3_8B_BlockT_Config()],
        metadata={"help": "List of block types to use in the hybrid model."}
    )
    block_type_sequence: List[int] = field(
        default_factory=lambda: [
            0, 0, 0, 1,
            0, 0, 0, 1,
            0, 0, 0, 1,
            0, 0, 0, 1,
            0, 0, 0, 1,
            0, 0, 0, 1,
            0, 0, 0, 1,
            0, 0, 0, 1,
        ],
        metadata={
            "help": "A list of integers where each integer is the index into hybrid_blocks for that layer. "
                    "The length must equal the total number of layers."
        }
    )
    @staticmethod
    def get_name():
        return "jamba-mini"


@dataclass
class Jamba_TinyModelConfig(BaseModelConfig):
    """Top-level model config for ai21labs/Jamba-tiny-dev."""
    num_layers: int = 16
    num_q_heads: int = 8
    num_kv_heads: int = 2
    embedding_dim: int = 512
    mlp_hidden_dim: int = 2048
    max_position_embeddings: int = 262144
    vocab_size: int = 65536
    hybrid: bool = True
    num_A_blocks: int = 2
    num_M_blocks: int = 14
    param_scale: float = 0.5
    hybrid_blocks: List[BaseBlockConfig] = field(
        default_factory=lambda: [Jamba_Tiny_BlockM_Config(), Attention_Jamba_Tiny_BlockT_Config()],
        metadata={"help": "List of block types to use in the hybrid model."}
    )
    block_type_sequence: List[int] = field(
        default_factory=lambda: [
            0, 0, 0, 0, 1, 0, 0, 0,
            0, 0, 0, 0, 1, 0, 0, 0,
        ],
        metadata={
            "help": "A list of integers where each integer is the index into hybrid_blocks for that layer. "
                    "The length must equal the total number of layers."
        }
    )

    @staticmethod
    def get_name():
        return "jamba-tiny"
    
@dataclass
class Mamba2_3BModelConfig(BaseModelConfig):
    num_layers: int = 64
    embedding_dim: int = 4096
    max_position_embeddings: int = 512
    param_scale = 0.6
    block_config: Mamba2_3B_BlockM_Config = field(
        default_factory=Mamba2_3B_BlockM_Config,
        metadata={"help": "Block configuration for Mamba2 3B model."}
    )
    hybrid: bool = False
    num_A_blocks: int = 0
    num_M_blocks: int = 64
    block_list: Optional[List[str]] = field(
        default= lambda: [
            "FC", "Conv1D", "Softplus", "Silu", "SSM", "RMS_Norm", "FC"
        ],
    )
    hybrid_blocks: List = field(
        default_factory= lambda: [Mamba2_3B_BlockM_Config(),Llama3_8B_BlockT_Config()],
        metadata={"help": "List of block types to use in the hybrid model."}
    )
    block_type_sequence: List[int] = field(
        default_factory=lambda: [0 for _ in range(64)],
        metadata={
            "help": "A list of integers where each integer is the index into hybrid_blocks for that layer. \
                    The length must equal the total number of layers."
        }
    )
    @staticmethod
    def get_name():
        return "mamba2-3b"
