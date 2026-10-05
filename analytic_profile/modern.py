"""Shape-explicit decoder rooflines shared by all network fidelity levels.

Each operator emits serial compute/HBM phases. Weights stream once per batch;
activations stay in SRAM when their live working set fits, otherwise the full
working set spills. Attention uses a causal, streaming score reduction. GDN uses
recurrent delta-rule work in both stages, not a calibrated chunked GPU kernel.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import inspect
import math

import Sim.common as common
from Sim.config.utils import get_all_subclasses
from Sim.entities.execution import ExecutionPhase
from Sim.metrics.monitor import analytics


@dataclass(frozen=True)
class OperatorWork:
    name: str
    macs: int = 0
    element_ops: int = 0
    weight_bytes: int = 0
    state_bytes: int = 0
    working_bytes: int = 0
    recurrent_state_bytes: int = 0
    recurrent_steps: int = 1


@dataclass(frozen=True)
class AcceleratorRates:
    macs_per_ns: float
    elements_per_ns: float
    sram_bytes_per_ns: float
    sram_capacity_bytes: int

    @classmethod
    def from_name(cls, name):
        family = name.split('_')[0]
        if family not in ('tscs', 'marca'):
            raise ValueError(f'Modern operator rates are not validated for {name}.')
        logic = common.logic_lib[name]
        config = logic['config']
        # The hardware library uses a 1 MiB SRAM macro; rates are decimal GB/s.
        return cls(config[family] * logic['core'],
                   config['out_ports'] * logic['core'],
                   config['sram_tp'] * logic['sram'],
                   1024 * 1024 * logic['sram'])


class BaseOperatorProfile(ABC):
    @classmethod
    def find(cls, name):
        from analytic_profile import attention, moe
        for subclass in get_all_subclasses(cls):
            if not inspect.isabstract(subclass) and subclass.get_name() == name:
                return subclass
        return None

    @staticmethod
    @abstractmethod
    def get_name():
        raise NotImplementedError

    @abstractmethod
    def work(self, **kwargs):
        raise NotImplementedError

    def profile(self, logic_name, ext_bw, **kwargs):
        if common.ByteperParam != 1:
            raise ValueError('Explicit decoder profiles require the one-byte precision contract.')
        if ext_bw <= 0:
            raise ValueError('External bandwidth must be positive.')
        rates = AcceleratorRates.from_name(logic_name)
        phases = []
        records = self.work(**kwargs)
        for item in records:
            spilled = item.working_bytes if item.working_bytes > rates.sram_capacity_bytes else 0
            memory_bytes = item.weight_bytes + item.state_bytes + spilled
            if item.recurrent_state_bytes > rates.sram_capacity_bytes:
                memory_bytes += 2 * item.recurrent_state_bytes * (item.recurrent_steps - 1)
            compute_ns = (item.macs / rates.macs_per_ns
                          + item.element_ops / rates.elements_per_ns)
            sram_ns = (item.weight_bytes + item.state_bytes + item.working_bytes) / rates.sram_bytes_per_ns
            phases.append(ExecutionPhase(max(compute_ns, sram_ns), memory_bytes))
        total = sum(max(p.compute_ns, p.memory_bytes / ext_bw) for p in phases)
        memory_time = sum(p.memory_bytes / ext_bw for p in phases)
        utilization = 100 * sum(p.compute_ns for p in phases) / total if total else 0
        return analytics(math.ceil(total), math.ceil(memory_time),
                         sum(p.memory_bytes for p in phases), [], utilization, 0, phases)


class GQAProfile(BaseOperatorProfile):
    @staticmethod
    def get_name():
        return 'ModernGQA'

    def work(self, embedding_dim, q_heads, kv_heads, head_dim, rotary_dim,
             output_gate, bs, batch_size, L_seq, stage, **kwargs):
        if q_heads % kv_heads or not 0 < rotary_dim <= head_dim:
            raise ValueError('Invalid GQA head geometry.')
        d, q, k = embedding_dim, q_heads * head_dim, kv_heads * head_dim
        n = batch_size * bs
        projection_dim = q * (1 + output_gate) + 2 * k
        cached = L_seq - bs if stage == 'prefill' else L_seq - 1
        pairs = bs * cached + bs * (bs + 1) // 2 if stage == 'prefill' else L_seq
        attention_macs = 2 * batch_size * q * pairs
        cache_read = batch_size * 2 * k * cached
        return [
            OperatorWork('input_norm', element_ops=5*n*d, weight_bytes=d, working_bytes=2*n*d),
            OperatorWork('qkv_gate_projection', macs=n*d*projection_dim,
                         weight_bytes=d*projection_dim, working_bytes=n*(d+projection_dim)),
            OperatorWork('qk_norm_rope', element_ops=n*(5*(q+k)+3*(q_heads+kv_heads)*rotary_dim),
                         weight_bytes=2*head_dim, working_bytes=2*n*(q+k)),
            OperatorWork('causal_attention', macs=attention_macs,
                         element_ops=5*batch_size*q_heads*pairs,
                         state_bytes=cache_read+2*n*k, working_bytes=n*(2*q+2*k)),
            OperatorWork('attention_gate', element_ops=2*n*q*output_gate,
                         working_bytes=2*n*q if output_gate else 0),
            OperatorWork('output_projection', macs=n*q*d, weight_bytes=q*d,
                         working_bytes=n*(q+d)),
            OperatorWork('attention_residual', element_ops=n*d, working_bytes=3*n*d),
        ]


class SwiGLUProfile(BaseOperatorProfile):
    @staticmethod
    def get_name():
        return 'ModernSwiGLU'

    def work(self, embedding_dim, intermediate_dim, bs, batch_size, **kwargs):
        d, h, n = embedding_dim, intermediate_dim, bs * batch_size
        return [
            OperatorWork('post_attention_norm', element_ops=5*n*d, weight_bytes=d, working_bytes=2*n*d),
            OperatorWork('up_gate_projection', macs=2*n*d*h, weight_bytes=2*d*h,
                         working_bytes=n*(d+2*h)),
            OperatorWork('silu_product', element_ops=5*n*h, working_bytes=3*n*h),
            OperatorWork('down_projection', macs=n*d*h, weight_bytes=d*h, working_bytes=n*(h+d)),
            OperatorWork('mlp_residual', element_ops=n*d, working_bytes=3*n*d),
        ]


class GatedDeltaNetProfile(BaseOperatorProfile):
    @staticmethod
    def get_name():
        return 'ModernGDN'

    def work(self, embedding_dim, key_heads, value_heads, key_head_dim,
             value_head_dim, conv_kernel_size, state_bytes, bs, batch_size,
             stage, **kwargs):
        if value_heads % key_heads:
            raise ValueError('GDN value heads must be divisible by key heads.')
        d, k, v = embedding_dim, key_heads*key_head_dim, value_heads*value_head_dim
        c, n = 2*k+v, batch_size*bs
        state = value_heads*key_head_dim*value_head_dim
        cached = batch_size*(state*state_bytes+c*conv_kernel_size)
        # State is initialized during prefill, then read and written once per call.
        state_traffic = cached if stage == 'prefill' else 2*cached
        return [
            OperatorWork('input_norm', element_ops=5*n*d, weight_bytes=d, working_bytes=2*n*d),
            OperatorWork('qkv_z_beta_decay_projection', macs=n*d*(c+v+2*value_heads),
                         weight_bytes=d*(c+v+2*value_heads), working_bytes=n*(d+c+v+2*value_heads)),
            OperatorWork('depthwise_conv_silu', macs=n*c*conv_kernel_size,
                         element_ops=4*n*c, weight_bytes=c*conv_kernel_size,
                         working_bytes=2*n*c),
            OperatorWork('qk_norm_beta_decay', element_ops=n*(5*2*k+8*value_heads),
                         weight_bytes=2*value_heads, working_bytes=2*n*(k+value_heads)),
            # Two state-vector products and a rank-one update, plus decay.
            OperatorWork('delta_recurrence', macs=3*n*state,
                         element_ops=n*(state+2*v), state_bytes=state_traffic,
                         working_bytes=batch_size*state*state_bytes+n*(c+v),
                         recurrent_state_bytes=batch_size*state*state_bytes, recurrent_steps=bs),
            OperatorWork('gated_output_norm', element_ops=9*n*v,
                         weight_bytes=value_head_dim, working_bytes=3*n*v),
            OperatorWork('output_projection', macs=n*v*d, weight_bytes=v*d,
                         working_bytes=n*(v+d)),
            OperatorWork('attention_residual', element_ops=n*d, working_bytes=3*n*d),
        ]
