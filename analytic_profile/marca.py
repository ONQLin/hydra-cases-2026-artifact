# ...existing code...
import numpy as np
from typing import Tuple, List, Dict, Any
import math
from Sim.metrics.monitor import analytics
from Sim.entities.execution import ExecutionPhase
import Sim.common as common

from .Base_Accmodel import BaseAccModel
import inspect


class MarcaAccModel(BaseAccModel):
    """
    Systolic / memory-controller-like accelerator performance model.
    Converted from top-level functions in marca.py into a BaseAccModel subclass.
    """
    acc_freq: float = 1e9  # default 1 GHz

    def __init__(self, acc_freq: float | None = None):
        if acc_freq is not None:
            self.acc_freq = acc_freq

    @staticmethod
    def get_name() -> str:
        return "marca"

    @classmethod
    def _logic(cls, logic_name: str) -> Dict[str, Any]:
        return cls._load_logic(logic_name) or common.logic_lib[logic_name]

    @classmethod
    def get_kernel(cls, kernel_name: str, *args, **kwargs) -> Any:
        kernel_calls = {
            'RMS_Norm': cls.rms_norm_est,
            'MHA_p': cls.get_pf_attention_latency,
            'MHA_d': cls.get_dc_attention_latency,
            'Softplus': cls.softplus_est,
            'Silu': cls.silu_est,
            'SSM_p': cls.get_pf_ssm_latency,
            'SSM_d': cls.get_dc_ssm_latency,
            'FC': cls.get_pf_mlp_latency,
            'Conv1D': cls.get_pf_conv1d_latency,
            'SquareRelu': cls.square_relu_est,
        }

        if kernel_name not in kernel_calls:
            raise ValueError(f"Unknown kernel: {kernel_name}")

        fn = kernel_calls[kernel_name]

        # inspect signature to filter args/kwargs to only those accepted by the target
        sig = inspect.signature(fn)
        params = list(sig.parameters.values())

        has_var_pos = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params)
        has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params)

        if has_var_pos:
            if has_var_kw:
                return fn(*args, **kwargs)
            else:
                allowed_kw = {p.name for p in params if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)}
                filtered_kwargs = {k: v for k, v in kwargs.items() if k in allowed_kw}
                return fn(*args, **filtered_kwargs)
        else:
            param_names = [p.name for p in params if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)]
            if param_names and param_names[0] in ("cls", "self"):
                param_names = param_names[1:]

            call_kwargs = {}
            for i, a in enumerate(args):
                if i < len(param_names):
                    call_kwargs[param_names[i]] = a

            if has_var_kw:
                call_kwargs.update(kwargs)
            else:
                allowed = set(param_names)
                for k, v in kwargs.items():
                    if k in allowed:
                        call_kwargs[k] = v

            return fn(**call_kwargs)

    # --- converted kernel implementations (original logic preserved) ---
    @classmethod
    def get_pf_ssm_latency(cls, batch_size: int, L_seq: int, ED: int, state_size: int,
                           logic_name="marca_d", ext_bw=128, acc_freq: float | None = None):
        acc_freq = acc_freq or cls.acc_freq
        specified_logic = cls._logic(logic_name)

        MACs = specified_logic['config']['marca'] * specified_logic['core']    # MACs / ns (tensor-core-like)
        GActs = specified_logic['config']['out_ports'] * specified_logic['core']     # GActs / s (elem-wise)
        SRAM_bw = specified_logic['config']['sram_tp'] * specified_logic['sram'] # GB/s
        SRAM_cap = 10e6 * specified_logic['sram']                                 # bytes
        elem_ops = specified_logic['config']['out_ports'] * specified_logic['core']

        B = batch_size
        L = L_seq
        D = ED
        n = state_size

        def sec_to_cycles(t_s: float) -> int:
            return int(t_s * acc_freq)

        def ns_to_cycles(t_ns: float) -> int:
            return int(t_ns * (acc_freq * 1e-9))

        # 1. Compute latency
        elem_ops_1 = B * L * D
        elem_ops_2 = B * L * D * n
        non_elem_ops_1 = B * L * D
        
        # Ops requirements/GOps
        Compute_latency1 = elem_ops_1/elem_ops + elem_ops_2/elem_ops + non_elem_ops_1/GActs
        Compute_latency1 = ns_to_cycles(Compute_latency1)

        elem_ops_3 = 3 * B * L * D * n
        Compute_latency2 = ns_to_cycles(elem_ops_3/elem_ops)
        macs_linear = B * L * D * n
        Compute_latency3 = ns_to_cycles(macs_linear / MACs)
        compute_latency = Compute_latency1 + Compute_latency2 + Compute_latency3

        # 2. SRAM latency
        u_in = B * L * D
        a_0 = D
        dt = B * L * D
        a_1 = B * L * D
        a_2 = B * L * D
        b_0 = B * L * n
        b_1 = B * L * D * n
        c_0 = B * L * n
        x_01 = 5 * B * L * D * n + u_in
        y_t = B * L * D + u_in
        total_sram_bytes = u_in + a_0 + dt + a_1 + a_2 + b_0 + b_1 + c_0 + x_01 + y_t
        sram_time_s = total_sram_bytes / max(1e-12, (SRAM_bw * 1e9))
        sram_latency = sec_to_cycles(sram_time_s)

        # 3. HBM traffic
        
        bytes_input  = (B * L * D) # f
        bytes_paramA = D
        bytes_paramB = L * n * B
        bytes_paramC = L * n * B
        bytes_output = B * L * D
        if bytes_input + bytes_paramA + bytes_paramB + bytes_paramC <= SRAM_cap:
            bytes_hbm = bytes_output
        else:
            n_tiles = math.ceil((bytes_input + bytes_paramA + bytes_paramB + bytes_paramC) / SRAM_cap)
            bytes_hbm = (n_tiles-1) * SRAM_cap + bytes_output

        hbm_time_s = bytes_hbm / max(1e-12, (ext_bw * 1e9))
        hbm_latency = sec_to_cycles(hbm_time_s)

        total_latency = max(hbm_latency, compute_latency, sram_latency)
        execution_phases = [ExecutionPhase(max(compute_latency, sram_latency), bytes_hbm)]

        return analytics(total_latency, -1, -1, [], 80, 200, execution_phases)

    @classmethod
    def get_dc_ssm_latency(cls, batch_size: int, ED: int, state_size: int,
                           logic_name="marca_d", ext_bw=128, acc_freq: float | None = None):
        acc_freq = acc_freq or cls.acc_freq
        specified_logic = cls._logic(logic_name)

        MACs_per_ns = specified_logic['config']['marca'] * specified_logic['core']
        GActs = specified_logic['config']['out_ports'] * specified_logic['core']
        SRAM_bw = specified_logic['config']['sram_tp'] * specified_logic['sram']
        SRAM_cap = 10e6 * specified_logic['sram']
        Elem_bw = specified_logic['config']['out_ports'] * specified_logic['core']

        B, D, N = batch_size, ED, state_size

        def sec_to_cycles(t_s: float) -> int:
            return int(t_s * acc_freq)

        def ns_to_cycles(t_ns: float) -> int:
            return int(t_ns * (acc_freq * 1e-9))

        elem_ops = B * D * N + B * D
        exp_ops  = B * D
        t_elem_s = elem_ops / max(1, Elem_bw) / 1e9
        t_exp_s  = exp_ops  / max(1, GActs)  / 1e9
        compute_stage1 = ns_to_cycles(t_elem_s + t_exp_s)

        recur_ops = 3 * B * D * N
        t_recur_s = recur_ops / max(1, Elem_bw) / 1e9
        compute_stage2 = ns_to_cycles(t_recur_s)

        mac_ops = B * D * N
        t_mac_ns = mac_ops / max(1, MACs_per_ns)
        compute_stage3 = ns_to_cycles(t_mac_ns)
        compute_latency = compute_stage1 + compute_stage2 + compute_stage3

        bytes_sram = 11 * (B * D) + (B * D * N) * 2 + (B * N) * 2
        sram_time_s = bytes_sram / max(1e-12, (SRAM_bw * 1e9))
        sram_latency = sec_to_cycles(sram_time_s)

        bytes_input  = (B * D)
        bytes_paramA = D 
        bytes_paramB = N * B
        bytes_paramC = N * B
        bytes_output = B * D
        
        if bytes_input + bytes_paramA + bytes_paramB + bytes_paramC <= SRAM_cap:
            bytes_hbm = bytes_output
        else:
            n_tiles = math.ceil((bytes_input + bytes_paramA + bytes_paramB + bytes_paramC) / SRAM_cap)
            bytes_hbm = (n_tiles-1) * SRAM_cap + bytes_output

        hbm_time_s = bytes_hbm / max(1e-12, (ext_bw * 1e9))
        hbm_latency = sec_to_cycles(hbm_time_s)

        total_latency = max(hbm_latency, compute_latency, sram_latency)
        execution_phases = [ExecutionPhase(max(compute_latency, sram_latency), bytes_hbm)]
        return analytics(total_latency, -1, -1, [], 80, 200, execution_phases)

    @classmethod
    def get_pf_mlp_latency(cls, bs: int, f_in: int, f_out: int, logic_name="marca_d", ext_bw=128,
                           acc_freq: float | None = None, batch_size: int = 1):
        acc_freq = acc_freq or cls.acc_freq
        specified_logic = cls._logic(logic_name)
        req_bs = batch_size

        MACs_per_ns = specified_logic['config']['marca'] * specified_logic['core']
        total_ops = 2 * bs * f_in * f_out * req_bs
        comp_time_ns = total_ops / MACs_per_ns
        comp_latency = int(comp_time_ns * (acc_freq * 1e-9))

        SRAM_bw = specified_logic['config']['sram_tp'] * specified_logic['sram']
        bytes_sram = (bs*f_in*req_bs + f_in*f_out + bs*f_out*req_bs)
        sram_time_s = bytes_sram / (SRAM_bw * 1e9)
        sram_latency = int(sram_time_s * acc_freq)

        SRAM_cap = 10e6 * specified_logic['sram']
        weight_bytes = f_in * f_out
        input_bytes = bs * f_in * req_bs
        output_bytes = bs * f_out * req_bs

        if weight_bytes <= SRAM_cap:
            bytes_hbm = weight_bytes + output_bytes
        else:
            n_tiles = math.ceil(weight_bytes / SRAM_cap)
            bytes_hbm = n_tiles * (weight_bytes / n_tiles) + output_bytes

        hbm_time_s = bytes_hbm / (ext_bw * 1e9)
        hbm_latency = int(hbm_time_s * acc_freq)

        total_latency = max(hbm_latency, comp_latency, sram_latency)
        execution_phases = [ExecutionPhase(max(comp_latency, sram_latency), bytes_hbm)]

        return analytics(total_latency, -1, -1, [], 80, 200, execution_phases)

    @classmethod
    def get_pf_conv1d_latency(cls, bs: int, c_in: int, c_out: int, f_in: int, f_out: int, kernel_size: int,
                              logic_name="marca_p", ext_bw=128, acc_freq: float | None = None, batch_size: int = 1):
        acc_freq = acc_freq or cls.acc_freq
        specified_logic = cls._logic(logic_name)
        req_bs = batch_size

        MACs_per_ns = specified_logic['config']['marca'] * specified_logic['core']
        total_ops = 2 * bs * c_in * kernel_size * req_bs
        comp_time_ns = total_ops / MACs_per_ns
        comp_latency = int(comp_time_ns * (acc_freq * 1e-9))

        SRAM_bw = specified_logic['config']['sram_tp'] * specified_logic['sram']
        bytes_sram = (bs*c_in*req_bs + c_in*kernel_size + bs*c_in*req_bs*req_bs)
        sram_time_s = bytes_sram / (SRAM_bw * 1e9)
        sram_latency = int(sram_time_s * acc_freq)

        SRAM_cap = 10e6 * specified_logic['sram']
        weight_bytes = c_in * kernel_size
        input_bytes = bs * c_in * req_bs
        output_bytes = bs * c_out * req_bs

        if weight_bytes <= SRAM_cap:
            bytes_hbm = output_bytes
        else:
            n_tiles = math.ceil((weight_bytes) / SRAM_cap)
            bytes_hbm = (n_tiles - 1) * SRAM_cap + output_bytes

        hbm_time_s = bytes_hbm / (ext_bw * 1e9)
        hbm_latency = int(hbm_time_s * acc_freq)

        total_latency = max(hbm_latency, comp_latency, sram_latency)
        execution_phases = [ExecutionPhase(max(comp_latency, sram_latency), bytes_hbm)]
        return analytics(total_latency, -1, -1, [], 80, 200, execution_phases)

    @classmethod
    def get_pf_attention_latency(cls, batch_size: int, L_seq: int, embedding_dim: int, q_heads: int, kv_heads: int,
                                 logic_name="marca_p", ext_bw=128, acc_freq: float | None = None) -> Tuple[int, int, int, List]:
        acc_freq = acc_freq or cls.acc_freq
        specified_logic = cls._logic(logic_name)

        MACs_per_ns = specified_logic['config']['marca'] * specified_logic['core']
        SRAM_bw = specified_logic['config']['sram_tp'] * specified_logic['sram']
        SRAM_cap = 10e6 * specified_logic['sram']

        B = batch_size
        L = L_seq
        D = embedding_dim
        d_k = D / kv_heads

        def sec_to_cycles(t_s: float) -> int:
            return int(t_s * acc_freq)

        def ns_to_cycles(t_ns: float) -> int:
            return int(t_ns * (acc_freq * 1e-9))

        ops_qkv = 6.0 * B * L * D * D
        comp_latency_qkv = ns_to_cycles(ops_qkv / MACs_per_ns)

        bytes_sram_qkv = (B * L * D) + (3 * D * D) + (3 * B * L * D)
        sram_time_qkv = bytes_sram_qkv / (SRAM_bw * 1e9)
        sram_latency_qkv = sec_to_cycles(sram_time_qkv)

        input_bytes = B * L * D
        weight_bytes = 3 * D * D
        kv_cache_wb = 2 * B * L * D
        Interm = weight_bytes + kv_cache_wb
        if weight_bytes <= SRAM_cap:
            bytes_hbm_qkv = weight_bytes + kv_cache_wb
            extra_hbm_bytes = 0
        else:
            n_tiles = math.ceil(Interm / SRAM_cap)
            bytes_hbm_qkv = weight_bytes + kv_cache_wb
            extra_hbm_bytes = (n_tiles - 1) * (weight_bytes)

        hbm_time_qkv = bytes_hbm_qkv / (ext_bw * 1e9)
        hbm_latency_qkv = sec_to_cycles(hbm_time_qkv)
        extra_hbm_latency = sec_to_cycles(extra_hbm_bytes / (ext_bw * 1e9))

        total_latency_qkv = max(hbm_latency_qkv, comp_latency_qkv, sram_latency_qkv) + extra_hbm_latency

        ops_qkt = 2.0 * B * q_heads * (L * L) * d_k
        comp_latency_qkt = ns_to_cycles(ops_qkt / MACs_per_ns)

        bytes_sram_qkt = (B * L * D) + (B * L * D) + (B * q_heads * L * L)
        sram_time_qkt = bytes_sram_qkt / (SRAM_bw * 1e9)
        sram_latency_qkt = sec_to_cycles(sram_time_qkt)

        Interm = (B * L * D) + (B * L * D) + (B * q_heads * L * L)
        if (B * L * D) + (B * L * D) <= SRAM_cap:
            hbm_latency_qkt = (B * L * D)
            extra_hbm_bytes = 0
        else:
            n_tiles = math.ceil(Interm / SRAM_cap)
            hbm_latency_qkt = 0
            extra_hbm_bytes = (n_tiles - 1) * (B * L * D + B * L * D)
        extra_hbm_latency = sec_to_cycles(extra_hbm_bytes / (ext_bw * 1e-9))
        total_latency_qkt = max(hbm_latency_qkt, comp_latency_qkt, sram_latency_qkt) + extra_hbm_latency

        size_qkt = B * q_heads * L * L
        GActs:float = specified_logic['config']['sfu'] * specified_logic['sfu']
        ops_av = 2.0 * B * q_heads * (L * L) * d_k
        lat_softmax = size_qkt / (GActs*1e9)
        comp_latency_av = ns_to_cycles(ops_av / MACs_per_ns)
        comp_latency_av = max(comp_latency_av, sec_to_cycles(lat_softmax))

        bytes_sram_av = (B * q_heads * L * L) + (B * L * D) + (B * L * D)
        sram_time_av = bytes_sram_av / (SRAM_bw * 1e9)
        sram_latency_av = sec_to_cycles(sram_time_av)

        Interm = (B * q_heads * L * L) + (B * L * D) + (B * L * D)
        if (B * L * D) + (B * L * D) <= SRAM_cap:
            hbm_latency_av = (B * L * D)
            extra_hbm_bytes = 0
        else:
            n_tiles = math.ceil(Interm / SRAM_cap)
            hbm_latency_av = 0
            extra_hbm_bytes = (n_tiles - 1) * (B * q_heads * L * L + B * L * D + B * L * D)

        extra_hbm_latency = sec_to_cycles(extra_hbm_bytes / (ext_bw * 1e-9))
        total_latency_av = max(hbm_latency_av, comp_latency_av, sram_latency_av) + extra_hbm_latency

        total_latency = total_latency_qkv + total_latency_qkt + total_latency_av
        return analytics(total_latency, -1, -1, [], 90, 250)

    @classmethod
    def get_dc_attention_latency(cls, batch_size: int, L_seq: int, embedding_dim: int, q_heads: int, kv_heads: int,
                                 logic_name="marca_d", ext_bw=128, acc_freq: float | None = None) -> Tuple[int, int, int, List]:
        acc_freq = acc_freq or cls.acc_freq
        specified_logic = cls._logic(logic_name)
        MACs_per_ns = specified_logic['config']['marca'] * specified_logic['core']
        SRAM_bw = specified_logic['config']['sram_tp'] * specified_logic['sram']
        SRAM_cap = 10e6 * specified_logic['sram']

        B = batch_size
        L = L_seq
        D = embedding_dim
        d_k = D / kv_heads

        def sec_to_cycles(t_s: float) -> int: return int(t_s * acc_freq)
        def ns_to_cycles(t_ns: float) -> int: return int(t_ns * (acc_freq * 1e-9))

        ops_q = B * D * D
        comp_latency_q = ns_to_cycles(ops_q / MACs_per_ns)
        bytes_sram_q = (B*D + D*D + B*D)
        sram_latency_q = sec_to_cycles(bytes_sram_q / (SRAM_bw * 1e9))

        Interm = B*D + D*D + B*D
        if Interm <= SRAM_cap:
            hbm_latency_q = (B*D + D*D)/(ext_bw * 1e-9)
            extra_hbm_bytes = 0
        else:
            n_tiles = math.ceil(Interm / SRAM_cap)
            hbm_latency_q = (B*D + D*D)/(ext_bw * 1e-9)
            extra_hbm_bytes = (n_tiles - 1) * (D*D)
        hbm_latency_q = sec_to_cycles(hbm_latency_q)
        extra_hbm_latency = sec_to_cycles(extra_hbm_bytes / (ext_bw * 1e-9))
        total_q = max(hbm_latency_q, comp_latency_q, sram_latency_q) + extra_hbm_latency

        ops_qkt = B * q_heads * L * d_k
        comp_latency_qkt = ns_to_cycles(ops_qkt / MACs_per_ns)
        bytes_sram_qkt = (B*D + B*L*D)
        sram_latency_qkt = sec_to_cycles(bytes_sram_qkt / (SRAM_bw*1e9))

        Interm = B*D + D
        if Interm <= SRAM_cap:
            hbm_latency_qkt = (L*D)/(ext_bw * 1e-9)
            extra_hbm_bytes = 0
        else:
            n_tiles = math.ceil(Interm / SRAM_cap)
            hbm_latency_qkt = (L*D)/(ext_bw * 1e-9)
            extra_hbm_bytes = (n_tiles - 1) * (L*D)
        hbm_latency_qkt = sec_to_cycles(hbm_latency_qkt)
        extra_hbm_latency = sec_to_cycles(extra_hbm_bytes / (ext_bw * 1e-9))
        total_qkt = max(comp_latency_qkt, sram_latency_qkt, hbm_latency_qkt) + extra_hbm_latency

        ops_av = B * q_heads * L * d_k
        comp_latency_av = ns_to_cycles(ops_av / MACs_per_ns)
        size_qkt = B * q_heads * L
        GActs:float = specified_logic['config']['sfu'] * specified_logic['sfu']
        lat_softmax = size_qkt / (GActs*1e-9)
        comp_latency_av = max(comp_latency_av, sec_to_cycles(lat_softmax))

        bytes_sram_av = (B*L*D + B*D)
        sram_latency_av = sec_to_cycles(bytes_sram_av / (SRAM_bw*1e9))

        Interm = B*D + D
        if Interm <= SRAM_cap:
            hbm_latency_av = (L*D)/(ext_bw * 1e-9)
            extra_hbm_bytes = 0
        else:
            n_tiles = math.ceil(Interm / SRAM_cap)
            hbm_latency_av = (L*D)/(ext_bw * 1e-9)
            extra_hbm_bytes = (n_tiles - 1) * (L*D)
        hbm_latency_av = sec_to_cycles(hbm_latency_av)
        extra_hbm_latency = sec_to_cycles(extra_hbm_bytes / (ext_bw * 1e-9))
        total_av = max(comp_latency_av, sram_latency_av, hbm_latency_av) + extra_hbm_latency

        total_latency = total_q + total_qkt + total_av
        return analytics(total_latency, -1, -1, [], 90, 250)

    @classmethod
    def rms_norm_est(cls, in_size, logic_name="marca_d", ext_bw=128, batch_size=1) -> int:
        specified_logic = cls._logic(logic_name)
        GActs:float = specified_logic['config']['sfu'] * specified_logic['sfu']

        in_size = in_size * batch_size
        Comp_latency = int(in_size / GActs)

        SRAM_bw = specified_logic['config']['sram_tp'] * specified_logic['sram']
        SRAM_latency = int(in_size / SRAM_bw)

        HBM_latency = int(in_size / ext_bw)

        Total_latency = max(Comp_latency, SRAM_latency, HBM_latency)

        return analytics(Total_latency, -1, -1, [], 30, 100,
                         [ExecutionPhase(max(Comp_latency, SRAM_latency), in_size)])

    @classmethod
    def silu_est(cls, in_size, logic_name="marca_d", ext_bw=128, batch_size=1) -> int:
        specified_logic = cls._logic(logic_name)
        GActs = specified_logic['config']['out_ports'] * specified_logic['core']     # GActs / s (elem-wise)
        in_size = in_size * batch_size
        Comp_latency = int(in_size / GActs)
        
        SRAM_bw = specified_logic['config']['sram_tp'] * specified_logic['sram']
        SRAM_latency = int(in_size / SRAM_bw)
        
        HBM_latency = int(in_size / (ext_bw))
        Total_latency = max(Comp_latency, SRAM_latency, HBM_latency)
        
        return analytics(Total_latency, -1, -1, [], 30, 100,
                         [ExecutionPhase(max(Comp_latency, SRAM_latency), in_size)])

    @classmethod
    def softplus_est(cls, in_size, logic_name="marca_d", ext_bw=128, batch_size=1) -> int:
        return cls.rms_norm_est(in_size, logic_name, ext_bw, batch_size)
    
    @classmethod
    def square_relu_est(cls, in_size, logic_name="marca_d", ext_bw=128, batch_size=1) -> int:
        return cls.rms_norm_est(in_size, logic_name, ext_bw, batch_size)