# ...existing code...
import numpy as np
from typing import Tuple, List, Dict, Any
import math
from Sim.metrics.monitor import analytics
import Sim.common as common

from .Base_Accmodel import BaseAccModel

import inspect


class B200AccModel(BaseAccModel):
    """
    Tensor-core-like accelerator performance model (converted from top-level functions).
    Implements BaseAccModel abstract APIs.
    """
    acc_freq: float = 1.72e9  # default 1.72 GHz

    def __init__(self, acc_freq: float | None = None):
        if acc_freq is not None:
            self.acc_freq = acc_freq

    @staticmethod
    def get_name() -> str:
        return "B200"

    # -- Helper accessors (use BaseAccModel helpers where appropriate) --
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

        # if function accepts *args, pass positional args through
        # generally it does not accept *args, so we map positional args to parameter names
        has_var_pos = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params)
        has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params)

        # build final kwargs to pass
        if has_var_pos:
            # allow all positional args, filter kwargs only if function doesn't accept **kwargs
            if has_var_kw:
                return fn(*args, **kwargs)
            else:
                allowed_kw = {p.name for p in params if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)}
                filtered_kwargs = {k: v for k, v in kwargs.items() if k in allowed_kw}
                return fn(*args, **filtered_kwargs)
        else:
            # map positional args to parameter names (skip 'cls' / 'self' if present)
            param_names = [p.name for p in params if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)]
            if param_names and param_names[0] in ("cls", "self"):
                param_names = param_names[1:]

            call_kwargs = {}
            for i, a in enumerate(args):
                if i < len(param_names):
                    call_kwargs[param_names[i]] = a

            if has_var_kw:
                # function accepts **kwargs, pass all kwargs through (overrides positional mapping)
                call_kwargs.update(kwargs)
            else:
                # filter kwargs to only allowed names
                allowed = set(param_names)
                for k, v in kwargs.items():
                    if k in allowed:
                        call_kwargs[k] = v

            return fn(**call_kwargs)
    
    @classmethod
    def get_pf_ssm_latency(cls, batch_size: int, L_seq: int, ED: int, state_size: int,
                           logic_name="B200", ext_bw=128, acc_freq: float | None = None,
                           concurrent_chiplets: int = 1, first_layer: bool = False, last_layer: bool = False) -> analytics:
        acc_freq = acc_freq or cls.acc_freq
        specified_logic = cls._logic(logic_name)

        # ---- Hardware capabilities ----
        MACs = specified_logic['config']['MACs'] * specified_logic['core']    # MACs / ns (tensor-core-like)
        GActs = specified_logic['config']['sfu'] * specified_logic['sfu']     # GActs / s  (elem-wise, incl. non-linear)
        SRAM_bw = specified_logic['config']['sram_tp'] * specified_logic['sram'] # GB/s
        SRAM_cap = (1024**2) * specified_logic['sram']                                 # bytes
        elem_ops = specified_logic['config']['out_ports'] * specified_logic['core']  # number of elem-wise ops that can be done in parallel

        B = batch_size
        L = L_seq
        D = ED
        n = state_size


        # first divide ED to concurrent chiplets
        D = math.ceil(D / concurrent_chiplets)

        # ---------- 1. Compute latency ----------
        non_linear_ops = B * L * D # for A delta
        elem_ops_0 = B * L * D  # for A delta
        elem_ops_1 = B * L * D * n # for Bu
        elem_ops_2 = B * L * D * n # for AX
        non_elem_ops_1 = B * L * D * n # for CX


        comp_non_linear_ns = non_linear_ops / GActs # in ns
        comp_non_linear_cycles = cls.ns_to_cycles(comp_non_linear_ns, acc_freq)
        Compute_latency1 = elem_ops_0/elem_ops + max(elem_ops_1/elem_ops, comp_non_linear_cycles)
        Compute_latency1 = cls.ns_to_cycles(Compute_latency1, acc_freq)

        Compute_latency2 = cls.ns_to_cycles(elem_ops_2/elem_ops, acc_freq)

        Compute_latency3 = cls.ns_to_cycles(non_elem_ops_1/MACs, acc_freq)
        compute_latency = Compute_latency1 + Compute_latency2 + Compute_latency3

        # ---------- 2. SRAM latency ----------
        u_in = B * L * D * common.ByteperParam
        a_0 = D * common.ByteperParam
        dt = B * L * D * common.ByteperParam
        a_1 = B * L * D * common.ByteperParam
        a_2 = B * L * D * common.ByteperParam
        b_0 = B * L * n * common.ByteperParam
        b_1 = B * L * D * n * common.ByteperParam
        c_0 = B * L * n * common.ByteperParam
        x_01 = B * L * D * n * common.ByteperParam
        y_t = B * L * D * common.ByteperParam + u_in * common.ByteperParam
        total_sram_bytes = u_in + a_0 + dt + a_1 + a_2 + b_0 + b_1 + c_0 + x_01 + y_t
        sram_latency = total_sram_bytes / (SRAM_bw)  # in ns

        # ------------------------------------------------------------
        # 3. HBM traffic (capacity-aware)
        # ------------------------------------------------------------

        # c0 and b0 are not needed at the same time
        Intrem = max(u_in + a_0 + dt + a_1 + b_1 + x_01, a_2 + x_01 + y_t)

        if Intrem <= SRAM_cap:
            bytes_hbm = 2*a_0 # a and D
        else:
            bytes_hbm = 2*a_0 + Intrem - SRAM_cap
        
        hbm_latency = bytes_hbm / (ext_bw)  # ns
        total_latency = max(hbm_latency, compute_latency, sram_latency)

        HBM_utilization = hbm_latency / total_latency if total_latency > 0 else 0
        compute_utilization = compute_latency / total_latency if total_latency > 0 else 0

        return analytics(total_latency, -1, HBM_utilization, [], compute_utilization, 200)

    @classmethod
    def get_dc_ssm_latency(cls, batch_size: int, ED: int, state_size: int,
                           logic_name="B200", ext_bw=128, acc_freq: float | None = None,
                           concurrent_chiplets: int = 1, first_layer: bool = False, last_layer: bool = False) -> analytics:
        acc_freq = acc_freq or cls.acc_freq
        specified_logic = cls._logic(logic_name)

        MACs = specified_logic['config']['MACs'] * specified_logic['core']    # MACs / ns (tensor-core-like)
        GActs = specified_logic['config']['sfu'] * specified_logic['sfu']       # GActs/s
        SRAM_bw = specified_logic['config']['sram_tp'] * specified_logic['sram']   # GB/s
        SRAM_cap = 10e6 * specified_logic['sram']                                   # bytes
        elem_ops = specified_logic['config']['out_ports'] * specified_logic['core']                           # elem-wise ALU width

        B, D, N = batch_size, ED, state_size

        # first divide ED to concurrent chiplets
        D = math.ceil(D / concurrent_chiplets)

        # ---------- 1. Compute latency ----------
        non_linear_ops = B * 1 * D # for A delta
        elem_ops_0 = B * 1 * D  # for A delta
        elem_ops_1 = B * 1 * D * N # for Bu
        elem_ops_2 = B * 1 * D * N # for AX
        non_elem_ops_1 = B * 1 * D * N # for CX

        comp_non_linear_ns = non_linear_ops / GActs # in ns
        comp_non_linear_cycles = cls.ns_to_cycles(comp_non_linear_ns, acc_freq)
        Compute_latency1 = elem_ops_0/elem_ops + max(elem_ops_1/elem_ops, comp_non_linear_cycles)
        Compute_latency1 = cls.ns_to_cycles(Compute_latency1, acc_freq)

        Compute_latency2 = cls.ns_to_cycles(elem_ops_2/elem_ops, acc_freq)

        Compute_latency3 = cls.ns_to_cycles(non_elem_ops_1/MACs, acc_freq)
        compute_latency = Compute_latency1 + Compute_latency2 + Compute_latency3

        # ---------- 2. SRAM latency ----------
        u_in = B * 1 * D * common.ByteperParam
        a_0 = D * common.ByteperParam
        dt = B * 1 * D * common.ByteperParam
        a_1 = B * 1 * D * common.ByteperParam
        a_2 = B * 1 * D * common.ByteperParam
        b_0 = B * 1 * N * common.ByteperParam
        b_1 = B * 1 * D * N * common.ByteperParam
        c_0 = B * 1 * N * common.ByteperParam
        x_01 = B * 1 * D * N * common.ByteperParam
        y_t = B * 1 * D * common.ByteperParam + u_in * common.ByteperParam
        total_sram_bytes = u_in + a_0 + dt + a_1 + a_2 + b_0 + b_1 + c_0 + x_01 + y_t
        sram_latency = total_sram_bytes / (SRAM_bw)  # in ns

        # ------------------------------------------------------------
        # 3. HBM traffic (capacity-aware)
        # ------------------------------------------------------------

        # c0 and b0 are not needed at the same time
        Intrem = max(u_in + a_0 + dt + a_1 + b_1 + x_01, a_2 + x_01 + y_t)

        if Intrem <= SRAM_cap:
            bytes_hbm = 2*a_0 + x_01 # a and D
        else:
            bytes_hbm = 2*a_0 + x_01 + Intrem - SRAM_cap
        
        hbm_latency = bytes_hbm / (ext_bw)  # ns
        total_latency = max(hbm_latency, compute_latency, sram_latency)

        HBM_utilization = hbm_latency / total_latency if total_latency > 0 else 0
        compute_utilization = compute_latency / total_latency if total_latency > 0 else 0

        return analytics(total_latency, -1, HBM_utilization, [], compute_utilization, 200)

    @classmethod
    def get_pf_mlp_latency(cls, bs: int, f_in: int, f_out: int, logic_name="B200", ext_bw=128,
                           acc_freq: float | None = None, batch_size: int = 1,
                           concurrent_chiplets: int = 1, first_layer: bool = False, last_layer: bool = False) -> analytics:
        acc_freq = acc_freq or cls.acc_freq
        specified_logic = cls._logic(logic_name)
        req_bs = batch_size

        # divide f_out by number of concurrent chiplets
        f_out = math.ceil(f_out / concurrent_chiplets)
        # ----------------------------
        # 1. Compute Latency
        # ----------------------------
        MACs_per_cycle = specified_logic['config']['MACs'] * specified_logic['core']  # MACs / ns
        total_ops = bs * f_in * f_out * req_bs  # MACs
        comp_time_cycles = total_ops / MACs_per_cycle
        comp_latency = cls.cycles_to_ns(comp_time_cycles, acc_freq)

        # ----------------------------
        # 2. SRAM Latency
        # ----------------------------
        weight_bytes = f_in * f_out * common.ByteperParam
        input_bytes = bs * f_in * req_bs * common.ByteperParam
        output_bytes = bs * f_out * req_bs * common.ByteperParam

        # SRAM bandwidth is in GB/s
        SRAM_bw = specified_logic['config']['sram_tp'] * specified_logic['sram']  # GB/s
        bytes_sram = weight_bytes + input_bytes + output_bytes # bytes
        sram_latency = bytes_sram / SRAM_bw # ns

        # ----------------------------
        # HBM traffic (capacity-aware)
        # ----------------------------
        SRAM_cap = (1024**2) * specified_logic['sram']  # bytes

        # need to load input only for first layer and output only for last layer
        if not first_layer:
            input_bytes = 0
        if not last_layer:
            output_bytes = 0

        if weight_bytes <= SRAM_cap:
            # All fits: load once
            bytes_hbm = weight_bytes + input_bytes + output_bytes
        else:
            # Tiled case: reload input for each tile
            n_tiles = math.ceil(weight_bytes / SRAM_cap)
            bytes_hbm = n_tiles * (input_bytes + weight_bytes / n_tiles) + output_bytes

        # Bw is in GB/s
        hbm_latency = bytes_hbm / (ext_bw) # ns

        # ----------------------------
        # 4. Total (overlap model)
        # ----------------------------
        total_latency = max(hbm_latency, comp_latency, sram_latency)

        HBM_BW_utilization = hbm_latency / total_latency if total_latency > 0 else 0
        compute_utilization = comp_latency / total_latency if total_latency > 0 else 0

        return analytics(total_latency, -1, HBM_BW_utilization, [], compute_utilization, 200)
    
    @classmethod
    def get_pf_conv1d_latency(cls, bs: int, c_in: int, c_out: int, f_in: int, f_out: int, kernel_size: int,
                              logic_name="tscs_p", ext_bw=128, acc_freq: float | None = None, batch_size: int = 1,
                              concurrent_chiplets: int = 1, first_layer: bool = False, last_layer: bool = False) -> analytics:
        acc_freq = acc_freq or cls.acc_freq
        specified_logic = cls._logic(logic_name)
        req_bs = batch_size

        # we can divide c_in by number of concurrent chiplets
        c_in = math.ceil(c_in / concurrent_chiplets)
        c_out = math.ceil(c_out / concurrent_chiplets)

        MACs_per_cycle = specified_logic['config']['MACs'] * specified_logic['core']
        total_ops = bs * c_in * kernel_size * req_bs
        comp_time_cycles = total_ops / MACs_per_cycle
        comp_latency = cls.cycles_to_ns(comp_time_cycles, acc_freq)

        # ----------------------------
        # 2. SRAM Latency
        # ----------------------------
        weight_bytes = c_in * kernel_size * common.ByteperParam
        input_bytes = bs* c_in * req_bs * common.ByteperParam
        output_bytes = bs * c_out * req_bs * common.ByteperParam

        SRAM_bw = specified_logic['config']['sram_tp'] * specified_logic['sram']  # GB/s
        bytes_sram = (weight_bytes + input_bytes + output_bytes) # bytes
        sram_latency = bytes_sram / SRAM_bw # ns
        
        # ----------------------------
        # HBM traffic (capacity-aware)
        # ----------------------------
        SRAM_cap = (1024**2) * specified_logic['sram']  # bytes
        
        # need to load input only for first layer and output only for last layer
        if not first_layer:
            input_bytes = 0
        if not last_layer:
            output_bytes = 0

        if weight_bytes <= SRAM_cap:
            # All fits: load once
            bytes_hbm = weight_bytes + input_bytes + output_bytes
        else:
            # Tiled case: reload input for each tile
            n_tiles = math.ceil(weight_bytes / SRAM_cap)
            bytes_hbm = n_tiles * (input_bytes + weight_bytes / n_tiles) + output_bytes

        hbm_latency = bytes_hbm / (ext_bw) # ns

        # ----------------------------
        # 4. Total (overlap model)
        # ----------------------------
        total_latency = max(hbm_latency, comp_latency, sram_latency)

        HBM_BW_utilization = hbm_latency / total_latency if total_latency > 0 else 0
        compute_utilization = comp_latency / total_latency if total_latency > 0 else 0

        return analytics(total_latency, -1, HBM_BW_utilization, [], compute_utilization, 200)

    @classmethod
    def get_pf_attention_latency(cls, batch_size: int, L_seq: int, embedding_dim: int, q_heads: int, kv_heads: int,
                                 logic_name="B200", ext_bw=128, acc_freq: float | None = None,
                                 concurrent_chiplets: int = 1, first_layer: bool = False, last_layer: bool = False) -> analytics:
        acc_freq = acc_freq or cls.acc_freq
        specified_logic = cls._logic(logic_name)

        # Common resources
        MACs_per_cycle = specified_logic['config']['MACs'] * specified_logic['core']   # MACs / ns
        SRAM_bw     = specified_logic['config']['sram_tp'] * specified_logic['sram']  # GB/s
        SRAM_cap    = (1024**2) * specified_logic['sram']  # bytes (effective on-chip capacity for modeling)

        B = batch_size # request batch size
        L = L_seq
        D = embedding_dim
        # Per-KV head inner dim; supports GQA where kv_heads may be < q_heads
        d_k = D / q_heads  # float on purpose to avoid floor issues

        # ------------------------------------------------------------
        # 1) QKV projection: [B,L,D] x [D,3D] → Q:[B,L,D], K:[B,L,D], V:[B,L,D]
        #    Ops = 3 * 2 * B * L * D * D
        #    HBM includes: load X, load W_qkv, write K,V cache to HBM
        # ------------------------------------------------------------
        # we can first compute total output dim and divide by number of concurrent chiplets

        # because of GQA, K and V may be smaller than Q
        qkv_out_dim = D + d_k * kv_heads * 2
        qkv_out_dim_chiplet = math.ceil(qkv_out_dim / concurrent_chiplets)

        ops_qkv =  B * L * D * qkv_out_dim_chiplet
        comp_latency_qkv = ops_qkv / MACs_per_cycle
        comp_latency_qkv = cls.cycles_to_ns(comp_latency_qkv, acc_freq)

        weight_bytes_qkv = D * qkv_out_dim_chiplet * common.ByteperParam
        input_bytes_qkv = B * L * D * common.ByteperParam
        output_bytes_qkv = B * L * qkv_out_dim_chiplet * common.ByteperParam
        # SRAM traffic (local movement) ~ X + W_qkv + (Q+K+V)
        bytes_sram_qkv = input_bytes_qkv + weight_bytes_qkv + output_bytes_qkv
        sram_latency_qkv = bytes_sram_qkv / SRAM_bw  # ns

        # HBM traffic: capacity-aware tiling over weights; plus KV cache writeback
        Interm = max(input_bytes_qkv + weight_bytes_qkv, output_bytes_qkv)
        input_bytes_hbm   = (input_bytes_qkv) if first_layer else 0  # X
        weight_bytes_hbm  = weight_bytes_qkv
        output_bytes_hbm = 0            # no need to write intermediates back to HBM
        if Interm <= SRAM_cap:
            bytes_hbm_qkv = input_bytes_hbm + weight_bytes_hbm 
        else:
            bytes_hbm_qkv = (input_bytes_hbm + weight_bytes_hbm) + Interm-SRAM_cap

        hbm_latency_qkv = bytes_hbm_qkv / (ext_bw)  # ns

        total_latency_qkv = max(hbm_latency_qkv, comp_latency_qkv, sram_latency_qkv)


        # ------------------------------------------------------------
        # 2) QK^T: per (query) head GEMM; with GQA this scales ~ q_heads * L^2 * d_k
        #    Ops ≈ 2 * B * q_heads * L * L * d_k
        #    HBM assumed 0 (Q,K produced on-chip and streamed)
        # ------------------------------------------------------------
        # This stage involves Bx(seq_lenxd_model * d_modelxseq_len = seq_lenxseq_len) computation
        # This stage involves Bx(seq_lenxd_model * d_modelxseq_len/chiplets = seq_lenxseq_len/chiplets) computation
        L_seq_chiplet = math.ceil(L / concurrent_chiplets)

        ops_qkt = B * D * (L * L_seq_chiplet)
        comp_latency_qkt = ops_qkt / MACs_per_cycle
        comp_latency_qkt = cls.cycles_to_ns(comp_latency_qkt, acc_freq)

        # SRAM traffic (local): Q + K + attention scores
        # Use Q,K as B*L*D each; scores as B*q_heads*L*L
        q_bytes = B * L * D * common.ByteperParam
        k_bytes = B * L_seq_chiplet * d_k * kv_heads * common.ByteperParam
        scores_bytes = B * L * L_seq_chiplet * common.ByteperParam
        bytes_sram_qkt = q_bytes + k_bytes + scores_bytes
        sram_latency_qkt = bytes_sram_qkt / (SRAM_bw)  # ns

        Interm = max(q_bytes + k_bytes, scores_bytes)
        if Interm <= SRAM_cap:
            hbm_bytes  = 0
        else:
            hbm_bytes = (Interm - SRAM_cap)

        qkt_hbm_latency = hbm_bytes / (ext_bw)  # ns
        total_latency_qkt = max(qkt_hbm_latency, comp_latency_qkt, sram_latency_qkt)

        # ------------------------------------------------------------
        # 4) softmax weights
        #    Ops ≈ exponential of all scores + division normalization
        #   2 * non linear ops per score
        # ------------------------------------------------------------
        GActs:float = specified_logic['config']['sfu'] * specified_logic['sfu']  # GActs / s

        in_size = B * L * L/concurrent_chiplets # number of scores

        comp_latency_exp = int(in_size / GActs) #ns
        
        sram_bytes_exp = in_size * common.ByteperParam
        sram_latency_exp = sram_bytes_exp / SRAM_bw  # ns

        hbm_bytes_exp = max(0, sram_bytes_exp - SRAM_cap)
        hbm_latency_exp = hbm_bytes_exp / ext_bw # ns 

        total_latency_exp = max(hbm_latency_exp, comp_latency_exp, sram_latency_exp)
        total_latency_softmax = 2*total_latency_exp # for exp + division

        # ------------------------------------------------------------
        # 4) AttnV: (softmax weights) x V; scales ~ q_heads * L^2 * d_k
        #    Ops ≈ 2 * B * q_heads * L * L * d_k
        #    HBM assumed 0 (V resident from stage 1; output stays on-chip for out-proj)
        # ------------------------------------------------------------
        # This stage does B(d_modelxseq_len * seq_lenxseq_len = d_modelxseq_len) computation
        L_seq_chiplet = math.ceil(L / concurrent_chiplets)

        ops_av = B * D * (L * L_seq_chiplet)
        comp_latency_av = ops_av / MACs_per_cycle
        comp_latency_av = cls.cycles_to_ns(comp_latency_av, acc_freq)

        v_bytes = B * L * d_k * kv_heads * common.ByteperParam
        scores_bytes = B * L * L_seq_chiplet * common.ByteperParam
        output_bytes = B * D * L_seq_chiplet * common.ByteperParam

        bytes_sram_av = v_bytes + scores_bytes + output_bytes
        sram_latency_av = bytes_sram_av / (SRAM_bw) # ns

        # HBM traffic (off-chip)
        Interm = max(v_bytes + scores_bytes, output_bytes)
        if Interm <= SRAM_cap:
            hbm_bytes = 0
        else:
            hbm_bytes = (Interm - SRAM_cap)
        av_hbm_latency = hbm_bytes / (ext_bw)  # ns

        total_latency_av = max(av_hbm_latency, comp_latency_av, sram_latency_av)

        # ------------------------------------------------------------
        # 5) output projection
        # ------------------------------------------------------------
        # This stage does (seq_lenxd_model * d_modelxd_model = seq_lenxd_model) computation
        d_model_chiplet = math.ceil(D / concurrent_chiplets)
        ops_out = B * L * D * d_model_chiplet
        comp_latency_out = ops_out / MACs_per_cycle
        comp_latency_outproj = cls.cycles_to_ns(comp_latency_out, acc_freq)

        # SRAM latency
        weight_bytes_outproj = D * d_model_chiplet * common.ByteperParam
        input_bytes_outproj = B * D * L * common.ByteperParam
        output_bytes_outproj = B * d_model_chiplet * L * common.ByteperParam

        bytes_sram_outproj = weight_bytes_outproj + input_bytes_outproj + output_bytes_outproj

        sram_latency_outproj = bytes_sram_outproj / (SRAM_bw)  # ns

        # HBM traffic (capacity-aware)
        Interm = max(input_bytes_outproj + weight_bytes_outproj, output_bytes_outproj)
        weight_bytes_hbm_outproj  = weight_bytes_outproj
        output_bytes_hbm_outproj = output_bytes_outproj if last_layer else 0  # final output assumed to stay on-chip
        if Interm <= SRAM_cap:
            bytes_hbm_outproj = weight_bytes_hbm_outproj + output_bytes_hbm_outproj
        else:
            bytes_hbm_outproj = (weight_bytes_hbm_outproj + output_bytes_hbm_outproj) + Interm-SRAM_cap

        hbm_latency_outproj = bytes_hbm_outproj / (ext_bw)  # ns
        total_latency_outproj = max(hbm_latency_outproj, comp_latency_outproj, sram_latency_outproj)
        # ------------------------------------------------------------

        total_latency = total_latency_qkv + total_latency_qkt + total_latency_softmax + total_latency_av + total_latency_outproj

        HBM_BW_utilization = ((hbm_latency_qkv + qkt_hbm_latency + 2*hbm_latency_exp +
                               av_hbm_latency + hbm_latency_outproj ) / total_latency if total_latency > 0 else 0)
        compute_utilization = (comp_latency_qkv + comp_latency_qkt + 2*comp_latency_exp +
                               comp_latency_av + comp_latency_outproj) / total_latency if total_latency > 0 else 0

        return analytics(total_latency, -1, HBM_BW_utilization, [], compute_utilization, 250)

    @classmethod
    def get_dc_attention_latency(cls, batch_size: int, L_seq: int, embedding_dim: int, q_heads: int, kv_heads: int,
                                 logic_name="B200", ext_bw=128, acc_freq: float | None = None,
                                 concurrent_chiplets: int = 1, first_layer: bool = False, last_layer: bool = False) -> analytics:

        acc_freq = acc_freq or cls.acc_freq
        specified_logic = cls._logic(logic_name)

        SRAM_bw     = specified_logic['config']['sram_tp'] * specified_logic['sram']  # GB/s
        MACS_per_cycle = specified_logic['config']['MACs'] * specified_logic['core']   # MACs / ns
        SRAM_cap    = (1024**2) * specified_logic['sram']  # bytes (effective on-chip capacity for modeling)
        
        B = batch_size
        L = L_seq
        D = embedding_dim
        d_k = D / q_heads

        # ------------------------------------------------------------
        # 1) QKV projection: [B,1,D] x [D,3D] → Q:[B,1,D], K:[B,1,D], V:[B,1,D]
        #    HBM includes: load X, load W_qkv
        # ------------------------------------------------------------
        # we can first compute total output dim and divide by number of concurrent chiplets
        qkv_out_dim = D + d_k * kv_heads * 2
        qkv_out_dim_chiplet = math.ceil(qkv_out_dim / concurrent_chiplets)

        ops_q = B * D * qkv_out_dim_chiplet
        comp_latency_qkv = ops_q / MACS_per_cycle
        comp_latency_qkv = cls.cycles_to_ns(comp_latency_qkv, acc_freq)

        weight_bytes_qkv = D * qkv_out_dim_chiplet * common.ByteperParam
        input_bytes_qkv = B * 1 * D * common.ByteperParam
        output_bytes_qkv = B * 1 * qkv_out_dim_chiplet * common.ByteperParam
        # SRAM traffic (local movement) ~ X + W_qkv + (Q+K+V)
        bytes_sram_qkv = input_bytes_qkv + weight_bytes_qkv + output_bytes_qkv
        sram_latency_qkv = bytes_sram_qkv / (SRAM_bw)  # ns

        # HBM traffic: capacity-aware tiling over weights
        Interm = max(input_bytes_qkv + weight_bytes_qkv, output_bytes_qkv)

        input_bytes_hbm   = (input_bytes_qkv) if first_layer else 0  # X
        weight_bytes_hbm  = weight_bytes_qkv
        output_bytes_hbm = 0            # no need to write intermediates back to HBM
        if Interm <= SRAM_cap:
            bytes_hbm_qkv = input_bytes_hbm + weight_bytes_hbm
        else:
            bytes_hbm_qkv = (input_bytes_hbm + weight_bytes_hbm) + Interm-SRAM_cap

        hbm_latency_qkv = bytes_hbm_qkv / (ext_bw)  # ns
        total_latency_qkv = max(hbm_latency_qkv, comp_latency_qkv, sram_latency_qkv)
        # ------------------------------------------------------------
        # 2) QK^T: per (query) head GEMM; with GQA this scales ~ q_heads * L^2 * d_k
        #    HBM assumed 0 (Q,K produced on-chip and streamed)
        # ------------------------------------------------------------
        # This stage involves Bx(1xd_model * d_modelxseq_len = 1xseq_len) computation
        # This stage involves Bx(1xd_model * d_modelxseq_len/chiplets = 1xseq_len/chiplets) computation
        L_seq_chiplet = math.ceil(L / concurrent_chiplets)

        ops_qkt = B * D * L_seq_chiplet
        comp_latency_qkt = ops_qkt / MACS_per_cycle
        comp_latency_qkt = cls.cycles_to_ns(comp_latency_qkt, acc_freq)

        # SRAM traffic (local): 
        q_bytes = B * 1 * D * common.ByteperParam
        k_bytes = B * L_seq_chiplet * d_k * kv_heads * common.ByteperParam
        scores_bytes = B * 1 * L_seq_chiplet * common.ByteperParam

        bytes_sram_qkt = q_bytes + k_bytes + scores_bytes
        sram_latency_qkt = bytes_sram_qkt / SRAM_bw

        Interm = max(q_bytes + k_bytes, scores_bytes)
        if Interm <= SRAM_cap:
            hbm_bytes = k_bytes  # only K needs to be streamed from HBM
        else:
            hbm_bytes = k_bytes + (Interm - SRAM_cap)

        qkt_hbm_latency = hbm_bytes / (ext_bw)  # ns
        total_latency_qkt = max(qkt_hbm_latency, comp_latency_qkt, sram_latency_qkt)

        # ------------------------------------------------------------
        # 3) softmax weights
        #    Ops ≈ exponential of all scores + division normalization
        #   2 * non linear ops per score
        # ------------------------------------------------------------
        GActs:float = specified_logic['config']['sfu'] * specified_logic['sfu']  # GActs / s
        in_size = B * 1 * L/concurrent_chiplets  # number of scores

        comp_latency_exp = int(in_size / GActs) #ns

        sram_bytes_exp = in_size * common.ByteperParam
        sram_latency_exp = sram_bytes_exp / (SRAM_bw) # ns

        hbm_bytes_exp = max(0, sram_bytes_exp - SRAM_cap)
        hbm_latency_exp = hbm_bytes_exp / (ext_bw) # ns

        total_latency_exp = max(hbm_latency_exp, comp_latency_exp, sram_latency_exp)
        total_latency_softmax = 2*total_latency_exp # for exp + division

        # ------------------------------------------------------------
        # 4) AttnV: (softmax weights) x V; scales ~ q_heads * L^2 * d_k
        #    Ops ≈ 2 * B * q_heads * L * L * d_k
        #    HBM assumed 0 (V resident from stage 1; output stays on-chip for out-proj)
        # ------------------------------------------------------------
        # This stage does B(d_modelxseq_len * seq_lenx1 = d_modelx1) computation
        # or B(1xseq_len * seq_lenx d_model/chiplets = 1xd_model/chiplets) computation
        d_model_chiplet = math.ceil(D / concurrent_chiplets)
        dK_chiplet = math.ceil(d_k / concurrent_chiplets)

        ops_av = B * 1 * L * d_model_chiplet
        comp_latency_av = ops_av / MACS_per_cycle
        comp_latency_av = cls.cycles_to_ns(comp_latency_av, acc_freq)

        v_bytes = B * L * dK_chiplet * kv_heads * common.ByteperParam
        scores_bytes = B * 1 * L * common.ByteperParam
        output_bytes = B * d_model_chiplet * 1 * common.ByteperParam

        bytes_sram_av = v_bytes + scores_bytes + output_bytes
        sram_latency_av = bytes_sram_av / (SRAM_bw)  # ns

        # HBM traffic (off-chip)
        Interm = max(v_bytes + scores_bytes, output_bytes)
        if Interm <= SRAM_cap:
            hbm_bytes = v_bytes  # only V needs to be streamed from HBM
        else:
            hbm_bytes = v_bytes + (Interm - SRAM_cap)

        av_hbm_latency = hbm_bytes / (ext_bw)  # ns
        total_latency_av = max(av_hbm_latency, comp_latency_av, sram_latency_av)


        # ------------------------------------------------------------
        # 5) output projection
        # ------------------------------------------------------------
        # This stage does (1xd_model * d_modelxd_model = 1xd_model) computation
        d_model_chiplet = math.ceil(D / concurrent_chiplets)
        ops_out = B * 1 * D * d_model_chiplet
        comp_latency_out = ops_out / MACS_per_cycle
        comp_latency_outproj = cls.cycles_to_ns(comp_latency_out, acc_freq)

        # SRAM latency
        weight_bytes_outproj = D * d_model_chiplet * common.ByteperParam
        input_bytes_outproj = B * D * 1 * common.ByteperParam
        output_bytes_outproj = B * d_model_chiplet * 1 * common.ByteperParam
        bytes_sram_outproj = weight_bytes_outproj + input_bytes_outproj + output_bytes_outproj
        sram_latency_outproj = bytes_sram_outproj / (SRAM_bw)  # ns

        # HBM traffic (capacity-aware)
        Interm = max(input_bytes_outproj + weight_bytes_outproj, output_bytes_outproj)
        weight_bytes_hbm_outproj  = weight_bytes_outproj
        output_bytes_hbm_outproj = output_bytes_outproj if last_layer else 0  # final output assumed to stay on-chip
        if Interm <= SRAM_cap:
            bytes_hbm_outproj = weight_bytes_hbm_outproj + output_bytes_hbm_outproj
        else:
            bytes_hbm_outproj = (weight_bytes_hbm_outproj + output_bytes_hbm_outproj) + Interm-SRAM_cap
        hbm_latency_outproj = bytes_hbm_outproj / (ext_bw)  # ns
        total_latency_outproj = max(hbm_latency_outproj, comp_latency_outproj, sram_latency_outproj)

        #------------------------------------------------------------
        # Total latency
        #------------------------------------------------------------
        total_latency = total_latency_qkv + total_latency_qkt + total_latency_softmax + total_latency_av + total_latency_outproj

        HBM_BW_utilization = (hbm_latency_qkv + qkt_hbm_latency + 2*hbm_latency_exp +
                               av_hbm_latency + hbm_latency_outproj) / total_latency if total_latency > 0 else 0
        compute_utilization = (comp_latency_qkv + comp_latency_qkt + 2*comp_latency_exp + 
                               comp_latency_av + comp_latency_outproj) / total_latency if total_latency > 0 else 0

        return analytics(total_latency, -1, HBM_BW_utilization, [], compute_utilization, 250)

    @classmethod
    def rms_norm_est(cls, in_size, logic_name="B200", ext_bw=128, batch_size=1,
                     concurrent_chiplets: int = 1, first_layer: bool = False, last_layer: bool = False) -> analytics:
        specified_logic = cls._logic(logic_name)
        GActs: float = specified_logic['config']['sfu'] * specified_logic['sfu']

        in_size = in_size * batch_size/concurrent_chiplets
        Comp_latency = int(in_size / GActs)

        SRAM_bw = specified_logic['config']['sram_tp'] * specified_logic['sram']
        SRAM_latency = int(in_size*common.ByteperParam / SRAM_bw) # ns

        HBM_bw = ext_bw
        # only first layer (read input) and last layer (write) need to pay HBM cost
        if not first_layer and not last_layer:
            HBM_latency = 1 # just to avoid zero
        else:
            HBM_latency = int(in_size*common.ByteperParam / HBM_bw) # ns

        Total_latency = max(Comp_latency, SRAM_latency, HBM_latency) 

        HBM_BW_utilization = HBM_latency / Total_latency if Total_latency > 0 else 0
        compute_utilization = Comp_latency / Total_latency if Total_latency > 0 else 0

        if Total_latency == 0:
            Total_latency = 1

        return analytics(Total_latency, -1, HBM_BW_utilization, [], compute_utilization, 100)

    @classmethod
    def silu_est(cls, in_size, logic_name="B200", ext_bw=128, batch_size=1,
                 concurrent_chiplets: int = 1, first_layer: bool = False, last_layer: bool = False) -> analytics:
        return cls.rms_norm_est(in_size, logic_name, ext_bw, batch_size, concurrent_chiplets, first_layer, last_layer)

    @classmethod
    def softplus_est(cls, in_size, logic_name="B200", ext_bw=128, batch_size=1,
                     concurrent_chiplets: int = 1, first_layer: bool = False, last_layer: bool = False) -> analytics:
        return cls.rms_norm_est(in_size, logic_name, ext_bw, batch_size, concurrent_chiplets, first_layer, last_layer)
    
    @classmethod
    def square_relu_est(cls, in_size, logic_name="B200", ext_bw=128, batch_size=1,
                        concurrent_chiplets: int = 1, first_layer: bool = False, last_layer: bool = False) -> analytics:
        return cls.rms_norm_est(in_size, logic_name, ext_bw, batch_size, concurrent_chiplets, first_layer, last_layer)

    # Helper: convert seconds→cycles and ns→cycles
    def sec_to_cycles(t_s: float, acc_freq: float) -> int:
        return int(t_s * acc_freq)

    def sec_to_ns(t_s: float) -> int:
        return int(t_s * 1e9)

    def ns_to_cycles(t_ns: float, acc_freq: float) -> int:
        return int(t_ns * (acc_freq * 1e-9))
    
    def cycles_to_ns(cycles: int, acc_freq: float) -> int:
        return int(cycles / (acc_freq * 1e-9))
