# AK TODO: Implement correct behavior for systolic arrays. 

# Now it is just a makeup model because not specify the arch

import numpy as np
from typing import Tuple, List, Dict, Any
import math
from Sim.metrics.monitor import analytics
import Sim.common as common

from .Base_Accmodel import BaseAccModel
import inspect

class SystolicArrayAccModel(BaseAccModel):
    """
    Systolic / memory-controller-like accelerator performance model.
    Converted from top-level functions in systlicarray.py into a BaseAccModel subclass.
    """
    acc_freq: float = 1e9  # default 1 GHz

    def __init__(self, acc_freq: float | None = None):
        if acc_freq is not None:
            self.acc_freq = acc_freq

    @staticmethod
    def get_name() -> str:
        return "systolicarray"
    
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

    @classmethod
    def get_pf_ssm_latency(cls, batch_size: int, L_seq: int, ED: int, state_size: int,
                        logic_name="systolicarray", ext_bw=256, acc_freq: float | None = None, 
                        concurrent_chiplets: int = 1, first_layer: bool = False, last_layer: bool = False) -> analytics:

        """
        Prefill latency (in cycles) for an SSM/Mamba2 layer on a our modified systolicarray accelerator.
        """
        acc_freq = acc_freq or cls.acc_freq
        specified_logic = cls._logic(logic_name)

        # ---- Hardware capabilities ----
        SRAM_bw     = specified_logic['config']['sram_tp'] * specified_logic['sram']  # GB/s
        SRAM_cap    = (1024**2) * specified_logic['sram']  # bytes (effective on-chip capacity for modeling)
        GActs       = specified_logic['config']['sfu']  * specified_logic['sfu']     # GActs / s  (elem-wise, incl. non-linear)

        B  = batch_size
        L  = L_seq
        D  = ED
        n  = state_size

        # first divide ED to concurrent chiplets
        D = math.ceil(D / concurrent_chiplets)

        # ------------------------------------------------------------
        # 1. Compute latency
        # For each chiplet, we need to first find #systolic array row groups and col groups.
        # row groups are determined by D/total_SA rows
        # col groups are determined by n/total_SA cols 
        # If total_SA_groups > total available SAs, we can further divide by batch size
        # ------------------------------------------------------------
        
        SA_rows = specified_logic['rows']
        SA_cols = specified_logic['cols']
        row_groups = math.ceil(D / SA_rows)
        col_groups = math.ceil(n / SA_cols)
        total_SA_groups = row_groups * col_groups

        if total_SA_groups < specified_logic['core']:
            div_factor = math.ceil(specified_logic['core'] / total_SA_groups)
            B_tile = math.ceil(B / div_factor)
        elif total_SA_groups > specified_logic['core']:
            extra_iterations = math.floor(total_SA_groups / specified_logic['core'])
            B_tile = B * extra_iterations

        # It takes total 3 cycles * L to finish the computation for one batch
        compute_cycles_SSM = B_tile * L * 3 + (3*SA_rows + SA_cols)
        compute_latency = cls.cycles_to_ns(compute_cycles_SSM, acc_freq)

        # since we already have non-linear units, equal to num_cols, we don't need to worry about 
        # ext(A dt) latency

        # ---------- 2. SRAM latency ----------
        #                  B,C    dt, u
        inputs_bytes = (B*2*L*n + 2*B*D*L)* common.ByteperParam
        total_weights = 2*D *common.ByteperParam #A,D
        total_outputs = B*D*L * common.ByteperParam
        total_sram_bytes = inputs_bytes + total_weights + total_outputs
        sram_latency = total_sram_bytes/SRAM_bw  # ns
        
        # ------------------------------------------------------------
        # 3. HBM traffic (capacity-aware)
        # ------------------------------------------------------------
        # Inputs/outputs no weights in this phase
        # SSM will never be first or last layer

        Intrem = max(inputs_bytes + total_weights, total_outputs)

        if Intrem <= SRAM_cap:
            bytes_hbm = total_weights
        else:
            bytes_hbm = total_weights + Intrem - SRAM_cap
        
        hbm_latency = bytes_hbm / (ext_bw)  # ns
        total_latency = max(hbm_latency, compute_latency, sram_latency)

        HBM_utilization = hbm_latency / total_latency if total_latency > 0 else 0
        compute_utilization = compute_latency / total_latency if total_latency > 0 else 0

        return analytics(total_latency, -1, HBM_utilization, [], compute_utilization, 200)

    @classmethod
    def get_dc_ssm_latency(cls, batch_size: int, ED: int, state_size: int,
                        logic_name="systolicarray", ext_bw=256, acc_freq: float | None = None,
                        concurrent_chiplets: int = 1, first_layer: bool = False, last_layer: bool = False) -> analytics:

        # all hardware configurations in the chiplets
        acc_freq = acc_freq or cls.acc_freq
        specified_logic = cls._logic(logic_name)
        """
        Decoding latency (in cycles) for one token of an SSM/Mamba2 layer on a tensor-core-like accelerator.
        """
        acc_freq = acc_freq or cls.acc_freq
        specified_logic = cls._logic(logic_name)

        # ---- Hardware capabilities ----
        SRAM_bw     = specified_logic['config']['sram_tp'] * specified_logic['sram']  # GB/s
        SRAM_cap    = (1024**2) * specified_logic['sram']  # bytes (effective on-chip capacity for modeling)
        GActs       = specified_logic['config']['sfu']  * specified_logic['sfu']     # GActs / s  (elem-wise, incl. non-linear)

        B  = batch_size
        D  = ED
        n  = state_size

        # first divide ED to concurrent chiplets
        D = math.ceil(D / concurrent_chiplets)

        # ------------------------------------------------------------
        # 1. Compute latency
        # ------------------------------------------------------------
        SA_rows = specified_logic['rows']
        SA_cols = specified_logic['cols']
        row_groups = math.ceil(D / SA_rows)
        col_groups = math.ceil(n / SA_cols)
        total_SA_groups = row_groups * col_groups

        if total_SA_groups < specified_logic['core']:
            div_factor = math.ceil(specified_logic['core'] / total_SA_groups)
            B_tile = math.ceil(B / div_factor)
        elif total_SA_groups > specified_logic['core']:
            extra_iterations = math.floor(total_SA_groups / specified_logic['core'])
            B_tile = B * extra_iterations

        compute_cycles_SSM = B_tile * 3*n  + SA_cols
        compute_latency = cls.cycles_to_ns(compute_cycles_SSM, acc_freq)

        # ------------------------------------------------------------
        # 2. SRAM latency (local)
        # ------------------------------------------------------------
        # Store small inputs (x_prev, X_t, params)
        inputs_bytes = (B*2*n + 2*B*D)* common.ByteperParam
        weight_bytes = 2*D *common.ByteperParam #A,D
        state_cache_bytes = B*n*D * common.ByteperParam
        output_bytes = B*D * common.ByteperParam
        total_sram_bytes = inputs_bytes + weight_bytes + output_bytes + state_cache_bytes
        sram_latency = total_sram_bytes/SRAM_bw  # ns
        
        # ------------------------------------------------------------
        # 3. HBM latency
        # ------------------------------------------------------------
        # Need to read x_prev, A,B,C (params), and write x_t back
        Intrem = max(inputs_bytes + weight_bytes, output_bytes)
        # If fits, all cached in SRAM; else fetch x_prev from HBM
        if Intrem <= SRAM_cap:
            bytes_hbm = weight_bytes + state_cache_bytes
        else:
            bytes_hbm = weight_bytes + state_cache_bytes + Intrem - SRAM_cap

        hbm_latency = bytes_hbm / (ext_bw)  # ns

        # ------------------------------------------------------------
        # 4. Total
        # ------------------------------------------------------------
        total_latency = max(hbm_latency, compute_latency, sram_latency)

        HBM_utilization = hbm_latency / total_latency if total_latency > 0 else 0
        compute_utilization = compute_latency / total_latency if total_latency > 0 else 0

        return analytics(total_latency, -1, HBM_utilization, [], compute_utilization, 200)

    @classmethod
    def get_pf_mlp_latency(cls, bs: int, f_in: int, f_out: int, logic_name="systolicarray", ext_bw=256,
                           acc_freq: float | None = None, batch_size: int = 1,
                           concurrent_chiplets: int = 1, first_layer: bool = False, last_layer: bool = False) -> analytics:
        """
        Estimate latency (in cycles) for an MLP layer on a tensor-core-like accelerator.
        """
        acc_freq = acc_freq or cls.acc_freq
        specified_logic = cls._logic(logic_name)
        req_bs = batch_size

        # divide f_out by number of concurrent chiplets
        f_out = math.ceil(f_out / concurrent_chiplets)
        # ----------------------------
        # 1. Compute Latency
        # ----------------------------
        # find which is bigger between f_in, f_out and bs*batch and divide that by number of systolic arrays
        # adding rows and cols, because that much extra we pay when we map to the systolic array
        max_dim = max(f_in+specified_logic['rows'], f_out+specified_logic['cols'], bs*req_bs)
        f_in_tile, f_out_tile, bs_tile = f_in, f_out, bs*req_bs
        if f_in+specified_logic['rows'] == max_dim:
            f_in_tile = math.ceil(f_in / specified_logic['core'])
        elif f_out+specified_logic['cols'] == max_dim:
            f_out_tile = math.ceil(f_out / specified_logic['core'])
        elif bs*batch_size == max_dim:
            bs_tile = math.ceil(bs*batch_size / specified_logic['core'])

        # now find the time for one systolic array
        row_iter = math.ceil(f_in_tile / specified_logic['rows'])
        col_iter = math.ceil(f_out_tile / specified_logic['cols'])

        comp_time_cycles = row_iter * col_iter * bs_tile + (specified_logic['rows'] + specified_logic['cols'])
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
                            logic_name="systolicarray", ext_bw=256, acc_freq: float | None = None, batch_size: int = 1, 
                            concurrent_chiplets: int = 1, first_layer: bool = False, last_layer: bool = False) -> analytics:
        """
        Estimate latency (in cycles) for Conv1D on a systolic array accelerator.
        """
        acc_freq = acc_freq or cls.acc_freq
        specified_logic = cls._logic(logic_name)
        req_bs = batch_size

        # we can divide c_in by number of concurrent chiplets
        c_in = math.ceil(c_in / concurrent_chiplets)
        c_out = math.ceil(c_out / concurrent_chiplets)
        # ----------------------------
        # 1. Compute Latency
        # ----------------------------
        # check which is bigger between c_in, bs*batch and divide that by number of systolic arrays
        max_dim = max(c_in+specified_logic['rows'], kernel_size+specified_logic['cols'], bs*req_bs)

        c_in_tile = c_in
        bs_tile = bs*req_bs
        kernel_size_tile = kernel_size

        if c_in+specified_logic['rows'] == max_dim:
            c_in_tile = math.ceil(c_in / specified_logic['core'])
        elif kernel_size+specified_logic['cols'] == max_dim:
            kernel_size_tile = math.ceil(kernel_size / specified_logic['core'])
        elif bs*req_bs == max_dim:
            bs_tile = math.ceil(bs*req_bs / specified_logic['core'])

        row_iter = math.ceil(c_in_tile / specified_logic['rows'])
        col_iter = math.ceil(kernel_size_tile / specified_logic['cols'])

        # Compute latency (in cycles)
        comp_latency = row_iter * col_iter * bs_tile + (specified_logic['rows'] + specified_logic['cols'])
        comp_latency = cls.cycles_to_ns(comp_latency, acc_freq)
        
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
                                logic_name="systolicarray", ext_bw=256, acc_freq: float | None = None,
                                concurrent_chiplets: int = 1, first_layer: bool = False, last_layer: bool = False) -> analytics:
        """
        Estimate latency (in cycles) of attention prefill phase on systolic array.
        Returns: (QKV_proj_latency, QK_T_latency, AttnV_latency, breakdown_list)
        """
        acc_freq = acc_freq or cls.acc_freq
        specified_logic = cls._logic(logic_name)

        # Common resources
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

        # check which is bigger between D, bs*batch and qkv_out_dim_chiplet and divide that by number of systolic arrays
        max_dim = max(D+specified_logic['rows'], qkv_out_dim_chiplet+specified_logic['cols'], B*L)

        D_tile = D
        bs_tile = B * L
        qkv_out_tile = qkv_out_dim_chiplet

        if D+specified_logic['rows'] == max_dim:
            D_tile = math.ceil(D / specified_logic['core'])
        elif qkv_out_dim_chiplet+specified_logic['cols'] == max_dim:
            qkv_out_tile = math.ceil(qkv_out_dim_chiplet / specified_logic['core'])
        elif B*L == max_dim:
            bs_tile = math.ceil(B * L / specified_logic['core'])

        row_iter = math.ceil(D_tile / specified_logic['rows'])
        col_iter = math.ceil(qkv_out_tile / specified_logic['cols'])
        # compute latency (in cycles)
        comp_latency_qkv = row_iter * col_iter * bs_tile + (specified_logic['rows'] + specified_logic['cols'])
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

        # Check which is bigger between D, L and L_seq_chiplet and divide that by number of systolic arrays
        max_dim = max(D+specified_logic['rows'], L_seq_chiplet+specified_logic['cols'], L)

        D_tile = D
        L_tile = L
        Lout_tile = L_seq_chiplet
        if D+specified_logic['rows'] == max_dim:
            D_tile = math.ceil(D / specified_logic['core'])
        elif L_seq_chiplet+specified_logic['cols'] == max_dim:
            Lout_tile = math.ceil(L_seq_chiplet / specified_logic['core'])
        elif L == max_dim:
            L_tile = math.ceil(L / specified_logic['core'])

        row_iter = math.ceil(D_tile / specified_logic['rows'])
        col_iter = math.ceil(Lout_tile / specified_logic['cols'])

        # compute latency (in cycles)
        comp_latency_qkt = row_iter * col_iter * L_tile + (specified_logic['rows'] + specified_logic['cols'])
        comp_latency_qkt = B * comp_latency_qkt  # scale by batch size, they don't share anything
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

        # max dim from D, L, L_seq_chiplet
        max_dim = max(L+specified_logic['rows'], L_seq_chiplet+specified_logic['cols'], D)

        D_tile = D
        L_tile = L
        Lout_tile = L_seq_chiplet

        if L+specified_logic['rows'] == max_dim:
            L_tile = math.ceil(L / specified_logic['core'])
        elif L_seq_chiplet+specified_logic['cols'] == max_dim:
            Lout_tile = math.ceil(L_seq_chiplet / specified_logic['core'])
        elif D == max_dim:
            D_tile = math.ceil(D / specified_logic['core'])
        
        row_iter = math.ceil(L_tile / specified_logic['rows'])
        col_iter = math.ceil(Lout_tile / specified_logic['cols'])

        # compute latency (in cycles)
        comp_latency_av = row_iter * col_iter * D_tile + (specified_logic['rows'] + specified_logic['cols'])
        comp_latency_av = B * comp_latency_av  # scale by batch size
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

        # Find max from D, d_model_chiplet, L
        max_dim = max(D+specified_logic['rows'], d_model_chiplet+specified_logic['cols'], L)

        D_tile = D
        d_model_tile = d_model_chiplet
        L_tile = B*L

        if D+specified_logic['rows'] == max_dim:
            D_tile = math.ceil(D / specified_logic['core'])
        elif d_model_chiplet+specified_logic['cols'] == max_dim:
            d_model_tile = math.ceil(d_model_chiplet / specified_logic['core'])
        elif B*L == max_dim:
            L_tile = math.ceil(B*L / specified_logic['core'])

        row_iter = math.ceil(D_tile / specified_logic['rows'])
        col_iter = math.ceil(d_model_tile / specified_logic['cols'])

        # compute latency (in cycles)
        comp_latency_outproj = row_iter * col_iter * L_tile + (specified_logic['rows'] + specified_logic['cols'])
        comp_latency_outproj = cls.cycles_to_ns(comp_latency_outproj, acc_freq)

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

        total_latency = total_latency_qkv + total_latency_qkt + total_latency_softmax + total_latency_av + total_latency_outproj

        HBM_BW_utilization = ((hbm_latency_qkv + qkt_hbm_latency + 2*hbm_latency_exp +
                               av_hbm_latency + hbm_latency_outproj ) / total_latency if total_latency > 0 else 0)
        compute_utilization = (comp_latency_qkv + comp_latency_qkt + 2*comp_latency_exp +
                               comp_latency_av + comp_latency_outproj) / total_latency if total_latency > 0 else 0

        return analytics(total_latency, -1, HBM_BW_utilization, [], compute_utilization, 250)

    @classmethod
    def get_dc_attention_latency(cls, batch_size: int, L_seq: int, embedding_dim: int, q_heads: int, kv_heads: int,
                                logic_name="systolicarray", ext_bw=256, acc_freq: float | None = None,
                                concurrent_chiplets: int = 1, first_layer: bool = False, last_layer: bool = False) -> analytics:
        """
        Estimate latency (in cycles) of attention decoding (one token) on a systolic array.
        Returns: (Q_proj_latency, QK_T_latency, AttnV_latency, breakdown_list)
        """
        acc_freq = acc_freq or cls.acc_freq
        specified_logic = cls._logic(logic_name)

        SRAM_bw     = specified_logic['config']['sram_tp'] * specified_logic['sram']  # GB/s
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

        # check which is bigger between D, bs*batch and qkv_out_dim_chiplet and divide that by number of systolic arrays
        max_dim = max(D+specified_logic['rows'], qkv_out_dim_chiplet+specified_logic['cols'], B) # since L=1 during decoding

        D_tile = D
        bs_tile = B
        qkv_out_tile = qkv_out_dim_chiplet

        if D+specified_logic['rows'] == max_dim:
            D_tile = math.ceil(D / specified_logic['core'])
        elif qkv_out_dim_chiplet+specified_logic['cols'] == max_dim:
            qkv_out_tile = math.ceil(qkv_out_dim_chiplet / specified_logic['core'])
        elif B == max_dim:
            bs_tile = math.ceil(B / specified_logic['core'])

        row_iter = math.ceil(D_tile / specified_logic['rows'])
        col_iter = math.ceil(qkv_out_tile / specified_logic['cols'])
        # compute latency (in cycles)
        comp_latency_qkv = row_iter * col_iter * bs_tile + (specified_logic['rows'] + specified_logic['cols'])
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

        # Check which is bigger between D, 1 and L_seq_chiplet and divide that by number of systolic arrays
        max_dim = max(D+specified_logic['rows'], L_seq_chiplet+specified_logic['cols'], 1)

        D_tile = D
        L_tile = 1
        Lout_tile = L_seq_chiplet

        if D+specified_logic['rows'] == max_dim:
            D_tile = math.ceil(D / specified_logic['core'])
        elif L_seq_chiplet+specified_logic['cols'] == max_dim:
            Lout_tile = math.ceil(L_seq_chiplet / specified_logic['core'])
        elif 1 == max_dim:
            L_tile = math.ceil(1 / specified_logic['core'])

        row_iter = math.ceil(D_tile / specified_logic['rows'])
        col_iter = math.ceil(Lout_tile / specified_logic['cols'])
        # compute latency (in cycles)
        comp_latency_qkt = row_iter * col_iter * L_tile + (specified_logic['rows'] + specified_logic['cols'])
        comp_latency_qkt = B * comp_latency_qkt  # scale by batch size
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

        # max dim from 1, L, d_model_chiplet
        max_dim = max(L+specified_logic['rows'], d_model_chiplet+specified_logic['cols'], 1)
        D_tile = d_model_chiplet
        L_tile = L
        Lout_tile = 1

        if L+specified_logic['rows'] == max_dim:
            L_tile = math.ceil(L / specified_logic['core'])
        elif d_model_chiplet+specified_logic['cols'] == max_dim:
            D_tile = math.ceil(d_model_chiplet / specified_logic['core'])
        elif 1 == max_dim:
            Lout_tile = math.ceil(1 / specified_logic['core'])
        
        row_iter = math.ceil(L_tile / specified_logic['rows'])
        col_iter = math.ceil(Lout_tile / specified_logic['cols'])

        # compute latency (in cycles)
        comp_latency_av = row_iter * col_iter * D_tile + (specified_logic['rows'] + specified_logic['cols'])
        comp_latency_av = B * comp_latency_av  # scale by batch size
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

        # Find max from D, d_model_chiplet, B
        max_dim = max(D+specified_logic['rows'], d_model_chiplet+specified_logic['cols'], B) # since L=1 during decoding
        D_tile = D
        d_model_tile = d_model_chiplet
        L_tile = B
        if D+specified_logic['rows'] == max_dim:
            D_tile = math.ceil(D / specified_logic['core'])
        elif d_model_chiplet+specified_logic['cols'] == max_dim:
            d_model_tile = math.ceil(d_model_chiplet / specified_logic['core'])
        elif B == max_dim:
            L_tile = math.ceil(B / specified_logic['core'])
        
        row_iter = math.ceil(D_tile / specified_logic['rows'])
        col_iter = math.ceil(d_model_tile / specified_logic['cols'])
        # compute latency (in cycles)
        comp_latency_outproj = row_iter * col_iter * L_tile + (specified_logic['rows'] + specified_logic['cols'])
        comp_latency_outproj = cls.cycles_to_ns(comp_latency_outproj, acc_freq)
        
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
    def rms_norm_est(cls, in_size, logic_name="systolicarray", ext_bw=256, batch_size=1, concurrent_chiplets: int = 1,
                     first_layer: bool = False, last_layer: bool = False) -> analytics:
        specified_logic = cls._logic(logic_name)

        GActs:float = specified_logic['config']['sfu'] * specified_logic['sfu'] #GActs/s

        in_size = in_size * batch_size/concurrent_chiplets # since each chiplet process part of the batch
        Comp_latency = int(in_size / GActs) #ns

        SRAM_bw = specified_logic['config']['sram_tp'] * specified_logic['sram'] # GBytes/s

        SRAM_latency = int(in_size*common.ByteperParam / SRAM_bw) # ns

        HBM_bw = ext_bw # GBytes/s

        # only first layer (read input) and last layer (write) need to pay HBM cost
        if not first_layer and not last_layer:
            HBM_latency = 1 # just to avoid zero
        else:
            HBM_latency = int(in_size*common.ByteperParam / HBM_bw) # ns

        Total_latency = max(Comp_latency, SRAM_latency, HBM_latency) 

        HBM_BW_utilization = HBM_latency / Total_latency
        compute_utilization = Comp_latency / Total_latency

        return analytics(Total_latency, -1, HBM_BW_utilization, [], compute_utilization, 100)

    @classmethod
    def silu_est(cls, in_size, logic_name="systolicarray", ext_bw=256, batch_size=1, concurrent_chiplets: int = 1,
                 first_layer: bool = False, last_layer: bool = False) -> analytics:
        return cls.rms_norm_est(in_size, logic_name, ext_bw, batch_size, concurrent_chiplets, first_layer, last_layer)

    @classmethod
    def softplus_est(cls, in_size, logic_name="systolicarray", ext_bw=128, batch_size=1, concurrent_chiplets: int = 1,
                     first_layer: bool = False, last_layer: bool = False) -> analytics:
        return cls.rms_norm_est(in_size, logic_name, ext_bw, batch_size, concurrent_chiplets, first_layer, last_layer)
    
    @classmethod
    def square_relu_est(cls, in_size, logic_name="systolicarray", ext_bw=128, batch_size=1, concurrent_chiplets: int = 1,
                        first_layer: bool = False, last_layer: bool = False) -> analytics:
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
