# ...existing code...
import numpy as np
from typing import Tuple, List, Dict, Any
import math
from Sim.metrics.monitor import analytics
import Sim.common as common

from .Base_Accmodel import BaseAccModel

import inspect


class UnifiedAccModel(BaseAccModel):
    """
    Tensor-core-like accelerator performance model (converted from top-level functions).
    Implements BaseAccModel abstract APIs.
    """
    acc_freq: float = 0.975e9  # default 1.72 GHz

    def __init__(self, acc_freq: float | None = None):
        if acc_freq is not None:
            self.acc_freq = acc_freq

    @staticmethod
    def get_name() -> str:
        return "unified"

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
                           logic_name="unified", ext_bw=128, acc_freq: float | None = None,
                           concurrent_chiplets: int = 1, first_layer: bool = False, last_layer: bool = False) -> analytics:
        acc_freq = acc_freq or cls.acc_freq
        specified_logic = cls._logic(logic_name)

        # ---- Hardware capabilities ----
        vu_rows = specified_logic['config']['VU_rows']
        vu_cols = specified_logic['config']['VU_cols']
        vu_width = specified_logic['config']['VU_width']
        vu_cores = specified_logic['vua_core']
        sa_rows = specified_logic['config']['sa_rows']
        sa_cols = specified_logic['config']['sa_cols']
        sa_cores = specified_logic['sa_core']
        SRAM_bw     = specified_logic['config']['sram_tp'] * specified_logic['sram']  # GB/s
        GActs = specified_logic['config']['sfu'] * specified_logic['sfu']     # GActs / s  (elem-wise, incl. non-linear)
        SRAM_bw = specified_logic['config']['sram_tp'] * specified_logic['sram'] # GB/s
        SRAM_cap = (1024**2) * specified_logic['sram']                                 # bytes

        B = batch_size
        L = L_seq
        D = ED
        n = state_size

        # ---------- 1. Compute latency ----------
        # first divide ED to concurrent chiplets
        D = math.ceil(D / concurrent_chiplets)
        # D_sa = math.ceil(D / 2)
        # D_vua = D_sa
        # SA
        D_sa = D
        row_groups = math.ceil(D_sa / sa_rows)
        col_groups = math.ceil(n / sa_cols)
        total_SA_groups = row_groups * col_groups

        if total_SA_groups < sa_cores:
            div_factor = math.ceil(sa_cores / total_SA_groups)
            B_tile = math.ceil(B / div_factor)
        elif total_SA_groups > sa_cores:
            extra_iterations = math.floor(total_SA_groups / sa_cores)
            B_tile = B * extra_iterations

        # It takes total 3 cycles * L to finish the computation for one batch
        compute_cycles_SSM = B_tile * L * 3 + (3*sa_rows + sa_cols)
        compute_latency_sa = cls.cycles_to_ns(compute_cycles_SSM, acc_freq)

        # # VUA
        # VU_groups = math.ceil(n/vu_width)
        # # total groups
        # total_groups = math.ceil(vu_cores*vu_rows*vu_cols/VU_groups)
        # if total_groups < B*D_vua:
        #     D_tile = math.ceil(B*D_vua / total_groups)
        # else:
        #     D_tile = 1

        # compute_cycle_ssm = L*D_tile*3 + B*(math.log2(vu_width) + VU_groups)
        # compute_latency_vua = cls.cycles_to_ns(compute_cycle_ssm, acc_freq)
        # compute_latency = max(compute_latency_sa, compute_latency_vua)
        compute_latency = compute_latency_sa

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
                           logic_name="unified", ext_bw=128, acc_freq: float | None = None,
                           concurrent_chiplets: int = 1, first_layer: bool = False, last_layer: bool = False) -> analytics:
        acc_freq = acc_freq or cls.acc_freq
        specified_logic = cls._logic(logic_name)

        # ---- Hardware capabilities ----
        vu_rows = specified_logic['config']['VU_rows']
        vu_cols = specified_logic['config']['VU_cols']
        vu_width = specified_logic['config']['VU_width']
        vu_cores = specified_logic['vua_core']
        sa_rows = specified_logic['config']['sa_rows']
        sa_cols = specified_logic['config']['sa_cols']
        sa_cores = specified_logic['sa_core']
        GActs = specified_logic['config']['sfu'] * specified_logic['sfu']     # GActs / s  (elem-wise, incl. non-linear)
        SRAM_bw = specified_logic['config']['sram_tp'] * specified_logic['sram'] # GB/s
        SRAM_cap = (1024**2) * specified_logic['sram']                                 # bytes

        B = batch_size
        D = ED
        n = state_size

        # ---------- 1. Compute latency ----------
        # first divide ED to concurrent chiplets
        D = math.ceil(D / concurrent_chiplets)
        # D_sa = math.ceil(D / 2)
        # D_vua = D_sa
        # # SA
        # row_groups = math.ceil(D_sa / sa_rows)
        # col_groups = math.ceil(n / sa_cols)
        # total_SA_groups = row_groups * col_groups

        # if total_SA_groups < sa_cores:
        #     div_factor = math.ceil(sa_cores / total_SA_groups)
        #     B_tile = math.ceil(B / div_factor)
        # elif total_SA_groups > sa_cores:
        #     extra_iterations = math.floor(total_SA_groups / sa_cores)
        #     B_tile = B * extra_iterations

        # compute_cycles_SSM = B_tile * 3*n  + sa_cols
        # compute_latency_sa = cls.cycles_to_ns(compute_cycles_SSM, acc_freq)

        # VUA
        D_vua = D
        VU_groups = math.ceil(n/vu_width)
        # total groups
        total_groups = math.ceil(vu_cores*vu_rows*vu_cols/VU_groups)
        if total_groups < B*D_vua:
            D_tile = math.ceil(B*D_vua / total_groups)
        else:
            D_tile = 1

        compute_cycle1 = D_tile*3 + B*(math.log2(vu_width) + VU_groups)  
        compute_latency_vua = cls.cycles_to_ns(compute_cycle1, acc_freq)
        # compute_latency = max(compute_latency_sa, compute_latency_vua)
        compute_latency = compute_latency_vua

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
    def get_pf_mlp_latency(cls, bs: int, f_in: int, f_out: int, logic_name="unified", ext_bw=128,
                           acc_freq: float | None = None, batch_size: int = 1,
                           concurrent_chiplets: int = 1, first_layer: bool = False, last_layer: bool = False) -> analytics:
        acc_freq = acc_freq or cls.acc_freq
        specified_logic = cls._logic(logic_name)
        req_bs = batch_size
        vu_rows = specified_logic['config']['VU_rows']
        vu_cols = specified_logic['config']['VU_cols']
        vu_width = specified_logic['config']['VU_width']
        vu_cores = specified_logic['vua_core']

        sa_rows = specified_logic['config']['sa_rows']
        sa_cols = specified_logic['config']['sa_cols']
        sa_cores = specified_logic['sa_core']

        # divide f_out by number of concurrent chiplets
        f_out = math.ceil(f_out / concurrent_chiplets)
        # divide into two equal parts of sa and vua
        f_out_sa = math.ceil(f_out / 2)
        f_out_vua = f_out_sa
        # ----------------------------
        # 1. Compute Latency
        # ----------------------------
        # SA
        # find which is bigger between f_in, f_out and bs*batch and divide that by number of systolic arrays
        # adding rows and cols, because that much extra we pay when we map to the systolic array
        max_dim = max(f_in+sa_rows, f_out_sa+sa_cols, bs*req_bs)
        f_in_tile, f_out_tile, bs_tile = f_in, f_out_sa, bs*req_bs
        if f_in+sa_rows == max_dim:
            f_in_tile = math.ceil(f_in / sa_cores)
        elif f_out_sa+sa_cols == max_dim:
            f_out_tile = math.ceil(f_out_sa / sa_cores)
        elif bs*req_bs == max_dim:
            bs_tile = math.ceil(bs*req_bs / sa_cores)

        # now find the time for one systolic array
        row_iter = math.ceil(f_in_tile / sa_rows)
        col_iter = math.ceil(f_out_tile / sa_cols)

        comp_time_cycles = row_iter * col_iter * bs_tile + (sa_rows + sa_cols)
        comp_latency_sa = cls.cycles_to_ns(comp_time_cycles, acc_freq)

        # VUA
        # find which is bigger between f_in, f_out and bs*batch and divide that by number of VU arrays
        # adding rows and cols, because that much extra we pay when we map to the VUA
        max_dim = max(f_in, f_out_vua, bs*req_bs)
        f_in_tile, f_out_tile, bs_tile = f_in, f_out_vua, bs*req_bs
        if f_in == max_dim:
            f_in_tile = math.ceil(f_in / vu_cores)
        elif f_out_vua == max_dim:
            f_out_tile = math.ceil(f_out_vua / vu_cores)
        elif bs*req_bs == max_dim:
            bs_tile = math.ceil(bs*req_bs / vu_cores)

        # map bs*batch_size to rows
        # f_in to VU_width
        # f_out to cols
        # Check if bs*batch_size or f_out is smaller then hardware dim, if so, we can relocate the vu_width
        local_bs, local_f_in, local_f_out = cls.retile_dim(vu_rows, vu_width, vu_cols, bs_tile, f_in_tile, f_out_tile)
        # local_f_in*local_f_out*local_bs should be >= f_in*f_out*bs*batch_size
        if local_f_in * local_f_out * local_bs < f_in_tile * f_out_tile * bs_tile:
            raise ValueError("in correct tiling calculation")

        # now find the time for one systolic array
        row_iter = math.ceil(local_bs / vu_rows)
        col_iter = math.ceil(local_f_out / vu_cols)
        width_iter = math.ceil(local_f_in / vu_width)

        comp_time_cycles = row_iter * col_iter * width_iter + math.log2(vu_width)
        comp_latency_vua = cls.cycles_to_ns(comp_time_cycles, acc_freq)

        comp_latency = max(comp_latency_sa, comp_latency_vua)

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
        vu_rows = specified_logic['config']['VU_rows']
        vu_cols = specified_logic['config']['VU_cols']
        vu_width = specified_logic['config']['VU_width']
        vu_cores = specified_logic['vua_core']

        sa_rows = specified_logic['config']['sa_rows']
        sa_cols = specified_logic['config']['sa_cols']
        sa_cores = specified_logic['sa_core']

        # we can divide c_in by number of concurrent chiplets
        c_in = math.ceil(c_in / concurrent_chiplets)
        c_out = math.ceil(c_out / concurrent_chiplets)
        c_in_sa = math.ceil(c_in / 2)
        c_in_vua = c_in_sa
        # ----------------------------
        # 1. Compute Latency
        # SA
        # ----------------------------
        max_dim = max(c_in_sa+sa_rows, kernel_size+sa_cols, bs*req_bs)

        c_in_tile = c_in_sa
        bs_tile = bs*req_bs
        kernel_size_tile = kernel_size

        if c_in_sa+sa_rows == max_dim:
            c_in_tile = math.ceil(c_in_sa / sa_cores)
        elif kernel_size+sa_cols == max_dim:
            kernel_size_tile = math.ceil(kernel_size / sa_cores)
        elif bs*req_bs == max_dim:
            bs_tile = math.ceil(bs*req_bs / sa_cores)

        row_iter = math.ceil(c_in_tile / sa_rows)
        col_iter = math.ceil(kernel_size_tile / sa_cols)

        # Compute latency (in cycles)
        comp_latency = row_iter * col_iter * bs_tile + (sa_rows + sa_cols)
        comp_latency_sa = cls.cycles_to_ns(comp_latency, acc_freq)
        
        # vua
        max_dim = max(c_in_vua, kernel_size, bs*req_bs)
        c_in_tile, kernel_tile, bs_tile = c_in_vua, kernel_size, bs*req_bs
        if c_in_vua == max_dim:
            c_in_tile = math.ceil(c_in_vua / vu_cores)
        elif kernel_size == max_dim:
            kernel_tile = math.ceil(kernel_size / vu_cores)
        elif bs*req_bs == max_dim:
            bs_tile = math.ceil(bs*req_bs / vu_cores)
        
        # c_in to rows
        # kernel_size to VU_width
        # bs*batch_size to cols
        row_iter = math.ceil(c_in_tile / vu_rows)
        col_iter = math.ceil(bs_tile / vu_cols)
        width_iter = math.ceil(kernel_tile / vu_width)
        comp_time_cycles = row_iter * col_iter * width_iter + math.log2(vu_width)
        comp_latency_vua = cls.cycles_to_ns(comp_time_cycles, acc_freq)

        comp_latency = max(comp_latency_sa, comp_latency_vua)

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
                                 logic_name="unified", ext_bw=128, acc_freq: float | None = None,
                                 concurrent_chiplets: int = 1, first_layer: bool = False, last_layer: bool = False) -> analytics:
        acc_freq = acc_freq or cls.acc_freq
        specified_logic = cls._logic(logic_name)

        # Common resources
        vu_rows = specified_logic['config']['VU_rows']
        vu_cols = specified_logic['config']['VU_cols']
        vu_width = specified_logic['config']['VU_width']
        vu_cores = specified_logic['vua_core']
        sa_rows = specified_logic['config']['sa_rows']
        sa_cols = specified_logic['config']['sa_cols']
        sa_cores = specified_logic['sa_core']
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
        # qkv_out_dim_chiplet_sa = math.ceil(qkv_out_dim_chiplet / 2)
        # qkv_out_dim_chiplet_vua = qkv_out_dim_chiplet_sa

        # SA
        # check which is bigger between D, bs*batch and qkv_out_dim_chiplet and divide that by number of systolic arrays
        qkv_out_dim_chiplet_sa = qkv_out_dim_chiplet
        max_dim = max(D+sa_rows, qkv_out_dim_chiplet_sa+sa_cols, B*L)

        D_tile = D
        bs_tile = B * L
        qkv_out_tile = qkv_out_dim_chiplet_sa

        if D+sa_rows == max_dim:
            D_tile = math.ceil(D / sa_cores)
        elif qkv_out_dim_chiplet_sa+sa_cols == max_dim:
            qkv_out_tile = math.ceil(qkv_out_dim_chiplet_sa / sa_cores)
        elif B*L == max_dim:
            bs_tile = math.ceil(B * L / sa_cores)

        row_iter = math.ceil(D_tile / sa_rows)
        col_iter = math.ceil(qkv_out_tile / sa_cols)
        # compute latency (in cycles)
        comp_latency_qkv_sa = row_iter * col_iter * bs_tile + (sa_rows + sa_cols)
        comp_latency_qkv_sa = cls.cycles_to_ns(comp_latency_qkv_sa, acc_freq)

        # # VUA
        # max_dim = max(D, qkv_out_dim_chiplet_vua, B * L)
        # D_tile, qkv_out_dim_tile, BL_tile = D, qkv_out_dim_chiplet_vua, B * L
        # if D == max_dim:
        #     D_tile = math.ceil(D / vu_cores)
        # elif qkv_out_dim_chiplet_vua == max_dim:
        #     qkv_out_dim_tile = math.ceil(qkv_out_dim_chiplet_vua / vu_cores)
        # elif B * L == max_dim:
        #     BL_tile = math.ceil(B * L / vu_cores)
        
        # local_bl, local_d, local_qkv_out = cls.retile_dim(vu_rows, vu_width, vu_cols, BL_tile, D_tile, qkv_out_dim_tile)
        # # local_d*local_qkv_out*local_bl should be >= D*qkv_out_dim_chiplet*B*L
        # if local_d * local_qkv_out * local_bl < D_tile * qkv_out_dim_tile * BL_tile:
        #     raise ValueError("in correct tiling calculation")
        # row_iter = math.ceil(local_bl / vu_rows)
        # col_iter = math.ceil(local_qkv_out / vu_cols)
        # width_iter = math.ceil(local_d / vu_width)

        # comp_latency_qkv_vua = row_iter * col_iter * width_iter + math.log2(vu_width)
        # comp_latency_qkv_vua = cls.cycles_to_ns(comp_latency_qkv_vua, acc_freq)
        # comp_latency_qkv = max(comp_latency_qkv_sa, comp_latency_qkv_vua)
        comp_latency_qkv = comp_latency_qkv_sa

        # 
        # SRAM traffic (local movement) ~ X + W_qkv + (Q+K+V)
        weight_bytes_qkv = D * qkv_out_dim_chiplet * common.ByteperParam
        input_bytes_qkv = B * L * D * common.ByteperParam
        output_bytes_qkv = B * L * qkv_out_dim_chiplet * common.ByteperParam
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
        # L_seq_chiplet_sa = math.ceil(L_seq_chiplet / 2)
        # L_seq_chiplet_vua = L_seq_chiplet_sa

        # SA
        # Check which is bigger between D, L and L_seq_chiplet and divide that by number of systolic arrays
        L_seq_chiplet_sa = L_seq_chiplet
        max_dim = max(D+sa_rows, L_seq_chiplet_sa+sa_cols, L)

        D_tile = D
        L_tile = L
        Lout_tile = L_seq_chiplet_sa
        if D+sa_rows == max_dim:
            D_tile = math.ceil(D / sa_cores)
        elif L_seq_chiplet_sa+sa_cols == max_dim:
            Lout_tile = math.ceil(L_seq_chiplet_sa / sa_cores)
        elif L == max_dim:
            L_tile = math.ceil(L / sa_cores)

        row_iter = math.ceil(D_tile / sa_rows)
        col_iter = math.ceil(Lout_tile / sa_cols)

        # compute latency (in cycles)
        comp_latency_qkt_sa = row_iter * col_iter * L_tile + (sa_rows + sa_cols)
        comp_latency_qkt_sa = B * comp_latency_qkt_sa  # scale by batch size, they don't share anything
        comp_latency_qkt_sa = cls.cycles_to_ns(comp_latency_qkt_sa, acc_freq)

        # # VUA
        # max_dim = max(D, L_seq_chiplet_vua, L)
        # D_tile, L_seq_tile, L_tile = D, L_seq_chiplet_vua, L
        # if D == max_dim:
        #     D_tile = math.ceil(D / vu_cores)
        # elif L_seq_chiplet_vua == max_dim:
        #     L_seq_tile = math.ceil(L_seq_chiplet_vua / vu_cores)
        # elif L == max_dim:
        #     L_tile = math.ceil(L / vu_cores)
        
        # local_L, local_D, local_L_seq = cls.retile_dim(vu_rows, vu_width, vu_cols, L_tile, D_tile, L_seq_tile)
        # # local_D*local_L_seq*local_L should be >= D*L_seq_chiplet*L
        # if local_D * local_L_seq * local_L < D_tile * L_seq_tile * L_tile:
        #     raise ValueError("in correct tiling calculation")
        # row_iter = math.ceil(local_L / vu_rows)
        # col_iter = math.ceil(local_L_seq / vu_cols)
        # width_iter = math.ceil(local_D / vu_width)

        # comp_latency_qkt_vua = B*row_iter * col_iter * width_iter + math.log2(vu_width)
        # comp_latency_qkt_vua = cls.cycles_to_ns(comp_latency_qkt_vua, acc_freq)
        # comp_latency_qkt = max(comp_latency_qkt_sa, comp_latency_qkt_vua)
        comp_latency_qkt = comp_latency_qkt_sa

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
        # L_seq_chiplet_sa = math.ceil(L_seq_chiplet / 2)
        # L_seq_chiplet_vua = L_seq_chiplet_sa
        # SA

        # max dim from D, L, L_seq_chiplet
        L_seq_chiplet_sa = L_seq_chiplet
        max_dim = max(L+sa_rows, L_seq_chiplet_sa+sa_cols, D)

        D_tile = D
        L_tile = L
        Lout_tile = L_seq_chiplet_sa

        if L+sa_rows == max_dim:
            L_tile = math.ceil(L / sa_cores)
        elif L_seq_chiplet_sa+sa_cols == max_dim:
            Lout_tile = math.ceil(L_seq_chiplet_sa / sa_cores)
        elif D == max_dim:
            D_tile = math.ceil(D / sa_cores)

        row_iter = math.ceil(L_tile / sa_rows)
        col_iter = math.ceil(Lout_tile / sa_cols)

        # compute latency (in cycles)
        comp_latency_av_sa = row_iter * col_iter * D_tile + (sa_rows + sa_cols)
        comp_latency_av_sa = B * comp_latency_av_sa  # scale by batch size
        comp_latency_av_sa = cls.cycles_to_ns(comp_latency_av_sa, acc_freq)

        # # VUA
        # max_dim = max(D, L_seq_chiplet_vua, L)
        # D_tile, L_seq_tile, L_tile = D, L_seq_chiplet_vua, L
        # if D == max_dim:
        #     D_tile = math.ceil(D / vu_cores)
        # elif L_seq_chiplet_vua == max_dim:  
        #     L_seq_tile = math.ceil(L_seq_chiplet_vua / vu_cores)
        # elif L == max_dim:
        #     L_tile = math.ceil(L / vu_cores)
        
        # local_D, local_L, local_L_seq = cls.retile_dim(vu_rows, vu_width, vu_cols, D_tile, L_tile, L_seq_tile)
        # # local_D*local_L_seq*local_L should be >= D*L_seq_chiplet*L
        # if local_D * local_L_seq * local_L < D_tile * L_seq_tile * L_tile:
        #     raise ValueError("in correct tiling calculation")
        # row_iter = math.ceil(local_D / vu_rows)
        # col_iter = math.ceil(local_L_seq / vu_cols)
        # width_iter = math.ceil(local_L / vu_width)

        # comp_latency_av_vua = B*row_iter * col_iter * width_iter + math.log2(vu_width)
        # comp_latency_av_vua = cls.cycles_to_ns(comp_latency_av_vua, acc_freq)
        # comp_latency_av = max(comp_latency_av_sa, comp_latency_av_vua)
        comp_latency_av = comp_latency_av_sa

        # SRAM traffic (local): V + scores + output
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
        # d_model_chiplet_sa = math.ceil(d_model_chiplet / 2)
        # d_model_chiplet_vua = d_model_chiplet_sa

        # SA
        # Find max from D, d_model_chiplet, L
        d_model_chiplet_sa = d_model_chiplet
        max_dim = max(D+sa_rows, d_model_chiplet_sa+sa_cols, B*L)

        D_tile = D
        d_model_tile = d_model_chiplet_sa
        L_tile = B*L

        if D+sa_rows == max_dim:
            D_tile = math.ceil(D / sa_cores)
        elif d_model_chiplet_sa+sa_cols == max_dim:
            d_model_tile = math.ceil(d_model_chiplet_sa / sa_cores)
        elif B*L == max_dim:
            L_tile = math.ceil(B*L / sa_cores)

        row_iter = math.ceil(D_tile / sa_rows)
        col_iter = math.ceil(d_model_tile / sa_cols)

        # compute latency (in cycles)
        comp_latency_outproj_sa = row_iter * col_iter * L_tile + (sa_rows + sa_cols)
        comp_latency_outproj_sa = cls.cycles_to_ns(comp_latency_outproj_sa, acc_freq)

        # # VUA
        # max_dim = max(D, d_model_chiplet_vua, B * L)
        # D_tile, d_model_tile, BL_tile = D, d_model_chiplet_vua, B * L
        # if D == max_dim:
        #     D_tile = math.ceil(D / vu_cores)
        # elif d_model_chiplet_vua == max_dim:
        #     d_model_tile = math.ceil(d_model_chiplet_vua / vu_cores)
        # elif B * L == max_dim:
        #     BL_tile = math.ceil(B * L / vu_cores)

        # local_bl, local_d, local_d_model = cls.retile_dim(vu_rows, vu_width, vu_cols, BL_tile, D_tile, d_model_tile)
        # # local_d*local_d_model*local_bl should be >= D*d_model_chiplet*B*L
        # if local_d * local_d_model * local_bl < D_tile * d_model_tile * BL_tile:
        #     raise ValueError("in correct tiling calculation")
        # row_iter = math.ceil(local_bl / vu_rows)
        # col_iter = math.ceil(local_d_model / vu_cols)
        # width_iter = math.ceil(local_d / vu_width)

        # comp_latency_out = row_iter * col_iter * width_iter + math.log2(vu_width)
        # comp_latency_outproj_vua = cls.cycles_to_ns(comp_latency_out, acc_freq)
        # comp_latency_outproj = max(comp_latency_outproj_sa, comp_latency_outproj_vua)
        comp_latency_outproj = comp_latency_outproj_sa

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
                                 logic_name="unified", ext_bw=128, acc_freq: float | None = None,
                                 concurrent_chiplets: int = 1, first_layer: bool = False, last_layer: bool = False) -> analytics:

        acc_freq = acc_freq or cls.acc_freq
        specified_logic = cls._logic(logic_name)

        vu_rows = specified_logic['config']['VU_rows']
        vu_cols = specified_logic['config']['VU_cols']
        vu_width = specified_logic['config']['VU_width']
        vu_cores = specified_logic['vua_core']
        sa_rows = specified_logic['config']['sa_rows']
        sa_cols = specified_logic['config']['sa_cols']
        sa_cores = specified_logic['sa_core']
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
        # qkv_out_dim_chiplet_sa = math.ceil(qkv_out_dim_chiplet / 2)
        # qkv_out_dim_chiplet_vua = qkv_out_dim_chiplet_sa
        
        # # SA
        # # check which is bigger between D, bs*batch and qkv_out_dim_chiplet and divide that by number of systolic arrays
        # max_dim = max(D+sa_rows, qkv_out_dim_chiplet_sa+sa_cols, B) # since L=1 during decoding

        # D_tile = D
        # bs_tile = B
        # qkv_out_tile = qkv_out_dim_chiplet_sa

        # if D+sa_rows == max_dim:
        #     D_tile = math.ceil(D / sa_cores)
        # elif qkv_out_dim_chiplet_sa+sa_cols == max_dim:
        #     qkv_out_tile = math.ceil(qkv_out_dim_chiplet_sa / sa_cores)
        # elif B == max_dim:
        #     bs_tile = math.ceil(B / sa_cores)

        # row_iter = math.ceil(D_tile / sa_rows)
        # col_iter = math.ceil(qkv_out_tile / sa_cols)
        # # compute latency (in cycles)
        # comp_latency_qkv_sa = row_iter * col_iter * bs_tile + (sa_rows + sa_cols)
        # comp_latency_qkv_sa = cls.cycles_to_ns(comp_latency_qkv_sa, acc_freq)

        # VUA
        qkv_out_dim_chiplet_vua = qkv_out_dim_chiplet  
        max_dim = max(D, qkv_out_dim_chiplet_vua, B)
        D_tile, qkv_out_dim_tile, B_tile = D, qkv_out_dim_chiplet_vua, B
        if D == max_dim:
            D_tile = math.ceil(D / vu_cores)
        elif qkv_out_dim_chiplet_vua == max_dim:
            qkv_out_dim_tile = math.ceil(qkv_out_dim_chiplet_vua / vu_cores)
        elif B == max_dim:
            B_tile = math.ceil(B / vu_cores)
        local_b, local_d, local_qkv_out = cls.retile_dim(vu_rows, vu_width, vu_cols, B_tile, D_tile, qkv_out_dim_tile)
        # local_d*local_qkv_out*local_b should be >= D*qkv_out_dim_chiplet*B
        if local_d * local_qkv_out * local_b < D_tile * qkv_out_dim_tile * B_tile:
            raise ValueError("in correct tiling calculation")
        row_iter = math.ceil(local_b / vu_rows)
        col_iter = math.ceil(local_qkv_out / vu_cols)
        width_iter = math.ceil(local_d / vu_width)

        comp_latency_qkv_vua = row_iter * col_iter * width_iter + math.log2(vu_width)
        comp_latency_qkv = cls.cycles_to_ns(comp_latency_qkv_vua, acc_freq)
        # comp_latency_qkv = max(comp_latency_qkv_sa, comp_latency_qkv_vua)

        # SRAM traffic (local movement) ~ X + W_qkv + (Q+K+V)
        weight_bytes_qkv = D * qkv_out_dim_chiplet * common.ByteperParam
        input_bytes_qkv = B * 1 * D * common.ByteperParam
        output_bytes_qkv = B * 1 * qkv_out_dim_chiplet * common.ByteperParam
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
        # L_seq_chiplet_sa = math.ceil(L_seq_chiplet / 2)
        # L_seq_chiplet_vua = L_seq_chiplet_sa
        # # SA
        # # Check which is bigger between D, 1 and L_seq_chiplet and divide that by number of systolic arrays
        # max_dim = max(D+sa_rows, L_seq_chiplet_sa+sa_cols, 1)

        # D_tile = D
        # L_tile = 1
        # Lout_tile = L_seq_chiplet_sa

        # if D+sa_rows == max_dim:
        #     D_tile = math.ceil(D / sa_cores)
        # elif L_seq_chiplet_sa+sa_cols == max_dim:
        #     Lout_tile = math.ceil(L_seq_chiplet_sa / sa_cores)
        # elif 1 == max_dim:
        #     L_tile = math.ceil(1 / sa_cores)

        # row_iter = math.ceil(D_tile / sa_rows)
        # col_iter = math.ceil(Lout_tile / sa_cols)
        # # compute latency (in cycles)
        # comp_latency_qkt_sa = row_iter * col_iter * L_tile + (sa_rows + sa_cols)
        # comp_latency_qkt_sa = B * comp_latency_qkt_sa  # scale by batch size
        # comp_latency_qkt_sa = cls.cycles_to_ns(comp_latency_qkt_sa, acc_freq)

        # VUA
        L_seq_chiplet_vua = L_seq_chiplet
        max_dim = max(D, L_seq_chiplet_vua, 1)
        D_tile, L_seq_tile, one_tile = D, L_seq_chiplet_vua, 1
        if D == max_dim:
            D_tile = math.ceil(D / vu_cores)
        elif L_seq_chiplet_vua == max_dim:
            L_seq_tile = math.ceil(L_seq_chiplet_vua / vu_cores)
        elif 1 == max_dim:
            one_tile = math.ceil(1 / vu_cores)
        local_1, local_D, local_L_seq = cls.retile_dim(vu_rows, vu_width, vu_cols, one_tile, D_tile, L_seq_tile)
        # local_D*local_L_seq*local_1 should be >= D*L_seq_chiplet*1
        if local_D * local_L_seq * local_1 < D_tile * L_seq_tile * one_tile:
            raise ValueError("in correct tiling calculation")
        row_iter = math.ceil(local_1 / vu_rows)
        col_iter = math.ceil(local_L_seq / vu_cols)
        width_iter = math.ceil(local_D / vu_width)

        comp_latency_qkt_vua = B*row_iter * col_iter * width_iter + math.log2(vu_width)
        comp_latency_qkt_vua = cls.cycles_to_ns(comp_latency_qkt_vua, acc_freq)
        # comp_latency_qkt = max(comp_latency_qkt_sa, comp_latency_qkt_vua)
        comp_latency_qkt = comp_latency_qkt_vua

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
        # d_model_chiplet_sa = math.ceil(d_model_chiplet / 2)
        # d_model_chiplet_vua = d_model_chiplet_sa

        # # SA

        # # max dim from 1, L, d_model_chiplet
        # max_dim = max(L+sa_rows, d_model_chiplet_sa+sa_cols, 1)
        # D_tile = d_model_chiplet_sa
        # L_tile = L
        # Lout_tile = 1

        # if L+sa_rows == max_dim:
        #     L_tile = math.ceil(L / sa_cores)
        # elif d_model_chiplet_sa+sa_cols == max_dim:
        #     D_tile = math.ceil(d_model_chiplet_sa / sa_cores)
        # elif 1 == max_dim:
        #     Lout_tile = math.ceil(1 / sa_cores)

        # row_iter = math.ceil(L_tile / sa_rows)
        # col_iter = math.ceil(Lout_tile / sa_cols)

        # # compute latency (in cycles)
        # comp_latency_av_sa = row_iter * col_iter * D_tile + (sa_rows + sa_cols)
        # comp_latency_av_sa = B * comp_latency_av_sa  # scale by batch size
        # comp_latency_av_sa = cls.cycles_to_ns(comp_latency_av_sa, acc_freq)
        
        # VUA
        d_model_chiplet_vua = d_model_chiplet
        max_dim = max(1, d_model_chiplet_vua, L)
        one_tile, d_model_tile, L_tile = 1, d_model_chiplet_vua, L
        if 1 == max_dim:
            one_tile = math.ceil(1 / vu_cores)
        elif d_model_chiplet_vua == max_dim:
            d_model_tile = math.ceil(d_model_chiplet_vua / vu_cores)
        elif L == max_dim:
            L_tile = math.ceil(L / vu_cores)
        local_1, local_L, local_d_model = cls.retile_dim(vu_rows, vu_width, vu_cols, one_tile, L_tile, d_model_tile)
        # local_d_model*local_L*local_1 should be >= d_model_chiplet*L*1
        if local_d_model * local_L * local_1 < d_model_tile * L_tile * one_tile:
            raise ValueError("in correct tiling calculation")
        row_iter = math.ceil(local_1 / vu_rows)
        col_iter = math.ceil(local_d_model / vu_cols)
        width_iter = math.ceil(local_L / vu_width)

        comp_latency_av_vua = B*row_iter * col_iter * width_iter + math.log2(vu_width)
        comp_latency_av_vua = cls.cycles_to_ns(comp_latency_av_vua, acc_freq)
        # comp_latency_av = max(comp_latency_av_sa, comp_latency_av_vua)
        comp_latency_av = comp_latency_av_vua

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
        # d_model_chiplet_sa = math.ceil(d_model_chiplet / 2)
        # d_model_chiplet_vua = d_model_chiplet_sa
        # # SA
        # # Find max from D, d_model_chiplet, B
        # max_dim = max(D+sa_rows, d_model_chiplet_sa+sa_cols, B) # since L=1 during decoding
        # D_tile = D
        # d_model_tile = d_model_chiplet_sa
        # L_tile = B
        # if D+sa_rows == max_dim:
        #     D_tile = math.ceil(D / sa_cores)
        # elif d_model_chiplet_sa+sa_cols == max_dim:
        #     d_model_tile = math.ceil(d_model_chiplet_sa / sa_cores)
        # elif B == max_dim:
        #     L_tile = math.ceil(B / sa_cores)

        # row_iter = math.ceil(D_tile / sa_rows)
        # col_iter = math.ceil(d_model_tile / sa_cols)
        # # compute latency (in cycles)
        # comp_latency_outproj_sa = row_iter * col_iter * L_tile + (sa_rows + sa_cols)
        # comp_latency_outproj_sa = cls.cycles_to_ns(comp_latency_outproj_sa, acc_freq)

        # VUA
        d_model_chiplet_vua = d_model_chiplet
        max_dim = max(D, d_model_chiplet_vua, B)
        D_tile, d_model_tile, B_tile = D, d_model_chiplet_vua, B
        if D == max_dim:
            D_tile = math.ceil(D / vu_cores)
        elif d_model_chiplet_vua == max_dim:
            d_model_tile = math.ceil(d_model_chiplet_vua / vu_cores)
        elif B == max_dim:
            B_tile = math.ceil(B / vu_cores)
        local_b, local_d, local_d_model = cls.retile_dim(vu_rows, vu_width, vu_cols, B_tile, D_tile, d_model_tile)
        # local_d*local_d_model*local_b should be >= D*d_model_chiplet*B
        if local_d * local_d_model * local_b < D_tile * d_model_tile * B_tile:
            raise ValueError("in correct tiling calculation")
        row_iter = math.ceil(local_b / vu_rows)
        col_iter = math.ceil(local_d_model / vu_cols)
        width_iter = math.ceil(local_d / vu_width)

        comp_latency_out = row_iter * col_iter * width_iter + math.log2(vu_width)
        comp_latency_outproj_vua = cls.cycles_to_ns(comp_latency_out, acc_freq)
        # comp_latency_outproj = max(comp_latency_outproj_sa, comp_latency_outproj_vua)
        comp_latency_outproj = comp_latency_outproj_vua

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
    def rms_norm_est(cls, in_size, logic_name="unified", ext_bw=128, batch_size=1,
                     concurrent_chiplets: int = 1, first_layer: bool = False, last_layer: bool = False) -> analytics:
        specified_logic = cls._logic(logic_name)
        GActs: float = specified_logic['config']['sfu'] * specified_logic['sfu']

        in_size = math.ceil(in_size * batch_size/concurrent_chiplets)
        Comp_latency = math.ceil(in_size / GActs)

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
    def silu_est(cls, in_size, logic_name="unified", ext_bw=128, batch_size=1,
                 concurrent_chiplets: int = 1, first_layer: bool = False, last_layer: bool = False) -> analytics:
        return cls.rms_norm_est(in_size, logic_name, ext_bw, batch_size, concurrent_chiplets, first_layer, last_layer)

    @classmethod
    def softplus_est(cls, in_size, logic_name="unified", ext_bw=128, batch_size=1,
                     concurrent_chiplets: int = 1, first_layer: bool = False, last_layer: bool = False) -> analytics:
        return cls.rms_norm_est(in_size, logic_name, ext_bw, batch_size, concurrent_chiplets, first_layer, last_layer)
    
    @classmethod
    def square_relu_est(cls, in_size, logic_name="unified", ext_bw=128, batch_size=1,
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

    def retile_dim(rows: int, vu_width: int, cols: int, M: int, N: int, K: int) -> tuple[int, int, int]:
        # M maps to rows
        # N maps to width
        # K maps to cols
        extra_rows = rows - M
        extra_width = vu_width - N
        extra_cols = cols - K
        local_M, local_N, local_K = M, N, K

        if extra_width < 0 and extra_rows > 0:
            local_M = rows
            local_N = math.ceil(N * M / local_M)
            extra_width = vu_width - local_N

        if extra_width < 0 and extra_cols > 0:
            local_K = cols
            local_N = math.ceil(K * N / local_K)

        return local_M, local_N, local_K
