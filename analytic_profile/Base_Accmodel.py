from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass, field

from Sim.config.utils import get_all_subclasses

# best-effort import of project "common" logic_lib (may be None in tests)

from Sim import common


class BaseAccModel(ABC):
    """
    Minimal base class for accelerator performance models.

    Responsibilities:
    - Provide common helper conversions helpers.
    - Define abstract kernel latency/count APIs subclasses should implement.
    """
    acc_freq: float  # in Hz

    @classmethod
    def config_params(cls, **params):
        args = {}
        for key, value in params.items():
            args[key] = value
        return args

    @classmethod
    def create_from_name(cls, name: str) -> Any:
        for subclass in get_all_subclasses(cls):
            if subclass.get_name() in name:
                return subclass
        raise ValueError(f"[{cls.__name__}] Invalid name: {name}")
    
    @staticmethod
    def _load_logic(logic_name: Optional[str]) -> Optional[Dict[str, Any]]:
        if logic_name is None:
            return None
        if common is None:
            return None
        try:
            return common.logic_lib[logic_name]
        except Exception:
            return None

    # --- conversion helpers ---
    @classmethod
    def sec_to_cycles(cls, t_s: float) -> int:
        """Convert seconds to cycles using acc_freq (cycles per second)."""
        if t_s <= 0:
            return 0
        return int(round(t_s * cls.acc_freq))
	
    @classmethod
    def ns_to_cycles(cls, t_ns: float) -> int:
        """Convert nanoseconds to cycles."""
        return cls.sec_to_cycles(t_ns * 1e-9)

    @classmethod
    def ext_bytes_per_s(cls, ext_bw: float) -> float:
        """External (HBM) bandwidth (GB/s) in bytes/sec."""
        return max(1e-12, ext_bw * 1024 ** 3)
    
    @classmethod
    def sram_bytes_per_s(cls, sram_gb_s: float) -> float:
        return max(1e-12, sram_gb_s * 1024 ** 3)

    @classmethod
    def sram_latency_for_bytes(cls, param_count: int, sram_gb_s: float) -> int:
        """Return cycles to move `param_count` parameters to/from SRAM (bytes -> cycles)."""
        t_s = param_count * common.ByteperParam / cls.sram_bytes_per_s(sram_gb_s)
        return cls.sec_to_cycles(t_s)
    
    @classmethod
    def hbm_latency_for_bytes(cls, param_count: int, ext_bw: float) -> int:
        """Return cycles to move `param_count` parameters to/from external memory (HBM)."""
        t_s = param_count * common.ByteperParam / cls.ext_bytes_per_s(ext_bw)
        return cls.sec_to_cycles(t_s)

    @classmethod
    def compute_latency_from_macs_ns(cls, mac_ops: float, macs_per_ns: float) -> int:
        """
        Given mac_ops in MACs (not time), compute cycles using macs_per_ns.
        macs_per_ns is MACs / ns -> time_ns = mac_ops / macs_per_ns
        then convert ns -> cycles.
        """
        if macs_per_ns <= 0:
            return 0
        time_ns = mac_ops / macs_per_ns
        return cls.ns_to_cycles(time_ns)

    @classmethod
    def profile_kernel(cls, kernel_name, **kwargs):
        """Dispatch explicit modern operators while preserving legacy profiles."""
        from analytic_profile.modern import BaseOperatorProfile
        profile = BaseOperatorProfile.find(kernel_name)
        if profile is not None:
            return profile().profile(**kwargs)
        return cls.get_kernel(kernel_name, **kwargs)

    @abstractmethod
    def get_kernel(cls, kernel_name: str, *args, **kwargs) -> Any:
        raise NotImplementedError
    
    # --- abstract kernel APIs to implement in subclasses ---
    @abstractmethod
    def get_pf_ssm_latency(cls, *args, **kwargs) -> Any:
        raise NotImplementedError

    @abstractmethod
    def get_dc_ssm_latency(cls, *args, **kwargs) -> Any:
        raise NotImplementedError

    @abstractmethod
    def get_pf_mlp_latency(cls, *args, **kwargs) -> Any:
        raise NotImplementedError

    @abstractmethod
    def get_pf_conv1d_latency(cls, *args, **kwargs) -> Any:
        raise NotImplementedError

    @abstractmethod
    def get_pf_attention_latency(cls, *args, **kwargs) -> Any:
        raise NotImplementedError
    
    @abstractmethod
    def get_dc_attention_latency(cls, *args, **kwargs) -> Any:
        raise NotImplementedError
    
    @abstractmethod
    def rms_norm_est(cls, *args, **kwargs) -> Any:
        raise NotImplementedError
    
    @abstractmethod
    def silu_est(cls, *args, **kwargs) -> Any:
        raise NotImplementedError
    
    @abstractmethod
    def softplus_est(cls, *args, **kwargs) -> Any:
        raise NotImplementedError

    @abstractmethod
    def get_name() -> Any:
        raise NotImplementedError
    
    
    def __str__(self):
        return self.get_name()