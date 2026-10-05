"""Independent DSE controls for agent admission, batching and retained KV."""

from dataclasses import dataclass
import math


@dataclass
class AgentSchedulerConfig:
    priority: str = 'fcfs'
    batching: str = 'immediate'
    max_batch_wait_s: float = 0.001
    service_estimator: str = 'fixed_output'
    estimated_output_tokens: int = 128
    prefill_token_s: float = 0.00001
    decode_token_s: float = 0.001
    default_session_slo_s: float = 10.0
    max_active_batches: int = 0  # Zero uses the existing hardware/worker limit.
    prefix_cache: str = 'none'
    prefix_capacity_bytes: int = 0
    host_dram_capacity_bytes: int = 256 * 1024**3
    host_link_bandwidth_gbps: float = 25.0
    host_dram_bandwidth_gbps: float = 100.0
    host_transfer_latency_s: float = 0.00001

    def __post_init__(self):
        if self.priority not in ('fcfs', 'sjf', 'hrrn', 'edf', 'least_slack'):
            raise ValueError('Unknown agent priority policy.')
        if self.batching not in ('immediate', 'timeout'):
            raise ValueError('Agent batching must be immediate or timeout.')
        if self.service_estimator not in ('fixed_output', 'oracle_output'):
            raise ValueError('Unknown service estimator.')
        for name in ('max_batch_wait_s', 'prefill_token_s', 'decode_token_s', 'default_session_slo_s'):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f'{name} must be finite nonnegative seconds.')
        if self.decode_token_s == 0 or self.default_session_slo_s == 0:
            raise ValueError('Decode estimate and default SLO must be positive.')
        if type(self.estimated_output_tokens) is not int or self.estimated_output_tokens < 1:
            raise ValueError('Estimated output length must be a positive integer.')
        if type(self.max_active_batches) is not int or self.max_active_batches < 0:
            raise ValueError('Maximum active batches must be a nonnegative integer.')
        if type(self.prefix_capacity_bytes) is not int or self.prefix_capacity_bytes < 0:
            raise ValueError('Prefix capacity must be nonnegative integer bytes.')
        if self.prefix_cache not in ('none', 'lru', 'lru_offload') or (self.prefix_cache != 'none' and not self.prefix_capacity_bytes):
            raise ValueError('Prefix cache requires none, or lru/lru_offload with positive capacity.')
        if self.prefix_cache == 'none' and self.prefix_capacity_bytes:
            raise ValueError('Set prefix_cache=lru to use a prefix capacity.')
        if type(self.host_dram_capacity_bytes) is not int or self.host_dram_capacity_bytes < 1:
            raise ValueError('Host DRAM capacity must be positive integer bytes.')
        for name in ('host_link_bandwidth_gbps', 'host_dram_bandwidth_gbps', 'host_transfer_latency_s'):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f'{name} must be finite and nonnegative.')
            if name.endswith('gbps') and value == 0:
                raise ValueError('Host transfer bandwidth must be positive decimal GB/s.')
