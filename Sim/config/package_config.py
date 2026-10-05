"""Static package pipeline and an effective full-duplex scale-up fabric."""

from dataclasses import dataclass
import math


@dataclass
class PackageConfig:
    # Existing architecture/placement fields describe ONE package.
    count: int = 1
    partition: str = 'contiguous'
    communication_model: str = 'alpha_beta'
    bandwidth_gbps: float = 25.0  # Decimal GB/s, per package, per direction.
    efficiency: float = 0.8
    latency_ns: float = 1000.0  # Assumed end-to-end startup; not a measured NVLink latency.
    token_bytes: int = 4

    def __post_init__(self):
        if type(self.count) is not int or self.count < 1:
            raise ValueError('Package count must be a positive integer.')
        if type(self.token_bytes) is not int or self.token_bytes < 1:
            raise ValueError('Token feedback size must be a positive integer.')
        if not math.isfinite(self.bandwidth_gbps) or self.bandwidth_gbps <= 0:
            raise ValueError('Package bandwidth must be positive and finite.')
        if not math.isfinite(self.efficiency) or not 0 < self.efficiency <= 1:
            raise ValueError('Package link efficiency must be in (0, 1].')
        if not math.isfinite(self.latency_ns) or self.latency_ns < 0:
            raise ValueError('Package startup latency must be finite and nonnegative.')

    def validate_execution(self, config):
        if self.count == 1:
            return
        if config.mapping_config.task_parallelism != 'pipeline' or config.mapping_config.mapping_strategy != 'static':
            raise ValueError('Multi-package execution requires static pipeline mapping.')
        if config.cluster_config.local_scheduler != 'static' or config.cluster_config.num_replicas != 1:
            raise ValueError('Multi-package execution uses one model instance with static batching.')
        if not config.placmt_config.two_d_grid:
            raise ValueError('Multi-package execution currently replicates uniform package meshes.')
        if config.simulator_backend not in ('hydra_sim', 'hydra_packet', 'chipsim_contended'):
            raise ValueError('Multi-package execution requires Analytic, Packet or contended CHIPSIM.')
