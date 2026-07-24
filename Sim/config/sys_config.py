import os
from abc import ABC
from dataclasses import dataclass, field
from Sim.config.base_poly_config import BasePolyConfig
from typing import Any, List, Optional, Tuple
import numpy as np
import networkx as nx
import Sim.config.utils as utils
from Sim.entities.chips_network import chip_graph
from Sim.config.model_config import BaseModelConfig
from Sim.config.utils import dataclass_to_dict, chiplets_lib
from Sim.scheduler import BaseReqScheduler
from Sim.placer import BasePlacer
import json


from Sim.logger import init_logger, _setup_logger
_setup_logger()   # root logger handles console + file
logger = init_logger(__name__)  # child logger inherits handlers

@dataclass
class ClusterConfig:
    num_replicas: int = field(
        default=1,
        metadata={"help": "Number of replicas."},
    )
    global_scheduler: str = field(
        default="rr",
        metadata={
            "help": "Global scheduler type."
        },
    )
    local_scheduler: str = field(
        default="static",
        metadata={"help": "Local scheduler configuration (static/vllm/vllm_latest)."},
    )
    batch_size: int = field(
        default=4,
        metadata={"help": "Batch size for the cluster."},
    )
    local_scheduler_inst: Any = field(
        default=None,
        metadata={"help": "Local scheduler instance."},
    )
    
    def __post_init__(self):
        self.local_scheduler_inst = BaseReqScheduler.BaseReqScheduler.create_from_name(self.local_scheduler)

@dataclass
class ArchConfig:
    num_nodes: int = field(
        default=24,
        metadata={"help": "Number of chiplets in the interposer."},
    )
    
    intp_width: int = field(
        default=6,
        metadata={"help": "Width of interposer."},
    )
    
    intp_height: int = field(
        default=4,
        metadata={"help": "Height of interposer."},
    )
    
@dataclass
class BaseRequestLengthGeneratorConfig(BasePolyConfig):
    seed: int = field(
        default=42,
        metadata={"help": "Seed for the random number generator."},
    )
    max_tokens: int = field(
        default=128000,
        metadata={"help": "Maximum tokens."},
    )

# TBD    
@dataclass
class BaseChipletPlacerConfig(BasePolyConfig):
    int_width: int = field(
        default=ArchConfig.intp_width,
        metadata={"help": "Width of the interposer."},
    )
    
    int_height: int = field(
        default=ArchConfig.intp_height,
        metadata={"help": "Height of the interposer."},
    )
    
    num_nodes: int = field( 
        default=ArchConfig.num_nodes,
        metadata={"help": "Number of chiplets in the interposer."},
    )
    
    chiplets_types: dict = field(
        default_factory=lambda: utils.chiplet_types_dict,
        metadata={"help": "Dictionary of chiplet types and their areas."},
    )
    
    chiplets_mapping: List[List[int]] = field(
        default_factory=lambda: [[0] * ArchConfig.intp_height for _ in range(ArchConfig.intp_width)],
        metadata={"help": "Mapping of chiplets to the interposer (2D list, converted to ndarray at runtime)."},
    )
    
    two_d_grid: bool = field(
        default=True,
        metadata={"help": "Whether to use a 2D uniform grid for the chiplets."},
    )

    placement_trace: Optional[List[Tuple[int, List[Tuple[int, int]]]]] = field(
        default=None,
        metadata={"help": "Placement trace for non-uniform grid graphs."},
    )

    placement_trace_file: Optional[str] = field(
        default='b200_package_placement_trace.csv',
        metadata={"help": "Placement trace CSV file for non-uniform grid graphs."},
    )
    # 7	1	12	1	3
    chiplet_alloc: dict = field(
        default_factory=lambda: {
            "marca_p": 2,
            "marca_d": 11,
            "tscs_p": 2,
            "tscs_d": 3,
            "systolicarray_p": 0,
            "HBM3": 6, # 16G
            "GDDR7": 0, # 16G
            "HBM3e": 0, # 16G
            "B200_p": 0,
            "vuarray_d": 0,
            "unified_p": 0,
        },
        metadata={"description": "Chiplet allocation in the system"},
    )
    
    placer_label: str = field(
        default="rr",
        metadata={"help": "Chiplet placer to use (e.g., bw, rr, trace)."},
    )
    
    placer_inst: Any = field(
        default=None,
        metadata={"help": "Local scheduler instance."},
    )

    def __post_init__(self):
        self.chiplets_mapping = np.array(self.chiplets_mapping)

        csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'../../analytic_profile/{self.placement_trace_file}')
        csv_path = os.path.normpath(csv_path)
        if self.placement_trace is None and os.path.exists(csv_path) and not self.two_d_grid:
            self.placement_trace = utils.load_placement_trace_from_csv(csv_path)

        self.chip_graph = chip_graph(self.num_nodes, self.int_width, self.int_height, self.chiplets_mapping, self.two_d_grid, self.placement_trace)
        self.placer_inst = BasePlacer.BasePlacer.create_from_name(self.placer_label)
@dataclass
class TraceRequestLengthGeneratorConfig(BaseRequestLengthGeneratorConfig):
    trace_file: str = field(
        default="dataset/longwriter/long_writer_transformer_1.3b_6k.csv",
        metadata={"help": "Path to the trace request length generator file. \
                  dataset/arxiv/arxiv_summarization_stats_llama3.csv \
                  dataset/bwb/bwb_translation_stats_llama3.csv  \
                  dataset/longwriter/long_writer_transformer_1.3b_6k.csv \
                      dataset/chat/chat1m_stats_llama3.csv"},
    )
    prefill_scale_factor: float = field(
        default=1,
        metadata={
            "help": "Prefill scale factor for the trace request length generator."
        },
    )
    decode_scale_factor: float = field(
        default=1,
        metadata={
            "help": "Decode scale factor for the trace request length generator."
        },
    )
    request_type: str = field(
        default="e2e",
        metadata={"help": "Type of requests to generate (prefill/decode/e2e)."},
    )

    # @staticmethod
    # def get_type():
    #     return RequestLengthGeneratorType.TRACE

@dataclass
class BaseRequestGeneratorConfig(BasePolyConfig):
    seed: int = field(
        default=42,
        metadata={"help": "Seed for the random number generator."},
    )
    
@dataclass
class BaseRequestIntervalGeneratorConfig(BasePolyConfig):
    seed: int = field(
        default=42,
        metadata={"help": "Seed for the random number generator."},
    )
    
@dataclass
class StaticRequestIntervalGeneratorConfig(BaseRequestIntervalGeneratorConfig):
    interval: float = field(
        default=0.01,
        metadata={"help": "Static request interval in seconds."},
    )
    


@dataclass
class TraceRequestGeneratorConfig(BaseRequestGeneratorConfig):
    num_requests: Optional[int] = field(
        default=10000,
        metadata={"help": "Number of requests for Trace Request Generator."},
    )
    trace_length_generator_config: BaseRequestLengthGeneratorConfig = field(
        default_factory=TraceRequestLengthGeneratorConfig,
        metadata={"help": "Request length generator configuration."},
    )
    trace_interval_generator_config: BaseRequestIntervalGeneratorConfig = field(
        default_factory=StaticRequestIntervalGeneratorConfig,
        metadata={"help": "Request interval generator configuration."},
    )
    


@dataclass
class BaseMappingConfig(BasePolyConfig):
    mapping_strategy: str = field(
        default="elastic",
        metadata={"help": "Mapping strategy to use (static/fcfs/worksteal/elastic)."},
    )

    task_parallelism: str = field(
        # default='tensor',
        default='pipeline',
        metadata={"help": "Task parallelism strategy (e.g., pipeline, tensor)."},
    )

    int_width: int = field(
        default=5,
        metadata={"help": "Width of the interposer."},
    )
    
    int_height: int = field(
        default=5,
        metadata={"help": "Height of the interposer."},
    )
    
    num_nodes: int = field( 
        default=25,
        metadata={"help": "Number of chiplets in the interposer."},
    )
    
    chiplets_types: dict = field(
        default_factory=lambda: utils.chiplet_types_dict,
        metadata={"help": "Dictionary of chiplet types and their areas."},
    )
    
    chiplets_mapping: List[List[int]] = field(
        default_factory=lambda: [[0] * 5 for _ in range(5)],
        metadata={"help": "Mapping of chiplets to the interposer (2D list)."},
    )
    
    two_d_grid: bool = field(
        default=True,
        metadata={"help": "Whether to use a 2D uniform grid for the chiplets."},
    )
    
    link_reserve: bool = field(
        default=True,
        metadata={"help": "Whether to reserve links for the mapping kernels."},
    )

    def __post_init__(self):
        self.chiplets_mapping = np.array(self.chiplets_mapping)

@dataclass
class WorkloadConfig:
    dataset : str = field(
        default="chat",
        metadata={"help": "Dataset to use for the workload. arxiv/bwb/chat/longwriter."},
    )
    
    model : str = field( 
        default="nemotronh-4b",
        metadata={"help": "Model to use for the workload llama3-8b/mamba2-3b/nemotronh-4b/zamba2-7b/jamba-mini/jamba-tiny."},
    )
    
    length_request: int = field(
        default=1000,
        metadata={"help": "Length of the request. -1 means the whole dataset"},
    )

    bytes_per_param: int = field(
        default=1,
        # default=2,
        metadata={"help": "Bytes per parameter."},
    )
    
    request_generator_config: BaseRequestGeneratorConfig = field(
        default_factory=TraceRequestGeneratorConfig,
        metadata={"help": "Request generator config."},
    )
    
    # model_config: BaseModelConfig = field(
    #     default_factory=lambda: BaseModelConfig.create_from_name("llama3-8b"),
    #     metadata={"help": "Model configuration."},
    # )
    
    def __post_init__(self):
        self.model_config: BaseModelConfig = BaseModelConfig.create_from_name(
            self.model
        )


    
@dataclass 
class ChipletsConfig:
    avail_chiplets: List[str] = field(
        default_factory=lambda: utils.avail_chiplets,
        metadata={"help": "List of available chiplets."},
    )
    
    D2D_NoI_bw: int = field(
        default=256,
        metadata={"help": "D2D NoI bandwidth GBps.(128/256/384/512/640)"},
    )
    
    HBM_IO_bw: int = field( 
        default=600,
        metadata={"help": "HBM IO bandwidth GBps."},
    )

    area_limit: int = field(
        default=144,
        metadata={"help": "Area limit for the chiplets (121/144) mm^2."},
    )

    area_utilization: float = field(
        default=1.0,
        metadata={"help": "Area utilization for the chiplets (0-1)."},
    )
    
    tech_node: str = field(
        default="28nm",
        metadata={"help": "Technology node for the chiplets."},
    )

    comp_libs: List[str] = field(
        default_factory=lambda: ['marca_d','marca_p','tscs_d','tscs_p'],
        metadata={"help": "List of component libraries."},
    )

    mem_libs: List[str] = field(
        # default_factory=lambda: ['gddr7'],
        default_factory=lambda: ['hbm3'],
        metadata={"help": "List of memory libraries."},
    )   
    
    mem_sizes: List[int] = field(
        default_factory=lambda: [16],
        metadata={"help": "List of memory sizes in GB."},
    )

    chips_lib: chiplets_lib = field(
        default_factory=chiplets_lib,
        metadata={"help": "Chiplets library."},
    )

    Fixed_chiplet_area: bool = field(
        default=True,
        metadata={"help": "Whether to use fixed chiplet area."},
    )

    def __post_init__(self):
        utils.NoI_bw = self.D2D_NoI_bw
        utils.Fixed_chiplet_area = self.Fixed_chiplet_area
        # TODO: not dynamic now
        utils.used_bws = utils.used_bws_dict[self.D2D_NoI_bw] # just a default
        utils.IO_bw = self.HBM_IO_bw
        if self.Fixed_chiplet_area:
            # This needs to be 128/256/384/512/640
            # This case area_limit makes a difference
            self.chips_lib = chiplets_lib(node=self.tech_node, logic_lib=self.comp_libs, mem_lib=self.mem_libs,
                                          area_limit=self.area_limit, area_utilization=self.area_utilization, NoI_bw=self.D2D_NoI_bw, hw_bw=self.D2D_NoI_bw)
        else:
            self.chips_lib = chiplets_lib(node=self.tech_node, logic_lib=self.comp_libs, mem_lib=self.mem_libs,
                                          area_limit=self.area_limit, area_utilization=self.area_utilization, NoI_bw=self.D2D_NoI_bw)

@dataclass
class MetricsConfig:
    """Metric configuration."""

    write_metrics: bool = field(
        default=True,
        metadata={"help": "Whether to write metrics."},
    )
    verbose: bool = field(
        default=False,
        metadata={"help": "Whether to record verbose metrics (mem,comp,bw)."},
    )
    output_dir: str = field(
        default="simulator_output",
        metadata={"help": "Output directory."},
    )
    cache_dir: str = field(
        default="cache",
        metadata={"help": "Cache directory."},
    )
    label_name: str = field(
        default="",
        metadata={"help": "Label name for the metrics."},
    )

    def __post_init__(self):
        utils.verbose = self.verbose
        import os
        from datetime import datetime
        self.output_dir = (
            f"{self.output_dir}/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')}+{self.label_name}"
        )
        os.makedirs(self.output_dir, exist_ok=True)

@dataclass
class HPSim_Config(ABC):
    seed: int = field(
        default=42,
        metadata={"help": "Seed for the random number generator."},
    )
    log_level: str = field(
        default="info",
        metadata={"help": "Logging level."},
    )
    time_limit: int = field(
        default=100,  # in seconds, 0 is no limit
        metadata={"help": "Time limit for simulation in seconds. 0 means no limit."},
    )
    
    cluster_config: ClusterConfig = field(
        default_factory=ClusterConfig,
        metadata={"help": "Cluster configuration."},
    )
    
    arch_config: ArchConfig = field(
        default_factory=ArchConfig,
        metadata={"help": "Architecture configuration."},
    )
    
    chips_config: ChipletsConfig = field(
        default_factory=ChipletsConfig,
        metadata={"help": "Chiplets configuration."},
    )
    
    placmt_config: BaseChipletPlacerConfig = field(
        default_factory=BaseChipletPlacerConfig,
        metadata={"help": "Placement configuration."},
    )
    
    workload_config: WorkloadConfig = field(
        default_factory=WorkloadConfig,
        metadata={"help": "Workload configuration."},
    )
    
    mapping_config: BaseMappingConfig = field(
        default_factory=BaseMappingConfig,
        metadata={"help": "Mapping configuration."},
    )
    
    metrics_config: MetricsConfig = field(
        default_factory=MetricsConfig,
        metadata={"help": "Metrics configuration."},
    )

    def __post_init__(self):
        self.write_config_to_file()
        
    def to_dict(self):
        if not hasattr(self, "__flat_config__"):
            logger.warning("Flat config not found. Returning the original config.")
            return self.__dict__

        return self.__flat_config__.__dict__

    def write_config_to_file(self):
        config_dict = dataclass_to_dict(self)
        with open(f"{self.metrics_config.output_dir}/config.json", "w") as f:
            json.dump(config_dict, f, indent=4)
    
    
