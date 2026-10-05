"""HYDRA configuration translator and subprocess bridge for CHIPSIM."""

import copy
import csv
from dataclasses import dataclass
import inspect
import json
import os
from pathlib import Path
from pprint import pformat
import re
import shutil
import subprocess
from typing import Any

import yaml

import Sim.common as common
import Sim.config.utils as utils
from Sim.config.model_config import BaseBlockConfig
from Sim.config.sys_config import ChipsimConfig, HPSim_Config
from Sim.entities.chips_network import chip_graph
from Sim.entities.comp_sys import comp_sys
from Sim.entities.mem_sys import mem_sys
from Sim.entities.static_mapper import static_mapper
from Sim.metrics.monitor import analytics
from analytic_profile.Base_Accmodel import BaseAccModel
from Sim.backends.BaseSimulationBackend import BaseSimulationBackend


@dataclass(frozen=True)
class ChipsimTranslation:
    """Files and metadata generated from one HYDRA configuration."""

    input_dir: Path
    generated_config: Path
    chiplet_specs: Path
    weight_placement: Path
    execution_mapping: Path
    manifest: Path
    model_name: str
    block_name: str
    sequence_length: int
    mesh_rows: int
    link_width_bits: int


@dataclass(frozen=True)
class HydraSystemSnapshot:
    """Placed HYDRA package with its loaded weights and static task mapping."""

    graph: chip_graph
    memory_system: mem_sys
    compute_system: comp_sys
    task_mapping: dict[str, dict[int, int]]
    hbm_groups: tuple[list[int], list[int]]


@dataclass(frozen=True)
class ChipsimHardwareTranslation:
    """CHIPSIM hardware files and stable cross-simulator identifiers."""

    adjacency_file: Path
    mapping_file: Path
    specs_file: Path
    type_mapping: dict[str, str]
    hydra_to_chipsim_id: dict[int, int]


class ChipsimBackendError(RuntimeError):
    """Raised when HYDRA cannot translate or execute a CHIPSIM run."""


class ChipsimBackend(BaseSimulationBackend):
    """Translate ``HPSim_Config`` and launch the CHIPSIM co-simulator."""
    display_name = 'CHIPSIM (snapshot)'


    @staticmethod
    def get_name() -> str:
        return "chipsim"

    def __init__(
        self,
        config: HPSim_Config,
        repo_root: str | Path | None = None,
    ) -> None:
        self.config: HPSim_Config = config
        self.backend_config: ChipsimConfig = config.chipsim_config
        self.repo_root: Path = Path(
            repo_root or Path(__file__).resolve().parents[2]
        )
        self.chipsim_root: Path = self._repo_path(
            self.backend_config.submodule_dir
        )
        self.python_executable: Path = self._python_path(
            self.backend_config.python_executable
        )
        self.requirements_file: Path = self._chipsim_path(
            self.backend_config.requirements_file
        )
        self.runner: Path = (
            self.repo_root / "integrations" / "chipsim" / "run_hydra_experiment.py"
        )
        self.garnet_binary: Path = (
            self.chipsim_root
            / "integrations"
            / "gem5"
            / "build"
            / "Garnet_standalone"
            / "gem5.opt"
        )
        self.translation: ChipsimTranslation | None = None

    def _repo_path(self, value: str) -> Path:
        path: Path = Path(value).expanduser()
        return path if path.is_absolute() else self.repo_root / path

    def _python_path(self, value: str) -> Path:
        path: Path = Path(value).expanduser()
        if path.is_absolute() or path.parent != Path("."):
            return path if path.is_absolute() else self.repo_root / path
        resolved: str | None = shutil.which(value)
        return Path(resolved) if resolved else path

    def _chipsim_path(self, value: str) -> Path:
        path: Path = Path(value).expanduser()
        return path if path.is_absolute() else self.chipsim_root / path

    def command(self) -> list[str]:
        if self.translation is None:
            raise ChipsimBackendError(
                "CHIPSIM inputs have not been translated from HPSim_Config."
            )
        return [
            str(self.python_executable),
            str(self.runner),
            "--config",
            str(self.translation.generated_config),
            "--chiplet-specs",
            str(self.translation.chiplet_specs),
            "--mesh-rows",
            str(self.translation.mesh_rows),
            "--link-width-bits",
            str(self.translation.link_width_bits),
            "--router-latency",
            str(self.backend_config.router_latency_cycles),
            "--link-latency",
            str(self.backend_config.link_latency_cycles),
            "--vcs-per-vnet",
            str(self.backend_config.virtual_channels_per_vnet),
        ]

    def _python_requirements(self) -> list[tuple[str, str | None]]:
        requirements: list[tuple[str, str | None]] = []
        with self.requirements_file.open("r", encoding="utf-8") as stream:
            for raw_line in stream:
                line: str = raw_line.split("#", 1)[0].strip()
                if not line:
                    continue
                package: str = re.split(r"[<>=!~;\s\[]", line, maxsplit=1)[0]
                version: str | None = line.split("==", 1)[1] if "==" in line else None
                requirements.append((package, version))
        return requirements

    def _check_python_environment(self) -> list[str]:
        requirements: list[tuple[str, str | None]] = self._python_requirements()
        import_names: dict[str, str] = {"PyYAML": "yaml", "scons": "SCons"}
        probe: str = """
import importlib
from importlib import metadata
import json
import sys

requirements = json.loads(sys.argv[1])
import_names = json.loads(sys.argv[2])
report = {"missing": [], "mismatched": [], "broken": []}
for package, expected in requirements:
    try:
        installed = metadata.version(package)
    except metadata.PackageNotFoundError:
        report["missing"].append(package)
        continue
    if expected and installed != expected:
        report["mismatched"].append(
            f"{package} (installed {installed}, required {expected})"
        )
    module = import_names.get(package, package.replace("-", "_").lower())
    try:
        importlib.import_module(module)
    except Exception as exc:
        report["broken"].append(f"{package} ({type(exc).__name__}: {exc})")
print(json.dumps(report))
"""
        try:
            completed: subprocess.CompletedProcess[str] = subprocess.run(
                [
                    str(self.python_executable),
                    "-c",
                    probe,
                    json.dumps(requirements),
                    json.dumps(import_names),
                ],
                cwd=self.chipsim_root,
                env={**os.environ, "MPLBACKEND": "Agg"},
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return [f"cannot run the CHIPSIM Python environment ({exc})"]

        if completed.returncode != 0:
            detail: str = completed.stderr.strip() or completed.stdout.strip()
            return [f"CHIPSIM Python dependency probe failed: {detail}"]

        try:
            report: dict[str, list[str]] = json.loads(completed.stdout.strip())
        except json.JSONDecodeError:
            return [
                "CHIPSIM Python dependency probe returned invalid output: "
                f"{completed.stdout.strip()}"
            ]

        issues: list[str] = []
        if report["missing"]:
            issues.append("missing Python packages: " + ", ".join(report["missing"]))
        if report["mismatched"]:
            issues.append(
                "incompatible Python packages: " + ", ".join(report["mismatched"])
            )
        if report["broken"]:
            issues.append("packages that fail to import: " + ", ".join(report["broken"]))
        return issues

    def _configuration_issues(self) -> list[str]:
        chipsim: ChipsimConfig = self.backend_config
        issues: list[str] = []
        if chipsim.translated_block_type.lower() != "transformer":
            issues.append("the initial translator supports block type 'transformer' only")
        if chipsim.translated_num_blocks < 1:
            issues.append("translated_num_blocks must be at least one")
        if chipsim.compute_frequency_hz <= 0 or chipsim.network_frequency_hz <= 0:
            issues.append("compute and network frequencies must be positive")
        if not 0 < chipsim.compute_efficiency <= 1:
            issues.append("compute_efficiency must be in the interval (0, 1]")
        if chipsim.compute_memory_capacity_mb <= 0:
            issues.append("compute_memory_capacity_mb must be positive")
        if chipsim.hbm_access_latency_ns < 0:
            issues.append("hbm_access_latency_ns cannot be negative")
        if chipsim.hbm_ports <= 0:
            issues.append("hbm_ports must be positive")
        if (
            chipsim.router_latency_cycles <= 0
            or chipsim.link_latency_cycles <= 0
            or chipsim.virtual_channels_per_vnet <= 0
        ):
            issues.append("router latency, link latency, and virtual channels must be positive")
        if chipsim.activation_precision_bits <= 0 or chipsim.weight_precision_bits <= 0:
            issues.append("activation and weight precisions must be positive")
        if chipsim.garnet_sim_cycles <= 0 or chipsim.garnet_ticks_per_cycle <= 0:
            issues.append("Garnet cycles and ticks per cycle must be positive")
        if chipsim.timeout_seconds < 0:
            issues.append("timeout_seconds cannot be negative")
        if chipsim.communication_method not in {"pipelined", "non-pipelined"}:
            issues.append("communication_method must be pipelined or non-pipelined")
        return issues

    def validate_environment(self) -> None:
        simulate_entry: Path = self.chipsim_root / "simulate.py"
        issues: list[str] = self._configuration_issues()
        if not simulate_entry.is_file():
            issues.append(
                f"CHIPSIM submodule is missing at {self.chipsim_root}; "
                "run 'git submodule update --init --recursive'."
            )
        if not self.runner.is_file():
            issues.append(f"HYDRA CHIPSIM runner is missing: {self.runner}")
        if not self.python_executable.is_file() or not os.access(
            self.python_executable, os.X_OK
        ):
            issues.append(
                f"CHIPSIM Python is missing at {self.python_executable}; "
                "run 'bash scripts/setup_chipsim.sh'."
            )
        if not self.requirements_file.is_file() and simulate_entry.is_file():
            issues.append(
                f"CHIPSIM requirements file is missing: {self.requirements_file}; "
                "update the submodule and rerun 'bash scripts/setup_chipsim.sh'."
            )
        if self.python_executable.is_file() and self.requirements_file.is_file():
            dependency_issues: list[str] = self._check_python_environment()
            issues.extend(
                f"{issue}; run 'bash scripts/setup_chipsim.sh'."
                for issue in dependency_issues
            )
        if not self.garnet_binary.is_file() or not os.access(
            self.garnet_binary, os.X_OK
        ):
            issues.append(
                f"Garnet is not built at {self.garnet_binary}; "
                "run 'bash scripts/setup_chipsim.sh'."
            )

        if issues:
            details: str = "\n".join(f"  - {issue}" for issue in issues)
            raise ChipsimBackendError(
                "CHIPSIM backend preflight failed:\n"
                f"{details}\n"
                "See integrations/chipsim/README.md for installation guidance."
            )

    def _configure_placement_runtime(self) -> None:
        config: HPSim_Config = self.config
        common.logic_lib = config.chips_config.chips_lib.logics
        common.mem_lib = config.chips_config.chips_lib.mem_lib
        common.component_lib = config.chips_config.chips_lib.acc_components
        common.mapping_strategy = config.mapping_config.mapping_strategy
        common.task_parallelism = config.mapping_config.task_parallelism
        common.local_scheduler = config.cluster_config.local_scheduler
        common.ByteperParam = config.workload_config.bytes_per_param
        common.batch_size = config.cluster_config.batch_size

    def _system_snapshot(self, input_dir: Path) -> HydraSystemSnapshot:
        """Run the same placement, loading, and mapping sequence as Simulator."""
        self._configure_placement_runtime()
        common.output_folder = str(input_dir)
        graph: chip_graph = copy.deepcopy(self.config.placmt_config.chip_graph)
        placer_class: type = self.config.placmt_config.placer_inst
        placer: Any = placer_class(
            chiplet_num=graph.num_nodes,
            chiplet_alloc=self.config.placmt_config.chiplet_alloc,
        )
        placement_args: dict[str, Any] = {
            "chip_graph": graph,
            "model_config": self.config.workload_config.model_config,
            "mem_size": self.config.chips_config.mem_sizes[0],
            "output_folder": str(input_dir),
        }
        parameters: dict[str, inspect.Parameter] = dict(
            inspect.signature(placer.make_chiplet_placement).parameters
        )
        if "logging" in parameters:
            placement_args["logging"] = False
        placement_result: tuple[list[int], list[int]] = (
            placer.make_chiplet_placement(**placement_args)
        )
        group_hbm_m: list[int]
        group_hbm_a: list[int]
        group_hbm_m, group_hbm_a = placement_result

        task_parallelism: str = self.config.mapping_config.task_parallelism
        if task_parallelism == "pipeline":
            load_label: str = self.config.placmt_config.placer_label
        elif task_parallelism == "tensor":
            load_label = "tp"
            try:
                utils.used_bws = utils.used_bws_dict_tensor[utils.NoI_bw]
            except KeyError as exc:
                raise ChipsimBackendError(
                    "Tensor-parallel CHIPSIM export has no bandwidth profile for "
                    f"{utils.NoI_bw} GB/s."
                ) from exc
        else:
            raise ChipsimBackendError(
                f"Unsupported HYDRA task parallelism: {task_parallelism}"
            )

        memory_system: mem_sys = mem_sys(graph)
        memory_system.load_model(
            label=load_label,
            mod_config=self.config.workload_config.model_config,
            group_HBM_M=group_hbm_m,
            group_HBM_A=group_hbm_a,
        )
        compute_system: comp_sys = comp_sys(graph)
        mapper: static_mapper = static_mapper()
        task_mapping: dict[str, dict[int, int]]
        task_mapping, _ = mapper.generate_mapping(
            self.config.workload_config.model_config,
            compute_system,
            memory_system,
        )
        return HydraSystemSnapshot(
            graph=graph,
            memory_system=memory_system,
            compute_system=compute_system,
            task_mapping=task_mapping,
            hbm_groups=(list(group_hbm_m), list(group_hbm_a)),
        )

    def _hydra_chiplet_name(self, chiplet_type: Any) -> str:
        reverse_types: dict[int, str] = {
            int(identifier): name
            for name, identifier in utils.chiplet_types_dict.items()
        }
        if isinstance(chiplet_type, str) and chiplet_type in utils.chiplet_types_dict:
            return chiplet_type
        try:
            return reverse_types[int(chiplet_type)]
        except (KeyError, TypeError, ValueError) as exc:
            raise ChipsimBackendError(
                f"Cannot translate HYDRA chiplet type {chiplet_type!r}."
            ) from exc

    def _chiplet_spec(
        self,
        hydra_type: str,
        chipsim_name: str,
        minimum_weight_capacity: int,
    ) -> dict[str, Any]:
        logic: dict[str, Any] = self.config.chips_config.chips_lib.logics.get(
            hydra_type, {}
        )
        cores: int = int(logic.get("core", 1))
        macs_per_core_cycle: int = int(
            logic.get(
                "core_MAC", self.backend_config.fallback_macs_per_core_cycle
            )
        )
        macs_per_second: float = (
            cores
            * macs_per_core_cycle
            * self.backend_config.compute_frequency_hz
            * self.backend_config.compute_efficiency
        )
        configured_weight_capacity: int = int(
            self.backend_config.compute_memory_capacity_mb
            * 1024
            * 1024
            * 8
            / self.backend_config.weight_precision_bits
        )
        specification: dict[str, Any] = {
            "name": chipsim_name,
            "type": "CMOS",
            "macs_per_second": macs_per_second,
            "energy_per_mac": self.backend_config.compute_energy_per_mac_fj,
            "crossbar_rows": 0,
            "crossbar_columns": 0,
            "bits_per_cell": 0,
            "dac_precision_bits": 0,
            "bits_per_weight": self.backend_config.weight_precision_bits,
            "input_precision_bits": self.backend_config.activation_precision_bits,
            "tiles_per_chiplet": 0,
            "crossbars_per_tile": 0,
            "total_memory_weights": max(
                configured_weight_capacity,
                minimum_weight_capacity,
            ),
        }
        if self.backend_config.use_hydra_compute_backend:
            specification["compute_backend"] = "HYDRA"
        return specification

    def _memory_spec(
        self,
        hydra_id: int,
        attributes: dict[str, Any],
        chipsim_name: str,
    ) -> dict[str, Any]:
        capacity_bytes: int = int(float(attributes["dram(g)"]) * 1024**3)
        bandwidth_bytes_per_second: float = (
            float(attributes["bandwidth"]) * 1e9
        )
        return {
            "name": chipsim_name,
            "type": "IO",
            "memory_backend": "HYDRA_HBM",
            "hydra_chiplet_id": hydra_id,
            "capacity_bytes": capacity_bytes,
            "bandwidth_bytes_per_second": bandwidth_bytes_per_second,
            "access_latency_ns": self.backend_config.hbm_access_latency_ns,
            "ports": self.backend_config.hbm_ports,
            "energy_per_mac": 0,
            "crossbar_rows": 0,
            "crossbar_columns": 0,
            "bits_per_cell": 0,
            "bits_per_weight": self.backend_config.weight_precision_bits,
            "tiles_per_chiplet": 0,
            "crossbars_per_tile": 0,
        }

    def _translate_hardware(
        self,
        graph: chip_graph,
        input_dir: Path,
        minimum_weight_capacity: int,
    ) -> ChipsimHardwareTranslation:
        if graph.is_2d_grid():
            nodes: list[tuple[Any, dict[str, Any]]] = sorted(
                graph.graph.nodes(data=True),
                key=lambda item: (int(item[0][1]), int(item[0][0])),
            )
        else:
            nodes = sorted(
                graph.graph.nodes(data=True), key=lambda item: int(item[1]["id"])
            )
        node_index: dict[Any, int] = {
            node: index for index, (node, _) in enumerate(nodes)
        }
        hydra_to_chipsim_id: dict[int, int] = {
            int(attributes["id"]): index
            for index, (_, attributes) in enumerate(nodes, start=1)
        }
        node_count: int = len(nodes)
        adjacency: list[list[int]] = [
            [0 for _ in range(node_count)] for _ in range(node_count)
        ]
        for source, destination in graph.graph.edges():
            source_index: int = node_index[source]
            destination_index: int = node_index[destination]
            adjacency[source_index][destination_index] = 1
            adjacency[destination_index][source_index] = 1

        adjacency_file: Path = input_dir / "hydra_topology.csv"
        with adjacency_file.open("w", encoding="utf-8", newline="") as stream:
            writer: csv.writer = csv.writer(stream, delimiter=" ")
            writer.writerows(adjacency)

        memory_types: set[str] = set(utils.chiplets_division["Memory"])
        type_mapping: dict[str, str] = {}
        chipsim_mapping: dict[int, str] = {}
        chiplet_specs: list[dict[str, Any]] = []
        for chipsim_id, (_, attributes) in enumerate(nodes, start=1):
            hydra_id: int = int(attributes["id"])
            hydra_type: str = self._hydra_chiplet_name(attributes["chiplet_type"])
            if hydra_type in memory_types:
                chipsim_type: str = f"HYDRA_HBM_{hydra_id}"
                chiplet_specs.append(
                    self._memory_spec(hydra_id, attributes, chipsim_type)
                )
                type_mapping[f"{hydra_type}:{hydra_id}"] = chipsim_type
            else:
                chipsim_type = "HYDRA_" + re.sub(
                    r"[^A-Za-z0-9_]", "_", hydra_type
                ).upper()
                if chipsim_type not in {spec["name"] for spec in chiplet_specs}:
                    chiplet_specs.append(
                        self._chiplet_spec(
                            hydra_type,
                            chipsim_type,
                            minimum_weight_capacity,
                        )
                    )
                type_mapping[hydra_type] = chipsim_type
            chipsim_mapping[chipsim_id] = chipsim_type

        mapping_file: Path = input_dir / "hydra_chiplet_mapping.yaml"
        with mapping_file.open("w", encoding="utf-8") as stream:
            yaml.safe_dump(chipsim_mapping, stream, sort_keys=True)

        specs_file: Path = input_dir / "hydra_chiplet_specs.json"
        with specs_file.open("w", encoding="utf-8") as stream:
            json.dump(chiplet_specs, stream, indent=2)
        return ChipsimHardwareTranslation(
            adjacency_file=adjacency_file,
            mapping_file=mapping_file,
            specs_file=specs_file,
            type_mapping=type_mapping,
            hydra_to_chipsim_id=hydra_to_chipsim_id,
        )

    def _select_block(self) -> BaseBlockConfig:
        model: Any = self.config.workload_config.model_config
        candidates: list[BaseBlockConfig] = []
        block_config: BaseBlockConfig | None = getattr(model, "block_config", None)
        if block_config is not None:
            candidates.append(block_config)
        candidates.extend(getattr(model, "hybrid_blocks", []) or [])

        target: str = self.backend_config.translated_block_type.lower()
        for block in candidates:
            block_name: str = str(getattr(block, "type_name", "")).lower()
            if target in block_name or (
                target == "transformer" and "attention" in block_name
            ):
                return block

        available: list[str] = [
            str(getattr(block, "type_name", type(block).__name__))
            for block in candidates
        ]
        raise ChipsimBackendError(
            f"Model '{self.config.workload_config.model}' has no {target} block. "
            f"Available blocks: {available}"
        )

    def _selected_block_ids(self, block: BaseBlockConfig) -> list[int]:
        model: Any = self.config.workload_config.model_config
        hybrid_blocks: list[BaseBlockConfig] = list(
            getattr(model, "hybrid_blocks", []) or []
        )
        if not hybrid_blocks:
            hybrid_blocks = [block]
        try:
            block_type_index: int = next(
                index
                for index, candidate in enumerate(hybrid_blocks)
                if candidate is block or candidate == block
            )
        except StopIteration as exc:
            raise ChipsimBackendError(
                f"Selected block '{block.type_name}' is absent from the model sequence."
            ) from exc

        sequence: list[int] = list(getattr(model, "block_type_sequence", []) or [])
        if not sequence:
            sequence = [block_type_index] * int(getattr(model, "num_layers", 1))
        matching_ids: list[int] = [
            global_id
            for global_id, candidate_index in enumerate(sequence)
            if int(candidate_index) == block_type_index
        ]
        requested: int = self.backend_config.translated_num_blocks
        if len(matching_ids) < requested:
            raise ChipsimBackendError(
                f"Requested {requested} '{block.type_name}' blocks, but the HYDRA "
                f"model contains {len(matching_ids)}."
            )
        return matching_ids[:requested]

    def _sequence_length(self, block: BaseBlockConfig) -> int:
        request_config: Any = self.config.workload_config.request_generator_config
        length_config: Any = request_config.trace_length_generator_config
        trace_file: Path = self._repo_path(length_config.trace_file)
        try:
            with trace_file.open("r", encoding="utf-8", newline="") as stream:
                first_request: dict[str, str] | None = next(csv.DictReader(stream), None)
        except OSError as exc:
            raise ChipsimBackendError(
                f"Cannot read HYDRA request trace {trace_file}: {exc}"
            ) from exc
        if first_request is None or "num_prefill_tokens" not in first_request:
            raise ChipsimBackendError(
                f"HYDRA request trace has no prefill length: {trace_file}"
            )
        sequence_length: int = max(
            1,
            int(
                float(first_request["num_prefill_tokens"])
                * float(length_config.prefill_scale_factor)
            ),
        )
        return min(
            sequence_length,
            int(length_config.max_tokens),
            int(block.max_position_embeddings),
        )

    def _linear_layer(
        self,
        input_size: int,
        output_size: int,
        sequence_length: int,
        operation: str,
    ) -> dict[str, Any]:
        return {
            "description": "Linear layer",
            "hydra_operation": operation,
            "parameters": {
                "batch_size": self.config.cluster_config.batch_size,
                "input_channels": input_size,
                "output_channels": output_size,
                "output_height": sequence_length,
                "input_precision": self.backend_config.activation_precision_bits,
                "output_precision": self.backend_config.activation_precision_bits,
                "weight_precision": self.backend_config.weight_precision_bits,
            },
        }

    def _attention_layers(
        self,
        operation: Any,
        sequence_length: int,
    ) -> list[dict[str, Any]]:
        embedding_dim: int = int(operation.embedding_dim)
        q_heads: int = int(operation.q_heads)
        kv_heads: int = int(operation.kv_heads)
        head_dim: int = embedding_dim // q_heads
        qkv_output: int = embedding_dim + 2 * kv_heads * head_dim
        return [
            self._linear_layer(
                embedding_dim, qkv_output, sequence_length, "MHA_QKV"
            ),
            {
                "description": "SelfAttention QK layer",
                "hydra_operation": "MHA_QK",
                "parameters": {
                    "input_channels": embedding_dim,
                    "output_channels": sequence_length,
                    "output_height": sequence_length,
                },
            },
            {
                "description": "SelfAttention AttnV layer",
                "hydra_operation": "MHA_AttnV",
                "parameters": {
                    "input_channels": sequence_length,
                    "output_channels": embedding_dim,
                    "output_height": sequence_length,
                },
            },
            self._linear_layer(
                embedding_dim, embedding_dim, sequence_length, "MHA_Output"
            ),
        ]

    def _translate_transformer_block(
        self,
        block: BaseBlockConfig,
        sequence_length: int,
        block_ids: list[int],
    ) -> tuple[dict[int, dict[str, Any]], list[str]]:
        layers: list[dict[str, Any]] = []
        skipped_operations: list[str] = []
        for block_id in block_ids:
            for operation_group in block.layer_configs:
                for operation in operation_group:
                    if all(
                        hasattr(operation, attribute)
                        for attribute in ("embedding_dim", "q_heads", "kv_heads")
                    ):
                        attention_layers: list[dict[str, Any]] = (
                            self._attention_layers(operation, sequence_length)
                        )
                        for layer in attention_layers:
                            layer["hydra_block_id"] = block_id
                        layers.extend(attention_layers)
                    elif all(
                        hasattr(operation, attribute)
                        for attribute in ("f_in", "f_out")
                    ):
                        linear_layer: dict[str, Any] = self._linear_layer(
                            int(operation.f_in),
                            int(operation.f_out),
                            sequence_length,
                            type(operation).__name__,
                        )
                        linear_layer["hydra_block_id"] = block_id
                        layers.append(linear_layer)
                    else:
                        skipped_operations.append(type(operation).__name__)

        if not layers:
            raise ChipsimBackendError(
                f"Transformer block '{block.type_name}' produced no CHIPSIM layers."
            )
        translated: dict[int, dict[str, Any]] = {}
        for layer_index, layer in enumerate(layers):
            layer["receiving_layers"] = (
                [layer_index + 1] if layer_index + 1 < len(layers) else [-1]
            )
            translated[layer_index] = layer
        return translated, skipped_operations

    def _block_for_global_id(self, block_id: int) -> BaseBlockConfig:
        model: Any = self.config.workload_config.model_config
        sequence: list[int] = list(getattr(model, "block_type_sequence", []) or [])
        hybrid_blocks: list[BaseBlockConfig] = list(
            getattr(model, "hybrid_blocks", []) or []
        )
        if sequence and hybrid_blocks:
            return hybrid_blocks[int(sequence[block_id])]
        block: BaseBlockConfig | None = getattr(model, "block_config", None)
        if block is None:
            if not hybrid_blocks:
                raise ChipsimBackendError("HYDRA model contains no block definition.")
            block = hybrid_blocks[0]
        return block

    def _profile_block(
        self,
        block: BaseBlockConfig,
        hydra_chiplet_type: int,
        sequence_length: int,
    ) -> tuple[float, float]:
        logic_name: str = utils.chiplet_types_list[hydra_chiplet_type]
        analytic_model: BaseAccModel = BaseAccModel.create_from_name(logic_name)
        total_latency_ns: float = 0.0
        total_energy_fj: float = 0.0
        for operation_group, operation_names in zip(
            block.layer_configs,
            block.layers,
        ):
            if len(operation_group) != len(operation_names):
                raise ChipsimBackendError(
                    f"Invalid layer description in block '{block.type_name}'."
                )
            for operation, operation_name in zip(
                operation_group,
                operation_names,
            ):
                kernel_name: str = operation_name
                if kernel_name in {"MHA", "SSM"}:
                    kernel_name += "_p"
                function_args: dict[str, Any] = analytic_model.config_params(
                    **vars(operation)
                )
                function_args["bs"] = sequence_length
                function_args["batch_size"] = self.config.cluster_config.batch_size
                function_args["L_seq"] = sequence_length
                if "in_size" in function_args:
                    function_args["in_size"] *= sequence_length
                function_args["logic_name"] = logic_name
                # CHIPSIM models HBM and NoI transfer separately for this path.
                function_args["ext_bw"] = 1e12
                result: analytics = analytic_model.get_kernel(
                    kernel_name,
                    **function_args,
                )
                latency_ns: float = float(result.total_latency)
                total_latency_ns += latency_ns
                total_energy_fj += float(result.power) * latency_ns * 1000.0
        return total_latency_ns / 1000.0, total_energy_fj

    def _layer_macs(self, layer: dict[str, Any]) -> int:
        parameters: dict[str, Any] = layer["parameters"]
        return int(
            parameters["input_channels"]
            * parameters["output_channels"]
            * parameters.get("output_height", 1)
        )

    def _attach_compute_profiles(
        self,
        layers: dict[int, dict[str, Any]],
        snapshot: HydraSystemSnapshot,
        hardware: ChipsimHardwareTranslation,
        sequence_length: int,
    ) -> None:
        if not self.backend_config.use_hydra_compute_backend:
            return

        layers_by_block: dict[int, list[dict[str, Any]]] = {}
        for layer in layers.values():
            block_id: int = int(layer["hydra_block_id"])
            layers_by_block.setdefault(block_id, []).append(layer)

        for block_id, block_layers in layers_by_block.items():
            hydra_compute_id: int = int(snapshot.task_mapping["P"][block_id])
            compute_chiplet: Any = snapshot.compute_system.find_chiplet_by_id(
                hydra_compute_id
            )
            hydra_type: int = int(compute_chiplet.chiplet_type)
            hydra_type_name: str = self._hydra_chiplet_name(hydra_type)
            chipsim_type: str = hardware.type_mapping[hydra_type_name]
            block: BaseBlockConfig = self._block_for_global_id(block_id)
            latency_us: float
            energy_fj: float
            latency_us, energy_fj = self._profile_block(
                block,
                hydra_type,
                sequence_length,
            )
            total_macs: int = sum(self._layer_macs(layer) for layer in block_layers)
            for layer in block_layers:
                fraction: float = self._layer_macs(layer) / max(total_macs, 1)
                layer["compute_backend"] = "HYDRA"
                layer["hydra_compute_profiles"] = {
                    chipsim_type: {
                        "latency_us": latency_us * fraction,
                        "energy_fj": energy_fj * fraction,
                        "frequency_hz": self.backend_config.compute_frequency_hz,
                    }
                }

    def _write_weight_placement(
        self,
        snapshot: HydraSystemSnapshot,
        hardware: ChipsimHardwareTranslation,
        layers: dict[int, dict[str, Any]],
        input_dir: Path,
    ) -> Path:
        model: Any = self.config.workload_config.model_config
        blocks: dict[str, dict[str, Any]] = {}
        sequence: list[int] = list(getattr(model, "block_type_sequence", []) or [])
        if not sequence:
            sequence = [0] * int(getattr(model, "num_layers", 1))
        for block_id in range(len(sequence)):
            block: BaseBlockConfig = self._block_for_global_id(block_id)
            weight_bytes: int = int(
                float(block.parameter_count)
                * float(self.config.workload_config.bytes_per_param)
            )
            allocation: int | list[int] = snapshot.memory_system.blocks_alloc[
                block_id
            ]
            memory_ids: list[int] = (
                [int(allocation)]
                if isinstance(allocation, int)
                else [int(memory_id) for memory_id in allocation]
            )
            shard_count: int = len(memory_ids)
            shards: list[dict[str, Any]] = []
            bytes_remaining: int = weight_bytes
            for shard_index, hydra_memory_id in enumerate(memory_ids):
                shard_bytes: int = (
                    bytes_remaining
                    if shard_index + 1 == shard_count
                    else weight_bytes // shard_count
                )
                bytes_remaining -= shard_bytes
                shards.append(
                    {
                        "hydra_memory_chiplet_id": hydra_memory_id,
                        "chipsim_memory_chiplet_id": hardware.hydra_to_chipsim_id[
                            hydra_memory_id
                        ],
                        "fraction": shard_bytes / max(weight_bytes, 1),
                        "weight_bytes": shard_bytes,
                    }
                )
            blocks[str(block_id)] = {
                "block_type": block.type_name,
                "weight_bytes": weight_bytes,
                "shards": shards,
            }

        memories: list[dict[str, Any]] = []
        for memory_chiplet in snapshot.memory_system._mem_chiplets:
            hydra_memory_id: int = int(memory_chiplet.chiplet_id)
            memories.append(
                {
                    "hydra_chiplet_id": hydra_memory_id,
                    "chipsim_chiplet_id": hardware.hydra_to_chipsim_id[
                        hydra_memory_id
                    ],
                    "capacity_bytes": int(memory_chiplet.dram_budget * 1024**2),
                    "bandwidth_bytes_per_second": float(
                        memory_chiplet.BW_budget
                    )
                    * 1e9,
                    "access_latency_ns": self.backend_config.hbm_access_latency_ns,
                    "ports": self.backend_config.hbm_ports,
                }
            )

        placement: dict[str, Any] = {
            "schema_version": 1,
            "model": self.config.workload_config.model,
            "bytes_per_parameter": self.config.workload_config.bytes_per_param,
            "hydra_to_chipsim_id": {
                str(hydra_id): chipsim_id
                for hydra_id, chipsim_id in hardware.hydra_to_chipsim_id.items()
            },
            "memories": memories,
            "layer_to_block": {
                str(layer_id): int(layer["hydra_block_id"])
                for layer_id, layer in layers.items()
            },
            "blocks": blocks,
        }
        placement_file: Path = input_dir / "hydra_weight_placement.json"
        with placement_file.open("w", encoding="utf-8") as stream:
            json.dump(placement, stream, indent=2)
        return placement_file

    def _write_execution_mapping(
        self,
        snapshot: HydraSystemSnapshot,
        hardware: ChipsimHardwareTranslation,
        layers: dict[int, dict[str, Any]],
        input_dir: Path,
    ) -> Path:
        layer_mapping: dict[str, int] = {}
        for layer_id, layer in layers.items():
            block_id: int = int(layer["hydra_block_id"])
            hydra_compute_id: int = int(snapshot.task_mapping["P"][block_id])
            layer_mapping[str(layer_id)] = hardware.hydra_to_chipsim_id[
                hydra_compute_id
            ]
        allowed_chiplets: list[int] = sorted(set(layer_mapping.values()))
        execution: dict[str, Any] = {
            "schema_version": 1,
            "model_index": 1,
            "stage": "prefill",
            "layer_mapping": layer_mapping,
            "allowed_chiplet_ids": allowed_chiplets,
            "preferred_start_chiplet_id": allowed_chiplets[0],
            "hydra_task_mapping": {
                stage: {
                    str(block_id): hardware.hydra_to_chipsim_id[int(chiplet_id)]
                    for block_id, chiplet_id in mapping.items()
                }
                for stage, mapping in snapshot.task_mapping.items()
            },
        }
        execution_file: Path = input_dir / "hydra_execution_mapping.json"
        with execution_file.open("w", encoding="utf-8") as stream:
            json.dump(execution, stream, indent=2)
        return execution_file

    def _noi_link_width_bits(self) -> int:
        bandwidth_bytes_per_second: float = (
            self.config.chips_config.D2D_NoI_bw * 1e9
        )
        bytes_per_cycle: float = (
            bandwidth_bytes_per_second
            / self.backend_config.network_frequency_hz
        )
        return max(8, int(round(bytes_per_cycle)) * 8)

    def translate_config(self) -> ChipsimTranslation:
        output_dir: Path = Path(self.config.metrics_config.output_dir).resolve()
        input_dir: Path = output_dir / "chipsim_inputs"
        input_dir.mkdir(parents=True, exist_ok=True)

        block: BaseBlockConfig = self._select_block()
        block_ids: list[int] = self._selected_block_ids(block)
        sequence_length: int = self._sequence_length(block)
        translated_block: tuple[dict[int, dict[str, Any]], list[str]] = (
            self._translate_transformer_block(
                block,
                sequence_length,
                block_ids,
            )
        )
        layers: dict[int, dict[str, Any]]
        skipped_operations: list[str]
        layers, skipped_operations = translated_block
        model_name: str = "hydra_" + re.sub(
            r"[^A-Za-z0-9_]", "_", self.config.workload_config.model
        ).lower() + "_transformer_block"

        snapshot: HydraSystemSnapshot = self._system_snapshot(input_dir)
        minimum_weight_capacity: int = sum(
            int(self._block_for_global_id(block_id).parameter_count)
            for block_id in block_ids
        )
        hardware: ChipsimHardwareTranslation = self._translate_hardware(
            snapshot.graph,
            input_dir,
            minimum_weight_capacity,
        )
        self._attach_compute_profiles(
            layers,
            snapshot,
            hardware,
            sequence_length,
        )
        weight_placement_file: Path = self._write_weight_placement(
            snapshot,
            hardware,
            layers,
            input_dir,
        )
        execution_mapping_file: Path = self._write_execution_mapping(
            snapshot,
            hardware,
            layers,
            input_dir,
        )

        model_file: Path = input_dir / "hydra_model_definitions.py"
        model_definitions: dict[str, Any] = {model_name: {"layers": layers}}
        model_file.write_text(
            "# Generated from HPSim_Config.\nMODEL_DEFINITIONS = "
            + pformat(model_definitions, sort_dicts=True)
            + "\n",
            encoding="utf-8",
        )

        workload_file: Path = input_dir / "hydra_workload.csv"
        with workload_file.open("w", encoding="utf-8", newline="") as stream:
            writer: csv.writer = csv.writer(stream)
            writer.writerow(["net_idx", "inject_time_us", "network", "num_inputs"])
            writer.writerow([1, 0, model_name, 1])

        experiment: dict[str, Any] = {
            "simulation": {
                "input_files": {
                    "workload": str(workload_file.resolve()),
                    "adj_matrix": str(hardware.adjacency_file.resolve()),
                    "chiplet_mapping": str(hardware.mapping_file.resolve()),
                    "model_defs": str(model_file.resolve()),
                    "hydra_weight_placement": str(
                        weight_placement_file.resolve()
                    ),
                    "hydra_execution_mapping": str(
                        execution_mapping_file.resolve()
                    ),
                },
                "core_settings": {
                    "clear_cache": False,
                    "comm_simulator": "Garnet",
                    "comm_method": self.backend_config.communication_method,
                    "enable_dsent": self.backend_config.enable_dsent,
                    "enable_comm_cache": self.backend_config.enable_comm_cache,
                    "blocking_age_threshold": 10,
                    "weight_stationary": True,
                    "weight_loading_strategy": "all_at_once",
                },
                "hardware_parameters": {
                    "bits_per_activation": self.backend_config.activation_precision_bits,
                    "bits_per_packet": self._noi_link_width_bits(),
                    "network_operation_frequency_hz": int(
                        self.backend_config.network_frequency_hz
                    ),
                },
                "gem5_parameters": {
                    "gem5_sim_cycles": self.backend_config.garnet_sim_cycles,
                    "gem5_injection_rate": 0.0,
                    "gem5_ticks_per_cycle": self.backend_config.garnet_ticks_per_cycle,
                    "gem5_deadlock_threshold": None,
                },
                "dsent_parameters": {"dsent_tech_node": "32"},
            },
            "post_processing": {
                "warmup_period_us": 0.0,
                "cooldown_period_us": 0.0,
                "run_wkld_agg_comm": False,
                "run_ind_comm": False,
                "run_net_agg_comm": False,
                "generate_plots": False,
                "generate_visualizations": False,
            },
        }
        experiment_file: Path = input_dir / "hydra_chipsim_experiment.yaml"
        with experiment_file.open("w", encoding="utf-8") as stream:
            yaml.safe_dump(experiment, stream, sort_keys=False)

        translation_manifest: dict[str, Any] = {
            "hydra_model": self.config.workload_config.model,
            "hydra_block": block.type_name,
            "hydra_block_ids": block_ids,
            "translated_num_blocks": self.backend_config.translated_num_blocks,
            "sequence_length": sequence_length,
            "batch_size": self.config.cluster_config.batch_size,
            "placer": self.config.placmt_config.placer_label,
            "D2D_NoI_bw_GBps": self.config.chips_config.D2D_NoI_bw,
            "HBM_IO_bw_GBps": self.config.chips_config.HBM_IO_bw,
            "noi_link_width_bits": self._noi_link_width_bits(),
            "chipsim_packet_size_bits": self._noi_link_width_bits(),
            "chiplet_type_mapping": hardware.type_mapping,
            "hydra_to_chipsim_id": hardware.hydra_to_chipsim_id,
            "hbm_groups": {
                "mamba": snapshot.hbm_groups[0],
                "attention": snapshot.hbm_groups[1],
            },
            "compute_backend": (
                "HYDRA"
                if self.backend_config.use_hydra_compute_backend
                else "CHIPSIM"
            ),
            "skipped_non_matrix_operations": sorted(set(skipped_operations)),
            "generated_files": {
                "experiment": str(experiment_file),
                "workload": str(workload_file),
                "model": str(model_file),
                "topology": str(hardware.adjacency_file),
                "mapping": str(hardware.mapping_file),
                "chiplet_specs": str(hardware.specs_file),
                "weight_placement": str(weight_placement_file),
                "execution_mapping": str(execution_mapping_file),
            },
        }
        manifest_file: Path = input_dir / "translation_manifest.json"
        with manifest_file.open("w", encoding="utf-8") as stream:
            json.dump(translation_manifest, stream, indent=2)

        self.translation = ChipsimTranslation(
            input_dir=input_dir,
            generated_config=experiment_file,
            chiplet_specs=hardware.specs_file,
            weight_placement=weight_placement_file,
            execution_mapping=execution_mapping_file,
            manifest=manifest_file,
            model_name=model_name,
            block_name=block.type_name,
            sequence_length=sequence_length,
            mesh_rows=snapshot.graph.intp_height,
            link_width_bits=self._noi_link_width_bits(),
        )
        return self.translation

    def _check_experiment(self, experiment: dict[str, Any]) -> list[str]:
        issues: list[str] = []
        simulation: Any = experiment.get("simulation")
        if not isinstance(simulation, dict):
            return ["generated experiment is missing the simulation section"]
        input_files: Any = simulation.get("input_files")
        if not isinstance(input_files, dict):
            return ["generated experiment is missing simulation.input_files"]
        for value in input_files.values():
            if not Path(value).is_file():
                issues.append(f"generated CHIPSIM input does not exist: {value}")
        return issues

    def validate_translation(self) -> None:
        if self.translation is None:
            raise ChipsimBackendError("HPSim_Config has not been translated.")
        try:
            with self.translation.generated_config.open(
                "r", encoding="utf-8"
            ) as stream:
                experiment: dict[str, Any] = yaml.safe_load(stream) or {}
        except (OSError, yaml.YAMLError) as exc:
            raise ChipsimBackendError(
                f"Cannot read generated CHIPSIM experiment: {exc}"
            ) from exc
        issues: list[str] = self._check_experiment(experiment)
        if not self.translation.chiplet_specs.is_file():
            issues.append(
                f"generated chiplet specifications do not exist: "
                f"{self.translation.chiplet_specs}"
            )
        if not self.translation.weight_placement.is_file():
            issues.append(
                "generated HYDRA weight placement does not exist: "
                f"{self.translation.weight_placement}"
            )
        if not self.translation.execution_mapping.is_file():
            issues.append(
                "generated HYDRA execution mapping does not exist: "
                f"{self.translation.execution_mapping}"
            )
        if issues:
            details: str = "\n".join(f"  - {issue}" for issue in issues)
            raise ChipsimBackendError(
                f"HYDRA-to-CHIPSIM translation validation failed:\n{details}"
            )

    def validate(self) -> None:
        self.validate_environment()
        if self.translation is not None:
            self.validate_translation()
        print("CHIPSIM preflight: dependencies and translated inputs are ready.")

    def _submodule_revision(self) -> str | None:
        result: subprocess.CompletedProcess[str] = subprocess.run(
            ["git", "-C", str(self.chipsim_root), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() if result.returncode == 0 else None

    def _latest_result(self) -> Path | None:
        results_root: Path = self.chipsim_root / "_results" / "raw_results"
        if not results_root.is_dir():
            return None
        candidates: list[Path] = [
            path for path in results_root.iterdir() if path.is_dir()
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda path: path.stat().st_mtime)

    def _write_manifest(self, returncode: int, result_dir: Path | None) -> None:
        if self.translation is None:
            raise ChipsimBackendError("Cannot record a run before translation.")
        output_dir: Path = Path(self.config.metrics_config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest: dict[str, Any] = {
            "simulator_backend": "chipsim",
            "chipsim_revision": self._submodule_revision(),
            "translation_manifest": str(self.translation.manifest),
            "generated_experiment": str(self.translation.generated_config),
            "command": self.command(),
            "returncode": returncode,
            "chipsim_result_dir": str(result_dir) if result_dir else None,
        }
        with (output_dir / "chipsim_launch.json").open(
            "w", encoding="utf-8"
        ) as stream:
            json.dump(manifest, stream, indent=2)

    def run(self) -> Path | None:
        self.validate_environment()
        translation: ChipsimTranslation = self.translate_config()
        self.validate_translation()
        print("CHIPSIM preflight: dependencies and translated inputs are ready.")
        print(f"Translated HYDRA block: {translation.block_name}")
        print(f"Translated CHIPSIM inputs: {translation.input_dir}")

        command: list[str] = self.command()
        timeout: int | None = self.backend_config.timeout_seconds or None
        environment: dict[str, str] = os.environ.copy()
        environment.setdefault("MPLBACKEND", "Agg")
        environment["PYTHONUNBUFFERED"] = "1"

        try:
            completed: subprocess.CompletedProcess[Any] = subprocess.run(
                command,
                cwd=self.chipsim_root,
                env=environment,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            self._write_manifest(124, self._latest_result())
            raise ChipsimBackendError(
                f"CHIPSIM exceeded the {timeout}-second timeout."
            ) from exc

        result_dir: Path | None = self._latest_result()
        self._write_manifest(completed.returncode, result_dir)
        if completed.returncode != 0:
            raise ChipsimBackendError(
                f"CHIPSIM exited with status {completed.returncode}."
            )
        if result_dir:
            print(f"CHIPSIM raw results: {result_dir}")
        return result_dir
