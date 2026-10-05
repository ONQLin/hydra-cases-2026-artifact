#!/usr/bin/env python3
"""Run a generated HYDRA experiment with per-run CHIPSIM chiplet specs."""

import argparse
import gzip
import json
from pathlib import Path
import sys
from typing import Any, Callable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--chiplet-specs", required=True)
    parser.add_argument("--mesh-rows", required=True, type=int)
    parser.add_argument("--link-width-bits", required=True, type=int)
    parser.add_argument("--router-latency", required=True, type=int)
    parser.add_argument("--link-latency", required=True, type=int)
    parser.add_argument("--vcs-per-vnet", required=True, type=int)
    return parser.parse_args()


def _rewrite_garnet_endpoint_ids(
    traffic_file: str,
    node_count: int,
) -> None:
    """Replace CHIPSIM's fixed 100-node directory offset."""
    text_path: Path = Path(traffic_file)
    lines: list[str] = text_path.read_text(encoding="utf-8").splitlines()
    translated: list[str] = []
    for line in lines:
        if not line or line.startswith("#"):
            translated.append(line)
            continue
        fields: list[str] = line.split()
        fields[3] = str(int(fields[4]) + node_count)
        translated.append(" ".join(fields))

    content: str = "\n".join(translated) + "\n"
    text_path.write_text(content, encoding="utf-8")
    with gzip.open(text_path.with_suffix(".gz"), "wt", encoding="utf-8") as stream:
        stream.write(content)


def main() -> int:
    args: argparse.Namespace = parse_args()
    chipsim_root: Path = Path.cwd()
    sys.path.insert(0, str(chipsim_root))

    from assets.chiplet_specs.chiplet_params import CHIPLET_TYPES

    with Path(args.chiplet_specs).open("r", encoding="utf-8") as stream:
        translated_specs: list[dict[str, Any]] = json.load(stream)
    CHIPLET_TYPES[:] = translated_specs

    from src.sim.communication_simulator import CommunicationSimulator

    original_command_builder: Callable[..., str] = (
        CommunicationSimulator._build_garnet_simulation_command
    )
    original_traffic_writer: Callable[..., tuple[str | None, str | None]] = (
        CommunicationSimulator._prepare_garnet_traffic_files
    )

    def prepare_garnet_traffic_files(
        simulator: CommunicationSimulator,
        traffic_matrices: list[Any],
        simulation_type: str,
    ) -> tuple[str | None, str | None]:
        text_file, gzip_file = original_traffic_writer(
            simulator,
            traffic_matrices,
            simulation_type,
        )
        if text_file is not None:
            _rewrite_garnet_endpoint_ids(
                text_file,
                int(simulator.system.num_chiplets),
            )
        return text_file, gzip_file

    def build_garnet_command(
        simulator: CommunicationSimulator,
        traffic_file: str,
        max_packets: int,
    ) -> str:
        command: str = original_command_builder(
            simulator, traffic_file, max_packets
        )
        return (
            f"{command} --mesh-rows={args.mesh_rows} "
            f"--link-width-bits={args.link_width_bits} "
            f"--router-latency={args.router_latency} "
            f"--link-latency={args.link_latency} "
            f"--vcs-per-vnet={args.vcs_per_vnet}"
        )

    CommunicationSimulator._prepare_garnet_traffic_files = (
        prepare_garnet_traffic_files
    )
    CommunicationSimulator._build_garnet_simulation_command = build_garnet_command

    from src.integrations.hydra.simulation_runner import HydraSimulationRunner
    from src.post.single_sim_processor import SingleSimProcessor
    from src.utils.config_loader import load_config

    topology_file: Path = (
        chipsim_root
        / "integrations"
        / "gem5"
        / "configs"
        / "topologies"
        / "myTopology.yaml"
    )
    original_topology: bytes | None = (
        topology_file.read_bytes() if topology_file.is_file() else None
    )
    try:
        config: dict[str, Any] = load_config(args.config)
        runner: HydraSimulationRunner = HydraSimulationRunner(config)
        raw_results_dir: str = runner.run()
        post_processing: dict[str, Any] | None = config.get("post_processing")
        if post_processing is not None:
            processor: SingleSimProcessor = SingleSimProcessor(
                raw_results_dir,
                post_processing,
            )
            processor.process()
        return 0
    finally:
        if original_topology is None:
            topology_file.unlink(missing_ok=True)
        else:
            topology_file.write_bytes(original_topology)


if __name__ == "__main__":
    raise SystemExit(main())
