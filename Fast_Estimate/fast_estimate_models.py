"""Fast DSE models for static and ET+DB configurations."""

from __future__ import annotations

import json
import sys
from collections import deque
from functools import lru_cache
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

import Sim.common as common
import Sim.config.utils as utils
from analytic_profile.Base_Accmodel import BaseAccModel
from Sim.config.model_config import BaseBlockConfig
from Sim.config.model_config import BaseModelConfig
from Sim.config.model_config import baselayerConfig
from Sim.config.sys_config import BaseChipletPlacerConfig
from Sim.config.sys_config import ChipletsConfig
from Sim.config.utils import used_bws_dict
from Sim.entities.chips_network import chip_graph
from Sim.entities.comp_sys import comp_sys
from Sim.entities.mem_chiplet import Params
from Sim.entities.mem_chiplet import mem_chiplet
from Sim.entities.mem_sys import mem_sys
from Sim.entities.static_mapper import static_mapper
from Sim.placer.BasePlacer import BasePlacer

MODEL_CONFIG_NAMES = {
    "LLAMA3": "llama3-8b",
    "MAMBA2": "mamba2-3b",
    "NEMO": "nemotronh-4b",
}

PROFILE_ROOT = REPO_ROOT / "Fast_Estimate/workloads_profiles"

OPERATOR_FAMILIES = {
    "A": {
        "block_keyword": "transformer",
        "prefill_chiplet": "tscs_p",
        "decode_chiplet": "tscs_d",
        "prefill_count": "Num Ap",
        "decode_count": "Num Ad",
    },
    "M": {
        "block_keyword": "mamba",
        "prefill_chiplet": "marca_p",
        "decode_chiplet": "marca_d",
        "prefill_count": "Num Mp",
        "decode_count": "Num Md",
    },
}


def normalize_model(value: str) -> str:
    value = value.strip().upper()
    aliases = {
        "LLAMA": "LLAMA3",
        "LLAMA3": "LLAMA3",
        "MAMBA": "MAMBA2",
        "MAMBA2": "MAMBA2",
        "NEMO": "NEMO",
    }
    if value not in aliases:
        raise ValueError(
            f"Unsupported model '{value}'. Expected LLAMA3, MAMBA2, or NEMO."
        )
    return aliases[value]


def normalize_dataset(value: str) -> str:
    value = value.strip().upper()
    aliases = {
        "ARXIV": "ARXIV",
        "BWB": "BWB",
        "CHAT": "CHAT",
        "LW": "LW",
        "LONGWRITER": "LW",
    }
    if value not in aliases:
        raise ValueError(
            f"Unsupported dataset '{value}'. Expected ARXIV, BWB, CHAT, or LW."
        )
    return aliases[value]


def workload_profile_path(model: str, dataset: str) -> Path:
    model_name = normalize_model(model).lower()
    dataset_name = normalize_dataset(dataset).lower()
    return PROFILE_ROOT / f"{model_name}_{dataset_name}.json"


def summary_csv_path(summary_root: Path, model: str, dataset: str) -> Path:
    return summary_root / f"results_summary_{normalize_model(model)}-{normalize_dataset(dataset)}_bw_static_static.csv"


def placement_cache_path(cache_dir: Path, model: str) -> Path:
    return cache_dir / f"bwplacement_cache_{normalize_model(model).lower()}.json"


def mapping_cache_path(cache_dir: Path, model: str) -> Path:
    return cache_dir / f"mapping_cache_{normalize_model(model).lower()}.json"


def config_key(config: dict) -> tuple[int, int, int, int, int, int, int]:
    return (
        int(config["batchsize"]),
        int(config["Num M"]),
        int(config["Num Mp"]),
        int(config["Num Md"]),
        int(config["Num Ap"]),
        int(config["Num Ad"]),
        int(config["NoI_bw(GBps)"]),
    )


def placement_key(config: dict) -> str:
    return (
        f"m{int(config['Num M'])}_mp{int(config['Num Mp'])}_md{int(config['Num Md'])}"
        f"_ap{int(config['Num Ap'])}_ad{int(config['Num Ad'])}_bw{int(config['NoI_bw(GBps)'])}"
    )


def mapping_key(model: str, dataset: str, config: dict, profile_data: dict) -> str:
    return (
        f"{normalize_model(model).lower()}_{normalize_dataset(dataset).lower()}_"
        f"pf{int(profile_data['prefill_len'])}_dc{int(profile_data['decode_len'])}_"
        f"ps{profile_data['param_scale']}_ma{profile_data['mem_access_per_token(GB)']}_"
        f"bs{int(config['batchsize'])}_{placement_key(config)}"
    )


def load_cache_entries(cache_path: Path) -> dict[str, dict]:
    if not cache_path.is_file():
        return {}
    try:
        with cache_path.open("r", encoding="utf-8") as handle:
            cache_data = json.load(handle)
    except json.JSONDecodeError:
        cache_path.unlink(missing_ok=True)
        return {}
    if (
        isinstance(cache_data, dict)
        and "entries" in cache_data
        and isinstance(cache_data["entries"], dict)
    ):
        return cache_data["entries"]
    return cache_data if isinstance(cache_data, dict) else {}


def save_cache_entries(cache_path: Path, metadata: dict[str, Any], entries: dict[str, dict]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_data = {
        "metadata": {**metadata, "entries": len(entries)},
        "entries": entries,
    }
    temp_path = cache_path.with_suffix(f"{cache_path.suffix}.tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(cache_data, handle, indent=2)
    temp_path.replace(cache_path)


def load_valid_summary_configs(
    summary_root: Path,
    model: str,
    dataset: str,
) -> set[tuple[int, int, int, int, int, int, int]] | None:
    summary_path = summary_csv_path(summary_root, model, dataset)
    if not summary_path.is_file():
        return None

    df = pd.read_csv(summary_path)
    mask = (
        np.isfinite(df["TTFT(s)"])
        & np.isfinite(df["tokens_per_sec"])
        & (df["TTFT(s)"] > 0)
        & (df["tokens_per_sec"] > 0)
    )
    valid_df = df.loc[mask].copy()
    return {config_key(row) for _, row in valid_df.iterrows()}


def make_chiplet_alloc(model: str, config: dict[str, int]) -> dict[str, int]:
    model = normalize_model(model)
    alloc = {
        "marca_p": 0,
        "marca_d": 0,
        "tscs_p": 0,
        "tscs_d": 0,
        "systolicarray_p": 0,
        "HBM3": int(config["Num M"]),
        "GDDR7": 0,
        "HBM3e": 0,
        "B200_p": 0,
        "vuarray_d": 0,
        "unified_p": 0,
    }
    if model == "LLAMA3":
        alloc["tscs_p"] = int(config["Num Ap"])
        alloc["tscs_d"] = int(config["Num Ad"])
    elif model == "MAMBA2":
        alloc["marca_p"] = int(config["Num Mp"])
        alloc["marca_d"] = int(config["Num Md"])
    else:
        alloc["marca_p"] = int(config["Num Mp"])
        alloc["marca_d"] = int(config["Num Md"])
        alloc["tscs_p"] = int(config["Num Ap"])
        alloc["tscs_d"] = int(config["Num Ad"])
    return alloc


@lru_cache(maxsize=None)
def chiplet_bandwidth_tuple(bw: int) -> tuple[float, float, float]:
    chiplet_config = ChipletsConfig(
        D2D_NoI_bw=int(bw),
        area_limit=144,
        Fixed_chiplet_area=True,
    )
    decode_link_bw = float(used_bws_dict[int(bw)][1])
    prefill_link_bw = float(used_bws_dict[int(bw)][0])
    return float(chiplet_config.HBM_IO_bw), prefill_link_bw, decode_link_bw


def batch_bandwidth_gain(profile_data: dict, batchsize: int) -> float:
    scale = float(profile_data["param_scale"])
    denom = scale / max(int(batchsize), 1) + 1.0 - scale
    return 1.0 / max(denom, 1e-12)


def runtime_batch_utilization(
    profile_data: dict,
    batchsize: int,
) -> float:
    batchsize = max(int(batchsize), 1)
    active_token_cost = max(
        float(profile_data["prefill_len"])
        + min(float(profile_data["decode_len"]), 96),
        1.0,
    )
    loss_exponent = (
        0.36 * active_token_cost / (active_token_cost + 100.0)
    )
    return float(batchsize ** (-loss_exponent))


def static_state_bounds(
    config: dict,
    profile_data: dict,
    mapping_entry: dict,
) -> dict[str, float]:
    batchsize = int(config["batchsize"])
    bw = int(config["NoI_bw(GBps)"])
    num_m = int(config["Num M"])
    num_compute = (
        int(config["Num Mp"])
        + int(config["Num Md"])
        + int(config["Num Ap"])
        + int(config["Num Ad"])
    )
    decode_len = max(int(profile_data["decode_len"]), 1)
    prefill_len = max(int(profile_data["prefill_len"]), 1)
    mem_access = max(float(profile_data["mem_access_per_token(GB)"]), 1e-12)
    pf_s = max(float(mapping_entry["max_prefill_latency_us"]) / 1e6, 1e-12)
    d_s = max(float(mapping_entry["max_decode_latency_us"]) / 1e6, 1e-12)
    hbm_io_bw, _, _ = chiplet_bandwidth_tuple(bw)
    batch_gain = batch_bandwidth_gain(profile_data, batchsize)

    mamba_only = (
        int(config["Num Mp"]) + int(config["Num Md"]) > 0
        and int(config["Num Ap"]) + int(config["Num Ad"]) == 0
    )
    prefill_efficiency = (
        0.50 if mamba_only else 0.70
    )
    decode_efficiency = (
        0.25 if mamba_only else 0.50
    )
    prefill_service_s = pf_s / prefill_efficiency
    decode_service_s = d_s / decode_efficiency
    prefill_support_tp = batchsize * decode_len / prefill_service_s
    decode_support_tp = batchsize / decode_service_s
    mem_tp = (
        num_m
        * hbm_io_bw
        * batch_gain
    ) / mem_access

    mesh_bisection_bw = float(16 * bw)
    endpoint_injection_bw = float(bw * min(max(num_compute, 1), 3 * num_m))
    bisection_bw = min(mesh_bisection_bw, endpoint_injection_bw)
    bisection_tp = (bisection_bw * batch_gain) / mem_access

    decode_stage_tp = min(decode_support_tp, mem_tp, bisection_tp)
    prefill_stage_tp = min(
        prefill_support_tp,
        decode_len * mem_tp,
        decode_len * bisection_tp,
    )

    prefill_req_rate = prefill_stage_tp / decode_len
    decode_req_rate = decode_stage_tp / decode_len
    request_completion_rate = min(prefill_req_rate, decode_req_rate)
    throughput_bound = request_completion_rate * decode_len

    stage_balance = min(prefill_stage_tp, decode_stage_tp) / max(
        max(prefill_stage_tp, decode_stage_tp),
        1e-12,
    )
    prefill_pressure = decode_req_rate / max(prefill_req_rate, 1e-12)
    bw_pressure = (prefill_support_tp + decode_support_tp) / max(
        min(mem_tp, bisection_tp),
        1e-12,
    )
    return {
        "prefill_service_s": float(prefill_service_s),
        "throughput_bound": float(throughput_bound),
        "stage_balance": float(stage_balance),
        "prefill_pressure": float(prefill_pressure),
        "bw_pressure": float(bw_pressure),
    }


def _profile_block_latency_us(
    block_config: BaseBlockConfig,
    stage: str,
    logic_name: str,
    batchsize: int,
    prefill_len: int,
    effective_bandwidth: float,
) -> float:
    analytical_model: BaseAccModel = BaseAccModel.create_from_name(logic_name)
    total_latency_us = 0.0
    for layer_config, layer_names in zip(
        block_config.layer_configs,
        block_config.layers,
    ):
        if len(layer_config) != len(layer_names):
            raise ValueError("Layer configuration and name counts do not match.")
        for layer, layer_name in zip(layer_config, layer_names):
            if not isinstance(layer, baselayerConfig):
                raise TypeError("Expected a baselayerConfig while profiling a block.")
            kernel_name = layer_name
            if kernel_name in {"MHA", "SSM"}:
                kernel_name += "_p" if stage == "P" else "_d"

            kernel_args = analytical_model.config_params(**vars(layer))
            if stage == "P":
                kernel_args["bs"] = prefill_len
                kernel_args["batch_size"] = batchsize
                kernel_args["L_seq"] = prefill_len
                if "in_size" in kernel_args:
                    kernel_args["in_size"] *= prefill_len
            else:
                kernel_args["bs"] = 1
                kernel_args["batch_size"] = batchsize
                kernel_args["L_seq"] = prefill_len
            kernel_args["logic_name"] = logic_name
            kernel_args["ext_bw"] = effective_bandwidth
            result = analytical_model.get_kernel(kernel_name, **kernel_args)
            total_latency_us += common.convert_analy_time(result.total_latency)
    return float(total_latency_us)


def runtime_chiplet_service_rates(
    model_config: BaseModelConfig,
    config: dict,
    profile_data: dict,
) -> dict[str, dict[str, dict[str, float]]]:
    batchsize = max(int(config["batchsize"]), 1)
    batch_utilization = runtime_batch_utilization(profile_data, batchsize)
    prefill_len = max(int(profile_data["prefill_len"]), 1)
    num_m = max(int(config["Num M"]), 1)
    num_compute = max(
        int(config["Num Mp"])
        + int(config["Num Md"])
        + int(config["Num Ap"])
        + int(config["Num Ad"]),
        1,
    )
    bw = int(config["NoI_bw(GBps)"])
    chiplet_config = ChipletsConfig(
        D2D_NoI_bw=bw,
        area_limit=144,
        Fixed_chiplet_area=True,
    )
    common.logic_lib = chiplet_config.chips_lib.logics
    _, prefill_link_bw, decode_link_bw = chiplet_bandwidth_tuple(bw)
    shared_hbm_per_compute = num_m * chiplet_config.HBM_IO_bw / num_compute
    phase_bandwidth = {
        "P": min(prefill_link_bw, shared_hbm_per_compute),
        "D": min(decode_link_bw, shared_hbm_per_compute),
    }
    mamba_only = (
        int(config["Num Mp"]) + int(config["Num Md"]) > 0
        and int(config["Num Ap"]) + int(config["Num Ad"]) == 0
    )
    phase_efficiency = {
        "P": 0.50 if mamba_only else 0.70,
        "D": 0.25 if mamba_only else 0.50,
    }

    rates: dict[str, dict[str, dict[str, float]]] = {}
    for family, spec in OPERATOR_FAMILIES.items():
        num_prefill = int(config[spec["prefill_count"]])
        num_decode = int(config[spec["decode_count"]])
        if num_prefill + num_decode == 0:
            continue

        family_blocks = [
            model_config.hybrid_blocks[type_index]
            for type_index in model_config.block_type_sequence
            if spec["block_keyword"]
            in model_config.hybrid_blocks[type_index].type_name.lower()
        ]
        if not family_blocks:
            continue

        family_rates: dict[str, dict[str, float]] = {"P": {}, "D": {}}
        for stage in ("P", "D"):
            stage_work = prefill_len if stage == "P" else 1
            for chiplet_role, chiplet_name in (
                ("P", spec["prefill_chiplet"]),
                ("D", spec["decode_chiplet"]),
            ):
                latency_us = sum(
                    _profile_block_latency_us(
                        block_config=block,
                        stage=stage,
                        logic_name=chiplet_name,
                        batchsize=batchsize,
                        prefill_len=prefill_len,
                        effective_bandwidth=phase_bandwidth[stage],
                    )
                    for block in family_blocks
                )
                latency_s = max(latency_us / 1e6, 1e-12)
                family_rates[stage][chiplet_role] = float(
                    phase_efficiency[stage]
                    * batchsize
                    * batch_utilization
                    * stage_work
                    * (
                        1.0
                        if stage == chiplet_role
                        else 0.75
                    )
                    / latency_s
                )
        rates[family] = family_rates

    if not rates:
        raise ValueError("The configuration has no chiplets for the model operators.")
    return rates


def _runtime_bandwidth_caps(
    config: dict,
    profile_data: dict,
) -> dict[str, float]:
    bw = int(config["NoI_bw(GBps)"])
    num_m = max(int(config["Num M"]), 1)
    num_compute = max(
        int(config["Num Mp"])
        + int(config["Num Md"])
        + int(config["Num Ap"])
        + int(config["Num Ad"]),
        1,
    )
    hbm_io_bw, _, _ = chiplet_bandwidth_tuple(bw)
    batch_gain = batch_bandwidth_gain(profile_data, int(config["batchsize"]))
    batch_utilization = runtime_batch_utilization(
        profile_data,
        int(config["batchsize"]),
    )
    mesh_bisection_bw = float(16 * bw)
    endpoint_injection_bw = float(bw * min(num_compute, 3 * num_m))
    shared_noi_bw = min(mesh_bisection_bw, endpoint_injection_bw)

    caps: dict[str, float] = {}
    for stage, length_key, memory_key in (
        ("P", "prefill_len", "prefill mem_access(GB)"),
        ("D", "decode_len", "decode mem_access(GB)"),
    ):
        length = max(float(profile_data[length_key]), 1.0)
        phase_memory = float(profile_data.get(memory_key, 0.0))
        if phase_memory <= 0:
            memory_per_token = max(
                float(profile_data["mem_access_per_token(GB)"]),
                1e-12,
            )
        else:
            memory_per_token = max(phase_memory / length, 1e-12)
        hbm_cap = (
            num_m
            * hbm_io_bw
            * batch_gain
            * batch_utilization
            / memory_per_token
        )
        noi_cap = (
            shared_noi_bw
            * batch_gain
            * batch_utilization
            / memory_per_token
        )
        caps[stage] = float(min(hbm_cap, noi_cap))
    return caps


def runtime_state_capacities(
    config: dict,
    profile_data: dict,
    service_rates: dict[str, dict[str, dict[str, float]]],
) -> list[dict[str, Any]]:
    family_counts: dict[str, tuple[int, int]] = {}
    for family in service_rates:
        spec = OPERATOR_FAMILIES[family]
        family_counts[family] = (
            int(config[spec["prefill_count"]]),
            int(config[spec["decode_count"]]),
        )

    families = tuple(sorted(family_counts))
    bandwidth_caps = _runtime_bandwidth_caps(config, profile_data)
    state_ranges = [
        range(sum(family_counts[family]) + 1) for family in families
    ]
    states: list[dict[str, Any]] = []
    for coordinates in product(*state_ranges):
        prefill_family_caps: list[float] = []
        decode_family_caps: list[float] = []
        allocation: dict[str, dict[str, int]] = {}

        for family, num_for_prefill in zip(families, coordinates):
            num_prefill_native, num_decode_native = family_counts[family]
            total = num_prefill_native + num_decode_native
            num_prefill_on_prefill = min(num_for_prefill, num_prefill_native)
            num_prefill_on_decode = max(
                0,
                num_for_prefill - num_prefill_native,
            )
            num_for_decode = total - num_for_prefill
            num_decode_on_decode = min(num_for_decode, num_decode_native)
            num_decode_on_prefill = max(
                0,
                num_for_decode - num_decode_native,
            )
            family_rate = service_rates[family]
            prefill_family_caps.append(
                num_prefill_on_prefill * family_rate["P"]["P"]
                + num_prefill_on_decode * family_rate["P"]["D"]
            )
            decode_family_caps.append(
                num_decode_on_decode * family_rate["D"]["D"]
                + num_decode_on_prefill * family_rate["D"]["P"]
            )
            allocation[family] = {
                "prefill_on_prefill": num_prefill_on_prefill,
                "prefill_on_decode": num_prefill_on_decode,
                "decode_on_decode": num_decode_on_decode,
                "decode_on_prefill": num_decode_on_prefill,
            }

        states.append(
            {
                "state": tuple(int(value) for value in coordinates),
                "families": families,
                "allocation": allocation,
                "Gp": float(
                    min(min(prefill_family_caps), bandwidth_caps["P"])
                ),
                "Gd": float(
                    min(min(decode_family_caps), bandwidth_caps["D"])
                ),
            }
        )
    return states


def _workload_trace_arrays(
    profile_data: dict,
    request_trace: (
        str
        | Path
        | pd.DataFrame
        | tuple[np.ndarray, np.ndarray]
        | None
    ),
) -> tuple[np.ndarray, np.ndarray]:
    if request_trace is None:
        prefill = np.asarray([float(profile_data["prefill_len"])])
        decode = np.asarray([float(profile_data["decode_len"])])
    elif isinstance(request_trace, tuple):
        prefill, decode = request_trace
        prefill = np.asarray(prefill, dtype=float)
        decode = np.asarray(decode, dtype=float)
    else:
        trace = (
            request_trace
            if isinstance(request_trace, pd.DataFrame)
            else pd.read_csv(request_trace)
        )
        required = {"num_prefill_tokens", "num_decode_tokens"}
        if not required.issubset(trace.columns):
            missing = ", ".join(sorted(required - set(trace.columns)))
            raise ValueError(f"Request trace is missing columns: {missing}")
        prefill = trace["num_prefill_tokens"].to_numpy(dtype=float)
        decode = trace["num_decode_tokens"].to_numpy(dtype=float)
    prefill = np.maximum(prefill, 1.0)
    decode = np.maximum(decode, 1.0)
    return prefill, decode


def _state_drain_score(
    state: dict[str, Any],
    prefill_work: float,
    decode_work: float,
) -> float:
    if prefill_work > 0 and state["Gp"] <= 0:
        return float("inf")
    if decode_work > 0 and state["Gd"] <= 0:
        return float("inf")
    prefill_time = (
        prefill_work / state["Gp"] if prefill_work > 0 else 0.0
    )
    decode_time = decode_work / state["Gd"] if decode_work > 0 else 0.0
    return float(max(prefill_time, decode_time))


def _next_runtime_state(
    current: tuple[int, ...],
    states_by_key: dict[tuple[int, ...], dict[str, Any]],
    prefill_work: float,
    decode_work: float,
) -> tuple[int, ...]:
    current_score = _state_drain_score(
        states_by_key[current],
        prefill_work,
        decode_work,
    )
    target = min(
        states_by_key,
        key=lambda key: (
            _state_drain_score(
                states_by_key[key],
                prefill_work,
                decode_work,
            ),
            sum(abs(a - b) for a, b in zip(key, current)),
            key,
        ),
    )
    target_score = _state_drain_score(
        states_by_key[target],
        prefill_work,
        decode_work,
    )
    if current_score <= target_score * 1.05:
        return current
    if target == current:
        return current

    neighbors: list[tuple[int, ...]] = []
    for index, (source_value, target_value) in enumerate(zip(current, target)):
        if source_value == target_value:
            continue
        candidate = list(current)
        candidate[index] += 1 if target_value > source_value else -1
        candidate_key = tuple(candidate)
        if candidate_key in states_by_key:
            neighbors.append(candidate_key)
    return min(
        neighbors,
        key=lambda key: (
            _state_drain_score(
                states_by_key[key],
                prefill_work,
                decode_work,
            ),
            key,
        ),
    )


def _active_token_cost(
    prefill_jobs: deque[dict[str, float]],
    decode_jobs: deque[dict[str, float]],
) -> float:
    prefill_cost = sum(
        job["remaining_prefill"]
        + min(job["decode_tokens"], 96)
        for job in prefill_jobs
    )
    decode_cost = sum(
        min(job["remaining_decode"], 96)
        for job in decode_jobs
    )
    return float(prefill_cost + decode_cost)


def _process_prefill_jobs(
    prefill_jobs: deque[dict[str, float]],
    decode_jobs: deque[dict[str, float]],
    capacity_tokens: float,
) -> None:
    while prefill_jobs and capacity_tokens > 0:
        job = prefill_jobs[0]
        amount = min(job["remaining_prefill"], capacity_tokens)
        job["remaining_prefill"] -= amount
        capacity_tokens -= amount
        if job["remaining_prefill"] <= 1e-12:
            prefill_jobs.popleft()
            decode_jobs.append(
                {"remaining_decode": job["decode_tokens"]}
            )


def _process_decode_jobs(
    decode_jobs: deque[dict[str, float]],
    capacity_tokens: float,
) -> float:
    processed = 0.0
    while decode_jobs and capacity_tokens > 0:
        job = decode_jobs[0]
        amount = min(job["remaining_decode"], capacity_tokens)
        job["remaining_decode"] -= amount
        capacity_tokens -= amount
        processed += amount
        if job["remaining_decode"] <= 1e-12:
            decode_jobs.popleft()
    return float(processed)


def estimate_runtime_markov(
    model_config: BaseModelConfig,
    config: dict,
    profile_data: dict,
    request_trace: (
        str
        | Path
        | pd.DataFrame
        | tuple[np.ndarray, np.ndarray]
        | None
    ) = None,
    arrival_interval_s: float = 0.01,
    duration_s: float = 100.0,
    step_s: float = 0.001,
    allocation_interval_s: float = 0.01,
    service_rates: (
        dict[str, dict[str, dict[str, float]]] | None
    ) = None,
    enable_token_budget_admission: bool = True,
) -> dict[str, Any]:
    if arrival_interval_s <= 0:
        raise ValueError("arrival_interval_s must be positive.")
    if duration_s <= 0 or step_s <= 0 or allocation_interval_s <= 0:
        raise ValueError("Runtime duration and step sizes must be positive.")

    if service_rates is None:
        service_rates = runtime_chiplet_service_rates(
            model_config,
            config,
            profile_data,
        )
    states = runtime_state_capacities(config, profile_data, service_rates)
    states_by_key = {state["state"]: state for state in states}
    preferred_state = tuple(
        int(config[OPERATOR_FAMILIES[family]["prefill_count"]])
        for family in states[0]["families"]
    )
    if preferred_state not in states_by_key:
        raise ValueError("The preferred chiplet allocation is not a valid state.")

    prefill_lengths, decode_lengths = _workload_trace_arrays(
        profile_data,
        request_trace,
    )
    trace_token_cost = prefill_lengths + np.minimum(
        decode_lengths,
        96,
    )
    mean_prefill = float(np.mean(prefill_lengths))
    max_batch = max(int(config["batchsize"]), 1)
    token_budget = max_batch * float(np.median(trace_token_cost))

    steps = max(1, int(np.ceil(duration_s / step_s)))
    allocation_steps = max(1, int(round(allocation_interval_s / step_s)))
    current_state = preferred_state
    pending_requests: deque[dict[str, float]] = deque()
    prefill_jobs: deque[dict[str, float]] = deque()
    decode_jobs: deque[dict[str, float]] = deque()
    trace_index = 0
    repeat_trace = request_trace is None
    next_arrival_s = 0.0
    completed_decode_tokens = 0.0
    state_dwell = {key: 0.0 for key in states_by_key}
    prefill_state_dwell = {key: 0.0 for key in states_by_key}
    transition_counts: dict[tuple[tuple[int, ...], tuple[int, ...]], int] = {}

    for step in range(steps):
        current_time_s = step * step_s
        while next_arrival_s <= current_time_s + 1e-12:
            if trace_index >= len(prefill_lengths):
                if repeat_trace:
                    trace_index = 0
                else:
                    next_arrival_s = float("inf")
                    break
            pending_requests.append(
                {
                    "prefill_tokens": float(prefill_lengths[trace_index]),
                    "decode_tokens": float(decode_lengths[trace_index]),
                }
            )
            trace_index += 1
            next_arrival_s += arrival_interval_s

        while pending_requests:
            active_count = len(prefill_jobs) + len(decode_jobs)
            if active_count >= max_batch:
                break
            candidate = pending_requests[0]
            candidate_cost = candidate["prefill_tokens"] + min(
                candidate["decode_tokens"],
                96,
            )
            active_cost = _active_token_cost(prefill_jobs, decode_jobs)
            if (
                enable_token_budget_admission
                and active_count > 0
                and active_cost + candidate_cost > token_budget
            ):
                break
            pending_requests.popleft()
            prefill_jobs.append(
                {
                    "remaining_prefill": candidate["prefill_tokens"],
                    "decode_tokens": candidate["decode_tokens"],
                }
            )

        prefill_work = sum(
            job["remaining_prefill"] for job in prefill_jobs
        )
        decode_work = sum(
            job["remaining_decode"] for job in decode_jobs
        )

        if step > 0 and step % allocation_steps == 0:
            next_state = _next_runtime_state(
                current=current_state,
                states_by_key=states_by_key,
                prefill_work=prefill_work,
                decode_work=decode_work,
            )
            if next_state != current_state:
                transition = (current_state, next_state)
                transition_counts[transition] = (
                    transition_counts.get(transition, 0) + 1
                )
                current_state = next_state

        state = states_by_key[current_state]
        state_dwell[current_state] += step_s
        if prefill_work > 0:
            prefill_state_dwell[current_state] += step_s

        _process_prefill_jobs(
            prefill_jobs,
            decode_jobs,
            state["Gp"] * step_s,
        )
        decode_processed = _process_decode_jobs(
            decode_jobs,
            state["Gd"] * step_s,
        )
        completed_decode_tokens += decode_processed

    total_dwell = sum(state_dwell.values())
    total_prefill_dwell = sum(prefill_state_dwell.values())
    occupancy_rows: list[dict[str, Any]] = []
    for state in states:
        key = state["state"]
        pi_s = state_dwell[key] / max(total_dwell, 1e-12)
        prefill_pi_s = prefill_state_dwell[key] / max(
            total_prefill_dwell,
            1e-12,
        )
        occupancy_rows.append(
            {
                **state,
                "pi_s": float(pi_s),
                "prefill_pi_s": float(prefill_pi_s),
            }
        )

    throughput_eq7 = sum(
        row["pi_s"] * row["Gd"] for row in occupancy_rows
    )
    ttft_terms = [
        row["prefill_pi_s"]
        * (max_batch * mean_prefill / row["Gp"])
        for row in occupancy_rows
        if row["prefill_pi_s"] > 0 and row["Gp"] > 0
    ]
    transition_rates = []
    for (source, target), count in sorted(transition_counts.items()):
        transition_rates.append(
            {
                "source": source,
                "target": target,
                "count": count,
                "rate_per_s": float(
                    count / max(state_dwell[source], 1e-12)
                ),
            }
        )

    return {
        "estimated_throughput": float(throughput_eq7),
        "estimated_ttft_s": float(sum(ttft_terms)),
        "realized_fluid_throughput": float(
            completed_decode_tokens / max(total_dwell, 1e-12)
        ),
        "state_occupancy": occupancy_rows,
        "transition_rates": transition_rates,
    }


def iter_candidate_configs(
    model: str,
    batchsizes: tuple[int, ...] = (2, 4, 8, 16, 32, 64),
    bws: tuple[int, ...] = (256, 384, 512, 640),
):
    model = normalize_model(model)
    if model in {"LLAMA3", "MAMBA2"}:
        for batchsize in batchsizes:
            for bw in bws:
                for num_m in range(2, 17):
                    remain = 24 - num_m
                    for first in range(1, remain):
                        second = remain - first
                        if model == "LLAMA3":
                            yield {
                                "batchsize": batchsize,
                                "Num M": num_m,
                                "Num Mp": 0,
                                "Num Md": 0,
                                "Num Ap": first,
                                "Num Ad": second,
                                "NoI_bw(GBps)": bw,
                            }
                        else:
                            yield {
                                "batchsize": batchsize,
                                "Num M": num_m,
                                "Num Mp": first,
                                "Num Md": second,
                                "Num Ap": 0,
                                "Num Ad": 0,
                                "NoI_bw(GBps)": bw,
                            }
        return

    for batchsize in batchsizes:
        for bw in bws:
            for num_m in range(2, 17, 2):
                remain = 24 - num_m
                if remain < 4:
                    continue
                for num_ap in range(1, remain - 2, 2):
                    for num_ad in range(1, remain - num_ap - 1, 2):
                        for num_mp in range(1, remain - num_ap - num_ad, 2):
                            num_md = remain - num_ap - num_ad - num_mp
                            if num_md < 1:
                                continue
                            yield {
                                "batchsize": batchsize,
                                "Num M": num_m,
                                "Num Mp": num_mp,
                                "Num Md": num_md,
                                "Num Ap": num_ap,
                                "Num Ad": num_ad,
                                "NoI_bw(GBps)": bw,
                            }


def max_concurrent_requests(mem_sys_inst: mem_sys, model_config: BaseModelConfig) -> int:
    max_requests = 10**9
    for mem_chiplet_inst in mem_sys_inst._mem_chiplets:
        assert isinstance(mem_chiplet_inst, mem_chiplet)
        left_budget = (mem_chiplet_inst.dram_budget - mem_chiplet_inst.inuse_budget) * 1024 * 1024
        mem_req_per_request = 0
        for block_params in mem_chiplet_inst.content_params.get("weights", []):
            assert isinstance(block_params, Params)
            type_idx = model_config.block_type_sequence[block_params.block_id]
            block_config = model_config.hybrid_blocks[type_idx]
            if block_config.cache_store:
                mem_req_per_request += block_config.states * model_config.max_position_embeddings + block_config.peak_intermed
            else:
                mem_req_per_request += block_config.states + block_config.peak_intermed
        if mem_req_per_request <= 0:
            continue
        chiplet_max = int(left_budget // mem_req_per_request)
        if chiplet_max < max_requests:
            max_requests = chiplet_max
    return max_requests


def to_json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: to_json_ready(inner) for key, inner in value.items()}
    if isinstance(value, tuple):
        return [to_json_ready(inner) for inner in value]
    if isinstance(value, list):
        return [to_json_ready(inner) for inner in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def serialize_node_attributes(resources_graph: chip_graph) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for _, attrs in resources_graph.graph.nodes(data=True):
        rows.append({key: to_json_ready(value) for key, value in attrs.items()})
    rows.sort(key=lambda row: int(row["id"]))
    return rows


def apply_node_attributes(resources_graph: chip_graph, node_attributes: list[dict[str, Any]]) -> None:
    by_id = {int(row["id"]): row for row in node_attributes}
    for _, attrs in resources_graph.graph.nodes(data=True):
        cached = by_id.get(int(attrs["id"]))
        if cached is None:
            continue
        for key, value in cached.items():
            if key == "coord" and isinstance(value, list):
                attrs[key] = tuple(value)
            else:
                attrs[key] = value


def resolve_placement_entry(
    model: str,
    model_config: BaseModelConfig,
    config: dict[str, int],
    placement_entries: dict[str, dict],
) -> dict | None:
    key = placement_key(config)
    cached = placement_entries.get(key)
    if cached is not None:
        return cached

    bw = int(config["NoI_bw(GBps)"])
    chiplet_alloc = make_chiplet_alloc(model, config)
    utils.used_bws = used_bws_dict[bw]
    utils.NoI_bw = bw

    try:
        placer_config = BaseChipletPlacerConfig(chiplet_alloc=chiplet_alloc, placer_label="bw")
        resources_graph: chip_graph = placer_config.chip_graph
        placer_inst: BasePlacer = placer_config.placer_inst(
            chiplet_num=resources_graph.num_nodes,
            chiplet_alloc=placer_config.chiplet_alloc,
        )
        group_hbm_m, group_hbm_a = placer_inst.make_chiplet_placement(
            chip_graph=resources_graph,
            model_config=model_config,
            logging=False,
        )
        mem_sys_inst = mem_sys(resources_graph)
        mem_sys_inst.load_model(
            label="bw",
            mod_config=model_config,
            group_HBM_M=group_hbm_m,
            group_HBM_A=group_hbm_a,
        )
        cached = {
            "valid": True,
            "group_hbm_m": to_json_ready(group_hbm_m),
            "group_hbm_a": to_json_ready(group_hbm_a),
            "node_attributes": serialize_node_attributes(resources_graph),
            "max_concurrent_requests": int(max_concurrent_requests(mem_sys_inst, model_config)),
        }
    except (ValueError, ZeroDivisionError, KeyError) as exc:
        cached = {
            "valid": False,
            "error": str(exc),
        }

    placement_entries[key] = cached
    return cached


def evaluate_config(
    model: str,
    dataset: str,
    profile_data: dict,
    model_config: BaseModelConfig,
    config: dict[str, int],
    placement_entries: dict[str, dict],
    mapping_entries: dict[str, dict],
) -> dict | None:
    map_key = mapping_key(model, dataset, config, profile_data)
    cached_mapping = mapping_entries.get(map_key)
    if cached_mapping is not None:
        if not cached_mapping.get("valid", False):
            return None
        return dict(config)

    bw = int(config["NoI_bw(GBps)"])
    batchsize = int(config["batchsize"])
    chiplet_alloc = make_chiplet_alloc(model, config)
    utils.used_bws = used_bws_dict[bw]
    utils.NoI_bw = bw

    placement_entry = resolve_placement_entry(model, model_config, config, placement_entries)
    if not placement_entry or not placement_entry.get("valid", False):
        mapping_entries[map_key] = {"valid": False, "reason": "invalid_placement"}
        return None
    if int(placement_entry["max_concurrent_requests"]) < batchsize:
        mapping_entries[map_key] = {
            "valid": False,
            "reason": "insufficient_memory_capacity",
            "max_concurrent_requests": int(placement_entry["max_concurrent_requests"]),
        }
        return None

    try:
        placer_config = BaseChipletPlacerConfig(chiplet_alloc=chiplet_alloc, placer_label="bw")
        resources_graph: chip_graph = placer_config.chip_graph
        apply_node_attributes(resources_graph, placement_entry["node_attributes"])

        mem_sys_inst = mem_sys(resources_graph)
        mem_sys_inst.load_model(
            label="bw",
            mod_config=model_config,
            group_HBM_M=placement_entry["group_hbm_m"],
            group_HBM_A=placement_entry["group_hbm_a"],
        )
        comp_sys_inst = comp_sys(resources_graph)
        mapper = static_mapper()
        chiplet_config = ChipletsConfig(
            D2D_NoI_bw=bw,
            area_limit=144,
            Fixed_chiplet_area=True,
        )
        common.logic_lib = chiplet_config.chips_lib.logics

        pf_len = profile_data["prefill_len"]
        dc_len = profile_data["decode_len"]
        num_compute = int(config["Num Mp"]) + int(config["Num Md"]) + int(config["Num Ap"]) + int(config["Num Ad"])
        if num_compute <= 0:
            mapping_entries[map_key] = {"valid": False, "reason": "no_compute_chiplets"}
            return None

        eff_bw = min(utils.used_bws[1], int(config["Num M"]) * chiplet_config.HBM_IO_bw / num_compute)
        _, lat_dict = mapper.generate_mapping(
            model_config,
            comp_sys_inst,
            mem_sys_inst,
            bs=batchsize,
            pf_c=pf_len,
            NoI_bw=eff_bw,
        )
    except (ValueError, ZeroDivisionError, KeyError) as exc:
        mapping_entries[map_key] = {"valid": False, "reason": str(exc)}
        return None

    p_lat_list = [lat_dict["P"][key] for key in sorted(lat_dict["P"].keys())]
    d_lat_list = [lat_dict["D"][key] for key in sorted(lat_dict["D"].keys())]
    if not p_lat_list or not d_lat_list:
        mapping_entries[map_key] = {"valid": False, "reason": "empty_latency_dict"}
        return None

    max_prefill_latency_us = float(np.max(p_lat_list))
    max_decode_latency_us = float(np.max(d_lat_list))
    t_comp = float(
        max_decode_latency_us
        + max_prefill_latency_us * (1 / profile_data["decode_len"])
    )
    comp_tp = (1e6 / t_comp) * batchsize
    mem_tp = int(config["Num M"]) * chiplet_config.HBM_IO_bw / (
        (profile_data["param_scale"] / batchsize + 1 - profile_data["param_scale"])
        * profile_data["mem_access_per_token(GB)"]
    )
    noi_tp = bw * min(18, 3 * int(config["Num M"])) / (
        (profile_data["param_scale"] / batchsize + 1 - profile_data["param_scale"])
        * profile_data["mem_access_per_token(GB)"]
    )
    throughput = float(min(comp_tp, mem_tp, noi_tp))
    if max_prefill_latency_us <= 0 or throughput <= 0:
        mapping_entries[map_key] = {"valid": False, "reason": "non_positive_metric"}
        return None

    cached_mapping = {
        "valid": True,
        "mapper_throughput": throughput,
        "max_prefill_latency_us": max_prefill_latency_us,
        "max_decode_latency_us": max_decode_latency_us,
        "prefill_len": int(profile_data["prefill_len"]),
        "decode_len": int(profile_data["decode_len"]),
    }
    mapping_entries[map_key] = cached_mapping

    return dict(config)


def ensure_dataset_candidates(
    model: str,
    dataset: str,
    summary_root: Path,
    placement_cache_dir: Path,
    mapping_cache_dir: Path,
) -> tuple[dict, dict[str, dict], list[dict]]:
    model = normalize_model(model)
    dataset = normalize_dataset(dataset)
    profile_path = workload_profile_path(model, dataset)
    with profile_path.open("r", encoding="utf-8") as handle:
        profile_data = json.load(handle)

    valid_summary_configs = load_valid_summary_configs(
        summary_root,
        model,
        dataset,
    )
    model_config = BaseModelConfig.create_from_name(MODEL_CONFIG_NAMES[model])
    placement_entries = load_cache_entries(placement_cache_path(placement_cache_dir, model))
    mapping_entries = load_cache_entries(mapping_cache_path(mapping_cache_dir, model))

    filtered_results: list[dict] = []
    for config in iter_candidate_configs(model):
        evaluated = evaluate_config(
            model=model,
            dataset=dataset,
            profile_data=profile_data,
            model_config=model_config,
            config=config,
            placement_entries=placement_entries,
            mapping_entries=mapping_entries,
        )
        if evaluated is None:
            continue
        if valid_summary_configs is None or config_key(evaluated) in valid_summary_configs:
            filtered_results.append(evaluated)

    save_cache_entries(
        placement_cache_path(placement_cache_dir, model),
        {
            "model": model,
            "placer": "bw",
            "key_fields": ["Num M", "Num Mp", "Num Md", "Num Ap", "Num Ad", "NoI_bw(GBps)"],
        },
        placement_entries,
    )
    save_cache_entries(
        mapping_cache_path(mapping_cache_dir, model),
        {
            "model": model,
            "dataset_scope": "shared_by_model",
            "key_fields": ["batchsize", "Num M", "Num Mp", "Num Md", "Num Ap", "Num Ad", "NoI_bw(GBps)"],
        },
        mapping_entries,
    )

    return profile_data, mapping_entries, filtered_results
