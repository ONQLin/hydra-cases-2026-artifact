"""Simple static roofline baseline for HYDRA configurations.

The baseline models a fixed request batch and the chiplets assigned to their
preferred phase. It does not model dynamic batching, elastic reassignment, or
runtime state transitions.
"""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from Fast_Estimate.fast_estimate_models import chiplet_bandwidth_tuple
from Fast_Estimate.fast_estimate_models import mapping_key


MESH_WIDTH = 6
MESH_HEIGHT = 4
CONFIG_FIELDS = (
    "batchsize",
    "Num M",
    "Num Mp",
    "Num Md",
    "Num Ap",
    "Num Ad",
    "NoI_bw(GBps)",
)


def config_from_row(row: pd.Series | dict) -> dict[str, int]:
    return {field: int(row[field]) for field in CONFIG_FIELDS}


def static_roofline_bounds(
    config: dict[str, int],
    profile_data: dict,
    mapping_entry: dict,
) -> dict[str, float]:
    """Return fixed-allocation compute, HBM, and NoI throughput ceilings."""

    batchsize = max(int(config["batchsize"]), 1)
    num_memory = max(int(config["Num M"]), 1)
    bandwidth = int(config["NoI_bw(GBps)"])
    decode_latency_s = max(
        float(mapping_entry["max_decode_latency_us"]) / 1e6,
        1e-12,
    )
    memory_access_gb = max(
        float(profile_data["mem_access_per_token(GB)"]),
        1e-12,
    )

    # The static batch reuses model parameters but does not admit new requests
    # while that batch is running.
    reuse = (
        float(profile_data["param_scale"]) / batchsize
        + 1.0
        - float(profile_data["param_scale"])
    )
    compute_throughput = batchsize / decode_latency_s

    hbm_bandwidth, _, _ = chiplet_bandwidth_tuple(bandwidth)
    hbm_throughput = (
        num_memory * hbm_bandwidth / (reuse * memory_access_gb)
    )

    mesh_bandwidth = bandwidth * min(
        (MESH_WIDTH + MESH_HEIGHT - 1) * 2,
        3 * num_memory,
    )
    noi_throughput = mesh_bandwidth / (reuse * memory_access_gb)

    return {
        "roofline_compute_throughput": float(compute_throughput),
        "roofline_hbm_throughput": float(hbm_throughput),
        "roofline_noi_throughput": float(noi_throughput),
        "roofline_throughput": float(
            min(compute_throughput, hbm_throughput, noi_throughput)
        ),
    }


def score_configurations(
    model: str,
    dataset: str,
    rows: Iterable[pd.Series],
    profile_data: dict,
    mapping_entries: dict[str, dict],
) -> pd.DataFrame:
    """Score configurations without consulting simulated performance."""

    records: list[dict[str, float | int]] = []
    for row in rows:
        config = config_from_row(row)
        entry = mapping_entries.get(
            mapping_key(model, dataset, config, profile_data)
        )
        if not entry or not entry.get("valid", False):
            continue
        records.append(
            {
                **config,
                **static_roofline_bounds(config, profile_data, entry),
            }
        )
    if not records:
        raise RuntimeError(f"No valid roofline inputs for {model}-{dataset}.")
    return pd.DataFrame(records).sort_values(
        "roofline_throughput",
        ascending=False,
        kind="stable",
    ).reset_index(drop=True)
