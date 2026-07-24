"""Static-state analytical candidate filtering for fast DSE.

The filter evaluates every macro-architecture with the static mapper and
physical service bounds, then retains a structurally diverse candidate set.
It does not select the final operating point; detailed simulation results
make that decision in the second stage.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from Fast_Estimate.fast_estimate_models import static_state_bounds


CONFIG_FIELDS = (
    "batchsize",
    "Num M",
    "Num Mp",
    "Num Md",
    "Num Ap",
    "Num Ad",
    "NoI_bw(GBps)",
)


def evaluate_states(
    rows: Iterable[dict],
    profile_data: dict,
    mapping_entries: dict[str, dict],
    mapping_key_for_config,
) -> pd.DataFrame:
    """Evaluate valid configurations without consulting measured performance."""

    records: list[dict[str, object]] = []
    for row in rows:
        config = {field: int(row[field]) for field in CONFIG_FIELDS}
        entry = mapping_entries.get(mapping_key_for_config(config))
        if not entry or not entry.get("valid", False):
            continue
        bounds = static_state_bounds(config, profile_data, entry)
        records.append(
            {
                **config,
                "static_mapper_throughput": float(
                    entry["mapper_throughput"]
                ),
                "static_mapper_ttft_s": float(
                    entry["max_prefill_latency_us"]
                )
                / 1e6,
                "estimated_throughput": float(
                    bounds["throughput_bound"]
                ),
                "estimated_prefill_service_s": float(
                    bounds["prefill_service_s"]
                ),
                "stage_balance": float(bounds["stage_balance"]),
                "prefill_pressure": float(bounds["prefill_pressure"]),
                "bandwidth_pressure": float(bounds["bw_pressure"]),
            }
        )

    if not records:
        raise RuntimeError("No valid static-state configurations were available.")

    frame = pd.DataFrame(records)
    frame["throughput_per_prefill_s"] = (
        frame["estimated_throughput"]
        / frame["estimated_prefill_service_s"].clip(lower=1e-12)
    )
    return frame


def _coverage_indices(
    frame: pd.DataFrame,
    metric_column: str,
    budget: int,
) -> list[int]:
    """Cover resource envelopes before filling by a physical metric."""

    if budget <= 0:
        return []

    groups: list[tuple[float, list[int]]] = []
    coarse_fields = ["batchsize", "Num M", "NoI_bw(GBps)"]
    for _, group in frame.groupby(coarse_fields, sort=False):
        ranked = group.sort_values(metric_column, ascending=False, kind="stable")
        groups.append((float(ranked.iloc[0][metric_column]), ranked.index.tolist()))
    groups.sort(key=lambda item: item[0], reverse=True)

    selected = [indices[0] for _, indices in groups[:budget]]
    selected_set = set(selected)
    if len(selected) < budget:
        for index in frame.sort_values(
            metric_column,
            ascending=False,
            kind="stable",
        ).index:
            if index not in selected_set:
                selected.append(int(index))
                selected_set.add(int(index))
            if len(selected) == budget:
                break
    return selected


def retain_candidates(
    evaluated: pd.DataFrame,
    candidate_budget: int = 256,
) -> pd.DataFrame:
    """Retain promising but non-duplicative static states.

    The grouping rule is shared by every model and workload. It keeps broad
    coverage over batch size, memory count, and bandwidth, then fills the
    remaining budget in physical-metric order.
    """

    if candidate_budget <= 0:
        raise ValueError("candidate_budget must be positive.")
    if candidate_budget >= len(evaluated):
        result = evaluated.copy()
        result["selection_sources"] = "all_valid"
        result["candidate_rank"] = np.arange(1, len(result) + 1)
        return result

    sources: dict[int, set[str]] = {}
    mt_budget = max(1, 3 * candidate_budget // 4)
    bstar_budget = candidate_budget - mt_budget
    mt_indices = _coverage_indices(
        evaluated,
        "estimated_throughput",
        mt_budget,
    )
    bstar_indices = _coverage_indices(
        evaluated,
        "throughput_per_prefill_s",
        bstar_budget,
    )
    for index in mt_indices:
        sources.setdefault(index, set()).add("mt")
    for index in bstar_indices:
        sources.setdefault(index, set()).add("bstar")

    selected = list(sources)

    if len(selected) < candidate_budget:
        mt_order = evaluated.sort_values(
            "estimated_throughput", ascending=False, kind="stable"
        ).index
        bstar_order = evaluated.sort_values(
            "throughput_per_prefill_s", ascending=False, kind="stable"
        ).index
        for mt_index, bstar_index in zip(mt_order, bstar_order):
            for index, source in (
                (int(mt_index), "mt_fill"),
                (int(bstar_index), "bstar_fill"),
            ):
                if index not in sources:
                    sources[index] = {source}
                    selected.append(index)
                if len(selected) == candidate_budget:
                    break
            if len(selected) == candidate_budget:
                break

    result = evaluated.loc[selected].copy()
    result["selection_sources"] = [
        "+".join(sorted(sources[index])) for index in selected
    ]
    result["candidate_rank"] = np.arange(1, len(result) + 1)
    return result.reset_index(drop=True)
