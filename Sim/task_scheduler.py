"""Runtime task-mapping policies for pipeline execution.

The request scheduler decides which batch runs. These policies decide which
compute chiplet executes the current block of that batch.
"""

import os

import simpy

import Sim.common as common
from Sim.entities.comp_chiplet import comp_chiplet
from Sim.entities.mem_chiplet import mem_chiplet
from Sim.logger import init_logger


logger = init_logger(__name__)


_fcfs_rr_cursors: dict[tuple[tuple[int, ...], tuple[int, ...]], int] = {}
_STARVATION_RETRY_TICKS = 200
_FCFS_DISPATCH_OVERHEAD_TICKS = int(os.environ.get("HYDRA_FCFS_DISPATCH_OVERHEAD_TICKS", "5"))
_WORKSTEAL_DISPATCH_OVERHEAD_TICKS = int(os.environ.get("HYDRA_WORKSTEAL_DISPATCH_OVERHEAD_TICKS", "8"))
_DEBUG_SCHEDULER = os.environ.get("HYDRA_SCHED_DEBUG", "0") == "1"


def _type_key(chiplet_types) -> tuple[int, ...]:
    key = []
    for item in chiplet_types:
        if isinstance(item, (list, tuple)):
            key.extend(int(v) for v in item)
        else:
            key.append(int(item))
    return tuple(key)


def _clear_congested(batch_id: int) -> None:
    from Sim.metrics.monitor import congestion_monitor

    congestion_monitor.clear_congested(batch_id)


def _inc_congested(batch_id: int) -> None:
    from Sim.metrics.monitor import congestion_monitor

    congestion_monitor.inc_congested(batch_id)


def _dispatch_overhead_delay(overhead_ticks: int) -> int:
    return max(0, overhead_ticks) * common.simulation_clk


def _try_reserve_slot(
    env: simpy.Environment,
    chip_graph,
    comp_sys,
    chiplet: comp_chiplet | None,
    target_weight_chiplet: mem_chiplet,
    select_bw: int,
    batch_id: int,
    require_queue_head: bool = False,
):
    if chiplet is None or not chiplet.is_available():
        return None
    if require_queue_head and (not chiplet.job_queue or chiplet.job_queue[0] != batch_id):
        return None
    if not chip_graph.check_bw_availability(chiplet.chiplet_id, select_bw):
        return None
    if not chip_graph.check_bw_availability(target_weight_chiplet.chiplet_id, select_bw):
        return None
    route = chip_graph.find_unused_path(
        chiplet.chiplet_loc,
        target_weight_chiplet.chiplet_loc,
        select_bw,
    )
    if route is None:
        return None
    comp_sys.set_busy_byid(chiplet.chiplet_id)
    chip_graph.load_bw_byid(chiplet.chiplet_id, select_bw, env.now)
    chip_graph.reserve_path(route, select_bw, env.now)
    chip_graph.load_bw_byid(target_weight_chiplet.chiplet_id, select_bw, env.now)
    _clear_congested(batch_id)
    return route


def _wait_reason(chip_graph, chiplet: comp_chiplet | None, target_weight_chiplet: mem_chiplet, select_bw: int, batch_id: int) -> str:
    if chiplet is None:
        return "no_chiplet"
    if not chiplet.is_available():
        return "chiplet_busy"
    if not chiplet.job_queue:
        return "empty_queue"
    if chiplet.job_queue[0] != batch_id:
        return "not_queue_head"
    if not chip_graph.check_bw_availability(chiplet.chiplet_id, select_bw):
        return "compute_bw"
    if not chip_graph.check_bw_availability(target_weight_chiplet.chiplet_id, select_bw):
        return "memory_bw"
    route = chip_graph.find_unused_path(chiplet.chiplet_loc, target_weight_chiplet.chiplet_loc, select_bw)
    if route is None:
        return "no_route"
    return "unknown"


def _available_candidates(comp_sys, chip_graph, block_type, fallback_types, select_bw: int) -> list[comp_chiplet]:
    candidates = comp_sys.find_available_chiplets_by_type(block_type, chip_graph, select_bw)
    if not candidates:
        candidates = comp_sys.find_available_chiplets_by_type(fallback_types, chip_graph, select_bw)
    return sorted(candidates, key=lambda chiplet: chiplet.chiplet_id)


def _compatible_candidates(comp_sys, block_type, fallback_types) -> list[comp_chiplet]:
    candidates = comp_sys.find_chiplets_by_type(block_type)
    if not candidates:
        candidates = comp_sys.find_chiplets_by_type(fallback_types)
    return sorted(candidates, key=lambda chiplet: chiplet.chiplet_id)


def _next_fcfs_chiplet(comp_sys, block_type, fallback_types) -> comp_chiplet | None:
    candidates = _compatible_candidates(comp_sys, block_type, fallback_types)
    if not candidates:
        return None
    key = (_type_key(block_type), _type_key(fallback_types))
    cursor = _fcfs_rr_cursors.get(key, 0)
    selected = candidates[cursor % len(candidates)]
    _fcfs_rr_cursors[key] = cursor + 1
    return selected


def _mapped_chiplet(comp_sys, block_id: int, infer_type: str) -> comp_chiplet:
    stage_key = "P" if infer_type == "prefill" else "D"
    selected_id = common.job_mapping[stage_key][block_id]
    return comp_sys.find_chiplet_by_id(selected_id)


def _select_static(
    owner,
    env: simpy.Environment,
    chip_graph,
    comp_sys,
    block_id: int,
    infer_type: str,
    target_weight_chiplet: mem_chiplet,
    select_bw: int,
):
    selected_chiplet = _mapped_chiplet(comp_sys, block_id, infer_type)
    selected_chiplet.enqueue_job(owner.batch_id)
    while True:
        route = _try_reserve_slot(
            env,
            chip_graph,
            comp_sys,
            selected_chiplet,
            target_weight_chiplet,
            select_bw,
            owner.batch_id,
            require_queue_head=True,
        )
        if route is not None:
            return selected_chiplet, route
        yield env.timeout(common.simulation_clk)


def _select_fcfs(
    owner,
    env: simpy.Environment,
    chip_graph,
    comp_sys,
    block_type: list[int],
    fallback_types: list[int],
    target_weight_chiplet: mem_chiplet,
    select_bw: int,
):
    # FCFS baseline: assign incoming tasks to compatible chiplets in a global
    # round-robin order, ignoring placement preference and communication cost.
    selected_chiplet = _next_fcfs_chiplet(comp_sys, block_type, fallback_types)
    if selected_chiplet is None:
        while True:
            _inc_congested(owner.batch_id)
            yield env.timeout(common.simulation_clk)

    overhead_delay = _dispatch_overhead_delay(_FCFS_DISPATCH_OVERHEAD_TICKS)
    if overhead_delay:
        yield env.timeout(overhead_delay)
    selected_chiplet.enqueue_job(owner.batch_id)
    wait_ticks = 0
    while True:
        route = _try_reserve_slot(
            env,
            chip_graph,
            comp_sys,
            selected_chiplet,
            target_weight_chiplet,
            select_bw,
            owner.batch_id,
            require_queue_head=True,
        )
        if route is not None:
            return selected_chiplet, route
        wait_ticks += 1
        reason = _wait_reason(chip_graph, selected_chiplet, target_weight_chiplet, select_bw, owner.batch_id)
        if _DEBUG_SCHEDULER and wait_ticks % _STARVATION_RETRY_TICKS == 0:
            logger.info(
                "fcfs wait batch=%s chiplet=%s reason=%s queue_len=%s",
                owner.batch_id,
                selected_chiplet.chiplet_id,
                reason,
                selected_chiplet.queue_length,
            )
        if wait_ticks >= _STARVATION_RETRY_TICKS and reason != "memory_bw":
            selected_chiplet.remove_job(owner.batch_id)
            next_chiplet = _next_fcfs_chiplet(comp_sys, block_type, fallback_types)
            if next_chiplet is not None:
                selected_chiplet = next_chiplet
                selected_chiplet.enqueue_job(owner.batch_id)
            wait_ticks = 0
        _inc_congested(owner.batch_id)
        yield env.timeout(common.simulation_clk)


def _select_worksteal(
    owner,
    env: simpy.Environment,
    chip_graph,
    comp_sys,
    block_type: list[int],
    fallback_types: list[int],
    block_id: int,
    infer_type: str,
    target_weight_chiplet: mem_chiplet,
    select_bw: int,
):
    # Work-steal baseline: start from the FCFS round-robin assignment, then
    # migrate only by queue/busy-state imbalance. It does not score the
    # communication overhead of the stolen placement.
    selected_chiplet = _next_fcfs_chiplet(comp_sys, block_type, fallback_types)
    if selected_chiplet is None:
        while True:
            _inc_congested(owner.batch_id)
            yield env.timeout(common.simulation_clk)

    overhead_delay = _dispatch_overhead_delay(_WORKSTEAL_DISPATCH_OVERHEAD_TICKS)
    if overhead_delay:
        yield env.timeout(overhead_delay)
    selected_chiplet.enqueue_job(owner.batch_id)
    wait_ticks = 0
    while True:
        route = _try_reserve_slot(
            env,
            chip_graph,
            comp_sys,
            selected_chiplet,
            target_weight_chiplet,
            select_bw,
            owner.batch_id,
            require_queue_head=True,
        )
        if route is not None:
            return selected_chiplet, route

        wait_ticks += 1
        reason = _wait_reason(chip_graph, selected_chiplet, target_weight_chiplet, select_bw, owner.batch_id)
        if _DEBUG_SCHEDULER and wait_ticks % _STARVATION_RETRY_TICKS == 0:
            logger.info(
                "worksteal wait batch=%s chiplet=%s reason=%s queue_len=%s",
                owner.batch_id,
                selected_chiplet.chiplet_id,
                reason,
                selected_chiplet.queue_length,
            )
        if reason == "memory_bw":
            _inc_congested(owner.batch_id)
            yield env.timeout(common.simulation_clk)
            continue
        current_load = selected_chiplet.queue_length + (0 if selected_chiplet.is_available() else 1)
        candidates = _compatible_candidates(comp_sys, block_type, fallback_types)
        candidates = sorted(
            candidates,
            key=lambda chiplet: (
                chiplet.queue_length + (0 if chiplet.is_available() else 1),
                chiplet.chiplet_id,
            ),
        )
        for candidate in candidates:
            if candidate.chiplet_id == selected_chiplet.chiplet_id:
                continue
            candidate_load = candidate.queue_length + (0 if candidate.is_available() else 1)
            if candidate_load >= current_load and wait_ticks < _STARVATION_RETRY_TICKS:
                continue
            candidate.enqueue_job(owner.batch_id)
            route = _try_reserve_slot(
                env,
                chip_graph,
                comp_sys,
                candidate,
                target_weight_chiplet,
                select_bw,
                owner.batch_id,
                require_queue_head=True,
            )
            if route is not None:
                selected_chiplet.remove_job(owner.batch_id)
                return candidate, route
            candidate.remove_job(owner.batch_id)
            break

        if wait_ticks >= _STARVATION_RETRY_TICKS and reason != "memory_bw":
            selected_chiplet.remove_job(owner.batch_id)
            next_chiplet = _next_fcfs_chiplet(comp_sys, block_type, fallback_types)
            if next_chiplet is not None:
                selected_chiplet = next_chiplet
                selected_chiplet.enqueue_job(owner.batch_id)
            wait_ticks = 0
        _inc_congested(owner.batch_id)
        yield env.timeout(common.simulation_clk)


def _select_elastic(
    owner,
    env: simpy.Environment,
    chip_graph,
    comp_sys,
    block_type: list[int],
    fallback_types: list[int],
    block_id: int,
    infer_type: str,
    target_weight_chiplet: mem_chiplet,
    select_bw: int,
):
    selected_chiplet = None
    elastic_wait_ticks = 0
    stage_key = "P" if infer_type == "prefill" else "D"
    while True:
        if getattr(common, "elastic_runtime_rebalance_enabled", False):
            comp_sys_chiplets = comp_sys.find_chiplets_by_type(block_type)
            if not comp_sys_chiplets:
                target_comp_chiplets = comp_sys.find_chiplets_by_type(fallback_types)
                if not target_comp_chiplets:
                    yield env.timeout(common.simulation_clk)
                    continue
            else:
                target_comp_chiplets = comp_sys_chiplets

            if selected_chiplet is None:
                selected_chiplet = owner._choose_elastic_chiplet_with_rebalance(
                    target_comp_chiplets=target_comp_chiplets,
                    block_id=block_id,
                    stage_key=stage_key,
                    chip_graph=chip_graph,
                    target_weight_chiplet=target_weight_chiplet,
                )
                if selected_chiplet is None:
                    yield env.timeout(common.simulation_clk)
                    continue
                selected_chiplet.enqueue_job(owner.batch_id)

            if not selected_chiplet.job_queue or selected_chiplet.job_queue[0] != owner.batch_id:
                elastic_wait_ticks += 1
            elif not selected_chiplet.is_available():
                elastic_wait_ticks += 1
            elif not chip_graph.check_bw_availability(selected_chiplet.chiplet_id, select_bw):
                elastic_wait_ticks += 1
            elif not chip_graph.check_bw_availability(target_weight_chiplet.chiplet_id, select_bw):
                elastic_wait_ticks += 1
            else:
                route = chip_graph.find_unused_path(
                    selected_chiplet.chiplet_loc,
                    target_weight_chiplet.chiplet_loc,
                    select_bw,
                )
                if route is not None:
                    comp_sys.set_busy_byid(selected_chiplet.chiplet_id)
                    chip_graph.load_bw_byid(selected_chiplet.chiplet_id, select_bw, env.now)
                    chip_graph.reserve_path(route, select_bw, env.now)
                    chip_graph.load_bw_byid(target_weight_chiplet.chiplet_id, select_bw, env.now)
                    _clear_congested(owner.batch_id)
                    return selected_chiplet, route
                elastic_wait_ticks += 1

            selected_chiplet, elastic_wait_ticks = owner._maybe_reselect_elastic_chiplet(
                selected_chiplet=selected_chiplet,
                target_comp_chiplets=target_comp_chiplets,
                block_id=block_id,
                stage_key=stage_key,
                chip_graph=chip_graph,
                target_weight_chiplet=target_weight_chiplet,
                wait_ticks=elastic_wait_ticks,
            )
            yield env.timeout(common.simulation_clk)
            continue

        candidates = comp_sys.find_available_chiplets_by_type(block_type, chip_graph, select_bw)
        if not candidates:
            candidates = comp_sys.find_available_chiplets_by_type(fallback_types, chip_graph, select_bw)

        best_chiplet = None
        best_route = None
        best_distance = float("inf")
        for candidate in candidates:
            if not chip_graph.check_bw_availability(target_weight_chiplet.chiplet_id, select_bw):
                continue
            route = chip_graph.find_unused_path(
                candidate.chiplet_loc,
                target_weight_chiplet.chiplet_loc,
                select_bw,
            )
            if route is None:
                continue
            distance = chip_graph.get_distance(candidate.chiplet_loc, target_weight_chiplet.chiplet_loc)
            if distance < best_distance:
                best_chiplet = candidate
                best_route = route
                best_distance = distance

        if best_chiplet is not None and best_route is not None:
            comp_sys.set_busy_byid(best_chiplet.chiplet_id)
            chip_graph.load_bw_byid(best_chiplet.chiplet_id, select_bw, env.now)
            chip_graph.reserve_path(best_route, select_bw, env.now)
            chip_graph.load_bw_byid(target_weight_chiplet.chiplet_id, select_bw, env.now)
            _clear_congested(owner.batch_id)
            return best_chiplet, best_route

        _inc_congested(owner.batch_id)
        yield env.timeout(common.simulation_clk)


def select_task_slot(
    owner,
    env: simpy.Environment,
    chip_graph,
    comp_sys,
    block_type: list[int],
    fallback_types: list[int],
    block_id: int,
    infer_type: str,
    target_weight_chiplet: mem_chiplet,
    select_bw: int,
):
    strategy = common.mapping_strategy.lower()
    if strategy == "static":
        return (
            yield from _select_static(
                owner,
                env,
                chip_graph,
                comp_sys,
                block_id,
                infer_type,
                target_weight_chiplet,
                select_bw,
            )
        )
    if strategy == "fcfs":
        return (
            yield from _select_fcfs(
                owner,
                env,
                chip_graph,
                comp_sys,
                block_type,
                fallback_types,
                target_weight_chiplet,
                select_bw,
            )
        )
    if strategy in {"worksteal", "work_steal", "work-steal"}:
        return (
            yield from _select_worksteal(
                owner,
                env,
                chip_graph,
                comp_sys,
                block_type,
                fallback_types,
                block_id,
                infer_type,
                target_weight_chiplet,
                select_bw,
            )
        )
    if strategy == "elastic":
        return (
            yield from _select_elastic(
                owner,
                env,
                chip_graph,
                comp_sys,
                block_type,
                fallback_types,
                block_id,
                infer_type,
                target_weight_chiplet,
                select_bw,
            )
        )
    raise ValueError(
        f"Unknown mapping strategy '{common.mapping_strategy}'. "
        "Expected static, fcfs, worksteal, or elastic."
    )
